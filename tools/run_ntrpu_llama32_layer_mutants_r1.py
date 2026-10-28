"""Require compiled ownership mutations to fail explicit protocol assertions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def mutations(original):
    changes={'bypass_preflight': [('else if(!gate_result_valid)begin', "else if(1'b0)begin")], 'front_descriptor_from_base': [('assign front_command_addr[0]=address[0];', 'assign front_command_addr[0]=allocation_base[0];')], 'post_residual_from_branch': [('assign post_command_input_address=address[0];', 'assign post_command_input_address=address[12];')], 'truncate_front_identity': [('front_completion_sequence==completion_sequence', 'front_completion_sequence[31:0]==completion_sequence[31:0]')], 'ignore_cache_revocation': [('cache_changed || foreign_peer || changed_peer', 'foreign_peer || changed_peer')], 'unmasked_late_result': [('&& !revoked && fault_now==OK &&', '&& !revoked &&')], 'tail_not_switched_to_ffn': [('if(ffn_command_valid && ffn_command_ready)tail_owner<=2;', 'if(ffn_command_valid && ffn_command_ready)tail_owner<=1;')], 'early_projection_ack': [('assign projection_completion_ready=state==ACK_O', 'assign projection_completion_ready=state==RESPONSE')], 'wrong_source_ram_bank': [('assign front_result_address=projection_source_result_address;', "assign front_result_address=projection_source_result_address ^ 11'h400;")], 'admit_while_busy': [('assign command_ready=admission_allowed && gate_command_ready;', "assign command_ready=1'b1;")], 'drop_revocation_flag': [('fault<=fault_now;revoked<=1;', 'fault<=fault_now;revoked<=0;')], 'release_tail_before_drain': [('if(ffn_command_valid && ffn_command_ready)tail_owner<=2;', 'if(ffn_command_valid && ffn_command_ready)tail_owner<=2;\n            if(lease_completion_ready)tail_owner<=0;')]}
    result={}
    for name,pairs in changes.items():
        text=original
        for old,new in pairs:
            assert text.count(old)==1,(name,old);text=text.replace(old,new)
        assert text!=original;result[name]=text
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[:2];os.sched_setaffinity(0,affinity)
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    for n,h in json.loads((a.source/'SNAPSHOT_FILES.json').read_text()).items():assert sha(a.source/n)==h,n
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    now=lambda:datetime.now(timezone.utc).isoformat();start=time.monotonic()
    info=dict(stage='V7-llama32-layer-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_llama32_layer_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_llama32_layer_protocol_r4.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_llama32_layer_protocol_r4','-o',str(binary),str(path),str(tb)]+[str(a.source/n) for n in ['fpga/rtl/ntrpu_llama32_layer_preflight_r1.sv','fpga/rtl/ntrpu_post_ffn_handoff_r2.sv','fpga/tb/ntrpu_llama32_layer_protocol_peers_r2.sv']])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'LAYER_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('LAYER_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline','progress','offered','offer reached','final coordinated drain')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
