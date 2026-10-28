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
    changes={'skip_write_alias_preflight': [('if(range_index==9)state<=CHECK_ALIASES', 'if(range_index==9)state<=SOURCE_OFFER')], 'ignore_cache_overlap': [(' || cache_overlap)', ')')], 'ignore_phase_position': [('held_coefficient_position!=completion_position ||', '')], 'ignore_lease_overflow': [("held_lease>64'hffffffffffffffe0 ||", '')], 'ignore_address_alignment': [('addr[5:0]==0 && base[5:0]==0', 'base[5:0]==0')], 'ignore_base_alignment': [('addr[5:0]==0 && base[5:0]==0', 'addr[5:0]==0')], 'ignore_size_alignment': [('bytes[5:0]==0 && bytes!=0', 'bytes!=0')], 'allow_wrapped_range': [('!last[64] && !limit[64] && addr>=base', 'addr>=base')], 'query_live_host_layer': [('cache_query_layer=owner_busy ? completion_layer:command_layer', 'cache_query_layer=command_layer')], 'lease_reads_previous_layer': [('.cache_committed_tokens(cache_query_committed_tokens)', '.cache_committed_tokens(cache_committed_tokens)')], 'producer_live_host_identity': [('producer_command_sequence=completion_sequence', 'producer_command_sequence=command_sequence')], 'producer_low_sequence': [('producer_command_source_sequence=held_source_sequence', "producer_command_source_sequence={32'd0,held_source_sequence[31:0]}")], 'early_producer_ack': [('producer_completion_ready=lease_owned ? lease_source_completion_ready:', "producer_completion_ready=producer_completion_valid ? 1'b1:")], 'early_lease_ack': [('lease_completion_ready=state==ACK_LEASE &&', 'lease_completion_ready=state==RESPONSE &&')], 'unreserved_preflight': [('cache_exclusive=owner_busy || producer_busy', 'cache_exclusive=producer_busy')], 'ignore_host_cancel': [('else if(cancel_valid && cancel_ready)fault_now=CANCELLED;', '')]}
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
    info=dict(stage='V7-input-attention-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_input_attention_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_input_attention_r3.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_input_attention_r3','-o',str(binary),str(path),str(a.source/'fpga/rtl/ntrpu_qkv_rope_handoff_r1.sv'),str(a.source/'fpga/tb/ntrpu_input_attention_protocol_peers_r1.sv'),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'IA_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('IA_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline','progress','offered','offer reached','final coordinated drain')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
