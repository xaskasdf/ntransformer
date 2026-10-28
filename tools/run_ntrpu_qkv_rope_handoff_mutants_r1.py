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
    changes={'lost_source_descriptor_guard': [('source_snapshot!=saved_source', "1'b0")], 'source_sequence_low32': [('source_sequence==completion_sequence', 'source_sequence[31:0]==completion_sequence[31:0]')], 'source_epoch_low32': [('source_epoch==completion_epoch', 'source_epoch[31:0]==completion_epoch[31:0]')], 'child_sequence_low32': [('child_sequence==completion_sequence', 'child_sequence[31:0]==completion_sequence[31:0]')], 'child_epoch_low32': [('child_epoch==completion_epoch', 'child_epoch[31:0]==completion_epoch[31:0]')], 'layer_truncated': [('child_command_layer=completion_layer', "child_command_layer={30'd0,completion_layer[1:0]}")], 'lease_truncated': [('child_command_lease_base=lease', "child_command_lease_base={32'd0,lease[31:0]}")], 'phase_reads_v_allocation': [('address[i==3 ? 9:i+6]', 'address[i==3 ? 8:i+6]')], 'source_ack_before_public': [('state==DRAIN_SOURCE && source_owned && source_identity && source_completion_valid && !child_busy', 'source_owned && source_identity && source_completion_valid')], 'source_ack_before_consumer_drain': [('state==DRAIN_SOURCE && source_owned && source_identity && source_completion_valid && !child_busy', 'state==DRAIN_CHILD && source_owned && source_identity && source_completion_valid')], 'child_ack_before_public': [('state==DRAIN_CHILD && child_owned && child_identity && child_completion_valid', 'state==RESPONSE && child_owned && child_identity && child_completion_valid')], 'late_authority_resurrects': [('completion_status==OK && !revoked && fault_now==OK &&', 'completion_status==OK &&')], 'ignore_cache_mutation': [('source_bad || child_bad || cache_bad', 'source_bad || child_bad')], 'allow_cache_overlap': [(' || cache_overlap)', ')')], 'allow_workspace_overlap': [('else if(alias_write)', "else if(1'b0)")], 'ignore_address_alignment': [('addr[5:0]==0 && base[5:0]==0', 'base[5:0]==0')], 'ignore_base_alignment': [('addr[5:0]==0 && base[5:0]==0', 'addr[5:0]==0')], 'ignore_extent_alignment': [('bytes[5:0]==0 && bytes!=0', 'bytes!=0')], 'allow_wrapped_span': [('!last[64] && !limit[64] && addr>=base', 'addr>=base')], 'ignore_child_byte_certificate': [('child_q_frames==128 && child_kv_bytes==4096 &&', 'child_q_frames==128 &&')], 'ignore_child_commit_certificate': [('child_kv_tokens==completion_position+1 && child_kv_committed &&', 'child_kv_tokens==completion_position+1 &&')], 'unauthorized_ram_request': [('child_result_ce=result_ce && result_valid', 'child_result_ce=result_ce')], 'readmit_held_response': [('command_ready=state==IDLE &&', 'command_ready=(state==IDLE || state==RESPONSE) &&')], 'cache_reservation_missing': [('cache_exclusive=busy', "cache_exclusive=1'b0")], 'live_host_identity': [('child_command_sequence=completion_sequence', 'child_command_sequence=command_sequence')]}
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
    info=dict(stage='V7-qkv-rope-handoff-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_qkv_rope_handoff_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_qkv_rope_handoff_r6.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_qkv_rope_handoff_r6','-o',str(binary),str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'QRL_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('QRL_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline','progress','offered','offer reached','final coordinated drain')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
