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
    changes={'source_identity_low32': [('source_sequence!=command_sequence || source_epoch!=command_epoch', 'source_sequence[31:0]!=command_sequence[31:0] || source_epoch[31:0]!=command_epoch[31:0]')], 'lease_overflow_ignored': [("command_lease_base>64'hffffffffffffffe0", "1'b0")], 'zero_lease_accepted': [('command_lease_base==0 ||', "1'b0 ||")], 'layer_range_ignored': [('command_layer>=16 ||', "1'b0 ||")], 'source_extent_ignored': [('source_qk_words!=2560 || source_v_beats!=32 || source_frames!=192', "1'b0")], 'prior_source_status_lost': [('else if(source_status!=OK)admission_status=source_status;', 'else if(source_status!=OK)admission_status=OK;')], 'wrong_kv_ram_address': [("8'd128+index", "8'd127+index")], 'wrong_q_ram_address': [('source_address=q_phase ? index:', "source_address=q_phase ? index+8'd1:")], 'frame_bytes_corrupted': [('assign data_bits=frame;', "assign data_bits=frame^512'd1;")], 'early_kv_last': [('assign data_last=index==63;', 'assign data_last=index==62;')], 'early_q_last': [('assign q_last=index==127;', 'assign q_last=index==126;')], 'commit_fault_lost': [('assign commit_status=commit_code;', 'assign commit_status=OK;')], 'nonfinite_accepted': [("if(source_q[lane*32+23+:8]==8'hff)finite_frame=0;", "if(1'b0)finite_frame=0;")], 'timeout_status_lost': [('else if(timeout_now)fault_now=TIMEOUT;', 'else if(timeout_now)fault_now=BAD_PEER;')], 'cancel_status_lost': [('else if(cancel_valid)fault_now=CANCELLED;', 'else if(cancel_valid)fault_now=BAD_IDENTITY;')], 'forgotten_revocation': [('if(busy && (epoch_bad || cancel_valid || source_bad || cache_bad || root_bad))revoked<=1;', "if(1'b0)revoked<=1;")], 'early_source_ack': [('source_completion_ready=state==DRAIN &&', 'source_completion_ready=(state==DRAIN || state==RESPONSE) &&')], 'foreign_source_ack': [('source_completion_valid && source_identity && !append_owned', 'source_completion_valid && !append_owned')], 'lost_cache_exclusion': [('cache_exclusive=busy;', 'cache_exclusive=busy && state!=RESPONSE;')], 'result_address_corrupted': [('root_result_address=result_address;', "root_result_address=result_address+11'd1;")], 'response_read_bubble_lost': [('result_q_valid=root_result_q_valid && result_valid;', 'result_q_valid=result_valid;')], 'kv_commit_lost_on_cancel': [('completion_kv_committed<=completion_append_status==OK &&', 'completion_kv_committed<=fault_now==OK && completion_append_status==OK &&')], 'root_status_lost': [('else if(root_status!=OK)completion_status<=root_status;', 'else if(root_status!=OK)completion_status<=OK;')], 'source_valid_change_ignored': [('(saved_source_status==OK && !source_result_valid)', "1'b0")], 'capture_sequence_low32': [('completion_sequence<=command_sequence;', "completion_sequence<={32'd0,command_sequence[31:0]};")]}
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
    info=dict(stage='V7-qkv-dispatch-mutants-r2',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_qkv_dispatch_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_qkv_dispatch_r4.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_qkv_dispatch_r4','-o',str(binary),str(path),str(a.source/'fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_append_r1.sv'),str(a.source/'fpga/rtl/ntrpu_attention_root_guard_r1.sv'),str(a.source/'fpga/rtl/ntrpu_attention_head_sequencer_r1.sv'),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'DISPATCH_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('DISPATCH_FAULT_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
