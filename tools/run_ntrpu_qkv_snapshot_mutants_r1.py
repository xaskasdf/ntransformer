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
    changes={
        'v_identity_low32':[('command_v_source_sequence!=command_sequence','command_v_source_sequence[31:0]!=command_sequence[31:0]'),('command_v_source_epoch!=command_epoch','command_v_source_epoch[31:0]!=command_epoch[31:0]')],
        'unchecked_bounds':[('state==CHECK && bounds_bad',"state==CHECK && 1'b0")],
        'region_overflow_ignored':[(' || region_end[64] ||'," || 1'b0 ||")],
        'wrong_qk_packing':[("<={source_q,packed_words[479:0]};","<={packed_words[479:0],source_q};")],
        'wrong_v_frame':[("frames[8'd160+","frames[8'd159+")],
        'early_frame_write':[("completion_qk_words[3:0]==15","completion_qk_words[3:0]==14")],
        'early_publication':[('result_valid=completion_valid &&','result_valid=(completion_valid || state==COPY_REQ) &&')],
        'cancel_status_lost':[('else if(cancel_valid)fault_now=CANCELLED;',"else if(cancel_valid)fault_now=BAD_IDENTITY;")],
        'timeout_status_lost':[('else if(timeout_now)fault_now=TIMEOUT;',"else if(timeout_now)fault_now=READ_ERROR;")],
        'forgotten_revocation':[('if(busy && (epoch_bad || cancel_valid || source_changed))revoked<=1;',"if(1'b0)revoked<=1;")],
        'early_reuse':[('command_ready=state==IDLE &&','command_ready=(state==IDLE || state==RESPONSE) &&')],
        'v_nonfinite_accepted':[("if(reader_data[lane*32+23+:8]==8'hff)beat_bad=1;","if(1'b0)beat_bad=1;")],
        'qk_nonfinite_accepted':[("state==COPY_RESP && source_q_valid && source_q[30:23]==8'hff","state==COPY_RESP && 1'b0")],
        'reader_status_ignored':[('reader_status!=0 || reader_bytes!=chunk_bytes || chunk_received!=chunk_beats',"1'b0")],
        'source_change_ignored':[('else if(source_changed)fault_now=BAD_PEER;',"else if(1'b0)fault_now=BAD_PEER;")],
        'early_source_ack':[('source_completion_ready=state==DRAIN &&','source_completion_ready=(state==DRAIN || state==RESPONSE) &&')],
        'foreign_source_ack':[('source_completion_valid && source_identity;',"source_completion_valid;")],
        'prior_status_lost':[('else if(source_status!=OK)admission_status=source_status;','else if(source_status!=OK)admission_status=OK;')],
        'layer_range_ignored':[('command_layer>=16 || command_position>=131072',"1'b0 || command_position>=131072")],
        'capture_sequence_low32':[('completion_sequence<=command_sequence;',"completion_sequence<={32'd0,command_sequence[31:0]};")],
        'consumer_bounds_ignored':[('result_pending<=result_ce && result_valid && result_address<192;','result_pending<=result_ce && result_valid;')],
        'source_position_ignored':[('source_position!=command_position ||',"1'b0 ||")],
        'coefficient_position_ignored':[('source_coefficient_position!=command_position',"1'b0")],
    }
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
    info=dict(stage='V7-qkv-snapshot-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_qkv_snapshot_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_qkv_snapshot_r3.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_qkv_snapshot_r3','-o',str(binary),str(path),str(a.source/'fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv'),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'SNAP_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('SNAP_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
