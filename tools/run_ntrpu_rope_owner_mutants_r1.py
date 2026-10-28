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
        'identity_low32': [('command_sequence!=source_sequence','command_sequence[31:0]!=source_sequence[31:0]'),('command_epoch!=source_epoch','command_epoch[31:0]!=source_epoch[31:0]'),('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]')],
        'wrong_input_plane':[('source_q[plane]:32','source_q[1-plane]:32')],
        'wrong_output_word':[('output_mem[hls_output_address]<=hls_output_d;',"output_mem[hls_output_address]<=hls_output_d^32'd1;")],
        'early_publication':[('result_valid=completion_valid &&','result_valid=(completion_valid || state==H_WAIT) &&')],
        'missing_count':[('else if(write_after!=2560)fault_now=BAD_MEMORY;',"else if(1'b0)fault_now=BAD_MEMORY;")],
        'missing_response_ignored':[('state==H_WAIT && input_bad',"state==H_WAIT && 1'b0")],
        'input_nonfinite_ignored':[(" || source_q[plane][30:23]==8'hff",'')],
        'ignore_cancel':[('else if(cancel_valid)fault_now=CANCELLED;',"else if(1'b0)fault_now=CANCELLED;")],
        'ignore_timeout':[('else if(timeout_now)fault_now=TIMEOUT;',"else if(1'b0)fault_now=TIMEOUT;")],
        'forgotten_revocation':[('if(busy && (epoch_bad || cancel_valid))revoked<=1;',"if(1'b0)revoked<=1;")],
        'early_source_release':[('source_completion_ready=state==RELEASE_SOURCE &&','source_completion_ready=(state==RELEASE_SOURCE || state==H_WAIT) &&')],
        'early_reuse':[('command_ready=state==IDLE &&','command_ready=(state==IDLE || state==RESPONSE) &&')],
        'wrong_position':[('hls_position=completion_position','hls_position=completion_position+1')],
        'wrong_heads':[("hls_heads=32'd40","hls_heads=32'd32")],
        'hls_failure_ignored':[('else if(hls_return!=0)fault_now=COMPUTE_ERROR;',"else if(1'b0)fault_now=COMPUTE_ERROR;")],
        'hls_ready_ignored':[('if(!hls_ready)fault_now=BAD_PEER;',"if(1'b0)fault_now=BAD_PEER;")],
        'format_ignored':[(' || word_bad)',')')],
        'output_order_ignored':[(" || hls_output_address!=write_count",'')],
        'foreign_source_acked':[('source_sequence==completion_sequence && source_epoch==completion_epoch;',"1'b1;")],
        'response_tied_to_current_ce':[('source_live && source_q_valid[plane] ?','source_live && hls_input_ce[plane] && source_q_valid[plane] ?')],
        'bad_position_accepted':[("command_position>=32'd131072","1'b0")],
        'source_position_ignored':[('source_position!=command_position',"1'b0")],
        'coefficient_position_ignored':[('source_coefficient_position!=command_position',"1'b0")],
        'position_drift_ignored':[('source_position!=completion_position',"1'b0")],
        'coefficient_drift_ignored':[('source_coefficient_position!=coefficient_position',"1'b0")],
        'source_bounds_ignored':[('active_hls && input_address_bad',"1'b0")],
        'consumer_bounds_ignored':[("result_pending<=result_ce && result_valid && result_address<12'd2560;",'result_pending<=result_ce && result_valid;')],
        'wrong_coefficient_position':[('hls_coefficient_position=coefficient_position','hls_coefficient_position=coefficient_position+1')],
        'capture_low32':[('completion_sequence<=command_sequence;',"completion_sequence<={32'd0,command_sequence[31:0]};")],
    }
    result={}
    for name,pairs in changes.items():
        text=original
        for old,new in pairs:
            assert old in text,(name,old);text=text.replace(old,new)
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
    info=dict(stage='V7-rope-owner-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_rope_owner_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_rope_owner_r1.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_rope_owner_r1','-o',str(binary),str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'ROPEOWNER_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('ROPEOWNER_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
