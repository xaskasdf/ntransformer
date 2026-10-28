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
        'identity_low32':[('command_sequence!=command_source_sequence','command_sequence[31:0]!=command_source_sequence[31:0]'),('command_epoch!=command_source_epoch','command_epoch[31:0]!=command_source_epoch[31:0]'),('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]')],
        'unchecked_bounds':[('state==CHECK && bounds_bad',"state==CHECK && 1'b0")],
        'qk_join_offset':[("plane==1 ? 12'd2048:12'd0","plane==1 ? 12'd2047:12'd0")],
        'wrong_lane':[('pending_data[pending_lane*32+:32]',"pending_data[(pending_lane^4'd1)*32+:32]")],
        'early_publication':[('result_valid=completion_valid &&','result_valid=(completion_valid || state==READ) &&')],
        'ignore_cancel':[('else if(cancel_valid && cancel_ready)fault_now=CANCELLED;',"else if(1'b0)fault_now=CANCELLED;")],
        'ignore_timeout':[('else if(timeout_now)fault_now=TIMEOUT;',"else if(1'b0)fault_now=TIMEOUT;")],
        'forgotten_revocation':[('if(busy && (epoch_bad || cancel_valid))revoked<=1;',"if(1'b0)revoked<=1;")],
        'early_reuse':[('command_ready=state==IDLE &&','command_ready=(state==IDLE || state==RESPONSE) &&')],
        'nonfinite_accepted':[("if(reader_data[lane*32+23+:8]==8'hff)beat_bad=1;","if(1'b0)beat_bad=1;")],
        'reader_status_ignored':[('reader_status!=0 || reader_bytes!=chunk_bytes || chunk_received!=chunk_beats',"1'b0")],
        'wrong_k_extent':[("1:return 8'd32;","1:return 8'd31;")],
        'wrong_phase_extent':[("2:return 8'd4;","2:return 8'd5;")],
        'position_range_ignored':[("command_position>=32'd131072","1'b0")],
        'source_position_ignored':[('command_source_position!=command_position',"1'b0")],
        'coefficient_position_ignored':[('command_coefficient_position!=command_position',"1'b0")],
        'capture_position_low16':[('completion_position<=command_position;',"completion_position<={16'd0,command_position[15:0]};")],
        'qk_consumer_bounds_ignored':[("qk_pending<=qk_ce && result_valid && qk_address<12'd2560;",'qk_pending<=qk_ce && result_valid;')],
        'coefficient_consumer_bounds_ignored':[("coefficient_pending<=coefficient_ce && result_valid && coefficient_address<12'd64;",'coefficient_pending<=coefficient_ce && result_valid;')],
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
    info=dict(stage='V7-rope-loader-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_rope_loader_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_rope_loader_r1.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_rope_loader_r1','-o',str(binary),str(path),str(a.source/'fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv'),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'FAIL RPELOAD' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('RPELOAD_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
