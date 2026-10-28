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
        'cancel_not_forwarded':[('child_cancel=busy && (cancel_seen || cancel_valid)',"child_cancel=1'b0")],
        'early_reuse':[('command_ready=state==IDLE &&','command_ready=(state==IDLE || state==RESPONSE) &&')],
        'early_owner_ack':[('owner_match && ((state==RESPONSE && completion_ready) || state==DRAIN)', 'owner_match && ((state==RESPONSE) || state==DRAIN)')],
        'public_data_changed':[("result_q=result_q_valid ? owner_q:32'd0","result_q=result_q_valid ? (owner_q^32'd1):32'd0")],
        'result_address_changed':[('.result_ce(owner_read),.result_address,',".result_ce(owner_read),.result_address(result_address^12'd1),")],
        'source_qk_swapped':[('.qk_q(source_q[0])','.qk_q(source_q[1])'),('.coefficient_q(source_q[1])','.coefficient_q(source_q[0])')],
        'wrong_hls_position':[('.command_position(completion_position)',".command_position(completion_position+32'd1)")],
        'bad_status_reported':[('completion_status<=owner_status;',"completion_status<=4'd0;")],
        'captured_position_low16':[('completion_position<=command_position;',"completion_position<={16'd0,command_position[15:0]};")],
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
    info=dict(stage='V7-rope-input-path-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_rope_input_path_r2.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_rope_input_path_r3.sv'
    variants={name:('path',text) for name,text in mutations(original).items()}
    owner=(a.source/'fpga/rtl/ntrpu_rope_owner_r2.sv').read_text()
    needle='        else if(cancel_valid)admission_status=CANCELLED;\n';assert owner.count(needle)==1
    variants['handoff_cancel_lost']=('owner',owner.replace(needle,''))
    for name,(target,text) in variants.items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_rope_input_path_r3','-o',str(binary),str(path) if target=='path' else str(a.source/'fpga/rtl/ntrpu_rope_input_path_r2.sv'),str(path) if target=='owner' else str(a.source/'fpga/rtl/ntrpu_rope_owner_r2.sv'),str(a.source/'fpga/rtl/ntrpu_rope_loader_r1.sv'),str(a.source/'fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv'),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'FAIL RPEPATH' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('RPEPATH_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,target=target,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
