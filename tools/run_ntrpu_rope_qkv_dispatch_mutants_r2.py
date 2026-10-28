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
    changes={'downstream_drain_abandoned': [('DRAIN:if(!rope_busy && !capture_busy)', 'DRAIN:if(!capture_completion_valid)')]}
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
    info=dict(stage='V7-rope-qkv-dispatch-mutants-r2',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_rope_qkv_dispatch_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_rope_qkv_dispatch_r3.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_rope_qkv_dispatch_r3','-o',str(binary),str(path)]+[str(a.source/n) for n in ['fpga/rtl/ntrpu_rope_input_real_r1.sv', 'fpga/rtl/ntrpu_rope_input_path_r2.sv', 'fpga/rtl/ntrpu_rope_owner_r2.sv', 'fpga/rtl/ntrpu_rope_loader_r1.sv', 'fpga/tb/ntrpu_rope_identity_protocol_model_r2.sv', 'fpga/rtl/ntrpu_qkv_capture_dispatch_r1.sv', 'fpga/rtl/ntrpu_qkv_snapshot_r1.sv', 'fpga/rtl/ntrpu_qkv_dispatch_r1.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_append_r1.sv', 'fpga/rtl/ntrpu_attention_root_guard_r1.sv', 'fpga/rtl/ntrpu_attention_head_sequencer_r1.sv', 'fpga/tb/tb_ntrpu_rope_qkv_dispatch_r3.sv']])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'DISPATCH_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('ROPECAP_PROTOCOL_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
