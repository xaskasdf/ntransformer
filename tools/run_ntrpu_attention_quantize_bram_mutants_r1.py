"""Bounded Icarus RTL preflight, separate from Vivado qualification."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[:2];os.sched_setaffinity(0,affinity)
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for name,sha in hashes.items():assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==sha,name
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    now=lambda:datetime.now(timezone.utc).isoformat();started=time.monotonic()
    info=dict(stage='V7-attention-quantize-bram-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,per_command_timeout_seconds=120,native_only=True,hardware_slot=False,status='running',commands=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    with (out/'version.log').open('wb') as f:subprocess.run(['iverilog','-V'],stdout=f,stderr=subprocess.STDOUT,check=True)
    mutants=[('private_leak', "assign result_q=result_q_valid ? result_data:64'd0;", 'assign result_q=result_data;'), ('unsolicited_response', 'result_pending<=result_ce && result_valid;', 'result_pending<=result_valid;'), ('wrong_address', 'result_data<=output_mem[result_address];', "result_data<=output_mem[result_address^9'd1];"), ('extra_latency', 'if(rst_n && state==RESPONSE)result_data<=output_mem[result_address];', 'if(rst_n && state==RESPONSE && result_pending)result_data<=output_mem[result_address];'), ('ignore_epoch', 'completion_epoch!=expected_epoch', 'completion_epoch[31:0]!=expected_epoch[31:0]'), ('revive_after_cancel', 'if(busy && (epoch_bad || cancel_valid))revoked<=1;', 'if(busy && (epoch_bad || cancel_valid))revoked<=0;')]
    base=(a.source/'fpga/rtl/ntrpu_attention_quantize_owner_r3.sv').read_text()
    commands=[];manifest=[]
    for name,before,after in mutants:
        assert base.count(before)==1
        rtl=out/(name+'.sv');rtl.write_text(base.replace(before,after))
        manifest.append(dict(name=name,before=before,after=after,sha256=hashlib.sha256(rtl.read_bytes()).hexdigest()))
        commands.append((name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_attention_quantize_owner_bram_r1','-o',str(out/(name+'.vvp')),str(a.source/'fpga/rtl/ntrpu_attention_quantize_owner_r2.sv'),str(rtl),str(a.source/'fpga/tb/tb_ntrpu_attention_quantize_owner_bram_r1.sv')]))
        commands.append((name+'-seed-1.log',['vvp',str(out/(name+'.vvp')),'+SEED=1']))
    (out/'MUTATIONS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    code=0
    for name,cmd in commands:
        t=time.monotonic()
        with (out/name).open('wb') as f:
            try:result=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120);code=result.returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-t))
        inv.write_text(json.dumps(info,indent=2)+'\n')
        if name.endswith('-compile.log') and code:break
        if name.endswith('-seed-1.log') and code!=1:break
    code=0 if len(info['commands'])==12 and all(row['exit_code']==(0 if i%2==0 else 1) for i,row in enumerate(info['commands'])) else 1
    info.update(exit_code=code,duration_seconds=time.monotonic()-started,finished_utc=now(),status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
