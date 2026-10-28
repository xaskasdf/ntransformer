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
    info=dict(stage='V7-attention-projection-control-native-r3',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,per_command_timeout_seconds=120,native_only=True,hardware_slot=False,status='running',commands=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    with (out/'version.log').open('wb') as f:subprocess.run(['iverilog','-V'],stdout=f,stderr=subprocess.STDOUT,check=True)
    commands=[('compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_attention_projection_control_r1','-o',str(out/'simulation.vvp'),
        str(a.source/'fpga/rtl/registered_watchdog_r3/ntrpu_attention_projection_control_r1.sv'),str(a.source/'fpga/tb/tb_ntrpu_attention_projection_control_r1.sv')])]
    commands += [(f'seed-{seed}.log',['vvp',str(out/'simulation.vvp'),f'+SEED={seed}']) for seed in (1,7,42)]
    commands += [('compile-exhaustion.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_attention_projection_control_r1',
        '-Ptb_ntrpu_attention_projection_control_r1.EXHAUSTION=1','-o',str(out/'exhaustion.vvp'),
        str(a.source/'fpga/rtl/registered_watchdog_r3/ntrpu_attention_projection_control_r1.sv'),str(a.source/'fpga/tb/tb_ntrpu_attention_projection_control_r1.sv')]),
        ('exhaustion.log',['vvp',str(out/'exhaustion.vvp'),'+SEED=1'])]
    code=0
    for name,cmd in commands:
        t=time.monotonic()
        with (out/name).open('wb') as f:
            try:result=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120);code=result.returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-t))
        inv.write_text(json.dumps(info,indent=2)+'\n')
        if code:break
    info.update(exit_code=code,duration_seconds=time.monotonic()-started,finished_utc=now(),status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
