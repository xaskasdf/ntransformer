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
    info=dict(stage='V7-residual-real-native-r2',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,per_command_timeout_seconds=120,native_only=True,hardware_slot=False,compile_only=False,actual_arithmetic_compiled=False,arithmetic_qualified=False,status='running',commands=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    with (out/'version.log').open('wb') as f:subprocess.run(['iverilog','-V'],stdout=f,stderr=subprocess.STDOUT,check=True)
    compiled=json.loads((a.source/'compiled_sources.json').read_text())
    assert len(compiled)==16 and len(set(compiled))==16
    assert sum(n.startswith('generated/') for n in compiled)==9
    attestation={n:hashlib.sha256((a.source/n).read_bytes()).hexdigest() for n in compiled}
    assert all(attestation[n]==hashes[n] for n in compiled)
    (out/'compiled_sources.json').write_text(json.dumps(attestation,indent=2)+'\n')
    commands=[('compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_residual_real_r1','-o',str(out/'protocol.vvp')]+[str(a.source/n) for n in compiled])]
    commands += [('seed-'+str(seed)+'.log',['vvp',str(out/'protocol.vvp'),'+SEED='+str(seed),'+FIXTURE='+str(a.source/'fixture')]) for seed in (1,7,42)]
    code=0
    for name,cmd in commands:
        t=time.monotonic()
        with (out/name).open('wb') as f:
            try:result=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120);code=result.returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-t))
        inv.write_text(json.dumps(info,indent=2)+'\n')
        if code:break
        if name=='compile.log':info['actual_arithmetic_compiled']=True
    info.update(exit_code=code,duration_seconds=time.monotonic()-started,finished_utc=now(),status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
