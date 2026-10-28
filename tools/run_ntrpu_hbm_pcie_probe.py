"""Generate two private VU47P IP profiles with bounded resources and exported evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10)
    os.environ['OMP_NUM_THREADS']='2'
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    command=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_hbm_pcie_probe.tcl'),'-tclargs',str(out/'suite')]
    r=dict(status='RUNNING',sources=sources,command=command,timeout=900,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0))
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        j=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=j.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(j.pid,signal.SIGTERM)
            try:code=j.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(j.pid,signal.SIGKILL);code=j.wait()
    # Export generated XCI, wrappers, constraints and the AMD example sources.
    # Project caches remain remote; every exported byte has a manifest entry.
    for p in (out/'suite').rglob('*'):
        if not p.is_file() or any(s.startswith('.') for s in p.relative_to(out/'suite').parts):continue
        if p.suffix.lower() in ('.xci','.v','.sv','.vh','.svh','.xdc','.tcl','.tsv','.rpt','.xpr'):
            target=out/'export'/p.relative_to(out/'suite');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    r.update(returncode=code,timed_out=expired,seconds=round(time.monotonic()-start,3),sources_after={n:sha(ROOT/n) for n in sources})
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.relative_to(out).parts[0]!='suite' and not any(s.startswith('.') for s in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)

if __name__=='__main__':main()
