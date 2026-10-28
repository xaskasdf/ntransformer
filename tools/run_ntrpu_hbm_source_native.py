"""Bounded private native gate, with immutable donor hashes and small exports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n')


def run(out, timeout):
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2]);os.nice(10)
    os.environ['OMP_NUM_THREADS']='2'
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    seed=Path((ROOT/'evidence/seed.txt').read_text().strip())
    expected=json.loads((ROOT/'evidence/seed_files.json').read_text())
    if any(not (seed/n).is_file() or sha(seed/n)!=h for n,h in expected.items()):
        raise ValueError('Seed differs from locally verified generation')
    cache=None;cache_files=None
    if (ROOT/'evidence/cache.txt').exists():
        cache=Path((ROOT/'evidence/cache.txt').read_text().strip())
        cache_files=json.loads((ROOT/'evidence/cache_files.json').read_text())
        if any(not (cache/n).is_file() or sha(cache/n)!=h for n,h in cache_files.items()):
            raise ValueError('Private compiled cache changed')
        # Copy preserves mtimes for incremental compilation; never open the donor.
        shutil.copytree(cache,out/'work/example',copy_function=shutil.copy2)
    command=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_hbm_source_native.tcl'),'-tclargs',str(out/'reports')]
    result=dict(status='RUNNING',sources=sources,seed_files=expected,command=command,timeout=timeout,cpus=sorted(os.sched_getaffinity(0)),nice=10,cache_files=cache_files)
    save(out/'result.json',result);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        p=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(p.pid,signal.SIGTERM)
            try:rc=p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
    result.update(returncode=rc,timed_out=expired,seconds=round(time.monotonic()-start,3),sources_after={n:sha(ROOT/n) for n in sources},seed_unchanged=all(sha(seed/n)==h for n,h in expected.items()))
    result['cache_unchanged']=cache is None or all(sha(cache/n)==h for n,h in cache_files.items())
    # Export the exact compiled HDL/header/XCI and simulator logs, never the cache.
    work=out/'work'
    for p in work.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.sv','.v','.vh','.svh','.xci','.xdc','.log','.prj','.tcl','.json') and not any(x.startswith('.') for x in p.relative_to(work).parts):
            dest=out/'export'/p.relative_to(work);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    result['native_hbm_enabled']=True
    result['host_exclusion_enforced']=False
    result['status']='PENDING_REVIEW' if rc==0 and not expired and result['seed_unchanged'] and result['cache_unchanged'] and sources==result['sources_after'] else 'FAILED'
    save(out/'result.json',result)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.relative_to(out).parts[0]!='work' and not any(x.startswith('.') for x in p.relative_to(out).parts) and p.name!='FILES.json'})
    print(json.dumps({k:result[k] for k in ('status','seconds','returncode')}),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--timeout',type=int,required=True)
    a=ap.parse_args();r=run(a.output.resolve(),a.timeout);raise SystemExit(0 if r['status']=='PENDING_REVIEW' else 1)
