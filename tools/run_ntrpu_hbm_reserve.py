"""Bounded reserve-pressure preflight; never qualifies the native HBM model."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from run_ntrpu_hbm_native_transport import ROOT, SOURCES as BASE_SOURCES, sha, save

SOURCES = list(dict.fromkeys(BASE_SOURCES + ['tools/run_ntrpu_hbm_reserve.py',
    'tools/prepare_ntrpu_hbm_reserve.py', 'fpga/tb/ntrpu_hbm_reserve_body.svh']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--beats', type=int, choices=(8,64), default=8)
    a = ap.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    os.nice(10); os.environ['OMP_NUM_THREADS'] = '2'
    sources = {n: sha(ROOT/n) for n in SOURCES}
    # Derive an isolated Tcl without modifying either approved source or DUT.
    tcl = (ROOT/'tools/ntrpu_hbm_native_transport.tcl').read_text()
    tcl = tcl.replace('set root [file normalize [file join [file dirname [info script]] ..]]',
                      'set root {'+str(ROOT)+'}')
    tcl = tcl.replace('prepare_ntrpu_hbm_native_transport.py', 'prepare_ntrpu_hbm_reserve.py')
    tcl = tcl.replace('prepare_ntrpu_hbm_reserve.py] $out $reference',
                      f'prepare_ntrpu_hbm_reserve.py] $out $reference --beats {a.beats}')
    tcl = tcl.replace('tb_ntrpu_hbm_native_transport', 'tb_ntrpu_hbm_reserve')
    tcl = tcl.replace('NC ALL_PASS cases=4 ', 'NR ALL_PASS cases=2 ')
    (out/'run.tcl').write_text(tcl)
    cmd = ['vivado', '-mode', 'batch', '-notrace', '-source', str(out/'run.tcl'),
           '-log', 'vivado.log', '-journal', 'vivado.jou', '-tclargs', str(out/'suite'), str(a.reference), 'preflight']
    r = dict(status='RUNNING', experiment='hbm_reserve', stage='preflight', native_hbm=False,frame_beats=a.beats,
             sources=sources, command=cmd, cpus=sorted(os.sched_getaffinity(0)), nice=os.nice(0),
             load_before=os.getloadavg(), timeout=900)
    save(out/'result.json', r); start=time.monotonic(); expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try: code=job.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,
             sources_after={n:sha(ROOT/n) for n in SOURCES})
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and sources==r['sources_after'] else 'FAILED_OR_CHANGED'
    for p in (out/'suite/project').rglob('*.log'):
        dest=out/'tool_logs'/p.relative_to(out/'suite/project');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()
         and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)
    return 0 if r['status']=='EXPORTED_PENDING_REVIEW' else 1


if __name__=='__main__': raise SystemExit(main())
