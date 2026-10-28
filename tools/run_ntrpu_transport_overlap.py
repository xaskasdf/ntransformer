"""Run a bounded, isolated overlap experiment and export compact evidence."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_ntrpu_transport_study import SOURCES as BASE, digest, save

ROOT = Path(__file__).resolve().parents[1]
SOURCES = BASE + ['fpga/tb/tb_ntrpu_transport_overlap_credit.sv',
    'tools/prepare_ntrpu_transport_overlap.py', 'tools/ntrpu_transport_overlap.tcl',
    'tools/run_ntrpu_transport_overlap.py']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--variant', choices=('ahead', 'compact'), required=True)
    ap.add_argument('--timeout', type=int, default=900)
    a = ap.parse_args()
    if not 1 <= a.timeout <= 900:raise ValueError('Timeout must be 1..900')
    out = a.output.resolve();out.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[-2:];os.sched_setaffinity(0, cpus);os.nice(10)
    os.environ['OMP_NUM_THREADS'] = '2'
    record = dict(status='RUNNING', variant=a.variant, sources={n:digest(ROOT/n) for n in SOURCES},
                  cpus=cpus, nice=os.nice(0), load_before=os.getloadavg(), timeout=a.timeout)
    save(out/'result.json', record)
    command = ['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_transport_overlap.tcl'),
               '-log',str(out/'vivado.log'),'-journal',str(out/'vivado.jou'),'-tclargs',str(out/'suite'),a.variant]
    start = time.monotonic();timed_out=False
    with (out/'console.log').open('w') as log:
        job = subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    record.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=timed_out,
                  sources_after={n:digest(ROOT/n) for n in SOURCES},load_after=os.getloadavg())
    record['status']='EXPORTED_PENDING_REVIEW' if code==0 and not timed_out else 'FAILED_OR_TIMEOUT'
    if record['sources']!=record['sources_after']:record['status']='INPUTS_CHANGED'
    save(out/'result.json',record)
    for p in (out/'suite/project').rglob('*.log'):
        dest=out/'tool_logs'/p.relative_to(out/'suite/project')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'FILES.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*')
        if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    return 0 if record['status']=='EXPORTED_PENDING_REVIEW' else 1


if __name__=='__main__':raise SystemExit(main())
