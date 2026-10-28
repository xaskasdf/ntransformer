"""Bounded post-route physical optimization and routed compact comparison."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_ntrpu_overlap_physical import SOURCES as BASE, digest, save

ROOT=Path(__file__).resolve().parents[1]
SOURCES=BASE+['tools/ntrpu_overlap_physopt.tcl','tools/ntrpu_overlap_physopt_restruct.tcl','tools/run_ntrpu_overlap_physopt.py']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('route','netlist'),required=True)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--timeout',type=int,default=900)
    ap.add_argument('--restructure',action='store_true',help='Enable combinational LUT restructuring in a second physical pass')
    a=ap.parse_args()
    if not 1<=a.timeout<=900:raise ValueError('Timeout must be 1..900')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);prior=a.prior.resolve()
    sources={n:digest(ROOT/n) for n in SOURCES};old=json.loads((prior/'result.json').read_text())
    assert old['stage']=='route' and old['variant']=='compact' and old['status']=='EXPORTED_PENDING_REVIEW'
    assert old['sources']==old['sources_after'] and all(sources[n]==old['sources'][n] for n in BASE),'Prior baseline sources changed'
    files=json.loads((prior/'FILES.json').read_text());artifact='physical_route.dcp' if a.stage=='route' else 'physical_funcsim.v'
    (out/'inputs').mkdir()
    for rel in ['suite/'+artifact,'suite/timing.rpt','suite/utilization.rpt','suite/constraints_after.xdc','result.json']:
        assert digest(prior/rel)==files[rel],'Prior hash mismatch'
        shutil.copy2(prior/rel,out/'inputs'/Path(rel).name)
    shutil.copy2(prior/'FILES.json',out/'inputs/FILES.json')
    for p in (prior/'rtl_evidence').iterdir():assert digest(p)==files['rtl_evidence/'+p.name]
    shutil.copytree(prior/'rtl_evidence',out/'rtl_evidence')
    inputs={p.name:digest(p) for p in (out/'inputs').iterdir()};rtl={p.name:digest(p) for p in (out/'rtl_evidence').iterdir()}
    cpus=sorted(os.sched_getaffinity(0))[-2:];os.sched_setaffinity(0,cpus);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    record=dict(status='RUNNING',stage=a.stage,variant='compact',experiment='post_route_physopt',sources=sources,
        restructure=a.restructure,
        timeout=a.timeout,cpus=cpus,nice=os.nice(0),load_before=os.getloadavg(),inputs=inputs,rtl_evidence=rtl,
        prior=dict(path=str(prior),stage='route',artifact=artifact,sha256=files['suite/'+artifact]))
    save(out/'result.json',record)
    script='ntrpu_overlap_physopt_restruct.tcl' if a.restructure else 'ntrpu_overlap_physopt.tcl'
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools'/script),
         '-log',str(out/'vivado.log'),'-journal',str(out/'vivado.jou'),'-tclargs',str(out/'suite'),a.stage]
    start=time.monotonic();timed_out=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    record.update(returncode=code,timed_out=timed_out,seconds=round(time.monotonic()-start,3),command=cmd,
        sources_after={n:digest(ROOT/n) for n in SOURCES},load_after=os.getloadavg(),
        inputs_after={n:digest(out/'inputs'/n) for n in inputs},rtl_evidence_after={n:digest(out/'rtl_evidence'/n) for n in rtl})
    record['status']='EXPORTED_PENDING_REVIEW' if code==0 and not timed_out else 'FAILED_OR_TIMEOUT'
    if sources!=record['sources_after'] or inputs!=record['inputs_after'] or rtl!=record['rtl_evidence_after']:record['status']='INPUTS_CHANGED'
    save(out/'result.json',record)
    for p in (out/'suite/project').rglob('*.log'):
        dest=out/'tool_logs'/p.relative_to(out/'suite/project');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'FILES.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()
        and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:record[k] for k in ('stage','status','seconds')}),flush=True)
    return 0 if record['status']=='EXPORTED_PENDING_REVIEW' else 1


if __name__=='__main__':raise SystemExit(main())
