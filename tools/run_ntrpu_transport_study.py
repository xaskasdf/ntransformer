"""Run one bounded performance/physical stage and retain compact verified evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_ntrpu_candidate_transport import SOURCES as BASE_SOURCES, digest, save

ROOT=Path(__file__).resolve().parents[1]
SOURCES=BASE_SOURCES+['tools/prepare_ntrpu_candidate_netlist.py',
    'tools/prepare_ntrpu_transport_study.py','tools/ntrpu_transport_study.tcl','tools/run_ntrpu_transport_study.py']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('perf','synth','place','route','netlist'),required=True)
    ap.add_argument('--prior',type=Path);ap.add_argument('--timeout',type=int,default=900)
    args=ap.parse_args()
    if not 1<=args.timeout<=900:raise ValueError('Bounded timeout in 1..900 required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources={n:digest(ROOT/n) for n in SOURCES}
    record=dict(status='RUNNING',stage=args.stage,sources=sources,timeout=args.timeout)
    inputs={}
    if args.stage in ('place','route','netlist'):
        if not args.prior:raise ValueError('Prior stage required')
        prior=args.prior.resolve();previous=json.loads((prior/'result.json').read_text())
        expected={'place':'synth','route':'place','netlist':'route'}[args.stage]
        if previous['stage']!=expected or previous['status']!='EXPORTED_PENDING_REVIEW' or previous['sources']!=previous['sources_after']:
            raise ValueError('Incomplete prior stage')
        logic=[n for n in sources if n.startswith('fpga/rtl/') or n.startswith('tools/prepare_')]
        if any(sources[n]!=previous['sources'][n] for n in logic):raise ValueError('Prior logic changed')
        manifest=json.loads((prior/'FILES.json').read_text())
        name='physical_funcsim.v' if args.stage=='netlist' else 'physical_'+expected+'.dcp'
        rel='suite/'+name
        if digest(prior/rel)!=manifest[rel]:raise ValueError('Prior artifact hash mismatch')
        (out/'inputs').mkdir();shutil.copy2(prior/rel,out/'inputs'/name)
        for meta in ('result.json','FILES.json'):shutil.copy2(prior/meta,out/'inputs'/meta)
        inputs={p.name:digest(p) for p in (out/'inputs').iterdir()}
        record['prior']=dict(path=str(prior),stage=expected,artifact=name,sha256=manifest[rel])
    cpus=sorted(os.sched_getaffinity(0))[-2:];os.sched_setaffinity(0,cpus);os.nice(10)
    os.environ['OMP_NUM_THREADS']='2'
    record.update(cpus=cpus,nice=os.nice(0),load_before=os.getloadavg(),inputs=inputs)
    save(out/'result.json',record)
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_transport_study.tcl'),
         '-log',str(out/'vivado.log'),'-journal',str(out/'vivado.jou'),'-tclargs',str(out/'suite'),args.stage]
    start=time.monotonic();timed_out=False
    with (out/'console.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    record.update(returncode=code,timed_out=timed_out,seconds=round(time.monotonic()-start,3),command=cmd,
                  sources_after={n:digest(ROOT/n) for n in SOURCES},load_after=os.getloadavg(),
                  inputs_after={n:digest(out/'inputs'/n) for n in inputs})
    record['status']='EXPORTED_PENDING_REVIEW' if code==0 and not timed_out else 'FAILED_OR_TIMEOUT'
    if record['sources']!=record['sources_after'] or inputs!=record['inputs_after']:record['status']='INPUTS_CHANGED'
    save(out/'result.json',record)
    for path in (out/'suite/project').rglob('*.log'):
        target=out/'tool_logs'/path.relative_to(out/'suite/project')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    manifest={}
    for p in out.rglob('*'):
        rel=p.relative_to(out)
        if p.is_file() and 'project' not in rel.parts and not any(n.startswith('.') for n in rel.parts):manifest[rel.as_posix()]=digest(p)
    save(out/'FILES.json',manifest)
    print(json.dumps({k:record[k] for k in ('stage','status','seconds')}),flush=True)
    return 0 if record['status']=='EXPORTED_PENDING_REVIEW' else 1


if __name__=='__main__':raise SystemExit(main())
