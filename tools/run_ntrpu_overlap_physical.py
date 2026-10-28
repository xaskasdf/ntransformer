"""Run one bounded compact physical stage, retaining provenance and evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_ntrpu_transport_study import SOURCES as BASE, digest, save
from prepare_ntrpu_transport_study import GENERATED

ROOT=Path(__file__).resolve().parents[1]
SOURCES=BASE+['tools/prepare_ntrpu_transport_overlap.py','tools/prepare_ntrpu_overlap_physical.py',
             'tools/ntrpu_overlap_physical.tcl','tools/run_ntrpu_overlap_physical.py']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('synth','place','route','netlist'),required=True)
    ap.add_argument('--prior',type=Path);ap.add_argument('--rtl-evidence',type=Path)
    ap.add_argument('--timeout',type=int,default=900);a=ap.parse_args()
    if not 1<=a.timeout<=900:raise ValueError('Bounded timeout in 1..900 required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources={n:digest(ROOT/n) for n in SOURCES}
    record=dict(status='RUNNING',stage=a.stage,variant='compact',sources=sources,timeout=a.timeout)
    if a.stage=='synth':
        if not a.rtl_evidence:raise ValueError('Validated compact RTL evidence required')
        evidence=a.rtl_evidence.resolve();old=json.loads((evidence/'result.json').read_text())
        assert old['variant']=='compact' and old['status']=='EXPORTED_PENDING_REVIEW' and old['sources']==old['sources_after']
        assert all(old['sources'][n]==sources[n] for n in [*BASE,'tools/prepare_ntrpu_transport_overlap.py'])
        manifest=json.loads((evidence/'FILES.json').read_text());(out/'rtl_evidence').mkdir()
        for rel in ['result.json',*[f'suite/generated/{n}.sv' for n in GENERATED]]:
            assert digest(evidence/rel)==manifest[rel],'RTL evidence hash mismatch'
            shutil.copy2(evidence/rel,out/'rtl_evidence'/Path(rel).name)
        shutil.copy2(evidence/'FILES.json',out/'rtl_evidence/FILES.json')
    else:
        if not a.prior:raise ValueError('Prior stage required')
        prior=a.prior.resolve();previous=json.loads((prior/'result.json').read_text())
        expected={'place':'synth','route':'place','netlist':'route'}[a.stage]
        assert previous['stage']==expected and previous['variant']=='compact' and previous['status']=='EXPORTED_PENDING_REVIEW'
        assert sources==previous['sources']==previous['sources_after'],'Physical stage sources changed'
        manifest=json.loads((prior/'FILES.json').read_text())
        artifact='physical_funcsim.v' if a.stage=='netlist' else 'physical_'+expected+'.dcp'
        rel='suite/'+artifact;assert digest(prior/rel)==manifest[rel]
        (out/'inputs').mkdir();shutil.copy2(prior/rel,out/'inputs'/artifact)
        for meta in ('result.json','FILES.json'):shutil.copy2(prior/meta,out/'inputs'/meta)
        for p in (prior/'rtl_evidence').iterdir():assert digest(p)==manifest['rtl_evidence/'+p.name]
        shutil.copytree(prior/'rtl_evidence',out/'rtl_evidence')
        record['prior']=dict(path=str(prior),stage=expected,artifact=artifact,sha256=manifest[rel])
    inputs={p.name:digest(p) for p in (out/'inputs').glob('*')}
    rtl={p.name:digest(p) for p in (out/'rtl_evidence').iterdir()}
    cpus=sorted(os.sched_getaffinity(0))[-2:];os.sched_setaffinity(0,cpus);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    record.update(cpus=cpus,nice=os.nice(0),load_before=os.getloadavg(),inputs=inputs,rtl_evidence=rtl)
    save(out/'result.json',record)
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_overlap_physical.tcl'),
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
