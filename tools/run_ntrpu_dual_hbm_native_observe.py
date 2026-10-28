"""Observe public startup signals using the identical reviewed native snapshot."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_dual_hbm import ROOT,sha,save
SOURCES=['tools/run_ntrpu_dual_hbm.py','tools/prepare_ntrpu_dual_hbm.py','tools/run_ntrpu_dual_hbm_native_observe.py','tools/ntrpu_dual_hbm_native_observe.tcl']
SIMDIR=Path('suite/project/hbm_adapter.sim/sim_1/behav/xsim')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    prior=a.prior.resolve();pr=json.loads(a.review.read_text());assert pr['status']=='DUAL_HBM_NATIVE_SNAPSHOT_REVIEWED'
    assert pr['result_sha256']==sha(prior/'result.json') and pr['snapshot_manifest_sha256']==sha(prior/'SNAPSHOT_INPUTS.json')
    assert all(sha(prior/n)==h for n,h in json.loads((prior/'FILES.json').read_text()).items())
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);cwd=out/'runtime';cwd.mkdir();donor=prior/SIMDIR
    inputs=json.loads((prior/'SNAPSHOT_INPUTS.json').read_text())
    for n,v in inputs.items():
        assert (donor/n).resolve().is_relative_to(donor.resolve()) and sha(donor/n)==v['sha256']
        p=cwd/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor/n,p);assert sha(p)==v['sha256']
    shutil.copy2(prior/'SNAPSHOT_INPUTS.json',out/'SNAPSHOT_INPUTS.json');shutil.copy2(a.review,out/'PRIOR_REVIEW.json')
    shutil.copy2(ROOT/'tools/ntrpu_dual_hbm_native_observe.tcl',cwd/'run.tcl')
    sources={n:sha(ROOT/n) for n in SOURCES}
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2';os.environ['RDI_DATADIR']=str(Path(os.environ['XILINX_VIVADO'])/'data')
    cmd=['xsim','tb_ntrpu_dual_hbm_behav','-tclbatch','run.tcl','-log','simulate.log']
    r=dict(status='RUNNING',experiment='dual_hbm_native_observe',sources=sources,command=cmd,timeout=1200,
           cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),prior_result_sha256=sha(prior/'result.json'))
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in SOURCES},
             donor_unchanged=all(sha(donor/n)==v['sha256'] for n,v in inputs.items()),runtime_inputs_unchanged=all(sha(cwd/n)==v['sha256'] for n,v in inputs.items()))
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and sources==r['sources_after'] and r['donor_unchanged'] and r['runtime_inputs_unchanged'] else 'FAILED_OR_CHANGED'
    for n in ('run.tcl','simulate.log','progress.tsv'):
        if (cwd/n).exists():
            p=out/'evidence'/n;p.parent.mkdir(exist_ok=True);shutil.copy2(cwd/n,p)
    save(out/'result.json',r);save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)
if __name__=='__main__':main()
