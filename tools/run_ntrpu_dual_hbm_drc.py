"""Inspect the reviewed dual-HBM routed DCP without editing or waiving it."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_dual_hbm import ROOT,sha,save
SOURCES=['tools/run_ntrpu_dual_hbm.py','tools/prepare_ntrpu_dual_hbm.py',
         'tools/run_ntrpu_dual_hbm_drc.py','tools/ntrpu_hbm_drc_audit.tcl']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    prior=a.prior.resolve();review=json.loads(a.review.read_text());assert review['status']=='DUAL_HBM_ROUTED_INTERNAL_TIMING_PASS'
    assert review['result_sha256']==sha(prior/'result.json')
    assert all(sha(prior/n)==h for n,h in json.loads((prior/'FILES.json').read_text()).items())
    donor=prior/'suite/routed.dcp';digest=sha(donor);assert review['checkpoint_sha256']==digest
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copy2(donor,out/'input.dcp');shutil.copy2(a.review,out/'PRIOR_REVIEW.json')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={n:sha(ROOT/n) for n in SOURCES}
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_hbm_drc_audit.tcl'),'-tclargs',str(out/'input.dcp')]
    r=dict(status='RUNNING',experiment='dual_hbm_drc',sources=sources,input_sha256=digest,
           prior_result_sha256=sha(prior/'result.json'),cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),timeout=300,command=cmd)
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=300)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in SOURCES},
             donor_unchanged=sha(donor)==digest,input_unchanged=sha(out/'input.dcp')==digest)
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] and r['donor_unchanged'] and r['input_unchanged'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',r);save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}))
if __name__=='__main__':main()
