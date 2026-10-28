"""Inspect the reviewed dual-HBM routed DCP without editing or waiving it."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
SOURCES=['tools/run_ntrpu_peer_read_drc.py','tools/ntrpu_hbm_drc_audit.tcl']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True)
    ap.add_argument('--timeout',type=int,default=420);a=ap.parse_args();assert 0<a.timeout<=900
    prior=a.prior.resolve();review=json.loads(a.review.read_text());assert review['status'] in ('PEER_READ_ROUTED_INTERNAL_TIMING_PASS','PEER_READ_ROUTED_INTERNAL_TIMING_OPEN')
    assert review['result_sha256']==sha(prior/'result.json')
    assert all(sha(prior/n)==h for n,h in json.loads((prior/'FILES.json').read_text()).items())
    donor=prior/'suite/routed.dcp';digest=sha(donor);assert review['checkpoint_sha256']==digest
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copy2(donor,out/'input.dcp');shutil.copy2(a.review,out/'PRIOR_REVIEW.json')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={n:sha(ROOT/n) for n in SOURCES}
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_hbm_drc_audit.tcl'),'-tclargs',str(out/'input.dcp')]
    r=dict(status='RUNNING',experiment='peer_read_drc',sources=sources,input_sha256=digest,
           prior_result_sha256=sha(prior/'result.json'),cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),timeout=a.timeout,command=cmd)
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=a.timeout)
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
