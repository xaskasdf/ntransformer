"""Replay the shared-source scenarios through an untouched routed public DUT."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_peer_read import sha,save
from prepare_ntrpu_peer_shared_routed import prepare
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();prior=a.prior.resolve();out=a.output.resolve()
    reviewed=json.loads(a.review.read_text());donor=json.loads((prior/'result.json').read_text())
    assert reviewed['status']=='SHARED_ROUTED_INTERNAL_TIMING_PASS'
    assert reviewed['result_sha256']==sha(prior/'result.json') and reviewed['checkpoint_sha256']==sha(prior/'suite/routed.dcp')
    assert donor['sources']==donor['sources_after']
    assert all(sha(ROOT/n)==h for n,h in donor['sources'].items())
    assert all(sha(prior/n)==h for n,h in json.loads((prior/'FILES.json').read_text()).items())
    out.mkdir(parents=True,exist_ok=False);(out/'suite').mkdir();shutil.copyfile(a.review,out/'PHYSICAL_REVIEW.json')
    netlist=prior/'suite/routed.v';digest=sha(netlist);prepare(ROOT,out/'suite',netlist)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    cmd=['vivado','-mode','batch','-notrace','-source',str(out/'suite/run.tcl'),'-tclargs',str(out/'suite'),str(ROOT)]
    result=dict(status='RUNNING',experiment='peer_shared_routed',sources=sources,command=cmd,timeout=900,
                prior_result_sha256=reviewed['result_sha256'],netlist_sha256=digest,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0))
    save(out/'result.json',result);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    result.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,
                  sources_after={n:sha(ROOT/n) for n in sources},donor_unchanged=sha(netlist)==digest and sha(prior/'result.json')==reviewed['result_sha256'])
    result['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and result['sources_after']==sources and result['donor_unchanged'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',result)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()
                          and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:result[k] for k in ('status','seconds','returncode')}))


if __name__=='__main__':main()
