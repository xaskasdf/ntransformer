"""Bounded functional simulation of a reviewed routed netlist, via public ports."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_read_return import ROOT,sha,save
from prepare_ntrpu_peer_read_routed import prepare


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    prior=a.prior.resolve();physical=json.loads(a.review.read_text());donor=json.loads((prior/'result.json').read_text())
    assert physical['evidence_validated'] and physical['result_sha256']==sha(prior/'result.json')
    assert donor['sources']==donor['sources_after']
    for name,digest in json.loads((prior/'FILES.json').read_text()).items():assert sha(prior/name)==digest,name
    for name in ('fpga/tb/tb_ntrpu_peer_read.sv','fpga/experimental/ntrpu_peer_read_pkg.sv'):
        assert sha(ROOT/name)==donor['sources'][name]
    netlist=prior/'suite/routed.v';digest=sha(netlist)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'suite').mkdir()
    shutil.copyfile(a.review,out/'PHYSICAL_REVIEW.json');prepare(ROOT,out/'suite',netlist)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    command=['vivado','-mode','batch','-notrace','-source',str(out/'suite/run.tcl'),'-tclargs',str(out/'suite'),str(ROOT)]
    result=dict(status='RUNNING',experiment='peer_read_routed',sources=sources,command=command,timeout=900,
                prior_result_sha256=physical['result_sha256'],netlist_sha256=digest,
                cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0))
    save(out/'result.json',result);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    result.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,
                  sources_after={name:sha(ROOT/name) for name in sources},donor_unchanged=sha(netlist)==digest and sha(prior/'result.json')==physical['result_sha256'])
    result['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and result['sources']==result['sources_after'] and result['donor_unchanged'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',result)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:result[k] for k in ('status','seconds','returncode')}))


if __name__=='__main__':main()
