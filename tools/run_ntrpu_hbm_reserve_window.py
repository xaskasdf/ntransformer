"""Repeat the identical native reserve snapshot with an explicitly extended window."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_hbm_reserve_native import ROOT,SOURCES,TOP,checked_files
from run_ntrpu_hbm_native_transport import SIMDIR,sha,save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);prior=a.prior.resolve();checked_files(prior)
    review=json.loads(a.review.read_text());previous=json.loads((prior/'result.json').read_text())
    assert review['status']=='NATIVE_RESERVE_SNAPSHOT_REVIEWED' and review['result_sha256']==sha(prior/'result.json')
    baseline={n:sha(ROOT/n) for n in SOURCES};assert previous['sources']==previous['sources_after']==baseline
    assert review['snapshot_manifest_sha256']==sha(prior/'SNAPSHOT_INPUTS.json')
    sources=dict(baseline);sources['tools/run_ntrpu_hbm_reserve_window.py']=sha(Path(__file__))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    donor=prior/SIMDIR;runtime=out/'runtime';runtime.mkdir()
    inputs=json.loads((prior/'SNAPSHOT_INPUTS.json').read_text())
    for n,v in inputs.items():
        assert (donor/n).resolve().is_relative_to(donor.resolve()) and sha(donor/n)==v['sha256']
        dest=runtime/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor/n,dest);assert sha(dest)==v['sha256']
    tcl=(ROOT/'tools/ntrpu_hbm_native_transport_run.tcl').read_text().replace('tb_ntrpu_hbm_native_transport',TOP)
    assert tcl.count('850000')==1
    tcl=tcl.replace('850000','1050000');(runtime/'run.tcl').write_text(tcl)
    shutil.copy2(a.review,out/'PRIOR_REVIEW.json');shutil.copy2(prior/'SNAPSHOT_INPUTS.json',out/'SNAPSHOT_INPUTS.json')
    cmd=['xsim',TOP+'_behav','-tclbatch','run.tcl','-log','simulate.log']
    r=dict(status='RUNNING',experiment='hbm_reserve_native',stage='traffic',operation='extended_window_same_snapshot',
           sources=sources,prior_result_sha256=sha(prior/'result.json'),timeout=1080,tcl_guard_seconds=1050,
           reason='850-second Tcl window ended before second native readback completed; no RTL or bench change.',
           cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),command=cmd,load_before=os.getloadavg())
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=1080)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,
             sources_after={n:sha(ROOT/n) for n in sources},
             donor_unchanged=all(sha(donor/n)==v['sha256'] for n,v in inputs.items()),
             runtime_inputs_unchanged=all(sha(runtime/n)==v['sha256'] for n,v in inputs.items()))
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] and r['donor_unchanged'] and r['runtime_inputs_unchanged'] else 'FAILED_OR_CHANGED'
    for n in ('simulate.log','run.tcl','progress.tsv'):
        dest=out/'evidence'/n;dest.parent.mkdir(exist_ok=True)
        if (runtime/n).exists():shutil.copy2(runtime/n,dest)
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)


if __name__=='__main__':main()
