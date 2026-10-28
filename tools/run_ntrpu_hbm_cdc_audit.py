"""Synthesize and report CDC for an immutable, functionally accepted composition."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_hbm_cdc import SOURCES as FROZEN_SOURCES
ROOT=Path(__file__).resolve().parents[1]
SOURCES=FROZEN_SOURCES+['tools/run_ntrpu_hbm_cdc_audit.py','tools/ntrpu_hbm_cdc_audit.tcl']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--simulation',type=Path,required=True);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    hashes={n:sha(ROOT/n) for n in SOURCES};old=json.loads((a.simulation/'result.json').read_text())
    review=json.loads(a.review.read_text());files=json.loads((a.simulation/'FILES.json').read_text())
    assert review['status']=='HBM_ASYNC_FUNCTIONAL_PASS'
    assert old['status']=='EXPORTED_PENDING_REVIEW' and old['sources']==old['sources_after']
    assert all(hashes[n]==h for n,h in old['sources'].items())
    assert old['generated']==old['generated_after']
    for n,h in old['generated'].items():
        src=a.simulation/n;assert sha(src)==h==files[n]
        dest=out/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    shutil.copy2(a.review,out/'FUNCTIONAL_REVIEW.json')
    generated=lambda:{p.relative_to(out).as_posix():sha(p) for p in (out/'suite/generated').iterdir() if p.is_file()}
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    r=dict(status='RUNNING',experiment='hbm_cdc_audit',sources=hashes,generated=generated(),
           simulation=str(a.simulation),simulation_result_sha256=sha(a.simulation/'result.json'),
           functional_review_sha256=sha(a.review),cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),timeout=900)
    save(out/'result.json',r)
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_hbm_cdc_audit.tcl'),
         '-log',str(out/'vivado.log'),'-journal',str(out/'vivado.jou'),'-tclargs',str(out/'suite')]
    start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,command=cmd,
             sources_after={n:sha(ROOT/n) for n in SOURCES},generated_after=generated())
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] and r['generated']==r['generated_after'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',r)
    for p in (out/'suite/project').rglob('*.log'):
        dest=out/'tool_logs'/p.relative_to(out/'suite/project');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds')}),flush=True)

if __name__=='__main__':main()
