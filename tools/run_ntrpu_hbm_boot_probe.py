"""Observe native HBM startup on a hash-verified cached XSIM snapshot."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['tools/run_ntrpu_hbm_boot_probe.py','tools/ntrpu_hbm_boot_progress.tcl','tools/ntrpu_hbm_apb_progress.tcl']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--donor',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--apb',action='store_true',help='Observe native initialization ROM/APB signals')
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    runtime=out/'runtime';runtime.mkdir();sources={n:sha(ROOT/n) for n in SOURCES}
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
    inputs=json.loads(a.manifest.read_text())
    r=dict(status='PREPARING',experiment='hbm_apb_progress' if a.apb else 'hbm_boot_progress',sources=sources,
           donor=str(a.donor),manifest_sha256=sha(a.manifest),cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),timeout=300)
    save(out/'result.json',r)
    for n,v in inputs.items():
        p=a.donor/n;assert p.resolve().is_relative_to(a.donor.resolve()) and sha(p)==v['sha256'],n
        dest=runtime/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==v['sha256']
    shutil.copy2(a.manifest,out/'SNAPSHOT_INPUTS.json')
    shutil.copy2(ROOT/'tools'/('ntrpu_hbm_apb_progress.tcl' if a.apb else 'ntrpu_hbm_boot_progress.tcl'),runtime/'run.tcl')
    r['status']='RUNNING';save(out/'result.json',r)
    cmd=['xsim','tb_ntrpu_hbm_boot_behav','-tclbatch','run.tcl','-log','simulate.log']
    start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    for p in runtime.rglob('*'):
        if p.is_file() and (p.suffix=='.log' or p.name in ('progress.tsv','visibility.txt','run.tcl')):
            dest=out/'evidence'/p.relative_to(runtime);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,command=cmd,
             sources_after={n:sha(ROOT/n) for n in SOURCES},donor_unchanged=all(sha(a.donor/n)==v['sha256'] for n,v in inputs.items()))
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] and r['donor_unchanged'] else 'FAILED_OR_TIMEOUT'
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts})
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
