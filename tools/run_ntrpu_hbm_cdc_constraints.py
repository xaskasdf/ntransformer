"""Audit targeted status CDC constraints on a frozen synthesis checkpoint."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['tools/run_ntrpu_hbm_cdc_constraints.py','tools/ntrpu_hbm_cdc_constraints.tcl','fpga/constraints/ntrpu_hbm_startup_cdc.xdc']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--prior',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((a.prior/'result.json').read_text());files=json.loads((a.prior/'FILES.json').read_text())
    assert prior['experiment']=='hbm_cdc_audit' and prior['status']=='EXPORTED_PENDING_REVIEW'
    assert prior['sources']==prior['sources_after'] and prior['generated']==prior['generated_after']
    source=a.prior/'suite/async_synth.dcp';assert sha(source)==files['suite/async_synth.dcp']
    dest=out/'input.dcp';shutil.copy2(source,dest);assert sha(dest)==sha(source)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
    r=dict(status='RUNNING',experiment='hbm_cdc_constraints',sources={n:sha(ROOT/n) for n in SOURCES},
           prior=str(a.prior),prior_result_sha256=sha(a.prior/'result.json'),checkpoint_sha256=sha(source),timeout=300,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0))
    save(out/'result.json',r)
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_hbm_cdc_constraints.tcl'),'-log',str(out/'vivado.log'),'-journal',str(out/'vivado.jou'),'-tclargs',str(out),str(dest)]
    start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    r.update(returncode=code,timed_out=expired,seconds=round(time.monotonic()-start,3),sources_after={n:sha(ROOT/n) for n in SOURCES},input_after=sha(dest),command=cmd)
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] and r['input_after']==r['checkpoint_sha256'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds')}),flush=True)
if __name__=='__main__':main()
