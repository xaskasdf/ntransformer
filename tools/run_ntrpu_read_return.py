"""Bounded private simulation of the full-frame HBM read return."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_dual_hbm import ROOT,SOURCES as WRITE_SOURCES,sha,save
from prepare_ntrpu_read_return import prepare
SOURCES=list(dict.fromkeys(WRITE_SOURCES+['tools/run_ntrpu_read_return.py','tools/prepare_ntrpu_read_return.py',
    'tools/ntrpu_read_return.tcl','fpga/tb/tb_ntrpu_read_return.sv']+
    ['fpga/experimental/'+n+'.sv' for n in ('ntrpu_hbm_read_frame','ntrpu_hbm_read_adapter','ntrpu_hbm_read_return')]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--stub',type=Path,required=True)
    ap.add_argument('--common',type=int,choices=(0,1),default=0);ap.add_argument('--native-profile',type=int,choices=(0,1),default=0);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);suite=out/'suite';suite.mkdir()
    prepare(a.binding,suite/'ntrpu_read_return_binding.svh');shutil.copy2(a.binding,suite/'native_binding.sv');shutil.copy2(a.stub,suite/'native_stub.sv')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={n:sha(ROOT/n) for n in SOURCES};command=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_read_return.tcl'),'-tclargs',str(suite),str(a.common),str(a.native_profile)]
    r=dict(status='RUNNING',experiment='read_return',common=a.common,native_profile=a.native_profile,sources=sources,
           binding_sha256=sha(a.binding),stub_sha256=sha(a.stub),cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),load_before=os.getloadavg(),command=command,timeout=900)
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in SOURCES})
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and sources==r['sources_after'] else 'FAILED_OR_CHANGED'
    for p in (suite/'project').rglob('*.log'):
        dest=out/'tool_logs'/p.relative_to(suite/'project');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'result.json',r);save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)
if __name__=='__main__':main()
