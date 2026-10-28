"""Bounded isolated dual-receiver CDC test with real AMD conversion IP."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from prepare_ntrpu_dual_hbm import prepare
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['tools/run_ntrpu_dual_hbm.py','tools/prepare_ntrpu_dual_hbm.py','tools/ntrpu_dual_hbm.tcl',
         'tools/ntrpu_hbm_adapter.tcl','fpga/tb/tb_ntrpu_dual_hbm.sv']
SOURCES+=['fpga/rtl/'+n+'.sv' for n in ('ntrpu_rv_elastic','ntrpu_axi_write_capacity_queue','ntrpu_hbm_write_adapter','ntrpu_hbm_startup')]
SOURCES+=['fpga/experimental/'+n+'.sv' for n in ('ntrpu_locality_allocator','ntrpu_dual_ingress_locality','ntrpu_dual_hbm_async')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--stub',type=Path,required=True)
    ap.add_argument('--prefer-local',type=int,choices=(0,1),default=1)
    ap.add_argument('--native-profile',type=int,choices=(0,1),default=0)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);suite=out/'suite';suite.mkdir()
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    prepare(a.binding,suite/'ntrpu_dual_hbm_binding.svh');shutil.copy2(a.stub,suite/'native_stub.sv');shutil.copy2(a.binding,suite/'native_binding.sv')
    sources={n:sha(ROOT/n) for n in SOURCES}
    command=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_dual_hbm.tcl'),'-tclargs',str(suite),str(a.prefer_local),str(a.native_profile)]
    r=dict(status='RUNNING',experiment='dual_hbm_async',prefer_local=a.prefer_local,native_profile=a.native_profile,sources=sources,
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
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)


if __name__=='__main__':main()
