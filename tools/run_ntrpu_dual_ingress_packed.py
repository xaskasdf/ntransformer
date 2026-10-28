"""Bounded isolated dual-input receiver simulation or synthesis, using CPUs 26/27."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['tools/run_ntrpu_dual_ingress_packed.py','tools/ntrpu_dual_ingress_packed.tcl',
 'fpga/experimental/ntrpu_dual_allocator.sv','fpga/experimental/ntrpu_dual_ingress_packed.sv',
 'fpga/tb/tb_ntrpu_dual_ingress_packed.sv']
# Hash helpers do not require imports of the complete transport runner graph.
import hashlib
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('sim','synth'),required=True);ap.add_argument('--serial',type=int,choices=(0,1),default=0)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[-4:-2] if len(cpus)>=4 else cpus[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={n:sha(ROOT/n) for n in SOURCES}
    command=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_dual_ingress_packed.tcl'),
      '-tclargs',str(out/'suite'),a.stage,str(a.serial)]
    r=dict(status='RUNNING',experiment='dual_ingress_packed',stage=a.stage,serial=a.serial,sources=sources,
           cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),load_before=os.getloadavg(),command=command,timeout=900)
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
    for p in (out/'suite/project').rglob('*.log'):
        dest=out/'tool_logs'/p.relative_to(out/'suite/project');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)


if __name__=='__main__':main()
