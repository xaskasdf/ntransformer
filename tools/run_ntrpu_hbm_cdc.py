"""Run bounded isolated native HBM boot and asynchronous transport experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from run_ntrpu_hbm_transport import SOURCES as TRANSPORT_SOURCES

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(TRANSPORT_SOURCES+[
    'tools/run_ntrpu_hbm_cdc.py','tools/ntrpu_hbm_cdc.tcl',
    'tools/ntrpu_hbm_cdc_async.tcl','tools/prepare_ntrpu_hbm_native_link.py',
    'tools/prepare_ntrpu_hbm_cdc.py','fpga/rtl/ntrpu_hbm_startup.sv',
    'fpga/rtl/ntrpu_hbm_transport_async.sv','fpga/tb/ntrpu_hbm_cdc_cases.svh']))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('boot','async'),required=True);ap.add_argument('--timeout',type=int,default=900)
    ap.add_argument('--reference',type=Path)
    a=ap.parse_args();assert 1<=a.timeout<=900
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=SOURCES
    hashes={n:digest(ROOT/n) for n in names}
    cpus=sorted(os.sched_getaffinity(0))[-2:];os.sched_setaffinity(0,cpus);os.nice(10)
    os.environ['OMP_NUM_THREADS']='2'
    r=dict(status='RUNNING',experiment='hbm_cdc',stage=a.stage,sources=hashes,cpus=cpus,nice=os.nice(0),load_before=os.getloadavg(),timeout=a.timeout)
    save(out/'result.json',r)
    if a.stage=='async':
        assert a.reference
        (out/'suite').mkdir()
        subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_hbm_cdc.py'),
                        str(out/'suite'),'--stage','async','--reference',str(a.reference)],check=True)
    generated=lambda:{p.relative_to(out).as_posix():digest(p) for p in (out/'suite/generated').glob('*') if p.is_file()}
    r['generated']=generated()
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_hbm_cdc.tcl'),
         '-log',str(out/'vivado.log'),'-journal',str(out/'vivado.jou'),'-tclargs',str(out/'suite'),a.stage]
    start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(returncode=code,timed_out=expired,seconds=round(time.monotonic()-start,3),sources_after={n:digest(ROOT/n) for n in names},command=cmd,generated_after=generated())
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired else 'FAILED_OR_TIMEOUT'
    if r['sources']!=r['sources_after'] or r['generated']!=r['generated_after']:r['status']='INPUTS_CHANGED'
    save(out/'result.json',r)
    for p in (out/'suite/project').rglob('*.log'):
        dest=out/'tool_logs'/p.relative_to(out/'suite/project');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'FILES.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()
                          and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('stage','status','seconds')}),flush=True)
    return 0 if r['status']=='EXPORTED_PENDING_REVIEW' else 1


if __name__=='__main__':raise SystemExit(main())
