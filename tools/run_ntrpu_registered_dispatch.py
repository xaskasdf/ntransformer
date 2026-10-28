"""Simulate and route one isolated registered-head R4 composition with bounded resources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['fpga/rtl/ntrpu_buffer_pool_allocator.sv', 'fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv', 'fpga/rtl/ntrpu_rv_elastic.sv', 'fpga/rtl/ntrpu_destination_dispatch.sv', 'fpga/rtl/ntrpu_reserved_dispatch.sv', 'fpga/rtl/ntrpu_peer_multibuffer_fillstage.sv', 'fpga/rtl/ntrpu_peer_multibuffer_elastic.sv', 'fpga/tb/tb_ntrpu_reserved_dispatch.sv', 'fpga/constraints/ntrpu_multibuffer_ooc.xdc', 'tools/prepare_ntrpu_reserved_receiver.py', 'tools/ntrpu_reserved_dispatch.tcl', 'tools/run_ntrpu_reserved_dispatch.py', 'fpga/rtl/ntrpu_registered_dispatch.sv', 'fpga/constraints/ntrpu_multibuffer_physical_ooc.xdc', 'tools/prepare_ntrpu_registered_dispatch.py', 'tools/ntrpu_registered_dispatch.tcl', 'tools/run_ntrpu_registered_dispatch.py', 'tools/ntrpu_registered_route.tcl']


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--timeout',type=int,default=900)
    ap.add_argument('--simulation-only',action='store_true',help='Repeat instrumentation without resynthesizing unchanged RTL')
    ap.add_argument('--resume-synth',type=Path,help='Resume physical implementation from this verified prior suite directory')
    ap.add_argument('--threads',type=int,choices=(2,4,8),default=2)
    ap.add_argument('--resume-stage',choices=('synth','physopt'),default='synth')
    args=ap.parse_args()
    if not 1<=args.timeout<=1800:raise ValueError('bounded timeout required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if args.resume_synth and args.simulation_only:raise ValueError('incompatible modes')
    cpus=sorted(os.sched_getaffinity(0))[-args.threads:];os.sched_setaffinity(0,cpus);os.nice(10)
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    before={n:digest(ROOT/n) for n in SOURCES}
    record=dict(status='RUNNING',sources=before,cpus=cpus,nice=os.nice(0),
                load_before=os.getloadavg(),timeout=args.timeout,simulation_only=args.simulation_only)
    save(output/'result.json',record)
    cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_registered_dispatch.tcl'),
         '-log',str(output/'vivado.log'),'-journal',str(output/'vivado.jou'),'-tclargs',str(output/'suite')]
    if args.simulation_only:cmd.append('simulation-only')
    if args.resume_synth:
        prior=args.resume_synth.resolve();previous=json.loads((prior.parent/'result.json').read_text())
        logic=[n for n in SOURCES if n.startswith('fpga/') or n in ('tools/prepare_ntrpu_reserved_receiver.py','tools/prepare_ntrpu_registered_dispatch.py')]
        if not all(previous['sources'][n]==before[n] for n in logic):raise ValueError('checkpoint inputs changed')
        if 'RS SUITE_PASS cases=8 ' not in (prior/'simulation.log').read_text():raise ValueError('prior simulation incomplete')
        suite=output/'suite';suite.mkdir()
        for name in ('simulation.log','utilization.rpt','timing_synth.rpt','structure.txt','registered_dispatch_synth.dcp'):
            shutil.copy2(prior/name,suite/name)
        if args.resume_stage=='physopt':
            for name in ('registered_dispatch_place.dcp','registered_dispatch_physopt.dcp','timing_place.rpt','timing_physopt.rpt'):
                shutil.copy2(prior/name,suite/name)
        shutil.copytree(prior/'generated',suite/'generated')
        checkpoint='registered_dispatch_'+args.resume_stage+'.dcp'
        record['resume']=dict(suite=str(prior),sources=previous['sources'],stage=args.resume_stage,
                              checkpoint_name=checkpoint,checkpoint_sha256=digest(prior/checkpoint))
        cmd=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_registered_route.tcl'),
             '-log',str(output/'vivado.log'),'-journal',str(output/'vivado.jou'),'-tclargs',str(suite),str(args.threads),args.resume_stage]
    start=time.monotonic()
    with (output/'console.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timeout=False
        try:code=p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    record.update(returncode=code,timed_out=timeout,seconds=round(time.monotonic()-start,3),command=cmd,
                  sources_after={n:digest(ROOT/n) for n in SOURCES},load_after=os.getloadavg())
    record['status']='EXPORTED_PENDING_REVIEW' if code==0 and not timeout else 'FAILED_OR_TIMEOUT'
    if before!=record['sources_after']:record['status']='SOURCES_CHANGED'
    save(output/'result.json',record)
    manifest={p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}
    save(output/'FILES.json',manifest)
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    save(output/'VERIFIED.json',dict(sha256=digest(archive),crc='PASS',hashes='PASS'))
    print(json.dumps(dict(status=record['status'],seconds=record['seconds'],sha256=digest(archive))),flush=True)
    return 0 if record['status']=='EXPORTED_PENDING_REVIEW' else 1


if __name__=='__main__':raise SystemExit(main())
