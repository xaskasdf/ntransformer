"""Bounded synthesis of the new client in the isolated HBM frontier namespace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tarfile
import time

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def run(a):
    allowed=Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-source').resolve()
    root=a.output.resolve();here=Path(__file__).resolve().parent
    if os.name!='posix' or not root.is_relative_to(allowed) or root.exists():raise ValueError('New private VM output required')
    names=['ntrpu_axi_bridge_hbm_source.sv','ntrpu_axi_bridge_hbm_client.sv','ntrpu_axi_bridge_guard_v3.sv',
           'ntrpu_axi_bridge_endpoint_client.sv','ntrpu_axi_bridge_mapper.sv','ntrpu_axi_bridge_writer.sv','ntrpu_xdma_local_master.sv',
           'synth_hbm_source.tcl','run_hbm_source_synth.py']
    hashes={n:sha(here/n) for n in names}
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for n in names:shutil.copy2(here/n,source/n)
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    args=['vivado','-mode','batch','-source',str(source/'synth_hbm_source.tcl'),'-journal',str(root/'vivado.jou'),'-log',str(root/'vivado.log'),'-tclargs',str(source),str(root/'output')]
    command='source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(args)
    started=time.monotonic();expired=False;error=None;code=None
    with (root/'synthesis.log').open('w') as log:
        p=subprocess.Popen(['bash','-c',command],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'own_process.json',dict(pid=p.pid,pgid=p.pid,timeout=600))
        try:code=p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    if code or expired or not (root/'output/STATUS.txt').exists():error='Synthesis incomplete'
    unchanged=hashes=={n:sha(here/n) for n in names}
    if not unchanged:error='Input sources changed'
    result=dict(status='HBM_SOURCE_CLIENT_SYNTH_EXPORTED' if error is None else 'FAILED',error=error,returncode=code,timed_out=expired,sources=hashes,sources_unchanged=unchanged,seconds=time.monotonic()-started,native_hbm_pcie_ip=False,physical_timing_qualified=False)
    save(root/'RESULT.json',result)
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(work) and '.Xil' not in p.parts]
    save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in files})
    archive=Path(str(root)+'.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:
        for p in files+[root/'FILES.json']:tar.add(p,arcname=p.relative_to(root).as_posix())
    result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps(result),flush=True)
    return error is None

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--settings',type=Path,required=True)
    a=ap.parse_args();raise SystemExit(0 if run(a) else 1)
