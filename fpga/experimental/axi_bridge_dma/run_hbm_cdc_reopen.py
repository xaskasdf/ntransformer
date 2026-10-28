"""Bounded read-only audit of a pinned routed CDC checkpoint."""
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
    allowed=Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-cdc-physical').resolve()
    root=a.output.resolve();here=Path(__file__).resolve().parent
    if os.name!='posix' or not root.is_relative_to(allowed) or root.exists():raise ValueError('New private VM output required')
    names=['constrain_hbm_cdc_physical.tcl','audit_hbm_cdc_physical.tcl','audit_hbm_cdc_topology.tcl','reopen_hbm_cdc_physical.tcl','run_hbm_cdc_reopen.py']
    donor=a.donor.resolve()
    if sha(donor)!=a.expected_donor:raise ValueError('Frozen CDC DCP mismatch')
    hashes={n:sha(here/n) for n in names}
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for n in names:shutil.copy2(here/n,source/n)
    shutil.copy2(donor,source/'donor.dcp')
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    args=['vivado','-mode','batch','-source',str(source/'reopen_hbm_cdc_physical.tcl'),'-journal',str(root/'vivado.jou'),'-log',str(root/'vivado.log'),'-tclargs',str(source),str(source/'donor.dcp'),str(root/'output')]
    command='source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(args)
    started=time.monotonic();expired=False;error=None;code=None
    with (root/'physical.log').open('w') as log:
        p=subprocess.Popen(['bash','-c',command],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'own_process.json',dict(pid=p.pid,pgid=p.pid,timeout=600))
        try:code=p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    if code or expired or not (root/'output/STATUS.txt').exists():error='Physical run incomplete'
    unchanged=hashes=={n:sha(here/n) for n in names}
    if not unchanged:error='Input sources changed'
    result=dict(status='HBM_CDC_REOPEN_EXPORTED' if error is None else 'FAILED',error=error,returncode=code,timed_out=expired,donor_sha256=sha(donor),sources=hashes,sources_unchanged=unchanged,seconds=time.monotonic()-started,native_hbm_pcie_ip=False,physical_timing_qualified=False)
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
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--settings',type=Path,required=True);ap.add_argument('--donor',type=Path,required=True);ap.add_argument('--expected-donor',required=True)
    a=ap.parse_args();raise SystemExit(0 if run(a) else 1)
