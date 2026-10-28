"""Bounded clock-tree probe of a copied checkpoint, private VM namespace only."""
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

SYNTH_SHA='4070592049040d4afc8ae4fe70c0625757bf2b7df3c7e72124079221c8e765cf'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def run(a):
    base=Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-clocktree').resolve()
    root=a.output.resolve();here=Path(__file__).resolve().parent;donor=a.donor.resolve()
    if os.name!='posix' or not root.is_relative_to(base) or root.exists():raise ValueError('New private VM output required')
    expected=a.donor_sha if a.mode=='reopen' else SYNTH_SHA
    if not expected or sha(donor)!=expected:raise ValueError('Checkpoint hash mismatch')
    names=['constrain_hbm_clocktree.tcl','insert_hbm_clocktree.tcl','audit_hbm_clocktree.tcl','audit_hbm_clocktree_topology.tcl','run_hbm_clocktree.tcl','run_hbm_clocktree.py']
    hashes={n:sha(here/n) for n in names}
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for n in names:shutil.copy2(here/n,source/n)
    shutil.copy2(donor,source/'donor.dcp')
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    limit=1200 if a.mode=='route' else 600
    args=['vivado','-mode','batch','-source',source/'run_hbm_clocktree.tcl','-journal',root/'vivado.jou','-log',root/'vivado.log','-tclargs',source,source/'donor.dcp',root/'output',a.mode]
    command='source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(list(map(str,args)))
    start=time.monotonic();expired=False
    with (root/'vivado_console.log').open('w') as log:
        p=subprocess.Popen(['bash','-c',command],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'own_process.json',dict(pid=p.pid,pgid=p.pid,timeout=limit))
        try:code=p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    error=None
    if code or expired or not (root/'output/STATUS.txt').exists():error='Incomplete clock-tree phase'
    unchanged=hashes=={n:sha(here/n) for n in names} and sha(donor)==expected and sha(source/'donor.dcp')==expected
    if not unchanged:error='Frozen input changed'
    result=dict(status='HBM_CLOCKTREE_EXPORTED' if error is None else 'FAILED',mode=a.mode,error=error,returncode=code,timed_out=expired,
                seconds=time.monotonic()-start,sources=hashes,sources_unchanged=unchanged,donor_sha256=expected,
                cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),native_hbm_pcie_ip=False,board_qualified=False)
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
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--mode',choices=('preflight','route','reopen'),required=True)
    ap.add_argument('--donor',type=Path,required=True);ap.add_argument('--donor-sha');ap.add_argument('--settings',type=Path,required=True)
    a=ap.parse_args();raise SystemExit(0 if run(a) else 1)
