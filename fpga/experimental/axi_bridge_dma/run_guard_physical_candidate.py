"""Bounded exploratory PCIe OOC implementation; preserves the source checkpoint."""
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


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def run(a):
    allowed=Path('/home/ubuntu/vivado/ntrpu-side/guard-integration').resolve()
    root=a.output.resolve();checkpoint=a.checkpoint.resolve();archive=Path(str(root)+'.tar.gz')
    if os.name!='posix' or not root.is_relative_to(allowed) or not checkpoint.is_relative_to(allowed):raise ValueError('Private integration namespace required')
    if root.exists() or archive.exists() or not a.settings.is_file():raise ValueError('New output and installed Vivado required')
    digest=sha(checkpoint);root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    here=Path(__file__).resolve().parent
    for name in ('implement_guard_candidate.tcl','run_guard_physical_candidate.py'):shutil.copy2(here/name,source/name)
    save(root/'INPUT.json',dict(checkpoint=str(checkpoint),sha256=digest))
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    started=time.monotonic();code=None;expired=False;error=None
    try:
        args=['vivado','-mode','batch','-source',str(source/'implement_guard_candidate.tcl'),'-journal',str(root/'vivado.jou'),
              '-log',str(root/'vivado.log'),'-tclargs',str(checkpoint),str(root/'output')]
        command='source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(args)
        with (root/'audit.log').open('w') as log:
            p=subprocess.Popen(['bash','-c',command],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'own_process.json',dict(pid=p.pid,pgid=p.pid,timeout=1200,command=args))
            try:code=p.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
        if code or expired or not (root/'output/STATUS.txt').exists():raise ValueError('Audit failed; inspect preserved evidence')
    except (OSError,ValueError,RuntimeError) as exc:error=str(exc)
    unchanged=sha(checkpoint)==digest
    if not unchanged:error='Input checkpoint changed'
    result=dict(status='NATIVE_PHYSICAL_CANDIDATE_EXPORTED' if error is None else 'FAILED',error=error,returncode=code,
                timed_out=expired,seconds=time.monotonic()-started,inputs_unchanged=unchanged,
                cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),physical_route_executed=(root/'output/routed.dcp').exists(),placement_executed=(root/'output/placed.dcp').exists(),board_timing_closed=False)
    save(root/'RESULT.json',result)
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(work) and '.Xil' not in p.parts]
    save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in paths})
    with tarfile.open(archive,'x:gz') as tar:
        for p in paths+[root/'FILES.json']:tar.add(p,arcname=p.relative_to(root).as_posix())
    result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps(result),flush=True)
    return error is None


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--settings',type=Path,required=True)
    ap.add_argument('--execute-native',action='store_true');a=ap.parse_args()
    if not a.execute_native:ap.error('--execute-native required')
    raise SystemExit(0 if run(a) else 1)
