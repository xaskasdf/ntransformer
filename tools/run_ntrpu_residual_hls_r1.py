"""Bounded, isolated Vitis HLS execution; terminal status is not numerical approval."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--vitis',type=Path,required=True);a=p.parse_args()
    assert os.name=='posix' and a.vitis.is_file() and not a.output.exists()
    affinity=sorted(os.sched_getaffinity(0))[:2]
    assert affinity
    os.sched_setaffinity(0,affinity)
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for name,sha in hashes.items():assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==sha,name
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir();source=out/'source';source.mkdir()
    paths=['fpga/hls/residual_fp32_r1/'+n for n in ('ntrpu_residual_fp32_r1.cpp','ntrpu_residual_fp32_r1.h','run_hls.tcl')]
    paths+=['tests/test_ntrpu_hls_residual_r1.cpp','fixture/llama32_residual_r1.bin']
    for name in paths:shutil.copyfile(a.source/name,source/Path(name).name)
    command=[str(a.vitis),'--mode','hls','--tcl','source/run_hls.tcl']
    env=os.environ.copy();env.update(NTRPU_HLS_PART='xcvu5p-flva2104-2-i',NTRPU_HLS_CLOCK_NS='5',NTRPU_HLS_STAGE='cosim')
    now=lambda:datetime.now(timezone.utc).isoformat()
    info=dict(stage='V7-residual-hls-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        command=command,timeout_seconds=2400,status='running',part=env['NTRPU_HLS_PART'],clock_ns=5,
        cpu_affinity=affinity,
        numerical_scope='Local-memory integer FP32 residual; 43 cases/51556 values, four real embedding rows with synthetic branches. No AXI, attention or full inference.')
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    start=time.monotonic()
    with (a.output/'console.log').open('wb') as log:
        proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    info.update(exit_code=code,duration_seconds=round(time.monotonic()-start,3),finished_utc=now(),status='awaiting_evidence_review')
    inv.write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)
if __name__=='__main__':main()
