"""Bounded isolated Vivado simulation/synthesis; numerical review is separate."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    affinity=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,affinity)
    for name,sha in json.loads((a.source/'SNAPSHOT_FILES.json').read_text()).items():
        assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==sha,name
    a.output.mkdir(parents=True)
    command=['timeout','--signal=TERM','--kill-after=30s','1200s','vivado','-mode','batch','-source',
        str(a.source/'tools/ntrpu_board_kv_tile_loader_r2.tcl'),'-log',str(a.output/'vivado.log'),'-journal',str(a.output/'vivado.jou'),
        '-tclargs',str(a.output/'output')]
    now=lambda:datetime.now(timezone.utc).isoformat();started=time.monotonic()
    info=dict(stage='V7-kv-loader-vivado-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,max_threads=2,timeout_seconds=1200,part='xcvu5p-flva2104-2-i',clock_ns=5,
        command=command,status='running',scope='340 RTL cases then OOC synthesis, no DDR hardware or routed timing.')
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    with (a.output/'environment.log').open('wb') as f:subprocess.run(['vivado','-version'],stdout=f,stderr=subprocess.STDOUT,check=True)
    with (a.output/'console.log').open('wb') as f:result=subprocess.run(command,cwd=a.output,stdout=f,stderr=subprocess.STDOUT)
    info.update(exit_code=result.returncode,duration_seconds=time.monotonic()-started,finished_utc=now(),status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
