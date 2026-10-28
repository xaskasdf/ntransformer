"""Compile the four-lane hierarchy with signatures only, without executing arithmetic."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    allowed=sorted(os.sched_getaffinity(0));affinity=allowed[2:4] if len(allowed)>=4 else allowed[:2];os.sched_setaffinity(0,affinity)
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for n,h in hashes.items():assert hashlib.sha256((a.source/n).read_bytes()).hexdigest()==h,n
    plan=json.loads((a.source/'build/cloud-setup/20261002-board/attention-ffn-quad-r1-plan.json').read_text())
    a.output.mkdir(parents=True);now=lambda:datetime.now(timezone.utc).isoformat();start=time.monotonic()
    command=['timeout','--signal=TERM','--kill-after=10s','120s','iverilog','-g2012','-Wall',
        '-s','tb_ntrpu_attention_ffn_quad_real_r1','-o',str(a.output/'hierarchy.vvp')]
    command+=['-D'+n for n in plan['defines']]+[str(a.source/n) for n in plan['native_sources']]
    info=dict(started_utc=now(),source=str(a.source),source_sha256=a.source_sha,cpu_affinity=affinity,
        command=command,compile_only=True,arithmetic_executed=False,timeout_seconds=120)
    with (a.output/'invocation.started.json').open('x') as f:json.dump(info,f,indent=2);f.write('\n')
    with (a.output/'compile.log').open('xb') as f:result=subprocess.run(command,cwd=a.output,stdout=f,stderr=subprocess.STDOUT)
    scopes={}
    if result.returncode==0:
        text=(a.output/'hierarchy.vvp').read_text()
        for module in ('ntrpu_q8_gemv_quad','ntrpu_q8_block_mac','ntrpu_q8_gemv_parallel'):
            scopes[module]=len(re.findall(r'\.scope module, "[^"\n]+" "'+module+r'"',text))
    info.update(exit_code=result.returncode,finished_utc=now(),elapsed_seconds=time.monotonic()-start,
        hierarchy_instances=scopes,source_unchanged=all(hashlib.sha256((a.source/n).read_bytes()).hexdigest()==h for n,h in hashes.items()))
    with (a.output/'invocation.json').open('x') as f:json.dump(info,f,indent=2);f.write('\n')
    print(json.dumps(info))


if __name__=='__main__':main()
