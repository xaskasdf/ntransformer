"""Run one bounded private phase; preserve failures and archive VM evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from review_hbm_clocked import review_case,review_source,sha

def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def run(a):
    root=a.output.resolve();prepared=a.prepared.resolve();here=Path(__file__).resolve().parent
    if root.exists():raise ValueError('New output required')
    if a.engine!='icarus' and (os.name!='posix' or not root.is_relative_to(Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-clocked').resolve())):raise ValueError('Private VM namespace required')
    inputs={p.name:p for p in prepared.iterdir() if p.is_file() and p.suffix in ('.sv','.py','.tcl','.json')}
    for n in ['ntrpu_axi_bridge_hbm_mem_watch.sv','tb_axi_bridge_hbm_clocked.sv','prepare_hbm_clocked.py','run_hbm_clocked.py','review_hbm_clocked.py','synth_hbm_clocked.tcl']:
        inputs[n]=here/n
    hashes={n:sha(p) for n,p in inputs.items()}
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for n,p in inputs.items():shutil.copy2(p,source/n)
    review_source(source)
    if os.name=='posix':os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    env=dict(os.environ);steps=[];error=None;started=time.monotonic()
    def execute(name,args,cwd,limit):
        if a.engine!='icarus':args=['bash','-c','source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(list(map(str,args)))]
        with (root/(name+'.log')).open('w') as log:
            t=time.monotonic();p=subprocess.Popen(list(map(str,args)),cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
            save(root/'own_process.json',dict(pid=p.pid,pgid=p.pid,step=name,timeout=limit));expired=False
            try:code=p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                expired=True
                if os.name=='posix':os.killpg(p.pid,signal.SIGTERM)
                else:p.terminate()
                try:code=p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name=='posix':os.killpg(p.pid,signal.SIGKILL)
                    else:p.kill()
                    code=p.wait()
        steps.append(dict(name=name,returncode=code,timed_out=expired,seconds=time.monotonic()-t));save(root/'steps.json',steps)
        text=(root/(name+'.log')).read_text(errors='replace')
        if code or expired or re.search(r'(?im)^\s*(?:Fatal|Error|FATAL|ERROR):',text):raise ValueError('Failed step: '+name)
        return text
    try:
        if a.engine=='synth':
            execute('synthesis',['vivado','-mode','batch','-source',source/'synth_hbm_clocked.tcl','-journal',root/'vivado.jou','-log',root/'vivado.log','-tclargs',source,root/'output'],work,1200)
        else:
            names=['ntrpu_axi_bridge_guard_v3','ntrpu_axi_bridge_guard_v3_reset_export','ntrpu_hbm_startup','ntrpu_axi_bridge_hbm_mem_watch','ntrpu_axi_bridge_hbm_cdc','ntrpu_bridge_async_fifo','tb_axi_bridge_hbm_clocked']
            rtl=[source/(n+'.sv') for n in names]
            if a.engine=='icarus':
                bins=a.toolchain.resolve()/'mingw64/bin';env['PATH']=str(bins)+os.pathsep+env.get('PATH','')
                execute('compile',[bins/'iverilog.exe','-g2012','-Wall','-s','tb_axi_bridge_hbm_clocked','-o',work/'bench.vvp',*rtl],work,90)
            else:
                execute('compile',['xvlog','--sv',*rtl],work,120)
                execute('elaborate',['xelab','--debug','typical','--mt','2','--snapshot','hbm_clocked','work.tb_axi_bridge_hbm_clocked'],work,120)
            for seed in range(2):
                for case in range(13):
                    name=f'case-{case:02}-seed-{seed}'
                    if a.engine=='icarus':args=[bins/'vvp.exe',work/'bench.vvp',f'+case={case}',f'+seed={seed}']
                    else:args=['xsim','hbm_clocked','-runall','-testplusarg',f'case={case}','-testplusarg',f'seed={seed}','-log',name+'.log']
                    review_case(execute(name,args,work,60),case,seed)
                    print(json.dumps(dict(case=case,seed=seed,complete=True)),flush=True)
    except (OSError,ValueError,RuntimeError) as exc:error=str(exc)
    unchanged=hashes=={n:sha(p) for n,p in inputs.items()}
    if not unchanged:error='Input sources changed'
    result=dict(status='HBM_CLOCKED_EXPORTED' if error is None else 'FAILED',error=error,engine=a.engine,
                sources=hashes,sources_unchanged=unchanged,steps=steps,seconds=time.monotonic()-started,
                native_hbm_pcie_ip=False,physical_timing_qualified=False)
    save(root/'RESULT.json',result)
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(work) and '.Xil' not in p.parts and 'project' not in p.relative_to(root).parts]
    save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in paths})
    if a.engine!='icarus':
        archive=Path(str(root)+'.tar.gz')
        with tarfile.open(archive,'x:gz') as tar:
            for p in paths+[root/'FILES.json']:tar.add(p,arcname=p.relative_to(root).as_posix())
        result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','sources')}),flush=True)
    return error is None

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--engine',choices=('icarus','xsim','synth'),required=True)
    ap.add_argument('--toolchain',type=Path);ap.add_argument('--settings',type=Path)
    a=ap.parse_args();raise SystemExit(0 if run(a) else 1)
