"""Bounded, private Icarus/XSIM tests of the reader-facing source frontier."""
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
from review_hbm_cdc_queues import review,sha,CASES,FIFO_SHA

def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def run(a):
    root=a.output.resolve();here=Path(__file__).resolve().parent
    if root.exists():raise ValueError('New output required')
    if a.engine=='xsim' and (os.name!='posix' or not root.is_relative_to(Path('/home/ubuntu/vivado/ntrpu-side').resolve())):raise ValueError('Private VM namespace required')
    names=['ntrpu_axi_bridge_hbm_cdc.sv','tb_axi_bridge_hbm_cdc_queues.sv','run_hbm_cdc_queues.py',
           'review_hbm_cdc_queues.py','run_hbm_cdc_queues_trace.tcl','review_hbm_source.py']
    inputs={n:here/n for n in names};inputs['ntrpu_bridge_async_fifo.sv']=a.fifo_source.resolve()
    hashes={n:sha(p) for n,p in inputs.items()}
    if hashes['ntrpu_bridge_async_fifo.sv']!=FIFO_SHA:raise ValueError('Frozen FIFO mismatch')
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for n,p in inputs.items():shutil.copy2(p,source/n)
    if os.name=='posix':os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    env=dict(os.environ);steps=[];reviews=[];error=None;started=time.monotonic()
    def execute(name,args,cwd,limit=90,extra=None):
        run_env=dict(env);run_env.update(extra or {})
        if a.engine=='xsim':args=['bash','-c','source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(list(map(str,args)))]
        with (root/(name+'.log')).open('w') as log:
            t=time.monotonic();p=subprocess.Popen(list(map(str,args)),cwd=cwd,env=run_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
            save(root/'own_process.json',dict(pid=p.pid,pgid=p.pid,step=name,timeout=limit))
            expired=False
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
        rtl=[source/n for n in inputs if n.endswith('.sv')]
        if a.engine=='icarus':
            bins=a.toolchain.resolve()/'mingw64/bin';env['PATH']=str(bins)+os.pathsep+env.get('PATH','')
            execute('compile',[bins/'iverilog.exe','-g2012','-Wall','-s','tb_axi_bridge_hbm_cdc_queues','-o',work/'bench.vvp',*rtl],work)
        else:
            execute('compile',['xvlog','--sv','--define','XSIM','--work','xil_defaultlib',*rtl],work,120)
            execute('elaborate',['xelab','--debug','typical','--mt','2','--snapshot','hbm_cdc_queues','xil_defaultlib.tb_axi_bridge_hbm_cdc_queues'],work,120)
        for seed in range(4):
            for case in CASES:
                name=f'case-{case:02}-seed-{seed}';out=root/name;out.mkdir()
                if a.engine=='icarus':text=execute(name,[bins/'vvp.exe',work/'bench.vvp',f'+case={case}',f'+seed={seed}'],out,30)
                else:text=execute(name,['xsim','hbm_cdc_queues','-tclbatch',source/'run_hbm_cdc_queues_trace.tcl','-testplusarg',f'case={case}','-testplusarg',f'seed={seed}','-log',name+'.log'],work,90,{'HBM_CDC_QUEUES_TRACE':str(out/'hbm_cdc_queues.vcd')})
                checked=review(text,case,seed);reviews.append(checked);save(out/'REVIEW.json',checked)
                print(json.dumps(dict(completed=len(reviews),total=4*len(CASES),case=case,seed=seed)),flush=True)
    except (ValueError,OSError,RuntimeError) as exc:error=str(exc)
    unchanged=hashes=={n:sha(p) for n,p in inputs.items()}
    if not unchanged:error='Input sources changed'
    result=dict(status='HBM_CDC_QUEUES_PASS' if error is None else 'FAILED',error=error,engine=a.engine,sources=hashes,sources_unchanged=unchanged,reviews=reviews,steps=steps,seconds=time.monotonic()-started,native_hbm_pcie_simulation=False)
    save(root/'RESULT.json',result)
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(work)]
    save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in paths})
    if a.engine=='xsim':
        archive=Path(str(root)+'.tar.gz')
        with tarfile.open(archive,'x:gz') as tar:
            for p in paths+[root/'FILES.json']:tar.add(p,arcname=p.relative_to(root).as_posix())
        result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','reviews','sources')}),flush=True)
    return error is None

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--engine',choices=('icarus','xsim'),required=True)
    ap.add_argument('--fifo-source',type=Path,required=True);ap.add_argument('--settings',type=Path);ap.add_argument('--toolchain',type=Path)
    a=ap.parse_args();raise SystemExit(0 if run(a) else 1)
