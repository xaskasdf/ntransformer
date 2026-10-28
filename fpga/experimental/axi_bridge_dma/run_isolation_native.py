"""Run on the Vivado VM, using private copies only; preserve and hash each batch."""
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


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def manifest(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def run(root,seed,example,settings):
    # Do not take ownership of any existing work or shared process group.
    allowed=Path('/home/ubuntu/vivado/ntrpu-side').resolve()
    for p in (root,seed,example):
        if not p.resolve().is_relative_to(allowed):raise ValueError('Outside private side namespace')
    if root.exists() or not (seed/'xsim.dir').is_dir():raise ValueError('New output and compiled seed required')
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    here=Path(__file__).resolve().parent
    names=['ntrpu_axi_bridge_isolation.sv','ntrpu_axi_bridge_guard.sv','run_isolation_trace.tcl','run_isolation_native.py']
    for name in names:shutil.copy2(here/name,source/name)
    # Preserve all original client sources with the new fixture for recovery.
    for name in ['ntrpu_axi_bridge_writer.sv','ntrpu_axi_bridge_mapper.sv','ntrpu_axi_bridge_endpoint_client.sv','ntrpu_xdma_local_master.sv']:
        shutil.copy2(seed.parent/'source'/name,source/name)
    board=(example/'imports/board.v').read_text()
    before_board=sha(example/'imports/board.v')
    if board.count('ntrpu_axi_bridge_native probe();')!=1:raise ValueError('Reviewed board binding required')
    (source/'board.v').write_text(board.replace('ntrpu_axi_bridge_native probe();','ntrpu_axi_bridge_isolation probe();'))
    shutil.copy2(example/'imports/board_common.vh',source/'board_common.vh')
    seed_before=manifest(seed);save(root/'SEED_FILES.json',seed_before)
    shutil.copytree(seed/'xsim.dir',work/'xsim.dir');shutil.copy2(seed/'xsim.ini',work/'xsim.ini')
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    env=dict(os.environ,NTRPU_NATIVE_OUT=str(root/'output'))
    steps=[];started=time.monotonic()
    def execute(name,args,limit):
        cmd='source '+shlex.quote(str(settings))+' && exec '+shlex.join(list(map(str,args)))
        start=time.monotonic();expired=False
        with (root/(name+'.log')).open('w') as log:
            p=subprocess.Popen(['bash','-c',cmd],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'own_process.json',dict(step=name,pid=p.pid,pgid=p.pid,timeout_seconds=limit))
            try:rc=p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(p.pid,signal.SIGTERM)
                try:rc=p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
        item=dict(step=name,returncode=rc,seconds=time.monotonic()-start,timed_out=expired,command=args)
        steps.append(item);save(root/'steps.json',steps)
        print(json.dumps(item),flush=True)
        if rc!=0 or expired:raise RuntimeError('Native step failed: '+name)
    error=None
    try:
        execute('compile',['xvlog','--incr','--relax','--sv','--work','xil_defaultlib','--include',str(source),str(source/'ntrpu_axi_bridge_guard.sv'),str(source/'ntrpu_axi_bridge_isolation.sv'),str(source/'board.v')],90)
        libs=['xil_defaultlib','axi_bram_ctrl_v4_1_14','gtwizard_ultrascale_v1_7_23','blk_mem_gen_v8_4_13','xdma_v4_2_4','uvm','xilinx_vip','unisims_ver','unimacro_ver','secureip','xpm']
        execute('elaborate',['xelab','--incr','--debug','typical','--relax','--mt','2',*[v for lib in libs for v in ['-L',lib]],'--snapshot','isolation_behav','xil_defaultlib.board','xil_defaultlib.glbl','-log','elaborate.log'],600)
        execute('simulate',['xsim','isolation_behav','-tclbatch',str(source/'run_isolation_trace.tcl'),'-testplusarg','TESTNAME=ntrpu_bridge_native','-log','simulate.log'],600)
    except (OSError,RuntimeError) as e:error=str(e)
    result=dict(status='SIMULATOR_RETURNED_REVIEW_REQUIRED' if error is None else 'FAILED',error=error,steps=steps,
                seconds=time.monotonic()-started,compiled_seed=str(seed),compiled_seed_files=len(seed_before),
                compiled_seed_unchanged=manifest(seed)==seed_before,board_seed_unchanged=sha(example/'imports/board.v')==before_board,
                cpus=sorted(os.sched_getaffinity(0)),priority_nice=os.nice(0),native_review_pass=False)
    save(root/'RESULT.json',result)
    # Small recoverable evidence package: original compiled seed already backed
    # up separately. New snapshot stays on VM; source and commands reproduce it.
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and 'xsim.dir' not in p.relative_to(root).parts]
    files={p.relative_to(root).as_posix():sha(p) for p in paths};save(root/'FILES.json',files)
    archive=root.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in paths+[root/'FILES.json']:tar.add(p,arcname=p.relative_to(root).as_posix())
    result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('--seed',type=Path,required=True)
    ap.add_argument('--example',type=Path,required=True);ap.add_argument('--settings',type=Path,required=True)
    a=ap.parse_args();result=run(a.root,a.seed,a.example,a.settings)
    raise SystemExit(0 if result['error'] is None else 1)
