"""Explicit, future VM execution of a prepared private clock-loss fixture."""
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

from review_clock_loss_native import review


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def manifest(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def run(root,seed,example,settings):
    if os.name!='posix':raise ValueError('Native runner requires the Linux Vivado VM')
    allowed=Path('/home/ubuntu/vivado/ntrpu-side').resolve()
    for path in (root,seed,example):
        if not path.resolve().is_relative_to(allowed):raise ValueError('Outside private side namespace')
    if root.resolve().is_relative_to(seed.resolve()) or root.resolve().is_relative_to(example.resolve()):
        raise ValueError('Output must not be nested inside either seed')
    archive=Path(str(root)+'.tar.gz')
    if root.exists() or archive.exists() or not (seed/'xsim.dir').is_dir() or not settings.is_file():
        raise ValueError('New output, compiled seed and Vivado settings required')
    here=Path(__file__).resolve().parent
    prepared=json.loads((here/'PREPARED.json').read_text())
    for name,digest in prepared['files'].items():
        if Path(name).name!=name or sha(here/name)!=digest:raise ValueError('Prepared source changed: '+name)
    for name in ('ntrpu_axi_bridge_writer.sv','ntrpu_axi_bridge_mapper.sv',
                 'ntrpu_axi_bridge_endpoint_client.sv','ntrpu_xdma_local_master.sv'):
        if sha(here/name)!=sha(seed.parent/'source'/name):raise ValueError('Compiled client seed differs: '+name)
    board=example/'imports/board.v';before_board=sha(board);binding=board.read_text()
    if binding.count('ntrpu_axi_bridge_native probe();')!=1:raise ValueError('Reviewed board binding required')
    seed_before=manifest(seed)
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for name in prepared['files']:shutil.copy2(here/name,source/name)
    shutil.copy2(here/'PREPARED.json',source/'PREPARED.json')
    (source/'board.v').write_text(binding.replace('ntrpu_axi_bridge_native probe();','ntrpu_axi_bridge_clock_loss probe();'))
    shutil.copy2(example/'imports/board_common.vh',source/'board_common.vh')
    save(root/'SEED_FILES.json',seed_before)
    shutil.copytree(seed/'xsim.dir',work/'xsim.dir');shutil.copy2(seed/'xsim.ini',work/'xsim.ini')
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    steps=[];reviews=[];started=time.monotonic();error=None

    def execute(name,args,limit,env=None):
        command='source '+shlex.quote(str(settings))+' && exec '+shlex.join(list(map(str,args)))
        start=time.monotonic();expired=False
        with (root/(name+'.log')).open('w') as log:
            p=subprocess.Popen(['bash','-c',command],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'own_process.json',dict(step=name,pid=p.pid,pgid=p.pid,timeout_seconds=limit))
            try:code=p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
        item=dict(step=name,returncode=code,timed_out=expired,seconds=time.monotonic()-start,command=args)
        steps.append(item);save(root/'steps.json',steps);print(json.dumps(item),flush=True)
        if code or expired:raise RuntimeError('Native step failed: '+name)

    try:
        execute('compile',['xvlog','--incr','--relax','--sv','--work','xil_defaultlib','--include',str(source),
            str(source/'ntrpu_axi_bridge_guard.sv'),str(source/'ntrpu_axi_bridge_clock_loss.sv'),str(source/'board.v')],90)
        libs=['xil_defaultlib','axi_bram_ctrl_v4_1_14','gtwizard_ultrascale_v1_7_23','blk_mem_gen_v8_4_13',
              'xdma_v4_2_4','uvm','xilinx_vip','unisims_ver','unimacro_ver','secureip','xpm']
        execute('elaborate',['xelab','--incr','--debug','typical','--relax','--mt','2',
            *[v for lib in libs for v in ['-L',lib]],'--snapshot','clock_loss_behav','xil_defaultlib.board','xil_defaultlib.glbl','-log','elaborate.log'],600)
        for level in (0,1):
            out=root/f'level-{level}'
            execute(f'simulate-{level}',['xsim','clock_loss_behav','-tclbatch',str(source/'run_clock_loss_trace.tcl'),
                '-testplusarg','TESTNAME=ntrpu_bridge_native','-testplusarg',f'clock_stop_level={level}',
                '-log',f'simulate-{level}.log'],600,dict(os.environ,NTRPU_NATIVE_OUT=str(out)))
            checked=review((root/f'simulate-{level}.log').read_text(errors='replace'),(out/'clock_loss.vcd').read_text())
            if checked['stop_level']!=level:raise ValueError('Native run case mismatch')
            checked.update(log_sha256=sha(root/f'simulate-{level}.log'),vcd_sha256=sha(out/'clock_loss.vcd'))
            save(out/'REVIEW.json',checked);reviews.append(checked)
    except (OSError,ValueError,RuntimeError) as exc:error=str(exc)
    unchanged=manifest(seed)==seed_before and sha(board)==before_board
    if not unchanged:error='Seed changed during execution'
    result=dict(status='NATIVE_AXI_BOUNDARY_CLOCK_INJECTION_PASS' if error is None and len(reviews)==2 else 'FAILED',
        error=error,steps=steps,reviews=reviews,seconds=time.monotonic()-started,seed_unchanged=unchanged,
        seed_files=len(seed_before),compiled_seed=str(seed),cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),
        physical_clock_failure_validated=False,physical_cdc_validated=False)
    save(root/'RESULT.json',result)
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and 'xsim.dir' not in p.relative_to(root).parts]
    save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in paths})
    with tarfile.open(archive,'x:gz') as tar:
        for p in paths+[root/'FILES.json']:tar.add(p,arcname=p.relative_to(root).as_posix())
    result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('--seed',type=Path,required=True)
    ap.add_argument('--example',type=Path,required=True);ap.add_argument('--settings',type=Path,required=True)
    ap.add_argument('--execute-native',action='store_true',help='Explicit opt-in; this script does not upload or SSH')
    a=ap.parse_args()
    if not a.execute_native:ap.error('Use --execute-native only when the private VM run is authorized')
    result=run(a.root,a.seed,a.example,a.settings)
    raise SystemExit(0 if result['status']=='NATIVE_AXI_BOUNDARY_CLOCK_INJECTION_PASS' else 1)
