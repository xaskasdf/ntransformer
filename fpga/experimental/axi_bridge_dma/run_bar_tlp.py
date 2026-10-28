"""Bounded PCIe PIPE traffic on a copied, already backed-up AMD compiled seed."""
import argparse
import hashlib
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


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def manifest(root):return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def run(a):
    root=a.output.resolve();seed=a.seed.resolve();prepared=a.prepared.resolve()
    ns=Path('/home/ubuntu/vivado/ntrpu-side').resolve()
    if os.name!='posix' or any(not p.is_relative_to(ns) or p==ns for p in (root,seed,prepared)):
        raise ValueError('Private namespace required')
    if root.exists() or not (seed/'xsim.dir').is_dir():raise ValueError('Fresh output and compiled seed required')
    if a.bar_base is not None and (a.bar_base<0 or a.bar_base>0xfffe0000 or a.bar_base%0x20000):
        raise ValueError('BAR base must be 128 KiB aligned within 32-bit memory')
    replay=None
    if a.replay:
        donor=json.loads((seed.parent/'RESULT.json').read_text())
        if donor['status']!='BAR_TLP_EXPORTED' or not donor['sources_unchanged']:
            raise ValueError('Replay requires a closed successful donor')
        if not (seed/'xsim.dir/bar_tlp_behav/xsimk').is_file():raise ValueError('Missing compiled traffic snapshot')
        for name in ('ntrpu_control_snapshot.sv','ntrpu_hbm_control_plane.sv','ntrpu_bar32_adapter.sv',
                     'ntrpu_bar_tlp_probe.sv','xilinx_axi_pcie4_ep.v','board.v','board_common.vh'):
            if sha(prepared/name)!=donor['sources'][name] or sha(seed.parent/'source'/name)!=donor['sources'][name]:
                raise ValueError('Replay HDL differs: '+name)
        replay=dict(donor=str(seed.parent),result_sha256=sha(seed.parent/'RESULT.json'),rtl_unchanged=True)
    root.mkdir();source=root/'source';shutil.copytree(prepared,source);work=root/'work';work.mkdir()
    hashes=manifest(source);seed_before=manifest(seed);save(root/'SEED_FILES.json',seed_before)
    shutil.copytree(seed/'xsim.dir',work/'xsim.dir');shutil.copy2(seed/'xsim.ini',work/'xsim.ini')
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    env=dict(os.environ,NTRPU_NATIVE_OUT=str(root/'output'))
    env.pop('NTRPU_BAR_BASE',None)
    if a.bar_base is not None:env['NTRPU_BAR_BASE']=format(a.bar_base,'x')
    steps=[];started=time.monotonic();error=None
    def execute(name,argv,limit):
        cmd='source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(argv)
        start=time.monotonic();expired=False
        with (root/(name+'.log')).open('w') as log:
            child=subprocess.Popen(['bash','-c',cmd],cwd=work,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with (root/'own_processes.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=name,pid=child.pid,pgid=child.pid,timeout=limit))+'\n')
            try:code=child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        step=dict(name=name,returncode=code,timed_out=expired,seconds=time.monotonic()-start,command=argv)
        steps.append(step);save(root/'steps.json',steps);print(json.dumps(step),flush=True)
        if code or expired:raise RuntimeError('Failed '+name)
    try:
        rtl=['ntrpu_control_snapshot.sv','ntrpu_hbm_control_plane.sv','ntrpu_bar32_adapter.sv',
             'ntrpu_bar_tlp_probe.sv','xilinx_axi_pcie4_ep.v','board.v']
        if not a.replay:
            execute('compile',['xvlog','--incr','--relax','--sv','--work','xil_defaultlib','--include',str(source),
                               *[str(source/name) for name in rtl]],90)
            libs=['xil_defaultlib','axi_bram_ctrl_v4_1_14','gtwizard_ultrascale_v1_7_23','blk_mem_gen_v8_4_13',
                  'xdma_v4_2_4','uvm','xilinx_vip','unisims_ver','unimacro_ver','secureip','xpm']
            execute('elaborate',['xelab','--incr','--debug','typical','--relax','--mt','2',
                *[x for lib in libs for x in ['-L',lib]],'--snapshot','bar_tlp_behav','xil_defaultlib.board',
                'xil_defaultlib.glbl','-log','elaborate.log'],900)
        execute('simulate',['xsim','bar_tlp_behav','-tclbatch',str(source/'run_bar_tlp_trace.tcl'),
                           '-testplusarg','TESTNAME=ntrpu_bridge_native','-log','simulate.log'],900)
        text=(root/'simulate.log').read_text(errors='replace')
        if len(re.findall(r'^BAR_TLP_PASS .+$',text,re.M))!=1 or re.search(r'(?im)^\s*(?:fatal|error):|BT CHECK|BT WATCHDOG',text):
            raise RuntimeError('Native traffic has no clean PASS')
    except (OSError,RuntimeError) as exc:error=str(exc)
    unchanged=manifest(prepared)==hashes==manifest(source) and manifest(seed)==seed_before
    if not unchanged:error='Input changed'
    result=dict(status='BAR_TLP_EXPORTED' if error is None else 'FAILED',error=error,steps=steps,sources=hashes,
        seconds=time.monotonic()-started,sources_unchanged=unchanged,compiled_seed=str(seed),compiled_seed_files=len(seed_before),
        cpus=sorted(os.sched_getaffinity(0)),priority_nice=os.nice(0),hbm_memory_simulated=False,physical_link_simulated=False,
        replay=replay,requested_bar_base=a.bar_base)
    save(root/'RESULT.json',result)
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and 'xsim.dir' not in p.relative_to(root).parts and p.suffix not in ('.wdb','.pb')]
    save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in paths})
    archive=Path(str(root)+'.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:
        for p in paths+[root/'FILES.json']:tar.add(p,arcname=p.relative_to(root).as_posix())
    receipt={k:v for k,v in result.items() if k not in ('sources','steps')}
    receipt.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps(receipt),flush=True)
    return error is None


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    for name in ('seed','prepared','settings'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replay',action='store_true',help='Reuse a closed identical-HDL traffic snapshot')
    p.add_argument('--bar-base',type=lambda value:int(value,0),help='Root Port MEM32 allocator seed')
    raise SystemExit(0 if run(p.parse_args()) else 1)
