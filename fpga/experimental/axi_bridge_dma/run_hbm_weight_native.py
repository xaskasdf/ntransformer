"""Bounded isolated native HBM/PCIe simulation; every attempt exports evidence."""
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
from itertools import groupby


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def manifest(root):return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def run(a):
    root=a.output.resolve();prepared=a.prepared.resolve()
    ns=Path('/home/ubuntu/vivado/ntrpu-side').resolve()
    if os.name!='posix' or root.exists() or any(not p.is_relative_to(ns) or p==ns for p in (root,prepared)):
        raise ValueError('Fresh private paths required')
    root.mkdir();source=root/'source';shutil.copytree(prepared,source);work=root/'work';work.mkdir()
    hashes=manifest(source);donors={};steps=[];error=None;start=time.monotonic()
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    env=dict(os.environ,NTRPU_NATIVE_OUT=str(root/'output'),NTRPU_BAR_BASE='80000000')
    def execute(name,argv,limit):
        before=time.monotonic();expired=False
        with (root/(name+'.log')).open('w') as log:
            child=subprocess.Popen(['bash','-c','source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(argv)],
                cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with (root/'own_processes.jsonl').open('a') as f:f.write(json.dumps(dict(step=name,pid=child.pid,pgid=child.pid,timeout=limit))+'\n')
            try:code=child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        step=dict(name=name,returncode=code,timed_out=expired,seconds=time.monotonic()-before,command=argv)
        steps.append(step);save(root/'steps.json',steps);print(json.dumps(step),flush=True)
        if code or expired:raise RuntimeError('Failed '+name)
    try:
        if a.stage=='models':
            execute('generate',['vivado','-mode','batch','-source',str(source/'generate_hbm_tlp_models.tcl'),
                '-log',str(root/'vivado.log'),'-journal',str(root/'vivado.jou'),'-tclargs',str(source),str(root/'output')],300)
            if (root/'output/STATUS.txt').read_text().strip()!='HBM_TLP_MODELS_EXPORTED':raise RuntimeError('Missing generation verdict')
        else:
            seed=a.seed.resolve();models=a.models.resolve()
            if any(not p.is_relative_to(ns) or p==ns for p in (seed,models)):raise ValueError('Private donors required')
            if json.loads((models/'RESULT.json').read_text())['status']!='HBM_TLP_EXPORTED':raise ValueError('Closed models required')
            donors[str(seed)]=manifest(seed);donors[str(models/'output')]=manifest(models/'output')
            shutil.copytree(seed/'xsim.dir',work/'xsim.dir');shutil.copy2(seed/'xsim.ini',work/'xsim.ini')
            shutil.copytree(models/'output',root/'models')
            binding=json.loads((source/'BINDING.json').read_text())
            reuse_models=a.resume or a.loader_update
            if reuse_models:
                prior=json.loads((seed.parent/'RESULT.json').read_text())
                if prior['stage']!='traffic' or not prior['sources_unchanged']:raise ValueError('Closed unchanged traffic donor required')
                changed=[];added=[]
                for name in binding['compile_names']+['board_common.vh']:
                    if name not in prior['sources']:
                        added.append(name);continue
                    if sha(seed.parent/'source'/name)!=prior['sources'][name]:raise ValueError('Changed donor HDL: '+name)
                    if sha(source/name)!=prior['sources'][name]:changed.append(name)
                allowed=['ntrpu_hbm_tlp_probe.sv','xilinx_axi_pcie4_ep.v','board.v'] if a.loader_update else []
                expected_added=['ntrpu_hbm_weight_loader.sv'] if a.loader_update else []
                if set(changed)!=set(allowed) or added!=expected_added:
                    raise ValueError('Unexpected integration differences: '+str((changed,added)))
                for name,digest in binding['frozen_rtl'].items():
                    if sha(source/name)!=digest or prior['sources'][name]!=digest:raise ValueError('Frozen core changed: '+name)
                save(root/'RESUME.json',dict(donor=str(seed.parent),result_sha256=sha(seed.parent/'RESULT.json'),
                    frozen_core_unchanged=True,changed_hdl=changed,added_hdl=added,
                    debug_level=a.debug,elaboration_limit=a.elaboration_limit))
            rows=[];include_paths=set()
            for line in (root/'models/compile_order.tsv').read_text().splitlines():
                lib,kind,name=line.split('\t');path=Path(name)
                copied=root/'models'/path.relative_to(models/'output')
                if kind=='Verilog Header':include_paths.add(str(copied.parent))
                if kind in ('Verilog','SystemVerilog','VHDL') and copied.name!='model_inventory.v':rows.append((lib,kind,str(copied)))
            for name in ('native_hbm.v','fabric_clocks.v','hbm_read_cdc.v','hbm_read_width.v','hbm_read_protocol.v'):
                if not any(Path(row[2]).name==name for row in rows):raise ValueError('Missing model '+name)
            includes=sorted(include_paths)
            for number,(key,entries) in enumerate(groupby(rows,key=lambda row:row[:2])):
                lib,kind=key;files=[row[2] for row in entries]
                if kind=='VHDL':argv=['xvhdl','--incr','--relax','--work',lib,*files]
                else:argv=['xvlog','--incr','--relax','--work',lib,
                    *(['--sv'] if kind=='SystemVerilog' else []),*[x for path in includes for x in ['-i',path]],*files]
                if not reuse_models:execute('model_compile_'+str(number),argv,120)
            for p in (root/'models').rglob('*.mem'):
                dest=work/p.name
                if dest.exists() and sha(dest)!=sha(p):raise ValueError('Conflicting HBM initialization '+p.name)
                shutil.copy2(p,dest)
            if not reuse_models or a.loader_update:
                names=binding['compile_names']
                execute('compile',['xvlog','--incr','--relax','--sv','--work','xil_defaultlib','-i',str(source),
                    *[str(source/name) for name in names]],120)
            libs=list(dict.fromkeys(['xil_defaultlib',*[row[0] for row in rows],
                'axi_bram_ctrl_v4_1_14','gtwizard_ultrascale_v1_7_23','blk_mem_gen_v8_4_13','xdma_v4_2_4',
                'uvm','xilinx_vip','unisims_ver','unimacro_ver','secureip','xpm']))
            execute('elaborate',['xelab','--incr','--debug',a.debug,'--relax','--mt','2',
                *[x for lib in libs for x in ['-L',lib]],'--snapshot','hbm_tlp_behav','xil_defaultlib.board',
                'xil_defaultlib.glbl','-log','elaborate.log'],a.elaboration_limit)
            execute('simulate',['xsim','hbm_tlp_behav','-tclbatch',str(source/'run_hbm_tlp_trace.tcl'),
                '-testplusarg','TESTNAME=ntrpu_bridge_native','-log','simulate.log'],3000)
            log=(root/'simulate.log').read_text(errors='replace')
            if len(re.findall(r'^HBM_WEIGHT_NATIVE_PASS .+$',log,re.M))!=1 or re.search(r'(?im)^\s*(?:fatal|error):|HBT WATCHDOG|BT CHECK',log):
                raise RuntimeError('No clean integrated traffic verdict')
    except (OSError,RuntimeError,ValueError) as exc:error=str(exc)
    unchanged=manifest(prepared)==hashes==manifest(source)
    for path,expected in donors.items():unchanged=unchanged and manifest(Path(path))==expected
    if not unchanged:error='Input changed'
    result=dict(status='HBM_WEIGHT_NATIVE_EXPORTED' if error is None else 'FAILED',stage=a.stage,error=error,steps=steps,sources=hashes,
                sources_unchanged=unchanged,seconds=time.monotonic()-start,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0))
    save(root/'RESULT.json',result);save(root/'DONORS.json',donors)
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and not set(p.relative_to(root).parts)&{'xsim.dir','.Xil'}
           and p.suffix not in ('.wdb','.pb','.str','.zip')]
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
    for name in ('prepared','settings'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=('traffic',),default='traffic')
    p.add_argument('--seed',type=Path);p.add_argument('--models',type=Path)
    reuse=p.add_mutually_exclusive_group()
    reuse.add_argument('--resume',action='store_true',help='Reuse the libraries of a closed, identical-HDL attempt')
    reuse.add_argument('--loader-update',action='store_true',help='Reuse vendor models; require frozen core unchanged and recompile integration')
    p.add_argument('--debug',choices=('typical','wave'),default='wave')
    p.add_argument('--elaboration-limit',type=int,choices=(1200,2400),default=2400)
    raise SystemExit(0 if run(p.parse_args()) else 1)
