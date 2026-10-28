"""Bounded, private simulation or implementation of the range-check variant."""
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
from prepare_hbm_range import variant
from review_hbm_source import review as review_source_case, CASES as SOURCE_CASES
from review_hbm_cdc import review as review_cdc_case, CASES as CDC_CASES

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def seal(root):save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and 'work' not in p.relative_to(root).parts and 'project' not in p.relative_to(root).parts and '.Xil' not in p.parts and p!=root/'FILES.json'})

def run(a):
    root=a.output.resolve();prepared=a.prepared.resolve()
    if root.exists():raise ValueError('New output required')
    if a.engine!='icarus' and (os.name!='posix' or not root.is_relative_to(Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-range').resolve())):raise ValueError('Private VM namespace required')
    names={p.name:p for p in prepared.iterdir() if p.is_file() and p.suffix in ('.sv','.tcl','.py','.json')}
    hashes={n:sha(p) for n,p in names.items()}
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for n,p in names.items():shutil.copy2(p,source/n)
    original=(source/'reference_hbm_source.sv').read_text().replace('module reference_hbm_source (','module ntrpu_axi_bridge_hbm_source (')
    if variant(original)!=(source/'ntrpu_axi_bridge_hbm_source.sv').read_text():raise ValueError('Extra RTL changes')
    if os.name=='posix':os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    env=dict(os.environ);steps=[];reviews=[];error=None;started=time.monotonic()
    if a.engine=='icarus':
        bins=a.toolchain.resolve()/'mingw64/bin';env['PATH']=str(bins)+os.pathsep+env.get('PATH','')
    def execute(name,args,cwd,limit):
        if a.engine!='icarus':args=['bash','-c','source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(list(map(str,args)))]
        with (root/(name+'.log')).open('w') as log:
            t=time.monotonic();p=subprocess.Popen(list(map(str,args)),cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
            process=dict(pid=p.pid,pgid=p.pid,step=name,timeout=limit)
            save(root/'own_process.json',process)
            with (root/'own_processes.jsonl').open('a') as f:f.write(json.dumps(process)+'\n')
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
        step=dict(name=name,returncode=code,timed_out=expired,seconds=time.monotonic()-t);steps.append(step);save(root/'steps.json',steps)
        text=(root/(name+'.log')).read_text(errors='replace')
        if code or expired or re.search(r'(?im)^\s*(?:Fatal|Error|FATAL|ERROR):',text):raise ValueError('Failed step: '+name)
        return text
    try:
        if a.engine in ('physical','reopen','refine'):
            if a.engine=='physical':
                execute('synthesis',['vivado','-mode','batch','-source',source/'synth_hbm_clocked.tcl','-journal',root/'synth.jou','-log',root/'synth.log','-tclargs',source,root/'synth'],work,1200)
                donor=root/'synth/synth.dcp'
            else:donor=a.donor.resolve()
            if a.donor_sha and sha(donor)!=a.donor_sha:raise ValueError('Donor hash mismatch')
            if a.engine=='refine':
                if not a.donor_sha:raise ValueError('Refinement requires pinned routed donor')
                shutil.copy2(donor,source/'refinement_donor.dcp')
                execute('refinement',['vivado','-mode','batch','-source',source/'refine_hbm_range.tcl','-journal',root/'refine.jou','-log',root/'refine.log','-tclargs',source/'refinement_donor.dcp',root/'refinement'],work,600)
                donor=root/'refinement/refined.dcp'
            phase=root/'physical';phase.mkdir();ps=phase/'source';ps.mkdir()
            helpers=['constrain_hbm_clocktree.tcl','insert_hbm_clocktree.tcl','audit_hbm_clocktree.tcl','audit_hbm_clocktree_topology.tcl','run_hbm_clocktree.tcl']
            for n in helpers:shutil.copy2(source/n,ps/n)
            shutil.copy2(donor,ps/'donor.dcp');mode='route' if a.engine=='physical' else 'reopen';phase_error=None
            try:
                execute('implementation',['vivado','-mode','batch','-source',ps/'run_hbm_clocktree.tcl','-journal',phase/'vivado.jou','-log',phase/'vivado.log','-tclargs',ps,ps/'donor.dcp',phase/'output',mode],work,1200 if mode=='route' else 600)
            except (ValueError,OSError) as exc:phase_error=str(exc)
            step=steps[-1]
            save(phase/'RESULT.json',dict(status='HBM_CLOCKTREE_EXPORTED' if phase_error is None else 'FAILED',mode=mode,error=phase_error,
                 returncode=step['returncode'],timed_out=step['timed_out'],sources_unchanged=all(sha(ps/n)==hashes[n] for n in helpers),
                 donor_sha256=sha(donor),sources={n:sha(ps/n) for n in helpers}))
            seal(phase)
            if phase_error:raise ValueError(phase_error)
        else:
            common=['ntrpu_axi_bridge_guard_v3','ntrpu_axi_bridge_writer','ntrpu_axi_bridge_hbm_source','ntrpu_axi_bridge_hbm_client',
                    'ntrpu_axi_bridge_endpoint_client','ntrpu_axi_bridge_mapper','ntrpu_hbm_read_frames','ntrpu_xdma_local_master']
            suites=[('range',['ntrpu_axi_bridge_hbm_source','reference_hbm_source'],None),
                    ('source',common,SOURCE_CASES),('cdc',common+['ntrpu_axi_bridge_hbm_cdc','ntrpu_bridge_async_fifo'],CDC_CASES)]
            for suite,modules,cases in suites:
                cwd=work/suite;cwd.mkdir();top='tb_axi_bridge_hbm_'+suite
                rtl=[source/(n+'.sv') for n in modules+[top]]
                if a.engine=='icarus':execute(suite+'-compile',[bins/'iverilog.exe','-g2012','-Wall','-s',top,'-o',cwd/'bench.vvp',*rtl],cwd,90)
                else:
                    execute(suite+'-compile',['xvlog','--sv','--define','XSIM',*rtl],cwd,120)
                    execute(suite+'-elaborate',['xelab','--debug','typical','--mt','2','--snapshot',suite,'work.'+top],cwd,120)
                for case,seed in ([(None,None)] if cases is None else [(c,s) for s in range(2) for c in cases]):
                    name=suite+'-predicate' if cases is None else f'{suite}-case-{case:02}-seed-{seed}'
                    args=[bins/'vvp.exe',cwd/'bench.vvp'] if a.engine=='icarus' else ['xsim',suite,'-runall','-log',name+'.log']
                    if cases is not None:
                        if a.engine=='icarus':args += [f'+case={case}',f'+seed={seed}']
                        else:args += ['-testplusarg',f'case={case}','-testplusarg',f'seed={seed}']
                    text=execute(name,args,cwd,180 if cases is None else 90)
                    if cases is None:
                        match=re.findall(r'HR PASS known=(\d+) unknown=(\d+)',text)
                        if match!=[('380448','28480')]:raise ValueError('Predicate coverage mismatch: '+str(match))
                        reviews.append(dict(suite=suite,known=380448,unknown=28480))
                    else:
                        check=review_source_case if suite=='source' else review_cdc_case
                        reviews.append(dict(suite=suite,**check(text,case,seed)))
                    print(json.dumps(dict(completed=name)),flush=True)
    except (ValueError,OSError,RuntimeError) as exc:error=str(exc)
    unchanged=hashes=={n:sha(p) for n,p in names.items()}
    if not unchanged:error='Input sources changed'
    result=dict(status='HBM_RANGE_EXPORTED' if error is None else 'FAILED',error=error,engine=a.engine,sources=hashes,sources_unchanged=unchanged,
                steps=steps,reviews=reviews,seconds=time.monotonic()-started,refinement_donor_sha256=a.donor_sha if a.engine=='refine' else None,native_hbm_pcie_ip=False,board_qualified=False)
    save(root/'RESULT.json',result);seal(root)
    if a.engine!='icarus':
        archive=Path(str(root)+'.tar.gz')
        with tarfile.open(archive,'x:gz') as tar:
            for n in list(json.loads((root/'FILES.json').read_text()))+['FILES.json']:tar.add(root/n,arcname=n)
        result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','sources','reviews')}),flush=True)
    return error is None

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--engine',choices=('icarus','xsim','physical','reopen','refine'),required=True);ap.add_argument('--settings',type=Path);ap.add_argument('--toolchain',type=Path)
    ap.add_argument('--donor',type=Path);ap.add_argument('--donor-sha');a=ap.parse_args();raise SystemExit(0 if run(a) else 1)
