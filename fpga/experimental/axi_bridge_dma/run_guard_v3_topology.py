"""Private Vivado/XSIM validation of the guard V3 shared and independent profiles."""
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

from review_guard import review
from review_guard_v2_controls import review_controls


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def run(a):
    allowed=Path('/home/ubuntu/vivado/ntrpu-side').resolve();root=a.output.resolve()
    if os.name!='posix' or not root.is_relative_to(allowed):raise ValueError('Private Linux namespace required')
    archive=Path(str(root)+'.tar.gz')
    if root.exists() or archive.exists() or not a.settings.is_file():raise ValueError('New output and Vivado required')
    here=Path(__file__).resolve().parent
    names=['ntrpu_axi_bridge_guard_v3.sv','ntrpu_axi_bridge_guard_v2.sv','run_guard_v3_topology.py','review_guard.py','review_guard_v2_controls.py']
    if a.phase=='rtl':names+=['ntrpu_axi_bridge_writer.sv','tb_ntrpu_axi_bridge_guard.sv','tb_guard_v2_controls.sv','run_guard_v2_trace.tcl','tb_guard_v3_topology.sv']
    else:names+=['audit_guard_cdc.tcl','synth_guard_v2.tcl']
    hashes={name:sha(here/name) for name in names}
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for name in names:shutil.copy2(here/name,source/name)
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    steps=[];reviews=[];started=time.monotonic();error=None
    def execute(name,args,limit,env=None,expected_failure=False):
        shell='source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(list(map(str,args)))
        t=time.monotonic();expired=False
        with (root/(name+'.log')).open('w') as log:
            p=subprocess.Popen(['bash','-c',shell],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'own_process.json',dict(pid=p.pid,pgid=p.pid,step=name,timeout=limit))
            try:code=p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
        text=(root/(name+'.log')).read_text(errors='replace')
        step=dict(name=name,returncode=code,timed_out=expired,seconds=time.monotonic()-t,command=list(map(str,args)))
        steps.append(step);save(root/'steps.json',steps)
        if not expected_failure and (code or expired or re.search(r'(?im)^\s*(?:fatal|error):',text)):raise ValueError('Failed step: '+name)
        if expected_failure and (expired or 'V3_SHARED_RESET_CONTRACT' not in text):raise ValueError('Expected contract rejection missing: '+name)
        return text
    try:
        if a.phase=='rtl':
            if hashes['ntrpu_axi_bridge_writer.sv']!='885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6':raise ValueError('Writer changed')
            profiles=[(2,5,1),(7,2,2)]
            execute('compile-topology',['xvlog','--sv','--work','xil_defaultlib',source/'ntrpu_axi_bridge_guard_v2.sv',source/'ntrpu_axi_bridge_guard_v3.sv',source/'tb_guard_v3_topology.sv'],60)
            execute('elaborate-topology',['xelab','--debug','typical','--mt','2','--snapshot','topology_v3','xil_defaultlib.tb_guard_v3_topology'],90)
            for i,clocks in enumerate(profiles):
                for bad in (0,1):
                    name=f'topology-{i}-{bad}';out=root/name
                    args=['xsim','topology_v3','-tclbatch',source/'run_guard_v2_trace.tcl','-testplusarg',f'bad={bad}','-log',name+'.log']
                    for key,value in zip(('axi_half','mgmt_half','phase'),clocks):args+=['-testplusarg',f'{key}={value}']
                    log=execute(name,args,90,dict(os.environ,NTRPU_V2_OUT=str(out),NTRPU_V2_TOP='tb_guard_v3_topology'),expected_failure=bool(bad))
                    if not bad and 'V3_TOPOLOGY_PASS' not in log:raise ValueError('Missing topology verdict')
                    checked=dict(name=name,expected_contract_rejection=bool(bad),log_sha256=sha(root/(name+'.log')),vcd_sha256=sha(out/'guard.vcd'))
                    save(out/'REVIEW.json',checked);reviews.append(checked)
        else:
            execute('synthesis',['vivado','-mode','batch','-source',source/'synth_guard_v2.tcl','-journal',root/'vivado.jou','-log',root/'vivado.log','-tclargs',source/'ntrpu_axi_bridge_guard_v2.sv',root/'reports'],300)
            if not (root/'reports/STATUS.txt').is_file():raise ValueError('Missing synthesis verdict')
    except (OSError,ValueError,RuntimeError) as exc:error=str(exc)
    unchanged=hashes=={name:sha(here/name) for name in names}
    if not unchanged:error='Staged inputs changed'
    result=dict(status=('GUARD_V3_TOPOLOGY_TRACE_PASS' if a.phase=='rtl' else 'GUARD_V2_SYNTHESIS_EXPORTED') if error is None else 'FAILED',
                error=error,phase=a.phase,steps=steps,reviews=reviews,seconds=time.monotonic()-started,sources=hashes,sources_unchanged=unchanged,
                cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),physical_cdc_validated=False)
    save(root/'RESULT.json',result)
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(work)]
    save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in paths})
    with tarfile.open(archive,'x:gz') as tar:
        for path in paths+[root/'FILES.json']:tar.add(path,arcname=path.relative_to(root).as_posix())
    result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps(result),flush=True)
    return error is None


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);ap.add_argument('--settings',type=Path,required=True)
    ap.add_argument('--phase',choices=('rtl',),required=True);ap.add_argument('--execute-native',action='store_true')
    a=ap.parse_args()
    if not a.execute_native:ap.error('--execute-native required')
    raise SystemExit(0 if run(a) else 1)
