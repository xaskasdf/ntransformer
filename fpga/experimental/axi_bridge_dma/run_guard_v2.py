"""Private Vivado/XSIM validation of the isolated guard V2 candidate."""
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
    names=['ntrpu_axi_bridge_guard_v2.sv','run_guard_v2.py','review_guard.py','review_guard_v2_controls.py']
    if a.phase=='rtl':names+=['ntrpu_axi_bridge_writer.sv','tb_ntrpu_axi_bridge_guard.sv','tb_guard_v2_controls.sv','run_guard_v2_trace.tcl']
    else:names+=['audit_guard_cdc.tcl','synth_guard_v2.tcl']
    hashes={name:sha(here/name) for name in names}
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for name in names:shutil.copy2(here/name,source/name)
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    steps=[];reviews=[];started=time.monotonic();error=None
    def execute(name,args,limit,env=None):
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
        if code or expired or re.search(r'(?im)^\s*(?:fatal|error):',text):raise ValueError('Failed step: '+name)
        return text
    try:
        if a.phase=='rtl':
            if hashes['ntrpu_axi_bridge_writer.sv']!='885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6':raise ValueError('Writer changed')
            bench=(source/'tb_ntrpu_axi_bridge_guard.sv').read_text()
            if bench.count('ntrpu_axi_bridge_guard #(')!=1:raise ValueError('Unknown legacy bench binding')
            bench=bench.replace('ntrpu_axi_bridge_guard #(','ntrpu_axi_bridge_guard_v2 #(')
            # XSIM supports only one VCD: Tcl owns it; keep the original bench frozen.
            bench,dumps=re.subn(r'\$dump(?:file|vars)\([^;]*\);','/* VCD owned by run_guard_v2_trace.tcl */',bench)
            if dumps!=2:raise ValueError('Unexpected legacy VCD instrumentation')
            (source/'bound_legacy_tb.sv').write_text(bench)
            execute('compile',['xvlog','--sv','--work','xil_defaultlib',source/'ntrpu_axi_bridge_guard_v2.sv',source/'ntrpu_axi_bridge_writer.sv',source/'bound_legacy_tb.sv',source/'tb_guard_v2_controls.sv'],60)
            for top,snapshot in [('tb_ntrpu_axi_bridge_guard','legacy_v2'),('tb_guard_v2_controls','controls_v2')]:
                execute('elaborate-'+snapshot,['xelab','--debug','typical','--mt','2','--snapshot',snapshot,'xil_defaultlib.'+top],90)
            profiles=[(2,5,1),(4,7,3),(7,2,2),(2,2,5)]
            cases=[(f'legacy-{i}-{case:02}','legacy_v2','tb_ntrpu_axi_bridge_guard',case,clocks) for i,clocks in enumerate(profiles) for case in range(17)]
            cases += [(f'controls-{case:02}','controls_v2','tb_guard_v2_controls',case,None) for case in range(16)]
            for name,snapshot,top,case,clocks in cases:
                out=root/name
                args=['xsim',snapshot,'-tclbatch',source/'run_guard_v2_trace.tcl','-testplusarg',f'case={case}','-log',name+'.log']
                if clocks:
                    for key,value in zip(('axi_half','mgmt_half','phase'),clocks):args+=['-testplusarg',f'{key}={value}']
                log=execute(name,args,30,dict(os.environ,NTRPU_V2_OUT=str(out),NTRPU_V2_TOP=top))
                vcd=(out/'guard.vcd').read_text()
                checked=review(log,vcd,case,clocks) if clocks else review_controls(log,vcd,case)
                checked.update(name=name,log_sha256=sha(root/(name+'.log')),vcd_sha256=sha(out/'guard.vcd'))
                save(out/'REVIEW.json',checked);reviews.append(checked)
                print(json.dumps(dict(completed=len(reviews),total=len(cases),case=name)),flush=True)
        else:
            execute('synthesis',['vivado','-mode','batch','-source',source/'synth_guard_v2.tcl','-journal',root/'vivado.jou','-log',root/'vivado.log','-tclargs',source/'ntrpu_axi_bridge_guard_v2.sv',root/'reports'],300)
            if not (root/'reports/STATUS.txt').is_file():raise ValueError('Missing synthesis verdict')
    except (OSError,ValueError,RuntimeError) as exc:error=str(exc)
    unchanged=hashes=={name:sha(here/name) for name in names}
    if not unchanged:error='Staged inputs changed'
    result=dict(status=('GUARD_V2_RTL_TRACE_PASS' if a.phase=='rtl' else 'GUARD_V2_SYNTHESIS_EXPORTED') if error is None else 'FAILED',
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
    ap.add_argument('--phase',choices=('rtl','cdc'),required=True);ap.add_argument('--execute-native',action='store_true')
    a=ap.parse_args()
    if not a.execute_native:ap.error('--execute-native required')
    raise SystemExit(0 if run(a) else 1)
