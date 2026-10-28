"""Fast local preflight of the exact prepared XSIM V3 benches and RTL."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from review_guard import review
from review_guard_v2_controls import review_controls
from review_guard_v3 import topology


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def run(output,prepared,toolchain):
    if output.exists():raise ValueError('New output required')
    output.mkdir(parents=True);source=output/'source';source.mkdir();bins=toolchain/'mingw64/bin'
    env=dict(os.environ);env['PATH']=str(bins)+os.pathsep+env.get('PATH','')
    names=['ntrpu_axi_bridge_guard_v2.sv','ntrpu_axi_bridge_guard_v3.sv','ntrpu_axi_bridge_writer.sv','tb_guard_v3_topology.sv']
    names += [f'{kind}_v3_{shared}.sv' for kind in ('legacy','controls') for shared in (0,1)]
    hashes={n:sha(prepared/n) for n in names}
    for n in names:shutil.copy2(prepared/n,source/n)
    for n in ('run_guard_v3_local.py','review_guard_v3.py','review_guard.py','review_guard_v2_controls.py','review_guard_integration.py','review_guard_physical_candidate.py'):
        shutil.copy2(Path(__file__).parent/n,source/n)
    tools={n:sha(bins/n) for n in ('iverilog.exe','vvp.exe')}
    profiles=[(2,5,1),(7,2,2)];reviews=[];steps=[];started=time.monotonic();error=None
    def execute(name,args,cwd,negative=False):
        t=time.monotonic();r=subprocess.run(list(map(str,args)),cwd=cwd,env=env,capture_output=True,text=True,timeout=45)
        log=r.stdout+r.stderr;(output/(name+'.log')).write_text(log)
        steps.append(dict(name=name,returncode=r.returncode,seconds=time.monotonic()-t,command=list(map(str,args))))
        if r.returncode and not negative:raise ValueError('Failed local step: '+name)
        return log
    try:
        for kind in ('legacy','controls','topology'):
            for shared in ((0,1) if kind!='topology' else (0,)):
                top=('tb_ntrpu_axi_bridge_guard' if kind=='legacy' else 'tb_guard_v2_controls')+f'_{shared}' if kind!='topology' else 'tb_guard_v3_topology'
                name=f'{kind}_v3_{shared}' if kind!='topology' else 'topology_v3'
                path=source/(name+'.sv') if kind!='topology' else source/'tb_guard_v3_topology.sv'
                text=path.read_text();text=text.replace('endmodule',f'initial begin $dumpfile("guard.vcd");$dumpvars(1,{top});end\nendmodule')
                local_source=source/(name+'_local.sv');local_source.write_text(text)
                image=output/(name+'.vvp')
                execute('compile-'+name,[bins/'iverilog.exe','-g2012','-Wall','-s',top,'-o',image,source/'ntrpu_axi_bridge_guard_v2.sv',source/'ntrpu_axi_bridge_guard_v3.sv',source/'ntrpu_axi_bridge_writer.sv',local_source],output)
                cases=[(f'{name}-{i}-{case:02}',case,c,0) for i,c in enumerate(profiles) for case in range(17)] if kind=='legacy' else [(f'{name}-{case:02}',case,None,0) for case in range(16)]
                if kind=='topology':cases=[(f'topology-{i}-{bad}',0,c,bad) for i,c in enumerate(profiles) for bad in (0,1)]
                for case_name,case,clocks,bad in cases:
                    out=output/case_name;out.mkdir()
                    args=[bins/'vvp.exe',image,f'+case={case}',f'+bad={bad}']
                    if clocks:
                        args += [f'+{k}={v}' for k,v in zip(('axi_half','mgmt_half','phase'),clocks)]
                    log=execute(case_name,args,out,negative=bool(bad));vcd=(out/'guard.vcd').read_text()
                    checked=topology(log,vcd,bad,clocks) if kind=='topology' else review(log,vcd,case,clocks) if kind=='legacy' else review_controls(log,vcd,case)
                    checked.update(name=case_name,log_sha256=sha(output/(case_name+'.log')),vcd_sha256=sha(out/'guard.vcd'))
                    save(out/'REVIEW.json',checked);reviews.append(checked)
    except (ValueError,OSError,subprocess.TimeoutExpired) as exc:error=str(exc)
    unchanged=hashes=={n:sha(prepared/n) for n in names} and tools=={n:sha(bins/n) for n in tools}
    if not unchanged:error='Inputs changed'
    result=dict(status='GUARD_V3_RTL_TRACE_PASS' if error is None else 'FAILED',error=error,tool='Icarus local preflight',sources=hashes,sources_unchanged=unchanged,toolchain=tools,steps=steps,reviews=reviews,seconds=time.monotonic()-started)
    save(output/'RESULT.json',result);save(output/'FILES.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps(dict(status=result['status'],error=error,cases=len(reviews),seconds=result['seconds'])))
    return error is None


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--prepared-source',type=Path,required=True);ap.add_argument('--toolchain',type=Path,required=True)
    a=ap.parse_args();raise SystemExit(0 if run(a.output.resolve(),a.prepared_source.resolve(),a.toolchain.resolve()) else 1)
