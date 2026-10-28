"""Local, isolated RTL regression; no Vivado, SSH, driver or board changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from check_ntrpu_xdma_completion_guard import review, negative_controls

ROOT = Path(__file__).resolve().parents[1]
RTL = ['fpga/rtl/ntrpu_xdma_local_master.sv',
       'fpga/experimental/ntrpu_xdma_completion_guard.sv',
       'fpga/tb/tb_ntrpu_xdma_completion_guard.sv']
SOURCES = RTL + ['tools/run_ntrpu_xdma_completion_guard.py',
                 'tools/check_ntrpu_xdma_completion_guard.py',
                 'tools/check_ntrpu_xdma_stop_inventory_r4.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, toolchain, seeds):
    output.mkdir(parents=True,exist_ok=False)
    bins = toolchain/'mingw64/bin'
    env = dict(os.environ)
    windows = Path(env.get('SystemRoot','C:/Windows'))
    env['PATH'] = os.pathsep.join(map(str,[bins,windows/'System32',windows]))
    initial = {n:sha(ROOT/n) for n in SOURCES}
    chain = {p.relative_to(toolchain).as_posix():sha(p) for p in toolchain.rglob('*') if p.is_file()}
    for name in SOURCES:
        dest=output/'repo'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    result = dict(status='FAILED', sources=initial, toolchain=chain, runs=[],
                  native_xdma_integrated=False, hbm_integrated=False,
                  operational_reset_qualified=False, physical_timing_qualified=False)
    start=time.monotonic()

    def execute(command,cwd,log):
        job=subprocess.run(list(map(str,command)),cwd=cwd,env=env,text=True,capture_output=True,timeout=30)
        text=job.stdout+job.stderr;(cwd/log).write_text(text,encoding='utf-8')
        if job.returncode:raise ValueError(f'{log}: exit {job.returncode}')
        return text

    try:
        execute([bins/'iverilog.exe','-V'],output,'compiler-version.log')
        execute([bins/'vvp.exe','-V'],output,'simulator-version.log')
        log=execute([bins/'iverilog.exe','-g2012','-Wall','-s','tb_ntrpu_xdma_completion_guard',
                     '-o',output/'guard.vvp',*[output/'repo'/n for n in RTL]],output,'compile.log')
        warnings=[line for line in log.splitlines() if 'warning:' in line]
        if any('is sensitive to all 2 words in array' not in line for line in warnings):
            raise ValueError('Unexpected compile warning')
        result['compile_warnings']=warnings
        for seed in range(seeds):
            case=output/f'seed-{seed:02}';case.mkdir()
            log=execute([bins/'vvp.exe',output/'guard.vvp',f'+seed={seed}'],case,'simulation.log')
            checked,rows=review(log,(case/'guard.vcd').read_text())
            result['runs'].append(dict(seed=seed,cases=checked))
            if seed==0:result['negative_controls']=negative_controls(rows)
        result['sources_after']={n:sha(ROOT/n) for n in SOURCES}
        result['toolchain_after']={p.relative_to(toolchain).as_posix():sha(p) for p in toolchain.rglob('*') if p.is_file()}
        if result['sources_after']!=initial or result['toolchain_after']!=chain:
            raise ValueError('Source/toolchain changed during regression')
        result['status']='COMPLETION_GUARD_RTL_PASS'
        result['scenarios']=sum(len(r['cases']) for r in result['runs'])
    except (ValueError,OSError,subprocess.TimeoutExpired) as error:
        result['error']=str(error)
    result['seconds']=round(time.monotonic()-start,6)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'FILES.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--toolchain',type=Path,required=True)
    ap.add_argument('--seeds',type=int,default=8)
    args=ap.parse_args()
    if not 1<=args.seeds<=32:ap.error('--seeds must be 1..32')
    result=run(args.output.resolve(),args.toolchain.resolve(),args.seeds)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','sources_after','toolchain','toolchain_after','runs','compile_warnings')}))
    raise SystemExit(0 if result['status']=='COMPLETION_GUARD_RTL_PASS' else 1)
