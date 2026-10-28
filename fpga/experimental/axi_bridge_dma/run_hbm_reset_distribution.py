"""Validate isolated HBM reset distribution, real consumer queues and native primitive topology."""
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


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2) + '\n')


def run(args):
    root = args.output.resolve(); prepared = args.prepared.resolve()
    ns = Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-native-join').resolve()
    if os.name != 'posix' or root.exists() or not root.is_relative_to(ns) or root == ns:
        raise ValueError('New private output required')
    root.mkdir(parents=True); source = root / 'source'; source.mkdir()
    work = root / 'work'; work.mkdir()
    inputs = {p.name: p for p in prepared.iterdir() if p.is_file()
              and p.suffix in ('.sv', '.v', '.vh', '.py', '.tcl', '.xci', '.json', '.tsv')}
    hashes = {n: sha(p) for n, p in inputs.items()}
    for n, p in inputs.items():
        shutil.copy2(p, source / n)
    os.nice(10); os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    steps = []; error = None; started = time.monotonic(); completed = []

    def execute(name, argv, limit):
        command = 'source ' + shlex.quote(str(args.settings)) + ' && exec ' + shlex.join(list(map(str, argv)))
        with (root / (name + '.log')).open('w') as log:
            child = subprocess.Popen(['bash', '-c', command], cwd=work, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with (root / 'own_processes.jsonl').open('a') as f:
                f.write(json.dumps(dict(pid=child.pid, pgid=child.pid, step=name, timeout=limit)) + '\n')
            expired = False; start = time.monotonic()
            try:
                code = child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                expired = True; os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); code = child.wait()
        steps.append(dict(name=name, returncode=code, timed_out=expired, seconds=time.monotonic()-start))
        save(root / 'steps.json', steps)
        text = (root / (name + '.log')).read_text(errors='replace')
        if code or expired or re.search(r'(?im)^\s*(?:error|fatal):', text):
            raise ValueError('Failed step: ' + name)
        return text

    try:
        modules = ['ntrpu_hbm_startup_distributed', 'ntrpu_hbm_startup_reference',
                   'ntrpu_hbm_read_adapter', 'tb_hbm_reset_distribution', 'tb_hbm_cold_release']
        execute('compile', ['xvlog','--sv',*[source/(n+'.sv') for n in modules]],120)
        def simulate(label, top, snapshot):
            execute(label+'-elaborate',['xelab','--debug','typical','--mt','2','--snapshot',snapshot,'work.'+top],120)
            return execute(label+'-run',['xsim',snapshot,'-runall'],120)
        text=simulate('positive','tb_hbm_reset_distribution','distribution_positive')
        cases=re.findall(r'RD CASE PASS case=(\d+) checks=(\d+) groups=(\d+) popped=(\d+)',text)
        if sorted(int(a) for a,_,_,_ in cases)!=[0,1,2,3] or 'RD PASS cases=4' not in text:
            raise ValueError('Missing distribution coverage')
        completed=[dict(case=int(a),checks=int(b),groups=int(c),popped=int(d)) for a,b,c,d in cases]
        if any(c['popped']!=6 or c['groups']<15 for c in completed):raise ValueError('Incomplete queue/reset coverage')
        regression=simulate('regression','tb_hbm_cold_release','distribution_regression')
        if 'RC PASS cases=4' not in regression or len(re.findall('RC CASE PASS',regression))!=4:
            raise ValueError('Original cold-release regression incomplete')
        rtl=(source/'ntrpu_hbm_startup_distributed.sv').read_text()
        mutations=[
            ('bypass','assign mem_rst_n=memory_distribution;','assign mem_rst_n=memory_release[2];',
             'distribution release at fourth edge'),
            ('synchronous_assert','always @(posedge mem_clk or negedge memory_release[2])begin',
             'always @(posedge mem_clk)begin','asynchronous assertion without clocks'),
            ('fault_reset','assign mem_rst_n=memory_distribution;',
             'assign mem_rst_n=memory_distribution && !fault;','fault must preserve reset epoch')]
        rejected=[]
        for label,anchor,replacement,reason in mutations:
            if rtl.count(anchor)!=1:raise ValueError('Mutation anchor differs: '+label)
            mutation=root/('mutation-'+label);mutation.mkdir()
            bad=mutation/'ntrpu_hbm_startup_distributed.sv';bad.write_text(rtl.replace(anchor,replacement,1))
            execute(label+'-compile',['xvlog','--sv',bad],120)
            execute(label+'-elaborate',['xelab','--debug','typical','--mt','2','--snapshot',label,
                                      'work.tb_hbm_reset_distribution'],120)
            failed=False
            try:execute(label+'-run',['xsim',label,'-runall'],120)
            except ValueError:failed=True
            text=(root/(label+'-run.log')).read_text()
            if not failed or reason not in text or steps[-1]['timed_out']:
                raise ValueError('Mutation not rejected for intended invariant: '+label)
            rejected.append(dict(name=label,reason=reason,source_sha256=sha(bad)))
        save(root/'NEGATIVE_CONTROLS.json',rejected)
        execute('synthesis',['vivado','-mode','batch','-source',source/'synthesize_hbm_reset_distribution.tcl',
                '-log',root/'synthesis-vivado.log','-journal',root/'synthesis-vivado.jou',
                '-tclargs',source,root/'output'],180)
        if (root/'output/STATUS.txt').read_text().strip()!='HBM_RESET_DISTRIBUTION_TOPOLOGY_EXPORTED':
            raise ValueError('Native topology proof missing')
    except (OSError, ValueError) as exc:
        error = str(exc)
    unchanged = all(sha(inputs[n]) == sha(source / n) == h for n, h in hashes.items())
    if not unchanged:
        error = (error + '; ' if error else '') + 'Sources changed'
    result = dict(status='HBM_RESET_DISTRIBUTION_SIM_EXPORTED' if error is None else 'FAILED',
                  error=error, sources=hashes, sources_unchanged=unchanged, steps=steps,
                  completed=completed, seconds=time.monotonic()-started,
                  native_pcie_hbm_traffic=False, physical_timing_qualified=False, native_frontend_integrated=False)
    save(root / 'RESULT.json', result)
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and 'work' not in p.relative_to(root).parts}
    save(root / 'FILES.json', files)
    archive = Path(str(root) + '.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for name in list(files) + ['FILES.json']:
            tar.add(root / name, arcname=name)
    print(json.dumps(dict(status=result['status'], archive=str(archive), archive_sha256=sha(archive),
                          archive_bytes=archive.stat().st_size)), flush=True)
    return error is None


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('output', type=Path)
    p.add_argument('--prepared', type=Path, required=True); p.add_argument('--settings', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)
