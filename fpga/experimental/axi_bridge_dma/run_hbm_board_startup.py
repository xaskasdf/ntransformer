"""Run isolated board-startup RTL, an actual negative mutation, and pad synthesis."""
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
        modules = ['ntrpu_hbm_board_startup', 'tb_hbm_board_startup']
        execute('compile', ['xvlog', '--sv', *[source / (n + '.sv') for n in modules]], 120)
        execute('elaborate', ['xelab', '--debug', 'typical', '--mt', '2', '--snapshot', 'board_startup',
                             'work.tb_hbm_board_startup'], 120)
        text = execute('positive', ['xsim', 'board_startup', '-runall'], 120)
        cases = re.findall(r'BS CASE PASS hold=(\d+) stable=(\d+) checks=(\d+) groups=(\d+)', text)
        if sorted((int(a),int(b)) for a,b,_,_ in cases) != [(1,1),(4,3),(16,32),(17,9)] or 'BS PASS cases=4' not in text:
            raise ValueError('Missing startup coverage')
        completed = [dict(hold=int(a),stable=int(b),checks=int(c),groups=int(d)) for a,b,c,d in cases]
        # Actual RTL mutation: lock loss illegally clears the cold epoch.
        mutation = root/'mutation'; mutation.mkdir()
        rtl = (source/'ntrpu_hbm_board_startup.sv').read_text()
        anchor = 'else if (!locked_sync[2]) lock_fault <= 1;'
        if rtl.count(anchor) != 1: raise ValueError('Mutation anchor changed')
        bad = mutation/'ntrpu_hbm_board_startup.sv'
        bad.write_text(rtl.replace(anchor,'else if (!locked_sync[2]) begin lock_fault <= 1; configuration_rst_n <= 0; end'))
        execute('mutation-compile', ['xvlog','--sv',bad,source/'tb_hbm_board_startup.sv'],120)
        execute('mutation-elaborate', ['xelab','--debug','typical','--mt','2','--snapshot','board_mutant','work.tb_hbm_board_startup'],120)
        rejected = False
        try: execute('mutation-run',['xsim','board_mutant','-runall'],120)
        except ValueError:
            rejected = True
        bad_log=(root/'mutation-run.log').read_text()
        if not rejected or 'lock loss preserves epoch' not in bad_log or steps[-1]['timed_out']:
            raise ValueError('Mutation was not rejected for the intended invariant')
        execute('synth',['vivado','-mode','batch','-source',source/'synth_hbm_board_frontend.tcl',
                         '-tclargs',source,root/'output'],300)
    except (OSError, ValueError) as exc:
        error = str(exc)
    unchanged = all(sha(inputs[n]) == sha(source / n) == h for n, h in hashes.items())
    if not unchanged:
        error = (error + '; ' if error else '') + 'Sources changed'
    result = dict(status='BOARD_STARTUP_SIM_EXPORTED' if error is None else 'FAILED',
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
