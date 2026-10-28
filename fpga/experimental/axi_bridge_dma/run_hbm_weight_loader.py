"""Bounded, isolated XSIM or OOC synthesis of the experimental HBM weight loader."""
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(args):
    root = args.output.resolve()
    prepared = args.prepared.resolve()
    namespace = Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-weight-loader').resolve()
    if os.name != 'posix' or root.exists() or root == namespace or not root.is_relative_to(namespace):
        raise ValueError('A new output inside the private side-thread namespace is required')
    root.mkdir(parents=True)
    source = root / 'source'; source.mkdir()
    work = root / 'work'; work.mkdir()
    output = root / 'output'; output.mkdir()
    inputs = {p.name: p for p in prepared.iterdir() if p.is_file()}
    hashes = {name: sha(path) for name, path in inputs.items()}
    for name, path in inputs.items():
        shutil.copy2(path, source / name)
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    started = time.monotonic(); steps = []; completed = []; error = None

    def execute(name, argv, limit):
        command = 'source ' + shlex.quote(str(args.settings)) + ' && exec ' + shlex.join(list(map(str, argv)))
        start = time.monotonic()
        with (root / (name + '.log')).open('w') as log:
            child = subprocess.Popen(['bash', '-c', command], cwd=work, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with (root / 'own_processes.jsonl').open('a') as records:
                records.write(json.dumps(dict(pid=child.pid, pgid=child.pid, step=name, timeout=limit)) + '\n')
            expired = False
            try:
                code = child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                expired = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
        steps.append(dict(name=name, returncode=code, timed_out=expired, seconds=time.monotonic()-start))
        save(root / 'steps.json', steps)
        text = (root / (name + '.log')).read_text(errors='replace')
        if code or expired or re.search(r'(?im)^\s*(?:error|fatal):', text):
            raise ValueError('Failed step: ' + name)
        return text

    try:
        modules = ['ntrpu_control_snapshot', 'ntrpu_hbm_control_plane',
                   'ntrpu_hbm_weight_loader', 'tb_hbm_weight_loader']
        for simulator in args.simulators:
            if simulator == 'icarus':
                execute('compile-icarus', ['iverilog', '-g2012', '-s', 'tb_hbm_weight_loader', '-o', 'loader.vvp',
                        *[source / (n + '.sv') for n in modules]], 120)
            else:
                execute('compile-xsim', ['xvlog', '--sv', *[source / (n + '.sv') for n in modules]], 120)
                execute('elaborate-xsim', ['xelab', '--debug', 'wave', '--mt', '2', '--snapshot', 'loader',
                        'work.tb_hbm_weight_loader'], 120)
            for seed in args.seeds:
                name = f'{simulator}-seed-{seed}'
                argv = ['vvp', 'loader.vvp', f'+seed={seed}'] if simulator == 'icarus' else [
                    'xsim', 'loader', '-runall', '-testplusarg', f'seed={seed}', '-log', name + '.log']
                text = execute(name, argv, 120)
                match = re.findall(r'WL PASS seed=(\d+) groups=(\d+) checks=(\d+) aw=(\d+) w=(\d+) b=(\d+)', text)
                if len(match) != 1 or int(match[0][0]) != seed or int(match[0][1]) != 8 or int(match[0][2]) < 100:
                    raise ValueError('Missing/incorrect simulation coverage: ' + name)
                if simulator == 'icarus':shutil.copy2(work/'loader.vcd', output/(name+'.vcd'))
                completed.append(dict(simulator=simulator, seed=seed, groups=int(match[0][1]), checks=int(match[0][2]),
                                      aw=int(match[0][3]), w=int(match[0][4]), b=int(match[0][5])))
                print(json.dumps(dict(completed=name)), flush=True)
    except (OSError, ValueError) as exc:
        error = str(exc)
    unchanged = all(sha(inputs[name]) == sha(source / name) == value for name, value in hashes.items())
    if not unchanged:
        error = (error + '; ' if error else '') + 'Source changed'
    result = dict(status='WEIGHT_LOADER_PREFLIGHT_EXPORTED' if error is None else 'FAILED',
                  phase='sim', error=error, sources=hashes, sources_unchanged=unchanged,
                  steps=steps, completed=completed, seeds=args.seeds,
                  seconds=time.monotonic()-started, native_bar_attached=False,
                  physical_cdc_qualified=False, native_pcie_hbm_traffic=False)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--simulators', choices=['icarus', 'xsim'], nargs='+', default=['xsim'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 3])
    raise SystemExit(0 if run(parser.parse_args()) else 1)
