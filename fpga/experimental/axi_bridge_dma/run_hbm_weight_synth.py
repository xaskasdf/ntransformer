"""Bounded, isolated XSIM or OOC synthesis of the experimental control plane."""
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
        text = execute('synth', ['vivado', '-mode', 'batch', '-notrace', '-source', source/'synth_hbm_weight_loader.tcl',
                        '-tclargs', source, output], 300)
        if 'WEIGHT_LOADER_OOC_SYNTH_COMPLETE' not in text or not (output/'loader.dcp').is_file():
            raise ValueError('OOC synthesis evidence missing')
    except (OSError, ValueError) as exc:
        error = str(exc)
    unchanged = all(sha(inputs[name]) == sha(source / name) == value for name, value in hashes.items())
    if not unchanged:
        error = (error + '; ' if error else '') + 'Source changed'
    result = dict(status='WEIGHT_LOADER_OOC_EXPORTED' if error is None else 'FAILED',
                  phase='synth', error=error, sources=hashes, sources_unchanged=unchanged,
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
    parser.add_argument('--phase', choices=['synth'], default='synth')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    raise SystemExit(0 if run(parser.parse_args()) else 1)
