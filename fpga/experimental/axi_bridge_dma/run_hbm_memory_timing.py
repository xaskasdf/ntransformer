"""Bounded post-route optimization and independent reopen in the side namespace."""
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

HELPERS = ('constrain_hbm_clocktree.tcl', 'insert_hbm_clocktree.tcl',
           'audit_hbm_clocktree.tcl', 'audit_hbm_clocktree_topology.tcl',
           'run_hbm_clocktree.tcl')
INPUTS = HELPERS + ('refine_hbm_memory_timing.tcl', 'run_hbm_memory_timing.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def seal(root):
    save(root / 'FILES.json', {p.relative_to(root).as_posix(): sha(p)
         for p in sorted(root.rglob('*')) if p.is_file()
         and 'work' not in p.relative_to(root).parts and '.Xil' not in p.parts
         and p != root / 'FILES.json'})


def run(args):
    root, prepared, donor = args.output.resolve(), args.prepared.resolve(), args.donor.resolve()
    namespace = Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-memory-timing').resolve()
    if os.name != 'posix' or not root.is_relative_to(namespace) or root == namespace or root.exists():
        raise ValueError('New private VM output required')
    if sha(donor) != args.donor_sha:
        raise ValueError('Input checkpoint checksum mismatch')
    hashes = {n: sha(prepared / n) for n in INPUTS}
    root.mkdir(parents=True)
    source, work = root / 'source', root / 'work'
    source.mkdir(); work.mkdir()
    for name in INPUTS:
        shutil.copy2(prepared / name, source / name)
    shutil.copy2(donor, source / 'donor.dcp')
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    started = time.monotonic()
    steps, error = [], None

    def execute(name, script, tclargs, log_root):
        command = ['vivado', '-mode', 'batch', '-source', str(script),
                   '-journal', str(log_root / 'vivado.jou'), '-log', str(log_root / 'vivado.log'),
                   '-tclargs', *map(str, tclargs)]
        command = ['bash', '-c', 'source ' + shlex.quote(str(args.settings)) + ' && exec ' + shlex.join(command)]
        with (root / (name + '.log')).open('w') as log:
            t = time.monotonic()
            child = subprocess.Popen(command, cwd=work, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(pid=child.pid, pgid=child.pid, step=name, timeout=600)
            save(root / 'own_process.json', record)
            with (root / 'own_processes.jsonl').open('a') as record_file:
                record_file.write(json.dumps(record) + '\n')
            expired = False
            try:
                code = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                expired = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
        steps.append(dict(name=name, returncode=code, timed_out=expired, seconds=time.monotonic()-t))
        save(root / 'steps.json', steps)
        log_text = (root / (name + '.log')).read_text(errors='replace')
        if code or expired or re.search(r'(?im)^\s*(?:fatal|error):', log_text):
            raise ValueError('Failed step: ' + name)

    try:
        execute('refinement', source / 'refine_hbm_memory_timing.tcl',
                [source / 'donor.dcp', root / 'refinement', args.strategy], root)
        phase = root / 'physical'; phase.mkdir()
        phase_source = phase / 'source'; phase_source.mkdir()
        for name in HELPERS:
            shutil.copy2(source / name, phase_source / name)
        refined = root / 'refinement/refined.dcp'
        shutil.copy2(refined, phase_source / 'donor.dcp')
        phase_error = None
        try:
            execute('reopen', phase_source / 'run_hbm_clocktree.tcl',
                    [phase_source, phase_source / 'donor.dcp', phase / 'output', 'reopen'], phase)
        except (OSError, ValueError) as exc:
            phase_error = str(exc)
        step = steps[-1]
        save(phase / 'RESULT.json', dict(status='HBM_CLOCKTREE_EXPORTED' if phase_error is None else 'FAILED',
             mode='reopen', error=phase_error, returncode=step['returncode'], timed_out=step['timed_out'],
             donor_sha256=sha(refined), sources={n: hashes[n] for n in HELPERS},
             sources_unchanged=all(sha(phase_source / n) == hashes[n] for n in HELPERS)))
        seal(phase)
        if phase_error:
            raise ValueError(phase_error)
    except (OSError, ValueError, RuntimeError) as exc:
        error = str(exc)
    unchanged = all(sha(prepared / n) == sha(source / n) == h for n, h in hashes.items())
    if not unchanged or sha(donor) != args.donor_sha or sha(source / 'donor.dcp') != args.donor_sha:
        error = 'Input mutation detected'
    result = dict(status='HBM_MEMORY_TIMING_EXPORTED' if error is None else 'FAILED', error=error,
                  strategy=args.strategy, donor_sha256=args.donor_sha, sources=hashes,
                  sources_unchanged=unchanged, steps=steps, seconds=time.monotonic()-started,
                  native_hbm_pcie_ip=False, board_qualified=False)
    save(root / 'RESULT.json', result); seal(root)
    archive = Path(str(root) + '.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for name in list(json.loads((root / 'FILES.json').read_text())) + ['FILES.json']:
            tar.add(root / name, arcname=name)
    receipt = {k: v for k, v in result.items() if k not in ('steps', 'sources')}
    receipt.update(archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)
    print(json.dumps(receipt), flush=True)
    return error is None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--donor', type=Path, required=True)
    parser.add_argument('--donor-sha', required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--strategy', choices=('replicate', 'restructure', 'explore'), required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
