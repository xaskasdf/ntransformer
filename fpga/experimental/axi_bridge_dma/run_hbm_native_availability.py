"""Run the availability/real-queue bank on XSIM in an isolated VM directory."""
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
        modules = ['ntrpu_axi_bridge_guard_v3', 'ntrpu_axi_bridge_guard_v3_reset_export',
                   'ntrpu_hbm_startup', 'ntrpu_axi_bridge_hbm_mem_watch', 'ntrpu_axi_bridge_hbm_cdc',
                   'ntrpu_bridge_async_fifo', 'ntrpu_hbm_native_availability', 'tb_hbm_native_availability']
        execute('compile', ['xvlog', '--sv', *[source / (n + '.sv') for n in modules]], 120)
        execute('elaborate', ['xelab', '--debug', 'typical', '--mt', '2', '--snapshot', 'native_availability',
                             'work.tb_hbm_native_availability'], 120)
        for seed in range(4):
            for case in range(20):
                name = f'case-{case:02}-seed-{seed}'
                text = execute(name, ['xsim', 'native_availability', '-runall', '-testplusarg', f'case={case}',
                                     '-testplusarg', f'seed={seed}', '-log', name + '.log'], 120)
                matches = re.findall(r'NA PASS case=(\d+) seed=(\d+) comparisons=(\d+) cold_domains=4 retained_lanes=2', text)
                if len(matches) != 1 or tuple(map(int, matches[0][:2])) != (case, seed) or int(matches[0][2]) <= 40:
                    raise ValueError('Missing/wrong coverage: ' + name)
                completed.append(dict(case=case, seed=seed, comparisons=int(matches[0][2])))
                print(json.dumps(dict(completed=name)), flush=True)
    except (OSError, ValueError) as exc:
        error = str(exc)
    unchanged = all(sha(inputs[n]) == sha(source / n) == h for n, h in hashes.items())
    if not unchanged:
        error = (error + '; ' if error else '') + 'Sources changed'
    result = dict(status='NATIVE_AVAILABILITY_SIM_EXPORTED' if error is None else 'FAILED',
                  error=error, sources=hashes, sources_unchanged=unchanged, steps=steps,
                  completed=completed, seconds=time.monotonic()-started,
                  native_pcie_hbm_traffic=False, physical_timing_qualified=False)
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
