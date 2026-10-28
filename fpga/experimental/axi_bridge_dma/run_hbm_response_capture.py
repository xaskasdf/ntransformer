"""Run bounded private XSIM tests and VU47P synthesis; archive success and failure."""
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
    root = args.output.resolve()
    prepared = args.prepared.resolve()
    ns = Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-native-join').resolve()
    if os.name != 'posix' or root.exists() or not root.is_relative_to(ns) or root == ns:
        raise ValueError('New private output required')
    root.mkdir(parents=True)
    source = root / 'source'; source.mkdir()
    work = root / 'work'; work.mkdir()
    inputs = {p.name: p for p in prepared.iterdir() if p.is_file() and p.suffix in ('.sv', '.py', '.tcl', '.json')}
    hashes = {n: sha(p) for n, p in inputs.items()}
    for n, p in inputs.items():
        shutil.copy2(p, source / n)
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    steps = []; error = None; started = time.monotonic(); rejected = []

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
        modules = ['reference_queue', 'reference_tracker', 'candidate_tracker',
                   'ntrpu_hbm_response_capture_queue', 'tb_hbm_response_capture', 'tb_hbm_capture_tracker']
        execute('compile', ['xvlog', '--sv', *[source / (n + '.sv') for n in modules]], 120)
        for label, top in [('queue', 'tb_hbm_response_capture'), ('tracker', 'tb_hbm_capture_tracker')]:
            execute(label+'-elaborate', ['xelab', '--debug', 'typical', '--mt', '2', '--snapshot', label, 'work.'+top], 120)
            log = execute(label+'-run', ['xsim', label, '-runall'], 120)
            if ('CAPTURE_'+label.upper()+' ALL PASS') not in log:
                raise ValueError('Missing simulation completion: '+label)
        rtl = (source / 'ntrpu_hbm_response_capture_queue.sv').read_text()
        mutations = [
            ('ownership', 'if(push)write_slot<=!write_slot;', 'if(s_ready)write_slot<=!write_slot;', 'ownership equivalence'),
            ('occupied', 'if(write_slot)slot1<=s_data;', 'if(!write_slot)slot1<=s_data;', 'slot'),
            ('full', 'if(s_ready)begin', 'if(rst_n)begin', 'full/reset payload frozen')]
        for label, anchor, replacement, reason in mutations:
            if rtl.count(anchor) != 1:
                raise ValueError('Mutation anchor differs: '+label)
            directory = root / ('mutation-'+label); directory.mkdir()
            bad = directory / 'ntrpu_hbm_response_capture_queue.sv'
            bad.write_text(rtl.replace(anchor, replacement, 1))
            execute(label+'-compile', ['xvlog', '--sv', bad], 120)
            execute(label+'-elaborate', ['xelab', '--debug', 'typical', '--mt', '2', '--snapshot', label,
                                      'work.tb_hbm_response_capture'], 120)
            failed = False
            try:
                execute(label+'-run', ['xsim', label, '-runall'], 120)
            except ValueError:
                failed = True
            log = (root / (label+'-run.log')).read_text()
            if not failed or 'CAPTURE_QUEUE width=' not in log or reason not in log or steps[-1]['timed_out']:
                raise ValueError('Mutation did not fail for intended invariant: '+label)
            rejected.append(dict(name=label, reason=reason, sha256=sha(bad)))
        save(root / 'NEGATIVE_CONTROLS.json', rejected)
        execute('synthesis', ['vivado', '-mode', 'batch', '-source', source / 'synthesize_hbm_response_capture.tcl',
                '-log', root / 'synthesis-vivado.log', '-journal', root / 'synthesis-vivado.jou',
                '-tclargs', source, root / 'output'], 300)
        if (root / 'output/STATUS.txt').read_text().strip() != 'HBM_RESPONSE_CAPTURE_TOPOLOGY_EXPORTED':
            raise ValueError('Missing native synthesis completion')
    except (OSError, ValueError) as exc:
        error = str(exc)
    unchanged = all(sha(inputs[n]) == sha(source / n) == h for n, h in hashes.items())
    if not unchanged:
        error = (error+'; ' if error else '')+'Sources changed'
    save(root / 'RESULT.json', dict(status='HBM_RESPONSE_CAPTURE_EXPORTED' if error is None else 'FAILED',
         error=error, sources=hashes, sources_unchanged=unchanged, steps=steps, negative_controls=rejected,
         seconds=time.monotonic()-started, full_board_integrated=False, physical_timing_qualified=False,
         native_pcie_hbm_traffic=False))
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and 'work' not in p.relative_to(root).parts}
    save(root / 'FILES.json', files)
    archive = Path(str(root)+'.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for name in list(files)+['FILES.json']:
            tar.add(root / name, arcname=name)
    print(json.dumps(dict(status='HBM_RESPONSE_CAPTURE_EXPORTED' if error is None else 'FAILED',
                         archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)), flush=True)
    return error is None


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--settings', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)
