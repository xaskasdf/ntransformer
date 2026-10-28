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
        child = subprocess.Popen(['bash', '-c', command], cwd=work, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        with (root / 'own_processes.jsonl').open('a') as f:
            f.write(json.dumps(dict(pid=child.pid, pgid=child.pid, step=name, timeout=limit)) + '\n')
        expired = False; start = time.monotonic()
        try:
            captured, _ = child.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            expired = True
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: captured, _ = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                captured, _ = child.communicate()
        code = child.returncode
        (root / (name + '.log')).write_bytes(captured)
        steps.append(dict(name=name, returncode=code, timed_out=expired, seconds=time.monotonic()-start))
        save(root / 'steps.json', steps)
        text = (root / (name + '.log')).read_text(errors='replace')
        if code or expired or re.search(r'(?im)^\s*(?:error|fatal):', text):
            raise ValueError('Failed step: ' + name)
        return text

    try:
        modules = ['reference_queue', 'reference_ready_queue', 'reference_tracker', 'reference_return',
                   'candidate_tracker', 'candidate_capture', 'candidate_validated', 'candidate_return',
                   'tb_hbm_registered_ready_queue', 'tb_hbm_registered_ready_tracker', 'tb_hbm_return_valid_queue']
        execute('compile', ['xvlog', '--sv', *[source / (n + '.sv') for n in modules]], 120)
        for label, top in [('queue', 'tb_hbm_registered_ready_queue'), ('tracker', 'tb_hbm_registered_ready_tracker'), ('return', 'tb_hbm_return_valid_queue')]:
            execute(label+'-elaborate', ['xelab', '--debug', 'typical', '--mt', '2', '--snapshot', label, 'work.'+top], 120)
            log = execute(label+'-run', ['xsim', label, '-runall'], 180)
            if (('RETURN_VALID_QUEUE' if label=='return' else 'REGISTERED_READY_'+label.upper())+' ALL PASS') not in log:
                raise ValueError('Missing simulation completion: '+label)
        rtl = (source / 'candidate_tracker.sv').read_text()
        mutations = [
            ('stale_space', 'q_space_next&&', 'q_space&&', 'reference controls'),
            ('stale_credit', '(capture_remaining_next!=0)', '(capture_remaining!=0)', 'reference controls'),
            ('late_fault', '&&!fault_next;', '&&!fault;', 'reference controls'),
            ('request_fault', 'unexpected_offer||predict_bad_request;', 'unexpected_offer;', 'active-edge fault predictor equivalence'),
            ('sync_reset', 'always @(posedge clk or negedge rst_n)', 'always @(posedge clk)', 'asynchronous READY clear')]
        mutations = [('tracker', *item) for item in mutations] + [
            ('return', 'return_stale', 'valid_reg<=predict_count!=0;', 'valid_reg<=count!=0;', 'post-edge valid model'),
            ('return', 'return_sync_reset', 'always @(posedge clk or negedge rst_n)', 'always @(posedge clk)', 'asynchronous VALID clear'),
            ('return', 'return_overwrite', 'if(s_ready)begin', 'if(1)begin', 'full/reset payload retained')]
        for kind, label, anchor, replacement, reason in mutations:
            filename = 'candidate_tracker.sv' if kind=='tracker' else 'candidate_return.sv'
            rtl = (source / filename).read_text()
            top = 'work.tb_hbm_registered_ready_tracker' if kind=='tracker' else 'work.tb_hbm_return_valid_queue'
            marker = 'REGISTERED_READY_TRACKER pending=' if kind=='tracker' else 'RETURN_VALID_QUEUE width='
            if rtl.count(anchor) != 1:
                raise ValueError('Mutation anchor differs: '+label)
            directory = root / ('mutation-'+label); directory.mkdir()
            bad = directory / filename
            bad.write_text(rtl.replace(anchor, replacement, 1))
            execute(label+'-compile', ['xvlog', '--sv', bad], 120)
            execute(label+'-elaborate', ['xelab', '--debug', 'typical', '--mt', '2', '--snapshot', label,
                                      top], 120)
            failed = False
            try:
                execute(label+'-run', ['xsim', label, '-runall'], 120)
            except ValueError:
                failed = True
            log = (root / (label+'-run.log')).read_text()
            if not failed or marker not in log or reason not in log or steps[-1]['timed_out']:
                raise ValueError('Mutation did not fail for intended invariant: '+label)
            rejected.append(dict(name=label, kind=kind, file=filename, reason=reason, sha256=sha(bad)))
        save(root / 'NEGATIVE_CONTROLS.json', rejected)
        execute('synthesis', ['vivado', '-mode', 'batch', '-source', source / 'synthesize_hbm_control_predictors.tcl',
                '-log', root / 'synthesis-vivado.log', '-journal', root / 'synthesis-vivado.jou',
                '-tclargs', source, root / 'output'], 600)
        if (root / 'output/STATUS.txt').read_text().strip() != 'HBM_CONTROL_PREDICTORS_TOPOLOGY_EXPORTED':
            raise ValueError('Missing native synthesis completion')
    except (OSError, ValueError) as exc:
        error = str(exc)
    unchanged = all(sha(inputs[n]) == sha(source / n) == h for n, h in hashes.items())
    if not unchanged:
        error = (error+'; ' if error else '')+'Sources changed'
    save(root / 'RESULT.json', dict(status='HBM_CONTROL_PREDICTORS_EXPORTED' if error is None else 'FAILED',
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
    print(json.dumps(dict(status='HBM_CONTROL_PREDICTORS_EXPORTED' if error is None else 'FAILED',
                         archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)), flush=True)
    return error is None


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--settings', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)
