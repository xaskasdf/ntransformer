"""Bounded private Vivado guard diagnostic; never launches SSH or main jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tarfile
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def run(args):
    root = args.output.resolve()
    allowed = Path('/home/ubuntu/vivado/ntrpu-side').resolve()
    if os.name != 'posix' or not root.is_relative_to(allowed):
        raise ValueError('Linux private side namespace required')
    archive = Path(str(root) + '.tar.gz')
    if root.exists() or archive.exists() or not args.settings.is_file():
        raise ValueError('New output and installed Vivado required')
    here = Path(__file__).resolve().parent
    rtl = here / 'ntrpu_axi_bridge_guard.sv'
    if sha(rtl) != 'd5b340ab788143cdca33e4cec760778722e76c783cb5e0e3ee63798df74d142d':
        raise ValueError('Guard differs from reviewed native source')
    checkpoint_sha = None
    if args.checkpoint:
        if not args.checkpoint.resolve().is_relative_to(allowed) or not args.checkpoint.is_file():
            raise ValueError('Private existing checkpoint required')
        checkpoint_sha = sha(args.checkpoint)
    root.mkdir(parents=True)
    source = root / 'source'; source.mkdir()
    names = ['ntrpu_axi_bridge_guard.sv', 'audit_guard_cdc.tcl', 'run_guard_cdc.py']
    if args.checkpoint: names.append('scope_guard_cdc.tcl')
    for name in names:
        shutil.copy2(here / name, source / name)
    work = root / 'work'; work.mkdir()
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    script = 'scope_guard_cdc.tcl' if args.checkpoint else 'audit_guard_cdc.tcl'
    input_path = args.checkpoint.resolve() if args.checkpoint else source / rtl.name
    command = ['vivado', '-mode', 'batch', '-source', str(source / script),
               '-journal', str(root / 'vivado.jou'), '-log', str(root / 'vivado.log'),
               '-tclargs', str(input_path), str(root / 'reports')]
    shell = 'source ' + shlex.quote(str(args.settings)) + ' && exec ' + shlex.join(command)
    started = time.monotonic(); timed_out = False
    with (root / 'console.log').open('w') as log:
        process = subprocess.Popen(['bash', '-c', shell], cwd=work, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'own_process.json', dict(pid=process.pid, pgid=process.pid, timeout=300))
        try:
            code = process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(process.pid, signal.SIGTERM)
            try: code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); code = process.wait()
    marker = root / 'reports/STATUS.txt'
    complete = code == 0 and not timed_out and marker.is_file()
    checkpoint_unchanged = args.checkpoint is None or sha(args.checkpoint) == checkpoint_sha
    complete = complete and checkpoint_unchanged
    result = dict(status='GUARD_CDC_DIAGNOSTIC_EXPORTED' if complete else 'FAILED',
                  returncode=code, timed_out=timed_out, seconds=time.monotonic()-started,
                  command=command, cpus=sorted(os.sched_getaffinity(0)), nice=os.nice(0),
                  guard_sha256=sha(rtl), timing_closed=False, physical_cdc_validated=False,
                  checkpoint=str(args.checkpoint) if args.checkpoint else None,
                  checkpoint_sha256=checkpoint_sha, checkpoint_unchanged=checkpoint_unchanged)
    save(root / 'RESULT.json', result)
    paths = [p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(work)]
    save(root / 'FILES.json', {p.relative_to(root).as_posix(): sha(p) for p in paths})
    with tarfile.open(archive, 'x:gz') as tar:
        for path in paths + [root / 'FILES.json']:
            tar.add(path, arcname=path.relative_to(root).as_posix())
    result.update(archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)
    print(json.dumps(result), flush=True)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--execute-native', action='store_true')
    parser.add_argument('--checkpoint', type=Path, help='Reuse immutable baseline for a scoped constraint trial')
    args = parser.parse_args()
    if not args.execute_native: parser.error('--execute-native required')
    raise SystemExit(0 if run(args) else 1)
