"""Bounded native integration stages, each in a new private directory."""
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
    root, prepared = args.output.resolve(), args.prepared.resolve()
    namespace = Path('/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-native-join').resolve()
    if os.name != 'posix' or not root.is_relative_to(namespace) or root == namespace or root.exists():
        raise ValueError('New private output required')
    names = {p.name: p for p in prepared.iterdir() if p.is_file() and p.suffix in ('.tcl', '.py', '.sv', '.v', '.vh', '.xci', '.json', '.tsv')}
    hashes = {n: sha(p) for n, p in names.items()}
    root.mkdir(parents=True); source = root / 'source'; source.mkdir()
    work = root / 'work'; work.mkdir()
    for name, path in names.items():
        shutil.copy2(path, source / name)
    os.nice(10); os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    tclargs = [source, root / 'output']
    donor = None
    checkpoint_hash = None
    if args.stage == 'inspect':
        donor = args.checkpoint.resolve()
        if not donor.is_relative_to(namespace) or donor.suffix != '.dcp':
            raise ValueError('Private checkpoint required')
        checkpoint_hash = sha(donor)
        tclargs = [donor, root / 'output']
    elif args.stage != 'generate':
        donor = args.generation.resolve()
        if not donor.is_relative_to(namespace) or (donor / 'STATUS.txt').read_text().strip() != 'HBM_NATIVE_JOIN_GENERATED':
            raise ValueError('Closed generation donor required')
        shutil.copytree(donor / 'project', root / 'project')
        tclargs += [root / 'project/native_join.xpr', args.stage]
    script = source / ({'generate': 'generate', 'inspect': 'inspect'}.get(args.stage, 'elaborate') + '_hbm_native_join.tcl')
    argv = ['vivado', '-mode', 'batch', '-source', str(script), '-log', str(root / 'vivado.log'),
            '-journal', str(root / 'vivado.jou'), '-tclargs', *map(str, tclargs)]
    started = time.monotonic(); expired = False
    with (root / 'tool.log').open('w') as log:
        child = subprocess.Popen(['bash', '-c', 'source ' + shlex.quote(str(args.settings)) + ' && exec ' + shlex.join(argv)],
            cwd=work, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'own_process.json', dict(pid=child.pid, pgid=child.pid, timeout=1200, stage=args.stage))
        try:
            code = child.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            expired = True; os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); code = child.wait()
    unchanged = all(sha(prepared / n) == sha(source / n) == h for n, h in hashes.items())
    if checkpoint_hash:
        unchanged = unchanged and sha(donor) == checkpoint_hash
    errors = bool(re.search(r'(?im)^\s*(?:fatal|error):', (root / 'tool.log').read_text(errors='replace')))
    result = dict(status='HBM_NATIVE_JOIN_EXPORTED' if code == 0 and not expired and not errors and unchanged else 'FAILED',
        stage=args.stage, returncode=code, timed_out=expired, sources_unchanged=unchanged, sources=hashes,
        seconds=time.monotonic()-started, generation=str(donor) if donor else None,
        checkpoint_sha256=checkpoint_hash,
        integrated_timing_qualified=False, native_traffic_qualified=False, board_qualified=False)
    save(root / 'RESULT.json', result)
    # Include generated HDL/XCI/XDC, reports and source; exclude regenerable tool databases.
    allowed = {'.sv', '.v', '.vh', '.vhd', '.xci', '.xdc', '.tcl', '.py', '.json', '.tsv', '.rpt', '.log', '.jou', '.txt', '.veo', '.vho', '.dcp', '.xpr', '.mem', '.coe', '.mif', '.dat'}
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()
        and p.suffix in allowed and 'work' not in p.relative_to(root).parts and '.Xil' not in p.parts}
    save(root / 'FILES.json', files)
    archive = Path(str(root) + '.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for name in list(files) + ['FILES.json']:
            tar.add(root / name, arcname=name)
    receipt = {k: v for k, v in result.items() if k != 'sources'}
    receipt.update(archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)
    print(json.dumps(receipt), flush=True)
    return result['status'] != 'FAILED'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--stage', choices=('generate', 'elaborate', 'synth', 'inspect'), required=True)
    parser.add_argument('--generation', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
