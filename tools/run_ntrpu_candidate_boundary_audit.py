"""Audit the external-only RTSTAT-10 finding of a preserved routed fixture."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import zipfile

from run_ntrpu_registered_dispatch import SOURCES as BASE_SOURCES, digest, save

ROOT = Path(__file__).resolve().parents[1]
SOURCES = BASE_SOURCES + ['tools/prepare_ntrpu_candidate_dispatch.py',
    'tools/ntrpu_candidate_dispatch.tcl', 'tools/run_ntrpu_candidate_dispatch.py',
    'tools/ntrpu_candidate_place.tcl', 'tools/run_ntrpu_candidate_place.py',
    'tools/ntrpu_candidate_route.tcl', 'tools/run_ntrpu_candidate_route.py',
    'tools/ntrpu_candidate_boundary_audit.tcl', 'tools/run_ntrpu_candidate_boundary_audit.py']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--routed', type=Path, required=True)
    ap.add_argument('--threads', type=int, choices=(2, 4), default=2)
    args = ap.parse_args()
    if not 1 <= args.timeout <= 180:
        raise ValueError('Expected timeout of at most 180 seconds')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[-args.threads:]
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    before = {n: digest(ROOT/n) for n in SOURCES}
    record = dict(status='RUNNING', sources=before, cpus=cpus, nice=os.nice(0),
                  load_before=os.getloadavg(), timeout=args.timeout, threads=args.threads)
    prior = args.routed.resolve()
    previous = json.loads((prior/'result.json').read_text())
    manifest = json.loads((prior/'FILES.json').read_text())
    if previous['status'] != 'EXPORTED_PENDING_REVIEW' or previous['sources'] != previous['sources_after']:
        raise ValueError('Expected unchanged completed route')
    logic = [n for n in SOURCES if n.startswith('fpga/') or 'prepare_ntrpu_' in n]
    if not all(previous['sources'][n] == before[n] for n in logic):
        raise ValueError('Checkpoint logic inputs changed')
    inputs = output/'inputs'
    inputs.mkdir()
    checkpoints = {}
    for variant in ('candidate',):
        name = variant + '_route_attempt.dcp'
        checkpoints[name] = digest(prior/name)
        if checkpoints[name] != manifest[name]:
            raise ValueError('Checkpoint hash mismatch')
        shutil.copy2(prior/name, inputs/name)
    record['resume'] = dict(routed=str(prior), checkpoints=checkpoints, sources=previous['sources'])
    save(output/'result.json', record)
    cmd = ['vivado', '-mode', 'batch', '-notrace', '-source', str(ROOT/'tools/ntrpu_candidate_boundary_audit.tcl'),
           '-log', str(output/'vivado.log'), '-journal', str(output/'vivado.jou'), '-tclargs', str(inputs/'candidate_route_attempt.dcp'), str(output/'suite')]
    start = time.monotonic()
    with (output/'console.log').open('w') as log:
        p = subprocess.Popen(cmd, cwd=output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                code = p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                code = p.wait()
    record.update(returncode=code, timed_out=timed_out, seconds=round(time.monotonic()-start, 3), command=cmd,
                  sources_after={n: digest(ROOT/n) for n in SOURCES}, load_after=os.getloadavg())
    record['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not timed_out else 'FAILED_OR_TIMEOUT'
    if before != record['sources_after']:
        record['status'] = 'SOURCES_CHANGED'
    save(output/'result.json', record)
    manifest = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}
    save(output/'FILES.json', manifest)
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for p in output.rglob('*'):
            if p.is_file():
                bundle.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        import hashlib
        assert all(hashlib.sha256(bundle.read(n)).hexdigest() == h for n, h in manifest.items())
    save(output/'VERIFIED.json', dict(sha256=digest(archive), crc='PASS', hashes='PASS'))
    print(json.dumps(dict(status=record['status'], seconds=record['seconds'], sha256=digest(archive))), flush=True)
    return 0 if record['status'] == 'EXPORTED_PENDING_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
