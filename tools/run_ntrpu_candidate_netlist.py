"""Bounded routed functional simulation; preserve source, netlist and failed runs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import zipfile

from run_ntrpu_candidate_route import SOURCES as BASE_SOURCES
from run_ntrpu_registered_dispatch import digest, save

ROOT = Path(__file__).resolve().parents[1]
SOURCES = BASE_SOURCES + ['tools/prepare_ntrpu_candidate_netlist.py',
    'tools/ntrpu_candidate_netlist.tcl', 'tools/run_ntrpu_candidate_netlist.py',
    'tools/check_ntrpu_candidate_netlist.py', 'tools/check_ntrpu_candidate_dispatch.py',
    'tools/check_ntrpu_registered_dispatch.py']
NETLIST = 'candidate_fixture_funcsim.v'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--routed', type=Path, required=True)
    ap.add_argument('--timeout', type=int, default=900)
    args = ap.parse_args()
    if not 1 <= args.timeout <= 900:
        raise ValueError('Timeout must be at most 900s')
    prior = args.routed.resolve()
    previous = json.loads((prior/'result.json').read_text())
    files = json.loads((prior/'FILES.json').read_text())
    if previous['status'] != 'EXPORTED_PENDING_REVIEW' or previous['sources'] != previous['sources_after']:
        raise ValueError('Routed source provenance incomplete')
    before = {n: digest(ROOT/n) for n in SOURCES}
    if any(before[n] != h for n, h in previous['sources'].items() if n.startswith('fpga/') or '/prepare_' in n):
        raise ValueError('Reference logic differs from routed logic')
    if digest(prior/NETLIST) != files[NETLIST]:
        raise ValueError('Routed netlist hash mismatch')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs';inputs.mkdir()
    for name in (NETLIST, 'result.json', 'FILES.json'):
        shutil.copy2(prior/name, inputs/name)
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0, cpus);os.nice(10)
    os.environ['OMP_NUM_THREADS'] = '2'
    record = dict(status='RUNNING', sources=before, cpus=cpus, nice=os.nice(0),
                  routed=str(prior), netlist_sha256=files[NETLIST], timeout=args.timeout,
                  sdf=False, load_before=os.getloadavg())
    save(output/'result.json', record)
    cmd = ['vivado', '-mode', 'batch', '-notrace', '-source', str(ROOT/'tools/ntrpu_candidate_netlist.tcl'),
           '-log', str(output/'vivado.log'), '-journal', str(output/'vivado.jou'),
           '-tclargs', str(output/'suite'), str(inputs/NETLIST)]
    start = time.monotonic()
    with (output/'console.log').open('w') as log:
        p = subprocess.Popen(cmd, cwd=output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True;os.killpg(p.pid, signal.SIGTERM)
            try:
                code = p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL);code = p.wait()
    record.update(returncode=code, timed_out=timed_out, seconds=round(time.monotonic()-start, 3), command=cmd,
                  sources_after={n: digest(ROOT/n) for n in SOURCES}, load_after=os.getloadavg(),
                  netlist_sha256_after=digest(inputs/NETLIST))
    record['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not timed_out else 'FAILED_OR_TIMEOUT'
    if before != record['sources_after'] or record['netlist_sha256'] != record['netlist_sha256_after']:
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
        import hashlib
        assert bundle.testzip() is None
        assert all(hashlib.sha256(bundle.read(n)).hexdigest() == h for n, h in manifest.items())
    save(output/'VERIFIED.json', dict(sha256=digest(archive), crc='PASS', hashes='PASS'))
    print(json.dumps(dict(status=record['status'], seconds=record['seconds'], sha256=digest(archive))), flush=True)
    return 0 if record['status'] == 'EXPORTED_PENDING_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
