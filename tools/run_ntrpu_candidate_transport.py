"""Bounded isolated transport validation; export evidence without duplicate ZIPs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    'fpga/rtl/ntrpu_buffer_pool_allocator.sv',
    'fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv',
    'fpga/rtl/ntrpu_rv_elastic.sv', 'fpga/rtl/ntrpu_destination_dispatch.sv',
    'fpga/rtl/ntrpu_registered_dispatch.sv',
    'fpga/rtl/ntrpu_peer_multibuffer_fillstage.sv',
    'fpga/rtl/ntrpu_peer_multibuffer_elastic.sv',
    'fpga/tb/tb_ntrpu_reserved_dispatch.sv',
    'fpga/rtl/ntrpu_candidate_transport.sv',
    'fpga/tb/tb_ntrpu_candidate_transport.sv',
    'tools/prepare_ntrpu_reserved_receiver.py',
    'tools/prepare_ntrpu_registered_dispatch.py',
    'tools/prepare_ntrpu_candidate_dispatch.py',
    'tools/run_ntrpu_candidate_transport.py',
    'tools/ntrpu_candidate_transport.tcl',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--simulation-only', action='store_true', help='For bench-only changes; review with prior synthesis evidence')
    args = ap.parse_args()
    if not 1 <= args.timeout <= 900:
        raise ValueError('Timeout must be in 1..900 seconds')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = {n: digest(ROOT/n) for n in SOURCES}
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    os.environ['OMP_NUM_THREADS'] = '2'
    record = dict(status='RUNNING', sources=before, cpus=cpus, nice=os.nice(0),
                  timeout=args.timeout, simulation_only=args.simulation_only, load_before=os.getloadavg())
    save(output/'result.json', record)
    cmd = ['vivado', '-mode', 'batch', '-notrace', '-source',
           str(ROOT/'tools/ntrpu_candidate_transport.tcl'), '-log', str(output/'vivado.log'),
           '-journal', str(output/'vivado.jou'), '-tclargs', str(output/'suite')]
    if args.simulation_only:
        cmd.append('simulation-only')
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
    record.update(returncode=code, timed_out=timed_out, command=cmd,
                  seconds=round(time.monotonic()-start, 3),
                  sources_after={n: digest(ROOT/n) for n in SOURCES}, load_after=os.getloadavg())
    record['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not timed_out else 'FAILED_OR_TIMEOUT'
    if record['sources_after'] != before:
        record['status'] = 'SOURCES_CHANGED'
    save(output/'result.json', record)
    # Capture logs even when compile/elaboration failed. Reproducible caches are
    # deliberately not archived; HDL, generated files, logs and checkpoint are.
    project = output/'suite/project'
    evidence = output/'tool_logs'
    evidence.mkdir()
    if project.exists():
        for path in project.rglob('*.log'):
            target = evidence/path.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    manifest = {}
    for path in output.rglob('*'):
        relative = path.relative_to(output)
        if path.is_file() and 'project' not in relative.parts and not any(p.startswith('.') for p in relative.parts):
            manifest[relative.as_posix()] = digest(path)
    save(output/'FILES.json', manifest)
    print(json.dumps(dict(status=record['status'], seconds=record['seconds'], evidence_files=len(manifest))), flush=True)
    return 0 if record['status'] == 'EXPORTED_PENDING_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
