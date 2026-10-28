"""Bounded post-opt connectivity and actual-device geometry probe for the AXI plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--expected-sha256', required=True)
    ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args()
    checkpoint = args.checkpoint.resolve(); plan = args.plan.resolve()
    assert sha(checkpoint) == args.expected_sha256
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    scripts = [Path(__file__), Path(__file__).with_name('ntrpu_overlap_axi_preflight.tcl'),
               Path(__file__).with_name('ntrpu_overlap_connectivity.tcl')]
    inputs = lambda: {'checkpoint': sha(checkpoint),
                      **{p.name: sha(p) for p in scripts},
                      **{'plan/'+p.relative_to(plan).as_posix(): sha(p) for p in plan.rglob('*') if p.is_file()}}
    cpus = sorted(os.sched_getaffinity(0))[-2:]; os.sched_setaffinity(0, cpus); os.nice(10)
    os.environ['OMP_NUM_THREADS'] = '2'
    record = dict(status='RUNNING', stage='preflight', cpus=cpus, timeout=300,
                  inputs=inputs(), checkpoint=str(checkpoint), plan=str(plan))
    save = lambda: (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    save(); start = time.monotonic(); expired = False
    command = ['vivado', '-mode', 'batch', '-notrace', '-source', str(scripts[1]),
               '-log', str(out/'vivado.log'), '-journal', str(out/'vivado.jou'),
               '-tclargs', str(checkpoint), str(out), str(plan)]
    with (out/'console.log').open('w') as log:
        job = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = job.wait(timeout=300)
        except subprocess.TimeoutExpired:
            expired = True; os.killpg(job.pid, signal.SIGTERM)
            try: code = job.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(job.pid, signal.SIGKILL); code = job.wait()
    record.update(returncode=code, timed_out=expired, seconds=round(time.monotonic()-start, 3),
                  inputs_after=inputs(), command=command)
    record['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not expired else 'FAILED_OR_TIMEOUT'
    if record['inputs'] != record['inputs_after']: record['status'] = 'INPUTS_CHANGED'
    save()
    files = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()
             and not any(n.startswith('.') for n in p.relative_to(out).parts)}
    (out/'FILES.json').write_text(json.dumps(files, indent=2)+'\n')
    print(json.dumps(record))
    return int(record['status'] != 'EXPORTED_PENDING_REVIEW')


if __name__ == '__main__':
    raise SystemExit(main())
