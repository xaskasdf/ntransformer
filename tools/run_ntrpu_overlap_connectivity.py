"""Export routed leaf-pin connectivity with a bounded, reproducible audit record."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path); ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--expected-sha256', required=True); ap.add_argument('--timeout', type=int, default=300)
    a = ap.parse_args()
    if not 1 <= a.timeout <= 300: raise ValueError('Timeout must be 1..300')
    checkpoint = a.checkpoint.resolve(); script = Path(__file__).with_name('ntrpu_overlap_connectivity.tcl')
    if sha(checkpoint) != a.expected_sha256: raise ValueError('Checkpoint identity mismatch')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[-2:]; os.sched_setaffinity(0, cpus)
    os.nice(10); os.environ['OMP_NUM_THREADS'] = '2'
    record = dict(status='RUNNING', stage='probe', cpus=cpus, timeout=a.timeout,
                  checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  script_sha256=sha(script), runner_sha256=sha(Path(__file__)))
    save = lambda: (out/'result.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    save(); start = time.monotonic(); timed_out = False
    command = ['vivado', '-mode', 'batch', '-notrace', '-source', str(script),
               '-log', str(out/'vivado.log'), '-journal', str(out/'vivado.jou'),
               '-tclargs', str(checkpoint), str(out)]
    with (out/'console.log').open('w') as log:
        job = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = job.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(job.pid, signal.SIGTERM)
            try: code = job.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(job.pid, signal.SIGKILL); code = job.wait()
    record.update(returncode=code, timed_out=timed_out, seconds=round(time.monotonic()-start, 3), command=command,
                  checkpoint_sha256_after=sha(checkpoint), script_sha256_after=sha(script),
                  runner_sha256_after=sha(Path(__file__)))
    record['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not timed_out else 'FAILED_OR_TIMEOUT'
    if any(record[n] != record[n+'_after'] for n in ('checkpoint_sha256', 'script_sha256', 'runner_sha256')):
        record['status'] = 'INPUTS_CHANGED'
    save()
    (out/'FILES.json').write_text(json.dumps({p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*')
                                            if p.is_file() and not any(n.startswith('.') for n in p.relative_to(out).parts)}, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ('status', 'stage', 'seconds')}))
    return 0 if record['status'] == 'EXPORTED_PENDING_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
