"""Bounded, private Vivado OOC review; does not touch principal runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tcl', default='tools/ntrpu_hbm_dma_cdc_probe.tcl')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.nice(10)
    os.environ['OMP_NUM_THREADS'] = '2'
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*') if p.is_file()}
    command = ['vivado', '-mode', 'batch', '-notrace', '-source', str(ROOT / args.tcl),
               '-tclargs', str(out / 'reports')]
    result = dict(status='RUNNING', sources=sources, command=command, timeout=args.timeout,
                  cpus=sorted(os.sched_getaffinity(0)), nice=os.nice(0))
    save(out / 'result.json', result)
    start = time.monotonic()
    expired = False
    with (out / 'console.log').open('w') as log:
        job = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
        try:
            code = job.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            expired = True
            os.killpg(job.pid, signal.SIGTERM)
            try:
                code = job.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(job.pid, signal.SIGKILL)
                code = job.wait()
    after = {name: sha(ROOT / name) for name in sources}
    result.update(returncode=code, timed_out=expired, seconds=round(time.monotonic() - start, 3),
                  sources_after=after)
    result['status'] = 'PENDING_REVIEW' if code == 0 and not expired and sources == after else 'FAILED'
    save(out / 'result.json', result)
    save(out / 'FILES.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*')
                            if p.is_file() and not any(x.startswith('.') for x in p.relative_to(out).parts)
                            and p.name != 'FILES.json'})
    print(json.dumps({k: result[k] for k in ('status', 'seconds', 'returncode')}), flush=True)
    raise SystemExit(0 if result['status'] == 'PENDING_REVIEW' else 1)


if __name__ == '__main__':
    main()
