"""Run the isolated differential bench with a bounded XSIM process group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from run_ntrpu_peer_read import sha, save

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    out = ap.parse_args().output.resolve(); out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2]); os.nice(10)
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*') if p.is_file()}
    commands = [['xvlog', '-sv', *[str(ROOT / n) for n in (
        'fpga/experimental/ntrpu_peer_read_pkg.sv', 'fpga/experimental/ntrpu_peer_admission_reference.sv',
        'fpga/experimental/ntrpu_peer_read_admission.sv', 'fpga/tb/tb_ntrpu_peer_interval.sv')]],
        ['xelab', 'tb_ntrpu_peer_interval', '-s', 'interval', '--mt', '2', '-debug', 'off'],
        ['xsim', 'interval', '-runall', '-log', 'simulation.log']]
    result = dict(status='RUNNING', experiment='peer_interval', sources=sources,
                  cpus=sorted(os.sched_getaffinity(0)), nice=os.nice(0), timeout=300)
    save(out / 'result.json', result); start = time.monotonic(); expired = False
    with (out / 'console.log').open('w') as log:
        for command in commands:
            job = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = job.wait(timeout=max(1, 300 - (time.monotonic() - start)))
            except subprocess.TimeoutExpired:
                expired = True; os.killpg(job.pid, signal.SIGTERM)
                try: code = job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL); code = job.wait()
            if code or expired: break
    after = {n: sha(ROOT / n) for n in sources}
    log = out / 'simulation.log'; text = log.read_text(errors='replace') if log.exists() else ''
    passed = 'INT ALL_PASS cases=324' in text and not any(n in text for n in ('Fatal:', 'INT_FAIL', 'INT_WATCHDOG'))
    result.update(status='EXPORTED_PENDING_REVIEW' if code == 0 and not expired and passed and sources == after else 'FAILED_OR_CHANGED',
                  returncode=code, timed_out=expired, seconds=round(time.monotonic() - start, 3), sources_after=after)
    save(out / 'result.json', result)
    save(out / 'FILES.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()
                             and 'xsim.dir' not in p.parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k: result[k] for k in ('status', 'returncode', 'seconds')}))


if __name__ == '__main__': main()
