"""Run an isolated P0 single-span experiment; no principal queue changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
NEW = ['fpga/experimental/ntrpu_peer_read_pkg.sv',
       'fpga/experimental/ntrpu_peer_read_lane.sv', 'fpga/tb/tb_ntrpu_peer_read.sv',
       'tools/run_ntrpu_peer_read.py', 'tools/ntrpu_peer_read.tcl',
       'tools/check_ntrpu_peer_read.py', 'fpga/experimental/ntrpu_peer_read_pair.sv']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--pair', action='store_true')
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.nice(10)
    os.environ['OMP_NUM_THREADS'] = '2'
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*') if p.is_file()}
    if a.full:
        commands = [['vivado', '-mode', 'batch', '-notrace', '-source',
                     str(ROOT/'tools/ntrpu_peer_read.tcl'), '-tclargs', str(out/'suite'), str(int(a.pair))]]
    else:
        sv = [*NEW[:3], 'fpga/experimental/ntrpu_hbm_read_frames.sv']
        commands = [['xvlog', '-sv', *[str(ROOT/n) for n in sv]],
                    ['xelab', 'tb_ntrpu_peer_read', '-s', 'peer', '-debug', 'off'],
                    ['xsim', 'peer', '-runall', '-log', 'simulation.log']]
    start = time.monotonic()
    result = dict(status='RUNNING', full=a.full, pair=a.pair, sources=sources, commands=commands,
                  cpus=sorted(os.sched_getaffinity(0)), nice=os.nice(0), timeout=900)
    save(out/'result.json', result)
    expired = False
    with (out/'console.log').open('w') as log:
        for command in commands:
            job = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
            try:
                code = job.wait(timeout=max(1, 900-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                expired = True
                os.killpg(job.pid, signal.SIGTERM)
                try:
                    code = job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL)
                    code = job.wait()
            if code or expired:
                break
    after = {n: sha(ROOT/n) for n in sources}
    path=out/('suite/simulation.log' if a.full else 'simulation.log')
    logtext=path.read_text(errors='replace') if path.exists() else ''
    functional=('PR ALL_PASS cases=21 full='+str(int(a.full))) in logtext and not any(x in logtext for x in ('Fatal:', 'PR_FAIL', 'PR_WATCHDOG'))
    result.update(returncode=code, timed_out=expired, sources_after=after,
                  seconds=round(time.monotonic()-start, 3), functional_marker=functional)
    result['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and functional and not expired and after == sources else 'FAILED_OR_CHANGED'
    save(out/'result.json', result)
    save(out/'FILES.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*')
                            if p.is_file() and 'project' not in p.parts and 'xsim.dir' not in p.parts
                            and not any(x.startswith('.') for x in p.relative_to(out).parts)})
    print(json.dumps(result | {'sources': len(sources), 'sources_after': len(after)}))


if __name__ == '__main__':
    main()
