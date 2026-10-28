"""Bounded, isolated functional-netlist regression and read-only routed DRC audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from run_ntrpu_hbm_cdc import SOURCES as CDC_SOURCES

ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys(CDC_SOURCES + [
    'tools/run_ntrpu_hbm_netlist.py', 'tools/prepare_ntrpu_hbm_netlist.py',
    'tools/prepare_ntrpu_return_cut_netlist.py', 'tools/ntrpu_hbm_drc_audit.tcl']))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--stage', choices=('drc', 'sim'), required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--mutant', choices=('none', 'payload', 'ready'), default='none')
    a = ap.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    review = json.loads(a.review.read_text())
    old = json.loads((a.prior/'result.json').read_text())
    files = json.loads((a.prior/'FILES.json').read_text())
    assert review['routed'] and review['evidence_validated'] and old['sources'] == old['sources_after']
    donor = a.prior/('suite/physical_route.dcp' if a.stage == 'drc' else 'suite/physical_funcsim.v')
    assert sha(donor) == files[donor.relative_to(a.prior).as_posix()]
    if a.stage == 'drc':
        assert sha(donor) == review['checkpoint_sha256']
    dest = out/('input.dcp' if a.stage == 'drc' else 'input.v')
    shutil.copy2(donor, dest); shutil.copy2(a.review, out/'PRIOR_REVIEW.json')
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0, cpus); os.nice(10); os.environ['OMP_NUM_THREADS'] = '2'
    timeout = 300 if a.stage == 'drc' else 900
    record = dict(status='RUNNING', experiment='hbm_netlist', stage=a.stage, mutant=a.mutant,
                  sources={n: sha(ROOT/n) for n in SOURCES}, cpus=cpus, nice=os.nice(0),
                  load_before=os.getloadavg(), timeout=timeout, input_sha256=sha(dest),
                  prior_result_sha256=sha(a.prior/'result.json'))
    save(out/'result.json', record)
    generated = lambda: {p.relative_to(out).as_posix(): sha(p) for p in (out/'suite/generated').glob('*') if p.is_file()}
    try:
        if a.stage == 'sim':
            assert a.reference
            subprocess.run([sys.executable, '-B', str(ROOT/'tools/prepare_ntrpu_hbm_netlist.py'), str(out/'suite'),
                            '--netlist', str(dest), '--reference', str(a.reference), '--mutant', a.mutant], check=True)
            script = out/'suite/run.tcl'; args = []
        else:
            script = ROOT/'tools/ntrpu_hbm_drc_audit.tcl'; args = ['-tclargs', str(dest)]
        record['generated'] = generated()
        command = ['vivado', '-mode', 'batch', '-notrace', '-source', str(script),
                   '-log', 'vivado.log', '-journal', 'vivado.jou'] + args
        record['command'] = command; save(out/'result.json', record)
        start = time.monotonic(); expired = False
        with (out/'console.log').open('w') as log:
            job = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = job.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                expired = True; os.killpg(job.pid, signal.SIGTERM)
                try:
                    code = job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL); code = job.wait()
        record.update(seconds=round(time.monotonic()-start, 3), returncode=code, timed_out=expired)
        record['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not expired else 'FAILED_OR_TIMEOUT'
    except Exception as exc:
        record.update(status='PREPARATION_FAILED', error=repr(exc))
    record.update(sources_after={n: sha(ROOT/n) for n in SOURCES}, input_after=sha(dest),
                  donor_after=sha(donor), generated_after=generated())
    if record['sources'] != record['sources_after'] or record['input_sha256'] != record['input_after'] or record['input_after'] != record['donor_after']:
        record['status'] = 'INPUTS_CHANGED'
    if record.get('generated', {}) != record['generated_after']:
        record['status'] = 'GENERATED_INPUTS_CHANGED'
    save(out/'result.json', record)
    for path in (out/'suite/project').rglob('*.log'):
        target = out/'tool_logs'/path.relative_to(out/'suite/project')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    save(out/'FILES.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()
                          and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps(record), flush=True)
    return 0 if record['status'] == 'EXPORTED_PENDING_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
