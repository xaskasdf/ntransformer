"""Bounded connectivity-derived bank placement using verified compact checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_ntrpu_overlap_physical import SOURCES as BASE, digest, save

ROOT = Path(__file__).resolve().parents[1]
SOURCES = BASE + ['tools/ntrpu_overlap_bankplan.tcl', 'tools/run_ntrpu_overlap_bankplan.py',
                  'tools/ntrpu_overlap_connectivity.tcl', 'tools/prepare_ntrpu_overlap_bankplan.py']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--stage', choices=('place', 'route', 'netlist'), required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--plan', type=Path, help='Plan and probe_evidence directory; required for place')
    a = ap.parse_args()
    if not 1 <= a.timeout <= 900:
        raise ValueError('Timeout must be 1..900')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    prior = a.prior.resolve()
    sources = {n: digest(ROOT/n) for n in SOURCES}
    old = json.loads((prior/'result.json').read_text())
    expected = {'place': 'synth', 'route': 'place', 'netlist': 'route'}[a.stage]
    assert old['stage'] == expected and old['variant'] == 'compact'
    assert old['status'] == 'EXPORTED_PENDING_REVIEW' and old['sources'] == old['sources_after']
    assert all(sources[n] == old['sources'][n] for n in BASE), 'Baseline sources changed'
    if a.stage != 'place':
        assert old['experiment'] == 'connectivity_bank_floorplan' and sources == old['sources']
    files = json.loads((prior/'FILES.json').read_text())
    artifact = 'physical_funcsim.v' if a.stage == 'netlist' else 'physical_'+expected+'.dcp'
    names = ['suite/'+artifact, 'suite/constraints_after.xdc', 'result.json']
    if a.stage != 'place':
        names += ['suite/bankplan.tsv', 'suite/members_after.tsv']
    (out/'inputs').mkdir()
    for n in names:
        assert digest(prior/n) == files[n], 'Prior artifact changed'
        shutil.copy2(prior/n, out/'inputs'/Path(n).name)
    shutil.copy2(prior/'FILES.json', out/'inputs/FILES.json')
    for p in (prior/'rtl_evidence').iterdir():
        assert digest(p) == files['rtl_evidence/'+p.name]
    shutil.copytree(prior/'rtl_evidence', out/'rtl_evidence')
    if a.stage == 'place':
        if not a.plan: raise ValueError('Connectivity plan required')
        plan_source = a.plan.resolve()
    else:
        plan_source = prior/'plan_evidence'
        assert all(digest(p) == files[p.relative_to(prior).as_posix()] for p in plan_source.rglob('*') if p.is_file())
    shutil.copytree(plan_source, out/'plan_evidence')
    plan_files = {p.relative_to(out/'plan_evidence').as_posix(): digest(p) for p in (out/'plan_evidence').rglob('*') if p.is_file()}
    plan = json.loads((out/'plan_evidence/BANK_PLAN.json').read_text())
    assert plan['generator_sha256'] == sources['tools/prepare_ntrpu_overlap_bankplan.py']
    from prepare_ntrpu_overlap_bankplan import derive
    assert derive(out/'plan_evidence/probe_evidence') == plan, 'Plan differs from connectivity evidence'
    inputs = {p.name: digest(p) for p in (out/'inputs').iterdir()}
    rtl = {p.name: digest(p) for p in (out/'rtl_evidence').iterdir()}
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0, cpus); os.nice(10); os.environ['OMP_NUM_THREADS'] = '2'
    record = dict(status='RUNNING', stage=a.stage, variant='compact', experiment='connectivity_bank_floorplan',
                  sources=sources, timeout=a.timeout, cpus=cpus, nice=os.nice(0), load_before=os.getloadavg(),
                  inputs=inputs, rtl_evidence=rtl, plan_evidence=plan_files,
                  prior=dict(path=str(prior), stage=expected, artifact=artifact, sha256=files['suite/'+artifact]))
    save(out/'result.json', record)
    cmd = ['vivado', '-mode', 'batch', '-notrace', '-source', str(ROOT/'tools/ntrpu_overlap_bankplan.tcl'),
           '-log', str(out/'vivado.log'), '-journal', str(out/'vivado.jou'), '-tclargs', str(out/'suite'), a.stage]
    start = time.monotonic(); timed_out = False
    with (out/'console.log').open('w') as log:
        job = subprocess.Popen(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = job.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(job.pid, signal.SIGTERM)
            try:
                code = job.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(job.pid, signal.SIGKILL); code = job.wait()
    record.update(returncode=code, timed_out=timed_out, seconds=round(time.monotonic()-start, 3), command=cmd,
                  sources_after={n: digest(ROOT/n) for n in SOURCES}, load_after=os.getloadavg(),
                  inputs_after={n: digest(out/'inputs'/n) for n in inputs},
                  rtl_evidence_after={n: digest(out/'rtl_evidence'/n) for n in rtl})
    record['plan_evidence_after'] = {n: digest(out/'plan_evidence'/n) for n in plan_files}
    record['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not timed_out else 'FAILED_OR_TIMEOUT'
    if sources != record['sources_after'] or inputs != record['inputs_after'] or rtl != record['rtl_evidence_after'] or plan_files != record['plan_evidence_after']:
        record['status'] = 'INPUTS_CHANGED'
    save(out/'result.json', record)
    for p in (out/'suite/project').rglob('*.log'):
        dest = out/'tool_logs'/p.relative_to(out/'suite/project')
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    save(out/'FILES.json', {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file()
                           and 'project' not in p.relative_to(out).parts
                           and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k: record[k] for k in ('stage', 'status', 'seconds')}), flush=True)
    return 0 if record['status'] == 'EXPORTED_PENDING_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
