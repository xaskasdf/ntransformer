"""Run an isolated, bounded sensitivity sweep over the qualified circuit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_ntrpu_return_cut import SOURCES as FROZEN_SOURCES
from run_ntrpu_overlap_physical import digest, save

ROOT = Path(__file__).resolve().parents[1]
SOURCES = FROZEN_SOURCES + ['tools/prepare_ntrpu_return_boundary.py',
    'tools/ntrpu_return_boundary.tcl', 'tools/run_ntrpu_return_boundary.py']


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--stage', choices=('boundary',), default='boundary')
    ap.add_argument('--timeout', type=int, default=900); a = ap.parse_args()
    assert 1 <= a.timeout <= 900
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    hashes = {n: digest(ROOT/n) for n in SOURCES}; ref = a.reference.resolve()
    old = json.loads((ref/'result.json').read_text()); files = json.loads((ref/'FILES.json').read_text())
    assert old['status'] == 'EXPORTED_PENDING_REVIEW' and old['stage'] == 'rtl'
    assert old['experiment'] == 'registered_return_axi'
    assert old['sources'] == old['sources_after'] == {n: hashes[n] for n in FROZEN_SOURCES}
    for directory in ('rtl_evidence', 'plan_evidence'):
        for p in (ref/directory).rglob('*'):
            if p.is_file(): assert digest(p) == files[p.relative_to(ref).as_posix()]
        shutil.copytree(ref/directory, out/directory)
    inputs = out/'inputs'; inputs.mkdir()
    for n in ['result.json','suite/generated/RETURN_CUT.json','suite/generated/AXI_INTEGRATION.json','suite/ip_config.tsv','suite/perf.log']:
        assert digest(ref/n) == files[n]; shutil.copy2(ref/n, inputs/Path(n).name)
    shutil.copy2(ref/'FILES.json', inputs/'FILES.json')
    tree = lambda d: {p.relative_to(out/d).as_posix(): digest(p) for p in (out/d).rglob('*') if p.is_file()}
    cpus = sorted(os.sched_getaffinity(0))[-2:]; os.sched_setaffinity(0, cpus); os.nice(10)
    os.environ['OMP_NUM_THREADS'] = '2'
    record = dict(status='RUNNING', experiment='return_boundary_sweep', stage='boundary', sources=hashes,
                  cpus=cpus, nice=os.nice(0), load_before=os.getloadavg(), timeout=a.timeout,
                  reference=str(ref), reference_result_sha256=files['result.json'])
    for d in ('inputs','rtl_evidence','plan_evidence'): record[d] = tree(d)
    save(out/'result.json', record)
    cmd = ['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_return_boundary.tcl'),
           '-log',str(out/'vivado.log'),'-journal',str(out/'vivado.jou'),'-tclargs',str(out/'suite')]
    start = time.monotonic(); expired = False
    with (out/'console.log').open('w') as log:
        job = subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code = job.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            expired = True; os.killpg(job.pid, signal.SIGTERM)
            try: code = job.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(job.pid, signal.SIGKILL); code = job.wait()
    record.update(returncode=code,timed_out=expired,seconds=round(time.monotonic()-start,3),command=cmd,
                  sources_after={n:digest(ROOT/n) for n in SOURCES})
    for d in ('inputs','rtl_evidence','plan_evidence'): record[d+'_after'] = tree(d)
    record['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not expired else 'FAILED_OR_TIMEOUT'
    if any(record[n] != record[n+'_after'] for n in ('sources','inputs','rtl_evidence','plan_evidence')):
        record['status'] = 'INPUTS_CHANGED'
    save(out/'result.json',record)
    for p in (out/'suite/project').rglob('*.log'):
        dest = out/'tool_logs'/p.relative_to(out/'suite/project'); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    save(out/'FILES.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()
                          and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:record[k] for k in ('stage','status','seconds')}),flush=True)
    return 0 if record['status'] == 'EXPORTED_PENDING_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
