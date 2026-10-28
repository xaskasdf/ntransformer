"""Bounded physical stages for the frozen, functionally reviewed HBM transport."""
import argparse, hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path
from run_ntrpu_hbm_cdc_audit import SOURCES as AUDIT_SOURCES

ROOT = Path(__file__).resolve().parents[1]
SOURCES = AUDIT_SOURCES + [
    'fpga/constraints/ntrpu_hbm_startup_cdc.xdc',
    'tools/run_ntrpu_hbm_physical.py', 'tools/ntrpu_hbm_physical.tcl']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v): p.write_text(json.dumps(v, indent=2)+'\n')

def wrapper(text):
    declaration = text.split('module ntrpu_hbm_transport_async (', 1)[1].split('\n);', 1)[0]
    return ('`timescale 1ns/1ps\nmodule ntrpu_hbm_physical ('+declaration+'\n);\n'
            '    wire core_clk, core_mem_clk;\n'
            '    BUFGCE source_clock(.I(clk), .CE(1\'b1), .O(core_clk));\n'
            '    BUFGCE memory_clock(.I(mem_clk), .CE(1\'b1), .O(core_mem_clk));\n'
            '    ntrpu_hbm_transport_async core(.clk(core_clk), .mem_clk(core_mem_clk), .*);\n'
            'endmodule\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--stage', choices=('synth', 'place', 'route'), required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    a = ap.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    suite = out/'suite'; suite.mkdir()
    old = json.loads((a.prior/'result.json').read_text())
    files = json.loads((a.prior/'FILES.json').read_text())
    review = json.loads(a.review.read_text()); hashes = {n:sha(ROOT/n) for n in SOURCES}
    assert old['status'] == 'EXPORTED_PENDING_REVIEW'
    assert old['sources'] == old['sources_after']
    updates={n:dict(before=h,after=hashes[n]) for n,h in old['sources'].items() if hashes[n]!=h}
    # Stage-driver corrections are recorded; RTL, IP setup, constraints and
    # previously qualified generators must remain byte-identical.
    assert set(updates)<= {'tools/run_ntrpu_hbm_physical.py','tools/ntrpu_hbm_physical.tcl'}
    checkpoint = None
    if a.stage == 'synth':
        assert review['status'] == 'CDC_STRUCTURE_AUDITED_TIMING_OPEN'
        assert old['experiment'] == 'hbm_cdc_audit'
        assert old['generated'] == old['generated_after']
        for n,h in old['generated'].items():
            assert sha(a.prior/n) == h == files[n]
            dest = out/n; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a.prior/n, dest)
        (suite/'generated/ntrpu_hbm_physical.sv').write_text(wrapper((ROOT/'fpga/rtl/ntrpu_hbm_transport_async.sv').read_text()))
    else:
        previous = {'place':'synth', 'route':'place'}[a.stage]
        assert old['stage'] == previous and review['evidence_validated']
        relative=review.get('checkpoint_relative_path','suite/physical_'+previous+'.dcp')
        source=a.prior/relative
        assert source.resolve().is_relative_to(a.prior.resolve())
        assert review['checkpoint_sha256'] == files[relative]
        assert sha(source) == review['checkpoint_sha256']
        checkpoint = out/'input.dcp'; shutil.copy2(source, checkpoint)
    shutil.copy2(a.review, out/'PRIOR_REVIEW.json')
    generated = lambda: {p.relative_to(out).as_posix():sha(p) for p in (suite/'generated').glob('*') if p.is_file()}
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:]); os.nice(10)
    os.environ['OMP_NUM_THREADS'] = '2'
    r = dict(status='RUNNING', experiment='hbm_physical', stage=a.stage,
             sources=hashes, generated=generated(), prior=str(a.prior), tool_updates=updates,
             prior_result_sha256=sha(a.prior/'result.json'), prior_review_sha256=sha(a.review),
             input_sha256=sha(checkpoint) if checkpoint else None,
             timeout=900, cpus=sorted(os.sched_getaffinity(0)), nice=os.nice(0))
    save(out/'result.json', r)
    cmd = ['vivado', '-mode', 'batch', '-notrace', '-source', str(ROOT/'tools/ntrpu_hbm_physical.tcl'),
           '-log', str(out/'vivado.log'), '-journal', str(out/'vivado.jou'),
           '-tclargs', str(suite), a.stage, str(checkpoint) if checkpoint else '-']
    start = time.monotonic(); expired = False
    with (out/'console.log').open('w') as log:
        job = subprocess.Popen(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try: code = job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired = True; os.killpg(job.pid, signal.SIGTERM)
            try: code = job.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(job.pid, signal.SIGKILL); code = job.wait()
    r.update(returncode=code, timed_out=expired, seconds=round(time.monotonic()-start, 3), command=cmd,
             sources_after={n:sha(ROOT/n) for n in SOURCES}, generated_after=generated(),
             input_after=sha(checkpoint) if checkpoint else None)
    r['status'] = 'EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] and r['generated']==r['generated_after'] and r['input_sha256']==r['input_after'] else 'FAILED_OR_CHANGED'
    save(out/'result.json', r)
    for p in (suite/'project').rglob('*.log'):
        dest = out/'tool_logs'/p.relative_to(suite/'project'); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    save(out/'FILES.json', {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('stage','status','seconds')}), flush=True)

if __name__ == '__main__': main()
