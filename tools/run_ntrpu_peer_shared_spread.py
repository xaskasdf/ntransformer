"""Place the reviewed compact design with AltSpreadLogic_high; keep timing constraints."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from run_ntrpu_peer_read import sha, save

ROOT = Path(__file__).resolve().parents[1]


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--stage', choices=('route',), required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--cpus', type=int, choices=range(1, 9), default=4)
    ap.add_argument('--timeout', type=int, default=1200)
    a = ap.parse_args()
    assert 0 < a.timeout <= 3600
    prior, out = a.prior.resolve(), a.output.resolve()
    donor = json.loads((prior / 'result.json').read_text())
    review = json.loads(a.review.read_text())
    assert donor['status'] == 'EXPORTED_PENDING_REVIEW' and donor['sources'] == donor['sources_after']
    assert all(sha(prior / n) == h for n, h in json.loads((prior / 'FILES.json').read_text()).items())
    assert all(sha(ROOT / n) == h for n, h in donor['sources'].items())
    assert review['result_sha256'] == sha(prior / 'result.json')
    if a.stage == 'synth':
        assert review['status'] == 'SHARED_TRACE_REVIEWED' and review['mode'] == 'full_behavioral'
        assert review['counters']['successful_retire'] == 27
    else:
        assert donor['experiment'] == 'peer_shared_physical' and donor['stage'] == 'synth'
        assert review['status'] == 'SHARED_SYNTHESIS_REVIEWED'
        assert sha(prior / 'suite/synthesized.dcp') == review['checkpoint_sha256']
    out.mkdir(parents=True, exist_ok=False)
    suite = out / 'suite'; suite.mkdir()
    shutil.copyfile(a.review, out / 'PRIOR_REVIEW.json')
    shutil.copyfile(prior / 'result.json', out / 'DONOR_RESULT.json')
    baseline = (ROOT / 'tools/ntrpu_peer_read_physical.tcl').read_text()
    baseline = once(baseline, '{ntrpu_peer_read_pkg ntrpu_peer_read_lane ntrpu_peer_read_pair}',
                    '{ntrpu_peer_read_pkg ntrpu_peer_read_lane ntrpu_peer_read_admission ntrpu_peer_read_shared}')
    baseline = baseline.replace('ntrpu_peer_read_physical', 'ntrpu_peer_shared_physical')
    baseline = baseline.replace('PEER_READ_PHYSICAL_EXPORTED', 'SHARED_PHYSICAL_EXPORTED')
    if a.stage == 'synth':
        declaration = (ROOT / 'fpga/experimental/ntrpu_peer_read_shared.sv').read_text().split(
            'module ntrpu_peer_read_shared (', 1)[1].split('\n);', 1)[0]
        wrapper = ('`timescale 1ns/1ps\nmodule ntrpu_peer_shared_physical (' + declaration + '\n);\n'
                   'wire core_clk,core_mem_clk;\n'
                   "BUFGCE source_clock(.I(clk),.CE(1'b1),.O(core_clk));\n"
                   "BUFGCE memory_clock(.I(mem_clk),.CE(1'b1),.O(core_mem_clk));\n"
                   'ntrpu_peer_read_shared core(.clk(core_clk),.mem_clk(core_mem_clk),.*);\nendmodule\n')
        (suite / 'ntrpu_peer_shared_physical.sv').write_text(wrapper)
        baseline = once(baseline, '    prepare_read_return $root $output',
                        '    set root {' + ROOT.as_posix() + '}\n    prepare_read_return $root $output\n    set_param general.maxThreads $threads')
        boundary = baseline.index('    opt_design\n')
        tail = baseline[baseline.index('    close_project\n'):]
        baseline = baseline[:boundary] + '''    report_timing_summary -report_unconstrained -file [file join $output synthesis_timing.rpt]
    check_timing -verbose -file [file join $output synthesis_check_timing.rpt]
    write_xdc -constraints all [file join $output synthesis_constraints.xdc]
    set f [open [file join $output synthesis_summary.tsv] w]
    puts $f "latches\\t[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LD*}]]"
    puts $f "blackboxes\\t[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]"
    puts $f "startup_async_registers\\t[llength [get_cells -hier -regexp {(^|.*/)startup/.*} -filter {ASYNC_REG == TRUE}]]"
    puts $f "internal_status_pins\\t[llength $ntrpu_status_internal_d]"
    puts $f "external_status_pins\\t[llength $ntrpu_status_apb_d]"
    close $f
''' + tail
    else:
        shutil.copyfile(prior / 'suite/synthesized.dcp', out / 'input.dcp')
        for p in (prior / 'suite').iterdir():
            if p.is_file() and p.suffix != '.dcp':
                shutil.copyfile(p, suite / p.name)
        header = '''set root [file normalize [file join [file dirname [info script]] .. repo]]
set output [file normalize [lindex $argv 0]]
if {[catch {
    set_param general.maxThreads [lindex $argv 1]
    open_checkpoint [file join $output .. input.dcp]
'''
        xdc = (ROOT / 'fpga/constraints/ntrpu_hbm_startup_cdc.xdc').read_text()
        header += ''.join('    ' + line + '\n' for line in xdc.splitlines() if line.startswith('set ntrpu_status_'))
        baseline = header + baseline[baseline.index('    opt_design\n'):]
    baseline = once(baseline, '    place_design\n', '    place_design -directive AltSpreadLogic_high\n')
    (out / 'flow.tcl').write_text(baseline)
    # Generated Tcl lives in output; its root must remain the frozen private repo.
    baseline = once(baseline, baseline.splitlines()[0], 'set root {' + ROOT.as_posix() + '}')
    (out / 'flow.tcl').write_text(baseline)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpus])
    os.nice(10); os.environ['OMP_NUM_THREADS'] = str(a.cpus)
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*') if p.is_file()}
    donor_hash = sha(prior / 'result.json')
    command = ['vivado', '-mode', 'batch', '-notrace', '-source', str(out / 'flow.tcl'),
               '-tclargs', str(suite), str(a.cpus)]
    result = dict(status='RUNNING', experiment='peer_shared_physical', stage=a.stage, sources=sources,
                  placement_directive='AltSpreadLogic_high', prior_result_sha256=donor_hash, cpus=sorted(os.sched_getaffinity(0)), nice=os.nice(0),
                  command=command, timeout=a.timeout, load_before=os.getloadavg())
    if a.stage == 'route':
        result['input_dcp_sha256'] = sha(out / 'input.dcp')
    save(out / 'result.json', result)
    start = time.monotonic(); expired = False
    with (out / 'console.log').open('w') as log:
        job = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = job.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            expired = True; os.killpg(job.pid, signal.SIGTERM)
            try:
                code = job.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(job.pid, signal.SIGKILL); code = job.wait()
    after = {n: sha(ROOT / n) for n in sources}
    unchanged = sha(prior / 'result.json') == donor_hash
    if a.stage == 'route':
        unchanged &= sha(prior / 'suite/synthesized.dcp') == result['input_dcp_sha256']
    result.update(seconds=round(time.monotonic() - start, 3), returncode=code, timed_out=expired,
                  sources_after=after, donor_unchanged=unchanged)
    marker = 'SHARED_PHYSICAL_EXPORTED' in (out / 'console.log').read_text()
    result['status'] = 'EXPORTED_PENDING_REVIEW' if code == 0 and not expired and sources == after and unchanged and marker else 'FAILED_OR_CHANGED'
    save(out / 'result.json', result)
    save(out / 'FILES.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*')
                             if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k: result[k] for k in ('status', 'stage', 'seconds', 'returncode')}))


if __name__ == '__main__':
    main()
