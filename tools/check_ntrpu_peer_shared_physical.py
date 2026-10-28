"""Audit shared-source synthesis/routing; keep timing and board claims separate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from check_ntrpu_hbm_audit import cdc, report_checks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def timing_summary(text):
    m = re.search(r'^\s*WNS\(ns\)[^\n]*\n\s*[- ]+\n\s*([^\n]+)', text, re.M)
    need(m is not None, 'missing design timing summary')
    values = list(map(float, m[1].split()))
    need(len(values) == 12, 'incomplete timing summary')
    result = {}
    for i, name in enumerate(('setup', 'hold', 'pulse_width')):
        worst, total, failed, endpoints = values[i * 4:i * 4 + 4]
        need(endpoints > 0 and endpoints.is_integer() and failed.is_integer() and 0 <= failed <= endpoints, 'timing endpoint counts')
        result[name] = dict(worst_slack_ns=worst, total_slack_ns=total,
                            failing_endpoints=int(failed), endpoints=int(endpoints))
    return result


def resources(text):
    def count(name):
        m = re.search(r'\|\s*' + re.escape(name) + r'\*?\s*\|\s*([\d.]+)', text)
        need(m is not None, 'missing resource: ' + name)
        return float(m[1])
    return dict(luts=count('CLB LUTs'), registers=count('CLB Registers'),
                bram36_equivalent=count('Block RAM Tile'))


def common_metrics(metrics, clocks, checks, counts):
    need(metrics['latches'] == metrics['blackboxes'] == 0, 'latch or black box')
    need(metrics['startup_async_registers'] == 21 and metrics['internal_status_pins'] == 3 and
         metrics['external_status_pins'] == 2, 'status synchronizer inventory')
    expected = ('create_clock -name transport_clk -period 5.000 [get_ports clk]',
                'create_clock -name memory_clk -period 2.222 [get_ports mem_clk]',
                'set_clock_uncertainty -setup 0.100 [get_clocks {transport_clk memory_clk}]',
                'set_clock_uncertainty -hold 0.050 [get_clocks {transport_clk memory_clk}]')
    need(all(line in clocks for line in expected) and 'set_clock_groups' not in clocks, 'clock constraints changed')
    need(all(checks[k] == 0 for k in ('no_clock', 'unconstrained_internal_endpoints', 'loops', 'latch_loops', 'multiple_clock')), 'internal timing coverage')
    need(set(counts) <= {'CDC-3', 'CDC-9'} and sum(counts.values()) > 0, 'unexpected CDC structure')


def negative_controls(metrics, clocks, checks, counts):
    controls = []
    for kind in ('latches', 'blackboxes', 'internal_status_pins', 'startup_async_registers'):
        m = dict(metrics); m[kind] += 1
        controls.append((kind, m, clocks, checks, counts))
    for kind in ('no_clock', 'unconstrained_internal_endpoints', 'loops'):
        c = dict(checks); c[kind] += 1
        controls.append((kind, metrics, clocks, c, counts))
    controls.append(('relaxed_clock', metrics, clocks.replace('period 2.222', 'period 3.000'), checks, counts))
    controls.append(('unsafe_cdc', metrics, clocks, checks, dict(counts, **{'CDC-1': 1})))
    rejected = []
    for name, m, cl, ch, cd in controls:
        try:
            common_metrics(m, cl, ch, cd)
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError('accepted audit negative control: ' + name)
    return rejected


def review(out):
    result = json.loads((out / 'result.json').read_text())
    need(result['experiment'] == 'peer_shared_physical' and result['status'] == 'EXPORTED_PENDING_REVIEW', 'incomplete physical run')
    need(result['returncode'] == 0 and not result['timed_out'] and result['donor_unchanged'] and
         result['sources'] == result['sources_after'], 'run or donor changed')
    for name, digest in json.loads((out / 'FILES.json').read_text()).items():
        path = (out / name).resolve()
        need(path.is_relative_to(out.resolve()) and sha(path) == digest, 'artifact hash: ' + name)
    for name, digest in result['sources'].items():
        path = (out.parent / 'repo' / name).resolve()
        need(path.is_relative_to((out.parent / 'repo').resolve()) and sha(path) == digest, 'source hash: ' + name)
    prior = json.loads((out / 'PRIOR_REVIEW.json').read_text())
    need(prior['result_sha256'] == result['prior_result_sha256'] == sha(out / 'DONOR_RESULT.json'), 'functional/synthesis donor binding')
    donor = json.loads((out / 'DONOR_RESULT.json').read_text())
    need(donor['sources'] == donor['sources_after'] and
         all(result['sources'].get(n) == h for n, h in donor['sources'].items()), 'donor sources differ')
    suite = out / 'suite'
    prefix = 'synthesis_' if result['stage'] == 'synth' else ''
    metrics = {k: float(v) for k, v in (line.split('\t') for line in (suite / (prefix + 'summary.tsv')).read_text().splitlines())}
    checks = report_checks(suite / (prefix + 'check_timing.rpt'))
    counts = cdc(suite / (prefix + 'cdc.rpt'))
    clocks = (suite / 'clocks.xdc').read_text()
    common_metrics(metrics, clocks, checks, counts)
    wrapper = (suite / 'ntrpu_peer_shared_physical.sv').read_text()
    need(wrapper.count('BUFGCE ') == 2 and 'ntrpu_peer_read_shared core' in wrapper, 'unexpected physical top')
    profile = json.loads((out.parent / 'repo/NATIVE_PROFILE.json').read_text())
    for name, config in profile['selected_read_configs'].items():
        observed = dict(line.split('\t', 1) for line in (suite / name).read_text().splitlines() if line.startswith('CONFIG.'))
        need(observed == config, 'read IP configuration changed: ' + name)
    summary = dict(stage=result['stage'], result_sha256=sha(out / 'result.json'), seconds=result['seconds'],
                   cdc=counts, coverage=checks, **resources((suite / (prefix + 'utilization.rpt')).read_text()),
                   global_timing=timing_summary((suite / (prefix + 'timing.rpt')).read_text()),
                   negative_controls=negative_controls(metrics, clocks, checks, counts),
                   native_hbm_in_design=False, external_io_qualified=False, post_route_simulated=False,
                   board_qualified=False)
    if result['stage'] == 'synth':
        need(prior['status'] == 'SHARED_TRACE_REVIEWED' and prior['mode'] == 'full_behavioral', 'unreviewed functional donor')
        summary.update(status='SHARED_SYNTHESIS_REVIEWED', qualified_timing=False,
                       checkpoint_sha256=sha(suite / 'synthesized.dcp'))
        return summary
    need(prior['status'] == 'SHARED_SYNTHESIS_REVIEWED' and
         result['input_dcp_sha256'] == sha(out / 'input.dcp'), 'unreviewed input checkpoint')
    if result.get('resumed'):
        resume = json.loads((out / 'RESUME_RESULT.json').read_text())
        placed = json.loads((out / 'PLACED_PROVENANCE.json').read_text())
        need(sha(out / 'RESUME_RESULT.json') == result['resume_result_sha256'] and
             resume['sources'] == resume['sources_after'] == placed['sources'] and
             all(result['sources'].get(n) == h for n, h in resume['sources'].items()), 'resume source binding')
        need(placed['checkpoint_sha256'] == result['input_dcp_sha256'] and
             placed['synthesis_result_sha256'] == resume['prior_result_sha256'] == prior['result_sha256'], 'placed checkpoint provenance')
    else:
        need(result['input_dcp_sha256'] == prior['checkpoint_sha256'], 'synthesis checkpoint binding')
    need(metrics['period_transport_clk'] == 5 and metrics['period_memory_clk'] == 2.222, 'implemented clock periods changed')
    ready = list(csv.DictReader((suite / 'ready_fanin.tsv').read_text().splitlines(), delimiter='\t'))
    need({v['output'] for v in ready} == {'h_rready[0]', 'h_rready[1]'} and all(v['class'] == 'pin' for v in ready), 'combinatorial native READY')
    text = (suite / 'cdc.rpt').read_text()
    for endpoint in ('fault_meta_reg[0]/D', 'fault_meta_reg[1]/D', 'ready_meta_reg[2]/D'):
        need(any(endpoint in line and 'CDC-3' in line and 'Max Delay Datapath Only' in line for line in text.splitlines()), 'missing targeted CDC constraint')
    drc = (suite / 'drc.rpt').read_text()
    rows = re.findall(r'^\|\s*([A-Z]+-\d+)\s*\|\s*(Critical Warning|Warning|Error)\s*\|[^|]+\|\s*(\d+)\s*\|', drc, re.M)
    need(sum(int(n) for _, _, n in rows) == int(re.search(r'Checks found:\s+(\d+)', drc)[1]), 'unparsed DRC')
    need(all(level == 'Warning' and rule in ('PDCN-1569', 'RTSTAT-10') for rule, level, n in rows), 'new DRC class requires review')
    routing = {k: int(v) for k, v in re.findall(r'# of (.+?)\.+\s*:\s*(\d+)', (suite / 'route_status.rpt').read_text())}
    need(routing['fully routed nets'] == routing['routable nets'] > 0 and routing['nets with routing errors'] == 0, 'incomplete routing')
    skew = (suite / 'bus_skew.rpt').read_text()
    skews = [tuple(map(float, row)) for row in re.findall(r'^\s+(?:Slow|Fast)\s+([.0-9]+)\s+([.0-9]+)\s+([-0-9.]+)\s*$', skew, re.M)]
    need(len(skews) == 8 and all(req == 2.222 and value >= 0 and abs(req - value - slack) < .002 for req, value, slack in skews), 'bus skew inventory')
    slacks = {k: v for k, v in metrics.items() if k.startswith(('max_', 'min_'))}
    closed = all(v >= 0 for v in slacks.values()) and all(v[2] >= 0 for v in skews) and all(
        v['worst_slack_ns'] >= 0 and v['failing_endpoints'] == 0 and v['total_slack_ns'] == 0 for v in summary['global_timing'].values())
    summary.update(status='SHARED_ROUTED_INTERNAL_TIMING_' + ('PASS' if closed else 'OPEN'), qualified_timing=closed,
                   clock_slack_ns=slacks, checkpoint_sha256=sha(suite / 'routed.dcp'),
                   fully_routed_nets=routing['fully routed nets'], minimum_bus_skew_slack_ns=min(v[2] for v in skews),
                   drc_findings=rows, drc_clean=not rows, drc_objects_individually_reviewed=False,
                   registered_native_ready_verified=True, ready_fanin_startpoints=len(ready))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()), sort_keys=True))
