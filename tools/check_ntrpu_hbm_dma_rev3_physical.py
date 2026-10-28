"""Verify revision-3 checkpoint provenance and the unchanged registered boundary."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from check_ntrpu_hbm_dma_rev2_review import digest, require, verify_stage
from check_ntrpu_hbm_dma_rev2_physical import capture, controls, rows, summary
from check_ntrpu_hbm_dma_boundary import boundary, inventory, polarity, reset_reproducer
from check_ntrpu_hbm_dma_boundary import exceptions as original_exceptions, negative_checks as original_negative_checks
from check_ntrpu_hbm_dma_rev3_capture import review as capture_review, cache_registers


def literal_ports(text):
    # write_xdc can serialize a collection as individual get_ports queries.
    # Normalize only literal forms, preserving each exact name; never eval Tcl.
    text = re.sub(r'\\\r?\n\s*', ' ', text)
    pattern = r'\[list ((?:\[get_ports (?:\{[^{}\n]+\}|[\w/]+)\]\s*)+)\]'
    def replace(match):
        names = re.findall(r'\[get_ports (\{[^{}\n]+\}|[\w/]+)\]', match[1])
        return '[get_ports [list ' + ' '.join(names) + ']]'
    return re.sub(pattern, replace, text)


def exceptions(text, manifest, reset):
    return original_exceptions(literal_ports(text), manifest, reset)


def negative_checks(manifest, connectivity, ports, effective, reset):
    count = original_negative_checks(manifest, connectivity, ports, literal_ports(effective), reset)
    altered = effective.replace('[get_ports {ack_event[0]}]', '[get_ports {ack_event[99999]}]', 1)
    if altered == effective:
        normalized = literal_ports(effective)
        lines = normalized.splitlines()
        index = next(i for i, l in enumerate(lines) if l.startswith('set_false_path -from [get_ports [list '))
        before = lines[index]
        lines[index] = before.replace('{ack_event[0]}', '{ack_event[99999]}', 1)
        require(lines[index] != before, 'Missing port for parser control')
        altered = '\n'.join(lines)
    try:
        exceptions(altered, boundary(manifest, connectivity, ports), reset)
    except ValueError:
        return count + 1
    raise ValueError('Changed literal port scope was accepted')


def reset_defaults(reports):
    model = rows((reports / 'reset_library.tsv').read_text())
    require(len(model) == 1 and model[0]['sha256'] ==
            '74d30d7c788e7db47f12b92d5b4b57b707974a0962708b309e4cc081d5f98b04' and
            model[0]['init_default'] == model[0]['clr_inverted_default'] == '0', 'Wrong FDCE default evidence')
    records = rows((reports / 'reset_polarity.tsv').read_text())
    polarity((reports / 'reset_polarity.tsv').read_text(), (reports / 'cold_reset_endpoints.txt').read_text())
    for r in records:
        raw = r['raw_clr_inverted']
        value = 0 if raw in ('', '0', "1'b0", 'false', 'FALSE') else 1 if raw in ('1', "1'b1", 'true', 'TRUE') else None
        require(value is not None and str(value) == r['clr_inverted'], 'Unexplained inversion normalization')
        require(r['raw_init'] in ('', "1'b0") and r['init'] == "1'b0", 'Unexplained INIT normalization')
    return dict(default_inversion=sum(r['raw_clr_inverted'] == '' for r in records),
                default_init=sum(r['raw_init'] == '' for r in records))


def review(behavior, synth, capture_stage, prepare, route, audit, placement, baseline=None, reset_probe=None, insertion=None):
    precheck = capture_review(behavior, synth, capture_stage)
    ports = synth / 'output/reports/ports.tsv'
    stages = [verify_stage(p) for p in (prepare, route, audit)]
    baseline_seconds = None
    physical = route
    if baseline is not None:
        br = verify_stage(baseline)
        baseline_seconds = br['seconds']
        require(digest(baseline / 'output/reports/routed.dcp') == digest(route / 'repo/evidence/routed.dcp'),
                'Optimized a different routed checkpoint')
        polarity((route / 'output/reports/before_reset/reset_polarity.tsv').read_text(),
                 (route / 'output/reports/before_reset/cold_reset_endpoints.txt').read_text())
        physical = baseline
    require(digest(prepare / 'repo/evidence/controls_constrained.dcp') ==
            precheck['constrained_dcp_sha256'], 'Wrong revision-3 constrained donor')
    if (prepare / 'repo/evidence/anchors_unconstrained.dcp').exists():
        require(insertion is not None, 'Recovered insertion needs original provenance')
        io = insertion / 'output'
        for n, sha in json.loads((io / 'FILES.json').read_text()).items():
            require(digest(io / n) == sha, 'Changed insertion evidence')
        ir = json.loads((io / 'result.json').read_text())
        require(ir['status'] == 'FAILED' and ir['returncode'] == 1 and not ir['timed_out'] and
                ir['sources'] == ir['sources_after'], 'Wrong insertion rejection')
        require('Unknown CLR polarity: sr_reg[2] = ' in (io / 'console.log').read_text(), 'Different insertion failure')
        for n, sha in ir['sources'].items():
            require(digest(insertion / 'repo' / n) == sha, 'Changed insertion source')
        require(digest(insertion / 'repo/evidence/controls_constrained.dcp') == precheck['constrained_dcp_sha256'], 'Wrong insertion donor')
        require(digest(io / 'reports/anchors_unconstrained.dcp') == digest(prepare / 'repo/evidence/anchors_unconstrained.dcp'), 'Wrong recovered checkpoint')
        require((io / 'reports/boundary.tsv').read_text() == (prepare / 'output/reports/boundary.tsv').read_text()
                and (io / 'reports/connectivity.tsv').read_text() == (prepare / 'output/reports/connectivity.tsv').read_text(),
                'Recovered topology differs')
    placement_seconds = None
    if (physical / 'repo/evidence/placed.dcp').exists():
        require(placement is not None, 'Resumed route requires placement provenance')
        po = placement / 'output'
        for n, sha in json.loads((po / 'FILES.json').read_text()).items():
            require(digest(po / n) == sha, 'Changed placement evidence')
        pr = json.loads((po / 'result.json').read_text())
        require(pr['status'] == 'PENDING_REVIEW' and pr['returncode'] == 0 and not pr['timed_out'] and pr['sources'] == pr['sources_after'],
                'Placement did not complete cleanly')
        for n, sha in pr['sources'].items():
            require(digest(placement / 'repo' / n) == sha, 'Changed placement source')
        require('place_design completed successfully' in (po / 'console.log').read_text(), 'Incomplete placement')
        require(digest(prepare / 'output/reports/prepared.dcp') == digest(placement / 'repo/evidence/prepared.dcp'),
                'Different placement input')
        require(digest(po / 'reports/placed.dcp') == digest(physical / 'repo/evidence/placed.dcp'), 'Different resumed placement')
        placement_seconds = pr['seconds']
    else:
        require(digest(prepare / 'output/reports/prepared.dcp') == digest(physical / 'repo/evidence/prepared.dcp'),
                'Wrong physical input checkpoint')
    require(digest(route / 'output/reports/routed.dcp') == digest(audit / 'repo/evidence/routed.dcp'),
            'Wrong audit input checkpoint')
    pr, rr, ar = [p / 'output/reports' for p in (prepare, route, audit)]
    read = lambda p, name: (p / name).read_text()
    manifest = boundary(read(pr, 'boundary.tsv'), read(pr, 'connectivity.tsv'), ports.read_text())
    for p in (pr, rr, ar):
        reset_defaults(p)
        require(read(p, 'boundary.tsv') == read(pr, 'boundary.tsv'), 'Boundary manifest changed')
        exceptions(read(p, 'effective_constraints.xdc'), manifest, read(p, 'cold_reset_endpoints.txt'))
        require('No ignored timing exceptions found.' in read(p, 'ignored_exceptions.rpt'), 'Ignored exception')
        anchors = rows(read(p, 'anchors.tsv'))
        require(len(anchors) == 11375 and
                {tuple(r[k] for k in ('port', 'direction', 'clock', 'anchor')) for r in anchors} ==
                {tuple(r[k] for k in ('port', 'direction', 'clock', 'anchor')) for r in manifest},
                'Incomplete anchor re-audit')
        if p != pr:
            require(all(r['loc'].startswith('SLICE_X') for r in anchors), 'Unplaced anchor')
    timing = summary(read(ar, 'timing.rpt'))
    require(timing == summary(read(rr, 'timing.rpt')), 'Reopened timing changed')
    require(read(ar, 'metrics.tsv') == read(rr, 'metrics.tsv'), 'Reopened metrics changed')
    final_anchors = rows(read(ar, 'anchors.tsv'))
    require(all(not r['core_ports'] and r['core_pins'] and r['core_route'] in ('ROUTED', 'INTRASITE')
                and r['outer_route'] == 'HIERPORT' for r in final_anchors), 'Unrouted or externally dependent core boundary')
    metrics = {r['metric']: r['value'] for r in rows(read(ar, 'metrics.tsv'))}
    require(metrics['async_registers'] == '98' and metrics['blackboxes'] == metrics['latches'] == '0',
            'Changed core inventory')
    clocks = rows(read(ar, 'clock_buffers.tsv'))
    require(len(clocks) == 2 and {r['clock'] for r in clocks} == {'reader_clk', 'dma_clk'}, 'Wrong physical clocks')
    for r in clocks:
        require(float(r['period']) == (5 if r['clock'] == 'reader_clk' else 4)
                and r['loc'].startswith('BUFGCE_X') and r['route_status'] == 'ROUTED'
                and int(r['loads']) > 0, 'Unbuffered/unrouted clock')
    routing = {k: int(v) for k, v in re.findall(r'# of (.+?)\.+\s*:\s*(\d+)', read(ar, 'route_status.rpt'))}
    require(routing['fully routed nets'] == routing['routable nets'] > 0 and routing['nets with routing errors'] == 0,
            'Incomplete routing')
    coverage = {k: int(v) for k, v in re.findall(r'checking ([a-z_]+) \((\d+)\)', read(ar, 'check_timing.rpt'))}
    require(set(coverage) == {'no_clock', 'constant_clock', 'pulse_width_clock', 'unconstrained_internal_endpoints',
                             'no_input_delay', 'no_output_delay', 'multiple_clock', 'generated_clocks', 'loops',
                             'partial_input_delay', 'partial_output_delay', 'latch_loops'}, 'Incomplete coverage report')
    require(all(v == (1 if k == 'no_input_delay' else 0) for k, v in coverage.items()), 'Timing coverage changed')
    cdc = inventory(read(ar, 'cdc.rpt'))
    clocked = inventory(read(ar, 'clocked_cdc.rpt'))
    require(cdc and clocked and not any(v['severity'] == 'Critical' for v in cdc.values()), 'Critical/unparsed CDC')
    require({k: v['count'] for k, v in clocked.items()} == {'CDC-3': 6, 'CDC-6': 14, 'CDC-15': 1318},
            'Clocked CDC inventory changed')
    reset_reproduction = None
    if cdc.get('CDC-26', {}).get('count', 0):
        require(reset_probe is not None, 'Reset CDC warnings require the controlled reproduction')
        reset_reproduction = reset_reproducer(reset_probe)
        require({k: v['count'] for k, v in cdc.items()} == {'CDC-3': 6, 'CDC-6': 14, 'CDC-15': 1318, 'CDC-26': 4},
                'Unreviewed raw CDC finding')
        destinations = re.findall(r'^\s*\d+\s+CDC-26[^\n]*cold_reset_n\s+(\S+)\s*$', read(ar, 'cdc.rpt'), re.M)
        require(len(destinations) == 4 and set(destinations) ==
                {'dr_reg[0]/CLR', 'sr_reg[0]/CLR', 'link_s1_reg/CLR', 'ready_s1_reg/CLR'}, 'Different CDC-26 endpoints')
    else:
        require({k: v['count'] for k, v in cdc.items()} == {'CDC-3': 6, 'CDC-6': 14, 'CDC-15': 1318, 'CDC-9': 2},
                'Unreviewed raw CDC finding')
    data_slack = capture(read(ar, 'data_bindings.tsv'), read(ar, 'data_paths.tsv'))
    control_slack = controls(read(ar, 'control_paths.tsv'), (prepare / 'repo/evidence/bindings.tsv').read_text())
    cov = read(ar, 'exceptions.rpt')
    groups = re.findall(r'^\d+\s+Max Delay Datapath Only\s+max_dpo=(\d+)\s+-\s+(\d+) cells\s+(\d+) cells\s+(\d+)\s+([\d.]+)\s+33\.33\s*$', cov, re.M)
    require(len(groups) == 19 and sum(int(x[3]) for x in groups) == 45, 'Missing control constraint coverage')
    require(sum(x[0] == '5' for x in groups) == 8 and sum(x[0] == '4' for x in groups) == 11, 'Wrong source-period bounds')
    require(all(int(src) >= int(dst) == int(n) and abs(float(percent) - 100 * int(n) / int(src)) < .01
                for _, src, dst, n, percent in groups), 'Invalid control coverage')
    data = re.findall(r'^\d+\s+Max Delay Datapath Only\s+max_dpo=(\d+)\s+-\s+(\d+) cells\s+(\d+) pins\s+(\d+)\s+100\.00\s+100\.00\s*$', cov, re.M)
    require(sorted(tuple(map(int, r)) for r in data) ==
            sorted([(4, 143, 143, 143), (4, 514, 514, 514), (5, 2, 2, 2)] * 2 + [(4, 1, 1, 1)]),
            'Capture/link constraint coverage changed')
    skew = [tuple(map(float, x)) for x in re.findall(r'^\s+(?:Slow|Fast)\s+([.0-9]+)\s+([.0-9]+)\s+([-0-9.]+)\s*$',
                                                        read(ar, 'bus_skew.rpt'), re.M)]
    require(len(skew) == 12 and all(req == 4 and abs(req - actual - slack) < .002 for req, actual, slack in skew),
            'Wrong skew coverage')
    skew_slack = min(v[2] for v in skew)
    link = re.search(r'Slack \((?:MET|VIOLATED)\)\s*:\s*([-\d.]+)ns', read(ar, 'link_timing.rpt'))
    require(link is not None, 'Missing link timing')
    link_slack = float(link[1])
    drc = re.findall(r'^\|\s*([A-Z]+-\d+)\s*\|\s*(Critical Warning|Warning|Error)\s*\|[^|]+\|\s*(\d+)\s*\|',
                     read(ar, 'drc.rpt'), re.M)
    require(sum(int(n) for _, _, n in drc) == int(re.search(r'Checks found:\s+(\d+)', read(ar, 'drc.rpt'))[1]),
            'Unparsed DRC')
    require(not any(level in ('Error', 'Critical Warning') for _, level, _ in drc), 'Severe DRC')
    require(drc == [('RTSTAT-10', 'Warning', '1')] and
            '3713 net(s) have no routable loads.' in read(ar, 'drc.rpt'), 'Unexpected unroutable-load inventory')
    listed_outer = re.findall(r'ntrpu_boundary_ff_\d{5}_outer', read(ar, 'drc.rpt'))
    require(listed_outer and set(listed_outer) <= {r['anchor'] + '_outer' for r in manifest if r['direction'] == 'OUT'},
            'DRC no-load warning reaches the measured core boundary')
    require(read(ar, 'cache_registers.tsv') == read(rr, 'cache_registers.tsv') and
            read(ar, 'cache_metrics.tsv') == read(rr, 'cache_metrics.tsv'), 'Reopened cache evidence changed')
    cache_counts = cache_registers(read(ar, 'cache_registers.tsv'), placed=True)
    cache_paths = rows(read(ar, 'cache_metrics.tsv'))
    require(len(cache_paths) == 4 and {r['group']: int(r['registers']) for r in cache_paths} == cache_counts,
            'Cache timing coverage mismatch')
    require(all(float(r['setup_ns']) >= timing['setup']['slack_ns'] - .002 and
                float(r['hold_ns']) >= timing['hold']['slack_ns'] - .002 for r in cache_paths), 'Inconsistent cache timing')
    closed = (all(v['slack_ns'] >= 0 and v['failed'] == 0 for v in timing.values()) and
              min(data_slack, control_slack, skew_slack, link_slack) >= 0)
    negative_count = negative_checks(read(pr, 'boundary.tsv'), read(pr, 'connectivity.tsv'), ports.read_text(),
                                     read(ar, 'effective_constraints.xdc'), read(ar, 'cold_reset_endpoints.txt'))
    reset_representation = polarity(read(ar, 'reset_polarity.tsv'), read(ar, 'cold_reset_endpoints.txt'))
    bad_polarity = read(ar, 'reset_polarity.tsv').replace('\tCLR_ATTRIBUTE\t1\t', '\tCLR_ATTRIBUTE\t0\t', 1)
    if bad_polarity == read(ar, 'reset_polarity.tsv'):
        bad_polarity = bad_polarity.replace('\tLUT1\t0\t', '\tLUT1\t1\t', 1)
    try:
        polarity(bad_polarity, read(ar, 'cold_reset_endpoints.txt'))
    except ValueError:
        negative_count += 1
    else:
        raise ValueError('Wrong reset polarity was accepted')
    return dict(status='REV3_BOUNDARY_ROUTED_TIMING_' + ('PASS' if closed else 'OPEN'),
                evidence_verified=True, capture_review=precheck, cache_registers=cache_counts, cache_timing=cache_paths,
                reset_defaults_prepared=reset_defaults(pr), reset_defaults_routed=reset_defaults(ar),
                recovered_insertion=insertion is not None,
                harness_timing_closed=closed, protocol_adapter_validated=False,
                native_hbm_xdma_integrated=False, external_io_qualified=False, board_qualified=False,
                anchors=11375, timing=timing, domain_slack_ns={k: float(v) for k, v in metrics.items() if '_clk_' in k},
                core_boundary_routing=dict(Counter(r['core_route'] for r in final_anchors)),
                excluded_outer_routing=dict(Counter(r['outer_route'] for r in final_anchors)),
                no_load_drc_external_capture_outputs_only=True,
                routing=routing, cdc=cdc, clocked_cdc=clocked, drc=drc, timing_coverage=coverage,
                clock_buffers=clocks, reset_integration_qualified=False,
                cdc26_tool_warnings_remaining=cdc.get('CDC-26', {}).get('count', 0),
                reset_diagnostic_reproduction=reset_reproduction,
                negative_controls=negative_count,
                placement_seconds=placement_seconds,
                baseline_seconds=baseline_seconds, baseline_original_review_rejected=False,
                post_route_optimization='Explore' if baseline is not None else None,
                reset_representation=dict(reset_representation),
                route_directive='RuntimeOptimized' if placement_seconds is not None else 'Default',
                min_capture_slack_ns=data_slack, min_control_slack_ns=control_slack,
                min_link_slack_ns=link_slack, min_gray_skew_slack_ns=skew_slack,
                stage_seconds=[s['seconds'] for s in stages], routed_dcp_sha256=digest(rr / 'routed.dcp'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('behavior', 'synth', 'capture_stage', 'prepare', 'route', 'audit', 'placement'):
        ap.add_argument(name, type=Path)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--reset-probe', type=Path)
    ap.add_argument('--insertion', type=Path)
    a = ap.parse_args()
    print(json.dumps(review(a.behavior, a.synth, a.capture_stage, a.prepare, a.route, a.audit,
                            a.placement, a.baseline, a.reset_probe, a.insertion), indent=2))
