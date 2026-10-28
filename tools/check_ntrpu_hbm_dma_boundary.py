"""Verify a registered timing harness; does not qualify protocol or board I/O."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from check_ntrpu_hbm_dma_rev2_review import digest, require, verify_stage
from check_ntrpu_hbm_dma_rev2_physical import capture, controls, rows, summary


def boundary(manifest, connectivity, ports):
    records = rows(manifest)
    original = {r['name']: r['direction'] for r in rows(ports)
                if r['name'] not in ('clk', 'dma_clk', 'cold_reset_n')}
    require(len(records) == len(original) == 11375, 'Incomplete boundary')
    require({r['port']: r['direction'] for r in records} == original, 'Port inventory changed')
    require(len({r['anchor'] for r in records}) == 11375, 'Duplicate anchor')
    require(Counter((r['direction'], r['clock']) for r in records) ==
            {('IN', 'reader_clk'): 7651, ('IN', 'dma_clk'): 11,
             ('OUT', 'reader_clk'): 2208, ('OUT', 'dma_clk'): 1505}, 'Wrong boundary domains')
    for r in records:
        base = r['port'].split('[')[0]
        dma = base in (('dma_link_up', 'dma_run', 'dma_error', 'dma_done', 'dsc_ready', 'axis_ready')
                       if r['direction'] == 'IN' else ('dma_rst_n', 'dma_fault', 'dsc_load', 'axis_valid',
                           'axis_last', 'dsc_src', 'dsc_dst', 'axis_keep', 'dsc_len', 'dsc_ctl', 'axis_data'))
        require(r['clock'] == ('dma_clk' if dma else 'reader_clk'), 'Misclocked boundary bit')
        require(re.fullmatch(r'ntrpu_boundary_ff_\d{5}', r['anchor']), 'Unknown anchor name')
    connections = rows(connectivity)
    require(len(connections) == 11375 and
            {r['port']: r['unchanged_original_pins'] for r in connections} ==
            {r['port']: r['original_pins'] for r in records}, 'ECO core connectivity changed')
    return records


def inventory(text):
    return {rule: dict(severity=severity, count=int(count)) for rule, severity, count in
            re.findall(r'^(CDC-\d+)\s+(Critical|Warning|Info)\s+(\d+)\s', text, re.M)}


def exceptions(text, manifest, reset):
    text = re.sub(r'\\\r?\n\s*', ' ', text)
    require(not re.search(r'^\s*(?:set_clock_groups|set_multicycle_path|set_disable_timing|create_waiver)\b',
                          text, re.M), 'Unexpected timing exclusion')
    require(len(re.findall(r'^set_max_delay\b', text, re.M)) == 26, 'Wrong max-delay count')
    require(len(re.findall(r'^set_bus_skew\b', text, re.M)) == 12, 'Wrong Gray-bus count')
    lines = re.findall(r'^set_false_path[^\n]*', text, re.M)
    require(len(lines) == 3, 'Only two outer measurement cuts and cold reset are allowed')
    # Parse only the literal query forms emitted by write_xdc; never evaluate
    # Tcl from an artifact. Vivado preserves lists of individual cell queries.
    def objects(line, side, kind):
        source, dest = line.removeprefix('set_false_path -from ').split(' -to ')
        query = source if side == 'from' else dest
        prefix = '[get_' + kind + ' [list '
        if query.startswith(prefix) and query.endswith(']]'):
            value = query[len(prefix):-2]
            tokens = re.findall(r'\{([^{}]+)\}|([^\s{}]+)', value)
            require(not re.sub(r'\{[^{}]+\}|[^\s{}]+', '', value).strip(), 'Unparsed literal object list')
            return [a or b for a, b in tokens]
        match = re.fullmatch(r'\[get_' + kind + r' ([\w/]+)\]', query)
        if match:
            return [match[1]]
        if query.startswith('[list ') and query.endswith(']'):
            value = query[6:-1]
            pattern = r'\[get_' + kind + r' ([\w/]+)\]'
            found = re.findall(pattern, value)
            require(found and not re.sub(pattern, '', value).strip(), 'Nonliteral exception object query')
            return found
        raise ValueError('Unparsed exception scope')
    reset_line = next((l for l in lines if 'cold_reset_n' in l), '')
    require(objects(reset_line, 'from', 'ports') == ['cold_reset_n'], 'Expanded cold reset source')
    require(reset_line == 'set_false_path -from [get_ports cold_reset_n] '
            '-to [all_fanout -flat -endpoints_only [get_ports cold_reset_n]]', 'Changed cold-reset query')
    required_reset = {f'{prefix}_reg[{i}]/CLR' for prefix in ('sr', 'dr') for i in range(3)}
    required_reset.update(f'{p}_reg/CLR' for p in ('link_s1', 'link_s2', 'ready_s1', 'ready_s2'))
    require(set(reset.splitlines()) == required_reset and len(reset.splitlines()) == 10, 'Wrong cold-reset fanout')
    outer = [l for l in lines if l != reset_line]
    launch = next((l for l in outer if '-from [get_ports ' in l), '')
    capture_line = next((l for l in outer if '-from [list [get_cells ' in l), '')
    require(set(objects(launch, 'from', 'ports')) == {r['port'] for r in manifest if r['direction'] == 'IN'},
            'Incorrect outer launch ports')
    require(set(objects(launch, 'to', 'pins')) == {r['anchor'] + '/D' for r in manifest if r['direction'] == 'IN'},
            'Cut reaches core-facing input paths')
    require(set(objects(capture_line, 'from', 'cells')) == {r['anchor'] for r in manifest if r['direction'] == 'OUT'},
            'Cut reaches core-facing output paths')
    require(set(objects(capture_line, 'to', 'ports')) == {r['port'] for r in manifest if r['direction'] == 'OUT'},
            'Incorrect outer capture ports')


def negative_checks(manifest_text, connectivity, ports, effective, reset):
    manifest = boundary(manifest_text, connectivity, ports)
    exceptions(effective, manifest, reset)
    mutations = [
        (boundary, ('\n'.join(manifest_text.splitlines()[:-1]), connectivity, ports)),
        (boundary, (manifest_text.replace('\treader_clk\t', '\tdma_clk\t', 1), connectivity, ports)),
        (boundary, (manifest_text, '\n'.join(connectivity.splitlines()[:-1]), ports)),
        (exceptions, (effective.replace('ntrpu_boundary_ff_00000/D]', 'ntrpu_boundary_ff_00000/Q]', 1), manifest, reset)),
        (exceptions, (effective + '\nset_false_path -from [all_inputs]\n', manifest, reset)),
        (exceptions, (effective + '\nset_clock_groups -asynchronous -group reader_clk -group dma_clk\n', manifest, reset)),
        (exceptions, (effective, manifest, reset + 'unexpected/CLR\n')),
    ]
    for check, args in mutations:
        try:
            check(*args)
        except ValueError:
            continue
        raise ValueError('Corrupted boundary evidence was accepted')
    return len(mutations)


def polarity(text, reset):
    records = rows(text)
    require(len(records) == 10 and {r['endpoint'] for r in records} == set(reset.splitlines()), 'Reset polarity coverage')
    for r in records:
        require(r['ref'] == 'FDCE' and r['async_reg'] == '1' and r['init'] == "1'b0", 'Changed reset register')
        require((r['representation'], r['clr_inverted']) in (('LUT1', '0'), ('CLR_ATTRIBUTE', '1')),
                'Reset is not active-low')
        require(r['clock'] == ('dma_clk' if r['endpoint'].startswith('dr_reg') else 'reader_clk'), 'Wrong reset clock')
    return Counter(r['representation'] for r in records)


def rejected_reset_stage(stage):
    """Validate provenance only, not success, for the specific old guard rejection."""
    out = stage / 'output'
    for n, sha in json.loads((out / 'FILES.json').read_text()).items():
        require(digest(out / n) == sha, 'Changed rejected-stage evidence')
    result = json.loads((out / 'result.json').read_text())
    require(result['status'] == 'FAILED' and result['returncode'] == 1 and not result['timed_out']
            and result['sources'] == result['sources_after'], 'Unrecognized reset-review failure')
    for n, sha in result['sources'].items():
        require(digest(stage / 'repo' / n) == sha, 'Changed rejected-stage source')
    log = (out / 'console.log').read_text()
    require('route_design completed successfully' in log and 'Unexpected raw reset loads' in log
            and 'reset_scope $out 0' in log, 'Not the documented post-route reset guard failure')
    require((out / 'reports/routed.dcp').is_file(), 'No completed routed checkpoint')
    return result


def reset_reproducer(stage):
    verify_stage(stage)
    r = stage / 'output/reports'
    cells = rows((r / 'optimized_reset/cells.tsv').read_text())
    require(cells and not any('RAM' in c['ref'] for c in cells), 'Reset reproduction contains RAM')
    pol = rows((r / 'reset_polarity.tsv').read_text())
    expected = {f'{p}_reg[{i}]/CLR' for p in ('sr', 'dr') for i in range(3)} | {'link_s1_reg/CLR', 'link_s2_reg/CLR'}
    require(len(pol) == 8 and {p['pin'] for p in pol} == expected and
            all(p['ref'] == 'FDCE' and p['clr_inverted'] == "1'b1" and p['async_reg'] == '1' for p in pol),
            'Wrong reproduction reset structure')
    cdc = inventory((r / 'optimized_reset/cdc.rpt').read_text())
    require({k: v['count'] for k, v in cdc.items()} == {'CDC-3': 1, 'CDC-26': 3}, 'Reset diagnostic not reproduced')
    return dict(ram_cells=0, inverted_clr=8, cdc26=3)


def review(prepare, route, audit, ports, placement=None, baseline=None, reset_probe=None):
    stages = [verify_stage(p) for p in (prepare, route, audit)]
    baseline_seconds = None
    physical = route
    if baseline is not None:
        br = rejected_reset_stage(baseline)
        baseline_seconds = br['seconds']
        require(digest(baseline / 'output/reports/routed.dcp') == digest(route / 'repo/evidence/routed.dcp'),
                'Optimized a different routed checkpoint')
        polarity((route / 'output/reports/before_reset/reset_polarity.tsv').read_text(),
                 (route / 'output/reports/before_reset/cold_reset_endpoints.txt').read_text())
        physical = baseline
    require(digest(prepare / 'repo/evidence/controls_constrained.dcp') ==
            '6c5ed2814728e8aa25273cd7862f2b947dce1c9ade8489ddef48a50aa4df324a', 'Wrong core donor')
    placement_seconds = None
    if (physical / 'repo/evidence/placed.dcp').exists():
        require(placement is not None, 'Resumed route requires placement provenance')
        po = placement / 'output'
        for n, sha in json.loads((po / 'FILES.json').read_text()).items():
            require(digest(po / n) == sha, 'Changed placement evidence')
        pr = json.loads((po / 'result.json').read_text())
        require(pr['status'] == 'FAILED' and pr['timed_out'] and pr['sources'] == pr['sources_after'],
                'Unexpected interrupted placement stage')
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
    return dict(status='BOUNDARY_ROUTED_TIMING_' + ('PASS' if closed else 'OPEN'),
                evidence_verified=True, harness_timing_closed=closed, protocol_adapter_validated=False,
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
                baseline_seconds=baseline_seconds, baseline_original_review_rejected=baseline is not None,
                post_route_optimization='Explore' if baseline is not None else None,
                reset_representation=dict(reset_representation),
                route_directive='RuntimeOptimized' if placement_seconds is not None else 'Default',
                min_capture_slack_ns=data_slack, min_control_slack_ns=control_slack,
                min_link_slack_ns=link_slack, min_gray_skew_slack_ns=skew_slack,
                stage_seconds=[s['seconds'] for s in stages], routed_dcp_sha256=digest(rr / 'routed.dcp'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('prepare', 'route', 'audit', 'ports'):
        ap.add_argument(name, type=Path)
    ap.add_argument('--placement', type=Path, help='Interrupted stage that supplied placed.dcp')
    ap.add_argument('--baseline', type=Path, help='Routed baseline rejected by the documented old reset guard')
    ap.add_argument('--reset-probe', type=Path, help='No-RAM reproducer of inverted-CLR CDC diagnostics')
    a = ap.parse_args()
    print(json.dumps(review(a.prepare, a.route, a.audit, a.ports, a.placement, a.baseline, a.reset_probe), indent=2))
