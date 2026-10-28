"""Verify isolated routed evidence, retaining timing failures as open results."""
import argparse
import csv
import io
import json
from pathlib import Path
import re

from check_ntrpu_hbm_dma_rev2_review import digest, require, verify_stage


def rows(text):
    return list(csv.DictReader(io.StringIO(text), delimiter='\t'))


def summary(text):
    match = re.search(r'^\s*WNS\(ns\)[^\n]*\n\s*[- ]+\n\s*([^\n]+)', text, re.M)
    require(match is not None, 'Missing complete timing summary')
    values = list(map(float, match[1].split()))
    require(len(values) == 12, 'Incomplete timing summary')
    result = {}
    for i, name in enumerate(('setup', 'hold', 'pulse_width')):
        worst, total, failed, endpoints = values[i * 4:i * 4 + 4]
        require(endpoints > 0 and endpoints.is_integer() and failed.is_integer()
                and 0 <= failed <= endpoints, 'Invalid endpoint accounting')
        result[name] = dict(slack_ns=worst, total_slack_ns=total,
                            failed=int(failed), endpoints=int(endpoints))
    return result


def constraints(effective, coverage, ignored, reset_text):
    require(not re.search(r'^\s*(?:set_clock_groups|set_multicycle_path|create_waiver|set_disable_timing)\b',
                          effective, re.M), 'Unexpected timing exclusion/waiver')
    require(len(re.findall(r'^set_false_path\b', effective, re.M)) == 1,
            'Only the enumerated cold-reset exception is allowed')
    false = next(line for line in effective.splitlines() if line.startswith('set_false_path'))
    require('cold_reset_n' in false and '-to' in false, 'Unscoped reset exception')
    expected = {f'{prefix}_reg[{bit}]/CLR' for prefix in ('sr', 'dr') for bit in range(3)}
    expected.update(f'{prefix}_reg/CLR' for prefix in ('ready_s1', 'ready_s2', 'link_s1', 'link_s2'))
    actual = reset_text.splitlines()
    require(len(actual) == 10 and set(actual) == expected, 'Unexpected cold-reset endpoints')
    require(len(re.findall(r'^set_max_delay\b', effective, re.M)) == 26, 'Missing/extra max-delay group')
    require(len(re.findall(r'^set_bus_skew\b', effective, re.M)) == 12, 'Missing/extra Gray constraint')
    require('No ignored timing exceptions found.' in ignored, 'Ignored timing exceptions')
    controls = re.findall(r'^\d+\s+Max Delay Datapath Only\s+max_dpo=(\d+)\s+-\s+(\d+) cells\s+(\d+) cells\s+(\d+)\s+([\d.]+)\s+33\.33\s*$', coverage, re.M)
    # Physical synthesis replicates each dma_fault source. The extra copy has
    # no path to the synchronizer, so its From coverage is 50%; all 45 real D
    # endpoints and their exact original drivers are independently verified.
    wanted = ([(5, 3, 3, 3, 100.)] * 6 + [(4, 3, 3, 3, 100.)] * 6
              + [(4, 2, 1, 1, 50.)] * 2 + [(4, 2, 2, 2, 100.)] * 2
              + [(4, 1, 1, 1, 100.)] + [(5, 1, 1, 1, 100.)] * 2)
    require(sorted(tuple(map(float, x)) for x in controls) == sorted(wanted), 'Control coverage changed')
    require(re.search(r'^\d+\s+Max Delay Datapath Only\s+max_dpo=4\s+-\s+1 ports\s+1 pins\s+1\s+100\.00\s+100\.00\s*$', coverage, re.M),
            'Missing link-status bound')
    data = re.findall(r'^\d+\s+Max Delay Datapath Only\s+max_dpo=(\d+)\s+-\s+(\d+) cells\s+(\d+) pins\s+(\d+)\s+100\.00\s+100\.00\s*$', coverage, re.M)
    require(sorted(tuple(map(int, x)) for x in data) == sorted([(4, 143, 143, 143), (4, 514, 514, 514), (5, 2, 2, 2)] * 2),
            'Capture constraints lost coverage')


def capture(bindings, paths):
    expected = {}
    endpoints = set()
    for lane in range(2):
        for fifo, width in [('commands', 143), ('payloads', 514), ('completions', 2)]:
            clocks = ('dma_clk', 'reader_clk') if fifo == 'completions' else ('reader_clk', 'dma_clk')
            expected[f'l{lane}_{fifo}'] = (width, width, *clocks)
            endpoints.update(f'lanes[{lane}].bridge/{fifo}/output_data_reg[{i}]/D' for i in range(width))
    banks = rows(bindings)
    require(len(banks) == 6 and {r['fifo']: (int(r['ram_cells']), int(r['bits']), r['source_clock'], r['dest_clock'])
                               for r in banks} == expected, 'Capture bank binding changed')
    records = rows(paths)
    require(len(records) == 1318 and {r['endpoint'] for r in records} == endpoints, 'Incomplete capture endpoints')
    for r in records:
        lane, fifo = r['fifo'].split('_', 1)
        prefix = f'lanes[{lane[1:]}].bridge/{fifo}/'
        require(r['fifo'] in expected and r['startpoint'].startswith(prefix + 'memory_reg_')
                and r['startpoint'].endswith('/CLK') and r['endpoint'].startswith(prefix), 'Cross-bank capture path')
    return min(float(r['slack']) for r in records)


def controls(text, reference):
    actual, expected = rows(text), rows(reference)
    keys = ('group', 'source', 'first', 'second', 'source_clock', 'dest_clock', 'gray')
    require(len(actual) == len(expected) == 45, 'Missing control path')
    require({tuple(r[k] for k in keys) for r in actual} == {tuple(r[k] for k in keys) for r in expected},
            'Control binding changed')
    return min(float(r['slack']) for r in actual)


def interface(text, ports):
    actual = rows(text)
    expected = {}
    for r in rows(ports):
        name = r['name']
        if name in ('clk', 'dma_clk', 'cold_reset_n'):
            continue
        base = name.split('[')[0]
        direction = r['direction']
        dma = base in (('dma_link_up', 'dma_run', 'dma_error', 'dma_done', 'dsc_ready', 'axis_ready')
                       if direction == 'IN' else ('dma_rst_n', 'dma_fault', 'dsc_load', 'axis_valid', 'axis_last',
                                                    'dsc_src', 'dsc_dst', 'axis_keep', 'dsc_len', 'dsc_ctl', 'axis_data'))
        expected[name] = (direction, 'dma_clk' if dma else 'reader_clk', .1 if direction == 'IN' else -.1, 1.)
    require(len(actual) == len(expected) == 11375, 'Incomplete interface inventory')
    require({r['port']: (r['direction'], r['clock'], float(r['min_ns']), float(r['max_ns']))
             for r in actual} == expected, 'Interface budget/domain mismatch')


def review(route, audit, placement=None):
    route_result, audit_result = verify_stage(route), verify_stage(audit)
    require(digest(route / 'repo/evidence/controls_constrained.dcp') ==
            '6c5ed2814728e8aa25273cd7862f2b947dce1c9ade8489ddef48a50aa4df324a',
            'Wrong registered-FIFO revision-2 donor')
    directive = 'Explore'
    if (route / 'repo/evidence/placed.dcp').exists():
        require(placement is not None, 'Resumed routing needs its placement provenance')
        po = placement / 'output'
        for name, sha in json.loads((po / 'FILES.json').read_text()).items():
            require(digest(po / name) == sha, 'Placement transfer changed: ' + name)
        pr = json.loads((po / 'result.json').read_text())
        require(pr['sources'] == pr['sources_after'], 'Placement sources changed')
        for name, sha in pr['sources'].items():
            require(digest(placement / 'repo' / name) == sha, 'Frozen placement source changed')
        require(pr['timed_out'] and pr['status'] == 'FAILED'
                and 'phys_opt_design completed successfully' in (po / 'console.log').read_text(),
                'Unexpected placement provenance')
        require(digest(po / 'reports/placed.dcp') == digest(route / 'repo/evidence/placed.dcp'),
                'Resumed a different placement')
        require(digest(placement / 'repo/evidence/controls_constrained.dcp') ==
                digest(route / 'repo/evidence/controls_constrained.dcp'), 'Placement donor changed')
        require('Quick: diagnostic legal routing;' in (route / 'output/reports/ROUTE_DIRECTIVE.txt').read_text(),
                'Missing diagnostic routing label')
        directive = 'Quick'
    require(digest(route / 'output/reports/routed.dcp') == digest(audit / 'repo/evidence/routed.dcp'),
            'Audited a different checkpoint')
    rr, ar = route / 'output/reports', audit / 'output/reports'
    read = lambda name: (ar / name).read_text()
    require('NTRPU_BRIDGE_REV2_PHYSICAL_EXPORTED' in (route / 'output/console.log').read_text(), 'Missing route completion')
    require('NTRPU_BRIDGE_REV2_PHYSICAL_AUDIT_EXPORTED' in (audit / 'output/console.log').read_text(), 'Missing audit completion')
    constraint_args = (read('effective_constraints.xdc'), read('exceptions.rpt'), read('ignored_exceptions.rpt'),
                       (rr / 'cold_reset_endpoints.txt').read_text())
    constraints(*constraint_args)
    require(read('cold_reset_endpoints.txt') == (rr / 'cold_reset_endpoints.txt').read_text(),
            'Cold-reset fanout changed during implementation')
    data_args = (read('data_bindings.tsv'), read('data_paths.tsv'))
    data_slack = capture(*data_args)
    control_slack = controls(read('control_paths.tsv'), (route / 'repo/evidence/bindings.tsv').read_text())
    interface((rr / 'interface.tsv').read_text(), (audit / 'repo/evidence/ports.tsv').read_text())
    clocks = rows(read('clock_buffers.tsv'))
    require(len(clocks) == 2 and {r['clock'] for r in clocks} == {'reader_clk', 'dma_clk'}, 'Wrong clock inventory')
    for r in clocks:
        require(r['ref'] == 'BUFGCE' and r['loc'].startswith('BUFGCE_X') and r['ce_driver'] == 'VCC'
                and r['route_status'] == 'ROUTED' and int(r['loads']) > 0, 'Clock is not physically buffered/routed')
        require(float(r['period']) == (5 if r['clock'] == 'reader_clk' else 4), 'Wrong clock period')
    flags = dict(line.split('\t') for line in read('design.tsv').splitlines())
    require(flags['PART'] == 'xcvu47p-fsvh2892-2-e' and flags['async_registers'] == '98'
            and flags['blackboxes'] == flags['latches'] == '0', 'Changed design profile')
    # These design properties return empty in this Vivado build. Do not turn
    # missing properties into Boolean evidence; use the route/DRC reports.
    unavailable_flags = [n for n in ('PLACED_FULLY', 'PARTIALLY_ROUTED', 'ROUTED_FULLY', 'ERRORS_IN_ROUTES') if flags[n] == '']
    routing = {k: int(v) for k, v in re.findall(r'# of (.+?)\.+\s*:\s*(\d+)', read('route_status.rpt'))}
    require(routing['fully routed nets'] == routing['routable nets'] > 0 and routing['nets with routing errors'] == 0,
            'Incomplete routing or errors')
    coverage = {k: int(v) for k, v in re.findall(r'checking ([a-z_]+) \((\d+)\)', read('check_timing.rpt'))}
    for k in ('no_clock', 'unconstrained_internal_endpoints', 'no_output_delay', 'multiple_clock',
              'loops', 'partial_output_delay', 'latch_loops'):
        require(coverage[k] == 0, 'Timing coverage gap: ' + k)
    require(coverage['no_input_delay'] <= 1, 'Unbudgeted input besides asynchronous cold reset')
    require(coverage['partial_input_delay'] == 1 and
            re.search(r'10\. checking partial_input_delay \(1\)[\s\S]+?\ndma_link_up\s+11\.', read('check_timing.rpt')),
            'Unexpected partial input-delay inventory')
    effective = read('effective_constraints.xdc')
    for mode, value in [('min', '0.100'), ('max', '1.000')]:
        require(f'set_input_delay -clock [get_clocks dma_clk] -{mode} {value} [get_ports dma_link_up]' in effective,
                'Missing actual DMA-link input budget')
    skew = read('bus_skew.rpt')
    values = [tuple(map(float, x)) for x in re.findall(r'^\s+(?:Slow|Fast)\s+([.0-9]+)\s+([.0-9]+)\s+([-0-9.]+)\s*$', skew, re.M)]
    require(len(values) == 12 and all(req == 4 and actual >= 0 and abs(req - actual - slack) < .002
                                    for req, actual, slack in values), 'Incomplete/wrong skew summary')
    require(len(re.findall(r'^Endpoints: 3$', skew, re.M)) == 12, 'Wrong Gray bus widths')
    timing = summary(read('timing.rpt'))
    require(timing == summary((rr / 'timing.rpt').read_text()), 'Timing changed on checkpoint reopen')
    subsets = rows(read('timing_subsets.tsv'))
    require(len(subsets) == 8 and len({(r['subset'], r['kind']) for r in subsets}) == 8, 'Missing timing subsets')
    cdc = {rule: {'severity': severity, 'count': int(count)} for rule, severity, count in
           re.findall(r'^(CDC-\d+)\s+(Critical|Warning|Info)\s+(\d+)\s', read('cdc.rpt'), re.M)}
    require(cdc and not any(v['severity'] == 'Critical' for v in cdc.values()), 'Critical/unparsed CDC')
    require({k: v['count'] for k, v in cdc.items()} == {'CDC-3': 5, 'CDC-6': 14, 'CDC-15': 1318, 'CDC-26': 5},
            'Changed CDC inventory needs a new review')
    drc_text = read('drc.rpt')
    drc = re.findall(r'^\|\s*([A-Z]+-\d+)\s*\|\s*(Critical Warning|Warning|Error)\s*\|[^|]+\|\s*(\d+)\s*\|', drc_text, re.M)
    require(sum(int(n) for _, _, n in drc) == int(re.search(r'Checks found:\s+(\d+)', drc_text)[1]), 'Unparsed DRC')
    require(not any(level in ('Error', 'Critical Warning') for _, level, _ in drc), 'Severe physical DRC')
    # Reject omissions, misattributed data, blanket exclusions and incorrect clocks.
    mutations = []
    changed = list(data_args);changed[1] = '\n'.join(changed[1].splitlines()[:-1]) + '\n';mutations.append((capture, changed))
    changed = list(data_args);changed[1] = changed[1].replace('/commands/memory_', '/payloads/memory_', 1);mutations.append((capture, changed))
    changed = list(data_args);changed[0] = changed[0].replace('reader_clk\tdma_clk', 'dma_clk\treader_clk', 1);mutations.append((capture, changed))
    changed = list(constraint_args);changed[0] += '\nset_clock_groups -asynchronous -group reader_clk -group dma_clk\n';mutations.append((constraints, changed))
    changed = list(constraint_args);changed[0] += '\nset_false_path -from [all_inputs]\n';mutations.append((constraints, changed))
    changed = list(constraint_args);changed[3] += 'unexpected/CLR\n';mutations.append((constraints, changed))
    for check, arguments in mutations:
        try:
            check(*arguments)
        except ValueError:
            continue
        raise ValueError('A corrupted physical report passed')
    skew_slack = min(v[2] for v in values)
    closed = (coverage['partial_input_delay'] == 0 and data_slack >= 0 and control_slack >= 0 and skew_slack >= 0
              and all(v['slack_ns'] >= 0 and v['failed'] == 0 and v['total_slack_ns'] == 0 for v in timing.values()))
    return dict(status='REV2_ROUTED_TIMING_' + ('PASS_PROVISIONAL_IO' if closed else 'OPEN'),
                evidence_verified=True, isolated_timing_closed=closed, external_io_qualified=False, board_qualified=False,
                native_hbm_xdma_integrated=False, post_route_simulated=False, timing=timing, timing_subsets=subsets,
                capture_bits=1318, min_capture_slack_ns=data_slack, control_bits=45, min_control_slack_ns=control_slack,
                gray_buses=12, min_gray_skew_slack_ns=skew_slack, cdc=cdc, drc=drc, timing_coverage=coverage,
                routing=routing, clock_buffers=clocks, negative_controls=len(mutations),
                route_directive=directive,
                unavailable_design_properties=unavailable_flags, timing_coverage_complete=False,
                unresolved_findings=['partial_input_delay on dma_link_up despite both persisted budgets',
                                     'five CDC-26 rows on cold reset/link input require investigation',
                                     'OOC interface ports have no physical partition anchors'],
                route_seconds=route_result['seconds'], audit_seconds=audit_result['seconds'],
                routed_dcp_sha256=digest(route / 'output/reports/routed.dcp'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('route', type=Path)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--placement', type=Path, help='Archived timed-out stage that produced the resumed placement')
    args = parser.parse_args()
    print(json.dumps(review(args.route, args.audit, args.placement), indent=2))
