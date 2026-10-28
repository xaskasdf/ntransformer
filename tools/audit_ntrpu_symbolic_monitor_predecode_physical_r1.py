"""Review OOC implementation evidence for diagnostic replay, never timing signoff.

Callers must independently bind reports, EDIF audits, constraints and scripts to
their immutable source snapshots and certified terminal archives.
"""
import collections
import hashlib
import re
from decimal import Decimal
from audit_ntrpu_timing_subset_r1 import compare_readback


class PhysicalReviewError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PhysicalReviewError(message)


def rows(text, width=None):
    result = [line.split('\t') for line in text.splitlines()[1:]]
    require(width is None or all(len(row) == width for row in result), 'Malformed TSV')
    require(len({tuple(row) for row in result}) == len(result), 'Duplicate TSV rows')
    return result


def table(text):
    data = rows(text, 2)
    require(len({row[0] for row in data}) == len(data), 'Duplicate TSV keys')
    return dict(data)


def timing_summary(text):
    pattern = r'^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s*$'
    matches = re.findall(pattern, text.split('Design Timing Summary')[-1], re.M)
    require(len(matches) == 1, 'Missing or ambiguous timing summary')
    return list(matches[0])


def temporal(checkpoint):
    return {name: [line.strip() for line in text.splitlines()
                  if re.match(r'^\s*(create_clock|set_input_delay|set_output_delay)\b', line)]
            for name, text in checkpoint.items()}


def warning_review(console, ports, readback=False):
    lines = [s for s in console.splitlines() if s.startswith(('WARNING:', 'CRITICAL WARNING:', 'ERROR:'))]
    counts = collections.Counter()
    boundary_ports = []
    drc = 'WARNING: [DRC 23-814] Not all possible (connectivity based) DRCs may have been run because this design is seen as Out of Context.'
    clock = 'WARNING: [Route 35-197] Clock port "clk" does not have an associated HD.CLK_SRC. Without this constraint, timing analysis may not be accurate and upstream checks cannot be done to ensure correct clock placement.'
    port_pattern = r'WARNING: \[Route 35-198\] Port "([^"]+)" does not have an associated HD.PARTPIN_LOCS, which will prevent the partial routing of the signal "([^"]+)"\. Without this partial route, timing analysis to/from this port will not be accurate, and no routing information for this port can be exported\.'
    for line in lines:
        if line == drc:
            counts['DRC 23-814'] += 1
        elif line == clock and not readback:
            counts['Route 35-197'] += 1
        else:
            match = re.fullmatch(port_pattern, line)
            require(not readback and match is not None, 'Unreviewed warning/error: ' + line)
            port, net = match.groups()
            require(port == net and port in ports and port != 'clk', 'Unknown boundary warning endpoint')
            boundary_ports.append(port)
            counts['Route 35-198'] += 1
    expected = {'DRC 23-814': 2} if readback else {'DRC 23-814': 3, 'Route 35-198': 100, 'Route 35-197': 1}
    require(dict(counts) == expected, 'Warning inventory differs from reviewed OOC flow')
    require(len(set(boundary_ports)) == len(boundary_ports), 'Repeated boundary warning endpoint')
    return dict(counts= dict(counts), lines=lines, emitted_boundary_ports=boundary_ports,
                cap_does_not_review_unemitted_cases=True)


def review(e, ae, dag, xdc, axdc, saved, asaved, script, top, marker):
    r = ae['reports']
    require(e['reports']['implementation_status.txt'].splitlines() ==
            ['return_code=0', 'phase=completed', 'message=completed'], 'Implementation incomplete')
    for name in ['report_errors.txt', 'final_report_errors.txt']:
        require(not e['reports'][name].strip(), 'Report command errors')
    require(e['console'].splitlines().count(marker + '_PLACEMENT_DIAGNOSTIC_EXPORTED') == 1, 'Missing placement marker')
    require(ae['console'].splitlines().count(marker + '_ROUTE_AUDIT_EXPORTED violations=0') == 1, 'Missing readback marker')
    require(re.findall(r'^\s*(route_design[^\n]*)', script, re.M) == ['route_design -directive Quick'], 'Route command changed')
    require(re.findall(r'^\s*(place_design[^\n]*)', script, re.M) == ['place_design'], 'Placement command changed')
    require(not re.search(r'^\s*(synth_design|opt_design)\b', script, re.M), 'Unexpected resynthesis/optimization')
    expected = ['create_clock -period 4.000 -name pcie_user_clk [get_ports clk]',
                'set_input_delay -clock pcie_user_clk 0.000 [get_ports -filter {DIRECTION == IN && NAME != clk}]',
                'set_output_delay -clock pcie_user_clk 0.000 [all_outputs]',
                'set_property CLOCK_BUFFER_TYPE BUFGCE [get_nets clk]', 'current_instance -quiet']
    require(xdc and axdc and saved and asaved, 'Missing saved constraints')
    for name, text in (xdc | axdc).items():
        commands = [s.strip() for s in text.splitlines() if s.strip() and not s.lstrip().startswith('#')
                    and not re.match(r'^\s*set_property (?:SRC_FILE_INFO|src_info) ', s)]
        require(commands == expected, 'Changed exported constraints: ' + name)
    for checkpoint in list(saved.values()) + list(asaved.values()):
        require(bool(checkpoint), 'DCP lacks saved constraints')
        for text in checkpoint.values():
            require(not re.search(r'\b(set_false_path|set_multicycle_path|set_max_delay|set_min_delay|set_clock_groups|set_disable_timing|HD.CLK_SRC)\b', text), 'Hidden timing exception or clock source')
        require(temporal(checkpoint) == temporal(saved['input.dcp']), 'DCP temporal constraints changed')
    require(saved['route_attempt.dcp'] == asaved['implemented_input.dcp'], 'DCP saved constraints changed on readback')
    require('No valid timing exceptions found.' in r['exceptions.rpt'], 'Active exceptions')
    require('No ignored timing exceptions found.' in r['ignored_exceptions.rpt'], 'Ignored exceptions')
    require(r['reference_clock.rpt'] == r['implemented_clock.rpt'], 'Clock properties changed')
    require(re.search(r'^PERIOD\s+double\s+true\s+4\.000$', r['implemented_clock.rpt'], re.M), 'Clock period changed')
    port_rows = rows(r['implemented_ports.tsv'], 3)
    require(port_rows == rows(r['reference_ports.tsv'], 3) == rows(e['reports']['final_ports.tsv'], 3), 'Port inventory changed')
    ports = {row[0]: row[1] for row in port_rows}
    require(len(ports) == len(port_rows) == 1831 and not any(row[2] for row in port_rows), 'OOC boundary changed')
    require(dict(collections.Counter(ports.values())) == dict(IN=1384, OUT=447), 'Port direction/width changed')
    require([row[:2] for row in port_rows] == rows(e['reports']['ports.tsv'], 2), 'Initial ports differ')
    primitives = table(r['reloaded_primitives.tsv'])
    require(primitives == table(r['implemented_primitives.tsv']) == table(e['reports']['primitives.tsv']), 'Primitive inventory mismatch')
    reference = table(r['reference_primitives.tsv'])
    changes = dict(added={k: v for k, v in primitives.items() if k not in reference},
                   removed={k: v for k, v in reference.items() if k not in primitives},
                   changed_type={k: [v, primitives[k]] for k, v in reference.items() if k in primitives and v != primitives[k]})
    require(changes == dag['primitive_changes'], 'EDIF primitive changes mismatch')
    require(not changes['removed'] and not changes['changed_type'], 'Original primitive removed/retyped')
    replica_origins = {}
    for name, kind in changes['added'].items():
        base = re.sub(r'_replica(?:_\d+)?$', '', name)
        require(kind == 'FDRE' and base != name and reference.get(base) == 'FDRE', 'Unreviewed added primitive: ' + name)
        replica_origins[name] = base
    # These names classify implementation changes only; no replica equivalence is inferred.
    counts = dict(collections.Counter(primitives.values()))
    require(counts == dag['primitive_counts'], 'EDIF primitive counts mismatch')
    registers = {n for n, kind in primitives.items() if kind == 'FDRE'}
    for name in ['implemented_register_names.txt', 'register_timing_endpoints.txt']:
        lines = r[name].splitlines()
        require(len(lines) == len(set(lines)) and set(lines) == registers, 'Register inventory mismatch')
    require(set(e['reports']['register_timing_endpoints.txt'].splitlines()) == registers, 'Original endpoint inventory mismatch')
    flags = table(r['route_flags.tsv'])
    require(flags == table(e['reports']['route_flags.tsv']) == dict(PLACED_FULLY='1', PARTIALLY_ROUTED='0', ROUTED_FULLY='1', ERRORS_IN_ROUTES='0'), 'Routing incomplete')
    nets = table(r['route_nets.tsv'])
    require(nets == table(e['reports']['route_nets.tsv']), 'Route inventory changed on readback')
    route_counts = dict(collections.Counter(nets.values()))
    require(set(route_counts) <= {'ROUTED', 'HIERPORT', 'INTRASITE'} and route_counts.get('ROUTED', 0) > 0, 'Incomplete/unknown net status')
    require(not rows(r['incomplete_nets.tsv']) and not rows(e['reports']['partial_routes.tsv']), 'Incomplete nets')
    for kind in ['PARTIAL', 'GAPS', 'UNPLACED', 'UNROUTED', 'CONFLICTS', 'ANTENNAS', 'NODRIVER', 'MULTI_DRIVER']:
        require(re.search(r'Total\s+: 0\s+Parents: 0', r['route_' + kind + '.rpt']), 'Route violations: ' + kind)
    require(not rows(r['drc_objects.tsv']), 'DRC objects reported')
    for data in [e, ae]:
        require(re.search(r'Checks found: 0\b', data['reports']['drc.rpt']), 'DRC checks reported')
    require(r['buffer_connectivity.tsv'] == e['reports']['buffer_connectivity.tsv'], 'Clock connectivity changed')
    connection_rows = rows(r['buffer_connectivity.tsv'], 7)
    connections = {row[0].rsplit('/', 1)[1]: row for row in connection_rows}
    require(len(connection_rows) == 3 and set(connections) == {'CE', 'I', 'O'}, 'Clock pin inventory changed')
    require(connections['I'][1:] == ['clk_BUFG', 'HIERPORT', 'clk', '', 'clk_BUFGCE_inst/I', ''], 'Clock boundary changed')
    require(connections['CE'][1:6] == ['VCC_1', 'ROUTED', '', 'VCC/P', 'clk_BUFGCE_inst/CE'], 'Clock enable changed')
    require(connections['CE'][6] and connections['O'][6], 'Clock/enable route absent')
    require(connections['O'][2] == 'ROUTED' and connections['O'][3:5] == ['', 'clk_BUFGCE_inst/O'], 'Clock driver changed')
    loads = connections['O'][5].split(';')
    require(len(loads) == len(set(loads)) and set(loads) == {n + '/C' for n in registers}, 'Clock does not drive every FDRE exactly once')
    require(dag['clock_buffer']['clock_loads'] == len(registers), 'EDIF clock load mismatch')
    locs = re.findall(r'^LOC\s+site\s+false\s+(BUFGCE_X\d+Y\d+)$', r['implemented_buffer.rpt'], re.M)
    require(len(locs) == 1, 'Missing/ambiguous placed BUFGCE location')
    summary = timing_summary(r['timing.rpt'])
    require(summary == timing_summary(e['reports']['timing.rpt']), 'Full timing summary changed on readback')
    subsets = {}
    for group in ['register_to_register', 'input_to_register', 'register_to_output', 'input_to_output']:
        sr, dr = group.startswith('register'), group.endswith('register')
        for kind in ['setup', 'hold']:
            key = group + '_' + kind
            subsets[key] = compare_readback(e['reports'][key + '.rpt'], r[key + '.rpt'], kind=kind, top=top,
                sources=registers if sr else {n for n, d in ports.items() if d == 'IN' and n != 'clk'},
                destinations=registers if dr else {n for n, d in ports.items() if d == 'OUT'},
                source_registers=sr, destination_registers=dr)
    for kind, index in [('setup', 0), ('hold', 4)]:
        worst = min(Decimal(v['worst_slack_ns']) for k, v in subsets.items() if k.endswith('_' + kind))
        require(abs(worst - Decimal(summary[index])) <= Decimal('0.002'), 'Subsets disagree with global slack')
    checks = {n: int(v) for n, v in re.findall(r'^\d+\. checking (\w+) \((\d+)\)', r['timing.rpt'], re.M)}
    require(len(checks) == 12 and not any(checks.values()), 'Unconstrained/invalid timing checks')
    methodology = dict(collections.Counter(re.findall(r'^([A-Z]+-\d+)#\d+ Warning', r['methodology.rpt'], re.M)))
    require(methodology == dict(collections.Counter(re.findall(r'^([A-Z]+-\d+)#\d+ Warning', e['reports']['methodology.rpt'], re.M))) == {'TIMING-15': 1000, 'TIMING-16': 1000, 'XDCH-2': 1830}, 'Methodology inventory changed')
    warnings = warning_review(e['console'], ports)
    readback_warnings = warning_review(ae['console'], ports, readback=True)
    resources = {}
    for name in ['CLB LUTs', 'CLB Registers', 'Block RAM Tile', 'URAM', 'DSPs', 'BUFGCE', 'CARRY8']:
        match = re.search(r'^\|\s*' + re.escape(name) + r'\s*\|\s*(\d+)\s*\|', r['utilization.rpt'], re.M)
        require(match is not None, 'Resource missing: ' + name)
        resources[name] = int(match[1])
        original = re.search(r'^\|\s*' + re.escape(name) + r'\s*\|\s*(\d+)\s*\|', e['reports']['utilization.rpt'], re.M)
        require(original is not None and int(original[1]) == resources[name], 'Resource changed on readback: ' + name)
    require(resources['CLB Registers'] == len(registers) and resources['BUFGCE'] == counts.get('BUFGCE') == 1, 'Register/clock resources disagree')
    require(resources['CARRY8'] == counts.get('CARRY8') and resources['CLB LUTs'] > 0, 'Logic resources disagree')
    require(all(resources[n] == 0 for n in ['Block RAM Tile', 'URAM', 'DSPs']), 'Unexpected memory/DSP resources')
    help_capture = {}
    for command in ['open_checkpoint', 'opt_design', 'place_design', 'route_design', 'report_route_status', 'report_timing_summary', 'report_timing', 'all_registers', 'report_high_fanout_nets', 'write_edif', 'help']:
        text = r['help_' + command + '.txt']
        emitted = re.findall(r'^NTRPU_HELP_CAPTURED ' + command + r' characters=(\d+)$', ae['console'], re.M)
        require(len(emitted) == 1 and int(emitted[0]) > 100 and len(text) > 100 and command in text, 'Missing installed command help')
        help_capture[command] = dict(bytes=len(text.encode()), sha256=hashlib.sha256(text.encode()).hexdigest())
    require('Vivado v2026.1' in r['VERSION.txt'], 'Unexpected tool version')
    return dict(resources=resources, primitive_changes=changes, replica_name_origins=replica_origins,
                replica_functional_equivalence_proven=False, route_flags=flags, route_hierarchical_counts=route_counts,
                clock_output_routed_loads=len(registers), buffer_loc=locs[0], path_subsets=subsets,
                timing_summary=summary, check_timing=checks, methodology_reported_checks=methodology,
                emitted_warning_counts=warnings['counts'], emitted_warnings=warnings['lines'],
                readback_warning_counts=readback_warnings['counts'], readback_warnings=readback_warnings['lines'],
                saved_temporal_constraints=temporal(saved['route_attempt.dcp']), help_capture=help_capture,
                ports=len(ports), inputs=1384, outputs=447, ports_without_partition_locations=len(ports),
                clock_input_hierport_without_physical_route=True, drc_reported_violations=0,
                drc_checks_incomplete_in_ooc=True, warning_caps_do_not_prove_unemitted_cases_reviewed=True,
                timing_qualified=False, physical_board_qualified=False, functional_replay_qualified=False)
