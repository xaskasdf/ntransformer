"""Check routed report identity and inventory changes; grant no qualification."""
from collections import Counter
from decimal import Decimal
import re
from audit_ntrpu_timing_subset_r1 import compare_readback
from audit_ntrpu_symbolic_monitor_predecode_physical_r2 import require, rows, table, timing_summary


def geometry(text, primitives):
    require(text.splitlines()[0] == 'cell\tprimitive\tloc\tbel', 'Geometry header changed')
    data = rows(text, 4)
    result = {row[0]: dict(primitive=row[1], loc=row[2], bel=row[3]) for row in data}
    require(len(result) == len(data), 'Duplicate geometry cell')
    expected = {n:k for n,k in primitives.items() if k not in ('VCC', 'GND')}
    require({n:v['primitive'] for n,v in result.items()} == expected, 'Geometry primitive inventory differs')
    return result


def measure(e, ae, top):
    original, reopened = e['reports'], ae['reports']
    require(original['implementation_status.txt'].splitlines() ==
            ['return_code=0', 'phase=completed', 'message=completed'], 'Implementation incomplete')
    for name in ('report_errors.txt', 'final_report_errors.txt'):
        require(not original[name].strip(), 'Report failure')
    primitives = table(reopened['implemented_primitives.tsv'])
    require(primitives == table(reopened['reloaded_primitives.tsv']) == table(original['primitives.tsv']),
            'Primitive readback mismatch')
    reference = table(reopened['reference_primitives.tsv'])
    registers = {n for n,k in primitives.items() if k == 'FDRE'}
    for text in (original['register_timing_endpoints.txt'], reopened['register_timing_endpoints.txt'],
                 reopened['implemented_register_names.txt']):
        names = text.splitlines()
        require(len(names) == len(set(names)) and set(names) == registers, 'Register inventory mismatch')
    port_rows = rows(reopened['implemented_ports.tsv'], 3)
    require(port_rows == rows(reopened['reference_ports.tsv'], 3) == rows(original['final_ports.tsv'], 3),
            'Port inventory changed')
    ports = {r[0]:r[1] for r in port_rows}
    require(len(ports) == len(port_rows) == 1831 and not any(r[2] for r in port_rows), 'OOC ports differ')
    require(Counter(ports.values()) == dict(IN=1384, OUT=447), 'Port widths differ')
    require([r[:2] for r in port_rows] == rows(original['ports.tsv'], 2), 'Initial ports differ')
    require(original['ed_registers.tsv'] == reopened['ed_registers.tsv'], 'E/D subset changed')
    ed_rows = rows(original['ed_registers.tsv'], 2)
    require(original['ed_registers.tsv'].splitlines()[0] == 'stage\tregister', 'E/D header changed')
    require(all(r[0] in ('E', 'D') for r in ed_rows), 'Unknown E/D stage')
    es = {n for stage,n in ed_rows if stage == 'E'}
    ds = {n for stage,n in ed_rows if stage == 'D'}
    require(es == {n for n in registers if n.startswith('e_stage/')}, 'E subset incomplete')
    require(ds == {n for n in registers if n.startswith('d_stage/symbolic_rows_reg')}, 'D subset incomplete')
    flags = table(original['route_flags.tsv'])
    require(flags == table(reopened['route_flags.tsv']) ==
            dict(PLACED_FULLY='1', PARTIALLY_ROUTED='0', ROUTED_FULLY='1', ERRORS_IN_ROUTES='0'), 'Routing incomplete')
    nets = table(original['route_nets.tsv'])
    require(nets == table(reopened['route_nets.tsv']), 'Net route states changed on readback')
    require(set(nets.values()) <= {'ROUTED', 'HIERPORT', 'INTRASITE'}, 'Unexpected route state')
    require(not rows(original['partial_routes.tsv']) and not rows(reopened['incomplete_nets.tsv']), 'Incomplete nets')
    summary = timing_summary(original['timing.rpt'])
    require(summary == timing_summary(reopened['timing.rpt']), 'Timing summary changed')
    groups = {}
    for group in ('register_to_register', 'input_to_register', 'register_to_output', 'input_to_output', 'ed'):
        sr, dr = group.startswith('register') or group == 'ed', group.endswith('register') or group == 'ed'
        sources = es if group == 'ed' else registers if sr else {n for n,d in ports.items() if d == 'IN' and n != 'clk'}
        destinations = ds if group == 'ed' else registers if dr else {n for n,d in ports.items() if d == 'OUT'}
        for kind in ('setup', 'hold'):
            name = group+'_'+kind
            groups[name] = compare_readback(original[name+'.rpt'], reopened[name+'.rpt'], kind=kind, top=top,
                sources=sources, destinations=destinations, source_registers=sr, destination_registers=dr)
    for kind, index in (('setup', 0), ('hold', 4)):
        worst = min(Decimal(v['worst_slack_ns']) for k,v in groups.items() if k.endswith('_'+kind) and not k.startswith('ed_'))
        require(abs(worst-Decimal(summary[index])) <= Decimal('0.002'), 'Global and subset slacks disagree')
    resources = {}
    for name in ('CLB LUTs', 'CLB Registers', 'CARRY8', 'BUFGCE', 'Block RAM Tile', 'URAM', 'DSPs'):
        section = {'CLB LUTs':'1. CLB Logic', 'CLB Registers':'1. CLB Logic', 'CARRY8':'1. CLB Logic',
                   'BUFGCE':'6. CLOCK', 'Block RAM Tile':'3. BLOCKRAM', 'URAM':'3. BLOCKRAM', 'DSPs':'4. ARITHMETIC'}[name]
        blocks = [re.findall(r'^'+re.escape(section)+r'\n-+\n(.*?)(?=^\d+(?:\.\d+)?\.? [^\n]+\n-+\n|\Z)',
                             data['utilization.rpt'], re.M | re.S) for data in (original, reopened)]
        require(all(len(block) == 1 for block in blocks), 'Missing/ambiguous primary resource section: '+section)
        values = [re.findall(r'^\|\s*'+re.escape(name)+r'\s*\|\s*(\d+)\s*\|', block[0], re.M) for block in blocks]
        require(len(values[0]) == 1 and values[0] == values[1], 'Resource mismatch: '+name)
        resources[name] = int(values[0][0])
    require(resources['CLB Registers'] == len(registers) and resources['BUFGCE'] == 1, 'Register/clock count mismatch')
    changes = dict(added={n:k for n,k in primitives.items() if n not in reference},
        removed={n:k for n,k in reference.items() if n not in primitives},
        changed_type={n:[k,primitives[n]] for n,k in reference.items() if n in primitives and primitives[n] != k})
    geom = dict(available=False)
    geometry_presence = ['input_geometry.tsv' in original, 'final_geometry.tsv' in original,
                         'reference_geometry.tsv' in reopened, 'implemented_geometry.tsv' in reopened]
    require(all(geometry_presence) or not any(geometry_presence), 'Partial geometry evidence')
    if all(geometry_presence):
        before = geometry(original['input_geometry.tsv'], reference)
        after = geometry(original['final_geometry.tsv'], primitives)
        require(before == geometry(reopened['reference_geometry.tsv'], reference), 'Seed geometry readback differs')
        require(after == geometry(reopened['implemented_geometry.tsv'], primitives), 'Final geometry readback differs')
        moved = {n:dict(before=before[n], after=v) for n,v in after.items() if n in before and before[n] != v}
        geom = dict(available=True, initial_cells=len(before), final_cells=len(after),
            changed_existing_cells=moved, added={n:v for n,v in after.items() if n not in before},
            removed={n:v for n,v in before.items() if n not in after},
            empty_initial_loc=sum(not v['loc'] for v in before.values()), empty_final_loc=sum(not v['loc'] for v in after.values()),
            unchanged=before == after)
    warnings = {}
    for name, data in (('implementation', e), ('readback', ae)):
        lines = [line for line in data['console'].splitlines() if line.startswith(('WARNING:', 'CRITICAL WARNING:', 'ERROR:'))]
        warnings[name] = dict(lines=lines, counts=dict(Counter(re.match(r'^(.*?\])', line)[1]
            if re.match(r'^(.*?\])', line) else line for line in lines)))
    return dict(timing_summary=summary, path_groups=groups, resources=resources, primitive_changes=changes,
        geometry=geom, warning_inventory=warnings, route_flags=flags, route_counts=dict(Counter(nets.values())),
        registers=len(registers), ports=len(ports), warnings_reviewed=False,
        structural_equivalence_proven=False, constraints_reviewed=False, timing_qualified=False,
        netlist_replay_allowed=False, physical_board_qualified=False)
