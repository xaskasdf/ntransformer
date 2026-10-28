"""Prove the one-net routing scope and preserve all inherited board checks."""
import math
import re
import review_hbm_bready_physical as inherited
from review_hbm_bready_placement import indexed, binding
from review_hbm_bready_balanced_placement import PLAN as PLACEMENT
from review_hbm_capture_physical import require, sha, table

DRIVER = 'weight_loader/active_reg[25]/Q'
SINK = 'weight_loader/address_reg[25]/D'
NET = 'weight_loader/active_reg_n_0_[25]'
PLAN = [dict(driver_pin=DRIVER, sink_pin=SINK, net=NET, method='delay')]
COMMAND = 'route_design -unroute -nets; route_design -nets -delay; route_design -finalize'


def route_proof(out, plan_path, mode):
    require(mode in ('eco', 'reopen'), 'Unknown routing mode')
    require(table(plan_path) == PLAN, 'Route plan changed')
    before = indexed(out / 'cells_before.tsv')
    after = indexed(out / 'cells_after.tsv')
    require(len(before) == 120241 and before == after, 'Primitive/function/placement changed')
    for row in PLACEMENT:
        cell = after[row['cell']]
        require(cell['primitive'] == row['primitive'] and cell['loc'] == row['after_loc'] and
                cell['bel'].split('.')[-1] == row['after_bel'] and
                cell['fixed_loc'] == cell['fixed_bel'] == '1', 'BREADY placement not preserved')
    for name, count in [('cone.tsv', 17), ('connections.tsv', 1699),
                        ('shared_drivers.tsv', 2), ('shared_endpoints.tsv', 2)]:
        rows = table(out / 'placement_before' / name)
        require(len(rows) == count and rows == table(out / 'placement_after' / name), 'BREADY cone changed')
    clock = table(out / 'clock_pins_before.tsv')
    require(len(clock) == 8321 and len({r['pin'] for r in clock}) == 8321 and
            {r['net'] for r in clock} == {'clock_generator/inst/clk_out2'} and
            clock == table(out / 'clock_pins_after.tsv'), 'Clock connectivity changed')
    pips = table(out / 'clock_pips_before.tsv')
    require(len(pips) == 2705 and len({r['pip'] for r in pips}) == 2705 and
            pips == table(out / 'clock_pips_after.tsv'), 'Clock route changed')
    require(not (out / 'clock_repair.tsv').exists(), 'Unexpected clock repair')
    routes_before = indexed(out / 'net_routes_before.tsv', 'net')
    names = indexed(out / 'net_routes_before.tsv.names.tsv', 'net')
    if mode == 'eco':
        routes_after = indexed(out / 'net_routes_after.tsv', 'net')
        names_after = indexed(out / 'net_routes_after.tsv.names.tsv', 'net')
    else:
        require(not (out / 'net_routes_after.tsv').exists() and not (out / 'net_routes_after.tsv.names.tsv').exists(),
                'Read-only reopen must provide one independent route snapshot')
        routes_after, names_after = routes_before, names
    require(len(routes_before) >= 100262 and set(routes_before) == set(routes_after), 'Canonical net inventory changed')
    require(len(names) >= 140842 and names == names_after, 'Logical net inventory changed')
    require(set(routes_before) <= set(names), 'Canonical routes outside logical inventory')
    for inventory in (routes_before, routes_after):
        require(all(re.fullmatch(r'[0-9a-f]{64}', r['route_sha256']) and r['route_bytes'].isdigit()
                    for r in inventory.values()), 'Malformed route digest')
    require(NET in routes_before, 'Missing target net')
    changed = {n for n in routes_before if routes_before[n] != routes_after[n]}
    require(changed <= ({NET} if mode == 'eco' else set()), 'Collateral routing change')
    for inventory in (routes_before, routes_after):
        require(inventory[NET]['status'] == 'ROUTED' and inventory[NET]['fixed'] == '0' and
                int(inventory[NET]['route_bytes']) > 0, 'Target net incomplete or fixed')
    pins = table(out / 'target_before/pins.tsv')
    require(len(pins) == 3 and len({r['pin'] for r in pins}) == 3 and
            {r['net'] for r in pins} == {NET} and
            {r['pin'] for r in pins if r['direction'] == 'OUT'} == {DRIVER} and
            len([r for r in pins if r['direction'] == 'IN']) == 2 and
            SINK in {r['pin'] for r in pins if r['direction'] == 'IN'}, 'Target net connectivity scope')
    require(pins == table(out / 'target_after/pins.tsv'), 'Target connectivity changed')
    expected = [dict(net=NET, command=COMMAND)] if mode == 'eco' else []
    require(table(out / 'route_actions.tsv') == expected, 'Routing action record changed')
    timing = {}
    for phase in ('before', 'after'):
        rows = table(out / f'target_{phase}/slow_max.tsv')
        require(len(rows) == 1, 'Target timing coverage')
        row = rows[0]
        require(row['STARTPOINT_PIN'] == DRIVER[:-1] + 'C' and row['ENDPOINT_PIN'] == SINK and
                row['STARTPOINT_CLOCK'] == 'axi_aclk_out' and row['ENDPOINT_CLOCK'] == 'clk_out2_fabric_clocks' and
                row['DELAY_TYPE'] == 'max' and abs(float(row['REQUIREMENT']) - 2.222) < .002,
                'Target timing binding changed')
        require(all(math.isfinite(float(row[k])) for k in ('SLACK', 'DATAPATH_DELAY')), 'Nonfinite timing')
        require('false' not in row['EXCEPTION'].lower() and 'multicycle' not in row['EXCEPTION'].lower(), 'Target timing waiver')
        timing[phase] = row
    if mode == 'reopen': require(timing['before'] == timing['after'], 'Reopen timing changed')
    return dict(primitive_cells=len(before), moved_cells=[], net_inventory=len(routes_before), logical_nets=len(names),
                route_snapshots=2 if mode == 'eco' else 1,
                changed_nets=sorted(changed), target_pins=pins, target_timing=timing,
                bready_placement_preserved=True, cone_connections_unchanged=True,
                clock_connectivity_unchanged=True, clock_route_unchanged=True,
                clock_pins=len(clock), clock_pips=len(pips), plan=PLAN)


def review(root, reference, donor):
    native, record = binding(root, donor)
    require(record['routing_only'] is True and record['clock_repair'] is False, 'Routing stage binding')
    mode = record['routing_mode']
    result = inherited.review(root, reference, donor, 'reopen')
    result.update(status='HBM_LOADER_SKEW_ROUTE_REVIEWED', routing_mode=mode,
                  routing=route_proof(root / 'output', root / 'source/route_plan.tsv', mode))
    if mode == 'eco': result['checkpoint_sha256'] = sha(root / 'output/routed.dcp')
    return result
