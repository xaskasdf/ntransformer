"""Review five explicit movements with a rebuilt clock route and board contracts."""
import json
from pathlib import Path
import review_hbm_bready_physical as inherited
from review_hbm_bready_placement import indexed, binding
from review_hbm_capture_physical import require, sha, table

TARGET = 'weight_loader/bready_predictor_i_2'
PLAN = [{'cell': 'weight_loader/bready_reg_reg[1]', 'primitive': 'FDCE', 'before_loc': 'SLICE_X41Y11', 'before_bel': 'SLICEM.CFF', 'after_loc': 'SLICE_X100Y4', 'after_bel': 'CFF'}, {'cell': 'weight_loader/bready_predictor/next_ready[1]_INST_0', 'primitive': 'LUT4', 'before_loc': 'SLICE_X41Y11', 'before_bel': 'SLICEM.C6LUT', 'after_loc': 'SLICE_X100Y4', 'after_bel': 'C6LUT'}, {'cell': 'weight_loader/bready_predictor/next_ready[1]_INST_0_i_1', 'primitive': 'LUT5', 'before_loc': 'SLICE_X41Y11', 'before_bel': 'SLICEM.H6LUT', 'after_loc': 'SLICE_X100Y4', 'after_bel': 'H6LUT'}, {'cell': 'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf_i_33', 'primitive': 'LUT2', 'before_loc': 'SLICE_X113Y11', 'before_bel': 'SLICEL.C6LUT', 'after_loc': 'SLICE_X122Y0', 'after_bel': 'A6LUT'}, {'cell': 'weight_loader/bready_predictor_i_2', 'primitive': 'LUT2', 'before_loc': 'SLICE_X7Y11', 'before_bel': 'SLICEL.D5LUT', 'after_loc': 'SLICE_X63Y11', 'after_bel': 'F6LUT'}]


def placement_proof(out, plan_path, mode):
    require(mode in ('eco', 'reopen'), 'Balanced placement mode')
    require(table(plan_path) == PLAN, 'Balanced placement plan changed')
    before = indexed(out / 'cells_before.tsv')
    after = indexed(out / 'cells_after.tsv')
    require(len(before) == 120241 and set(before) == set(after), 'Primitive inventory changed')
    for name in before:
        require(all(before[name][k] == after[name][k] for k in ('primitive', 'init')), 'Primitive function changed')
    changed = {name for name in before if before[name] != after[name]}
    require(changed == ({r['cell'] for r in PLAN} if mode == 'eco' else set()), 'Collateral placement/property change')
    for row in PLAN:
        name = row['cell']; target = after[name]
        require(target['primitive'] == row['primitive'], 'Balanced primitive differs')
        require(target['loc'] == row['after_loc'] and target['bel'].split('.')[-1] == row['after_bel'], 'Balanced destination differs')
        require(target['fixed_loc'] == target['fixed_bel'] == '1', 'Balanced placement not fixed')
        if mode == 'eco':
            require(before[name]['loc'] == row['before_loc'] and before[name]['bel'] == row['before_bel'], 'Balanced origin differs')
            require(before[name]['fixed_loc'] == before[name]['fixed_bel'] == '0', 'Fixed balanced donor')
    require(after[TARGET]['init'] == "4'h1", 'Shared predictor function changed')
    require(table(out / 'placement_actions.tsv') == PLAN, 'Physical action record differs')
    expected = {}
    for row in PLAN:
        dest = row['after_loc'] + '/' + row['after_bel']
        expected[dest] = '' if mode == 'eco' else row['cell']
        if dest.endswith('6LUT'): expected[dest[:-4] + '5LUT'] = ''
    require(table(out / 'destination_before.tsv') == [dict(bel=k, cells=v) for k,v in sorted(expected.items())], 'Balanced destination occupancy differs')
    cone_before = indexed(out / 'placement_before/cone.tsv')
    cone_after = indexed(out / 'placement_after/cone.tsv')
    required = {r['cell'] for r in PLAN} | {f'weight_loader/bready_reg_reg[{ch}]' for ch in (0, 1)}
    require(len(cone_before) == 17 and set(cone_before) == set(cone_after) and required <= set(cone_before), 'Both-channel cone inventory changed')
    for name in cone_before:
        require(cone_before[name] == before[name] and cone_after[name] == after[name], 'Cone/global inventory differs')
        require(all(cone_before[name][k] == cone_after[name][k] for k in ('primitive', 'init')), 'Cone function changed')
    connections = table(out / 'placement_before/connections.tsv')
    require(len(connections) == 1699 and len({(r['cell'], r['pin']) for r in connections}) == 1699 and
            connections == table(out / 'placement_after/connections.tsv'), 'Both-channel connectivity changed')
    endpoints = table(out / 'placement_before/shared_endpoints.tsv')
    require(endpoints == table(out / 'placement_after/shared_endpoints.tsv'), 'Shared endpoints changed')
    require(len(endpoints) == 2 and {x['pin'] for x in endpoints} ==
            {f'weight_loader/bready_reg_reg[{ch}]/D' for ch in (0, 1)}, 'Shared endpoint coverage')
    drivers = table(out / 'placement_before/shared_drivers.tsv')
    require(drivers == table(out / 'placement_after/shared_drivers.tsv'), 'Shared drivers changed')
    require(len(drivers) == 2 and {x['pin'] for x in drivers} == {TARGET+'/I0', TARGET+'/I1'} and
            {x['driver'] for x in drivers} == {f'weight_loader/FSM_sequential_ms_reg[{ch}]/Q' for ch in (0, 1)}, 'Shared driver coverage')
    clock = table(out / 'clock_pins_before.tsv')
    require(len(clock) == 8321 and len({r['pin'] for r in clock}) == len(clock) and
            {r['net'] for r in clock} == {'clock_generator/inst/clk_out2'}, 'Clock connectivity coverage')
    require(clock == table(out / 'clock_pins_after.tsv'), 'Clock connectivity changed')
    require('weight_loader/bready_reg_reg[1]/C' in {r['pin'] for r in clock}, 'Missing moved register clock')
    before_pips = table(out / 'clock_pips_before.tsv'); after_pips = table(out / 'clock_pips_after.tsv')
    for rows in (before_pips, after_pips):
        require(len(rows) > 2000 and len({r['pip'] for r in rows}) == len(rows), 'Clock route coverage')
        require(all(r['pip'] for r in rows), 'Empty clock PIP')
    if mode == 'eco':
        require(table(out / 'clock_repair.tsv') == [dict(net='clock_generator/inst/clk_out2', command='route_design -unroute -nets; update_clock_routing -net')], 'Clock repair record')
    else:
        require(before_pips == after_pips and not (out / 'clock_repair.tsv').exists(), 'Reopen altered clock route')
    bp = {r['pip'] for r in before_pips}; ap = {r['pip'] for r in after_pips}
    return dict(primitive_cells=len(before), moved_cells=sorted(changed), plan=PLAN,
                cone_cells=len(cone_before), cone_connections=len(connections), shared_endpoints=endpoints,
                shared_drivers=drivers, collateral_placement_changes=0,
                primitive_types_and_init_unchanged=True, cone_connections_unchanged=True,
                affected_clock_pins=len(clock), clock_connectivity_unchanged=True,
                clock_pips_before=len(bp), clock_pips_after=len(ap), clock_pips_added=len(ap-bp),
                clock_pips_removed=len(bp-ap), clock_route_unchanged=bp == ap)



def review(root, reference, donor):
    native, record = binding(root, donor)
    mode = record['placement_mode']
    require(record['balanced_predictor_placement'] is True and record['clock_repair'] is True, 'Balanced experiment binding')
    result = inherited.review(root, reference, donor, 'reopen')
    result.update(status='BREADY_BALANCED_PLACEMENT_REVIEWED', placement_mode=mode, full_rtl_resynthesis=False,
                  placement=placement_proof(root / 'output', root / 'source/placement_plan.tsv', mode))
    if mode == 'eco':
        result['checkpoint_sha256'] = sha(root / 'output/routed.dcp')
    return result
