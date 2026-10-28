"""Review one shared-predictor movement with unchanged clocks and board contracts."""
import json
from pathlib import Path
import review_hbm_bready_physical as inherited
from review_hbm_bready_placement import indexed, binding
from review_hbm_capture_physical import require, sha, table

TARGET = 'weight_loader/bready_predictor_i_2'
PLAN = dict(cell=TARGET, primitive='LUT2', before_loc='SLICE_X7Y11',
            before_bel='SLICEL.D5LUT', after_loc='SLICE_X63Y11', after_bel='F6LUT')


def placement_proof(out, plan_path, mode):
    require(mode in ('eco', 'reopen'), 'Shared placement mode')
    require(table(plan_path) == [PLAN], 'Shared placement plan changed')
    before = indexed(out / 'cells_before.tsv')
    after = indexed(out / 'cells_after.tsv')
    require(len(before) == 120241 and set(before) == set(after), 'Primitive inventory changed')
    for name in before:
        require(all(before[name][k] == after[name][k] for k in ('primitive', 'init')), 'Primitive function changed')
    changed = {name for name in before if before[name] != after[name]}
    require(changed == ({TARGET} if mode == 'eco' else set()), 'Collateral placement/property change')
    target = after[TARGET]
    require(target['primitive'] == 'LUT2' and target['init'] == "4'h1", 'Shared predictor function changed')
    require(target['loc'] == PLAN['after_loc'] and target['bel'].split('.')[-1] == PLAN['after_bel'], 'Shared destination differs')
    require(target['fixed_loc'] == target['fixed_bel'] == '1', 'Shared placement not fixed')
    if mode == 'eco':
        require(before[TARGET]['loc'] == PLAN['before_loc'] and before[TARGET]['bel'] == PLAN['before_bel'], 'Shared origin differs')
        require(before[TARGET]['fixed_loc'] == before[TARGET]['fixed_bel'] == '0', 'Fixed shared donor')
    require(table(out / 'placement_actions.tsv') == [PLAN], 'Physical action record differs')
    occupancy = table(out / 'destination_before.tsv')
    expected = [{'bel': 'SLICE_X63Y11/F5LUT', 'cells': ''},
                {'bel': 'SLICE_X63Y11/F6LUT', 'cells': '' if mode == 'eco' else TARGET}]
    require(occupancy == expected, 'Shared destination occupancy differs')
    cone_before = indexed(out / 'placement_before/cone.tsv')
    cone_after = indexed(out / 'placement_after/cone.tsv')
    required = {TARGET} | {f'weight_loader/bready_reg_reg[{ch}]' for ch in (0, 1)}
    require(len(cone_before) == 17 and set(cone_before) == set(cone_after) and required <= set(cone_before), 'Both-channel cone inventory changed')
    for name in cone_before:
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
    pips = table(out / 'clock_pips_before.tsv')
    require(len(pips) == 2706 and len({r['pip'] for r in pips}) == len(pips), 'Clock route coverage')
    require(pips == table(out / 'clock_pips_after.tsv'), 'Clock route changed')
    require(not (out / 'clock_repair.tsv').exists(), 'Unexpected clock repair')
    return dict(primitive_cells=len(before), moved_cells=sorted(changed), plan=[PLAN],
                cone_cells=len(cone_before), cone_connections=len(connections), shared_endpoints=endpoints,
                shared_drivers=drivers, collateral_placement_changes=0,
                primitive_types_and_init_unchanged=True, cone_connections_unchanged=True,
                affected_clock_pins=len(clock), clock_connectivity_unchanged=True,
                clock_pips=len(pips), clock_route_unchanged=True)


def review(root, reference, donor):
    native, record = binding(root, donor)
    mode = record['placement_mode']
    require(record['shared_predictor_placement'] is True and record['clock_repair'] is False, 'Shared experiment binding')
    result = inherited.review(root, reference, donor, 'reopen')
    result.update(status='BREADY_SHARED_PLACEMENT_REVIEWED', placement_mode=mode, full_rtl_resynthesis=False,
                  placement=placement_proof(root / 'output', root / 'source/placement_plan.tsv', mode))
    if mode == 'eco':
        result['checkpoint_sha256'] = sha(root / 'output/routed.dcp')
    return result
