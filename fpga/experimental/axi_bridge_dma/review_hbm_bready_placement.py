"""Review a bounded physical BREADY experiment without accepting timing waivers."""
import json
from pathlib import Path
import review_hbm_bready_physical as inherited
from review_hbm_capture_physical import closed, require, sha, table


def indexed(path, key='cell'):
    rows = table(path)
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows) and rows, 'Unique physical inventory required')
    return result


def binding(root, donor):
    native = closed(root)
    record = json.loads((root / 'source/CAPTURE_STAGE_BINDING.json').read_text())
    require(native['checkpoint_sha256'] == record['checkpoint_sha256'] == sha(donor), 'Placement input checkpoint changed')
    for field in ('prepared_files', 'frozen_audits'):
        for name, digest in record[field].items():
            require(sha(root / 'source' / name) == digest, 'Placement source changed')
    require(record['full_rtl_resynthesis'] is False, 'Placement is not RTL resynthesis')
    return native, record


def probe_review(root, reference, donor):
    native, record = binding(root, donor)
    out = root / 'output'
    require(record['placement_mode'] == 'probe' and record['baseline_read_only'], 'Placement probe binding')
    require((out / 'STATUS.txt').read_text().strip() == 'BREADY_PLACEMENT_PROBE_EXPORTED', 'Incomplete placement probe')
    proof = inherited.bready_proof(out, True)
    vendor = inherited.vendor_proof(out)
    previous = json.loads((reference / 'COMPARISON.json').read_text())
    require(proof == previous['bready'] and vendor == previous['vendor_feedback'], 'Read-only placement baseline differs')
    cone = indexed(out / 'placement/cone.tsv')
    require('weight_loader/bready_reg_reg[1]' in cone, 'Missing placement register')
    sites = indexed(out / 'placement/sites.tsv', 'site')
    require(len(sites) >= 100 and all(r['x'].isdigit() and r['y'].isdigit() for r in sites.values()), 'Physical site coverage')
    native_site = table(out / 'placement/native_site.tsv')
    require(len(native_site) == 1 and native_site[0]['x'].isdigit() and native_site[0]['y'].isdigit(), 'Native physical coordinates')
    return dict(status='BREADY_PLACEMENT_PROBE_REVIEWED', seconds=native['seconds'],
                checkpoint_sha256=sha(donor), cone_cells=len(cone), sites=len(sites),
                empty_sites=sum(not r['cells'] for r in sites.values()), bready=proof, vendor_feedback=vendor)


def placement_proof(out, plan_path, mode):
    plan = indexed(plan_path)
    register = 'weight_loader/bready_reg_reg[1]'
    gate = 'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf_i_33'
    allowed = {register, gate, 'weight_loader/bready_predictor/next_ready[1]_INST_0',
               'weight_loader/bready_predictor/next_ready[1]_INST_0_i_1'}
    require({register, gate} <= set(plan) <= allowed, 'Placement target scope changed')
    before = indexed(out / 'cells_before.tsv')
    after = indexed(out / 'cells_after.tsv')
    require(len(before) > 10000 and set(before) == set(after), 'Primitive inventory changed')
    require(2 <= len(plan) <= 6, 'Placement plan size')
    for name in before:
        require(all(before[name][k] == after[name][k] for k in ('primitive', 'init')), 'Primitive function changed')
    changed = {name for name in before if before[name] != after[name]}
    require(changed == (set(plan) if mode == 'eco' else set()), 'Collateral placement/property change')
    for name, row in plan.items():
        require(name in after and after[name]['primitive'] == row['primitive'], 'Placement target primitive')
        require(after[name]['loc'] == row['after_loc'] and after[name]['bel'].split('.')[-1] == row['after_bel'], 'Placement destination differs')
        if mode == 'eco':
            require(before[name]['loc'] == row['before_loc'] and before[name]['bel'] == row['before_bel'], 'Placement donor location differs')
    require(table(out / 'placement_actions.tsv') == table(plan_path), 'Physical action record differs')
    require(table(out / 'placement_before/connections.tsv') == table(out / 'placement_after/connections.tsv'), 'BREADY cone connectivity changed')
    cone_before = indexed(out / 'placement_before/cone.tsv')
    cone_after = indexed(out / 'placement_after/cone.tsv')
    require(set(cone_before) == set(cone_after) and set(plan) <= set(cone_before), 'BREADY cone inventory changed')
    for name in cone_before:
        require(all(cone_before[name][k] == cone_after[name][k] for k in ('primitive', 'init')), 'BREADY cone function changed')
    clock_before = table(out / 'clock_pins_before.tsv')
    clock_after = table(out / 'clock_pins_after.tsv')
    require(len(clock_before) == 8321 and clock_before == clock_after, 'Clock connectivity changed')
    require(len({r['pin'] for r in clock_before}) == len(clock_before), 'Duplicate clock pins')
    require({r['net'] for r in clock_before} == {'clock_generator/inst/clk_out2'}, 'Clock repair scope changed')
    require(register + '/C' in {r['pin'] for r in clock_before}, 'Missing moved register clock')
    if mode == 'eco':
        repair = table(out / 'clock_repair.tsv')
        require(repair == [dict(net='clock_generator/inst/clk_out2',
                               command='route_design -unroute -nets; update_clock_routing -net')], 'Clock repair record')
    return dict(primitive_cells=len(before), moved_cells=sorted(changed), plan=list(plan.values()),
                cone_cells=len(cone_before), cone_connections=len(table(out / 'placement_after/connections.tsv')),
                collateral_placement_changes=0, primitive_types_and_init_unchanged=True, cone_connections_unchanged=True,
                affected_clock_pins=len(clock_before), clock_connectivity_unchanged=True)


def review(root, reference, donor):
    native, record = binding(root, donor)
    mode = record['placement_mode']
    require(mode in ('eco', 'reopen'), 'Unknown physical placement stage')
    result = inherited.review(root, reference, donor, 'reopen')
    result.update(status='BREADY_PLACEMENT_REVIEWED', placement_mode=mode,
                  full_rtl_resynthesis=False,
                  placement=placement_proof(root / 'output', root / 'source/placement_plan.tsv', mode))
    if mode == 'eco':
        result['checkpoint_sha256'] = sha(root / 'output/routed.dcp')
    return result
