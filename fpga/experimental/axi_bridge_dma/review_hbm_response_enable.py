"""Review HBM-domain physical optimization and the unchanged response enable."""
import argparse
import itertools
from collections import Counter
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from review_hbm_reset_integration import (
    closed, sha, require, table, lines, stable, complete_timing, route_status,
    normalized_timing, semantic_xdc, reset_topology, path_slack, findings, PINS,
    loader_paths, loader_sync, hbm_connections, snapshot_bounds)

PREFIX = 'core/reader/channels[0].adapter/tracker/response_cut'
DRIVER = PREFIX + '/slot1[264]_i_1'
PINS_EXPECTED = {f'{PREFIX}/slot1_reg[{i}]/CE' for i in range(265)}


def tcl_names(value):
    names = re.findall(r'\{([^{}]+)\}|([^\s{}]+)', value)
    return [a or b for a, b in names]


def properties(path):
    rows = table(path)
    result = {(r['cell'], r['property']): r['value'] for r in rows}
    require(len(result) == len(rows), 'Duplicate logical property')
    return result


def driver_functions(path):
    props = properties(path)
    result = {}
    for cell in {c for c, _ in props}:
        require(props[cell, 'REF_NAME'] == 'LUT4', 'Enable driver is not a LUT4')
        init = re.fullmatch(r"16'h([0-9A-Fa-f]{4})", props[cell, 'INIT'])
        require(init, 'Unexpected LUT INIT encoding')
        inputs = [tcl_names(props[cell, f'I{i}']) for i in range(4)]
        require(all(len(x) == 1 for x in inputs), 'Enable input is not singly driven')
        signals = [x[0] for x in inputs]
        require(len(set(signals)) == 4, 'Unexpected tied enable inputs')
        order = sorted(signals)
        truth = []
        for values in itertools.product((0, 1), repeat=4):
            binding = dict(zip(order, values))
            address = sum(binding[s] << i for i, s in enumerate(signals))
            truth.append((int(init[1], 16) >> address) & 1)
        result[cell] = (order, truth)
    return result, props


def logic_proof(before, after):
    a, ap = driver_functions(before / 'driver_logic.tsv')
    b, bp = driver_functions(after / 'driver_logic.tsv')
    require(set(a) == {DRIVER}, 'Unexpected original enable driver')
    require(all(value == a[DRIVER] for value in b.values()), 'Enable truth table changed')
    require(all(c == DRIVER or c.startswith(DRIVER + '_replica') for c in b), 'Unexpected replica name')
    loads = table(after / 'enable_loads.tsv')
    require(len(loads) == 265 and {r['pin'] for r in loads} == PINS_EXPECTED, 'Enable destination set changed')
    require(all(r['cell_type'] == 'FDRE' and r['location'] and r['bel'] for r in loads), 'Response registers changed/unplaced')
    require(all(r['driver'].removesuffix('/O') in b for r in loads), 'Unreviewed enable driver')
    driver_loads = {cell: tcl_names(bp[cell, 'O_LOADS']) for cell in b}
    require(Counter(p for values in driver_loads.values() for p in values) == Counter(PINS_EXPECTED),
            'Replica has extra, missing or duplicated consumers')
    require(all(r['pin'] in driver_loads[r['driver'].removesuffix('/O')] for r in loads), 'Driver/load views disagree')
    cells_before, cells_after = set(lines(before / 'cells.txt')), set(lines(after / 'cells.txt'))
    require(not cells_before - cells_after and cells_after - cells_before == set(b) - {DRIVER}, 'Unrelated cell edit')
    x, y = properties(before / 'tracker_logic.tsv'), properties(after / 'tracker_logic.tsv')
    required_registers={f'{PREFIX}/slot{s}_reg[{i}]' for s in (0,1) for i in range(265)}
    for props in (x,y):
        require(required_registers <= {c for c,k in props if k=='PIN_C'}, 'Missing tracker clock/connectivity coverage')
    def canonical(props, drivers):
        result = {}
        for (cell, prop), value in props.items():
            if cell in drivers:
                continue  # All 16 input combinations are independently enumerated above.
            if prop.startswith('PIN_'):
                value = ' '.join(DRIVER + '/O' if v.removesuffix('/O') in drivers else v for v in tcl_names(value))
            result[cell, prop] = value
        return result
    require(canonical(x, a) == canonical(y, b), 'Collateral tracker logic/connectivity edit')
    return dict(enable_destinations=265, original_drivers=1, resulting_drivers=len(b),
                truth_assignments_per_driver=16, equivalent=True,
                tracker_cells_compared=len({c for c,k in x}),
                tracker_input_connections_compared=sum(k.startswith('PIN_') for c,k in x),
                added_cells=sorted(cells_after-cells_before),
                added_nets=sorted(set(lines(after/'nets.txt'))-set(lines(before/'nets.txt'))),
                removed_nets=sorted(set(lines(before/'nets.txt'))-set(lines(after/'nets.txt'))),
                fanout_by_driver={c:len(v) for c,v in driver_loads.items()})


def timing_sample(path):
    rows = table(path)
    require(len(rows) == 265 and {r['ENDPOINT_PIN'] for r in rows} == PINS_EXPECTED, 'Incomplete enable timing sample')
    require(all(r['STARTPOINT_CLOCK'] == r['ENDPOINT_CLOCK'] == 'clk_out2_fabric_clocks' for r in rows), 'Enable clock changed')
    return dict(paths=265, failing=sum(float(r['SLACK']) < 0 for r in rows),
                minimum_slack_ns=min(float(r['SLACK']) for r in rows),
                maximum_datapath_ns=max(float(r['DATAPATH_DELAY']) for r in rows))


def snapshot_review(root,reference,donor):
    # A deliberately interrupted runner is NOT a successful full-board reopen.
    # Review only the two complete snapshots preceding its observed phase marker.
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},'Snapshot archive inventory differs')
    require(all(sha(root/n)==h for n,h in files.items()),'Snapshot archive hash differs')
    native=json.loads((root/'RESULT.json').read_text())
    marker=json.loads((root/'SNAPSHOT_COLLECTION_STOP.json').read_text())
    binding=json.loads((root/'source/RESPONSE_ENABLE_BINDING.json').read_text())
    require(native['status']=='FAILED' and native['returncode']==-15 and not native['timed_out'] and
            native['sources_unchanged'],'Unexpected snapshot runner disposition')
    require(marker['status']=='COMPLETE_READ_ONLY_SNAPSHOTS_COLLECTED' and
            marker['created_by']=='external private-job observer' and
            marker['native_full_reopen_audit_completed'] is False,'Incorrect snapshot completion claim')
    require(all(sha(root/n)==h for n,h in marker['files'].items()),'Snapshot changed after observer stop')
    require(all(sha(root/'source'/n)==h for n,h in native['sources'].items()),'Snapshot source changed')
    require(binding['mode']=='reopen' and native['checkpoint_sha256']==binding['checkpoint_sha256']==sha(donor),
            'Wrong snapshot checkpoint')
    candidate=reference/'candidate';out=root/'output'
    require(binding['reference_sha256']==sha(candidate/'routed.dcp')==
            (out/'reference_sha256.txt').read_text().strip(),'Wrong original snapshot')
    require(binding['script_sha256']==sha(root/'source/inspect_hbm_response_enable.tcl'),'Snapshot script changed')
    require((binding['timeout_seconds'],binding['cpu_count'],binding['nice'])==(1200,2,10),'Snapshot resource boundary changed')
    require(not re.search(r'^ERROR:',(root/'tool.log').read_text(),re.M),'Native snapshot error')
    rows=table(out/'experiment.tsv')
    require(len(rows)==1 and rows[0]['mode']=='reopen' and rows[0]['command']=='open_checkpoint' and
            rows[0]['clock_or_exception_edits']=='0','Snapshot phase did not complete')
    for label in ('reference','before'):
        folder=out/label
        require(all('output/'+label+'/'+n in marker['files'] for n in ('tracker_logic.tsv','cells.txt','nets.txt',
                'enable_loads.tsv','driver_logic.tsv','enable_max.tsv','enable_min.tsv','timing.rpt','route.rpt','effective.xdc')),
                'Snapshot file coverage incomplete')
        reset_topology(folder,lines(candidate/'reset_consumers.txt'))
        require(semantic_xdc((folder/'effective.xdc').read_text())==semantic_xdc((candidate/'reports/effective.xdc').read_text()),
                'Snapshot scoped constraints changed')
        route_status(folder/'route.rpt',True)
    proof=logic_proof(out/'reference',out/'before')
    require(not proof['added_cells'] and not proof['added_nets'] and not proof['removed_nets'],'Snapshot netlist changed')
    return dict(status='HBM_RESPONSE_ENABLE_SNAPSHOTS_REVIEWED',mode='reopen',seconds=native['seconds'],
        native_runner_status='FAILED_CONTROLLED_STOP',native_full_reopen_audit_completed=False,
        donor_sha256=sha(donor),checkpoint_sha256=sha(donor),logic=proof,
        timing=complete_timing(out/'before/timing.rpt'),before_timing=complete_timing(out/'reference/timing.rpt'),
        route=route_status(out/'before/route.rpt',True),
        enable_before={t:timing_sample(out/'reference'/('enable_'+t+'.tsv')) for t in ('max','min')},
        enable_after={t:timing_sample(out/'before'/('enable_'+t+'.tsv')) for t in ('max','min')},
        timing_qualified=False,board_qualified=False,native_traffic_qualified=False)


def review(root, reference, donor, mode, probe_root=None):
    native = closed(root); out = root / 'output'; candidate = reference / 'candidate'
    require(native['checkpoint_sha256'] == sha(donor), 'Wrong input checkpoint')
    binding = json.loads((root/'source/RESPONSE_ENABLE_BINDING.json').read_text())
    require(binding['mode'] == mode and binding['checkpoint_sha256'] == sha(donor), 'Preparation mismatch')
    require((binding['timeout_seconds'],binding['cpu_count'],binding['nice']) == (1200,2,10), 'Resource boundary changed')
    require(binding['script_sha256'] == sha(root/'source/inspect_hbm_response_enable.tcl'), 'Native script changed')
    for name, digest in binding['frozen_audits'].items():
        require(sha(root/'source'/name) == sha(reference/'prepared-audits'/name) == digest, 'Frozen audit changed')
    require((out/'STATUS.txt').read_text().strip() == f'HBM_RESPONSE_ENABLE_{mode}_EXPORTED_REVIEW_REQUIRED', 'Incomplete native stage')
    expected = table(out/'experiment.tsv')
    command = 'phys_opt_design -placement_opt -routing_opt -path_groups' if mode=='placement' else 'open_checkpoint'
    require(len(expected)==1 and expected[0]['mode']==mode and expected[0]['command']==command and
            expected[0]['clock_or_exception_edits']=='0', 'Unexpected physical command')
    if mode=='placement':
        require(expected[0]['target']=='clk_out2_fabric_clocks', 'Wrong physical path group')
    probe=root if mode=='reopen' else probe_root
    require(probe is not None,'Independent semantic reopen required')
    if mode=='reopen':return snapshot_review(root,reference,donor)
    snapshot=snapshot_review(probe,reference,out/'routed.dcp')
    probe_native=json.loads((probe/'RESULT.json').read_text())
    probe_binding=json.loads((probe/'source/RESPONSE_ENABLE_BINDING.json').read_text())
    candidate_sha=sha(out/'routed.dcp') if mode=='placement' else sha(donor)
    require(probe_native['checkpoint_sha256']==probe_binding['checkpoint_sha256']==candidate_sha and
            probe_binding['mode']=='reopen' and
            probe_binding['reference_sha256']==sha(candidate/'routed.dcp')==
            (probe/'output/reference_sha256.txt').read_text().strip(), 'Wrong semantic probe checkpoint pair')
    before=probe/'output/reference';after=probe/'output/before'
    proof = logic_proof(before,after)
    require(not proof['added_cells'] and not proof['added_nets'] and not proof['removed_nets'], 'Physical-only run changed netlist inventory')
    for path in (out/'before.xdc',out/'before/effective.xdc',out/'after/effective.xdc',out/'reports/effective.xdc'):
        require(normalized_timing(path)==normalized_timing(candidate/'reports/effective.xdc'), 'Timing constraints changed')
    # The exact native properties retain their scopes; reject all new timing exceptions.
    require(semantic_xdc((out/'reports/effective.xdc').read_text()) == semantic_xdc((out/'before.xdc').read_text()),
            'Scoped constraints or physical properties changed')
    for stage in ('before','after'):
        reset_topology(out/stage,lines(candidate/'reset_consumers.txt'))
        route_status(out/stage/'route.rpt',True)
    calibration_probe=reference/'calibration-probe'
    require(closed(calibration_probe)['checkpoint_sha256']==sha(candidate/'routed.dcp'), 'Wrong calibration reference checkpoint')
    require(table(out/'calibration_selectors.tsv') == table(calibration_probe/'output/current_calibration_selectors.tsv'), 'PCIe calibration membership changed')
    contract=out/'contracts'; old=candidate/'contracts'
    require((contract/'AUDIT_STATUS.txt').read_text().strip()=='BOARD_IMPLEMENTATION_CONTRACTS_EXPORTED_REVIEW_REQUIRED', 'Incomplete scoped audit')
    for path in old.glob('*.tsv'):
        require(stable(table(contract/path.name))==stable(table(path)), 'Board contract changed: '+path.name)
    domains=table(contract/'domain_clocks.tsv');clocks={x['logical']:x['clock'] for x in domains}
    loader_paths(table(contract/'loader_bounds.tsv'),clocks)
    loader_sync(table(contract/'loader_synchronizers.tsv'),table(contract/'loader_interstage.tsv'),clocks)
    hbm_connections(table(contract/'loader_hbm_connections.tsv'));snapshot_bounds(table(contract/'snapshot_bounds.tsv'))
    scoped={name:path_slack(table(contract/name)) for name in ('paths.tsv','snapshot_bounds.tsv','loader_bounds.tsv',
            'snapshot_interstage.tsv','loader_interstage.tsv','readiness_source_paths.tsv','synchronous_paths.tsv','board_lock_paths.tsv')}
    require(all(v['failing']==0 for v in scoped.values()), 'Scoped timing failure')
    pins=table(out/'trial_pins.tsv')
    require(len(pins)==11 and {x['port']:(x['ball'],x['standard']) for x in pins}==PINS, 'Trial pinout changed')
    reports=out/'reports'
    skew_text=(reports/'bus_skew.rpt').read_text().split('2. Bus Skew Report Per Constraint')[1]
    skew=[float(x[3]) for x in re.findall(r'^\s*(Slow|Fast)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$',skew_text,re.M)]
    require(len(skew)==18 and min(skew)>=0,'Bus skew regression')
    findings_result={name:findings(reports/(name+'.rpt')) for name in ('drc','methodology')}
    for name,current in findings_result.items():
        require(current==findings(candidate/'reports'/(name+'.rpt')), 'Physical findings changed: '+name)
    require('No waivers found.' in (reports/'waivers.rpt').read_text(),'Waivers added')
    cdc=lambda p:[list(row) for row in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+',p.read_text(),re.M)]
    require(cdc(reports/'cdc.rpt')==cdc(candidate/'reports/cdc.rpt'), 'CDC findings changed')
    coverage=lambda p:dict(re.findall(r'^\d+\. checking (\w+) \((\d+)\)$',p.read_text(),re.M))
    require(coverage(reports/'check_timing.rpt')==coverage(candidate/'reports/check_timing.rpt'), 'Timing coverage changed')
    distribution=table(out/'distribution_paths.tsv')
    require(stable(distribution)==stable(table(candidate/'distribution_paths.tsv')) and
            all(float(x['slack_ns'])>=0 for x in distribution),'Distribution recovery/removal changed or failed')
    return dict(status='HBM_RESPONSE_ENABLE_REVIEWED',mode=mode,seconds=native['seconds'],
                donor_sha256=sha(donor),checkpoint_sha256=sha(out/'routed.dcp') if mode=='placement' else sha(donor),
                logic=proof,scoped_timing=scoped,bounded_paths=2195,bus_skew_minimum_slack_ns=min(skew),
                distribution_paths=distribution,findings=findings_result,cdc=cdc(reports/'cdc.rpt'),
                route=route_status(reports/'route_status.rpt',True),
                before_timing=complete_timing(out/'before/timing.rpt'),timing=complete_timing(reports/'timing.rpt'),
                enable_before={t:timing_sample(before/('enable_'+t+'.tsv')) for t in ('max','min')},
                enable_after={t:timing_sample(after/('enable_'+t+'.tsv')) for t in ('max','min')},
                native_enable_before={t:timing_sample(out/'before'/('enable_'+t+'.tsv')) for t in ('max','min')},
                native_enable_after={t:timing_sample(out/'after'/('enable_'+t+'.tsv')) for t in ('max','min')},
                semantic_probe_checkpoint_sha256=candidate_sha,
                timing_qualified=False,board_qualified=False,native_traffic_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','donor'):p.add_argument(name,type=Path)
    p.add_argument('--mode',choices=('placement','reopen'),required=True)
    p.add_argument('--probe-root',type=Path)
    a=p.parse_args()
    result=snapshot_review(a.root,a.reference,a.donor) if a.mode=='reopen' else review(a.root,a.reference,a.donor,a.mode,a.probe_root)
    print(json.dumps(result,indent=2))
