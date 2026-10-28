"""Review availability fixes and narrowly proven native constant optimization."""
import argparse
import copy
import json
from pathlib import Path
import re

from review_hbm_native_join import closed, config, require, semantic, sha, table, xci


def module(text, name):
    matches = list(re.finditer(r'^module ' + re.escape(name) + r'\b', text, re.M))
    require(len(matches) == 1, 'Missing/duplicate module: ' + name)
    start = matches[0].start()
    return text[start:text.index('endmodule', start)]


def optimized(e):
    """Do not count the three optimized pins as literal net continuity."""
    expected = {'core/m_awsize[1]': ('ntrpu_bridge_i/s_axib_awsize[1]', '1', '0'),
                'core/m_awsize[2]': ('ntrpu_bridge_i/s_axib_awsize[2]', '1', '0'),
                'core/m_bresp[0]': ('ntrpu_bridge_i/s_axib_bresp[0]', '0', '')}
    rows = {r['core_pin']: r for r in e['optimized']}
    require(len(e['optimized']) == len(rows) == 3 and set(rows) == set(expected), 'Optimization coverage')
    for pin, (peer, own, other) in expected.items():
        r = rows[pin]
        require((r['peer_pin'], r['core_constant'], r['peer_constant']) == (peer, own, other), 'Optimization constants')
        require(int(r['peer_net_count']) == 0 if pin.endswith('bresp[0]') else int(r['peer_net_count']) > 0,
                'Optimization net evidence')
    for r in e['wires']:
        require(int(r['optimized_bits']) == {'m_awsize': 2, 'm_bresp': 1}.get(r['core_pin'], 0), 'Optimization scope')
    source = e['top_source']
    for signal, width, own, peer in [('m_awsize', '2', 'm_awsize', 's_axib_awsize'),
                                     ('m_bresp', '1', 'm_bresp', 's_axib_bresp')]:
        declaration = re.search(r'wire\s+\[' + width + r':0\]\s+' + signal + r'\s*;', source)
        require(declaration is not None and declaration.start() < source.index('ntrpu_bridge ntrpu_bridge_i'), 'Declared native width')
        require(len(re.findall(r'\.' + peer + r'\s*\(\s*' + signal + r'\s*\)', source)) == 1, 'Source native connection')
        core = re.search(r'ntrpu_axi_bridge_hbm_clocked core\((.*?)\);', source, re.S)
        require(core is not None and '.*' in core[1] and not re.search(r'\.' + own + r'\s*\(', core[1]),
                'Core wildcard native connection')
    require("assign m_awsize=3'd6;" in re.sub(r'\s+', ' ', e['writer']).replace(' = ', '='), 'Writer transfer size')
    bridge = module(e['netlist'], 'ntrpu_bridge')
    require(re.search(r'GND\s+GND\s*\(\.G\(\\<const0>\s*\)\);', bridge), 'Native GND provenance')
    require(re.search(r'assign s_axib_bresp\[0\] = \\<const0>\s*;', bridge), 'Native BRESP zero')
    require('input [2:0]s_axib_awsize;' in bridge, 'Native AWSIZE declaration')
    # Its only occurrences are port list, input declaration and the callee's
    # named port with a literal constant. No use of the input survives.
    require(len(re.findall(r'\bs_axib_awsize\b', bridge)) == 3, 'AWSIZE has a surviving use')
    require(".s_axib_awsize({1'b0,1'b0,1'b0})" in bridge, 'Specialized native AWSIZE')
    top = module(e['netlist'], 'ntrpu_axi_bridge_hbm_native_join')
    require('.m_awsize(m_awsize)' in top, 'Writer AWSIZE output')
    require(".m_bresp({m_bresp,1'b0})" in top, 'Writer BRESP bit zero')
    require('.s_axib_bresp({m_bresp,NLW_ntrpu_bridge_i_s_axib_bresp_UNCONNECTED[0]})' in top, 'Preserved BRESP error bit')


def availability(e):
    require(len(e['inputs']) == 2 and {r['pin'] for r in e['inputs']} ==
            {f'availability/apb_complete_mgmt[{i}]' for i in range(2)}, 'APB completion input coverage')
    for r in e['inputs']:
        require(r['driver'].startswith('hbm_binding/memory/') and 'apb_switch_r_reg' in r['driver']
                and r['driver_reference'].startswith('FD') and abs(float(r['period'])-10.) < .001,
                'APB completion clock provenance')
    flops = {r['cell']: r for r in e['flops']}
    names = {f'availability/lock_sync_reg[{i}]' for i in range(3)}
    names |= {'availability/ready_mgmt_reg[0]', 'availability/ready_mgmt_reg[1]',
              'availability/armed_reg', 'availability/fault_reg'}
    require(len(e['flops']) == 7 and set(flops) == names, 'Availability register coverage')
    for name, r in flops.items():
        require(r['reference'] == 'FDCE' and abs(float(r['period'])-10.) < .001, 'Management register clock')
        # Vivado all_fanin -startpoints_only reports a register's launch C pin.
        require(r['reset_startpoints'] == 'core/endpoint/guard/cold_m_reg[2]/C', 'Availability reset launch register')
    net = module(e['netlist'], 'ntrpu_hbm_native_availability')
    inversion = "LUT1 #( .INIT(2'h1)) \\ready_mgmt[1]_i_2 (.I0(cold_rst_n), .O(\\ready_mgmt[1]_i_2_n_0 ));"
    require(inversion in re.sub(r'\s+', ' ', net), 'Availability cold-reset inversion')
    require(net.count('.CLR(\\ready_mgmt[1]_i_2_n_0 )') == 7, 'All availability resets share cold inversion')
    top = module(e['netlist'], 'ntrpu_axi_bridge_hbm_native_join')
    require(top.count('.cold_rst_n(management_cold_rst_n)') == 1 and
            '.APB_0_PRESET_N(management_cold_rst_n)' in top and
            '.management_cold_rst_n(management_cold_rst_n)' in top, 'Availability reset aliases canonical APB reset')
    for i in range(3):
        r = flops[f'availability/lock_sync_reg[{i}]']
        require(r['async_reg'].upper() in ('TRUE', '1'), 'LOCKED synchronizer attribute')
        if i:
            require(r['data_drivers'] == f'availability/lock_sync_reg[{i-1}]/Q', 'LOCKED synchronizer chain')
        else:
            require(r['data_drivers'].startswith('clock_generator/') and r['data_drivers'].endswith('/LOCKED'), 'Raw MMCM lock source')
    edges = {r['pin']: r['driver'] for r in e['edges']}
    require(len(e['edges']) == 2 and edges == {
        f'core/reader/startup/ready_meta_reg[{i}]/D': f'availability/ready_mgmt_reg[{i}]/Q' for i in range(2)},
        'Registered availability crossing')
    require(e['cdc'].get('CDC-10', {}).get('count', 0) == 0, 'Combinational logic before synchronizer')
    require(not e['duplicate_clock'] and not e['implicit_wires'], 'Wrapper/clock diagnostics remain')


def simulation(root, synthesis):
    files = json.loads((root / 'FILES.json').read_text())
    require(all(sha(root / n) == h for n, h in files.items()), 'Simulation manifest')
    r = json.loads((root / 'RESULT.json').read_text())
    require(r['status'] == 'NATIVE_AVAILABILITY_SIM_EXPORTED' and r['error'] is None and r['sources_unchanged'], 'Simulation result')
    require(all(sha(root / 'source' / n) == h for n, h in r['sources'].items()), 'Simulation source hashes')
    require(len(r['steps']) == 82 and all(s['returncode'] == 0 and not s['timed_out'] for s in r['steps']), 'Simulation steps')
    expected = {(case, seed) for case in range(20) for seed in range(4)}
    require(len(r['completed']) == 80 and {(c['case'], c['seed']) for c in r['completed']} == expected, 'Simulation coverage')
    for c in r['completed']:
        text = (root / f"case-{c['case']:02}-seed-{c['seed']}.log").read_text()
        matches = re.findall(r'NA PASS case=(\d+) seed=(\d+) comparisons=(\d+) cold_domains=4 retained_lanes=2', text)
        require(matches == [(str(c['case']), str(c['seed']), str(c['comparisons']))] and c['comparisons'] > 40, 'Simulation pass evidence')
        require(not re.search(r'(?im)^\s*(?:fatal|error):', text), 'Simulation error')
    require(not r['native_pcie_hbm_traffic'] and not r['physical_timing_qualified'], 'Simulation scope')
    for name in ('ntrpu_hbm_native_availability', 'ntrpu_axi_bridge_guard_v3_reset_export',
                 'ntrpu_hbm_startup', 'ntrpu_axi_bridge_hbm_mem_watch', 'ntrpu_axi_bridge_hbm_cdc', 'ntrpu_bridge_async_fifo'):
        require(sha(root / 'source' / (name + '.sv')) == sha(synthesis / 'source' / (name + '.sv')), 'Tested/synthesized RTL mismatch')
    return r


def review(generation, synthesis, inspection, sim, self_test=False):
    closed(generation, 'generate'); sr = closed(synthesis, 'synth'); ir = closed(inspection, 'inspect')
    require(ir['checkpoint_sha256'] == sha(synthesis / 'output/synth.dcp'), 'Checkpoint identity')
    require((inspection / 'output/STATUS.txt').read_text().strip() == 'HBM_NATIVE_FIX_OBSERVATIONS_EXPORTED', 'Observation status')
    require(not (inspection / 'output/findings.txt').read_text().strip(), 'Unexpected wire findings')
    for name, h in json.loads((generation / 'source/INPUTS.json').read_text()).items():
        if Path(name).suffix in ('.sv', '.v', '.xci'):
            require(sha(synthesis / 'source' / name) == h, 'Frozen input changed: ' + name)
    fixed = json.loads((synthesis / 'source/FIX_PREPARED.json').read_text())
    require(fixed['top_sha256'] == sha(synthesis / 'source/ntrpu_axi_bridge_hbm_native_join.sv'), 'Fixed top identity')
    require(fixed['availability_sha256'] == sha(synthesis / 'source/ntrpu_hbm_native_availability.sv'), 'Availability identity')
    out = generation / 'output'
    e = dict(primitives={r['reference']: int(r['count']) for r in table(inspection / 'output/primitives.tsv')},
        pins=table(inspection / 'output/clock_consumers.tsv'), resets=table(inspection / 'output/resets.tsv'),
        clocks=table(inspection / 'output/clocks.tsv'), original={}, generated={}, reader_configs={},
        wire_pairs=table(inspection / 'source/wire_pairs.tsv'), wires=table(inspection / 'output/wires.tsv'),
        optimized=table(inspection / 'output/optimized_bits.tsv'), flops=table(inspection / 'output/availability_flops.tsv'),
        edges=table(inspection / 'output/availability_edges.tsv'), inputs=table(inspection / 'output/availability_inputs.tsv'),
        netlist=(inspection / 'output/wrapper_netlist.v').read_text(),
        top_source=(synthesis / 'source/ntrpu_axi_bridge_hbm_native_join.sv').read_text(),
        writer=(synthesis / 'source/ntrpu_axi_bridge_writer.sv').read_text())
    for name in ('native_hbm', 'ntrpu_bridge', 'axi_bram_ctrl_0'):
        e['original'][name] = xci(generation / 'source' / (name + '.xci'))
        e['generated'][name] = xci(out / f'project/native_join.srcs/sources_1/ip/{name}/{name}.xci')
    for name in ('native_hbm', 'ntrpu_bridge', 'axi_bram_ctrl_0', 'fabric_clocks',
                 'hbm_read_cdc', 'hbm_read_width', 'hbm_read_protocol'):
        rel = f'project/native_join.srcs/sources_1/ip/{name}/{name}.xci'
        require(xci(out / rel) == xci(synthesis / rel), 'Synthesized IP configuration: ' + name)
    memories = list((synthesis / 'project').rglob('*.mem'))
    require(len(memories) == 4 and all(p.stat().st_size > 1000 for p in memories), 'HBM initialization files')
    e['selected_reader_configs'] = json.loads((generation / 'source/HEAD_NATIVE_PROFILE.json').read_text())['selected_read_configs']
    for name in e['selected_reader_configs']:
        e['reader_configs'][name] = config(out / name)
    log = (synthesis / 'tool.log').read_text()
    e['duplicate_clock'] = '[Constraints 18-1056]' in log
    e['implicit_wires'] = bool(re.search(r'(?i)WARNING:.*(?:used before|implicitly declared)', log))
    e['cdc'] = {name: dict(severity=severity, count=int(count)) for name, severity, count in
        re.findall(r'(?m)^(CDC-\d+)\s+(\w+)\s+(\d+)\s+', (synthesis / 'output/cdc.rpt').read_text())}
    def validate(value):
        # Keep the old diagnostic gate's exact accounting (1455/1458).
        # The additional proof below accounts separately for the three pins.
        semantic(value, diagnostic=True); optimized(value); availability(value)
    validate(e)
    checks = {k: int(v) for k, v in re.findall(r'checking (\w+) \((\d+)\)', (synthesis / 'output/check_timing.rpt').read_text())}
    for name in ('no_clock', 'multiple_clock', 'generated_clocks', 'loops', 'latch_loops'):
        require(checks.get(name) == 0, 'Structural timing: ' + name)
    sim_result = simulation(sim, synthesis)
    rejected = []
    if self_test:
        mutations = {
            'missing_hbm': lambda x: x['primitives'].update(HBM_SNGLBLI_INTF_AXI=31),
            'wrong_clock': lambda x: x['pins'][0].update(period='8.0'),
            'wrong_reset': lambda x: x['resets'][0].update(actual_driver='ntrpu_bridge_i/axi_aresetn'),
            'truncated_wire': lambda x: x['wires'][0].update(bits_connected='1'),
            'wrong_optimized_constant': lambda x: x['optimized'][0].update(core_constant='0'),
            'missing_optimization': lambda x: x['optimized'].pop(),
            'source_size_truncated': lambda x: x.update(top_source=x['top_source'].replace('wire [2:0] m_awsize;', 'wire m_awsize;')),
            'surviving_awsize_use': lambda x: x.update(netlist=x['netlist'].replace('input [2:0]s_axib_awsize;', 'input [2:0]s_axib_awsize; wire illicit=s_axib_awsize[1];')),
            'wrong_bresp_constant': lambda x: x.update(netlist=x['netlist'].replace('assign s_axib_bresp[0] = \\<const0>', 'assign s_axib_bresp[0] = \\<const1>')),
            'missing_sync_flop': lambda x: x['flops'].pop(),
            'wrong_reset_polarity': lambda x: x.update(netlist=x['netlist'].replace(".INIT(2'h1)", ".INIT(2'h2)")),
            'wrong_apb_clock': lambda x: x['inputs'][0].update(period='5.0'),
            'raw_availability_crossing': lambda x: x['edges'][0].update(driver='clock_generator/mmcm_adv_inst/LOCKED'),
            'critical_cdc': lambda x: x['cdc'].update({'CDC-10': dict(severity='Critical', count=1)}),
            'duplicate_clock': lambda x: x.update(duplicate_clock=True),
        }
        for name, mutate in mutations.items():
            bad = copy.deepcopy(e); mutate(bad)
            try:
                validate(bad)
            except ValueError:
                rejected.append(name)
            else:
                raise ValueError('Accepted negative control: ' + name)
    return dict(status='NATIVE_FIX_STRUCTURAL_AND_AVAILABILITY_REVIEWED', structural_acceptance_passed=True,
        checkpoint_sha256=ir['checkpoint_sha256'], clock_consumers=41, canonical_reset_consumers=34,
        native_connections=48, wire_bits_continuous_or_equal_constant=1455, optimized_bits_proven=3,
        availability_registers=7, registered_ready_crossings=2, simulation_cases=80,
        simulation_seconds=sim_result['seconds'], synthesis_seconds=sr['seconds'],
        negative_controls_rejected=rejected, cdc_qualified=False, native_traffic_qualified=False,
        integrated_timing_qualified=False, board_qualified=False, remaining_cdc=e['cdc'], check_timing=checks)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('generation', 'synthesis', 'inspection', 'simulation'):
        p.add_argument(name, type=Path)
    p.add_argument('--self-test', action='store_true'); a = p.parse_args()
    print(json.dumps(review(a.generation, a.synthesis, a.inspection, a.simulation, a.self_test), indent=2))
