"""Review closed native composition evidence; never infer board or timing closure."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closed(root, stage, diagnostic=False):
    files = json.loads((root / 'FILES.json').read_text())
    require(all(sha(root / n) == h for n, h in files.items()), 'Manifest mismatch')
    result = json.loads((root / 'RESULT.json').read_text())
    expected_status = 'FAILED' if diagnostic else 'HBM_NATIVE_JOIN_EXPORTED'
    require(result['stage'] == stage and result['status'] == expected_status, 'Unexpected stage result')
    require(result['returncode'] == (1 if diagnostic else 0) and not result['timed_out'] and result['sources_unchanged'], 'Runner failure')
    require(all(sha(root / 'source' / n) == h for n, h in result['sources'].items()), 'Source mismatch')
    require(not re.search(r'(?im)^\s*(?:ERROR|FATAL):', (root / 'tool.log').read_text()), 'Tool error')
    return result


def table(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def config(path):
    return dict(line.split('\t', 1) for line in path.read_text().splitlines())


def xci(path):
    obj = json.loads(path.read_text())['ip_inst']['parameters']['component_parameters']
    return {k: v[0]['value'] for k, v in obj.items()}


def semantic(e, diagnostic=False):
    p = e['primitives']
    for name, count in {'HBM_REF_CLK': 2, 'HBM_SNGLBLI_INTF_APB': 2,
                        'HBM_SNGLBLI_INTF_AXI': 32, 'PCIE4CE4': 1}.items():
        require(p.get(name) == count, 'Native primitive count: ' + name)
    require(p.get('MMCME4_ADV', 0) >= 1, 'Missing clock generator')
    for name in ('native_hbm', 'axi_bram_ctrl_0', 'ntrpu_bridge'):
        expected = dict(e['original'][name])
        if name == 'ntrpu_bridge':
            expected['pipe_sim'] = 'false'
        require(e['generated'][name] == expected, 'Native IP configuration: ' + name)
    require(e['reader_configs'] == e['selected_reader_configs'], 'Reader profile changed')
    pins = {r['pin']: r for r in e['pins']}
    expected = {'core/clk': 4., 'core/mgmt_clk': 10., 'core/reader_clk': 5.,
                'core/mem_clk': 1000/450, 'hbm_binding/axi_clk': 1000/450}
    native = [r for r in e['pins'] if r['pin'] not in expected]
    require(Counter(r['consumer_reference'] for r in native) ==
            {'HBM_SNGLBLI_INTF_AXI': 32, 'HBM_SNGLBLI_INTF_APB': 2, 'HBM_REF_CLK': 2}, 'Hard-block clock coverage')
    for r in native:
        ref = r['consumer_reference']
        suffix, period = {'HBM_SNGLBLI_INTF_AXI': ('ACLK', 1000/450),
                          'HBM_SNGLBLI_INTF_APB': ('PCLK', 10.), 'HBM_REF_CLK': ('REF_CLK', 10.)}[ref]
        require(r['pin'].startswith('hbm_binding/memory/') and r['pin'].endswith('/' + suffix), 'Native clock pin identity')
        expected[r['pin']] = period
    require(len(e['pins']) == len(pins) == 41 and set(pins) == set(expected), 'Clock coverage')
    for name, period in expected.items():
        r = pins[name]
        require(abs(float(r['period']) - period) <= .001, 'Clock period: ' + name)
        require(bool(r['driver']) and 'BUFG' in r['driver_reference'], 'Buffered clock: ' + name)
    clocks = {r['clock']: r for r in e['clocks']}
    for name in ('core/reader_clk', 'core/mem_clk'):
        r = clocks[pins[name]['clock']]
        require(r['generated'] == '1' and bool(r['master']), 'Generated fabric clock: ' + name)
        require('clock_generator/' in pins[name]['driver'], 'Fabric clock origin: ' + name)
    require('ntrpu_bridge_i/' in pins['core/clk']['driver'], 'PCIe clock origin')
    resets = {r['pin']: r for r in e['resets']}
    expected_resets = {}
    for r in native:
        ref = r['consumer_reference']; cell = r['pin'].rsplit('/', 1)[0]
        if ref == 'HBM_SNGLBLI_INTF_AXI':
            expected_resets[cell + '/ARESET_N'] = 'core/reader/startup/memory_release_reg[2]/Q'
        elif ref == 'HBM_SNGLBLI_INTF_APB':
            expected_resets[cell + '/PRESET_N'] = 'core/endpoint/guard/cold_m_reg[2]/Q'
    require(len(e['resets']) == len(resets) == 34 and set(resets) == set(expected_resets), 'Reset coverage')
    for name, driver in expected_resets.items():
        require(resets[name]['actual_driver'] == driver, 'Canonical reset: ' + name)
    expected_wires = {r['core_pin']: r for r in e['wire_pairs']}
    require(len(expected_wires) == len(e['wires']) == 48, 'Wire coverage')
    require({r['core_pin'] for r in e['wires']} == set(expected_wires), 'Missing wire')
    wire_findings = {}
    for r in e['wires']:
        expected = expected_wires[r['core_pin']]
        require(r['peer_pin'] == expected['peer_pin'] and r['width'] == expected['width'], 'Wire identity: ' + r['core_pin'])
        if r['bits_connected'] != r['width']:
            wire_findings[r['core_pin']] = [int(r['bits_connected']), int(r['width'])]
    if diagnostic:
        # This explicit mode reports the failed acceptance gate. It must never
        # turn these unresolved bits into a successful integration verdict.
        require(wire_findings == {'m_awsize': [1, 3], 'm_bresp': [1, 2]}, 'Unexpected diagnostic findings')
    else:
        require(not wire_findings, 'Unresolved native wires: ' + str(wire_findings))
    return wire_findings


def review(generation, synthesis, inspection, self_test=False, diagnostic=False):
    closed(generation, 'generate'); closed(synthesis, 'synth'); ir = closed(inspection, 'inspect', diagnostic)
    require(ir['checkpoint_sha256'] == sha(synthesis / 'output/synth.dcp'), 'Inspection checkpoint mismatch')
    findings = (inspection / 'output/findings.txt').read_text().strip()
    if diagnostic:
        require(findings.splitlines() == [
            'Wire mismatch m_awsize -> ntrpu_bridge_i/s_axib_awsize: 1/3',
            'Wire mismatch m_bresp -> ntrpu_bridge_i/s_axib_bresp: 1/2'], 'Unexpected failed gate')
        require(not (inspection / 'output/STATUS.txt').exists(), 'Failed gate exported a passing status')
    else:
        require((inspection / 'output/STATUS.txt').read_text().strip() == 'HBM_NATIVE_JOIN_CLOCK_RESET_EXPORTED', 'Inspection gate missing')
        require(not findings, 'Inspection findings')
    for name, h in json.loads((generation / 'source/INPUTS.json').read_text()).items():
        if Path(name).suffix in ('.sv', '.v', '.xci'):
            require(sha(synthesis / 'source' / name) == h, 'Frozen input changed: ' + name)
    out = generation / 'output'
    e = dict(primitives={r['reference']: int(r['count']) for r in table(inspection / 'output/primitives.tsv')},
        pins=table(inspection / 'output/clock_consumers.tsv'), resets=table(inspection / 'output/resets.tsv'),
        clocks=table(inspection / 'output/clocks.tsv'), original={}, generated={}, reader_configs={},
        wire_pairs=table(inspection / 'source/wire_pairs.tsv'), wires=table(inspection / 'output/wires.tsv'))
    for name in ('native_hbm', 'ntrpu_bridge', 'axi_bram_ctrl_0'):
        e['original'][name] = xci(generation / 'source' / (name + '.xci'))
        e['generated'][name] = xci(out / f'project/native_join.srcs/sources_1/ip/{name}/{name}.xci')
    e['selected_reader_configs'] = json.loads((generation / 'source/HEAD_NATIVE_PROFILE.json').read_text())['selected_read_configs']
    for name in e['selected_reader_configs']:
        e['reader_configs'][name] = config(out / name)
    wire_findings = semantic(e, diagnostic)
    checks = {k: int(v) for k, v in re.findall(r'checking (\w+) \((\d+)\)',
                                             (synthesis / 'output/check_timing.rpt').read_text())}
    for name in ('no_clock', 'multiple_clock', 'generated_clocks', 'loops', 'latch_loops'):
        require(checks.get(name) == 0, 'Structural timing check: ' + name)
    cdc = {name: {'severity': severity, 'count': int(count)} for name, severity, count in
           re.findall(r'(?m)^(CDC-\d+)\s+(\w+)\s+(\d+)\s+', (synthesis / 'output/cdc.rpt').read_text())}
    rejected = []
    if self_test:
        mutations = {
            'missing_pcie': lambda x: x['primitives'].pop('PCIE4CE4'),
            'missing_hbm_stack': lambda x: x['primitives'].update(HBM_REF_CLK=1),
            'pipe_simulation': lambda x: x['generated']['ntrpu_bridge'].update(pipe_sim='true'),
            'reader_config': lambda x: x['reader_configs']['hbm_read_cdc_config.tsv'].update({'CONFIG.ID_WIDTH': '5'}),
            'missing_clock': lambda x: x['pins'].pop(),
            'wrong_clock_period': lambda x: x['pins'][0].update(period='8.000'),
            'missing_reset': lambda x: x['resets'].pop(),
            'noncanonical_reset': lambda x: x['resets'][0].update(actual_driver='ntrpu_bridge_i/axi_aresetn'),
            'truncated_wire': lambda x: x['wires'][0].update(bits_connected='1'),
            'missing_wire': lambda x: x['wires'].pop(),
        }
        for name, mutate in mutations.items():
            bad = copy.deepcopy(e); mutate(bad)
            try:
                semantic(bad, diagnostic)
            except ValueError:
                rejected.append(name)
            else:
                raise ValueError('Accepted negative control: ' + name)
    return dict(status='NATIVE_JOIN_DIAGNOSTIC_WITH_OPEN_FINDINGS' if diagnostic else 'NATIVE_JOIN_STRUCTURAL_CLOCK_RESET_REVIEWED',
        acceptance_passed=not diagnostic, clock_reset_consumers_validated=True,
        checkpoint_sha256=ir['checkpoint_sha256'], clock_consumers=41, canonical_reset_consumers=34,
        wire_connections_checked=48, wire_bits=sum(int(r['width']) for r in e['wires']),
        wire_bits_validated=sum(int(r['bits_connected']) for r in e['wires']),
        native_wiring_qualified=not bool(wire_findings), unresolved_wire_bits=wire_findings,
        constant_alias_bits=sum(int(r['constant_bits']) for r in e['wires']),
        negative_controls_rejected=rejected, integrated_timing_qualified=False,
        native_traffic_qualified=False, cdc_qualified=False, board_qualified=False,
        open_findings=dict(check_timing=checks, cdc=cdc,
            drc=['AVAL-326: missing refclk IBUFDS_GTE4 LOC', 'NSTD-1: I/O standards', 'UCIO-1: pin locations'],
            clock_override='fabric_refclk replaces fabric_ref_clk_in; both 10 ns'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('generation', 'synthesis', 'inspection'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--diagnostic', action='store_true', help='Report the known failed wire gate; never grant acceptance')
    args = parser.parse_args()
    print(json.dumps(review(args.generation, args.synthesis, args.inspection, args.self_test, args.diagnostic), indent=2))
