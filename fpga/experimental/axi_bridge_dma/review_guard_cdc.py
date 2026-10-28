"""Review exported guard structure and scoped constraints; never certify board CDC."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re

CHAINS = {
    'cold_m': ('FDCE', 'mgmt_clk', 'VCC/P'),
    'cold_a': ('FDCE', 'axi_clk', 'VCC/P'),
    'healthy_m': ('FDCE', 'mgmt_clk', 'VCC/P'),
    'admit_a': ('FDCE', 'axi_clk', 'VCC/P'),
    'stop_a': ('FDPE', 'axi_clk', 'GND/G'),
    'echo_a': ('FDRE', 'axi_clk', 'request_reg/Q'),
    'echo_m': ('FDRE', 'mgmt_clk', 'echo_a_reg[2]/Q'),
}


def require(value, message):
    if not value: raise ValueError(message)


def rows(text):
    return list(csv.DictReader(io.StringIO(text), delimiter='\t'))


def load(path):
    return {p.name: p.read_text(errors='replace') for p in path.iterdir()
            if p.suffix in ('.rpt', '.tsv', '.xdc', '.txt')}


def structure(report):
    cells = {r['cell']: r for r in rows(report['cells.tsv'])}
    pins = {r['pin']: r for r in rows(report['pins.tsv'])}
    expected = {f'{name}_reg[{i}]' for name in CHAINS for i in range(3)}
    require({k for k, v in cells.items() if v['async_reg'].lower() in ('1', 'true')} == expected,
            'Exact 21 ASYNC_REG cells')

    def sources(pin, active=frozenset()):
        require(pin not in active, 'Acyclic driver cones')
        row = pins[pin]
        if row['direction'] == 'OUT':
            cell = pin.rsplit('/', 1)[0]
            if cells[cell]['ref'].startswith('FD') or cells[cell]['ref'] in ('GND', 'VCC'):
                return {pin}
            require(cells[cell]['ref'].startswith('LUT'), 'Reviewed combinational driver types')
            inputs = [p for p, r in pins.items() if p.rsplit('/', 1)[0] == cell and r['direction'] == 'IN']
            return set().union(*(sources(p, active | {pin}) for p in inputs))
        result = set(filter(None, row['input_ports'].split(',')))
        for driver in filter(None, row['drivers'].split(',')):
            result |= sources(driver, active | {pin})
        return result

    native = {'perst_n', 'native_link_up', 'native_axi_reset_n', 'native_ctl_reset_n'}
    controls = dict(cold_m={'cold_rst_n'}, cold_a={'cold_rst_n'}, healthy_m=native,
                    admit_a=native | {'cold_rst_n', 'armed_reg/Q', 'fault_reg/Q', 'cold_m_reg[2]/Q'},
                    stop_a={'fault_reg/Q'})
    endpoints = set()
    for name, (ref, clock, first) in CHAINS.items():
        for i in range(3):
            cell = f'{name}_reg[{i}]'
            require(cells[cell]['ref'] == ref and cells[cell]['init'] == "1'b0", 'Chain type and INIT: '+cell)
            require(pins[cell+'/C']['clocks'] == clock, 'Chain clock: '+cell)
            require(pins[cell+'/CE']['drivers'] == 'VCC/P', 'Chain enable: '+cell)
            driver = first if i == 0 else f'{name}_reg[{i-1}]/Q'
            require(pins[cell+'/D']['drivers'] == driver, 'Chain continuity: '+cell)
            if i:
                consumers = {p for p, r in pins.items() if r['direction'] == 'IN' and r['net'] == pins[driver]['net']}
                require(consumers == {cell+'/D'}, 'Exclusive interstage input fanout: '+cell)
                endpoints.add(cell+'/D')
            if ref == 'FDRE':
                require(pins[cell+'/R']['drivers'] == 'GND/G', 'Data chain reset: '+cell)
            else:
                pin = cell + ('/CLR' if ref == 'FDCE' else '/PRE')
                require(sources(pin) == controls[name], 'Asynchronous driver cone: '+cell)
    return cells, pins, endpoints


def review(baseline, scoped):
    bcells, bpins, endpoints = structure(baseline)
    scells, spins, sendpoints = structure(scoped)
    require(bcells == scells and bpins == spins and endpoints == sendpoints, 'Netlist structure unchanged')
    require(sum(c['ref'].startswith('FD') for c in bcells.values()) == 36, '36 synthesized FF')
    for report in (baseline, scoped):
        require('No waivers found.' in report['waivers.rpt'], 'No waived CDC or methodology findings')
        counts = {m[0]: (m[1], int(m[2])) for m in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s', report['cdc.rpt'], re.M)}
        require(counts == {'CDC-3': ('Info', 2), 'CDC-9': ('Info', 2), 'CDC-10': ('Critical', 1)},
                'Retain all CDC findings, including unresolved CDC-10')
        for key, count in (('no_clock', 0), ('unconstrained_internal_endpoints', 0), ('no_input_delay', 5), ('no_output_delay', 8)):
            require(f'checking {key} ({count})' in report['check_timing.rpt'], 'Explicit timing coverage: '+key)
        require(len(re.findall(r'^LUTAR-1#\d+ Warning', report['methodology.rpt'], re.M)) == 2,
                'Retain both combinational asynchronous-control warnings')
    bpaths, spaths = rows(baseline['paths.tsv']), rows(scoped['paths.tsv'])
    for paths in (bpaths, spaths):
        require(max(Counter((r['source_clock'],r['destination_clock'],r['delay']) for r in paths).values()) < 1000,
                'Path export not truncated')
    intra = lambda paths: Counter(tuple(sorted(r.items())) for r in paths if r['source_clock'] == r['destination_clock'])
    require(intra(bpaths) == intra(spaths), 'All exported same-clock paths unchanged')
    stages = rows(scoped['stage_paths.tsv'])
    require(len(stages) == 28 and {(r['endpoint'],r['delay']) for r in stages} == {(e,d) for e in endpoints for d in ('max','min')},
            'All 14 stage connections checked for setup and hold')
    for r in stages:
        require(r['before_slack'] == r['after_slack'] and float(r['after_slack']) > 0, 'Interstage margins unchanged and positive')
        actual = [float(p['slack']) for p in bpaths if p['endpoint']==r['endpoint'] and p['delay']==r['delay']]
        require(actual and min(actual) == float(r['before_slack']), 'Interstage values match independent baseline export')
    commands = lambda report: [s.strip() for s in report['effective.xdc'].splitlines() if s.strip() and not s.lstrip().startswith('#')]
    before, after = commands(baseline), commands(scoped)
    base_commands = {s for s in before if s.startswith(('create_clock ', 'current_instance '))}
    require(set(before) == base_commands and len([s for s in before if s.startswith('create_clock ')])==2, 'Baseline has only two clocks')
    additions = [s for s in after if s not in base_commands]
    require(set(before).issubset(after) and len(additions)==14, 'Exactly fourteen new scoped exceptions')
    pairs = set(); maxima = {}
    select = r'\[get_pins -quiet -filter \{NAME == "([^"]+)"\}\]'
    for line in additions:
        false_path = re.fullmatch(r'set_false_path -from '+select+' -to '+select, line)
        max_delay = re.fullmatch(r'set_max_delay -datapath_only -from '+select+' -to '+select+r' ([0-9.]+)', line)
        require(false_path or max_delay, 'Exact source and endpoint pins in every exception')
        if false_path:
            pairs.add(false_path.groups())
        else:
            maxima[max_delay.groups()[:2]] = float(max_delay[3])
    expected = {(s, f'admit_a_reg[{i}]/CLR') for s in ('armed_reg/C','fault_reg/C','cold_m_reg[2]/C') for i in range(3)}
    expected |= {('fault_reg/C', f'stop_a_reg[{i}]/PRE') for i in range(3)}
    require(pairs == expected, 'Exactly twelve asynchronous source/control-pin pairs')
    require(maxima == {('request_reg/C','echo_a_reg[0]/D'):4.0, ('echo_a_reg[2]/C','echo_m_reg[0]/D'):10.0},
            'Only the two reviewed first-stage data bounds')
    require(not re.search(r'^TIMING-[67]#', scoped['methodology.rpt'], re.M), 'Unrelated clock phase checks removed')
    require('No ignored timing exceptions found.' in scoped['exceptions_ignored.rpt'], 'No ignored timing exceptions')
    return dict(status='GUARD_SCOPED_CONSTRAINT_AUDIT_WITH_OPEN_CDC_FINDINGS', synchronizer_chains=7,
                async_reg_cells=21, flip_flops=36, same_clock_paths_preserved=sum(intra(bpaths).values()),
                interstage_connections=14, interstage_setup_hold_checks=28,
                interstage_min_setup_ns=min(float(r['after_slack']) for r in stages if r['delay']=='max'),
                interstage_min_hold_ns=min(float(r['after_slack']) for r in stages if r['delay']=='min'),
                scoped_false_path_pairs=12, bounded_data_paths=2, cdc10_critical=1, lutar1_warnings=2,
                ports_without_input_delay=5, ports_without_output_delay=8, physical_timing_closed=False)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline',type=Path);ap.add_argument('scoped',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();result=review(load(a.baseline),load(a.scoped))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
