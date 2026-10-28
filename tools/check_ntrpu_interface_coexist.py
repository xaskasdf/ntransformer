"""Audit combined DDR/PCIe pin preservation and targeted voltage/DCI reports."""
import argparse
import csv
import hashlib
import json
import pathlib
import re


def check(condition, message):
    if not condition:
        raise ValueError(message)


def review(inventory, output, console):
    expected = {}
    original_io = {}
    inputs = []
    for interface, count in [('ddr', 154), ('pcie', 67)]:
        path = inventory / (interface + '_ports.tsv')
        inputs.append(path)
        rows = list(csv.DictReader(path.read_text().splitlines(), delimiter='\t'))
        check(len(rows) == count, 'Wrong inventory port count: ' + interface)
        for row in rows:
            name = interface + '_' + row['port']
            check(name not in expected, 'Duplicate inventory port')
            expected[name] = row
        io_path = inventory / (interface + '_io.rpt')
        inputs.append(io_path)
        for line in io_path.read_text().splitlines():
            fields = [v.strip() for v in line.strip().strip('|').split('|')]
            if len(fields) > 15 and interface + '_' + fields[1] in expected:
                name = interface + '_' + fields[1]
                check(name not in original_io, 'Duplicate original report port')
                original_io[name] = fields
    check(set(original_io) == set(expected), 'Original report port set differs')
    check(len({r['ball'] for r in expected.values()}) == 221, 'Ball collision')
    check(all(r['ball'] for r in expected.values()), 'Unassigned ball')
    check(expected['pcie_sys_rst_n']['standard'] == 'LVCMOS18', 'Unexpected original PERST')

    io_path = output / 'io.rpt'
    inputs.append(io_path)
    actual = {}
    for line in io_path.read_text().splitlines():
        fields = [v.strip() for v in line.strip().strip('|').split('|')]
        if len(fields) > 15 and fields[1].startswith(('ddr_', 'pcie_')):
            check(fields[1] not in actual, 'Duplicate report port')
            actual[fields[1]] = fields
    check(set(actual) == set(expected), 'Combined report port set differs')
    direction = {'IN': 'INPUT', 'OUT': 'OUTPUT', 'INOUT': 'BIDIR'}
    for name, row in expected.items():
        fields = actual[name]
        check(fields[0] == row['ball'], 'Ball changed: ' + name)
        check(fields[5] == direction[row['direction']], 'Direction changed: ' + name)
        standard = 'LVCMOS12' if name == 'pcie_sys_rst_n' else row['standard']
        check(fields[6] == standard, 'IOSTANDARD changed: ' + name)
        for key, index in [('drive', 8), ('slew', 10), ('pulltype', 15)]:
            # Invalid GT properties are empty in Tcl, although report_io
            # displays a default SLOW for TX. Compare those against the
            # original report, not a fabricated constraint.
            check(fields[index] == original_io[name][index], key + ' changed: ' + name)
            if row[key]:
                check(fields[index] == row[key], key + ' constraint changed: ' + name)
        for index in (2, 3, 4, 7, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21):
            check(fields[index] == original_io[name][index], 'IO report property changed: ' + name)
    check(actual['pcie_sys_rst_n'][0] == 'AM17', 'PERST ball is not AM17')
    check(actual['pcie_sys_rst_n'][7] == '65', 'PERST bank is not 65')

    counts = {}
    for name, required in [('mixed_voltage', 1), ('shared_1v2', 0), ('dci', 0)]:
        path = output / (name + '.rpt')
        inputs.append(path)
        matches = re.findall(r'Checks found:\s*(\d+)', path.read_text())
        check(matches == [str(required)], 'Unexpected targeted DRC: ' + name)
        counts[name] = required
    path = output / 'clocks.rpt'
    inputs.append(path)
    clock_text = path.read_text()
    for name, period, port in [('ntrpu_board_ref', '4.000', 'ddr_c0_sys_clk_p'),
                               ('sys_clk', '10.000', 'pcie_sys_clk_p')]:
        pattern = '^' + re.escape(name) + r'\s+' + re.escape(period) + r'\s+.*\{' + re.escape(port) + r'\}\s*$'
        check(re.search(pattern, clock_text, re.MULTILINE), 'Missing primary clock: ' + name)
    inputs.append(console)
    text = console.read_text()
    check(not re.search(r'^ERROR:', text, re.MULTILINE), 'Console contains ERROR')
    check(re.search(r'^NTRPU_INTERFACE_COEXISTENCE_TARGETED_PASS ports=221 mixed=1 shared=0 dci=0$',
                    text, re.MULTILINE), 'Missing actual completion marker')
    return {
        'status': 'TARGETED_PASS', 'ports': 221, 'unique_balls': 221,
        'targeted_drc_counts': counts,
        'only_io_change': 'pcie_sys_rst_n AM17 bank65 LVCMOS18 -> LVCMOS12',
        'scope': 'Preserved pins/properties and primary clocks; voltage/DCI only. Requires external PERST translation. No placement, timing, DIMM or board approval.',
        'files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('console', type=pathlib.Path)
    parser.add_argument('--json', required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = review(args.inventory, args.output, args.console)
    with args.json.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('NTRPU_COEXISTENCE_REVIEW_TARGETED_PASS ports=221 unique_balls=221')
