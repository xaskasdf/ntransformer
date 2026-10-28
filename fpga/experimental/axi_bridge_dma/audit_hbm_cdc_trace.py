"""Check watchdog latency and retained ownership from top-level VCD transitions."""
import argparse
import json
import re
from pathlib import Path
from review_hbm_source import require


def trace(path, case):
    return trace_text(path.read_text(),case)


def trace_text(text, case):
    scale = re.search(r'\$timescale\s+(\d+)\s*(\w+)\s+\$end', text)
    require(scale is not None, 'Missing trace timescale')
    factor = int(scale[1]) * {'ps': 1, 'ns': 1000, 'fs': .001}[scale[2]]
    names = ('clk', 'reader_clk', 'mgmt_clk', 'reader_run', 'reader_management_fault',
             'reader_fault', 'source_fault', 'owned', 'source_owned', 'cold_rst_n', 'reader_started')
    changes = {name: [] for name in names}
    symbols = {}
    depth = 0
    for line in text.splitlines():
        if line.startswith('$scope '):
            depth += 1
        elif line.startswith('$upscope '):
            depth -= 1
        elif line.startswith('$enddefinitions'):
            break
        elif depth == 1:
            declaration = re.match(r'\$var\s+\w+\s+(\d+)\s+(\S+)\s+(\w+)', line)
            if declaration and declaration[3] in changes:
                width, symbol, name = declaration.groups()
                require(int(width) == 1 and symbol not in symbols, 'Unexpected trace alias')
                symbols[symbol] = name
    require(set(symbols.values()) == set(names), 'Missing public trace signals')
    time = 0
    for line in text.split('$enddefinitions $end')[-1].splitlines():
        line = line.strip()
        if line.startswith('#'):
            time = int(line[1:]) * factor
        elif line and line[0] in '01xzXZ' and line[1:] in symbols:
            changes[symbols[line[1:]]].append((time, line[0].lower()))
    stops = [t for t, value in changes['reader_run'] if value == '0' and t > 0]
    require(len(stops) == 1, 'Missing unique clock halt')
    stopped = stops[0]
    resumed = next(t for t, value in changes['reader_run'] if value == '1' and t > stopped)
    for name, period in [('clk', 4000), ('reader_clk', 5000), ('mgmt_clk', 10000)]:
        edges = [t for t, value in changes[name] if value == '1' and t < stopped]
        require(len(edges) >= 20 and all(b-a == period for a, b in zip(edges, edges[1:])), 'Wrong clock profile: ' + name)
    require(not any(stopped < t < resumed for t, _ in changes['reader_clk']), 'Reader clock did not stop')

    def at(name, when):
        return next(value for t, value in reversed(changes[name]) if t <= when)

    require(at('reader_clk', stopped) == str(case-22), 'Wrong stopped clock level')
    delays = {}
    for name, bound in [('reader_management_fault', 800000), ('reader_fault', 820000), ('source_fault', 820000)]:
        asserted = next(t for t, value in changes[name] if value == '1' and t >= stopped)
        require(0 < asserted-stopped <= bound, 'Fault exceeded watchdog bound: ' + name)
        require(all(value == '1' for t, value in changes[name] if t >= asserted), 'Fault was cleared: ' + name)
        delays[name + '_ns'] = (asserted-stopped)/1000
    for name in ('owned', 'source_owned', 'cold_rst_n'):
        require(at(name, stopped) == '1' and all(value == '1' for t, value in changes[name] if t >= stopped), 'Lost retained state: ' + name)
    require(at('reader_started', resumed) == '0' and all(value == '0' for t, value in changes['reader_started'] if t >= resumed), 'Restart reopens failed epoch')
    return dict(case=case,halt_ns=stopped/1000,resume_ns=resumed/1000,**delays)


def audit(root):
    checks = [trace(root/f'case-{case:02}-seed-{seed}/hbm_cdc.vcd',case) | {'seed':seed}
              for seed in range(2) for case in (22,23)]
    return dict(status='HBM_CDC_WATCHDOG_TRACE_PASS',checks=checks,clock_profile_mhz=[250,200,100])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',type=Path)
    print(json.dumps(audit(parser.parse_args().phase),indent=2))
