"""Verify revision-2 registered-capture evidence; never return CDC/board signoff."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_stage(stage):
    out = stage / 'output'
    for name, sha in json.loads((out / 'FILES.json').read_text()).items():
        require(digest(out / name) == sha, 'Transferred evidence changed: ' + name)
    result = json.loads((out / 'result.json').read_text())
    require(result['status'] == 'PENDING_REVIEW' and result['returncode'] == 0,
            'Vivado stage did not complete')
    require(result['sources'] == result['sources_after'], 'Sources changed during run')
    for name, sha in result['sources'].items():
        require(digest(stage / 'repo' / name) == sha, 'Frozen source changed: ' + name)
    return result


def check_reports(binding_text, coverage, ignored, skew, cdc, effective):
    rows = list(csv.DictReader(io.StringIO(binding_text), delimiter='\t'))
    expected = {}
    for lane in range(2):
        for fifo in ('commands', 'payloads', 'completions'):
            for source, dest, second in (('wg', 'ws1', 'ws2'), ('rg', 'rs1', 'rs2')):
                group = f'l{lane}_{fifo}_{source}'
                prefix = f'lanes[{lane}].bridge/{fifo}/'
                forward = (fifo != 'completions') == (source == 'wg')
                clocks = ('reader_clk', 'dma_clk') if forward else ('dma_clk', 'reader_clk')
                for bit in range(3):
                    expected[prefix + f'{dest}_reg[{bit}]'] = (
                        group, prefix + f'{source}_reg[{bit}]', prefix + f'{second}_reg[{bit}]', *clocks, '1')
        prefix = f'lanes[{lane}].bridge/'
        expected[prefix + 'stop_d1_reg'] = (
            f'l{lane}_stop', prefix + 'stop_source_reg', prefix + 'stop_d2_reg', 'reader_clk', 'dma_clk', '0')
        expected[prefix + 'fault_s1_reg'] = (
            f'l{lane}_fault', prefix + 'dma_fault_reg', prefix + 'fault_s2_reg', 'dma_clk', 'reader_clk', '0')
        for bit in range(2):
            expected[prefix + f'sent_s1_reg[{bit}]'] = (
                f'l{lane}_sent', prefix + f'sent_generation_reg[{bit}]', prefix + f'sent_s2_reg[{bit}]',
                'dma_clk', 'reader_clk', '0')
    expected['ready_s1_reg'] = ('ready', 'dr_reg[2]', 'ready_s2_reg', 'dma_clk', 'reader_clk', '0')
    actual = {r['first']: tuple(r[k] for k in ('group', 'source', 'second', 'source_clock', 'dest_clock', 'gray'))
              for r in rows}
    require(len(rows) == len(actual) == 45 and actual == expected, 'Missing, duplicated or incorrect synchronizer binding')
    matches = re.findall(r'^\d+\s+Max Delay Datapath Only\s+max_dpo=(\d+)\s+-\s+(\d+) cells\s+(\d+) cells\s+(\d+)\s+100\.00\s+33\.33\s*$', coverage, re.M)
    require(len(matches) == 19, 'Expected 19 covered max-delay constraints')
    require(sum(int(m[3]) for m in matches) == 45, 'Expected 45 max-delay endpoints')
    require(sum(m[0] == '5' for m in matches) == 8 and sum(m[0] == '4' for m in matches) == 11,
            'Wrong source-period budgets')
    require(all(a == b == c for _, a, b, c in matches), 'Unexpected endpoint cardinality')
    require('No ignored timing exceptions found.' in ignored, 'Ignored constraints exist')
    require(re.findall(r'^Id: (\d+)$', skew, re.M) == [str(i) for i in range(1, 13)], 'Expected 12 skew buses')
    require(len(re.findall(r'^Requirement: 4\.000ns$', skew, re.M)) == 12, 'Wrong bus-skew budget')
    require(len(re.findall(r'^Endpoints: 3$', skew, re.M)) == 12, 'Wrong skew endpoint count')
    counts = {code: int(count) for code, count in re.findall(r'^(CDC-\d+)\s+(?:Critical|Warning|Info)\s+(\d+)\s', cdc, re.M)}
    require(counts == {'CDC-3': 5, 'CDC-6': 14, 'CDC-15': 1318},
            'CDC findings changed; a new review is required')
    require(not re.search(r'^\s*(?:set_false_path|set_clock_groups|create_waiver)\b', effective, re.M),
            'Unexpected exclusion or waiver')
    require(len(re.findall(r'^set_max_delay\b', effective, re.M)) == 25, 'Effective max-delay count')
    require(len(re.findall(r'^set_bus_skew\b', effective, re.M)) == 12, 'Effective bus-skew count')
    return counts


def check_capture(bindings, paths, coverage):
    rows = list(csv.DictReader(io.StringIO(bindings), delimiter='\t'))
    require(len(rows) == 6, 'Expected six FIFO capture banks')
    expected = {}
    for lane in range(2):
        for fifo, width, unused in [('commands', 143, 25), ('payloads', 514, 78), ('completions', 2, 2)]:
            clocks = ('dma_clk', 'reader_clk') if fifo == 'completions' else ('reader_clk', 'dma_clk')
            expected[f'l{lane}_{fifo}'] = (width, width, *clocks, unused)
    actual = {r['fifo']: (int(r['ram_cells']), int(r['bits']), r['source_clock'], r['dest_clock'], int(r['unused_outputs'])) for r in rows}
    require(actual == expected, 'Wrong data bank clock, width or dangling-output accounting')
    records = list(csv.DictReader(io.StringIO(paths), delimiter='\t'))
    require(len(records) == 1318, 'Incomplete capture paths')
    endpoints = set()
    slacks = []
    for r in records:
        require(r['fifo'] in expected, 'Unknown capture bank')
        lane, fifo = r['fifo'].split('_', 1)
        prefix = f'lanes[{lane[1:]}].bridge/{fifo}/'
        require(r['startpoint'].startswith(prefix + 'memory_reg_') and r['startpoint'].endswith('/CLK'), 'Cross-bank or non-RAM source')
        m = re.fullmatch(re.escape(prefix) + r'output_data_reg\[(\d+)\]/D', r['endpoint'])
        require(m is not None and int(m[1]) < expected[r['fifo']][1], 'Wrong capture endpoint')
        require(r['endpoint'] not in endpoints, 'Duplicate capture endpoint')
        endpoints.add(r['endpoint']);slacks.append(float(r['slack']))
    require(min(slacks) >= 0, 'Estimated data max-delay violation')
    matches = re.findall(r'^\d+\s+Max Delay Datapath Only\s+max_dpo=(\d+)\s+-\s+(\d+) cells\s+(\d+) pins\s+(\d+)\s+100\.00\s+100\.00\s*$', coverage, re.M)
    wanted = sorted([(4, 143, 143, 143), (4, 514, 514, 514), (5, 2, 2, 2)] * 2)
    require(sorted(tuple(map(int, m)) for m in matches) == wanted, 'Wrong data max-delay coverage or budget')
    return min(slacks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('synth', type=Path)
    parser.add_argument('controls', type=Path)
    args = parser.parse_args()
    synth = verify_stage(args.synth)
    controls = verify_stage(args.controls)
    require(digest(args.synth / 'output/reports/synth.dcp') == digest(args.controls / 'repo/evidence/synth.dcp'),
            'Wrong donor checkpoint')
    console = (args.synth / 'output/console.log').read_text()
    require('Synth 8-7137' not in console and 'Synth 8-4767' not in console, 'RAM/reset synthesis warning remains')
    utilization = (args.synth / 'output/reports/utilization.rpt').read_text()
    row = next(line for line in utilization.splitlines() if '| ntrpu_hbm_dma_bridge ' in line)
    fields = [v.strip() for v in row.split('|')]
    require([int(v) for v in fields[3:8]] == [8671, 7903, 768, 0, 9268], 'Unexpected resource profile; review required')
    reports = args.controls / 'output/reports'
    names = ('bindings.tsv', 'exceptions.rpt', 'ignored_exceptions.rpt', 'bus_skew_estimated.rpt', 'cdc.rpt', 'effective_constraints.xdc')
    texts = [(reports / name).read_text() for name in names]
    counts = check_reports(*texts)
    capture_texts = [(reports / n).read_text() for n in ('data_bindings.tsv', 'data_paths.tsv', 'exceptions.rpt')]
    capture_slack = check_capture(*capture_texts)
    require('NTRPU_BRIDGE_REV2_CAPTURE_REVIEW_DONE' in (args.controls / 'output/console.log').read_text(),
            'Missing completed topology review')
    # Independent evidence corruptions: omissions, clock attribution and masked CDC.
    corruptions = []
    changed = texts.copy(); changed[0] = '\n'.join(changed[0].splitlines()[:-1]) + '\n'; corruptions.append(changed)
    changed = texts.copy(); changed[0] = changed[0].replace('reader_clk\tdma_clk', 'dma_clk\treader_clk', 1); corruptions.append(changed)
    changed = texts.copy(); changed[1] = changed[1].replace('max_dpo=5', 'max_dpo=6', 1); corruptions.append(changed)
    changed = texts.copy(); changed[5] += '\nset_clock_groups -asynchronous -group reader_clk -group dma_clk\n'; corruptions.append(changed)
    changed = texts.copy(); changed[4] += '\nCDC-1 Critical 1 unexpected crossing\n'; corruptions.append(changed)
    for changed in corruptions:
        try:
            check_reports(*changed)
        except ValueError:
            continue
        raise ValueError('A corrupted report passed')
    capture_corruptions = []
    changed = capture_texts.copy(); changed[1] = '\n'.join(changed[1].splitlines()[:-1]) + '\n'; capture_corruptions.append(changed)
    changed = capture_texts.copy(); changed[0] = changed[0].replace('reader_clk\tdma_clk', 'dma_clk\treader_clk', 1); capture_corruptions.append(changed)
    changed = capture_texts.copy(); changed[1] = changed[1].replace('/commands/memory_', '/payloads/memory_', 1); capture_corruptions.append(changed)
    for changed in capture_corruptions:
        try:
            check_capture(*changed)
        except ValueError:
            continue
        raise ValueError('Corrupted data capture evidence passed')
    print(json.dumps(dict(status='REV2_CAPTURE_REVIEW_VERIFIED_NOT_QUALIFIED', max_delay_groups=25,
                         first_stage_bits=45, gray_buses=12, ignored_exceptions=0, cdc=counts,
                         negative_controls=len(corruptions) + len(capture_corruptions), synth_seconds=synth['seconds'],
                         controls_seconds=controls['seconds'], physical_cdc_qualified=False,
                         routed=False, board_qualified=False, lut_total=8671, lut_logic=7903, lut_ram=768, flip_flops=9268, capture_bits=1318, capture_min_estimated_slack=capture_slack), indent=2))


if __name__ == '__main__':
    main()
