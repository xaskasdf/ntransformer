"""Verify the frozen 2026.1 structural review; never return CDC/board signoff."""
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
        expected[prefix + 'fault_s1_reg'] = (
            f'l{lane}_fault', prefix + 'dma_fault_reg', prefix + 'fault_s2_reg', 'dma_clk', 'reader_clk', '0')
        for bit in range(2):
            expected[prefix + f'sent_s1_reg[{bit}]'] = (
                f'l{lane}_sent', prefix + f'sent_generation_reg[{bit}]', prefix + f'sent_s2_reg[{bit}]',
                'dma_clk', 'reader_clk', '0')
    expected['ready_s1_reg'] = ('ready', 'dr_reg[2]', 'ready_s2_reg', 'dma_clk', 'reader_clk', '0')
    actual = {r['first']: tuple(r[k] for k in ('group', 'source', 'second', 'source_clock', 'dest_clock', 'gray'))
              for r in rows}
    require(len(rows) == len(actual) == 43 and actual == expected, 'Missing, duplicated or incorrect synchronizer binding')
    matches = re.findall(r'^\d+\s+Max Delay Datapath Only\s+max_dpo=(\d+)\s+-\s+(\d+) cells\s+(\d+) cells\s+(\d+)\s+100\.00\s+33\.33\s*$', coverage, re.M)
    require(len(matches) == 17, 'Expected 17 covered max-delay constraints')
    require(sum(int(m[3]) for m in matches) == 43, 'Expected 43 max-delay endpoints')
    require(sum(m[0] == '5' for m in matches) == 6 and sum(m[0] == '4' for m in matches) == 11,
            'Wrong source-period budgets')
    require(all(a == b == c for _, a, b, c in matches), 'Unexpected endpoint cardinality')
    require('No ignored timing exceptions found.' in ignored, 'Ignored constraints exist')
    require(re.findall(r'^Id: (\d+)$', skew, re.M) == [str(i) for i in range(1, 13)], 'Expected 12 skew buses')
    require(len(re.findall(r'^Requirement: 4\.000ns$', skew, re.M)) == 12, 'Wrong bus-skew budget')
    require(len(re.findall(r'^Endpoints: 3$', skew, re.M)) == 12, 'Wrong skew endpoint count')
    counts = {code: int(count) for code, count in re.findall(r'^(CDC-\d+)\s+(?:Critical|Warning|Info)\s+(\d+)\s', cdc, re.M)}
    require(counts == {'CDC-1': 2358, 'CDC-3': 5, 'CDC-6': 14, 'CDC-10': 2, 'CDC-15': 226},
            'CDC findings changed; a new review is required')
    require(not re.search(r'^\s*(?:set_false_path|set_clock_groups|create_waiver)\b', effective, re.M),
            'Unexpected exclusion or waiver')
    require(len(re.findall(r'^set_max_delay\b', effective, re.M)) == 17, 'Effective max-delay count')
    require(len(re.findall(r'^set_bus_skew\b', effective, re.M)) == 12, 'Effective bus-skew count')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('synth', type=Path)
    parser.add_argument('controls', type=Path)
    args = parser.parse_args()
    synth = verify_stage(args.synth)
    controls = verify_stage(args.controls)
    require(digest(args.synth / 'output/reports/synth.dcp') == digest(args.controls / 'repo/evidence/synth.dcp'),
            'Wrong donor checkpoint')
    reports = args.controls / 'output/reports'
    names = ('bindings.tsv', 'exceptions.rpt', 'ignored_exceptions.rpt', 'bus_skew_estimated.rpt', 'cdc.rpt', 'effective_constraints.xdc')
    texts = [(reports / name).read_text() for name in names]
    counts = check_reports(*texts)
    require('NTRPU_BRIDGE_CONTROLS_REVIEW_DONE' in (args.controls / 'output/console.log').read_text(),
            'Missing completed topology review')
    # Independent evidence corruptions: omissions, clock attribution and masked CDC.
    corruptions = []
    changed = texts.copy(); changed[0] = '\n'.join(changed[0].splitlines()[:-1]) + '\n'; corruptions.append(changed)
    changed = texts.copy(); changed[0] = changed[0].replace('reader_clk\tdma_clk', 'dma_clk\treader_clk', 1); corruptions.append(changed)
    changed = texts.copy(); changed[1] = changed[1].replace('max_dpo=5', 'max_dpo=6', 1); corruptions.append(changed)
    changed = texts.copy(); changed[5] += '\nset_clock_groups -asynchronous -group reader_clk -group dma_clk\n'; corruptions.append(changed)
    changed = texts.copy(); changed[4] = re.sub(r'(CDC-10\s+Critical\s+)2', r'\g<1>0', changed[4], count=1); corruptions.append(changed)
    for changed in corruptions:
        try:
            check_reports(*changed)
        except ValueError:
            continue
        raise ValueError('A corrupted report passed')
    print(json.dumps(dict(status='STRUCTURAL_REVIEW_VERIFIED_NOT_QUALIFIED', max_delay_groups=17,
                         first_stage_bits=43, gray_buses=12, ignored_exceptions=0, cdc=counts,
                         negative_controls=len(corruptions), synth_seconds=synth['seconds'],
                         controls_seconds=controls['seconds'], physical_cdc_qualified=False,
                         routed=False, board_qualified=False), indent=2))


if __name__ == '__main__':
    main()
