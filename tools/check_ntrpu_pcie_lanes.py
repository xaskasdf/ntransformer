"""Cross-check synthesized lane connectivity and Vivado report_io ball functions."""
import argparse
import csv
import hashlib
import json
import pathlib
import re


def review(trace_text, io_text):
    rows = list(csv.DictReader(trace_text.splitlines(), delimiter='\t'))
    if len(rows) != 64:
        raise ValueError('Expected 64 scalar lane ports')
    io = {}
    for line in io_text.splitlines():
        fields = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(fields) > 3 and re.fullmatch(r'pci_exp_(?:rx|tx)[pn]\[\d+\]', fields[1]):
            if fields[1] in io:
                raise ValueError('Duplicate report_io lane port')
            io[fields[1]] = (fields[0], fields[3])
    expected = {f'pci_exp_{side}{polarity}[{lane}]'
                for side in ('rx', 'tx') for polarity in ('p', 'n') for lane in range(16)}
    if set(io) != expected or {r['port'] for r in rows} != expected:
        raise ValueError('Missing/extra lane ports')
    balls = set()
    result = []
    for row in rows:
        match = re.fullmatch(r'pci_exp_(rx|tx)(p|n)\[(\d+)\]', row['port'])
        side, polarity, lane = match.groups()
        lane = int(lane)
        site_y = 15-lane
        bank = 224+site_y//4
        signal = side.upper()+polarity.upper()
        if row['gt_site'] != f'GTYE4_CHANNEL_X1Y{site_y}' or row['bank'] != str(bank):
            raise ValueError('Unexpected lane site/bank: '+row['port'])
        if not row['primitive_pin'].endswith('/GTY'+signal):
            raise ValueError('Wrong primitive pin: '+row['port'])
        ball, function = io[row['port']]
        if function != f'MGTY{signal}{site_y%4}_{bank}':
            raise ValueError('Package pin direction/polarity/channel mismatch: '+row['port'])
        if ball not in row['package_candidates'].split() or ball in balls:
            raise ValueError('Duplicate/noncandidate ball: '+row['port'])
        balls.add(ball)
        result.append(dict(row, package_ball=ball, pin_function=function))
    return {'status': 'PASS', 'lanes': 16, 'scalar_ports': 64, 'unique_balls': len(balls),
            'mapping': result,
            'scope': 'Synthesized serial endpoint connectivity and ball-function cross-check; no placement/routing/board approval'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--json', required=True, type=pathlib.Path)
    args = parser.parse_args()
    paths = [args.output/name for name in ('lane_trace.tsv', 'io.rpt')]
    result = review(*(p.read_text() for p in paths))
    result['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with args.json.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('NTRPU_LANE_BALL_REVIEW_PASS lanes=16 scalar_ports=64 unique_balls=64')
