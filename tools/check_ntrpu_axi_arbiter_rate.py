"""Measure registered AXI data flow from per-beat, per-interface cycle traces."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def fields(line):
    return {key: value if key == 'channel' else int(value, 16 if key in {'data', 'strb'} else 10)
            for key, value in re.findall(r'(\w+)=([A-Z0-9a-f]+)', line)}


def review(text, interval=1):
    def need(ok, why):
        if not ok:
            raise ValueError(why)

    need(interval in (1, 2), 'supported reference interval')
    need(not re.search(r'(?im)^\s*(Fatal:|ERROR:|FAIL\b)', text), 'failure diagnostic')
    cases = []
    done = []
    beats = []
    passes = []
    active = None
    for line in text.splitlines():
        if not line.startswith('NTRPU_RATE_'):
            continue
        kind = line.split()[0]
        row = fields(line)
        if kind == 'NTRPU_RATE_CASE':
            ident = len(cases)+1
            need(active is None and not passes and row == dict(id=ident, owner=(ident-1)//2,
                 pause=(ident-1) % 2, beats=64, interval=interval), 'case identity/mode/order')
            active = ident
            cases.append(row)
        elif kind == 'NTRPU_RATE_BEAT':
            need(row.get('id') == active and not passes, 'beat outside active case')
            beats.append(row)
        elif kind == 'NTRPU_RATE_DONE':
            need(row.get('id') == active and not passes, 'completion order')
            active = None
            done.append(row)
        elif kind == 'NTRPU_RATE_PASS':
            need(active is None, 'PASS with active case')
            passes.append(row)
        else:
            raise ValueError('unknown trace event '+kind)
    need(len(cases) == len(done) == 4 and len(beats) == 1024, 'complete coverage')
    need(passes == [dict(cases=4, interval=interval, beats=1024)], 'single terminal PASS')
    results = []
    for case, summary in zip(cases, done):
        ident, owner = case['id'], case['owner']
        by_channel = {name: [b for b in beats if b['id'] == ident and b.get('channel') == name]
                      for name in ('SW', 'MW', 'MR', 'SR')}
        need(sum(map(len, by_channel.values())) == 256, 'channel names')
        for channel, rows in by_channel.items():
            write = channel.endswith('W')
            need([b.get('beat') for b in rows] == list(range(64)), 'beat count/order '+channel)
            cycles = [b.get('cycle', 0) for b in rows]
            need(min(cycles) > 0 and all(b > a for a, b in zip(cycles, cycles[1:])), 'cycle order '+channel)
            for beat, row in enumerate(rows):
                payload = bytes((owner*97+beat*7+lane*11+int(write)*31) & 255 for lane in range(64))
                need(row.get('data') == int.from_bytes(payload, 'little') and row.get('last') == int(beat == 63), 'payload/last '+channel)
                if write:
                    mask = ((1 << 64)-1, 0x5555555555555555, 0xaaaaaaaaaaaaaaaa)[beat % 3]
                    need(row.get('strb') == mask, 'byte enables '+channel)
                else:
                    need(row.get('resp') == (2, 3, 0)[beat % 3], 'read response '+channel)
        result = dict(id=ident, owner=owner, paused=bool(case['pause']))
        for prefix, incoming, outgoing in [('w', 'SW', 'MW'), ('r', 'MR', 'SR')]:
            source = by_channel[incoming]
            target = by_channel[outgoing]
            need(all(b['cycle'] > a['cycle'] for a, b in zip(source, target)), 'registered forwarding '+prefix)
            deltas = Counter()
            for row in source:
                deltas[row['cycle']] += 1
            for row in target:
                deltas[row['cycle']] -= 1
            credit = peak = 0
            for cycle in sorted(deltas):
                credit += deltas[cycle]
                peak = max(peak, credit)
                need(0 <= credit <= (2 if interval == 1 else 1), 'queue credit '+prefix)
            need(credit == 0, 'data drained '+prefix)
            overlap = len({b['cycle'] for b in source} & {b['cycle'] for b in target})
            span = target[-1]['cycle']-target[0]['cycle']+1
            need(summary.get(prefix+'_span') == span and summary.get(prefix+'_overlap') == overlap and
                 summary.get('max_'+prefix) == peak, 'measured summary '+prefix)
            if not case['pause']:
                need(all(b['cycle']-a['cycle'] == interval for a, b in zip(target, target[1:])), 'sustained interval '+prefix)
                need(span == 63*interval+1, 'sustained window '+prefix)
            elif interval == 1:
                need(peak == 2 and overlap > 0, 'second buffer slot and concurrent flow '+prefix)
            result.update({prefix+'_span_cycles': span, prefix+'_overlap_cycles': overlap, prefix+'_peak_queue': peak})
        need(summary.get('cycles', 0) >= max(result['r_span_cycles'], result['w_span_cycles']), 'case elapsed bound')
        results.append(result)
    return dict(status='PASS', interval=interval, cases=results, handshake_events=1024,
                scope='64-beat full-duplex bursts, both initiators separately, no-stall and paused cases. '+
                      'AXI protocol peers at a simulated 5 ns clock; no physical DDR/PCIe bandwidth claim.')


def mutations(text):
    def change(prefix, field, hexadecimal=False):
        return re.sub(r'(^'+prefix+r'[^\n]*\b'+field+r'=)([0-9a-f]+)',
                      lambda m: m[1]+(format(int(m[2], 16)^1, 'x') if hexadecimal else str(int(m[2])+1)),
                      text, count=1, flags=re.M)
    return {
        'fatal': text+'\nFatal: injected evidence fault\n',
        'missing_pass': text.replace('NTRPU_RATE_PASS ', 'REMOVED_PASS ', 1),
        'owner': change('NTRPU_RATE_CASE', 'owner'),
        'data': change('NTRPU_RATE_BEAT[^\n]*channel=MW', 'data', True),
        'strobe': change('NTRPU_RATE_BEAT[^\n]*channel=MW', 'strb', True),
        'response': change('NTRPU_RATE_BEAT[^\n]*channel=SR', 'resp'),
        'cycle': change('NTRPU_RATE_BEAT[^\n]*channel=MW', 'cycle'),
        'interval': change('NTRPU_RATE_CASE', 'interval'),
        'credit': change('NTRPU_RATE_DONE', 'max_r'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--interval', type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    print(json.dumps(review(args.log.read_text(encoding='utf-8'), args.interval), indent=2))
