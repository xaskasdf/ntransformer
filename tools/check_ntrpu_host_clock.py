"""Independently reconstruct offers and acknowledgements across AMD CDC peers.

AXI-Lite may acknowledge the sender after the receiver consumes a transfer.
Ordering is therefore checked against the actual stable offer, not an assumed
global order of the two clock-domain handshakes. Behavioral FIFO timing is not
a measurement of metastability, silicon latency, or physical bandwidth.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def fields(line):
    return {k: v if k == 'ch' else int(v, 16 if k in {'addr', 'meta', 'data', 'strb'} else 10)
            for k, v in re.findall(r'(\w+)=([A-Z0-9a-f]+)', line)}


def review(text, profile):
    def need(ok, why):
        if not ok:
            raise ValueError(why)

    need(profile in (0, 1, 2), 'clock profile')
    need(not re.search(r'(?mi)^\s*(Fatal:|ERROR:|FAIL\b)', text), 'failure diagnostic')
    source_half, target_half, phase = [(2000, 2500, 700), (3100, 2300, 900), (1700, 2900, 300)][profile]
    groups = {}
    active = {0: None, 1: None}
    next_case = {0: 1, 1: 1}
    bank_passes = []
    suite = []
    for line in text.splitlines():
        if line.startswith('NTRPU_HOST_CLOCK_PASS '):
            need(all(x is None for x in active.values()) and len(bank_passes) == 2, 'suite order')
            suite.append(fields(line))
            continue
        if not line.startswith('NTRPU_CDC_'):
            continue
        kind = line.split()[0].removeprefix('NTRPU_CDC_')
        row = fields(line)
        lite = row.get('lite')
        need(lite in (0, 1) and not suite, 'interface identity/terminal ordering')
        if kind == 'PASS':
            need(active[lite] is None and next_case[lite] == 10, 'bank completeness')
            bank_passes.append(row)
            continue
        ident = row.get('case')
        if kind == 'CASE':
            need(active[lite] is None and ident == next_case[lite], 'case order')
            need(row == dict(lite=lite, case=ident, abort=ident//2 if ident % 2 == 0 else 0,
                             profile=profile, source_half_ps=source_half, target_half_ps=target_half,
                             phase_ps=phase, tx=12, time_ps=row.get('time_ps')), 'case contract')
            groups[lite, ident] = dict(start=row, events=[], end=None, stalls=None, stop=None)
            active[lite] = ident
        elif kind in {'DONE', 'ABORT'}:
            need(active[lite] == ident, 'terminal case identity')
            group = groups[lite, ident]
            need((kind == 'ABORT') == bool(group['start']['abort']), 'completion kind')
            group['end'] = row
            active[lite] = None
            next_case[lite] += 1
        elif kind in {'STALLS', 'STOP'}:
            need((lite, ident) in groups and groups[lite, ident]['end'] is not None, 'trailer after case')
            key = 'stalls' if kind == 'STALLS' else 'stop'
            need(groups[lite, ident][key] is None, 'single trailer')
            groups[lite, ident][key] = row
        elif kind in {'ADDR', 'DATA'}:
            need(active[lite] == ident, 'traffic in active case')
            groups[lite, ident]['events'].append((kind, row))
        else:
            raise ValueError('unknown event '+kind)
    need(suite == [dict(banks=18, resets=8)], 'single suite PASS')
    need(sorted(bank_passes, key=lambda r: r['lite']) ==
         [dict(lite=l, profile=profile, cases=9, resets=4) for l in (0, 1)], 'two bank PASS markers')
    results = []
    for (lite, ident), group in sorted(groups.items()):
        byte_count = 4 if lite else 64
        length = lambda tx: 1 if lite else (1, 64, 2, 16, 3, 4)[tx % 6]
        tag = lambda tx: 0 if lite else (tx+ident*3) % 16
        pairs = [(tx, beat) for tx in range(12) for beat in range(length(tx))]
        start, end = group['start'], group['end']
        need(end is not None and end['time_ps'] > start['time_ps'], 'positive case duration')
        channels = defaultdict(list)
        for kind, row in group['events']:
            channels[row['ch']].append(row)
            need(start['time_ps'] < row['time_ps'] <= end['time_ps'], 'event within measured window')
            if row['ch'].startswith('M'):
                need((row['time_ps']-phase-target_half) % (2*target_half) == 0, 'target rising edge')
            else:
                residue = row['time_ps'] % (2*source_half)
                need(residue == source_half if ident < 9 else residue in (0, source_half), 'source rising edge')
        need(set(channels) <= {'SAWO', 'SAW', 'MAW', 'SARO', 'SAR', 'MAR', 'SWO', 'SW', 'MW', 'MBO', 'MB', 'SB', 'MRO', 'MR', 'SR'}, 'channel names')
        for ch, rows in channels.items():
            need(all(b['time_ps'] > a['time_ps'] for a, b in zip(rows, rows[1:])), 'one transfer per edge '+ch)
            address = ch in {'SAWO', 'SAW', 'MAW', 'SARO', 'SAR', 'MAR'}
            if address:
                need([r['tx'] for r in rows] == list(range(len(rows))) and len(rows) <= 12, 'address transaction order')
                for row in rows:
                    tx = row['tx']
                    a = (0x80000000 if lite else 0x8000000000)+ident*0x100000+int('AW' in ch)*0x40000+tx*4096+(4096-byte_count if tx % 6 == 0 else 0)
                    meta = tx % 8 if lite else (((((((((length(tx)-1) << 3 | 6) << 2 | 1) << 4 | tag(tx)) << 4 | tx % 16) << 3 | tx % 8) << 1) << 4 | (tx ^ 5) % 16) << 4 | (tx ^ 10) % 16)
                    need(row['addr'] == a and row['meta'] == meta, 'raw address/sidebands '+ch)
                    need(a % byte_count == 0 and a % 4096+length(tx)*byte_count <= 4096, 'aligned page-contained burst')
            else:
                is_b = ch in {'MBO', 'MB', 'SB'}
                expected = [(tx, 0) for tx in range(12)] if is_b else pairs
                need([(r['tx'], r['beat']) for r in rows] == expected[:len(rows)] and len(rows) <= len(expected), 'data order '+ch)
                for row in rows:
                    tx, beat = row['tx'], row['beat']
                    write = ch in {'SWO', 'SW', 'MW'}
                    data = 0 if is_b else int.from_bytes(bytes((ident*53+tx*17+beat*7+lane*11+int(write)*31) % 256 for lane in range(byte_count)), 'little')
                    mask_kind = (tx+beat) % 4
                    mask = sum(1 << lane for lane in range(byte_count) if mask_kind == 0 or mask_kind == 1 and lane % 2 == 0 or mask_kind == 2 and lane % 2 == 1) if write else 0
                    need(row['data'] == data and row['strb'] == mask, 'data/epoch/strobe '+ch)
                    need(row['last'] == int(is_b or beat == length(tx)-1), 'last '+ch)
                    need(row['tag'] == (0 if write else tag(tx)) and row['resp'] == (0 if write else (0, 2, 3)[(tx+beat) % 3]), 'ID/response '+ch)
        # An offer is stable until acknowledgement. Acknowledgement may return
        # after consumption on the other domain, especially for AXI4-Lite.
        for offer, endpoints in [('SAWO', ('SAW', 'MAW')), ('SARO', ('SAR', 'MAR')),
                                 ('SWO', ('SW', 'MW')), ('MBO', ('MB', 'SB')), ('MRO', ('MR', 'SR'))]:
            for endpoint in endpoints:
                need(len(channels[endpoint]) <= len(channels[offer]), 'no unoffered transfer '+endpoint)
                for offered, consumed in zip(channels[offer], channels[endpoint]):
                    need(offered['time_ps'] <= consumed['time_ps'], 'offer precedes confirmation '+endpoint)
                    need({k: v for k, v in offered.items() if k not in {'ch', 'time_ps'}} ==
                         {k: v for k, v in consumed.items() if k not in {'ch', 'time_ps'}}, 'offer payload preserved '+endpoint)
        for row in channels['MBO']:
            tx = row['tx']
            need(tx < len(channels['MAW']), 'B requires accepted AW')
            writes = [r for r in channels['MW'] if r['tx'] == tx]
            need(len(writes) == length(tx) and writes[-1]['time_ps'] < row['time_ps'] and
                 channels['MAW'][tx]['time_ps'] < row['time_ps'], 'B after destination address and full write')
        for row in channels['MRO']:
            need(row['tx'] < len(channels['MAR']) and channels['MAR'][row['tx']]['time_ps'] < row['time_ps'], 'R after accepted AR')
        mode = start['abort']
        if not mode:
            need(all(len(channels[ch]) == count for ch, count in
                 [('SAWO', 12), ('SAW', 12), ('MAW', 12), ('SARO', 12), ('SAR', 12), ('MAR', 12),
                  ('SWO', len(pairs)), ('SW', len(pairs)), ('MW', len(pairs)), ('MRO', len(pairs)),
                  ('MR', len(pairs)), ('SR', len(pairs)), ('MBO', 12), ('MB', 12), ('SB', 12)]), 'complete drained traffic')
            need(end['wbeats'] == end['rbeats'] == len(pairs) and end['sc'] > 0 and end['mc'] > 0, 'summary counts')
            need(end['time_ps']-start['time_ps'] == end['sc']*2*source_half, 'source cycles independently measured')
            stalls = group['stalls']
            need(stalls is not None and group['stop'] is None, 'complete case trailers')
            for a, b in [('saw', 'maw'), ('sw', 'mw'), ('sb', 'mb'), ('sar', 'mar'), ('sr', 'mr')]:
                need(stalls[a] >= 0 and stalls[b] >= 0 and stalls[a]+stalls[b] > 0, 'backpressure on '+a)
        else:
            need(end['mode'] == mode and end['stopped'] == int(mode == 4), 'abort mode')
            for field, ch in [('saw', 'SAW'), ('sar', 'SAR'), ('sw', 'SW'), ('sb', 'SB'), ('sr', 'SR'),
                              ('maw', 'MAW'), ('mar', 'MAR'), ('mw', 'MW'), ('mb', 'MB'), ('mr', 'MR')]:
                need(end[field] == len(channels[ch]), 'aborted prefix count '+ch)
            if mode == 1:
                need(channels['SARO'] and not channels['MAR'], 'AR stalled reset')
            elif mode == 2:
                need(channels['MRO'] and not channels['SR'], 'R stalled reset')
            elif mode == 3:
                need(len(channels['MW']) == 1 and len(channels['SWO']) > 1, 'partial W reset')
            else:
                need(channels['MBO'] and not channels['SB'], 'B stalled reset')
                stop = group['stop']
                need(stop is not None and stop['source_edges'] == 0 and stop['target_falling_edges'] == 32, 'clock stop observed')
                need(stop['begin_ps'] >= end['time_ps'] and 31*2*target_half <= stop['end_ps']-stop['begin_ps'] <= 32*2*target_half, 'stopped clock interval')
        backward = sum(dst['time_ps'] < src['time_ps'] for a, b in [('SAW', 'MAW'), ('SAR', 'MAR'), ('SW', 'MW'), ('MB', 'SB'), ('MR', 'SR')]
                       for src, dst in zip(channels[a], channels[b]))
        results.append(dict(lite=lite, case=ident, abort=mode, elapsed_ps=end['time_ps']-start['time_ps'],
                            source_cycles=end.get('sc'), target_cycles=end.get('mc'),
                            source_write_beats=len(channels['SW']), destination_write_beats=len(channels['MW']),
                            source_read_beats=len(channels['SR']), target_read_beats=len(channels['MR']),
                            receiver_before_sender_ack=backward))
    return dict(status='PASS', profile=profile, cases=results, resets=8,
                scope='Generated AMD AXI4/AXI4-Lite converters with simulated peers and coordinated reset. '+
                      'Behavioral FIFO omits synchronization delays; no physical latency, MTBF, XDMA/FFN integration or DDR qualification.')


def mutations(text):
    def change(prefix, field, hexadecimal=False):
        return re.sub(r'(^'+prefix+r'[^\n]*\b'+field+r'=)([0-9a-f]+)',
                      lambda m: m[1]+(format(int(m[2], 16)^1, 'x') if hexadecimal else str(int(m[2])+1)), text, count=1, flags=re.M)
    return dict(fatal=text+'\nFatal: injected error\n', missing_pass=text.replace('NTRPU_HOST_CLOCK_PASS ', 'REMOVED_PASS ', 1),
                address=change('NTRPU_CDC_ADDR[^\n]*ch=MAW', 'addr', True),
                sideband=change('NTRPU_CDC_ADDR[^\n]*ch=MAW', 'meta', True),
                data=change('NTRPU_CDC_DATA[^\n]*ch=SR', 'data', True),
                mask=change('NTRPU_CDC_DATA[^\n]*ch=MW', 'strb', True),
                response=change('NTRPU_CDC_DATA[^\n]*ch=SB', 'resp'),
                beat=change('NTRPU_CDC_DATA[^\n]*ch=SR', 'beat'),
                edge=change('NTRPU_CDC_DATA[^\n]*ch=MW', 'time_ps'),
                reset=change('NTRPU_CDC_ABORT', 'mode'),
                clock_stopped=change('NTRPU_CDC_STOP', 'source_edges'),
                cycles=change('NTRPU_CDC_DONE', 'sc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--profile', type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    print(json.dumps(review(args.log.read_text(encoding='utf-8'), args.profile), indent=2))
