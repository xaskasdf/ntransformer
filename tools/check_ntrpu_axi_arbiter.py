"""Independently reconstruct both sides of the registered AXI arbiter's trace."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import re


def burst_length(tx):
    return (1, 2, 3, 4, 16, 64)[tx % 6]


def address(owner, tx, write):
    return (1 << 32) + (owner << 36) + (int(write) << 29) + tx*4096 + (4032 if tx % 6 == 0 else 0)


def data(owner, tx, beat, write):
    values = bytes((owner*71 + tx*13 + beat*17 + lane*3 + int(write)*29) & 255 for lane in range(64))
    return int.from_bytes(values, 'little')


def strobe(tx, beat):
    if tx % 4 == 0:
        return (1 << 64)-1
    if tx % 4 == 1 and beat == 0:
        return 0
    return 0x0123456789abcdef ^ (0x1111111111111111 * (beat % 16))


def response(tx):
    return {3: 2, 4: 3}.get(tx % 5, 0)


def review(text, seed, pipelined=False):
    def need(ok, why):
        if not ok:
            raise ValueError(why)

    need(not re.search(r'(?im)^\s*(Fatal:|ERROR:|FAIL\b)', text), 'tool/test failure')
    lines = [line for line in text.splitlines() if line.startswith('NTRPU_ARB_')]
    need(lines and lines[0] == f'NTRPU_ARB_SEED seed={seed}', 'seed')
    config = [line for line in lines if line.startswith('NTRPU_ARB_PIPELINED ')]
    need(config == (['NTRPU_ARB_PIPELINED value=1'] if pipelined else []), 'anticipated request mode')
    lines = [line for line in lines if not line.startswith('NTRPU_ARB_PIPELINED ')]
    summary = None
    banks = []
    totals = Counter()
    current = None
    for line in lines[1:]:
        event, _, raw = line.partition(' ')
        pairs = re.findall(r'(\w+)=([0-9a-f,]+)', raw)
        row = {k: ([int(n) for n in v.split(',')] if k == 'stalls' else
                   int(v, 16 if k in {'addr', 'data', 'strb'} else 10)) for k, v in pairs}
        event = event.removeprefix('NTRPU_ARB_')
        if event == 'BANK':
            need(current is None and summary is None, 'overlapping bank/after PASS')
            ident = len(banks)+1
            need(row == dict(id=ident, count=24 if ident == 1 else 4, abort=ident//2 if ident % 2 == 0 else 0), 'bank contract')
            current = row
            active_r = active_w = None
            source_w = deque()
            target_r = deque()
            memory_b = None
            r_source = [0, 0]
            w_source = [0, 0]
            r_grants = []
            w_grants = []
            counts = Counter()
            continue
        if event == 'PASS':
            need(current is None and summary is None and len(banks) == 9, 'single terminal PASS after all banks')
            summary = row
            continue
        need(current is not None and row.get('bank') == current['id'] and summary is None, 'event bank/order')
        if event in {'DONE', 'RESET'}:
            if current['abort']:
                need(event == 'RESET' and row.get('kind') == current['abort'], 'reset kind')
                need(row.get('reads') == counts['reads'] and row.get('writes') == counts['writes'], 'reset accounting')
                kind = current['abort']
                need((kind != 1 or active_r is not None and not active_r['issued']) and
                     (kind != 2 or bool(target_r)) and
                     (kind != 3 or active_w is not None and active_w['forwarded'] > 0 and not active_w['issued']) and
                     (kind != 4 or memory_b is not None), 'reset hit the intended in-flight channel')
            else:
                count = current['count']
                need(event == 'DONE' and counts['reads'] == counts['writes'] == count*2, 'completed bursts')
                need(row.get('reads') == counts['reads'] and row.get('writes') == counts['writes'], 'completion accounting')
                need(active_r is active_w is memory_b is None and not source_w and not target_r, 'fully drained bank')
                expected = [(o, tx) for tx in range(count) for o in (0, 1)]
                need(r_grants == w_grants == expected, 'sustained round-robin fairness')
                need(all(row.get(k, 0) > 0 for k in ('overlap', 'aw_wait_w')), 'independent directions / AW waits W coverage')
            banks.append(dict(**current, **dict(counts)))
            totals.update(counts)
            current = None
            continue
        owner, tx = row.get('owner'), row.get('tx')
        if event != 'RAW_W':
            need(owner in (0, 1) and tx is not None and 0 <= tx < current['count'], 'transaction identity')
        if event in {'S_AR', 'S_AW'}:
            write = event == 'S_AW'
            need(row.get('addr') == address(owner, tx, write), 'source high address')
            next_tx = w_source if write else r_source
            need(tx == next_tx[owner], 'source transaction order')
            next_tx[owner] += 1
            state = dict(owner=owner, tx=tx, issued=False, accepted=0, forwarded=0, returned=0, checked=0)
            if write:
                need(active_w is None, 'write ownership held until B consumption')
                active_w = state
            else:
                need(active_r is None, 'read ownership held until RLAST consumption')
                active_r = state
        elif event in {'M_AR', 'M_AW'}:
            write = event == 'M_AW'
            state = active_w if write else active_r
            need(state is not None and (state['owner'], state['tx']) == (owner, tx) and not state['issued'], 'memory grant association')
            need(row == dict(bank=current['id'], owner=owner, tx=tx, addr=address(owner, tx, write),
                             beats=burst_length(tx), id=tx % 16, cache=tx % 16, prot=tx % 8), 'complete address/sideband forwarding')
            state['issued'] = True
            (w_grants if write else r_grants).append((owner, tx))
        elif event == 'S_W':
            need(active_w is not None, 'write data owner')
            state = active_w
            beat = state['accepted']
            need((owner, tx) == (state['owner'], state['tx']) and row.get('beat') == beat, 'source W association/order')
            need(beat < burst_length(tx) and row.get('data') == data(owner, tx, beat, True) and
                 row.get('strb') == strobe(tx, beat) and row.get('last') == int(beat == burst_length(tx)-1), 'source W data/mask/last')
            source_w.append({k: row[k] for k in ('last', 'strb', 'data')})
            state['accepted'] += 1
        elif event == 'RAW_W':
            need(active_w is not None and bool(source_w), 'unsolicited memory W')
            need({k: row.get(k) for k in ('last', 'strb', 'data')} == source_w.popleft(), 'registered W data/mask/last forwarding')
            active_w['forwarded'] += 1
            counts['wbeats'] += 1
        elif event == 'M_W':
            need(active_w is not None and active_w['issued'], 'memory write check without address')
            state = active_w
            beat = state['checked']
            need((owner, tx) == (state['owner'], state['tx']) and row.get('beat') == beat and beat < state['forwarded'], 'memory W count/order')
            need(row.get('data') == data(owner, tx, beat, True) and row.get('strb') == strobe(tx, beat) and
                 row.get('last') == int(beat == burst_length(tx)-1), 'memory W numerical check')
            state['checked'] += 1
        elif event in {'M_R', 'S_R'}:
            need(active_r is not None and active_r['issued'], 'read response without address')
            state = active_r
            beat = state['forwarded' if event == 'M_R' else 'returned']
            need((owner, tx) == (state['owner'], state['tx']) and row.get('beat') == beat and beat < burst_length(tx), 'read response owner/order')
            need(row.get('data') == data(owner, tx, beat, False) and row.get('rid') == tx % 16 and
                 row.get('resp') == response(tx) and row.get('last') == int(beat == burst_length(tx)-1), 'read data/ID/response/last')
            if event == 'M_R':
                state['forwarded'] += 1
                target_r.append(row)
            else:
                need(bool(target_r) and row == target_r.popleft(), 'registered R forwarding')
                state['returned'] += 1
                counts['rbeats'] += 1
                if row['last']:
                    need(not target_r, 'duplicate R after last')
                    active_r = None
                    counts['reads'] += 1
        elif event in {'M_B', 'S_B'}:
            need(active_w is not None, 'B owner')
            state = active_w
            need((owner, tx) == (state['owner'], state['tx']) and state['issued'] and
                 state['accepted'] == state['forwarded'] == state['checked'] == burst_length(tx), 'B follows complete AW/W')
            need(row.get('bid') == tx % 16 and row.get('resp') == response(tx), 'write ID/status')
            if event == 'M_B':
                need(memory_b is None, 'duplicate B')
                memory_b = row
            else:
                need(row == memory_b and not source_w, 'registered B forwarding')
                memory_b = active_w = None
                counts['writes'] += 1
        else:
            raise ValueError('unknown evidence event: '+event)
    need(summary is not None and summary.get('banks') == 9 and summary.get('resets') == 4, 'bank/reset coverage')
    for key in ('reads', 'writes', 'rbeats', 'wbeats'):
        need(summary.get(key) == totals[key], 'summary '+key)
    need(len(summary.get('stalls', [])) == 7 and min(summary['stalls']) > 0, 'all downstream requests and upstream responses stalled')
    need(all(summary.get(k, 0) > 0 for k in ('overlap', 'aw_wait_w', 'w_before_aw')), 'contention/independent AW-W coverage')
    return dict(status='PASS', seed=seed, pipelined=pipelined, summary=summary, banks=banks,
                scope='Two protocol initiators and one memory peer, single-clock AXI arbiter; no FFN/PCIe/DDR integration.')


def mutations(text):
    def xor_field(prefix, field):
        return re.sub(r'(^'+prefix+r'[^\n]*\b'+field+r'=)([0-9a-f]+)',
                      lambda m: m[1]+format(int(m[2], 16)^1, 'x'), text, count=1, flags=re.M)
    return {
        'seed': text.replace('NTRPU_ARB_SEED seed=1', 'NTRPU_ARB_SEED seed=2', 1),
        'fatal': text+'\nFatal: injected evidence fault\n',
        'missing_pass': text.replace('NTRPU_ARB_PASS ', 'REMOVED_PASS ', 1),
        'source_high_address': xor_field('NTRPU_ARB_S_AR', 'addr'),
        'memory_high_address': xor_field('NTRPU_ARB_M_AW', 'addr'),
        'memory_write_data': xor_field('NTRPU_ARB_RAW_W', 'data'),
        'memory_write_mask': xor_field('NTRPU_ARB_RAW_W', 'strb'),
        'read_data': xor_field('NTRPU_ARB_S_R', 'data'),
        'read_id': xor_field('NTRPU_ARB_S_R', 'rid'),
        'write_status': xor_field('NTRPU_ARB_S_B', 'resp'),
        'misroute': text.replace('NTRPU_ARB_S_R bank=1 owner=0', 'NTRPU_ARB_S_R bank=1 owner=1', 1),
        'missing_reset': re.sub(r'^NTRPU_ARB_RESET bank=2[^\n]+\n', '', text, count=1, flags=re.M),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--pipelined', action='store_true')
    args = parser.parse_args()
    print(json.dumps(review(args.log.read_text(encoding='utf-8'), args.seed, args.pipelined), indent=2))
