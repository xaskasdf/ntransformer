"""Audit physical AXI traffic through private Q/K/V/state and published results.

The arithmetic peer is a deterministic checksum, not attention. No physical
DDR, generated HLS binding, routed timing or model inference is qualified.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def plan():
    rows = [(1, 0, h, 2, 0) for h in range(32)] + [(1, 15, 31, 0, 0)]
    rows += [(1, 0, 0, 0, m) for m in range(1, 13)]
    rows += [(1, 0, 0, 0, 15), (1, 0, 0, 0, 16), (1, 16, 0, 0, 0),
             (1, 0, 32, 0, 0), (1, 0, 0, 3, 0), (1, 0, 0, 3, 0)]
    rows += [(2, 15, 31, p, m) for p, m in [(63, 0), (64, 0), (128, 0),
             (128, 0), (64, 13), (64, 14), (0, 0)]]
    rows += [(g, 15, 31, 0, 0) for g in (3, 4, 5)]
    assert len(rows) == 61
    return rows


def pattern(g, layer, pos, index):
    return sum((0x3f000000 | ((g*100003+layer*4099+pos*97+index*23+lane*3) & 0x7fffff))
               << (32*lane) for lane in range(16))


def qword(q, lane):
    return 0x3e000000 | ((q['generation']*131071+q['layer']*4099+q['head']*131+lane) & 0x7fffff)


def rotate(value, count):
    return ((value << count) | (value >> (32-count))) & 0xffffffff


def address(cache, q, plane, pos, word):
    return cache['base']+(((q['layer']*2+plane)*8+q['head']//4)*cache['capacity']+pos)*256+word*64


def state_value(q, ending, index, acc):
    if index >= 40:
        return acc[index-40]
    return (q['generation'] << 48) | (q['layer'] << 40) | (q['head'] << 32) | (ending << 8) | index


def review(text, seed, watchdog):
    assert not re.search(r'(?im)^\s*(?:fatal:|error:|ATM_FAIL)', text), 'simulator failure'
    hexkeys = {'seq', 'sequence', 'epoch', 'lease', 'address', 'base', 'data', 'key', 'value'}
    events = []
    for line in text.splitlines():
        if not line.startswith('ATM_'):
            continue
        tag, *parts = line.split()
        pairs = [p.split('=') for p in parts]
        assert all(len(p) == 2 and re.fullmatch('[0-9a-f]+', p[1]) for p in pairs), line
        assert len(dict(pairs)) == len(pairs), 'duplicate field'
        events.append((tag.removeprefix('ATM_'), {k: int(v, 16 if k in hexkeys else 10) for k, v in pairs}))
    assert events[0][0] == 'RESET' and events[-1][0] == 'PASS'
    counts = Counter(t for t, _ in events)
    fixed = dict(BIND=5, BIND_DONE=5, APPEND=139, APPEND_DONE=139, QUERY=61, RESET=4,
                 QUERY_RESET=3, PEER=57, PEER_DONE=57, COMPUTE=58, FINISH=57,
                 Q=3648, KV=40192, STATE_R=624, STATE_W=5928, OUTPUT_W=3200,
                 RESULT=2560, COMPLETE=58, CERTIFICATE=56, LEASE=112, LEASE_DONE=112,
                 AW=2350, W=8836, B=2350, AR=224, R=5038, PASS=1)
    assert counts == Counter(fixed), ('event coverage', counts)
    summary = events[-1][1]
    want = dict(seed=seed, watchdog=watchdog, binds=5, appends=139, queries=61, resets=4,
                peers=57, q_words=3648, kv_words=80384, state_words=624, result_words=2560,
                completions=58, aw=2350, w=8836, b=2350, ar=224, r=5038)
    assert {k: summary[k] for k in want} == want and summary['overlap'] > 0
    mem = {}; prefix = [0]*16; cache = None; binding = None; appends = {}; queries = {}
    aw = ar = None; w = []; applied = False; lease = request = None; q = peer = compute = None
    generation = published = resets = 0; overlaps = set(); metrics = []; last_cycle = -1
    for tag, r in events:
        if tag == 'PASS':
            continue
        assert r['cycle'] >= last_cycle, 'event time moved backwards'
        last_cycle = r['cycle']
        if tag == 'RESET':
            resets += 1
            assert r['count'] == resets and r['query'] == (0 if resets == 1 else 56+resets)
            if q:
                assert q['reset_phase'] == resets-2
            prefix = [0]*16; cache = binding = lease = request = q = peer = compute = aw = ar = None
            w = []; applied = False
        elif tag == 'BIND':
            generation += 1
            assert not lease and q is None and r['generation'] == generation
            cap, offset = (4, 4032) if generation == 1 else ((130, 4032) if generation == 2 else (1, 64))
            assert (r['capacity'], r['base'], r['sequence'], r['epoch']) == (
                cap, 0xa000000000+offset, 0x1234567800000000+generation, 0xfedcba9800000000+generation)
            binding = r; prefix = [0]*16
        elif tag == 'BIND_DONE':
            assert binding and r['id'] == binding['id'] and r['status'] == 0
            cache = binding; binding = None
        elif tag == 'APPEND':
            assert cache and r['generation'] == generation and r['prefix'] == prefix[r['layer']]
            assert r['position'] == prefix[r['layer']] < cache['capacity'] and r['mode'] in (0, 1)
            assert r['id'] not in appends
            appends[r['id']] = dict(r, words=0, responses=0, done=False)
        elif tag == 'APPEND_DONE':
            a = appends[r['id']]; ok = a['mode'] == 0
            assert not a['done'] and r['status'] == (0 if ok else 3)
            assert a['words'] == (64 if ok else 4) and r['bytes'] == a['words']*64
            assert a['responses'] > 0 and aw is None and not w
            if ok:
                prefix[a['layer']] += 1; published += 4096
            else:
                assert a['responses'] == 1
            assert r['committed'] == prefix[a['layer']]; a['done'] = True
        elif tag == 'AW':
            assert aw is None and r['address'] % 64 == 0 and 1 <= r['beats'] <= 4
            assert r['address'] % 4096+r['beats']*64 <= 4096
            aw = r; applied = False
        elif tag == 'W':
            assert len(w) < 4 and (not w or not w[-1]['last']); w.append(r)
        elif tag == 'B':
            assert aw and applied and r['id'] == aw['id'] and len(w) == aw['beats']
            a = appends[r['id']]
            assert r['ordinal'] == a['responses'] and r['resp'] == (2 if a['mode'] else 0)
            a['responses'] += 1; aw = None; w = []; applied = False
        elif tag == 'QUERY':
            assert q is None and peer is None and compute is None and lease is None and request is None
            n = len(queries)+1; assert r['query'] == n
            assert tuple(r[k] for k in ('generation', 'layer', 'head', 'position', 'mode')) == plan()[n-1]
            assert generation == r['generation'] and r['epoch'] == cache['epoch']
            assert r['seq'] == cache['sequence'] ^ (0x100000000 if r['mode'] == 16 else 0)
            assert r['lease'] == 0x1122334400000000+n-1
            q = dict(r, reads=[], loaded={}, next_start=0, peers=0, tokens=0, tiles=0,
                     state=None, result=[], certificate=None, reset_phase=None, compute_count=0)
            queries[n] = q
        elif tag in ('LEASE', 'LEASE_DONE'):
            assert q and r['query'] == q['query']
            if tag == 'LEASE':
                assert request is None and r['lease'] == q['lease']
                if r['release']:
                    assert lease and peer is None and compute is None and ar is None, 'early lease release'
                else:
                    assert not lease and q['peers'] == 0
                request = r
            else:
                assert request and r['release'] == request['release']
                expected = 0 if r['release'] else (2 if q['mode'] == 16 else (1 if prefix[q['layer']] <= q['position'] else 0))
                assert r['status'] == expected and r['active'] == int(not r['release'] and expected == 0)
                lease = None if r['release'] or expected else q['lease']; request = None
        elif tag == 'AR':
            assert lease and q and r['id'] == q['query'] and ar is None
            assert r['address'] % 64 == 0 and 1 <= r['beats'] <= 256 and r['address'] % 4096+r['beats']*64 <= 4096
            ar = dict(r, seen=0)
        elif tag == 'R':
            assert ar and q and r['id'] == ar['id'] == q['query']
            assert r['address'] == ar['address']+64*ar['seen'] and r['last'] == int(ar['seen']+1 == ar['beats'])
            assert r['data'] == mem[r['address']], 'physical read differs from written memory'
            ordinal = len(q['reads']); cursor = ordinal; start = 0
            while cursor >= min(64, q['position']+1-start)*8:
                cursor -= min(64, q['position']+1-start)*8; start += 64
                assert start <= q['position'], 'excess physical reads'
            count = min(64, q['position']+1-start); plane, sub = divmod(cursor, count*4)
            assert r['address'] == address(cache, q, plane, start+sub//4, sub % 4), 'causal read layout'
            assert r['resp'] == (2 if q['mode'] == 1 and ordinal == 3 else 0)
            q['reads'].append(r); q['loaded'][r['address']] = r['data']; ar['seen'] += 1
            if ar['seen'] == ar['beats']:
                ar = None
        elif tag == 'COMPUTE':
            assert q and lease and peer is None and compute is None and ar is None
            assert (r['query'], r['start'], r['tokens'], r['layer'], r['head'], r['kv'], r['lease']) == (
                q['query'], q['next_start'], min(64, q['position']+1-q['next_start']), q['layer'], q['head'], q['head']//4, q['lease'])
            assert len(q['reads']) >= (r['start']+r['tokens'])*8
            compute = r; q['compute_count'] += 1
        elif tag == 'PEER':
            assert q and lease and compute and peer is None
            mode = 0 if q['mode'] == 14 and r['start'] == 0 else q['mode']
            assert (r['query'], r['start'], r['count'], r['mode']) == (q['query'], compute['start'], compute['tokens'], mode)
            peer = dict(r, qwords=[], state_reads=0, kv=0, writes=0, outputs=0, acc=[], done=False)
            q['peers'] += 1
        elif tag in ('Q', 'STATE_R', 'KV', 'STATE_W', 'OUTPUT_W', 'PEER_DONE'):
            assert q and peer and lease and r['query'] == q['query'] and not peer['done']
            if 'start' in r:
                assert r['start'] == peer['start']
            if tag == 'Q':
                lane = len(peer['qwords']); assert r['lane'] == lane < 64 and r['data'] == qword(q, lane)
                peer['qwords'].append(r['data']); peer['acc'].append(r['data'])
            elif tag == 'STATE_R':
                index = peer['state_reads']
                assert peer['start'] and len(peer['qwords']) == 64 and r['index'] == index < 104
                assert q['state'] and r['data'] == q['state'][index], 'continuation state'
                if index >= 40:
                    peer['acc'][index-40] = r['data']
                peer['state_reads'] += 1
            elif tag == 'KV':
                index = peer['kv']; assert r['index'] == index < peer['count']*64
                assert len(peer['qwords']) == 64 and peer['state_reads'] == (104 if peer['start'] else 0)
                valid = not (peer['mode'] == 4 or (peer['mode'] in (5, 6) and index >= 3))
                assert r['valid'] == int(valid)
                for plane, key in [(0, 'key'), (1, 'value')]:
                    pos = peer['start']+index//64; lane = index % 64
                    physical = address(cache, q, plane, pos, lane//16)
                    expected = (q['loaded'][physical] >> (32*(lane % 16))) & 0xffffffff
                    independent = (pattern(generation, q['layer'], pos, plane*32+(q['head']//4)*4+lane//16) >> (32*(lane % 16))) & 0xffffffff
                    assert expected == independent and r[key] == (expected if valid else 0), 'KV consumption/revocation'
                lane = index % 64
                peer['acc'][lane] = rotate(peer['acc'][lane], 3) ^ r['key'] ^ rotate(r['value'], 7) ^ peer['qwords'][lane]
                peer['kv'] += 1
            elif tag == 'STATE_W':
                index = peer['writes']; assert r['index'] == index < 104 and peer['kv'] == peer['count']*64
                assert r['we'] == int(not (peer['mode'] == 8 and index == 103))
                assert r['data'] == state_value(q, peer['start']+peer['count'], index, peer['acc'])
                peer['writes'] += 1
            elif tag == 'OUTPUT_W':
                lane = peer['outputs']; assert r['lane'] == lane < 64 and peer['writes'] == 104
                assert peer['start']+peer['count'] == q['position']+1
                assert r['we'] == int(not (peer['mode'] == 11 and lane == 63))
                wanted = 0x7f800001 if peer['mode'] == 10 and lane == 9 else 0x3f000000 | (peer['acc'][lane] & 0x7fffff)
                assert r['data'] == wanted; peer['outputs'] += 1
            else:
                assert peer['kv'] == peer['count']*64 and peer['writes'] == 104
                assert peer['outputs'] == (64 if peer['start']+peer['count'] == q['position']+1 else 0)
                assert r['aborted'] == int(peer['mode'] in (4, 5, 6, 9)); peer['done'] = True
        elif tag == 'FINISH':
            assert q and peer and peer['done'] and lease and compute
            assert (r['query'], r['start']) == (q['query'], peer['start'])
            status = {4:7, 5:7, 6:2, 7:10, 8:4, 9:7, 10:4, 11:4, 14:10}.get(peer['mode'], 0)
            assert r['status'] == status and r['abort'] == int(status != 0)
            if status == 0:
                q['next_start'] += peer['count']; q['tokens'] += peer['count']; q['tiles'] += 1
                q['state'] = [state_value(q, q['next_start'], i, peer['acc']) for i in range(104)]
                q['result'] = [0x3f000000 | (v & 0x7fffff) for v in peer['acc']]
            peer = compute = None
        elif tag == 'CERTIFICATE':
            assert q and lease is None and request is None and peer is None and compute is None and ar is None
            status = {1:3, 2:7, 3:8, 4:7, 5:7, 6:2, 7:7, 8:7, 9:7, 10:7, 11:7, 13:7, 14:7, 16:2}.get(q['mode'], 0)
            if q['query'] == 50:
                status = 1
            assert (r['query'], r['status'], r['tokens'], r['tiles'], r['lease']) == (q['query'], status, q['tokens'], q['tiles'], q['lease'])
            assert q['certificate'] is None; q['certificate'] = r; q['host_words'] = 0
            if status == 0:
                assert q['tokens'] == q['position']+1 and len(q['reads']) == (q['position']+1)*8
        elif tag == 'RESULT':
            assert q and q['certificate'] and q['certificate']['status'] == 0 and lease is None
            lane = q['host_words']; assert r['query'] == q['query'] and r['lane'] == lane < 64
            assert r['data'] == q['result'][lane]; q['host_words'] += 1
        elif tag == 'COMPLETE':
            assert q and lease is None and request is None and peer is None and ar is None and compute is None
            status = {1:3, 2:7, 3:8, 4:7, 5:7, 6:2, 7:10, 8:4, 9:7, 10:4, 11:4, 13:7, 14:10, 15:4, 16:2}.get(q['mode'], 0)
            if q['query'] in (48, 49, 50):
                status = 1
            assert (r['query'], r['status'], r['peers'], r['revoke']) == (q['query'], status, q['peers'], int(q['mode'] == 12))
            assert (q['certificate'] is None) == (q['query'] in (46, 48, 49))
            assert q.get('host_words', 0) == (64 if status == 0 else 0)
            # Bench timer starts before waiting for the command's falling edge.
            assert r['elapsed'] == r['cycle']-q['cycle']+1 and r['elapsed'] > 0
            metrics.append(dict(query=r['query'], status=status, cycles=r['elapsed'], position=q['position'], mode=q['mode']))
            q = None
        elif tag == 'QUERY_RESET':
            assert q and r['query'] == q['query'] == 58+r['phase'] and r['phase'] in (0, 1, 2)
            assert (ar is not None) == (r['phase'] == 0)
            assert (compute is not None) == (r['phase'] == 1)
            assert (q['certificate'] is not None) == (r['phase'] == 2)
            assert q.get('host_words', 0) == 0; q['reset_phase'] = r['phase']
        else:
            raise AssertionError(tag)
        if aw and not applied and len(w) == aw['beats'] and w[-1]['last']:
            a = appends[aw['id']]
            for index, beat in enumerate(w):
                assert beat['id'] == aw['id'] and beat['last'] == int(index+1 == len(w))
                ordinal = a['words']; plane, rest = divmod(ordinal, 32); head, word = divmod(rest, 4)
                physical = cache['base']+(((a['layer']*2+plane)*8+head)*cache['capacity']+a['position'])*256+word*64
                assert aw['address']+index*64 == physical and beat['data'] == pattern(a['generation'], a['layer'], a['position'], ordinal)
                mem[physical] = beat['data']; a['words'] += 1
            applied = True
        if aw and ar:
            overlaps.add((aw['id'], aw['address'], ar['id'], ar['address']))
    assert q is None and lease is None and request is None and peer is None and ar is None and aw is None and not w
    assert len(queries) == 61 and all(a['done'] for a in appends.values()) and published == 138*4096 and overlaps
    assert summary['cycles'] >= last_cycle
    return dict(summary=summary, publishable_write_bytes=published, physical_read_bytes=5038*64,
                kv_words_compared=80384, q_words_compared=3648, continuation_words_compared=624,
                result_words_compared=2560, observed_overlap_burst_pairs=len(overlaps), queries=metrics,
                generated_hls_bound=False, attention_arithmetic_qualified=False, physical_timing_qualified=False)


def controls(text, seed, watchdog):
    changes = [('W', 'data'), ('R', 'data'), ('AW', 'address'), ('AR', 'address'),
               ('LEASE', 'lease'), ('LEASE_DONE', 'status'), ('COMPUTE', 'kv'),
               ('Q', 'data'), ('STATE_R', 'data'), ('KV', 'key'), ('KV', 'value'),
               ('KV', 'valid'), ('STATE_W', 'data'), ('STATE_W', 'we'), ('OUTPUT_W', 'data'),
               ('FINISH', 'status'), ('CERTIFICATE', 'tokens'), ('RESULT', 'data'),
               ('COMPLETE', 'status'), ('QUERY_RESET', 'phase')]
    changes += [(tag, None) for tag in ('QUERY', 'PEER', 'FINISH', 'RESULT', 'PASS')]
    lines = text.splitlines(); rejected = []
    for tag, key in changes:
        bad = lines.copy(); index = next(i for i, line in enumerate(bad) if line.startswith('ATM_'+tag+' '))
        if key is None:
            del bad[index]
        else:
            base = 16 if key in ('data', 'address', 'lease', 'key', 'value') else 10
            bad[index] = re.sub(r'\b'+key+'=([0-9a-f]+)', lambda m: key+'='+format(int(m[1], base)^1, 'x' if base == 16 else 'd'), bad[index], count=1)
        try:
            review('\n'.join(bad), seed, watchdog)
        except (AssertionError, KeyError, IndexError, ValueError):
            rejected.append(tag+':'+str(key))
        else:
            raise AssertionError('Accepted altered evidence: '+tag+':'+str(key))
    return rejected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('log', type=Path)
    parser.add_argument('--seed', type=int, default=1); parser.add_argument('--watchdog', type=int, default=32)
    args = parser.parse_args(); text = args.log.read_text(); result = review(text, args.seed, args.watchdog)
    result['controls'] = controls(text, args.seed, args.watchdog); print(json.dumps(result, indent=2))
