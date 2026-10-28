"""Audit physical AXI traffic, real HLS RAM ports and Decimal90 attention.

A passing transcript is evidence review only. The queue must separately bind
the generated HLS source, simulation invocation and archived source hashes.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
import check_ntrpu_attention_axi_reference_r1 as numeric
import check_ntrpu_attention_axi_fixture_r1 as layout

MANIFEST_SHA = '4e80daca26e268eb84f07b1a7f749ba509454ec1ab3457467540d96432193711'
SCHEMA = {
    'BIND': 'cycle id generation capacity base sequence epoch',
    'BIND_DONE': 'cycle id status',
    'APPEND': 'cycle id generation layer position mode prefix',
    'APPEND_DONE': 'cycle id status bytes committed',
    'AW': 'cycle id address beats', 'W': 'cycle id last data',
    'B': 'cycle id ordinal resp', 'AR': 'cycle id address beats',
    'R': 'cycle id address resp last data',
    'QUERY': 'cycle query fixture image generation layer head position mode sequence epoch lease',
    'START': 'cycle query fixture start count position layer head kv sequence epoch',
    'Q': 'cycle query start lane data',
    'K': 'cycle query start index valid data', 'V': 'cycle query start index valid data',
    'STATE_R': 'cycle query start index data', 'STATE': 'cycle query start index data',
    'OUTPUT': 'cycle query lane data', 'DONE': 'cycle query start status aborting',
    'FINISH': 'cycle query start status', 'CERTIFICATE': 'cycle query status tokens tiles lease',
    'LEASE': 'cycle query release lease', 'LEASE_DONE': 'cycle query release status active',
    'RESULT': 'cycle query fixture lane data',
    'COMPLETE': 'cycle query fixture status tiles elapsed revoke',
    'CANCEL': 'cycle query revoke', 'RESET': 'cycle count query',
    'REVOKE': 'cycle query epoch valid q_valid data',
    'REVOKED': 'cycle query epoch valid q_valid data',
    'QUERY_RESET': 'cycle query phase',
    'PASS': 'seed watchdog cases queries completions results resets binds appends starts dones q k v state_r state_w output_w aw w b ar r cycles',
}
HEX = {'base', 'sequence', 'epoch', 'lease', 'address', 'data'}
RESET_QUERIES = {150: 0, 152: 1, 154: 2}


def load_fixture(folder, parent):
    raw = (folder / 'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA
    manifest = json.loads(raw)
    files = {name: (folder / name).read_bytes() for name in manifest['files']}
    for name, sha in manifest['files'].items():
        assert hashlib.sha256(files[name]).hexdigest() == sha, name
    jobs, original = layout.parent_jobs(parent)
    proof = layout.review_payload(manifest, files, jobs, original)
    oracle = numeric.fixture(folder / 'native-fixture')
    return dict(manifest=manifest, jobs=jobs, oracle=oracle, layout=proof,
                beats=layout.hexwords(files['kv_beats.hex'], 128, 47808))


def parse(text):
    assert not re.search(r'(?mi)^\s*(?:Fatal:|ERROR:|FAIL:|AHE_FAIL)', text), 'Failure diagnostic'
    events = []
    for line in text.splitlines():
        if not line.startswith('AHE_'):
            continue
        tag, *fields = line[4:].split()
        assert tag in SCHEMA, ('Unknown event', tag)
        assert [f.split('=')[0] for f in fields] == SCHEMA[tag].split(), ('Event schema', tag)
        row = {}
        for field in fields:
            key, value = field.split('=')
            assert re.fullmatch('[0-9a-f]+' if key in HEX else '[0-9]+', value), (tag, field)
            row[key] = int(value, 16 if key in HEX else 10)
        events.append((tag, row))
    assert events and events[-1][0] == 'PASS'
    return events


def physical(cache, layer, plane, head, position, word):
    return cache['base'] + (((layer * 2 + plane) * 8 + head) * cache['capacity'] + position) * 256 + word * 64


def state_numeric(query, tile, oracle):
    state = [tile['state'][i] for i in range(104)]
    prefix = next(p for p in oracle['prefixes'] if p['start'] == tile['start'])
    assert state[:6] == [0x4e54524154543031, query['sequence'], query['epoch'],
                         query['position'], (query['layer'] << 32) | query['head'], prefix['end']]
    assert state[6] == prefix['maximum'], 'Ordered FP64 score maximum'
    assert state[8:40] == [oracle['q'][2*i] | (oracle['q'][2*i+1] << 32) for i in range(32)]
    maximum = Decimal(0)
    with localcontext() as ctx:
        ctx.prec = 90
        expected = [(state[7], prefix['normalizer'], prefix['normalizer'])]
        expected += [(state[40+i], prefix['accumulators'][i], prefix['absolute'][i]) for i in range(64)]
        for bits, want, magnitude in expected:
            number = numeric.f64(bits)
            assert math.isfinite(number), 'Nonfinite private state'
            error = abs(Decimal.from_float(number) - want) / max(magnitude, Decimal('1e-300'))
            assert error <= Decimal('2e-12'), 'Independent state tolerance'
            maximum = max(maximum, error)
    final = prefix['end'] == oracle['tokens']
    assert set(tile['outputs']) == (set(range(64)) if final else set())
    max_ulp = 0
    if final:
        for lane, want in enumerate(prefix['expected']):
            got = tile['outputs'][lane]
            assert math.isfinite(numeric.f32(got))
            delta = numeric.ulp(got, want)
            assert delta <= 1, 'Independent output tolerance'
            max_ulp = max(max_ulp, delta)
    return maximum, max_ulp


def review(text, fixture, seed=1):
    events = parse(text)
    counts = Counter(tag for tag, _ in events)
    fixed = dict(BIND=19, BIND_DONE=19, APPEND=756, APPEND_DONE=756, QUERY=155,
                 RESET=4, QUERY_RESET=3, COMPLETE=152, RESULT=9600, CANCEL=2,
                 START=158, DONE=157, FINISH=157, CERTIFICATE=153, PASS=1,
                 LEASE=308, LEASE_DONE=308, W=48384, R=8768, REVOKE=1, REVOKED=1)
    for tag, count in fixed.items():
        assert counts[tag] == count, ('Event coverage', tag, counts[tag], count)
    summary = events[-1][1]
    assert summary['seed'] == seed and summary['watchdog'] == 65536 and summary['cases'] == 143
    aliases = dict(queries='QUERY', completions='COMPLETE', results='RESULT', resets='RESET',
                   binds='BIND', appends='APPEND', starts='START', dones='DONE', q='Q',
                   k='K', v='V', state_r='STATE_R', state_w='STATE', output_w='OUTPUT',
                   aw='AW', w='W', b='B', ar='AR', r='R')
    assert all(summary[key] == counts[tag] for key, tag in aliases.items()), 'Summary differs from events'
    cache = binding = append = query = tile = lease = request = aw = ar = None
    memory = {}; writes = []; applied = False
    generation = prefix = resets = last_op = query_count = 0
    last_cycle = -1; max_error = Decimal(0); max_ulp = 0; numeric_tiles = 0
    jobs = fixture['jobs']; images = fixture['manifest']['images']; metrics = []
    for tag, row in events:
        if tag == 'PASS':
            continue
        assert row['cycle'] >= last_cycle, ('Event clock moved backwards', tag)
        last_cycle = row['cycle']
        if tag == 'RESET':
            resets += 1
            assert row['count'] == resets and row['query'] == (0 if resets == 1 else 146 + resets*2)
            if query:
                assert query['reset_phase'] == RESET_QUERIES[query['query']]
            cache = binding = append = query = tile = lease = request = aw = ar = None
            memory = {}; writes = []; applied = False; prefix = 0
        elif tag == 'BIND':
            assert not any((query, lease, request, append, aw, ar, binding))
            generation += 1; last_op += 1
            image = images[min(generation, 16)-1]
            assert (row['id'], row['generation'], row['capacity'], row['base'], row['sequence'], row['epoch']) == (
                last_op, generation, image['capacity'], 0xa000000000+image['cache_offset'],
                0x1234567800000000+generation, 0xfedcba9800000000+generation)
            binding = row; memory = {}; prefix = 0
        elif tag == 'BIND_DONE':
            assert binding and row['id'] == binding['id'] and row['status'] == 0
            cache = binding; binding = None
        elif tag == 'APPEND':
            assert cache and not any((append, query, aw, ar, lease)) and not writes
            last_op += 1
            assert (row['id'], row['generation'], row['layer'], row['position'], row['prefix'], row['mode']) == (
                last_op, generation, 0, prefix, prefix, 0)
            assert prefix < cache['capacity']
            append = dict(row, words=0, responses=0)
        elif tag == 'APPEND_DONE':
            assert append and row['id'] == append['id'] and append['words'] == 64
            assert append['responses'] and aw is None and not writes
            prefix += 1
            assert (row['status'], row['bytes'], row['committed']) == (0, 4096, prefix)
            append = None
        elif tag == 'AW':
            assert append and aw is None and row['id'] == append['id']
            assert row['address'] % 64 == 0 and 1 <= row['beats'] <= 4
            assert row['address'] % 4096 + row['beats']*64 <= 4096
            aw = row; applied = False
        elif tag == 'W':
            assert append and row['id'] == append['id'] and len(writes) < 4
            assert not writes or not writes[-1]['last']
            writes.append(row)
        elif tag == 'B':
            assert append and aw and applied and row['id'] == aw['id'] == append['id']
            assert row['ordinal'] == append['responses'] and row['resp'] == 0
            append['responses'] += 1; aw = None; writes = []; applied = False
        elif tag == 'QUERY':
            assert cache and not any((query, tile, append, lease, request, aw, ar)) and not writes
            query_count += 1; ident = query_count if query_count <= 143 else 143
            job = jobs[ident-1]; image_id = 1 if ident <= 128 else ident-127
            wanted_generation = image_id if query_count <= 150 else (query_count-151)//2+17
            mode = {144: 1, 146: 2, 148: 3}.get(query_count, 0)
            assert (row['query'], row['fixture'], row['image'], row['generation'], row['layer'],
                    row['head'], row['position'], row['mode']) == (
                query_count, ident, image_id, wanted_generation, job['layer'], job['query_head'], job['tokens']-1, mode)
            assert generation == wanted_generation and prefix == cache['capacity']
            assert (row['sequence'], row['epoch'], row['lease']) == (
                cache['sequence'], cache['epoch'], 0x1122334400000000+query_count-1)
            query = dict(row, reads=[], loaded={}, next_start=0, starts=0, finished=0,
                         result=[], state=None, outputs=None, certificate=None,
                         cancelled=False, reset_phase=None, revoked=0)
        elif tag in ('LEASE', 'LEASE_DONE'):
            assert query and row['query'] == query['query']
            if tag == 'LEASE':
                assert request is None and row['lease'] == query['lease']
                if row['release']:
                    assert lease and tile is None and ar is None, 'Lease released before HLS/AXI drain'
                else:
                    assert not lease and query['starts'] == 0
                request = row
            else:
                assert request and row['release'] == request['release'] and row['status'] == 0
                assert row['active'] == int(not row['release'])
                lease = None if row['release'] else query['lease']; request = None
        elif tag == 'AR':
            assert lease and query and tile is None and ar is None and row['id'] == query['query']
            assert row['address'] % 64 == 0 and 1 <= row['beats'] <= 256
            assert row['address'] % 4096 + row['beats']*64 <= 4096
            ar = dict(row, seen=0)
        elif tag == 'R':
            assert query and ar and lease and row['id'] == query['query'] == ar['id']
            assert row['address'] == ar['address'] + ar['seen']*64
            assert row['last'] == int(ar['seen']+1 == ar['beats']) and row['resp'] == 0
            assert row['data'] == memory[row['address']], 'AXI read differs from published write'
            ordinal = len(query['reads']); cursor = ordinal; start = 0
            while cursor >= min(64, query['position']+1-start)*8:
                cursor -= min(64, query['position']+1-start)*8; start += 64
                assert start <= query['position'], 'Excess read'
            count = min(64, query['position']+1-start); plane, sub = divmod(cursor, count*4)
            assert row['address'] == physical(cache, query['layer'], plane, query['head']//4, start+sub//4, sub%4)
            query['reads'].append(row); query['loaded'][row['address']] = row['data']; ar['seen'] += 1
            if ar['seen'] == ar['beats']:
                ar = None
        elif tag == 'START':
            assert query and lease and tile is None and ar is None and not query['cancelled']
            start = query['next_start']; count = min(64, query['position']+1-start)
            assert tuple(row[k] for k in ('query', 'fixture', 'start', 'count', 'position', 'layer', 'head', 'kv', 'sequence', 'epoch')) == (
                query['query'], query['fixture'], start, count, query['position'], query['layer'], query['head'], query['head']//4, query['sequence'], query['epoch'])
            assert len(query['reads']) == (start+count)*8
            tile = dict(row, q=set(), k=set(), v=set(), state_reads=set(), state={}, outputs={}, done=False)
            query['starts'] += 1
        elif tag in ('Q', 'K', 'V', 'STATE_R', 'STATE', 'OUTPUT', 'DONE'):
            assert query and tile and lease and row['query'] == query['query'] and not tile['done']
            if 'start' in row:
                assert row['start'] == tile['start']
            job = jobs[query['fixture']-1]
            if tag == 'Q':
                assert 0 <= row['lane'] < 64 and row['data'] == job['query'][row['lane']]
                tile['q'].add(row['lane'])
            elif tag in ('K', 'V'):
                index = row['index']; assert 0 <= index < tile['count']*64
                position, lane = divmod(index, 64); position += tile['start']; plane = int(tag == 'V')
                address = physical(cache, query['layer'], plane, query['head']//4, position, lane//16)
                expected = (query['loaded'][address] >> (32*(lane%16))) & 0xffffffff
                assert expected == job['values' if plane else 'keys'][position*64+lane]
                live = not query['cancelled']
                assert row['valid'] == int(live) and row['data'] == (expected if live else 0), 'KV RAM content/revocation'
                tile[tag.lower()].add(index)
            elif tag == 'STATE_R':
                assert tile['start'] and 0 <= row['index'] < 104 and query['state']
                assert row['data'] == query['state'][row['index']], 'Private continuation differs from prior output'
                tile['state_reads'].add(row['index'])
            elif tag == 'STATE':
                assert 0 <= row['index'] < 104 and row['index'] not in tile['state']
                tile['state'][row['index']] = row['data']
            elif tag == 'OUTPUT':
                assert tile['start']+tile['count'] == query['position']+1
                assert 0 <= row['lane'] < 64 and row['lane'] not in tile['outputs']
                tile['outputs'][row['lane']] = row['data']
            else:
                assert row['aborting'] == int(query['cancelled'])
                if not query['cancelled']:
                    assert row['status'] == 0 and tile['q'] == set(range(64))
                    assert tile['k'] == tile['v'] == set(range(tile['count']*64))
                    assert tile['state_reads'] == (set(range(104)) if tile['start'] else set())
                    assert set(tile['state']) == set(range(104))
                    error, delta = state_numeric(query, tile, fixture['oracle'][query['fixture']-1])
                    max_error = max(max_error, error); max_ulp = max(max_ulp, delta); numeric_tiles += 1
                tile['done'] = True
        elif tag == 'CANCEL':
            assert query and tile and not tile['done'] and not query['cancelled']
            assert query['mode'] in (1, 2) and row['query'] == query['query']
            assert row['revoke'] == int(query['mode'] == 2)
            query['cancelled'] = True
        elif tag == 'FINISH':
            assert query and tile and tile['done'] and lease
            status = {1: 7, 2: 2}.get(query['mode'], 0)
            assert (row['query'], row['start'], row['status']) == (query['query'], tile['start'], status)
            if status == 0:
                query['next_start'] += tile['count']; query['finished'] += 1
                query['state'] = tile['state']; query['outputs'] = tile['outputs']
            tile = None
        elif tag == 'CERTIFICATE':
            assert query and not any((tile, lease, request, ar)) and query['certificate'] is None
            status = {1: 7, 2: 2}.get(query['mode'], 0)
            assert tuple(row[k] for k in ('query', 'status', 'tokens', 'tiles', 'lease')) == (
                query['query'], status, query['next_start'], query['finished'], query['lease'])
            if status == 0:
                assert query['next_start'] == query['position']+1
            query['certificate'] = row
        elif tag == 'RESULT':
            assert query and query['certificate'] and query['certificate']['status'] == 0
            assert not query['revoked']
            lane = len(query['result'])
            assert (row['query'], row['fixture'], row['lane']) == (query['query'], query['fixture'], lane)
            assert lane < 64 and row['data'] == query['outputs'][lane], 'Host result differs from real HLS write'
            query['result'].append(row['data'])
        elif tag in ('REVOKE', 'REVOKED'):
            assert query and query['mode'] == 3 and query['certificate'] and len(query['result']) == 64
            assert row['query'] == query['query'] and row['valid'] == row['q_valid'] == row['data'] == 0
            if tag == 'REVOKE':
                assert query['revoked'] == 0 and row['epoch'] == query['epoch']+1
                query['revoked'] = 1
            else:
                assert query['revoked'] == 1 and row['epoch'] == query['epoch']
                query['revoked'] = 2
        elif tag == 'COMPLETE':
            assert query and query['certificate'] and not any((tile, lease, request, ar))
            status = {1: 7, 2: 2}.get(query['mode'], 0)
            assert tuple(row[k] for k in ('query', 'fixture', 'status', 'tiles', 'revoke')) == (
                query['query'], query['fixture'], status, query['starts'], int(query['mode'] == 3))
            assert len(query['result']) == (0 if status else 64)
            assert query['revoked'] == (2 if query['mode'] == 3 else 0)
            assert len(query['reads']) == (query['position']+1)*8
            assert row['elapsed'] == row['cycle']-query['cycle']+1
            metrics.append(dict(query=query['query'], fixture=query['fixture'], status=status,
                                tokens=query['position']+1, tiles=query['starts'], cycles=row['elapsed']))
            query = None
        elif tag == 'QUERY_RESET':
            assert query and row['query'] == query['query'] and row['phase'] == RESET_QUERIES[query['query']]
            assert (ar is not None) == (row['phase'] == 0)
            assert (tile is not None) == (row['phase'] == 1)
            assert (query['certificate'] is not None) == (row['phase'] == 2)
            assert not query['result']; query['reset_phase'] = row['phase']
        else:
            raise AssertionError(tag)
        if aw and not applied and len(writes) == aw['beats'] and writes[-1]['last']:
            image = images[min(generation, 16)-1]
            for i, beat in enumerate(writes):
                assert beat['id'] == aw['id'] and beat['last'] == int(i+1 == len(writes))
                ordinal = append['words']; assert ordinal < 64
                plane, sub = divmod(ordinal, 32); head, word = divmod(sub, 4)
                address = physical(cache, append['layer'], plane, head, append['position'], word)
                assert aw['address']+i*64 == address
                assert beat['data'] == fixture['beats'][image['first_beat']+append['position']*64+ordinal]
                memory[address] = beat['data']; append['words'] += 1
            applied = True
    assert not any((query, tile, lease, request, ar, aw, append, binding)) and not writes
    assert resets == 4 and generation == 19 and query_count == 155 and numeric_tiles == 155
    assert summary['cycles'] >= last_cycle
    return dict(status='PASS_ATTENTION_ENGINE_R1_TRANSCRIPT', summary=summary,
                physical_write_bytes=counts['W']*64, physical_read_bytes=counts['R']*64,
                numeric_tiles=numeric_tiles, decimal90_max_output_ulp=max_ulp,
                decimal90_max_scaled_state_error=str(max_error),
                queries=metrics, physical_timing_qualified=False,
                generated_source_provenance_requires_external_review=True)


def controls(text, fixture, seed):
    rejected = []
    changes = [('BIND', 'epoch'), ('AW', 'address'), ('W', 'data'), ('B', 'resp'),
               ('AR', 'address'), ('R', 'data'), ('QUERY', 'head'), ('START', 'sequence'),
               ('Q', 'data'), ('K', 'data'), ('V', 'valid'), ('STATE_R', 'data'),
               ('STATE', 'data'), ('OUTPUT', 'data'), ('DONE', 'status'),
               ('FINISH', 'status'), ('CERTIFICATE', 'tokens'), ('RESULT', 'data'),
               ('COMPLETE', 'status'), ('LEASE_DONE', 'active'), ('QUERY_RESET', 'phase'),
               ('REVOKE', 'q_valid'), ('REVOKED', 'valid')]
    for tag, key in changes:
        pattern = rf'(?m)^(AHE_{tag} .*?\b{key}=)([0-9a-f]+)'
        def alter(match):
            number = int(match[2], 16 if key in HEX else 10) ^ 1
            return match[1] + (format(number, 'x') if key in HEX else str(number))
        bad, n = re.subn(pattern, alter, text, count=1)
        assert n == 1 and bad != text, ('Missing control event', tag)
        try:
            review(bad, fixture, seed)
        except (AssertionError, ValueError, KeyError, IndexError, StopIteration):
            rejected.append(tag+'_'+key)
        else:
            raise AssertionError('Accepted corruption: '+tag+'_'+key)
    for tag in ('START', 'STATE', 'RESULT', 'PASS'):
        bad, n = re.subn(rf'(?m)^AHE_{tag} .*\n?', '', text, count=1)
        assert n == 1
        try:
            review(bad, fixture, seed)
        except (AssertionError, ValueError, KeyError, IndexError, StopIteration):
            rejected.append('missing_'+tag)
        else:
            raise AssertionError('Accepted missing '+tag)
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    fixture = load_fixture(args.fixture, args.parent)
    text = args.log.read_text()
    result = review(text, fixture, args.seed)
    if args.controls:
        result['rejected_controls'] = controls(text, fixture, args.seed)
    result.update(log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
