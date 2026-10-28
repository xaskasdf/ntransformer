"""Causal review of original RoPE and attention HLS over physically written KV.

Transcript acceptance requires a separate source/invocation/archive review.
Synthetic checker tests do not execute RTL or qualify either arithmetic core.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import ntrpu_input_attention_fixture_r1 as fixture_module
import ntrpu_input_attention_source_review_r1 as producer_review
import check_ntrpu_attention_engine_r1 as attention

SCHEMA = {
    'BIND': 'cycle group sequence epoch capacity',
    'BIND_ACK': 'cycle group status sequence epoch',
    'BEGIN': 'cycle job sample mode group prefix sequence epoch',
    'ADMIT': 'cycle job sequence epoch layer position',
    'AR': 'cycle job port address beats', 'R': 'cycle job port address last data',
    'AW': 'cycle job address beats', 'W': 'cycle job index last data',
    'B': 'cycle job beats response', 'ROPE_START': 'cycle job position',
    'ROPE_READ': 'cycle job plane index data', 'ROPE_WRITE': 'cycle job index data',
    'ROPE_DONE': 'cycle job status writes',
    'HEAD': 'cycle job head sequence epoch lease query',
    'AT_START': 'cycle job head count sequence epoch',
    'Q': 'cycle job head index data', 'K': 'cycle job head index data', 'V': 'cycle job head index data',
    'STATE': 'cycle job head index data', 'OUTPUT': 'cycle job head index data',
    'AT_DONE': 'cycle job head status', 'HEAD_ACK': 'cycle job head',
    'COMPLETE': 'cycle job status child committed tokens bytes heads',
    'RESULT': 'cycle job index data', 'ACK': 'cycle job', 'RELEASE': 'cycle job prefix',
    'PASS': 'seed jobs binds errors rope heads words source values kv_reads writes state output q k v cycles',
}
CONSUMER_TAGS = set(SCHEMA)
SCHEMA.update(producer_review.SCHEMA)
SCHEMA['PRODUCER_SUMMARY'] = 'starts dones acks consumer_starts consumer_acks norm_starts norm_dones mac reads writes'
HEX = {'tag', 'sequence', 'epoch', 'lease', 'query', 'address', 'data'}
BASE = fixture_module.KV_BASE


def need(ok, why):
    if not ok:
        raise ValueError(why)


def packed(words):
    return sum(x << (32*i) for i, x in enumerate(words))


def physical(plane, head, token, frame):
    return BASE + (((plane*8+head)*4+token)*4+frame)*64


def parse(text):
    need(not re.search(r'(?mi)^\s*(?:Fatal:|ERROR:|FAIL[: ]|IAR_FAIL)', text), 'Failure diagnostic')
    events = []
    for line in text.splitlines():
        if not line.startswith('IAR_'):
            continue
        tag, *fields = line[4:].split()
        need(tag in SCHEMA, 'Unknown event')
        need([s.split('=')[0] for s in fields] == SCHEMA[tag].split(), 'Event schema')
        row = {}
        for field in fields:
            key, value = field.split('=')
            need(re.fullmatch('[0-9a-f]+' if key in HEX else '[0-9]+', value), 'Finite event field')
            row[key] = int(value, 16 if key in HEX else 10)
        events.append((tag, row))
    need(events and events[-1][0] == 'PASS', 'Terminal summary')
    return events


def make_checker(folder):
    manifest, producer_fixture, samples, refs = fixture_module.load(folder)
    # Prove parent attention K/V equal these rotated/source values as well as Q.
    raw = (folder/'consumer/attention-reference/llama32_attention_r1.bin').read_bytes()
    cursor = 12
    for i in range(128):
        ident, n, layer, head, kv, width, seq, epoch = struct.unpack_from('<6I2Q', raw, cursor)
        cursor += 40
        need((ident, n, layer, head, kv, width) == (i+1, i//32+1, 0, i%32, (i%32)//4, 64), 'Parent identity')
        for key, expected in [('q', samples[n-1]['rope'][head*64:(head+1)*64]),
                              ('k', sum([s['rope'][2048+kv*64:2048+(kv+1)*64] for s in samples[:n]], [])),
                              ('v', sum([s['v'][kv*64:(kv+1)*64] for s in samples[:n]], []))]:
            got = list(struct.unpack_from('<'+'I'*len(expected), raw, cursor)); cursor += 4*len(expected)
            need(got == expected, 'Parent '+key+' correspondence')
        cursor += 256

    def job_review(events, spec, ident, memory):
        ordinal = sum(p['mode'] != 1 for p in manifest['plan'][:ident])-1
        producer = producer_review.review_source(events, producer_fixture, spec, ident, ordinal)
        events = [(tag, row) for tag, row in events if tag in CONSUMER_TAGS]
        if spec['mode'] == 1:
            need([tag for tag, _ in events] == ['BEGIN', 'ADMIT', 'COMPLETE', 'ACK', 'RELEASE'], 'Preflight rejects before all arithmetic/AXI')
            start, admit, complete, ack, release = [row for _, row in events]
            seq, epoch = 0x1234567800000000+spec['group'], 0xfedcba9800000000+spec['group']
            need(start == dict(cycle=start['cycle'], job=ident, sample=spec['sample'], mode=1, group=spec['group'], prefix=spec['prefix'], sequence=seq, epoch=epoch), 'Planned rejected command')
            need(admit == dict(cycle=admit['cycle'], job=ident, sequence=seq, epoch=epoch, layer=0, position=spec['sample']), 'Rejected admission identity')
            need(complete == dict(cycle=complete['cycle'], job=ident, status=1, child=0, committed=0, tokens=0, bytes=0, heads=0), 'Rejected command certificate')
            need(ack == dict(cycle=ack['cycle'], job=ident) and release == dict(cycle=release['cycle'], job=ident, prefix=spec['prefix']), 'Rejected ownership release')
            need(start['cycle'] < admit['cycle'] < complete['cycle'] < ack['cycle'] < release['cycle'] and ack['cycle'] >= complete['cycle']+7 and release['cycle']-ack['cycle'] < 100, 'Rejected lifetime')
            return dict(job=ident, mode=1, prefix=spec['prefix'], heads=0, words=0, exact_rope_words=0, kv_reads=0, writes=0, max_output_ulp=0, max_scaled_state_error='0', producer=producer['summary'])
        by = defaultdict(list)
        read_slots = set()
        for tag, row in events:
            need(row.get('job') == ident, 'Job identity')
            if tag in ('ROPE_READ', 'Q', 'K', 'V'):
                slot = (tag, row['cycle'], row.get('plane', 0))
                need(slot not in read_slots, 'At most one observation per physical RAM port/cycle')
                read_slots.add(slot)
            by[tag].append(row)
        for tag in ('BEGIN', 'ADMIT', 'ROPE_START', 'ROPE_DONE', 'COMPLETE', 'ACK', 'RELEASE'):
            need(len(by[tag]) == 1, 'Unique '+tag)
        start, admit, rs, rd, complete, ack, release = [by[t][0] for t in
            ('BEGIN', 'ADMIT', 'ROPE_START', 'ROPE_DONE', 'COMPLETE', 'ACK', 'RELEASE')]
        sample, mode, group, prefix = (spec[k] for k in ('sample', 'mode', 'group', 'prefix'))
        seq, epoch = 0x1234567800000000+group, 0xfedcba9800000000+group
        source = samples[sample]
        need(start == dict(cycle=start['cycle'], job=ident, sample=sample, mode=mode, group=group,
                           prefix=prefix, sequence=seq, epoch=epoch), 'Planned job')
        need(admit == dict(cycle=admit['cycle'], job=ident, sequence=seq, epoch=epoch, layer=0, position=sample), 'Retained admission')
        need(start['cycle'] < admit['cycle'] < rs['cycle'] < rd['cycle'] < complete['cycle'] < ack['cycle'] < release['cycle'], 'Ownership order')
        need(rs['position'] == sample and rd['status'] == 0 and rd['writes'] == 2560, 'RoPE certificate')
        need(ack['cycle'] >= complete['cycle']+7, 'Held public response')
        need(complete['cycle']-admit['cycle'] < 20000010 and release['cycle']-ack['cycle'] < 100, 'Bounded job/drain')
        source_frames = [packed(source['input'][i:i+16]) for i in range(0, 2624, 16)]
        value_frames = [packed(source['v'][i:i+16]) for i in range(0, 512, 16)]
        source_addresses = ([fixture_module.address(6)+i*64 for i in range(128)] +
                            [fixture_module.address(7)+i*64 for i in range(32)] + [fixture_module.address(9)+i*64 for i in range(4)])
        value_addresses = [fixture_module.address(8)+i*64 for i in range(32)]
        pending = {}; reads = [[], [], []]; loaded = {}; rotated = []; rp_seen = [set(), set()]
        aw = None; burst = []; writes = []; planned = 0; responses = []; successful = set()
        head = tile = None; head_count = head_acks = 0; head_outputs = {}; result_count = 0
        max_error = Decimal(0); max_ulp = 0
        for tag, row in events:
            c = row['cycle']
            if tag in ('BEGIN', 'ADMIT', 'ACK', 'RELEASE'):
                continue
            need(admit['cycle'] < c < ack['cycle'], 'Transaction within ownership')
            if tag == 'AR':
                port = row['port']; need(port in (0, 1, 2) and port not in pending, 'One read per master')
                i = len(reads[port])
                if port < 2:
                    addresses = source_addresses if port == 0 else value_addresses
                    need(i < len(addresses), 'Source address extent')
                    a = addresses[i]
                    remaining = (128-i if i < 128 else 160-i if i < 160 else 164-i) if port == 0 else 32-i
                else:
                    need(head is not None and tile is None, 'KV fetch before owned tile')
                    i = len(loaded); n = sample+1; plane, offset = divmod(i, n*4)
                    need(plane < 2, 'KV fetch extent')
                    a = physical(plane, head['head']//4, offset//4, offset%4)
                    remaining = n*4-offset
                beats = min(remaining, 64-(a%4096)//64)
                need(row['address'] == a and row['beats'] == beats, 'Exact physical read layout/page extent')
                pending[port] = dict(row, seen=0)
            elif tag == 'R':
                port = row['port']; need(port in pending, 'Read owns AR'); request = pending[port]
                a = request['address']+request['seen']*64
                need(c > request['cycle'] and row['address'] == a and row['last'] == int(request['seen']+1 == request['beats']), 'Read framing and address')
                if port < 2:
                    expected = (source_frames if port == 0 else value_frames)[len(reads[port])]
                    if a in producer['memory']:
                        actual = producer['memory'][a]
                        need(actual['cycle'] < c and row['data'] == actual['data'], 'Consumer Q/K/V must equal an earlier successful producer B commit')
                    else:
                        need(port == 0 and fixture_module.address(9) <= a < fixture_module.address(9)+256, 'Only phase coefficients are external immutable consumer inputs')
                else:
                    need(a in memory and memory[a]['group'] == group, 'KV read owns a successful current binding write')
                    expected = memory[a]['data']; need(memory[a]['cycle'] < c and a not in loaded, 'KV causal uniqueness')
                    loaded[a] = row['data']
                need(row['data'] == expected, 'Physical read bytes')
                reads[port].append(row['data']); request['seen'] += 1
                if row['last']: del pending[port]
            elif tag == 'ROPE_START':
                need(len(reads[0]) == 164 and 0 not in pending, 'RoPE after all source responses')
            elif tag == 'ROPE_READ':
                plane, i = row['plane'], row['index']
                need(plane in (0, 1) and 0 <= i < (2560 if plane == 0 else 64), 'RoPE RAM bounds')
                need(rs['cycle'] <= c <= rd['cycle'], 'RoPE RAM lifetime')
                offset = i+2560*plane
                need(row['data'] == (reads[0][offset//16] >> (32*(offset%16))) & 0xffffffff, 'RoPE input from accepted physical response')
                rp_seen[plane].add(i)
            elif tag == 'ROPE_WRITE':
                i = len(rotated)
                need(rs['cycle'] <= c <= rd['cycle'] and row['index'] == i < 2560, 'Ordered RoPE publication')
                need(row['data'] == source['rope'][i], 'Independent rational RoPE value'); rotated.append(row['data'])
            elif tag == 'ROPE_DONE':
                need(len(rotated) == 2560, 'Full RoPE before completion')
            elif tag in ('AW', 'W', 'B'):
                need(mode != 1 and c > rd['cycle'] and head_count == 0 and len(reads[1]) == 32, 'KV writes after successful capture')
                if tag == 'AW':
                    need(aw is None and planned < 64, 'One write address')
                    a = physical(planned//32, (planned//4)%8, sample, planned%4)
                    n = min(4-planned%4, 64-(a%4096)//64)
                    need((row['address'], row['beats']) == (a, n), 'Exact physical append layout')
                    aw = dict(row, start=planned); planned += n
                elif tag == 'W':
                    i = len(writes)
                    need(row['index'] == i < 64 and len(burst) < 4, 'Ordered write bytes')
                    expected = packed(rotated[2048+i*16:2048+(i+1)*16]) if i < 32 else reads[1][i-32]
                    need(row['data'] == expected, 'Write equals actual rotated K or accepted V')
                    writes.append(row); burst.append(row)
                else:
                    need(aw is not None and c > aw['cycle'] and len(burst) == row['beats'] == aw['beats'], 'B owns AW and complete W')
                    need([w['index'] for w in burst] == list(range(aw['start'], planned)), 'AW/W correspondence')
                    need(all(w['cycle'] < c and w['last'] == int(i == len(burst)-1) for i, w in enumerate(burst)), 'WLAST and B order')
                    need(row['response'] == (2 if mode == 2 and planned == 64 else 0), 'Final B failure stimulus')
                    if row['response'] == 0:
                        for i, w in enumerate(burst):
                            memory[aw['address']+i*64] = dict(data=w['data'], group=group, cycle=c)
                            successful.add(w['index'])
                    responses.append(row); aw = None; burst = []
            elif tag == 'HEAD':
                need(mode == 0 and len(successful) == 64 and aw is None and not burst and not pending, 'Attention only after complete successful write drain')
                need(head is None and row['head'] == head_count < 32 and c > responses[-1]['cycle'], 'Sequential heads')
                need((row['sequence'], row['epoch'], row['lease']) == (seq, epoch, 0x1000+ident*64+head_count), 'Head identity and lease')
                need(row['query'] == packed(rotated[head_count*64:(head_count+1)*64]), 'Head Q comes from actual RoPE')
                head = row; head_count += 1; loaded = {}
            elif tag == 'AT_START':
                need(head is not None and tile is None and row['head'] == head['head'], 'Unique tile per head')
                need(len(loaded) == (sample+1)*8 and 2 not in pending, 'Full physical KV tile before arithmetic')
                need((row['count'], row['sequence'], row['epoch']) == (sample+1, seq, epoch), 'HLS tile identity')
                tile = dict(start=0, cycle=c, state={}, outputs={}, q=set(), k=set(), v=set(), done=None)
            elif tag in ('Q', 'K', 'V', 'STATE', 'OUTPUT', 'AT_DONE'):
                need(head is not None and tile is not None and row['head'] == head['head'], 'HLS owned head')
                need(c >= tile['cycle'] and (tile['done'] is None or (tag in ('Q', 'K', 'V') and c == tile['done'])), 'HLS transaction lifetime')
                if tag in ('Q', 'K', 'V'):
                    i = row['index']; need(0 <= i < (64 if tag == 'Q' else (sample+1)*64), 'HLS RAM bounds')
                    if tag == 'Q': expected = (head['query'] >> (32*i)) & 0xffffffff
                    else:
                        a = physical(int(tag == 'V'), head['head']//4, i//64, (i%64)//16)
                        need(a in loaded, 'HLS RAM derives from physical response')
                        expected = (loaded[a] >> (32*(i%16))) & 0xffffffff
                    need(row['data'] == expected, 'HLS RAM value'); tile[tag.lower()].add(i)
                elif tag in ('STATE', 'OUTPUT'):
                    key, limit = ('state', 104) if tag == 'STATE' else ('outputs', 64)
                    i = row['index']; need(0 <= i < limit and i not in tile[key], 'Unique complete HLS publication')
                    tile[key][i] = row['data']
                else:
                    need(row['status'] == 0 and len(tile['state']) == 104 and len(tile['outputs']) == 64, 'Complete HLS result')
                    query = dict(sequence=seq, epoch=epoch, position=sample, layer=0, head=head['head'])
                    error, ulps = attention.state_numeric(query, tile, refs[sample*32+head['head']])
                    max_error = max(max_error, error); max_ulp = max(max_ulp, ulps); tile['done'] = c
            elif tag == 'HEAD_ACK':
                need(head is not None and tile is not None and tile['done'] is not None and c > tile['done'], 'Head ACK after HLS drain')
                need(row['head'] == head['head'] and tile['q'] == set(range(64)) and tile['k'] == tile['v'] == set(range((sample+1)*64)), 'Every HLS input observed')
                head_outputs[row['head']] = tile['outputs']; head_acks += 1; head = tile = None
            elif tag == 'COMPLETE':
                need(not pending and aw is None and not burst and head is tile is None, 'Public completion after all drains')
                need(len(reads[0]) == 164 and len(reads[1]) == 32 and len(rotated) == 2560, 'Full capture')
                need(rp_seen == [set(range(2560)), set(range(64))], 'Every actual RoPE input observed')
                need(len(writes) == planned == (0 if mode == 1 else 64), 'Exact append extent')
                need(head_count == head_acks == (32 if mode == 0 else 0), 'Exact head extent')
                need((row['status'], row['child'], row['committed'], row['tokens'], row['bytes'], row['heads']) ==
                     (spec['status'], 3 if mode == 2 else 0, int(mode == 0), sample+1 if mode == 0 else prefix if mode == 2 else 0,
                      0 if mode == 1 else 4096, head_count), 'Public append certificate')
            elif tag == 'RESULT':
                i = (result_count*17+31)%2048
                need(mode == 0 and c > complete['cycle'] and row['index'] == i and result_count < 2048, 'Authorized permuted public result')
                need(row['data'] == head_outputs[i//64][i%64], 'Public result equals actual HLS publication')
                need(math.isfinite(attention.numeric.f32(row['data'])) and attention.numeric.ulp(row['data'], source['attention'][i]) <= 1, 'Public Decimal90 tolerance')
                result_count += 1
            else: raise ValueError('Unexpected in-job event '+tag)
        need(events[0][0] == 'BEGIN' and events[-1][0] == 'RELEASE', 'Job boundary')
        need(result_count == (2048 if mode == 0 else 0), 'Public result extent')
        need(release['prefix'] == (sample+1 if mode == 0 else prefix), 'Prefix success/failure retention')
        return dict(job=ident, mode=mode, prefix=release['prefix'], heads=head_count, words=result_count,
                    exact_rope_words=len(rotated), kv_reads=len(reads[2]), writes=len(writes),
                    max_output_ulp=max_ulp, max_scaled_state_error=str(max_error), producer=producer['summary'])

    def review_events(events, seed=1):
        need(events and events[-1][0] == 'PASS', 'Terminal summary')
        counts = Counter(tag for tag, _ in events); summary = events[-1][1]
        need(counts['PASS'] == 1 and summary['seed'] == seed, 'One seeded summary')
        need(summary['cycles'] < 120000000, 'Global simulation deadline')
        previous = -1
        for tag, row in events[:-2]:
            need(row['cycle'] >= previous, 'Monotonic cycle'); previous = row['cycle']
        need(summary['cycles'] >= previous, 'Summary cycle')
        need(len(events) >= 2 and events[-2][0] == 'PRODUCER_SUMMARY' and counts['PRODUCER_SUMMARY'] == 1, 'Producer summary required')
        producer_summary = events[-2][1]
        need(producer_summary == dict(starts=6, dones=6, acks=6, consumer_starts=6, consumer_acks=6, norm_starts=6, norm_dones=6, mac=18432, reads=665088, writes=1536), 'Full producer summary')
        producer_aliases = dict(starts='PSTART', dones='PDONE', acks='PACK', consumer_starts='CSTART', consumer_acks='CACK', norm_starts='PNSTART', norm_dones='PNDONE', mac='PMAC', reads='PR', writes='PW')
        need(all(producer_summary[k] == counts[t] for k,t in producer_aliases.items()), 'Producer summary matches actual events')
        metrics = []; memory = {}; cursor = prefix = 0
        for ident, spec in enumerate(manifest['plan'], 1):
            if spec['rebind']:
                for tag, fields in [('BIND', dict(capacity=4)), ('BIND_ACK', dict(status=0))]:
                    need(cursor < len(events) and events[cursor][0] == tag, 'Binding boundary')
                    row = events[cursor][1]; cursor += 1; group = spec['group']
                    need(row == dict(cycle=row['cycle'], group=group, sequence=0x1234567800000000+group,
                                     epoch=0xfedcba9800000000+group, **fields), 'Binding identity')
                    if tag == 'BIND': bind_cycle = row['cycle']
                    else: need(row['cycle'] >= bind_cycle+5, 'Held bind acknowledgement')
                prefix = 0  # Keep physical memory; only logical ownership resets.
            need(spec['prefix'] == prefix, 'Causal committed prefix')
            end = cursor
            while end < len(events) and events[end][0] != 'RELEASE': end += 1
            need(end < len(events), 'Missing job release')
            metrics.append(job_review(events[cursor:end+1], spec, ident, memory)); cursor = end+1
            prefix = metrics[-1]['prefix']
        need(cursor == len(events)-2, 'Trailing events')
        aliases = dict(jobs='BEGIN', binds='BIND', rope='ROPE_DONE', heads='HEAD', words='RESULT',
                       writes='W', state='STATE', output='OUTPUT', q='Q', k='K', v='V')
        need(all(summary[k] == counts[t] for k, t in aliases.items()), 'Summary event counts')
        for key, port in [('source', 0), ('values', 1), ('kv_reads', 2)]:
            need(summary[key] == sum(t == 'R' and r['port'] == port for t, r in events), 'Summary read counts')
        fixed = dict(jobs=8, binds=2, errors=3, rope=6, heads=160, words=10240, source=984,
                     values=192, kv_reads=2816, writes=384, state=16640, output=10240)
        need(all(summary[k] == v for k, v in fixed.items()), 'Whole suite coverage')
        return dict(status='PASS_INPUT_ATTENTION_REAL_R1_TRANSCRIPT', summary=summary, producer_summary=producer_summary, metrics=metrics,
                    scope='Original-core trace review; source binding and XSim execution need separate archive review. No physical DDR or complete inference claim.')

    def review(text, seed=1): return review_events(parse(text), seed)

    def controls(text, seed=1):
        events = parse(text); review_events(events, seed); rejected = []
        probes = [('PR','data'), ('PB','response'), ('PDONE','bytes'), ('CSTART','epoch'),
                  ('R','data'), ('ROPE_WRITE','data'), ('STATE','data'), ('OUTPUT','data'),
                  ('RESULT','data'), ('HEAD','lease'), ('COMPLETE','committed'), ('RELEASE','prefix')]
        for tag,key in probes:
            i=next(i for i,(kind,_) in enumerate(events) if kind==tag)
            altered=list(events);row=dict(events[i][1]);row[key]^=1;altered[i]=(tag,row)
            try:review_events(altered,seed)
            except (ValueError,AssertionError,KeyError,IndexError):rejected.append(tag+'.'+key)
            else:raise AssertionError('Accepted corrupted '+tag+'.'+key)
        for tag in ('PACK','CACK','PRODUCER_SUMMARY','PASS'):
            i=next(i for i,(kind,_) in enumerate(events) if kind==tag)
            try:review_events(events[:i]+events[i+1:],seed)
            except (ValueError,AssertionError,KeyError,IndexError):rejected.append(tag+'.missing')
            else:raise AssertionError('Accepted missing '+tag)
        return rejected

    return dict(review=review, events=review_events, controls=controls, samples=samples, refs=refs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('log', type=Path)
    p.add_argument('--fixture', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--controls', action='store_true'); a = p.parse_args(); checker = make_checker(a.fixture)
    text = a.log.read_text(); result = checker['review'](text)
    if a.controls: result['rejected_controls'] = checker['controls'](text)
    result.update(log_sha256=fixture_module.sha(a.log), checker_sha256=fixture_module.sha(Path(__file__)))
    with a.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))
