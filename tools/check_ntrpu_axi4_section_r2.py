"""Independent bus histories for the active admission / actual AMD CDC section.

Checks complete cycle observations, full attributes, causal response queues,
finite closure, retained local replies, proof identities and stopped clocks.
This bank is a transaction fixture, not an XDMA/DDR or physical timing test.
"""
from collections import Counter, deque
import re
from check_ntrpu_axi4_drain import pattern, mask, require
from check_ntrpu_axi4_pair_r2 import FIFO_WARNING

LENS = [1, 2, 4, 16, 64, 256, 3]
FIELDS = {'aw': ['id', 'addr', 'len', 'size', 'prot', 'burst', 'cache', 'lock', 'region', 'qos'],
          'w': ['data', 'strb', 'last'], 'b': ['id', 'resp'],
          'ar': ['id', 'addr', 'len', 'size', 'prot', 'burst', 'cache', 'lock', 'region', 'qos'],
          'r': ['data', 'id', 'resp', 'last']}
HEX = {'awaddr', 'araddr', 'wdata', 'wstrb', 'rdata', 'gen', 'tag', 'proof_tag', 'drain_tag'}
NAMES = ['START', 'PARTIAL_GAP_REQUEST', 'FINITE_PREFIX', 'HELD_RESPONSES', 'STOP_DEVICE',
         'DEVICE_WAIT', 'RESUME_DEVICE', 'PROOF_ONE', 'LOCAL_HELD', 'CDC_RESET_ASSERT',
         'CDC_RESET_RELEASE', 'STOP_HOST_NEW_TAG', 'HOST_WAIT', 'RESUME_HOST', 'PROOF_TWO',
         'NEW_GENERATION', 'PROOF_THREE', 'INJECT_ORPHAN_R', 'STOP_DEVICE_FAULT',
         'DEVICE_FAULT_REVOKED', 'RESUME_DEVICE_FAULT', 'STOP_HOST_FAULT',
         'HOST_FAULT_REVOKED', 'FAULT_FRESH_REQUEST', 'RESUME_HOST_FAULT', 'LOCAL_RELEASE', 'TERMINAL']


def parse(line):
    values = dict(x.split('=') for x in line.split()[1:])
    return {k: (v if k in {'side', 'name'} else None if re.search('[xzXZ]', v)
                else int(v, 16 if k in HEX else 10)) for k, v in values.items()}


def response(n):
    return [0, 2, 3][n % 3]


def review(text, period):
    lines = text.splitlines()
    require(lines.count(FIFO_WARNING) == 5, 'five documented functional-only FIFO notices')
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|Error:|Warning:|WARNING:)',
                          '\n'.join(x for x in lines if x != FIFO_WARNING), re.M), 'runtime diagnostic')
    require(lines.count(f'AXI4_SECTION_SUITE_PASS device_period_ps={period} writes=24 reads=24 local_writes=1 local_reads=1 proofs=3 cdc_resets=1 native_endpoint=0') == 1, 'suite marker')
    require(lines.count(f'AXI4_SECTION_PROFILE device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000 transactions=24 write_depth=32') == 1, 'profile')
    events = [parse(x) for x in lines if x.startswith('AXI4_SECTION_EVENT ')]
    reset_names = ['HOST_RESET_ASSERT','DEVICE_RESET_ASSERT','HOST_RESET_RELEASE','DEVICE_RESET_RELEASE']
    require([e['name'] for e in events if e['name'] not in reset_names] == NAMES, 'complete ordered event inventory')
    require(all(sum(e['name'] == n for e in events) == 1 for n in reset_names), 'four per-domain reset transitions')
    require(all(a['t'] <= b['t'] for a, b in zip(events, events[1:])), 'monotonic events')
    by = {e['name']: e for e in events}
    buses = {s: [parse(x) for x in lines if x.startswith('AXI4_SECTION_BUS side=' + s + ' ')] for s in ['S', 'H', 'D']}
    hosts = [parse(x) for x in lines if x.startswith('AXI4_SECTION_HOST ')]
    mgmt = [parse(x) for x in lines if x.startswith('AXI4_SECTION_MGMT ')]
    reset_measurements = {}
    for side, clock in [('HOST',7000),('DEVICE',period)]:
        first = by[side+'_RESET_ASSERT']['t']; last = by[side+'_RESET_RELEASE']['t']
        clock_offset = 0 if side == 'HOST' else 600
        require((first-1-clock_offset) % (clock//2) == 0 and (last-1-clock_offset) % (clock//2) == 0, 'reset transitions follow local clock edges')
        require(last-first >= 64*clock, 'at least 64 local clock periods under reset')
        require(first <= by['CDC_RESET_ASSERT']['t'] < last <= by['CDC_RESET_RELEASE']['t'], 'overlapping domain reset order')
        reset_measurements[side] = dict(assert_ps=first,release_ps=last,duration_ps=last-first)
    overlap = min(v['release_ps'] for v in reset_measurements.values())-max(v['assert_ps'] for v in reset_measurements.values())
    require(overlap >= 16*max(period,7000), 'reset overlap exceeds PG059 slowest-clock recommendation')
    times = {}; snapshots = {}; transfers = {}; stats = {}; offers = {}
    pause_names = [('D', 'STOP_DEVICE', 'RESUME_DEVICE'), ('H', 'STOP_HOST_NEW_TAG', 'RESUME_HOST'),
                   ('D', 'STOP_DEVICE_FAULT', 'RESUME_DEVICE_FAULT'), ('H', 'STOP_HOST_FAULT', 'RESUME_HOST_FAULT')]
    for side, rows in buses.items():
        clock_side = 'H' if side == 'S' else side
        step = 7000 if clock_side == 'H' else period
        offset = 0 if clock_side == 'H' else 600
        times[side] = [r['t'] for r in rows]
        require(rows and all(a < b for a, b in zip(times[side], times[side][1:])), 'strict bus timestamps')
        first_phase = offset + step // 2
        first = first_phase + ((by['START']['t'] - first_phase) // step + 1) * step
        require(times[side][0] == first, 'first bus observation')
        require(0 <= by['TERMINAL']['t'] - times[side][-1] < step, 'last bus observation')
        require(all((t - offset) % (step // 2) == 0 for t in times[side]), 'clock grid')
        pauses = [(by[a]['t'], by[b]['t']) for s, a, b in pause_names if s == clock_side]
        for a, b in zip(times[side], times[side][1:]):
            if not any(a <= end and start <= b for start, end in pauses):
                require(b - a == step, 'missing clock observation')
        for start, end in pauses:
            require(not any(start < t < end for t in times[side]), 'source clock advanced during declared stop')
        aw = deque(); w = deque(); partial = []; aseq = rseq = wseq = 0
        writes = {i: deque() for i in range(16)}; reads = {i: deque() for i in range(16)}
        ap = wp = 0; observed = {ch: [] for ch in FIELDS}; raw_offers = []
        handshakes = Counter(); stalls = Counter(); maximum = Counter(); previous = None
        state_at = {}; first_w = first_aw = None
        for row in rows:
            when = row['t']
            domain = 'HOST' if clock_side == 'H' else 'DEVICE'
            reset = reset_measurements[domain]
            expected_reset = not (reset['assert_ps'] < when < reset['release_ps'])
            require(row['rst'] == expected_reset, 'actual CDC reset pins follow declared interval')
            if side != 'S' and reset['assert_ps'] + 16*step < when < reset['release_ps']:
                require(not any(row[ch+s] for ch in FIELDS for s in ['valid','ready']), 'both connected raw interface ends stay in reset state')
            state_at[when] = dict(ap=ap, wp=wp, ar=sum(map(len, reads.values())), partial=len(partial),
                                  aw=aseq, wc=wseq, rc=rseq,
                                  b=handshakes['b'], r=handshakes['r'])
            for ch, fields in FIELDS.items():
                require(row[ch + 'valid'] in {0, 1} and row[ch + 'ready'] in {0, 1}, 'known handshakes')
                if row[ch + 'valid']:
                    require(all(row[ch + f] is not None for f in fields), 'known offered payload')
                if previous and previous[ch + 'valid'] and not previous[ch + 'ready']:
                    require(row[ch + 'valid'] and all(row[ch + f] == previous[ch + f] for f in fields), f'{side} held {ch} changed')
                if row[ch + 'valid'] and not row[ch + 'ready']: stalls[ch] += 1
                if row[ch + 'valid'] and row[ch + 'ready']:
                    handshakes[ch] += 1; observed[ch].append(tuple(row[ch + f] for f in fields))
            if row['bvalid']:
                require(writes[row['bid']], f'{side} orphan B')
                desc = writes[row['bid']][0]
                require(row['bresp'] == (2 if desc['seq'] == 24 else response(desc['seq'])), 'B response semantics')
                if row['bready']: writes[row['bid']].popleft(); ap -= 1; wp -= 1
            if row['rvalid']:
                q = reads[row['rid']]
                if not q:
                    require(side != 'S' and when >= by['INJECT_ORPHAN_R']['t'], 'unexpected orphan R')
                    require((row['rid'], row['rresp'], row['rlast'], row['rdata']) == (3, 2, 1, pattern(0xbad4c0de)), 'precise external orphan injection')
                    if side == 'H': require(not row['rready'], 'orphan must be contained before protected source')
                    raw_offers.append(when)
                else:
                    desc = q[0]; local = desc['seq'] == 24
                    require(row['rlast'] == (desc['beat'] == desc['len'] - 1), 'R LAST association')
                    require(row['rresp'] == (2 if local else response(desc['beat'])), 'R response semantics')
                    expected_data = 0 if local else pattern(desc['seq'] * 4096 ^ (desc['beat'] << 8) ^ row['rid'])
                    require(row['rdata'] == expected_data, 'complete R payload')
                    if row['rready']:
                        desc['beat'] += 1
                        if row['rlast']: q.popleft()
            for ch in ['aw', 'ar']:
                if row[ch + 'valid'] and row[ch + 'ready']:
                    seq = aseq if ch == 'aw' else rseq
                    require(seq < (25 if side == 'S' else 24), 'no request after finite prefix')
                    expected = [seq % 16, (0x100000000 if ch == 'aw' else 0x200000000) + seq * 4096,
                                LENS[seq % 7] - 1, 4 if seq % 7 == 5 else 6, seq % 8, 1,
                                3 + 4 * (seq % 4), 0, seq % 16, 15 - seq % 16]
                    require([row[ch + f] for f in FIELDS[ch]] == expected, 'all address attributes including REGION/QOS')
                    require((row[ch + 'addr'] % 4096) + (row[ch + 'len'] + 1) * (1 << row[ch + 'size']) <= 4096, '4K burst boundary')
                    if ch == 'aw':
                        first_aw = when if first_aw is None else first_aw
                        aw.append(dict(seq=seq, id=seq % 16, len=LENS[seq % 7], size=expected[3])); aseq += 1; ap += 1
                    else: reads[seq % 16].append(dict(seq=seq, len=LENS[seq % 7], beat=0)); rseq += 1
            if row['wvalid'] and row['wready']:
                first_w = when if first_w is None else first_w
                partial.append((row['wdata'], row['wstrb']))
                if row['wlast']: w.append(partial); partial = []; wp += 1; wseq += 1
            while aw and w:
                desc = aw.popleft(); beats = w.popleft()
                require(len(beats) == desc['len'], 'AW/W burst association')
                for beat, (data, strb) in enumerate(beats):
                    require(data == pattern(desc['seq'] * 4096 ^ (beat << 8)), 'all W payload bits')
                    require(strb == mask(beat, desc['size']), 'all W strobe bits')
                writes[desc['id']].append(desc)
            maximum['aw'] = max(maximum['aw'], ap); maximum['w'] = max(maximum['w'], wp)
            maximum['ar'] = max(maximum['ar'], sum(map(len, reads.values())))
            previous = row
        expected = dict(aw=25, w=1061, b=25, ar=25, r=1061) if side == 'S' else dict(aw=24, w=1045, b=24, ar=24, r=1045 + (side == 'D'))
        require(dict(handshakes) == expected, f'{side} full transaction totals')
        require(not (ap or wp or partial or aw or w or any(reads.values()) or any(writes.values())), 'terminal causal queues empty')
        require(first_w < first_aw, 'real W before AW')
        if side != 'S': require(raw_offers, 'real orphan reached both raw boundaries')
        snapshots[side] = state_at; transfers[side] = observed; offers[side] = raw_offers
        stats[side] = dict(samples=len(rows), handshakes=dict(handshakes), stalls=dict(stalls), maximum=dict(maximum))
    for ch in FIELDS:
        raw_d = transfers['D'][ch][:-1] if ch == 'r' else transfers['D'][ch]
        require(transfers['H'][ch] == raw_d, 'normal raw CDC conservation')
        size = len(transfers['H'][ch]); require(transfers['S'][ch][:size] == transfers['H'][ch], 'protected normal payload conservation')
    for domain, side in [('HOST','H'),('DEVICE','D')]:
        for transition in ['assert_ps','release_ps']:
            require(reset_measurements[domain][transition]-1 in times[side], 'reset transition follows an observed local rising edge')
    require(times['S'] == times['H'] == [x['t'] for x in hosts], 'complete admission observations')
    # Retained host inventories are measured before the accepting edge. Source
    # inventory includes local replies only after the section was proved empty.
    for row, source, raw in zip(hosts, buses['S'], buses['H']):
        when = row['t']; h = snapshots['H'][when]; s = snapshots['S'][when]
        require(row['cold'] == 1, 'no cold reset during retained-lifetime tests')
        require(row['hrst'] == source['rst'], 'host reset diagnostic')
        require((row['hap'], row['hwp'], row['har']) == (h['ap'], h['wp'], h['ar']), 'raw host passive ledger')
        if row['state'] != 2:
            require((row['ap'], row['wp'], row['arp'], row['partial']) == (s['ap'], s['wp'], s['ar'], s['partial']), 'protected reply inventory')
        else:
            require(s['b'] >= 24 and s['r'] >= 1045, 'REJECT requires all protected normal responses consumed')
            require((row['ap'], row['wp'], row['arp'], row['partial']) == (0, 0, 0, 0), 'downstream guard remains empty')
            require(not any(raw[ch + 'valid'] for ch in ['aw', 'w', 'ar']), 'no reopening in REJECT')
        if row['state'] == 1:
            require((row['awleft'], row['wleft'], row['arleft']) == (24 - h['aw'], 24 - h['wc'], 0), 'finite prefix independent of ring indices')
        if when > by['LOCAL_HELD']['t'] and when < by['LOCAL_RELEASE']['t']:
            require(row['state'] == 2 and row['fenced'] == 1 and row['idle'] == 0 and source['bvalid'] == source['rvalid'] == 1, 'retained local replies through reset/proof/fault')
        require(row['af'] == (2 if offers['H'] and when > offers['H'][0] else 0), 'direct registered admission fault')
        require(row['hf'] == (4 if offers['H'] and when > offers['H'][0] else 0), 'direct registered host monitor fault')
        if row['proof_ack']:
            require(row['proof_req'] and row['proof_tag'] == row['drain_tag'] and row['state'] == 2 and not row['af'], 'fresh producer proof')
    # Management samples are post-edge, including all fault CDC updates.
    require(mgmt and all(b['t'] - a['t'] == 10000 for a, b in zip(mgmt, mgmt[1:])), 'complete management clock observations')
    require(mgmt[0]['t'] == 5001 + ((by['START']['t'] - 5001) // 10000 + 1) * 10000, 'first management observation')
    require(0 <= by['TERMINAL']['t'] - mgmt[-1]['t'] <= 10000, 'last management observation')
    last_fault = 0
    for row in mgmt:
        require((row['fm'] | last_fault) == row['fm'], 'management faults retained')
        require(row['fm'] in {0, 32768, 32770, 33280, 33282}, 'only causal declared faults')
        last_fault = row['fm']
        require(row['ack'] == (row['uc'] and row['pc']), 'complete section certificate')
        if row['fm']: require(not (row['ack'] or row['uc'] or row['pc']), 'immediate synchronized fault masks all cached certificates')
        if row['ack']:
            require(row['req'] and row['fenced'] and row['state'] == 2 and not row['fm'], 'valid section certificate')
            require(not any(row[k] for k in ['ap', 'wp', 'arp', 'partial', 'hap', 'hwp', 'har', 'dap', 'dwp', 'dar']), 'all section inventories empty at certificate')
        if by['STOP_HOST_NEW_TAG']['t'] <= row['t'] < by['RESUME_HOST']['t']:
            require(row['ack'] == row['uc'] == row['pc'] == 0, 'new tag cannot reuse cached certificate while host stopped')
    forbidden = ['PARTIAL_GAP_REQUEST', 'FINITE_PREFIX', 'HELD_RESPONSES', 'DEVICE_WAIT',
                 'STOP_HOST_NEW_TAG', 'HOST_WAIT', 'NEW_GENERATION', 'DEVICE_FAULT_REVOKED',
                 'HOST_FAULT_REVOKED', 'FAULT_FRESH_REQUEST', 'TERMINAL']
    for name in forbidden: require(by[name]['ack'] == 0, 'forbidden certificate at ' + name)
    proofs = [by[n] for n in ['PROOF_ONE', 'PROOF_TWO', 'PROOF_THREE']]
    for proof in proofs:
        sampled = next(r for r in reversed(mgmt) if r['t'] <= proof['t'])
        require((proof['gen'], proof['tag']) == (sampled['gen'], sampled['tag']), 'proof event identity matches management trace')
        require(proof['ack'] == proof['uc'] == proof['pc'] == proof['hc'] == proof['dc'] == 1 and proof['fm'] == 0, 'complete fresh proof chain')
        require(proof['sbs'] == 24 and proof['srs'] == proof['drs'] == 1045, 'protected normal completions before proof')
    require(proofs[0]['gen'] == proofs[1]['gen'] != proofs[2]['gen'] and proofs[0]['tag'] != proofs[1]['tag'] == proofs[2]['tag'], 'distinct tag and generation lifetimes')
    for left, right in [('PROOF_ONE', 'PROOF_TWO'), ('PROOF_TWO', 'PROOF_THREE')]:
        lo, hi = by[left]['t'], by[right]['t']
        require(any(lo < r['t'] < hi and not r['proof_req'] and not r['proof_ack'] for r in hosts), 'producer fresh request gap')
        require(any(lo < r['t'] < hi and r['proof_req'] and not r['proof_ack'] for r in hosts), 'producer fresh low-before-ack')
    finite = by['FINITE_PREFIX']
    require(finite['partial'] == 2 and finite['wleft'] == 1 and finite['awleft'] == finite['arleft'] == 0, 'partial W gap witness')
    for name in ['LOCAL_HELD', 'CDC_RESET_ASSERT', 'CDC_RESET_RELEASE', 'PROOF_TWO', 'PROOF_THREE']:
        e = by[name]; require(e['ack'] == e['sbv'] == e['srv'] == 1 and not e['idle'], 'local pending distinct from section proof')
    require(by['CDC_RESET_RELEASE']['t'] - by['CDC_RESET_ASSERT']['t'] >= 310000, 'actual sustained CDC-only reset')
    latencies = {}
    for side, bit, name in [('device', 15, 'STOP_DEVICE_FAULT'), ('admission', 1, 'STOP_HOST_FAULT')]:
        found = next(r for r in mgmt if r['fm'] & (1 << bit))
        latency = found['t'] - by[name]['t']; require(0 <= latency <= 30001, 'fault transport without another source edge')
        latencies[side] = latency
    end = by['TERMINAL']; require(end['fm'] == 33282 and end['af'] == 2 and end['hf'] == end['df'] == 4, 'permanent independent source faults')
    require(end['sbv'] == end['srv'] == end['ub'] == end['hb'] == end['db'] == 0 and end['idle'] == 0, 'local replies completed but fault forbids reopening')
    return dict(status='PASS_ACTIVE_AXI4_SECTION_REAL_CDC', device_period_ps=period, profiles=stats,
                normal_writes=24, normal_reads=24, normal_write_beats=1045, normal_read_beats=1045,
                local_write_beats=16, local_read_beats=16, fresh_certificates=3, cdc_only_resets=1,
                fault_reporting_latency_ps=latencies, reset_measurements=reset_measurements, reset_overlap_ps=overlap, vendor_model_notices=5,
                data_cdc_synchronization_delay_modeled=False,
                scope='Actual AMD CDC + retained admission + three fresh tagged mailboxes; full attributes and orphan containment. No native endpoint reset, DDR or physical performance qualification.')
