"""Independent transaction and VCD review of the composed drain fixture.

No XDMA endpoint, qualified external provider, physical timing or reset claim.
Uses accepted bus traffic, every logged AXI4 clock edge and actual VCD state;
HDL assertions and PASS markers alone are insufficient.
"""
from collections import Counter, deque
import re

from check_ntrpu_axi4_section import FIELDS, LENS, parse, response
from check_ntrpu_axi4_drain import pattern, mask, require
from check_ntrpu_axi4_pair_r2 import FIFO_WARNING
from check_ntrpu_xdma_stop_inventory_r4 import markers, vcd_rows as raw_vcd_rows


def vcd_rows(text):
    # XSIM 2026.1 writes constant parameters as width 0 while retaining their
    # declared [msb:lsb] range and binary value. Recover only this exact form;
    # signal declarations and malformed value widths remain strict.
    header, body = text.split('$enddefinitions $end', 1)
    header = re.sub(r'(?m)^(\$var parameter) 0 (\S+ \S+) \[(\d+):(\d+)\] \$end$',
                    lambda m: f'{m[1]} {abs(int(m[3])-int(m[4]))+1} {m[2]} [{m[3]}:{m[4]}] $end', header)
    yield from raw_vcd_rows(header+'$enddefinitions $end'+body)


def axi4_history(rows, side, external, shared, fault_start):
    aw, w, partial = deque(), deque(), []
    writes = {i: deque() for i in range(16)}
    reads = {i: deque() for i in range(16)}
    aseq = rseq = 0
    accepted = {ch: [] for ch in FIELDS}
    stalls = Counter()
    previous = None
    first_aw = first_w = None
    orphan = []
    for row in rows:
        require(row['rst'] == 1 and row['phase'] in {0, 1}, 'AXI4 operational reset/phase')
        if previous:
            require(row['t'] > previous['t'], 'strict AXI4 timestamps')
        for ch, fields in FIELDS.items():
            require(row[ch+'valid'] in {0, 1} and row[ch+'ready'] in {0, 1}, 'known handshake')
            if row[ch+'valid']:
                require(all(row[ch+f] is not None for f in fields), 'known offered payload')
            if previous and previous[ch+'valid'] and not previous[ch+'ready']:
                require(row[ch+'valid'] and all(row[ch+f] == previous[ch+f] for f in fields),
                        side+' held '+ch+' payload/VALID changed')
            if row[ch+'valid'] and not row[ch+'ready']:
                stalls[ch] += 1
            if row[ch+'valid'] and row[ch+'ready']:
                accepted[ch].append(tuple(row[ch+f] for f in fields))
        if row['bvalid']:
            require(writes[row['bid']], side+' orphan B')
            desc = writes[row['bid']][0]
            require(row['bresp'] == (2 if desc['seq'] == 24 else response(desc['seq'])), 'B response')
            if row['bready']:
                writes[row['bid']].popleft()
        if row['rvalid']:
            q = reads[row['rid']]
            if not q:
                require(shared and side != 'S' and row['t'] >= fault_start, 'unexpected orphan R')
                require((row['rid'], row['rresp'], row['rlast'], row['rdata']) ==
                        (3, 2, 1, pattern(0xbad4c0de)), 'exact orphan injection')
                if side == 'H':
                    require(not row['rready'], 'orphan R containment')
                orphan.append(row['t'])
            else:
                desc = q[0]
                local = desc['seq'] == 24
                require(row['rlast'] == (desc['beat'] == desc['len']-1), 'R LAST')
                require(row['rresp'] == (2 if local else response(desc['beat'])), 'R response')
                expected = 0 if local else pattern(desc['seq']*4096 ^ (desc['beat'] << 8) ^ row['rid'])
                require(row['rdata'] == expected, 'full R payload')
                if row['rready']:
                    desc['beat'] += 1
                    if row['rlast']:
                        q.popleft()
        for ch in ('aw', 'ar'):
            if row[ch+'valid'] and row[ch+'ready']:
                seq = aseq if ch == 'aw' else rseq
                limit = 25 + (external and ch == 'aw') if side == 'S' else 24
                require(seq < limit, 'finite accepted request prefix')
                expected = [seq % 16, (0x100000000 if ch == 'aw' else 0x200000000)+seq*4096,
                            LENS[seq % 7]-1, 4 if seq % 7 == 5 else 6, seq % 8, 1,
                            3+4*(seq % 4), 0, seq % 16, 15-seq % 16]
                require([row[ch+f] for f in FIELDS[ch]] == expected, 'all address attributes')
                require((expected[1] % 4096)+(expected[2]+1)*(1 << expected[3]) <= 4096,
                        '4 KiB burst boundary')
                if ch == 'aw':
                    first_aw = row['t'] if first_aw is None else first_aw
                    aw.append(dict(seq=seq, id=seq % 16, length=LENS[seq % 7], size=expected[3]))
                    aseq += 1
                else:
                    reads[seq % 16].append(dict(seq=seq, len=LENS[seq % 7], beat=0))
                    rseq += 1
        if row['wvalid'] and row['wready']:
            first_w = row['t'] if first_w is None else first_w
            partial.append((row['wdata'], row['wstrb']))
            if row['wlast']:
                w.append(partial)
                partial = []
        while aw and w:
            desc, beats = aw.popleft(), w.popleft()
            require(len(beats) == desc['length'], 'AW/W burst association')
            for beat, (data, strobe) in enumerate(beats):
                require(data == pattern(desc['seq']*4096 ^ (beat << 8)), 'full W payload')
                require(strobe == mask(beat, desc['size']), 'full W strobe')
            writes[desc['id']].append(desc)
        previous = row
    counts = {ch: len(v) for ch, v in accepted.items()}
    expected = dict(aw=25+external, w=1061, b=25, ar=25, r=1061) if side == 'S' else \
        dict(aw=24, w=1045, b=24, ar=24, r=1045+int(shared and side == 'D'))
    require(counts == expected, side+' complete transaction counts')
    require(not (w or partial or any(reads.values()) or any(writes.values())), 'all responses retired')
    require([d['seq'] for d in aw] == ([25] if external and side == 'S' else []),
            'only declared late AW remains unretired')
    require(first_w is not None and first_aw is not None and first_w < first_aw, 'W before AW witness')
    require(bool(orphan) == bool(shared and side != 'S'), 'orphan boundary evidence')
    return accepted, dict(samples=len(rows), handshakes=counts, stalls=dict(stalls), orphan_cycles=len(orphan))


LITE_FIELDS = {'AW': ('addr',), 'W': ('data', 'strb'), 'B': ('resp',),
               'AR': ('addr',), 'R': ('data', 'resp')}


def lite_history(rows):
    result = {}
    for side in ('U', 'H', 'D'):
        channels = {c: [r for r in rows if r['side'] == side and r['channel'] == c] for c in LITE_FIELDS}
        count = 6 if side == 'U' else 5
        require(all(len(v) == count for v in channels.values()), 'AXI-Lite transaction totals')
        values = {c: [tuple(int(r[f], 16) for f in LITE_FIELDS[c]) for r in rs] for c, rs in channels.items()}
        aw = [(0x7000+16*n,) for n in range(4)]+[(0x7400,)]
        ar = [(0x7000+16*n,) for n in range(4)]+[(0x7000,)]
        writes = [(0xa1b2c3d4+n, n+1) for n in range(4)]+[(0x89abcdef, 10)]
        words = []
        for n in range(4):
            word = (0x13579bdf ^ ((0x1c00+n*4)*0x01020408)) & 0xffffffff
            for b in range(4):
                if (n+1) & (1 << b):
                    word = (word & ~(255 << (b*8))) | (((0xa1b2c3d4+n) >> (b*8) & 255) << (b*8))
            words.append((word, 0))
        expected = dict(AW=aw, W=writes, B=[(0,)]*5, AR=ar, R=words+[words[0]])
        if side == 'U':
            for c, value in dict(AW=(0x9000,), W=(0xaabbccdd, 5), B=(2,), AR=(0x9000,), R=(0, 2)).items():
                expected[c] = expected[c]+[value]
        require(values == expected, 'AXI-Lite full data/address/response semantics')
        for c, rs in channels.items():
            require(all(int(a['ps']) < int(b['ps']) for a, b in zip(rs, rs[1:])), 'AXI-Lite channel order')
            require(all(r['case'] == '0' and r['close'] in {'0', '1'} for r in rs), 'AXI-Lite declared scenario')
        for n in range(count):
            require(max(int(channels[c][n]['ps']) for c in ('AW', 'W')) < int(channels['B'][n]['ps']), 'AXI-Lite write causality')
            require(int(channels['AR'][n]['ps']) < int(channels['R'][n]['ps']), 'AXI-Lite read causality')
        result[side] = {c: len(v) for c, v in channels.items()}
    return result


EVENT_MAP = dict(gen='generation', tag='tag', pg='pg', pt='pt', req='req', present='present',
                 ready='ready', proof='proof_req', ack='ack', accepted='accepted', complete='complete',
                 revoked='revoked', fault='retained_faults', sections='section_ack', idle='local_idle',
                 pending='replies_pending', localbusy='local_busy', localreq='local_peer_req',
                 localack='local_peer_ack', localfault='local_revoked', enable='host_enable')


def review(sim, vcd, period, external, kind='original'):
    require(kind in {'original', 'r2', 'shared'} and external in {0, 1} and period in {5000, 37000}, 'known profile')
    shared = kind == 'shared'
    require(not shared or external == 0, 'shared profile uses absent external contract')
    lines = sim.splitlines()
    require(lines.count(FIFO_WARNING) == 5, 'five functional FIFO notices')
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|Error:|Warning:|WARNING:|CRITICAL WARNING:|DRAIN_CAPTURE_FATAL)',
                          '\n'.join(x for x in lines if x != FIFO_WARNING), re.M), 'simulator/capture diagnostic')
    pass_name = {'original': 'DRAIN_SECTIONS_PASS', 'r2': 'DRAIN_SECTIONS_R2_PASS', 'shared': 'DRAIN_SHARED_FAULT_PASS'}[kind]
    require(lines.count(f'{pass_name} external={external} period={period} actual_sections=2 local_providers=2 native_endpoint=0 external_providers_qualified=0 reset_driven=0') == 1, 'unique scoped PASS')
    finishes = re.findall(r'\$finish called at time\s*:\s*(\d+) ps', sim)
    require(len(finishes) == 1, 'unique native finish at ps precision')
    finish = int(finishes[0])
    events = markers(sim, 'DRAIN_SECTIONS_EVENT')
    names = ['BEGIN_CLOSE', 'FORWARD_HELD', 'LOCAL_HELD']
    if external:
        names += ['LOCAL_PROOFS_PENDING']
    names += ['REPLACE_IDENTITY', 'STOPPED_NEW_ID', 'SECTIONS_RENEWED', 'AXIL_LOCAL_DRAINED', 'LOCAL_DRAINED']
    if external:
        names += ['READINESS_LOST', 'READINESS_REPROVEN', 'LATE_REQUEST_SOURCE_STOPPED', 'LATE_REQUEST_REVOKED', 'FAULT_RETAINED']
    if shared:
        names += ['SHARED_FAULT_MGMT_STOPPED', 'SHARED_FAULT_SOURCE_STOPPED', 'SHARED_FAULT_DELIVERED', 'SHARED_FAULT_RETAINED']
    names += ['TERMINAL']
    require([e['name'] for e in events] == names, 'ordered scenario inventory')
    require(all(int(a['ps']) < int(b['ps']) for a, b in zip(events, events[1:])), 'event times')
    require(all(int(e['period']) == period and int(e['external']) == external for e in events), 'event profile')
    by = {e['name']: e for e in events}
    require(finish == int(by['TERMINAL']['ps'])+1, 'terminal marker immediately precedes finish')
    number = lambda name, key: int(by[name][key], 16 if key in {'gen', 'tag', 'pg', 'pt'} else 2)
    require(number('LOCAL_HELD', 'sections') == 3 and number('LOCAL_HELD', 'idle') == 0 and
            number('LOCAL_HELD', 'pending') == 3 and number('LOCAL_HELD', 'complete') == 0, 'local replies block composed proof')
    require(number('AXIL_LOCAL_DRAINED', 'idle') == 1 and number('AXIL_LOCAL_DRAINED', 'pending') & 2 and
            not number('AXIL_LOCAL_DRAINED', 'complete'), 'independent AXI4 local hold')
    require(number('STOPPED_NEW_ID', 'enable') == 1 and not number('STOPPED_NEW_ID', 'sections') & 2 and
            not number('STOPPED_NEW_ID', 'complete'), 'stopped peer cannot renew')
    require(number('LOCAL_DRAINED', 'idle') == 3 and number('LOCAL_DRAINED', 'pending') == 0 and
            number('LOCAL_DRAINED', 'complete') == external, 'external contract gate')
    if external:
        require(min(int(by['LOCAL_DRAINED'][k]) for k in ('obsolete_l', 'obsolete_a')) >= 1, 'obsolete local mailboxes retired')
    buses = {s: [parse(l) for l in lines if l.startswith('AXI4_SECTION_BUS side='+s+' ')] for s in ('S', 'H', 'D')}
    histories, stats = {}, {}
    for side, rows in buses.items():
        histories[side], stats[side] = axi4_history(rows, side, external, shared,
            int(by['SHARED_FAULT_MGMT_STOPPED']['ps']) if shared else 0)
    for ch in FIELDS:
        raw = histories['D'][ch][:-1] if shared and ch == 'r' else histories['D'][ch]
        require(histories['H'][ch] == raw, 'raw CDC payload conservation')
        require(histories['S'][ch][:len(raw)] == raw, 'protected normal payload conservation')
    lite = markers(sim, 'FENCE_BUS')
    lite_stats = lite_history(lite)
    require(vcd.count('$var ') > 100, 'nonempty capture')
    root_match = re.search(r'\$scope module (\S+) \$end', vcd)
    require(root_match, 'VCD root')
    root = '/'+root_match[1]+'/'
    require(root_match[1].lstrip('\\').split('(')[0] ==
            {'original': 'tb_ntrpu_drain_sections', 'r2': 'tb_ntrpu_drain_sections_r2', 'shared': 'tb_ntrpu_drain_shared_fault'}[kind], 'correct VCD top')
    bus_lookup = {(s, r['t']): r for s, rs in buses.items() for r in rs}
    require(len(bus_lookup) == sum(map(len, buses.values())), 'unique AXI4 cycle samples')
    lite_lookup = {(r['side'], int(r['ps']), r['channel']): r for r in lite}
    require(len(lite_lookup) == len(lite), 'unique AXI-Lite handshakes')
    seen_bus, seen_lite = set(), set()
    event_index = 0
    mgmt_edges = positive_edges = complete_rises = 0
    fresh_low = 0
    identity = None
    last_fault = 0
    source_revoked = global_revoked = shared_source = shared_delivered = None
    host_edges, management_edges = [], []
    clocks = {n: [] for n in ('lite/host_clk', 'lite/device_clk', 'axi4/device_clk')}
    final = 0
    def covered_rows():
        last = None
        for row in vcd_rows(vcd):
            last = row
            yield row
        # VCD omits timestamps with no changes. The final two mark() calls
        # add four 1-ps delays without signal changes; bound this tail strictly.
        require(last is not None and 0 <= finish-last[0] <= 4, 'VCD reaches native finish')
        if last[0] < finish:
            yield finish, last[2], last[2]

    for t, before, after in covered_rows():
        final = t
        get = lambda name, state=after: state[root+name]
        rising = lambda name: get(name, before) == 0 and get(name) == 1
        for clock in clocks:
            if rising(clock):
                clocks[clock].append(t)
        if t >= int(by['BEGIN_CLOSE']['ps']):
            require(get('cold') == 1 and get('peer_cold') == 3 and
                    all(get(n) == 1 for n in ('lite/manager_reset_n', 'lite/hn', 'lite/dn',
                                             'axi4/host_rst_n', 'axi4/device_rst_n')), 'no operational reset during composed proof')
        if int(by['LOCAL_HELD']['ps']) <= t < int(by['SECTIONS_RENEWED']['ps']):
            require(get('replies_pending') == 3 and get('local_idle') == 0 and get('complete') == 0,
                    'local replies continuously retained across identity replacement')
        while event_index < len(events) and int(events[event_index]['ps']) <= t:
            e = events[event_index]
            qt = int(e['ps'])
            expected = {root+signal: int(e[k], 16 if k in {'gen', 'tag', 'pg', 'pt'} else 2) for k, signal in EVENT_MAP.items()}
            expected[root+'obsolete'] = int(e['obsolete_l']) | (int(e['obsolete_a']) << 32)
            if shared:
                expected.update({root+'mgmt_enable': int(e['mgmt_enable'], 2), root+'axi4_sources': int(e['asrc'], 2),
                                 root+'axi4_fault': int(e['local_axi4_fault'], 2)})
            candidates = [before, after] if qt == t else [before]
            require(any(all(s[k] == v for k, v in expected.items()) and
                        (not shared or (s[root+'axi4/fault_mgmt'] & 127) == int(e['asrc_mgmt'], 2)) for s in candidates),
                    'whole event/VCD snapshot: '+e['name'])
            event_index += 1
        for side, clock in [('S', 'host_clk'), ('H', 'host_clk'), ('D', 'device_clk')]:
            if rising('axi4/'+clock) and get('axi4/active', before):
                key = (side, t)
                require(key in bus_lookup, 'missing AXI4 cycle sample')
                row = bus_lookup[key]
                for ch, fields in FIELDS.items():
                    keys = [ch+'valid', ch+'ready']+[ch+f for f in fields] if row[ch+'valid'] else [ch+'valid', ch+'ready']
                    for key_field in keys:
                        require(get('axi4/'+side.lower()+'_'+key_field, before) == row[key_field],
                                'AXI4 VCD/log '+side+' '+key_field+' at '+str(t))
                require(get('axi4/phase', before) == row['phase'] and
                        get('axi4/'+('device_rst_n' if side == 'D' else 'host_rst_n'), before) == row['rst'], 'AXI4 phase/reset VCD')
                seen_bus.add(key)
        for side, clock, reset in [('U', 'host_clk', 'manager_reset_n'), ('H', 'host_clk', 'hn'), ('D', 'device_clk', 'dn')]:
            if rising('lite/'+clock) and get('lite/'+reset, before):
                for ch, fields in LITE_FIELDS.items():
                    prefix = 'lite/'+side.lower()+'_'+ch.lower()
                    if get(prefix+'valid', before) and get(prefix+'ready', before):
                        key = (side, t, ch)
                        require(key in lite_lookup, 'missing AXI-Lite handshake')
                        row = lite_lookup[key]
                        require(all(get(prefix+f, before) == int(row[f], 16) for f in fields), 'AXI-Lite VCD payload')
                        require(get('lite/shutdown_request', before) == int(row['close'], 2), 'AXI-Lite close marker')
                        seen_lite.add(key)
        if rising('axi4/host_clk'):
            host_edges.append(t)
        current_identity = (get('pg'), get('pt'))
        if current_identity != identity or not get('cold'):
            fresh_low = 0
            identity = current_identity
        if rising('mgmt_clk'):
            management_edges.append(t)
            mgmt_edges += 1
            if get('cold', before) and get('req', before) and (get('pg', before), get('pt', before)) == identity:
                fresh_low |= get('proof_req', before) & (~get('ack', before) & 127)
            if get('complete'):
                positive_edges += 1
        if get('cold'):
            faults = get('retained_faults')
            require(faults | last_fault == faults, 'retained management fault monotonicity')
            last_fault = faults
            if not external:
                require(get('complete') == 0 and not (get('proof_req') & 4 or get('ack') & 4), 'absent XDMA contract always blocks')
        if get('complete'):
            require(external and get('cold') == get('req') == 1 and all(get(n) == 127 for n in ('present', 'ready', 'ack', 'accepted')) and
                    all(get(n) == 0 for n in ('retained_faults', 'fault', 'revoked', 'replay')), 'every complete state is authorized')
            require(get('generation') == get('pg') == get('cg') and get('tag') == get('pt') == get('ct') and fresh_low == 127,
                    'current identity and seven independent low challenges')
            for i in range(7):
                require((get('ack_gen') >> (64*i)) & ((1 << 64)-1) == get('pg') and
                        (get('ack_tag') >> (64*i)) & ((1 << 64)-1) == get('pt'), 'per-slot ACK identity')
            if not get('complete', before):
                complete_rises += 1
                require(get('local_idle') == 3 and get('requests_present') == get('replies_pending') == 0, 'local drain at complete assertion')
        for scope in ('local_lite/', 'local_axi4/'):
            if get(scope+'peer_answer'):
                require(get(scope+'observed_idle') == get(scope+'peer_current') == 1 and
                        get(scope+'source_faults') == get(scope+'revoked_source') == 0 and get(scope+'quiet') == 3 and
                        get(scope+'saved_generation') == get(scope+'peer_generation') and get(scope+'saved_tag') == get(scope+'peer_tag'), 'valid local peer answer')
        if get('local_revoked') is not None and get('local_revoked') & 2 and source_revoked is None:
            source_revoked = t
        if get('retained_faults') is not None and get('retained_faults') & 64 and global_revoked is None:
            global_revoked = t
        # Ignore configuration-time X values only while the composed bank
        # remains in cold initialization. Active fault observations stay strict.
        if shared and get('cold') == 1:
            if get('axi4_sources') & 2 and shared_source is None:
                shared_source = t
            if get('axi4_fault') & 1 and shared_delivered is None:
                shared_delivered = t
            require(not (get('local_revoked') & 2 or get('axi4_fault') & 2), 'shared fault not hidden by own revocation')
    require(seen_bus == set(bus_lookup) and seen_lite == set(lite_lookup), 'all logged bus observations witnessed in VCD')
    require(event_index == len(events) and 0 <= final-int(by['TERMINAL']['ps']) <= 2, 'complete capture through terminal')
    lo, hi = int(by['REPLACE_IDENTITY']['ps']), int(by['STOPPED_NEW_ID']['ps'])
    require(hi-lo >= 480000 and not any(lo < t < hi for t in host_edges), 'stopped host clock across identity replacement')
    for clock, times in clocks.items():
        step = 7000 if clock == 'lite/host_clk' else period
        offset = 700 if clock == 'lite/host_clk' else 1100 if clock == 'lite/device_clk' else 600
        require(times and times[0] == offset+step//2 and all(b-a == step for a, b in zip(times, times[1:])),
                'continuous clock grid '+clock)
        require(0 <= finish-times[-1] < step, 'clock captured through finish '+clock)
    for a, b in zip(host_edges, host_edges[1:]):
        if not (a < lo and b > hi):
            require(b-a == 7000, 'AXI4 host edges outside declared stop')
    for a, b in zip(management_edges, management_edges[1:]):
        if shared and a < int(by['SHARED_FAULT_MGMT_STOPPED']['ps']) < int(by['SHARED_FAULT_SOURCE_STOPPED']['ps']) < b:
            continue
        require(b-a == 10000, 'management clock grid')
    latencies = {}
    if external:
        require(complete_rises == 2 and positive_edges > 0, 'initial and fresh readiness proof')
        require(source_revoked is not None and global_revoked is not None and 0 < global_revoked-source_revoked <= 30000,
                'bounded three-stage local fault crossing')
        require(not any(source_revoked < t for t in host_edges), 'fault delivered without another host edge')
        latencies['own_revocation_ps'] = global_revoked-source_revoked
    else:
        require(complete_rises == positive_edges == 0, 'no complete proof with missing contract')
    if shared:
        require(shared_source is not None and shared_delivered is not None and global_revoked == shared_delivered, 'shared fault observed by joiner')
        stop = int(by['SHARED_FAULT_MGMT_STOPPED']['ps'])
        resume = int(by['SHARED_FAULT_SOURCE_STOPPED']['ps'])
        require(stop < shared_source < resume and not any(stop < t < resume for t in management_edges), 'registered source while management stopped')
        require(not any(shared_source < t for t in host_edges) and 0 < shared_delivered-resume <= 30000, 'shared fault crosses without another host edge')
        require(number('SHARED_FAULT_RETAINED', 'fault') & 80 == 80, 'both section/local fault retained after new identity')
        latencies['shared_delivery_after_resume_ps'] = shared_delivered-resume
    return dict(status='DRAIN_SECTION_TRANSACTION_VCD_PASS', kind=kind, external=external, period_ps=period,
                axi4=stats, axil=lite_stats, events=len(events), mgmt_edges=mgmt_edges, positive_edges=positive_edges,
                complete_rises=complete_rises, latency_ps=latencies, final_ps=final, native_endpoint=False,
                external_providers_qualified=False, reset_driven=False, physical_qualification=False)
