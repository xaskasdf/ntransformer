"""Independent transaction and certificate histories for actual-CDC fault cases."""
from collections import Counter, deque
import re
from check_ntrpu_axi4_drain import pattern, mask, require
from check_ntrpu_axi4_pair_r2 import FIFO_WARNING
from check_ntrpu_axi4_section_cancel import parse, FIELDS


def review(text, stage, channel, period):
    lines = text.splitlines()
    control = channel == 2
    require(stage in range(4) and channel in range(3) and (not control or stage == 3), 'case parameters')
    require(lines.count(FIFO_WARNING) == 5, 'five functional-only FIFO notices')
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|Error:|WARNING:|Warning:)',
                          '\n'.join(line for line in lines if line != FIFO_WARNING), re.M), 'runtime diagnostic')
    require(lines.count(f'AXI4_LIFETIME_PROFILE stage={stage} channel={channel} device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000') == 1, 'profile')
    require(lines.count(f'AXI4_LIFETIME_SUITE_PASS stage={stage} channel={channel} device_period_ps={period} normal_beats=16 native_endpoint=0') == 1, 'suite marker')
    events = [parse(line) for line in lines if line.startswith('AXI4_LIFETIME_EVENT ')]
    names = ['START', 'HEALTHY_HELD', 'STAGE_READY', 'REPLIES_RELEASE', 'TERMINAL']
    if stage: names += ['PROOF_REQUEST']
    if stage >= 2: names += ['NORMAL_DRAINED', 'LOCAL_HELD']
    if stage == 2: names += ['STAGE_HOST_STOP', 'STAGE_HOST_RESUME']
    if stage == 3: names += ['STAGE_DEVICE_STOP', 'STAGE_DEVICE_RESUME']
    if control:
        names += ['GENERATION_REPLACE', 'GENERATION_WAIT', 'PROOF_COMPLETE']
    else:
        names += ['INJECT_FAULT', 'DEVICE_FAULT_STOP', 'DEVICE_REPORTED', 'DEVICE_FAULT_RESUME',
                  'HOST_FAULT_STOP', 'HOST_REPORTED', 'FAULT_FRESH_REQUEST', 'FAULT_REQUEST_WAIT',
                  'HOST_FAULT_RESUME', 'HELD_AFTER_FAULT']
    require(Counter(e['name'] for e in events) == Counter(names), 'complete unique event inventory')
    require(all(a['t'] <= b['t'] for a, b in zip(events, events[1:])), 'monotonic events')
    by = {e['name']: e for e in events}
    require(by['START']['t'] < by['HEALTHY_HELD']['t'] <= by['STAGE_READY']['t'] < by['REPLIES_RELEASE']['t'] < by['TERMINAL']['t'], 'main lifetime order')
    buses = {side: [parse(line) for line in lines if line.startswith('AXI4_LIFETIME_BUS side=' + side + ' ')] for side in ['S', 'H', 'D']}
    host = [parse(line) for line in lines if line.startswith('AXI4_LIFETIME_HOST ')]
    mgmt = [parse(line) for line in lines if line.startswith('AXI4_LIFETIME_MGMT ')]
    boxes = {box: [parse(line) for line in lines if line.startswith('AXI4_LIFETIME_MB box=' + box + ' ')] for box in ['upstream', 'host', 'device']}
    pauses = []
    if stage == 2: pauses += [('H', 'STAGE_HOST_STOP', 'STAGE_HOST_RESUME')]
    if stage == 3: pauses += [('D', 'STAGE_DEVICE_STOP', 'STAGE_DEVICE_RESUME')]
    if not control: pauses += [('H', 'HOST_FAULT_STOP', 'HOST_FAULT_RESUME'), ('D', 'DEVICE_FAULT_STOP', 'DEVICE_FAULT_RESUME')]
    stats = {}; transfers = {}; snapshots = {}; orphan_offers = {}; times = {}
    for side, rows in buses.items():
        domain = 'H' if side == 'S' else side
        step, offset = (7000, 0) if domain == 'H' else (period, 600)
        times[side] = [row['t'] for row in rows]
        require(rows and all(a < b for a, b in zip(times[side], times[side][1:])), 'strict bus timestamps')
        first_phase = offset + step // 2
        require(times[side][0] == first_phase + ((by['START']['t'] - first_phase) // step + 1) * step, 'first bus observation')
        require(0 <= by['TERMINAL']['t'] - times[side][-1] < step, 'last bus observation')
        require(all((t - offset) % (step // 2) == 0 for t in times[side]), 'clock grid')
        stopped = [(by[a]['t'], by[b]['t']) for d, a, b in pauses if d == domain]
        for a, b in zip(times[side], times[side][1:]):
            if not any(a <= end and start <= b for start, end in stopped): require(b - a == step, 'missing clock observation')
        for start, end in stopped:
            require(start < end and not any(start < t < end for t in times[side]), 'clock advances while declared stopped')
        aw = deque(); w = deque(); reads = deque(); writes = deque(); partial = []
        counts = Counter(); stalls = Counter(); observed = {ch: [] for ch in FIELDS}
        state_at = {}; raw_orphans = []; previous = None; awseq = arseq = wseq = 0; ap = wp = 0
        first_aw = first_w = None
        for row in rows:
            when = row['t']
            require(row['rst'] == 1, 'no transport reset in fault lifetime bank')
            state_at[when] = dict(ap=ap, wp=wp, ar=len(reads), partial=len(partial), b=counts['b'], r=counts['r'])
            for ch, fields in FIELDS.items():
                require(row[ch+'valid'] in {0, 1} and row[ch+'ready'] in {0, 1}, 'known handshake')
                if row[ch+'valid']: require(all(row[ch+field] is not None for field in fields), 'known valid payload')
                if previous and previous[ch+'valid'] and not previous[ch+'ready']:
                    require(row[ch+'valid'] and all(row[ch+f] == previous[ch+f] for f in fields), side+' held '+ch+' changed')
                if row[ch+'valid'] and not row[ch+'ready']: stalls[ch] += 1
            for ch in ['b', 'r']:
                if not row[ch+'valid']: continue
                if row[ch+'id'] == 9:
                    require(not control and side != 'S' and channel == (ch == 'r'), 'orphan only on declared raw channel')
                    require(when >= by['INJECT_FAULT']['t'], 'orphan before external injection')
                    expected = (9, 3) if ch == 'b' else (pattern(0xbaddf00d), 9, 2, 1)
                    require(tuple(row[ch+f] for f in FIELDS[ch]) == expected, 'exact external orphan payload')
                    if side == 'H': require(row[ch+'ready'] == 0, 'orphan contained at raw host boundary')
                    raw_orphans.append(when)
                    if row[ch+'ready']: counts['orphan'] += 1
                    continue
                require(row[ch+'id'] == 3, 'only known healthy response ID')
                q = writes if ch == 'b' else reads
                require(q, side+' response lacks causal request')
                desc = q[0]
                local = desc['seq'] == 1
                if ch == 'b': require(row['bresp'] == (2 if local else 0), 'healthy B semantics')
                else:
                    require(row['rlast'] == (desc['beat'] == 15), 'R LAST association')
                    require(row['rresp'] == (2 if local else [0, 2, 3][desc['beat'] % 3]), 'healthy R semantics')
                    require(row['rdata'] == (0 if local else pattern(3*4096 ^ (desc['beat'] << 8) ^ 3)), 'full healthy R payload')
                if row[ch+'ready']:
                    counts[ch] += 1; observed[ch].append(tuple(row[ch+f] for f in FIELDS[ch]))
                    if ch == 'b': q.popleft(); ap -= 1; wp -= 1
                    else:
                        desc['beat'] += 1
                        if row['rlast']: q.popleft()
            for ch in ['aw', 'ar']:
                if not (row[ch+'valid'] and row[ch+'ready']): continue
                seq = awseq if ch == 'aw' else arseq
                require(seq < (2 if side == 'S' and stage >= 2 else 1), 'no extra forwarded requests')
                expected = [3, (0x100000000 if ch == 'aw' else 0x200000000)+3*4096, 15, 6, 3, 1, 15, 0, 3, 12]
                require([row[ch+f] for f in FIELDS[ch]] == expected, 'complete address attributes')
                require(row[ch+'addr'] % 4096 + 16*64 <= 4096, '4K burst boundary')
                counts[ch] += 1; observed[ch].append(tuple(row[ch+f] for f in FIELDS[ch]))
                if ch == 'aw':
                    first_aw = when if first_aw is None else first_aw
                    aw.append(dict(seq=seq)); awseq += 1; ap += 1
                else: reads.append(dict(seq=seq, beat=0)); arseq += 1
            if row['wvalid'] and row['wready']:
                first_w = when if first_w is None else first_w
                beat = len(partial)
                require(beat < 16 and row['wlast'] == (beat == 15), 'W LAST boundary')
                require(row['wdata'] == pattern(3*4096 ^ (beat << 8)) and row['wstrb'] == mask(beat, 6), 'full W data and strobe')
                partial.append((row['wdata'], row['wstrb'])); counts['w'] += 1
                observed['w'].append(tuple(row['w'+f] for f in FIELDS['w']))
                if row['wlast']: w.append(partial); partial = []; wp += 1; wseq += 1
            while aw and w:
                desc = aw.popleft(); beats = w.popleft(); require(len(beats) == 16, 'AW/W causal association'); writes.append(desc)
            previous = row
        factor = 2 if side == 'S' and stage >= 2 else 1
        expected_counts = dict(aw=factor, w=16*factor, ar=factor, b=factor, r=16*factor)
        if not control and side == 'D': expected_counts['orphan'] = 1
        require(dict(counts) == expected_counts, side+' complete transfer totals')
        require(not (aw or w or partial or reads or writes or ap or wp), 'terminal healthy queues empty')
        require(first_w < first_aw, 'real W before AW')
        require(bool(raw_orphans) == (not control and side != 'S'), 'orphan observed only at raw boundaries')
        stats[side] = dict(samples=len(rows), transfers=dict(counts), stalls=dict(stalls))
        transfers[side] = observed; snapshots[side] = state_at; orphan_offers[side] = raw_orphans
    for ch in FIELDS:
        require(transfers['H'][ch] == transfers['D'][ch], 'complete normal CDC conservation')
        require(transfers['S'][ch][:len(transfers['H'][ch])] == transfers['H'][ch], 'protected normal conservation')
    require(times['S'] == times['H'] == [row['t'] for row in host], 'complete host diagnostics')
    for row, raw, protected in zip(host, buses['H'], buses['S']):
        require((row['sbv'],row['srv']) == (protected['bvalid'],protected['rvalid']), 'protected offer diagnostics match actual bus')
        source = snapshots['S'][row['t']]; downstream = snapshots['H'][row['t']]
        require(row['cold'] == row['hrst'] == row['drst'] == 1, 'retention not reset')
        require((row['hap'], row['hwp'], row['har']) == (downstream['ap'], downstream['wp'], downstream['ar']), 'raw host retained inventory')
        if row['state'] != 2:
            require((row['ap'], row['wp'], row['arp'], row['partial']) == (source['ap'], source['wp'], source['ar'], source['partial']), 'protected response inventory')
        else:
            require(source['b'] >= 1 and source['r'] >= 16, 'REJECT only after all protected normal replies')
            require((row['ap'], row['wp'], row['arp'], row['partial']) == (0, 0, 0, 0), 'guard inventory excludes local replies')
            require(not any(raw[ch+'valid'] for ch in ['aw', 'w', 'ar']), 'no reopening after REJECT')
        detected = not control and orphan_offers['H'] and row['t'] > orphan_offers['H'][0]
        require(row['af'] == ((1 if channel == 0 else 2) if detected else 0), 'registered admission fault from real offer')
        require(row['hf'] == ((2 if channel == 0 else 4) if detected else 0), 'registered host fault from real offer')
        if row['proof_ack']: require(row['proof_req'] and row['proof_tag'] == row['drain_tag'] and row['state'] == 2 and not row['af'], 'fresh producer proof')
    require(mgmt and all(b['t']-a['t'] == 10000 for a, b in zip(mgmt, mgmt[1:])), 'complete management observations')
    require(mgmt[0]['t'] == 5001 + ((by['START']['t']-5001)//10000+1)*10000, 'first management observation')
    require(0 <= by['TERMINAL']['t']-mgmt[-1]['t'] <= 10000, 'last management observation')
    first_change = by['GENERATION_REPLACE']['t'] if control else by['FAULT_FRESH_REQUEST']['t']
    final_fault = 0 if control else (16641 if channel == 0 else 33282)
    last_fault = 0
    for row in mgmt:
        changed = row['t'] >= first_change
        require(row['gen'] == 0x100000001 + (0x100000001 if changed else 0), 'full-width generation timeline')
        require(row['tag'] == 0x200000001 + (0x100000001 if changed and not control else 0), 'full-width tag timeline')
        expected_req = (stage != 0 and row['t'] >= by['PROOF_REQUEST']['t']) or (not control and changed)
        require(row['req'] == expected_req, 'external request timeline')
        require((row['fm'] | last_fault) == row['fm'] and not (row['fm'] & ~final_fault), 'only declared sticky fault sources')
        last_fault = row['fm']
        require(row['ack'] == (row['uc'] and row['pc']), 'complete section proof')
        if not control: require(not row['ack'], 'fault case never obtains section certificate')
        if row['fm']: require(not (row['ack'] or row['uc'] or row['pc']), 'reported fault revokes certificates')
        if row['ack']:
            require(control and changed and row['state'] == 2 and row['fenced'] and not row['fm'], 'only fresh control generation certifies')
            require(not any(row[k] for k in ['ap','wp','arp','partial','hap','hwp','har','dap','dwp','dar']), 'complete section inventory empty at proof')
    for box, rows in boxes.items():
        require([row['t'] for row in rows] == [row['t'] for row in mgmt], 'complete mailbox observations')
        previous = None
        counted_obsolete = 0
        for row, m in zip(rows, mgmt):
            require(all(v is not None for v in row.values()), 'known mailbox fields')
            expected_req = m['req'] and not m['fm']
            if box != 'upstream': expected_req = expected_req and m['uc']
            if box == 'device': expected_req = expected_req and m['hc']
            require(row['req'] == expected_req and (row['gen'],row['tag'],row['kind']) == (m['gen'],m['tag'],1), 'mailbox current identity and gating')
            if row['ack']: require(row['req'] and (row['ack_gen'],row['ack_tag'],row['ack_kind']) == (m['gen'],m['tag'],1), 'no obsolete mailbox acknowledgement')
            if previous:
                if previous['send'] and not previous['rcv']:
                    require(row['send'] and (row['payload_gen'],row['payload_tag'],row['payload_kind']) == (previous['payload_gen'],previous['payload_tag'],previous['payload_kind']), 'issued mailbox transfer cannot be cleared or overwritten')
                require(previous['obsolete'] <= row['obsolete'] <= previous['obsolete']+1, 'obsolete completions retained')
                if control and box == 'device':
                    retired = previous['send'] and not row['send']
                    if retired:
                        require(previous['rcv'], 'issued transfer retires only after source receipt')
                        counted_obsolete += previous['payload_gen'] != row['gen']
                    require(row['obsolete'] == counted_obsolete, 'obsolete count matches retired old generation')
            previous = row
        if control: require(rows[-1]['obsolete'] == (box == 'device'), 'only old device generation retires obsolete')
    for index, m in enumerate(mgmt):
        u, h, d = [boxes[box][index] for box in ['upstream','host','device']]
        require(m['uc'] == (u['ack'] and not m['fm']) and m['hc'] == h['ack'] and m['dc'] == d['ack'], 'certificates bound to mailbox identities')
        require(m['pc'] == (h['ack'] and d['ack'] and not m['fm']), 'pair certificate binding')
    held, ready, terminal = by['HEALTHY_HELD'], by['STAGE_READY'], by['TERMINAL']
    require(held['sbs'] == 0 and held['srs'] == 15 and held['sbv'] == held['srv'] == 1 and held['ap'] == held['wp'] == held['arp'] == 1, 'healthy protected last replies present')
    require(ready['ack'] == ready['fm'] == 0, 'case starts without completed section proof or fault')
    if stage == 0: require(ready['req'] == 0 and ready['state'] == 0, 'fault precedes first management request')
    if stage == 1: require(ready['ub'] == ready['proof_req'] == 1 and ready['uc'] == ready['proof_ack'] == 0, 'fault during upstream proof')
    if stage == 2: require(ready['uc'] == ready['hb'] == 1 and ready['hc'] == 0 and ready['state'] == 2, 'fault during host pair proof')
    if stage == 3: require(ready['uc'] == ready['hc'] == ready['db'] == 1 and ready['dc'] == 0 and ready['state'] == 2, 'fault or replacement during device proof')
    for name in ['HELD_AFTER_FAULT'] if not control else ['PROOF_COMPLETE']:
        event = by[name]; require(event['sbv'] == event['srv'] == 1 and not event['idle'], 'healthy replies retained before final release')
        require(event['sbs'] == (0 if stage < 2 else 1) and event['srs'] == (15 if stage < 2 else 16), 'only intended healthy replies remain held')
    require(terminal['sbs'] == (1 if stage < 2 else 2) and terminal['srs'] == (16 if stage < 2 else 32), 'healthy terminal completions')
    require(terminal['fm'] == final_fault and terminal['sbv'] == terminal['srv'] == 0 and terminal['fenced'] == 1, 'terminal fault and response ownership')
    latencies = {}
    if control:
        require(terminal['ack'] == terminal['idle'] == 1, 'healthy generation control terminal')
        lo, hi = by['GENERATION_REPLACE']['t'], by['STAGE_DEVICE_RESUME']['t']
        require(lo < by['GENERATION_WAIT']['t'] <= hi and by['GENERATION_WAIT']['t']-lo >= 70000, 'sustained stopped-device generation replacement')
        selected = [row for row in boxes['device'] if lo <= row['t'] < hi]
        require(len(selected) >= 7 and all(row['send'] == row['busy'] == 1 and row['rcv'] == row['ack'] == 0 and row['payload_gen'] == 0x100000001 and row['gen'] == 0x200000002 for row in selected), 'old issued generation retained until peer resumes')
        proof = by['PROOF_COMPLETE']; sampled = next(row for row in reversed(mgmt) if row['t'] <= proof['t'])
        require(proof['ack'] == 1 and (proof['gen'],proof['tag']) == (sampled['gen'],sampled['tag']) == (0x200000002,0x200000001), 'new generation confirmed by actual management samples')
    else:
        require(terminal['ack'] == terminal['idle'] == 0 and terminal['state'] == (1 if stage < 2 else 2), 'quarantine is not idle or reopened')
        for side, name, bit in [('D','DEVICE_FAULT_STOP',14 if channel == 0 else 15), ('H','HOST_FAULT_STOP',0 if channel == 0 else 1)]:
            require(by[name]['t'] == orphan_offers[side][0]+1, 'stop immediately after registered real bus fault')
            found = next(row for row in mgmt if row['fm'] & (1 << bit))
            latencies[side] = found['t']-by[name]['t']; require(0 <= latencies[side] <= 30001, 'fault crosses without another source edge')
        require(by['DEVICE_REPORTED']['t'] < by['FAULT_FRESH_REQUEST']['t'] < by['FAULT_REQUEST_WAIT']['t'] <= by['HOST_FAULT_RESUME']['t'], 'fault and fresh-request exercise order')
        require(by['FAULT_REQUEST_WAIT']['t']-by['FAULT_FRESH_REQUEST']['t'] >= 190000, 'sustained fresh request rejected after fault')
        require(by['FAULT_REQUEST_WAIT']['ack'] == 0 and by['FAULT_REQUEST_WAIT']['fm'] == final_fault, 'new identity cannot evade retained fault')
        if stage == 0: require(all(not row['busy'] for rows in boxes.values() for row in rows), 'no mailbox launch after pre-request fault')
        if stage == 1: require(boxes['upstream'][-1]['busy'] == boxes['upstream'][-1]['send'] == 1, 'unacknowledged upstream proof retained after fault')
        if stage == 3: require(boxes['device'][-1]['busy'] == boxes['device'][-1]['send'] == 1, 'unacknowledged device proof retained after fault')
    return dict(status='PASS_ACTUAL_AXI4_FAULT_LIFETIME' if not control else 'PASS_BUSY_GENERATION_REPLACEMENT',
                stage=stage, channel=channel, device_period_ps=period, profiles=stats, normal_write_beats=16, normal_read_beats=16,
                local_write_beats=16 if stage >= 2 else 0, local_read_beats=16 if stage >= 2 else 0,
                fault_reporting_latency_ps=latencies, final_fault=final_fault,
                terminal_mailbox_busy={box: rows[-1]['busy'] for box, rows in boxes.items()},
                obsolete_completions={box: rows[-1]['obsolete'] for box, rows in boxes.items()},
                physical_timing_qualified=False, native_endpoint_qualified=False,
                scope='Real external orphan B/R at four proof stages or healthy busy-generation control; actual AMD CDC and unchanged retained production RTL.')
