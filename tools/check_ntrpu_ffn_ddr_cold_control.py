"""ABI2 lifecycle for three cold MIG FFNs with cumulative counters.

The caller independently checks DDR/host bytes, regions, clocks and reset.
"""
from check_ntrpu_ffn_host import records
from check_ntrpu_ffn_irq_async import expected_abi2_writes as legacy_writes

CAPACITY = 128 * 1024**3

def region_base(ident, buffer):
    if ident == 3 and buffer == 0:
        return 2**32 - 4096
    if ident == 3 and buffer == 7:
        return CAPACITY - 4160
    return 2**32 + ((buffer + ident - 1) % 4) * 32 * 1024**3 + (buffer // 4) * 1024**2

def expected_abi2_writes(ident, dim, hidden, scenario, tiles):
    rows = legacy_writes(ident, dim, hidden, scenario, tiles)
    # Replace only the eight address/base pairs in the initial descriptor.
    # Post-START shadow poisoning and all service/ACK checks remain intact.
    for buffer in range(8):
        base = region_base(ident, buffer)
        for field, value in enumerate([base + 4032, base]):
            for half in range(2):
                index = 7 + buffer * 7 + field * 2 + half
                addr, _, resp = rows[index]
                rows[index] = (addr, (value >> (32 * half)) & 0xffffffff, resp)
    return rows

def review_control(text, seed, host_period_ps, result):
    errors = result['errors']
    def need(ok, why):
        if not ok: errors.append(why)
    fixture = None
    specs = [(32,32,0,0)]*3
    tiles = (2,2,2)
    n = 3
    ids = finished = list(range(1,4))
    aborted = []
    host = records(text, 'NTRPU_FFN_HOST_CASE')
    writes = records(text, 'NTRPU_FFN_HOST_WRITE')
    reads = records(text, 'NTRPU_FFN_IRQ_READ')
    memory = {c['id']: c for c in result['cases']}
    events = {name: records(text, 'NTRPU_FFN_IRQ_'+name) for name in
              ['BEGIN', 'JOB', 'DONE', 'EVENT', 'EDGE', 'ACK', 'SERVICE_REQUEST', 'SERVICE', 'RETIRE', 'END']}
    for name in ['BEGIN', 'JOB']:
        need([r.get('id') for r in events[name]] == ids, 'one '+name+' per admitted job')
    for name in ['DONE', 'EVENT', 'SERVICE_REQUEST', 'SERVICE', 'RETIRE', 'END']:
        need([r.get('id') for r in events[name]] == finished, 'one '+name+' per non-aborted job')
    need([r.get('id') for r in host] == ids, 'host completion/reset order')
    for rows in [reads, writes, events['EDGE'], events['ACK']]:
        need(all(r.get('id') in ids for r in rows), 'unknown case identity')
    need([w.get('order') for w in writes] == [i % 3 for i in range(len(writes))], 'AW/W skew schedule')
    for name in ['JOB', 'DONE', 'EVENT', 'SERVICE', 'RETIRE']:
        for r in events[name]:
            need(r.get('time_ps', 0) % 5000 == result['device_phase_ps'], 'source edge '+name)
    for name in ['EDGE', 'ACK']:
        for r in events[name]:
            need((r.get('time_ps', 0)-700) % host_period_ps == host_period_ps//2, 'host edge '+name)
    accepted = completed = 0
    timings = []
    for ident, spec in enumerate(specs, 1):
        dim, hidden, scenario, target = spec
        mode = (ident-1) % 3
        ws = [w for w in writes if w.get('id') == ident]
        rs = [r for r in reads if r.get('id') == ident]
        er = {name: [r for r in rows if r.get('id') == ident] for name, rows in events.items()}
        hs = [r for r in host if r.get('id') == ident]
        need(len(hs) == 1, 'one host outcome '+str(ident))
        expected = expected_abi2_writes(ident, dim, hidden, scenario, tiles)
        actual = [(r.get('addr'), r.get('data'), r.get('resp')) for r in ws]
        if scenario == 4:
            # A real in-flight reset can interrupt post-START shadow poisoning.
            # It must preserve an exact ordered prefix, including all descriptor
            # fields and the accepted START (first64 writes), with no later ACK.
            need(64 <= len(actual) <= len(expected) and actual == expected[:len(actual)],
                 'reset-aborted MMIO prefix '+str(ident))
        else:
            need(actual == expected, 'complete ABI2 MMIO write trace '+str(ident))
        need(len(rs) >= 6, 'initial MMIO read count')
        accepted = completed = ident - 1
        init = [(r.get('addr'), r.get('data')) for r in rs[:6]]
        need(init[:5] == [(0, 0x4652544e), (4, 0x20000), (8, 15), (0x18, accepted), (0x1c, completed)], 'ABI2 identity/acceptance counters')
        if len(init) == 6:
            need(init[5][0] == 0xc and init[5][1] & 0x187 == 0x100, 'ready for START')
        for name in ['BEGIN', 'JOB', 'DONE', 'EVENT', 'SERVICE', 'RETIRE']:
            for r in er[name]:
                need((r.get('sequence'), r.get('epoch')) == (0xfffffffe, ident+600), 'owner identity '+name)
        for r in er['BEGIN']:
            need(r.get('mode') == mode and r.get('fault') == 0, 'IRQ mode or injected defect')
        if len(er['BEGIN']) != 1 or len(er['JOB']) != 1 or len(hs) != 1:
            continue
        begin, job, h = er['BEGIN'][0], er['JOB'][0], hs[0]
        need(job['time_ps'] > begin['time_ps'], 'START/compute acceptance order')
        if scenario == 4:
            need(h == dict(id=ident, reset=1), 'reset abort cannot publish a completion')
            need(all(not er[name] for name in ['DONE', 'EVENT', 'EDGE', 'ACK', 'SERVICE', 'RETIRE', 'END']), 'aborted computation has no notification/retirement')
            later = [a['time_ps'] for a in reset_assert if a['time_ps'] > job['time_ps']]
            need(bool(later), 'in-flight reset after acceptance')
            accepted = completed = 0
            continue
        accepted = completed = ident
        if any(len(er[name]) != 1 for name in ['DONE', 'EVENT', 'SERVICE_REQUEST', 'SERVICE', 'RETIRE', 'END']):
            continue
        done, event, request, service, retire, end = [er[k][0] for k in ['DONE', 'EVENT', 'SERVICE_REQUEST', 'SERVICE', 'RETIRE', 'END']]
        mem = memory.get(ident, {})
        status = 6 if scenario in (1, 2) else 7 if scenario == 3 else 1 if scenario == 5 else 0
        stages = 4 if scenario == 0 else 0 if scenario == 5 else target
        need(h.get('reset') == 0 and h.get('status') == status and h.get('stages') == stages and h.get('flags') == 0, 'host result classification')
        need((h.get('sequence'), h.get('epoch')) == (0xfffffffe, ident+600), 'host completion identity')
        need(done.get('status') == status and done.get('stages') == stages, 'actual compute result classification')
        need(h.get('bytes') == done.get('bytes') == mem.get('down_bytes') and
             h.get('stage_bytes') == done.get('stage_bytes') == mem.get('bytes'), 'physical/compute/CSR byte agreement')
        need(h.get('cancel') == int(scenario == 3), 'cancel coverage')
        need(done['time_ps']-job['time_ps'] == h.get('compute_cycles', -1)*5000, 'hardware elapsed count vs compute handshakes')
        need(job['time_ps'] < done['time_ps'] < event['time_ps'] < request['time_ps'] <= service['time_ps'] < retire['time_ps'] < end['time_ps'], 'completion/event/service/retirement ordering')
        need(event.get('enable') == int(mode != 2) and end.get('mode') == mode, 'captured IRQ mode')
        need(0 < end.get('retire_polls', 0) < 128, 'bounded observed retirement polls')
        need(h.get('writes') == len(ws), 'MMIO write count')
        need(h.get('reads') == len(rs) == h.get('polls', -999)+19+end['retire_polls']+int(mode == 1), 'MMIO read count')
        need(h.get('polls', 0) > 0, 'completion polling occurred')
        need(0 < h.get('compute_cycles', 0) < h.get('total_cycles', 0) < (250000 if fixture is None else 1000000), 'device elapsed range')
        need(0 < h.get('program_cycles', 0) and h['program_cycles']+h.get('poll_cycles', 0) < h['total_cycles'], 'host lifecycle duration')
        need(h['total_cycles'] > mem.get('cycles', 0), 'consumer finishes before retirement')
        p = h.get('polls', 0)
        middle = rs[6+p:6+p+13]
        addresses = list(range(0x40, 0x64, 4))+[0x64, 0x68, 0x18, 0x1c]
        need([r.get('addr') for r in middle] == addresses, 'metadata/owner/counter read order')
        if len(middle) == 13:
            values = {r['addr']: r['data'] for r in middle}
            need(values[0x40] & 15 == status and (values[0x40] >> 8) & 7 == stages, 'completion status register')
            need(values[0x44] == h['bytes'] and values[0x48] == h['stage_bytes'], 'byte register values')
            need(values[0x4c] == values[0x64] == 0xfffffffe and values[0x50] == values[0x68] == ident+600, 'returned and admitted identities')
            need(values[0x54] == values[0x60] == 0 and values[0x58]+(values[0x5c] << 32) == h['compute_cycles'], 'metadata diagnostic/elapsed register')
            need(values[0x18] == accepted and values[0x1c] == completed, 'exactly-once counters')
        polls = rs[6:6+p]
        need(len(polls) == p and all(r.get('addr') == 0xc for r in polls), 'poll addresses')
        if polls:
            need(polls[-1]['data'] & 4 and all(not r['data'] & 4 for r in polls[:-1]), 'first observed completion')
        trailing = rs[6+p+13:]
        need(all(r.get('addr') == 0xc for r in trailing), 'retirement status reads')
        if trailing:
            need(trailing[-1]['data'] & 0x187 == 0x100, 'idle after retirement')
            if mode == 1:
                need(trailing[0]['data'] & 0x24 == 0x24, 'software ACK retained completion')
        for r in rs:
            if r.get('addr') == 0xc:
                value = r['data']
                need(value & ~0x1ff == 0 and not value & 0x80, 'status reserved/fault bits')
                need(not (value & 4 and value & 3), 'completion cannot overlap next job')
        if mode == 2:
            need(not er['EDGE'] and not er['ACK'], 'polling requires no interrupt or transport ACK')
        else:
            need([r.get('level') for r in er['EDGE']] == [1, 0] and len(er['ACK']) == 1, 'one held IRQ and transport ACK')
            if len(er['EDGE']) == 2 and len(er['ACK']) == 1:
                rise, fall = [r['time_ps'] for r in er['EDGE']]
                ack = er['ACK'][0]['time_ps']
                need(event['time_ps'] < rise <= ack < fall < retire['time_ps'], 'host IRQ/ACK/drain order')
                need(fall > service['time_ps'], 'IRQ cannot fall before software service')
                need((ack < request['time_ps']) if mode == 0 else (ack > service['time_ps']), 'two acknowledgement orders')
        timings.append(dict(id=ident,mode=mode,compute_cycles=h['compute_cycles'],
                            event_to_retire_ps=retire['time_ps']-event['time_ps'],
                            service_to_retire_ps=retire['time_ps']-service['time_ps'],
                            event_to_irq_ps=(er['EDGE'][0]['time_ps']-event['time_ps']) if er['EDGE'] else None))
    for i in range(1, n):
        prev = [r for r in events['END'] if r.get('id') == i]
        nex = [r for r in events['BEGIN'] if r.get('id') == i+1]
        if prev and nex:
            need(prev[0]['time_ps'] < nex[0]['time_ps'], 'new job only after retirement')
    result.update(status='FAIL' if errors else 'PASS', host_cases=host, mmio_writes=len(writes),
                  mmio_reads=len(reads), irq_timings=timings, completed_ids=finished, aborted_ids=aborted,
                  host_period_ps=host_period_ps,
                  scope='Real FFN/HLS -> ABI2 CSR -> XPM IRQ transport with separate MMIO service and protocol host ACK; '+
                        'AMD asynchronous AXI4/AXI4-Lite, supervised cold admission and generic128GiB MIG/model. No real XDMA, warm-reset qualification, exact DIMM, physical board or full inference.')
    return result
