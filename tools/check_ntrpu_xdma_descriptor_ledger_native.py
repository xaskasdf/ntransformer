"""Correlate actual fixture descriptor bytes, native status and ledger outputs."""
from check_ntrpu_xdma_local_master_native import review as review_master
from check_ntrpu_xdma_stop_inventory_r4 import require, markers, vcd_rows


def review_ledger(sim, vcd, native):
    p = '/board/ntrpu_stop_probe/'
    require(markers(sim, 'XDMA_LEDGER_NATIVE_PASS') == [dict(
        t=str(native['finish_ps']), publications='2', done_events='2', revoked='1',
        native_xdma='1', driver_closure='0', global_drain='0', reset_tested='0')], 'native ledger bounded scope')
    images = []
    for source, dest in [(0x400, 0x1000), (0x1000, 0x600)]:
        raw = bytearray(32)
        for offset, value in [(0, 0xad4b0013), (4, 64), (8, source), (16, dest)]:
            raw[offset:offset+4] = value.to_bytes(4, 'little')
        images.append(int.from_bytes(raw, 'little'))
    active = closed = faults = started = 0
    published, completed, previous_run = [0, 0], [0, 0], [0, 0]
    publications, done, arm_times, revoked, accounted_times = [], [[], []], [], [], []
    held_samples = {'h2c': 0, 'c2h': 0}
    checkpoints = {int(row['t']): row for row in markers(sim, 'XDMA_LEDGER_ACCOUNTED')}
    require(len(checkpoints) == 1, 'one logged descriptor accounting point')
    csr = {row['response_ps']: row for row in native['local_reads'] if row['address'] in (0x48, 0x1048)}
    csr_compared = []
    seen_checkpoints = 0
    for t, before, after in vcd_rows(vcd):
        if t < native['start_ps']:
            continue
        a = {name[len(p):]: value for name, value in after.items() if name.startswith(p)}
        b = {name[len(p):]: value for name, value in before.items() if name.startswith(p)}
        observed = [n for n in a if n.startswith('ledger_') and n not in ('ledger_h2c_image', 'ledger_c2h_image')]
        require(all(a[n] is not None for n in observed), 'known ledger interface')
        require(a['ledger_ready'] == 1, 'native readiness remains asserted')
        edge = before['/board/EP/user_clk'] == 0 and after['/board/EP/user_clk'] == 1
        if edge:
            if b['ledger_arm_valid'] and b['ledger_arm_ready']:
                require(not active and not arm_times, 'one native ledger session')
                active = 1
                arm_times.append(t)
            if b['ledger_publish_valid']:
                c = b['ledger_publish_channel']
                require(active and published[c] == 0 and b['ledger_publish_index'] == 0, 'one owned descriptor per channel')
                require(b['ledger_publish_address'] == (0x100 if c == 0 else 0x200), 'actual descriptor address')
                require(b['ledger_publish_image'] == images[c] == b['ledger_' + ('h2c' if c == 0 else 'c2h') + '_image'],
                        'notification contains actual complete host descriptor bytes')
                require(not (b['h2c_status'] & 0x41 or b['c2h_status'] & 0x41), 'publication precedes engine enable')
                published[c] += 1
                publications.append(dict(t=t, channel=c, index=0, address=b['ledger_publish_address'],
                                         image=images[c], generation=31, tag=301))
            for c, status in enumerate(('h2c_status', 'c2h_status')):
                run = (b[status] >> 6) & 1
                if run and not previous_run[c]:
                    require(published[c] == 1 and not (started & (1 << c)) and not closed, 'one published native Run lifetime')
                    started |= 1 << c
                if b[status] & 8:
                    require(active and started & (1 << c) and completed[c] < published[c], 'Done requires owned native descriptor')
                    completed[c] += 1
                    done[c].append(t)
                previous_run[c] = run
            if closed and (not b['ledger_close_request'] or not b['ledger_host_closed'] or
                           (b['ledger_host_generation'], b['ledger_host_tag']) != (31, 301)):
                if not faults:
                    revoked.append(t)
                faults |= 256
            if active and b['ledger_close_request'] and b['ledger_host_closed']:
                require((b['ledger_host_generation'], b['ledger_host_tag']) == (31, 301), 'current fixture closure identity')
                closed = 1
        require(a['ledger_active'] == active and a['ledger_closed'] == closed, 'retained native ledger lifecycle')
        require(a['ledger_faults'] == faults and a['ledger_started'] == started, 'native ledger faults and Run association')
        require((a['ledger_generation'], a['ledger_tag']) == ((31, 301) if active else (0, 0)), 'retained ledger identity')
        for c in (0, 1):
            for field, wanted in [('published', published[c]), ('completed', completed[c]),
                                  ('outstanding', published[c]-completed[c])]:
                require((a['ledger_' + field] >> (32*c)) & 0xffffffff == wanted, 'event-derived native ' + field)
            if published[c]:
                require(a['ledger_' + ('h2c' if c == 0 else 'c2h') + '_image'] == images[c], 'published descriptor bytes never change')
        idle = not (a['h2c_status'] & 0x41 or a['c2h_status'] & 0x41)
        accounted = int(active and closed and not faults and a['ledger_close_request'] and a['ledger_host_closed'] and
                        (a['ledger_host_generation'], a['ledger_host_tag']) == (31, 301) and idle and
                        not a['ledger_publish_valid'] and not (a['h2c_status'] & 8 or a['c2h_status'] & 8) and published == completed)
        require(a['ledger_accounted'] == accounted, 'continuous descriptor-only accounting predicate')
        require(a['ledger_arm_ready'] == int(not active and not faults and idle), 'native ledger arming predicate')
        if accounted:
            accounted_times.append(t)
        for c, name in enumerate(('h2c', 'c2h')):
            if a['phase'] == (3 if c == 0 else 5) and a['hold_dma_' + ('b' if c == 0 else 'r')] and published[c] > completed[c]:
                require(not accounted, 'held native descriptor cannot be retired')
                held_samples[name] += 1
        if t in csr:
            row = csr[t]
            c = int(row['address'] == 0x1048)
            require(row['value'] == completed[c], 'native CSR count agrees with observed Done obligations')
            csr_compared.append(dict(channel=c, t=t, count=completed[c]))
        if t in checkpoints:
            row = checkpoints[t]
            require(accounted and a['ledger_published'] == int(row['published'], 16) ==
                    a['ledger_completed'] == int(row['completed'], 16) == 0x100000001,
                    'accounting checkpoint matches trace')
            require(row['global_drain'] == '0' and int(row['host_bytes']) == a['host_bytes'] == 64,
                    'fixture receipt does not claim global drain')
            seen_checkpoints += 1
    parsed = [{k: int(v, 10 if k == 't' else 2 if k == 'channel' else 16) for k, v in row.items()}
              for row in markers(sim, 'XDMA_LEDGER_PUBLISH')]
    require(parsed == publications and [r['channel'] for r in publications] == [0, 1], 'publication log/VCD correspondence')
    require(done == [native['descriptor_done_ps']['h2c'], native['descriptor_done_ps']['c2h']], 'native Done event correspondence')
    require([len(rows) for rows in done] == [1, 1] and len(arm_times) == len(revoked) == seen_checkpoints == 1,
            'complete native ledger event coverage')
    require(len(csr_compared) == len(csr) and all(n > 20 for n in held_samples.values()), 'CSR and retained descriptor observations')
    require(accounted_times and revoked[0] > accounted_times[-1] and faults == 256 and not accounted and active,
            'approval revoked permanently while identity and counts remain retained')
    for publication in publications:
        fetch = next(row for row in native['native']['host_reads'] if row['address'] == publication['address'])
        require(publication['t'] < fetch['request_ps'] < fetch['completion_ps'] < done[publication['channel']][0],
                'published bytes fetched through native PCIe before descriptor completion')
    return dict(publications=publications, done_ps=done, csr_comparisons=csr_compared,
                armed_ps=arm_times[0], held_observation_samples=held_samples,
                first_accounted_ps=accounted_times[0], revocation_fault_ps=revoked[0],
                final_faults=faults, descriptor_bytes_verified=64, fixture_closure_only=True,
                publication_to_done_ps=[done[c][0]-publications[c]['t'] for c in (0, 1)])


def review(sim, vcd):
    native = review_master(sim, vcd)
    ledger = review_ledger(sim, vcd, native)
    native.update(status='PASS_DESCRIPTOR_LEDGER_NATIVE_XDMA_PUBLIC_EVENTS', ledger=ledger,
                  scope='Two immutable host-fixture descriptor publications, native PCIe fetch bytes, Done events and CSR counts correlated under retained DMA responses; fixture closure revoked. No real driver/CDC closure, global posted/BAR inventory, operational reset, physical timing or inference approval.')
    return native
