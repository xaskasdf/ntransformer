"""Reconstruct native H2C seed, retained C2H R, posted CQ receipt and host memory."""
import re
from check_ntrpu_xdma_stop_inventory_r4 import vcd_rows, markers, require


def review(sim, vcd):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|Error:|ERROR:|CRITICAL WARNING:)', sim, re.M), 'simulator failure')
    unique = {}
    for name in ['BEGIN', 'SEEDED', 'PENDING', 'RELEASE', 'HOST_BEGIN', 'HOST_END', 'PASS']:
        rows = markers(sim, 'XDMA_C2H_'+name)
        require(len(rows) == 1, 'unique '+name)
        unique[name] = rows[0]
    start, seeded, pending, release, host_begin, host_end, finish = [int(unique[n]['t']) for n in unique]
    require(90000000 < start < seeded < pending < release < host_begin < host_end < finish < 200000000, 'bounded causality')
    require(unique['BEGIN']['dma_bar'] == '1' and unique['BEGIN']['payload_bytes'] == '64', 'initial scope')
    require({k: unique['PASS'][k] for k in ['payload_bytes', 'host_packets', 'reset_tested', 'global_drain_tested']} ==
            dict(payload_bytes='64', host_packets='1', reset_tested='0', global_drain_tested='0'), 'final scope')
    require('$finish called at time' in sim, 'simulator finish')
    require({k: unique['PENDING'][k] for k in ['ar', 'r', 'host_bytes']} == dict(ar='1', r='0', host_bytes='0'), 'pending inventory')
    regs = markers(sim, 'XDMA_C2H_REG')
    expected = [(1, 0x40, 0), (1, 0x1040, 0), (1, 0x40, 0), (1, 0x48, 1),
                (1, 4, 0xfffe7e), (2, 0x1040, 1), (2, 0x1048, 0)]
    for phase, busy in [(3, 1), (3, 1), (4, 0), (4, 0)]:
        expected += [(phase, 0x1004, 0xfffe7e), (phase, 0x1040, busy), (phase, 0x1048, 1-busy),
                     (phase, 4, 0xfffe7e), (phase, 0x40, 0), (phase, 0x48, 1)]
    require(len(regs) == len(expected), 'all register observations')
    for row, (phase, address, value) in zip(regs, expected):
        actual = int(row['value'], 16)
        require(int(row['phase']) == phase and int(row['address'], 16) == address, 'register order')
        require((actual & 1 == value and actual & ~0x47 == 0) if address in (0x40, 0x1040) else actual == value,
                'native status/control/count')
    writes = [(int(r['phase']), int(r['address'], 16), int(r['value'], 16)) for r in markers(sim, 'XDMA_C2H_WRITE')]
    require(writes == [(1, 0x4080, 0x100), (1, 0x4084, 0), (1, 0x4088, 0), (1, 4, 0xfffe7f), (1, 0xc, 1),
                      (2, 0x5080, 0x200), (2, 0x5084, 0), (2, 0x5088, 0), (2, 0x1004, 0xfffe7f), (3, 0x100c, 1)],
            'native control writes')
    ep, tb, rp = '/board/EP/', '/board/ntrpu_stop_probe/', '/board/RP/'
    fields = {'AW': ['awid', 'awaddr', 'awlen', 'awsize', 'awburst'], 'W': ['wdata', 'wstrb', 'wlast'],
              'B': ['bid', 'bresp'], 'AR': ['arid', 'araddr', 'arlen', 'arsize'], 'R': ['rid', 'rdata', 'rresp', 'rlast']}
    accepted = {k: [] for k in fields}
    packets = {k: [] for k in ['s_axis_rq', 'm_axis_rc', 'm_axis_cq', 's_axis_cc']}
    queries = {int(r['t']): r for r in regs}
    require(len(queries) == len(regs), 'unique register times')
    qtimes, qi, phases, held, last = sorted(queries), 0, [], 0, 0
    retained = None
    memory, written, host_count, in_host = bytearray(64), 0, 0, False
    host_beats, memory_events = [], []
    for t, before, after in vcd_rows(vcd):
        last = t
        while qi < len(qtimes) and qtimes[qi] <= t:
            qt = qtimes[qi]; row = queries[qt]
            samples = [before, after] if qt == t else [before]
            require(any(s[rp+'tx_usrapp/P_READ_DATA'] == int(row['value'], 16) and s[tb+'phase'] == int(row['phase']) for s in samples),
                    'register log/VCD identity')
            qi += 1
        if t < start:
            continue
        require(all(after[n] == 1 for n in ['/board/sys_rst_n', ep+'user_resetn', ep+'user_lnk_up', rp+'user_lnk_up']),
                'continuous native readiness')
        phase = after[tb+'phase']; require(phase in (1, 2, 3, 4, 5), 'known phase')
        if not phases or phases[-1] != phase:
            phases.append(phase)
        require(after[tb+'hold_dma_b'] == after[tb+'hold_bypass_b'] == 0 and after[tb+'hold_dma_r'] == int(phase < 4),
                'public response delay controls')
        if pending <= t < release:
            require(after[ep+'ntrpu_app_dma_rvalid'] == 1 and after[ep+'ntrpu_app_dma_rready'] == 0 and
                    after[ep+'m_axi_rvalid'] == 0, 'continuously retained application R')
            payload = [after[ep+'m_axi_'+n] for n in fields['R']]
            require(None not in payload, 'known retained R payload')
            if retained is None:
                retained = payload
            require(retained == payload, 'stable retained R payload')
        if before[ep+'user_clk'] == 0 and after[ep+'user_clk'] == 1:
            if pending <= t < release:
                held += 1
            for p in ['m_axib_aw', 'm_axib_w', 'm_axib_b', 'm_axib_ar', 'm_axib_r']:
                require(not (before[ep+p+'valid'] == 1 and before[ep+p+'ready'] == 1), 'no BAR data transfers')
            for channel, names in fields.items():
                p = ep+'m_axi_'; low = channel.lower()
                require(before[p+low+'valid'] in (0, 1) and before[p+low+'ready'] in (0, 1), 'known AXI handshake')
                if before[p+low+'valid'] == before[p+low+'ready'] == 1:
                    row = [t]+[before[p+n] for n in names]
                    require(None not in row, 'known accepted AXI payload')
                    accepted[channel].append(row)
        if before[rp+'user_clk'] == 0 and after[rp+'user_clk'] == 1:
            for prefix, rows in packets.items():
                p = rp+prefix+'_'
                require(before[p+'tvalid'] in (0, 1), 'known PCIe valid')
                if before[p+'tvalid'] == 1:
                    require(before[p+'tready'] is not None, 'known PCIe ready')
                if before[p+'tvalid'] == 1 and before[p+'tready'] is not None and before[p+'tready'] & 1:
                    row = [t]+[before[p+n] for n in ['tdata', 'tkeep', 'tlast', 'tuser']]
                    require(None not in row, 'known PCIe payload'); rows.append(row)
                    if prefix == 'm_axis_cq':
                        _, data, keep, final, user = row
                        if not in_host and (data >> 75) & 15 == 1:
                            require(host_count == 0 and written == 0 and data & ((1 << 64)-1) == 0x600 and
                                    (data >> 64) & 2047 == 16 and user & 65535 == 0xf0f, 'host write header')
                            # Native C2H posted data has ATTR=0; native descriptor
                            # and H2C payload reads below have ATTR=2. Do not infer
                            # one direction's ordering policy from the other.
                            require((data >> 121) & 7 == 0 and (data >> 124) & 7 == 0, 'host write TC/attributes')
                            in_host = True
                        if in_host:
                            first = written == 0
                            enable = (user >> 16) & ((1 << 64)-1)
                            require(keep == (65535 if first else 15) and final == int(not first), 'CQ write framing')
                            require((user >> 80) & 3 == int(first) and (user >> 82) & 15 == 0 and
                                    (user >> 86) & 3 == int(not first) and (user >> 96) & 1 == 0, 'CQ boundary/discontinue')
                            if final:
                                require((user >> 88) & 15 == 3, 'CQ final DWORD')
                            require(enable == (((1 << 48)-1) << 16 if first else 65535), 'CQ enabled payload lanes')
                            for lane in range(64):
                                if enable >> lane & 1:
                                    require(written < 64, 'host write bound')
                                    memory[written] = (data >> (8*lane)) & 255
                                    written += 1
                            host_beats.append(row)
                            memory_events.append(dict(time_ps=t, accepted_bytes=written, memory=memory.hex()))
                            if final:
                                require(written == 64, 'full host packet')
                                in_host = False; host_count += 1
        require(after[tb+'host_memory'] == int.from_bytes(memory, 'little') and
                after[tb+'host_written'] == (1 << written)-1 and after[tb+'host_bytes'] == written and
                after[tb+'host_packets'] == host_count and after[tb+'host_active'] == int(in_host),
                'host memory changes only at accepted CQ bytes')
    require(phases == [1, 2, 3, 4, 5] and finish-1 <= last <= finish and qi == len(regs), 'complete VCD scope')
    require(held >= 64 and host_count == 1 and not in_host, 'retention and host receipt')
    pattern = bytes(((i*13)^0xa5)&255 for i in range(64))
    require(bytes(memory) == pattern and int(unique['PASS']['memory'], 16) == int.from_bytes(pattern, 'little') ==
            int(unique['HOST_END']['memory'], 16), 'host memory end-to-end data')
    for channel in fields:
        rows = markers(sim, 'XDMA_C2H_'+channel)
        require(len(rows) == len(accepted[channel]) == 1, 'one accepted '+channel)
        keys = {'AW': ['id', 'addr', 'len', 'size'], 'W': ['data', 'strb', 'last'], 'B': ['id', 'resp'],
                'AR': ['id', 'addr', 'len', 'size'], 'R': ['id', 'data', 'resp', 'last']}[channel]
        logged = [int(rows[0]['t'])]+[int(rows[0][k], 2 if k == 'last' else 16) for k in keys]
        actual = accepted[channel][0]
        require(logged == (actual[:-1] if channel == 'AW' else actual), 'AXI log/VCD identity')
    aw, w, b, ar, r = [accepted[n][0] for n in fields]
    require(aw[2:] == [0x1000, 0, 6, 1] and w[1].to_bytes(64, 'little') == pattern and
            w[2:] == [(1 << 64)-1, 1] and b[1:] == [aw[1], 0], 'H2C seed payload/response')
    require(ar[2:] == [0x1000, 0, 6] and r[1:] == [ar[1], int.from_bytes(pattern, 'little'), 0, 1] == retained,
            'C2H address/data/identity')
    require(max(aw[0], w[0]) < b[0] < seeded < ar[0] < pending < release < r[0] < host_begin,
            'H2C seed, retained R and posted write causality')
    reads, posted = [], []
    for t, data, keep, final, user in packets['s_axis_rq']:
        kind = (data >> 75) & 15
        require(kind in (0, 1) and (data >> 64) & 2047 == 1 and data & 3 == 0 and final == 1 and
                keep == (15 if kind == 0 else 31), 'native register request geometry')
        (reads if kind == 0 else posted).append([t, data & ((1 << 64)-1), (data >> 128) & 0xffffffff])
    require(len(reads) == len(packets['m_axis_rc']) == len(regs), 'native register inventory')
    base = reads[0][1]-0x40; require(base == 0x10000, 'enumerated BAR1')
    previous = -1
    for request, completion, log in zip(reads, packets['m_axis_rc'], regs):
        t, data, keep, final, user = completion
        require(previous < request[0] < t <= int(log['t']) and request[1] == base+int(log['address'], 16), 'serial register correlation')
        require(keep == 15 and final == 1 and (data >> 32) & 2047 == 1 and (data >> 43) & 15 == 0 and
                (data >> 96) & 0xffffffff == int(log['value'], 16), 'native register completion')
        previous = t
    require([(p[1], p[2]) for p in posted] == [(base+a, v) for _, a, v in writes], 'posted control writes')
    cq = packets['m_axis_cq']; cc = packets['s_axis_cc']
    require(len(cq) == 5 and len(cc) == 4 and cq[3:] == host_beats, 'exact host read/write inventory')
    desc = {}
    for address, source, destination in [(0x100, 0x400, 0x1000), (0x200, 0x1000, 0x600)]:
        payload = bytearray(32)
        for offset, value in [(0, 0xad4b0013), (4, 64), (8, source), (16, destination)]:
            payload[offset:offset+4] = value.to_bytes(4, 'little')
        desc[address] = bytes(payload)
    payloads = {**desc, 0x400: pattern}; host_reads = []; offset = 0
    for request, address in zip(cq[:3], [0x100, 0x400, 0x200]):
        t, data, keep, final, user = request
        payload = payloads[address]; length = len(payload)
        count = 1 if length == 32 else 2
        group = cc[offset:offset+count]; offset += count; header = group[0][1]
        require(data & ((1 << 64)-1) == address and (data >> 64) & 2047 == length//4 and
                (data >> 75) & 15 == 0 and keep == 15 and final == 1 and user & 65535 == 0xf0f,
                'host descriptor/payload fetch')
        require((data >> 121) & 7 == 0 and (data >> 124) & 7 == 2, 'host fetch attributes')
        tag, requester = (data >> 96) & 255, (data >> 80) & 65535
        require(t < group[0][0] <= group[-1][0] and (header >> 64) & 255 == tag and
                (header >> 48) & 65535 == requester, 'CC request identity')
        require(header & 127 == 0 and (header >> 16) & 4095 == length and (header >> 32) & 2047 == length//4 and
                (header >> 43) & 31 == 0 and (header >> 31) & 1 == 0 and (header >> 88) & 1 == 1 and
                (header >> 89) & 7 == 0 and (header >> 92) & 7 == 2, 'CC header')
        require([g[2:] for g in group] == ([[2047, 1, 0xa41]] if length == 32 else [[65535, 0, 1], [7, 1, 0x240]]), 'CC segmentation')
        actual = group[0][1].to_bytes(64, 'little')[12:12+min(length, 52)]
        if count == 2:
            require(group[0][0] < group[1][0], 'ordered CC beats')
            actual += group[1][1].to_bytes(64, 'little')[:12]
        require(actual == payload, 'all descriptor and seed bytes')
        host_reads.append(dict(address=address, bytes=length, request_ps=t, completion_ps=group[-1][0], tag=tag))
    require(host_reads[0]['completion_ps'] < aw[0] and host_reads[1]['completion_ps'] < w[0] and
            seeded < host_reads[2]['request_ps'] < host_reads[2]['completion_ps'] < ar[0], 'fetch causality')
    starts, beats, ends = [markers(sim, 'NTRPU_HOST_CPL_'+n) for n in ['BEGIN', 'BEAT', 'END']]
    require(len(starts) == len(ends) == 3 and len(beats) == 4, 'completion driver log inventory')
    require([int(x['offset']) for x in beats] == [0, 0, 52, 0], 'completion log offsets')
    for log, actual in zip(beats, cc):
        require([int(log['t']), int(log['data'], 16), int(log['keep'], 16), int(log['last']), int(log['user'], 16)] == actual,
                'CC log/VCD identity')
    for begin, end, item in zip(starts, ends, host_reads):
        require(int(begin['address'], 16) == item['address'] and int(begin['tag'], 16) == int(end['tag'], 16) == item['tag'] and
                int(begin['bytes']) == int(end['bytes']) == item['bytes'] and int(begin['attr'], 16) == 2 and int(begin['tc'], 16) == 0,
                'completion driver request identity')
        require(item['request_ps'] <= int(begin['t']) < item['completion_ps'] < int(end['t']), 'completion driver lifetime')
    logs = markers(sim, 'XDMA_C2H_HOST_BEAT'); require(len(logs) == 2, 'host write log inventory')
    for log, actual, count in zip(logs, host_beats, [48, 64]):
        require([int(log['t']), int(log['data'], 16), int(log['keep'], 16), int(log['last']), int(log['user'], 16)] == actual and
                int(log['bytes']) == count, 'host posted byte accounting')
    require(host_beats[0][0] == host_begin and host_beats[-1][0] == host_end and
            int(unique['HOST_BEGIN']['address'], 16) == 0x600 and unique['HOST_BEGIN']['bytes'] == '64', 'host memory lifetime')
    clear = next(r for r in markers(sim, 'XDMA_C2H_WRITE') if int(r['address'], 16) == 0x100c)
    readback = next(r for r in regs if int(r['phase']) == 3 and int(r['address'], 16) == 0x1004)
    return dict(status='PASS_NATIVE_C2H_RETAINED_RESPONSE_AND_HOST_RECEIPT', start_ps=start, finish_ps=finish,
        payload_bytes=64, held_r_cycles=held, held_r_ps=release-pending, accepted=accepted, host_reads=host_reads,
        host_memory_events=memory_events, host_write_begin_ps=host_begin, host_write_end_ps=host_end,
        r_to_host_receipt_ps=host_end-r[0], run_clear_posted_ps=int(clear['t']), run_clear_readback_ps=int(readback['t']),
        register_observations=len(regs), native_posted_controls=len(posted), native_host_cq_beats=len(cq),
        qualified_bounded_c2h=True, qualified_cancellation=False, qualified_global_drain=False,
        qualified_endpoint_reset=False, qualified_physical_design=False,
        native_register_correlation='serial one-outstanding register reads; core tag allocation not traced',
        scope='One 64-byte native H2C/C2H round trip with retained AXI R and passive CQ host memory; no OS memory/coherence or physical throughput qualification.')
