"""Independently reconstruct bridge CSRs and Root Port CQ bytes from public traces."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from collections import defaultdict


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def vcd_rows(text):
    header, body = text.split('$enddefinitions $end', 1)
    require(re.search(r'\$timescale\s+1ps\s+\$end', header), 'VCD timescale')
    scopes, names = [], {}
    for line in header.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == '$scope':
            scopes.append(fields[2])
        elif fields[0] == '$upscope':
            scopes.pop()
        elif fields[0] == '$var':
            names['/' + '/'.join(scopes + [fields[4]])] = fields[3]
    require('/board/probe/owned' in names, 'VCD hierarchy')
    aliases = defaultdict(list)
    for name, code in names.items():
        aliases[code].append(name)
    current, changes, now = {}, {}, 0
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('$'):
            continue
        if line.startswith('#'):
            after = {**current, **changes}
            yield now, current, after
            current, changes = after, {}
            value = int(line[1:]); require(value >= now, 'VCD timestamp order'); now = value
        else:
            raw, code = line[1:].split() if line.startswith('b') else (line[0], line[1:])
            value = None if re.search('[xzXZ]', raw) else int(raw, 2)
            require(code in aliases, 'VCD identifier')
            for name in aliases[code]:
                changes[name] = value
    yield now, current, {**current, **changes}


def review_vcd(text, log, expected_period):
    # Sample the stable values immediately BEFORE each clock edge, not the NBA
    # updates in the same VCD timestamp. Compare every observed bus handshake.
    p = '/board/probe/'
    specs = [
        ('CFG', 'cfg_valid', 'cfg_ready', {'session': ('cfg_session', 16), 'generation': ('cfg_generation', 16), 'axi': ('cfg_axi_base', 16), 'pcie': ('cfg_pcie_base', 16)}),
        ('CFG_R', 'cfg_rsp_valid', 'cfg_rsp_ready', {'error': ('cfg_rsp_error', 10)}),
        ('AW', 'm_axil_awvalid', 'm_axil_awready', {'address': ('m_axil_awaddr', 16)}),
        ('W', 'm_axil_wvalid', 'm_axil_wready', {'data': ('m_axil_wdata', 16), 'strobe': ('m_axil_wstrb', 16)}),
        ('B', 'm_axil_bvalid', 'm_axil_bready', {'response': ('m_axil_bresp', 10)}),
        ('AR', 'm_axil_arvalid', 'm_axil_arready', {'address': ('m_axil_araddr', 16)}),
        ('R', 'm_axil_rvalid', 'm_axil_rready', {'data': ('m_axil_rdata', 16), 'response': ('m_axil_rresp', 10)}),
        ('DMA_AW', 'm_awvalid', 'm_awready', {'address': ('m_awaddr', 16), 'len': ('m_awlen', 10), 'size': ('m_awsize', 10), 'burst': ('m_awburst', 10), 'id': ('m_awid', 16)}),
        ('DMA_W', 'm_wvalid', 'm_wready', {'data': ('m_wdata', 16), 'strb': ('m_wstrb', 16), 'last': ('m_wlast', 10)}),
        ('DMA_B', 'm_bvalid', 'm_bready', {'response': ('m_bresp', 10), 'id': ('m_bid', 16)}),
        ('ACK', 'ack_valid', 'ack_ready', {'tag': ('ack_tag', 16), 'generation': ('ack_generation', 16), 'bytes': ('ack_bytes', 10)}),
        ('RESULT', 'result_valid', 'result_ready', {'tag': ('result_tag', 16), 'bytes': ('result_bytes', 10)}),
    ]
    cq_fields = {'data': ('tdata', 16), 'keep': ('tkeep', 16), 'last': ('tlast', 10), 'user': ('tuser', 16)}
    formats = {kind: fields for kind, _, _, fields in specs}; formats['CQ'] = cq_fields
    expected, observed = defaultdict(list), defaultdict(list)
    for line in log.splitlines():
        if not line.startswith('BN '):
            continue
        kind = line.split()[1]
        if kind not in formats:
            continue
        fields = dict(re.findall(r'(\w+)=\s*([^\s]+)', line))
        expected[kind].append((int(fields['case']), tuple(int(fields[k], radix) for k, (_, radix) in formats[kind].items())))
    last_clock, edges, retained = None, 0, False
    clk = '/board/AXI_PCIE_EP/axi_aclk_out'
    rp_clk = '/board/RP/user_clk'
    for now, before, after in vcd_rows(text):
        phase = before.get(p + 'phase')
        if phase in (1, 2):
            require(before.get(p + 'cold_rst_n') == 1 and before.get('/board/sys_rst_n') == 1, 'No reset between native maps')
            if before.get(clk) == 0 and after.get(clk) == 1:
                if last_clock is not None:
                    require(now - last_clock == expected_period, 'VCD native clock period')
                last_clock = now; edges += 1
                for key in ('link_up',):
                    require(before.get(p + key) == 1, 'VCD native availability')
                for kind, valid, ready, fields in specs:
                    if before.get(p + valid) == 1 and before.get(p + ready) == 1:
                        values = tuple(before.get(p + signal) for signal, _ in fields.values())
                        require(None not in values, 'Known VCD handshake data')
                        observed[kind].append((before[p + 'case_id'], values))
            if before.get(rp_clk) == 0 and after.get(rp_clk) == 1:
                prefix = '/board/RP/m_axis_cq_'
                if before.get(prefix + 'tvalid') == 1 and before.get(prefix + 'tready') == 1:
                    values = tuple(before.get(prefix + signal) for signal, _ in cq_fields.values())
                    require(None not in values, 'Known VCD CQ data')
                    observed['CQ'].append((before[p + 'case_id'], values))
        if (after.get(p + 'phase') == 3 and after.get('/board/sys_rst_n') == 0 and
                all(after.get(p + k) == 1 for k in ('cold_rst_n', 'owned', 'fault', 'mapper_fault'))):
            require(all(after.get(p + k) == 0 for k in ('cmd_ready', 'cfg_ready', 'result_valid')), 'Reset closes admission')
            retained = True
    require(edges >= 32 and retained, 'VCD clock and reset coverage')
    require(dict(expected) == dict(observed), 'VCD/public-log handshake agreement')
    return dict(status='VCD_PUBLIC_HANDSHAKES_MATCH', observed_native_clock_edges=edges,
                handshakes={k: len(v) for k, v in observed.items()}, cold_reset_held_between_maps=True,
                pending_source_context_retained_after_perst=True)


def review(log):
    require(not re.search(r'(^|\n)(?:Fatal:|ERROR:|BN CHECK|BN TIMEOUT)', log), 'Simulation error')
    end = re.findall(r'^BRIDGE_NATIVE_PASS bytes=192 maps=2 clock_period_ps=(\d+) host_isolation=unproven$', log, re.M)
    require(len(end) == 1, 'Unique native completion')
    periods = re.findall(r'^BN CLOCK t=\d+ period=(\d+)$', log, re.M)
    require(len(periods) == 32 and set(periods) == set(end), 'Measured stable clock')
    period = int(end[0])
    require(period in (4000, 8000), 'Supported measured clock')
    require('BN RESET_PASS owned=1 fault=1 mapper_fault=1 cold_rst_n=1' in log,
            'Operational reset must retain context and cold reset')
    events = []
    for line in log.splitlines():
        if line.startswith('BN ') and 'case=' in line:
            kind = line.split()[1]
            fields = dict(re.findall(r'(\w+)=\s*([^\s]+)', line))
            events.append((kind, fields))
    summaries = []
    for case, destination in enumerate((0x1234567800001000, 0x9ABCDEF000002000)):
        count = (64, 128)[case]
        trace = [(k, f) for k, f in events if int(f['case']) == case]
        kinds = [k for k, _ in trace]
        require(all(kinds.count(k) == 1 for k in ('CFG', 'CFG_R', 'DMA_AW', 'DMA_B', 'ACK', 'RESULT', 'CASE_PASS')),
                'Exactly one transaction lifecycle')
        cfg = next(f for k, f in trace if k == 'CFG')
        require([int(cfg[x], 16) for x in ('session', 'generation', 'axi', 'pcie')] ==
                [9, case + 1, 0x10000000, destination], 'Configuration identity')
        aw = wd = ar = None
        controls = []
        sent = bytearray()
        received = {}
        active = None
        packets = 0
        for index, (kind, f) in enumerate(trace):
            if kind == 'AW':
                require(aw is None, 'Single control AW'); aw = int(f['address'], 16)
            elif kind == 'W':
                require(wd is None and int(f['strobe'], 16) == 15, 'Single full control W')
                wd = int(f['data'], 16)
            elif kind == 'B':
                require(aw is not None and wd is not None and int(f['response']) == 0, 'Control B causality')
                controls.append(('W', aw, wd)); aw = wd = None
            elif kind == 'AR':
                require(ar is None, 'Single control AR'); ar = int(f['address'], 16)
            elif kind == 'R':
                require(ar is not None and ar == int(f['address'], 16) and int(f['response']) == 0, 'Control R causality')
                controls.append(('R', ar, int(f['data'], 16))); ar = None
            elif kind == 'CFG_R':
                require(len(controls) == 6 and int(f['error']) == 0, 'Verified mapping response')
            elif kind == 'DMA_AW':
                require(kinds.index('CFG_R') < index and len(controls) == 6, 'DMA after verified map')
                require(int(f['address'], 16) == 0x10000000 and int(f['id'], 16) == 0 and
                        int(f['size']) == 6 and int(f['burst']) == 1 and (int(f['len']) + 1) * 64 == count,
                        'AXI writer address and shape')
            elif kind == 'DMA_W':
                require(int(f['strb'], 16) == (1 << 64) - 1, 'Full DMA beat')
                sent.extend(int(f['data'], 16).to_bytes(64, 'little'))
                require(int(f['last']) == int(len(sent) == count), 'DMA last')
            elif kind == 'DMA_B':
                require(int(f['response']) == 0 and int(f['id'], 16) == 0 and len(sent) == count, 'Native B after payload')
            elif kind == 'CQ':
                data, keep, user = [int(f[x], 16) for x in ('data', 'keep', 'user')]
                sop = (user >> 80) & 3
                first = 0
                if active is None:
                    require(sop == 1 and (data >> 75) & 15 == 1 and user & 0xFFFF == 0x0F0F, 'CQ memory-write header')
                    address = data & ((1 << 64) - 1)
                    length = ((data >> 64) & 0x7FF) * 4
                    require(length > 0, 'CQ nonempty packet')
                    active = [address, length, 0]; first = 16
                else:
                    require(sop == 0, 'CQ nonstraddled continuation')
                require((user >> 96) & 1 == 0, 'CQ not discontinued')
                for lane in range(64):
                    if (user >> (16 + lane)) & 1:
                        address, length, offset = active
                        require(lane >= first and (keep >> (lane // 4)) & 1 and offset < length, 'CQ payload lane')
                        target = address + offset
                        require(destination <= target < destination + count and target not in received,
                                'CQ exact destination and unique byte')
                        received[target] = (data >> (lane * 8)) & 255
                        active[2] += 1
                if int(f['last']):
                    require(active[2] == active[1], 'CQ complete packet length'); active = None; packets += 1
            elif kind == 'ACK':
                require(active is None and len(received) == count and kinds.index('DMA_B') < index,
                        'ACK follows native B and received payload')
                require(int(f['tag'], 16) == (9 << 64) | (case + 1) and int(f['generation'], 16) == case + 1
                        and int(f['bytes']) == count, 'ACK identity')
            elif kind == 'RESULT':
                require(kinds.index('ACK') < index and int(f['tag'], 16) == (9 << 64) | (case + 1)
                        and int(f['bytes']) == count, 'Result identity and causality')
        canonical = [('R', 0x130), ('R', 0x204), ('W', 0x208), ('W', 0x20C), ('R', 0x208), ('R', 0x20C)]
        require([(k, a) for k, a, _ in controls] == canonical and aw is None and wd is None and ar is None,
                'Exact drained CSR sequence')
        require(controls[0][2] & 0xA == 8 and controls[1][2] & 0xFFFF0000 == 0x03800000, 'Native bridge identity fields')
        require([v for _, _, v in controls[2:]] == [destination >> 32, destination & 0xFFFFFFFF] * 2, 'Native translation readback')
        expected = bytes(((case * 13 + i * 37) ^ (i >> 3)) & 255 for i in range(count))
        require(bytes(sent) == expected, 'Source payload')
        require(bytes(received[destination + i] for i in range(count)) == expected, 'Native received payload')
        summaries.append(dict(case=case, destination=f'{destination:016x}', bytes=count, native_packets=packets,
                              native_id_register=f'{controls[0][2]:08x}', native_vsec=f'{controls[1][2]:08x}'))
    return dict(status='NATIVE_BRIDGE_MAPPING_AND_DATA_PASS', cases=summaries, clock_period_ps=period,
                measured_axi_mhz=1000000 / period, operational_reset_context_retained=True,
                host_isolation_proven=False, source_reader_modeled=True, remote_ack_modeled=True,
                board_or_timing_validated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--vcd', type=Path)
    args = parser.parse_args()
    content = args.log.read_text(errors='replace')
    result = review(content)
    result['log_sha256'] = hashlib.sha256(args.log.read_bytes()).hexdigest()
    if args.vcd:
        result['vcd_review'] = review_vcd(args.vcd.read_text(), content, result['clock_period_ps'])
        result['vcd_sha256'] = hashlib.sha256(args.vcd.read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result))
