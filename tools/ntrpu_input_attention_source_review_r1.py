"""Review producer arithmetic and the physical writes feeding full attention."""
from collections import defaultdict
import ntrpu_input_attention_fixture_r1 as layout

SCHEMA = {
    'PSTART': 'cycle job sequence epoch', 'PNSTART': 'cycle job',
    'PNWRITE': 'cycle job index data', 'PNDONE': 'cycle job status words',
    'PMATRIX': 'cycle job phase tag rows', 'PMAC': 'cycle job phase index data',
    'PAR': 'cycle job kind phase address beats',
    'PR': 'cycle job kind phase address beat last data',
    'PAW': 'cycle job kind phase address beats',
    'PW': 'cycle job kind phase index last data',
    'PB': 'cycle job kind phase address beats response',
    'PMACK': 'cycle job phase tag status rows bytes',
    'PDONE': 'cycle job status norm qkv matrix hls reads records phase matrices bytes',
    'CSTART': 'cycle job sequence epoch layer position', 'CACK': 'cycle job', 'PACK': 'cycle job',
}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def bursts(address, byte_count):
    result = []
    while byte_count:
        beats = min(byte_count // 64, (4096 - address % 4096) // 64)
        need(0 < beats <= 64, 'Invalid planned burst')
        result.append((address, beats))
        address += beats * 64
        byte_count -= beats * 64
    return result


def read_plan():
    plan = []
    for plane in range(2):
        plan += [(0, 0, a, n, -1) for a, n in bursts(layout.address(plane), 8192)]
    for phase, rows in enumerate((2048, 512, 512)):
        for group in range(rows // 32):
            intervals = [(layout.address(2), 4096),
                         (layout.address(3 + phase) + group * 69632, 34816),
                         (layout.address(3 + phase) + group * 69632 + 34816, 34816)]
            for a, n in intervals:
                plan += [(1, phase, address, beats, group) for address, beats in bursts(a, n)]
    return plan


def write_plan():
    plan = [(0, 0, layout.address(2) + i * 64, 1) for i in range(64)]
    for phase, rows in enumerate((2048, 512, 512)):
        for group in range(rows // 32):
            plan += [(1, phase, a, n) for a, n in bursts(layout.address(6 + phase) + group * 128, 128)]
    return plan


READ_PLAN, WRITE_PLAN = read_plan(), write_plan()


def review_source(events, fixture, spec, ident, ordinal):
    """Return actual B-committed memory; never return reference-populated memory."""
    rows = [(tag, row) for tag, row in events if tag in SCHEMA]
    if spec['mode'] == 1:
        need(not rows, 'Preflight-rejected position started producer/consumer work')
        return dict(memory={}, summary=dict(producer_started=False))
    by = defaultdict(list)
    for tag, row in rows:
        need(set(row) == set(SCHEMA[tag].split()) and row['job'] == ident, 'Producer event schema/identity')
        by[tag].append(row)
    for tag in ('PSTART', 'PNSTART', 'PNDONE', 'PDONE', 'CSTART', 'CACK', 'PACK'):
        need(len(by[tag]) == 1, 'Unique ' + tag)
    boundary = {tag: [row for kind, row in events if kind == tag] for tag in ('ADMIT', 'COMPLETE', 'ACK', 'RELEASE', 'ROPE_START')}
    need(all(len(v) == 1 for v in boundary.values()), 'Full source ownership boundaries')
    boundary = {k: v[0] for k, v in boundary.items()}
    seq, epoch = 0x1234567800000000 + spec['group'], 0xfedcba9800000000 + spec['group']
    start, done, consumer, ack = (by[t][0] for t in ('PSTART', 'PDONE', 'CSTART', 'PACK'))
    need((start['sequence'], start['epoch']) == (seq, epoch), 'Producer command identity')
    need((consumer['sequence'], consumer['epoch'], consumer['layer'], consumer['position']) ==
         (seq, epoch, 0, spec['sample']), 'Consumer command identity')
    need(boundary['ADMIT']['cycle'] < start['cycle'] < done['cycle'] < consumer['cycle'] <
         boundary['ROPE_START']['cycle'] < boundary['COMPLETE']['cycle'] < boundary['ACK']['cycle'] <
         by['CACK'][0]['cycle'] < ack['cycle'] < boundary['RELEASE']['cycle'], 'Producer/consumer/public lifetime')
    need(tuple(done[k] for k in ('status', 'norm', 'qkv', 'matrix', 'hls', 'reads', 'records', 'phase', 'matrices', 'bytes')) ==
         (0, 0, 0, 0, 0, 256, 64, 2, 3, 12288), 'Producer full certificate')
    need(len(by['PMATRIX']) == len(by['PMACK']) == 3, 'Three actual matrices and releases')
    for phase, count in enumerate((2048, 512, 512)):
        matrix, released = by['PMATRIX'][phase], by['PMACK'][phase]
        tag = ordinal * 192 + phase * 64
        need((matrix['phase'], matrix['tag'], matrix['rows']) == (phase, tag, count), 'Matrix descriptor and rolling tag')
        need((released['phase'], released['tag'], released['status'], released['rows'], released['bytes']) ==
             (phase, tag, 0, count, count * 4), 'Matrix completion certificate')
        need(matrix['cycle'] < released['cycle'] < boundary['RELEASE']['cycle'], 'Matrix lifetime')
        if phase < 2:
            need(released['cycle'] < by['PMATRIX'][phase+1]['cycle'] < done['cycle'], 'Matrix serialization')
        else:
            need(boundary['ACK']['cycle'] < released['cycle'], 'Final matrix not released before public ACK')
    sample = fixture['samples'][spec['sample']]
    norm_words = fixture['norm']['source32.hex'][spec['sample']*4096:(spec['sample']+1)*4096]
    abi_words = fixture['norm']['abi64.hex'][spec['sample']*512:(spec['sample']+1)*512]
    expected_writes = sample['activation'] + sum(sample['fp32'], [])
    memory, read_request, aw, buffered = {}, None, None, []
    reads = writes = read_cursor = write_cursor = norm_published = 0
    norm_started = norm_done = False
    active_phase = -1
    mac = [0, 0, 0]
    phase_reads = [0, 0, 0]
    phase_committed = [0, 0, 0]
    committed_abi = 0
    previous = -1
    for kind, row in rows:
        cycle = row['cycle']
        need(previous <= cycle < boundary['RELEASE']['cycle'] and cycle >= start['cycle'], 'Producer monotonic owned time')
        previous = cycle
        if kind not in ('PMACK', 'CSTART', 'CACK', 'PACK'):
            need(cycle <= done['cycle'], 'Producer operation after completion')
        if kind == 'PNSTART':
            need(reads == 256 and read_request is None and not norm_started and writes == 0, 'Norm starts after exact input reads')
            norm_started = True
        elif kind == 'PNWRITE':
            need(norm_started and not norm_done and row['index'] == norm_published < 512, 'Norm ordered owned words')
            need(row['data'] == abi_words[norm_published], 'Actual normalization arithmetic')
            norm_published += 1
        elif kind == 'PNDONE':
            need(norm_started and not norm_done and norm_published == 512 and row['status'] == 0 and row['words'] == 512, 'Norm completion extent')
            norm_done = True
        elif kind == 'PMATRIX':
            need(norm_done and committed_abi == 64 and read_request is None and aw is None and not buffered, 'Matrix reads only committed ABI1')
            need(row['phase'] == active_phase + 1, 'Matrix phase order')
            active_phase += 1
        elif kind == 'PAR':
            need(read_request is None and read_cursor < len(READ_PLAN), 'One producer AR')
            read_kind, phase, address, beats, group = READ_PLAN[read_cursor]
            need((row['kind'], row['phase'], row['address'], row['beats']) == (read_kind, phase, address, beats), 'Exact producer read layout')
            if read_kind:
                need(phase == active_phase and phase_committed[phase] >= group * 2, 'Next group waits for prior output commits')
            else:
                need(not norm_started, 'Late input read')
            read_request = dict(row, seen=0)
            read_cursor += 1
        elif kind == 'PR':
            need(read_request is not None and cycle > read_request['cycle'], 'Producer R owns earlier AR')
            a = read_request['address'] + read_request['seen'] * 64
            need((row['kind'], row['phase'], row['address'], row['beat'], row['last']) ==
                 (read_request['kind'], read_request['phase'], a, read_request['seen'], int(read_request['seen']+1 == read_request['beats'])), 'Producer read framing')
            if not row['kind']:
                plane = 0 if a < layout.address(1) else 1
                offset = plane * 2048 + (a - layout.address(plane)) // 4
                expected = sum(norm_words[offset+i] << (32*i) for i in range(16))
            elif layout.address(2) <= a < layout.address(2) + 4096:
                need(a in memory and memory[a]['cycle'] < cycle and committed_abi == 64, 'ABI read depends on successful B')
                expected = memory[a]['data']
            else:
                phase = row['phase']
                expected = fixture['weights'][phase][(a - ((phase+4) << 40)) // 64]
            need(row['data'] == expected, 'Actual producer read bytes')
            read_request['seen'] += 1
            reads += 1
            if row['kind']:
                phase_reads[row['phase']] += 1
            if row['last']:
                read_request = None
        elif kind == 'PMAC':
            phase, index = row['phase'], row['index']
            need(phase == active_phase and index == mac[phase] < (2048 if phase == 0 else 512), 'Exact MAC coverage/order')
            need(phase_reads[phase] >= (index//32+1)*1152, 'MAC after accepted group inputs')
            need(row['data'] == sample['fp64'][phase][index], 'Actual FP64 matrix arithmetic')
            mac[phase] += 1
        elif kind == 'PAW':
            need(aw is None and write_cursor < len(WRITE_PLAN), 'One producer AW')
            k, p, a, n = WRITE_PLAN[write_cursor]
            need((row['kind'], row['phase'], row['address'], row['beats']) == (k, p, a, n), 'Exact ABI/QKV write layout')
            if k:
                need(p == active_phase and mac[p] >= ((a-layout.address(p+6))//128+1)*32, 'Output address follows actual MACs')
            else:
                need(norm_done, 'ABI address after norm completion')
            aw = dict(row)
            write_cursor += 1
        elif kind == 'PW':
            need(len(buffered) < 2 and row['index'] == writes < 256, 'Producer write order/extent')
            need(row['kind'] == int(writes >= 64), 'Producer write owner')
            need(row['phase'] == (active_phase if writes >= 64 else 0), 'Producer write phase')
            need(row['data'] == expected_writes[writes], 'Actual ABI1/FP32 write arithmetic')
            if writes < 64:
                need(norm_done, 'ABI data after complete norm output')
            else:
                phase_index = writes - 64 - (0 if active_phase == 0 else 128 if active_phase == 1 else 160)
                need(mac[active_phase] >= (phase_index//2+1)*32, 'Output data follows actual MACs')
            buffered.append(row)
            writes += 1
        elif kind == 'PB':
            need(aw is not None and len(buffered) == aw['beats'] == row['beats'] and cycle > aw['cycle'], 'B after complete AW/W')
            need((row['kind'], row['phase'], row['address'], row['response']) == (aw['kind'], aw['phase'], aw['address'], 0), 'Successful matching producer B')
            for i, word in enumerate(buffered):
                need(word['cycle'] < cycle and word['last'] == int(i == len(buffered)-1), 'Producer WLAST/B causality')
                address = aw['address'] + i * 64
                need(address not in memory, 'Unique memory publication')
                memory[address] = dict(data=word['data'], cycle=cycle)
                if row['kind']:
                    phase_committed[row['phase']] += 1
                else:
                    committed_abi += 1
            aw = None
            buffered = []
        elif kind in ('PMACK', 'PDONE'):
            need(read_request is None and aw is None and not buffered, 'Completion after physical drain')
            if kind == 'PMACK':
                phase = row['phase']
                count = 2048 if phase == 0 else 512
                need(mac[phase] == count and phase_reads[phase] == count//32*1152 and phase_committed[phase] == count//16,
                     'Full matrix memory and arithmetic extent')
    need(read_cursor == len(READ_PLAN) and write_cursor == len(WRITE_PLAN), 'All planned bursts')
    need(reads == 110848 and writes == 256 and norm_published == 512 and mac == [2048, 512, 512], 'Full-size producer coverage')
    need(len(memory) == 256 and committed_abi == 64 and phase_committed == [128, 32, 32], 'Complete successful physical publication')
    return dict(memory=memory, summary=dict(producer_started=True, input_reads=reads, writes=writes,
                norm_words=norm_published, mac_words=sum(mac), abi_commits=committed_abi,
                qkv_commits=sum(phase_committed), cycles=done['cycle']-start['cycle']))
