"""Independent reconstruction of the source join's public transaction trace."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def words(text, n, tail=0):
    value = int(text, 16) >> tail
    need(value < 1 << (64*n), 'Oversized structure')
    return [(value >> (64*(n-1-i))) & ((1 << 64)-1) for i in range(n)]


def decode(text):
    need('SJ ASSERT' not in text and 'FATAL:' not in text and 'watchdog' not in text, 'Simulation failed')
    rows = [line.split()[1:] for line in text.splitlines() if line.startswith('SJ ')]
    need(rows and rows[-1][0] == 'PASS' and int(rows[-1][1]) == 15, 'Missing completion marker')
    cases = defaultdict(list)
    for row in rows[:-1]:
        cases[int(row[1])].append([row[0], *row[2:]])
    need(sorted(cases) == list(range(1, 16)), 'Scenario coverage')
    return dict(cases)


def check(cases):
    reviewed = []
    for phase, rows in sorted(cases.items()):
        rejected = set()
        last_submit = None
        for i, row in enumerate(rows):
            if row[0] == 'SUBMIT':
                last_submit = i
            elif row[0] == 'REJECT':
                need(last_submit is not None and last_submit not in rejected, 'Unattributed rejection')
                rejected.add(last_submit)
        requests = [words(row[2], 16) for i, row in enumerate(rows) if row[0] == 'SUBMIT' and i not in rejected]
        need(len(requests) == (8 if phase == 3 else 4), 'Accepted transfer inventory')
        request_by_seq = {r[2]: r for r in requests}
        need(len(request_by_seq) == len(requests), 'Repeated accepted sequence')
        need([r[2] for r in requests] == list(range(1, len(requests)+1)), 'Request sequence')
        for req in requests:
            index = req[2]-1
            generation = 2 if index >= 4 else 1
            pc = index % 2
            if phase == 15:
                pc ^= (index//2) % 2
            expected = [1, 0, index+1, 1, 0, 10, generation, 1, 1, 20, generation,
                        generation, 7, 0x200000000*pc+(index//2)*4096,
                        0x800+(index % 2)*0x100000000+(index//2)*4096, (64, 128, 256, 4096)[index % 4]]
            need(req == expected, 'Request context/profile mismatch')
        lane_requests = [[r for r in requests if (r[2]-1) % 2 == lane] for lane in (0, 1)]
        commands = [[], []]
        ars = [[], []]
        descriptors = [[], []]
        packets = [[], []]
        payload = [bytearray(), bytearray()]
        native_payload = [bytearray(), bytearray()]
        good_cpl = set()
        acknowledgements = set()
        results = [[], []]
        done = [0, 0]
        credits = [0, 0]
        pending_ar = None
        mgmt_reads = 0
        last_count = [0, 0]
        certificates = [0, 0]
        final = None
        for row in rows:
            kind = row[0]
            if kind == 'MGMT':
                if row[1] == 'AR':
                    need(pending_ar is None, 'Management ordering')
                    pending_ar = int(row[2], 16)
                    need(pending_ar in (0x1004, 0x1040, 0x1048, 0x1104, 0x1140, 0x1148), 'Read-only register whitelist')
                else:
                    need(pending_ar is not None and int(row[3], 16) == 0, 'Management response')
                    lane = (pending_ar >> 8) & 1
                    offset = pending_ar & 255
                    value = int(row[2], 16)
                    if offset == 4:
                        need(value == 0xf83e7f, 'Management configuration')
                        certificates[lane] = last_count[lane]
                    elif offset == 0x40:
                        need(value & ~5 == 0, 'Management status')
                    else:
                        need(last_count[lane] <= value <= done[lane], 'Completion count inventory')
                        last_count[lane] = value
                    pending_ar = None
                    mgmt_reads += 1
                continue
            if kind in ('CMD', 'AR', 'DESC', 'AXIS', 'R', 'CPL', 'ACK', 'RESULT', 'DONE', 'CREDIT'):
                lane = int(row[1])
                need(lane in (0, 1), 'Lane index')
            if kind == 'CMD':
                address, tag, length = [int(v, 16) for v in row[2:]]
                req = lane_requests[lane][len(commands[lane])]
                need(tag == req[2] and length+1 == req[-1]//64, 'Command identity/length')
                base = 0x8000 if tag > 4 else 0x4000
                need(address == base+req[13], 'Pinned HBM translation')
                if commands[lane] and commands[lane][-1][0] >> 29 != address >> 29:
                    need(commands[lane][-1][2] in good_cpl, 'HBM domain changed before local CPL')
                commands[lane].append((address, length, tag))
            elif kind == 'AR':
                address, length = [int(v, 16) for v in row[2:]]
                need(len(ars[lane]) < len(commands[lane]), 'AR before reservation')
                need((address, length) == commands[lane][len(ars[lane])][:2], 'AR translation/length')
                ars[lane].append((address, length))
            elif kind == 'DESC':
                wb, target, length, control = [int(v, 16) for v in row[2:]]
                index = len(descriptors[lane])
                req = lane_requests[lane][index]
                need(target == 0x100000000+req[14] and length == req[15] and control == 0x10, 'DMA descriptor')
                need(wb == 0x300001000+128*lane+64*(index % 2), 'Writeback identity')
                descriptors[lane].append((wb, target, length, control))
            elif kind == 'R':
                data, rid, resp, last = [int(v, 16) for v in row[2:]]
                native_payload[lane].extend(data.to_bytes(64, 'little'))
                if phase not in (4, 5, 6):
                    need(rid == 0 and resp == 0, 'Unexpected HBM error')
            elif kind == 'AXIS':
                data, keep, last = [int(v, 16) for v in row[2:]]
                index = len(packets[lane])
                need(index < len(descriptors[lane]), 'Stream before descriptor')
                req = lane_requests[lane][index]
                chunk = data.to_bytes(64, 'little')
                offset = len(payload[lane])
                expected = bytes(((i*37) ^ (i >> 7) ^ (req[2]*53)) & 255 for i in range(offset, offset+64))
                need(chunk == expected and keep == (1 << 64)-1, 'Payload corruption')
                payload[lane].extend(chunk)
                need(last == (len(payload[lane]) == req[-1]), 'Payload boundary')
                if last:
                    packets[lane].append(bytes(payload[lane]))
                    payload[lane].clear()
            elif kind == 'CPL':
                tag, status, count = [int(v, 16) for v in row[2:]]
                need(tag in request_by_seq, 'CPL attribution')
                if phase == 13 and lane == 0:
                    need(status == 0 and count == request_by_seq[tag][-1] ^ 64, 'Injected CPL error')
                elif status == 0:
                    need(count == request_by_seq[tag][-1] and tag not in good_cpl, 'CPL byte/duplicate')
                    good_cpl.add(tag)
            elif kind == 'DONE':
                done[lane] += 1
                need(done[lane] <= len(packets[lane]), 'Unbacked native Done')
            elif kind == 'CREDIT':
                credits[lane] += 1
                need(credits[lane] <= done[lane] and credits[lane] <= certificates[lane], 'Uncertified credit')
            elif kind == 'ACK':
                event = words(row[2], 9, 4)
                need(int(row[2], 16) & 15 == 0 and event[2] in request_by_seq, 'ACK status/identity')
                req = request_by_seq[event[2]]
                need(event == req[:3]+req[7:12]+[req[15]], 'Full ACK attribution')
                acknowledgements.add(event[2])
            elif kind == 'RESULT':
                context, status, count = row[2:]
                req = lane_requests[lane][len(results[lane])]
                need(words(context, 16) == req, 'Full result attribution')
                status, count = int(status, 16), int(count, 16)
                if status == 0:
                    need(req[2] in good_cpl and req[2] in acknowledgements and
                         len(results[lane]) < credits[lane] and count == req[-1], 'Premature success')
                else:
                    need(status == 1 and count == 0 and phase in (*range(4, 10), 12, 13, 14), 'Error result semantics')
                results[lane].append(status)
            elif kind == 'CASE':
                need(final is None, 'Duplicate final state')
                final = list(map(int, row[1:]))
        need(final is not None, 'Missing final state')
        success = phase <= 3 or phase in (10, 11, 15)
        expected_held = 0 if success else 10
        need(final[:2] == [expected_held]*2, 'Ownership release/quarantine')
        need(final[2:4] == [len(r) for r in lane_requests], 'Result count')
        need(final[4:6] == [len(c) for c in commands], 'Command count')
        need(final[6:8] == [len(p) for p in packets], 'Packet count')
        need(final[8] == len(rejected) == (5 if phase == 10 else 0), 'Rejection count')
        if phase == 10:
            need([int(row[1], 16) for row in rows if row[0] == 'REJECT'] == [1, 3, 5, 6, 7], 'Rejection reasons')
        need([len(r) for r in results] == final[2:4], 'Result evidence missing')
        if success:
            need(all(status == 0 for lane in results for status in lane), 'Success scenario errors')
            need(credits == done == [len(r) for r in lane_requests], 'DMA retirement inventory')
            for lane in (0, 1):
                need(bytes(native_payload[lane]) == b''.join(packets[lane]), 'HBM/DMA byte conservation')
        if phase == 7:
            need(not any(commands) and not any(ars), 'Stop leaked reader command')
        if phase == 9:
            need(results == [[0, 1], [0, 1]], 'Held success/error quarantine')
        need(mgmt_reads >= 8, 'No management bootstrap')
        reviewed.append(dict(phase=phase, requests=len(requests), rejects=len(rejected),
                             commands=sum(map(len, commands)), results=sum(map(len, results)),
                             payload_bytes=sum(sum(map(len, p)) for p in packets)+sum(map(len, payload)),
                             hbm_bytes=sum(map(len, native_payload)), management_reads=mgmt_reads,
                             retained_contexts=0 if success else 4))
    return reviewed


def review(text):
    cases = decode(text)
    reviewed = check(cases)
    mutations = [
        ('payload', 'AXIS', 2, lambda v: hex(int(v, 16)^1)[2:]),
        ('hbm_address', 'AR', 2, lambda v: hex(int(v, 16)^64)[2:]),
        ('command_tag', 'CMD', 3, lambda v: hex(int(v, 16)+1)[2:]),
        ('result_identity', 'RESULT', 2, lambda v: hex(int(v, 16)^(1 << 1023))[2:]),
        ('descriptor_high_address', 'DESC', 3, lambda v: hex(int(v, 16)^(1 << 32))[2:]),
        ('cpl_bytes', 'CPL', 4, lambda v: hex(int(v, 16)+64)[2:]),
    ]
    rejected = []
    for name, kind, column, change in mutations:
        altered = deepcopy(cases)
        row = next(row for row in altered[1] if row[0] == kind)
        row[column] = change(row[column])
        try:
            check(altered)
        except (ValueError, IndexError):
            rejected.append(name)
        else:
            raise ValueError('Accepted evidence mutation: '+name)
    altered = deepcopy(cases)
    row = next(row for row in altered[4] if row[0] == 'CASE')
    row[1] = row[2] = '0'
    try:
        check(altered)
    except ValueError:
        rejected.append('error_releases_ownership')
    else:
        raise ValueError('Accepted error release')
    return dict(status='SOURCE_JOIN_TRACE_REVIEWED', cases=reviewed, negative_controls=rejected,
                log_sha256=hashlib.sha256(text.encode()).hexdigest(), native_hbm=False, native_xdma=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log', type=Path)
    a = ap.parse_args()
    print(json.dumps(review(a.log.read_text(encoding='utf-8')), indent=2))
