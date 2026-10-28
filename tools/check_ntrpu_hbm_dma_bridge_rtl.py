"""Replay successful RTL transactions against the frozen bridge contract model.

Fault scenarios are checked by the SystemVerilog bench's assertions and required
markers. This is sampled transaction refinement, not exhaustive RTL equivalence.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from ntrpu_hbm_dma_bridge_model import BridgeModel, Route, Span, Event
from ntrpu_transport_model import RegionHandle, TransferId


TARGET = 1 << 36
WRITEBACK = 1 << 44


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pattern(seq, length):
    return bytes(((i * 37) ^ (i >> 7) ^ (seq * 53)) & 255 for i in range(length))


def replay(lines, case):
    route = Route(RegionHandle(1, 0, 0, 1), RegionHandle(1, 1, 0, 1),
                  1, 7, 0, 1 << 34, TARGET, 1 << 33, WRITEBACK)
    model = BridgeModel(payload_depth=4)
    model.configure(route)
    model.set_run(0, True); model.set_run(1, True)
    spans = {}; accepted = Counter(); memory = {}; writebacks = {}; pool = {}
    retirements = 0
    for line in lines:
        tokens = line.split()
        if len(tokens) < 3 or tokens[0] != 'BR' or tokens[2] != str(case):
            continue
        kind = tokens[1]
        if kind not in {'ADMIT', 'DESC', 'PUSH', 'DATA', 'LOCAL', 'ACK', 'DONE', 'EVENT', 'RESULT'}:
            continue
        c = int(tokens[3])
        if kind == 'ADMIT':
            seq, size = map(int, tokens[4:6]); destination = int(tokens[6], 16)
            offset = destination - TARGET - (c << 32)
            s = Span(TransferId(1, 0, seq), route.source, route.target, 1, 7,
                     (c << 33) + offset, destination, size)
            require(seq not in spans, 'Duplicate identity in trace')
            model.reserve(c, s); spans[seq] = s
        elif kind == 'DESC':
            seq = int(tokens[4]); wb, destination = [int(v, 16) for v in tokens[5:7]]
            size = int(tokens[7]); d = model.issue(c, ready=True)
            require(d is not None and (d.identity.sequence, d.dst_addr, d.length) == (seq, destination, size),
                    'Descriptor differs from model')
            require(wb in (WRITEBACK + c * 128, WRITEBACK + c * 128 + 64), 'Writeback outside channel pool')
            require(wb not in pool, 'Reused writeback before retirement')
            # Model uses first free slot; RTL uses a rotating pointer. Both
            # must obey this externally visible lifetime/range contract.
            pool[wb] = seq; writebacks[seq] = wb
        elif kind == 'PUSH':
            seq, last = map(int, tokens[4:6]); s = spans[seq]; off = accepted[seq]
            data = pattern(seq, s.length)[off:off + 64]
            require(model.push(c, s.identity, data, last=bool(last)), 'RTL exceeded payload capacity')
            accepted[seq] += 64
        elif kind == 'DATA':
            seq, offset, last = map(int, tokens[4:7]); s = spans[seq]
            data = int(tokens[7], 16).to_bytes(64, 'little')
            beat = model.take(c, ready=True)
            require(beat is not None and beat.identity == s.identity and beat.address == s.target_address + offset
                    and beat.last == bool(last) and beat.data == data, 'Payload/address differs from model')
            require(beat.address not in memory, 'Duplicate or aliased destination')
            memory[beat.address] = data
        elif kind == 'LOCAL':
            s = spans[int(tokens[4])]; model.local_complete(s.identity, s.length)
        elif kind == 'ACK':
            s = spans[int(tokens[4])]
            model.acknowledge(Event(s.identity, s.target, s.window_generation, s.length))
        elif kind == 'DONE':
            model.observe_done(c)
        elif kind == 'EVENT':
            slot, status = map(int, tokens[4:6]); e = model.poll_done(c, ready=True)
            require(e is not None and status == 0, 'Missing or failed completion event')
            require(writebacks[e.identity.sequence] == WRITEBACK + c * 128 + slot * 64, 'Completion slot mismatch')
        elif kind == 'RESULT':
            seq, status, size = map(int, tokens[4:7]); result = model.retire(c)
            require((result.identity.sequence, result.length, status) == (seq, size, 0), 'Premature/incorrect retirement')
            del pool[writebacks[seq]]; retirements += 1
    expected = {s.target_address + i: pattern(seq, s.length)[i:i + 64]
                for seq, s in spans.items() for i in range(0, s.length, 64)}
    require(memory == expected and bool(memory), 'Missing or incorrect memory image')
    expected_count = 4 if case == 0 else 8 if case == 18 else 1
    require(retirements == expected_count and len(spans) == expected_count, 'Incomplete successful scenario')
    require(not pool and model.resources()['contexts'] == [0, 0], 'Leaked ownership')
    image = b''.join(a.to_bytes(8, 'little') + v for a, v in sorted(memory.items()))
    return dict(case=case, transfers=retirements, bytes=len(memory) * 64,
                memory_sha256=hashlib.sha256(image).hexdigest())


def audit_log(text, full):
    require(not re.search(r'FATAL|WATCHDOG|ERROR', text), 'Simulator failure marker')
    lines = text.splitlines()
    passes = [int(line.split()[2]) for line in lines if line.startswith('BR PASS ')]
    expected = list(range(25)) if full else [0]
    require(passes == expected, 'Missing, repeated or reordered scenario')
    require(text.count('BR ALL_PASS ' + str(len(expected)) + '\n') == 1, 'Missing final completion')
    return [replay(lines, case) for case in ([0, 12, 13, 14, 15, 16, 17, 18] if full else [0])]


def negative_controls(text):
    lines = text.splitlines()
    changes = {}
    for label, prefix, token, value in [
            ('truncated_destination', 'BR DESC 0 ', 6, '0000000000000000'),
            ('writeback_is_hbm', 'BR DESC 0 ', 5, '0000000000000000'),
            ('corrupt_payload', 'BR DATA 0 ', 7, '0'),
            ('wrong_result_bytes', 'BR RESULT 0 ', 6, '1')]:
        index = next(i for i, line in enumerate(lines) if line.startswith(prefix))
        new = list(lines); fields = new[index].split();fields[token] = value;new[index] = ' '.join(fields)
        changes[label] = '\n'.join(new) + '\n'
    for label, prefix in [('missing_done', 'BR DONE 0 '), ('missing_ack', 'BR ACK 0 '),
                          ('missing_event', 'BR EVENT 0 '), ('missing_scenario', 'BR PASS 23 ')]:
        new = list(lines);index = next(i for i, line in enumerate(new) if line.startswith(prefix));new.pop(index)
        changes[label] = '\n'.join(new) + '\n'
    rejected = []
    for label, altered in changes.items():
        try:
            audit_log(altered, True)
        except (ValueError, KeyError) as error:
            rejected.append(dict(name=label, reason=str(error)))
        else:
            raise ValueError('Accepted corrupted evidence: ' + label)
    return rejected


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--happy-only', action='store_true')
    args = parser.parse_args()
    text = args.log.read_text(encoding='utf-8')
    print(json.dumps(dict(replay=audit_log(text, not args.happy_only),
                         negative_controls=[] if args.happy_only else negative_controls(text)), indent=2))
