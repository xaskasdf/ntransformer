"""Strict fragment control review: explicit ABI and frozen rejection snapshots; no PCIe drain proof."""
import argparse
import json
from pathlib import Path
import re

CASES = [
    'register_identity_and_decode', 'fragment_profile_and_readonly_telemetry',
    'atomic_snapshot_and_response_stability', 'strobe_and_command_validation',
    'reject_invalid_arm', 'immutable_pending_arm',
    'barrier_is_read_only_and_not_synthesized', 'stop_keeps_outstanding_lease',
    'identity_checked_retirement', 'recovery_without_reset',
    'epoch_exhaustion_no_wrap', 'unexpected_result_fail_closed',
]
SCHEMA = {
    'TELEMETRY': 'n frames identity code faults cycle',
    'CASE': 'id name', 'WRITE': 'n addr data strb resp cycle',
    'READ': 'n addr data resp cycle',
    'ARM': 'n epoch tag window dest cycle',
    'RETIRE': 'n epoch tag status cycle',
    'PASS': 'cases writes reads arms retires checks cycles b_hold r_hold arm_hold seed',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write_fixture():
    """Independent host fixture; every request and B response must be witnessed."""
    rows = []
    def w(a,d,st=15,resp=0): rows.append((a,d,st,resp))
    def desc(e,t,a,d):
        for offset,value in enumerate((e,t,a,d)):
            w(0x20+offset*8,value & 0xffffffff);w(0x24+offset*8,value >> 32)
    def ack(e,t):
        for offset,value in enumerate((e,t)):
            w(0x40+offset*8,value & 0xffffffff);w(0x44+offset*8,value >> 32)
    w(0,1,15,2);w(0xb0,1,15,3);w(0x10000020,1,15,3);w(0x21,1,15,3)
    for a in range(0x90,0xb0,4): w(a,0xffffffff,15,2)
    w(0x10,8,0);w(0x10,8);w(0x10,8)
    w(0x20,0x11223344);w(0x20,0xaabbccdd,5);w(0x20,0,0)
    w(0x10,0xffffffff,0);w(0x10,3,15,2);w(0x14,2,15,2);w(0x14,1,1);w(0x14,0,2)
    desc(0,1,0x100000000,0x100000040);w(0x10,1,15,2)
    w(0x20,1);w(0x30,1);w(0x10,1,15,2);w(0x30,0);w(0x38,0x41);w(0x10,1,15,2)
    e=0x100000001;t=0x8877665544332211
    desc(e,t,0x1000fffc0,0x1fffffffc0);w(0x10,1)
    for a,d in ((0x20,7),(0x2c,0),(0x10,1),(0x10,2),(0x10,4),(0x78,3),(0x7c,1),(0x80,1)): w(a,d,15,2)
    w(0x10,2)
    for a,d in ((0x10,2),(0x10,4),(0x10,1),(0x38,0),(0x10,4)): w(a,d,15,2)
    ack(1,t);w(0x10,4,15,2);ack(e,0x44332211);w(0x10,4,15,2)
    ack(e,t);w(0x10,4);w(0x10,4,15,2)
    desc(e,2,0x100000000,0);w(0x10,1,15,2)
    e+=1;t=0xffffffff00000002
    desc(e,t,0x100000000,0);w(0x10,1);ack(e,t);w(0x10,4)
    e=0xffffffffffffffff;t=3
    desc(e,t,0x100000000,0);w(0x10,1);ack(e,t);w(0x10,4)
    desc(0,4,0x100000000,0);w(0x10,1,15,2)
    for a,d in ((0x10,1),(0x10,4),(0x8c,0),(0x20,1)): w(a,d,15,2)
    w(0x10,8)
    assert len(rows)==128
    return rows


def review(text, seed):
    need(not re.search(r'^(Fatal:|ERROR:|CRITICAL WARNING:|WARNING:)', text, re.M), 'diagnostic')
    events = []
    for line in text.splitlines():
        if not line.startswith('NTRPU_CONTROL_'):
            continue
        label, *words = line.split()
        kind = label.removeprefix('NTRPU_CONTROL_')
        fields = dict(w.split('=', 1) for w in words)
        need(kind in SCHEMA and len(fields) == len(words) and
             set(fields) == set(SCHEMA[kind].split()), 'schema')
        events.append((kind, fields))
    need(events and events[-1][0] == 'PASS' and sum(k == 'PASS' for k, _ in events) == 1, 'PASS count')
    counts = dict(WRITE=0, READ=0, ARM=0, RETIRE=0, CASE=0, TELEMETRY=0)
    chronology = 0
    writes = []
    # Replay the host descriptor and acknowledgement words independently.
    descriptor = [0]*8
    acknowledgement = [0]*4
    expected_arms = [
        (0x100000001, 0x8877665544332211, 0x1000fffc0, 0x1fffffffc0),
        (0x100000002, 0xffffffff00000002, 0x100000000, 0),
        (0xffffffffffffffff, 3, 0x100000000, 0),
    ]
    expected_reads = [
        (0,0x4652544e,0),(4,0x10000,0),(8,0x1f,0),(0xb0,0,3),
        (0x10000000,0,3),(1,0,3),
        (0x90,0,0),(0x94,0,0),(0x98,0,0),(0x9c,0x5052544e,0),
        (0xa0,0x31304655,0),(0xa4,64,0),(0xa8,32,0),(0xac,0,0),
        (0x94,0,0),(0xac,0,0),
        (0xac,0,0),(0x90,0,0),(0xac,1,0),(0x90,0x80000001,0),
        (0x94,0x40000002,0),(0x98,0x307,0),(0x90,0x80000001,0),
        (0x90,0xffffffff,0),(0x94,0x80000004,0),(0x98,0x422,0),
        (0x90,0xffffffff,0),(0x94,0x80000004,0),(0x98,0x422,0),
        (0x20,0x11bb33dd,0),
        (0x20,0x11bb33dd,0),(0x14,1,0),(0x78,1,0),(0x7c,1,0),
        (0x80,0xffffffff,0),(0x78,3,0),(0x18,1,0),(0x84,1,0),
        (0x88,1,0),(0x60,1,0),(0x64,1,0),(0x68,0x44332211,0),
        (0x6c,0x88776655,0),(0x70,1,0),(0x1c,1,0),(0x18,2,0),
        (0x1c,2,0),(0x8c,1,0),(0x90,9,0),(0x94,5,0),(0x98,4,0),(0xac,1,0),
    ]
    snapshot = None
    accepted = None
    command_pending = 0
    for kind, f in events:
        if 'cycle' in f:
            need(int(f['cycle']) >= chronology, 'chronology')
            chronology = int(f['cycle'])
        if kind in counts:
            counts[kind] += 1
        if kind == 'CASE':
            need(int(f['id']) == counts[kind] and f['name'] == CASES[counts[kind]-1], 'scenario')
        elif kind in {'WRITE', 'READ', 'ARM', 'RETIRE', 'TELEMETRY'}:
            need(int(f['n']) == counts[kind], 'event count')
        if kind == 'WRITE':
            address, data, strobe = [int(f[k], 16) for k in ('addr', 'data', 'strb')]
            resp = int(f['resp'])
            need(strobe < 16 and resp in {0, 2, 3}, 'write fields')
            writes.append((address, data, strobe, resp))
            mask = sum(0xff << (i*8) for i in range(4) if strobe & (1 << i))
            if resp == 0 and 0x20 <= address <= 0x3c:
                index = (address-0x20)//4
                descriptor[index] = (descriptor[index] & ~mask) | (data & mask)
            if resp == 0 and 0x40 <= address <= 0x4c:
                index = (address-0x40)//4
                acknowledgement[index] = (acknowledgement[index] & ~mask) | (data & mask)
            if address == 0x10 and data & mask == 1 and resp == 0:
                snapshot = tuple(descriptor[i] | descriptor[i+1] << 32 for i in range(0, 8, 2))
                command_pending += 1
        elif kind == 'TELEMETRY':
            expected = [(0x80000001,0x40000002,7,0x30), (0xffffffff,0x80000004,2,0x42), (9,5,4,0)]
            need(tuple(int(f[k],16) for k in ('frames','identity','code','faults')) == expected[counts[kind]-1], 'telemetry fixture')
        elif kind == 'READ':
            need((int(f['addr'],16),int(f['data'],16),int(f['resp'])) ==
                 expected_reads[counts[kind]-1], 'read fixture')
        elif kind == 'ARM':
            actual = tuple(int(f[k], 16) for k in ('epoch', 'tag', 'window', 'dest'))
            need(actual == expected_arms[counts[kind]-1] == snapshot and command_pending == 1, 'arm descriptor')
            command_pending = 0
            accepted = actual
        elif kind == 'RETIRE':
            actual = tuple(int(f[k], 16) for k in ('epoch', 'tag'))
            ack = tuple(acknowledgement[i] | acknowledgement[i+1] << 32 for i in (0, 2))
            need(accepted and actual == accepted[:2] == ack, 'retirement identity')
            need(int(f['status']) == (1 if counts[kind] == 1 else 0), 'retirement status')
            accepted = None
    p = {k: int(v) for k, v in events[-1][1].items()}
    need(counts['CASE'] == p['cases'] == 12 and counts['ARM'] == p['arms'] == 3 and
         counts['RETIRE'] == p['retires'] == 3, 'coverage')
    need(counts['WRITE'] == p['writes'] == 128 and counts['READ'] == p['reads'] == 52 and p['seed'] == seed, 'totals')
    need(p['checks'] > 500 and p['cycles'] >= chronology and p['b_hold'] > 100 and
         p['r_hold'] > 50 and p['arm_hold'] >= 100, 'backpressure coverage')
    for address in (0x78, 0x7c, 0x80, 0x8c):
        need(any(a == address and r == 2 for a, _, _, r in writes), 'read-only barrier/fault')
    need(writes == write_fixture(), 'complete write/response fixture')
    need(counts['TELEMETRY'] == 3, 'telemetry updates')
    need(not command_pending and accepted is None, 'outstanding work')
    return dict(summary=p, events=len(events), scenarios=CASES,
                rejected_writes=sum(r != 0 for _, _, _, r in writes),
                accepted_descriptors=expected_arms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.log.read_text(), args.seed), indent=2))
