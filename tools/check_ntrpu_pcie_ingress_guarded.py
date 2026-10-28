"""Independent source-byte and memory-write replay of the capacity composition.

The fixture sends three aligned but invalid 64-bit destinations and requires
zero downstream transactions, then recovers without reset. Scope is one clock,
one line, a compliant destination and externally trusted isolation barriers.
"""
import argparse
import json
from pathlib import Path
import re

CASES = [
    ('complete', [0]*16, 0), ('sparse_and_identical_duplicate', [0]*33, 0),
    ('conflicting_duplicate', [0, 2], 4), ('holes_stop', [0]*16, 1),
    ('upper_address_rejection', [3], 3),
    *[(f'stop_phase_{i}', [0]*16, 1) for i in range(1, 6)],
    ('destination_slverr', [0]*16, 5), ('destination_decerr', [0]*16, 5),
    ('first_invalid_destination', [0]*16, 5),
    ('high_bit_destination_alias', [0]*16, 5),
    ('missing_final_b', [0]*16, 0), ('recovery_without_reset', [0]*16, 0),
    ('stop_partial_aw', [2], 1),
    ('zero_destination', [0]*16, 0), ('above_4gib_destination', [0]*16, 0),
    ('destination_end_overflow', [0]*16, 5), ('zero_recovery_after_rejections', [0]*16, 0),
]
WRITE_JOBS = {1, 2, *range(6, 17), *range(18, 22)}
MEMORY_JOBS = WRITE_JOBS - {13, 14, 20}


def destination(job):
    return {13: 0x2000000000, 14: 0x1000000000000000, 18: 0,
            19: 0x100000040, 20: 0xffffffffffffffc0, 21: 0}.get(job, 0x1fffffffc0)

FIELDS = {
    'CASE': 'job name', 'ARM': 'epoch tag window dest cycle',
    'AW': 'addr id len size cycle', 'W': 'data strb last cycle',
    'B': 'id resp cycle', 'MAW': 'addr id len size cycle',
    'MW': 'data strb last cycle', 'MB': 'id resp cycle',
    'PHASE': 'phase busy idle status cycle', 'STOP': 'phase cycle',
    'HOLD': 'busy idle result arm epoch peer cycle',
    'RESULT': 'epoch tag status cycle',
    'PASS': 'jobs results writes foreign aw w b cycles aw_stall w_stall b_stall result_stall held seed',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def fixture(job, index):
    base = 0x1000fffc0 if job % 2 else 0x100000000
    offset, mask = 0, 15
    if job in WRITE_JOBS - {2}:
        offset = index*4
    elif job == 2:
        if index < 2:
            mask = 5
        else:
            offset = ((index-1)//2)*4
            mask = 5 if (index-1)%2 == 0 else 10
    elif job == 4:
        offset, mask = index*4, 1
    elif job == 5:
        base ^= 1 << 32
    first = (mask & -mask).bit_length()-1
    return base+offset+first, mask << offset


def review(text, seed):
    need(not re.search(r'^(Fatal:|ERROR:|WARNING:|CRITICAL WARNING:)', text, re.M), 'positive diagnostic')
    events = []
    for line in text.splitlines():
        if not line.startswith('NTRPU_GUARDED_'):
            continue
        label, *values = line.split()
        kind = label.removeprefix('NTRPU_GUARDED_')
        f = dict(v.split('=', 1) for v in values)
        need(kind in FIELDS and len(f) == len(values) and set(f) == set(FIELDS[kind].split()), 'event schema')
        events.append((kind, f))
    need(events and events[-1][0] == 'PASS' and sum(k == 'PASS' for k, _ in events) == 1, 'terminal PASS')
    job = 0
    counts = dict(aw=0, w=0, b=0, writes=0, foreign=0, results=0, held=0)
    last_cycle = 0
    byte_writes = 0
    measurements = []
    for kind, f in events:
        if 'cycle' in f:
            cycle = int(f['cycle'])
            need(cycle >= last_cycle, 'chronology')
            last_cycle = cycle
        if kind == 'CASE':
            if job:
                need(retired and address is None and beat is None and responses == len(CASES[job-1][1]), 'unfinished previous case')
                need(phases == ([1, 2, 3, 4, 5, 0] if job in WRITE_JOBS else []), 'bridge sequence')
            job += 1
            need(job <= 21 and int(f['job']) == job and f['name'] == CASES[job-1][0], 'case order')
            armed = retired = False
            address = beat = maw = mw = mb = None
            bank = [None]*64
            responses = 0
            phases, stops, holds, foreign = [], [], [], []
            arm_cycle = last_b_cycle = 0
        elif kind == 'ARM':
            need(job and not armed and not retired, 'unexpected ARM')
            need(int(f['epoch'], 16) == job and int(f['tag'], 16) == 0x1000+job, 'ARM identity')
            window = 0x1000fffc0 if job % 2 else 0x100000000
            need(int(f['window'], 16) == window and int(f['dest'], 16) == destination(job), 'ARM address')
            armed, arm_cycle = True, cycle
        elif kind in {'AW', 'W'}:
            need(armed and not retired and responses < len(CASES[job-1][1]), 'source context')
            addr, mask = fixture(job, responses)
            if kind == 'AW':
                need(address is None, 'overwritten AW')
                address = {k: int(v, 16 if k in {'addr', 'id'} else 10) for k, v in f.items()}
                need(address['addr'] == addr and address['len'] == 0 and address['size'] == 2, 'source address')
                expected_id = 5 if job == 17 else (counts['b']+3)%16
                need(address['id'] == expected_id, 'source request ID')
                counts['aw'] += 1
            else:
                need(beat is None, 'overwritten W')
                beat = {k: int(v, 16 if k in {'data', 'strb'} else 10) for k, v in f.items()}
                data = int.from_bytes(bytes((job*37+i*19+seed*7)&255 for i in range(64)), 'little')
                if job == 3 and responses == 1:
                    data ^= 1
                need(beat['data'] == data and beat['strb'] == mask and beat['last'] == 1, 'source data/mask/last')
                counts['w'] += 1
        elif kind == 'B':
            need(address is not None and beat is not None and cycle > max(address['cycle'], beat['cycle']), 'source B before channels')
            need(int(f['id'], 16) == address['id'] and int(f['resp']) == CASES[job-1][1][responses], 'source B status/ID')
            if int(f['resp']) == 0:
                need(address['addr']//64*64 == window, 'reservation address')
                raw = beat['data'].to_bytes(64, 'little')
                for lane in range(64):
                    if beat['strb'] >> lane & 1:
                        need(address['addr'] <= window+lane < address['addr']//4*4+4, 'DWORD byte bounds')
                        need(bank[lane] is None or bank[lane] == raw[lane], 'conflicting byte accepted')
                        bank[lane] = raw[lane]
                        byte_writes += 1
            address = beat = None
            responses += 1
            counts['b'] += 1
            last_b_cycle = cycle
        elif kind in {'MAW', 'MW'}:
            need(job in MEMORY_JOBS and responses == len(CASES[job-1][1]) and address is None and beat is None, 'destination before source drain')
            need(cycle > last_b_cycle and None not in bank, 'destination before complete bytes')
            if kind == 'MAW':
                need(maw is None and int(f['addr'], 16) == destination(job) and int(f['id'], 16) == 0 and int(f['len']) == 0 and int(f['size']) == 6, 'destination AW')
                maw = cycle
            else:
                need(mw is None and int(f['data'], 16).to_bytes(64, 'little') == bytes(bank) and int(f['strb'], 16) == (1 << 64)-1 and f['last'] == '1', 'destination byte reference')
                mw = cycle
                counts['writes'] += 1
        elif kind == 'MB':
            need(maw is not None and mw is not None and cycle > max(maw, mw) and mb is None, 'destination B order')
            ident, resp = int(f['id'], 16), int(f['resp'])
            need(ident == 0 and resp == ({11: 2, 12: 3}.get(job, 0)), 'destination B response')
            mb = cycle
        elif kind == 'PHASE':
            phase, busy, idle, status = (int(f[k]) for k in ['phase', 'busy', 'idle', 'status'])
            expected = [1, 2, 3, 4, 5, 0]
            need(job in WRITE_JOBS and len(phases) < 6 and phase == expected[len(phases)], 'phase order')
            if phase in {1, 2, 3}:
                need(status == 0 and busy == 0 and idle == 1, 'early phase')
            if phase in {4, 5, 0}:
                need(status == (4 if job in {11, 12, 13, 14, 20} else 0), 'destination error propagation')
                need(busy == 0 and idle == 1, 'guard/receiver not drained')
                if job in MEMORY_JOBS:
                    need(mb is not None and cycle > mb, 'bridge before final B')
                else:
                    need(job in {13, 14, 20} and maw is mw is mb is None, 'rejected destination reached memory')
            phases.append(phase)
        elif kind == 'STOP':
            need(job in set(range(6, 11)) and not stops and int(f['phase']) == job-5, 'stop phase coverage')
            stops.append(cycle)
        elif kind == 'HOLD':
            need(job == 15 and mb is None and maw is not None and mw is not None, 'hold context')
            need(f['busy'] == '1' and f['idle'] == '0' and f['result'] == '0' and f['arm'] == '0' and int(f['epoch'], 16) == job and int(f['peer'], 16) == job, 'hold lease/epoch')
            need(not holds or cycle == holds[-1]+1, 'hold continuity')
            holds.append(cycle)
            counts['held'] += 1
        elif kind == 'RESULT':
            need(armed and not retired and responses == len(CASES[job-1][1]) and address is None and beat is None, 'retirement source order')
            need(int(f['epoch'], 16) == job and int(f['tag'], 16) == 0x1000+job and int(f['status']) == CASES[job-1][2], 'result identity/status')
            if job in MEMORY_JOBS:
                need(maw is not None and mw is not None and mb is not None and cycle > mb and phases == [1, 2, 3, 4, 5, 0], 'retirement destination order')
            elif job in WRITE_JOBS:
                need(job in {13, 14, 20} and maw is mw is mb is None and phases == [1, 2, 3, 4, 5, 0], 'rejection completion path')
            else:
                need(maw is mw is mb is None and not phases, 'failed partial line wrote destination')
            need(len(holds) == (100 if job == 15 else 0), 'bounded missing B coverage')
            need(len(stops) == (1 if job in set(range(6, 11)) else 0), 'stop coverage')
            need(len(foreign) == 0, 'foreign B coverage')
            measurements.append(dict(job=job, arm=arm_cycle, source_b=last_b_cycle, aw=maw, w=mw, final_b=mb, result=cycle, holds=len(holds), status=int(f['status'])))
            retired = True
            counts['results'] += 1
        elif kind == 'PASS':
            need(job == 21 and retired, 'incomplete suite')
            value = {k: int(v) for k, v in f.items()}
            need(all(value[k] == n for k, n in counts.items()), 'summary counts')
            need(value['jobs'] == 21 and value['seed'] == seed and value['cycles'] >= last_cycle, 'summary identity/time')
            need(counts == dict(aw=309, w=309, b=309, writes=14, foreign=0, results=21, held=100) and byte_writes == 1110, 'fixture totals')
            need(all(value[k] > 0 for k in ['aw_stall', 'w_stall', 'b_stall', 'result_stall']), 'stall coverage')
    return dict(summary=value, byte_writes=byte_writes, destination_bytes=counts['writes']*64, jobs=measurements)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('log', type=Path)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    print(json.dumps(review(a.log.read_text(), a.seed), indent=2))
