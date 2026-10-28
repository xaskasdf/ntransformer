"""Independent fragment/control/receiver/capacity composition replay.

The fixture sends three aligned but invalid 64-bit destinations and requires
zero downstream transactions, then recovers without reset. Scope is one clock,
one line, a compliant destination and externally trusted isolation barriers.
"""
import argparse
import json
from pathlib import Path
import re

CASES = [
    ('complete', [0,0], 0), ('stale_identity_and_identical_duplicate', [2,2,2,0,0,0], 0),
    ('conflicting_duplicate', [0,2], 4), ('holes_stop', [0], 1),
    ('upper_address_rejection', [3,0,0], 0),
    *[(f'stop_phase_{i}', [0,0], 1) for i in range(1,6)],
    ('destination_slverr',[0,0],5), ('destination_decerr',[0,0],5),
    ('first_invalid_destination',[0,0],5), ('high_bit_destination_alias',[0,0],5),
    ('missing_final_b',[0,0],0), ('recovery_without_reset',[2,0,0],0),
    ('stop_partial_aw',[2],1), ('zero_destination',[0,0],0),
    ('above_4gib_destination',[0,0],0), ('destination_end_overflow',[0,0],5),
    ('zero_recovery_after_rejections',[0,0],0),
]
WRITE_JOBS = set(range(1,22)) - {3,4,17}
MEMORY_JOBS = WRITE_JOBS - {13,14,20}


def telemetry(job):
    frames=7 if job>=17 else 6 if job>=16 else 5 if job>=5 else 4 if job>=3 else 3 if job>=2 else 0
    identities=3 if job>=16 else 2 if job>=2 else 0
    reason=0x1 if job>=17 else 0x307 if job>=16 else 0x43 if job>=5 else 0x4 if job>=3 else 0x82 if job>=2 else 0
    return frames,identities,reason


def destination(job):
    return {13: 0x2000000000, 14: 0x1000000000000000, 18: 0,
            19: 0x100000040, 20: 0xffffffffffffffc0, 21: 0}.get(job, 0x1fffffffc0)

FIELDS = {
    'TELEMETRY': 'frames identity reason cycle',
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


def fixture(job, index, seed):
    window = 0x1000fffc0 if job % 2 else 0x100000000
    e,t=0x100000000+job,0x8000000100001000+job
    pattern_job=job
    offset=32*index
    if job==2:
        offset=[0,0,0,32,32,0][index]
        if index==0: e ^= 1<<40
        if index==1: t ^= 1<<40
    elif job==3: offset=0
    elif job==5:
        offset=32*max(0,index-1)
        if index==0: window ^= 1<<32
    elif job==16:
        offset=32*max(0,index-1)
        if index==0: e-=2;t-=2;pattern_job=14
    payload=bytes((pattern_job*37+i*19+seed*7)&255 for i in range(64))
    if job==3 and index==1: payload=bytes([payload[0]^1])+payload[1:]
    packet=b'NTRPUF01'+e.to_bytes(8,'little')+t.to_bytes(8,'little')+offset.to_bytes(8,'little')+payload[offset:offset+32]
    if job==2 and index==2: packet=bytes([packet[0]^1])+packet[1:]
    return window,(1<<64)-1,int.from_bytes(packet,'little')


def review_data(text, seed):
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
            telemetry_seen=False
        elif kind == 'ARM':
            need(job and not armed and not retired, 'unexpected ARM')
            need(int(f['epoch'], 16) == 0x100000000+job and int(f['tag'], 16) == 0x8000000100001000+job, 'ARM identity')
            window = 0x1000fffc0 if job % 2 else 0x100000000
            need(int(f['window'], 16) == window and int(f['dest'], 16) == destination(job), 'ARM address')
            armed, arm_cycle = True, cycle
        elif kind in {'AW', 'W'}:
            need(armed and not retired and responses < len(CASES[job-1][1]), 'source context')
            addr, mask, data = fixture(job, responses, seed)
            if kind == 'AW':
                need(address is None, 'overwritten AW')
                address = {k: int(v, 16 if k in {'addr', 'id'} else 10) for k, v in f.items()}
                need(address['addr'] == addr and address['len'] == 0 and address['size'] == 6, 'source address')
                expected_id = 5 if job == 17 else (counts['b']+3)%16
                need(address['id'] == expected_id, 'source request ID')
                counts['aw'] += 1
            else:
                need(beat is None, 'overwritten W')
                beat = {k: int(v, 16 if k in {'data', 'strb'} else 10) for k, v in f.items()}
                need(beat['data'] == data and beat['strb'] == mask and beat['last'] == 1, 'source data/mask/last')
                counts['w'] += 1
        elif kind == 'B':
            need(address is not None and beat is not None and cycle > max(address['cycle'], beat['cycle']), 'source B before channels')
            need(int(f['id'], 16) == address['id'] and int(f['resp']) == CASES[job-1][1][responses], 'source B status/ID')
            if int(f['resp']) == 0:
                need(address['addr']//64*64 == window, 'reservation address')
                raw = beat['data'].to_bytes(64, 'little')
                need(raw[:8] == b'NTRPUF01' and int.from_bytes(raw[8:16],'little') == 0x100000000+job and
                     int.from_bytes(raw[16:24],'little') == 0x8000000100001000+job, 'accepted frame identity')
                offset=int.from_bytes(raw[24:32],'little')
                need(offset in {0,32}, 'accepted frame offset')
                for lane in range(32):
                    dest=offset+lane
                    need(bank[dest] is None or bank[dest] == raw[32+lane], 'conflicting byte accepted')
                    bank[dest]=raw[32+lane]
                    byte_writes+=1
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
            need(f['busy'] == '1' and f['idle'] == '0' and f['result'] == '0' and f['arm'] == '0' and int(f['epoch'], 16) == 0x100000000+job and int(f['peer'], 16) == 0x100000000+job, 'hold lease/epoch')
            need(not holds or cycle == holds[-1]+1, 'hold continuity')
            holds.append(cycle)
            counts['held'] += 1
        elif kind == 'TELEMETRY':
            need(not telemetry_seen and not retired, 'duplicate telemetry')
            need((int(f['frames']),int(f['identity']),int(f['reason'],16)) == telemetry(job), 'telemetry counts/reason')
            telemetry_seen=True
        elif kind == 'RESULT':
            need(telemetry_seen, 'missing telemetry')
            need(armed and not retired and responses == len(CASES[job-1][1]) and address is None and beat is None, 'retirement source order')
            need(int(f['epoch'], 16) == 0x100000000+job and int(f['tag'], 16) == 0x8000000100001000+job and int(f['status']) == CASES[job-1][2], 'result identity/status')
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
            need(counts == dict(aw=46, w=46, b=46, writes=15, foreign=0, results=21, held=100) and byte_writes == 1248, 'fixture totals')
            need(all(value[k] > 0 for k in ['aw_stall', 'w_stall', 'b_stall', 'result_stall']), 'stall coverage')
    return dict(summary=value, byte_writes=byte_writes, destination_bytes=counts['writes']*64, jobs=measurements)


def review(text, seed):
    result = review_data(text, seed)
    writes = []
    reads = [(0,0x4652544e),(4,0x10000),(8,0x1f),(0x9c,0x5052544e),(0xa0,0x31304655),(0xa4,64),(0xa8,32)]
    for job in range(1, 22):
        win = 0x1000fffc0 if job % 2 else 0x100000000
        tag = 0x8000000100001000+job
        current_epoch=0x100000000+job
        dest = destination(job)

        def desc(epoch, window, address):
            for index, value in enumerate((epoch, tag, window, address)):
                writes.extend([(0x20+index*8,value & 0xffffffff,0),
                               (0x24+index*8,value >> 32,0)])

        desc(current_epoch, win, dest)
        writes.extend([(0x10,1,0),(0x20,0,2),(0x10,4,2)])
        if job == 2:
            writes.append((0x10,8,0));reads.append((0x90,0))
        if job in {4,6,7,8,9,10,17}:
            writes.append((0x10,2,0))
        if job == 15:
            writes.extend([(0x10,1,2),(0x10,4,2)])
        writes.append((0x10,8,0))
        rejected,identities,reason=telemetry(job)
        reads.extend([(0x90,rejected),(0x94,identities),(0x98,reason),(0xac,1)])
        reads.extend([(0x60,job),(0x64,1),(0x68,tag & 0xffffffff),(0x6c,tag >> 32),(0x70,CASES[job-1][2])])
        writes.extend([(0x40,job,0),(0x44,1,0),(0x48,tag & 0xffffffff,0),(0x4c,(tag >> 32)^1,0),
                       (0x10,4,2),(0x4c,tag >> 32,0),(0x10,4,0)])
        for epoch,window,address in [(current_epoch,win,dest),(current_epoch+1,win+1,dest),(current_epoch+1,win,dest+1)]:
            desc(epoch,window,address)
            writes.append((0x10,1,2))
    actual_writes = []
    actual_reads = []
    terminal = []
    last_cycle = 0
    for line in text.splitlines():
        if not line.startswith('NTRPU_MANAGED_'):
            continue
        label,*words=line.split()
        f=dict(word.split('=',1) for word in words)
        need(len(f)==len(words), 'duplicate field')
        if label=='NTRPU_MANAGED_MMIO':
            need(set(f)==set('n addr data resp cycle'.split()), 'write schema')
            actual_writes.append((int(f['addr'],16),int(f['data'],16),int(f['resp'])))
            need(int(f['n'])==len(actual_writes), 'write numbering')
        elif label=='NTRPU_MANAGED_READ':
            need(set(f)==set('n addr data cycle'.split()), 'read schema')
            actual_reads.append((int(f['addr'],16),int(f['data'],16)))
            need(int(f['n'])==len(actual_reads), 'read numbering')
        elif label=='NTRPU_MANAGED_PASS':
            need(set(f)==set('writes reads jobs seed'.split()), 'summary schema')
            terminal.append({k:int(v) for k,v in f.items()})
        else:
            raise ValueError('unknown managed event')
        if 'cycle' in f:
            need(int(f['cycle'])>=last_cycle, 'MMIO chronology')
            last_cycle=int(f['cycle'])
    need(actual_writes==writes and len(writes)==976, 'MMIO fixture')
    need(actual_reads==reads and len(reads)==197, 'readback fixture')
    need(terminal==[dict(writes=976,reads=197,jobs=21,seed=seed)], 'managed summary')
    result.update(control=terminal[0], rejected_control_writes=sum(r!=0 for _,_,r in writes))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();print(json.dumps(review(a.log.read_text(),a.seed),indent=2))
