"""Boundary-oriented binary descriptors for the four-lane registered proposal unit."""
import hashlib
import json
import random
from pathlib import Path


def descriptor(count, byte_count, address, terminal, error=0, status=0, poison=0, salt=0):
    h=address | error<<12 | byte_count<<16 | terminal<<30 | count<<32 | status<<43 | poison<<46
    # Populate opaque fields independently; the arithmetic must not use them.
    return h | (salt & 0xffff)<<48 | ((salt>>16)&0xff)<<64 | ((salt>>24)&0xffffff)<<72 | ((salt>>48)&0xffffffff)<<96


def generate():
    rng=random.Random(0x52435052)
    rows=[];phases=[]
    def phase(name):phases.append(dict(id=len(phases),name=name,start=len(rows),rows=0))
    def emit(headers, valid=15, rst=1):
        assert len(headers)==4
        data=sum(h<<(128*i) for i,h in enumerate(headers))
        rows.append(f'{phases[-1]["id"]} {rst:x} {valid:x} {data:0128x}')
        phases[-1]['rows']+=1
    phase('reset_and_inactive_outputs')
    emit([rng.getrandbits(128) for _ in range(4)],rst=0)
    emit([rng.getrandbits(128) for _ in range(4)],valid=0)
    phase('all_counts_low_offsets_and_boundary_byte_counts')
    pending=[]
    for count in range(2048):
        for low in range(4):
            available=count*4-low
            counts={0,1,2,3,4,4095,4096,4097,8190,8191}
            counts.update(v for v in [available-1,available,available+1] if 0<=v<=8191)
            for bc in sorted(counts):
                for terminal in range(2):
                    pending.append(descriptor(count,bc,(rng.randrange(1024)<<2)|low,terminal,salt=rng.getrandbits(80)))
                    if len(pending)==4:emit(pending);pending=[]
    if pending:emit(pending+[0]*(4-len(pending)),valid=(1<<len(pending))-1)
    phase('non_normal_status_error_poison')
    for error in range(16):
        for status in range(8):
            for poison in range(2):
                emit([descriptor(rng.randrange(2048),rng.randrange(8192),rng.randrange(4096),rng.randrange(2),error,status,poison,rng.getrandbits(80)) for _ in range(4)])
    phase('random_full_width_and_lane_masks')
    for i in range(16384):
        headers=[descriptor(rng.randrange(2048),rng.randrange(8192),rng.randrange(4096),rng.randrange(2),
            error=0 if i%3 else rng.randrange(16),status=0,poison=int(i%29==0),salt=rng.getrandbits(80)) for _ in range(4)]
        emit(headers,valid=i%16,rst=int(i%997!=0))
    phase('reset_after_contention')
    emit([rng.getrandbits(128) for _ in range(4)],rst=0)
    emit([rng.getrandbits(128) for _ in range(4)],valid=0)
    text='\n'.join(rows)+'\n'
    return text,dict(rows=len(rows),phases=phases,seed=0x52435052,
        vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
        scope='Binary proposal arithmetic only; expected metadata, ownership, framing, payload context and inventory are external.',
        lanes=4,latency_cycles=1)


if __name__=='__main__':
    text,manifest=generate();base=Path(__file__).resolve().parents[1]/'fpga/tb/data/pcie_rc_proposal'
    with base.with_suffix('.txt').open('x',newline='\n') as f:f.write(text)
    with base.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))
