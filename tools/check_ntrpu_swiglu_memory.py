"""Independent traffic and exact ABI review of the memory-fed activation bank."""
import re
import struct

GATE = 0x100000FC0
UP = 0x8000000FC0
CASES = [(1, 0), (31, 0), (32, 0), (33, 0), (720, 0), (0, 1), (8193, 1)]
CASES += [(33 if mode == 26 else 64, mode) for mode in range(2, 30)]


def expected_reads(elements):
    result = []
    for first in range(0, elements, 32):
        size = ((min(elements - first, 32) + 7) // 8) * 64
        for base in [GATE, UP]:
            address, remaining = base + first * 8, size
            while remaining:
                count = min(remaining, 4096 - address % 4096)
                result.append((address, count // 64))
                address += count
                remaining -= count
    return result


def expected_record(case_id, elements, block, varying_gate=False):
    data = bytearray(64)
    for lane in range(min(32, elements - block * 32)):
        value = 127 if lane == 0 else -1 if case_id % 3 == 0 else (lane * 11 + block * 37) % 255 - 127
        data[lane] = value % 256
    # SiLU(g) rounds to g in FP32 for these large positive integral inputs.
    # Each block contains up[0]=127, so amax/127 is exactly g and q=up.
    data[32:40] = struct.pack('<d', 257 + 2 * block if varying_gate else 257)
    return bytes(data)


def review(text, seed, large=False, varying_gate=False):
    errors = []

    def need(ok, why):
        if not ok:
            errors.append(why)

    contracts = [(8192, 0)] if large else CASES
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)', text, re.M | re.I), 'Executed failure')
    need(re.findall(r'^NTRPU_SWIGLU_MEMORY_SEED seed=(\d+)\s*$', text, re.M) == [str(seed)], 'Seed')
    patterns = re.findall(r'^NTRPU_SWIGLU_MEMORY_PATTERN gate=(\w+)\s*$',text,re.M)
    need(patterns == (['257_plus_2_block'] if varying_gate else []), 'Gate pattern')
    rows = [{k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
            for line in text.splitlines() if line.startswith('NTRPU_SWIGLU_MEMORY_CASE ')]
    need([row.get('id') for row in rows] == list(range(1, len(contracts) + 1)), 'Case identities')
    ars = [(int(i), int(a, 16), int(n)) for i, a, n in re.findall(
        r'^NTRPU_SWIGLU_AR id=(\d+) address=([0-9a-f]{16}) beats=(\d+)\s*$', text, re.M)]
    writes = [(int(i), int(b), h) for i, b, h in re.findall(
        r'^NTRPU_SWIGLU_W id=(\d+) block=(\d+) data=([0-9a-f]{128})\s*$', text, re.M)]
    need(len(ars) == len(re.findall(r'^NTRPU_SWIGLU_AR ', text, re.M)), 'AR record format')
    need(len(writes) == len(re.findall(r'^NTRPU_SWIGLU_W ', text, re.M)), 'W record format')
    need([i for i, _, _ in ars] == sorted(i for i, _, _ in ars), 'AR case order')
    need([i for i, _, _ in writes] == sorted(i for i, _, _ in writes), 'W case order')
    for row, (elements, mode) in zip(rows, contracts):
        ident = row.get('id')
        status = 0
        if mode in {1, 2, 3, 4, 5, 27, 28, 29}: status = 1
        if mode == 6: status = 2
        if mode in {7, 8, 9}: status = 3
        if mode in {10, 11}: status = 5
        if mode in {12, 13}: status = 4
        if mode in {14, 15, 16, 17, 18}: status = 7
        if mode in {19, 20}: status = 8
        need(set(row) == {'id','elements','scenario','status','reset','cycles','ar','r','aw','w','b','stored','hls','cancel'}, 'Fields')
        need((row.get('elements'),row.get('scenario'),row.get('status')) == (elements,mode,status), 'Case contract')
        need(row.get('reset') == int(21 <= mode <= 25) and row.get('cancel') == int(status == 7), 'Reset/cancel contract')
        full_reads = expected_reads(elements)
        blocks = (elements + 31) // 32
        # (AR, R, AW, W, B, physically stored records, HLS launches)
        traffic = (len(full_reads), 2 * ((elements + 7) // 8), blocks, blocks, blocks, blocks, blocks)
        if status in {1,2}: traffic = (0,0,0,0,0,0,0)
        traffic = {
            7:(1,1,0,0,0,0,0), 8:(3,5,0,0,0,0,0), 9:(5,12,1,1,1,1,1),
            10:(4,8,0,0,0,0,1), 11:(6,16,1,1,1,1,2), 12:(6,16,2,2,2,2,2),
            13:(4,8,1,1,1,1,1), 14:(2,4,0,0,0,0,0), 15:(2,4,0,0,0,0,0),
            16:(4,8,0,0,0,0,1), 17:(4,8,1,1,1,1,1), 18:(5,12,1,1,1,1,1),
            19:(2,4,0,0,0,0,0), 20:(4,8,1,1,1,1,1), 21:(1,1,0,0,0,0,0),
            22:(4,8,0,0,0,0,1), 23:(4,8,0,1,0,0,1), 24:(4,8,1,1,0,1,1),
            25:(4,8,1,1,1,1,1)}.get(mode, traffic)
        need(tuple(row.get(k) for k in ['ar','r','aw','w','b','stored','hls']) == traffic, 'Traffic '+str(ident))
        need(0 < row.get('cycles',0) < 2000000, 'Latency')
        if status == 8: need(row.get('cycles',0) >= 4144, 'Timeout stimulus')
        need([(a,n) for i,a,n in ars if i==ident] == full_reads[:traffic[0]], 'Independent AR plan '+str(ident))
        actual = [(b,h) for i,b,h in writes if i==ident]
        expected = [(b,expected_record(ident,elements,b,varying_gate)[::-1].hex()) for b in range(traffic[3])]
        need(actual == expected, 'Independent ABI records '+str(ident))
    need(len(ars) == sum(row.get('ar',0) for row in rows), 'No extra AR cases')
    need(len(writes) == sum(row.get('w',0) for row in rows), 'No extra W cases')
    markers = re.findall(r'^NTRPU_SWIGLU_MEMORY_PASS (.+)$',text,re.M)
    need(len(markers)==1,'Suite marker')
    summary = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',markers[0])} if markers else {}
    need(summary.get('cases') == len(contracts), 'Suite count')
    for key in ['ar','r','w','b']: need(summary.get(key)==sum(row.get(key,0) for row in rows),'Total '+key)
    for key in ['ar_stalls','aw_stalls','w_stalls','done_stalls']: need(summary.get(key,0)>0,'Coverage '+key)
    for key in ['resets','cancels']: need(summary.get(key)==(0 if large else 5),'Coverage '+key)
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=rows,
                records_checked=len(writes),summary=summary)
