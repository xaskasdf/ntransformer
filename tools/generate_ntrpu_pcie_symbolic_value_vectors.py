"""Raw contexts that must survive independently of tag reuse and current flags."""
import hashlib
import json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors, place, rq, rc, rc_header
from ntrpu_pcie_straddle import LogicWord


def request(tag, address=0x100, count=1, tc=0, attr=0, first=15, last=None):
    data, user = rq(tag, address, count, first, last)
    return data | tc << 121 | attr << 124, user


def completion(tag, payload=b'abcd', address=0x100, remaining=None, tc=0, attr=0):
    data, user = rc(tag, payload, address, remaining)
    return data | tc << 89 | attr << 92, user


def start(tag, remaining, terminal, tc=0, attr=0):
    raw = rc_header(tag, 16, remaining, 0x100, terminal, tc=tc, attr=attr) + bytes(range(52))
    return place((0, raw)), ((1 << 52) - 1) << 12 | 1 << 64


def tail(discontinue=False):
    return place((0, bytes(range(52, 64)))), 0xfff | 1 << 76 | 2 << 80 | int(discontinue) << 96


def change_unknown_to_z(v, field, bit):
    fields = v.rows[-1].split()
    data = list(fields[field])
    assert len(data) == 512 and data[511-bit] == 'x'
    data[511-bit] = 'z'
    fields[field] = ''.join(data)
    v.rows[-1] = ' '.join(fields)


def generate():
    v = Vectors()
    for terminal in (True, False):
        for pause in (1, 257):
            v.case_begin(f'value_retained_terminal_{int(terminal)}_pause_{pause}')
            v.step(rq_beat=request(61, count=16 if terminal else 20, tc=5, attr=3))
            v.step(rc_beat=start(61, 64 if terminal else 80, terminal, 5, 3),
                   rq_beat=request(61, 0x200, tc=2, attr=1) if terminal else None)
            for _ in range(pause):
                v.step()
            v.step(rc_beat=tail())
            if terminal:
                v.step(rc_beat=completion(61, address=0x200, tc=2, attr=1))
            else:
                v.step(rc_beat=completion(61, bytes(16), 0x140, 16, 5, 3))
            for _ in range(3):
                v.step()
            v.end()
    for terminal in (True, False):
        v.case_begin(f'value_late_discontinue_terminal_{int(terminal)}')
        v.step(rq_beat=request(62, count=16 if terminal else 20, tc=6, attr=2))
        v.step(rc_beat=start(62, 64 if terminal else 80, terminal, 6, 2),
               rq_beat=request(62, 0x300, tc=1, attr=1) if terminal else None)
        for _ in range(17):
            v.step()
        v.step(rc_beat=tail(True), expect_error=True)
        v.step(rc_beat=completion(62, address=0x300, tc=1, attr=1), expect_error=True)
        v.end()
    v.case_begin('value_replaced_incomplete_context')
    v.step(rq_beat=request(70, count=20, tc=7, attr=3))
    v.step(rq_beat=request(71, 0x200, tc=1, attr=2))
    v.step(rc_beat=start(70, 80, False, 7, 3))
    for _ in range(33):
        v.step()
    v.step(rc_beat=completion(71, address=0x200, tc=1, attr=2), expect_error=True)
    v.step(rq_beat=request(71, 0x300, tc=2, attr=1))
    v.step(rc_beat=completion(71, address=0x300, tc=2, attr=1))
    v.end()
    for offset in range(4):
        v.case_begin(f'value_zero_count_chain_offset_{offset}')
        v.step(rq_beat=request(72, count=8, first=(15 << offset) & 15))
        zero = place((0, rc_header(72, 0, 32-offset, 0x100+offset, False))), 1 << 64 | 1 << 76 | 2 << 80
        for _ in range(65):
            v.step(rc_beat=zero, expect_error=True)
        v.step(rc_beat=completion(72, remaining=32-offset, address=0x100+offset), expect_error=True)
        v.end()
    for unknown in ('x', 'z'):
        for which in ('rq_address', 'rc_count'):
            v.case_begin(f'value_unknown_{which}_{unknown}')
            if which == 'rq_address':
                data, user = request(73, count=8)
                bit = 8
                v.step(rq_beat=(LogicWord(data, ((1 << 512)-1) ^ (1 << bit), 512), user), expect_error=True)
                if unknown == 'z':
                    change_unknown_to_z(v, 10, bit)
            else:
                v.step(rq_beat=request(73, count=8))
                data = place((0, rc_header(73, 1, 32, 0x100, False) + b'abcd'))
                bit = 32
                v.step(rc_beat=(LogicWord(data, ((1 << 512)-1) ^ (1 << bit), 512),
                    15 << 12 | 1 << 64 | 1 << 76 | 3 << 80), expect_error=True)
                if unknown == 'z':
                    change_unknown_to_z(v, 14, bit)
            for _ in range(3):
                v.step(rc_beat=(place((0, rc_header(73, 1, 32, 0x100, False) + b'abcd')),
                    15 << 12 | 1 << 64 | 1 << 76 | 3 << 80), expect_error=True)
            v.step(rc_beat=completion(73, remaining=32), expect_error=True)
            v.end()
    v.case_begin('value_count_sum_saturation_across_beats')
    v.step(rq_beat=request(74, 0, count=2047), expect_error=True)
    for count in (1024, 1024, 1, 0, 2047, 0):
        v.step(rc_beat=(place((0, rc_header(74, count, 8188, 0, False) + b'abcd')),
            15 << 12 | 1 << 64 | 1 << 76 | 3 << 80), expect_error=True)
    v.step(rc_beat=completion(74, address=0, remaining=8188), expect_error=True)
    v.end()
    v.case_begin('value_cold_reset_during_detached_payload')
    v.step(rq_beat=request(75, count=16, tc=3, attr=2))
    v.step(rc_beat=start(75, 64, True, 3, 2), rq_beat=request(75, 0x200, tc=6, attr=1))
    for _ in range(7):
        v.step()
    # The next scenario starts with cold reset; pending work is intentionally canceled.
    v.case.update(last_vector=len(v.rows)-1, reference_fault=False,
        peak_tags=v.model.peak_tags, peak_reads=v.model.peak_reads, peak_span_bytes=v.model.peak_span_bytes)
    v.case_begin('value_after_cold_reset_reuses_tag')
    v.step(rq_beat=request(75, 0x300, tc=4, attr=3))
    v.step(rc_beat=completion(75, address=0x300, tc=4, attr=3))
    v.end()
    return v


if __name__ == '__main__':
    import sys
    output = Path(sys.argv[1])
    v = generate()
    text = '\n'.join(v.rows) + '\n'
    with output.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    manifest = dict(rows=len(v.rows), scenarios=v.scenarios,
        vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), native_rows=0,
        scope='Raw symbolic value lifetime supplement; Python before faults and independent R7 for every post-fault row. Explicit x/z and cold cancellation during retained payload.')
    with output.with_suffix('.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2); f.write('\n')
    print(json.dumps(dict(rows=len(v.rows), scenarios=len(v.scenarios), sha256=manifest['vectors_sha256'])))
