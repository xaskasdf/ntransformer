"""Additional raw-input scenarios for effective RC ancestry and byte prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors, place, rq, rc, rc_header


def quad(tag, remaining, address, counts=(1, 1, 1, 1), terminal_last=False,
         ends=(3, 7, 11, 15), starts=(0, 1, 2, 3)):
    pieces = []
    user = 15 << 64 | 15 << 76
    for i, (sop, eop) in enumerate(zip(starts, ends)):
        user |= sop << (68 + 2 * i) | eop << (80 + 4 * i)
    for i, words in enumerate(counts):
        terminal = terminal_last and i == 3
        pieces.append((16 * i, rc_header(tag, words, remaining, address, terminal) + bytes([i] * 4)))
        delta = min(remaining, (4 * words - (address & 3)) & 32767)
        # There is exactly one physical payload DWORD per header. Bad counts
        # intentionally disagree with it; do not construct oversized user fields.
        enabled = min(delta, 4 - (address & 3))
        user |= ((1 << enabled) - 1) << (16 * i + 12 + (address & 3))
        if not terminal:
            remaining -= delta
            address = (address + delta) & 4095
    return place(*pieces), user


def generate():
    v = Vectors()
    for words in (4, 8, 16):
        for offset in range(4):
            v.case_begin(f'prefix_four_splits_{words}_{offset}')
            v.step(rq_beat=rq(41, count=words, first=(15 << offset) & 15))
            v.step(rc_beat=quad(41, 4 * words - offset, 0x100 + offset, terminal_last=words == 4))
            if words > 4:
                remaining = 4 * words - 16
                v.step(rc_beat=rc(41, bytes(remaining), address=0x110, remaining=remaining))
            v.end()
    for counts in [(0, 1, 1, 1), (1, 0, 1, 1), (1024, 1, 0, 1), (2047, 0, 1, 0), (1, 1, 1, 0)]:
        for offset in range(4):
            v.case_begin('prefix_fault_counts_' + '_'.join(map(str, (*counts, offset))))
            v.step(rq_beat=rq(42, count=8, first=(15 << offset) & 15))
            v.step(rc_beat=quad(42, 32 - offset, 0x100 + offset, counts, terminal_last=True),
                   rq_beat=rq(42, 0x200), expect_error=True)
            v.step(rc_beat=rc(42, address=0x200))
            v.end()
    for offset in range(4):
        v.case_begin(f'prefix_retained_then_three_{offset}')
        v.step(rq_beat=rq(43, count=20, first=(15 << offset) & 15))
        raw = rc_header(43, 16, 80 - offset, 0x100 + offset, terminal=False) + bytes(range(64))
        v.step(rc_beat=(place((0, raw[:64])), ((1 << (52 - offset)) - 1) << (12 + offset) | 1 << 64))
        pieces = [(0, raw[64:])]
        user = 0xfff | 7 << 64 | 15 << 76 | 2 << 80
        for i in range(3):
            pieces.append((16 * (i + 1), rc_header(43, 1, 16 - 4 * i, 0x140 + 4 * i, terminal=False) + bytes(4)))
            user |= (i + 1) << (68 + 2 * i) | (7 + 4 * i) << (84 + 4 * i) | 15 << (28 + 16 * i)
        v.step(rc_beat=(place(*pieces), user))
        v.step(rc_beat=rc(43, address=0x14c, remaining=4))
        v.end()
    for request_count, first, last in [(0, 1, 0), (0, 8, 1), (1025, 15, 15), (2047, 15, 15)]:
        first_index = (first & -first).bit_length() - 1
        last_index = last.bit_length() - 1 if last else 0
        remaining = ((request_count - 1) * 4 + last_index - first_index + 1) & 8191
        for counts in [(1, 1, 1, 1), (0, 1024, 0, 2047), (2047, 0, 1, 0)]:
            v.case_begin(f'prefix_fault_seed_{request_count}_{first}_{counts[0]}')
            v.step(rq_beat=rq(44, 0, count=request_count, first=first, last=last), expect_error=True)
            v.step(rc_beat=quad(44, remaining, first_index, counts, terminal_last=True), expect_error=True)
            v.end()
    for kind, ends, starts in [('reverse_eop', (15, 11, 7, 3), (0, 1, 2, 3)),
                               ('duplicate_eop', (3, 3, 11, 15), (0, 1, 2, 3)),
                               ('reverse_sop', (3, 7, 11, 15), (3, 1, 0, 2))]:
        for offset in range(4):
            v.case_begin(f'prefix_fault_{kind}_{offset}')
            v.step(rq_beat=rq(45, count=8, first=(15 << offset) & 15))
            v.step(rc_beat=quad(45, 32 - offset, 0x100 + offset, ends=ends, starts=starts), expect_error=True)
            v.end()
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    v = generate()
    text = '\n'.join(v.rows) + '\n'
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    result = dict(rows=len(v.rows), scenarios=v.scenarios, vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_rows=0, scope='Raw-input prefix integration supplement. Python state expectations stop on malformed traffic; the frozen R7 RTL reference and full table/state comparisons remain active after faults.')
    with args.output.with_suffix('.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(rows=len(v.rows), scenarios=len(v.scenarios), sha256=result['vectors_sha256'])))


if __name__ == '__main__':
    main()
