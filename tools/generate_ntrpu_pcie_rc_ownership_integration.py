"""Raw RC ownership/release cases with post-fault traffic and RC-before-RQ reuse."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors, place, rq, rc, rc_header


def finish(v):
    if v.fault:
        v.step(rq_beat=rq(180, 0x700))
        v.step(rc_beat=rc(180, address=0x700))
    else:
        for (_, tag), entry in list(v.model.tags.items()):
            remaining = entry['remaining']
            v.step(rc_beat=rc(tag, bytes(remaining), address=entry['next_address'], remaining=remaining))
    v.end()


def generate():
    v = Vectors()
    partitions = [p for p in itertools.product(range(4), repeat=4)
                  if p[0] == 0 and all(p[i] <= max(p[:i]) + 1 for i in range(1, 4))]
    for part, pattern in enumerate(partitions):
        tags = [7 + n for n in pattern]
        expected_terminal = sum(1 << i for i, tag in enumerate(tags) if tag not in tags[i+1:])
        for allocation in range(16):
            for terminal in range(16):
                v.case_begin(f'rc_ownership_{part}_{allocation}_{terminal}')
                for i in range(4):
                    if allocation >> i & 1:
                        v.step(rq_beat=rq(7+i, count=max(1, tags.count(7+i))))
                pieces = []
                user = 15 << 64 | 0xe4 << 68 | 15 << 76 | 0xfb73 << 80
                for i, tag in enumerate(tags):
                    remaining = 4 * tags[i:].count(tag)
                    address = 0x100 + 4 * tags[:i].count(tag)
                    pieces.append((16*i, rc_header(tag, 1, remaining, address, bool(terminal >> i & 1)) + bytes(4)))
                    user |= 15 << (16*i+12)
                bad = terminal != expected_terminal or any(not (allocation >> (tag-7) & 1) for tag in tags)
                first_terminal = next((i for i in range(4) if terminal >> i & 1), None)
                reuse = rq(tags[first_terminal], 0x300) if first_terminal is not None else None
                v.step(rc_beat=(place(*pieces), user), rq_beat=reuse, expect_error=bad)
                finish(v)
    for retained_terminal in (False, True):
        for terminal in range(8):
            for discontinue in (False, True):
                v.case_begin(f'rc_retained_ownership_{int(retained_terminal)}_{terminal}_{int(discontinue)}')
                total = 64 if retained_terminal else 80
                v.step(rq_beat=rq(7, count=total//4))
                raw = rc_header(7, 16, total, 0x100, retained_terminal) + bytes(64)
                v.step(rc_beat=(place((0, raw[:64])), ((1 << 52)-1) << 12 | 1 << 64),
                       rq_beat=rq(7, 0x200, count=3) if retained_terminal else None)
                pieces = [(0, raw[64:])]
                user = 0xfff | 7 << 64 | 0x39 << 68 | 15 << 76 | 0xfb72 << 80 | int(discontinue) << 96
                for i in range(3):
                    remaining = (12 if retained_terminal else 16) - 4*i
                    address = (0x200 if retained_terminal else 0x140) + 4*i
                    pieces.append((16*(i+1), rc_header(7, 1, remaining, address, bool(terminal >> i & 1)) + bytes(4)))
                    user |= 15 << (28+16*i)
                bad = discontinue or terminal != (4 if retained_terminal else 0)
                v.step(rc_beat=(place(*pieces), user), expect_error=bad)
                finish(v)
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    v = generate()
    text = '\n'.join(v.rows) + '\n'
    manifest = dict(rows=len(v.rows), scenarios=v.scenarios, vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), native_rows=0,
                    scope='Raw RC ownership supplement; independent Python before faults and immutable reference RTL for every post-fault row; all binary rows replayed on own netlist.')
    with args.output.open('x', encoding='utf-8', newline='\n') as f: f.write(text)
    with args.output.with_suffix('.json').open('x', encoding='utf-8') as f: json.dump(manifest, f, indent=2); f.write('\n')
    print(json.dumps(dict(rows=len(v.rows), scenarios=len(v.scenarios), sha256=manifest['vectors_sha256'])))


if __name__ == '__main__':
    main()
