"""Independent event-list reference for binary RC write-selection controls."""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path


def expected(flags, bitmap, retained, data, user):
    if flags != 7:
        return 0
    thermometers = {0: 0, 1: 1, 3: 2, 7: 3, 15: 4}
    sop = {4 * ((user >> (68 + 2 * i)) & 3) for i in range(thermometers.get((user >> 64) & 15, 0))}
    eop = {(user >> (80 + 4 * i)) & 15: i for i in range(thermometers.get((user >> 76) & 15, 0))}
    tag = retained & 255
    owned, terminal, normal = [(retained >> bit) & 1 for bit in (8, 9, 10)]
    if (user >> 96) & 1:
        normal = 0
    released, writes = set(), []
    result = 0
    for lane in sorted(sop | eop.keys()):
        if lane in sop:
            header = data >> (32 * lane)
            tag, terminal = (header >> 64) & 255, (header >> 30) & 1
            normal = not ((header >> 12) & 15 or (header >> 43) & 15)
            owned = ((bitmap >> tag) & 1) and tag not in released
            preceding = [event for event in writes if event['tag'] == tag]
            if preceding:
                newest = max(preceding, key=lambda event: event['lane'])
                result |= 1 << (lane // 4)
                result |= newest['slot'] << (4 + 2 * (lane // 4))
            if terminal and owned:
                released.add(tag)
        if lane in eop:
            if owned and not terminal and normal:
                writes.append(dict(tag=tag, slot=eop[lane], lane=lane))
            owned = False
    last_by_tag = {event['tag']: event for event in writes}
    for tag, event in last_by_tag.items():
        result |= 1 << (12 + tag)
        result |= event['slot'] << (268 + 2 * tag)
    return result


def user_bits(starts, ends, sop, eop, discontinue=0):
    return (starts << 64 | ends << 76 | discontinue << 96 |
            sum(p << (68 + 2 * i) for i, p in enumerate(sop)) |
            sum(p << (80 + 4 * i) for i, p in enumerate(eop)))


def headers(tags, terminal=0, abnormal=0):
    return sum((tag << 64 | ((terminal >> i) & 1) << 30 |
                ((abnormal >> i) & 1) << 12) << (128 * i) for i, tag in enumerate(tags))


def vectors():
    # Every arrangement of four EOP pointers, including duplicates/reversed order.
    for ends in itertools.product(range(16), repeat=4):
        yield (7, 1 << 7, 7 | 256 | 1024, headers([7] * 4),
               user_bits(15, 15, [0, 1, 2, 3], ends))
    # Every arrangement of SOP pointers, with terminal releases, retained context,
    # alternate tags, malformed masks, discontinue and overlapping SOP/EOP.
    masks = [0, 1, 3, 7, 15, 2, 5, 8, 11, 14, 15, 15, 15, 15, 15, 15]
    for starts in itertools.product(range(4), repeat=4):
        for kind in range(16):
            yield (7, (1 << 7) | (1 << 8), 7 | 256 | (kind & 1) * 512 | 1024,
                   headers([7, 8, 7, 8] if kind & 2 else [7] * 4, kind, kind >> 1),
                   user_bits(masks[kind], masks[15 - kind], starts,
                             [(kind + 3 * n) % 16 for n in range(4)], kind & 1))
    rng = random.Random(0x52435345)
    for _ in range(20000):
        yield (rng.randrange(8), rng.getrandbits(256), rng.getrandbits(11),
               rng.getrandbits(512), rng.getrandbits(161))


def generate():
    lines = []
    for flags, bitmap, retained, data, user in vectors():
        value = expected(flags, bitmap, retained, data, user)
        lines.append(f'{flags:x} {bitmap:064x} {retained:03x} {data:0128x} {user:041x} {value:0195x}')
    assert len(lines) == 89632
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    text = generate()
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    manifest = dict(rows=89632, exhaustive_eop_pointer_rows=65536,
                    sop_pointer_and_context_rows=4096, sampled_rows=20000,
                    random_seed=0x52435345, sha256=hashlib.sha256(text.encode()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scope='Binary control selection only. No byte arithmetic, four-state universal equivalence or full monitor qualification.')
    with args.output.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
