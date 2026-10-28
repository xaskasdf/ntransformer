"""Deterministic binary traffic for all tag addresses and contending update lanes."""
import hashlib
import json
import random
from pathlib import Path


def pack(values, width):
    return sum(value << (width * index) for index, value in enumerate(values))


def generate():
    rng = random.Random(0x50434945)
    rows, phases = [], []

    def phase(name):
        phases.append(dict(id=len(phases), name=name, start=len(rows), rows=0))

    def emit(reset=1, queries=None, writes=None, valid=15, full=0x30, write_valid=63):
        queries = queries if queries is not None else [(0, i) for i in range(4)]
        writes = writes if writes is not None else [(0, rng.getrandbits(30)) for _ in range(6)]
        values = [reset, valid, pack([q[0] for q in queries], 8),
            pack([q[1] for q in queries], 3), write_valid, full,
            pack([w[0] for w in writes], 8), pack([w[1] for w in writes], 30)]
        rows.append(str(phases[-1]['id']) + ' ' + ' '.join(f'{v:x}' for v in values))
        phases[-1]['rows'] += 1

    def read_all():
        for tag in range(0, 256, 4):
            emit(queries=[(tag+i, 0) for i in range(4)], write_valid=0)

    phase('cold_reset_overrides_all_ports')
    emit(reset=0, full=63)
    read_all()
    phase('fill_and_read_every_tag')
    for base in range(0, 256, 6):
        emit(writes=[((base+i) % 256, rng.getrandbits(30)) for i in range(6)],
             queries=[((base+i) % 256, 6) for i in range(4)], full=63)
    read_all()
    phase('all_write_valid_and_full_masks')
    for mask in range(64):
        for full in range(64):
            emit(write_valid=mask, full=full,
                queries=[(0, (mask+full+i) % 7) for i in range(4)])
    phase('same_tag_every_cycle_full_rate')
    for i in range(4096):
        emit(queries=[(17, (i+j) % 7) for j in range(4)],
             writes=[(17, rng.getrandbits(30)) for _ in range(6)],
             full=(i % 64))
    phase('six_distinct_tags_same_low_bank_bits')
    for i in range(4096):
        tags = [16*((i+j) % 16) + 5 for j in range(6)]
        emit(queries=[(tags[j], (i+j) % 7) for j in range(4)],
             writes=[(tag, rng.getrandbits(30)) for tag in tags])
    phase('random_aliases_masks_cutoffs_and_reset')
    for i in range(16384):
        tags = [rng.randrange(256) for _ in range(3)]
        emit(reset=int(i % 997 != 0), valid=rng.randrange(16),
             queries=[(rng.choice(tags), rng.randrange(8)) for _ in range(4)],
             writes=[(rng.choice(tags), rng.getrandbits(30)) for _ in range(6)],
             write_valid=rng.randrange(64), full=rng.randrange(64))
    phase('invalid_cutoff_does_not_discard_writes')
    emit(queries=[(23, 7)]*4, writes=[(23, rng.getrandbits(30)) for _ in range(6)], full=63)
    emit(queries=[(23, 0)]*4, write_valid=0)
    phase('inactive_queries_do_not_expose_data')
    emit(valid=0, queries=[(23, 7)]*4)
    phase('reset_under_contention_and_full_table_scan')
    emit(reset=0)
    read_all()
    text = '\n'.join(rows) + '\n'
    return text, dict(rows=len(rows), phases=phases, seed=0x50434945,
        vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
        scope='Known binary metadata primitive inputs; no PCIe protocol/native traffic, allocation bitmap or live payload context.',
        metadata=dict(tags=256, width=30, lookups=4, updates=6, latency=1))


if __name__ == '__main__':
    text, manifest = generate()
    base = Path(__file__).resolve().parents[1] / 'fpga/tb/data/pcie_tag_metadata'
    with base.with_suffix('.txt').open('x', newline='\n') as f:
        f.write(text)
    with base.with_suffix('.json').open('x') as f:
        json.dump(manifest, f, indent=2); f.write('\n')
    print(json.dumps(manifest))
