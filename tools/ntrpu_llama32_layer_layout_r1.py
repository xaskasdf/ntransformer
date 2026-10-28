"""One-layer allocation ABI and independent preflight stimulus; no model weights."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOCATIONS = (
    ('input', 8192, False), ('input_norm_weight', 8192, False), ('input_abi1', 4096, True),
    ('q_weight', 4456448, False), ('k_weight', 1114112, False), ('v_weight', 1114112, False),
    ('q_projection', 8192, True), ('k_projection', 2048, True), ('v_projection', 2048, True),
    ('rope_coefficients', 256, False), ('o_weight', 4456448, False), ('o_abi1', 4096, True),
    ('o_projection', 8192, True), ('post_norm_weight', 8192, False), ('post_residual', 8192, True),
    ('ffn_abi1', 4096, True), ('gate_weight', 17825792, False), ('up_weight', 17825792, False),
    ('down_weight', 17825792, False), ('gate_fp64', 65536, True), ('up_fp64', 65536, True),
    ('swiglu_abi1', 16384, True), ('down_projection', 8192, True), ('final_residual', 8192, True),
)
FIELDS = ('sequence', 'epoch', 'expected_epoch', 'source_valid', 'source_sequence', 'source_epoch',
          'lease_base', 'layer', 'position', 'coefficient_position', 'cache_valid', 'metadata_fault',
          'cache_base', 'cache_region', 'cache_bytes', 'cache_sequence', 'cache_epoch', 'capacity',
          'query_valid', 'prefix', 'status', 'injection', 'inject_cycle', 'reserved')
FAULTS = ('none', 'cancel', 'epoch', 'cache_valid', 'metadata_fault', 'query_valid', 'cache_base',
          'cache_region', 'cache_bytes', 'cache_sequence', 'cache_epoch', 'capacity', 'prefix')


def baseline():
    sequence, epoch = 0x1234567800000001, 0xFEDCBA9800000001
    words = [sequence, epoch, epoch, 1, sequence, epoch, 0xFFFFFFFFFFFFFFE0, 5, 0, 0,
             1, 0, (25 << 40) + 4032, 25 << 40, 262144 + 4032, sequence, epoch, 4, 1, 0, 0, 0, 0, 0]
    for i, (_, span, _) in enumerate(ALLOCATIONS):
        words += [((i + 1) << 40) + 4032, (i + 1) << 40, span + 4032]
    assert len(words) == 96
    return words


def plan():
    rows = []

    def add(name, status=0, *, fields=None, allocation=None, words=None, injection=0, at=0):
        values = list(words or baseline())
        for field, value in (fields or {}).items():
            values[FIELDS.index(field)] = value
        for i, triple in (allocation or {}).items():
            values[24 + i * 3:27 + i * 3] = triple
        values[20:23] = [status, injection, at]
        assert all(0 <= v < 1 << 64 for v in values)
        rows.append(dict(id=len(rows), name=name, expected_status=status,
                         injection=FAULTS[injection], inject_cycle=at, words=values))

    add('baseline')
    add('maximum_capacity_and_last_token', fields=dict(capacity=131072, position=131071,
        coefficient_position=131071, prefix=131071, cache_bytes=(131072 << 16) + 4032, layer=15))
    add('minimum_lease', fields=dict(lease_base=1))
    # Every unordered pair: 221 hazards rejected, 55 read/read aliases permitted.
    for i in range(24):
        for j in range(i + 1, 24):
            addr, base = ((i + 1) << 40) + 4032, (i + 1) << 40
            triple = [addr, base, max(ALLOCATIONS[i][1], ALLOCATIONS[j][1]) + 4032]
            hazard = ALLOCATIONS[i][2] or ALLOCATIONS[j][2]
            add(f'alias_{i}_{j}', int(hazard), allocation={i: triple, j: triple})
    for i, (_, span, _) in enumerate(ALLOCATIONS):
        addr, base, size = baseline()[24 + i * 3:27 + i * 3]
        variants = {
            'short': [addr, base, size - 64], 'empty': [addr, base, 0],
            'unaligned_address': [addr + 1, base, size + 64],
            'unaligned_base': [addr, base + 1, size],
            'unaligned_size': [addr, base, size + 1],
            'address_before_base': [base - 64, base, size],
            'address_wrap': [(1 << 64) - 64, (1 << 64) - 64, 64],
            'allocation_wrap': [addr, base, (1 << 64) - base + 64],
        }
        for name, triple in variants.items():
            add(f'range_{i}_{name}', 1, allocation={i: triple})
        add(f'cache_alias_{i}', 1, allocation={i: [(25 << 40) + 4032, 25 << 40, span + 4032]})
        # Different allocation bases may still overlap beyond their actual data spans.
        if ALLOCATIONS[i][2]:
            j = (i + 1) % 24
            broad = (1 << 40) + ALLOCATIONS[j][1] + 4032
            if j > i:
                add(f'allocation_tail_alias_{i}_{j}', 1, allocation={i: [addr, base, broad]})
    for field, value in [('sequence', 0), ('epoch', 0), ('source_valid', 0),
                         ('source_sequence', 1), ('source_epoch', 1), ('cache_sequence', 1),
                         ('cache_epoch', 1), ('expected_epoch', 1)]:
        add('identity_' + field, 2, fields={field: value})
    for field, value in [('cache_valid', 0), ('metadata_fault', 1), ('query_valid', 0),
                         ('capacity', 0), ('capacity', 131073), ('layer', 16), ('layer', 1 << 31),
                         ('position', 4), ('prefix', 1), ('coefficient_position', 1),
                         ('lease_base', 0), ('lease_base', 0xFFFFFFFFFFFFFFE1),
                         ('cache_base', (25 << 40) + 4033), ('cache_region', (25 << 40) + 1),
                         ('cache_bytes', 262144 + 4031), ('cache_bytes', 262144 + 3968),
                         ('cache_bytes', 0), ('cache_bytes', (1 << 64) - (25 << 40) + 64)]:
        add(f'context_{field}_{value}', 1, fields={field: value})
    for injection in range(1, len(FAULTS)):
        status = 7 if injection == 1 else 2 if injection == 2 else 11
        for at in (1, 10, 60):
            add(f'transient_{FAULTS[injection]}_{at}', status, injection=injection, at=at)
        add('late_' + FAULTS[injection], injection=injection)
    return rows


def write_plan():
    rows = plan()
    path = ROOT / 'fpga/tb/llama32_layer_preflight_plan_r1.mem'
    raw = ''.join(f'{word:016x}\n' for row in rows for word in row['words']).encode()
    if path.exists():
        assert path.read_bytes() == raw, 'Refuse to change a published plan'
    else:
        path.write_bytes(raw)
    return dict(cases=len(rows), allocation_pairs=276, rejected_write_aliases=221,
                permitted_read_aliases=55, sha256=hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    print(json.dumps(write_plan()))
