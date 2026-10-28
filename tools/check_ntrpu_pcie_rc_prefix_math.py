"""Binary arithmetic study; this does not validate RTL or event selection."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


def step(remaining, address, words):
    # RTL stores signed available in 15 bits, then compares against unsigned R.
    available_bits = (4 * words - (address & 3)) & 32767
    delivered = np.where(available_bits < remaining, available_bits, remaining)
    return remaining - delivered, (address + delivered) & 4095


def prefix(remaining, address, total_words, first_words):
    delta = np.minimum(remaining, 4 * total_words - (address & 3))
    delta = np.where((first_words == 0) & ((address & 3) != 0), remaining, delta)
    return remaining - delta, (address + delta) & 4095


def same(actual, expected):
    assert all(np.array_equal(a, e) for a, e in zip(actual, expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.monotonic()
    single = pairs = random_prefixes = 0
    words = np.arange(2048, dtype=np.int64)[None, :]
    for start in range(0, 8192, 64):
        remaining = np.arange(start, start + 64, dtype=np.int64)[:, None]
        for offset in range(4):
            address = np.full((64, 2048), offset, dtype=np.int64)
            r = np.broadcast_to(remaining, address.shape)
            same(step(r, address, words), prefix(r, address, words, words))
            single += address.size
    # All R/low-address states, boundary word-count pairs and wrapping addresses.
    values = [0, 1, 2, 3, 1023, 1024, 2046, 2047]
    r = np.repeat(np.arange(8192, dtype=np.int64), 4)
    a = np.tile(np.arange(4092, 4096, dtype=np.int64), 8192)
    for first in values:
        for second in values:
            same(step(*step(r, a, first), second), prefix(r, a, first + second, first))
            pairs += r.size
    rng = np.random.Generator(np.random.PCG64(0x4E545250))
    r0 = rng.integers(0, 8192, size=250000, dtype=np.int64)
    a0 = rng.integers(0, 4096, size=250000, dtype=np.int64)
    chain = rng.integers(0, 2048, size=(4, 250000), dtype=np.int64)
    chain[0, ::3] = 0
    chain[1, ::5] = 0
    r, a = r0, a0
    total = np.zeros_like(r0)
    for w in chain:
        r, a = step(r, a, w)
        total += w
        same((r, a), prefix(r0, a0, total, chain[0]))
        random_prefixes += r0.size
    # Concrete counterexamples reject two tempting, but invalid simplifications.
    controls = []
    for name, initial, counts, bad_delta in [
        ('omit_zero_count_misaligned_rule', (10, 1), [0, 1], 3),
        ('truncate_prefix_count_to_11_bits', (8191, 0), [1024, 1024], 0),
    ]:
        r, a = initial
        for count in counts:
            r, a = step(r, a, count)
        bad = (initial[0] - bad_delta, (initial[1] + bad_delta) & 4095)
        assert (r, a) != bad
        controls.append(dict(name=name, initial=initial, counts=counts,
                             expected=[int(r), int(a)], incorrect=bad, rejected=True))
    assert (single, pairs, random_prefixes) == (67108864, 2097152, 1000000)
    result = dict(status='PASS_BINARY_RC_PREFIX_ARITHMETIC_MODEL',
                  single_step_cases=single, boundary_pair_cases=pairs,
                  sampled_prefixes=random_prefixes, random_seed=0x4E545250,
                  controls=controls, duration_seconds=round(time.monotonic() - started, 3),
                  python=platform.python_version(), numpy=np.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='All binary R13/offset2/W11 single steps; boundary pairs over all R13 and offsets at address wrap; 250000 sampled chains at each length1..4. No exhaustive chain proof, X/Z, partial retained contexts, event decoding, RTL, netlist, native or physical qualification.')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
