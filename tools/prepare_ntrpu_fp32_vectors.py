#!/usr/bin/env python3
"""Pin an IEEE conversion fixture using Python's native binary32 packing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct


def bits(value):
    return int.from_bytes(struct.pack('<d', value), 'little')


def reference(word):
    value = struct.unpack('<d', word.to_bytes(8, 'little'))[0]
    if not math.isfinite(value):
        return 2, 0
    maximum = struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0]
    if abs(value) > maximum:
        return 3, 0
    return 0, int.from_bytes(struct.pack('<f', value), 'little')


def prepare(output, brandon):
    output.mkdir(parents=True, exist_ok=False)
    words = []
    # Every binary64 exponent, both signs, and carry/tie-sensitive fractions.
    fractions = [0, 1, (1 << 28)-1, 1 << 28, (1 << 28)+1,
                 (1 << 29)-1, 1 << 29, 1 << 51, (1 << 52)-1]
    for exp in range(2048):
        for frac in fractions:
            for sign in range(2):
                words.append((sign << 63) | (exp << 52) | frac)
    # Native FP32 adjacent-value midpoint and both adjacent FP64 numbers.
    for exp in range(255):
        for fraction in [0, 1, 0x3fffff, 0x7ffffe]:
            small = (exp << 23) | fraction
            lo = struct.unpack('<f', small.to_bytes(4, 'little'))[0]
            hi = struct.unpack('<f', (small+1).to_bytes(4, 'little'))[0]
            midpoint = (lo+hi)/2
            for value in [lo, math.nextafter(midpoint, -math.inf), midpoint,
                          math.nextafter(midpoint, math.inf), hi]:
                words.extend([bits(value), bits(-value)])
    rng = random.Random(20261003)
    words.extend(rng.getrandbits(64) for _ in range(4096))
    # Do not derive the converter's expected values from an HLS helper.
    # Check the existing native/independent FFN references and then add down.
    from check_ntrpu_ffn_memory_fixture import load_fixture
    fixture = load_fixture(brandon)
    real_start = len(words)
    for job in fixture['jobs']:
        raw = bytes.fromhex(job['reference_buffers'][7])
        words.extend(struct.unpack('<256Q', raw))
    data = ''.join(f'{w:016x} {reference(w)[0]} {reference(w)[1]:08x}\n' for w in words)
    (output/'vectors.txt').write_bytes(data.encode('ascii'))
    manifest = dict(format='NTRFPCAST01', count=len(words),
                    vector_sha256=hashlib.sha256(data.encode('ascii')).hexdigest(),
                    brandon_manifest_sha256=hashlib.sha256((brandon/'manifest.json').read_bytes()).hexdigest(),
                    brandon_start=real_start, brandon_count=768,
                    statuses={str(s):sum(reference(w)[0]==s for w in words) for s in [0,2,3]},
                    seed=20261003, oracle='Python struct.pack binary32; explicit finite/FLT_MAX contract')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--brandon', required=True, type=Path)
    args=parser.parse_args()
    prepare(args.output, args.brandon)
