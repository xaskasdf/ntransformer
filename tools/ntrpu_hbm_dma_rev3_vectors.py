"""Admission vectors with unbounded-integer interval arithmetic as the oracle."""
from collections import Counter
import random

LIMIT = 1 << 64
MASK = LIMIT - 1


def lease():
    return dict(source_base=0, source_size=1 << 34, published_offset=0, published_bytes=1 << 34,
                target_base=1 << 36, target_size=1 << 33, window_offset=0, window_bytes=1 << 33,
                flags=15)


def request():
    return dict(source_offset=0, target_offset=0, bytes=64, sequence=1, bad_handle=False)


def expected(c, r):
    if r['bad_handle']:
        return 1
    if c['flags'] != 15:
        return 2
    if r['sequence'] == 0:
        return 6
    s, t, n = r['source_offset'], r['target_offset'], r['bytes']
    source_first, target_first = c['source_base'] + s, c['target_base'] + t
    if (n == 0 or not 0 <= s <= c['source_size'] - n or not 0 <= t <= c['target_size'] - n or
            not c['window_offset'] <= t <= c['window_offset'] + c['window_bytes'] - n or
            source_first + n > LIMIT or target_first + n > LIMIT):
        return 3
    if not c['published_offset'] <= s <= c['published_offset'] + c['published_bytes'] - n:
        return 4
    if (n > 4096 or n % 64 or source_first % 64 or target_first % 64 or
            source_first >= 1 << 34 or source_first + n > 1 << 34 or
            source_first // 4096 != (source_first + n - 1) // 4096):
        return 5
    return 0


def packed(fields):
    value = 0
    for f in fields:
        assert 0 <= f < LIMIT
        value = (value << 64) | f
    return value


def generate(path, seed=3109):
    profiles = [lease()]
    for change in [dict(source_base=MASK - 63, source_size=MASK, published_bytes=MASK),
                   dict(source_base=1, source_size=MASK, published_bytes=MASK),
                   dict(source_base=(1 << 34) - 64), dict(source_base=MASK),
                   dict(target_base=LIMIT - 4096, target_size=4096, window_bytes=4096),
                   dict(target_base=0, target_size=MASK - 255, window_bytes=MASK),
                   dict(window_offset=64, window_bytes=128),
                   dict(window_offset=MASK - 63, window_bytes=128),
                   dict(target_base=0, target_size=MASK - 255, window_offset=LIMIT - 512, window_bytes=1024),
                   dict(window_offset=0, window_bytes=0),
                   dict(published_offset=64, published_bytes=128),
                   dict(published_offset=MASK - 63, published_bytes=MASK),
                   dict(published_offset=0, published_bytes=0),
                   dict(source_size=1, target_size=1),
                   dict(source_size=MASK, published_bytes=MASK),
                   dict(source_size=MASK, published_offset=MASK - 63, published_bytes=128)]:
        c = lease(); c.update(change); profiles.append(c)
    samples = []
    lengths = [0, 1, 63, 64, 65, 128, 4096, 4097, MASK]
    for c in profiles:
        offsets = {0, 1, 63, 64, 4032, 4095, 4096, (1 << 34) - 64, 1 << 34, MASK - 63, MASK}
        for name in ('source_size', 'target_size', 'window_offset', 'published_offset'):
            offsets.update((max(0, c[name] - 64), c[name], min(MASK, c[name] + 64)))
        for offset in sorted(offsets):
            for length in lengths:
                for field in ('source_offset', 'target_offset'):
                    r = request(); r.update({field: offset, 'bytes': length}); samples.append((dict(c), r))
    # Priority combinations: handle > permission > replay > range > unpublished > span.
    for bad_handle in (False, True):
        for flags in (15, 14, 13, 11, 7, 0):
            for seq in (0, 1):
                for change in ({}, dict(bytes=0), dict(bytes=63), dict(source_offset=MASK)):
                    c = lease(); c['flags'] = flags
                    r = request(); r.update(change, bad_handle=bad_handle, sequence=seq)
                    samples.append((c, r))
    rng = random.Random(seed)
    for i in range(6000):
        c = dict(rng.choice(profiles))
        r = request()
        r['bytes'] = rng.choice(lengths) if i % 3 else rng.getrandbits(64)
        for key in ('source_offset', 'target_offset'):
            r[key] = rng.getrandbits(64) if i % 4 == 0 else rng.randrange(0, 16384)
            if i % 4 == 2:
                r[key] &= ~63
        if i % 4 == 2:
            r['bytes'] = 64 * rng.randrange(1, 65)
        samples.append((c, r))
    counts = Counter()
    with path.open('x', encoding='ascii') as f:
        for generation, (c, r) in enumerate(samples, 2):
            end = c['target_base'] + c['target_size']
            wb = 0 if c['target_base'] >= 256 else LIMIT - 256
            assert end <= LIMIT and (wb + 256 <= c['target_base'] or end <= wb)
            cfields = [1, 0, 0, 1, 1, 1, 0, 1, c['source_base'], c['source_size'],
                       c['published_offset'], c['published_bytes'], c['target_base'], c['target_size'],
                       generation, c['window_offset'], c['window_bytes'], 7]
            rfields = [1, 0, r['sequence'], 1, 0, 0, 2 if r['bad_handle'] else 1,
                       1, 1, 0, 1, generation, 7, r['source_offset'], r['target_offset'], r['bytes']]
            code = expected(c, r); counts[code] += 1
            f.write(f'{(packed(cfields) << 4) | c["flags"]:0289x} {packed(rfields):0256x} {wb:016x} {code}\n')
    assert set(counts) == {0, 1, 2, 3, 4, 5, 6}
    return dict(vectors=len(samples), expected_codes=dict(counts), seed=seed,
                oracle='Unbounded integer closed containment and absolute address/page checks')
