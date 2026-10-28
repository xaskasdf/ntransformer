"""Materialize host-packed immutable AXI images for Brandon and maximum shapes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import struct

from check_ntrpu_gemv_fixture import load_fixture
from ntrpu_gemv_layout import pack_weights

BRANDON_MANIFEST = '7758590dec8fac758b7f2920703d06ccf73b3a5d571b7c3af0626ffe1624cf97'
SHAPES = [(32, 64), (32, 256), (1, 256), (31, 64)]
HALF_BITS = [0x3c00, 0xb800, 0x0001, 0x03ff, 0x4200, 0xbc01]


def synthetic(rows, k_blocks, pattern):
    """Independent scalar binary64 oracle and dense Q8 source for shape tests."""
    qx = [bytes((((k*13+lane*7+pattern*3) & 255)-128) & 255 for lane in range(32))
          for k in range(k_blocks)]
    sx = [1.0/(k % 5+1) for k in range(k_blocks)]
    source = bytearray()
    reference = []
    for r in range(rows):
        acc = 0.0
        for k in range(k_blocks):
            scale = HALF_BITS[(r+k) % 6].to_bytes(2, 'little')
            w = [((r*29+k*11+lane*17+pattern) & 255)-128 for lane in range(32)]
            source.extend(scale+bytes(v & 255 for v in w))
            a = [v if v < 128 else v-256 for v in qx[k]]
            dot = sum(x*y for x, y in zip(w, a))
            acc += (struct.unpack('<e', scale)[0]*sx[k])*float(dot)
        reference.append(struct.pack('>d', acc).hex())
    return bytes(source), qx, sx, reference


def prepare(brandon, output):
    if hashlib.sha256((brandon/'manifest.json').read_bytes()).hexdigest() != BRANDON_MANIFEST:
        raise ValueError('Expected the reviewed immutable Brandon fixture')
    original = load_fixture(brandon)
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for job in original['jobs']:
        j = dict(job, category='real_brandon_projection')
        stem = f"job-{j['id']:03d}"
        for suffix in ['weights', 'activations', 'scales', 'reference']:
            shutil.copyfile(brandon/(stem+'-'+suffix+'.hex'), output/(stem+'-'+suffix+'.hex'))
        jobs.append(j)
    for pattern, (rows, k) in enumerate(SHAPES, 1):
        ident = len(jobs)+1
        stem = f'job-{ident:03d}'
        payload, qx, sx, reference = synthetic(rows, k, pattern)
        files = {'weights': ''.join(f'{v:02x}\n' for v in payload),
                 'activations': ''.join(v[::-1].hex()+'\n' for v in qx),
                 'scales': ''.join(struct.pack('>d', v).hex()+'\n' for v in sx),
                 'reference': '\n'.join(reference)+'\n'}
        for suffix, text in files.items():
            (output/(stem+'-'+suffix+'.hex')).write_text(text, encoding='ascii')
        jobs.append(dict(id=ident, tensor=f'synthetic/{rows}x{k*32}', first_row=0,
                         rows=rows, k_blocks=k, tile_k=32, reference_fp64_bits=reference,
                         category='synthetic_dimensions', synthetic_pattern_id=pattern))
    for j in jobs:
        stem = f"job-{j['id']:03d}"
        source = bytes.fromhex((output/(stem+'-weights.hex')).read_text())
        packed = pack_weights(source, j['rows'], j['k_blocks'], j['tile_k'], layout=2)
        if len(packed.image) > 524288:
            raise ValueError('Image exceeds simulation memory capacity')
        (output/(stem+'-image.hex')).write_text(''.join(f'{v:02x}\n' for v in packed.image), encoding='ascii')
        (output/(stem+'-tiles.hex')).write_text(''.join(
            f'{t.address:016x}{t.k_base:08x}{t.k_count:08x}\n' for t in packed.tiles), encoding='ascii')
        (output/(stem+'-image.txt')).write_text(f'{len(packed.image)} {len(packed.tiles)}\n', encoding='ascii')
        j.update(layout=2, image_bytes=len(packed.image), image_sha256=hashlib.sha256(packed.image).hexdigest(),
                 dense_source_sha256=hashlib.sha256(source).hexdigest(), tiles=[asdict(t) for t in packed.tiles])
    (output/'jobs.txt').write_text('NTRGEMV02 '+str(len(jobs))+'\n'+''.join(
        f"{j['id']} {j['rows']} {j['k_blocks']} {j['tile_k']}\n" for j in jobs), encoding='ascii')
    manifest = dict(original, format='NTRGEMV02', layout=2, jobs=jobs,
                    base_fixture_manifest_sha256=BRANDON_MANIFEST, region_base=0x100000000,
                    region_bytes=0x200000,
                    scope='Host-packed immutable AXI images:54 real Brandon projections and4 synthetic maximum/odd shapes; no Llama weights, FFN or physical DDR',
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(jobs=len(jobs), rows=sum(j['rows'] for j in jobs),
                         blocks=sum(j['rows']*j['k_blocks'] for j in jobs),
                         manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brandon-fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.brandon_fixture, args.output)
