"""Pack dense row-major Q8 blocks into immutable GEMV weight tiles.

Layout1: [row][K block][34 bytes]. Layout2: [K block][row][34 bytes].
The input always contains little-endian FP16 scale followed by 32 INT8 bytes.
No quantization, row padding, DMA or device buffer registration is performed.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Tile:
    k_base: int
    k_count: int
    address: int
    offset: int
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class PackedWeights:
    image: bytes
    tiles: tuple[Tile, ...]


def integer(name, value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')


def pack_weights(q8, rows, k_blocks, tile_k, *, layout=2,
                 region_base=0x100000000, region_bytes=0x200000,
                 first_offset=0xfc0, tile_stride=0x10000,
                 max_image_bytes=64*1024*1024):
    """Return a fresh image and ordered descriptors; reject unsupported geometry.

    All gaps are initialized to 0xa5, so a valid zero is distinguishable from
    an unwritten gap. The image begins at region_base; its descriptors address
    disjoint, 64-byte-aligned payloads inside the supplied registered range.
    """
    integer('rows', rows, 1, 32)
    integer('k_blocks', k_blocks, 1, 256)
    integer('tile_k', tile_k, 1, 32)
    integer('layout', layout, 1, 2)
    integer('region_base', region_base, 0, (1 << 64)-1)
    integer('region_bytes', region_bytes, 1, (1 << 64)-1)
    integer('first_offset', first_offset, 0, (1 << 64)-1)
    integer('tile_stride', tile_stride, 64, (1 << 64)-1)
    integer('max_image_bytes', max_image_bytes, 1, (1 << 64)-1)
    if not isinstance(q8, bytes) or len(q8) != rows*k_blocks*34:
        raise ValueError('Expected exactly rows*k_blocks*34 immutable Q8 bytes')
    if region_base+region_bytes >= 1 << 64:
        raise ValueError('Registered range overflows the RTL 64-bit contract')
    if (region_base+first_offset) % 64 or tile_stride % 64:
        raise ValueError('Tile addresses must be 64-byte aligned')
    for offset in range(0, len(q8), 34):
        scale = int.from_bytes(q8[offset:offset+2], 'little')
        if scale & 0x7c00 == 0x7c00:
            raise ValueError(f'Nonfinite FP16 weight scale in source block {offset//34}')
    payloads = []
    end = 0
    for index, base in enumerate(range(0, k_blocks, tile_k)):
        count = min(tile_k, k_blocks-base)
        if rows*count % 32:
            raise ValueError('Tile requires explicit row/K padding before packing')
        offset = first_offset+index*tile_stride
        size = rows*count*34
        if offset < end:
            raise ValueError('Tile payloads overlap')
        end = offset+size
        if end > region_bytes or end > max_image_bytes:
            raise ValueError('Tile exceeds registered range or image-size budget')
        order = ((r, k) for r in range(rows) for k in range(base, base+count)) if layout == 1 else (
            (r, k) for k in range(base, base+count) for r in range(rows))
        payload = b''.join(q8[(r*k_blocks+k)*34:(r*k_blocks+k+1)*34] for r, k in order)
        payloads.append((Tile(base, count, region_base+offset, offset, size,
                              hashlib.sha256(payload).hexdigest()), payload))
    image = bytearray([0xa5])*end
    for tile, payload in payloads:
        image[tile.offset:tile.offset+tile.byte_count] = payload
    return PackedWeights(bytes(image), tuple(t for t, _ in payloads))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Dense row-major Q8 byte file')
    parser.add_argument('--output', type=Path, required=True)
    for name in ['rows', 'k-blocks', 'tile-k']:
        parser.add_argument('--'+name, type=int, required=True)
    parser.add_argument('--layout', type=int, choices=[1, 2], default=2)
    parser.add_argument('--region-base', type=lambda s: int(s, 0), default=0x100000000)
    parser.add_argument('--region-bytes', type=lambda s: int(s, 0), default=0x200000)
    args = parser.parse_args()
    source = args.input.read_bytes()
    packed = pack_weights(source, args.rows, args.k_blocks, args.tile_k,
                          layout=args.layout, region_base=args.region_base, region_bytes=args.region_bytes)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'weights-image.bin').write_bytes(packed.image)
    (args.output/'weights-image.hex').write_text(''.join(f'{b:02x}\n' for b in packed.image), encoding='ascii')
    manifest = dict(format='NTRTILE02', layout=args.layout, rows=args.rows,
                    k_blocks=args.k_blocks, tile_k=args.tile_k, region_base=args.region_base,
                    region_bytes=args.region_bytes, source_sha256=hashlib.sha256(source).hexdigest(),
                    image_sha256=hashlib.sha256(packed.image).hexdigest(), image_bytes=len(packed.image),
                    tiles=[asdict(t) for t in packed.tiles])
    (args.output/'tiles.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
