"""Compact full-matrix Q8 layout for the RTL matrix descriptor sequencer."""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
from ntrpu_gemv_layout import integer


@dataclass(frozen=True)
class MatrixTile:
    first_row: int
    rows: int
    k_base: int
    k_count: int
    address: int
    offset: int
    byte_count: int


def matrix_tiles(rows, k_blocks, tile_k, *, region_base=0x100000000,
                 region_bytes=0x10000000, first_offset=0xfc0):
    """Plan contiguous [row group][K tile][K block][row][34 bytes] payloads."""
    integer('rows', rows, 1, 8192)
    integer('k_blocks', k_blocks, 1, 256)
    integer('tile_k', tile_k, 1, 32)
    integer('region_base', region_base, 0, (1 << 64)-1)
    integer('region_bytes', region_bytes, 1, (1 << 64)-1)
    integer('first_offset', first_offset, 0, (1 << 64)-1)
    if region_base+region_bytes >= 1 << 64:
        raise ValueError('Registered range overflows the RTL contract')
    if (region_base+first_offset) % 64:
        raise ValueError('First weight address must be 64-byte aligned')
    if first_offset+rows*k_blocks*34 > region_bytes:
        raise ValueError('Full matrix exceeds the registered range')
    tiles=[]
    offset=first_offset
    for first_row in range(0, rows, 32):
        count_rows=min(32, rows-first_row)
        for k_base in range(0, k_blocks, tile_k):
            count_k=min(tile_k, k_blocks-k_base)
            if count_rows*count_k % 32:
                raise ValueError('Tail requires explicit row/K padding')
            size=count_rows*count_k*34
            tiles.append(MatrixTile(first_row,count_rows,k_base,count_k,
                                    region_base+offset,offset,size))
            offset+=size
    return tuple(tiles)


def pack_matrix(q8, rows, k_blocks, tile_k, *, region_base=0x100000000,
                region_bytes=0x10000000, first_offset=0xfc0,
                max_image_bytes=128*1024*1024):
    """Pack immutable dense row-major Q8 blocks; perform no quantization/padding."""
    tiles=matrix_tiles(rows,k_blocks,tile_k,region_base=region_base,
                       region_bytes=region_bytes,first_offset=first_offset)
    integer('max_image_bytes',max_image_bytes,1,(1 << 64)-1)
    if not isinstance(q8,bytes) or len(q8)!=rows*k_blocks*34:
        raise ValueError('Expected exactly rows*k_blocks*34 immutable Q8 bytes')
    end=tiles[-1].offset+tiles[-1].byte_count
    if end>max_image_bytes:
        raise ValueError('Image exceeds allocation budget')
    for offset in range(0,len(q8),34):
        if int.from_bytes(q8[offset:offset+2],'little') & 0x7c00 == 0x7c00:
            raise ValueError('Nonfinite FP16 weight scale')
    image=bytearray([0xa5])*end
    for tile in tiles:
        cursor=tile.offset
        for k in range(tile.k_base,tile.k_base+tile.k_count):
            for row in range(tile.first_row,tile.first_row+tile.rows):
                source=(row*k_blocks+k)*34
                image[cursor:cursor+34]=q8[source:source+34]
                cursor+=34
    return bytes(image),tiles


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    for name in ['rows','k-blocks','tile-k']:
        parser.add_argument('--'+name,type=int,required=True)
    parser.add_argument('--region-base',type=lambda s:int(s,0),default=0x100000000)
    parser.add_argument('--region-bytes',type=lambda s:int(s,0),default=0x10000000)
    args=parser.parse_args()
    source=args.input.read_bytes()
    image,tiles=pack_matrix(source,args.rows,args.k_blocks,args.tile_k,
                            region_base=args.region_base,region_bytes=args.region_bytes)
    manifest=dict(format='NTRMAT01',layout=2,rows=args.rows,k_blocks=args.k_blocks,tile_k=args.tile_k,
                  region_base=args.region_base,region_bytes=args.region_bytes,weight_address=tiles[0].address,
                  source_sha256=hashlib.sha256(source).hexdigest(),image_sha256=hashlib.sha256(image).hexdigest(),
                  image_bytes=len(image),tiles=[asdict(t) for t in tiles])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'weights-image.bin').write_bytes(image)
    (args.output/'weights-image.hex').write_text(''.join(f'{b:02x}\n' for b in image),encoding='ascii')
    (args.output/'matrix.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='tiles'}))


if __name__=='__main__':
    main()
