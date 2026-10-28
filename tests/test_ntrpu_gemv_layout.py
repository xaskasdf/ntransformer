"""Host packing ABI and input-range rejection tests; no vendor tools needed."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from ntrpu_gemv_layout import pack_weights


def block(row, k):
    return b'\x00\x3c'+bytes([row, k])+bytes((row*7+k*3+i) % 256 for i in range(30))


def matrix(rows, k):
    return b''.join(block(r, c) for r in range(rows) for c in range(k))


class LayoutTests(unittest.TestCase):
    def test_literal_row_and_block_order(self):
        source = matrix(32, 2)
        for layout in [1, 2]:
            p = pack_weights(source, 32, 2, 2, layout=layout)
            self.assertEqual(p.image[:4032], b'\xa5'*4032)
            t = p.tiles[0]
            payload = p.image[t.offset:t.offset+t.byte_count]
            self.assertEqual(payload[:34], block(0, 0))
            self.assertEqual(payload[34:68], block(0, 1) if layout == 1 else block(1, 0))
            self.assertEqual(payload[1088:1122], block(16, 0) if layout == 1 else block(0, 1))
            self.assertEqual(hashlib.sha256(payload).hexdigest(), t.sha256)

    def test_tail_maximum_and_odd_rows(self):
        for rows, k, chunk in [(32, 256, 32), (31, 64, 32), (32, 23, 8), (16, 8, 4)]:
            source = matrix(rows, k)
            p = pack_weights(source, rows, k, chunk)
            reconstructed = bytearray(len(source))
            for t in p.tiles:
                self.assertEqual(t.address, 0x100000000+t.offset)
                self.assertEqual(t.byte_count % 64, 0)
                for c in range(t.k_count):
                    for r in range(rows):
                        pos = t.offset+(c*rows+r)*34
                        dest = (r*k+t.k_base+c)*34
                        reconstructed[dest:dest+34] = p.image[pos:pos+34]
            self.assertEqual(bytes(reconstructed), source)
            self.assertEqual(p.tiles[-1].k_base+p.tiles[-1].k_count, k)

    def test_reject_geometry_addresses_and_payloads(self):
        source = matrix(32, 64)
        bad = [dict(rows=0), dict(rows=33), dict(k_blocks=257), dict(tile_k=0),
               dict(layout=0), dict(layout=3), dict(layout=True), dict(region_base=-1),
               dict(region_base=(1 << 64)-64), dict(region_bytes=64), dict(first_offset=1),
               dict(tile_stride=65), dict(tile_stride=64), dict(max_image_bytes=1024)]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                args = dict(rows=32, k_blocks=64, tile_k=32)
                args.update(change)
                pack_weights(source, **args)
        with self.assertRaises(ValueError):
            pack_weights(matrix(31, 33), 31, 33, 32)
        for payload in [source[:-1], source+b'\0', bytearray(source), b'\0\x7c'+source[2:]]:
            with self.assertRaises(ValueError):
                pack_weights(payload, 32, 64, 32)


if __name__ == '__main__':
    unittest.main()
