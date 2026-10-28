import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ntrpu_matrix_layout import matrix_tiles,pack_matrix


class MatrixLayoutTests(unittest.TestCase):
    def test_group_tile_order_and_tail(self):
        blocks=[b'\x00\x3c'+bytes([r,k])*16 for r in range(33) for k in range(64)]
        dense=b''.join(blocks)
        image,tiles=pack_matrix(dense,33,64,32)
        self.assertEqual([(t.first_row,t.rows,t.k_base,t.byte_count) for t in tiles],
                         [(0,32,0,34816),(0,32,32,34816),(32,1,0,1088),(32,1,32,1088)])
        self.assertEqual(image[:4032],b'\xa5'*4032)
        for t in tiles:
            self.assertEqual(t.address,0x100000000+t.offset)
            for kb in range(t.k_count):
                for row in range(t.rows):
                    off=t.offset+(kb*t.rows+row)*34
                    self.assertEqual(image[off:off+34],b'\x00\x3c'+bytes([t.first_row+row,t.k_base+kb])*16)
        self.assertEqual(len(image),4032+33*64*34)

    def test_maximum_descriptor_geometry(self):
        tiles=matrix_tiles(8192,256,1)
        self.assertEqual(len(tiles),65536)
        self.assertEqual((tiles[-1].first_row,tiles[-1].rows,tiles[-1].k_base,tiles[-1].k_count),(8160,32,255,1))
        self.assertEqual(tiles[-1].offset+tiles[-1].byte_count,4032+8192*256*34)
        self.assertTrue(all(a.offset+a.byte_count==b.offset for a,b in zip(tiles,tiles[1:])))

    def test_reject_invalid_descriptors_and_data(self):
        for shape in [(0,32,32),(8193,32,32),(32,0,32),(32,257,32),(32,32,0),(32,32,33),(33,33,32)]:
            with self.subTest(shape=shape),self.assertRaises(ValueError):matrix_tiles(*shape)
        for kwargs in [dict(first_offset=1),dict(region_bytes=4032+32*32*34-1),dict(region_base=(1<<64)-64)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):matrix_tiles(32,32,32,**kwargs)
        valid=(b'\x00\x3c'+bytes(32))*32
        for data in [valid[:-1],bytearray(valid),b'\x00\x7c'+valid[2:]]:
            with self.assertRaises(ValueError):pack_matrix(data,1,32,32)
        with self.assertRaises(ValueError):pack_matrix(valid,1,32,32,max_image_bytes=64)


if __name__=='__main__':
    unittest.main()
