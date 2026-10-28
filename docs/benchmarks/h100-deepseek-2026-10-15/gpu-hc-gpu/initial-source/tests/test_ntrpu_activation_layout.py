"""CPU ABI golden vectors and invalid numeric inputs, independent of RTL."""
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ntrpu_activation_layout import pack_activations


class ActivationLayoutTests(unittest.TestCase):
    def test_byte_order_and_padding(self):
        values=bytes(range(64));scales=struct.pack('<dd',0.5,2.0)
        expected=values[:32]+bytes.fromhex('000000000000e03f')+bytes(24)
        expected+=values[32:]+bytes.fromhex('0000000000000040')+bytes(24)
        self.assertEqual(pack_activations(values,scales),expected)

    def test_boundary_bits_and_maximum(self):
        for bits in [0,1,0x8000000000000000,0x4780204070e1c387]:
            record=pack_activations(bytes(32),bits.to_bytes(8,'little'))
            self.assertEqual(record[32:40],bits.to_bytes(8,'little'))
        self.assertEqual(len(pack_activations(bytes(8192),bytes(2048))),16384)

    def test_reject_invalid(self):
        for bits in [0xbff0000000000000,0x8000000000000001,0x4780204070e1c388,
                     0x7ff0000000000000,0x7ff8000000000000,0xfff0000000000000]:
            with self.subTest(bits=bits),self.assertRaises(ValueError):
                pack_activations(bytes(32),bits.to_bytes(8,'little'))
        for values,scales in [(b'',b''),(bytes(31),bytes(8)),(bytes(32),bytes(7)),
                              (bytes(8224),bytes(2056)),(bytearray(32),bytes(8))]:
            with self.subTest(size=len(values)),self.assertRaises(ValueError):
                pack_activations(values,scales)


if __name__=='__main__':unittest.main()
