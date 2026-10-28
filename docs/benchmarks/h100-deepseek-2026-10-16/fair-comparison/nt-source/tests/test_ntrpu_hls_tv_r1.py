"""Endian, framing and fail-closed tests for the HLS binary TV reader."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from read_ntrpu_hls_tv_r1 import BMARK,transactions,text_transactions

class HlsTvTest(unittest.TestCase):
    def test_32_and_64_bit_frames(self):
        for width,values in [(32,[0x12345678,0xfedcba98]),(64,[0x0123456789abcdef,0xfedcba9876543210])]:
            raw=b''.join((2).to_bytes(8,'big')+b''.join(v.to_bytes(width//8,'big') for v in row)
                for row in [values,list(reversed(values))])+BMARK
            expected=[[f'0x{v:0{width//4}x}' for v in row] for row in [values,list(reversed(values))]]
            self.assertEqual(transactions(raw,2,width=width,depth=2),expected)
            self.assertEqual(transactions(text_transactions(expected).encode(),2,width=width,depth=2),expected)
            second_header=8+2*(width//8)
            for bad in [raw[:-1],raw+b'\0',raw[:-8]+bytes(8),raw[1:],raw[:16]+raw[17:],
                        (3).to_bytes(8,'big')+raw[8:],raw[:second_header]+bytes(8)+raw[second_header+8:]]:
                with self.subTest(width=width,bad=bad.hex()):
                    with self.assertRaises(AssertionError):transactions(bad,2,width=width,depth=2)
            with self.assertRaises(AssertionError):transactions(raw,3,width=width,depth=2)
            with self.assertRaises(AssertionError):transactions(raw,2,width=width)

    def test_text_unknown_overflow_order(self):
        good=text_transactions([['0x12345678'],['0xabcdef01']])
        for bad in [good.replace('0x12345678','0xxxxxxxxx'),good.replace('0x12345678','0x100000000'),
                    good.replace('[[transaction]] 1','[[transaction]] 0'),good.replace('[[/transaction]]','',1)]:
            with self.assertRaises((AssertionError,ValueError)):transactions(bad.encode(),2,width=32,depth=1)

if __name__=='__main__':unittest.main()
