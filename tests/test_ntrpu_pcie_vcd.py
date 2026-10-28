"""Four-state native trace data must stay unknown wherever it is meaningful."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ntrpu_pcie_straddle import LogicWord,StraddleDecoder,decode_completion,decode_memory_request
from ntrpu_pcie_vcd import logic_vcd_rows
from test_ntrpu_pcie_straddle import place,completion,request


class FourStateTests(unittest.TestCase):
    def test_vcd_extension_and_aliases(self):
        vcd='''$timescale 1ps $end
$scope module top $end
$var wire 8 ! data [7:0] $end
$var wire 8 ! alias [7:0] $end
$upscope $end
$enddefinitions $end
#0
bx !
#2
b1 !
#4
b0xz1 !
'''
        rows=list(logic_vcd_rows(vcd,{'/top/data'}))
        self.assertEqual(rows[-1][1]['/top/data'],LogicWord(1,255,8))
        self.assertEqual(rows[-1][2]['/top/data'],LogicWord(1,249,8))
        self.assertIsNone(rows[-1][2]['/top/alias'])
        self.assertEqual(rows[1][2]['/top/data'].known,0)

    def test_unused_rc_lanes_may_be_unknown(self):
        data=place((0,completion(1,1,4)+b'abcd'))
        packet=StraddleDecoder('rc').accept(10,LogicWord(data,(1<<128)-1,512),
                    15<<12 | 1<<64 | 1<<76 | 3<<80)[0]
        self.assertEqual(decode_completion(packet)['enabled_payload'],b'abcd')

    def test_unknown_descriptor_or_enabled_payload_rejected(self):
        data=place((0,completion(1,1,4)+b'abcd'))
        for bit in (0,12,16,30,48,64,96,127):
            with self.subTest(bit=bit),self.assertRaisesRegex(ValueError,'unknown'):
                packet=StraddleDecoder('rc').accept(10,LogicWord(data,((1<<512)-1)^(1<<bit),512),
                            15<<12 | 1<<64 | 1<<76 | 3<<80)[0]
                decode_completion(packet)

    def test_disabled_payload_byte_can_be_unknown(self):
        data=place((0,completion(1,1,2,address=1)+b'abcd'))
        known=((1<<512)-1) ^ (255<<96) ^ (255<<120)
        packet=StraddleDecoder('rc').accept(10,LogicWord(data,known,512),
                    6<<12 | 1<<64 | 1<<76 | 3<<80)[0]
        self.assertEqual(decode_completion(packet)['enabled_payload'],b'bc')

    def test_timeout_invalid_fields_can_be_unknown(self):
        data=place((0,completion(17,0,0,error=9,function=3)))
        known=15<<12 | 1<<30 | 255<<48 | 255<<64
        packet=StraddleDecoder('rc').accept(10,LogicWord(data,known,512),1<<64 | 1<<76 | 2<<80)[0]
        row=decode_completion(packet)
        self.assertEqual((row['tag'],row['requester_function'],row['request_completed']),(17,3,True))
        self.assertFalse(row['normal'])

    def test_rq_descriptor_known_even_when_idle_lanes_unknown(self):
        data=place((0,request(0x100,1,7)))
        user=15 | 1<<20 | 1<<26 | 3<<28
        p=StraddleDecoder('rq').accept(10,LogicWord(data,(1<<128)-1,512),user)[0]
        self.assertEqual(decode_memory_request(p)['tag'],7)
        p=StraddleDecoder('rq').accept(10,LogicWord(data,((1<<128)-1) ^ (1<<96),512),user)[0]
        with self.assertRaisesRegex(ValueError,'unknown RQ'):decode_memory_request(p)


if __name__=='__main__':unittest.main()
