"""Independent transaction scenarios for tag lifetime and completion payloads."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ntrpu_pcie_request_inventory import RequestInventory, AcceptedBeatMonitor
from test_ntrpu_pcie_straddle import place, request, completion


def rq(tag, address=0x100, count=1, first=15, last=None):
    if last is None:
        last = 0 if count == 1 else 15
    return place((0, request(address, count, tag))), first | last << 8 | 1 << 20 | 1 << 26 | 3 << 28


def rc(tag, payload=b'abcd', address=0x100, remaining=None, final=True, error=0, function=0):
    count = (len(payload)+3)//4
    if remaining is None:
        remaining = len(payload)
    raw = completion(tag, count, remaining, address, final, error, function)+payload
    assert len(raw) <= 64 and len(payload) % 4 == 0
    delivered = min(remaining, len(payload)-(address & 3)) if not error else 0
    enable = ((1 << delivered)-1) << (12+(address & 3))
    return place((0, raw)), enable | 1 << 64 | 1 << 76 | (2+count) << 80


class InventoryTests(unittest.TestCase):
    def test_backpressure_counts_only_acceptance(self):
        for interface,ready in [('rq',15),('rc',1)]:
            m=AcceptedBeatMonitor(interface)
            self.assertIsNone(m.sample(0,ready,None,None))
            for _ in range(7):self.assertIsNone(m.sample(1,0,123,456))
            self.assertEqual(m.sample(1,ready,123,456),(123,456))
            self.assertEqual((m.stalled_cycles,m.accepted_beats),(7,1))
            m.finish()

    def test_backpressure_mutations_rejected(self):
        for beat in [(0,0,123,456),(1,0,124,456),(1,15,123,457)]:
            m=AcceptedBeatMonitor('rq');m.sample(1,0,123,456)
            with self.assertRaisesRegex(ValueError,'backpressure'):m.sample(*beat)
        for ready in (1,3,7,None):
            with self.assertRaisesRegex(ValueError,'handshake'):AcceptedBeatMonitor('rq').sample(1,ready,0,0)
        m=AcceptedBeatMonitor('rc');m.sample(1,0,0,0)
        with self.assertRaisesRegex(ValueError,'unaccepted'):m.finish()

    def test_full_4k_payload_across_65_beats(self):
        m=RequestInventory({0:0});m.cycle(10,rq=rq(9,0,count=1024))
        payload=bytes(n % 251 for n in range(4096))
        raw=completion(9,1024,4096,0)+payload
        for index,offset in enumerate(range(0,len(raw),64)):
            chunk=raw[offset:offset+64]
            enable=sum(1 << lane for lane in range(len(chunk)) if offset+lane>=12)
            user=enable | (1 << 64 if index==0 else 0)
            if offset+64>=len(raw):user |= 1 << 76 | (len(chunk)//4-1) << 80
            m.cycle(20+index,rc=(place((0,chunk)),user))
        m.finish()
        self.assertEqual(m.rc.accepted_beats,65)
        self.assertEqual(m.completed_reads[0]['payload'],payload)
        self.assertEqual(m.peak_span_bytes,4096)

    def test_tag_reuse_while_old_completion_payload_remains(self):
        m = RequestInventory({0: 0})
        m.cycle(10, rq=rq(7, count=16))
        m.cycle(20, rc=(place((0, completion(7, 16, 64, 0x100)), (12, bytes(range(52)))),
                       ((1 << 52)-1) << 12 | 1 << 64), rq=rq(7, 0x200))
        self.assertEqual((len(m.tags), len(m.live), len(m.payloads)), (1, 2, 1))
        self.assertFalse(m.empty)
        self.assertEqual(m.peak_reads, 2)
        m.cycle(30, rc=(place((0, bytes(range(52, 64)))), 0xfff | 1 << 76 | 2 << 80))
        self.assertEqual((len(m.tags), len(m.live)), (1, 1))
        self.assertEqual(m.completed_reads[0]['payload'], bytes(range(64)))
        self.assertEqual((m.completed_reads[0]['terminal_header_ps'], m.completed_reads[0]['payload_end_ps']), (20,30))
        m.cycle(40, rc=rc(7, address=0x200))
        m.finish()

    def test_table_empty_does_not_drop_terminal_payload(self):
        m = RequestInventory({0: 0})
        m.cycle(10, rq=rq(7, count=16))
        m.cycle(20, rc=(place((0, completion(7,16,64,0x100))), ((1 << 52)-1) << 12 | 1 << 64))
        self.assertFalse(m.tags)
        self.assertFalse(m.empty)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            m.finish()

    def test_split_same_tag_and_reordering_other_tag(self):
        m = RequestInventory({0: 0})
        m.cycle(10, rq=rq(1, count=8))
        m.cycle(20, rq=rq(2, 0x200))
        m.cycle(30, rc=rc(1, bytes(range(16)), remaining=32, final=False))
        self.assertEqual(m.live[0]['remaining'], 16)
        m.cycle(40, rc=rc(2, address=0x200))
        m.cycle(50, rc=rc(1, bytes(range(16,32)), address=0x110, remaining=16))
        m.finish()
        self.assertEqual([r['serial'] for r in m.completed_reads], [1,0])
        self.assertEqual(m.completed_reads[1]['payload'], bytes(range(32)))
        self.assertEqual(m.completed_reads[1]['completion_count'], 2)

    def test_two_splits_same_tag_in_one_beat(self):
        m=RequestInventory({0:0});m.cycle(10,rq=rq(8,count=8))
        raw=place((0,completion(8,4,32,0x100,False)+bytes(range(16))),
                  (32,completion(8,4,16,0x110,True)+bytes(range(16,32))))
        user=0xffff << 12 | 0xffff << 44 | 3 << 64 | 2 << 70 | 3 << 76 | 6 << 80 | 14 << 84
        m.cycle(20,rc=(raw,user));m.finish()
        self.assertEqual(m.completed_reads[0]['payload'],bytes(range(32)))
        self.assertEqual(m.completed_reads[0]['completion_count'],2)

    def test_late_discontinue_keeps_error_after_terminal_header(self):
        m=RequestInventory({0:0});m.cycle(10,rq=rq(7,count=16))
        m.cycle(20,rc=(place((0,completion(7,16,64,0x100))),((1 << 52)-1) << 12 | 1 << 64))
        m.cycle(30,rc=(0,0xfff | 1 << 76 | 2 << 80 | 1 << 96))
        self.assertTrue(m.empty)
        self.assertFalse(m.completed_reads[0]['successful'])
        with self.assertRaisesRegex(ValueError,'sticky'):m.finish()

    def test_256_tags_live_then_four_completions_each_cycle(self):
        m = RequestInventory({0: 0})
        for pair in range(128):
            raw = place((0, request(0x100,1,2*pair)), (32,request(0x100,1,2*pair+1)))
            user = 0xff | 3 << 20 | 2 << 24 | 3 << 26 | 3 << 28 | 11 << 32
            m.cycle(10+pair, rq=(raw,user))
        self.assertEqual((m.peak_tags,m.peak_reads,m.peak_span_bytes), (256,256,1024))
        for batch in range(64):
            parts = []
            user = 15 << 64 | 0xe4 << 68 | 15 << 76 | 0xfb73 << 80
            for slot in range(4):
                tag = 255-(4*batch+slot)
                parts.append((16*slot, completion(tag,1,4,0x100)+tag.to_bytes(4,'little')))
                user |= 15 << (16*slot+12)
            m.cycle(200+batch, rc=(place(*parts),user))
        m.finish()
        self.assertEqual((len(m.completed_reads),m.rq.max_starts_per_beat,m.rc.max_starts_per_beat), (256,2,4))
        self.assertEqual([r['tag'] for r in m.completed_reads],list(reversed(range(256))))

    def test_capacity_and_duplicate_tags_rejected(self):
        for new_tag, message in [(1,'reused'),(2,'capacity')]:
            m = RequestInventory({0:0},capacity=1)
            m.cycle(10,rq=rq(1))
            with self.assertRaisesRegex(ValueError,message):m.cycle(20,rq=rq(new_tag))

    def test_holes_in_byte_enables_count_span_not_popcount(self):
        m = RequestInventory({0:0})
        m.cycle(10,rq=rq(1,first=5))
        self.assertEqual((m.requests[0]['requested_bytes'],m.requests[0]['completion_span_bytes']), (2,3))
        m.cycle(20,rc=rc(1,b'abcd',remaining=3))
        m.finish()
        self.assertEqual(m.completed_reads[0]['payload'], b'ac')

    def test_unaligned_and_zero_length_reads(self):
        for first, low, span, expected in [(6,0x101,2,b'bc'),(0,0x100,1,b'')]:
            with self.subTest(first=first):
                m = RequestInventory({0:0})
                m.cycle(10,rq=rq(1,first=first))
                m.cycle(20,rc=rc(1,b'abcd',address=low,remaining=span))
                m.finish()
                self.assertEqual(m.completed_reads[0]['payload'],expected)

    def test_remaining_address_identity_terminal_and_enable_corruption(self):
        base_data, base_user = rc(1)
        variants = [
            (base_data ^ 1 << 16,base_user,'byte count'),
            (base_data ^ 1,base_user,'address'),
            (base_data ^ 1 << 56,base_user,'requester ID'),
            (base_data ^ 1 << 89,base_user,'TC/attributes'),
            (base_data ^ 1 << 92,base_user,'TC/attributes'),
            (base_data ^ 1 << 30,base_user,'terminal'),
            (base_data,base_user ^ 1 << 12,'enabled byte'),
        ]
        for data,user,message in variants:
            with self.subTest(message=message):
                m = RequestInventory({0:0});m.cycle(10,rq=rq(1))
                with self.assertRaisesRegex(ValueError,message):m.cycle(20,rc=(data,user))

    def test_unknown_completion_cannot_match_same_cycle_request(self):
        m = RequestInventory({0:0})
        with self.assertRaisesRegex(ValueError,'unknown'):m.cycle(10,rq=rq(1),rc=rc(1))

    def test_timeout_and_ur_retire_tag_but_remain_failed(self):
        for error in (2,9):
            with self.subTest(error=error):
                m = RequestInventory({0:0});m.cycle(10,rq=rq(1))
                if error == 9:
                    data=place((0,completion(1,2047,8191,0xfff,True,error=9)))
                    beat=(data,1 << 64 | 1 << 76 | 2 << 80)
                else:
                    beat=rc(1,b'',remaining=0,error=error)
                m.cycle(20,rc=beat)
                self.assertTrue(m.empty)
                self.assertFalse(m.completed_reads[0]['successful'])
                with self.assertRaisesRegex(ValueError,'sticky'):m.finish()

    def test_effective_bus_id_is_not_raw_requester_descriptor(self):
        m=RequestInventory({0:0x1a0});m.cycle(10,rq=rq(1))
        data,user=rc(1)
        m.cycle(20,rc=(data | 0x1a0 << 48,user));m.finish()
        self.assertEqual(m.requests[0]['requester_id'],0)
        self.assertEqual(m.requests[0]['effective_requester_id'],0x1a0)

    def test_posted_write_is_not_a_completed_read(self):
        m=RequestInventory({0:0})
        data=place((0,request(0x600,1,0,True)+b'abcd'))
        m.cycle(10,rq=(data,15 | 1 << 20 | 1 << 26 | 4 << 28))
        m.finish()
        self.assertTrue(m.empty)
        self.assertEqual(len(m.requests),1)
        self.assertEqual(m.completed_reads,[])

    def test_pending_read_and_illegal_4k_crossing(self):
        m=RequestInventory({0:0});m.cycle(10,rq=rq(1))
        with self.assertRaisesRegex(ValueError,'outstanding'):m.finish()
        with self.assertRaisesRegex(ValueError,'4KiB'):
            RequestInventory({0:0}).cycle(10,rq=rq(1,0xffc,count=2))


if __name__ == '__main__':
    unittest.main()
