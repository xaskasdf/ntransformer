"""Protocol vectors for the generated 512-bit RQ2/RC4 evidence decoder."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ntrpu_pcie_straddle import StraddleDecoder, decode_completion, decode_memory_request


def place(*segments):
    data = bytearray(64)
    for lane, raw in segments:
        data[lane:lane+len(raw)] = raw
    return int.from_bytes(data, 'little')


def request(address, count, tag, write=False):
    return (address | count << 64 | int(write) << 75 | tag << 96).to_bytes(16, 'little')


def completion(tag, count, remaining, address=0, final=True, error=0, function=0):
    return (address | error << 12 | remaining << 16 | int(final) << 30 |
            count << 32 | function << 48 | tag << 64).to_bytes(12, 'little')


class StraddleTests(unittest.TestCase):
    def test_two_requests_one_beat(self):
        d = StraddleDecoder('rq')
        # SOP at DWORD0/8, EOP at DWORD3/11, independent BE nibbles.
        user = 0xffff | 3 << 20 | 2 << 24 | 3 << 26 | 3 << 28 | 11 << 32
        packets = d.accept(100, place((0, request(0x100, 16, 0xa1)),
                                      (32, request(0x200, 8, 0xb2))), user)
        rows = [decode_memory_request(p) for p in packets]
        self.assertEqual([(r['address'], r['tag'], r['requested_bytes']) for r in rows],
                         [(0x100, 0xa1, 64), (0x200, 0xb2, 32)])
        self.assertEqual((d.max_starts_per_beat, d.max_ends_per_beat), (2, 2))
        d.finish()

    def test_write_continuation_then_read(self):
        d = StraddleDecoder('rq')
        pattern = bytes(range(64))
        self.assertEqual(d.accept(100, place((0, request(0x600, 16, 7, True)),
                                             (16, pattern[:48])), 0xf0f | 1 << 20), [])
        self.assertIsNotNone(d.pending)
        packets = d.accept(200, place((0, pattern[48:]), (32, request(0x400, 1, 8))),
                           5 | 1 << 20 | 2 << 22 | 3 << 26 | 3 << 28 | 11 << 32)
        a, b = [decode_memory_request(p) for p in packets]
        self.assertEqual(a['enabled_payload'], pattern)
        self.assertEqual((a['start_ps'], a['end_ps']), (100, 200))
        self.assertEqual((b['byte_offsets'], b['tag'], b['start_ps']), ([0, 2], 8, 200))
        d.finish()

    def test_four_completions_one_beat(self):
        d = StraddleDecoder('rc')
        parts = []
        user = 15 << 64 | (0 | 1 << 2 | 2 << 4 | 3 << 6) << 68 | 15 << 76
        user |= (3 | 7 << 4 | 11 << 8 | 15 << 12) << 80
        for n in range(4):
            parts.append((16*n, completion(0x90+n, 1, 4) + bytes([n+1])*4))
            user |= 15 << (16*n+12)
        rows = [decode_completion(p) for p in d.accept(100, place(*parts), user)]
        self.assertEqual([r['tag'] for r in rows], [0x90, 0x91, 0x92, 0x93])
        self.assertEqual([r['enabled_payload'] for r in rows], [bytes([n])*4 for n in range(1,5)])
        self.assertTrue(all(r['normal'] and r['request_completed'] for r in rows))
        self.assertEqual((d.max_starts_per_beat, d.max_ends_per_beat), (4, 4))
        d.finish()

    def test_completion_continuation_then_new_packet(self):
        d = StraddleDecoder('rc')
        pattern = bytes(range(64))
        self.assertEqual(d.accept(100, place((0, completion(21,16,64)), (12, pattern[:52])),
                                  ((1 << 52)-1) << 12 | 1 << 64), [])
        user = 0xfff | 15 << 28 | 1 << 64 | 1 << 68 | 3 << 76 | 2 << 80 | 7 << 84
        packets = d.accept(200, place((0, pattern[52:]),
                           (16, completion(22,1,4) + b'abcd')), user)
        rows = [decode_completion(p) for p in packets]
        self.assertEqual([r['enabled_payload'] for r in rows], [pattern,b'abcd'])
        self.assertEqual([r['start_ps'] for r in rows], [100,200])
        d.finish()

    def test_split_descriptors_keep_remaining_count_and_final_separate(self):
        d = StraddleDecoder('rc')
        p1 = d.accept(100, place((0,completion(8,4,32,0x40,False)+bytes(range(16)))),
                      0xffff << 12 | 1 << 64 | 1 << 76 | 6 << 80)[0]
        p2 = d.accept(200, place((0,completion(8,4,16,0x50,True)+bytes(range(16,32)))),
                      0xffff << 12 | 1 << 64 | 1 << 76 | 6 << 80)[0]
        a,b = map(decode_completion,[p1,p2])
        self.assertEqual((a['byte_count'],b['byte_count']), (32,16))
        self.assertFalse(a['request_completed']); self.assertTrue(b['request_completed'])
        self.assertEqual(a['enabled_payload']+b['enabled_payload'],bytes(range(32)))

    def test_timeout_invalid_fields_not_interpreted_as_success(self):
        d = StraddleDecoder('rc')
        raw=bytearray(completion(0xac,2047,8191,0xfff,True,error=9,function=3))
        row=decode_completion(d.accept(100,place((0,raw)),1 << 64 | 1 << 76 | 2 << 80)[0])
        self.assertEqual((row['tag'],row['requester_function'],row['error_code']), (0xac,3,9))
        self.assertTrue(row['request_completed']); self.assertFalse(row['normal'])
        for field in ['requester_id','lower_address','byte_count','dword_count','completion_status','poisoned','tc','attributes']:
            self.assertIsNone(row[field])

    def test_error_completion_preserves_error(self):
        d=StraddleDecoder('rc')
        row=decode_completion(d.accept(100,place((0,completion(5,0,0,error=2))),
                              1 << 64 | 1 << 76 | 2 << 80)[0])
        self.assertTrue(row['request_completed']); self.assertFalse(row['normal'])
        self.assertEqual(row['error_code'],2)

    def test_discontinued_continuation_is_retained_as_failed(self):
        d=StraddleDecoder('rq')
        d.accept(100,place((0,request(0x600,32,9,True))),0xf0f | 1 << 20)
        p=d.accept(200,0,1 << 36 | 1 << 26 | 3 << 28)[0]
        self.assertTrue(decode_memory_request(p)['discontinued'])
        self.assertTrue(d.discontinued)

    def test_invalid_masks_and_boundaries(self):
        for interface,user in [('rq',2 << 20),('rc',5 << 64),('rq',2 << 26),('rc',5 << 76),
                               ('rq',1 << 20 | 1 << 22),('rq',1 << 26 | 3 << 28),
                               ('rc',1 << 76 | 2 << 80),('rq',1 << 20 | 1 << 26 | 2 << 28),
                               ('rc',1 << 64 | 1 << 76 | 1 << 80),
                               ('rc',3 << 64 | 3 << 76 | 3 << 80 | 7 << 84)]:
            with self.subTest(interface=interface,user=user), self.assertRaises(ValueError):
                StraddleDecoder(interface).accept(100,0,user)

    def test_rc_byte_enables_exclude_descriptor_and_idle_lanes(self):
        for enable in [1,1 << 20]:
            with self.subTest(enable=enable), self.assertRaises(ValueError):
                StraddleDecoder('rc').accept(100,0,enable | 1 << 64 | 1 << 76 | 2 << 80)

    def test_pending_packet_prevents_finish_and_overlapping_sop(self):
        d=StraddleDecoder('rq');d.accept(100,place((0,request(0,64,1,True))),0xf0f | 1 << 20)
        with self.assertRaisesRegex(ValueError,'incomplete'):d.finish()
        with self.assertRaisesRegex(ValueError,'overlaps'):d.accept(200,0,1 << 20)

    def test_lengths_and_byte_enables_checked_against_header(self):
        for header,user in [(request(0,1,1),0xf0f),(request(0,2,1),15),
                            (request(0,2,1,True),0xf0f)]:
            with self.subTest(header=header.hex(),user=user), self.assertRaises(ValueError):
                packet=StraddleDecoder('rq').accept(100,place((0,header)),user | 1 << 20 | 1 << 26 | 3 << 28)[0]
                decode_memory_request(packet)


if __name__=='__main__':
    unittest.main()
