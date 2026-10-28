"""Contract bench with an independent sparse-memory sink; no vendor IP or RTL."""
from collections import deque
from dataclasses import replace
import itertools
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ntrpu_hbm_dma_bridge_model import BridgeModel, Route, Span, Event
from ntrpu_transport_model import RegionHandle, TransferId, TransportError, TransportModel


TARGET = 1 << 36
WRITEBACK = 1 << 44
SUMMARY = []


def pattern(sequence, length):
    return bytes(((i * 37) ^ (i >> 7) ^ (sequence * 53)) & 255 for i in range(length))


def setup_model(depth=4):
    route = Route(RegionHandle(1, 0, 0, 1), RegionHandle(1, 1, 0, 1),
                  1, 7, 0, 1 << 34, TARGET, 1 << 33, WRITEBACK)
    model = BridgeModel(payload_depth=depth)
    model.configure(route)
    return model, route


def span(route, sequence=1, channel=0, length=128, offset=0):
    return Span(TransferId(1, 0, sequence), route.source, route.target, 1, 7,
                (channel << 33) + offset, TARGET + (channel << 32) + offset, length)


def ack(request):
    return Event(request.identity, request.target, request.window_generation, request.length)


class Destination:
    """Checks public descriptors/beats against a test-owned transfer manifest.

    Uses no model private state. Source patterns, full destination addresses
    and writeback bytes are independently derived; unwritten addresses retain
    sentinels. Logical completion is separate from observed memory contents.
    """
    def __init__(self, requests):
        self.expected = {s.identity: s for s in requests}
        self.descriptors = [deque(), deque()]
        self.memory = {}
        self.writebacks = {}
        self.finished = []
        self.offsets = {}

    def descriptor(self, value):
        s = self.expected[value.identity]
        expected_channel = (s.target_address - TARGET) >> 32
        if value.channel != expected_channel:
            raise AssertionError('channel identity mismatch')
        if (value.dst_addr, value.length, value.control) != (s.target_address, s.length, 0x10):
            raise AssertionError('descriptor range/control mismatch')
        low = WRITEBACK + value.channel * 128
        if value.src_addr not in (low, low + 64):
            raise AssertionError('writeback is not an authorized channel slot')
        self.descriptors[value.channel].append(value)
        self.offsets[value.identity] = 0

    def beat(self, channel, value):
        if not self.descriptors[channel]:
            raise AssertionError('data without descriptor')
        d = self.descriptors[channel][0]
        s = self.expected[d.identity]
        offset = self.offsets[d.identity]
        expected = pattern(s.identity.sequence, s.length)[offset:offset + 64]
        if (value.identity != d.identity or value.address != s.target_address + offset
                or value.data != expected or value.keep != (1 << 64) - 1
                or value.last != (offset + 64 == s.length)):
            raise AssertionError('payload, address or boundary mismatch')
        self.memory[value.address] = value.data
        self.offsets[d.identity] += 64
        if value.last:
            self.descriptors[channel].popleft()
            self.writebacks[d.src_addr] = (0x52b40001).to_bytes(4, 'little') + s.length.to_bytes(4, 'little')
            self.finished.append(s.identity)

    def verify(self):
        expected = {}
        for s in self.expected.values():
            data = pattern(s.identity.sequence, s.length)
            expected.update({s.target_address + i: data[i:i + 64] for i in range(0, s.length, 64)})
        if self.memory != expected:
            raise AssertionError('full-address memory differs from independent image')
        if len(self.finished) != len(self.expected):
            raise AssertionError('missing packet completion')


class BridgeTests(unittest.TestCase):
    def code(self, expected, fn, *args, **kwargs):
        with self.assertRaises(TransportError) as caught:
            fn(*args, **kwargs)
        self.assertEqual(caught.exception.code, expected)

    def transfer(self, *, length=128, depth=4):
        model, route = setup_model(depth)
        request = span(route, length=length)
        descriptor = model.reserve(0, request)
        model.set_run(0, True)
        self.assertEqual(model.issue(0, ready=True), descriptor)
        data = pattern(1, length)
        for offset in range(0, length, 64):
            self.assertTrue(model.push(0, request.identity, data[offset:offset + 64], last=offset + 64 == length))
            self.assertIsNotNone(model.take(0, ready=True))
        return model, route, request

    def test_two_channels_random_schedules_and_sparse_memory(self):
        # Two independent logical clocks. This is not a CDC electrical model.
        for seed in range(24):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                model, route = setup_model(depth=1 + seed % 4)
                requests = [span(route, 1, 0, 64), span(route, 2, 1, 128),
                            span(route, 3, 0, 4096, 4096), span(route, 4, 1, 192, 4096)]
                producers = [deque(), deque()]
                for i, s in enumerate(requests):
                    model.reserve(i % 2, s)
                    producers[i % 2].append([s, 0])
                sink = Destination(requests)
                for c in (0, 1):
                    model.set_run(c, True)
                local_pending, done_pending, ack_pending = [], [], []
                retired = []
                for ns in range(30000):
                    if ns % 5 == 0:
                        for c in (0, 1):
                            if producers[c] and rng.randrange(4):
                                s, offset = producers[c][0]
                                data = pattern(s.identity.sequence, s.length)[offset:offset + 64]
                                if model.push(c, s.identity, data, last=offset + 64 == s.length):
                                    producers[c][0][1] += 64
                                    if offset + 64 == s.length:
                                        producers[c].popleft()
                                        local_pending.append(s)
                            model.poll_done(c, ready=rng.randrange(4) != 0)
                        if local_pending and rng.randrange(3):
                            s = local_pending.pop(0)
                            model.local_complete(s.identity, s.length)
                        if ack_pending and rng.randrange(3):
                            model.acknowledge(ack(ack_pending.pop(0)))
                        for c in (0, 1):
                            if model.result(c) and rng.randrange(3):
                                retired.append(model.retire(c).identity)
                    if ns % 4 == seed % 4:
                        for c in (0, 1):
                            d = model.issue(c, ready=rng.randrange(3) != 0)
                            if d:
                                sink.descriptor(d)
                            beat = model.take(c, ready=rng.randrange(4) != 0)
                            if beat:
                                sink.beat(c, beat)
                                if beat.last:
                                    s = sink.expected[beat.identity]
                                    done_pending.append((c, s))
                                    ack_pending.append(s)
                        if done_pending and rng.randrange(3):
                            c, s = done_pending.pop(0)
                            self.assertEqual(model.observe_done(c).identity, s.identity)
                    if len(retired) == 4:
                        break
                self.assertEqual(set(retired), set(s.identity for s in requests))
                sink.verify()
                self.assertEqual(len(sink.writebacks), 4)
                self.assertEqual(model.resources()['contexts'], [0, 0])
                self.assertEqual(model.resources()['events'], [0, 0])
                self.assertFalse(model.halted)
                SUMMARY.append(dict(seed=seed, logical_ns=ns, bytes=sum(s.length for s in requests),
                                    events=len(model.trace), channels=2, retired=4,
                                    trace_sha256=hashlib.sha256(json.dumps(model.trace, sort_keys=True).encode()).hexdigest(),
                                    memory_sha256=hashlib.sha256(b''.join(a.to_bytes(8, 'little') + data
                                                 for a, data in sorted(sink.memory.items()))).hexdigest()))

    def test_context_limit_and_writeback_slots_hold_until_ack(self):
        model, route = setup_model()
        descriptors = [model.reserve(i % 2, span(route, i + 1, i % 2, 64, (i // 2) * 4096))
                       for i in range(4)]
        self.assertEqual(len({d.src_addr for d in descriptors}), 4)
        self.code('CONTEXT_FULL', model.reserve, 0, span(route, 5, 0, 64, 8192))
        self.assertEqual(model.resources()['contexts'], [2, 2])

    def test_descriptor_waits_for_ready_and_run(self):
        model, route = setup_model()
        d = model.reserve(1, span(route, channel=1))
        for _ in range(8):
            self.assertIsNone(model.issue(1, ready=True))
            self.assertEqual(model.descriptor(1), d)
        model.set_run(1, True)
        self.assertIsNone(model.issue(1, ready=False))
        self.assertEqual(model.issue(1, ready=True), d)
        self.assertIsNone(model.issue(1, ready=True))

    def test_payload_stability_and_bounded_backpressure(self):
        model, route = setup_model(depth=1)
        s = span(route)
        model.reserve(0, s)
        model.push(0, s.identity, b'a' * 64, last=False)
        self.assertIsNone(model.peek(0))
        self.assertFalse(model.push(0, s.identity, b'b' * 64, last=True))
        model.set_run(0, True)
        model.issue(0, ready=True)
        offered = model.peek(0)
        for _ in range(20):
            self.assertIsNone(model.take(0, ready=False))
            self.assertEqual(model.peek(0), offered)
        self.assertEqual(model.take(0, ready=True), offered)
        self.assertTrue(model.push(0, s.identity, b'b' * 64, last=True))

    def test_all_three_completion_orders_require_result_acceptance(self):
        for order in itertools.permutations(['local', 'dma', 'ack']):
            model, route, s = self.transfer()
            operations = {'local': lambda: model.local_complete(s.identity, s.length),
                          'dma': lambda: (model.observe_done(0), model.poll_done(0, ready=True)),
                          'ack': lambda: model.acknowledge(ack(s))}
            for index, operation in enumerate(order):
                operations[operation]()
                if index < 2:
                    self.assertIsNone(model.result(0))
            self.assertEqual(model.resources()['contexts'], [1, 0])
            self.assertEqual(model.retire(0), ack(s))
            self.assertEqual(model.resources()['contexts'], [0, 0])

    def test_consecutive_done_pulses_survive_consumer_stall(self):
        model, route = setup_model()
        requests = [span(route, i + 1, 1, 64, i * 64) for i in range(2)]
        model.set_run(1, True)
        for s in requests:
            model.reserve(1, s)
            model.issue(1, ready=True)
            model.push(1, s.identity, b'x' * 64, last=True)
            model.take(1, ready=True)
        for s in requests:
            self.assertEqual(model.observe_done(1).identity, s.identity)
        for _ in range(50):
            self.assertIsNone(model.poll_done(1, ready=False))
        self.assertEqual(model.resources()['events'], [0, 2])
        self.assertEqual([model.poll_done(1, ready=True).identity for _ in range(2)],
                         [s.identity for s in requests])
        self.assertEqual(model.resources()['contexts'], [0, 2])

    def test_extra_or_early_done_quarantines(self):
        model, route = setup_model()
        s = span(route)
        model.reserve(0, s)
        model.set_run(0, True)
        model.issue(0, ready=True)
        self.code('EARLY_DONE', model.observe_done, 0)
        self.assertEqual(model.resources()['quarantined'], 1)
        model, _, s = self.transfer()
        model.observe_done(0)
        self.code('UNEXPECTED_DONE', model.observe_done, 0)
        self.code('QUARANTINED', model.retire, 0)

    def test_malformed_last_or_cpl_never_success(self):
        for last in (False, True):
            model, route = setup_model()
            s = span(route, length=64 if not last else 128)
            model.reserve(0, s)
            self.code('LAST_MISMATCH', model.push, 0, s.identity, b'x' * 64, last=last)
            self.assertEqual(model.resources()['payload'], [0, 0])
            self.code('QUARANTINED', model.retire, 0)
        for length, status in [(64, 'OK'), (128, 'HBM_ERROR')]:
            model, _, s = self.transfer()
            self.code('LOCAL_CPL', model.local_complete, s.identity, length, status=status)
            self.assertEqual(model.result(0).length, 0)

    def test_no_packet_interleaving_within_channel(self):
        model, route = setup_model()
        a, b = span(route), span(route, 2, offset=4096)
        model.reserve(0, a)
        model.reserve(0, b)
        model.push(0, a.identity, b'a' * 64, last=False)
        self.code('PACKET_INTERLEAVE', model.push, 0, b.identity, b'b' * 64, last=False)
        self.assertEqual(model.resources()['quarantined'], 2)

    def test_dma_error_after_partial_payload_retains_every_context(self):
        model, route = setup_model()
        a, b = span(route), span(route, 2, 1)
        for c, s in enumerate([a, b]):
            model.reserve(c, s)
            model.set_run(c, True)
            model.issue(c, ready=True)
        model.push(0, a.identity, b'a' * 64, last=False)
        model.take(0, ready=True)
        self.assertEqual(model.observe_done(0, error='DMA_ERROR').length, 0)
        self.assertEqual(model.poll_done(0, ready=True).status, 'DMA_ERROR')
        self.assertEqual(model.resources()['quarantined'], 2)
        for c in (0, 1):
            self.assertEqual(model.result(c).length, 0)
            self.code('QUARANTINED', model.retire, c)

    def test_ack_authentication_early_duplicate_and_stale(self):
        mutations = [dict(target=RegionHandle(1, 1, 0, 2)), dict(window_generation=2),
                     dict(length=64), dict(status='REMOTE_ERROR'),
                     dict(identity=TransferId(2, 0, 1))]
        for fields in mutations:
            model, _, s = self.transfer()
            self.code('UNKNOWN_EVENT' if 'identity' in fields else 'BAD_ACK',
                      model.acknowledge, replace(ack(s), **fields))
            self.assertEqual(model.resources()['quarantined'], 1)
        model, route = setup_model()
        s = span(route)
        model.reserve(0, s)
        self.code('BAD_ACK', model.acknowledge, ack(s))
        model, _, s = self.transfer()
        model.acknowledge(ack(s))
        self.code('BAD_ACK', model.acknowledge, ack(s))

    def test_link_loss_at_each_boundary_and_recovery_does_not_release(self):
        for stage in range(7):
            model, route = setup_model()
            s = span(route)
            model.reserve(0, s)
            model.set_run(0, True)
            if stage >= 1:
                model.issue(0, ready=True)
            if stage >= 2:
                model.push(0, s.identity, b'a' * 64, last=False)
            if stage >= 3:
                model.take(0, ready=True)
                model.push(0, s.identity, b'b' * 64, last=True)
                model.take(0, ready=True)
                model.local_complete(s.identity, s.length)
            if stage >= 4:
                model.observe_done(0)
            if stage >= 5:
                model.poll_done(0, ready=True)
            if stage >= 6:
                model.acknowledge(ack(s))
            offered = model.peek(0)
            model.set_link(False)
            self.assertEqual(model.peek(0), offered)
            for _ in range(100):
                self.assertEqual(model.resources()['contexts'], [1, 0])
            model.set_link(True)
            self.code('QUARANTINED', model.retire, 0)
            self.code('HALTED', model.reserve, 1, span(route, 2, 1))

    def test_map_change_blocked_through_ack_and_result(self):
        model, route, s = self.transfer()
        new_route = replace(route, window_generation=2)
        model.local_complete(s.identity, s.length)
        model.observe_done(0)
        model.poll_done(0, ready=True)
        self.code('MAP_BUSY', model.configure, new_route)
        model.acknowledge(ack(s))
        self.code('MAP_BUSY', model.configure, new_route)
        model.retire(0)
        model.configure(new_route)
        self.code('STALE_CONTEXT', model.reserve, 0, span(route, 2))
        self.code('REPLAY', model.reserve, 0, replace(s, window_generation=2))

    def test_stale_ack_cannot_complete_reused_slot(self):
        model, route, first = self.transfer()
        model.local_complete(first.identity, first.length)
        model.observe_done(0); model.poll_done(0, ready=True)
        model.acknowledge(ack(first)); model.retire(0)
        second = span(route, 2)
        model.reserve(0, second)
        self.code('UNKNOWN_EVENT', model.acknowledge, ack(first))
        self.assertEqual(model.result(0).identity, second.identity)
        self.assertEqual(model.result(0).length, 0)
        self.code('QUARANTINED', model.retire, 0)

    def test_range_alignment_and_overlap_rejections_have_no_publication(self):
        changes = [dict(length=0), dict(length=65), dict(length=4160),
                   dict(source_address=(1 << 34) - 64), dict(target_address=(1 << 64) - 64),
                   dict(source_address=4096 - 64), dict(target_address=TARGET - 64)]
        for fields in changes:
            model, route = setup_model()
            with self.assertRaises(TransportError):
                model.reserve(0, replace(span(route), **fields))
            self.assertIsNone(model.descriptor(0))
            self.assertEqual(model.resources()['contexts'], [0, 0])
        model, route = setup_model()
        model.reserve(0, span(route))
        self.code('TARGET_BUSY', model.reserve, 1, span(route, 2))
        self.code('WRITEBACK_OVERLAP', setup_model()[0].configure,
                  replace(route, writeback_base=TARGET))

    def test_independent_sink_rejects_address_payload_and_writeback_corruption(self):
        model, route = setup_model()
        s = span(route, channel=1, length=64)
        d = model.reserve(1, s)
        for change in [dict(dst_addr=d.dst_addr & 0xffffffff), dict(src_addr=s.source_address),
                       dict(channel=0)]:
            with self.assertRaises(AssertionError):
                Destination([s]).descriptor(replace(d, **change))
        model.set_run(1, True)
        model.issue(1, ready=True)
        model.push(1, s.identity, pattern(1, 64), last=True)
        beat = model.take(1, ready=True)
        for change in [dict(data=bytes(64)), dict(address=beat.address & 0xffffffff), dict(last=False)]:
            sink = Destination([s]); sink.descriptor(d)
            with self.assertRaises(AssertionError):
                sink.beat(1, replace(beat, **change))

    def test_transport_oracle_agrees_on_bytes_and_remote_publication(self):
        model, route = setup_model()
        s = span(route, length=256)
        transport = TransportModel(fragment_bytes=64)
        source = transport.register(0, 0, 4096)
        target = transport.register(1, TARGET, 4096)
        transport.host_write(source, 0, pattern(1, 256))
        window = transport.bind_window(target)
        identity = transport.submit(1, 1, source, window, 256)
        d = model.reserve(0, s)
        sink = Destination([s]); sink.descriptor(d)
        model.set_run(0, True); model.issue(0, ready=True)
        for index in range(4):
            packet = transport.send(identity, index)
            transport.receive(packet); transport.commit(packet)
            data = pattern(1, 256)[index * 64:(index + 1) * 64]
            model.push(0, s.identity, data, last=index == 3)
            sink.beat(0, model.take(0, ready=True))
        model.local_complete(s.identity, 256)
        model.observe_done(0); model.poll_done(0, ready=True)
        self.assertIsNone(model.result(0))
        completion = transport.confirm_remote(identity)
        model.acknowledge(ack(s)); result = model.retire(0)
        self.assertEqual(result.length, completion.valid_bytes)
        self.assertEqual(transport.read(target, 0, 256), b''.join(sink.memory[a] for a in sorted(sink.memory)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
