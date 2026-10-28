"""Observable contract tests for the isolated transport model; no RTL involved."""

from dataclasses import replace
import pathlib
import random
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from ntrpu_transport_model import (  # noqa: E402
    Fence, PacketId, TransferId, TransportError, TransportModel, U64_MAX,
)


class TransportTests(unittest.TestCase):
    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(TransportError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def fixture(self, length=256, *, region_size=None, base=0, **config):
        model = TransportModel(**config)
        size = region_size or length
        source = model.register(0, base, size)
        target = model.register(1, base, size)
        payload = bytes((i * 37 + 11) % 256 for i in range(length))
        model.host_write(source, 0, payload)
        window = model.bind_window(target)
        return model, source, target, window, payload

    def complete(self, model, identity, length, order=None):
        count = (length + model.fragment_bytes - 1) // model.fragment_bytes
        for index in range(count) if order is None else order:
            packet = model.send(identity, index)
            model.receive(packet)
            model.commit(packet)
        return model.confirm_remote(identity)

    def test_discovery_and_configuration(self):
        model = TransportModel()
        self.assertEqual(model.discover()["capacities"], (512 << 30, 512 << 30))
        self.assertEqual(model.discover()["alignment"], 64)
        self.assert_code("VERSION_MISMATCH", model.discover, 2)
        self.assert_code("INVALID_U64", TransportModel, credits=0)
        self.assert_code("ALIGNMENT", TransportModel, fragment_bytes=65)
        self.assert_code("TWO_NODES_REQUIRED", TransportModel, capacities=(1024,))

    def test_success_needs_commit_and_remote_confirmation(self):
        model, source, target, window, payload = self.fixture(128)
        identity = model.submit(1, 1, source, window, 128)
        # Receiving/committing the last fragment alone cannot publish success.
        packet = model.send(identity, 1)
        self.assert_code("NOT_RECEIVED", model.commit, packet)
        model.receive(packet)
        self.assert_code("NOT_VISIBLE", model.confirm_remote, identity)
        model.commit(packet)
        self.assert_code("NOT_VISIBLE", model.confirm_remote, identity)
        packet = model.send(identity, 0)
        model.receive(packet)
        model.commit(packet)
        self.assertEqual(model.debug_read_raw(target, 0, 128), payload)
        self.assert_code("REGION_BUSY", model.read, target, 0, 128)
        self.assert_code("NOT_COMPLETE", model.completion, identity)
        result = model.confirm_remote(identity)
        self.assertEqual((result.status, result.valid_bytes), ("SUCCESS", 128))
        self.assertEqual(model.read(target, 0, 128), payload)
        self.assertEqual(model.retire(identity), result)
        self.assert_code("UNKNOWN_TRANSFER", model.retire, identity)

    def test_credits_hold_until_memory_commit(self):
        model, source, target, window, payload = self.fixture(192, credits=1)
        identity = model.submit(1, 1, source, window, 192)
        first = model.send(identity, 0)
        self.assertEqual(model.resources()["credits"], (1, 0))
        self.assert_code("NO_CREDIT", model.send, identity, 1)
        model.receive(first)
        self.assert_code("NO_CREDIT", model.send, identity, 1)
        model.commit(first)
        for index in (1, 2):
            packet = model.send(identity, index)
            model.receive(packet)
            model.commit(packet)
        model.confirm_remote(identity)
        self.assertEqual(model.read(target, 0, 192), payload)
        self.assertEqual(model.resources()["credits"], (1, 1))

    def test_completion_does_not_need_a_data_credit(self):
        model, source, target, window, payload = self.fixture(192, credits=1)
        first = model.submit(1, 1, source, window, 64)
        second = model.submit(1, 2, source, window, 64, source_offset=64, target_offset=64)
        packet = model.send(first, 0)
        model.receive(packet)
        model.commit(packet)
        outstanding = model.send(second, 0)
        self.assertEqual(model.resources()["credits"][1], 0)
        model.confirm_remote(first)
        model.retire(first)
        self.assertEqual(model.read(target, 0, 64), payload[:64])
        model.receive(outstanding)
        model.commit(outstanding)
        model.confirm_remote(second)
        self.assertEqual(model.read(target, 64, 64), payload[64:128])

    def test_queue_space_includes_unretired_completions(self):
        model, source, target, window, _ = self.fixture(192, queue_depth=1)
        first = model.submit(1, 1, source, window, 64)
        self.complete(model, first, 64)
        self.assertEqual(model.resources()["queue_used"], (1, 0))
        self.assert_code("QUEUE_FULL", model.submit, 1, 2, source, window, 64,
                         source_offset=64, target_offset=64)
        model.retire(first)
        second = model.submit(1, 2, source, window, 64, source_offset=64, target_offset=64)
        self.complete(model, second, 64)
        model.retire(second)
        self.assertEqual(model.resources()["queue_used"], (0, 0))

    def test_fence_orders_only_earlier_work(self):
        model, source, target, window, _ = self.fixture(256)
        first = model.submit(1, 3, source, window, 64)
        barrier = model.fence(0)
        second = model.submit(1, 7, source, window, 64, target_offset=64)
        self.assertEqual(model.fence_status(barrier), "PENDING")
        self.complete(model, first, 64)
        model.retire(first)
        self.assertEqual(model.fence_status(barrier), "SUCCESS")
        self.assertEqual(model.fence_status(model.fence(0)), "PENDING")
        self.complete(model, second, 64)
        self.assertEqual(model.fence_status(model.fence(0)), "SUCCESS")
        self.assert_code("INVALID_FENCE", model.fence_status, Fence(1, 0, 8))

    def test_completions_out_of_order_preserve_identity_and_fence(self):
        model, source, target, window, payload = self.fixture(256)
        first = model.submit(1, 1, source, window, 64)
        second = model.submit(1, 2, source, window, 64, source_offset=64, target_offset=64)
        barrier = model.fence(0)
        result = self.complete(model, second, 64)
        self.assertEqual(result.transfer, second)
        model.retire(second)
        self.assertEqual(model.fence_status(barrier), "PENDING")
        self.assert_code("NOT_COMPLETE", model.retire, first)
        self.complete(model, first, 64)
        self.assertEqual(model.fence_status(barrier), "SUCCESS")
        self.assertEqual(model.read(target, 0, 128), payload[:128])

    def test_bidirectional_credits_are_independent(self):
        model = TransportModel(credits=1)
        a = model.register(0, 0, 128)
        b = model.register(1, 0, 128)
        model.host_write(a, 0, b"a" * 64)
        model.host_write(b, 0, b"b" * 64)
        wa, wb = model.bind_window(a), model.bind_window(b)
        ab = model.submit(1, 1, a, wb, 64, target_offset=64)
        ba = model.submit(1, 1, b, wa, 64, target_offset=64)
        packets = [model.send(ab, 0), model.send(ba, 0)]
        self.assertEqual(model.resources()["credits"], (0, 0))
        for packet in reversed(packets):
            model.receive(packet)
            model.commit(packet)
            model.confirm_remote(packet.transfer)
        self.assertEqual(model.read(a, 64, 64), b"b" * 64)
        self.assertEqual(model.read(b, 64, 64), b"a" * 64)

    def test_source_and_destination_conflicts(self):
        model, source, target, window, payload = self.fixture(256)
        identity = model.submit(1, 1, source, window, 128)
        self.assert_code("REGION_BUSY", model.host_write, source, 64, bytes(64))
        self.assert_code("REGION_BUSY", model.host_write, target, 0, bytes(64))
        self.assert_code("REGION_BUSY", model.submit, 1, 2, source, window, 128,
                         target_offset=64)
        # Disjoint source ranges may be written without changing queued bytes.
        model.host_write(source, 128, bytes(64))
        self.complete(model, identity, 128)
        self.assertEqual(model.read(target, 0, 128), payload[:128])
        self.assert_code("REGION_BUSY", model.host_write, target, 0, bytes(64))
        model.retire(identity)
        model.host_write(target, 0, bytes(64))

    def test_unregister_closes_admission_but_allows_existing_drain(self):
        model, source, target, window, payload = self.fixture(128)
        identity = model.submit(1, 1, source, window, 64)
        for handle in (source, target):
            self.assert_code("REGION_BUSY", model.unregister, handle)
        self.assert_code("REGION_CLOSING", model.submit, 1, 2, source, window, 64,
                         target_offset=64)
        self.complete(model, identity, 64)
        self.assertEqual(model.debug_read_raw(target, 0, 64), payload[:64])
        model.retire(identity)
        model.unregister(source)
        model.unregister(target)
        self.assert_code("STALE_REGION", model.read, target, 0, 64)

    def test_window_cannot_rebind_until_retirement(self):
        model, source, target, window, _ = self.fixture(128)
        identity = model.submit(1, 1, source, window, 64)
        self.assert_code("WINDOW_BUSY", model.bind_window, target)
        self.complete(model, identity, 64)
        self.assert_code("WINDOW_BUSY", model.bind_window, target)
        model.retire(identity)
        new_window = model.bind_window(target)
        self.assertNotEqual(window, new_window)
        self.assert_code("STALE_WINDOW", model.submit, 1, 2, source, window, 64)
        model.submit(1, 2, source, new_window, 64)

    def test_region_generation_prevents_reused_slot_alias(self):
        model = TransportModel()
        old = model.register(0, 0, 64)
        model.host_write(old, 0, b"x" * 64)
        model.unregister(old)
        new = model.register(0, 0, 64)
        self.assertEqual(new.slot, old.slot)
        self.assertNotEqual(new.generation, old.generation)
        self.assert_code("STALE_REGION", model.host_write, old, 0, bytes(64))
        self.assertEqual(model.debug_read_raw(new, 0, 64), b"x" * 64)
        self.assert_code("INVALID_DATA", model.read, new, 0, 64)

    def test_high_addresses_and_capacity_boundary(self):
        for base in ((1 << 32) + 64, (512 << 30) - 128):
            with self.subTest(base=base):
                model, source, target, window, payload = self.fixture(128, base=base)
                identity = model.submit(1, 1, source, window, 128)
                self.complete(model, identity, 128)
                self.assertEqual(model.read(target, 0, 128), payload)
                self.assertEqual(model.resources()["materialized_bytes"], (128, 128))
                self.assert_code("OUT_OF_RANGE", model.register, 0, 512 << 30, 64)
                self.assert_code("OUT_OF_RANGE", model.register, 0, (512 << 30) - 64, 128)

    def test_bad_descriptors_have_no_destination_side_effects(self):
        model, source, target, window, _ = self.fixture(256)
        sentinel = b"z" * 256
        model.host_write(target, 0, sentinel)
        cases = [("INVALID_U64", dict(length=0)),
                 ("ALIGNMENT", dict(length=65)),
                 ("ALIGNMENT", dict(source_offset=1)),
                 ("OUT_OF_RANGE", dict(target_offset=256)),
                 ("INVALID_U64", dict(source_offset=U64_MAX + 1)),
                 ("OUT_OF_RANGE", dict(source_offset=U64_MAX)),
                 ("MODEL_EPOCH", dict(model_epoch=2))]
        for code, extra in cases:
            arguments = dict(length=64)
            arguments.update(extra)
            self.assert_code(code, model.submit, 1, 1, source, window, **arguments)
            self.assertEqual(model.read(target, 0, 256), sentinel)
            self.assertEqual(model.resources()["queue_used"], (0, 0))

    def test_window_extent_limits_target_access(self):
        model, source, target, _, _ = self.fixture(256)
        window = model.bind_window(target, offset=128, size=64)
        for offset in (0, 64, 192):
            self.assert_code("WINDOW_RANGE", model.submit, 1, 1, source, window, 64,
                             target_offset=offset)
        identity = model.submit(1, 1, source, window, 64, target_offset=128)
        self.complete(model, identity, 64)

    def test_permissions_uninitialized_data_and_wrong_owner(self):
        model, source, target, window, _ = self.fixture(64)
        unreadable = model.register(0, 128, 64, "w")
        self.assert_code("PERMISSIONS", model.submit, 1, 1, unreadable, window, 64)
        uninitialized = model.register(0, 256, 64)
        self.assert_code("INVALID_DATA", model.submit, 1, 1, uninitialized, window, 64)
        readonly = model.register(1, 128, 64, "r")
        readonly_window = model.bind_window(readonly)
        self.assert_code("PERMISSIONS", model.submit, 1, 1, source, readonly_window, 64)
        self.assert_code("PERMISSIONS", model.host_write, readonly, 0, bytes(64))
        self.assert_code("REMOTE_ONLY", model.submit, 1, 1, target, readonly_window, 64)
        self.assert_code("STALE_REGION", model.read, replace(source, generation=99), 0, 64)

    def test_replay_and_duplicate_packets_do_not_create_success(self):
        model, source, target, window, payload = self.fixture(128)
        identity = model.submit(1, 1, source, window, 128)
        self.assert_code("REPLAY", model.submit, 1, 1, source, window, 64)
        packet = model.send(identity, 0)
        self.assert_code("DUPLICATE_FRAGMENT", model.send, identity, 0)
        model.receive(packet)
        self.assert_code("DUPLICATE_RECEIVE", model.receive, packet)
        model.commit(packet)
        self.assert_code("UNKNOWN_PACKET", model.commit, packet)
        self.assert_code("NOT_VISIBLE", model.confirm_remote, identity)
        packet = model.send(identity, 1)
        model.receive(packet)
        model.commit(packet)
        model.confirm_remote(identity)
        self.assert_code("ALREADY_COMPLETE", model.confirm_remote, identity)
        model.retire(identity)
        self.assert_code("REPLAY", model.submit, 1, 1, source, window, 64)
        self.assertEqual(model.read(target, 0, 128), payload)

    def test_partial_failure_keeps_dirty_destination_quarantined(self):
        model, source, target, window, payload = self.fixture(192, credits=2)
        model.host_write(target, 0, b"z" * 192)
        identity = model.submit(1, 1, source, window, 192)
        first, late = model.send(identity, 0), model.send(identity, 1)
        model.receive(first)
        model.commit(first)
        barrier = model.fence(0)
        model.fail("TIMEOUT")
        result = model.completion(identity)
        self.assertEqual((result.status, result.valid_bytes), ("TIMEOUT", 0))
        self.assertEqual(model.debug_read_raw(target, 0, 192), payload[:64] + b"z" * 128)
        self.assert_code("REGION_BUSY", model.read, target, 0, 192)
        self.assert_code("LINK_DOWN", model.bind_window, target)
        model.retire(identity)
        self.assertEqual(model.fence_status(barrier), "FAILED")
        self.assert_code("INFLIGHT_NOT_DRAINED", model.recover)
        # An already issued write may arrive AFTER failure completion is retired.
        model.receive(late)
        model.commit(late)
        self.assertEqual(model.debug_read_raw(target, 0, 192), payload[:128] + b"z" * 64)
        self.assert_code("REGION_BUSY", model.unregister, target)
        self.assertFalse(any(event["kind"] == "SUCCESS" for event in model.events))
        model.recover()
        self.assert_code("STALE_SESSION", model.receive, late)
        new_target = model.register(1, 0, 192)
        self.assertEqual(model.debug_read_raw(new_target, 0, 192), payload[:128] + b"z" * 64)
        self.assert_code("INVALID_DATA", model.read, new_target, 0, 192)

    def test_revocation_is_explicit_before_recovery(self):
        model, source, target, window, _ = self.fixture(64)
        identity = model.submit(1, 1, source, window, 64)
        packet = model.send(identity, 0)
        self.assert_code("LINK_STILL_READY", model.revoke_inflight)
        model.fail()
        self.assert_code("INFLIGHT_NOT_DRAINED", model.recover)
        model.revoke_inflight()
        self.assert_code("UNKNOWN_PACKET", model.receive, packet)
        self.assert_code("COMPLETIONS_PENDING", model.recover)
        model.retire(identity)
        model.recover()
        self.assertEqual(model.session, 2)
        for operation in (model.completion, model.retire):
            self.assert_code("STALE_SESSION", operation, identity)
        self.assert_code("STALE_SESSION", model.submit, 1, 2, source, window, 64)
        self.assert_code("STALE_SESSION", model.read, target, 0, 64)

    def test_new_session_allows_new_ids_without_old_ack_alias(self):
        model, source, target, window, payload = self.fixture(64)
        old = model.submit(1, 1, source, window, 64)
        barrier = model.fence(0)
        model.fail()
        model.retire(old)
        model.recover()
        source, target = model.register(0, 0, 64), model.register(1, 0, 64)
        model.host_write(source, 0, payload)
        window = model.bind_window(target)
        new = model.submit(2, 1, source, window, 64)
        self.assertNotEqual(old, new)
        self.assert_code("STALE_SESSION", model.retire, old)
        self.assert_code("STALE_SESSION", model.fence_status, barrier)
        self.complete(model, new, 64)
        self.assertEqual(model.read(target, 0, 64), payload)

    def test_detected_corruption_aborts_without_committing_packet(self):
        model, source, target, window, _ = self.fixture(64, credits=1)
        identity = model.submit(1, 1, source, window, 64)
        packet = model.send(identity, 0)
        self.assert_code("INTEGRITY_ERROR", model.receive, packet, integrity_ok=False)
        self.assertEqual(model.completion(identity).valid_bytes, 0)
        self.assertEqual(model.debug_read_raw(target, 0, 64), bytes(64))
        self.assertEqual(model.resources()["credits"][1], 0)
        self.assert_code("LINK_DOWN", model.confirm_remote, identity)
        model.revoke_inflight()
        model.retire(identity)
        model.recover()

    def test_memory_budget_failure_can_follow_partial_commit(self):
        model = TransportModel(max_blocks=2)
        source = model.register(0, 0, 128)
        target = model.register(1, 0, 192)
        model.host_write(source, 0, b"x" * 128)
        model.host_write(target, 128, b"z" * 64)
        window = model.bind_window(target)
        identity = model.submit(1, 1, source, window, 128)
        first = model.send(identity, 0)
        model.receive(first)
        model.commit(first)
        second = model.send(identity, 1)
        model.receive(second)
        self.assert_code("MEMORY_BUDGET", model.commit, second)
        self.assertEqual(model.completion(identity).status, "MEMORY_ERROR")
        self.assertEqual(model.debug_read_raw(target, 0, 192), b"x" * 64 + bytes(64) + b"z" * 64)
        self.assertEqual(model.resources()["materialized_bytes"], (128, 128))

    def test_fragment_tail_and_limits(self):
        model, source, target, window, payload = self.fixture(192, fragment_bytes=128)
        identity = model.submit(1, 1, source, window, 192)
        self.assert_code("FRAGMENT_RANGE", model.send, identity, 2)
        self.assert_code("FRAGMENT_RANGE", model.send, identity, -1)
        self.complete(model, identity, 192, [1, 0])
        self.assertEqual(model.read(target, 0, 192), payload)
        model.retire(identity)
        self.assert_code("INVALID_U64", model.submit, 1, U64_MAX + 1, source, window, 64)

    def test_bounded_regions_and_sparse_memory(self):
        model = TransportModel(max_regions=1, max_blocks=1)
        whole_node = model.register(0, 0, 512 << 30)
        self.assertEqual(model.resources()["materialized_bytes"], (0, 0))
        self.assert_code("REGION_LIMIT", model.register, 0, 0, 64)
        model.host_write(whole_node, (512 << 30) - 64, b"v" * 64)
        self.assertEqual(model.read(whole_node, (512 << 30) - 64, 64), b"v" * 64)
        self.assert_code("MEMORY_BUDGET", model.host_write, whole_node, 0, bytes(64))
        self.assert_code("INVALID_DATA", model.read, whole_node, 0, 64)
        other = TransportModel()
        other.register(0, 64, 128)
        self.assert_code("OVERLAP", other.register, 0, 128, 128)

    def test_seeded_interleavings_compare_against_payload_oracle(self):
        # Independent byte oracle, repeated bidirectional fragments, bounded steps.
        for seed in range(20):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                model = TransportModel(credits=2, queue_depth=3)
                handles = [model.register(node, (1 << 32) + 64, 768) for node in (0, 1)]
                expected = [bytearray(768), bytearray(768)]
                for node in (0, 1):
                    expected[node][:384] = rng.randbytes(384)
                    model.host_write(handles[node], 0, bytes(expected[node][:384]))
                windows = [model.bind_window(handle) for handle in handles]
                work = []
                for node in (0, 1):
                    for sequence in range(1, 4):
                        offset = (sequence - 1) * 128
                        identity = model.submit(1, sequence, handles[node], windows[1-node], 128,
                                                source_offset=offset, target_offset=384 + offset)
                        expected[1-node][384+offset:512+offset] = expected[node][offset:128+offset]
                        work.append(identity)
                unsent = [PacketId(identity, index) for identity in work for index in (0, 1)]
                in_flight, received, committed = [], [], {identity: 0 for identity in work}
                remaining = set(work)
                for _ in range(500):
                    actions = []
                    for packet in unsent:
                        if model.resources()["credits"][1-packet.transfer.source]:
                            actions.append(("send", packet))
                    actions += [("receive", p) for p in in_flight]
                    actions += [("commit", p) for p in received]
                    actions += [("confirm", identity) for identity in
                                sorted(remaining, key=lambda key: (key.source, key.sequence))
                                if committed[identity] == 2]
                    if not actions:
                        break
                    action, item = rng.choice(actions)
                    if action == "send":
                        self.assertEqual(model.send(item.transfer, item.index), item)
                        unsent.remove(item)
                        in_flight.append(item)
                    elif action == "receive":
                        model.receive(item)
                        in_flight.remove(item)
                        received.append(item)
                    elif action == "commit":
                        model.commit(item)
                        received.remove(item)
                        committed[item.transfer] += 1
                    else:
                        model.confirm_remote(item)
                        model.retire(item)
                        remaining.remove(item)
                    credits = model.resources()["credits"]
                    for node in (0, 1):
                        held = sum(p.transfer.source == 1-node for p in in_flight + received)
                        self.assertEqual(credits[node] + held, 2)
                self.assertFalse(remaining, "scheduler stalled before all completions")
                self.assertEqual(model.resources()["queue_used"], (0, 0))
                for node in (0, 1):
                    self.assertEqual(model.read(handles[node], 0, 768), bytes(expected[node]))
                    self.assertEqual(model.fence_status(model.fence(node)), "SUCCESS")


if __name__ == "__main__":
    unittest.main()
