"""Behavioral adapter tests. They do not execute AMD IP or RTL."""

import pathlib
import random
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from ntrpu_pcie_axi_model import PcieAxiModel, plan_writes
from ntrpu_transport_model import TransportError, U64_MAX


class AdapterTests(unittest.TestCase):
    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(TransportError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def fixture(self, length=256, *, base=0, **config):
        config.setdefault("fragment_bytes", 1024)
        model = PcieAxiModel(**config)
        source = model.register(0, 0, length)
        target = model.register(1, base, length)
        payload = bytes((i * 29 + 17) % 256 for i in range(length))
        model.host_write(source, 0, payload)
        window = model.bind_window(target)
        work = model.submit(1, 1, source, window, length)
        return model, source, target, work, payload

    def finish(self, model, burst, *, delivered=False, launched=False, w_first=False):
        if not delivered:
            model.deliver(burst)
        if not launched:
            model.launch(burst)
        beats = model.describe(burst)["length"] // 64
        if not w_first:
            model.accept_aw(burst)
        for index in range(beats):
            model.accept_w(burst, last=index == beats - 1)
        if w_first:
            model.accept_aw(burst)
        for _ in range(beats):
            model.memory_step(burst)
        model.accept_b(burst)

    def quiesce(self, model):
        for node in (0, 1):
            model.source_stop_ack(node)
            model.isolate_ingress_ack(node)

    def test_split_both_address_spaces_and_preserve_every_byte(self):
        spans = plan_writes(0x100000fc0, 0x200000f80, 768)
        self.assertEqual([span.length for span in spans], [64, 64, 256, 256, 128])
        offset = 0
        for span in spans:
            self.assertEqual((span.offset, span.pcie_address, span.axi_address),
                             (offset, 0x100000fc0 + offset, 0x200000f80 + offset))
            for address in (span.pcie_address, span.axi_address):
                self.assertEqual(address // 4096, (address + span.length - 1) // 4096)
            offset += span.length
        self.assertEqual(offset, 768)

    def test_planner_limits_and_overflow(self):
        self.assertEqual([span.length for span in plan_writes(0, 0, 320,
                         mps=4096, max_axi_beats=2)], [128, 128, 64])
        self.assertEqual(plan_writes(U64_MAX - 63, U64_MAX - 63, 64)[0].length, 64)
        for args in ((U64_MAX - 63, 0, 128), (0, U64_MAX - 63, 128)):
            self.assert_code("ADDRESS_OVERFLOW", plan_writes, *args)
        self.assert_code("INVALID_MPS", plan_writes, 0, 0, 64, mps=64)
        self.assert_code("AXI_BEAT_LIMIT", plan_writes, 0, 0, 64, max_axi_beats=257)
        self.assert_code("ALIGNMENT", plan_writes, 1, 0, 64)
        self.assert_code("PLAN_LIMIT", plan_writes, 0, 0, 1 << 40)

    def test_local_bridge_ack_does_not_confirm_remote_memory(self):
        model, _, target, work, _ = self.fixture(512)
        bursts = model.post(work, 0)
        for burst in bursts:
            model.acknowledge_posted(burst)
        self.assertEqual(model.debug_read_raw(target, 0, 512), bytes(512))
        self.assert_code("RESPONSES_PENDING", model.confirm_remote, work)
        self.assert_code("NOT_COMPLETE", model.completion, work)
        self.assert_code("DUPLICATE_LOCAL_ACK", model.acknowledge_posted, bursts[0])

    def test_aw_and_w_independent_and_final_b_required(self):
        model, _, target, work, payload = self.fixture(128)
        burst, = model.post(work, 0)
        model.deliver(burst)
        model.launch(burst)
        model.accept_w(burst, last=False)
        self.assert_code("AW_PENDING", model.memory_step, burst)
        self.assert_code("AW_W_PENDING", model.accept_b, burst)
        model.accept_w(burst, last=True)
        model.accept_aw(burst)
        for _ in range(2):
            model.memory_step(burst)
        self.assertEqual(model.debug_read_raw(target, 0, 128), payload)
        self.assert_code("RESPONSES_PENDING", model.confirm_remote, work)
        model.accept_b(burst)
        model.confirm_remote(work)
        self.assertEqual(model.read(target, 0, 128), payload)

    def test_nonbufferable_profile_rejects_early_response(self):
        model, _, target, work, payload = self.fixture(128)
        burst, = model.post(work, 0)
        model.deliver(burst)
        model.launch(burst)
        model.accept_aw(burst)
        for _ in range(2):
            model.accept_w(burst)
        self.assert_code("EARLY_BRESP", model.accept_b, burst)
        model.memory_step(burst)
        self.assertEqual(model.debug_read_raw(target, 0, 128), payload[:64] + bytes(64))
        self.assert_code("EARLY_BRESP", model.accept_b, burst)
        self.assert_code("REGION_BUSY", model.read, target, 0, 128)
        model.memory_step(burst)
        model.accept_b(burst)
        model.confirm_remote(work)
        self.assertEqual(model.read(target, 0, 128), payload)

    def test_high_address_translation_across_page(self):
        model, _, target, work, payload = self.fixture(768, base=0x100000fc0)
        bursts = model.post(work, 0)
        self.assertEqual([model.describe(b)["length"] for b in bursts], [64, 256, 256, 192])
        offset = 0
        for burst in bursts:
            descriptor = model.describe(burst)
            self.assertEqual(descriptor["axi_address"], 0x100000fc0 + offset)
            self.assertEqual(descriptor["pcie_address"], 0x200000000 + offset)
            self.assertEqual((descriptor["awcache"], descriptor["awid"],
                              descriptor["awsize"], descriptor["awburst"]), (0, 0, 6, "INCR"))
            self.assertEqual(descriptor["wstrb"], (1 << 64) - 1)
            offset += descriptor["length"]
            self.finish(model, burst, w_first=True)
        model.confirm_remote(work)
        self.assertEqual(model.read(target, 0, 768), payload)

    def test_aperture_maps_last_bytes_of_large_region(self):
        model = PcieAxiModel(fragment_bytes=256)
        source = model.register(0, 0, 256)
        target = model.register(1, 0, 512 << 30)
        model.host_write(source, 0, b"q" * 256)
        offset = (512 << 30) - 256
        window = model.bind_window(target, offset=offset, size=256)
        work = model.submit(1, 1, source, window, 256, target_offset=offset)
        burst, = model.post(work, 0)
        self.assertEqual(model.describe(burst)["pcie_address"], 0x200000000)
        self.assertEqual(model.describe(burst)["axi_address"], offset)
        self.finish(model, burst)
        model.confirm_remote(work)
        self.assertEqual(model.read(target, offset, 256), b"q" * 256)

    def test_bridge_backpressure_and_fifo_order(self):
        model, _, _, work, _ = self.fixture(512, bridge_depth=1)
        first, second = model.post(work, 0)
        self.assert_code("PCIE_ORDER", model.deliver, second)
        model.deliver(first)
        self.assert_code("BRIDGE_FULL", model.deliver, second)
        model.launch(first)
        self.assert_code("BRIDGE_FULL", model.deliver, second)
        self.finish(model, first, delivered=True, launched=True)
        self.finish(model, second)
        model.confirm_remote(work)

    def test_one_active_axi_burst_until_response(self):
        model, _, _, work, _ = self.fixture(512)
        first, second = model.post(work, 0)
        model.deliver(first)
        model.deliver(second)
        self.assert_code("AXI_ORDER", model.launch, second)
        model.launch(first)
        self.assert_code("AXI_BUSY", model.launch, second)
        self.finish(model, first, delivered=True, launched=True)
        self.finish(model, second, delivered=True)

    def test_fragment_credit_and_window_lease_have_different_lifetimes(self):
        model, _, target, work, _ = self.fixture(128, credits=1)
        burst, = model.post(work, 0)
        model.deliver(burst)
        model.launch(burst)
        model.accept_aw(burst)
        for _ in range(2):
            model.accept_w(burst)
            model.memory_step(burst)
        self.assertEqual(model.resources()["credits"], (1, 1))
        self.assert_code("WINDOW_BUSY", model.bind_window, target)
        self.assert_code("RESPONSES_PENDING", model.confirm_remote, work)
        model.accept_b(burst)
        model.confirm_remote(work)
        self.assert_code("WINDOW_BUSY", model.bind_window, target)
        model.retire(work)
        model.bind_window(target)

    def test_missing_logical_fragment_prevents_success(self):
        model, _, _, work, _ = self.fixture(256, fragment_bytes=128)
        first, = model.post(work, 0)
        self.finish(model, first)
        self.assert_code("NOT_VISIBLE", model.confirm_remote, work)
        second, = model.post(work, 1)
        self.finish(model, second)
        self.assertEqual(model.confirm_remote(work).valid_bytes, 256)

    def test_quiesce_requires_all_sources_and_ingress_gates(self):
        model, _, target, work, _ = self.fixture(512)
        first, late = model.post(work, 0)
        model.deliver(first)
        model.fail("TIMEOUT")
        model.retire(work)
        self.assert_code("SOURCE_STOP_PENDING", model.revoke_inflight)
        model.source_stop_ack(0)
        self.assert_code("SOURCE_STOP_PENDING", model.revoke_inflight)
        model.source_stop_ack(1)
        self.assert_code("INGRESS_ISOLATION_PENDING", model.revoke_inflight)
        model.isolate_ingress_ack(1)
        self.assert_code("INGRESS_ISOLATED", model.deliver, late)
        self.assert_code("INGRESS_ISOLATED", model.launch, first)
        model.isolate_ingress_ack(0)
        model.revoke_inflight()
        model.recover()
        self.assert_code("STALE_SESSION", model.deliver, late)
        self.assert_code("STALE_SESSION", model.read, target, 0, 64)

    def test_launched_axi_write_cannot_be_discarded_by_revocation(self):
        model, _, target, work, payload = self.fixture(256)
        burst, = model.post(work, 0)
        model.deliver(burst)
        model.launch(burst)
        model.accept_aw(burst)
        model.accept_w(burst)
        model.memory_step(burst)
        model.fail("TIMEOUT")
        self.assertEqual(model.completion(work).valid_bytes, 0)
        model.retire(work)
        self.quiesce(model)
        self.assert_code("AXI_DRAIN_PENDING", model.revoke_inflight)
        self.assertEqual(model.debug_read_raw(target, 0, 256), payload[:64] + bytes(192))
        for _ in range(3):
            model.accept_w(burst)
            model.memory_step(burst)
        self.assert_code("AXI_DRAIN_PENDING", model.recover)
        model.accept_b(burst)
        self.assertEqual(model.debug_read_raw(target, 0, 256), payload)
        self.assert_code("REGION_BUSY", model.read, target, 0, 256)
        model.recover()
        fresh = model.register(1, 0, 256)
        self.assertEqual(model.debug_read_raw(fresh, 0, 256), payload)
        self.assert_code("INVALID_DATA", model.read, fresh, 0, 256)

    def test_launch_before_any_handshake_still_requires_drain(self):
        model, _, _, work, _ = self.fixture(64)
        burst, = model.post(work, 0)
        model.deliver(burst)
        model.launch(burst)
        model.fail()
        self.quiesce(model)
        self.assert_code("AXI_DRAIN_PENDING", model.revoke_inflight)
        self.finish(model, burst, delivered=True, launched=True, w_first=True)
        model.revoke_inflight()
        self.assert_code("COMPLETIONS_PENDING", model.recover)
        model.retire(work)
        model.recover()

    def test_axi_errors_leave_partial_data_unpublished(self):
        for response in ("SLVERR", "DECERR"):
            with self.subTest(response=response):
                model, _, target, work, payload = self.fixture(128)
                burst, = model.post(work, 0)
                model.deliver(burst)
                model.launch(burst)
                model.accept_aw(burst)
                for _ in range(2):
                    model.accept_w(burst)
                model.memory_step(burst)
                model.accept_b(burst, response)
                self.assertEqual(model.completion(work).status, "MEMORY_ERROR")
                self.assertEqual(model.debug_read_raw(target, 0, 128), payload[:64] + bytes(64))
                self.assert_code("NOT_ACTIVE", model.memory_step, burst)
                self.assert_code("RESPONSES_PENDING", model.confirm_remote, work)
                self.quiesce(model)
                model.retire(work)
                model.recover()

    def test_channel_duplicates_and_invalid_last_are_rejected(self):
        model, _, _, work, _ = self.fixture(128)
        burst, = model.post(work, 0)
        model.deliver(burst)
        self.assert_code("DUPLICATE_DELIVERY", model.deliver, burst)
        model.launch(burst)
        model.accept_aw(burst)
        self.assert_code("DUPLICATE_AW", model.accept_aw, burst)
        self.assert_code("WLAST", model.accept_w, burst, last=True)
        model.accept_w(burst, last=False)
        self.assert_code("WLAST", model.accept_w, burst, last=False)
        model.accept_w(burst, last=True)
        self.assert_code("EXTRA_W", model.accept_w, burst)
        self.assert_code("INVALID_BRESP", model.accept_b, burst, "EXOKAY")

    def test_generic_api_cannot_skip_adapter_stages(self):
        model, _, _, work, _ = self.fixture(64)
        self.assert_code("USE_POST", model.send, work, 0)
        burst, = model.post(work, 0)
        self.assert_code("USE_DELIVER", model.receive, burst.packet)
        self.assert_code("USE_MEMORY_STEP", model.commit, burst.packet)

    def test_seeded_bidirectional_channel_interleavings(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                model = PcieAxiModel(fragment_bytes=256, mps=128, credits=1, bridge_depth=2)
                regions = [model.register(node, (1 << 32) + 64, 512) for node in (0, 1)]
                payloads = [rng.randbytes(256), rng.randbytes(256)]
                for node in (0, 1):
                    model.host_write(regions[node], 0, payloads[node])
                windows = [model.bind_window(region) for region in regions]
                work = [model.submit(1, 1, regions[node], windows[1-node], 256,
                                     target_offset=256) for node in (0, 1)]
                bursts = [model.post(identity, 0) for identity in work]
                delivered, finished, launched = [0, 0], [0, 0], [0, 0]
                active = [None, None]
                aw, w, visible = [False, False], [0, 0], [0, 0]
                for _ in range(1000):
                    actions = []
                    for source in (0, 1):
                        if delivered[source] < len(bursts[source]) and delivered[source] - finished[source] < 2:
                            actions.append(("deliver", source))
                        if active[source] is None and launched[source] < delivered[source]:
                            actions.append(("launch", source))
                        if active[source] is not None:
                            count = model.describe(active[source])["length"] // 64
                            if not aw[source]:
                                actions.append(("aw", source))
                            if w[source] < count:
                                actions.append(("w", source))
                            if aw[source] and visible[source] < w[source]:
                                actions.append(("memory", source))
                            if aw[source] and visible[source] == count:
                                actions.append(("b", source))
                    if not actions:
                        break
                    if rng.random() < 0.25:  # Explicit pause with no handshake.
                        continue
                    action, source = rng.choice(actions)
                    if action == "deliver":
                        model.deliver(bursts[source][delivered[source]])
                        delivered[source] += 1
                    elif action == "launch":
                        active[source] = bursts[source][launched[source]]
                        model.launch(active[source])
                        launched[source] += 1
                        aw[source], w[source], visible[source] = False, 0, 0
                    elif action == "aw":
                        model.accept_aw(active[source])
                        aw[source] = True
                    elif action == "w":
                        model.accept_w(active[source])
                        w[source] += 1
                    elif action == "memory":
                        model.memory_step(active[source])
                        visible[source] += 1
                    else:
                        model.accept_b(active[source])
                        active[source] = None
                        finished[source] += 1
                self.assertEqual(finished, [len(row) for row in bursts], "no forward progress")
                for source in (0, 1):
                    model.confirm_remote(work[source])
                    self.assertEqual(model.read(regions[1-source], 256, 256), payloads[source])
                    model.retire(work[source])
                self.assertEqual(model.resources()["credits"], (1, 1))
                self.assertEqual(model.resources()["queue_used"], (0, 0))
                self.assertEqual(model.resources()["axi_active"], (False, False))


if __name__ == "__main__":
    unittest.main()
