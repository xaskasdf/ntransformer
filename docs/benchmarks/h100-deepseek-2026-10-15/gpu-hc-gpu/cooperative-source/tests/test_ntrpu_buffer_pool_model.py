"""Scheduling/conservation tests only; these do not execute FPGA RTL."""

from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ntrpu_buffer_pool_model import Config, Job, Stalls, simulate


class BufferPoolTests(unittest.TestCase):
    def run_model(self, config, jobs, stalls=Stalls(), **kwargs):
        return simulate(config, jobs, stalls, audit=True, **kwargs)

    def test_one_frame_matches_hand_timeline(self):
        for beats in (1, 4, 64):
            for delay in (0, 1, 19):
                r = self.run_model(Config(channels=1, dedicated=1, reserve=0, b_delay=delay),
                                   [Job(0, 0, beats)])
                self.assertEqual(r["metrics"]["cycles"], 2 * beats + max(1, delay) + 1)
                t = r["trace"][0]
                self.assertEqual((t["admitted"], t["filled"], t["first_w"], t["last_w"]),
                                 (0, beats - 1, beats, 2 * beats - 1))
                self.assertEqual(t["bank_freed"], t["last_w"])
                self.assertEqual(t["b"], t["last_w"] + max(1, delay))

    def test_non_power_of_two_pool_wraps_without_loss(self):
        jobs = [Job(i, i % 2, (i % 4 + 1) * 4) for i in range(40)]
        r = self.run_model(Config(dedicated=0, reserve=3), jobs)
        self.assertTrue(r["metrics"]["drained"])
        self.assertEqual(r["metrics"]["payload_bytes"], sum(j.beats * 64 for j in jobs))
        self.assertEqual({x["bank"] for x in r["trace"]}, {0, 1, 2})
        self.assertTrue(all(x["received_beats"] == x["beats"] for x in r["trace"]))

    def test_more_banks_do_not_cure_one_outstanding_limit(self):
        jobs = [Job(i, 0, 4) for i in range(40)]
        c = Config(channels=1, dedicated=0, reserve=2, outstanding=1, b_delay=100)
        a = self.run_model(c, jobs)["metrics"]
        b = self.run_model(replace(c, reserve=32), jobs)["metrics"]
        faster = self.run_model(replace(c, reserve=2, outstanding=32), jobs)["metrics"]
        self.assertEqual(a["cycles"], b["cycles"])
        self.assertLess(faster["cycles"], a["cycles"] // 2)

    def test_early_release_needs_metadata_not_payload_until_b(self):
        jobs = [Job(i, 0, 4) for i in range(40)]
        c = Config(channels=1, dedicated=2, reserve=0, outstanding=32, b_delay=100)
        early = self.run_model(c, jobs)
        late = self.run_model(replace(c, release_on_b=True), jobs)
        self.assertLess(early["metrics"]["cycles"], late["metrics"]["cycles"])
        self.assertGreater(early["metrics"]["peak_outstanding_total"], 2)
        self.assertTrue(all(t["bank_freed"] < t["b"] for t in early["trace"]))
        self.assertTrue(all(t["bank_freed"] == t["b"] for t in late["trace"]))

    def test_ordered_ingress_blocks_other_destination(self):
        jobs = [Job(i, 0, 4) for i in range(8)] + [Job(i + 8, 1, 4) for i in range(8)]
        c = Config(reserve=0)
        stalled = Stalls(w=((0, 0, 10000),))
        ordered = self.run_model(replace(c, ingress="ordered"), jobs, stalled, max_cycles=200)
        eligible = self.run_model(c, jobs, stalled, max_cycles=200)
        self.assertEqual(ordered["metrics"]["completed_by_channel"][1], 0)
        self.assertEqual(eligible["metrics"]["completed_by_channel"][1], 8)
        self.assertFalse(eligible["metrics"]["drained"])

    def test_quota_prevents_one_destination_consuming_reserve(self):
        jobs = [Job(i, 0, 4) for i in range(20)]
        c = Config(reserve=4, reserve_quota=2)
        r = self.run_model(c, jobs, Stalls(w=((0, 0, 1000),)), max_cycles=100)
        self.assertEqual(r["metrics"]["admitted"], 4)
        self.assertEqual(r["metrics"]["peak_reserve"], 2)
        unbounded = self.run_model(replace(c, reserve_quota=None), jobs,
                                   Stalls(w=((0, 0, 1000),)), max_cycles=100)
        self.assertEqual(unbounded["metrics"]["admitted"], 6)

    def test_shared_read_port_limits_catch_up(self):
        jobs = [Job(i, i % 2, 4) for i in range(48)]
        c = Config(dedicated=0, reserve=48, outstanding=32, b_delay=1)
        stalled = Stalls(w=((-1, 0, 256),))
        one = self.run_model(replace(c, reserve_read_ports=1), jobs, stalled)["metrics"]
        two = self.run_model(c, jobs, stalled)["metrics"]
        self.assertGreater(one["cycles"], two["cycles"])
        self.assertGreater(one["counters"].get("read_port_stalled_channel_cycles", 0), 0)

    def test_result_reservations_bound_admission_and_resume(self):
        jobs = [Job(i, i % 2, 4) for i in range(20)]
        c = Config(result_slots=2, b_delay=1)
        stalls = Stalls(result=((0, 500),))
        held = self.run_model(c, jobs, stalls, max_cycles=200)
        self.assertEqual(held["metrics"]["admitted"], 4)
        self.assertEqual(held["metrics"]["completed"], 0)
        done = self.run_model(c, jobs, stalls)
        self.assertTrue(done["metrics"]["drained"])
        self.assertLessEqual(done["metrics"]["peak_result_reservations"], 4)

    def test_aw_and_b_pauses_are_independent_of_w(self):
        c = Config(channels=1, dedicated=2, reserve=2, outstanding=4, b_delay=1)
        jobs = [Job(i, 0, 4) for i in range(10)]
        r = self.run_model(c, jobs, Stalls(aw=((0, 0, 40),), b=((0, 0, 120),)))
        self.assertTrue(r["metrics"]["drained"])
        self.assertTrue(all(t["aw"] >= 40 and t["b"] >= 120 for t in r["trace"]))
        self.assertTrue(any(t["last_w"] < 120 for t in r["trace"]))

    def test_small_frames_consume_whole_bank(self):
        r = self.run_model(Config(), [Job(i, i % 2, 4) for i in range(12)])
        self.assertEqual(r["metrics"]["payload_capacity_fraction"], 256 / 4096)

    def test_admission_respects_availability_and_finite_budget(self):
        c = Config(channels=1)
        r = self.run_model(c, [Job(0, 0, 1, available=20)])
        self.assertEqual(r["trace"][0]["admitted"], 20)
        r = self.run_model(c, [Job(0, 0, 1, available=20)], max_cycles=10)
        self.assertFalse(r["metrics"]["drained"])
        self.assertEqual(r["metrics"]["admitted"], 0)

    def test_seeded_mixed_lengths_stalls_and_backpressure(self):
        for seed in range(12):
            rng = random.Random(seed)
            c = Config(dedicated=rng.choice((0, 2)), reserve=rng.choice((3, 4, 8)),
                       outstanding=rng.choice((1, 4, 8)), descriptors=8, result_slots=8,
                       b_delay=rng.randrange(1, 40), read_ports=rng.choice((1, 2)),
                       reserve_read_ports=rng.choice((1, 2)),
                       release_on_b=bool(seed % 2), ingress=rng.choice(("ordered", "eligible")))
            jobs = [Job(i, rng.randrange(2), rng.choice((4, 8, 16, 64))) for i in range(24)]
            stalls = Stalls(aw=((0, 100, 130),), b=((1, 180, 240),),
                            result=((250, 400),), w_period=97, w_closed=21)
            r = self.run_model(c, jobs, stalls, max_cycles=10000)
            self.assertTrue(r["metrics"]["drained"], seed)
            self.assertEqual(r["metrics"]["payload_bytes"], sum(j.beats * 64 for j in jobs))
            for p in range(2):
                writes = sorted((t for t in r["trace"] if t["channel"] == p), key=lambda t: t["first_w"])
                self.assertEqual([t["tag"] for t in writes], [j.tag for j in jobs if j.channel == p])

    def test_invalid_configuration_or_workload_fails(self):
        for kwargs in ({"dedicated": 0, "reserve": 0}, {"bank_bytes": 65},
                       {"outstanding": 0}, {"reserve_quota": -1}, {"ingress": "guess"}):
            with self.assertRaises(ValueError):
                Config(**kwargs)
        for jobs in ([], [Job(0, 4, 1)], [Job(0, 0, 65)], [Job(0, 0, 1)] * 2):
            with self.assertRaises(ValueError):
                self.run_model(Config(), jobs)


if __name__ == "__main__":
    unittest.main()
