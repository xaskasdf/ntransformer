"""Bounded descriptor visibility tests; no RTL or PCIe execution."""
from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from ntrpu_buffer_pool_model import Config, Job, Stalls, simulate


class LookaheadTests(unittest.TestCase):
    def test_one_entry_matches_ordered_exactly(self):
        jobs = [Job(i, (i//3)%2, 1+i%8, available=(i//8)*100) for i in range(32)]
        stalls = Stalls(w=((0, 0, 180),), b=((1, 150, 300),))
        base = Config(reserve=4, descriptors=16, result_slots=16)
        a = simulate(replace(base, ingress='ordered'), jobs, stalls, audit=True)
        b = simulate(replace(base, ingress_window=1), jobs, stalls, audit=True)
        self.assertEqual(a, b)

    def test_small_window_cannot_see_past_blocked_prefix(self):
        jobs = [Job(i, int(i>=8), 4) for i in range(16)]
        stalls = Stalls(w=((0, 0, 10000),))
        small = simulate(Config(reserve=0, ingress_window=2), jobs, stalls, max_cycles=200, audit=True)
        large = simulate(Config(reserve=0, ingress_window=8), jobs, stalls, max_cycles=200, audit=True)
        self.assertEqual(small['metrics']['completed_by_channel'], [0,0])
        self.assertEqual(large['metrics']['completed_by_channel'], [0,8])
        self.assertFalse(large['metrics']['drained'])

    def test_window_and_destination_order_on_seeded_workloads(self):
        for window in (1,2,4,8):
            rng = random.Random(200+window)
            jobs = [Job(i, rng.randrange(2), rng.choice((1,8,64))) for i in range(32)]
            result = simulate(Config(ingress_window=window), jobs,
                              Stalls(w=((0,0,320),), b=((1,500,800),)), audit=True)
            self.assertTrue(result['metrics']['drained'])
            remaining = [j.tag for j in jobs]
            for row in result['trace']:
                self.assertIn(row['tag'], remaining[:window])
                remaining.remove(row['tag'])
            for channel in (0,1):
                self.assertEqual([r['tag'] for r in result['trace'] if r['channel']==channel],
                                 [j.tag for j in jobs if j.channel==channel])

    def test_full_window_matches_unbounded_queues(self):
        jobs = [Job(i, int(i%4==3), 8) for i in range(32)]
        stalls = Stalls(w=((0,0,300),))
        a = simulate(Config(), jobs, stalls, audit=True)
        b = simulate(Config(ingress_window=len(jobs)), jobs, stalls, audit=True)
        self.assertEqual(a,b)

    def test_invalid_window_rejected(self):
        for value in (0,-1,1.5,'8'):
            with self.assertRaises(ValueError):
                Config(ingress_window=value)
        with self.assertRaises(ValueError):
            Config(ingress='ordered', ingress_window=2)


if __name__ == '__main__':
    unittest.main()
