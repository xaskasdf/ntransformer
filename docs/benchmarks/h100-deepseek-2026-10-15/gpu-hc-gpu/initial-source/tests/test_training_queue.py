"""CPU-only tests for the completed-pipeline gate."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from queue_brandon_training import gate_decision


class GateTests(unittest.TestCase):
    def setUp(self):
        self.summary = dict(status='complete', phase='pretrain', completed_steps=32768, planned_steps=32768)

    def test_pretrain_completion_does_not_interrupt_sft(self):
        for stage in ('pretrain', 'pretrain-diagnostic', 'sft', 'sft-diagnostic'):
            self.assertEqual(gate_decision(dict(state='running', stage=stage), self.summary, True)[0], 'waiting')

    def test_failed_pipeline_is_not_gpu_availability(self):
        self.assertEqual(gate_decision(dict(state='failed'), self.summary, True)[0], 'blocked')

    def test_require_consistent_completion_metadata(self):
        self.assertEqual(gate_decision(dict(state='complete'), None, True)[0], 'blocked')
        self.assertEqual(gate_decision(dict(state='complete'), self.summary, False)[0], 'blocked')
        for key, value in [('status', 'paused'), ('phase', 'sft'), ('completed_steps', 100), ('planned_steps', None)]:
            self.assertEqual(gate_decision(dict(state='complete'), dict(self.summary, **{key: value}), True)[0], 'blocked')

    def test_ready_still_requires_external_lock_and_idle_gpu_checks(self):
        self.assertEqual(gate_decision(dict(state='complete'), self.summary, True)[0], 'ready')

    def test_unknown_state_does_not_launch(self):
        self.assertEqual(gate_decision({}, self.summary, True)[0], 'waiting')


if __name__ == '__main__':
    unittest.main()
