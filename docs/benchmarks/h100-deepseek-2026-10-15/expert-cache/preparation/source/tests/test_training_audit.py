"""CPU checks for the exact optimizer/RNG-state acceptance gate."""
import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_verified_instructions import exact_state


class StateAuditTests(unittest.TestCase):
    def setUp(self):
        self.state = [{'state': {0: {'exp_avg': torch.tensor([1., 2.]),
                                     'step': torch.tensor(100.)}},
                       'param_groups': [{'params': [0], 'lr': 5e-6}]}]

    def test_equal_state(self):
        self.assertTrue(exact_state(self.state, copy.deepcopy(self.state)))

    def test_optimizer_moment_change(self):
        changed = copy.deepcopy(self.state)
        changed[0]['state'][0]['exp_avg'][0] += 1e-6
        self.assertFalse(exact_state(self.state, changed))

    def test_optimizer_setting_change(self):
        changed = copy.deepcopy(self.state)
        changed[0]['param_groups'][0]['lr'] *= 2
        self.assertFalse(exact_state(self.state, changed))

    def test_dtype_shape_and_inventory_changes(self):
        value = torch.ones(2)
        for changed in (value.double(), value.reshape(1, 2), [value]):
            self.assertFalse(exact_state(value, changed))
        self.assertFalse(exact_state({'a': value}, {'b': value}))

    def test_nonfinite_state_rejected_even_when_equal(self):
        for value in (float('nan'), float('inf')):
            tensor = torch.tensor([value])
            self.assertFalse(exact_state(tensor, tensor.clone()))
            self.assertFalse(exact_state(value, value))

    def test_rng_bytes_change(self):
        self.assertTrue(exact_state(torch.tensor([1, 2], dtype=torch.uint8),
                                    torch.tensor([1, 2], dtype=torch.uint8)))
        self.assertFalse(exact_state(torch.tensor([1, 2], dtype=torch.uint8),
                                     torch.tensor([1, 3], dtype=torch.uint8)))


if __name__ == '__main__':
    unittest.main()
