"""CPU tests for elapsed-window accounting and the frozen baseline contract."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from ntransformer_training.measurement import file_sha256, validate_baseline, window_summary


class MeasurementTests(unittest.TestCase):
    def test_window_includes_logging_and_other_gaps(self):
        records = [dict(measured=False, tokens=100, seconds=99),
                   dict(measured=True, tokens=100, seconds=1),
                   dict(measured=True, tokens=100, seconds=3)]
        summary = window_summary(records, 5)
        self.assertEqual(summary['tokens_per_second'], 40)
        self.assertEqual(summary['mean_step_seconds'], 2.5)
        self.assertEqual(summary['instrumented_step_mean_seconds'], 2)

    def test_empty_or_zero_window_rejected(self):
        with self.assertRaises(ValueError):
            window_summary([], 1)
        with self.assertRaises(ValueError):
            window_summary([dict(measured=True, tokens=1, seconds=1)], 0)

    def test_preserved_snapshot_hashes(self):
        root = ROOT / 'docs/benchmarks/h100-training-2026-10-13/baseline-stage1-20261013-a'
        for name, expected in json.loads((root / 'SHA256SUMS.json').read_text()).items():
            self.assertEqual(file_sha256(root / name.replace('\\', '/')), expected, name)

    def test_baseline_contract_rejects_changes(self):
        root = ROOT / 'docs/benchmarks/h100-training-2026-10-13/baseline-stage1-20261013-a'
        path = root / 'baseline.json'
        baseline = json.loads(path.read_text())
        checkpoint = dict(config=copy.deepcopy(baseline['config']['model']),
                          settings=copy.deepcopy(baseline['config']['pretrain']),
                          manifest_sha256=baseline['captured_file_hashes']['data-manifest.json']['sha256'],
                          world_size=2)
        checkpoint['config']['norm_eps'] = 1e-5
        software = dict(torch='2.14.1+cu130', cuda_build='13.0', nccl=(2, 30, 7))
        validate_baseline(checkpoint, path, root / 'source', 107604288, software)
        for section, key, value in [('config', 'n_heads', 8), ('settings', 'batch_size', 8),
                                     ('settings', 'learning_rate', 1e-3)]:
            bad = copy.deepcopy(checkpoint)
            bad[section][key] = value
            with self.assertRaises(ValueError):
                validate_baseline(bad, path, root / 'source', 107604288, software)
        with self.assertRaises(ValueError):
            validate_baseline(checkpoint, path, root / 'source', 107604288, dict(software, nccl=(2, 29, 0)))
        with self.assertRaises(ValueError):
            validate_baseline(dict(checkpoint, manifest_sha256='changed'), path, root / 'source', 107604288, software)


if __name__ == '__main__':
    unittest.main()
