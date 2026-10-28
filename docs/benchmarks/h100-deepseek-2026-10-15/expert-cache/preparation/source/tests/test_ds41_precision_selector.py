import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

import numpy as np

spec = importlib.util.spec_from_file_location('selector', Path(__file__).parents[1] / 'tools/select_ds41_precision.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class PrecisionSelection(unittest.TestCase):
    limits = dict(mean_kl=.005, p99_kl=.05, nll_increase=.01, relative_l2=None, speedup=1.03)

    def metric(self, **changes):
        return dict(mean_kl=.001, p99_kl=.003, nll_increase=.002, max_relative_l2=.004,
                    speedup=1.2, conservative_speedup=1.1) | changes

    def test_fast_but_inaccurate_candidate_is_rejected(self):
        selected, decisions = selector.choose({'f16': [self.metric()] * 2,
            'q8': [self.metric(speedup=2, nll_increase=.1)] * 2}, self.limits)
        self.assertEqual(selected, 'f16')
        self.assertFalse(decisions['q8']['eligible'])

    def test_validation_regression_overrides_calibration_gain(self):
        selected, _ = selector.choose({'q8': [self.metric(speedup=2), self.metric(p99_kl=.5)]}, self.limits)
        self.assertEqual(selected, 'fp32')

    def test_noise_or_missing_validation_cannot_select_lower_precision(self):
        for metrics in [[self.metric()], [self.metric(conservative_speedup=.99)] * 2,
                        [self.metric(mean_kl=float('nan'))] * 2]:
            selected, _ = selector.choose({'q8': metrics}, self.limits)
            self.assertEqual(selected, 'fp32')

    def test_raw_logit_distance_is_diagnostic_by_default(self):
        metrics = [self.metric(mean_kl=0, p99_kl=0, nll_increase=0, max_relative_l2=100)] * 2
        selected, _ = selector.choose({'f16': metrics}, self.limits)
        self.assertEqual(selected, 'f16')
        selected, _ = selector.choose({'f16': metrics}, self.limits | {'relative_l2': .01})
        self.assertEqual(selected, 'fp32')

    def case(self, path, logits):
        path.mkdir(exist_ok=True)
        setup = dict(kind='setup', teacher_forcing=True, prompt_tokens=2, decode_steps=2, vocab=3,
                     layers=1, shards=1, model_file_bytes=100, tokenizer_pre='test')
        rows = [setup]
        for index in range(4):
            rows += [dict(kind='phase', phase='decode', round=index, tokens=2, seconds=1),
                     dict(kind='completion', round=index, repeat_bitwise_equal=True)]
        (path / 'measurements.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        (path / 'prompt-ids.json').write_text('[0, 1]')
        (path / 'target-ids.json').write_text('[1, 2, 0]')
        np.asarray(logits, dtype='<f4').tofile(path / 'all-logits.f32')
        (path.parent / (path.name + '-command.json')).write_text(json.dumps(dict(returncode=0, environment={'DS41_VRAM_GB': '2'})))
        (path.parent / 'environment.json').write_text(json.dumps(dict(binary_sha256='fixture', gpu='fixture')))
        return selector.read_case(path)

    def test_comparison_invariant_to_constant_logit_offset_and_token_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self.case(root / 'reference', [[0, 1, 2]] * 3)
            candidate = self.case(root / 'candidate', [[100, 101, 102]] * 3)
            metrics = selector.compare(reference, candidate)
            self.assertAlmostEqual(metrics['mean_kl'], 0)
            self.assertAlmostEqual(metrics['nll_increase'], 0)
            self.assertFalse(metrics['bitwise_equal'])
            candidate['targets'][0] = 2
            with self.assertRaisesRegex(ValueError, 'different tokens'):
                selector.compare(reference, candidate)

    def test_mismatched_cache_and_nonfinite_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self.case(root / 'reference', [[0, 1, 2]] * 3)
            candidate = self.case(root / 'candidate', [[0, 1, 2]] * 3)
            candidate['command']['environment']['DS41_VRAM_GB'] = '62'
            with self.assertRaisesRegex(ValueError, 'runtime settings'):
                selector.compare(reference, candidate)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                self.case(root / 'nonfinite', [[0, float('nan'), 2]] * 3)


if __name__ == '__main__':
    unittest.main()
