"""NumPy tool checks; run separately so core CTest needs no NumPy dependency."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_ntrpu_brandon import BrandonWeights, quantize_q8, q8_gemv, swiglu


class BrandonToolTests(unittest.TestCase):
    def test_packing_ties_and_padding(self):
        w = np.zeros((1, 33), dtype=np.float32)
        w[0, :6] = [127, -127, 0.5, -0.5, 63.5, -63.5]
        packed, values, scales = quantize_q8(w)
        self.assertEqual(len(packed), 68)
        self.assertEqual(packed[:8], bytes([0, 60, 127, 129, 1, 255, 64, 192]))
        self.assertEqual(packed[34:], bytes(34))
        self.assertEqual(scales[0, 0], 1)
        self.assertEqual(int(values[0, 0, 3]), -1)

    def test_integer_reference(self):
        packed, values, scales = quantize_q8(np.full((2, 32), 127, dtype=np.float32))
        result = q8_gemv(values, scales, np.ones(32, dtype=np.float32))
        np.testing.assert_array_equal(result, [4064, 4064])
        np.testing.assert_array_equal(q8_gemv(values, scales, np.zeros(32)), [0, 0])

    def test_invalid_and_extremes(self):
        for array in (np.zeros(1), np.zeros((1, 0)), np.array([[np.nan]]), np.array([[np.inf]])):
            with self.assertRaises(ValueError):
                quantize_q8(array)
        np.testing.assert_allclose(swiglu([100, -100, 0, 1], [1, 1, 2, 2]),
                                   [100, 0, 0, 1.46211715726], rtol=1e-7, atol=1e-40)

    def test_revision_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned revision"):
                BrandonWeights(path)


if __name__ == "__main__":
    unittest.main()
