"""Exercise the binary boundary without NumPy, downloads or a GPU."""

import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

EXECUTABLE = sys.argv.pop(1)


def fixture():
    # H=2, D=32: gate=1, up=1/2, down=(a+a)/32 with zero padding.
    block = struct.pack("<e", 1 / 32) + bytes([1] * 32)
    gate = block * 2
    up = (struct.pack("<e", 1 / 64) + bytes([1] * 32)) * 2
    down = (struct.pack("<e", 1 / 32) + bytes([1, 1] + [0] * 30)) * 32
    expected = (0.5 / (1 + math.exp(-1))) / 16
    return (b"NTRFFN01" + struct.pack("<IIII", 32, 2, 32, 1) + gate + up + down
            + struct.pack("<32f", *([1] * 32)) + struct.pack("<32f", *([expected] * 32)) * 2)


class FixtureTests(unittest.TestCase):
    def invoke(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(data)
            return subprocess.run([EXECUTABLE, str(path)], capture_output=True, text=True)

    def test_valid(self):
        process = self.invoke(fixture())
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_malformed(self):
        valid = fixture()
        for data in (b"", b"BADMAGIC" + valid[8:], valid[:23], valid[:-1], valid + b"extra",
                     valid[:8] + struct.pack("<I", 0xFFFFFFFF) + valid[12:],
                     valid[:16] + struct.pack("<I", 64) + valid[20:]):
            with self.subTest(length=len(data), header=data[:24]):
                self.assertNotEqual(self.invoke(data).returncode, 0)

    def test_reference_rejected(self):
        valid = bytearray(fixture())
        # Last reference vector: mismatch and nonfinite must fail, even in Release.
        for value in (77.0, float("nan")):
            struct.pack_into("<f", valid, len(valid) - 4, value)
            self.assertNotEqual(self.invoke(valid).returncode, 0)


if __name__ == "__main__":
    unittest.main()
