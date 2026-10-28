"""Check provenance and non-destructive HLS staging without claiming a tool run."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runner", ROOT / "scripts/run_ntrpu_hls.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_dry_run_preserves_sources_and_previous_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = ["fpga/hls/gemv_int8/ntrpu_gemv_int8.cpp", "fpga/hls/gemv_int8/ntrpu_gemv_int8.h",
                       "fpga/hls/gemv_int8/run_hls.tcl", "tests/test_ntrpu_hls.cpp"]
            for name in sources:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            args = ["runner", "--part", "xczu7ev-ffvc1156-2-e", "--stage", "cosim", "--dry-run"]
            with patch.object(runner, "ROOT", root), patch("sys.argv", args), \
                    patch.object(runner.shutil, "which", return_value=None), \
                    patch.object(runner.subprocess, "Popen") as process, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
                first = next((root / "build/ntrpu-hls").iterdir())
                manifest_bytes = (first / "manifest.json").read_bytes()
                self.assertEqual(runner.main(), 0)
                self.assertEqual((first / "manifest.json").read_bytes(), manifest_bytes)
                self.assertEqual(len(list((root / "build/ntrpu-hls").iterdir())), 2)
                process.assert_not_called()
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["result"], "dry_run")
            self.assertFalse(manifest["tool_executed"])
            self.assertIsNone(manifest["returncode"])
            self.assertEqual(manifest["reports"], [])
            for name in sources:
                expected = (root / name).read_bytes()
                self.assertEqual((first / "source" / Path(name).name).read_bytes(), expected)
                self.assertEqual(manifest["source_sha256"][name], hashlib.sha256(expected).hexdigest())

    def test_missing_tool_does_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "ROOT", Path(directory)), \
                patch("sys.argv", ["runner", "--part", "xczu7ev-ffvc1156-2-e"]), \
                patch.object(runner.shutil, "which", return_value=None), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(), 2)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_q8_kernel_stages_its_own_top_and_testbench(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["fpga/hls/gemv_q8_0/" + name for name in
                     ("ntrpu_gemv_q8_0.cpp", "ntrpu_gemv_q8_0.h", "run_hls.tcl")]
            names.append("tests/test_ntrpu_hls_q8_0.cpp")
            for name in names:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            args = ["runner", "--part", "xczu7ev-ffvc1156-2-e", "--kernel", "gemv_q8_0", "--dry-run"]
            with patch.object(runner, "ROOT", root), patch("sys.argv", args), \
                    patch.object(runner.shutil, "which", return_value=None), \
                    patch.object(runner.subprocess, "Popen") as process, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
                process.assert_not_called()
            run = next((root / "build/ntrpu-hls").iterdir())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["top"], "ntrpu_gemv_q8_0")
            self.assertEqual(set(manifest["source_sha256"]), set(names))
            for name in names:
                self.assertEqual((run / "source" / Path(name).name).read_bytes(), (ROOT / name).read_bytes())

    def test_invalid_parameters(self):
        for parameters in (["--part", "x;bad"], ["--part", "x", "--clock-ns", "nan"],
                           ["--part", "x", "--clock-ns", "0"], ["--part", "x", "--stage", "bad"],
                           ["--part", "x", "--test-mode", "smoke"],
                           ["--part", "x", "--kernel", "ffn_q8_0", "--test-mode", "smoke",
                            "--fixture", str(__file__)]):
            with patch("sys.argv", ["runner"] + parameters), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    runner.parse_args()
                self.assertNotEqual(caught.exception.code, 0)

    def test_ffn_smoke_records_reduced_scope_without_a_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "ROOT", root), patch("sys.argv", ["runner", "--part", "x",
                    "--kernel", "ffn_q8_0", "--test-mode", "smoke", "--dry-run"]), \
                    patch.object(runner.shutil, "which", return_value=None), \
                    patch.object(runner.Path, "read_bytes", return_value=b"provenance test"), \
                    patch.object(runner.subprocess, "Popen") as process, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
                process.assert_not_called()
            run = next((root / "build/ntrpu-hls").iterdir())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["test_mode"], "smoke")
            self.assertEqual(manifest["environment_overrides"]["NTRPU_HLS_TEST_MODE"], "smoke")
            self.assertEqual(manifest["environment_overrides"]["NTRPU_HLS_FIXTURE"], "0")
            self.assertIsNone(manifest["fixture_sha256"])

    def test_connected_kernel_snapshots_dependencies_and_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["fpga/hls/gemv_fp32_q8_0/" + name for name in
                     ("ntrpu_gemv_fp32_q8_0.cpp", "ntrpu_gemv_fp32_q8_0.h", "run_hls.tcl")]
            names.append("tests/test_ntrpu_hls_connected.cpp")
            for dependency in ("gemv_q8_0", "quantize_q8_0"):
                names.extend(f"fpga/hls/{dependency}/ntrpu_{dependency}{suffix}" for suffix in (".h", ".cpp"))
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            fixture = root / "external.bin"
            fixture.write_bytes(b"fixture snapshot test")
            args = ["runner", "--part", "xczu7ev-ffvc1156-2-e", "--kernel", "gemv_fp32_q8_0",
                    "--fixture", str(fixture), "--dry-run"]
            with patch.object(runner, "ROOT", root), patch("sys.argv", args), \
                    patch.object(runner.shutil, "which", return_value=None), \
                    patch.object(runner.subprocess, "Popen") as process, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
                process.assert_not_called()
            run = next((root / "build/ntrpu-hls").iterdir())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(set(manifest["source_sha256"]), set(names))
            self.assertEqual(manifest["fixture_sha256"], hashlib.sha256(fixture.read_bytes()).hexdigest())
            fixture.write_bytes(b"changed after staging")
            self.assertEqual((run / "source/brandon_tiles.bin").read_bytes(), b"fixture snapshot test")
            for name in names:
                self.assertEqual((run / "source" / Path(name).name).read_bytes(), (ROOT / name).read_bytes())

    def test_ffn_snapshots_activation_and_uses_distinct_fixture_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["fpga/hls/ffn_q8_0/" + name for name in
                     ("ntrpu_ffn_q8_0.cpp", "ntrpu_ffn_q8_0.h", "run_hls.tcl")]
            names.append("tests/test_ntrpu_hls_ffn.cpp")
            for dependency in ("gemv_q8_0", "quantize_q8_0", "swiglu"):
                names.extend(f"fpga/hls/{dependency}/ntrpu_{dependency}{suffix}" for suffix in (".h", ".cpp"))
            for name in names:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            fixture = root / "external.bin"
            fixture.write_bytes(b"FFN provenance test")
            with patch.object(runner, "ROOT", root), patch("sys.argv", ["runner", "--part", "test-part",
                    "--kernel", "ffn_q8_0", "--fixture", str(fixture), "--dry-run"]), \
                    patch.object(runner.shutil, "which", return_value=None), \
                    patch.object(runner.subprocess, "Popen") as process, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
                process.assert_not_called()
            run = next((root / "build/ntrpu-hls").iterdir())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(set(manifest["source_sha256"]), set(names))
            self.assertEqual(manifest["math_backend"], "amd_hls_math")
            self.assertEqual(manifest["fixture_sha256"], hashlib.sha256(fixture.read_bytes()).hexdigest())
            fixture.write_bytes(b"changed after snapshot")
            self.assertEqual((run / "source/brandon_ffn.bin").read_bytes(), b"FFN provenance test")
            for name in names:
                self.assertEqual(manifest["source_sha256"][name], hashlib.sha256((ROOT / name).read_bytes()).hexdigest())
                self.assertEqual((run / "source" / Path(name).name).read_bytes(), (ROOT / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
