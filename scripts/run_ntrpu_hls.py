"""Record and run the standalone NTRPU HLS flow; never reset an old project."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTBENCHES = {"gemv_int8": "test_ntrpu_hls.cpp", "gemv_q8_0": "test_ntrpu_hls_q8_0.cpp",
               "quantize_q8_0": "test_ntrpu_hls_quantize.cpp",
               "gemv_fp32_q8_0": "test_ntrpu_hls_connected.cpp",
               "swiglu": "test_ntrpu_hls_swiglu.cpp", "ffn_q8_0": "test_ntrpu_hls_ffn.cpp",
               "q8_block_mac": "test_ntrpu_hls_block_mac.cpp",
               "swiglu_q8_block": "test_ntrpu_hls_swiglu_block.cpp"}
DEPENDENCIES = {"gemv_fp32_q8_0": ("gemv_q8_0", "quantize_q8_0"),
                "ffn_q8_0": ("gemv_q8_0", "quantize_q8_0", "swiglu"),
                "swiglu_q8_block": ("quantize_q8_0", "swiglu")}
FIXTURE_FILES = {"gemv_fp32_q8_0": "brandon_tiles.bin", "ffn_q8_0": "brandon_ffn.bin"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", required=True, help="Exact FPGA part, no assumed board")
    parser.add_argument("--kernel", choices=tuple(TESTBENCHES), default="gemv_int8")
    parser.add_argument("--clock-ns", type=float, default=5.0)
    parser.add_argument("--stage", choices=("csim", "synth", "cosim", "export"), default="csim")
    parser.add_argument("--tool", help="Path to vitis-run; otherwise discover it on PATH")
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/ntrpu-hls",
                        help="Parent for a new isolated run; existing runs are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Snapshot inputs only; do not invoke Vitis")
    parser.add_argument("--fixture", type=Path, help="Optional local Brandon fixture for connected GEMV or FFN")
    parser.add_argument("--test-mode", choices=("full", "smoke"), default="full",
                        help="smoke runs only the small FFN regression; excludes maximum shape and real weights")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.part):
        parser.error("--part must be an FPGA part name")
    if not math.isfinite(args.clock_ns) or not 0 < args.clock_ns < 1000:
        parser.error("--clock-ns must be finite and between 0 and 1000")
    if args.fixture and (args.kernel not in FIXTURE_FILES or not args.fixture.is_file()):
        parser.error("--fixture requires gemv_fp32_q8_0 or ffn_q8_0 and an existing file")
    if args.test_mode == "smoke" and (args.kernel != "ffn_q8_0" or args.fixture):
        parser.error("--test-mode smoke requires ffn_q8_0 without --fixture")
    return args


def main():
    args = parse_args()
    tool = shutil.which(args.tool or "vitis-run")
    if not tool and not args.dry_run:
        print("vitis-run not found. Load the AMD tool environment or provide --tool. "
              "Use --dry-run to prepare sources without running HLS.", file=sys.stderr)
        return 2

    build = args.output_root.resolve()
    build.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(tempfile.mkdtemp(prefix=f"{stamp}-{args.stage}-", dir=build))
    source = run / "source"
    source.mkdir()
    top = "ntrpu_" + args.kernel
    paths = [ROOT / "fpga/hls" / args.kernel / name for name in
             (top + ".h", top + ".cpp", "run_hls.tcl")]
    bench = TESTBENCHES[args.kernel]
    paths.append(ROOT / "tests" / bench)
    for dependency in DEPENDENCIES.get(args.kernel, ()):
        paths.extend(ROOT / "fpga/hls" / dependency / ("ntrpu_" + dependency + suffix)
                     for suffix in (".h", ".cpp"))
    hashes = {}
    for path in paths:
        data = path.read_bytes()
        (source / path.name).write_bytes(data)
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()

    command = [str(Path(tool).resolve()) if tool else "vitis-run",
               "--mode", "hls", "--tcl", "source/run_hls.tcl"]
    settings = {"NTRPU_HLS_PART": args.part, "NTRPU_HLS_CLOCK_NS": str(args.clock_ns),
                "NTRPU_HLS_STAGE": args.stage, "NTRPU_HLS_FIXTURE": "0",
                "NTRPU_HLS_TEST_MODE": args.test_mode}
    fixture_hash = None
    if args.fixture:
        data = args.fixture.read_bytes()
        (source / FIXTURE_FILES[args.kernel]).write_bytes(data)
        fixture_hash = hashlib.sha256(data).hexdigest()
        settings["NTRPU_HLS_FIXTURE"] = "1"
    manifest = {"created_utc": stamp, "result": "dry_run" if args.dry_run else "prepared",
                "part": args.part, "clock_ns": args.clock_ns, "stage": args.stage,
                "flow": "vivado", "top": top, "kernel": args.kernel, "command": command,
                "test_mode": args.test_mode,
                "environment_overrides": settings, "source_sha256": hashes,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "fixture_sha256": fixture_hash,
                "math_backend": "amd_hls_math" if args.kernel in ("swiglu", "ffn_q8_0", "swiglu_q8_block") else None,
                "tool_executed": False, "returncode": None, "reports": []}

    def save():
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    print(f"Run directory: {run}", flush=True)
    if args.dry_run:
        print("Dry run: sources copied and hashed. No C simulation, RTL or synthesis results.")
        return 0

    environment = os.environ.copy()
    environment.update(settings)
    try:
        version = subprocess.run([command[0], "--version"], cwd=run, env=environment,
                                 capture_output=True, timeout=30, check=False)
        (run / "tool-version.log").write_bytes(version.stdout + version.stderr)
        manifest["version_returncode"] = version.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        manifest["version_error"] = str(error)
    manifest["result"] = "running"
    save()
    try:
        with (run / "tool.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=run, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT)
            manifest["tool_executed"] = True
            save()
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                manifest["result"] = "interrupted"
                manifest["returncode"] = process.returncode
                save()
                return 130
        manifest["returncode"] = returncode
        manifest["result"] = "tool_succeeded" if returncode == 0 else "failed"
        manifest["reports"] = sorted(path.relative_to(run).as_posix()
                                     for path in (run / "project").rglob("*")
                                     if path.is_file() and path.suffix in (".rpt", ".xml", ".log"))
        save()
        print(f"Tool exit: {returncode}; log: {run / 'tool.log'}")
        return returncode if 0 <= returncode <= 255 else 1
    except OSError as error:
        manifest["result"] = "failed"
        manifest["error"] = str(error)
        save()
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
