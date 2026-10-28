#!/usr/bin/env python3
"""Snapshot and validate NTRPU weight transport RTL with Vivado."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

DESIGNS = {
    "reader": {"top": "ntrpu_axi_read_master", "testbench": "tb_ntrpu_axi_read_master",
               "rtl": ["ntrpu_axi_read_master.sv"], "cases": 52,
               "marker": "NTRPU_AXI_READER_PASS", "metric": "bytes"},
    "tile_buffer": {"top": "ntrpu_weight_tile_buffer", "testbench": "tb_ntrpu_weight_tile_buffer",
                    "rtl": ["ntrpu_axi_read_master.sv", "ntrpu_weight_tile_buffer.sv"], "cases": 50,
                    "marker": "NTRPU_TILE_BUFFER_PASS", "metric": "tiles"},
    "q8_unpacker": {"top": "ntrpu_q8_unpacker", "testbench": "tb_ntrpu_q8_unpacker",
                    "rtl": ["ntrpu_q8_unpacker.sv"], "cases": 35,
                    "marker": "NTRPU_Q8_UNPACKER_PASS", "metric": "blocks"},
    "q8_stream": {"top": "ntrpu_q8_weight_stream", "testbench": "tb_ntrpu_q8_weight_stream",
                  "rtl": ["ntrpu_axi_read_master.sv", "ntrpu_weight_tile_buffer.sv",
                          "ntrpu_q8_unpacker.sv", "ntrpu_q8_weight_stream.sv"], "cases": 10,
                  "marker": "NTRPU_Q8_STREAM_PASS", "metric": "blocks"}
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=DESIGNS, default="reader")
    parser.add_argument("--output-root", type=Path, default=Path("build/ntrpu-axi"))
    parser.add_argument("--seed", type=int, action="append", help="repeat for multiple seeds")
    parser.add_argument("--synth", action="store_true", help="also run out-of-context synthesis")
    parser.add_argument("--part", default="xczu7ev-ffvc1156-2-e")
    args = parser.parse_args()
    design = DESIGNS[args.design]
    seeds = args.seed or [1, 7, 42]
    root = Path(__file__).resolve().parents[1]
    for tool in ["xvlog", "xelab", "xsim"] + (["vivado"] if args.synth else []):
        if not shutil.which(tool):
            parser.error(f"{tool} unavailable; source Vivado settings64.sh first")
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(tempfile.mkdtemp(prefix=stamp + "-", dir=args.output_root)).resolve()
    source = run / "source"
    source.mkdir()
    rtl_paths = ["fpga/rtl/" + name for name in design["rtl"]]
    paths = rtl_paths + [f"fpga/tb/{design['testbench']}.sv", "scripts/run_ntrpu_axi_reader.py"]
    hashes = {}
    for name in paths:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (run / "run.tcl").write_text("run all\nquit\n", encoding="utf-8")
    (run / "clock.xdc").write_text("create_clock -name clk -period 5.000 [get_ports clk]\n", encoding="utf-8")
    read_commands = "\n".join(f"read_verilog -sv source/{name}" for name in rtl_paths)
    (run / "synth.tcl").write_text(read_commands + "\n" + """read_xdc clock.xdc
synth_design -top @TOP@ -mode out_of_context -part $::env(NTRPU_PART)
report_utilization -file utilization.rpt
report_timing_summary -file timing_synthesis.rpt
write_checkpoint -force reader_synth.dcp
puts NTRPU_AXI_SYNTH_PASS
exit
""".replace("@TOP@", design["top"]), encoding="utf-8")
    manifest = {"run_id": run.name, "result": "running", "source_sha256": hashes,
                "seeds": seeds, "part": args.part, "synthesis_requested": args.synth,
                "design": args.design,
                "validation_scope": "RTL with synthetic memory; no HLS/DDR IP/PCIe integration",
                "commands": [], "simulations": []}

    def save():
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def invoke(command, log_name, timeout=180):
        manifest["commands"].append(command)
        save()
        print("Running:", " ".join(command), flush=True)
        import os
        env = os.environ.copy()
        env["NTRPU_PART"] = args.part
        with (run / log_name).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout, check=False)
        output = (run / log_name).read_text(encoding="utf-8", errors="replace")
        if result.returncode or re.search(r"(?im)^\s*(?:ERROR:|FATAL(?:_ERROR)?:|Fatal:|FAIL\b)", output):
            raise RuntimeError(f"{log_name} failed (exit {result.returncode}); inspect {run}")
        return output

    print(f"Run: {run}", flush=True)
    save()
    try:
        invoke(["xvlog", "--sv"] + ["source/" + name for name in paths if name.endswith(".sv")], "compile.log")
        invoke(["xelab", design["testbench"], "-s", "reader_tb"], "elaborate.log")
        for index, seed in enumerate(seeds):
            output = invoke(["xsim", "reader_tb", "-tclbatch", "run.tcl",
                             "-testplusarg", f"SEED={seed}", "-log", f"xsim-{index}.log"],
                            f"simulation-{index}.log")
            match = re.search(design["marker"] + r" cases=(\d+) bursts=(\d+) " + design["metric"] + r"=(\d+)", output)
            if not match or int(match[1]) != design["cases"]:
                raise RuntimeError(f"seed {seed}: missing complete {design['cases']}-case PASS marker")
            manifest["simulations"].append(dict(seed=seed, cases=int(match[1]),
                                               bursts=int(match[2]), **{design["metric"]: int(match[3])}))
            save()
        if args.synth:
            output = invoke(["vivado", "-mode", "batch", "-source", "synth.tcl"], "synthesis.log", 300)
            if not re.search(r"(?m)^NTRPU_AXI_SYNTH_PASS\s*$", output):
                raise RuntimeError("missing synthesis PASS marker")
        manifest["result"] = "passed"
    except Exception as exc:
        manifest["result"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
