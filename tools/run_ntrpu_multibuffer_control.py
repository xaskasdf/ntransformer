"""Isolated cached-return regression against the compact RTL and existing oracle."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("fpga/rtl/ntrpu_buffer_pool_allocator.sv",
           "fpga/rtl/ntrpu_buffer_pool_allocator_parallel.sv",
           "fpga/rtl/ntrpu_peer_multibuffer_compact.sv",
           "fpga/rtl/ntrpu_peer_multibuffer_control.sv",
           "fpga/tb/tb_ntrpu_peer_multibuffer.sv",
           "tools/check_ntrpu_peer_multibuffer.py",
           "tools/run_ntrpu_peer_multibuffer.py",
           "tools/run_ntrpu_multibuffer_control.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--ivl-base")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "inputs"
    for name in SOURCES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    hashes = {name: digest(snapshot / name) for name in SOURCES}
    sys.path.insert(0, str(snapshot / "tools"))
    from check_ntrpu_peer_multibuffer import replay
    from run_ntrpu_peer_multibuffer import PROFILES
    profiles = {**PROFILES, "odd_beats": (2, 4, 3, 16, 4)}
    bench = (snapshot / SOURCES[4]).read_text()
    monitor = """
    genvar cached_port;
    generate for (cached_port=0; cached_port<CHANNELS; cached_port=cached_port+1) begin: cache_monitor
        integer checks = 0;
        always @(posedge clk) begin
            #1;
            if (rst_n && dut.active[cached_port]) begin
                if (dut.active_bank[cached_port] !== dut.bank[cached_port*DESCRIPTORS+dut.send_head[cached_port]] ||
                    dut.active_generation[cached_port] !== dut.generation[cached_port*DESCRIPTORS+dut.send_head[cached_port]])
                    $fatal(1, "cached return lease mismatch port=%0d", cached_port);
                checks = checks + 1;
            end
        end
        final $display("CACHE_MONITOR port=%0d checks=%0d", cached_port, checks);
    end endgenerate
"""
    assert bench.count("endmodule") == 1
    for variant in ("compact", "control"):
        specialized = bench.replace("ntrpu_peer_multibuffer #(", f"ntrpu_peer_multibuffer_{variant} #(")
        if variant == "control":
            specialized = specialized.replace("endmodule", monitor + "endmodule")
        (output / (variant + "_tb.sv")).write_text(specialized)
    report = dict(status="RUNNING", created_utc=datetime.now(timezone.utc).isoformat(),
                  input_sha256=hashes, runs=[], mutations=[])

    def command(argv, label):
        start = time.monotonic()
        with (output / (label + ".stdout.log")).open("w") as stdout, (output / (label + ".stderr.log")).open("w") as stderr:
            run = subprocess.run(argv, cwd=output, stdout=stdout, stderr=stderr, timeout=45)
        save(output / (label + ".command.json"), dict(argv=argv, returncode=run.returncode,
                                                     seconds=time.monotonic() - start))
        return run.returncode, (output / (label + ".stdout.log")).read_text()

    def compile_rtl(variant, label, profile, mutant=None):
        cmd = [args.iverilog] + (["-B", args.ivl_base] if args.ivl_base else [])
        top = "tb_ntrpu_peer_multibuffer"
        cmd += ["-g2012", "-Wall", "-s", top]
        keys = ("CHANNELS", "RESERVE_BANKS", "BANK_BEATS", "DESCRIPTORS", "MAX_OUTSTANDING")
        cmd += [f"-P{top}.{key}={value}" for key, value in zip(keys, profile)]
        exe = output / (label + ".vvp")
        files = [snapshot / SOURCES[0], snapshot / SOURCES[1],
                 mutant or snapshot / f"fpga/rtl/ntrpu_peer_multibuffer_{variant}.sv",
                 output / (variant + "_tb.sv")]
        cmd += ["-o", str(exe)] + list(map(str, files))
        if command(cmd, "compile_" + label)[0]:
            raise RuntimeError("compile failed: " + label)
        return exe

    started = time.monotonic()
    try:
        traces = {}
        for variant in ("compact", "control"):
            for name, profile in profiles.items():
                label = variant + "_" + name
                exe = compile_rtl(variant, label, profile)
                for seed in (1, 7):
                    code, text = command([args.vvp, str(exe), f"+SEED={seed}"], f"{label}_seed{seed}")
                    if code:
                        raise RuntimeError("simulation failed: " + label)
                    result = replay(text, profile, seed)
                    trace = [line for line in text.splitlines() if line.startswith(("TRACE ", "CASE ", "PASS "))]
                    if variant == "compact":
                        traces[name, seed] = trace
                    elif trace != traces[name, seed]:
                        raise RuntimeError("cycle trace mismatch: " + label)
                    monitors = {int(p): int(n) for p, n in re.findall(r"CACHE_MONITOR port=(\d+) checks=(\d+)", text)}
                    if variant == "control" and (set(monitors) != set(range(profile[0])) or sum(monitors.values()) == 0):
                        raise RuntimeError("cache invariant was not exercised: " + label)
                    report["runs"].append(dict(variant=variant, profile=name, seed=seed, **result,
                                               cache_checks=monitors, trace_lines=len(trace)))
        source = (snapshot / SOURCES[3]).read_text()
        for name, old, new in (
            ("wrong_cached_bank", "active_bank[i] <= bank[i*DESCRIPTORS+send_head[i]];",
             "active_bank[i] <= bank[i*DESCRIPTORS+send_head[i]] ^ 1'b1;"),
            ("stale_cached_generation", "active_generation[i] <= generation[i*DESCRIPTORS+send_head[i]];",
             "active_generation[i] <= 0;"),
        ):
            assert source.count(old) == 1
            mutant = output / (name + ".sv")
            mutant.write_text(source.replace(old, new))
            exe = compile_rtl("control", name, profiles["full4"], mutant)
            code, text = command([args.vvp, str(exe), "+SEED=1"], name)
            if code == 0 or "FATAL:" not in text or "cached return lease mismatch" not in text:
                raise RuntimeError("cache mutation escaped: " + name)
            report["mutations"].append(dict(name=name, returncode=code, reason="cached return lease mismatch"))
        report["status"] = "CONTROL_RTL_CYCLE_EQUIVALENCE_PASS_NOT_INTEGRATED"
    except Exception as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["sources_after"] = {name: digest(ROOT / name) for name in SOURCES}
        if hashes != report["sources_after"]:
            report["status"] = "SOURCES_CHANGED"
        save(output / "REVIEW.json", report)
        manifest = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*") if p.is_file()}
        save(output / "FILES.json", manifest)
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
            for p in output.rglob("*"):
                if p.is_file():
                    bundle.write(p, p.relative_to(output).as_posix())
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            assert all(hashlib.sha256(bundle.read(n)).hexdigest() == h for n, h in manifest.items())
        save(output / "VERIFIED.json", dict(sha256=digest(archive), crc="PASS", hashes="PASS"))
        print(json.dumps({k: report[k] for k in ("status", "seconds")}))


if __name__ == "__main__":
    main()
