"""Shared fill-generation regression against decoded RTL and preserved transaction oracle."""

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
           "fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv",
           "fpga/rtl/ntrpu_peer_multibuffer_decoded.sv",
           "fpga/rtl/ntrpu_peer_multibuffer_fillstage.sv",
           "fpga/tb/tb_ntrpu_peer_multibuffer.sv",
           "tools/check_ntrpu_peer_multibuffer.py",
           "tools/run_ntrpu_peer_multibuffer.py",
           "tools/run_ntrpu_multibuffer_fillstage.py")


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
    reg [63:0] expected_lease [0:CHANNELS*DESCRIPTORS-1];
    integer lease_slot, lease_port;
    integer fill_checks=0, descriptor_checks=0, return_checks=0, reset_checks=0;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (lease_slot=0; lease_slot<CHANNELS*DESCRIPTORS; lease_slot=lease_slot+1)
                expected_lease[lease_slot] = 0;
        end else if (cmd_valid && cmd_ready) begin
            expected_lease[cmd_channel*DESCRIPTORS+dut.tail[cmd_channel]] = dut.alloc_generation;
        end
        #1;
        if (!rst_n) begin
            if (dut.fill_generation !== 64'd0) $fatal(1, "fill lease reset mismatch");
            reset_checks = reset_checks + 1;
        end else begin
            if (dut.fill_active) begin
                if (dut.fill_generation !== expected_lease[dut.fill_channel*DESCRIPTORS+dut.fill_slot])
                    $fatal(1, "fill lease capture mismatch");
                fill_checks = fill_checks + 1;
            end
            for (lease_slot=0; lease_slot<CHANNELS*DESCRIPTORS; lease_slot=lease_slot+1) begin
                if (dut.state[lease_slot] == dut.READY || dut.state[lease_slot] == dut.WRITE) begin
                    if (dut.generation[lease_slot] !== expected_lease[lease_slot])
                        $fatal(1, "descriptor lease commit mismatch");
                    descriptor_checks = descriptor_checks + 1;
                end
            end
            for (lease_port=0; lease_port<CHANNELS; lease_port=lease_port+1) begin
                if (dut.active[lease_port]) begin
                    if (dut.active_generation[lease_port] !== expected_lease[lease_port*DESCRIPTORS+dut.send_head[lease_port]])
                        $fatal(1, "active return lease mismatch");
                    return_checks = return_checks + 1;
                end
            end
        end
    end
    final $display("FILL_MONITOR fill=%0d descriptor=%0d active=%0d reset=%0d", fill_checks, descriptor_checks, return_checks, reset_checks);
"""
    assert bench.count("endmodule") == 1
    main_start = bench.index('    initial begin\n        if(!$value$plusargs')
    main_end = bench.index('    initial begin #5000000;', main_start)
    init_end = bench.index('        $display("CASE traffic_and_isolation");', main_start)
    directed_main = bench[main_start:init_end] + """
        $display("DIRECTED reset_during_fill");
        send_job(0,1); drain_all;
        begin_partial(BANK_BEATS);
        if (BANK_BEATS > 1) begin
            data_valid=1; data=pattern(directed_tag,0);
            do @(posedge clk); while(!data_ready);
            @(negedge clk); data_valid=0;
        end
        repeat(5) @(negedge clk);
        if (!dut.fill_active) $fatal(1,"reset case did not interrupt a fill");
        rst_n=0;
        $display("DIRECTED RESET_ABORT %0d",cycle);
        repeat(3) @(negedge clk);
        if (!drained || banks_used != 0 || pending_counts != 0 || fault)
            $fatal(1,"reset left owned state");
        rst_n=1;
        repeat(8) begin
            @(negedge clk);
            if ((|m_awvalid) || (|m_wvalid) || (|cpl_valid)) $fatal(1,"aborted work escaped reset");
        end
        send_job(0,1); drain_all;
        $display("DIRECTED fault_during_paused_fill");
        cold_reset;
        error_tag=next_tag;
        send_job(0,1);
        begin_partial(BANK_BEATS);
        wait(fault);
        @(negedge clk);
        if (!dut.fill_active || !data_ready) $fatal(1,"fault case missed active fill");
        repeat(12) @(negedge clk);
        finish_partial(BANK_BEATS);
        drain_all;
        if (!fault) $fatal(1,"fault lost during drain");
        cold_reset;
        $display("DIRECTED stop_through_final_beat");
        begin_partial(BANK_BEATS);
        stop=1;
        repeat(12) @(negedge clk);
        finish_partial(BANK_BEATS);
        drain_all;
        probe_closed_admission;
        stop=0;
        send_job(0,1); drain_all;
        if (fault) $fatal(1,"post-reset recovery failed");
        $display("DIRECTED_PASS seed=%0d cycles=%0d",seed,cycle);
        $finish;
    end
"""
    directed_tasks = """
    reg [63:0] directed_tag;
    task begin_partial;
        input integer beats;
        begin
            directed_tag=next_tag; next_tag=next_tag+1;
            @(negedge clk);
            cmd_valid=1; cmd_channel=0; cmd_tag=directed_tag;
            cmd_addr=BASE+(directed_tag<<12); cmd_len=beats-1;
            do @(posedge clk); while(!cmd_ready);
            @(negedge clk); cmd_valid=0;
        end
    endtask
    task finish_partial;
        input integer beats;
        integer beat;
        begin
            for(beat=0;beat<beats;beat=beat+1) begin
                @(negedge clk); data_valid=1; data=pattern(directed_tag,beat);
                do @(posedge clk); while(!data_ready);
            end
            @(negedge clk); data_valid=0;
        end
    endtask
"""
    for variant in ("decoded", "fillstage"):
        specialized = bench.replace("ntrpu_peer_multibuffer #(", f"ntrpu_peer_multibuffer_{variant} #(")
        if variant == "fillstage":
            specialized = specialized.replace("endmodule", monitor + "endmodule")
        (output / (variant + "_tb.sv")).write_text(specialized)
        directed = (bench[:main_start] + directed_main + bench[main_end:]).replace(
            "ntrpu_peer_multibuffer #(", f"ntrpu_peer_multibuffer_{variant} #(")
        directed = directed.replace("endmodule", directed_tasks + (monitor if variant == "fillstage" else "") + "endmodule")
        (output / (variant + "_directed_tb.sv")).write_text(directed)
    report = dict(status="RUNNING", created_utc=datetime.now(timezone.utc).isoformat(),
                  input_sha256=hashes, runs=[], directed=[], mutations=[])

    def command(argv, label):
        start = time.monotonic()
        with (output / (label + ".stdout.log")).open("w") as stdout, (output / (label + ".stderr.log")).open("w") as stderr:
            run = subprocess.run(argv, cwd=output, stdout=stdout, stderr=stderr, timeout=45)
        save(output / (label + ".command.json"), dict(argv=argv, returncode=run.returncode,
                                                     seconds=time.monotonic() - start))
        return run.returncode, (output / (label + ".stdout.log")).read_text()

    def compile_rtl(variant, label, profile, mutant=None, directed=False):
        cmd = [args.iverilog] + (["-B", args.ivl_base] if args.ivl_base else [])
        top = "tb_ntrpu_peer_multibuffer"
        cmd += ["-g2012", "-Wall", "-s", top]
        keys = ("CHANNELS", "RESERVE_BANKS", "BANK_BEATS", "DESCRIPTORS", "MAX_OUTSTANDING")
        cmd += [f"-P{top}.{key}={value}" for key, value in zip(keys, profile)]
        exe = output / (label + ".vvp")
        files = [snapshot / SOURCES[0], snapshot / SOURCES[1],
                 mutant or snapshot / f"fpga/rtl/ntrpu_peer_multibuffer_{variant}.sv",
                 output / (variant + ("_directed_tb.sv" if directed else "_tb.sv"))]
        cmd += ["-o", str(exe)] + list(map(str, files))
        if command(cmd, "compile_" + label)[0]:
            raise RuntimeError("compile failed: " + label)
        return exe

    started = time.monotonic()
    try:
        traces = {}
        for variant in ("decoded", "fillstage"):
            for name, profile in profiles.items():
                label = variant + "_" + name
                exe = compile_rtl(variant, label, profile)
                for seed in (1, 7):
                    code, text = command([args.vvp, str(exe), f"+SEED={seed}"], f"{label}_seed{seed}")
                    if code:
                        raise RuntimeError("simulation failed: " + label)
                    result = replay(text, profile, seed)
                    trace = [line for line in text.splitlines() if line.startswith(("TRACE ", "CASE ", "PASS "))]
                    if variant == "decoded":
                        traces[name, seed] = trace
                    elif trace != traces[name, seed]:
                        raise RuntimeError("cycle trace mismatch: " + label)
                    match = re.search(r"FILL_MONITOR fill=(\d+) descriptor=(\d+) active=(\d+) reset=(\d+)", text)
                    monitors = dict(zip(("fill", "descriptor", "active", "reset"), map(int, match.groups()))) if match else {}
                    if variant == "fillstage" and (not monitors or not all(monitors.values())):
                        raise RuntimeError("lease invariant was not exercised: " + label)
                    report["runs"].append(dict(variant=variant, profile=name, seed=seed, **result,
                                               lease_checks=monitors, trace_lines=len(trace)))
        directed_traces = {}
        for variant in ("decoded", "fillstage"):
            for name in ("full4", "full8", "one_beat"):
                label = variant + "_directed_" + name
                exe = compile_rtl(variant, label, profiles[name], directed=True)
                for seed in (1, 7):
                    code, log = command([args.vvp, str(exe), f"+SEED={seed}"], f"{label}_seed{seed}")
                    if code or not re.search(rf"^DIRECTED_PASS seed={seed} cycles=\d+$", log, re.M) or "FATAL:" in log:
                        raise RuntimeError("directed scenario failed: " + label)
                    trace = [line for line in log.splitlines() if line.startswith(("TRACE ", "DIRECTED"))]
                    if variant == "decoded":
                        directed_traces[name, seed] = trace
                    elif trace != directed_traces[name, seed]:
                        raise RuntimeError("directed cycle trace mismatch: " + label)
                    report["directed"].append(dict(variant=variant, profile=name, seed=seed, trace_lines=len(trace)))
        source = (snapshot / SOURCES[3]).read_text()
        for name, old, new, reason in (
            ("lost_capture", "fill_generation <= alloc_generation;", "fill_generation <= 0;", "fill lease capture mismatch"),
            ("live_generation_commit", "generation[fill_channel*DESCRIPTORS+fill_slot] <= fill_generation;",
             "generation[fill_channel*DESCRIPTORS+fill_slot] <= alloc_generation;", "descriptor lease commit mismatch"),
            ("blocked_commit_on_stop", "generation[fill_channel*DESCRIPTORS+fill_slot] <= fill_generation;",
             "if (!stop) generation[fill_channel*DESCRIPTORS+fill_slot] <= fill_generation;", "descriptor lease commit mismatch"),
            ("blocked_commit_on_fault", "generation[fill_channel*DESCRIPTORS+fill_slot] <= fill_generation;",
             "if (!fault) generation[fill_channel*DESCRIPTORS+fill_slot] <= fill_generation;", "descriptor lease commit mismatch"),
            ("missing_reset", "fill_generation <= 0;", "fill_generation <= fill_generation;", "fill lease reset mismatch"),
        ):
            assert source.count(old) == 1
            mutant = output / (name + ".sv")
            mutant.write_text(source.replace(old, new))
            exe = compile_rtl("fillstage", name, profiles["full4"], mutant,
                              directed=name in ("blocked_commit_on_stop", "blocked_commit_on_fault"))
            code, text = command([args.vvp, str(exe), "+SEED=1"], name)
            if code == 0 or "FATAL:" not in text or reason not in text:
                raise RuntimeError("lease mutation escaped: " + name)
            report["mutations"].append(dict(name=name, returncode=code, reason=reason))
        report["status"] = "FILLSTAGE_RTL_CYCLE_EQUIVALENCE_PASS_NOT_INTEGRATED"
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

    if report["status"] != "FILLSTAGE_RTL_CYCLE_EQUIVALENCE_PASS_NOT_INTEGRATED":
        raise RuntimeError(report["status"])


if __name__ == "__main__":
    main()
