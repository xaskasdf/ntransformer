"""Isolated local regression and ablations for compact RAM / parallel allocator."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "fpga/rtl/ntrpu_buffer_pool_allocator.sv", "fpga/rtl/ntrpu_buffer_pool_allocator_parallel.sv",
    "fpga/rtl/ntrpu_peer_multibuffer.sv", "fpga/rtl/ntrpu_peer_multibuffer_compact.sv",
    "fpga/tb/tb_ntrpu_buffer_pool_allocator.sv", "fpga/tb/tb_ntrpu_peer_multibuffer.sv",
    "tools/check_ntrpu_buffer_pool_allocator.py", "tools/check_ntrpu_peer_multibuffer.py",
    "tools/run_ntrpu_buffer_pool_allocator.py", "tools/run_ntrpu_peer_multibuffer.py",
    "tools/run_ntrpu_multibuffer_compact.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("--iverilog", default="iverilog")
    ap.add_argument("--vvp", default="vvp")
    ap.add_argument("--ivl-base")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snap = out / "inputs"
    for name in SOURCES:
        dest = snap / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    hashes = {name: digest(snap / name) for name in SOURCES}
    sys.path.insert(0, str(snap / "tools"))
    from check_ntrpu_buffer_pool_allocator import replay as allocator_replay
    from check_ntrpu_peer_multibuffer import replay as datapath_replay
    from run_ntrpu_buffer_pool_allocator import PROFILES as allocator_profiles
    from run_ntrpu_peer_multibuffer import PROFILES as datapath_profiles
    datapath_profiles = {**datapath_profiles, "odd_beats": (2, 4, 3, 16, 4)}
    alloc_tb = out / "allocator_tb.sv"
    alloc_tb.write_text((snap / SOURCES[4]).read_text().replace(
        "ntrpu_buffer_pool_allocator #(", "ntrpu_buffer_pool_allocator_parallel #("))
    data_tb = out / "datapath_tb.sv"
    data_tb.write_text((snap / SOURCES[5]).read_text().replace(
        "ntrpu_peer_multibuffer #(", "ntrpu_peer_multibuffer_compact #("))
    report = dict(status="RUNNING", created_utc=datetime.now(timezone.utc).isoformat(),
                  input_sha256=hashes, allocator=[], datapath=[], mutations=[])
    start = time.monotonic()

    def command(argv, name):
        began = time.monotonic()
        with (out / (name + ".stdout.log")).open("w") as stdout, (out / (name + ".stderr.log")).open("w") as stderr:
            process = subprocess.run(argv, cwd=out, stdout=stdout, stderr=stderr, timeout=45)
        save(out / (name + ".command.json"), dict(argv=list(map(str, argv)), returncode=process.returncode,
                                                seconds=time.monotonic()-began))
        return process.returncode, (out / (name + ".stdout.log")).read_text()

    def compile_rtl(name, top, profile, allocator=False, mode="compact", mutant=None):
        cmd = [args.iverilog]
        if args.ivl_base:
            cmd += ["-B", args.ivl_base]
        cmd += ["-g2012", "-Wall", "-s", top]
        keys = (("CHANNELS", "DEDICATED_BANKS", "RESERVE_BANKS", "RESERVE_QUOTA", "GENERATION_BITS")
                if allocator else ("CHANNELS", "RESERVE_BANKS", "BANK_BEATS", "DESCRIPTORS", "MAX_OUTSTANDING"))
        cmd += [f"-P{top}.{key}={value}" for key, value in zip(keys, profile)]
        if allocator:
            files = [mutant or snap / SOURCES[1], alloc_tb]
        elif mode == "reference":
            files = [snap / SOURCES[0], snap / SOURCES[2], snap / SOURCES[5]]
        else:
            if mode == "ram_only":
                cmd += ["-DNTRPU_REFERENCE_ALLOCATOR"]
            if mode == "allocator_only":
                # Override the DUT parameter through a specialized bench copy.
                bench = out / "unpacked_tb.sv"
                if not bench.exists():
                    bench.write_text(data_tb.read_text().replace("#(.CHANNELS", "#(.PACKED_RAM(0),.CHANNELS", 1))
            else:
                bench = data_tb
            files = [snap / SOURCES[0], snap / SOURCES[1], mutant or snap / SOURCES[3], bench]
        exe = out / (name + ".vvp")
        cmd += ["-o", str(exe)] + list(map(str, files))
        if command(cmd, "compile_" + name)[0]:
            raise RuntimeError("compile failed: " + name)
        return exe

    def trace(text):
        return [line for line in text.splitlines() if line.startswith(("TRACE ", "CASE ", "PASS "))]

    try:
        for name, profile in allocator_profiles.items():
            exe = compile_rtl("alloc_"+name, "tb_ntrpu_buffer_pool_allocator", profile, allocator=True)
            for seed in (1, 7, 20261004):
                label = f"alloc_{name}_seed{seed}"
                code, log = command([args.vvp, str(exe), f"+SEED={seed}"], label)
                if code:
                    raise RuntimeError("simulation failed: " + label)
                report["allocator"].append(dict(name=name, **allocator_replay(log, profile, seed)))
        reference_traces = {}
        for mode in ("compact", "ram_only", "allocator_only", "reference"):
            profiles = datapath_profiles if mode == "compact" else {k: datapath_profiles[k] for k in ("full4", "full8")}
            for name, profile in profiles.items():
                label = mode + "_" + name
                exe = compile_rtl(label, "tb_ntrpu_peer_multibuffer", profile, mode=mode)
                for seed in (1, 7):
                    code, log = command([args.vvp, str(exe), f"+SEED={seed}"], f"{label}_seed{seed}")
                    if code:
                        raise RuntimeError(f"simulation failed: {label}, seed {seed}")
                    result = datapath_replay(log, profile, seed)
                    if name in ("full4", "full8"):
                        key = name, seed
                        if mode == "compact":
                            reference_traces[key] = trace(log)
                        elif trace(log) != reference_traces[key]:
                            raise RuntimeError(f"cycle trace differs: {label}, seed {seed}")
                    report["datapath"].append(dict(mode=mode, name=name, seed=seed, **result))
        alloc_text = (snap / SOURCES[1]).read_text()
        alloc_mutations = {
            "ignore_generation": ("reserve4", "generation[return_bank[p*BANK_BITS +: BANK_BITS]] !=\n"
                                  "                             return_generation[p*GENERATION_BITS +: GENERATION_BITS]", "1'b0"),
            "lose_reallocation": ("reserve4", "used[g] <= 1;", "used[g] <= !returned[g];"),
            "ignore_quota": ("quota2", "loans < RESERVE_QUOTA", "1'b1"),
            "wrap_generation": ("generation3", " && generation[g] != MAX_GENERATION", ""),
        }
        for name, (profile_name, old, new) in alloc_mutations.items():
            if alloc_text.count(old) != 1:
                raise RuntimeError("mutation anchor: " + name)
            mutant = out / (name + ".sv")
            mutant.write_text(alloc_text.replace(old, new))
            profile = allocator_profiles[profile_name]
            exe = compile_rtl(name, "tb_ntrpu_buffer_pool_allocator", profile, allocator=True, mutant=mutant)
            code, log = command([args.vvp, str(exe), "+SEED=1"], name)
            try:
                allocator_replay(log, profile, 1)
            except RuntimeError as error:
                reason = str(error)
                if not reason.startswith(("pre-edge mismatch", "post-edge mismatch")):
                    raise
            else:
                raise RuntimeError("mutation escaped: " + name)
            report["mutations"].append(dict(name=name, reason=reason, returncode=code))
        data_text = (snap / SOURCES[3]).read_text()
        for name, old, new in (
            ("corrupt_replica", "payload[pool_write_address] <= data;", "payload[pool_write_address] <= data ^ (g == 1 ? 512'd1 : 512'd0);"),
            ("alias_banks", "fill_bank * BANK_BEATS + fill_index", "fill_index"),
        ):
            if data_text.count(old) != 1:
                raise RuntimeError("mutation anchor: " + name)
            mutant = out / (name + ".sv")
            mutant.write_text(data_text.replace(old, new))
            profile = datapath_profiles["full4"]
            exe = compile_rtl(name, "tb_ntrpu_peer_multibuffer", profile, mutant=mutant)
            code, log = command([args.vvp, str(exe), "+SEED=1"], name)
            if code == 0 or "FATAL:" not in log or "payload mismatch" not in log:
                raise RuntimeError("memory mutation escaped byte checks: " + name)
            report["mutations"].append(dict(name=name, reason="bench payload mismatch", returncode=code))
        report["status"] = "COMPACT_RTL_AND_ABLATION_PASS_NOT_INTEGRATED"
    except Exception as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        report["sources_after"] = {name: digest(ROOT / name) for name in SOURCES}
        if hashes != report["sources_after"]:
            report["status"] = "SOURCES_CHANGED"
        save(out / "REVIEW.json", report)
        manifest = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*") if p.is_file()}
        save(out / "FILES.json", manifest)
        with zipfile.ZipFile(out.with_suffix(".zip"), "x", compression=zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(out).as_posix())
        print(json.dumps({key: report[key] for key in ("status", "seconds")}))
    if report["status"] != "COMPACT_RTL_AND_ABLATION_PASS_NOT_INTEGRATED":
        raise RuntimeError(report["status"])


if __name__ == "__main__":
    main()
