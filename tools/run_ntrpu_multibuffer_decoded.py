"""Compare bank-decoded leases and datapath cycle traces with preserved baselines."""

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
    "fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv", "fpga/rtl/ntrpu_peer_multibuffer_control.sv",
    "fpga/rtl/ntrpu_peer_multibuffer_decoded.sv", "fpga/tb/tb_ntrpu_buffer_pool_allocator.sv",
    "fpga/tb/tb_ntrpu_peer_multibuffer.sv", "tools/check_ntrpu_buffer_pool_allocator.py",
    "tools/check_ntrpu_peer_multibuffer.py", "tools/run_ntrpu_buffer_pool_allocator.py",
    "tools/run_ntrpu_peer_multibuffer.py", "tools/run_ntrpu_multibuffer_decoded.py")


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
    from check_ntrpu_buffer_pool_allocator import replay as allocator_replay
    from check_ntrpu_peer_multibuffer import replay as datapath_replay
    from run_ntrpu_buffer_pool_allocator import PROFILES as allocator_profiles
    from run_ntrpu_peer_multibuffer import PROFILES as datapath_profiles
    datapath_profiles = {**datapath_profiles, "odd_beats": (2, 4, 3, 16, 4)}
    report = dict(status="RUNNING", created_utc=datetime.now(timezone.utc).isoformat(),
                  input_sha256=hashes, allocator=[], datapath=[], mutations=[])

    def command(argv, label):
        start = time.monotonic()
        with (output / (label + ".stdout.log")).open("w") as stdout, (output / (label + ".stderr.log")).open("w") as stderr:
            run = subprocess.run(argv, cwd=output, stdout=stdout, stderr=stderr, timeout=45)
        save(output / (label + ".command.json"), dict(argv=argv, returncode=run.returncode,
                                                     seconds=time.monotonic() - start))
        return run.returncode, (output / (label + ".stdout.log")).read_text()

    def compile_rtl(kind, variant, label, profile, mutant=None):
        allocator = kind == "allocator"
        base = "ntrpu_buffer_pool_allocator" if allocator else "ntrpu_peer_multibuffer"
        top = "tb_" + base
        bench = output / (kind + "_" + variant + "_tb.sv")
        if not bench.exists():
            bench.write_text((snapshot / "fpga/tb" / (top + ".sv")).read_text().replace(
                base + " #(", base + "_" + variant + " #("))
        cmd = [args.iverilog] + (["-B", args.ivl_base] if args.ivl_base else [])
        cmd += ["-g2012", "-Wall", "-s", top]
        keys = (("CHANNELS", "DEDICATED_BANKS", "RESERVE_BANKS", "RESERVE_QUOTA", "GENERATION_BITS")
                if allocator else ("CHANNELS", "RESERVE_BANKS", "BANK_BEATS", "DESCRIPTORS", "MAX_OUTSTANDING"))
        cmd += [f"-P{top}.{key}={value}" for key, value in zip(keys, profile)]
        rtl = snapshot / "fpga/rtl"
        files = [mutant or rtl / (base + "_" + variant + ".sv"), bench]
        if not allocator:
            files = [rtl / ("ntrpu_buffer_pool_allocator" + suffix + ".sv")
                     for suffix in ("", "_parallel", "_decoded")] + files
        exe = output / (label + ".vvp")
        cmd += ["-o", str(exe)] + list(map(str, files))
        if command(cmd, "compile_" + label)[0]:
            raise RuntimeError("compile failed: " + label)
        return exe

    started = time.monotonic()
    try:
        for kind, profiles, variants, seeds, replay in (
            ("allocator", allocator_profiles, ("parallel", "decoded"), (1, 7, 20261004), allocator_replay),
            ("datapath", datapath_profiles, ("control", "decoded"), (1, 7), datapath_replay),
        ):
            traces = {}
            for variant in variants:
                for name, profile in profiles.items():
                    label = kind + "_" + variant + "_" + name
                    exe = compile_rtl(kind, variant, label, profile)
                    for seed in seeds:
                        code, log = command([args.vvp, str(exe), f"+SEED={seed}"], f"{label}_seed{seed}")
                        if code:
                            raise RuntimeError("simulation failed: " + label)
                        result = replay(log, profile, seed)
                        trace = [line for line in log.splitlines() if line.startswith(("TRACE ", "CASE ", "PASS "))]
                        if variant == variants[0]:
                            traces[name, seed] = trace
                        elif not trace or trace != traces[name, seed]:
                            raise RuntimeError("cycle trace mismatch: " + label)
                        report[kind].append({**result, "variant": variant, "profile": name,
                                             "seed": seed, "trace_lines": len(trace)})
        source = (snapshot / "fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv").read_text()
        for name, profile_name, old, new in (
            ("ignore_generation", "reserve4", "generation[bank_index] == return_generation[port_index*GENERATION_BITS +: GENERATION_BITS]", "1'b1"),
            ("lose_reallocation", "reserve4", "used[g] <= 1;", "used[g] <= !returned[g];"),
            ("ignore_quota", "quota2", "loans < RESERVE_QUOTA", "1'b1"),
            ("wrap_generation", "generation3", " && generation[g] != MAX_GENERATION", ""),
        ):
            if source.count(old) != 1:
                raise RuntimeError("mutation anchor: " + name)
            mutant = output / (name + ".sv")
            mutant.write_text(source.replace(old, new))
            profile = allocator_profiles[profile_name]
            exe = compile_rtl("allocator", "decoded", name, profile, mutant)
            code, log = command([args.vvp, str(exe), "+SEED=1"], name)
            try:
                allocator_replay(log, profile, 1)
            except RuntimeError as error:
                reason = str(error)
                if not reason.startswith(("pre-edge mismatch", "post-edge mismatch")):
                    raise
            else:
                raise RuntimeError("mutation escaped: " + name)
            report["mutations"].append(dict(name=name, returncode=code, reason=reason))
        report["status"] = "DECODED_RTL_CYCLE_EQUIVALENCE_PASS_NOT_INTEGRATED"
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
            if bundle.testzip() is not None or any(hashlib.sha256(bundle.read(n)).hexdigest() != h for n, h in manifest.items()):
                raise RuntimeError("archive verification failed")
        save(output / "VERIFIED.json", dict(sha256=digest(archive), crc="PASS", hashes="PASS"))
        print(json.dumps({k: report[k] for k in ("status", "seconds")}))
    if report["status"] != "DECODED_RTL_CYCLE_EQUIVALENCE_PASS_NOT_INTEGRATED":
        raise RuntimeError(report["status"])


if __name__ == "__main__":
    main()
