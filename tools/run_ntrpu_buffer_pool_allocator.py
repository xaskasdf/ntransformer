"""Local Icarus allocator bench, independent replay, and negative mutations.

Writes immutable inputs/logs/results to a new directory; no cloud or Vivado.
Use a Linux/WSL Python and pass relocated Icarus paths when needed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("fpga/rtl/ntrpu_buffer_pool_allocator.sv", "fpga/tb/tb_ntrpu_buffer_pool_allocator.sv",
           "tools/check_ntrpu_buffer_pool_allocator.py", "tools/run_ntrpu_buffer_pool_allocator.py")
PROFILES = {
    "reserve4": (2, 2, 4, 4, 64), "reserve8": (2, 2, 8, 8, 64),
    "quota2": (2, 2, 4, 2, 64), "odd9": (3, 2, 3, 3, 64),
    "shared3": (2, 0, 3, 3, 64), "single": (1, 1, 0, 0, 64),
    "generation3": (1, 1, 0, 0, 3), "no_reserve": (2, 2, 0, 0, 64),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--ivl-base")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    hashes = {}
    for relative in SOURCES:
        source = ROOT / relative
        target = inputs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    sys.path.insert(0, str(inputs / "tools"))
    from check_ntrpu_buffer_pool_allocator import replay

    def command(argv, name):
        started = time.perf_counter()
        run = subprocess.run(argv, text=True, capture_output=True, timeout=45, cwd=output)
        (output / (name + ".stdout.log")).write_text(run.stdout, encoding="utf-8")
        (output / (name + ".stderr.log")).write_text(run.stderr, encoding="utf-8")
        (output / (name + ".command.json")).write_text(json.dumps({
            "argv": argv, "returncode": run.returncode, "seconds": time.perf_counter() - started
        }, indent=2) + "\n", encoding="utf-8")
        return run

    def compile_rtl(name, profile, rtl):
        executable = output / (name + ".vvp")
        argv = [args.iverilog]
        if args.ivl_base:
            argv += ["-B", args.ivl_base]
        argv += ["-g2012", "-Wall", "-s", "tb_ntrpu_buffer_pool_allocator"]
        for key, value in zip(("CHANNELS", "DEDICATED_BANKS", "RESERVE_BANKS", "RESERVE_QUOTA", "GENERATION_BITS"), profile):
            argv += [f"-Ptb_ntrpu_buffer_pool_allocator.{key}={value}"]
        argv += ["-o", str(executable), str(rtl), str(inputs / SOURCES[1])]
        result = command(argv, "compile_" + name)
        if result.returncode:
            raise RuntimeError(f"compile failed: {name}")
        return executable

    start = time.perf_counter()
    results = []
    for name, profile in PROFILES.items():
        executable = compile_rtl(name, profile, inputs / SOURCES[0])
        for seed in (1, 7, 20261004):
            result = command([args.vvp, str(executable), f"+SEED={seed}"], f"{name}_seed{seed}")
            if result.returncode:
                raise RuntimeError(f"simulation failed: {name}, seed {seed}")
            results.append({"name": name, **replay(result.stdout, profile, seed)})

    rtl_text = (inputs / SOURCES[0]).read_text(encoding="utf-8")
    mutations = {
        "ignore_generation": ("reserve4", "generation[return_bank[p*BANK_BITS +: BANK_BITS]] !=\n"
                              "                             return_generation[p*GENERATION_BITS +: GENERATION_BITS]", "1'b0"),
        "lose_reallocated_bank": ("reserve4", "used[alloc_bank] <= 1;", "used[alloc_bank] <= !returned[alloc_bank];"),
        "ignore_quota": ("quota2", "loans[channel] < RESERVE_QUOTA", "1'b1"),
        "wrap_generation": ("generation3", " && generation[b] != MAX_GENERATION", ""),
    }
    negatives = []
    for name, (profile_name, old, new) in mutations.items():
        if rtl_text.count(old) != 1:
            raise RuntimeError(f"mutation anchor is not unique: {name}")
        mutant = output / (name + ".sv")
        mutant.write_text(rtl_text.replace(old, new), encoding="utf-8")
        profile = PROFILES[profile_name]
        executable = compile_rtl(name, profile, mutant)
        result = command([args.vvp, str(executable), "+SEED=1"], name)
        try:
            replay(result.stdout, profile, 1)
        except RuntimeError as error:
            reason = str(error)
        else:
            raise RuntimeError(f"mutation escaped the checker: {name}")
        # A process failure alone is insufficient: require an actual edge mismatch.
        if not (reason.startswith("pre-edge mismatch") or reason.startswith("post-edge mismatch")):
            raise RuntimeError(f"mutant did not reach a checked signal mismatch: {name}: {reason}")
        negatives.append({"name": name, "status": "DETECTED", "reason": reason,
                          "simulator_returncode": result.returncode})

    report = {"status": "ISOLATED_ALLOCATOR_RTL_PASS_NOT_INTEGRATED", "created_utc": datetime.now(timezone.utc).isoformat(),
              "wall_seconds": time.perf_counter() - start, "input_sha256": hashes,
              "runs": results, "negative_mutations": negatives}
    (output / "REVIEW.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "runs": len(results),
                      "cycles": sum(r["cycles"] for r in results), "mutants_detected": len(negatives),
                      "wall_seconds": report["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
