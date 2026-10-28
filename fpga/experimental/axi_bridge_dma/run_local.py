"""Frozen, bounded local RTL regression; no Vivado, cloud or production integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


HERE = Path(__file__).resolve().parent
FILES = ["ntrpu_axi_bridge_writer.sv", "tb_ntrpu_axi_bridge_writer.sv", "run_local.py", "README.md"]
LENGTHS = {i: (4096 if i == 2 else 128 if i in (1, 3, 13) else 64) for i in range(16)}
BUS_CASES = {0, 1, 2, 3, 6, 7, 8, 12, 15}
FAULTS = {6: 1, 7: 4, 8: 8, 9: 2, 10: 16, 11: 2, 12: 4, 14: 8, 15: 1}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(log, seed):
    require("FATAL" not in log and "ERROR" not in log, "Simulator diagnostics")
    require(f"WRITER_LOCAL_PASS seed={seed} cases=16" in log, "Missing complete marker")
    passes = re.findall(r"CASE PASS id=(\d+) fault=([0-9a-f]+) owned=([01]) aw=(\d+) bytes=(\d+) b=(\d+) results=(\d+)", log)
    require([int(p[0]) for p in passes] == list(range(16)), "Case coverage/order")
    rows = {i: [] for i in range(16)}
    for line in log.splitlines():
        if line.startswith("EV "):
            _, kind, *fields = line.split()
            data = dict(field.split("=", 1) for field in fields)
            case = int(data.pop("case"))
            require(case in rows, "Unknown event case")
            rows[case].append((kind, data))
    for p in passes:
        case, fault, owned, aw, count, b, results = [int(p[0]), int(p[1], 16), *map(int, p[2:])]
        has_bus = case in BUS_CASES
        expected_results = int(case in (0, 1, 2, 3, 15))
        require(fault == FAULTS.get(case, 0), f"Fault classification {case}")
        require(owned == int(case in FAULTS and case != 10), f"Ownership {case}")
        require((aw, count, b, results) == (int(has_bus), LENGTHS[case] if has_bus else 0, int(has_bus), expected_results), f"Counters {case}")
        events = rows[case]
        addresses = [e for k, e in events if k == "AW"]
        writes = [e for k, e in events if k == "W"]
        responses = [e for k, e in events if k == "B"]
        outcomes = [e for k, e in events if k == "RESULT"]
        mappings = [e for k, e in events if k == "MAP"]
        require(len(mappings) == 1, f"One mapping image {case}")
        require(mappings[0] == dict(session="0000000000000009", generation="0000000000000001", axi="0000000010000000", pcie="0000000100000000"), f"Frozen mapping {case}")
        require(len(addresses) == aw and len(responses) == b and len(outcomes) == results, f"Observed event counts {case}")
        payload = bytearray()
        for index, item in enumerate(writes):
            require(int(item["offset"]) == len(payload), f"Contiguous W offsets {case}")
            require(int(item["last"]) == int(index == len(writes)-1), f"W last {case}")
            payload.extend(int(item["data"], 16).to_bytes(64, "little"))
        expected = bytes(((case*13+offset*37) ^ (offset >> 3)) & 255 for offset in range(count))
        require(bytes(payload) == expected, f"Payload bytes {case}")
        if has_bus:
            require(int(addresses[0]["address"], 16) == 0x10000000 and (int(addresses[0]["len"])+1)*64 == count, f"AXI target {case}")
            require(int(responses[0]["response"]) == (2 if case == 7 else 0) and int(responses[0]["id"]) == (1 if case == 12 else 0), f"B identity/status {case}")
        if expected_results:
            order = [k for k, _ in events]
            require(order.index("RESULT") > order.index("B") and order.index("RESULT") > order.index("ACK"), f"Result causality {case}")
            require(int(outcomes[0]["bytes"]) == LENGTHS[case], f"Result length {case}")
    source = [e for k, e in rows[2] if k == "SRC"]
    require(len(source) == 1 and int(source[0]["address"], 16) == (1 << 39)-4096, "High source address")
    coverage = re.search(r"COVER aw=(\d+) w=(\d+) result=(\d+) map=(\d+) source=(\d+)", log)
    require(coverage is not None and all(int(n) > 0 for n in coverage.groups()), "Real backpressure coverage")
    return dict(cases=16, payload_bytes=sum(LENGTHS[c] for c in BUS_CASES), native_ip=False)


def corruption_checks(log, seed):
    first_w = next(line for line in log.splitlines() if line.startswith("EV W "))
    altered_w = first_w[:-1] + ("0" if first_w[-1] != "0" else "1")
    first_b = next(line for line in log.splitlines() if line.startswith("EV B "))
    variants = {
        "payload": log.replace(first_w, altered_w, 1),
        "missing_response": log.replace(first_b, "", 1),
        "premature_release": log.replace("CASE PASS id=6 fault=01 owned=1", "CASE PASS id=6 fault=01 owned=0", 1),
    }
    for name, candidate in variants.items():
        require(candidate != log, "Mutation not applied: " + name)
        try:
            review(candidate, seed)
        except ValueError:
            continue
        raise ValueError("Undetected evidence corruption: " + name)
    return list(variants)


def run(output, toolchain, seeds):
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "source"
    frozen.mkdir()
    source_hashes = {name: sha(HERE / name) for name in FILES}
    for name in FILES:
        shutil.copy2(HERE / name, frozen / name)
    bins = toolchain / "mingw64/bin"
    tools = {str(p.relative_to(toolchain)): sha(p) for p in toolchain.rglob("*") if p.is_file()}
    env = dict(os.environ)
    windows = Path(env.get("SystemRoot", "C:/Windows"))
    env["PATH"] = os.pathsep.join(map(str, [bins, windows / "System32", windows]))
    result = dict(status="FAILED", sources=source_hashes, toolchain=tools, runs=[],
                  native_ip=False, source_memory_validated=False, hardware_exclusion_validated=False,
                  synthesized=False, board_qualified=False)
    start = time.monotonic()

    def execute(command, cwd, name):
        job = subprocess.run(list(map(str, command)), cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
        log = job.stdout + job.stderr
        (cwd / name).write_text(log, encoding="utf-8")
        require(job.returncode == 0, f"{name}: exit {job.returncode}")
        return log

    try:
        execute([bins / "iverilog.exe", "-V"], output, "compiler.log")
        execute([bins / "vvp.exe", "-V"], output, "simulator.log")
        compiled = execute([bins / "iverilog.exe", "-g2012", "-Wall", "-s", "tb_ntrpu_axi_bridge_writer",
                            "-o", output / "writer.vvp", frozen / FILES[0], frozen / FILES[1]], output, "compile.log")
        require(not compiled.strip(), "Unexpected compiler diagnostics; inspect compile.log")
        for seed in range(seeds):
            case = output / f"seed-{seed:02}"
            case.mkdir()
            log = execute([bins / "vvp.exe", output / "writer.vvp", f"+seed={seed}"], case, "simulation.log")
            checked = review(log, seed)
            checked["negative_controls"] = corruption_checks(log, seed)
            (case / "review.json").write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")
            result["runs"].append(dict(seed=seed, **checked))
        # Compile deliberate RTL defects privately: prove that the bench detects
        # mutable addressing, discarded ownership and completion without an ACK.
        original = (frozen / FILES[0]).read_text(encoding="utf-8")
        mutations = {
            "live_descriptor": ("assign m_awaddr=address[AXI_ADDR_BITS-1:0];",
                                "assign m_awaddr=cfg_axi_base+cmd_offset;", "AXI address and burst"),
            "release_on_fault": ("fault<=1;", "fault<=1;owned<=0;", "fault preserves outstanding AXI offers and ownership"),
            "completion_without_ack": ("owned && b_ok && ack_ok &&", "owned && b_ok &&", "B cannot replace ACK"),
        }
        result["rtl_negative_controls"] = []
        for name, (before, after, diagnostic) in mutations.items():
            require(original.count(before) == 1, "Unique mutation site: " + name)
            mutant = output / "mutations" / name
            mutant.mkdir(parents=True)
            source = mutant / "mutant.sv"
            source.write_text(original.replace(before, after), encoding="utf-8")
            log = execute([bins / "iverilog.exe", "-g2012", "-Wall", "-s", "tb_ntrpu_axi_bridge_writer",
                           "-o", mutant / "writer.vvp", source, frozen / FILES[1]], mutant, "compile.log")
            require(not log.strip(), "Mutant compilation diagnostics")
            job = subprocess.run([str(bins / "vvp.exe"), str(mutant / "writer.vvp"), "+seed=0"],
                                 cwd=mutant, env=env, capture_output=True, text=True, timeout=30)
            log = job.stdout + job.stderr
            (mutant / "simulation.log").write_text(log, encoding="utf-8")
            require(job.returncode != 0 and diagnostic in log and "TIMEOUT" not in log,
                    "RTL mutation was not rejected semantically: " + name)
            result["rtl_negative_controls"].append(dict(name=name, diagnostic=diagnostic, returncode=job.returncode))
        result["status"] = "AXI_BRIDGE_WRITER_LOCAL_PASS"
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        result["error"] = str(error)
    result["seconds"] = round(time.monotonic()-start, 6)
    result["sources_after"] = {name: sha(HERE / name) for name in FILES}
    result["toolchain_unchanged"] = tools == {str(p.relative_to(toolchain)): sha(p) for p in toolchain.rglob("*") if p.is_file()}
    if source_hashes != result["sources_after"] or not result["toolchain_unchanged"]:
        result.update(status="FAILED", error="Sources or toolchain changed during run")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "FILES.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("--toolchain", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.seeds <= 16:
        ap.error("--seeds must be 1..16")
    result = run(args.output.resolve(), args.toolchain.resolve(), args.seeds)
    print(json.dumps({key: result[key] for key in ("status", "seconds", "error") if key in result}))
    raise SystemExit(0 if result["status"] == "AXI_BRIDGE_WRITER_LOCAL_PASS" else 1)
