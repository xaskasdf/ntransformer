"""Review isolated synthesis evidence without treating OOC timing as board closure."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def review(root):
    output = root / "output"
    record = json.loads((output / "result.json").read_text())
    variant = record.get("variant", "reference")
    require(variant in ("reference", "compact", "control", "decoded", "fillstage", "elastic"), "unknown variant")
    require(record["status"] == "EXPORTED_PENDING_REVIEW", "incomplete run")
    reserves = record.get("reserves", [4, 8])
    require(reserves and len(set(reserves)) == len(reserves) and set(reserves).issubset({4, 8}), "invalid profiles")
    require([r["reserve"] for r in record["runs"]] == list(reserves) and
            all(r["returncode"] == 0 and not r.get("timeout", False) for r in record["runs"]), "failed profile")
    require(record["sources"] == record["sources_after"], "inputs changed during run")
    for name, expected in record["sources"].items():
        require(hashlib.sha256((root / "inputs" / name).read_bytes()).hexdigest() == expected, "input hash: " + name)
    for name, expected in json.loads((output / "FILES.json").read_text()).items():
        require(hashlib.sha256((output / name).read_bytes()).hexdigest() == expected, "output hash: " + name)
    profiles = []
    for reserve in reserves:
        folder = output / f"reserve_{reserve}"
        summary = dict(line.split("\t", 1) for line in (folder / "summary.tsv").read_text().splitlines())
        if variant in ("control", "decoded", "fillstage", "elastic"):
            require(summary.get("core_top") == "ntrpu_peer_multibuffer_" + variant, "wrong variant core")
        for key, expected in dict(part="xcvu5p-flva2104-2-i", tool_version="2026.1", reserve_banks=str(reserve),
                                  period_ns="5.000", clock_count="1", max_threads="2", latch_count="0", blackbox_count="0").items():
            require(summary.get(key) == expected, "unexpected " + key)
        checks = dict((name, int(count)) for name, count in re.findall(
            r"checking (\w+) \((\d+)\)", (folder / "check_timing.rpt").read_text()))
        timing_row = re.search(r"^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)",
                               (folder / "timing.rpt").read_text(), re.M)
        require(timing_row, "missing design timing summary")
        values = timing_row[1].split()
        require(len(values) == 12, "unexpected timing columns")
        timing_summary = dict(zip(("wns_ns", "tns_ns", "setup_failing_endpoints", "setup_total_endpoints",
                                   "whs_ns", "ths_ns", "hold_failing_endpoints", "hold_total_endpoints",
                                   "wpws_ns", "tpws_ns", "pulse_failing_endpoints", "pulse_total_endpoints"),
                                  map(float, values)))
        require(timing_summary["wns_ns"] == float(summary["worst_setup_slack_ns"]), "setup summary mismatch")
        for key in ("no_clock", "constant_clock", "unconstrained_internal_endpoints", "multiple_clock", "loops", "latch_loops"):
            require(checks.get(key) == 0, "internal timing issue: " + key)
        require(re.search(r"Checks found:\s+0\b", (folder / "drc.rpt").read_text()), "DRC findings require review")
        xdc = (folder / "applied.xdc").read_text()
        require(len(re.findall(r"^create_clock ", xdc, re.M)) == 1, "clock count")
        require(not re.search(r"^\s*set_(false_path|multicycle_path|clock_groups)\b", xdc, re.M), "timing exceptions")
        util = (folder / "utilization.rpt").read_text()
        resources = {}
        for label in ("CLB LUTs", "LUT as Logic", "LUT as Memory", "CLB Registers", "CARRY8", "Block RAM Tile", "RAMB36/FIFO", "RAMB18", "URAM", "DSPs"):
            match = re.search(r"^\|\s*" + re.escape(label) + r"\*?\s*\|\s*([\d.,]+)\s*\|", util, re.M)
            require(match, "missing utilization: " + label)
            resources[label] = float(match[1].replace(",", ""))
        memories = [line.split("\t") for line in (folder / "memories.tsv").read_text().splitlines()[1:]]
        primitive_counts = Counter(row[1] for row in memories)
        bank_counts = Counter()
        for row in memories:
            pattern = r"packed_pool\.reader\[(\d+)\]" if variant in ("compact", "control", "decoded", "fillstage", "elastic") else r"memory_bank\[(\d+)\]"
            match = re.search(pattern, row[0])
            require(match, "unreviewed memory outside payload banks: " + row[0])
            bank_counts[int(match[1])] += 1
        require(set(bank_counts) == set(range(2 if variant in ("compact", "control", "decoded", "fillstage", "elastic") else 4 + reserve)), "missing memory bank/replica")
        # Capture observations rather than assume a specific mapping beforehand.
        raw_bits = sum((36864 if row[1].startswith("RAMB36") else 18432 if row[1].startswith("RAMB18") else 0)
                       for row in memories)
        require(all(row[1].startswith(("RAMB18", "RAMB36")) for row in memories), "unreviewed RAM type")
        log = (output / f"reserve_{reserve}.console.log").read_text()
        require(not re.search(r"^(ERROR|CRITICAL WARNING):", log, re.M), "tool error/critical warning")
        warnings = Counter(re.findall(r"^WARNING: \[([^]]+)\]", log, re.M))
        methodology = [dict(rule=rule, severity=severity, description=description.strip(), count=int(count))
                       for rule, severity, description, count in re.findall(
                           r"^\|\s*([A-Z]+-\d+)\s*\|\s*(\w+)\s*\|\s*([^|]+)\|\s*(\d+)",
                           (folder / "methodology.rpt").read_text(), re.M)]
        path_text = (folder / "critical_paths.rpt").read_text()
        critical = {}
        for label in ("Source", "Destination", "Data Path Delay", "Logic Levels"):
            match = re.search(r"^\s*" + label + r":\s*(.+)$", path_text, re.M)
            require(match, "missing critical path field: " + label)
            critical[label] = match[1].strip()
        require(resources["RAMB36/FIFO"] == primitive_counts.get("RAMB36E2", 0), "RAMB36 count mismatch")
        require(resources["RAMB18"] == primitive_counts.get("RAMB18E2", 0), "RAMB18 count mismatch")
        require(resources["Block RAM Tile"] == resources["RAMB36/FIFO"] + resources["RAMB18"] / 2,
                "BRAM tile count mismatch")
        if variant in ("compact", "control", "decoded", "fillstage", "elastic"):
            allocator = "decoded" if variant in ("decoded", "fillstage", "elastic") else "parallel"
            require("ntrpu_buffer_pool_allocator_" + allocator in (folder / "hierarchy.rpt").read_text(),
                    "expected allocator absent from hierarchy")
        profiles.append(dict(reserve=reserve, payload_bytes=(4 + reserve) * 4096, resources=resources,
            ram_primitives=dict(primitive_counts), memory_groups=dict(bank_counts), raw_ram_bits=raw_bits,
            memory_group_kind="reader_replica" if variant in ("compact", "control", "decoded", "fillstage", "elastic") else "logical_bank",
            payload_over_raw_ram_fraction=(4 + reserve) * 4096 * 8 / raw_bits,
            internal_setup_slack_ns=float(summary["worst_setup_slack_ns"]), timing_checks=checks,
            timing_summary=timing_summary, warnings=dict(warnings), methodology=methodology, critical_path=critical))
        if variant == "reference":
            profiles[-1]["primitive_count_by_bank"] = dict(bank_counts)
    return dict(status="SYNTHESIS_EVIDENCE_REVIEWED_WITH_OPEN_BOUNDARIES", variant=variant, profiles=profiles,
                internal_setup_target_met=all(p["internal_setup_slack_ns"] >= 0 for p in profiles),
                limits=["Synthesis only, no placement/routing", "No external input/output delays or CDC qualification",
                        "OOC DRC does not cover board connectivity", "Warnings/methodology retained for explicit review"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = review(args.run)
    with (args.run / "REVIEW.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))
