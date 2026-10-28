"""Review isolated peer receiver synthesis, preserving OOC limitations."""

import argparse
import collections
import hashlib
import json
import pathlib
import re


def need(value, message):
    if not value:
        raise RuntimeError(message)


def review(run):
    output = run / "output"
    record = json.loads((output / "result.json").read_text())
    need(record["status"] == "EXPORTED_PENDING_REVIEW", "synthesis did not finish")
    need(len(record["runs"]) == 2 and all(r["returncode"] == 0 for r in record["runs"]),
         "missing successful synthesis profiles")
    need(record["sources"] == record["sources_after"], "sources changed during synthesis")
    for name, expected in record["sources"].items():
        need(hashlib.sha256((run / "inputs" / name).read_bytes()).hexdigest() == expected,
             f"source snapshot mismatch: {name}")
    manifest = json.loads((output / "FILES.json").read_text())
    for name, expected in manifest.items():
        need(hashlib.sha256((output / name).read_bytes()).hexdigest() == expected,
             f"exported evidence mismatch: {name}")
    profiles = []
    for beats in (4, 64):
        folder = output / f"beats_{beats}"
        summary = dict(line.split("\t", 1) for line in (folder / "summary.tsv").read_text().splitlines())
        expected = dict(part="xcvu5p-flva2104-2-i", max_beats=str(beats), tool_version="2026.1",
                        max_threads="2", clock_count="1", period_ns="5.000",
                        latch_count="0", blackbox_count="0")
        need(all(summary.get(k) == v for k, v in expected.items()), "unexpected synthesis profile")
        need(float(summary["worst_setup_slack_ns"]) >= 0, "negative internal setup estimate")
        checks = dict((name, int(count)) for name, count in re.findall(
            r"checking (\w+) \((\d+)\)", (folder / "check_timing.rpt").read_text()))
        for name in ("no_clock", "constant_clock", "unconstrained_internal_endpoints",
                     "multiple_clock", "loops", "latch_loops"):
            need(checks.get(name) == 0, f"internal timing concern: {name}")
        need(checks.get("no_input_delay") == 858 and checks.get("no_output_delay") == 790,
             "external timing context changed; re-review required")
        need(re.search(r"Checks found:\s+0\b", (folder / "drc.rpt").read_text()), "DRC findings")
        xdc = (folder / "applied.xdc").read_text()
        need(len(re.findall(r"^create_clock ", xdc, re.M)) == 1, "wrong clock constraints")
        need(not re.search(r"^\s*set_(false_path|multicycle_path|clock_groups)\b", xdc, re.M),
             "unexpected timing exceptions")
        methodology = (folder / "methodology.rpt").read_text()
        rules = re.findall(r"^\|\s*([A-Z]+-\d+)\s*\|\s*(\w+)\s*\|", methodology, re.M)
        need(set(rules) == {("TIMING-18", "Warning")}, "unexpected methodology rules")
        log = (output / f"beats_{beats}.console.log").read_text()
        need(not re.search(r"^(ERROR|CRITICAL WARNING):", log, re.M), "tool error/critical warning")
        warnings = collections.Counter(re.findall(r"^WARNING: \[([^]]+)\]", log, re.M))
        need(set(warnings) == {"Synth 8-3917", "Netlist 29-101", "Timing 38-242", "DRC 23-814"},
             "unexpected warning class; review explicitly")
        for line in log.splitlines():
            if line.startswith("WARNING: [Synth 8-3917]"):
                need("driven by constant" in line, "unreviewed synthesis warning")
        util = (folder / "utilization.rpt").read_text()
        resources = {}
        for label in ("CLB LUTs*", "LUT as Logic", "LUT as Memory", "CLB Registers", "Block RAM Tile", "DSPs"):
            match = re.search(r"^\|\s*" + re.escape(label) + r"\s*\|\s*(\d+)\s*\|", util, re.M)
            need(match is not None, f"missing utilization: {label}")
            resources[label] = int(match[1])
        profiles.append(dict(beats=beats, resources=resources,
                             internal_setup_estimate_ns=float(summary["worst_setup_slack_ns"]),
                             checks=checks, warning_counts=dict(warnings)))
    return dict(status="SYNTHESIS_REVIEW_PASS_WITH_OPEN_BOUNDARIES", profiles=profiles,
                limits=["No placement/routing or external I/O timing",
                        "Missing HD.CLK_SRC: parent clock delay/skew not estimated",
                        "OOC connectivity DRC is incomplete",
                        "Not a CDC, PCIe, DDR/HBM or physical board qualification"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path, help="Downloaded run with inputs/ and output/")
    args = parser.parse_args()
    result = review(args.run)
    target = args.run / "REVIEW.json"
    with target.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))
