"""Review frequency audits without claiming probe-port or board timing closure."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from check_ntrpu_multibuffer_route import digest, require


def review(root, source_checkpoint):
    output = root / "output"
    record = json.loads((output / "result.json").read_text())
    require(record["status"] == "EXPORTED_PENDING_REVIEW" and record["returncode"] == 0, "failed audit")
    require(record["sources"] == record["sources_after"], "audit source changed")
    for name, value in record["sources"].items():
        require(digest(root / "inputs" / name) == value, "source hash: " + name)
    for name, value in json.loads((output / "FILES.json").read_text()).items():
        require(digest(output / name) == value, "output hash: " + name)
    require(digest(source_checkpoint) == digest(output / "audit/input.dcp") ==
            record["checkpoint_sha256"] == record["checkpoint_sha256_after"], "routed checkpoint mismatch")
    rows = list(csv.DictReader((output / "audit/summary.tsv").read_text().splitlines(), delimiter="\t"))
    periods = record.get("periods_ns", ["6.667", "8.000"])
    require(periods and len(set(periods)) == len(periods) and
            set(periods).issubset({"5.715", "6.000", "6.667", "8.000"}), "unreviewed periods")
    require([row["period_ns"] for row in rows] == list(periods), "unexpected period audit")
    profiles = []
    for index, row in enumerate(rows):
        require(row["physical_identity_preserved"] == "1", "placement or routing changed")
        folder = output / "audit" / ("period_" + row["period_ns"])
        text = (folder / "timing.rpt").read_text()
        match = re.search(r"^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)", text, re.M)
        require(match and len(match[1].split()) == 12, "missing timing summary")
        names = ("wns_ns", "tns_ns", "setup_failing_endpoints", "setup_total_endpoints",
                 "whs_ns", "ths_ns", "hold_failing_endpoints", "hold_total_endpoints",
                 "wpws_ns", "tpws_ns", "pulse_failing_endpoints", "pulse_total_endpoints")
        timing = dict(zip(names, map(float, match[1].split())))
        require(timing["wns_ns"] == float(row["worst_setup_ns"]) and
                timing["whs_ns"] == float(row["worst_hold_ns"]), "slack mismatch")
        checks = {name: int(count) for name, count in re.findall(r"checking (\w+) \((\d+)\)",
                    (folder / "check_timing.rpt").read_text())}
        for key in ("no_clock", "constant_clock", "unconstrained_internal_endpoints", "multiple_clock", "loops", "latch_loops"):
            require(checks.get(key) == 0, "internal timing coverage: " + key)
        xdc = (folder / "applied.xdc").read_text()
        # write_xdc retains source commands and later overrides. Check their
        # exact sequence AND the effective clock report, not raw command count.
        definitions = re.findall(r"^create_clock -period ([\d.]+) -name fixture_clk \[get_ports clk\]$", xdc, re.M)
        expected_history = ["5.000"] + list(periods[:index + 1])
        require(definitions == expected_history and len(re.findall(r"^create_clock ", xdc, re.M)) == len(definitions),
                "unexpected clock override history")
        effective = re.findall(r"^(\S+)\s+([\d.]+)\s+\{[\d.]+\s+[\d.]+\}\s+\S+\s+\{clk\}",
                               (folder / "clocks.rpt").read_text(), re.M)
        require(effective == [("fixture_clk", row["period_ns"])], "wrong effective clock")
        require(not re.search(r"^\s*set_(false_path|multicycle_path|clock_groups|max_delay|min_delay|disable_timing|case_analysis)\b",
                              xdc, re.M), "unreviewed exception")
        for constraint in ("set_clock_uncertainty -setup 0.100", "set_clock_uncertainty -hold 0.050"):
            require(constraint in xdc, "uncertainty changed")
        nominal = {"5.715": 175, "6.000": 166.667, "6.667": 150, "8.000": 125}
        profiles.append(dict(period_ns=float(row["period_ns"]), nominal_mhz=nominal[row["period_ns"]],
                             timing=timing, checks=checks, clock_override_history_ns=list(map(float, definitions)),
                             scoped_timing_target_met=all(timing[k] >= 0 for k in ("wns_ns", "whs_ns", "wpws_ns"))))
    log = (output / "console.log").read_text()
    require("MULTIBUFFER_PERIOD_AUDIT_EXPORTED" in log and not re.search(r"^ERROR:", log, re.M), "tool audit incomplete")
    return dict(status="PERIOD_AUDIT_EVIDENCE_REVIEWED", reserve=record["reserve"], profiles=profiles,
                physical_implementation_unchanged=True, source_checkpoint_sha256=digest(source_checkpoint),
                warnings=dict(Counter(re.findall(r"^WARNING: \[([^]]+)\]", log, re.M))),
                board_or_parent_closure=False, limits=["STA on fixed routed registered fixture; no new P&R",
                    "Probe I/O remains unqualified; no PCIe/HBM system or board timing proof", "Not an Fmax search"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source_checkpoint", type=Path)
    args = parser.parse_args()
    result = review(args.run, args.source_checkpoint)
    with (args.run / "REVIEW.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
