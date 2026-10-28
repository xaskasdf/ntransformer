"""Review OOC routed evidence; never equate assumed context with board closure."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utilization(path):
    values = {}
    text = path.read_text()
    for label in ("CLB LUTs", "LUT as Logic", "LUT as Memory", "CLB Registers", "CARRY8",
                  "Block RAM Tile", "RAMB36/FIFO", "RAMB18", "URAM", "DSPs", "BUFGCE"):
        match = re.search(r"^\|\s*" + re.escape(label) + r"\*?\s*\|\s*([\d.,]+)\s*\|", text, re.M)
        require(match, "missing resource: " + label)
        values[label] = float(match[1].replace(",", ""))
    return values


def review(root, reserves=None):
    output = root / "output"
    record = json.loads((output / "result.json").read_text())
    context = record.get("context", "assumed")
    require(context in ("assumed", "registered"), "unknown physical context")
    if reserves is None:
        require(record["status"] == "EXPORTED_PENDING_REVIEW", "incomplete implementation")
        reserves = record.get("reserves", [4, 8])
    require(reserves and len(set(reserves)) == len(reserves), "invalid profile selection")
    runs = {run["reserve"]: run for run in record["runs"]}
    require(all(r in runs and runs[r]["returncode"] == 0 and not runs[r]["timeout"] for r in reserves), "failed selected profile")
    require(record["sources"] == record["sources_after"], "sources changed")
    require(record["checkpoints"] == record["checkpoints_after"], "checkpoints changed")
    require(record.get("resumed_files", {}) == record.get("resumed_files_after", {}), "resume inputs changed")
    for name, expected in record.get("resumed_files", {}).items():
        relative = name.partition("/inputs/")[2]
        require(relative and digest(root / "inputs" / relative) == expected, "resumed input hash: " + name)
    for name, expected in record["sources"].items():
        require(digest(root / "inputs" / name) == expected, "input hash: " + name)
    for name, expected in json.loads((output / "FILES.json").read_text()).items():
        require(digest(output / name) == expected, "output hash: " + name)
    profiles = []
    for reserve in reserves:
        folder = output / f"reserve_{reserve}"
        original = root / "inputs" / "checkpoints" / f"reserve_{reserve}" / "multibuffer_synth.dcp"
        require(digest(folder / "input.dcp") == digest(original), "wrong input checkpoint")
        require(digest(original) in record["checkpoints"].values(), "checkpoint absent from provenance")
        summary = dict(line.split("\t", 1) for line in (folder / "summary.tsv").read_text().splitlines())
        expected = dict(part="xcvu5p-flva2104-2-i", tool_version="2026.1", reserve_banks=str(reserve),
                        period_ns="5.000", clock_count="1",
                        max_threads=str(record.get("threads", 2)), latch_count="0", blackbox_count="0", route_command_failed="0")
        expected["clock_site" if context == "registered" else "assumed_clock_site"] = "BUFGCE_X0Y60"
        if context == "registered":
            expected.update(clock_route_status="ROUTED", route_string_gap_nets="0")
            expected["route_directive"] = record.get("route_directive", "Default")
            if "place_directive" in summary:
                expected["place_directive"] = record.get("place_directive", "Default")
        for key, value in expected.items():
            require(summary.get(key) == value, "unexpected " + key)
        require((folder / "route_status.txt").read_text().startswith("return_code=0\n"), "route command failed")
        text = (folder / "timing.rpt").read_text()
        match = re.search(r"^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)", text, re.M)
        require(match and len(match[1].split()) == 12, "missing timing summary")
        timing = dict(zip(("wns_ns", "tns_ns", "setup_failing_endpoints", "setup_total_endpoints",
                           "whs_ns", "ths_ns", "hold_failing_endpoints", "hold_total_endpoints",
                           "wpws_ns", "tpws_ns", "pulse_failing_endpoints", "pulse_total_endpoints"),
                          map(float, match[1].split())))
        checks = {name: int(count) for name, count in re.findall(r"checking (\w+) \((\d+)\)",
                                                               (folder / "check_timing.rpt").read_text())}
        for key in ("no_clock", "constant_clock", "unconstrained_internal_endpoints", "multiple_clock",
                    "loops", "latch_loops", "partial_input_delay",
                    "partial_output_delay"):
            require(checks.get(key) == 0, "incomplete timing coverage: " + key)
        if context == "assumed":
            require(checks.get("no_input_delay") == 0 and checks.get("no_output_delay") == 0, "missing I/O delays")
        else:
            fixture = json.loads((folder / "fixture/fixture.json").read_text())
            top = "ntrpu_peer_multibuffer_" + record.get("variant", "compact")
            require(fixture.get("core_top", "ntrpu_peer_multibuffer_compact") == top, "wrong core variant")
            if "core_top" in summary:
                require(summary["core_top"] == top, "core variant summary mismatch")
            require(fixture["reserve"] == reserve and fixture["register_bits"]["input"] == 673, "wrong fixture")
            require(int(summary["probe_register_cells"]) == sum(fixture["register_bits"].values()), "lost boundary registers")
            require(checks.get("no_input_delay") == fixture["register_bits"]["input"], "unexpected unqualified inputs")
            require(checks.get("no_output_delay") == fixture["register_bits"]["output"], "unexpected unqualified outputs")
        xdc = (folder / "implemented.xdc").read_text()
        require(len(re.findall(r"^create_clock ", xdc, re.M)) == 1, "unexpected clocks")
        require(not re.search(r"^\s*set_(false_path|multicycle_path|clock_groups|max_delay|min_delay|disable_timing|case_analysis)\b",
                              xdc, re.M), "unreviewed timing exception")
        tokens = ("BUFGCE_X0Y60", "set_clock_uncertainty")
        if context == "assumed":
            tokens += ("HD.CLK_SRC", "set_input_delay", "set_output_delay")
        else:
            require("HD.CLK_SRC" not in xdc, "assumed clock left in registered fixture")
        for token in tokens:
            require(token in xdc, "missing applied constraint: " + token)
        resources = utilization(folder / "utilization.rpt")
        paths = list(csv.DictReader((folder / "paths.tsv").read_text().splitlines(), delimiter="\t"))
        scopes = ("all", "core", "ingress", "egress") if context == "registered" else ("all", "internal", "input", "output")
        require({(p["scope"], p["type"]) for p in paths} ==
                {(s, t) for s in scopes for t in ("max", "min")}, "missing path scope")
        for path in paths:
            path["slack_ns"] = float(path["slack_ns"])
        for delay, key in (("max", "wns_ns"), ("min", "whs_ns")):
            require(next(p["slack_ns"] for p in paths if p["scope"] == "all" and p["type"] == delay) == timing[key],
                    "path and summary disagree")
        counts = {k: int(v) for k, v in (line.split("\t") for line in (folder / "route_counts.tsv").read_text().splitlines())}
        routing = {}
        route_report = (folder / "route.rpt").read_text()
        for label in ("# of logical nets", "# of routable nets", "# of fully routed nets", "# of nets with routing errors"):
            found = re.search(re.escape(label) + r"\.+\s*:\s*(\d+)", route_report)
            require(found, "missing route status: " + label)
            routing[label] = int(found[1])
        if context == "registered":
            require(routing["# of routable nets"] == routing["# of fully routed nets"] and
                    routing["# of nets with routing errors"] == 0, "incomplete fixture routing")
            require(set(counts).issubset({"ROUTED", "INTRASITE", "HIERPORT"}), "unreviewed net route status")
        gaps = (folder / "route_gaps.txt").read_text()
        require(int(re.search(r"ROUTE_STRING_GAP_NETS (\d+)", gaps)[1]) == int(summary["route_string_gap_nets"]),
                "gap counts disagree")
        log = (output / f"reserve_{reserve}.console.log").read_text()
        require(not re.search(r"^ERROR:", log, re.M), "tool error")
        marker = "MULTIBUFFER_REGISTERED_ROUTE_EXPORTED" if context == "registered" else "MULTIBUFFER_COMPACT_ROUTE_EXPORTED"
        require(f"{marker} RESERVE={reserve}" in log, "missing completion marker")
        drc = (folder / "drc.rpt").read_text()
        match = re.search(r"Checks found:\s+(\d+)\b", drc)
        require(match, "missing DRC count")
        methodology = [dict(rule=rule, severity=severity, description=desc.strip(), count=int(count))
                       for rule, severity, desc, count in re.findall(
                           r"^\|\s*([A-Z]+-\d+)\s*\|\s*(\w+)\s*\|\s*([^|]+)\|\s*(\d+)",
                           (folder / "methodology.rpt").read_text(), re.M)]
        ports = []
        if context == "assumed":
            ports = list(csv.DictReader((folder / "port_context.tsv").read_text().splitlines(), delimiter="\t"))
            require(sum(bool(re.fullmatch(r"banks_used\[\d+\]", p["port"])) for p in ports) == 4 + reserve,
                    "checkpoint does not have the expected logical bank count")
        profiles.append(dict(reserve=reserve, resources=resources, timing=timing, timing_checks=checks,
                             paths=paths, route_status_counts=counts, routing=routing,
                             route_string_gap_nets=int(summary["route_string_gap_nets"]),
                             port_count=len(ports) if context == "assumed" else 1 + sum(fixture["register_bits"].values()),
                             ports_with_partpin_locations=sum(bool(p["HD.PARTPIN_LOCS"]) for p in ports),
                             drc_findings=int(match[1]), methodology=methodology,
                             warnings=dict(Counter(re.findall(r"^WARNING: \[([^]]+)\]", log, re.M))),
                             critical_warnings=dict(Counter(re.findall(r"^CRITICAL WARNING: \[([^]]+)\]", log, re.M))),
                             timing_target_met=all(timing[k] >= 0 for k in ("wns_ns", "whs_ns", "wpws_ns"))))
        if context == "registered":
            require(resources["BUFGCE"] == 1, "missing physical clock buffer")
            before_import = utilization(folder / "core_before_export.rpt")
            after_import = utilization(folder / "core_after_import.rpt")
            require(before_import == after_import, "core resources changed during EDIF transfer")
            profiles[-1].update(fixture=fixture, clock_route_status=summary["clock_route_status"],
                                route_directive=summary.get("route_directive", "Default"),
                                place_directive=summary.get("place_directive", "Default"),
                                resumed_placed=summary.get("resumed_placed") == "1",
                                core_register_cells=int(summary["core_register_cells"]),
                                probe_register_cells=int(summary["probe_register_cells"]),
                                core_resources_before_import=before_import,
                                core_resources_after_route=utilization(folder / "core_utilization.rpt"))
    limits = (["Assumed parent BUFG location; no physical source buffer in this block",
               "Proposed synchronous I/O budgets, not measured PCIe/HBM timing"] if context == "assumed" else
              ["Real BUFG and registered probe boundaries; fixture is not an AXI adapter",
               "Raw probe ports intentionally lack external timing constraints; not a board interface"])
    return dict(status="PHYSICAL_OOC_EVIDENCE_REVIEWED", context=context, profiles=profiles,
                batch_status=record["status"], reviewed_reserves=list(reserves),
                excluded_runs=[r for r in record["runs"] if r["reserve"] not in reserves],
                scoped_timing_target_met=all(p["timing_target_met"] for p in profiles),
                board_or_parent_closure=False,
                limits=limits + ["No PCIe/HBM parent floorplan, CDC, board I/O or integrated transport",
                                 "Route gaps, DRC and methodology must be read alongside slack; no waivers"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reserves", type=int, choices=(4, 8), nargs="+")
    args = parser.parse_args()
    result = review(args.run, args.reserves)
    with (args.run / "REVIEW.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
