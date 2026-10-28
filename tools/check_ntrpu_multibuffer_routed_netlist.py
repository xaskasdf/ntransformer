"""Replay a routed compact core and compare its trace with synthesized evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from check_ntrpu_multibuffer_route import digest, require
from check_ntrpu_peer_multibuffer import replay


def review(root, routed_profile, baseline_log, reserve):
    output = root / "output"
    record = json.loads((output / "result.json").read_text())
    require(record["status"] == "EXPORTED_PENDING_REVIEW", "incomplete simulation")
    require(record["variant"] in ("compact", "control", "decoded", "fillstage"), "wrong variant")
    require(record["sources"] == record["sources_after"], "sources changed")
    for name, expected in record["sources"].items():
        require(digest(root / "inputs" / name) == expected, "source hash: " + name)
    for name, expected in json.loads((output / "FILES.json").read_text()).items():
        require(digest(output / name) == expected, "output hash: " + name)
    selected = [run for run in record["runs"] if run["reserve"] == reserve]
    require(len(selected) == 1, "missing or duplicate profile")
    run = selected[0]
    require(run["returncode"] == 0 and not run["timeout"], "failed simulation")
    folder = output / f"reserve_{reserve}"
    netlist_hash = digest(routed_profile / "multibuffer_funcsim.v")
    require(digest(folder / "input_netlist.v") == netlist_hash, "wrong routed core")
    recorded = [value for name, value in record["netlists"].items()
                if name.endswith(f"/reserve_{reserve}/multibuffer_funcsim.v")]
    require(recorded == [netlist_hash], "netlist provenance mismatch")
    log = (folder / "simulation.log").read_text()
    console = (output / f"reserve_{reserve}.console.log").read_text()
    for text in (log, console):
        require(not re.search(r"^\s*(ERROR:|Fatal:|FAIL )", text, re.M | re.I), "tool failure")
        require(not re.search(r"^CRITICAL WARNING:", text, re.M), "unreviewed critical warning")
    require(f"MULTIBUFFER_NETLIST_EXPORTED RESERVE={reserve}" in console, "missing export marker")
    result = replay(log, (2, reserve, 64, 16, 4), 1, check_bank_leases=False)
    require(result == run["pin_replay"], "downloaded replay differs")
    trace = lambda text: [line for line in text.splitlines()
                          if line.startswith(("TRACE ", "CASE ", "PASS "))]
    actual, baseline = trace(log), trace(baseline_log.read_text())
    require(actual and actual == baseline, "trace differs from synthesized compact core")
    return dict(status="ROUTED_CORE_PUBLIC_PIN_REVIEW_PASS", reserve=reserve,
                pin_replay=result, trace_identical_to_synthesized_compact=True,
                trace_lines=len(actual), core_netlist_sha256=netlist_hash,
                warnings=dict(Counter(re.findall(r"^WARNING: \[([^]]+)\]", console, re.M))),
                limits=["Functional simulation, 10 ns clock, no SDF",
                        "Only core, not timing fixture as an AXI adapter",
                        "No internal lease observation in netlist"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("routed_profile", type=Path)
    parser.add_argument("baseline_log", type=Path)
    parser.add_argument("--reserve", type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    result = review(args.run, args.routed_profile, args.baseline_log, args.reserve)
    with (args.run / "REVIEW.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
