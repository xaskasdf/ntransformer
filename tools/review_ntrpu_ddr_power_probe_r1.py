"""Transcribe native power reports, retaining missing data and diagnostic limits."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import check_ntrpu_ddr_power_probe_r1 as checker


def tables(evidence):
    text = evidence["files"]["reports/power.rpt"]["text"]
    normalize = lambda value: " ".join(value.split())
    text_rows = {tuple(normalize(v) for v in line.strip().strip("|").split("|"))
                 for line in text.splitlines() if line.startswith("|")}
    root = ET.fromstring(evidence["files"]["reports/power.xml"]["text"])
    result = {}
    checked = 0
    for section in root.iter("section"):
        rows = []
        for table in section.findall("table"):
            for row in table.findall("tablerow"):
                values = [normalize(cell.attrib["contents"]) for cell in row]
                assert tuple(values) in text_rows, (section.attrib["title"], values)
                rows.append(values)
                checked += 1
        if rows:
            assert section.attrib["title"] not in result
            result[section.attrib["title"]] = rows
    return result, checked, root.attrib


def number(value):
    if value is None or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    review = checker.review(evidence)
    review["rejected_controls"] = checker.controls(evidence)
    raw, checked, xml_header = tables(evidence)
    summary = dict(raw["Summary"])
    environment = dict(raw["Environment"])
    supply = raw["Power Supply Summary"]
    assert supply[0] == ["Source", "Voltage (V)", "Total (A)", "Dynamic (A)",
                         "Static (A)", "Powerup (A)", "Budget (A)", "Margin (A)"]
    keys = ["voltage_v", "total_a", "dynamic_a", "static_a", "powerup_a", "budget_a", "margin_a"]
    rails = [dict(source=row[0], **dict(zip(keys, map(number, row[1:])))) for row in supply[1:]]
    power = evidence["files"]["reports/power.rpt"]["text"]
    header = dict(re.findall(r"^\|\s*([^:|]+?)\s*:\s*(.*?)\s*$", power, re.M))
    advisory = [[v.strip() for v in line.strip().strip("|").split("|")]
                for line in power.splitlines() if line.startswith("|")]
    resets = [row for row in advisory if len(row) == 8 and row[1] == "Excessive Set/Reset/Preset/Clear"]
    metrics = dict(
        total_on_chip_w=number(summary.get("Total On-Chip Power (W)")),
        dynamic_w=number(summary.get("Dynamic (W)")),
        device_static_w=number(summary.get("Device Static (W)")),
        confidence=summary.get("Confidence Level"),
        effective_tja_c_per_w=number(summary.get("Effective TJA (C/W)")),
        junction_temperature_c=number(summary.get("Junction Temperature (C)")),
        max_ambient_c=number(summary.get("Max Ambient (C)")),
        ambient_temperature_c=number(environment.get("Ambient Temp (C)")),
        board_temperature_c=number(environment.get("Board Temperature (C)")),
        airflow_lfm=number(environment.get("Airflow (LFM)")),
        theta_sa_c_per_w=number(environment.get("ThetaSA (C/W)")),
        heat_sink=environment.get("Heat Sink"),
        board_selection=environment.get("Board Selection"),
        board_layers=environment.get("# of Board Layers"),
        design_power_budget_w=number(summary.get("Design Power Budget (W)")),
        power_budget_margin_w=number(summary.get("Power Budget Margin (W)")),
        rails=rails,
    )
    review.update(
        run=args.run, reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest(),
        metrics=metrics, report_header=header, xml_header=xml_header,
        raw_tables=raw, text_xml_rows_compared=checked,
        activity=dict(mode="inherited/default; vector-less propagation", new_saif_loaded=False,
                      simulation_activity_file=None, matched_nets=None,
                      annotated_nets_percent=None, measured_window_seconds=None,
                      confidence_rows=raw["Confidence Level"]),
        reset_advisory=dict(excessive_reset_rows=len(resets), examples=resets[:6]),
        scope="One generic routed DDR4 example; no board power or exact DIMM qualification",
        no_extrapolation_to_four_channels=True,
    )
    assert summary["Simulation Activity File"] == "---" and summary["Design Nets Matched"] == "NA"
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(review, stream, indent=2);stream.write("\n")
    print(json.dumps(dict(status=review["status"], run=args.run, metrics=metrics,
                          text_xml_rows_compared=checked, reset_advisory_rows=len(resets),
                          controls=len(review["rejected_controls"]))), flush=True)


if __name__ == "__main__":
    main()
