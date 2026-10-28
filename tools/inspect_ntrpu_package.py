#!/usr/bin/env python3
"""Compare NTRPU bank reservations with an offline AMD package-pin ZIP."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

from check_ntrpu_io_plan import DEFAULT_PLAN, check_profile


PACKAGE_FILES = {
    ("XCVU5P", "FLVA2104"): "usaall/xcvu5pflva2104pkg.txt",
    ("XCVU9P", "FLGA2104"): "usaall/xcvu9pflga2104pkg.txt",
}
FIELDS = ("ball", "function", "memory_byte_group", "bank", "io_type", "slr")


def parse_package(data, device):
    text = data.decode("utf-8-sig")
    metadata = {}
    for key in ("Device", "Date", "Revision", "Status"):
        match = re.search(rf"^--\s+{key}\s*:\s*(.+)$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing package metadata: {key}")
        metadata[key.lower()] = match.group(1).strip()
    if metadata["device"].lower() != device.lower():
        raise ValueError("Package header does not match selected device/package")
    pins = []
    seen = set()
    for line in text.splitlines():
        if not re.match(r"^[A-Z]+[0-9]+\s", line):
            continue
        values = line.split()
        if len(values) != len(FIELDS):
            raise ValueError(f"Unexpected pin row format: {values[0]}")
        pin = dict(zip(FIELDS, values))
        if pin["ball"] in seen:
            raise ValueError(f"Duplicate package ball: {pin['ball']}")
        seen.add(pin["ball"])
        for key in ("bank", "slr"):
            pin[key] = None if pin[key] == "NA" else int(pin[key])
        pins.append(pin)
    total = re.search(r"Total Number of Pins\s+(\d+)", text)
    if total is None or int(total.group(1)) != 2104 or len(pins) != 2104:
        raise ValueError("Expected exactly 2104 unique balls and a matching footer")
    return metadata, pins


def inspect_package(profile, data, include_pins=False):
    base = check_profile(profile)
    if not base["documentary_checks_passed"]:
        raise ValueError("Invalid documentary plan: " + "; ".join(base["errors"]))
    metadata, pins = parse_package(data, profile["device"] + profile["package"])
    hp = [p for p in pins if p["io_type"] == "HP" and p["function"].startswith("IO_")]
    banks = sorted({p["bank"] for p in hp})
    expected_banks = {b for group in profile["bonded_hp_groups"] for b in group["banks"]}
    errors = []
    if set(banks) != expected_banks:
        errors.append("Bonded HP bank set differs from the documentary plan")
    if len(hp) != 832:
        errors.append(f"Expected 832 HP user I/O; found {len(hp)}")
    bank_summary = []
    for bank in banks:
        members = [p for p in hp if p["bank"] == bank]
        slrs = sorted({p["slr"] for p in members}, key=str)
        if len(members) != 52 or len(slrs) != 1 or slrs[0] is None:
            errors.append(f"Bank {bank}: unexpected HP count or SLR data")
        bank_summary.append({
            "bank": bank, "hp_user_io": len(members), "package_slrs": slrs,
            "gc_pins": [p for p in members if "GC" in p["function"].split("_")],
        })
    channels = []
    for channel in profile["ddr_channels"]:
        slrs = {p["slr"] for p in hp if p["bank"] in channel["banks"]}
        if len(slrs) != 1 or None in slrs:
            errors.append(f"DDR{channel['id']}: banks do not share a known package SLR")
        channels.append({"id": channel["id"], "banks": channel["banks"],
                         "package_slrs": sorted(slrs, key=str)})
    gt = []
    for bank in profile["pcie_gen3_x16"]["reserved_gt_quads"]:
        members = [p for p in pins if p["bank"] == bank]
        counts = {kind: sum(bool(re.fullmatch(rf"MGTY{kind}[0-3]_{bank}", p["function"]))
                            for p in members) for kind in ("TXP", "TXN", "RXP", "RXN")}
        if any(count != 4 for count in counts.values()):
            errors.append(f"GT quad {bank}: expected four differential TX/RX lanes")
        gt.append({"quad": bank, "signal_counts": counts,
                   "reference_clock_candidates": [p for p in members if p["function"].startswith("MGTREFCLK")]})
    report = {
        "status": "package_inventory_checked_not_vivado_validated",
        "profile": profile["id"], "package_metadata": metadata,
        "package_file_sha256": hashlib.sha256(data).hexdigest(),
        "package_ball_count": len(pins), "hp_user_io_count": len(hp),
        "io_type_counts_including_non_user_pins": dict(Counter(p["io_type"] for p in pins)),
        "inventory_checks_passed": not errors, "errors": errors,
        "ddr_channels": channels, "hp_banks": bank_summary, "pcie_quads": gt,
        "configuration_bank0": [p for p in pins if p["io_type"] == "CONFIG" and p["bank"] == 0],
        "pcie_reset_candidates": [p for p in pins if "PERSTN" in p["function"]],
        "limitations": ["No MIG, DRC, timing, voltage or lane-order validation",
                        "GC and MGTREFCLK entries are candidates, not assigned clocks",
                        "Package SLR numbers must still be compared with Vivado",
                        "PERSTN and DDR use of bank 65 require a pin-level review"],
    }
    if include_pins:
        report["connection_worksheet"] = [dict(p, rtl_port=None, pcb_net=None,
                                                assignment_status="unassigned") for p in pins]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Official AMD usaall.zip, already downloaded")
    parser.add_argument("profile", help="Profile ID from io_plan.json")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--include-pins", action="store_true", help="Include unassigned ECAD worksheet")
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are refused")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        if plan["schema_version"] != 1 or plan["status"] != "preliminary_not_vivado_validated":
            raise ValueError("Expected a schema-1 preliminary I/O plan")
        profiles = [p for p in plan["profiles"] if p["id"] == args.profile]
        if len(profiles) != 1:
            raise ValueError("Profile must match exactly one plan entry")
        profile = profiles[0]
        member = PACKAGE_FILES[(profile["device"], profile["package"])]
        with ZipFile(args.archive) as archive:
            if archive.getinfo(member).file_size > 2_000_000:
                raise ValueError("Unexpectedly large package file")
            report = inspect_package(profile, archive.read(member), args.include_pins)
        with args.archive.open("rb") as stream:
            archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        report["source"] = {"archive_member": member, "archive_sha256": archive_hash}
        output = json.dumps(report, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(output)
            print(f"Wrote {args.output}; inventory_checks_passed={report['inventory_checks_passed']}")
        else:
            print(output, end="")
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as error:
        parser.exit(2, f"Package inspection failed: {error}\n")
    return 0 if report["inventory_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
