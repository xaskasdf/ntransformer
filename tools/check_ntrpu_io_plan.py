#!/usr/bin/env python3
"""Check documentary bank reservations, not MIG placement or electrical legality."""

import argparse
import json
from pathlib import Path


DEFAULT_PLAN = Path(__file__).resolve().parents[1] / "docs/fpga/board/io_plan.json"
# PG213 lane-0 choices for x16 in these exact packages; not a full device catalog.
PCIE_X16 = {
    ("XCVU5P", "FLVA2104"): ("PCIE4_X1Y0", 227, [224, 225, 226, 227]),
    ("XCVU9P", "FLGA2104"): ("PCIE4_X1Y2", 227, [224, 225, 226, 227]),
}


def bank_list(value):
    if not isinstance(value, list) or any(type(b) is not int or b < 0 for b in value):
        raise ValueError("Bank lists must contain non-negative integers")
    return value


def check_profile(profile):
    errors = []
    locations = {}
    for group in profile["bonded_hp_groups"]:
        location = (group["column"], group["physical_slr_zone"])
        for bank in bank_list(group["banks"]):
            if bank in locations:
                errors.append(f"Duplicate inventory bank {bank}")
            locations[bank] = location

    channels = profile["ddr_channels"]
    ids = [channel["id"] for channel in channels]
    if any(type(i) is not int for i in ids) or sorted(ids) != [0, 1, 2, 3]:
        errors.append("This study requires exactly four channels numbered 0..3")
    used = set()
    for channel in channels:
        banks = bank_list(channel["banks"])
        label = f"DDR{channel['id']}"
        if len(banks) != 3 or len(set(banks)) != 3:
            errors.append(f"{label}: this study reserves three distinct banks")
        if banks and sorted(banks) != list(range(min(banks), min(banks) + len(banks))):
            errors.append(f"{label}: bank skipping is not allowed")
        unknown = set(banks) - locations.keys()
        if unknown:
            errors.append(f"{label}: banks absent from bonded inventory: {sorted(unknown)}")
        if len({locations[b] for b in banks if b in locations}) > 1:
            errors.append(f"{label}: crosses a column or physical SLR zone")
        if used.intersection(banks):
            errors.append(f"{label}: bank sharing is outside this study")
        used.update(banks)

    free = bank_list(profile["unassigned_hp_banks"])
    if len(set(free)) != len(free) or set(free) != locations.keys() - used:
        errors.append("Unassigned bank list must equal bonded inventory minus DDR reservations")
    config = profile["configuration"]
    if type(config["tandem"]) is not bool:
        raise ValueError("configuration.tandem must be a boolean")
    if 65 in used and (config["tandem"] or config["bank65_configuration_signals"]):
        errors.append("Bank 65 DDR conflicts with Tandem or configuration signals in this plan")
    if config["dedicated_bank"] != 0 or config["mode"] != "jtag_or_qspi_x4_internal_configuration_clock":
        errors.append("Configuration mode is outside this bank-0-only study")

    expected = PCIE_X16.get((profile["device"], profile["package"]))
    pcie = profile["pcie_gen3_x16"]
    actual = (pcie["block"], pcie["lane0_gt_quad"], pcie["reserved_gt_quads"])
    if expected is None or actual != expected:
        errors.append("PCIe reservation differs from the documented x16 package choice")
    return {
        "profile": profile["id"],
        "documentary_checks_passed": not errors,
        "reserved_ddr_banks": sorted(used),
        "unassigned_hp_banks": free,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        if plan["schema_version"] != 1 or plan["status"] != "preliminary_not_vivado_validated":
            raise ValueError("Expected schema 1 and preliminary_not_vivado_validated status")
        profiles = plan["profiles"]
        if not profiles or len({p["id"] for p in profiles}) != len(profiles):
            raise ValueError("Expected nonempty profiles with unique IDs")
        results = [check_profile(profile) for profile in profiles]
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Invalid plan: {error}\n")
    print(json.dumps({
        "status": "preliminary_not_vivado_validated",
        "scope": "Consistency against manually transcribed bank groups and documented PCIe choices",
        "limitations": [
            "Does not verify the transcription against Vivado or AMD package files",
            "Does not check individual balls, byte lanes, clocks, voltages, capacity or timing",
            "Does not prove DDR4/PCIe coexistence or hardware operation",
        ],
        "profiles": results,
    }, indent=2))
    return 0 if all(result["documentary_checks_passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
