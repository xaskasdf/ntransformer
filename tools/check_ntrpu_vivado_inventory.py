#!/usr/bin/env python3
"""Compare actual Vivado report_property exports against an AMD package worksheet."""
import argparse
import hashlib
import json
from pathlib import Path
import re

# Reviewed on xcvu5p-flva2104-2-i, Vivado 2026.1. Other representations
# require review rather than silently accepting non-HP differences.
DEDICATED_BANK_ASSOCIATIONS = {
    "MGTRREF_RN": 231, "MGTAVTTRCAL_RN": 231,
    "MGTRREF_RS": 226, "MGTAVTTRCAL_RS": 226,
    "MGTRREF_LS": 126, "MGTAVTTRCAL_LS": 126,
    "MGTRREF_LN": 131, "MGTAVTTRCAL_LN": 131,
    "RSVDGND": 0, "VREFN": 0, "VREFP": 0, "DXN": 0, "DXP": 0,
    "POR_OVERRIDE": 0, "VBATT": 0,
}

def read_objects(path):
    objects = []
    for block in path.read_text(encoding="utf-8").split("Property"):
        obj = {}
        for line in block.splitlines():
            match = re.fullmatch(r"(\S+)\s+\S+\s+(?:true|false)\s*(.*)", line)
            if match:
                obj[match[1]] = match[2].strip()
        if "NAME" in obj:
            objects.append(obj)
    names = [obj["NAME"] for obj in objects]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate object names")
    return {obj["NAME"]: obj for obj in objects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("worksheet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    worksheet = json.loads(args.worksheet.read_text(encoding="utf-8"))
    pins = read_objects(args.inventory / "package_pins.rpt")
    banks = read_objects(args.inventory / "iobanks.rpt")
    official = {pin["ball"]: pin for pin in worksheet["connection_worksheet"]}
    errors, differences = [], []
    if pins.keys() != official.keys():
        errors.append({"missing_balls": sorted(official.keys()-pins.keys()),
                       "extra_balls": sorted(pins.keys()-official.keys())})
    hp_count = 0
    for ball in sorted(pins.keys() & official.keys()):
        actual, expected = pins[ball], official[ball]
        if actual["PIN_FUNC"] != expected["function"]:
            errors.append({"ball": ball, "pin_function": [actual["PIN_FUNC"], expected["function"]]})
        actual_bank = int(actual["BANK"]) if actual.get("BANK") else None
        if actual_bank != expected["bank"]:
            explained = (worksheet["profile"] == "vu5p_flva2104_bench"
                         and expected["bank"] is None
                         and expected["function"] in DEDICATED_BANK_ASSOCIATIONS
                         and actual_bank == DEDICATED_BANK_ASSOCIATIONS[expected["function"]])
            differences.append({"ball": ball, "function": expected["function"],
                                "bank_vivado": actual_bank, "bank_package": expected["bank"],
                                "reviewed_dedicated_pin_association": explained})
            if not explained:
                errors.append({"ball": ball, "unreviewed_bank_difference": True})
        if expected["io_type"] == "HP" and expected["function"].startswith("IO_"):
            hp_count += 1
            if actual_bank != expected["bank"]:
                errors.append({"ball": ball, "hp_bank_mismatch": True})
            if int(banks[str(expected["bank"])]["SLR_INDEX"]) != expected["slr"]:
                errors.append({"ball": ball, "hp_slr_mismatch": True})
    if len(pins) != 2104 or hp_count != 832:
        errors.append("Expected 2104 balls and 832 HP I/O")
    result = {"inventory_matches": not errors, "ball_count": len(pins), "hp_io_count": hp_count,
              "errors": errors, "bank_representation_differences": differences,
              "hp_banks": {name: bank for name, bank in banks.items() if bank.get("BANK_TYPE") == "BT_HIGH_PERFORMANCE"},
              "worksheet_sha256": hashlib.sha256(args.worksheet.read_bytes()).hexdigest(),
              "limitations": "Package inventory comparison only; no controller placement or electrical validation."}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"inventory_matches": not errors, "errors": errors[:10], "bank_differences": len(differences), "balls": len(pins), "hp_io": hp_count}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
