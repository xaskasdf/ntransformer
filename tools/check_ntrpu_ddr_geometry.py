"""Check a declared DDR4 wrapper profile, not DIMM functionality."""
import argparse
import hashlib
import json
import pathlib
import re


def review(text, profile="generic-dual"):
    if profile not in {"generic-dual", "catalog-3ds128"}:
        raise ValueError("Unknown memory profile")
    expected = {"ROW_WIDTH": "18", "COL_WIDTH": "10", "BANK_WIDTH": "2",
                "BANK_GROUP_WIDTH": "2", "RANKS": "4", "CS_WIDTH": "4",
                "S_HEIGHT": "1", "DQ_WIDTH": "72", "APP_DATA_WIDTH": "512",
                "C_S_AXI_ADDR_WIDTH": "37", "ECC": '"ON"',
                "SLOT0_CONFIG": "8'b0000_0011", "SLOT1_CONFIG": "8'b0000_1100"}
    if profile == "catalog-3ds128":
        expected.update(ROW_WIDTH="17", RANKS="2", CS_WIDTH="2", S_HEIGHT="4",
                        SLOT0_CONFIG="{{(8-CS_WIDTH){1'b0}},{CS_WIDTH{1'b1}}}",
                        SLOT1_CONFIG="8'b0000_0000")
    found = {}
    errors = []
    for name, value in expected.items():
        matches = re.findall(r"^\s*parameter\s+(?:integer\s+)?" + name +
                             r"\s*=\s*([^\n]+)", text, re.MULTILINE)
        if len(matches) != 1:
            errors.append(f"Expected one generated parameter {name}")
        else:
            found[name] = matches[0].strip().removesuffix(",").rstrip()
            if found[name] != value:
                errors.append(f"{name}: expected {value}, got {found[name]}")
    capacity = None
    if not errors:
        address_cells = 1 << sum(int(found[k]) for k in
                                ["ROW_WIDTH", "COL_WIDTH", "BANK_WIDTH", "BANK_GROUP_WIDTH"])
        # ECC is ON: a burst of eight transfers contains APP_DATA_WIDTH bits
        # of payload, so each transfer contains APP_DATA_WIDTH/64 useful bytes.
        capacity = address_cells * int(found["RANKS"]) * int(found["S_HEIGHT"]) * (int(found["APP_DATA_WIDTH"]) // 64)
        if capacity != 137438953472 or capacity != 1 << int(found["C_S_AXI_ADDR_WIDTH"]):
            errors.append("Derived useful capacity is not 128 GiB / 37 address bits")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "profile": profile,
            "generated_parameters": found, "derived_useful_capacity_bytes": capacity,
            "scope": "Generated wrapper geometry only; no calibration, alias or DIMM qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wrapper", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--profile", choices=["generic-dual", "catalog-3ds128"], default="generic-dual")
    args = parser.parse_args()
    data = args.wrapper.read_bytes()
    result = review(data.decode("utf8"), args.profile)
    result.update(wrapper=str(args.wrapper), wrapper_sha256=hashlib.sha256(data).hexdigest())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json:
        with args.json.open("x", encoding="utf8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
