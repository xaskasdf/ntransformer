"""Review the generic 128 GiB DDR simulation; not physical DIMM qualification."""
import argparse
import hashlib
import json
import pathlib
import re


def review(text, ecc=False, reset=False):
    # Vivado echoes Tcl source with '#'; strings in that source are not evidence.
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    required = [
        "NTRPU_DDR_SCOPE generic_128gib_two_slot_x4_ecc time_unit=ps",
        "NTRPU_DDR_SPARSE_COMPARE_PASS locations=36 address_bits=6..36",
        "NTRPU_DDR_MASKED_COMPARE_PASS locations=36",
        "NTRPU_DDR_CAPACITY_PASS locations=36 checked_bytes=5632",
    ]
    if ecc:
        required += [
            "NTRPU_DDR_ECC_REQUESTED",
            "NTRPU_DDR_ECC_CORRECTION_PASS bits=1 status=2 ce_count=1 irq=1",
            "NTRPU_DDR_ECC_DETECTION_PASS bits=2 status=1 rresp=2 irq=1",
            "NTRPU_DDR_ECC_PASS corrected=1 detected=1 clean_rewrites=2 irq_events=2",
        ]
    if reset:
        required += [
            "NTRPU_DDR_RESET_REQUESTED",
            "NTRPU_DDR_RESET_PASS interrupted_reads=1 aborted_beats=6 stale_responses=0 checked_bytes=640",
        ]
    errors = ["Expected exactly one marker: " + s for s in required if lines.count(s) != 1]
    diagnostic = re.compile(
        r"\b(?:ERROR|FATAL|FAILED|MISMATCH|VIOLATION|VIOLATED|INCORRECT_PARAMETER)\b"
        r"|NTRPU_DDR_INCOMPLETE|SIMULATIONS NOT SUPPORTED|TIMED\s+OUT"
        r"|timing\s+(?:check\s+)?fail", re.IGNORECASE)
    errors.extend(line for line in lines if diagnostic.search(line))
    rows = [s.removeprefix("NTRPU_DDR_METRICS ") for s in lines
            if s.startswith("NTRPU_DDR_METRICS ")]
    metrics = {}
    if len(rows) != 1:
        errors.append("Expected exactly one final metrics record")
    else:
        pairs = [re.fullmatch(r"([a-z0-9_]+)=(\d+)", field) for field in rows[0].split()]
        if not pairs or not all(pairs):
            errors.append("Malformed or unknown numeric metric")
        else:
            metrics = {m[1]: int(m[2]) for m in pairs}
            if len(metrics) != len(pairs):
                errors.append("Duplicate metric keys")
        keys = {"complete", "calibrated_ps", "first_axi_ps", "done_ps", "cycles",
                "writes", "reads", "checked_bytes", "w_stall", "r_stall",
                "rank0", "rank1", "rank2", "rank3"}
        if metrics.keys() != keys:
            errors.append("Missing or unexpected metrics")
        else:
            expected = {"complete": 1, "writes": 84 if reset else 68 if ecc else 64,
                        "reads": 102 if reset else 92 if ecc else 88,
                        "checked_bytes": 6464 if reset else 5824 if ecc else 5632}
            for key, value in expected.items():
                if metrics[key] != value:
                    errors.append(f"Unexpected {key}: {metrics[key]}, expected {value}")
            if not 0 < metrics["calibrated_ps"] <= metrics["first_axi_ps"] < metrics["done_ps"] < 2_000_000_000:
                errors.append("Invalid calibration/test timestamps or exceeded simulation budget")
            if metrics["cycles"] <= 0:
                errors.append("No observed UI cycles")
            for key in ["rank0", "rank1", "rank2", "rank3", "w_stall", "r_stall"]:
                if metrics[key] <= 0:
                    errors.append("Missing rank/backpressure coverage: " + key)
            for key in ["w_stall", "r_stall"]:
                if metrics[key] >= metrics["cycles"]:
                    errors.append("Impossible stall count: " + key)
            marker = f'NTRPU_DDR_CALIBRATED time_ps={metrics["calibrated_ps"]}'
            if lines.count(marker) != 1:
                errors.append("Missing or inconsistent calibration marker")
            finishes = [re.fullmatch(r"\$finish called at time : (\d+) ps : .*", s) for s in lines]
            finishes = [int(m[1]) for m in finishes if m]
            if finishes != [metrics["done_ps"]]:
                errors.append("Missing, early or inconsistent simulator finish")
            if reset:
                asserted = [re.fullmatch(r"NTRPU_DDR_RESET_ASSERT time_ps=(\d+) aborted_read_beats=6", s) for s in lines]
                recovered = [re.fullmatch(r"NTRPU_DDR_RESET_RECALIBRATED time_ps=(\d+) elapsed_ps=(\d+)", s) for s in lines]
                asserted, recovered = [m for m in asserted if m], [m for m in recovered if m]
                if len(asserted) != 1 or len(recovered) != 1:
                    errors.append("Missing or duplicate reset/recalibration timestamps")
                elif not (metrics["first_axi_ps"] < int(asserted[0][1]) < int(recovered[0][1]) < metrics["done_ps"]
                          and int(recovered[0][2]) == int(recovered[0][1]) - int(asserted[0][1])):
                    errors.append("Inconsistent reset/recalibration timestamps")
    return {
        "status": "PASS" if not errors else "FAIL", "errors": errors, "metrics": metrics,
        "warnings": [s for s in lines if re.match(r"WARNING\b", s, re.IGNORECASE)],
        "ecc_fault_test_required": ecc,
        "reset_recovery_required": reset,
        "scope": "Sparse generic two-slot x4+ECC 128 GiB simulation; not exhaustive capacity, exact DIMM, pinout or physical timing qualification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--ecc", action="store_true", help="Require single/double bit injection and recovery evidence")
    parser.add_argument("--reset", action="store_true", help="Require reset with outstanding read and recalibration (also needs --ecc)")
    args = parser.parse_args()
    if args.reset and not args.ecc:
        parser.error("--reset requires --ecc")
    data = args.log.read_bytes()
    result = review(data.decode("utf8", errors="replace"), args.ecc, args.reset)
    result.update(log=str(args.log), log_sha256=hashlib.sha256(data).hexdigest())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json:
        with args.json.open("x", encoding="utf8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
