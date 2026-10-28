"""Fail-closed review of the isolated AMD XDMA dma_test0 simulation log."""
import argparse
import hashlib
import json
import pathlib
import re


def review(text, expected_bytes, legacy_fs=False):
    required = [
        "Transaction Link Is Up...",
        "Check Max Link Speed = 8.0GT/s - PASSED",
        "*** H2C Transfer Data MATCHES ***",
        "*** C2H Transfer Data MATCHES ***",
        "XDMA H2C Test Completed Successfully",
        "XDMA C2H Test Completed Successfully",
        "NTRPU_PCIE_MONITOR_COMPLETE",
    ]
    errors = ["Missing marker: " + marker for marker in required if marker not in text]
    version2 = "NTRPU_PCIE_MONITOR_FORMAT version=2 time_unit=ps" in text
    if legacy_fs:
        if version2 or "Time resolution is 1 fs" not in text:
            errors.append("Legacy conversion requires v1 monitor and explicit 1 fs simulator resolution")
    elif not version2:
        errors.append("Unknown monitor time format; v1 requires explicit --legacy-time-format-fs")
    if not re.search(r"Check Negotiated Link Width = 5'h0*10 - PASSED", text):
        errors.append("No confirmed negotiated x16 link")
    diagnostics = [line for line in text.splitlines() if re.search(
        r"\b(?:FATAL|ERROR|FAILED|MISMATCH|TIMED\s+OUT)\b|NTRPU_PCIE_MONITOR_INCOMPLETE",
        line, re.IGNORECASE)]
    errors.extend(diagnostics)
    rows = re.findall(r"^NTRPU_PCIE_METRICS (.+)$", text, re.MULTILINE)
    metrics = {}
    if len(rows) != 1:
        errors.append("Expected exactly one final metrics record")
    else:
        metrics = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", rows[0])}
        needed = ["cycles", "clock_ps", "link_up_ps", "wbytes", "wbeats", "rbeats",
                  "aw", "ar", "b", "w_stall", "r_stall", "first_write_ps",
                  "last_write_ps", "first_read_ps", "last_read_ps"]
        if any(key not in metrics for key in needed):
            errors.append("Missing or unknown numeric metrics")
        else:
            if metrics["wbytes"] != expected_bytes:
                errors.append("Unexpected H2C payload byte count")
            if any(metrics[key] <= 0 for key in ["clock_ps", "link_up_ps", "aw", "ar", "rbeats"]):
                errors.append("Missing link, clock or AXI activity")
            if metrics["b"] != metrics["aw"]:
                errors.append("AXI writes have missing/extra responses")
            if metrics["wbeats"] != (expected_bytes + 63) // 64:
                errors.append("Unexpected number of aligned AXI write beats")
            for direction in ["write", "read"]:
                first, last = metrics[f"first_{direction}_ps"], metrics[f"last_{direction}_ps"]
                if not 0 < first <= last:
                    errors.append("Invalid " + direction + " timestamps")
    raw_metrics = dict(metrics)
    if legacy_fs:
        for key in metrics:
            if key.endswith("_ps"):
                if metrics[key] % 1000:
                    errors.append("Nonintegral legacy fs to ps conversion: " + key)
                metrics[key] //= 1000
    if metrics and metrics.get("clock_ps") != 4000:
        errors.append("Expected 250 MHz / 4000 ps AXI user clock")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "metrics": metrics, "raw_metrics": raw_metrics,
            "time_conversion": "v1 %t values in fs divided by 1000" if legacy_fs else "v2 decimal module time in ps",
            "expected_payload_bytes": expected_bytes,
            "scope": "Isolated XDMA example; not DDR, sustained throughput or physical hardware"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--expected-bytes", type=int, default=16)
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--legacy-time-format-fs", action="store_true",
                        help="For the original v1 snapshot only: correct %%t display at 1 fs resolution")
    args = parser.parse_args()
    if args.expected_bytes <= 0:
        parser.error("Expected bytes must be positive")
    data = args.log.read_bytes()
    result = review(data.decode("utf8", errors="replace"), args.expected_bytes, args.legacy_time_format_fs)
    result.update(log=str(args.log), log_sha256=hashlib.sha256(data).hexdigest())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json:
        with args.json.open("x", encoding="utf8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
