"""Review bounded 768-byte or 64-bit chained XDMA simulations (not board DDR)."""
import argparse
import hashlib
import json
import pathlib
import re


def review(text, require_stalls=False, require_host64=False, require_all_stalls=False):
    errors = []
    host64 = require_host64 or "NTRPU_PCIE_HOST64_FORMAT" in text
    payload = 1536 if host64 else 768
    # Vivado's outer console echoes Tcl source with #/## prefixes, including
    # unevaluated `error` branches. These are code, not runtime diagnostics.
    informational = "Review numerical/status markers and error logs; this marker alone is not a PASS."
    lines = text.splitlines()
    runtime = [line for line in lines if not line.lstrip().startswith("#") and line != informational]
    format_marker = ("NTRPU_PCIE_HOST64_FORMAT version=1 time_unit=ps payload=1536" if host64 else
                     "NTRPU_PCIE_MULTI_FORMAT version=1 time_unit=ps payload=768")
    required = [format_marker,
                "NTRPU_MULTI_CONFIG ep_mps=256 ep_mrrs=256",
                "NTRPU_MULTI_H2C_BYTES_OBSERVED", "NTRPU_MULTI_C2H_BYTES_OBSERVED",
                "Check Max Link Speed = 8.0GT/s - PASSED", "Transaction Link Is Up..."]
    for marker in required:
        if text.count(marker) != 1:
            errors.append("Expected exactly one marker: " + marker)
    if not re.search(r"Check Negotiated Link Width = 5'h0*10 - PASSED", text):
        errors.append("Missing Gen3 x16 width confirmation")
    errors.extend(line for line in runtime if re.search(
        r"\b(?:FATAL|ERROR|FAILED|MISMATCH|TIMED\s+OUT)\b|NTRPU_PCIE_MULTI_INCOMPLETE|Received an invalid frame",
        line, re.IGNORECASE))

    def record(prefix, keys):
        rows = re.findall(r"^" + prefix + r" (.+)$", text, re.MULTILINE)
        if len(rows) != 1:
            errors.append("Expected exactly one record: " + prefix)
            return {}
        values = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", rows[0]))
        if set(keys) - values.keys():
            errors.append("Missing numeric fields: " + prefix)
        return values

    cq = record("NTRPU_CQ_COMPLETE", ["c2h_bytes", "c2h_packets", "h2c_bytes", "h2c_requests",
                "descriptors", "beats", "stalls", "first_c2h_ps", "last_c2h_ps"])
    axi = record("NTRPU_PCIE_MULTI_COMPLETE", ["wbytes", "rbytes", "wbeats", "rbeats", "aw", "ar", "b",
                 "w_stalls", "r_stalls", "clock_ps", "link_ps", "first_w_ps", "last_w_ps",
                 "first_r_ps", "last_r_ps", "rp_mps_raw", "rp_mrrs_raw"])
    for key, expected in {"c2h_bytes": payload, "h2c_bytes": payload, "c2h_packets": 8 if host64 else 3,
                          "h2c_requests": 8 if host64 else 3, "descriptors": 4 if host64 else 2,
                          "beats": 44 if host64 else 20}.items():
        if cq.get(key) != expected:
            errors.append(f"CQ {key}: expected {expected}")
    for key, expected in {"wbytes": payload, "rbytes": payload, "wbeats": payload//64,
                          "rbeats": payload//64, "clock_ps": 4000}.items():
        if axi.get(key) != expected:
            errors.append(f"AXI {key}: expected {expected}")
    if not 0 < axi.get("aw", 0) == axi.get("b", -1) or axi.get("ar", 0) <= 0:
        errors.append("Missing AXI bursts/responses")
    if not (0 < axi.get("link_ps", 0) < axi.get("first_w_ps", 0) <= axi.get("last_w_ps", 0)
            < axi.get("first_r_ps", 0) <= axi.get("last_r_ps", 0)
            <= cq.get("last_c2h_ps", 0)):
        errors.append("Invalid simulation time ordering")
    requests = [(int(t), int(a, 16), int(n)) for t, a, n in re.findall(
        r"^NTRPU_CQ_REQUEST time_ps=\d+ type=(\d+) address=([0-9a-fA-F]+) bytes=(\d+)$", text, re.MULTILINE)]
    expected = [(0, 0x100, 32), (0, 0x400, 256), (0, 0x500, 256), (0, 0x600, 256),
                (0, 0x300, 32), (1, 0x800, 256), (1, 0x900, 256), (1, 0xa00, 256)]
    host = {}
    if host64:
        desc = [(0, (high << 32) + (0x100 if high < 3 else 0x300), 32) for high in range(1, 5)]
        data_requests = [(0 if high < 3 else 1, (high << 32)+0xfc0+offset, size)
                         for high in range(1, 5) for offset, size in [(0, 64), (64, 256), (320, 256), (576, 192)]]
        if sorted(requests) != sorted(desc+data_requests) or [r for r in requests if r not in desc] != data_requests:
            errors.append("Unexpected 64-bit request addresses, page split or descriptor set")
        required_host = ["NTRPU_HOST_MEMORY_INITIALIZED regions=8 descriptors=4 payload=1536 address_bits=64",
                         "NTRPU_HOST64_DESCRIPTORS direction=H2C count=2",
                         "NTRPU_HOST64_DESCRIPTORS direction=C2H count=2"]
        for marker in required_host:
            if text.count(marker) != 1:
                errors.append("Expected host marker: " + marker)
        host = record("NTRPU_HOST_MEMORY_COMPLETE", ["read_bytes", "written_bytes", "regions", "descriptors"])
        if host != {"read_bytes": 1664, "written_bytes": 1536, "regions": 8, "descriptors": 4}:
            errors.append("Incomplete host memory read/write coverage")
    elif requests != expected:
        errors.append("Unexpected request sequence, address or length")
    gates = {}
    control = {}
    all_stalls = require_all_stalls or "NTRPU_PCIE_STALLS version=2" in text
    if require_stalls or all_stalls or "NTRPU_PCIE_STALLS" in text or "NTRPU_AXI_GATE_COMPLETE" in text:
        config = ("NTRPU_PCIE_STALLS version=2 channels=AW,W,B,AR,R aw_delay=7 ar_delay=11 b_delay=13" if all_stalls
                  else "NTRPU_PCIE_STALLS version=1 channels=W,R")
        if text.count(config) != 1 or len(re.findall(r"^NTRPU_PCIE_STALLS ", text, re.M)) != 1:
            errors.append("Missing or duplicated AXI stall configuration")
        for channel in ("W", "R"):
            gate = record("NTRPU_AXI_GATE_COMPLETE channel=" + channel,
                          ["beats", "source_stalls", "sink_stalls"])
            gates[channel] = gate
            if gate.get("beats") != payload//64 or gate.get("source_stalls", 0) <= 0:
                errors.append("Missing AXI " + channel + " stall/beat coverage")
        if gates["W"].get("source_stalls") != axi.get("w_stalls"):
            errors.append("XDMA W stall count differs from gate observation")
        if all_stalls:
            control = record("NTRPU_AXI_CONTROL_COMPLETE", ["aw_stalls", "ar_stalls", "b_sink_stalls",
                             "bid_checked", "rid_beats_checked"])
            for channel, delay in [("AW", 7), ("AR", 11), ("B", 13)]:
                gate = record("NTRPU_AXI_GATE_COMPLETE channel=" + channel,
                              ["beats", "source_stalls", "sink_stalls"])
                gates[channel] = gate
                count = axi.get(channel.lower(), 0)
                if count <= 0 or gate.get("beats") != count or gate.get("source_stalls", 0) < count*delay:
                    errors.append("Missing deterministic AXI " + channel + " stall/beat coverage")
            for channel in ("AW", "AR"):
                if control.get(channel.lower()+"_stalls") != gates[channel].get("source_stalls"):
                    errors.append("Independent " + channel + " stall count differs")
            if control.get("b_sink_stalls") != gates["B"].get("sink_stalls"):
                errors.append("Independent B stall count differs")
            if control.get("bid_checked") != axi.get("b") or control.get("rid_beats_checked") != axi.get("rbeats"):
                errors.append("Missing AXI response ID coverage")
    return {"status": "FAIL" if errors else "PASS", "errors": errors,
            "excluded_tcl_echo_lines": sum(line.lstrip().startswith("#") for line in lines),
            "excluded_informational_notes": lines.count(informational),
            "cq": cq, "axi": axi, "requests": requests, "gates": gates, "host": host,
            "host64": host64, "payload_bytes": payload, "all_stalls": all_stalls, "control": control,
            "scope": f"Isolated AMD XDMA with BRAM; {payload} aligned bytes per direction; not DDR or sustained/physical throughput"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--require-stalls", action="store_true")
    parser.add_argument("--require-host64", action="store_true")
    parser.add_argument("--require-all-stalls", action="store_true")
    args = parser.parse_args()
    data = args.log.read_bytes()
    result = review(data.decode("utf8", errors="replace"), require_stalls=args.require_stalls,
                    require_host64=args.require_host64, require_all_stalls=args.require_all_stalls)
    result.update(log=str(args.log), log_sha256=hashlib.sha256(data).hexdigest())
    output = json.dumps(result, indent=2) + "\n"
    if args.json:
        with args.json.open("x", encoding="utf8") as stream:
            stream.write(output)
    print(output, end="")
    return result["status"] != "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
