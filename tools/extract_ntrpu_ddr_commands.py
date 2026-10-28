"""Extract one public chip's selected commands from the diagnostic XSIM VCD."""
import argparse
import bisect
import hashlib
import json
import pathlib

from inspect_ntrpu_ddr_reset_trace import read_trace


def extract(text):
    signals = read_trace(text)
    candidates = [name[:-4] for name in signals if name.endswith("/CS_n")]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one traced command interface")
    prefix = candidates[0]
    names = ["RESET_n", "CKE", "CS_n", "ACT_n", "RAS_n_A16", "CAS_n_A15", "WE_n_A14",
             "BG", "BA", "ADDR", "ADDR_17", "ODT", "TEN", "PARITY"]
    def value(name, time):
        events = signals[prefix + name]
        index = bisect.bisect_right(events, (time, chr(127))) - 1
        return events[index][1] if index >= 0 else "x"
    commands = []
    clocks = signals[prefix + "CK"]
    for (_, old), (time, new) in zip(clocks, clocks[1:]):
        if old in {"01", "1"} and new == "10":
            row = {name: value(name, time) for name in names}
            if row["RESET_n"] == row["CKE"] == "1" and row["CS_n"] == "0":
                row["time_ps"] = time
                commands.append(row)
    return {"scope": prefix, "commands": commands,
            "scope_limit": "Observed selected commands, not protocol or timing qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vcd", type=pathlib.Path)
    parser.add_argument("json", type=pathlib.Path)
    args = parser.parse_args()
    data = args.vcd.read_bytes()
    result = extract(data.decode("utf8"))
    result["vcd_sha256"] = hashlib.sha256(data).hexdigest()
    with args.json.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"command_count": len(result["commands"]), "vcd_sha256": result["vcd_sha256"]}))


if __name__ == "__main__":
    main()
