"""Measure public DDR reset/clock traces; never infer functional PASS from them."""
import argparse
import collections
import hashlib
import json
import pathlib
import re


def read_trace(text):
    if not re.search(r"\$timescale\s+1\s*ps\s+\$end", text):
        raise ValueError("Expected a 1 ps VCD timescale")
    scopes, signals = [], {}
    events = collections.defaultdict(list)
    now = 0
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "$scope":
            scopes.append(words[2])
        elif words[0] == "$upscope":
            scopes.pop()
        elif words[0] == "$var":
            name = "/".join(scopes[1:] + [words[4]])
            if name in signals:
                raise ValueError("Duplicate signal path: " + name)
            signals[name] = (words[3], int(words[2]))
        elif line.startswith("#"):
            now = int(line[1:])
        elif line[0] == "b":
            value, code = line[1:].split()
            events[code].append((now, value))
        elif line[0] in "01xz":
            events[line[1:]].append((now, line[0]))
    return {name: events[code] for name, (code, _) in signals.items()}


def reset_window(events, after):
    edges = [(t, v) for t, v in events if t >= after]
    if len(edges) != 2 or [v for _, v in edges] != ["0", "1"]:
        raise ValueError("Expected exactly one completed warm active-low reset")
    return edges[0][0], edges[1][0]


def measure(reset_events, clock_events, after):
    start, end = reset_window(reset_events, after)
    clocks = [(t, v) for t, v in clock_events if start <= t < end]
    if any(re.search("[xz]", v) for _, v in clocks):
        raise ValueError("Unknown clock during warm reset")
    # Public RDIMM ck[1] and component CK[1] are the positive clock legs.
    rising = sum(1 for (ta, va), (tb, vb) in zip(clock_events, clock_events[1:])
                 if start <= tb < end and not re.search("[xz]", va + vb)
                 and not (int(va, 2) & 2) and (int(vb, 2) & 2))
    gaps = [b[0] - a[0] for a, b in zip(clocks, clocks[1:])]
    return {"reset_assert_ps": start, "reset_release_ps": end,
            "reset_low_ps": end - start, "clock_changes_while_reset": len(clocks),
            "positive_clock_edges_while_reset": rising,
            "largest_clock_gap_ps": max(gaps, default=None),
            "first_clock_change_ps": clocks[0][0] if clocks else None,
            "last_clock_change_ps": clocks[-1][0] if clocks else None}


def inspect(text, components=False):
    signals = read_trace(text)
    resets = [t for t, v in signals["sys_rst"] if t > 0 and v == "1"]
    if len(resets) != 1:
        raise ValueError("Expected exactly one warm system reset")
    after = resets[0]
    slots = {}
    for slot in range(2):
        prefix = f"dimms[{slot}].memory/"
        slots[str(slot)] = measure(signals[prefix + "reset_n"], signals[prefix + "ck"], after)
    component_results = {}
    for name in signals:
        if re.search(r"[/.]iDDR4/RESET_n$", name):
            component_results[name] = measure(signals[name], signals[name[:-7] + "CK"], after)
            component_results[name]["cke_events_after_system_reset"] = [
                {"time_ps": t, "value": v} for t, v in signals[name[:-7] + "CKE"] if t >= after]
    if components and len(component_results) != 72:
        raise ValueError(f"Expected 72 component interfaces, found {len(component_results)}")
    return {"status": "TRACE_ANALYZED", "functional_reset_pass": False,
            "scope": "Public signal observations only; requires separate model/protocol review",
            "system_reset_assert_ps": after, "slots": slots,
            "component_count": len(component_results), "components": component_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vcd", type=pathlib.Path)
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    data = args.vcd.read_bytes()
    result = inspect(data.decode("utf8"), args.components)
    result["vcd_sha256"] = hashlib.sha256(data).hexdigest()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json:
        with args.json.open("x", encoding="utf8") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
