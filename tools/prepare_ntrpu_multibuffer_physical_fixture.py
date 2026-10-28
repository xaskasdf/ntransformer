"""Create a timing fixture, not an AXI transport wrapper, from a DCP port stub."""

import argparse
import json
from pathlib import Path
import re
import zipfile


def prepare(checkpoint, output, reserve):
    with zipfile.ZipFile(checkpoint) as archive:
        tops = [top for top in ("ntrpu_peer_multibuffer_compact", "ntrpu_peer_multibuffer_control", "ntrpu_peer_multibuffer_decoded", "ntrpu_peer_multibuffer_fillstage", "ntrpu_peer_multibuffer_elastic")
                if top + "_stub.v" in archive.namelist()]
        if len(tops) != 1:
            raise RuntimeError("Unexpected checkpoint top")
        top = tops[0]
        stub = archive.read(top + "_stub.v").decode()
    clean = re.sub(r"/\*.*?\*/", "", stub, flags=re.S)
    ports = re.findall(r"^\s*(input|output)\s*(\[\d+:\d+\])?\s*(\w+)\s*;", clean, re.M)
    if len(ports) != 37 or ports[0] != ("input", "", "clk"):
        raise RuntimeError("Unexpected checkpoint port contract")
    bank = next(width for _, width, name in ports if name == "banks_used")
    if bank != f"[{3 + reserve}:0]" or f'(* RESERVE_BANKS = "{reserve}" *)' not in stub:
        raise RuntimeError("Wrong reserve checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    (output / "core_top.txt").write_text(top + "\n")
    (output / "core_stub.v").write_text(stub.replace(f"module {top}(",
        f'(* black_box = "yes" *) module {top}('))
    lines = ["// Timing measurement only: raw probe pins are not an AXI interface.",
             "// Every core data/control port is bounded by a preserved synchronous register.",
             "module ntrpu_multibuffer_physical_fixture ("]
    lines += [f"    {direction} wire {width} {name}" + ("," if i != len(ports) - 1 else "")
              for i, (direction, width, name) in enumerate(ports)]
    lines += [");", "    wire core_clock;",
              '    (* DONT_TOUCH = "yes" *) BUFGCE clock_buffer (.I(clk), .CE(1\'b1), .O(core_clock));']
    connections = [".clk(core_clock)"]
    counts = dict(input=0, output=0)
    for direction, width, name in ports[1:]:
        bits = 1 if not width else int(width[1:].split(":")[0]) + 1
        counts[direction] += bits
        lines.append(f'    (* DONT_TOUCH = "yes" *) reg {width} probe_{name};')
        if direction == "input":
            lines.append(f"    always @(posedge core_clock) probe_{name} <= {name};")
            connections.append(f".{name}(probe_{name})")
        else:
            lines += [f"    wire {width} core_{name};",
                      f"    always @(posedge core_clock) probe_{name} <= core_{name};",
                      f"    assign {name} = probe_{name};"]
            connections.append(f".{name}(core_{name})")
    lines += [f'    (* KEEP_HIERARCHY = "yes" *) {top} core (',
              "        " + ",\n        ".join(connections), "    );", "endmodule"]
    (output / "fixture.sv").write_text("\n".join(lines) + "\n")
    (output / "fixture.json").write_text(json.dumps(dict(reserve=reserve, core_top=top, register_bits=counts,
        scope="Synchronous timing fixture; probe pins have no external timing contract; not an AXI adapter"), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reserve", type=int, choices=(4, 8))
    args = parser.parse_args()
    prepare(args.checkpoint, args.output, args.reserve)
