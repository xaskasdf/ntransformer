"""Specialize the public-pin bench for an already elaborated compact netlist."""
import argparse
from pathlib import Path
from prepare_ntrpu_multibuffer_netlist_tb import prepare


def specialized(source, reserve):
    text = prepare(source)
    old = "ntrpu_peer_multibuffer dut(.*);"
    if text.count(old) != 1:
        raise RuntimeError("DUT anchor changed")
    text = text.replace(old, "ntrpu_peer_multibuffer_compact dut(.*);")
    old = "parameter integer CHANNELS=2, RESERVE_BANKS=4, BANK_BEATS=64, DESCRIPTORS=16, MAX_OUTSTANDING=4;"
    if text.count(old) != 1 or reserve not in (4, 8):
        raise RuntimeError("profile anchor/selection changed")
    return text.replace(old, f"localparam integer CHANNELS=2, RESERVE_BANKS={reserve}, BANK_BEATS=64, DESCRIPTORS=16, MAX_OUTSTANDING=4;")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reserve", type=int)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as output:
        output.write(specialized(args.input.read_text(encoding="utf-8"), args.reserve))
