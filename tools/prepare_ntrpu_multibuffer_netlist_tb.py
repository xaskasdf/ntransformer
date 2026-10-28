"""Make a public-pin-only bench for a specialized Vivado functional netlist.

Keep all byte/order/backpressure/error checks. Internal lease/pipeline checks
remain covered by RTL runs; netlist replay must use check_bank_leases=False.
"""
import argparse
from pathlib import Path


def prepare(source):
    replacements = {
        "ntrpu_peer_multibuffer #(.CHANNELS(CHANNELS),.RESERVE_BANKS(RESERVE_BANKS),\n"
        "        .BANK_BEATS(BANK_BEATS),.DESCRIPTORS(DESCRIPTORS),.MAX_OUTSTANDING(MAX_OUTSTANDING)) dut(.*);":
        "ntrpu_peer_multibuffer dut(.*);",
        "dut.alloc_bank,dut.alloc_generation": "0,0",
        '            if(data_valid && data_ready && dut.ram_read_enable[dut.fill_bank])\n'
        '                $fatal(1,"simultaneous read and fill of the same bank");\n': "",
        '                if(dut.fifo_count[p]>2 || int\'(dut.fifo_count[p])+int\'(dut.read_pending[p])>2)\n'
        '                    $fatal(1,"read pipeline credit overflow");\n'
        '                if(dut.occupied[p]>DESCRIPTORS) $fatal(1,"descriptor overflow");\n'
        '                if(dut.read_issue[p] && (!banks_used[dut.bank[p*DESCRIPTORS+dut.send_head[p]]] ||\n'
        '                    dut.allocator.owner[dut.bank[p*DESCRIPTORS+dut.send_head[p]]] != p))\n'
        '                    $fatal(1,"RAM read without ownership");\n': "",
        "        repeat(4) @(negedge clk); rst_n=1;":
        "        #200; // Wait for vendor GSR, then apply explicit cold reset.\n"
        "        repeat(4) @(negedge clk); rst_n=1;",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"bench transformation must match exactly once: {old[:80]}")
        source = source.replace(old, new)
    if "dut." in source:
        raise RuntimeError("unhandled internal netlist reference")
    return "// Generated public-pin bench; CMD bank/gen fields are unobserved placeholders.\n" + source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as output:
        output.write(prepare(args.input.read_text(encoding="utf-8")))
