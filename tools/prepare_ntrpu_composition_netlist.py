"""Adapt the preserved public-interface credit bench for the fixed routed core."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    source = Path(__file__).resolve().parents[1]/'fpga/tb/tb_ntrpu_peer_multibuffer_credits.sv'
    text = source.read_text()
    old = 'ntrpu_peer_multibuffer_credits #(.GUARD_SLOTS(GUARD_SLOTS)) dut(.*);'
    assert text.count(old) == 1
    text = text.replace(old, 'ntrpu_composition_core dut(.*);')
    text, count = re.subn(r'^        if\(negative==[123]\) force dut\.[^\n]+\n', '', text, flags=re.M)
    assert count == 3 and 'force dut.' not in text
    old = 'if(!$value$plusargs("NEGATIVE=%d",negative)) negative=0;'
    assert text.count(old) == 1
    text = text.replace(old, old+'\n        if(negative!=0 || GUARD_SLOTS!=4) $fatal(1,"NETLIST_FIXED_PROFILE");')
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output/'netlist_tb.sv'
    target.write_text(text)
    (args.output/'transformation.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        changes=['Fixed routed top without parameter override',
                 'Removed three hierarchical RTL-only fault injection statements',
                 'Assert four-credit positive configuration; public scoreboards unchanged']), indent=2)+'\n')


if __name__ == '__main__': main()
