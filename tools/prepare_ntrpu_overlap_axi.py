"""Generate an isolated compact transport with two real AMD AXI write slices."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_ntrpu_candidate_dispatch import ports
from prepare_ntrpu_registered_dispatch import one
from prepare_ntrpu_transport_study import DIRECT, GENERATED, comparison

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = dict(awvalid=1, awready=1, awaddr=64, awlen=8, awid=4,
              awcache=4, awsize=3, awburst=2, wvalid=1, wready=1,
              wlast=1, wdata=512, wstrb=64, bready=1, bvalid=1, bid=4, bresp=2)


def integrated(text):
    wires = '\n'.join(f'    wire [{2*w-1}:0] ax_{n};' for n, w in WIDTHS.items())
    text = one(text, '    localparam [2:0] OPEN=', wires+'\n    localparam [2:0] OPEN=')
    pins = ','.join(f'.m_{n}(ax_{n})' for n in WIDTHS)
    text = one(text, '.fault(core_fault),.drained(core_drained),.queued(core_queued),.*',
               '.fault(core_fault),.drained(core_drained),.queued(core_queued),'+pins+',.*')
    lines = ['    for(genvar lane=0;lane<2;lane++)begin: memory_boundary',
             '        ntrpu_mem_regslice axi (.aclk(clk),.aresetn(rst_n),']
    for n, w in WIDTHS.items():
        sel = '[lane]' if w == 1 else f'[lane*{w}+:{w}]'
        lines += [f'            .s_axi_{n}(ax_{n}{sel}),.m_axi_{n}(m_{n}{sel}),']
    lines += ["            .s_axi_awlock(1'b0),.s_axi_awprot(3'b0),",
              "            .s_axi_awqos(4'b0),.s_axi_awregion(4'b0),",
              '            .m_axi_awlock(),.m_axi_awprot(),.m_axi_awqos(),.m_axi_awregion());',
              '    end', 'endmodule']
    return one(text, 'endmodule', '\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    ap.add_argument('--netlist', type=Path); a = ap.parse_args(); out = a.output.resolve()
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/prepare_ntrpu_overlap_physical.py'), str(out)], check=True)
    original = (ROOT/'fpga/rtl/ntrpu_candidate_transport.sv').read_text()
    modified = integrated(original); (out/'ntrpu_candidate_transport.sv').write_text(modified)
    (out/'AXI_INTEGRATION.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True))))
    # IP primitive startup is honored without editing glbl or forcing signals.
    for name in ('tb_ntrpu_candidate_transport', 'tb_ntrpu_transport_overlap_extra'):
        p = out/(name+'.sv'); p.write_text(one(p.read_text(), '    initial begin\n        reset_case(1);', '    initial begin\n        #150;\n        reset_case(1);'))
    p = out/'tb_ntrpu_transport_perf.sv'
    p.write_text(one(p.read_text(), '    initial begin\n        for(integer mode=', '    initial begin\n        #150;\n        for(integer mode='))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source_paths = {n: out/(n+'.sv') if n == 'ntrpu_candidate_transport' or n in GENERATED else ROOT/'fpga/rtl'/(n+'.sv') for n in DIRECT+GENERATED}
    fixture = dict(variant='compact_axi', reference_transport_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   integrated_transport_sha256=sha(out/'ntrpu_candidate_transport.sv'),
                   logic_sha256={n: sha(p) for n, p in source_paths.items()},
                   axi_data_width=512, axi_address_width=64, axi_id_width=4,
                   instances=2, clock_period_ns=5, coordinated_cold_reset=True, startup_ns=150)
    if a.netlist:
        signals = ports((ROOT/'fpga/rtl/ntrpu_registered_dispatch.sv').read_text())
        net = a.netlist.read_text(); top = net.split('module ntrpu_transport_physical\n', 1)[1].split('\n  wire', 1)[0]
        assert re.findall(r'^  (input|output) (\[[^\]]+\])?\s*(\w+);', top, re.M) == signals
        sources = {n: p.read_text() for n, p in source_paths.items()}
        modules = {m for t in sources.values() for m in re.findall(r'^module (\w+)', t, re.M)} | {'ntrpu_mem_regslice'}
        mapping = {n: 'ref_'+n for n in modules}; inverse = {v: k for k, v in mapping.items()}
        assert not set(mapping.values()) & set(re.findall(r'^module (\w+)', net, re.M))
        regex = re.compile(r'\b('+'|'.join(sorted(mapping))+r')\b')
        reverse = re.compile(r'\b('+'|'.join(sorted(inverse))+r')\b')
        for n, text in sources.items():
            renamed = regex.sub(lambda m: mapping[m[0]], text)
            assert reverse.sub(lambda m: inverse[m[0]], renamed) == text
            (out/('ref_'+n+'.sv')).write_text(renamed)
        for source, target in [('tb_ntrpu_candidate_transport', 'tb_ntrpu_transport_netlist'),
                               ('tb_ntrpu_transport_overlap_extra', 'tb_ntrpu_overlap_netlist_extra')]:
            text = (out/(source+'.sv')).read_text().replace('        #150;\n', '')
            if source != 'tb_ntrpu_candidate_transport':
                text = one(text, 'module '+source+';', 'module tb_ntrpu_candidate_transport;')
            text = comparison(text, signals)
            if target != 'tb_ntrpu_transport_netlist':
                text = one(text, 'module tb_ntrpu_transport_netlist;', 'module '+target+';')
            (out/(target+'.sv')).write_text(text)
        fixture['comparison'] = dict(netlist_sha256=sha(a.netlist), latency_edges=0, startup_ns=150, sdf=False, force=False, reversible_renaming=True)
    (out/'AXI_INTEGRATION.json').write_text(json.dumps(fixture, indent=2)+'\n')


if __name__ == '__main__':
    main()
