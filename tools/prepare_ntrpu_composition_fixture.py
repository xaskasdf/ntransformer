"""Generate fixed R4/four-credit, three-domain physical measurement ports.

These registers bound internal timing; they are NOT AXI protocol slices.
"""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root/'fpga/rtl/ntrpu_peer_multibuffer_credits.sv').read_text()
    header = source.split(') (', 1)[1].split('\n);', 1)[0]
    constants = {'CHANNELS': 2, 'MEMORY_ADDR_WIDTH': 37, 'CHANNEL_BITS': 1,
                 'BANKS': 8, 'COUNT_BITS': 5}
    ports = []
    for line in header.splitlines():
        match = re.fullmatch(r'\s*(input|output) wire (?:\[([^]]+)\] )?([\w, ]+),?', line)
        if not match:
            if line.strip(): raise ValueError(line)
            continue
        direction, width, names = match.groups()
        bits = 1
        if width:
            hi, lo = width.split(':')
            for key, value in constants.items():
                hi = re.sub(r'\b'+key+r'\b', str(value), hi)
            if not re.fullmatch(r'[0-9 *+()-]+', hi): raise ValueError(hi)
            bits = eval(hi, {'__builtins__': {}}, {}) - int(lo) + 1
        for name in names.rstrip(',').split(','):
            ports.append((direction, bits, name.strip()))
    def span(bits): return f'[{bits-1}:0] ' if bits > 1 else ''
    lines = ['`timescale 1ns/1ps', '// Generated measurement fixture, NOT a transport adapter.',
             'module ntrpu_composition_fixture (']
    lines += ['    '+d+' wire '+span(w)+n+(',' if i<len(ports)-1 else '')
              for i, (d,w,n) in enumerate(ports)]
    lines += [');', '    wire core_clk;', '    wire [1:0] core_mem_clk;',
              '    (* DONT_TOUCH="true" *) BUFGCE source_buf (.I(clk), .CE(1\'b1), .O(core_clk));']
    for g in range(2):
        lines += [f'    (* DONT_TOUCH="true" *) BUFGCE mem{g}_buf (.I(mem_clk[{g}]), .CE(1\'b1), .O(core_mem_clk['+str(g)+']));']
    bindings = ['.clk(core_clk)', '.mem_clk(core_mem_clk)']
    counts = dict(source=0, memory0=0, memory1=0)
    for direction, bits, name in ports:
        if name in ('clk', 'mem_clk'): continue
        lines += [f'    (* DONT_TOUCH="true" *) reg {span(bits)}probe_{name};']
        if direction == 'output':
            lines += [f'    wire {span(bits)}core_{name};', f'    assign {name} = probe_{name};']
            bindings += [f'.{name}(core_{name})']
            lhs, rhs = 'probe_'+name, 'core_'+name
        else:
            bindings += [f'.{name}(probe_{name})']
            lhs, rhs = 'probe_'+name, name
        if name.startswith('m_') or name == 'mem_rst_n':
            assert bits % 2 == 0
            width = bits//2
            for g in range(2):
                sl = f'[{g*width} +:{width}]'
                lines += [f'    always @(posedge core_mem_clk[{g}]) {lhs}{sl} <= {rhs}{sl};']
                counts['memory'+str(g)] += width
        else:
            lines += [f'    always @(posedge core_clk) {lhs} <= {rhs};']
            counts['source'] += bits
    lines += ['    (* KEEP_HIERARCHY="yes" *) ntrpu_peer_multibuffer_credits #(.GUARD_SLOTS(4)) core (',
              '        '+',\n        '.join(bindings), '    );', 'endmodule', '']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'fixture.sv').write_text('\n'.join(lines))
    (args.output/'fixture.json').write_text(json.dumps(dict(register_bits=counts, ports=ports,
        warning='Measurement boundary only; not an AXI adapter'), indent=2)+'\n')


if __name__ == '__main__': main()
