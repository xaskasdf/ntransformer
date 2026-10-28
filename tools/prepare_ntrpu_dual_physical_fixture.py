"""Registered measurement boundary for the preserved dual-MIG receiver.

The fixture is not an AXI adapter. External reset stays asynchronous and direct.
"""
import argparse
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    text=(root/'fpga/rtl/ntrpu_dual_multibuffer_mig.sv').read_text()
    header=text.split('module ntrpu_dual_multibuffer_mig (',1)[1].split('\n);',1)[0]
    ports=[]
    for line in header.splitlines():
        if not line.strip():continue
        match=re.fullmatch(r'\s*(input|output) wire (?:\[(\d+):0\] )?([\w, ]+),?',line)
        if not match:raise ValueError(line)
        direction,high,names=match.groups();bits=int(high)+1 if high else 1
        ports.extend((direction,bits,name.strip()) for name in names.rstrip(',').split(','))
    def span(bits):return f'[{bits-1}:0] ' if bits>1 else ''
    lines=['`timescale 1ns/1ps','// Timing measurement only, NOT a protocol adapter.',
           'module ntrpu_dual_physical_fixture (']
    lines += ['    '+d+' wire '+span(w)+n+(',' if i<len(ports)-1 else '')
              for i,(d,w,n) in enumerate(ports)]
    lines += [');','    wire core_source_clk,core_mgmt_clk;','    wire [1:0] core_ui_clk;']
    for name,port,out in [('source','source_clk','core_source_clk'),('mgmt','mgmt_clk','core_mgmt_clk'),
                          ('memory0','ui_clk[0]','core_ui_clk[0]'),('memory1','ui_clk[1]','core_ui_clk[1]')]:
        lines += [f'    (* DONT_TOUCH="true" *) BUFGCE {name}_buf (.I({port}), .CE(1\'b1), .O({out}));']
    bindings=['.source_clk(core_source_clk)','.mgmt_clk(core_mgmt_clk)','.ui_clk(core_ui_clk)',
              '.external_reset_async(external_reset_async)']
    management={'rearm','running','rearm_ready','restart_required','peers_reset_req','source_alive','memory_alive'}
    source={'stop','source_reset_n','admit','cmd_valid','cmd_channel','data_valid','cmd_ready','data_ready',
            'cmd_addr','cmd_tag','cmd_len','data','cpl_valid','cpl_ready','cpl_tag','cpl_status','fault',
            'drained','banks_used','pending_counts'}
    counts=dict(source=0,mgmt=0,memory0=0,memory1=0);domains={}
    for direction,bits,name in ports:
        if name in {'source_clk','mgmt_clk','ui_clk','external_reset_async'}:continue
        lines += [f'    (* DONT_TOUCH="true" *) reg {span(bits)}probe_{name};']
        if direction=='output':
            lines += [f'    wire {span(bits)}core_{name};',f'    assign {name}=probe_{name};']
            bindings.append(f'.{name}(core_{name})');lhs,rhs='probe_'+name,'core_'+name
        else:
            bindings.append(f'.{name}(probe_{name})');lhs,rhs='probe_'+name,name
        if name.startswith('m_') or name in {'calibrated','ui_rst','memory_reset_n'}:
            assert bits%2==0
            width=bits//2;domains[name]='memory0,memory1'
            for g in range(2):
                sl=f'[{g*width} +:{width}]'
                lines += [f'    always @(posedge core_ui_clk[{g}]) {lhs}{sl}<={rhs}{sl};']
                counts[f'memory{g}']+=width
        else:
            assert name in management|source,name
            domain='mgmt' if name in management else 'source';domains[name]=domain
            lines += [f'    always @(posedge core_{domain}_clk) {lhs}<={rhs};'];counts[domain]+=bits
    lines += ['    (* KEEP_HIERARCHY="yes" *) ntrpu_dual_multibuffer_mig core (',
              '        '+',\n        '.join(bindings),'    );','endmodule','']
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'fixture.sv').write_text('\n'.join(lines))
    (args.output/'fixture.json').write_text(json.dumps(dict(register_bits=counts,ports=ports,domains=domains,
        asynchronous_direct_port='external_reset_async',warning='OOC measurement only; not an AXI adapter'),indent=2)+'\n')


if __name__=='__main__':main()
