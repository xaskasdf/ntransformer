"""Bind two write ports to an exported AMD HBM template, checking every port."""
import argparse
import hashlib
import json
from pathlib import Path
import re

WRITE=dict(awaddr=('input',34),awburst=('input',2),awid=('input',6),awlen=('input',4),
    awsize=('input',3),awvalid=('input',1),awready=('output',1),wdata=('input',256),
    wstrb=('input',32),wlast=('input',1),wvalid=('input',1),wready=('output',1),
    bid=('output',6),bresp=('output',2),bvalid=('output',1),bready=('input',1))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('template',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();text=a.template.read_text();pins=[]
    for line in text.splitlines():
        m=re.search(r'\.([A-Za-z0-9_]+)\([^)]*\),?\s*// (input|output) wire(?: \[(\d+)\s*:\s*(\d+)\])? (\w+)',line)
        if m:
            n,d,hi,lo,comment=m.groups();assert n==comment and (lo is None or lo=='0')
            pins.append((n,d,1 if hi is None else int(hi)+1))
    assert len({n for n,_,_ in pins})==len(pins)>100
    actual={n:(d,w) for n,d,w in pins}
    for index in (0,16):
        for signal,value in WRITE.items():assert actual[f'AXI_{index:02}_{signal.upper()}']==value
    axi_ports=sorted({int(m[1]) for n in actual if (m:=re.fullmatch(r'AXI_(\d+)_ACLK',n))})
    assert axi_ports==list(range(32))
    management=[(n,d,w) for n,d,w in pins if not n.startswith('AXI_')]
    interface=[('axi_clk','input',1),('axi_cold_reset_n','input',1)]
    interface += [('s_'+n,d,w*2) for n,(d,w) in WRITE.items()]+management
    lines=['`timescale 1ns/1ps','// Native HBM port binding only; external management owns clocks, initialization',
           '// and coordinated cold reset. Not a startup/recovery controller.',
           '// Port 00 and 16 expose one ingress per stack. No read traffic.',
           'module ntrpu_hbm_native_link (']
    lines += [f'    {d} wire '+(f'[{w-1}:0] ' if w>1 else '')+n+(',' if i<len(interface)-1 else '') for i,(n,d,w) in enumerate(interface)]
    lines += [');','    wire [63:0] parity;', '    for(genvar k=0;k<64;k++)begin: byte_parity',
              '        assign parity[k]=^s_wdata[k*8+:8];','    end','    native_hbm memory (']
    connections={}
    for n,d,w in pins:
        m=re.fullmatch(r'AXI_(\d+)_(\w+)',n)
        if not m:v=n
        else:
            index=int(m[1]);key=m[2].lower();lane=0 if index==0 else 1
            if key=='aclk':v='axi_clk'
            elif key=='areset_n':v='axi_cold_reset_n'
            elif index in (0,16) and key in WRITE:
                v='s_'+key+(f'[{lane*w}+:{w}]' if w>1 else f'[{lane}]')
            elif index in (0,16) and key=='wdata_parity':v=f'parity[{lane*32}+:32]'
            elif d=='output':v=''
            elif key in ('arburst','awburst'):v=f"{w}'d1"
            elif key in ('arsize','awsize'):v=f"{w}'d5"
            elif key in ('rready','bready'):v="1'b1"
            else:v=f"{w}'b0"
        connections[n]=v
    lines += [f'        .{n}({v})'+(',' if i<len(connections)-1 else '') for i,(n,v) in enumerate(connections.items())]
    lines += ['    );','endmodule',''];a.output.mkdir(parents=True,exist_ok=False)
    source=a.output/'ntrpu_hbm_native_link.sv';source.write_text('\n'.join(lines))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (a.output/'BINDING.json').write_text(json.dumps(dict(status='NATIVE_PORT_BINDING_GENERATED',
        native_template_sha256=sha(a.template),binding_sha256=sha(source),axi_ports=axi_ports,
        active_write_ports=[0,16],connections=connections,management=management,
        clock_and_reset='External management; common AXI clock and coordinated cold reset',
        instantiated_hard_controller=True,initialized_or_simulated=False),indent=2)+'\n')

if __name__=='__main__':main()
