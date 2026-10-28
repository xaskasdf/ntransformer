"""Extend an isolated native binding with readback; preserve the qualified link."""
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path
from prepare_ntrpu_hbm_native_link import WRITE
from prepare_ntrpu_hbm_cdc import native_pins

ROOT=Path(__file__).resolve().parents[1]
READ=dict(araddr=('input',34),arburst=('input',2),arid=('input',6),arlen=('input',4),
          arsize=('input',3),arvalid=('input',1),arready=('output',1),rdata=('output',256),
          rdata_parity=('output',32),rid=('output',6),rlast=('output',1),rresp=('output',2),
          rvalid=('output',1),rready=('input',1))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);a=ap.parse_args();out=a.output
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_hbm_native_link.py'),
                    str(out/'native_hbm.veo'),str(out/'write_binding')],check=True)
    old=json.loads((out/'write_binding/BINDING.json').read_text());connections=dict(old['connections'])
    template=(out/'native_hbm.veo').read_text()
    ports={m[1]:(m[2],int(m[3])+1 if m[3] else 1) for m in re.finditer(
        r'\.([A-Za-z0-9_]+)\([^)]*\),?\s*// (input|output) wire(?: \[(\d+)\s*:\s*0\])?',template)}
    for lane,index in enumerate((0,16)):
        for name,(direction,width) in READ.items():
            native=f'AXI_{index:02}_{name.upper()}';assert ports[native]==(direction,width)
            connections[native]='s_'+name+(f'[{lane*width}+:{width}]' if width>1 else f'[{lane}]')
    interface=[('axi_clk','input',1),('axi_cold_reset_n','input',1)]
    interface += [('s_'+n,d,2*w) for n,(d,w) in (WRITE|READ).items()]+old['management']
    lines=['`timescale 1ns/1ps','// Isolated native AXI3 read/write probe; no transport/CDC implementation.',
           'module ntrpu_hbm_native_rw (']
    lines += [f'    {d} wire '+(f'[{w-1}:0] ' if w>1 else '')+n+(',' if i+1<len(interface) else '') for i,(n,d,w) in enumerate(interface)]
    lines += [');','    wire [63:0] parity;', '    for(genvar k=0;k<64;k++)begin: byte_parity',
              '        assign parity[k]=^s_wdata[k*8+:8];','    end','    native_hbm memory (']
    lines += [f'        .{n}({v})'+(',' if i+1<len(connections) else '') for i,(n,v) in enumerate(connections.items())]
    lines += ['    );','endmodule','']
    (out/'ntrpu_hbm_native_rw.sv').write_text('\n'.join(lines))
    declarations,pins=native_pins(old)
    for name,(direction,width) in (WRITE|READ).items():
        declarations.append(f'    {"reg" if direction=="input" else "wire"} [{2*width-1}:0] s_{name}'+("=0;" if direction=='input' else ';'))
        pins.append(f'.s_{name}(s_{name})')
    bench=(ROOT/'fpga/tb/ntrpu_hbm_native_traffic_body.svh').read_text()
    header='''`timescale 1ns/1ps
module tb_ntrpu_hbm_native_traffic;
    reg mem_clk=0,ref_clk=0,apb_clk=0;
    always #1.111 mem_clk=~mem_clk;
    always #5 ref_clk=~ref_clk;
    initial begin #1.7;forever #5 apb_clk=~apb_clk;end
    reg mem_rst_n=0,apb_rst_n=0;
'''
    (out/'tb_ntrpu_hbm_native_traffic.sv').write_text(header+'\n'.join(declarations)+
        '\n    ntrpu_hbm_native_rw dut(\n        '+',\n        '.join(pins)+');\n'+bench+'\nendmodule\n')
    changes={n:dict(before=old['connections'][n],after=v) for n,v in connections.items() if old['connections'][n]!=v}
    assert len(changes)==2*len(READ)
    (out/'BINDING.json').write_text(json.dumps(dict(status='NATIVE_RW_BINDING_GENERATED',
        original=old,connections=connections,changed_connections=changes,read_ports=[0,16],write_ports=[0,16],
        binding_sha256=sha(out/'ntrpu_hbm_native_rw.sv'),bench_sha256=sha(out/'tb_ntrpu_hbm_native_traffic.sv'),
        parity_checked=False,user_parity_enabled=False,reset_after_traffic=False),indent=2)+'\n')

if __name__=='__main__':main()
