"""Bind the inspected Vitis post-O core without adding RAM response latency."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from bind_ntrpu_attention_hls_ports_r1 import ports

TOP='ntrpu_post_attention_r1'

def contract(actual):
    mapping={};roles=[]
    def bind(name,direction,width,signal):
        assert actual.get(name)==dict(direction=direction,width=width),(name,actual.get(name))
        assert name not in mapping
        mapping[name]=signal;roles.append(dict(port=name,direction=direction,width=width,signal=signal))
    for name,direction,width,signal in [
        ('ap_clk','input',1,'clk'),('ap_rst','input',1,'~rst_n'),('ap_start','input',1,'hls_start'),
        ('ap_done','output',1,'hls_done'),('ap_ready','output',1,'hls_ready'),('ap_idle','output',1,'unused_idle'),
        ('ap_return','output',32,'hls_return'),('elements','input',32,'hls_elements'),('epsilon_bits','input',32,'hls_epsilon_bits'),
    ]:bind(name,direction,width,signal)
    for p,name in enumerate(('residual_input','branch','norm_weight')):
        bind(name+'_address0','output',11,f'hls_input_address[{p}]')
        bind(name+'_ce0','output',1,f'hls_input_ce[{p}]')
        bind(name+'_q0','input',32,f'hls_input_q[{p}]')
    for name,role,aw,dw in [('residual_output','residual',11,32),('ffn_input','ffn',9,64)]:
        for suffix,width,signal in [('address0',aw,'address'),('ce0',1,'ce'),('we0',1,'we'),('d0',dw,'d')]:
            bind(name+'_'+suffix,'output',width,'hls_'+role+'_'+signal)
    assert set(mapping)==set(actual) and len(mapping)==26
    return mapping,roles

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rtl',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    raw=args.rtl.read_bytes();actual,body=ports(raw.decode(),TOP);mapping,roles=contract(actual)
    assert re.search(r'always\s*@\s*\(posedge\s+ap_clk\s*\)',body)
    assert re.search(r'if\s*\(ap_rst\s*==\s*1\x27b1\)',body)
    assert not re.search(r'(?:posedge|negedge)\s+ap_rst',body)
    sv='// Inspected generated core SHA256: '+hashlib.sha256(raw).hexdigest()+'\n'
    sv+='''`default_nettype none
module ntrpu_post_attention_hls_ports_r1 (
    input wire clk,rst_n,hls_start,
    output wire hls_done,hls_ready,
    output wire [31:0] hls_return,
    input wire [31:0] hls_elements,hls_epsilon_bits,
    output wire [2:0] hls_input_ce,
    output wire [10:0] hls_input_address [0:2],
    input wire [31:0] hls_input_q [0:2],
    output wire hls_residual_ce,hls_residual_we,
    output wire [10:0] hls_residual_address,
    output wire [31:0] hls_residual_d,
    output wire hls_ffn_ce,hls_ffn_we,
    output wire [8:0] hls_ffn_address,
    output wire [63:0] hls_ffn_d
);
    wire unused_idle;
    ntrpu_post_attention_r1 core (
'''
    sv+=',\n'.join('        .'+name+'('+signal+')' for name,signal in mapping.items())+'\n    );\nendmodule\n`default_nettype wire\n'
    args.output.mkdir(parents=True);out=args.output/'ntrpu_post_attention_hls_ports_r1.sv'
    out.write_text(sv,encoding='utf-8',newline='\n')
    report=dict(status='INSPECTED_POST_ATTENTION_R1_GENERATED_PORTS',core_path=str(args.rtl),core_sha256=hashlib.sha256(raw).hexdigest(),
        binding_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ports=roles,synchronous_active_high_reset=True,added_latency_cycles=0,arithmetic_qualified=False,physical_timing_qualified=False)
    (args.output/'binding.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
