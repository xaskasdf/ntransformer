"""Check the lease's intentionally narrowed source subregions, not parent allocations."""
from build_ntrpu_llama32_layer_protocol_r2 import render as previous,ROOT

def render():
    tb,peers=previous()
    changes={
        'dut.ffn.command_base[0]==saved_base[15]':'dut.ffn.command_base[0]==saved_address[15]',
        'dut.ffn.command_bytes[0]==saved_bytes[15]':"dut.ffn.command_bytes[0]==64'd4096",
        'dut.ffn.command_residual_base==saved_base[14]':'dut.ffn.command_residual_base==saved_address[14]',
        'dut.ffn.command_residual_bytes==saved_bytes[14]':"dut.ffn.command_residual_bytes==64'd8192",
        '`CHECK(completion_status==wanted_status,"public status")':'if(completion_status!=wanted_status)$display("LAYER_STATUS actual=%0d expected=%0d stage=%0d gate=%0d fault=%0d",completion_status,wanted_status,completion_stage,completion_preflight_status,dut.fault);\n        `CHECK(completion_status==wanted_status,"public status")',
    }
    for old,new in changes.items():assert tb.count(old)==1,old;tb=tb.replace(old,new)
    return tb.replace('tb_ntrpu_llama32_layer_protocol_r2','tb_ntrpu_llama32_layer_protocol_r3'),peers

if __name__=='__main__':
    tb,_=render()
    with (ROOT/'fpga/tb/tb_ntrpu_llama32_layer_protocol_r3.sv').open('x') as f:f.write(tb)
