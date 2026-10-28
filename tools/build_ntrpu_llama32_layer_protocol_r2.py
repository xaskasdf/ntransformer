"""Correct the synthetic cache binding port name before protocol execution."""
from build_ntrpu_llama32_layer_protocol_r1 import render as original,ROOT

def render():
    tb,peers=original()
    return (tb.replace('bind_token_capacity','bind_capacity').replace('tb_ntrpu_llama32_layer_protocol_r1','tb_ntrpu_llama32_layer_protocol_r2'),
            peers.replace('bind_token_capacity','bind_capacity'))

if __name__=='__main__':
    tb,peers=render()
    for name,text in [('tb_ntrpu_llama32_layer_protocol_r2.sv',tb),('ntrpu_llama32_layer_protocol_peers_r2.sv',peers)]:
        with (ROOT/'fpga/tb'/name).open('x') as f:f.write(text)
