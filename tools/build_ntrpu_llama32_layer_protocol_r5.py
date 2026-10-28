"""Assert projection ACK ordering directly instead of relying on eventual drain."""
from build_ntrpu_llama32_layer_protocol_r4 import render as previous,ROOT

def render():
    tb,peers=previous()
    old='        if(cycle>1000000)$fatal'
    new='''        if(rst_n && dut.projection_completion_ready)begin
            `CHECK(!completion_valid && !dut.lease.busy && !dut.post.busy && !dut.ffn.busy,"projection ACK preceded downstream drain")
        end
'''
    assert tb.count(old)==1;tb=tb.replace(old,new+old)
    return tb.replace('tb_ntrpu_llama32_layer_protocol_r4','tb_ntrpu_llama32_layer_protocol_r5'),peers

if __name__=='__main__':
    tb,_=render()
    with (ROOT/'fpga/tb/tb_ntrpu_llama32_layer_protocol_r5.sv').open('x') as f:f.write(tb)
