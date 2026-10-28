"""Permit repeated HLS RAM requests while checking every synchronous return."""
from build_ntrpu_llama32_layer_real_tb_r1 import render as previous,ROOT
from ntrpu_llama32_layer_fixture_r1 import digest

def render():
    text=previous()
    old='    integer source_reads=0;bit source_pending=0;integer source_address=0;'
    new='''    integer source_reads=0,hls_inputs=0;bit source_pending=0;integer source_address=0;
    bit hls_input_pending=0;integer hls_input_address=0;reg [2047:0] source_seen=0;'''
    assert text.count(old)==1;text=text.replace(old,new)
    old='            if(dut.front_result_ce)begin'
    new='''            if(hls_input_pending)begin
                `CHECK(dut.projection.quantize_store.hls_input_q===attention_expected[hls_input_address],"Actual synchronous HLS input from retained front RAM")
                $display("LREAL_HLS_INPUT cycle=%0d index=%0d data=%08h",cycle,hls_input_address,dut.projection.quantize_store.hls_input_q);hls_inputs++;
            end
            hls_input_pending=dut.front_result_ce;hls_input_address=int'(dut.front_result_address);
'''
    assert text.count(old)==1;text=text.replace(old,new+old)
    text=text.replace('dut.front_result_address==source_reads','source_reads<8192')
    text=text.replace('source_reads==2048','source_reads>=2048 && (&source_seen) && hls_inputs==source_reads')
    old='$display("LREAL_SOURCE cycle=%0d index=%0d data=%08h",cycle,source_reads,dut.front_result_q);source_reads++;source_pending=0;'
    new='$display("LREAL_SOURCE cycle=%0d index=%0d data=%08h",cycle,source_address,dut.front_result_q);source_seen[source_address]=1;source_reads++;source_pending=0;'
    assert text.count(old)==1;text=text.replace(old,new)
    return text.replace('tb_ntrpu_llama32_layer_real_r1','tb_ntrpu_llama32_layer_real_r2')

if __name__=='__main__':
    path=ROOT/'fpga/tb/tb_ntrpu_llama32_layer_real_r2.sv'
    with path.open('x') as f:f.write(render())
    print(digest(path))
