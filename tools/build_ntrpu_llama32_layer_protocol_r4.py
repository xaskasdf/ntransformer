"""Exercise all four foreign identities, reused prefixes and late FFN revocation."""
from build_ntrpu_llama32_layer_protocol_r3 import render as previous,ROOT

def render():
    tb,peers=previous()
    old='for(integer n=0;n<4;n=n+1)begin defaults();admission();wait_response();release_job();end'
    new='''for(integer n=0;n<4;n=n+1)begin
            defaults();admission();wait_response();release_job();
            // Reuse the same binding and committed layer prefix without a reset.
            command_sequence=wanted_sequence;command_epoch=wanted_epoch;command_source_sequence=wanted_sequence;
            command_source_epoch=wanted_epoch;command_source_valid=1;command_layer=wanted_layer;
            command_position=1;command_coefficient_position=1;command_lease_base=64'h8000000012340100;
            addr_driver=saved_address;base_driver=saved_base;bytes_driver=saved_bytes;
            admission();wait_response();release_job();
        end'''
    assert tb.count(old)==1;tb=tb.replace(old,new)
    old='            tick();waits=waits+1;\n            `CHECK(waits<14000,"public response deadline")'
    new=''
    for i,role in enumerate(('front','projection','post','ffn'),1):
        new+=f'''            if(mode==7 && !injected && target=={i} && dut.{role}.completion_valid)begin
                repeat(4)begin tick();`CHECK(!dut.{role}.completion_ready,"foreign identity acknowledged") end
                dut.{role}.cfg_foreign=0;injected=1;
            end
'''
    assert tb.count(old)==1;tb=tb.replace(old,new+old)
    tb=tb.replace('for(target=0;target<3;target=target+1)','for(target=0;target<4;target=target+1)')
    tb=tb.replace('2:dut.post.cfg_revoke=1;endcase','2:dut.post.cfg_revoke=1;3:dut.ffn.cfg_revoke=1;endcase')
    tb=tb.replace('dut.projection.cfg_revoke=0;dut.post.cfg_revoke=0;','dut.projection.cfg_revoke=0;dut.post.cfg_revoke=0;dut.ffn.cfg_revoke=0;')
    old='        $display("LAYER_PASS'
    new='''        mode=7;wanted_status=11;
        for(target=1;target<=4;target=target+1)begin
            defaults();case(target)1:dut.front.cfg_foreign=1;2:dut.projection.cfg_foreign=1;3:dut.post.cfg_foreign=1;4:dut.ffn.cfg_foreign=1;endcase
            admission();wait_response();`CHECK(injected,"foreign identity not injected") release_job();
        end
'''
    assert tb.count(old)==1;tb=tb.replace(old,new+old)
    return tb.replace('tb_ntrpu_llama32_layer_protocol_r3','tb_ntrpu_llama32_layer_protocol_r4'),peers

if __name__=='__main__':
    tb,_=render()
    with (ROOT/'fpga/tb/tb_ntrpu_llama32_layer_protocol_r4.sv').open('x') as f:f.write(tb)
