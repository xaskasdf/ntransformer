"""Exercise the real bank's unchanged AXI memory service without arithmetic."""
import hashlib
from pathlib import Path
from build_ntrpu_llama32_layer_r1 import render,ROOT

def build():
    source=(ROOT/'fpga/tb/tb_ntrpu_llama32_layer_real_r2.sv').read_text()
    body=source[source.index('    localparam [63:0] SEQ='):source.index('`define RP')]
    assert 'dut.' not in body
    _,proof=render();bus={}
    for n,(d,w,a) in proof['public'].items():
        if n.startswith('tail_axi_'):bus[n[9:]]=(d,w)
    lines=['`timescale 1ns/1ps','`define CHECK(C,M) if((C)!==1\'b1) $fatal(1,"LREAL_FAIL cycle=%0d %s",cycle,M);',
        'module tb_ntrpu_llama32_layer_memory_r1;',"    localparam integer FRAMES=1017348;reg clk=0,rst_n=0;reg [1:0] tail_selected=0;integer mode=0;"]
    for suffix,(d,w) in bus.items():lines.append(f'    reg {w} bus_{suffix}[0:4];')
    lines += [body,r'''
    task step;begin @(posedge clk);#1;@(negedge clk);#0.1;end endtask
    task write_one(input integer p,r,before_aw,input bit corrupt);
        integer count;reg [511:0] value;
        begin
            value=golden[starts[r]];if(corrupt)value[0]=!value[0];
            bus_awaddr[p]=region_address(r);bus_awlen[p]=0;bus_awsize[p]=6;bus_awburst[p]=1;
            bus_wdata[p]=value;bus_wstrb[p]='1;bus_wlast[p]=mode!=7;
            if(before_aw)begin
                bus_wvalid[p]=1;count=0;while(!bus_wready[p])begin step();count++;`CHECK(count<20,"test W deadline") end
                step();bus_wvalid[p]=0;
            end
            bus_awvalid[p]=1;count=0;while(!bus_awready[p])begin step();count++;`CHECK(count<20,"test AW deadline") end
            step();bus_awvalid[p]=0;
            if(!before_aw)begin
                bus_wvalid[p]=1;count=0;while(!bus_wready[p])begin step();count++;`CHECK(count<20,"test W deadline") end
                step();bus_wvalid[p]=0;
            end
            count=0;while(!bus_bvalid[p])begin step();count++;`CHECK(count<20,"test B deadline") end
            bus_bready[p]=1;step();bus_bready[p]=0;
        end
    endtask
    task read_one(input integer p,r,input bit crossing);
        integer count;begin
            bus_araddr[p]=region_address(r);bus_arlen[p]=crossing ? 1:0;bus_arsize[p]=6;bus_arburst[p]=1;
            bus_arvalid[p]=1;count=0;while(!bus_arready[p])begin step();count++;`CHECK(count<20,"test AR deadline") end
            step();bus_arvalid[p]=0;count=0;
            while(!bus_rvalid[p])begin step();count++;`CHECK(count<20,"test R deadline") end
            `CHECK(bus_rdata[p]===memory[starts[r]],"memory test returned incorrect bytes")
            // Keep the memory response stalled for three clocks, then accept.
            repeat(3)begin step();`CHECK(bus_rvalid[p] && bus_rdata[p]===memory[starts[r]],"stalled R changed") end
            bus_rready[p]=1;step();bus_rready[p]=0;
        end
    endtask
    initial begin
        if(!$value$plusargs("MODE=%d",mode))mode=0;
@INIT_BUS@
        for(integer p=0;p<5;p++)begin
            read_active[p]=0;aw_active[p]=0;r_count[p]=0;r_index[p]=0;w_count[p]=0;w_beats[p]=0;
            ars[p]=0;rs[p]=0;aws[p]=0;ws[p]=0;bs[p]=0;r_stalls[p]=0;w_stalls[p]=0;
            ar_stalled[p]=0;aw_stalled[p]=0;w_stalled[p]=0;
            for(integer r=0;r<25;r++)begin read_hash[p][r]=64'hcbf29ce484222325;region_reads[p][r]=0;end
        end
        for(integer r=0;r<25;r++)begin starts[r]=r*128;sizes[r]=128;writable[r]=1;writes[r]=0;end
        writable[0]=0;memory[0]=512'habcdef1234567890;valid_frame[0]=1;
        golden[starts[2]]=512'h102030405060708090a0b0c0;golden[starts[6]]=512'hffeeddccbbaa99887766554433221100;
        repeat(3)step();rst_n=1;step();
        case(mode)
            0:begin
                read_one(0,0,0);write_one(0,2,1,0);read_one(0,2,0);
                write_one(0,6,0,0);read_one(1,6,0);
                `CHECK(writes[2]==1 && writes[6]==1 && rs[0]==2 && rs[1]==1,"Memory model accounting")
                $display("LAYER_MEMORY_PASS writes=2 reads=3 W_before_AW=1 cross_master=1");$finish;
            end
            1:read_one(0,2,0);
            2:write_one(0,0,0,0);
            3:write_one(0,2,0,1);
            4:begin write_one(0,2,0,0);write_one(0,2,0,0);end
            5:read_one(0,0,1);
            6:read_one(2,0,0);
            7:write_one(0,2,0,0);
        endcase
        $fatal(1,"MEMORY_TEST_FAILED_TO_REJECT mode=%0d",mode);
    end
    initial begin #10000;$fatal(1,"MEMORY_TEST_DEADLINE");end
endmodule
`undef CHECK
''']
    text='\n'.join(lines).replace('@INIT_BUS@','\n'.join(f"        for(integer p=0;p<5;p++)bus_{suffix}[p]='0;" for suffix in bus))
    return text,dict(bank_sha256=hashlib.sha256((ROOT/'fpga/tb/tb_ntrpu_llama32_layer_real_r2.sv').read_bytes()).hexdigest(),
        memory_body_sha256=hashlib.sha256(body.encode()).hexdigest(),arithmetic_executed=False)

if __name__=='__main__':
    text,proof=build()
    with (ROOT/'fpga/tb/tb_ntrpu_llama32_layer_memory_r1.sv').open('x') as f:f.write(text)
    print(proof)
