`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_kv_read_guard_r1;
    logic clk=0;always #2.5 clk=~clk;
    logic rst_n=0,core_ce=0;
    logic [31:0] tile_count=1,context_q=32'hdeadbeef;
    logic [11:0] core_address=0;
    wire context_ce,discarded_request;
    wire [11:0] context_address;
    wire [31:0] core_q;
    ntrpu_attention_kv_read_guard_r1 dut(.*);
    integer seed=1,checks=0;
    // A one-cycle memory peer with recognizable data for every address.
    always @(posedge clk)if(context_ce)context_q<=32'h3f000000 | {20'd0,context_address};
    task automatic step(input integer count,input integer address,input logic enabled);
        logic drop;
        logic [31:0] want;
        @(negedge clk);tile_count=count;core_address=address;core_ce=enabled;#1;
        drop=enabled && count>0 && count<64 && address==count*64;
        if(context_ce!==(enabled&&!drop) || discarded_request!==drop || context_address!==12'(address))
            $fatal(1,"KV_READ_GUARD_FAIL request");
        want=drop ? 32'd0:(context_ce ? (32'h3f000000 | 32'(address)):context_q);
        @(posedge clk);#1;
        if(core_q!==want)$fatal(1,"KV_READ_GUARD_FAIL response");
        checks=checks+1;
        $display("KV_READ_GUARD_CHECK count=%0d address=%0d ce=%0d drop=%0d data=%08h",count,address,enabled,drop,core_q);
    endtask
    initial begin
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        repeat(2)@(negedge clk);rst_n=1;
        for(integer count=0;count<=65;count=count+1)begin
            // Exhaust the address space, including malformed and full tiles.
            for(integer address=0;address<4096;address=address+1)
                step(count,address,1);
            step(count,(count*64)%4096,0);
        end
        // The final useful RAM response must survive next-address advancement.
        @(negedge clk);tile_count=1;core_ce=1;core_address=63;
        @(posedge clk);#0.1;core_address=64;#0.9;
        if(core_q!==32'h3f00003f || context_ce)$fatal(1,"KV_READ_GUARD_FAIL last valid word lost");
        @(posedge clk);#1;if(core_q!==0)$fatal(1,"KV_READ_GUARD_FAIL sentinel response");
        @(negedge clk);rst_n=0;core_ce=0;@(posedge clk);#1;
        if(core_q!==context_q)$fatal(1,"KV_READ_GUARD_FAIL reset pending");
        $display("KV_READ_GUARD_PASS seed=%0d checks=%0d tail=1 reset=1",seed,checks);$finish;
    end
endmodule
`default_nettype wire
