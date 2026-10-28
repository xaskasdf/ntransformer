`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_state_read_guard_r1;
    logic [31:0] tile_start;
    logic core_ce;
    logic [6:0] core_address;
    logic [63:0] context_q;
    wire context_ce;
    wire [6:0] context_address;
    wire [63:0] core_q;
    ntrpu_attention_state_read_guard_r1 dut(.*);
    integer seed=1,checks=0;
    initial begin
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        for(integer t=0;t<3;t=t+1)begin
            tile_start=t==0 ? 0:t==1 ? 64:131008;
            for(integer a=0;a<128;a=a+1)begin
                for(integer c=0;c<2;c=c+1)begin
                    core_address=a;core_ce=c;context_q=64'hdeadbeef01234567^(64'(seed)<<32)^64'(a);#1;
                    if(context_address!==7'(a) || context_ce!==(c!=0 && (t!=0 || a>=104)) ||
                       core_q!==(t==0 ? 64'd0:context_q))$fatal(1,"STATE_GUARD_FAIL");
                    $display("STATE_GUARD_CHECK tile=%0d address=%0d ce=%0d forwarded=%0d data=%016h",tile_start,a,c,context_ce,core_q);
                    checks=checks+1;
                end
            end
        end
        $display("STATE_GUARD_PASS seed=%0d checks=%0d",seed,checks);$finish;
    end
endmodule
`default_nettype wire
