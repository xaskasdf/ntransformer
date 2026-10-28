`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_state_read_guard_r2;
    logic clk=0,rst_n=0;
    always #5 clk=~clk;
    logic [31:0] tile_start=0;
    logic core_ce=0;
    logic [6:0] core_address=0;
    logic [63:0] context_q=0;
    wire context_ce,discarded_request;
    wire [6:0] context_address;
    wire [63:0] core_q;
    integer seed=1,checks=0;
    logic [63:0] prior=0,expected=0;
    ntrpu_attention_state_read_guard_r2 dut(.*);
    function automatic logic [63:0] word_at(input int a);
        return 64'hdeadbeef01234500^(64'(seed)<<32)^64'(a);
    endfunction
    always @(posedge clk)begin
        if(!rst_n)context_q<=0;
        else if(context_ce)context_q<=word_at(int'(context_address));
    end
    initial begin
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        repeat(2)@(negedge clk);rst_n=1;
        for(integer t=0;t<3;t=t+1)begin
            for(integer a=0;a<128;a=a+1)begin
                for(integer c=0;c<2;c=c+1)begin
                    @(negedge clk);tile_start=t==0 ? 0:t==1 ? 64:131008;
                    core_address=a;core_ce=c;#1;
                    if(context_address!==7'(a) || discarded_request!==(c!=0 && t!=0 && a==104) ||
                       context_ce!==(c!=0 && (t==0 ? a>=104:a!=104)))$fatal(1,"STATE_GUARD_FAIL request");
                    if(c!=0 && (t==0 ? a>=104:a!=104))prior=word_at(a);
                    expected=(t==0 || (c!=0 && a==104)) ? 64'd0:prior;
                    @(posedge clk);#1;
                    if(core_q!==expected)$fatal(1,"STATE_GUARD_FAIL response");
                    $display("STATE_GUARD_CHECK tile=%0d address=%0d ce=%0d forwarded=%0d drop=%0d data=%016h",
                        tile_start,a,c,context_ce,discarded_request,core_q);checks++;
                end
            end
        end
        @(negedge clk);tile_start=64;core_ce=1;core_address=103;
        @(posedge clk);#1;if(core_q!==word_at(103))$fatal(1,"STATE_GUARD_FAIL tail load");
        @(negedge clk);core_address=104;#1;
        if(core_q!==word_at(103) || context_ce)$fatal(1,"STATE_GUARD_FAIL tail advance");
        @(posedge clk);#1;if(core_q!==0)$fatal(1,"STATE_GUARD_FAIL sentinel response");
        @(negedge clk);rst_n=0;core_ce=0;
        @(posedge clk);#1;if(core_q!==0 || dut.discarded_response)$fatal(1,"STATE_GUARD_FAIL reset");
        @(negedge clk);rst_n=1;core_ce=1;core_address=0;
        @(posedge clk);#1;if(core_q!==word_at(0))$fatal(1,"STATE_GUARD_FAIL recovery");
        $display("STATE_GUARD_PASS seed=%0d checks=%0d tail=1 reset=1 recovery=1",seed,checks);$finish;
    end
endmodule
`default_nettype wire
