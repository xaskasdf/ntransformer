`timescale 1ns/1ps
// Contract-level test: a receiver offer may disappear before candidate commit.
// Resource/payload allocation is exercised separately by the integrated bench.
module tb_ntrpu_candidate_credit;
    reg clk=0;always #2.5 clk=~clk;
    reg rst_n=0,stop=0,fault_in=0;
    reg [1:0] submit_valid=0,reserve_can=0;
    reg [463:0] submit_desc=0;
    wire [1:0] submit_ready,head_valid,head_good,reserve_take;
    wire [463:0] head_desc;
    wire [7:0] queued;
    wire candidate_valid,candidate_channel;
    wire [231:0] candidate_desc;
    wire cmd_valid,cmd_channel,fetch_valid,fetch_channel;
    wire [231:0] cmd_desc,fetch_desc;
    reg cmd_ready=1,fetch_ready=1;
    wire frame_done_ready;
    wire frame_done_valid=frame_done_ready;
    wire [63:0] frame_done_tag=fetch_desc[199:136];
    wire [31:0] frame_done_epoch=fetch_desc[231:200];
    wire [8:0] frame_done_beats={1'b0,fetch_desc[7:0]}+9'd1;
    reg fence_valid=0,fence_done_ready=1,sink_drained=1;
    reg [63:0] fence_id=0;
    wire fence_ready,fence_done_valid,fault,local_empty,drained;
    wire [63:0] fence_done_id;
    integer cycle=0,takes=0,echoes=0,fetches=0,dones=0,case_id=0;
    integer channels[0:3];
    ntrpu_destination_candidate dut(.*);

    function [231:0] descriptor(input integer tag);
        descriptor={32'h12345678,64'(tag),64'h100000000+64'(tag)*4096,64'h800000000,8'd7};
    endfunction
    task need(input bit condition,input string label);
        if(!condition)$fatal(1,"CREDIT_%s case=%0d cycle=%0d",label,case_id,cycle);
    endtask
    task step;@(negedge clk);#0.2;endtask
    task reset_case(input integer number);
        rst_n=0;reserve_can=0;submit_valid=0;stop=0;fault_in=0;
        repeat(4)step;case_id=number;rst_n=1;step;
        submit_valid=3;submit_desc={descriptor(11),descriptor(10)};
        @(posedge clk);need(submit_ready==3,"BOTH_SUBMITTED");step;submit_valid=0;
        repeat(3)step;need(head_valid==3&&head_good==3&&queued==8'h11,"TWO_HEADS");
    endtask
    task wait_drain;
        begin: waiting
            for(integer i=0;i<100;i++)begin step;if(drained)disable waiting;end
            need(0,"DRAIN_TIMEOUT");
        end
        need(takes==2&&echoes==2&&fetches==2&&dones==2&&!fault,"EXACTLY_ONCE");
    endtask

    always @(posedge clk)begin
        cycle++;
        if(!rst_n)begin takes=0;echoes=0;fetches=0;dones=0;end
        else begin
            $display("CU S %0d %0d %b %b %b %b %h %h",cycle,case_id,
                reserve_can,candidate_valid,candidate_channel,reserve_take,queued,candidate_desc);
            need(reserve_take!=3&&((reserve_take&reserve_can)==reserve_take),"LIVE_OFFER_REQUIRED");
            if(|reserve_take)begin
                need(candidate_valid&&reserve_take==(1<<candidate_channel),"PREPARED_IDENTITY");
                need(candidate_desc===descriptor(10+int'(candidate_channel)),"CANDIDATE_METADATA");
                channels[takes]=int'(candidate_channel);takes++;
            end
            if(cmd_valid&&cmd_ready)echoes++;
            if(fetch_valid&&fetch_ready)fetches++;
            if(frame_done_valid&&frame_done_ready)dones++;
        end
    end

    initial begin
        reset_case(1);
        reserve_can=3;step;
        need(candidate_valid&&!candidate_channel&&takes==0&&queued==8'h11,"PREPARE_ZERO");
        reserve_can=2;#0.1;need(reserve_take==0,"WITHDRAW_PREVIOUS_OFFER");
        step;need(candidate_valid&&candidate_channel&&takes==0&&queued==8'h11,"REPICK_PEER_WITHOUT_POP");
        step;reserve_can=3;wait_drain;
        need(channels[0]==1&&channels[1]==0,"PEER_PROGRESS_AFTER_WITHDRAWAL");
        $display("CU CASE_PASS 1");

        reset_case(2);
        reserve_can=3;step;need(candidate_valid&&takes==0,"PREPARE_BEFORE_PAUSE");
        reserve_can=0;#0.1;need(reserve_take==0,"WITHDRAW_ALL");
        repeat(8)step;
        need(!candidate_valid&&takes==0&&queued==8'h11,"NO_STALE_GRANT");
        reserve_can=3;step;stop=1;wait_drain;
        need(submit_ready==0&&channels[0]==0&&channels[1]==1,"STOP_DRAINS_REPREPARED");
        $display("CU CASE_PASS 2");
        $display("CU SUITE_PASS cases=2");$finish;
    end
    initial begin #10000;$fatal(1,"CREDIT_WATCHDOG");end
endmodule
