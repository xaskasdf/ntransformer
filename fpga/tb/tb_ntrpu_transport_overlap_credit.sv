`timescale 1ns/1ps
// Metadata preparation is legal without credit; allocation still requires a
// live offer at its accepting edge. No force or synthetic resource returns.
module tb_ntrpu_transport_overlap_credit;
    reg clk=0;always #2.5 clk=~clk;
    reg rst_n=0,stop=0,fault_in=0,hold_done=0,echo_seen=0;
    reg [1:0] submit_valid=0,reserve_can=0;
    reg [463:0] submit_desc=0;
    wire [1:0] submit_ready,head_valid,head_good,reserve_take;
    wire [463:0] head_desc;
    wire [7:0] queued;
    wire candidate_valid,candidate_channel;
    wire [231:0] candidate_desc;
    wire cmd_valid,cmd_channel,fetch_valid,fetch_channel;
    wire [231:0] cmd_desc,fetch_desc;
    wire cmd_ready=!echo_seen;
    reg fetch_ready=1;
    wire frame_done_ready;
    wire frame_done_valid=frame_done_ready&&!hold_done;
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
    task need(input bit ok,input string why);
        if(!ok)$fatal(1,"OVERLAP_CREDIT_%s case=%0d cycle=%0d",why,case_id,cycle);
    endtask
    task step;@(negedge clk);#0.2;endtask
    task reset_case(input integer number);
        rst_n=0;reserve_can=0;submit_valid=0;stop=0;fault_in=0;hold_done=0;fetch_ready=1;
        repeat(4)step;case_id=number;rst_n=1;step;
        submit_valid=3;submit_desc={descriptor(11),descriptor(10)};
        @(posedge clk);need(submit_ready==3,"BOTH_SUBMITTED");step;submit_valid=0;
        repeat(4)step;
        need(candidate_valid&&!candidate_channel&&takes==0&&queued==8'h11,"METADATA_WITHOUT_CREDIT");
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
        if(!rst_n)begin takes=0;echoes=0;fetches=0;dones=0;echo_seen<=0;end
        else begin
            $display("OC S %0d %0d %b %b %b %b %h %h %0d %b",cycle,case_id,
                reserve_can,candidate_valid,candidate_channel,reserve_take,queued,candidate_desc,dut.state,fault);
            need(reserve_take!=3&&((reserve_take&reserve_can)==reserve_take),"LIVE_OFFER");
            if(|reserve_take)begin
                need(!fault&&dut.state==0&&candidate_valid&&reserve_take==(1<<candidate_channel),"COMMIT_STATE");
                need(candidate_desc===descriptor(10+int'(candidate_channel)),"PREPARED_IDENTITY");
                need(takes<2,"NO_DUPLICATE_ALLOCATION");channels[takes]=int'(candidate_channel);takes++;
                echo_seen<=0;
            end
            if(cmd_valid&&cmd_ready)begin echoes++;echo_seen<=1;end
            if(fetch_valid&&fetch_ready)fetches++;
            if(frame_done_valid&&frame_done_ready)dones++;
        end
    end
    initial begin
        reset_case(1);
        // Withdraw the preferred offer before its accepting edge.
        reserve_can=3;#0.1;need(reserve_take==1,"PREPARED_PREFERRED");reserve_can=2;
        #0.1;need(reserve_take==0,"WITHDRAW_SELECTED");
        step;need(candidate_channel&&takes==0&&queued==8'h11,"REPICK_WITHOUT_POP");
        step;reserve_can=3;wait_drain;
        need(channels[0]==1&&channels[1]==0,"PEER_AFTER_WITHDRAWAL");$display("OC CASE_PASS 1");

        reset_case(2);repeat(8)begin step;need(takes==0&&queued==8'h11,"NO_CREDIT_NO_ALLOCATION");end
        stop=1;reserve_can=3;wait_drain;need(submit_ready==0,"STOP_DRAINS_ACCEPTED");$display("OC CASE_PASS 2");

        reset_case(3);fault_in=1;reserve_can=3;step;fault_in=0;repeat(8)step;
        need(fault&&!candidate_valid&&takes==0&&queued==8'h11&&!fetch_valid,"FAULT_QUARANTINE");$display("OC CASE_PASS 3");

        reset_case(4);hold_done=1;reserve_can=3;
        begin: wait_frame
            for(integer i=0;i<40;i++)begin step;if(frame_done_ready&&candidate_valid)disable wait_frame;end
            need(0,"FRAME_TIMEOUT");
        end
        need(takes==1&&candidate_channel&&candidate_desc===descriptor(11),"NEXT_METADATA_WHILE_BUSY");
        reserve_can=0;hold_done=0;repeat(8)step;
        need(takes==1&&candidate_valid&&queued==8'h10&&dut.state==0,"RECHECK_AFTER_BUSY");
        reserve_can=2;wait_drain;$display("OC CASE_PASS 4");

        reset_case(5);fetch_ready=0;reserve_can=3;
        begin: wait_fetch
            for(integer i=0;i<40;i++)begin step;if(fetch_valid)disable wait_fetch;end
            need(0,"FETCH_TIMEOUT");
        end
        repeat(8)begin step;need(fetch_valid&&fetch_desc===descriptor(10)&&takes==1&&echoes==1,"HELD_FETCH_NO_REECHO");end
        fetch_ready=1;wait_drain;$display("OC CASE_PASS 5");
        $display("OC SUITE_PASS cases=5");$finish;
    end
    initial begin #10000;$fatal(1,"OVERLAP_CREDIT_WATCHDOG");end
endmodule
