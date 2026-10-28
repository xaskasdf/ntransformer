`timescale 1ns/1ps
module tb_ntrpu_destination_dispatch;
    parameter integer DEPTH=8;
    localparam integer COUNT_BITS=$clog2(DEPTH+1);
    reg clk=0;always #2.5 clk=~clk;
    reg rst_n=0,stop=0,fault_in=0;
    reg [1:0] submit_valid=0,reserve_can=0;
    wire [1:0] submit_ready,head_valid,reserve_take;
    reg [463:0] submit_desc=0;
    wire [463:0] head_desc;
    wire [2*COUNT_BITS-1:0] queued;
    wire cmd_valid,cmd_channel,fetch_valid,fetch_channel,frame_done_ready;
    reg cmd_ready=0,fetch_ready=0,frame_done_valid=0;
    wire [231:0] cmd_desc,fetch_desc;
    reg [63:0] frame_done_tag=0;
    reg [31:0] frame_done_epoch=0;
    reg [8:0] frame_done_beats=0;
    reg fence_valid=0,fence_done_ready=0,sink_drained=1;
    wire fence_ready,fence_done_valid,fault,local_empty,drained;
    reg [63:0] fence_id=0;
    wire [63:0] fence_done_id;
    reg automatic_service=0;
    integer cycle=0,case_id=0,submitted[0:1],taken[0:1],commands=0,fetches=0,finished=0;
    reg [1:0] saw_submit,saw_take;
    reg saw_command,saw_fetch,saw_done;
    ntrpu_destination_dispatch #(.DEPTH(DEPTH)) dut(.*);

    function [231:0] descriptor(input integer tag);
        descriptor={32'h12345678,64'(tag),64'h100000000+64'(tag)*4096,
                    64'h800000000+64'(tag)*4096,8'(tag%2 ? 63 : 7)};
    endfunction
    task need(input bit ok,input string why);
        if(!ok)$fatal(1,"DISPATCH_%s case=%0d cycle=%0d depth=%0d",why,case_id,cycle,DEPTH);
    endtask
    task step;
        @(negedge clk);
        if(automatic_service)begin
            cmd_ready=1;fetch_ready=1;frame_done_valid=frame_done_ready;
            frame_done_tag=cmd_desc[199:136];frame_done_epoch=cmd_desc[231:200];
            frame_done_beats={1'b0,cmd_desc[7:0]}+1;
        end
        #0.1;
        saw_submit=submit_valid&submit_ready;saw_take=reserve_take;
        saw_command=cmd_valid&&cmd_ready;saw_fetch=fetch_valid&&fetch_ready;
        saw_done=frame_done_valid&&frame_done_ready&&frame_done_tag==cmd_desc[199:136]&&
            frame_done_epoch==cmd_desc[231:200]&&frame_done_beats=={1'b0,cmd_desc[7:0]}+1;
        $display("DQ E %0d %0d %0d %b %b %b %b %b %h %b %b %h %b %b %b %b %b %h %b %b %b %h %b %b %h %h %h %b %b %h %b %b %h %b %b %b",
            cycle,case_id,DEPTH,rst_n,stop,fault_in,submit_valid,submit_ready,submit_desc,
            head_valid,reserve_can,queued,reserve_take,cmd_valid,cmd_ready,cmd_channel,fetch_channel,cmd_desc,
            fetch_valid,fetch_ready,frame_done_ready,fetch_desc,frame_done_valid,sink_drained,
            frame_done_tag,frame_done_epoch,frame_done_beats,fence_valid,fence_ready,fence_id,
            fence_done_valid,fence_done_ready,fence_done_id,fault,local_empty,drained);
        if(rst_n)begin
            for(integer p=0;p<2;p++)begin
                if(saw_submit[p])submitted[p]++;
                if(saw_take[p])taken[p]++;
            end
            commands+=saw_command;fetches+=saw_fetch;finished+=saw_done;
        end
        @(posedge clk);#0.1;cycle++;
    endtask
    task reset_case(input integer number);
        rst_n=0;stop=0;fault_in=0;submit_valid=0;reserve_can=0;
        cmd_ready=0;fetch_ready=0;frame_done_valid=0;
        fence_valid=0;fence_done_ready=0;sink_drained=1;automatic_service=0;
        case_id=number;step;step;
        for(integer p=0;p<2;p++)begin submitted[p]=0;taken[p]=0;end
        commands=0;fetches=0;finished=0;rst_n=1;step;
    endtask
    task push_one(input integer channel,input integer tag);
        submit_valid=1<<channel;submit_desc[channel*232+:232]=descriptor(tag);
        begin: wait_accept
            for(integer k=0;k<1000;k++)begin step;if(saw_submit[channel])disable wait_accept;end
            need(0,"SUBMIT_TIMEOUT");
        end
        submit_valid=0;
    endtask
    task push_pair(input integer tag);
        submit_valid=3;submit_desc={descriptor(tag+1),descriptor(tag)};
        begin: wait_both
            for(integer k=0;k<1000;k++)begin
                step;submit_valid=submit_valid&~saw_submit;
                if(submit_valid==0)disable wait_both;
            end
            need(0,"PAIR_TIMEOUT");
        end
    endtask
    task until_empty;
        begin: waiting
            for(integer k=0;k<1000;k++)begin step;if(local_empty)disable waiting;end
            need(0,"DRAIN_TIMEOUT");
        end
        need(commands==fetches && fetches==finished,"LEASE_BALANCE");
    endtask
    task until_frame;
        cmd_ready=1;fetch_ready=1;
        begin: waiting
            for(integer k=0;k<100;k++)begin step;if(frame_done_ready)disable waiting;end
            need(0,"FRAME_TIMEOUT");
        end
        cmd_ready=0;fetch_ready=0;
    endtask
    task pass_case;
        $display("DQ CASE_PASS %0d depth=%0d submitted0=%0d submitted1=%0d taken0=%0d taken1=%0d commands=%0d fetches=%0d finished=%0d",
            case_id,DEPTH,submitted[0],submitted[1],taken[0],taken[1],commands,fetches,finished);
    endtask

    initial begin
        // Port 0 cannot reserve, but its full queue does not block port 1.
        reset_case(1);
        for(integer i=0;i<DEPTH;i++)push_one(0,10+i);
        need(!submit_ready[0],"FULL_QUEUE");reserve_can=2;automatic_service=1;
        for(integer i=0;i<3*DEPTH;i++)push_one(1,100+i);
        begin: wait_peer
            for(integer k=0;k<1000;k++)begin step;if(finished==3*DEPTH)disable wait_peer;end
            need(0,"PEER_DRAIN_TIMEOUT");
        end
        need(taken[0]==0&&taken[1]==3*DEPTH&&finished==3*DEPTH,"PEER_PROGRESS");
        need(!local_empty&&!drained,"RETAIN_BLOCKED_QUEUE");
        reserve_can=3;until_empty;need(drained,"FINAL_DRAIN");pass_case;

        // Simultaneous enqueue, fair grants, full/empty and pointer wrap.
        reset_case(2);
        for(integer i=0;i<DEPTH;i++)push_pair(200+2*i);
        reserve_can=3;automatic_service=1;
        for(integer i=0;i<4*DEPTH+3;i++)push_pair(300+2*i);
        until_empty;need(taken[0]==taken[1]&&!fault,"RR_WRAP");pass_case;

        // A committed lease survives stop and command/fetch/payload delays.
        reset_case(3);push_pair(500);reserve_can=3;
        step;need(cmd_valid,"COMMAND_EXPOSED");stop=1;
        repeat(5)begin step;need(cmd_valid&&!submit_ready&&!reserve_take,"COMMAND_HELD");end
        cmd_ready=1;step;cmd_ready=0;need(fetch_valid,"FETCH_AFTER_COMMAND");
        repeat(5)begin step;need(fetch_valid&&!reserve_take,"FETCH_HELD");end
        fetch_ready=1;step;fetch_ready=0;
        repeat(5)begin step;need(frame_done_ready&&!local_empty&&!reserve_take,"FRAME_HELD");end
        automatic_service=1;until_empty;
        need(finished==2&&!submit_ready&&!fault,"STOP_DRAINS_QUEUED");pass_case;

        // Global fence closes new submissions and waits for sink completions.
        reset_case(4);push_pair(600);sink_drained=0;
        fence_valid=1;fence_id=64'habcdef;submit_valid=3;submit_desc={descriptor(701),descriptor(700)};
        step;need(saw_submit==0,"FENCE_PRIORITY");fence_valid=0;submit_valid=0;
        reserve_can=3;automatic_service=1;until_empty;
        repeat(5)begin step;need(!fence_done_valid&&!drained&&!submit_ready,"FENCE_WAIT_SINK");end
        sink_drained=1;step;step;need(fence_done_valid,"FENCE_COMPLETE");
        sink_drained=0;
        repeat(5)begin step;need(fence_done_valid&&fence_done_id==64'habcdef&&!submit_ready,"FENCE_RESULT_HELD");end
        sink_drained=1;fence_done_ready=1;step;fence_done_ready=0;step;
        need(drained&&submit_ready==3&&!fault,"FENCE_REOPENS");pass_case;

        // A stale tag, stale epoch or truncated frame cannot release the lease.
        for(integer bad=0;bad<3;bad++)begin
            reset_case(5+bad);push_pair(800+bad*10);reserve_can=3;until_frame;
            frame_done_tag=cmd_desc[199:136];frame_done_epoch=cmd_desc[231:200];
            frame_done_beats={1'b0,cmd_desc[7:0]}+1;
            if(bad==0)frame_done_tag=frame_done_tag^1;
            if(bad==1)frame_done_epoch=frame_done_epoch^1;
            if(bad==2)frame_done_beats=frame_done_beats-1;
            frame_done_valid=1;step;frame_done_valid=0;
            need(fault&&frame_done_ready&&!submit_ready&&!reserve_take&&!drained,"BAD_DONE_QUARANTINE");
            automatic_service=1;step;automatic_service=0;frame_done_valid=0;
            repeat(5)step;
            need(fault&&finished==1&&!local_empty&&!reserve_take,"FAULT_RETAINS_QUEUE");pass_case;
        end

        reset_case(8);push_pair(900);reserve_can=3;step;need(cmd_valid,"EXTERNAL_FAULT_SETUP");
        fault_in=1;step;fault_in=0;
        repeat(5)begin step;need(fault&&cmd_valid&&!submit_ready,"FAULT_COMMAND_STABLE");end
        automatic_service=1;repeat(10)step;
        need(fault&&finished==1&&!local_empty&&!reserve_take,"FAULT_DRAINS_LEASE_ONLY");pass_case;
        $display("DQ SUITE_PASS depth=%0d cases=8 cycles=%0d",DEPTH,cycle);$finish;
    end
    initial begin #200000;$fatal(1,"DISPATCH_WATCHDOG");end
endmodule
