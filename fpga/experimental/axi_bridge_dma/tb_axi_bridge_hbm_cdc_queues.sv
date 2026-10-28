`timescale 1ns/1ps
`default_nettype none
// Transport fixtures deliberately fill both directions. Real reader/writer
// composition is tested separately by tb_axi_bridge_hbm_cdc.
module tb_axi_bridge_hbm_cdc_queues;
    integer seed=0,stage=0,pc=0,rc=0;
    reg p_clk=0,r_clk=0,mgmt_clk=0,cold=1,r_run=0;
    always #2 p_clk=~p_clk;
    always #5 mgmt_clk=~mgmt_clk;
    initial begin
        if($value$plusargs("seed=%d",seed))begin end
        #(0.1+seed*0.7);forever begin #2.5;if(r_run)r_clk=~r_clk;end
    end
    // Explicit cold assertion also exercises reset while the reader has no
    // edges; do not depend on simulator scheduling of a time-zero constant.
    reg [2:0] pr=7,rr=7,mr=7;
    initial begin #0.2;cold=0;end
    always @(posedge p_clk or negedge cold)if(!cold)pr<=0;else pr<={pr[1:0],1'b1};
    always @(posedge r_clk or negedge cold)if(!cold)rr<=0;else rr<={rr[1:0],1'b1};
    always @(posedge mgmt_clk or negedge cold)if(!cold)mr<=0;else mr<={mr[1:0],1'b1};
    wire p_cold_rst_n=pr[2],r_cold_rst_n=rr[2],m_cold_rst_n=mr[2];
    reg p_stop=0,r_started=0,r_fault=0,drain_commands=0,drain_returns=0;
    wire p_started,p_fault,r_stop,clock_alive,armed,management_fault;
    reg [1:0] p_cmd_valid=0,r_cmd_ready=0,r_data_valid=0,r_data_last=0,r_cpl_valid=0;
    reg [1:0] p_data_ready=0,p_cpl_ready=0;
    wire [1:0] p_cmd_ready,r_cmd_valid,r_data_ready,r_cpl_ready,p_data_valid,p_data_last,p_cpl_valid;
    reg [127:0] p_cmd_addr=0,p_cmd_tag=0,r_data_tag=0,r_cpl_tag=0;
    reg [15:0] p_cmd_len=0;
    reg [1023:0] r_data=0;
    reg [3:0] r_cpl_status=0;
    reg [25:0] r_cpl_bytes=0;
    wire [127:0] r_cmd_addr,r_cmd_tag,p_data_tag,p_cpl_tag;
    wire [15:0] r_cmd_len;
    wire [1023:0] p_data;
    wire [3:0] p_cpl_status;
    wire [25:0] p_cpl_bytes;
    ntrpu_axi_bridge_hbm_cdc dut(.*);
    task require(input bit ok,input string msg);
        if(ok!==1'b1)$fatal(1,"HQ seed=%0d stage=%0d %s",seed,stage,msg);
    endtask
    task ptick(input integer n);repeat(n)@(negedge p_clk);endtask
    task rtick(input integer n);repeat(n)@(negedge r_clk);endtask
    function automatic [63:0] tag(input integer lane,input integer frame);
        return 64'habcd123400000001+(64'(lane)<<40)+64'(frame);
    endfunction
    function automatic [511:0] payload(input integer lane,input integer frame,input integer beat);
        for(integer b=0;b<64;b++)payload[b*8+:8]=8'((beat*64+b)*37 ^ (frame*13) ^ (lane*91) ^ seed);
    endfunction
    integer cmd_in[0:1],cmd_out[0:1],event_in[0:1],event_out[0:1];
    integer command_backpressure=0,return_backpressure=0,simultaneous=0;
    reg [135:0] commands[0:1][0:127];
    reg [592:0] events[0:1][0:255];
    reg [1:0] command_stalled=0,event_stalled=0;
    reg [135:0] held_command[0:1];
    reg [592:0] held_event[0:1];
    wire [592:0] p_event[0:1];
    for(genvar c=0;c<2;c++)begin
        assign p_event[c]=p_cpl_valid[c]?
            {1'b1,1'b0,p_cpl_tag[c*64+:64],p_cpl_status[c*2+:2],p_cpl_bytes[c*13+:13],512'b0}:
            {1'b0,p_data_last[c],p_data_tag[c*64+:64],2'b0,13'b0,p_data[c*512+:512]};
    end
    always @(negedge p_clk)begin
        for(integer c=0;c<2;c++)begin
            p_data_ready[c]=drain_returns&&(pc+seed+c)%5!=0;
            p_cpl_ready[c]=drain_returns&&(pc+seed+c)%7!=0;
        end
    end
    always @(negedge r_clk)begin
        for(integer c=0;c<2;c++)r_cmd_ready[c]=drain_commands&&(rc+seed+c)%4!=0;
    end
    always @(posedge p_clk)begin
        pc++;
        if(p_cold_rst_n)for(integer c=0;c<2;c++)begin
            if(p_cmd_valid[c]&&!p_cmd_ready[c])command_backpressure++;
            if(p_cmd_valid[c]&&p_cmd_ready[c])begin
                commands[c][cmd_in[c]]={p_cmd_addr[c*64+:64],p_cmd_tag[c*64+:64],p_cmd_len[c*8+:8]};cmd_in[c]++;
            end
            if(event_stalled[c])require((p_data_valid[c]||p_cpl_valid[c])&&p_event[c]===held_event[c],"return offer changed during stall/fault");
            event_stalled[c]=(p_data_valid[c]&&!p_data_ready[c])||(p_cpl_valid[c]&&!p_cpl_ready[c]);
            held_event[c]=p_event[c];
            require(!(p_data_valid[c]&&p_cpl_valid[c]),"one ordered output kind");
            if((p_data_valid[c]&&p_data_ready[c])||(p_cpl_valid[c]&&p_cpl_ready[c]))begin
                require(event_out[c]<event_in[c]&&p_event[c]===events[c][event_out[c]],"lost/reordered return event");
                $display("HQ EVENT stage=%0d lane=%0d index=%0d cpl=%b last=%b tag=%h status=%0d bytes=%0d data=%h",stage,c,event_out[c],p_event[c][592],p_event[c][591],p_event[c][590:527],p_event[c][526:525],p_event[c][524:512],p_event[c][511:0]);
                event_out[c]++;
            end
        end else event_stalled=0;
    end
    always @(posedge r_clk)begin
        rc++;
        if(r_cold_rst_n)for(integer c=0;c<2;c++)begin
            if(command_stalled[c])require(r_cmd_valid[c]&&{r_cmd_addr[c*64+:64],r_cmd_tag[c*64+:64],r_cmd_len[c*8+:8]}===held_command[c],"command offer changed during stall/fault");
            command_stalled[c]=r_cmd_valid[c]&&!r_cmd_ready[c];
            held_command[c]={r_cmd_addr[c*64+:64],r_cmd_tag[c*64+:64],r_cmd_len[c*8+:8]};
            if(r_cmd_valid[c]&&r_cmd_ready[c])begin
                require(cmd_out[c]<cmd_in[c]&&held_command[c]===commands[c][cmd_out[c]],"lost/reordered command");
                $display("HQ CMD stage=%0d lane=%0d index=%0d addr=%h tag=%h len=%0d",stage,c,cmd_out[c],r_cmd_addr[c*64+:64],r_cmd_tag[c*64+:64],r_cmd_len[c*8+:8]);cmd_out[c]++;
            end
            if(r_data_valid[c]&&!r_data_ready[c])return_backpressure++;
            if(r_data_valid[c]&&r_cpl_valid[c])begin
                simultaneous++;require(!r_cpl_ready[c],"CPL cannot overtake simultaneous data");
            end
            if(r_data_valid[c]&&r_data_ready[c])begin
                events[c][event_in[c]]={1'b0,r_data_last[c],r_data_tag[c*64+:64],2'b0,13'b0,r_data[c*512+:512]};event_in[c]++;
            end
            if(r_cpl_valid[c]&&r_cpl_ready[c])begin
                events[c][event_in[c]]={1'b1,1'b0,r_cpl_tag[c*64+:64],r_cpl_status[c*2+:2],r_cpl_bytes[c*13+:13],512'b0};event_in[c]++;
            end
        end else command_stalled=0;
    end
    task automatic send_commands(input integer lane,input integer first,input integer count);
        for(integer i=first;i<first+count;i++)begin
            @(negedge p_clk);p_cmd_valid[lane]=1;
            p_cmd_addr[lane*64+:64]=64'h100001000+(64'(lane)<<33)+64'(i*4096);
            p_cmd_tag[lane*64+:64]=tag(lane,i);p_cmd_len[lane*8+:8]=8'(i);
            do @(posedge p_clk);while(!p_cmd_ready[lane]);
            @(negedge p_clk);p_cmd_valid[lane]=0;
        end
    endtask
    task automatic send_frame(input integer lane,input integer frame,input integer beats);
        for(integer i=0;i<beats;i++)begin
            @(negedge r_clk);r_data_valid[lane]=1;r_data_last[lane]=i==beats-1;
            r_data[lane*512+:512]=payload(lane,frame,i);r_data_tag[lane*64+:64]=tag(lane,frame);
            if(i==beats-1)begin
                r_cpl_valid[lane]=1;r_cpl_tag[lane*64+:64]=tag(lane,frame);
                r_cpl_bytes[lane*13+:13]=13'(beats*64);r_cpl_status[lane*2+:2]=2'(frame%3);
            end
            do @(posedge r_clk);while(!r_data_ready[lane]);
            @(negedge r_clk);r_data_valid[lane]=0;
        end
        do @(posedge r_clk);while(!r_cpl_ready[lane]);
        @(negedge r_clk);r_cpl_valid[lane]=0;
    endtask
    initial begin
        for(integer c=0;c<2;c++)begin cmd_in[c]=0;cmd_out[c]=0;event_in[c]=0;event_out[c]=0;end
`ifndef XSIM
        $dumpfile("hbm_cdc_queues.vcd");$dumpvars(1,tb_axi_bridge_hbm_cdc_queues);
`endif
        ptick(8);cold=1;r_started=1;ptick(200);
        require(!p_started&&!armed&&!clock_alive,"stopped clock cannot qualify at boot");
        r_run=1;wait(p_started);ptick(5);stage=1;
        fork
            send_commands(0,0,8);
            send_commands(1,0,8);
            begin ptick(80);require(cmd_in[0]==5&&cmd_in[1]==5,"four RAM entries plus output slot");drain_commands=1;end
        join
        wait(cmd_out[0]==8&&cmd_out[1]==8);rtick(5);stage=2;
        fork
            send_frame(0,0,64);
            send_frame(1,0,64);
            begin rtick(100);require(event_in[0]==5&&event_in[1]==5,"return queue must apply backpressure");drain_returns=1;end
        join
        wait(event_out[0]==65&&event_out[1]==65);ptick(10);
        // Retain already queued data/commands during operational stop, then
        // drain those obligations without granting the source a new context.
        stage=3;drain_commands=0;drain_returns=0;ptick(3);
        fork send_commands(0,8,1);send_commands(1,8,1);send_frame(0,1,2);send_frame(1,1,2);join
        wait((&r_cmd_valid)&&(&p_data_valid));ptick(4);p_stop=1;wait(p_fault&&r_stop);ptick(50);
        require((&r_cmd_valid)&&(&p_data_valid)&&cmd_out[0]==8&&event_out[0]==65,"fault cannot erase queued offers");
        $display("HQ RETAIN commands=2 returns=6 fault=1");
        drain_commands=1;drain_returns=1;
        wait(cmd_out[0]==9&&cmd_out[1]==9&&event_out[0]==68&&event_out[1]==68);ptick(5);
        p_stop=0;ptick(20);require(p_fault&&r_stop&&!p_started,"operational recovery cannot clear stop");
        // Queue stale content; coordinated cold reset alone may discard it.
        stage=4;drain_commands=0;drain_returns=0;ptick(3);
        fork send_commands(0,9,1);send_commands(1,9,1);send_frame(0,2,1);send_frame(1,2,1);join
        wait((&r_cmd_valid)&&(&p_data_valid));@(negedge r_clk);r_run=0;#0.1;cold=0;r_started=0;
        ptick(5);require(!p_data_valid&&!p_cpl_valid&&!r_cmd_valid,"cold assertion invalidates stale offers with stopped clock");
        command_stalled=0;event_stalled=0;
        for(integer c=0;c<2;c++)begin cmd_in[c]=0;cmd_out[c]=0;event_in[c]=0;event_out[c]=0;end
        cold=1;ptick(200);require(!p_started&&!p_data_valid&&!p_cpl_valid,"no stale epoch after partial release");
        r_run=1;rtick(8);r_started=1;wait(p_started);ptick(10);
        require(!p_fault&&!r_stop&&!p_data_valid&&!p_cpl_valid&&!r_cmd_valid,"cold reset permits fresh epoch only");
        stage=5;drain_commands=1;drain_returns=1;
        fork send_commands(0,10,1);send_commands(1,10,1);send_frame(0,3,1);send_frame(1,3,1);join
        wait(cmd_out[0]==1&&cmd_out[1]==1&&event_out[0]==2&&event_out[1]==2);ptick(20);
        require(!p_data_valid&&!p_cpl_valid&&!r_cmd_valid,"exactly-once fresh epoch");
        require(command_backpressure>0&&return_backpressure>0&&simultaneous>=8,"required stress coverage");
        $display("HQ COVER command_stalls=%0d return_stalls=%0d simultaneous=%0d",command_backpressure,return_backpressure,simultaneous);
        $display("HQ PASS seed=%0d cold_recovery=1",seed);$finish;
    end
    initial begin #100000;$fatal(1,"HQ TIMEOUT");end
endmodule
`default_nettype wire
