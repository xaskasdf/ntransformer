`timescale 1ns / 1ps
// Real XPM CDCs/probes, retained request/response channels, controlled core ACK.
// This bank does not instantiate PCIe, AXI Clock Converters or the NTRD core.
module tb_ntrpu_host_loss_guard_r2;
    reg mgmt_clk=0,host_clk=0,source_clk=0,host_run=1,source_run=1;
    always #5 mgmt_clk=~mgmt_clk;
    initial begin #0.731;forever begin #3.65;if(host_run)host_clk=~host_clk;end end
    initial begin #0.319;forever begin #2.5;if(source_run)source_clk=~source_clk;end end
    reg cold=0,link_up=0,ep_reset=1,arm=0,peers_held=0,peers_ready=0;
    reg ack_enable=1,core_ack=0,source_fault=0;
    integer ack_age=0,scenario=-1;
    wire host_rst,source_rst,reset_req,host_admit,source_admit,host_alive,source_alive;
    wire invalid,loss_mgmt,waiting_ack,running;
    wire [4:0] causes;
    wire [3:0] state_debug;
    wire [63:0] first_tick;
    reg qv=0,pv=0,qr=0,pr=0;
    reg [16:0] qdata=0,pdata=0;
    wire qready,pready,qvalid,pvalid,qinvalid,pinvalid,qfence,pfence;
    wire [16:0] qout,pout;
    wire [1:0] qpending,ppending,qfault,pfault;
    wire [63:0] qa,qd,qb,pa,pd,pb;
    ntrpu_host_loss_guard dut(.mgmt_clk(mgmt_clk),.host_clk(host_clk),.source_clk(source_clk),
        .cold_rst_n(cold),.link_up_async(link_up),.endpoint_reset_async(ep_reset),
        .arm(arm),.peers_held(peers_held),.peers_ready(peers_ready),
        .source_invalid_ack(core_ack),.source_fault(source_fault),
        .host_reset_n(host_rst),.source_reset_n(source_rst),.peers_reset_req(reset_req),
        .host_admit(host_admit),.source_admit(source_admit),.host_alive(host_alive),.source_alive(source_alive),
        .source_invalidate(invalid),.loss_mgmt(loss_mgmt),.causes(causes),.first_loss_tick(first_tick),
        .waiting_ack(waiting_ack),.running(running),.state_debug(state_debug));
    ntrpu_retained_channel #(.WIDTH(17),.TO_BOUNDARY(0)) request_cut(
        .clk(source_clk),.cold_rst_n(cold),.boundary_rst_n(source_rst),.loss(invalid),
        .s_valid(qv),.s_ready(qready),.s_data(qdata),.m_valid(qvalid),.m_ready(qr),.m_data(qout),
        .invalid(qinvalid),.fenced(qfence),.pending(qpending),.accepted(qa),.delivered(qd),.abandoned(qb),.fault(qfault));
    ntrpu_retained_channel #(.WIDTH(17),.TO_BOUNDARY(1)) response_cut(
        .clk(source_clk),.cold_rst_n(cold),.boundary_rst_n(source_rst),.loss(invalid),
        .s_valid(pv),.s_ready(pready),.s_data(pdata),.m_valid(pvalid),.m_ready(pr),.m_data(pout),
        .invalid(pinvalid),.fenced(pfence),.pending(ppending),.accepted(pa),.delivered(pd),.abandoned(pb),.fault(pfault));
    always @(posedge source_clk) begin
        if(!cold) begin core_ack<=0;ack_age<=0;end
        else if(invalid && !core_ack) begin
            ack_age<=ack_age+1;
            if(ack_enable && ack_age>=3) core_ack<=1;
        end
        if(scenario>=0) begin
            $display("GUARD_S case=%0d t=%0.0f cold=%b reset=%b invalid=%b ack=%b fault_in=%b qv=%b qi=%b qr=%b qo=%b qdata=%h qout=%h qp=%0d qa=%0d qd=%0d qb=%0d qfault=%0d pv=%b pi=%b pr=%b po=%b pdata=%h pout=%h pp=%0d pa=%0d pd=%0d pb=%0d pfence=%b pfault=%0d",scenario,$realtime*1000,cold,source_rst,invalid,core_ack,source_fault,qv,qready,qr,qvalid,qdata,qout,qpending,qa,qd,qb,qfault,pv,pready,pr,pvalid,pdata,pout,ppending,pa,pd,pb,pfence,pfault);
        end
    end
    always @(posedge host_clk) if(scenario>=0)
        $display("GUARD_H case=%0d t=%0.0f cold=%b reset=%b admit=%b",scenario,$realtime*1000,cold,host_rst,host_admit);
    always @(posedge mgmt_clk) begin
        #0.001;
        if(scenario>=0) $display("GUARD_M case=%0d t=%0.0f cold=%b link=%b ep=%b arm=%b held=%b ready=%b hr=%b sr=%b ha=%b sa=%b hrun=%b srun=%b halive=%b salive=%b req=%b invalid=%b ack=%b ack_en=%b loss=%b causes=%0d tick=%0d wait=%b running=%b state=%0d",scenario,$realtime*1000,cold,link_up,ep_reset,arm,peers_held,peers_ready,host_rst,source_rst,host_admit,source_admit,host_run,source_run,host_alive,source_alive,reset_req,invalid,core_ack,ack_enable,loss_mgmt,causes,first_tick,waiting_ack,running,state_debug);
    end
    task automatic event_mark(input string name);
        $display("GUARD_EVENT case=%0d t=%0.0f name=%s",scenario,$realtime*1000,name);
    endtask
    task automatic mgmt_wait(input integer count);
        repeat(count) @(negedge mgmt_clk);
    endtask
    task automatic wait_running;
        integer n=0;
        while(!(running && host_admit && source_admit)) begin
            mgmt_wait(1);n++;if(n>1600)$fatal(1,"GUARD_BOOT_TIMEOUT");
        end
    endtask
    task automatic wait_loss;
        integer n=0;
        while(!loss_mgmt) begin mgmt_wait(1);n++;if(n>100)$fatal(1,"GUARD_LOSS_TIMEOUT");end
    endtask
    task automatic wait_faulted;
        integer n=0;
        while(state_debug!=8) begin mgmt_wait(1);n++;if(n>1600)$fatal(1,"GUARD_FAULT_TIMEOUT");end
        mgmt_wait(6);
        if(host_admit || source_admit || running || !invalid || !loss_mgmt || !host_rst || !source_rst)
            $fatal(1,"GUARD_REARM_OR_RESET");
    endtask
    task automatic start_case(input integer n);
        @(negedge mgmt_clk);scenario=n;cold=0;host_run=1;source_run=1;
        link_up=0;ep_reset=1;arm=0;peers_held=0;peers_ready=0;ack_enable=1;source_fault=0;
        qv=0;pv=0;qr=0;pr=0;qdata=0;pdata=0;
        mgmt_wait(32);cold=1;
        if(n==0) host_run=0;
        mgmt_wait(90);
        if(loss_mgmt || invalid || host_admit || source_admit || !reset_req)$fatal(1,"GUARD_INITIAL_ABSENCE");
        host_run=1;link_up=1;ep_reset=0;
        mgmt_wait(100);
        if(running || !reset_req || loss_mgmt)$fatal(1,"GUARD_ARM_REQUIRED");
        peers_held=1;arm=1;
        mgmt_wait(20);
        if(running || host_admit || source_admit)$fatal(1,"GUARD_PEER_READY_REQUIRED");
        peers_ready=1;wait_running();event_mark("running");
        for(integer k=0;k<2;k=k+1) begin
            @(negedge source_clk);qv=1;pv=1;qdata=17'h12340+k;pdata=17'h055a0+k;
            @(posedge source_clk);if(!qready || !pready)$fatal(1,"GUARD_FILL_BLOCKED");
        end
        @(negedge source_clk);qv=0;pv=0;
        mgmt_wait(5);
    endtask
    task automatic pulse_reset;
        @(negedge mgmt_clk);#1;event_mark("inject_reset");ep_reset=1;#2;ep_reset=0;
    endtask
    task automatic finish_case(input integer expected_cause);
        if(expected_cause!=0) begin
            wait_faulted();
            @(negedge source_clk);qv=1;qdata=17'h1ffff;
            repeat(8) @(negedge source_clk);
            pv=1;pdata=17'h066bb;
            @(posedge source_clk);if(!pready)$fatal(1,"GUARD_LATE_RESPONSE_BLOCKED");
            @(negedge source_clk);pv=0;qr=1;
        end else begin @(negedge source_clk);qr=1;pr=1;end
        mgmt_wait(20);
        if(qa!=2 || qd!=2 || qb!=0 || qpending!=0 || ppending!=0 || qfault!=0 || pfault!=0)
            $fatal(1,"GUARD_REQUEST_ACCOUNTING");
        if(expected_cause==0) begin
            if(pa!=2 || pd!=2 || pb!=0 || invalid || loss_mgmt)$fatal(1,"GUARD_NORMAL_RESPONSE");
        end else if(pa!=3 || pd!=0 || pb!=3 || causes!=expected_cause || !invalid || !core_ack)
            $fatal(1,"GUARD_FAULT_RESPONSE");
        $display("GUARD_CASE_PASS case=%0d causes=%0d qa=%0d qd=%0d qb=%0d pa=%0d pd=%0d pb=%0d",scenario,causes,qa,qd,qb,pa,pd,pb);
        @(negedge mgmt_clk);scenario=-1;cold=0;
    endtask
    initial begin
        #250;
        start_case(0);finish_case(0);
        start_case(1);pulse_reset();finish_case(2);
        start_case(2);@(negedge mgmt_clk);#1;event_mark("inject_link");link_up=0;#1;link_up=1;finish_case(1);
        for(integer n=3;n<=4;n=n+1) begin
            start_case(n);
            if(n==3) @(negedge host_clk);else @(posedge host_clk);
            #0.2;event_mark("stop_host");host_run=0;
            wait_loss();mgmt_wait(100);
            if(host_rst || source_rst || !invalid)$fatal(1,"GUARD_STOPPED_HOST_NOT_HELD");
            event_mark("resume_host");host_run=1;finish_case(4);
        end
        start_case(5);@(negedge source_clk);#0.2;event_mark("stop_source");source_run=0;
        wait_loss();mgmt_wait(32);
        if(reset_req || !host_rst || !source_rst || core_ack)$fatal(1,"GUARD_RESET_WITHOUT_SOURCE_ACK");
        event_mark("resume_source");source_run=1;finish_case(8);
        start_case(6);ack_enable=0;pulse_reset();wait_loss();mgmt_wait(200);
        if(reset_req || !invalid || !waiting_ack || !qvalid || !pvalid)$fatal(1,"GUARD_RESET_BEFORE_ACK");
        event_mark("allow_ack");ack_enable=1;finish_case(2);
        start_case(7);@(negedge source_clk);event_mark("inject_source_fault");source_fault=1;
        @(negedge source_clk);source_fault=0;finish_case(16);
        start_case(8);peers_held=0;pulse_reset();wait_loss();mgmt_wait(200);
        if(!reset_req || host_rst || source_rst || state_debug!=6)$fatal(1,"GUARD_PEERS_HELD_REQUIRED");
        event_mark("allow_peers_held");peers_held=1;finish_case(2);
        start_case(9);pulse_reset();wait_faulted();
        @(negedge mgmt_clk);#1;event_mark("inject_link_again");link_up=0;#1;link_up=1;
        mgmt_wait(20);
        if(!reset_req)$fatal(1,"GUARD_REPEATED_RESET_IGNORED");
        finish_case(3);
        $display("GUARD_SUITE_PASS cases=10");$finish;
    end
    initial begin #200000;$fatal(1,"GUARD_WATCHDOG");end
endmodule
