`timescale 1ns / 1ps
// Actual AMD AXI-Lite CDC + retained ledgers and management mailboxes.
// Peer/host behavioral drivers are protocol fixtures; no PCIe/MIG instance.
module tb_ntrpu_axil_pair_drain #(parameter integer DEVICE_PERIOD_PS=5000);
    reg mgmt_clk=0,host_clk=0,device_clk=0,host_run=1,device_run=1;
    always #5 mgmt_clk=~mgmt_clk;
    initial begin #0.7;forever begin #3.5;if(host_run)host_clk=~host_clk;end end
    initial begin #1.1;forever begin #(DEVICE_PERIOD_PS/2000.0);if(device_run)device_clk=~device_clk;end end
    reg cold=0,hold_reset=1,counter_clear=1;
    wire hn,dn,hd,dd,hdm,ddm;
    ntrpu_reset_release hr(.clk(host_clk),.hold_async(hold_reset),.released(hn));
    ntrpu_reset_release dr(.clk(device_clk),.hold_async(hold_reset),.released(dn));
    ntrpu_reset_hold_counter hc(.clk(host_clk),.clear_async(counter_clear),.done(hd));
    ntrpu_reset_hold_counter dc(.clk(device_clk),.clear_async(counter_clear),.done(dd));
    ntrpu_reset_release hqual(.clk(mgmt_clk),.hold_async(!hd),.released(hdm));
    ntrpu_reset_release dqual(.clk(mgmt_clk),.hold_async(!dd),.released(ddm));
    integer held_h=0,held_d=0;
    always @(posedge host_clk)if(!hn)held_h=held_h+1;
    always @(posedge device_clk)if(!dn)held_d=held_d+1;

    reg [31:0] h_awaddr=0,h_wdata=0,h_araddr=0;
    reg [3:0] h_wstrb=15;
    reg h_awvalid=0,h_wvalid=0,h_bready=0,h_arvalid=0,h_rready=0;
    wire h_awready,h_wready,h_bvalid,h_arready,h_rvalid;
    wire [1:0] h_bresp,h_rresp;
    wire [31:0] h_rdata;
    wire [31:0] d_awaddr,d_wdata,d_araddr;
    wire [3:0] d_wstrb;
    wire [2:0] d_awprot,d_arprot;
    wire d_awvalid,d_wvalid,d_bready,d_arvalid,d_rready;
    wire d_awready,d_wready,d_arready;
    reg d_bvalid=0,d_rvalid=0;
    reg [1:0] d_bresp=0,d_rresp=0,read_code=0;
    reg [31:0] d_rdata=0;
    ntrpu_axil_cdc_recode cdc(
        .s_axi_aclk(host_clk),.s_axi_aresetn(hn),.m_axi_aclk(device_clk),.m_axi_aresetn(dn),
        .s_axi_awaddr(h_awaddr),.s_axi_wdata(h_wdata),.s_axi_araddr(h_araddr),
        .s_axi_awprot(3'b000),.s_axi_arprot(3'b000),.s_axi_wstrb(h_wstrb),
        .s_axi_awvalid(h_awvalid),.s_axi_wvalid(h_wvalid),.s_axi_bready(h_bready),.s_axi_arvalid(h_arvalid),.s_axi_rready(h_rready),
        .s_axi_awready(h_awready),.s_axi_wready(h_wready),.s_axi_bvalid(h_bvalid),.s_axi_arready(h_arready),.s_axi_rvalid(h_rvalid),
        .s_axi_bresp(h_bresp),.s_axi_rresp(h_rresp),.s_axi_rdata(h_rdata),
        .m_axi_awaddr(d_awaddr),.m_axi_wdata(d_wdata),.m_axi_araddr(d_araddr),
        .m_axi_awprot(d_awprot),.m_axi_arprot(d_arprot),.m_axi_wstrb(d_wstrb),
        .m_axi_awvalid(d_awvalid),.m_axi_wvalid(d_wvalid),.m_axi_bready(d_bready),.m_axi_arvalid(d_arvalid),.m_axi_rready(d_rready),
        .m_axi_awready(d_awready),.m_axi_wready(d_wready),.m_axi_bvalid(d_bvalid),.m_axi_arready(d_arready),.m_axi_rvalid(d_rvalid),
        .m_axi_bresp(d_bresp),.m_axi_rresp(d_rresp),.m_axi_rdata(d_rdata));

    reg aw_seen=0,w_seen=0,r_pending=0,allow_b=1,allow_r=1;
    reg [31:0] aw_saved=0,w_saved=0;
    reg [3:0] strb_saved=0;
    reg [31:0] mem[0:3];
    initial begin mem[0]=0;mem[1]=0;mem[2]=0;mem[3]=0;end
    assign d_awready=dn && !aw_seen && !d_bvalid;
    assign d_wready=dn && !w_seen && !d_bvalid;
    assign d_arready=dn && !r_pending;
    always @(posedge device_clk)begin
        if(!dn)begin aw_seen<=0;w_seen<=0;r_pending<=0;d_bvalid<=0;d_rvalid<=0;end
        else begin
            if(d_awvalid && d_awready)begin aw_seen<=1;aw_saved<=d_awaddr;end
            if(d_wvalid && d_wready)begin w_seen<=1;w_saved<=d_wdata;strb_saved<=d_wstrb;end
            if(aw_seen && w_seen && !d_bvalid && allow_b)begin
                for(integer b=0;b<4;b=b+1)if(strb_saved[b])mem[aw_saved[3:2]][b*8+:8]<=w_saved[b*8+:8];
                d_bvalid<=1;d_bresp<=0;
            end
            if(d_bvalid && d_bready)begin d_bvalid<=0;aw_seen<=0;w_seen<=0;end
            if(d_arvalid && d_arready)begin r_pending<=1;d_rdata<=mem[d_araddr[3:2]];d_rresp<=read_code;end
            if(r_pending && !d_rvalid && allow_r)d_rvalid<=1;
            if(d_rvalid && d_rready)begin d_rvalid<=0;r_pending<=0;end
        end
    end

    reg request=0,fenced=0;
    reg [63:0] tag=64'h100000000;
    wire pair_ack,hcert,dcert,hbusy,dbusy,hfault,dfault;
    wire [3:0] ha,hw,hq,da,dw,dq;
    ntrpu_axil_pair_drain pair_drain(
        .mgmt_clk(mgmt_clk),.host_clk(host_clk),.device_clk(device_clk),.cold_rst_n(cold),
        .req_valid(request),.generation(64'h100000001),.tag(tag),.host_source_fenced(fenced),
        .host_bus({h_awvalid,h_awready,h_wvalid,h_wready,h_bvalid,h_bready,h_arvalid,h_arready,h_rvalid,h_rready}),
        .device_bus({d_awvalid,d_awready,d_wvalid,d_wready,d_bvalid,d_bready,d_arvalid,d_arready,d_rvalid,d_rready}),
        .external_drain_ack(pair_ack),.host_certificate(hcert),.device_certificate(dcert),
        .host_mailbox_busy(hbusy),.device_mailbox_busy(dbusy),.host_fault(hfault),.device_fault(dfault),
        .host_aw_pending(ha),.host_w_pending(hw),.host_ar_pending(hq),
        .device_aw_pending(da),.device_w_pending(dw),.device_ar_pending(dq));

    // Read-only diagnostic probes of the installed public AMD RTL. They are
    // NOT inputs to the hardware certificate/reset controller.
`define NATIVE cdc.bridge.inst.gen_clock_conv.gen_async_lite_conv
`define CHANNEL_STATE(n) {`NATIVE.n.src_state,`NATIVE.n.dest_state,`NATIVE.n.src_send,`NATIVE.n.src_rcv,`NATIVE.n.dest_req,`NATIVE.n.dest_ack}
    wire [39:0] native_state={`CHANNEL_STATE(clock_conv_lite_fwd_aw),`CHANNEL_STATE(clock_conv_lite_fwd_w),`CHANNEL_STATE(clock_conv_lite_resp_b),`CHANNEL_STATE(clock_conv_lite_fwd_ar),`CHANNEL_STATE(clock_conv_lite_resp_r)};
`undef CHANNEL_STATE
`undef NATIVE
    integer scenario=-1,passes=0;
    integer haw=0,hwc=0,hbc=0,har=0,hrc=0,daw=0,dwc=0,dbc=0,dar=0,drc=0;
    task check(input bit ok,input string name);if(!ok)$fatal(1,"PAIR_ASSERT case=%0d %s",scenario,name);endtask
    task mtick(input integer n);repeat(n)@(negedge mgmt_clk);endtask
    task htick(input integer n);repeat(n)@(negedge host_clk);endtask
    task dtick(input integer n);repeat(n)@(negedge device_clk);endtask
    task mark(input string name);
        $display("PAIR_EVENT case=%0d name=%s ps=%0.0f hcert=%b dcert=%b ack=%b ha=%0d hw=%0d hq=%0d da=%0d dw=%0d dq=%0d hn=%b dn=%b hd=%b dd=%b held_h=%0d held_d=%0d native=%h",scenario,name,$realtime*1000,hcert,dcert,pair_ack,ha,hw,hq,da,dw,dq,hn,dn,hdm,ddm,held_h,held_d,native_state);
    endtask
    task ask;@(negedge mgmt_clk);tag=tag+1;request=1;endtask
    task await_pair;
        integer n;begin n=0;while(!pair_ack)begin mtick(1);n=n+1;check(n<2000,"bounded pair certificate");end
        check(!(hfault || dfault) && {ha,hw,hq,da,dw,dq}==0,"certified external inventory");mark("DRAIN");end
    endtask
    task end_request;
        integer n;begin @(negedge mgmt_clk);request=0;n=0;
        while(hbusy || dbusy)begin mtick(1);n=n+1;check(n<2000,"mailboxes drain before next case");end
        fenced=0;check(!pair_ack,"request withdrawal");end
    endtask
    task take_aw(input [31:0] address);
        integer n;begin @(negedge host_clk);h_awaddr=address;h_awvalid=1;n=0;
        @(posedge host_clk);while(!h_awready)begin @(posedge host_clk);n=n+1;check(n<1000,"AW timeout");end
        @(negedge host_clk);h_awvalid=0;end
    endtask
    task take_w(input [31:0] data,input [3:0] strb);
        integer n;begin @(negedge host_clk);h_wdata=data;h_wstrb=strb;h_wvalid=1;n=0;
        @(posedge host_clk);while(!h_wready)begin @(posedge host_clk);n=n+1;check(n<1000,"W timeout");end
        @(negedge host_clk);h_wvalid=0;end
    endtask
    task take_b;
        integer n;begin @(negedge host_clk);h_bready=1;n=0;
        @(posedge host_clk);while(!h_bvalid)begin @(posedge host_clk);n=n+1;check(n<1000,"B timeout");end
        check(h_bresp==0,"write response preserved");@(negedge host_clk);h_bready=0;end
    endtask
    task take_ar(input [31:0] address,input [1:0] response);
        integer n;begin read_code=response;@(negedge host_clk);h_araddr=address;h_arvalid=1;n=0;
        @(posedge host_clk);while(!h_arready)begin @(posedge host_clk);n=n+1;check(n<1000,"AR timeout");end
        @(negedge host_clk);h_arvalid=0;end
    endtask
    task take_r(input [31:0] expected,input [1:0] response);
        integer n;begin @(negedge host_clk);h_rready=1;n=0;
        @(posedge host_clk);while(!h_rvalid)begin @(posedge host_clk);n=n+1;check(n<1000,"R timeout");end
        check(h_rdata==expected && h_rresp==response,"read data and response preserved");@(negedge host_clk);h_rready=0;end
    endtask
    task release_reset;
        integer n;begin n=0;while(!(hdm && ddm))begin mtick(1);n=n+1;check(n<2000,"both real hold counters");end
        check(held_h>=64 && held_d>=64,"64 real reset edges per domain");
        hold_reset=0;n=0;while(!(hn && dn))begin mtick(1);n=n+1;check(n<100,"reset release synchronization");end
        dtick(12);htick(12);end
    endtask
    task pass_case(input string name);passes=passes+1;$display("PAIR_CASE_PASS case=%0d name=%s ps=%0.0f",scenario,name,$realtime*1000);endtask
    always @(posedge host_clk)if(hn)begin
        if(h_awvalid && h_awready)begin haw=haw+1;$display("PAIR_BUS side=H channel=AW ps=%0.0f addr=%h",$realtime*1000,h_awaddr);end
        if(h_wvalid && h_wready)begin hwc=hwc+1;$display("PAIR_BUS side=H channel=W ps=%0.0f data=%h strb=%h",$realtime*1000,h_wdata,h_wstrb);end
        if(h_bvalid && h_bready)begin hbc=hbc+1;$display("PAIR_BUS side=H channel=B ps=%0.0f resp=%h",$realtime*1000,h_bresp);end
        if(h_arvalid && h_arready)begin har=har+1;$display("PAIR_BUS side=H channel=AR ps=%0.0f addr=%h",$realtime*1000,h_araddr);end
        if(h_rvalid && h_rready)begin hrc=hrc+1;$display("PAIR_BUS side=H channel=R ps=%0.0f data=%h resp=%h",$realtime*1000,h_rdata,h_rresp);end
    end
    always @(posedge device_clk)if(dn)begin
        if(d_awvalid && d_awready)begin daw=daw+1;$display("PAIR_BUS side=D channel=AW ps=%0.0f addr=%h",$realtime*1000,d_awaddr);end
        if(d_wvalid && d_wready)begin dwc=dwc+1;$display("PAIR_BUS side=D channel=W ps=%0.0f data=%h strb=%h",$realtime*1000,d_wdata,d_wstrb);end
        if(d_bvalid && d_bready)begin dbc=dbc+1;$display("PAIR_BUS side=D channel=B ps=%0.0f resp=%h",$realtime*1000,d_bresp);end
        if(d_arvalid && d_arready)begin dar=dar+1;$display("PAIR_BUS side=D channel=AR ps=%0.0f addr=%h",$realtime*1000,d_araddr);end
        if(d_rvalid && d_rready)begin drc=drc+1;$display("PAIR_BUS side=D channel=R ps=%0.0f data=%h resp=%h",$realtime*1000,d_rdata,d_rresp);end
    end
    always @(posedge mgmt_clk)if(cold)check(!hfault && !dfault,"no unexpected/ghost response or inventory overflow");

    initial begin
        $display("PAIR_PROFILE device_period_ps=%0d host_period_ps=7000 mgmt_period_ps=10000",DEVICE_PERIOD_PS);
        #300;mtick(3);counter_clear=0;release_reset();cold=1;mtick(2);
        scenario=0;take_aw(0);fenced=1;ask();mtick(60);
        check(ha==1 && hw==0 && da==1 && dw==0 && !pair_ack,"AW-only is not drained");mark("AW_ONLY");
        take_w(32'hdeadbeef,15);mtick(70);
        check(h_bvalid && !pair_ack && ha==1 && hw==1,"blocked B prevents certificate");mark("B_BLOCKED");
        take_b();await_pair();end_request();pass_case("aw_before_w_blocked_b");

        scenario=1;take_w(32'h11223344,5);fenced=1;ask();mtick(60);
        check(hw==1 && ha==0 && dw==1 && da==0 && !pair_ack,"W-only is not drained");mark("W_ONLY");
        take_aw(0);take_b();await_pair();end_request();pass_case("w_before_aw_masked_write");

        scenario=2;allow_r=0;take_ar(0,3);fenced=1;ask();mtick(60);
        check(hq==1 && dq==1 && !pair_ack,"AR without response is not drained");mark("AR_PENDING");
        allow_r=1;mtick(70);check(h_rvalid && !pair_ack,"blocked R prevents certificate");mark("R_BLOCKED");
        take_r(32'hde22be44,3);await_pair();end_request();pass_case("read_pending_blocked_r");

        scenario=3;take_aw(4);take_w(32'hcafebabe,15);fenced=1;ask();take_b();
        begin:wait_host_proof integer n;n=0;while(!pair_drain.host_mailbox.dest_ack)begin htick(1);n=n+1;check(n<1000,"host proof accepted locally");end end
        host_run=0;mark("HOST_STOP_AFTER_PROOF");await_pair();
        mark("EXTERNAL_DRAIN_WITH_CLOCK_STOP");
        if(DEVICE_PERIOD_PS==37000)check(native_state!=0,"slow profile exposes native handshake tail");
        // Coordinated reset is a separate, explicit protocol action. Keep all
        // peers in reset; never interpret stopped-clock elapsed time as edges.
        held_h=0;held_d=0;hold_reset=1;counter_clear=1;mtick(8);counter_clear=0;
        mtick(300);check(!hn && !dn && !hdm && ddm && held_h==0,"stopped host cannot finish hold");mark("HOLD_CLOCK_STOPPED");
        host_run=1;
        begin:await_holds integer n;n=0;while(!(hdm && ddm))begin mtick(1);n=n+1;check(n<2000,"hold after clock return");end end
        check(native_state==0 && held_h>=64 && held_d>=64,"native state empty before release");mark("RESET_HELD_COMPLETE");
        release_reset();mark("RESET_RELEASED");end_request();pass_case("tail_and_coordinated_reset_clock_stop");

        scenario=4;take_ar(4,0);take_r(32'hcafebabe,0);
        take_aw(8);take_w(32'h76543210,15);take_b();take_ar(8,1);take_r(32'h76543210,1);
        take_ar(0,2);take_r(32'hde22be44,2);fenced=1;ask();await_pair();end_request();
        pass_case("post_reset_traffic_all_read_responses");

        scenario=5;ask();mtick(100);check(!hcert && !dcert && !pair_ack,"missing host isolation prevents certificate");mark("NO_FENCE");
        fenced=1;await_pair();
        @(negedge mgmt_clk);tag=tag+1;#0.001;check(!pair_ack,"new tag cannot combine old certificates");mark("NEW_TAG");
        await_pair();end_request();pass_case("fence_and_fresh_pair_identity");
        dtick(50);htick(50);
        check(haw==4 && hwc==4 && hbc==4 && daw==4 && dwc==4 && dbc==4,"four conserved writes");
        check(har==4 && hrc==4 && dar==4 && drc==4,"four conserved reads");
        check(native_state==0 && passes==6,"terminal internal drain and case count");
        $display("PAIR_SUITE_PASS cases=6 writes=4 reads=4 device_period_ps=%0d native_pcie=0 complete_endpoint_drain=0",DEVICE_PERIOD_PS);$finish;
    end
    initial begin #200000;$fatal(1,"PAIR_TIMEOUT");end
endmodule
