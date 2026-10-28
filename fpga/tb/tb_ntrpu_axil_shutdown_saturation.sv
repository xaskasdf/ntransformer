`timescale 1ns / 1ps
// Actual AMD AXI-Lite CDC + retained ledgers and management mailboxes.
// Peer/host behavioral drivers are protocol fixtures; no PCIe/MIG instance.
module tb_ntrpu_axil_shutdown_saturation #(parameter integer DEVICE_PERIOD_PS=5000);
    reg mgmt_clk=0,host_clk=0,device_clk=0,host_run=1,device_run=1;
    always #5 mgmt_clk=~mgmt_clk;
    initial begin #0.7;forever begin #3.5;if(host_run)host_clk=~host_clk;end end
    initial begin #1.1;forever begin #(DEVICE_PERIOD_PS/2000.0);if(device_run)device_clk=~device_clk;end end
    reg cold=0,actuator_cold=0,hold_request=0,phase_valid=0;
    reg [2:0] phase_kind=0;
    reg [63:0] phase_tag=64'h300000000;
    wire hn,dn,hdm,ddm,hold_reset,phase_ack,host_up,device_up;
    wire [2:0] ack_kind,actuator_state;
    wire [63:0] ack_gen,ack_tag;
    // The fixture manager/subordinate use the same reset rails. External peer
    // phase inputs are held true here; independent negative tests cover them.
    ntrpu_local_reset_actuator actuator(
        .mgmt_clk(mgmt_clk),.host_clk(host_clk),.device_clk(device_clk),.cold_rst_n(actuator_cold),
        .req_valid(phase_valid),.hold_request(hold_request),.peers_held(1'b1),.peers_ready(1'b1),
        .req_kind(phase_kind),.req_generation(64'h100000001),.req_tag(phase_tag),
        .ack_valid(phase_ack),.ack_kind(ack_kind),.ack_generation(ack_gen),.ack_tag(ack_tag),
        .host_reset_n(hn),.device_reset_n(dn),.peers_reset_req(hold_reset),
        .host_held(hdm),.device_held(ddm),.host_up(host_up),.device_up(device_up),.state_debug(actuator_state));
    integer held_h=0,held_d=0;
    always @(posedge host_clk)if(!hn)held_h=held_h+1;
    always @(posedge device_clk)if(!dn)held_d=held_d+1;

    reg [31:0] u_awaddr=0,u_wdata=0,u_araddr=0;
    reg [3:0] u_wstrb=15;
    reg u_awvalid=0,u_wvalid=0,u_bready=0,u_arvalid=0,u_rready=0;
    wire u_awready,u_wready,u_bvalid,u_arready,u_rvalid;
    wire [1:0] u_bresp,u_rresp;
    wire [31:0] u_rdata;
    wire [31:0] d_awaddr,d_wdata,d_araddr;
    wire [3:0] d_wstrb;
    wire [2:0] d_awprot,d_arprot;
    wire d_awvalid,d_wvalid,d_bready,d_arvalid,d_rready;
    wire d_awready,d_wready,d_arready;
    reg d_bvalid=0,d_rvalid=0;
    reg [1:0] d_bresp=0,d_rresp=0,read_code=0;
    reg [31:0] d_rdata=0;
    wire [31:0] h_awaddr,h_wdata,h_araddr,h_rdata;
    wire [2:0] h_awprot,h_arprot;
    wire [3:0] h_wstrb;
    wire [1:0] h_bresp,h_rresp;
    wire h_awvalid,h_wvalid,h_bready,h_arvalid,h_rready;
    wire h_awready,h_wready,h_bvalid,h_arready,h_rvalid;
    reg shutdown_request=0;
    wire shutdown_active,shutdown_requested,wr_shutdown,rd_shutdown,irq,wr_irq,rd_irq,manager_reset_n;
    ntrpu_reset_release manager_cold_release(.clk(host_clk),.hold_async(!cold),.released(manager_reset_n));
    ntrpu_host_axil_shutdown shutdown_manager(
        .clk(host_clk),.resetn(manager_reset_n),.request_shutdown(shutdown_request),
        .shutdown_requested(shutdown_requested),.in_shutdown(shutdown_active),
        .wr_in_shutdown(wr_shutdown),.rd_in_shutdown(rd_shutdown),.irq(irq),.wr_irq(wr_irq),.rd_irq(rd_irq),
        .s_axi_awaddr(u_awaddr),.s_axi_awprot(3'b010),.s_axi_awvalid(u_awvalid),.s_axi_awready(u_awready),
        .s_axi_wdata(u_wdata),.s_axi_wstrb(u_wstrb),.s_axi_wvalid(u_wvalid),.s_axi_wready(u_wready),
        .s_axi_bresp(u_bresp),.s_axi_bvalid(u_bvalid),.s_axi_bready(u_bready),
        .s_axi_araddr(u_araddr),.s_axi_arprot(3'b101),.s_axi_arvalid(u_arvalid),.s_axi_arready(u_arready),
        .s_axi_rdata(u_rdata),.s_axi_rresp(u_rresp),.s_axi_rvalid(u_rvalid),.s_axi_rready(u_rready),
        .m_axi_awaddr(h_awaddr),.m_axi_awprot(h_awprot),.m_axi_awvalid(h_awvalid),.m_axi_awready(h_awready),
        .m_axi_wdata(h_wdata),.m_axi_wstrb(h_wstrb),.m_axi_wvalid(h_wvalid),.m_axi_wready(h_wready),
        .m_axi_bresp(h_bresp),.m_axi_bvalid(h_bvalid),.m_axi_bready(h_bready),
        .m_axi_araddr(h_araddr),.m_axi_arprot(h_arprot),.m_axi_arvalid(h_arvalid),.m_axi_arready(h_arready),
        .m_axi_rdata(h_rdata),.m_axi_rresp(h_rresp),.m_axi_rvalid(h_rvalid),.m_axi_rready(h_rready));
    wire ufault,upstream_idle;
    wire [7:0] ua,uw,uq;
    ntrpu_axil_drain_monitor #(.COUNT_WIDTH(8)) upstream_ledger(
        .clk(host_clk),.cold_rst_n(cold),.req_valid(1'b0),.req_tag(64'b0),.source_fenced(1'b0),
        .awvalid(u_awvalid),.awready(u_awready),.wvalid(u_wvalid),.wready(u_wready),
        .bvalid(u_bvalid),.bready(u_bready),.arvalid(u_arvalid),.arready(u_arready),
        .rvalid(u_rvalid),.rready(u_rready),.ack_valid(),.ack_tag(),.protocol_fault(ufault),
        .aw_pending(ua),.w_pending(uw),.ar_pending(uq));
    assign upstream_idle=!ufault && {ua,uw,uq}==0 && !(u_awvalid || u_wvalid || u_arvalid || u_bvalid || u_rvalid);
    ntrpu_axil_cdc_recode cdc(
        .s_axi_aclk(host_clk),.s_axi_aresetn(hn),.m_axi_aclk(device_clk),.m_axi_aresetn(dn),
        .s_axi_awaddr(h_awaddr),.s_axi_wdata(h_wdata),.s_axi_araddr(h_araddr),
        .s_axi_awprot(h_awprot),.s_axi_arprot(h_arprot),.s_axi_wstrb(h_wstrb),
        .s_axi_awvalid(h_awvalid),.s_axi_wvalid(h_wvalid),.s_axi_bready(h_bready),.s_axi_arvalid(h_arvalid),.s_axi_rready(h_rready),
        .s_axi_awready(h_awready),.s_axi_wready(h_wready),.s_axi_bvalid(h_bvalid),.s_axi_arready(h_arready),.s_axi_rvalid(h_rvalid),
        .s_axi_bresp(h_bresp),.s_axi_rresp(h_rresp),.s_axi_rdata(h_rdata),
        .m_axi_awaddr(d_awaddr),.m_axi_wdata(d_wdata),.m_axi_araddr(d_araddr),
        .m_axi_awprot(d_awprot),.m_axi_arprot(d_arprot),.m_axi_wstrb(d_wstrb),
        .m_axi_awvalid(d_awvalid),.m_axi_wvalid(d_wvalid),.m_axi_bready(d_bready),.m_axi_arvalid(d_arvalid),.m_axi_rready(d_rready),
        .m_axi_awready(d_awready),.m_axi_wready(d_wready),.m_axi_bvalid(d_bvalid),.m_axi_arready(d_arready),.m_axi_rvalid(d_rvalid),
        .m_axi_bresp(d_bresp),.m_axi_rresp(d_rresp),.m_axi_rdata(d_rdata));

    reg aw_seen=0,w_seen=0,r_pending=0,allow_b=1,allow_r=1,allow_aw=1,allow_w=1;
    reg [31:0] aw_saved=0,w_saved=0;
    reg [3:0] strb_saved=0;
    reg [31:0] mem[0:8191];
    initial for(integer n=0;n<8192;n=n+1)mem[n]=32'h13579bdf ^ (n*32'h01020408);
    assign d_awready=dn && allow_aw && !aw_seen && !d_bvalid;
    assign d_wready=dn && allow_w && !w_seen && !d_bvalid;
    assign d_arready=dn && !r_pending;
    always @(posedge device_clk)begin
        if(!dn)begin aw_seen<=0;w_seen<=0;r_pending<=0;d_bvalid<=0;d_rvalid<=0;end
        else begin
            if(d_awvalid && d_awready)begin aw_seen<=1;aw_saved<=d_awaddr;end
            if(d_wvalid && d_wready)begin w_seen<=1;w_saved<=d_wdata;strb_saved<=d_wstrb;end
            if(aw_seen && w_seen && !d_bvalid && allow_b)begin
                for(integer b=0;b<4;b=b+1)if(strb_saved[b])mem[aw_saved[14:2]][b*8+:8]<=w_saved[b*8+:8];
                d_bvalid<=1;d_bresp<=aw_saved[3:2];
            end
            if(d_bvalid && d_bready)begin d_bvalid<=0;aw_seen<=0;w_seen<=0;end
            if(d_arvalid && d_arready)begin r_pending<=1;d_rdata<=mem[d_araddr[14:2]];d_rresp<=d_araddr[3:2];end
            if(r_pending && !d_rvalid && allow_r)d_rvalid<=1;
            if(d_rvalid && d_rready)begin d_rvalid<=0;r_pending<=0;end
        end
    end

    reg request=0;
    wire fenced=shutdown_active;
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
    integer uaw=0,uwc=0,ubc=0,uar=0,urc=0,daw=0,dwc=0,dbc=0,dar=0,drc=0;
    task check(input bit ok,input string name);if(!ok)$fatal(1,"SHUT_ASSERT case=%0d %s",scenario,name);endtask
    task mtick(input integer n);repeat(n)@(negedge mgmt_clk);endtask
    task htick(input integer n);repeat(n)@(negedge host_clk);endtask
    task dtick(input integer n);repeat(n)@(negedge device_clk);endtask
    task mark(input string name);
        $display("SHUT_EVENT case=%0d name=%s ps=%0.0f hcert=%b dcert=%b ack=%b ha=%0d hw=%0d hq=%0d da=%0d dw=%0d dq=%0d hn=%b dn=%b hd=%b dd=%b held_h=%0d held_d=%0d native=%h shut=%b ua=%0d uw=%0d uq=%0d up_idle=%b manager_reset_n=%b",scenario,name,$realtime*1000,hcert,dcert,pair_ack,ha,hw,hq,da,dw,dq,hn,dn,hdm,ddm,held_h,held_d,native_state,shutdown_active,ua,uw,uq,upstream_idle,manager_reset_n);
    endtask
    task ask;@(negedge mgmt_clk);tag=tag+1;request=1;endtask
    task await_pair;
        integer n;begin n=0;while(!pair_ack)begin mtick(1);n=n+1;check(n<2000,"bounded pair certificate");end
        check(!(hfault || dfault) && {ha,hw,hq,da,dw,dq}==0,"certified external inventory");mark("DRAIN");end
    endtask
    task end_request;
        integer n;begin @(negedge mgmt_clk);request=0;#0.001;n=0;
        while(hbusy || dbusy)begin mtick(1);n=n+1;check(n<2000,"mailboxes drain before next case");end
        check(!pair_ack,"request withdrawal");end
    endtask
    task take_aw(input [31:0] address);
        integer n;begin @(negedge host_clk);u_awaddr=address;u_awvalid=1;n=0;
        @(posedge host_clk);while(!u_awready)begin @(posedge host_clk);n=n+1;check(n<1000,"AW timeout");end
        @(negedge host_clk);u_awvalid=0;end
    endtask
    task take_w(input [31:0] data,input [3:0] strb);
        integer n;begin @(negedge host_clk);u_wdata=data;u_wstrb=strb;u_wvalid=1;n=0;
        @(posedge host_clk);while(!u_wready)begin @(posedge host_clk);n=n+1;check(n<1000,"W timeout");end
        @(negedge host_clk);u_wvalid=0;end
    endtask
    task take_b(input [1:0] expected);
        integer n;begin @(negedge host_clk);u_bready=1;n=0;
        @(posedge host_clk);while(!u_bvalid)begin @(posedge host_clk);n=n+1;check(n<1000,"B timeout");end
        check(u_bresp==expected,"write response preserved");@(negedge host_clk);u_bready=0;end
    endtask
    task take_ar(input [31:0] address,input [1:0] response);
        integer n;begin read_code=response;@(negedge host_clk);u_araddr=address;u_arvalid=1;n=0;
        @(posedge host_clk);while(!u_arready)begin @(posedge host_clk);n=n+1;check(n<1000,"AR timeout");end
        @(negedge host_clk);u_arvalid=0;end
    endtask
    task take_r(input [31:0] expected,input [1:0] response);
        integer n;begin @(negedge host_clk);u_rready=1;n=0;
        @(posedge host_clk);while(!u_rvalid)begin @(posedge host_clk);n=n+1;check(n<1000,"R timeout");end
        check((response==2 || u_rdata==expected) && u_rresp==response,"read data and response preserved");@(negedge host_clk);u_rready=0;end
    endtask
    task start_hold;
        begin @(negedge mgmt_clk);held_h=0;held_d=0;phase_tag=phase_tag+1;
        phase_kind=2;phase_valid=1;hold_request=1;mtick(8);
        check(hold_reset && !hn && !dn,"actuator asserts both rails");
        $display("AUTO_RESET event=START ps=%0.0f tag=%h",$realtime*1000,phase_tag);end
    endtask
    task release_reset;
        integer n;begin n=0;while(!phase_ack)begin mtick(1);n=n+1;check(n<2000,"held phase ACK");end
        check(ack_kind==2 && ack_tag==phase_tag && ack_gen==64'h100000001 && held_h>=64 && held_d>=64,"held identity and actual edges");
        $display("AUTO_RESET event=HELD ps=%0.0f tag=%h he=%0d de=%0d",$realtime*1000,phase_tag,held_h,held_d);
        @(negedge mgmt_clk);phase_tag=phase_tag+1;phase_kind=5;hold_request=0;#0.001;check(!phase_ack,"no old held response on ready phase");
        n=0;while(!phase_ack)begin mtick(1);n=n+1;check(n<2000,"ready phase ACK");end
        check(ack_kind==5 && ack_tag==phase_tag && ack_gen==64'h100000001 && hn && dn && !hold_reset,"ready identity and release");
        $display("AUTO_RESET event=READY ps=%0.0f tag=%h",$realtime*1000,phase_tag);
        phase_valid=0;dtick(12);htick(12);end
    endtask
    task pass_case(input string name);passes=passes+1;$display("SHUT_CASE_PASS case=%0d name=%s ps=%0.0f",scenario,name,$realtime*1000);endtask
    always @(posedge host_clk)if(manager_reset_n)begin
        if(u_awvalid && u_awready)begin uaw=uaw+1;$display("SAT_BUS case=%0d close=%b side=U channel=AW ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,u_awaddr);end
        if(u_wvalid && u_wready)begin uwc=uwc+1;$display("SAT_BUS case=%0d close=%b side=U channel=W ps=%0.0f data=%h strb=%h",scenario,shutdown_request,$realtime*1000,u_wdata,u_wstrb);end
        if(u_bvalid && u_bready)begin ubc=ubc+1;$display("SAT_BUS case=%0d close=%b side=U channel=B ps=%0.0f resp=%h",scenario,shutdown_request,$realtime*1000,u_bresp);end
        if(u_arvalid && u_arready)begin uar=uar+1;$display("SAT_BUS case=%0d close=%b side=U channel=AR ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,u_araddr);end
        if(u_rvalid && u_rready)begin urc=urc+1;$display("SAT_BUS case=%0d close=%b side=U channel=R ps=%0.0f data=%h resp=%h",scenario,shutdown_request,$realtime*1000,u_rdata,u_rresp);end
    end
    integer haw=0,hwc=0,hbc=0,har=0,hrc=0;
    always @(posedge host_clk)if(hn)begin
        if(h_awvalid && h_awready)begin haw=haw+1;$display("SAT_BUS case=%0d close=%b side=H channel=AW ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,h_awaddr);end
        if(h_wvalid && h_wready)begin hwc=hwc+1;$display("SAT_BUS case=%0d close=%b side=H channel=W ps=%0.0f data=%h strb=%h",scenario,shutdown_request,$realtime*1000,h_wdata,h_wstrb);end
        if(h_bvalid && h_bready)begin hbc=hbc+1;$display("SAT_BUS case=%0d close=%b side=H channel=B ps=%0.0f resp=%h",scenario,shutdown_request,$realtime*1000,h_bresp);end
        if(h_arvalid && h_arready)begin har=har+1;$display("SAT_BUS case=%0d close=%b side=H channel=AR ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,h_araddr);end
        if(h_rvalid && h_rready)begin hrc=hrc+1;$display("SAT_BUS case=%0d close=%b side=H channel=R ps=%0.0f data=%h resp=%h",scenario,shutdown_request,$realtime*1000,h_rdata,h_rresp);end
    end
    always @(posedge device_clk)if(dn)begin
        if(d_awvalid && d_awready)begin daw=daw+1;$display("SAT_BUS case=%0d close=%b side=D channel=AW ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,d_awaddr);end
        if(d_wvalid && d_wready)begin dwc=dwc+1;$display("SAT_BUS case=%0d close=%b side=D channel=W ps=%0.0f data=%h strb=%h",scenario,shutdown_request,$realtime*1000,d_wdata,d_wstrb);end
        if(d_bvalid && d_bready)begin dbc=dbc+1;$display("SAT_BUS case=%0d close=%b side=D channel=B ps=%0.0f resp=%h",scenario,shutdown_request,$realtime*1000,d_bresp);end
        if(d_arvalid && d_arready)begin dar=dar+1;$display("SAT_BUS case=%0d close=%b side=D channel=AR ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,d_araddr);end
        if(d_rvalid && d_rready)begin drc=drc+1;$display("SAT_BUS case=%0d close=%b side=D channel=R ps=%0.0f data=%h resp=%h",scenario,shutdown_request,$realtime*1000,d_rdata,d_rresp);end
    end
    always @(posedge mgmt_clk)if(cold)check(!hfault && !dfault && !ufault,"no unexpected/ghost response or inventory overflow");

    task enter_shutdown;
        begin @(negedge host_clk);shutdown_request=1;ask();end
    endtask
    task reopen;
        integer n;begin check(upstream_idle,"reopen fixture only after all upstream completions");
        @(negedge host_clk);shutdown_request=0;n=0;
        while(shutdown_active)begin htick(1);n=n+1;check(n<1000,"bounded fixture reopen");end
        htick(12);end
    endtask
    task await_bvalid;
        integer n;begin n=0;while(!u_bvalid)begin htick(1);n=n+1;check(n<2000,"B presented");end end
    endtask
    task await_rvalid;
        integer n;begin n=0;while(!u_rvalid)begin htick(1);n=n+1;check(n<2000,"R presented");end end
    endtask
    // Closing must preserve every offered VALID and its payload until transfer.
    reg aw_stall=0,w_stall=0,ar_stall=0,b_stall=0,r_stall=0;
    reg [34:0] aw_payload=0,ar_payload=0;
    reg [35:0] w_payload=0;
    reg [1:0] b_payload=0;
    reg [33:0] r_payload=0;
    integer aw_stalls=0,w_stalls=0,ar_stalls=0,b_stalls=0,r_stalls=0;
    always @(posedge host_clk)if(manager_reset_n)begin
        if(aw_stall)check(h_awvalid && {h_awaddr,h_awprot}==aw_payload,"held AW preserved");
        if(w_stall)check(h_wvalid && {h_wdata,h_wstrb}==w_payload,"held W preserved");
        if(ar_stall)check(h_arvalid && {h_araddr,h_arprot}==ar_payload,"held AR preserved");
        if(b_stall)check(u_bvalid && u_bresp==b_payload,"held B preserved");
        if(r_stall)check(u_rvalid && {u_rdata,u_rresp}==r_payload,"held R preserved");
        aw_stall=h_awvalid && !h_awready;w_stall=h_wvalid && !h_wready;ar_stall=h_arvalid && !h_arready;
        b_stall=u_bvalid && !u_bready;r_stall=u_rvalid && !u_rready;
        aw_payload={h_awaddr,h_awprot};w_payload={h_wdata,h_wstrb};ar_payload={h_araddr,h_arprot};b_payload=u_bresp;r_payload={u_rdata,u_rresp};
        if(aw_stall)aw_stalls=aw_stalls+1;if(w_stall)w_stalls=w_stalls+1;if(ar_stall)ar_stalls=ar_stalls+1;
        if(b_stall)b_stalls=b_stalls+1;if(r_stall)r_stalls=r_stalls+1;
    end
    always @(posedge device_clk)if(dn)begin
        if(d_awvalid && d_awready)check(d_awprot==3'b010,"AWPROT preserved");
        if(d_arvalid && d_arready)check(d_arprot==3'b101,"ARPROT preserved");
    end

    // All five probes retain the 129th-style blocked offer (128 total here)
    // until its handshake. No VALID cancellation and no reset escape.
    localparam integer OFFERS=128;
    integer base_aw,base_w,base_b,base_ar,base_r;
    integer max_aw=0,max_w=0,max_ar=0;
    bit aw_done=1,w_done=1,ar_done=1;
    bit started=0;
    function [31:0] pattern(input integer n);
        pattern=32'hac000000 ^ (scenario<<16) ^ (n*32'h01030507);
    endfunction
    task send_aw_batch;
        begin aw_done=0;for(integer n=0;n<OFFERS;n=n+1)take_aw(32'h1000+scenario*1024+n*4);aw_done=1;end
    endtask
    task send_w_batch;
        begin w_done=0;for(integer n=0;n<OFFERS;n=n+1)take_w(pattern(n),(n%15)+1);w_done=1;end
    endtask
    task send_ar_batch;
        begin ar_done=0;for(integer n=0;n<OFFERS;n=n+1)take_ar(n*4,0);ar_done=1;end
    endtask
    task plateau(input integer mode);
        integer quiet,n,last,now;
        begin quiet=0;n=0;last=-1;
            while(quiet<256)begin
                htick(1);now=uaw+uwc+uar;
                if(now==last)quiet=quiet+1;else quiet=0;
                last=now;n=n+1;check(n<16000,"bounded acceptance plateau");
            end
            if(mode!=1 && mode!=3)check(u_awvalid && !u_awready && uaw>base_aw && uaw-base_aw<OFFERS,"AW saturation with held offer");
            if(mode!=0 && mode!=3)check(u_wvalid && !u_wready && uwc>base_w && uwc-base_w<OFFERS,"W saturation with held offer");
            if(mode>=3)check(u_arvalid && !u_arready && uar>base_ar && uar-base_ar<OFFERS,"AR saturation with held offer");
            if(mode==0)check(uwc==base_w && ubc==base_b,"AW-only plateau");
            if(mode==1)check(uaw==base_aw && ubc==base_b,"W-only plateau");
            check(ubc==base_b && urc==base_r,"responses really blocked");
            $display("SAT_PLATEAU case=%0d ps=%0.0f aw=%0d w=%0d ar=%0d ua=%0d uw=%0d uq=%0d quiet=%0d awv=%b awr=%b wv=%b wr=%b arv=%b arr=%b bv=%b rv=%b",scenario,$realtime*1000,uaw-base_aw,uwc-base_w,uar-base_ar,ua,uw,uq,quiet,u_awvalid,u_awready,u_wvalid,u_wready,u_arvalid,u_arready,u_bvalid,u_rvalid);
        end
    endtask
    task finish_batch(input integer writes,input integer reads);
        integer n;
        begin n=0;
            while(ubc-base_b<writes || urc-base_r<reads || !aw_done || !w_done || !ar_done)begin
                htick(1);n=n+1;check(n<16000,"all offers must complete without reset");
            end
            htick(128);u_bready=0;u_rready=0;
            check(uaw-base_aw==writes && uwc-base_w==writes && ubc-base_b==writes && uar-base_ar==reads && urc-base_r==reads,"exact upstream completion counts");
            await_pair();check(upstream_idle,"all upstream obligations retired");
            check(haw==hwc && haw==hbc && har==hrc && daw==dwc && daw==dbc && dar==drc && haw==daw && har==dar,"every downstream operation completed once");
            mark("SAT_DRAIN");end_request();
        end
    endtask
    task reopen_probe;
        integer before_w,before_r;
        reg [31:0] address,initial_word,expected;
        begin before_w=daw;before_r=dar;reopen();htick(128);dtick(128);
            check(daw==before_w && dar==before_r && upstream_idle,"rejected operations cannot replay after reopen");
            $display("SAT_REOPEN case=%0d ps=%0.0f daw=%0d dar=%0d idle=%b",scenario,$realtime*1000,daw,dar,upstream_idle);
            address=32'h7000+scenario*4;
            initial_word=32'h13579bdf ^ ((address>>2)*32'h01020408);
            expected=(initial_word & 32'hff00ff00) | ((32'ha1b2c3d4+scenario)&32'h00ff00ff);
            take_aw(address);take_w(32'ha1b2c3d4+scenario,5);take_b(address[3:2]);
            take_ar(address,0);take_r(expected,address[3:2]);htick(128);dtick(128);
            check(upstream_idle && native_state==0 && daw==before_w+1 && dar==before_r+1,"fresh post-reopen traffic and all native tails complete");
            $display("SAT_CASE_PASS case=%0d ps=%0.0f",scenario,$realtime*1000);
        end
    endtask
    // Upstream request stability, complementing H requests and U responses.
    reg uaws=0,uws=0,uars=0;
    reg [31:0] uap=0,urp=0;
    reg [35:0] uwp=0;
    integer uas=0,uws_count=0,urs=0;
    always @(posedge host_clk)if(started)begin
        check(cold && manager_reset_n && hn,"no operational reset during saturation");
        if(uaws)check(u_awvalid && u_awaddr==uap,"upstream AW held until accepted");
        if(uws)check(u_wvalid && {u_wdata,u_wstrb}==uwp,"upstream W held until accepted");
        if(uars)check(u_arvalid && u_araddr==urp,"upstream AR held until accepted");
        uaws=u_awvalid && !u_awready;uws=u_wvalid && !u_wready;uars=u_arvalid && !u_arready;
        uap=u_awaddr;uwp={u_wdata,u_wstrb};urp=u_araddr;
        if(uaws)uas=uas+1;if(uws)uws_count=uws_count+1;if(uars)urs=urs+1;
        #0.001;
        if(ua>max_aw)max_aw=ua;if(uw>max_w)max_w=uw;if(uq>max_ar)max_ar=uq;
    end
    always @(posedge device_clk)if(started)check(dn,"no device reset during saturation");
    initial begin
        $display("SAT_PROFILE device_period_ps=%0d host_period_ps=7000 mgmt_period_ps=10000 observer_width=8 offers=128",DEVICE_PERIOD_PS);
        #300;actuator_cold=1;mtick(3);start_hold();release_reset();cold=1;htick(64);started=1;
        for(integer mode=0;mode<5;mode=mode+1)begin
            scenario=mode;base_aw=uaw;base_w=uwc;base_b=ubc;base_ar=uar;base_r=urc;
            if(mode==0)fork send_aw_batch();join_none
            if(mode==1)fork send_w_batch();join_none
            if(mode==2)fork send_aw_batch();send_w_batch();join_none
            if(mode==3)fork send_ar_batch();join_none
            if(mode==4)fork send_aw_batch();send_w_batch();send_ar_batch();join_none
            plateau(mode);enter_shutdown();htick(64);mark("SAT_CLOSED_STILL_BLOCKED");
            // Requests already offered stay valid while the opposite half
            // is supplied and every response (including SLVERR) is consumed.
            if(mode==0)fork send_w_batch();join_none
            if(mode==1)fork send_aw_batch();join_none
            @(negedge host_clk);u_bready=1;u_rready=1;
            finish_batch(mode==3?0:OFFERS,mode>=3?OFFERS:0);
            reopen_probe();
        end
        enter_shutdown();await_pair();end_request();htick(128);dtick(128);
        check(upstream_idle && native_state==0,"terminal zero inventory");
        check(uas>0 && uws_count>0 && urs>0 && b_stalls>0 && r_stalls>0,"saturated request and response channels");
        $display("SAT_MAX aw=%0d w=%0d ar=%0d",max_aw,max_w,max_ar);
        $display("SAT_STALLS uaw=%0d uw=%0d uar=%0d hb=%0d hr=%0d",uas,uws_count,urs,b_stalls,r_stalls);
        $display("SAT_SUITE_PASS cases=5 writes=517 reads=261 device_period_ps=%0d native_endpoint=0 complete_dma_drain=0",DEVICE_PERIOD_PS);$finish;
    end
    initial begin #1000000;$fatal(1,"SAT_TIMEOUT");end
endmodule
