`timescale 1ns / 1ps
// Actual AMD AXI-Lite CDC + retained ledgers and management mailboxes.
// Peer/host behavioral drivers are protocol fixtures; no PCIe/MIG instance.
module ntrpu_axil_section_lifetime_r2_case #(parameter integer DEVICE_PERIOD_PS=5000,MODE=0,EXPECT_REVOKE=1,STOP_SOURCE=1)(output reg done=0);
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
    wire c_bvalid,c_bready,c_rvalid,c_rready;
    wire [1:0] c_bresp,c_rresp;
    wire [31:0] c_rdata;
    wire [2:0] guard_fault_sources;
    wire [7:0] guard_aw_pending,guard_w_pending,guard_ar_pending;
    wire shutdown_request;
    wire shutdown_active,shutdown_requested,wr_shutdown,rd_shutdown,irq,wr_irq,rd_irq,manager_reset_n;
    ntrpu_reset_release manager_cold_release(.clk(host_clk),.hold_async(!cold),.released(manager_reset_n));

    wire [31:0] g_awaddr,g_wdata,g_araddr,g_rdata;
    wire [2:0] g_awprot,g_arprot;
    wire [3:0] g_wstrb;
    wire [1:0] g_bresp,g_rresp,gate_state;
    wire g_awvalid,g_wvalid,g_bready,g_arvalid,g_rready;
    wire g_awready,g_wready,g_bvalid,g_arready,g_rvalid;
    wire producer_fenced,gate_ack,local_idle,gate_fault,proof_request;
    wire [63:0] proof_tag,gate_tag;
    wire [7:0] ga,gw,gq,aw_left,w_left;
    wire ar_left;
    reg request=0;
    reg [63:0] tag=64'h100000000,generation=64'h100000001;
    wire [3:0] fault_mgmt;
    wire pair_ack,section_pair,upstream_certificate,hcert,dcert,hbusy,dbusy,ubusy,hfault,dfault;
    wire [3:0] ha=section.certificate.ha,hw=section.certificate.hw,hq=section.certificate.hq;
    wire [3:0] da=section.certificate.da,dw=section.certificate.dw,dq=section.certificate.dq;
    wire [7:0] cert_diag;
    assign hcert=cert_diag[7];assign dcert=cert_diag[6];
    wire [1:0] gate_fault_sources;
    ntrpu_axil_fence_section_r3 section(
        .host_clk(host_clk),
        .cold_rst_n(cold),
        .close_host(shutdown_request),
        .host_proof_request(proof_request),
        .host_proof_tag(proof_tag),
        .s_awaddr(u_awaddr),
        .s_wdata(u_wdata),
        .s_araddr(u_araddr),
        .s_awprot(3'b010),
        .s_arprot(3'b101),
        .s_wstrb(u_wstrb),
        .s_awvalid(u_awvalid),
        .s_wvalid(u_wvalid),
        .s_bready(u_bready),
        .s_arvalid(u_arvalid),
        .s_rready(u_rready),
        .s_awready(u_awready),
        .s_wready(u_wready),
        .s_bvalid(u_bvalid),
        .s_arready(u_arready),
        .s_rvalid(u_rvalid),
        .s_bresp(u_bresp),
        .s_rresp(u_rresp),
        .s_rdata(u_rdata),
        .m_awaddr(g_awaddr),
        .m_wdata(g_wdata),
        .m_araddr(g_araddr),
        .m_awprot(g_awprot),
        .m_arprot(g_arprot),
        .m_wstrb(g_wstrb),
        .m_awvalid(g_awvalid),
        .m_wvalid(g_wvalid),
        .m_bready(g_bready),
        .m_arvalid(g_arvalid),
        .m_rready(g_rready),
        .m_awready(g_awready),
        .m_wready(g_wready),
        .m_bvalid(g_bvalid),
        .m_arready(g_arready),
        .m_rvalid(g_rvalid),
        .m_bresp(g_bresp),
        .m_rresp(g_rresp),
        .m_rdata(g_rdata),
        .source_fenced(producer_fenced),
        .drain_ack(gate_ack),
        .drain_tag(gate_tag),
        .local_idle(local_idle),
        .protocol_fault(gate_fault),
        .aw_pending(ga),
        .w_pending(gw),
        .ar_pending(gq),
        .aw_left(aw_left),
        .w_left(w_left),
        .ar_left(ar_left),
        .state_debug(gate_state),
        .mgmt_clk(mgmt_clk),
        .device_clk(device_clk),
        .req_valid(request),
        .generation(generation),
        .tag(tag),
        .shutdown_active(shutdown_active),
        .host_requests({h_awvalid,h_awready,h_wvalid,h_wready,h_arvalid,h_arready}),
        .c_bvalid(c_bvalid),.c_bready(c_bready),.c_bresp(c_bresp),
        .c_rvalid(c_rvalid),.c_rready(c_rready),.c_rresp(c_rresp),.c_rdata(c_rdata),
        .h_bvalid(h_bvalid),.h_bready(h_bready),.h_bresp(h_bresp),
        .h_rvalid(h_rvalid),.h_rready(h_rready),.h_rresp(h_rresp),.h_rdata(h_rdata),
        .guard_fault_sources(guard_fault_sources),.guard_aw_pending(guard_aw_pending),
        .guard_w_pending(guard_w_pending),.guard_ar_pending(guard_ar_pending),
        .device_bus({d_awvalid,d_awready,d_wvalid,d_wready,d_bvalid,d_bready,d_arvalid,d_arready,d_rvalid,d_rready}),
        .section_ack(pair_ack),
        .upstream_certificate(upstream_certificate),
        .pair_certificate(section_pair),
        .host_fault(hfault),
        .device_fault(dfault),
        .fault_mgmt(fault_mgmt),
        .host_mailbox_busy(ubusy),
        .pair_host_busy(hbusy),
        .pair_device_busy(dbusy),
        .diagnostics(cert_diag),
        .fault_sources(gate_fault_sources));
    ntrpu_host_axil_shutdown shutdown_manager(
        .clk(host_clk),.resetn(manager_reset_n),.request_shutdown(shutdown_request),
        .shutdown_requested(shutdown_requested),.in_shutdown(shutdown_active),
        .wr_in_shutdown(wr_shutdown),.rd_in_shutdown(rd_shutdown),.irq(irq),.wr_irq(wr_irq),.rd_irq(rd_irq),
        .s_axi_awaddr(g_awaddr),.s_axi_awprot(g_awprot),.s_axi_awvalid(g_awvalid),.s_axi_awready(g_awready),
        .s_axi_wdata(g_wdata),.s_axi_wstrb(g_wstrb),.s_axi_wvalid(g_wvalid),.s_axi_wready(g_wready),
        .s_axi_bresp(g_bresp),.s_axi_bvalid(g_bvalid),.s_axi_bready(g_bready),
        .s_axi_araddr(g_araddr),.s_axi_arprot(g_arprot),.s_axi_arvalid(g_arvalid),.s_axi_arready(g_arready),
        .s_axi_rdata(g_rdata),.s_axi_rresp(g_rresp),.s_axi_rvalid(g_rvalid),.s_axi_rready(g_rready),
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
        .s_axi_awvalid(h_awvalid),.s_axi_wvalid(h_wvalid),.s_axi_bready(c_bready),.s_axi_arvalid(h_arvalid),.s_axi_rready(c_rready),
        .s_axi_awready(h_awready),.s_axi_wready(h_wready),.s_axi_bvalid(c_bvalid),.s_axi_arready(h_arready),.s_axi_rvalid(c_rvalid),
        .s_axi_bresp(c_bresp),.s_axi_rresp(c_rresp),.s_axi_rdata(c_rdata),
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



    // Read-only diagnostic probes of the installed public AMD RTL. They are
    // NOT inputs to the hardware certificate/reset controller.
`define NATIVE cdc.bridge.inst.gen_clock_conv.gen_async_lite_conv
`define CHANNEL_STATE(n) {`NATIVE.n.src_state,`NATIVE.n.dest_state,`NATIVE.n.src_send,`NATIVE.n.src_rcv,`NATIVE.n.dest_req,`NATIVE.n.dest_ack}
    wire [39:0] native_state={`CHANNEL_STATE(clock_conv_lite_fwd_aw),`CHANNEL_STATE(clock_conv_lite_fwd_w),`CHANNEL_STATE(clock_conv_lite_resp_b),`CHANNEL_STATE(clock_conv_lite_fwd_ar),`CHANNEL_STATE(clock_conv_lite_resp_r)};
`undef CHANNEL_STATE
`undef NATIVE
    integer scenario=MODE,passes=0;
    reg fault_test_active=0,record_revocation=0;
        time first_fault_ps=0,revoked_ps=0;
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
        while(hbusy || dbusy || ubusy)begin mtick(1);n=n+1;check(n<2000,"mailboxes drain before next case");end
        check(!pair_ack,"request withdrawal");end
    endtask
    // Allow56us per offer: an excluded offer waits for the entire32-entry
    // slow-domain prefix, local reset and fresh proof, not only one transfer.
    task take_aw(input [31:0] address);
        integer n;begin @(negedge host_clk);u_awaddr=address;u_awvalid=1;n=0;
        @(posedge host_clk);while(!u_awready)begin @(posedge host_clk);n=n+1;check(n<8000,"AW timeout");end
        @(negedge host_clk);u_awvalid=0;end
    endtask
    task take_w(input [31:0] data,input [3:0] strb);
        integer n;begin @(negedge host_clk);u_wdata=data;u_wstrb=strb;u_wvalid=1;n=0;
        @(posedge host_clk);while(!u_wready)begin @(posedge host_clk);n=n+1;check(n<8000,"W timeout");end
        @(negedge host_clk);u_wvalid=0;end
    endtask
    task take_b(input [1:0] expected);
        integer n;begin @(negedge host_clk);u_bready=1;n=0;
        @(posedge host_clk);while(!u_bvalid)begin @(posedge host_clk);n=n+1;check(n<8000,"B timeout");end
        check(u_bresp==expected,"write response preserved");@(negedge host_clk);u_bready=0;end
    endtask
    task take_ar(input [31:0] address,input [1:0] response);
        integer n;begin read_code=response;@(negedge host_clk);u_araddr=address;u_arvalid=1;n=0;
        @(posedge host_clk);while(!u_arready)begin @(posedge host_clk);n=n+1;check(n<8000,"AR timeout");end
        @(negedge host_clk);u_arvalid=0;end
    endtask
    task take_r(input [31:0] expected,input [1:0] response);
        integer n;begin @(negedge host_clk);u_rready=1;n=0;
        @(posedge host_clk);while(!u_rvalid)begin @(posedge host_clk);n=n+1;check(n<8000,"R timeout");end
        check((response==2 || u_rdata==expected) && u_rresp==response,"read data and response preserved");@(negedge host_clk);u_rready=0;end
    endtask
    task start_hold;
        begin @(negedge mgmt_clk);held_h=0;held_d=0;phase_tag=phase_tag+1;
        phase_kind=2;phase_valid=1;hold_request=1;mtick(8);
        check(hold_reset && !hn && !dn,"actuator asserts both rails");
        $display("AUTO_RESET case=%0d event=START ps=%0.0f tag=%h",MODE,$realtime*1000,phase_tag);end
    endtask
    task release_reset;
        integer n;begin n=0;while(!phase_ack)begin mtick(1);n=n+1;check(n<2000,"held phase ACK");end
        check(ack_kind==2 && ack_tag==phase_tag && ack_gen==64'h100000001 && held_h>=64 && held_d>=64,"held identity and actual edges");
        $display("AUTO_RESET case=%0d event=HELD ps=%0.0f tag=%h he=%0d de=%0d",MODE,$realtime*1000,phase_tag,held_h,held_d);
        @(negedge mgmt_clk);phase_tag=phase_tag+1;phase_kind=5;hold_request=0;#0.001;check(!phase_ack,"no old held response on ready phase");
        n=0;while(!phase_ack)begin mtick(1);n=n+1;check(n<2000,"ready phase ACK");end
        check(ack_kind==5 && ack_tag==phase_tag && ack_gen==64'h100000001 && hn && dn && !hold_reset,"ready identity and release");
        $display("AUTO_RESET case=%0d event=READY ps=%0.0f tag=%h",MODE,$realtime*1000,phase_tag);
        phase_valid=0;dtick(12);htick(12);end
    endtask
    task pass_case(input string name);passes=passes+1;$display("SHUT_CASE_PASS case=%0d name=%s ps=%0.0f",scenario,name,$realtime*1000);endtask
    always @(posedge host_clk)if(manager_reset_n)begin
        if(u_awvalid && u_awready)begin uaw=uaw+1;$display("FENCE_BUS case=%0d close=%b side=U channel=AW ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,u_awaddr);end
        if(u_wvalid && u_wready)begin uwc=uwc+1;$display("FENCE_BUS case=%0d close=%b side=U channel=W ps=%0.0f data=%h strb=%h",scenario,shutdown_request,$realtime*1000,u_wdata,u_wstrb);end
        if(u_bvalid && u_bready)begin ubc=ubc+1;$display("FENCE_BUS case=%0d close=%b side=U channel=B ps=%0.0f resp=%h",scenario,shutdown_request,$realtime*1000,u_bresp);end
        if(u_arvalid && u_arready)begin uar=uar+1;$display("FENCE_BUS case=%0d close=%b side=U channel=AR ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,u_araddr);end
        if(u_rvalid && u_rready)begin urc=urc+1;$display("FENCE_BUS case=%0d close=%b side=U channel=R ps=%0.0f data=%h resp=%h",scenario,shutdown_request,$realtime*1000,u_rdata,u_rresp);end
    end
    integer haw=0,hwc=0,hbc=0,har=0,hrc=0;
    always @(posedge host_clk)if(hn)begin
        if(h_awvalid && h_awready)begin haw=haw+1;$display("FENCE_BUS case=%0d close=%b side=H channel=AW ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,h_awaddr);end
        if(h_wvalid && h_wready)begin hwc=hwc+1;$display("FENCE_BUS case=%0d close=%b side=H channel=W ps=%0.0f data=%h strb=%h",scenario,shutdown_request,$realtime*1000,h_wdata,h_wstrb);end
        if(h_bvalid && h_bready)begin hbc=hbc+1;$display("FENCE_BUS case=%0d close=%b side=H channel=B ps=%0.0f resp=%h",scenario,shutdown_request,$realtime*1000,h_bresp);end
        if(h_arvalid && h_arready)begin har=har+1;$display("FENCE_BUS case=%0d close=%b side=H channel=AR ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,h_araddr);end
        if(h_rvalid && h_rready)begin hrc=hrc+1;$display("FENCE_BUS case=%0d close=%b side=H channel=R ps=%0.0f data=%h resp=%h",scenario,shutdown_request,$realtime*1000,h_rdata,h_rresp);end
    end
    always @(posedge device_clk)if(dn)begin
        if(d_awvalid && d_awready)begin daw=daw+1;$display("FENCE_BUS case=%0d close=%b side=D channel=AW ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,d_awaddr);end
        if(d_wvalid && d_wready)begin dwc=dwc+1;$display("FENCE_BUS case=%0d close=%b side=D channel=W ps=%0.0f data=%h strb=%h",scenario,shutdown_request,$realtime*1000,d_wdata,d_wstrb);end
        if(d_bvalid && d_bready)begin dbc=dbc+1;$display("FENCE_BUS case=%0d close=%b side=D channel=B ps=%0.0f resp=%h",scenario,shutdown_request,$realtime*1000,d_bresp);end
        if(d_arvalid && d_arready)begin dar=dar+1;$display("FENCE_BUS case=%0d close=%b side=D channel=AR ps=%0.0f addr=%h",scenario,shutdown_request,$realtime*1000,d_araddr);end
        if(d_rvalid && d_rready)begin drc=drc+1;$display("FENCE_BUS case=%0d close=%b side=D channel=R ps=%0.0f data=%h resp=%h",scenario,shutdown_request,$realtime*1000,d_rdata,d_rresp);end
    end
    always @(posedge mgmt_clk)if(cold && !fault_test_active)check(!hfault && !dfault && !ufault && !gate_fault,"no unexpected/ghost response or inventory overflow");

    task enter_shutdown;
        begin ask();end
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


    localparam integer OFFERS=128;
    integer base_aw,base_w,base_b,base_ar,base_r;
    integer max_aw=0,max_w=0,max_ar=0;
    bit aw_done=1,w_done=1,ar_done=1;
    bit operational=0;
    integer gaw=0,gwc=0,gbc=0,gar=0,grc=0;
    function [31:0] pattern(input integer n);
        pattern=32'hac000000 ^ (MODE<<16) ^ (n*32'h01030507);
    endfunction
    task send_aw_batch;
        begin aw_done=0;for(integer n=0;n<OFFERS;n=n+1)take_aw(32'h1000+MODE*1024+n*4);aw_done=1;end
    endtask
    task send_w_batch;
        begin w_done=0;for(integer n=0;n<OFFERS;n=n+1)take_w(pattern(n),(n%15)+1);w_done=1;end
    endtask
    task send_ar_batch;
        begin ar_done=0;for(integer n=0;n<OFFERS;n=n+1)take_ar(n*4,0);ar_done=1;end
    endtask
    task snapshot(input string name);
        $display("FENCE_EVENT case=%0d name=%s ps=%0.0f state=%0d ga=%0d gw=%0d gq=%0d al=%0d wl=%0d rl=%b fenced=%b gate_ack=%b up_cert=%b pair=%b ack=%b local_idle=%b u_idle=%b close=%b shut=%b hn=%b dn=%b bvalid=%b rvalid=%b",MODE,name,$realtime*1000,gate_state,ga,gw,gq,aw_left,w_left,ar_left,producer_fenced,gate_ack,upstream_certificate,section_pair,pair_ack,local_idle,upstream_idle,shutdown_request,shutdown_active,hn,dn,u_bvalid,u_rvalid);
    endtask
    task plateau;
        integer quiet,n,last,now;
        begin quiet=0;n=0;last=-1;
            while(quiet<256)begin
                htick(1);now=uaw+uwc+uar;
                if(now==last)quiet=quiet+1;else quiet=0;
                last=now;n=n+1;check(n<16000,"bounded plateau");
            end
            if(MODE!=1)check(u_awvalid && !u_awready && uaw-base_aw==32,"32 AW credit cap");
            if(MODE==1)check(u_wvalid && !u_wready && uwc-base_w>0 && uwc-base_w<32,"W-only downstream saturation");
            if(MODE==2)check(u_wvalid && !u_wready && u_arvalid && !u_arready && uwc-base_w==32 && uar-base_ar==32,"32 W/AR caps");
            check(ubc==base_b && urc==base_r,"responses blocked");snapshot("PLATEAU");
        end
    endtask
    task await_gate;
        integer n;begin n=0;while(gate_state!=2)begin htick(1);n=n+1;check(n<16000,"finite prefix must drain");end end
    endtask
    task await_section;
        integer n;begin n=0;while(!pair_ack)begin mtick(1);n=n+1;check(n<2000,"fresh complete section certificate");end
            check(gate_state==2 && producer_fenced && {ga,gw,gq}==0 && {ha,hw,hq,da,dw,dq}==0,"certificate requires real stopped producer and zero inventory");
        end
    endtask
    // No spontaneous ledger reset; continuous independent G trace and bounds.
    always @(posedge host_clk)if(cold)begin
        if(g_awvalid && g_awready)begin gaw=gaw+1;$display("FENCE_BUS case=%0d close=%b side=G channel=AW ps=%0.0f addr=%h",MODE,shutdown_request,$realtime*1000,g_awaddr);end
        if(g_wvalid && g_wready)begin gwc=gwc+1;$display("FENCE_BUS case=%0d close=%b side=G channel=W ps=%0.0f data=%h strb=%h",MODE,shutdown_request,$realtime*1000,g_wdata,g_wstrb);end
        if(g_bvalid && g_bready)begin gbc=gbc+1;$display("FENCE_BUS case=%0d close=%b side=G channel=B ps=%0.0f resp=%h",MODE,shutdown_request,$realtime*1000,g_bresp);end
        if(g_arvalid && g_arready)begin gar=gar+1;$display("FENCE_BUS case=%0d close=%b side=G channel=AR ps=%0.0f addr=%h",MODE,shutdown_request,$realtime*1000,g_araddr);end
        if(g_rvalid && g_rready)begin grc=grc+1;$display("FENCE_BUS case=%0d close=%b side=G channel=R ps=%0.0f data=%h resp=%h",MODE,shutdown_request,$realtime*1000,g_rdata,g_rresp);end
        if(operational)check(manager_reset_n,"manager retained throughout local reset");
        #0.001;check(ga<=32 && gw<=32 && gq<=32,"hard pending limits");
        if(ga>max_aw)max_aw=ga;if(gw>max_w)max_w=gw;if(gq>max_ar)max_ar=gq;
        if(pair_ack)check(producer_fenced && gate_state==2 && shutdown_request,"certificate cannot reopen producer");
    end
    reg uas=0,uws=0,urs=0,gas=0,gws=0,grs=0;
    reg [31:0] uap=0,urp=0,gap=0,grp=0;
    reg [35:0] uwp=0,gwp=0;
    always @(posedge host_clk)if(manager_reset_n)begin
        if(uas)check(u_awvalid && u_awaddr==uap,"U AW held");
        if(uws)check(u_wvalid && {u_wdata,u_wstrb}==uwp,"U W held");
        if(urs)check(u_arvalid && u_araddr==urp,"U AR held");
        if(gas)check(g_awvalid && g_awaddr==gap,"G AW cannot be cut on STOP");
        if(gws)check(g_wvalid && {g_wdata,g_wstrb}==gwp,"G W cannot be cut on STOP");
        if(grs)check(g_arvalid && g_araddr==grp,"G AR cannot be cut on STOP");
        uas=u_awvalid && !u_awready;uws=u_wvalid && !u_wready;urs=u_arvalid && !u_arready;
        gas=g_awvalid && !g_awready;gws=g_wvalid && !g_wready;grs=g_arvalid && !g_arready;
        uap=u_awaddr;uwp={u_wdata,u_wstrb};urp=u_araddr;
        gap=g_awaddr;gwp={g_wdata,g_wstrb};grp=g_araddr;
    end

    always @(posedge mgmt_clk)begin
        #0.001;
        if(fault_test_active && fault_mgmt!=0)
            check(!pair_ack && !upstream_certificate && !section_pair && hn && dn,
                  "retained management fault masks every exposed certificate without reset");
    end
    always @(gate_fault or hfault or dfault)begin
        if(fault_test_active && (gate_fault || hfault || dfault) && first_fault_ps==0)
            first_fault_ps=$realtime*1000;
    end
    always @(negedge pair_ack)begin
        if(record_revocation && first_fault_ps!=0 && revoked_ps==0)revoked_ps=$realtime*1000;
    end
    task fault_snapshot(input string name);
        $display("FAULT_EVENT case=%0d name=%s ps=%0.0f ack=%b gate_fault=%b h_fault=%b d_fault=%b fault_mgmt=%0d first_ps=%0d revoked_ps=%0d host_run=%b device_run=%b closed=%b hn=%b dn=%b gate_sources=%0d",MODE,name,$realtime*1000,pair_ack,gate_fault,hfault,dfault,fault_mgmt,first_fault_ps,revoked_ps,host_run,device_run,shutdown_request,hn,dn,gate_fault_sources);
    endtask
    // H/D inject an orphan response handshake at the observed boundary.
    // Both VALID and READY are testbench-forced because a correctly idle IP
    // need not accept an unsolicited response. Ledger fault logic is real RTL.
    task probe_fault;
        integer n;
        reg [2:0] remembered;
        begin
            ask();await_section();fault_snapshot("BASELINE_CERTIFIED");
            fault_test_active=1;record_revocation=1;
            if(MODE==0)begin
                @(negedge host_clk);force g_bvalid=1'b1;
                @(posedge host_clk);@(negedge host_clk);release g_bvalid;
                check(gate_fault,"admission detects unsolicited response after closure");if(STOP_SOURCE)host_run=0;
            end else if(MODE==1)begin
                @(negedge host_clk);force h_bvalid=1'b1;force h_bready=1'b1;n=0;
                @(posedge host_clk);while(!h_bready)begin @(posedge host_clk);n=n+1;check(n<2000,"injected H response handshake");end
                @(negedge host_clk);release h_bvalid;release h_bready;check(hfault,"H ledger detects response underflow");if(STOP_SOURCE)host_run=0;
            end else begin
                @(negedge device_clk);force d_bvalid=1'b1;force d_bready=1'b1;n=0;
                @(posedge device_clk);while(!d_bready)begin @(posedge device_clk);n=n+1;check(n<2000,"injected D response handshake");end
                @(negedge device_clk);release d_bvalid;release d_bready;check(dfault,"D ledger detects response underflow");if(STOP_SOURCE)device_run=0;
            end
            fault_snapshot("SOURCE_FAULT_OBSERVED");mtick(16);
            if(EXPECT_REVOKE)begin
                check(!pair_ack && fault_mgmt!=0,"fault must revoke cached proof in management");
                check(first_fault_ps>0 && revoked_ps>=first_fault_ps && revoked_ps-first_fault_ps<=30000,"revocation within three management periods");
            end else check(pair_ack && first_fault_ps>0 && revoked_ps==0,"original controller retains stale certificate");
            fault_snapshot("AFTER_PROPAGATION");record_revocation=0;
            @(negedge mgmt_clk);request=0;mtick(2);generation=generation+1;tag=tag+1;request=1;mtick(32);
            check(!pair_ack,"fault cannot authorize fresh identity");fault_snapshot("FRESH_ID_BLOCKED");
            host_run=1;device_run=1;htick(32);dtick(32);
            check(!pair_ack && hn && dn && shutdown_request,"fault does not force resets or reopen");
            if(EXPECT_REVOKE)begin
                remembered=fault_mgmt;
                force gate_fault=1'b0;force hfault=1'b0;force dfault=1'b0;
                force section.certificate.admission_fault_sources=2'b00;
                mtick(32);check(fault_mgmt==remembered && fault_mgmt!=0 && !pair_ack,"management fault remains after reported source bits clear");
                fault_snapshot("RETAINED_AFTER_SOURCE_CLEAR");
                release section.certificate.admission_fault_sources;release gate_fault;release hfault;release dfault;mtick(2);
            end
            fault_snapshot("FAULT_TERMINAL");
            $display("FAULT_CASE_PASS case=%0d expect_revoke=%0d first_ps=%0d revoked_ps=%0d fault_mgmt=%0d",MODE,EXPECT_REVOKE,first_fault_ps,revoked_ps,fault_mgmt);
        end
    endtask

    reg guard_b_seen=0,guard_r_seen=0;
    always @(posedge host_clk)if(cold)begin
        if(c_bvalid && !guard_b_seen)begin
            $display("GUARD_RAW case=%0d channel=B ps=%0.0f ready=%b protected_valid=%b protected_ready=%b resp=%0d aw=%0d w=%0d",MODE,$realtime*1000,c_bready,h_bvalid,h_bready,c_bresp,guard_aw_pending,guard_w_pending);
        end
        if(c_rvalid && !guard_r_seen)begin
            $display("GUARD_RAW case=%0d channel=R ps=%0.0f ready=%b protected_valid=%b protected_ready=%b data=%h resp=%0d ar=%0d",MODE,$realtime*1000,c_rready,h_rvalid,h_rready,c_rdata,c_rresp,guard_ar_pending);
        end
        guard_b_seen=c_bvalid;guard_r_seen=c_rvalid;
        if(guard_fault_sources[0])check(c_bvalid && !c_bready && !h_bvalid,
            "orphan retained below guard without manager acceptance");
    end
    bit life_active=0,hold_send_guard=0;
    time report_ps=0,close_ps=0;
    reg [63:0] old_generation=0,old_tag=0;
    wire source_send=section.certificate.upstream_mailbox.send;
    wire source_received=section.certificate.upstream_mailbox.received_ack;
    wire [63:0] source_generation=section.certificate.upstream_mailbox.payload[130:67];
    wire [63:0] source_tag=section.certificate.upstream_mailbox.payload[66:3];
    wire [31:0] obsolete=section.certificate.upstream_mailbox.obsolete_completions;
    always @(fault_mgmt)if(life_active && fault_mgmt!=0 && report_ps==0)report_ps=$realtime*1000;
    always @(posedge shutdown_request)if(fault_test_active && close_ps==0)close_ps=$realtime*1000;
    always @(posedge mgmt_clk)if(life_active)begin
        check(hn && dn && manager_reset_n && !hold_reset,"no operational reset during lifetime fault");
        if(hold_send_guard)check(source_send && ubusy && !source_received,
            "stopped peer cannot truncate the outstanding XPM send");
    end
    always @(negedge source_send)if(life_active && MODE==1)
        check(source_received,"source send falls only after XPM response");
    task life_snapshot(input string name);
        $display("LIFE_EVENT case=%0d name=%s ps=%0.0f req=%b generation=%0d tag=%0d ack=%b close=%b state=%0d source_tag=%0d source_generation=%0d send=%b received=%b busy=%b obsolete=%0d host_run=%b device_run=%b gate_sources=%0d gate_fault=%b h_fault=%b d_fault=%b fault_mgmt=%0d first_ps=%0d report_ps=%0d close_ps=%0d bvalid=%b bresp=%0d rvalid=%b rresp=%0d rdata=%h ua=%0d uw=%0d uq=%0d hn=%b dn=%b guard_sources=%0d raw_bvalid=%b raw_bready=%b protected_bvalid=%b",MODE,name,$realtime*1000,request,generation,tag,pair_ack,shutdown_request,gate_state,source_tag,source_generation,source_send,source_received,ubusy,obsolete,host_run,device_run,gate_fault_sources,gate_fault,hfault,dfault,fault_mgmt,first_fault_ps,report_ps,close_ps,u_bvalid,u_bvalid?u_bresp:0,u_rvalid,u_rvalid?u_rresp:0,u_rvalid?u_rdata:32'b0,ua,uw,uq,hn,dn,guard_fault_sources,c_bvalid,c_bready,h_bvalid);
    endtask
    initial begin
        $display("LIFE_PROFILE case=%0d device_period_ps=%0d host_period_ps=7000 mgmt_period_ps=10000 stop_source=%0d",MODE,DEVICE_PERIOD_PS,STOP_SOURCE);
        #300;actuator_cold=1;mtick(3);start_hold();release_reset();cold=1;htick(64);operational=1;
        take_aw(32'h7000+MODE*4);take_w(32'ha1b2c3d4+MODE,5);take_b(MODE);
        begin reg [31:0] original,expected;
            original=32'h13579bdf ^ ((32'h1c00+MODE)*32'h01020408);
            expected=(original & 32'hff00ff00) | ((32'ha1b2c3d4+MODE)&32'h00ff00ff);
            take_ar(32'h7000+MODE*4,0);take_r(expected,MODE);htick(64);dtick(64);
        end
        check(upstream_idle && !pair_ack && !shutdown_request && gate_fault_sources==0,
            "clean open section before lifetime scenario");
        life_active=1;life_snapshot("START");mtick(1);
        if(MODE==1)begin
            @(negedge host_clk);host_run=0;ask();old_generation=generation;old_tag=tag;mtick(6);
            check(ubusy && source_send && !proof_request && !pair_ack && !shutdown_request,
                "real proof mailbox outstanding before fault with stopped peer");
            hold_send_guard=1;
        end else if(MODE==2)begin
            ask();await_section();old_generation=generation;old_tag=tag;
            take_aw(32'h7800);take_w(32'hdeadbeef,9);await_bvalid();
            take_ar(32'h7800,2);await_rvalid();
            check(u_bresp==2 && u_rresp==2 && u_rdata==0 && !local_idle && !upstream_idle && pair_ack,
                "certified section excludes held local SLVERR responses");
        end
        life_snapshot("ARMED");fault_test_active=1;
        if(MODE==0)begin
            // An orphan read response is consumed at G and U. Deterministic
            // injected payload; actual retained admission ledger detects it.
            @(negedge host_clk);u_rready=1;force g_rvalid=1'b1;
            force g_rresp=2'b10;force g_rdata=32'hbad00bad;
            @(posedge host_clk);@(negedge host_clk);
            release g_rvalid;release g_rresp;release g_rdata;u_rready=0;
            check(gate_fault_sources==1,"pre-proof admission ledger fault source");
            if(STOP_SOURCE)host_run=0;
        end else if(MODE==2)begin
            @(negedge host_clk);force g_bvalid=1'b1;
            @(posedge host_clk);@(negedge host_clk);release g_bvalid;
            check(gate_fault_sources==2,"late response admission fault source");
            if(STOP_SOURCE)host_run=0;
        end else begin
            // Orphan B handshake at the real D observer; forcing both sides
            // deliberately violates protocol and reaches retained fault RTL.
            @(negedge device_clk);force d_bvalid=1'b1;force d_bready=1'b1;
            @(posedge device_clk);@(negedge device_clk);release d_bvalid;release d_bready;
            check(dfault,"device ledger fault before complete proof");
            if(STOP_SOURCE)device_run=0;
        end
        life_snapshot("SOURCE_FAULT");mtick(16);
        check(!pair_ack && fault_mgmt!=0 && report_ps>=first_fault_ps &&
              report_ps-first_fault_ps<=30000,"management report within three periods");
        if(MODE==1)check(!shutdown_request && source_send && ubusy,
            "stopped host waits for retained closure without cancelling send");
        else check(shutdown_request,"fault closes or retains closure");
        if(MODE==3)check(!proof_request && !request && close_ps>=first_fault_ps &&
            close_ps-first_fault_ps<=65000,"device fault alone closes host within bounded running-clock path");
        life_snapshot("PROPAGATED");
        @(negedge mgmt_clk);request=0;mtick(2);generation=generation+1;tag=tag+1;request=1;mtick(32);
        check(!pair_ack,"lifetime fault blocks fresh identity");life_snapshot("FRESH_BLOCKED");
        hold_send_guard=0;host_run=1;device_run=1;mtick(40);htick(32);dtick(32);
        check(shutdown_request && !pair_ack && hn && dn,"retained closure after clock return");
        if(MODE==1)check(source_tag==old_tag && source_generation==old_generation &&
            ((ubusy && source_send && obsolete==0) || (!ubusy && !source_send && obsolete==1)),
            "old mailbox may wait or complete obsolete, never relabel to fresh identity");
        if(MODE==2)begin
            check(u_bvalid && u_bresp==2 && u_rvalid && u_rresp==2 && u_rdata==0,
                "held local responses unchanged through fault and clock stop");
            take_b(2);take_r(0,2);htick(32);
            check(upstream_idle && local_idle,"local errors retire only on real upstream handshakes");
        end
        force section.certificate.admission_fault_sources=2'b00;force hfault=1'b0;force dfault=1'b0;
        force section.certificate.guard_fault_sources=3'b000;
        mtick(32);check(fault_mgmt==(MODE==0 || MODE==2 ? 1:12) && !pair_ack,
            "retained management fault after every source report removed");
        life_snapshot("SOURCE_REPORTS_CLEARED");
        release section.certificate.admission_fault_sources;release hfault;release dfault;
        release section.certificate.guard_fault_sources;mtick(2);
        check(shutdown_request && hn && dn && !pair_ack,"terminal lifetime fault remains closed without reset");
        life_snapshot("TERMINAL");
        $display("LIFE_CASE_PASS case=%0d first_ps=%0d report_ps=%0d close_ps=%0d fault_mgmt=%0d uaw=%0d uar=%0d ubc=%0d urc=%0d gaw=%0d gar=%0d gbc=%0d grc=%0d daw=%0d dar=%0d dbc=%0d drc=%0d",MODE,first_fault_ps,report_ps,close_ps,fault_mgmt,uaw,uar,ubc,urc,gaw,gar,gbc,grc,daw,dar,dbc,drc);
        done=1;
    end
    initial begin #1000000;if(!done)$fatal(1,"FENCE_TIMEOUT case=%0d",MODE);end
endmodule

module tb_ntrpu_axil_section_lifetime_r2 #(parameter integer DEVICE_PERIOD_PS=5000,EXPECT_REVOKE=1,STOP_SOURCE=1);
    wire [3:0] done;
    for(genvar mode=0;mode<4;mode=mode+1)begin:cases
        ntrpu_axil_section_lifetime_r2_case #(.DEVICE_PERIOD_PS(DEVICE_PERIOD_PS),.MODE(mode),.EXPECT_REVOKE(EXPECT_REVOKE),.STOP_SOURCE(STOP_SOURCE)) bench(.done(done[mode]));
    end
    initial begin wait(&done);#100;
        $display("LIFE_SUITE_PASS cases=4 device_period_ps=%0d stop_source=%0d native_endpoint=0 complete_dma_drain=0",DEVICE_PERIOD_PS,STOP_SOURCE);$finish;
    end
endmodule
