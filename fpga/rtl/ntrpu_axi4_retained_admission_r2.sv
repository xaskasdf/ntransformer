`timescale 1ns / 1ps
// Retained active AXI4 boundary, 512-bit data / 64-bit addresses / four-bit ID.
// Permanent RUN -> FINISH -> REJECT until FPGA reconfiguration. No operational
// reset port. A fresh drain proof covers the downstream section, NOT local
// error replies or protected XDMA queues; inspect local_idle separately.
module ntrpu_axi4_retained_admission_r2 #(
    parameter integer WRITE_DEPTH=32, READ_DEPTH=8
) (
    input wire clk,cold_rst_n,close_request,proof_request,
    input wire [63:0] proof_tag,
    input wire s_awvalid,
    output wire s_awready,
    input wire [63:0] s_awaddr,
    input wire [3:0] s_awid,
    input wire [7:0] s_awlen,
    input wire [2:0] s_awsize,
    input wire [2:0] s_awprot,
    input wire [1:0] s_awburst,
    input wire [3:0] s_awcache,
    input wire s_awlock,
    input wire s_wvalid,
    output wire s_wready,
    input wire [511:0] s_wdata,
    input wire [63:0] s_wstrb,
    input wire s_wlast,
    output wire s_bvalid,
    input wire s_bready,
    output wire [3:0] s_bid,
    output wire [1:0] s_bresp,
    input wire s_arvalid,
    output wire s_arready,
    input wire [63:0] s_araddr,
    input wire [3:0] s_arid,
    input wire [7:0] s_arlen,
    input wire [2:0] s_arsize,
    input wire [2:0] s_arprot,
    input wire [1:0] s_arburst,
    input wire [3:0] s_arcache,
    input wire s_arlock,
    output wire s_rvalid,
    input wire s_rready,
    output wire [511:0] s_rdata,
    output wire [3:0] s_rid,
    output wire [1:0] s_rresp,
    output wire s_rlast,
    input wire [3:0] s_awregion,
    input wire [3:0] s_awqos,
    input wire [3:0] s_arregion,
    input wire [3:0] s_arqos,
    output wire m_awvalid,
    input wire m_awready,
    output wire [63:0] m_awaddr,
    output wire [3:0] m_awid,
    output wire [7:0] m_awlen,
    output wire [2:0] m_awsize,
    output wire [2:0] m_awprot,
    output wire [1:0] m_awburst,
    output wire [3:0] m_awcache,
    output wire m_awlock,
    output wire m_wvalid,
    input wire m_wready,
    output wire [511:0] m_wdata,
    output wire [63:0] m_wstrb,
    output wire m_wlast,
    input wire m_bvalid,
    output wire m_bready,
    input wire [3:0] m_bid,
    input wire [1:0] m_bresp,
    output wire m_arvalid,
    input wire m_arready,
    output wire [63:0] m_araddr,
    output wire [3:0] m_arid,
    output wire [7:0] m_arlen,
    output wire [2:0] m_arsize,
    output wire [2:0] m_arprot,
    output wire [1:0] m_arburst,
    output wire [3:0] m_arcache,
    output wire m_arlock,
    input wire m_rvalid,
    output wire m_rready,
    input wire [511:0] m_rdata,
    input wire [3:0] m_rid,
    input wire [1:0] m_rresp,
    input wire m_rlast,
    output wire [3:0] m_awregion,
    output wire [3:0] m_awqos,
    output wire [3:0] m_arregion,
    output wire [3:0] m_arqos,
    output wire source_fenced,drain_ack,local_idle,protocol_fault,
    output wire [63:0] drain_tag,
    output wire [6:0] fault_sources,
    output wire [$clog2(WRITE_DEPTH+1)-1:0] aw_pending,w_pending,
    output wire [$clog2(16*READ_DEPTH+1)-1:0] ar_pending,
    output wire [8:0] w_partial,
    output reg [$clog2(WRITE_DEPTH+2)-1:0] aw_left=0,w_left=0,
    output reg ar_left=0,
    output wire [1:0] state_debug
);
    localparam [1:0] RUN=0,FINISH=1,REJECT=2;
    localparam integer PC=$clog2(WRITE_DEPTH+2);
    reg [1:0] state=RUN;
    reg request_fault=0,local_write_fault=0,state_fault=0;
    wire [3:0] guard_faults;
    assign fault_sources={state_fault,local_write_fault,request_fault,guard_faults};
    assign protocol_fault=|fault_sources;
    assign state_debug=state;
    wire forwarding=state==RUN || state==FINISH;
    wire aw_capacity,w_capacity,ar_capacity,guard_idle;
    // Preserve upstream offers while blocked, including the first local request
    // waiting behind the finite forwarded prefix. A source protocol violation
    // is retained as a fault; it cannot alter an already presented request.
    reg keep_aw=0,keep_w=0,keep_ar=0;
    reg [96:0] saved_aw=0,saved_ar=0;
    reg [576:0] saved_w=0;
    wire [96:0] raw_aw,raw_ar;
    wire [576:0] raw_w={s_wdata,s_wstrb,s_wlast};
    wire up_awvalid=s_awvalid || keep_aw,up_wvalid=s_wvalid || keep_w,up_arvalid=s_arvalid || keep_ar;
    wire [96:0] selected_aw=keep_aw?saved_aw:raw_aw,selected_ar=keep_ar?saved_ar:raw_ar;
    wire [576:0] selected_w=keep_w?saved_w:raw_w;
    wire request_bad=(keep_aw && (!s_awvalid || raw_aw!=saved_aw)) ||
        (keep_w && (!s_wvalid || raw_w!=saved_w)) ||
        (keep_ar && (!s_arvalid || raw_ar!=saved_ar));
    // Once a downstream VALID was offered, a later fault/capacity change must
    // not withdraw it. The preserved upstream payload backs these grants.
    reg held_aw=0,held_w=0,held_ar=0;
    wire permit_aw=held_aw || (aw_capacity && (state==RUN || aw_left!=0));
    wire permit_w=held_w || (w_capacity && (state==RUN || w_left!=0));
    wire permit_ar=held_ar || (ar_capacity && (state==RUN || ar_left));
    assign m_awvalid=cold_rst_n && forwarding && up_awvalid && permit_aw;
    assign m_wvalid=cold_rst_n && forwarding && up_wvalid && permit_w;
    assign m_arvalid=cold_rst_n && forwarding && up_arvalid && permit_ar;
    wire af=m_awvalid && m_awready,wf=m_wvalid && m_wready,arf=m_arvalid && m_arready;
    wire wc=wf && m_wlast;
    wire [PC:0] offered_aw={1'b0,aw_pending}+up_awvalid;
    wire [PC:0] offered_w={1'b0,w_pending}+(w_partial!=0 || up_wvalid);
    wire [PC:0] prefix_writes=offered_aw>offered_w?offered_aw:offered_w;
    assign source_fenced=cold_rst_n && state!=RUN && aw_left==0 && w_left==0 && !ar_left &&
        !(held_aw || held_w || held_ar || m_awvalid || m_wvalid || m_arvalid);
    wire g_bvalid,g_rvalid,g_rlast;
    wire [3:0] g_bid,g_rid;
    wire [1:0] g_bresp,g_rresp;
    wire [511:0] g_rdata;
    reg local_aw=0,local_wdone=0,local_b=0,local_r=0;
    reg [3:0] local_bid=0,local_rid=0;
    reg [8:0] local_awbeats=0,local_wbeats=0,local_rleft=0;
    assign s_awready=cold_rst_n && (state==REJECT ? (!local_aw && !local_b && !local_write_fault) : forwarding && permit_aw && m_awready);
    assign s_wready=cold_rst_n && (state==REJECT ? (!local_wdone && !local_b && !local_write_fault) : forwarding && permit_w && m_wready);
    assign s_arready=cold_rst_n && (state==REJECT ? !local_r : forwarding && permit_ar && m_arready);
    assign s_bvalid=cold_rst_n && (state==REJECT?local_b:g_bvalid);
    assign s_rvalid=cold_rst_n && (state==REJECT?local_r:g_rvalid);
    assign s_bid=state==REJECT?local_bid:g_bid;
    assign s_bresp=state==REJECT?2'b10:g_bresp;
    assign s_rid=state==REJECT?local_rid:g_rid;
    assign s_rresp=state==REJECT?2'b10:g_rresp;
    assign s_rdata=state==REJECT?512'b0:g_rdata;
    assign s_rlast=state==REJECT?(local_rleft==1):g_rlast;
    assign local_idle=cold_rst_n && state==REJECT && !protocol_fault &&
        !(local_aw || local_wdone || local_wbeats!=0 || local_b || local_r || up_awvalid || up_wvalid || up_arvalid);
    ntrpu_axi4_response_guard_r2 #(.WRITE_DEPTH(WRITE_DEPTH),.READ_DEPTH(READ_DEPTH)) guard(
        .clk(clk),.cold_rst_n(cold_rst_n),.awvalid(m_awvalid),.awready(m_awready),.awid(m_awid),.awlen(m_awlen),
        .wvalid(m_wvalid),.wready(m_wready),.wlast(m_wlast),.arvalid(m_arvalid),.arready(m_arready),.arid(m_arid),.arlen(m_arlen),
        .c_bvalid(m_bvalid),.c_bready(m_bready),.c_bid(m_bid),.c_bresp(m_bresp),
        .c_rvalid(m_rvalid),.c_rready(m_rready),.c_rid(m_rid),.c_rresp(m_rresp),.c_rdata(m_rdata),.c_rlast(m_rlast),
        .h_bvalid(g_bvalid),.h_bready(s_bready && forwarding),.h_bid(g_bid),.h_bresp(g_bresp),
        .h_rvalid(g_rvalid),.h_rready(s_rready && forwarding),.h_rid(g_rid),.h_rresp(g_rresp),.h_rdata(g_rdata),.h_rlast(g_rlast),
        .fault_sources(guard_faults),.aw_capacity(aw_capacity),.w_capacity(w_capacity),.ar_capacity(ar_capacity),
        .local_idle(guard_idle),.aw_pending(aw_pending),.w_pending(w_pending),.ar_pending(ar_pending),.w_partial(w_partial));
    reg captured=0;
    reg [63:0] captured_tag=0;
    reg [1:0] quiet=0;
    assign drain_tag=captured_tag;
    assign drain_ack=cold_rst_n && proof_request && captured && proof_tag==captured_tag &&
        state==REJECT && guard_idle && !protocol_fault && quiet==2;
    always @(posedge clk)begin
        if(!cold_rst_n)begin
            state<=RUN;aw_left<=0;w_left<=0;ar_left<=0;
            request_fault<=0;local_write_fault<=0;state_fault<=0;
            keep_aw<=0;keep_w<=0;keep_ar<=0;saved_aw<=0;saved_w<=0;saved_ar<=0;
            held_aw<=0;held_w<=0;held_ar<=0;
            local_aw<=0;local_wdone<=0;local_b<=0;local_r<=0;
            local_bid<=0;local_rid<=0;local_awbeats<=0;local_wbeats<=0;local_rleft<=0;
            captured<=0;captured_tag<=0;quiet<=0;
        end else begin
            request_fault<=request_fault || request_bad;
            keep_aw<=up_awvalid && !s_awready;
            keep_w<=up_wvalid && !s_wready;
            keep_ar<=up_arvalid && !s_arready;
            if(up_awvalid && !s_awready && !keep_aw)saved_aw<=raw_aw;
            if(up_wvalid && !s_wready && !keep_w)saved_w<=raw_w;
            if(up_arvalid && !s_arready && !keep_ar)saved_ar<=raw_ar;
            held_aw<=m_awvalid && !m_awready;
            held_w<=m_wvalid && !m_wready;
            held_ar<=m_arvalid && !m_arready;
            case(state)
                RUN:if(close_request || protocol_fault || request_bad)begin
                    aw_left<=prefix_writes-aw_pending-af;
                    w_left<=prefix_writes-w_pending-wc;
                    ar_left<=up_arvalid && !arf;
                    state<=FINISH;
                end
                FINISH:begin
                    if(af)aw_left<=aw_left-1'b1;
                    if(wc)w_left<=w_left-1'b1;
                    if(arf)ar_left<=0;
                    if(source_fenced && guard_idle && !protocol_fault)state<=REJECT;
                end
                REJECT:begin
                    if(up_awvalid && s_awready)begin local_aw<=1;local_bid<=m_awid;local_awbeats<={1'b0,m_awlen}+9'd1;end
                    if(up_wvalid && s_wready)begin
                        if(local_wbeats==255 && !m_wlast)local_write_fault<=1;
                        else begin local_wbeats<=local_wbeats+9'd1;if(m_wlast)local_wdone<=1;end
                    end
                    if(local_aw && local_wdone && !local_b)begin
                        if(local_awbeats==local_wbeats)local_b<=1;
                        else local_write_fault<=1;
                    end
                    if(local_b && s_bready)begin local_b<=0;local_aw<=0;local_wdone<=0;local_wbeats<=0;end
                    if(up_arvalid && s_arready)begin local_r<=1;local_rid<=m_arid;local_rleft<={1'b0,m_arlen}+9'd1;end
                    if(local_r && s_rready)begin
                        if(local_rleft==1)begin local_r<=0;local_rleft<=0;end
                        else local_rleft<=local_rleft-9'd1;
                    end
                end
                default:state_fault<=1;
            endcase
            if(!proof_request)begin captured<=0;quiet<=0;end
            else if(!captured || captured_tag!=proof_tag)begin captured<=1;captured_tag<=proof_tag;quiet<=0;end
            else if(state!=REJECT || !guard_idle || protocol_fault)quiet<=0;
            else if(quiet<2)quiet<=quiet+1'b1;
        end
    end
    assign raw_aw={s_awaddr,s_awid,s_awlen,s_awsize,s_awprot,s_awburst,s_awcache,s_awlock,s_awregion,s_awqos};
    assign {m_awaddr,m_awid,m_awlen,m_awsize,m_awprot,m_awburst,m_awcache,m_awlock,m_awregion,m_awqos}=selected_aw;
    assign raw_ar={s_araddr,s_arid,s_arlen,s_arsize,s_arprot,s_arburst,s_arcache,s_arlock,s_arregion,s_arqos};
    assign {m_araddr,m_arid,m_arlen,m_arsize,m_arprot,m_arburst,m_arcache,m_arlock,m_arregion,m_arqos}=selected_ar;
    assign {m_wdata,m_wstrb,m_wlast}=selected_w;
    // synthesis translate_off
    initial if(WRITE_DEPTH<2 || READ_DEPTH<2)$fatal(1,"AXI4_ADMISSION_PARAMETERS");
    // synthesis translate_on
endmodule
