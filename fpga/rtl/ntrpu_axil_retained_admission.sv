`timescale 1ns / 1ps
// Configuration-lifetime AXI-Lite boundary. Closing is permanent until FPGA
// reconfiguration. Finish the finite prefix already accepted/presented, then
// become a local SLVERR subordinate. Never gate an already offered VALID with
// close. Credits are per AW/W/AR, retired by B/R; write halves remain independent.
// The downstream producer is stopped by this RTL, not by an external idle ACK.
// This does NOT stop the upstream PCIe endpoint or certify its protected queues.
module ntrpu_axil_retained_admission #(
    parameter integer MAX_PENDING=32
) (
    input wire clk,cold_rst_n,close_request,proof_request,
    input wire [63:0] proof_tag,
    input wire [31:0] s_awaddr,s_wdata,s_araddr,
    input wire [2:0] s_awprot,s_arprot,
    input wire [3:0] s_wstrb,
    input wire s_awvalid,s_wvalid,s_bready,s_arvalid,s_rready,
    output wire s_awready,s_wready,s_bvalid,s_arready,s_rvalid,
    output wire [1:0] s_bresp,s_rresp,
    output wire [31:0] s_rdata,
    output wire [31:0] m_awaddr,m_wdata,m_araddr,
    output wire [2:0] m_awprot,m_arprot,
    output wire [3:0] m_wstrb,
    output wire m_awvalid,m_wvalid,m_bready,m_arvalid,m_rready,
    input wire m_awready,m_wready,m_bvalid,m_arready,m_rvalid,
    input wire [1:0] m_bresp,m_rresp,
    input wire [31:0] m_rdata,
    output wire source_fenced,drain_ack,local_idle,protocol_fault,
    output wire [63:0] drain_tag,
    output wire [7:0] aw_pending,w_pending,ar_pending,
    output reg [7:0] aw_left=0,w_left=0,
    output reg ar_left=0,
    output wire [1:0] state_debug
);
    localparam [1:0] RUN=0,FINISH=1,REJECT=2;
    reg [1:0] state=RUN;
    reg reject_aw=0,reject_w=0,reject_b=0,reject_r=0,late_response=0;
    wire fault,ledger_ack;
    wire forward=state==RUN || state==FINISH;
    wire permit_aw=cold_rst_n && forward && aw_pending<MAX_PENDING && (state==RUN || aw_left!=0);
    wire permit_w=cold_rst_n && forward && w_pending<MAX_PENDING && (state==RUN || w_left!=0);
    wire permit_ar=cold_rst_n && forward && ar_pending<MAX_PENDING && (state==RUN || ar_left);
    assign m_awaddr=s_awaddr;assign m_awprot=s_awprot;
    assign m_wdata=s_wdata;assign m_wstrb=s_wstrb;
    assign m_araddr=s_araddr;assign m_arprot=s_arprot;
    assign m_awvalid=s_awvalid && permit_aw;
    assign m_wvalid=s_wvalid && permit_w;
    assign m_arvalid=s_arvalid && permit_ar;
    assign m_bready=cold_rst_n && forward && s_bready;
    assign m_rready=cold_rst_n && forward && s_rready;
    assign s_awready=state==REJECT ? cold_rst_n && !reject_aw && !reject_b : permit_aw && m_awready;
    assign s_wready=state==REJECT ? cold_rst_n && !reject_w && !reject_b : permit_w && m_wready;
    assign s_arready=state==REJECT ? cold_rst_n && !reject_r : permit_ar && m_arready;
    assign s_bvalid=cold_rst_n && (state==REJECT ? reject_b : m_bvalid);
    assign s_rvalid=cold_rst_n && (state==REJECT ? reject_r : m_rvalid);
    assign s_bresp=state==REJECT ? 2'b10 : m_bresp;
    assign s_rresp=state==REJECT ? 2'b10 : m_rresp;
    assign s_rdata=state==REJECT ? 32'b0 : m_rdata;
    wire aw_fire=m_awvalid && m_awready,w_fire=m_wvalid && m_wready;
    wire ar_fire=m_arvalid && m_arready;
    wire [8:0] offered_aw={1'b0,aw_pending}+s_awvalid;
    wire [8:0] offered_w={1'b0,w_pending}+s_wvalid;
    wire [8:0] prefix_writes=offered_aw>offered_w ? offered_aw:offered_w;
    assign source_fenced=cold_rst_n && state!=RUN && aw_left==0 && w_left==0 && !ar_left;
    assign protocol_fault=fault || late_response;
    assign drain_ack=ledger_ack && state==REJECT && !protocol_fault;
    assign local_idle=cold_rst_n && state==REJECT &&
        !(reject_aw || reject_w || reject_b || reject_r || s_awvalid || s_wvalid || s_arvalid);
    assign state_debug=state;
    ntrpu_axil_drain_monitor #(.COUNT_WIDTH(8)) ledger(
        .clk(clk),.cold_rst_n(cold_rst_n),.req_valid(proof_request),.req_tag(proof_tag),
        .source_fenced(source_fenced),.awvalid(m_awvalid),.awready(m_awready),
        .wvalid(m_wvalid),.wready(m_wready),.bvalid(m_bvalid),.bready(m_bready),
        .arvalid(m_arvalid),.arready(m_arready),.rvalid(m_rvalid),.rready(m_rready),
        .ack_valid(ledger_ack),.ack_tag(drain_tag),.protocol_fault(fault),
        .aw_pending(aw_pending),.w_pending(w_pending),.ar_pending(ar_pending));
    always @(posedge clk)begin
        if(!cold_rst_n)begin
            state<=RUN;aw_left<=0;w_left<=0;ar_left<=0;
            reject_aw<=0;reject_w<=0;reject_b<=0;reject_r<=0;late_response<=0;
        end else begin
            case(state)
                RUN:if(close_request)begin
                    // Counts describe the instant BEFORE this edge. Include
                    // each currently offered half once, whether READY or not;
                    // subtract handshakes at this edge, not response retirements.
                    aw_left<=prefix_writes-{1'b0,aw_pending}-aw_fire;
                    w_left<=prefix_writes-{1'b0,w_pending}-w_fire;
                    ar_left<=s_arvalid && !ar_fire;
                    state<=FINISH;
                end
                FINISH:begin
                    if(aw_fire)aw_left<=aw_left-1'b1;
                    if(w_fire)w_left<=w_left-1'b1;
                    if(ar_fire)ar_left<=0;
                    if(source_fenced && {aw_pending,w_pending,ar_pending}==0 &&
                       !(m_bvalid || m_rvalid) && !protocol_fault)state<=REJECT;
                end
                REJECT:begin
                    if(m_bvalid || m_rvalid)late_response<=1;
                    if(s_awvalid && s_awready)reject_aw<=1;
                    if(s_wvalid && s_wready)reject_w<=1;
                    if(reject_aw && reject_w && !reject_b)reject_b<=1;
                    if(reject_b && s_bready)begin reject_b<=0;reject_aw<=0;reject_w<=0;end
                    if(s_arvalid && s_arready)reject_r<=1;
                    if(reject_r && s_rready)reject_r<=0;
                end
                default:late_response<=1;
            endcase
        end
    end
    // synthesis translate_off
    initial if(MAX_PENDING<1 || MAX_PENDING>254)$fatal(1,"AXIL_ADMISSION_PARAMETERS");
    // synthesis translate_on
endmodule
