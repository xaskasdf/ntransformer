`timescale 1ns / 1ps
// NTRPUF01 single-clock write-only ingress -> peer receiver -> capacity firewall.
// Preserve all 64 address bits until the firewall accepts the complete span.
// Isolation and permissions are externally trusted; reset is cold/coordinated.
module ntrpu_pcie_fragment_guarded #(parameter integer MEMORY_ADDR_WIDTH=37) (
    input wire clk, rst_n, stop,
    input wire source_stopped, ingress_isolated,
    input wire [63:0] quiesce_epoch,
    input wire arm_valid,
    output wire arm_ready,
    input wire [63:0] arm_epoch, arm_tag, arm_window, arm_destination,
    output wire [63:0] epoch,
    output wire lease_held, failed,
    output wire [3:0] error_code,
    output wire [31:0] rejected_frames, rejected_identity,
    output wire [3:0] last_reject_code,
    output wire [6:0] last_reject_faults,

    output wire transport_idle,
    output wire guard_idle,
    output wire [63:0] valid_bytes,
    output wire [63:0] peer_epoch,
    output wire peer_quarantined, peer_axi_busy,
    output wire [2:0] bridge_phase,
    output wire [2:0] destination_status,

    input wire [63:0] s_axi_awaddr,
    input wire [3:0] s_axi_awid,
    input wire [7:0] s_axi_awlen,
    input wire [2:0] s_axi_awsize,
    input wire [1:0] s_axi_awburst,
    input wire s_axi_awlock, s_axi_awvalid,
    output wire s_axi_awready,
    input wire [511:0] s_axi_wdata,
    input wire [63:0] s_axi_wstrb,
    input wire s_axi_wlast, s_axi_wvalid,
    output wire s_axi_wready,
    output wire [3:0] s_axi_bid,
    output wire [1:0] s_axi_bresp,
    output wire s_axi_bvalid,
    input wire s_axi_bready,
    output wire result_valid,
    input wire result_ready,
    output wire [63:0] result_epoch, result_tag,
    output wire [3:0] result_status,

    output wire [MEMORY_ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awcache, m_axi_awid,
    output wire [2:0] m_axi_awprot,
    output wire m_axi_awlock, m_axi_awvalid,
    input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast, m_axi_wvalid,
    input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,
    input wire [3:0] m_axi_bid,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    wire peer_arm_ready;
    assign arm_ready = peer_arm_ready && guard_idle;
    wire [63:0] peer_awaddr;
    wire [7:0] peer_awlen;
    wire [2:0] peer_awsize;
    wire [1:0] peer_awburst;
    wire [3:0] peer_awcache;
    wire [3:0] peer_awid;
    wire [2:0] peer_awprot;
    wire peer_awlock;
    wire peer_awvalid;
    wire peer_awready;
    wire [511:0] peer_wdata;
    wire [63:0] peer_wstrb;
    wire peer_wlast;
    wire peer_wvalid;
    wire peer_wready;
    wire [1:0] peer_bresp;
    wire [3:0] peer_bid;
    wire peer_bvalid;
    wire peer_bready;
    ntrpu_pcie_fragment_peer peer (
        .clk(clk),
        .rst_n(rst_n),
        .stop(stop),
        .source_stopped(source_stopped),
        .ingress_isolated(ingress_isolated),
        .quiesce_epoch(quiesce_epoch),
        .arm_valid(arm_valid && guard_idle),
        .arm_ready(peer_arm_ready),
        .arm_epoch(arm_epoch),
        .arm_tag(arm_tag),
        .arm_window(arm_window),
        .arm_destination(arm_destination),
        .epoch(epoch),
        .lease_held(lease_held),
        .failed(failed),
        .error_code(error_code),
        .rejected_frames(rejected_frames),.rejected_identity(rejected_identity),
        .last_reject_code(last_reject_code),.last_reject_faults(last_reject_faults),

        .transport_idle(transport_idle),
        .valid_bytes(valid_bytes),
        .peer_epoch(peer_epoch),
        .peer_quarantined(peer_quarantined),
        .peer_axi_busy(peer_axi_busy),
        .bridge_phase(bridge_phase),
        .destination_status(destination_status),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awid(s_axi_awid),
        .s_axi_awlen(s_axi_awlen),
        .s_axi_awsize(s_axi_awsize),
        .s_axi_awburst(s_axi_awburst),
        .s_axi_awlock(s_axi_awlock),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wlast(s_axi_wlast),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bid(s_axi_bid),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .result_epoch(result_epoch),
        .result_tag(result_tag),
        .result_status(result_status),
        .m_axi_awaddr(peer_awaddr),
        .m_axi_awlen(peer_awlen),
        .m_axi_awsize(peer_awsize),
        .m_axi_awburst(peer_awburst),
        .m_axi_awcache(peer_awcache),
        .m_axi_awid(peer_awid),
        .m_axi_awprot(peer_awprot),
        .m_axi_awlock(peer_awlock),
        .m_axi_awvalid(peer_awvalid),
        .m_axi_awready(peer_awready),
        .m_axi_wdata(peer_wdata),
        .m_axi_wstrb(peer_wstrb),
        .m_axi_wlast(peer_wlast),
        .m_axi_wvalid(peer_wvalid),
        .m_axi_wready(peer_wready),
        .m_axi_bresp(peer_bresp),
        .m_axi_bid(peer_bid),
        .m_axi_bvalid(peer_bvalid),
        .m_axi_bready(peer_bready)
    );
    // Application stop must not close admission on an exposed line.
    // Memory reset/admission policy belongs to the coordinated domain reset.
    // Downstream responses must obey AXI; this filter is not a corrupt-B sink.
    ntrpu_axi_capacity_guard #(.MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH)) capacity (
        .clk(clk),.rst_n(rst_n),.admit(rst_n),.idle(guard_idle),
        .s_axi_araddr(64'd0),.s_axi_arid(4'd0),.s_axi_arlen(8'd0),
        .s_axi_arsize(3'd6),.s_axi_arburst(2'd1),.s_axi_arcache(4'd0),
        .s_axi_arprot(3'd0),.s_axi_arlock(1'b0),.s_axi_arvalid(1'b0),
        .s_axi_arready(),.s_axi_rdata(),.s_axi_rresp(),.s_axi_rid(),
        .s_axi_rlast(),.s_axi_rvalid(),.s_axi_rready(1'b1),
        .m_axi_araddr(),.m_axi_arid(),.m_axi_arlen(),.m_axi_arsize(),
        .m_axi_arburst(),.m_axi_arcache(),.m_axi_arprot(),.m_axi_arlock(),
        .m_axi_arvalid(),.m_axi_arready(1'b0),.m_axi_rdata(512'd0),
        .m_axi_rresp(2'd0),.m_axi_rid(4'd0),.m_axi_rlast(1'b0),
        .m_axi_rvalid(1'b0),.m_axi_rready(),
        .s_axi_awaddr(peer_awaddr),
        .s_axi_awlen(peer_awlen),
        .s_axi_awsize(peer_awsize),
        .s_axi_awburst(peer_awburst),
        .s_axi_awcache(peer_awcache),
        .s_axi_awid(peer_awid),
        .s_axi_awprot(peer_awprot),
        .s_axi_awlock(peer_awlock),
        .s_axi_awvalid(peer_awvalid),
        .s_axi_awready(peer_awready),
        .s_axi_wdata(peer_wdata),
        .s_axi_wstrb(peer_wstrb),
        .s_axi_wlast(peer_wlast),
        .s_axi_wvalid(peer_wvalid),
        .s_axi_wready(peer_wready),
        .s_axi_bresp(peer_bresp),
        .s_axi_bid(peer_bid),
        .s_axi_bvalid(peer_bvalid),
        .s_axi_bready(peer_bready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awid(m_axi_awid),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bid(m_axi_bid),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready)
    );
endmodule
