`timescale 1ns / 1ps
// Cold/coordinated host transport around the retained NTRD/two-MIG core.
// s_axi/s_axil use host_clk; diagnostics/barrier use source_clk.
// host_rst_n and transport_rst_n must cover both complete transport domains.
// Neither is the session reset. Runtime PCIe reset/clock loss requires system
// invalidation/quarantine and is not qualified by this cold-only wrapper.
// source_stopped/ingress_isolated must prove upstream and both CDCs drained;
// local transport_idle alone is insufficient. IRQ is source-domain only.
module ntrpu_pcie_fragment_dual_host_cold #(parameter [63:0] FIRST_COOKIE=64'd1) (

    input wire host_clk, host_rst_n, transport_rst_n,
    input wire mgmt_clk, source_clk, external_reset_async, rearm,
    input wire [1:0] ui_clk, calibrated, ui_rst,
    input wire [1:0] m_rvalid,
    output wire source_reset_n, admit, running, rearm_ready, restart_required,
    output wire peers_reset_req, source_alive, lifetime_reset_n, restart_required_source,
    output wire [1:0] memory_reset_n, memory_alive,
    output wire [7:0] banks_used,
    output wire [9:0] pending_counts,
    output wire [1:0] m_awvalid, m_wvalid, m_wlast, m_bready,
    input wire [1:0] m_awready, m_wready, m_bvalid,
    output wire [73:0] m_awaddr,
    output wire [15:0] m_awlen,
    output wire [7:0] m_awid, m_awcache,
    output wire [5:0] m_awsize,
    output wire [3:0] m_awburst,
    output wire [1023:0] m_wdata,
    output wire [127:0] m_wstrb,
    input wire [7:0] m_bid,
    input wire [3:0] m_bresp,



    output wire stop,
    input wire source_stopped, ingress_isolated,
    input wire [63:0] quiesce_epoch,
    output wire arm_valid,
    output wire arm_ready,
    output wire [63:0] arm_epoch, arm_tag, arm_window, arm_destination,
    output wire [63:0] epoch,
    output wire lease_held, failed,
    output wire [3:0] error_code, // Lease errors:1stop,4conflict,5destination,6late write; frame counters also2format,3range,7identity
    output wire transport_idle,
    output wire [63:0] valid_bytes,
    output wire [31:0] rejected_frames, rejected_identity,
    output wire [3:0] last_reject_code,
    output wire [6:0] last_reject_faults,

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

    output wire line_valid,
    output wire line_ready,
    output wire [63:0] line_epoch, line_tag, line_address,
    output wire [511:0] line_data,
    output wire done_valid,
    output wire done_ready,
    output wire [63:0] done_epoch, done_tag,
    output wire done_error,
    output wire result_valid,
    output wire result_ready,
    output wire [63:0] result_epoch, result_tag,
    output wire [3:0] result_status,

    output wire arm_channel,
    output wire ack_valid,
    output wire ack_ready,
    output wire [63:0] ack_epoch, ack_tag,
    output wire pool_admit, pool_restart_required, pool_fault, pool_drained,
    output wire cmd_valid,
    output wire cmd_ready,
    output wire cmd_channel,
    output wire [63:0] cmd_addr, cmd_tag,
    output wire [7:0] cmd_len,
    output wire data_valid,
    output wire data_ready,
    output wire [511:0] data,
    output wire [1:0] cpl_valid,
    output wire [1:0] cpl_ready,
    output wire [127:0] cpl_tag,
    output wire [3:0] cpl_status,
    output wire bridge_lease_held,
    output wire [63:0] bridge_epoch,
    output wire bridge_idle, bridge_arm_ready, collector_arm_ready, retire_ready,
    output wire [2:0] bridge_phase,
    output wire [3:0] done_status,
    output wire [1:0] done_pool_status,
    output wire protocol_fault, session_lost, session_invalid, exhausted,
    output wire [63:0] next_cookie,
    output wire [31:0] rejected_completions,
    input wire [31:0] s_axil_awaddr, s_axil_wdata, s_axil_araddr,
    input wire [3:0] s_axil_wstrb,
    input wire s_axil_awvalid, s_axil_wvalid, s_axil_bready,
    input wire s_axil_arvalid, s_axil_rready,
    output wire s_axil_awready, s_axil_wready, s_axil_arready,
    output wire s_axil_bvalid, s_axil_rvalid,
    output wire [1:0] s_axil_bresp, s_axil_rresp,
    output wire [31:0] s_axil_rdata,
    output wire irq, control_fault
);
    wire [63:0] d_axi_awaddr;
    wire [3:0] d_axi_awid;
    wire [7:0] d_axi_awlen;
    wire [2:0] d_axi_awsize;
    wire [1:0] d_axi_awburst;
    wire  d_axi_awlock;
    wire  d_axi_awvalid;
    wire  d_axi_awready;
    wire [511:0] d_axi_wdata;
    wire [63:0] d_axi_wstrb;
    wire  d_axi_wlast;
    wire  d_axi_wvalid;
    wire  d_axi_wready;
    wire [3:0] d_axi_bid;
    wire [1:0] d_axi_bresp;
    wire  d_axi_bvalid;
    wire  d_axi_bready;
    wire [31:0] d_axil_awaddr;
    wire [31:0] d_axil_wdata;
    wire [31:0] d_axil_araddr;
    wire [3:0] d_axil_wstrb;
    wire  d_axil_awvalid;
    wire  d_axil_wvalid;
    wire  d_axil_bready;
    wire  d_axil_arvalid;
    wire  d_axil_rready;
    wire  d_axil_awready;
    wire  d_axil_wready;
    wire  d_axil_arready;
    wire  d_axil_bvalid;
    wire  d_axil_rvalid;
    wire [1:0] d_axil_bresp;
    wire [1:0] d_axil_rresp;
    wire [31:0] d_axil_rdata;
    ntrpu_pcie_fragment_dual_mig #(.FIRST_COOKIE(FIRST_COOKIE)) core (
        .s_axi_awaddr(d_axi_awaddr),
        .s_axi_awid(d_axi_awid),
        .s_axi_awlen(d_axi_awlen),
        .s_axi_awsize(d_axi_awsize),
        .s_axi_awburst(d_axi_awburst),
        .s_axi_awlock(d_axi_awlock),
        .s_axi_awvalid(d_axi_awvalid),
        .s_axi_awready(d_axi_awready),
        .s_axi_wdata(d_axi_wdata),
        .s_axi_wstrb(d_axi_wstrb),
        .s_axi_wlast(d_axi_wlast),
        .s_axi_wvalid(d_axi_wvalid),
        .s_axi_wready(d_axi_wready),
        .s_axi_bid(d_axi_bid),
        .s_axi_bresp(d_axi_bresp),
        .s_axi_bvalid(d_axi_bvalid),
        .s_axi_bready(d_axi_bready),
        .s_axil_awaddr(d_axil_awaddr),
        .s_axil_wdata(d_axil_wdata),
        .s_axil_araddr(d_axil_araddr),
        .s_axil_wstrb(d_axil_wstrb),
        .s_axil_awvalid(d_axil_awvalid),
        .s_axil_wvalid(d_axil_wvalid),
        .s_axil_bready(d_axil_bready),
        .s_axil_arvalid(d_axil_arvalid),
        .s_axil_rready(d_axil_rready),
        .s_axil_awready(d_axil_awready),
        .s_axil_wready(d_axil_wready),
        .s_axil_arready(d_axil_arready),
        .s_axil_bvalid(d_axil_bvalid),
        .s_axil_rvalid(d_axil_rvalid),
        .s_axil_bresp(d_axil_bresp),
        .s_axil_rresp(d_axil_rresp),
        .s_axil_rdata(d_axil_rdata),
        .*);
    ntrpu_axil_cdc_recode control_cdc (
        .s_axi_aclk(host_clk), .s_axi_aresetn(host_rst_n),
        .m_axi_aclk(source_clk), .m_axi_aresetn(transport_rst_n),
        .s_axi_awprot(3'b0), .s_axi_arprot(3'b0), .m_axi_awprot(), .m_axi_arprot(),
        .s_axi_awaddr(s_axil_awaddr), .m_axi_awaddr(d_axil_awaddr),
        .s_axi_wdata(s_axil_wdata), .m_axi_wdata(d_axil_wdata),
        .s_axi_araddr(s_axil_araddr), .m_axi_araddr(d_axil_araddr),
        .s_axi_wstrb(s_axil_wstrb), .m_axi_wstrb(d_axil_wstrb),
        .s_axi_awvalid(s_axil_awvalid), .m_axi_awvalid(d_axil_awvalid),
        .s_axi_wvalid(s_axil_wvalid), .m_axi_wvalid(d_axil_wvalid),
        .s_axi_bready(s_axil_bready), .m_axi_bready(d_axil_bready),
        .s_axi_arvalid(s_axil_arvalid), .m_axi_arvalid(d_axil_arvalid),
        .s_axi_rready(s_axil_rready), .m_axi_rready(d_axil_rready),
        .s_axi_awready(s_axil_awready), .m_axi_awready(d_axil_awready),
        .s_axi_wready(s_axil_wready), .m_axi_wready(d_axil_wready),
        .s_axi_bvalid(s_axil_bvalid), .m_axi_bvalid(d_axil_bvalid),
        .s_axi_arready(s_axil_arready), .m_axi_arready(d_axil_arready),
        .s_axi_rvalid(s_axil_rvalid), .m_axi_rvalid(d_axil_rvalid),
        .s_axi_bresp(s_axil_bresp), .m_axi_bresp(d_axil_bresp),
        .s_axi_rresp(s_axil_rresp), .m_axi_rresp(d_axil_rresp),
        .s_axi_rdata(s_axil_rdata), .m_axi_rdata(d_axil_rdata)
    );
    ntrpu_ingress_axi_cdc ingress_cdc (
        .s_axi_aclk(host_clk),
        .s_axi_aresetn(host_rst_n),
        .m_axi_aclk(source_clk),
        .m_axi_aresetn(transport_rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .m_axi_awaddr(d_axi_awaddr),
        .s_axi_awid(s_axi_awid),
        .m_axi_awid(d_axi_awid),
        .s_axi_awlen(s_axi_awlen),
        .m_axi_awlen(d_axi_awlen),
        .s_axi_awsize(s_axi_awsize),
        .m_axi_awsize(d_axi_awsize),
        .s_axi_awburst(s_axi_awburst),
        .m_axi_awburst(d_axi_awburst),
        .s_axi_awlock(s_axi_awlock),
        .m_axi_awlock(d_axi_awlock),
        .s_axi_awvalid(s_axi_awvalid),
        .m_axi_awvalid(d_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .m_axi_awready(d_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .m_axi_wdata(d_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .m_axi_wstrb(d_axi_wstrb),
        .s_axi_wlast(s_axi_wlast),
        .m_axi_wlast(d_axi_wlast),
        .s_axi_wvalid(s_axi_wvalid),
        .m_axi_wvalid(d_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .m_axi_wready(d_axi_wready),
        .s_axi_bid(s_axi_bid),
        .m_axi_bid(d_axi_bid),
        .s_axi_bresp(s_axi_bresp),
        .m_axi_bresp(d_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .m_axi_bvalid(d_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .m_axi_bready(d_axi_bready),
        .s_axi_awcache(4'd0),
        .m_axi_awcache(),
        .s_axi_awprot(3'd0),
        .m_axi_awprot(),
        .s_axi_awqos(4'd0),
        .m_axi_awqos(),
        .s_axi_awregion(4'd0),
        .m_axi_awregion(),
        .s_axi_araddr(64'd0),
        .m_axi_araddr(),
        .s_axi_arid(4'd0),
        .m_axi_arid(),
        .s_axi_arlen(8'd0),
        .m_axi_arlen(),
        .s_axi_arsize(3'd2),
        .m_axi_arsize(),
        .s_axi_arburst(2'd1),
        .m_axi_arburst(),
        .s_axi_arlock(1'b0),
        .m_axi_arlock(),
        .s_axi_arcache(4'd0),
        .m_axi_arcache(),
        .s_axi_arprot(3'd0),
        .m_axi_arprot(),
        .s_axi_arqos(4'd0),
        .m_axi_arqos(),
        .s_axi_arregion(4'd0),
        .m_axi_arregion(),
        .s_axi_arvalid(1'b0),
        .m_axi_arvalid(),
        .s_axi_rready(1'b1),
        .m_axi_rready(),
        .s_axi_arready(),
        .m_axi_arready(1'b0),
        .s_axi_rdata(),
        .m_axi_rdata(512'd0),
        .s_axi_rid(),
        .m_axi_rid(4'd0),
        .s_axi_rresp(),
        .m_axi_rresp(2'd0),
        .s_axi_rlast(),
        .m_axi_rlast(1'b0),
        .s_axi_rvalid(),
        .m_axi_rvalid(1'b0)
    );
endmodule
