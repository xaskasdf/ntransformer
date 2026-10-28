`timescale 1ns / 1ps
// NTRF control and NTRPUF01 data: host AXI/AXI-Lite cross through real AMD bridges to UI-clock ingress.
// External trusted barrier includes both upstream producers and queued data.
// Diagnostic/control outputs are UI-domain; reset is cold/coordinated only.
module ntrpu_pcie_fragment_host #(parameter integer MEMORY_ADDR_WIDTH=37) (
    input wire [31:0] s_axil_awaddr, s_axil_wdata, s_axil_araddr,
    input wire [3:0] s_axil_wstrb,
    input wire s_axil_awvalid, s_axil_wvalid, s_axil_bready,
    input wire s_axil_arvalid, s_axil_rready,
    output wire s_axil_awready, s_axil_wready, s_axil_arready,
    output wire s_axil_bvalid, s_axil_rvalid,
    output wire [1:0] s_axil_bresp, s_axil_rresp,
    output wire [31:0] s_axil_rdata,
    output wire irq, protocol_fault,
    input wire clk, rst_n, host_clk, host_rst_n,
    output wire stop,
    input wire source_stopped, ingress_isolated,
    input wire [63:0] quiesce_epoch,
    output wire arm_valid,
    output wire arm_ready,
    output wire [63:0] arm_epoch, arm_tag, arm_window, arm_destination,
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
    output wire result_ready,
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
    wire [31:0] d_axil_awaddr;
    wire [31:0] d_axil_wdata;
    wire [31:0] d_axil_araddr;
    wire [3:0] d_axil_wstrb;
    wire d_axil_awvalid;
    wire d_axil_wvalid;
    wire d_axil_bready;
    wire d_axil_arvalid;
    wire d_axil_rready;
    wire d_axil_awready;
    wire d_axil_wready;
    wire d_axil_bvalid;
    wire d_axil_arready;
    wire d_axil_rvalid;
    wire [1:0] d_axil_bresp;
    wire [1:0] d_axil_rresp;
    wire [31:0] d_axil_rdata;
    ntrpu_pcie_fragment_control control (
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
        .s_axil_bvalid(d_axil_bvalid),
        .s_axil_arready(d_axil_arready),
        .s_axil_rvalid(d_axil_rvalid),
        .s_axil_bresp(d_axil_bresp),
        .s_axil_rresp(d_axil_rresp),
        .s_axil_rdata(d_axil_rdata),
        .*);
    ntrpu_pcie_fragment_async #(.MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH)) data_path (.*);
    ntrpu_axil_cdc_recode control_cdc (
        .s_axi_aclk(host_clk), .s_axi_aresetn(host_rst_n),
        .m_axi_aclk(clk), .m_axi_aresetn(rst_n),
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
endmodule
