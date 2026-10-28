`timescale 1ns/1ps
// Independent AMD CDC followed by the preserved 64-bit capacity guard.
// Cold/coordinated resets only; all peers and both clock domains must overlap.
// No PCIe framing, window authorization, controller calibration or recovery supervisor.
module ntrpu_peer_memory_port #(parameter integer MEMORY_ADDR_WIDTH=37) (
    input wire clk, rst_n, mem_clk, mem_rst_n,
    input wire s_awvalid,
    output wire s_awready,
    input wire [63:0] s_awaddr,
    input wire [7:0] s_awlen,
    input wire [3:0] s_awid,
    input wire [3:0] s_awcache,
    input wire [2:0] s_awsize,
    input wire [1:0] s_awburst,
    input wire s_wvalid,
    input wire s_wlast,
    output wire s_wready,
    input wire [511:0] s_wdata,
    input wire [63:0] s_wstrb,
    output wire s_bvalid,
    input wire s_bready,
    output wire [3:0] s_bid,
    output wire [1:0] s_bresp,
    output wire m_awvalid,
    input wire m_awready,
    output wire [MEMORY_ADDR_WIDTH-1:0] m_awaddr,
    output wire [7:0] m_awlen,
    output wire [3:0] m_awid,
    output wire [3:0] m_awcache,
    output wire [2:0] m_awsize,
    output wire [1:0] m_awburst,
    output wire m_wvalid,
    output wire m_wlast,
    input wire m_wready,
    output wire [511:0] m_wdata,
    output wire [63:0] m_wstrb,
    input wire m_bvalid,
    output wire m_bready,
    input wire [3:0] m_bid,
    input wire [1:0] m_bresp
);
    wire d_awvalid;
    wire d_awready;
    wire [63:0] d_awaddr;
    wire [7:0] d_awlen;
    wire [3:0] d_awid;
    wire [3:0] d_awcache;
    wire [2:0] d_awsize;
    wire [1:0] d_awburst;
    wire d_wvalid;
    wire d_wlast;
    wire d_wready;
    wire [511:0] d_wdata;
    wire [63:0] d_wstrb;
    wire d_bvalid;
    wire d_bready;
    wire [3:0] d_bid;
    wire [1:0] d_bresp;
    wire [2:0] d_awprot;
    wire d_awlock;
    ntrpu_peer_memory_cdc cdc (
        .s_axi_aclk(clk), .s_axi_aresetn(rst_n),
        .m_axi_aclk(mem_clk), .m_axi_aresetn(mem_rst_n),
        .s_axi_awvalid(s_awvalid),
        .s_axi_awready(s_awready),
        .s_axi_awaddr(s_awaddr),
        .s_axi_awlen(s_awlen),
        .s_axi_awid(s_awid),
        .s_axi_awcache(s_awcache),
        .s_axi_awsize(s_awsize),
        .s_axi_awburst(s_awburst),
        .s_axi_wvalid(s_wvalid),
        .s_axi_wlast(s_wlast),
        .s_axi_wready(s_wready),
        .s_axi_wdata(s_wdata),
        .s_axi_wstrb(s_wstrb),
        .s_axi_bvalid(s_bvalid),
        .s_axi_bready(s_bready),
        .s_axi_bid(s_bid),
        .s_axi_bresp(s_bresp),
        .m_axi_awvalid(d_awvalid),
        .m_axi_awready(d_awready),
        .m_axi_awaddr(d_awaddr),
        .m_axi_awlen(d_awlen),
        .m_axi_awid(d_awid),
        .m_axi_awcache(d_awcache),
        .m_axi_awsize(d_awsize),
        .m_axi_awburst(d_awburst),
        .m_axi_wvalid(d_wvalid),
        .m_axi_wlast(d_wlast),
        .m_axi_wready(d_wready),
        .m_axi_wdata(d_wdata),
        .m_axi_wstrb(d_wstrb),
        .m_axi_bvalid(d_bvalid),
        .m_axi_bready(d_bready),
        .m_axi_bid(d_bid),
        .m_axi_bresp(d_bresp),
        .s_axi_awprot(3'b0), .s_axi_awlock(1'b0), .s_axi_awqos(4'b0), .s_axi_awregion(4'b0),
        .m_axi_awprot(d_awprot), .m_axi_awlock(d_awlock), .m_axi_awqos(), .m_axi_awregion(),
        .s_axi_araddr(64'b0), .s_axi_arlen(8'b0), .s_axi_arsize(3'b0), .s_axi_arburst(2'b01),
        .s_axi_arid(4'b0), .s_axi_arcache(4'b0), .s_axi_arprot(3'b0), .s_axi_arlock(1'b0),
        .s_axi_arqos(4'b0), .s_axi_arregion(4'b0), .s_axi_arvalid(1'b0), .s_axi_arready(),
        .s_axi_rdata(), .s_axi_rid(), .s_axi_rresp(), .s_axi_rlast(), .s_axi_rvalid(), .s_axi_rready(1'b1),
        .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(), .m_axi_arid(),
        .m_axi_arcache(), .m_axi_arprot(), .m_axi_arlock(), .m_axi_arqos(), .m_axi_arregion(),
        .m_axi_arvalid(), .m_axi_arready(1'b0), .m_axi_rdata(512'b0), .m_axi_rid(4'b0),
        .m_axi_rresp(2'b0), .m_axi_rlast(1'b0), .m_axi_rvalid(1'b0), .m_axi_rready()
    );
    ntrpu_axi_capacity_guard #(.MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH)) capacity (
        .clk(mem_clk), .rst_n(mem_rst_n), .admit(1'b1), .idle(),
        .s_axi_awvalid(d_awvalid),
        .s_axi_awready(d_awready),
        .s_axi_awaddr(d_awaddr),
        .s_axi_awlen(d_awlen),
        .s_axi_awid(d_awid),
        .s_axi_awcache(d_awcache),
        .s_axi_awsize(d_awsize),
        .s_axi_awburst(d_awburst),
        .s_axi_wvalid(d_wvalid),
        .s_axi_wlast(d_wlast),
        .s_axi_wready(d_wready),
        .s_axi_wdata(d_wdata),
        .s_axi_wstrb(d_wstrb),
        .s_axi_bvalid(d_bvalid),
        .s_axi_bready(d_bready),
        .s_axi_bid(d_bid),
        .s_axi_bresp(d_bresp),
        .m_axi_awvalid(m_awvalid),
        .m_axi_awready(m_awready),
        .m_axi_awaddr(m_awaddr),
        .m_axi_awlen(m_awlen),
        .m_axi_awid(m_awid),
        .m_axi_awcache(m_awcache),
        .m_axi_awsize(m_awsize),
        .m_axi_awburst(m_awburst),
        .m_axi_wvalid(m_wvalid),
        .m_axi_wlast(m_wlast),
        .m_axi_wready(m_wready),
        .m_axi_wdata(m_wdata),
        .m_axi_wstrb(m_wstrb),
        .m_axi_bvalid(m_bvalid),
        .m_axi_bready(m_bready),
        .m_axi_bid(m_bid),
        .m_axi_bresp(m_bresp),
        .s_axi_awprot(d_awprot), .s_axi_awlock(d_awlock), .m_axi_awprot(), .m_axi_awlock(),
        .s_axi_araddr(64'b0), .s_axi_arlen(8'b0), .s_axi_arsize(3'b0), .s_axi_arburst(2'b01),
        .s_axi_arid(4'b0), .s_axi_arcache(4'b0), .s_axi_arprot(3'b0), .s_axi_arlock(1'b0),
        .s_axi_arvalid(1'b0), .s_axi_arready(), .s_axi_rdata(), .s_axi_rid(),
        .s_axi_rresp(), .s_axi_rlast(), .s_axi_rvalid(), .s_axi_rready(1'b1),
        .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(), .m_axi_arid(),
        .m_axi_arcache(), .m_axi_arprot(), .m_axi_arlock(), .m_axi_arvalid(), .m_axi_arready(1'b0),
        .m_axi_rdata(512'b0), .m_axi_rid(4'b0), .m_axi_rresp(2'b0),
        .m_axi_rlast(1'b0), .m_axi_rvalid(1'b0), .m_axi_rready()
    );
endmodule
