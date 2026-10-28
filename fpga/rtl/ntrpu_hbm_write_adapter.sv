`timescale 1ns/1ps
// Experimental, single-clock write adapter for a zero-based 16-GiB window.
// Four parent credits; ID zero; aligned INCR 64-byte source beats within 4 KiB.
// AMD hbm_width: AXI4 512->256, MAX_SPLIT_BEATS=16, no CDC.
// AMD hbm_protocol: AXI4->AXI3, protected translation. No hard HBM controller.
// Cold coordinated reset only. Malformed source W streams are not recoverable.
module ntrpu_hbm_write_adapter (
    input wire clk,rst_n,admit,
    output wire idle,
    output reg fault,
    input wire [63:0] s_axi_awaddr,
    input wire [7:0] s_axi_awlen,
    input wire [2:0] s_axi_awsize,s_axi_awprot,
    input wire [1:0] s_axi_awburst,
    input wire [3:0] s_axi_awid,s_axi_awcache,
    input wire s_axi_awlock,s_axi_awvalid,
    output wire s_axi_awready,
    input wire [511:0] s_axi_wdata,
    input wire [63:0] s_axi_wstrb,
    input wire s_axi_wlast,s_axi_wvalid,
    output wire s_axi_wready,
    output wire [3:0] s_axi_bid,
    output wire [1:0] s_axi_bresp,
    output wire s_axi_bvalid,
    input wire s_axi_bready,
    output wire [33:0] m_axi_awaddr,
    output wire [3:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output wire [5:0] m_axi_awid,
    output wire m_axi_awvalid,
    input wire m_axi_awready,
    output wire [255:0] m_axi_wdata,
    output wire [31:0] m_axi_wstrb,
    output wire m_axi_wlast,m_axi_wvalid,
    input wire m_axi_wready,
    input wire [5:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    wire [33:0] g_awaddr,d_awaddr;
    wire [7:0] g_awlen,d_awlen;
    wire [2:0] g_awsize,g_awprot,d_awsize,d_awprot;
    wire [1:0] g_awburst,g_bresp,d_awburst,d_bresp;
    wire [3:0] g_awid,g_awcache,g_bid,d_awcache,d_awregion,d_awqos,p_awid,p_wid;
    wire g_awlock,g_awvalid,g_awready,g_wlast,g_wvalid,g_wready,g_bvalid,g_bready;
    wire d_awlock,d_awvalid,d_awready,d_wlast,d_wvalid,d_wready,d_bvalid,d_bready,p_bready;
    wire [511:0] g_wdata;
    wire [63:0] g_wstrb;
    wire [255:0] d_wdata;
    wire [31:0] d_wstrb;
    // Check all six BID bits before the four-bit protocol converter.
    // Keep bad responses unaccepted; committed requests retain their state.
    wire bad_response=m_axi_bvalid&&(m_axi_bid!=0||m_axi_bresp==2'b01);
    wire guard_admit=admit&&!fault&&!bad_response;
    always @(posedge clk)begin
        if(!rst_n)fault<=0;
        else if(bad_response)fault<=1;
    end
    assign m_axi_bready=p_bready&&!bad_response;
    assign m_axi_awid={2'b00,p_awid};

    ntrpu_axi_write_capacity_queue #(.MEMORY_ADDR_WIDTH(34),.DEPTH(4)) guard (
        .clk(clk),.rst_n(rst_n),.admit(guard_admit),.idle(idle),
        .s_axi_awaddr(s_axi_awaddr),.s_axi_awlen(s_axi_awlen),.s_axi_awsize(s_axi_awsize),
        .s_axi_awprot(s_axi_awprot),.s_axi_awburst(s_axi_awburst),.s_axi_awid(s_axi_awid),
        .s_axi_awcache(s_axi_awcache),.s_axi_awlock(s_axi_awlock),.s_axi_awvalid(s_axi_awvalid),.s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),.s_axi_wstrb(s_axi_wstrb),.s_axi_wlast(s_axi_wlast),.s_axi_wvalid(s_axi_wvalid),.s_axi_wready(s_axi_wready),
        .s_axi_bid(s_axi_bid),.s_axi_bresp(s_axi_bresp),.s_axi_bvalid(s_axi_bvalid),.s_axi_bready(s_axi_bready),
        .m_axi_awaddr(g_awaddr),.m_axi_awlen(g_awlen),.m_axi_awsize(g_awsize),.m_axi_awprot(g_awprot),
        .m_axi_awburst(g_awburst),.m_axi_awid(g_awid),.m_axi_awcache(g_awcache),.m_axi_awlock(g_awlock),.m_axi_awvalid(g_awvalid),.m_axi_awready(g_awready),
        .m_axi_wdata(g_wdata),.m_axi_wstrb(g_wstrb),.m_axi_wlast(g_wlast),.m_axi_wvalid(g_wvalid),.m_axi_wready(g_wready),
        .m_axi_bid(g_bid),.m_axi_bresp(g_bresp),.m_axi_bvalid(g_bvalid),.m_axi_bready(g_bready)
    );
    hbm_width width_converter (
        .s_axi_aclk(clk),.s_axi_aresetn(rst_n),
        .s_axi_awaddr(g_awaddr),.s_axi_awlen(g_awlen),.s_axi_awsize(g_awsize),.s_axi_awprot(g_awprot),
        .s_axi_awburst(g_awburst),.s_axi_awid(g_awid),.s_axi_awcache(g_awcache),.s_axi_awlock(g_awlock),
        .s_axi_awregion(4'b0),.s_axi_awqos(4'b0),.s_axi_awvalid(g_awvalid),.s_axi_awready(g_awready),
        .s_axi_wdata(g_wdata),.s_axi_wstrb(g_wstrb),.s_axi_wlast(g_wlast),.s_axi_wvalid(g_wvalid),.s_axi_wready(g_wready),
        .s_axi_bid(g_bid),.s_axi_bresp(g_bresp),.s_axi_bvalid(g_bvalid),.s_axi_bready(g_bready),
        .m_axi_awaddr(d_awaddr),.m_axi_awlen(d_awlen),.m_axi_awsize(d_awsize),.m_axi_awprot(d_awprot),
        .m_axi_awburst(d_awburst),.m_axi_awcache(d_awcache),.m_axi_awlock(d_awlock),
        .m_axi_awregion(d_awregion),.m_axi_awqos(d_awqos),.m_axi_awvalid(d_awvalid),.m_axi_awready(d_awready),
        .m_axi_wdata(d_wdata),.m_axi_wstrb(d_wstrb),.m_axi_wlast(d_wlast),.m_axi_wvalid(d_wvalid),.m_axi_wready(d_wready),
        .m_axi_bresp(d_bresp),.m_axi_bvalid(d_bvalid),.m_axi_bready(d_bready)
    );
    hbm_protocol protocol_converter (
        .aclk(clk),.aresetn(rst_n),
        .s_axi_awaddr(d_awaddr),.s_axi_awlen(d_awlen),.s_axi_awsize(d_awsize),.s_axi_awprot(d_awprot),
        .s_axi_awburst(d_awburst),.s_axi_awid(4'b0),.s_axi_awcache(d_awcache),.s_axi_awlock(d_awlock),
        .s_axi_awregion(d_awregion),.s_axi_awqos(d_awqos),.s_axi_awvalid(d_awvalid),.s_axi_awready(d_awready),
        .s_axi_wdata(d_wdata),.s_axi_wstrb(d_wstrb),.s_axi_wlast(d_wlast),.s_axi_wvalid(d_wvalid),.s_axi_wready(d_wready),
        .s_axi_bid(),.s_axi_bresp(d_bresp),.s_axi_bvalid(d_bvalid),.s_axi_bready(d_bready),
        .m_axi_awaddr(m_axi_awaddr),.m_axi_awlen(m_axi_awlen),.m_axi_awsize(m_axi_awsize),.m_axi_awprot(),
        .m_axi_awburst(m_axi_awburst),.m_axi_awid(p_awid),.m_axi_awcache(),.m_axi_awlock(),.m_axi_awqos(),
        .m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),
        .m_axi_wid(p_wid),.m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),.m_axi_wlast(m_axi_wlast),.m_axi_wvalid(m_axi_wvalid),.m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid[3:0]),.m_axi_bresp(m_axi_bresp),.m_axi_bvalid(m_axi_bvalid&&!bad_response),.m_axi_bready(p_bready)
    );
endmodule
