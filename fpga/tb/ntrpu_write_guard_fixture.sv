`timescale 1ns/1ps
// Register fixture for common 200 MHz OOC measurement, not a board top.
module ntrpu_write_guard_fixture #(parameter integer MEMORY_ADDR_WIDTH=37, GUARD_SLOTS=4) (
input wire clk, rst_n, admit,
    output reg idle,
    input wire [63:0] s_axi_awaddr,
    input wire [7:0] s_axi_awlen,
    input wire [2:0] s_axi_awsize, s_axi_awprot,
    input wire [1:0] s_axi_awburst,
    input wire [3:0] s_axi_awid, s_axi_awcache,
    input wire s_axi_awlock, s_axi_awvalid,
    output reg s_axi_awready,
    input wire [511:0] s_axi_wdata,
    input wire [63:0] s_axi_wstrb,
    input wire s_axi_wlast, s_axi_wvalid,
    output reg s_axi_wready,
    output reg [3:0] s_axi_bid,
    output reg [1:0] s_axi_bresp,
    output reg s_axi_bvalid,
    input wire s_axi_bready,
    output reg [MEMORY_ADDR_WIDTH-1:0] m_axi_awaddr,
    output reg [7:0] m_axi_awlen,
    output reg [2:0] m_axi_awsize, m_axi_awprot,
    output reg [1:0] m_axi_awburst,
    output reg [3:0] m_axi_awid, m_axi_awcache,
    output reg m_axi_awlock, m_axi_awvalid,
    input wire m_axi_awready,
    output reg [511:0] m_axi_wdata,
    output reg [63:0] m_axi_wstrb,
    output reg m_axi_wlast, m_axi_wvalid,
    input wire m_axi_wready,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output reg m_axi_bready
);
    wire core_clk;
    BUFG clock_buffer(.I(clk),.O(core_clk));
    reg  q_rst_n;
    reg  q_admit;
    wire  q_idle;
    reg [63:0] q_s_axi_awaddr;
    reg [7:0] q_s_axi_awlen;
    reg [2:0] q_s_axi_awsize;
    reg [2:0] q_s_axi_awprot;
    reg [1:0] q_s_axi_awburst;
    reg [3:0] q_s_axi_awid;
    reg [3:0] q_s_axi_awcache;
    reg  q_s_axi_awlock;
    reg  q_s_axi_awvalid;
    wire  q_s_axi_awready;
    reg [511:0] q_s_axi_wdata;
    reg [63:0] q_s_axi_wstrb;
    reg  q_s_axi_wlast;
    reg  q_s_axi_wvalid;
    wire  q_s_axi_wready;
    wire [3:0] q_s_axi_bid;
    wire [1:0] q_s_axi_bresp;
    wire  q_s_axi_bvalid;
    reg  q_s_axi_bready;
    wire [MEMORY_ADDR_WIDTH-1:0] q_m_axi_awaddr;
    wire [7:0] q_m_axi_awlen;
    wire [2:0] q_m_axi_awsize;
    wire [2:0] q_m_axi_awprot;
    wire [1:0] q_m_axi_awburst;
    wire [3:0] q_m_axi_awid;
    wire [3:0] q_m_axi_awcache;
    wire  q_m_axi_awlock;
    wire  q_m_axi_awvalid;
    reg  q_m_axi_awready;
    wire [511:0] q_m_axi_wdata;
    wire [63:0] q_m_axi_wstrb;
    wire  q_m_axi_wlast;
    wire  q_m_axi_wvalid;
    reg  q_m_axi_wready;
    reg [3:0] q_m_axi_bid;
    reg [1:0] q_m_axi_bresp;
    reg  q_m_axi_bvalid;
    wire  q_m_axi_bready;
    always @(posedge core_clk) begin
        q_rst_n <= rst_n;
        q_admit <= admit;
        idle <= q_idle;
        q_s_axi_awaddr <= s_axi_awaddr;
        q_s_axi_awlen <= s_axi_awlen;
        q_s_axi_awsize <= s_axi_awsize;
        q_s_axi_awprot <= s_axi_awprot;
        q_s_axi_awburst <= s_axi_awburst;
        q_s_axi_awid <= s_axi_awid;
        q_s_axi_awcache <= s_axi_awcache;
        q_s_axi_awlock <= s_axi_awlock;
        q_s_axi_awvalid <= s_axi_awvalid;
        s_axi_awready <= q_s_axi_awready;
        q_s_axi_wdata <= s_axi_wdata;
        q_s_axi_wstrb <= s_axi_wstrb;
        q_s_axi_wlast <= s_axi_wlast;
        q_s_axi_wvalid <= s_axi_wvalid;
        s_axi_wready <= q_s_axi_wready;
        s_axi_bid <= q_s_axi_bid;
        s_axi_bresp <= q_s_axi_bresp;
        s_axi_bvalid <= q_s_axi_bvalid;
        q_s_axi_bready <= s_axi_bready;
        m_axi_awaddr <= q_m_axi_awaddr;
        m_axi_awlen <= q_m_axi_awlen;
        m_axi_awsize <= q_m_axi_awsize;
        m_axi_awprot <= q_m_axi_awprot;
        m_axi_awburst <= q_m_axi_awburst;
        m_axi_awid <= q_m_axi_awid;
        m_axi_awcache <= q_m_axi_awcache;
        m_axi_awlock <= q_m_axi_awlock;
        m_axi_awvalid <= q_m_axi_awvalid;
        q_m_axi_awready <= m_axi_awready;
        m_axi_wdata <= q_m_axi_wdata;
        m_axi_wstrb <= q_m_axi_wstrb;
        m_axi_wlast <= q_m_axi_wlast;
        m_axi_wvalid <= q_m_axi_wvalid;
        q_m_axi_wready <= m_axi_wready;
        q_m_axi_bid <= m_axi_bid;
        q_m_axi_bresp <= m_axi_bresp;
        q_m_axi_bvalid <= m_axi_bvalid;
        m_axi_bready <= q_m_axi_bready;
    end
    (* keep_hierarchy="yes" *) ntrpu_write_guard_select #(.MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH), .GUARD_SLOTS(GUARD_SLOTS)) guard (
        .clk(core_clk),
        .rst_n(q_rst_n),
        .admit(q_admit),
        .idle(q_idle),
        .s_axi_awaddr(q_s_axi_awaddr),
        .s_axi_awlen(q_s_axi_awlen),
        .s_axi_awsize(q_s_axi_awsize),
        .s_axi_awprot(q_s_axi_awprot),
        .s_axi_awburst(q_s_axi_awburst),
        .s_axi_awid(q_s_axi_awid),
        .s_axi_awcache(q_s_axi_awcache),
        .s_axi_awlock(q_s_axi_awlock),
        .s_axi_awvalid(q_s_axi_awvalid),
        .s_axi_awready(q_s_axi_awready),
        .s_axi_wdata(q_s_axi_wdata),
        .s_axi_wstrb(q_s_axi_wstrb),
        .s_axi_wlast(q_s_axi_wlast),
        .s_axi_wvalid(q_s_axi_wvalid),
        .s_axi_wready(q_s_axi_wready),
        .s_axi_bid(q_s_axi_bid),
        .s_axi_bresp(q_s_axi_bresp),
        .s_axi_bvalid(q_s_axi_bvalid),
        .s_axi_bready(q_s_axi_bready),
        .m_axi_awaddr(q_m_axi_awaddr),
        .m_axi_awlen(q_m_axi_awlen),
        .m_axi_awsize(q_m_axi_awsize),
        .m_axi_awprot(q_m_axi_awprot),
        .m_axi_awburst(q_m_axi_awburst),
        .m_axi_awid(q_m_axi_awid),
        .m_axi_awcache(q_m_axi_awcache),
        .m_axi_awlock(q_m_axi_awlock),
        .m_axi_awvalid(q_m_axi_awvalid),
        .m_axi_awready(q_m_axi_awready),
        .m_axi_wdata(q_m_axi_wdata),
        .m_axi_wstrb(q_m_axi_wstrb),
        .m_axi_wlast(q_m_axi_wlast),
        .m_axi_wvalid(q_m_axi_wvalid),
        .m_axi_wready(q_m_axi_wready),
        .m_axi_bid(q_m_axi_bid),
        .m_axi_bresp(q_m_axi_bresp),
        .m_axi_bvalid(q_m_axi_bvalid),
        .m_axi_bready(q_m_axi_bready)
    );
endmodule
