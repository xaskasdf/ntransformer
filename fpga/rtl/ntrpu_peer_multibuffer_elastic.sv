`timescale 1ns / 1ps

// Isolated AXI write boundary for fillstage: two slots each for AW, W and B.
// Custom command/data/completion ports are unchanged; this is not a CDC bridge.
// A bank returns after its last INTERNAL W transfer (payload now queue-owned).
// Fault/admission reflect B processing by the core, not arrival at this wrapper.
// Reset aborts buffered traffic; the attached target must share the reset epoch.
module ntrpu_peer_multibuffer_elastic #(
    parameter integer PACKED_RAM = 1,
    parameter integer CHANNELS = 2,
    parameter integer RESERVE_BANKS = 4,
    parameter integer BANK_BEATS = 64,
    parameter integer DESCRIPTORS = 16,
    parameter integer MAX_OUTSTANDING = 4,
    parameter integer BANKS = 2 * CHANNELS + RESERVE_BANKS,
    parameter integer BANK_BITS = BANKS <= 1 ? 1 : $clog2(BANKS),
    parameter integer CHANNEL_BITS = CHANNELS <= 1 ? 1 : $clog2(CHANNELS),
    parameter integer COUNT_BITS = $clog2(DESCRIPTORS + 1)
) (
    input wire clk, rst_n, stop,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [CHANNEL_BITS-1:0] cmd_channel,
    input wire [63:0] cmd_addr, cmd_tag,
    input wire [7:0] cmd_len,
    input wire data_valid,
    output wire data_ready,
    input wire [511:0] data,
    output wire [CHANNELS-1:0] m_awvalid,
    input wire [CHANNELS-1:0] m_awready,
    output wire [CHANNELS*64-1:0] m_awaddr,
    output wire [CHANNELS*8-1:0] m_awlen,
    output wire [CHANNELS*4-1:0] m_awid, m_awcache,
    output wire [CHANNELS*3-1:0] m_awsize,
    output wire [CHANNELS*2-1:0] m_awburst,
    output wire [CHANNELS-1:0] m_wvalid, m_wlast,
    input wire [CHANNELS-1:0] m_wready,
    output wire [CHANNELS*512-1:0] m_wdata,
    output wire [CHANNELS*64-1:0] m_wstrb,
    input wire [CHANNELS-1:0] m_bvalid,
    output wire [CHANNELS-1:0] m_bready,
    input wire [CHANNELS*4-1:0] m_bid,
    input wire [CHANNELS*2-1:0] m_bresp,
    output wire [CHANNELS-1:0] cpl_valid,
    input wire [CHANNELS-1:0] cpl_ready,
    output wire [CHANNELS*64-1:0] cpl_tag,
    output wire [CHANNELS*2-1:0] cpl_status,
    output wire fault,
    output wire drained,
    output wire [BANKS-1:0] banks_used,
    output wire [CHANNELS*COUNT_BITS-1:0] pending_counts
);
    wire [CHANNELS-1:0] i_awvalid;
    wire [CHANNELS-1:0] i_awready;
    wire [CHANNELS*64-1:0] i_awaddr;
    wire [CHANNELS*8-1:0] i_awlen;
    wire [CHANNELS*4-1:0] i_awid, i_awcache;
    wire [CHANNELS*3-1:0] i_awsize;
    wire [CHANNELS*2-1:0] i_awburst;
    wire [CHANNELS-1:0] i_wvalid, i_wlast;
    wire [CHANNELS-1:0] i_wready;
    wire [CHANNELS*512-1:0] i_wdata;
    wire [CHANNELS*64-1:0] i_wstrb;
    wire [CHANNELS-1:0] i_bvalid;
    wire [CHANNELS-1:0] i_bready;
    wire [CHANNELS*4-1:0] i_bid;
    wire [CHANNELS*2-1:0] i_bresp;

    wire core_drained;
    wire [CHANNELS-1:0] aw_empty, w_empty, b_empty;
    assign drained = core_drained && (&aw_empty) && (&w_empty) && (&b_empty);
    ntrpu_peer_multibuffer_fillstage #(
        .PACKED_RAM(PACKED_RAM), .CHANNELS(CHANNELS), .RESERVE_BANKS(RESERVE_BANKS),
        .BANK_BEATS(BANK_BEATS), .DESCRIPTORS(DESCRIPTORS), .MAX_OUTSTANDING(MAX_OUTSTANDING)
    ) core (
        .m_awvalid(i_awvalid),
        .m_awready(i_awready),
        .m_awaddr(i_awaddr),
        .m_awlen(i_awlen),
        .m_awid(i_awid),
        .m_awcache(i_awcache),
        .m_awsize(i_awsize),
        .m_awburst(i_awburst),
        .m_wvalid(i_wvalid),
        .m_wlast(i_wlast),
        .m_wready(i_wready),
        .m_wdata(i_wdata),
        .m_wstrb(i_wstrb),
        .m_bvalid(i_bvalid),
        .m_bready(i_bready),
        .m_bid(i_bid),
        .m_bresp(i_bresp),
        .drained(core_drained), .*
    );

    genvar g;
    generate for (g=0; g<CHANNELS; g=g+1) begin: boundary
        ntrpu_rv_elastic #(.WIDTH(85)) aw (
            .clk(clk), .rst_n(rst_n), .s_valid(i_awvalid[g]), .s_ready(i_awready[g]),
            .s_data({i_awaddr[g*64 +:64], i_awlen[g*8 +:8], i_awid[g*4 +:4],
                     i_awcache[g*4 +:4], i_awsize[g*3 +:3], i_awburst[g*2 +:2]}),
            .m_valid(m_awvalid[g]), .m_ready(m_awready[g]),
            .m_data({m_awaddr[g*64 +:64], m_awlen[g*8 +:8], m_awid[g*4 +:4],
                     m_awcache[g*4 +:4], m_awsize[g*3 +:3], m_awburst[g*2 +:2]}),
            .empty(aw_empty[g])
        );
        ntrpu_rv_elastic #(.WIDTH(577)) w (
            .clk(clk), .rst_n(rst_n), .s_valid(i_wvalid[g]), .s_ready(i_wready[g]),
            .s_data({i_wlast[g], i_wstrb[g*64 +:64], i_wdata[g*512 +:512]}),
            .m_valid(m_wvalid[g]), .m_ready(m_wready[g]),
            .m_data({m_wlast[g], m_wstrb[g*64 +:64], m_wdata[g*512 +:512]}),
            .empty(w_empty[g])
        );
        ntrpu_rv_elastic #(.WIDTH(6)) b (
            .clk(clk), .rst_n(rst_n), .s_valid(m_bvalid[g]), .s_ready(m_bready[g]),
            .s_data({m_bid[g*4 +:4], m_bresp[g*2 +:2]}),
            .m_valid(i_bvalid[g]), .m_ready(i_bready[g]),
            .m_data({i_bid[g*4 +:4], i_bresp[g*2 +:2]}), .empty(b_empty[g])
        );
    end endgenerate
endmodule
