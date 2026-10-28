`timescale 1ns/1ps
// Isolated write-side composition. No application-reset gating in the drain path.
// drained is valid only within a live coordinated transport/reset epoch.
module ntrpu_peer_multibuffer_memory #(
    parameter integer MEMORY_ADDR_WIDTH = 37,
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
    input wire [CHANNELS-1:0] mem_clk, mem_rst_n,
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
    output wire [CHANNELS*MEMORY_ADDR_WIDTH-1:0] m_awaddr,
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
    wire [CHANNELS-1:0] u_awvalid;
    wire [CHANNELS-1:0] u_awready;
    wire [CHANNELS*64-1:0] u_awaddr;
    wire [CHANNELS*8-1:0] u_awlen;
    wire [CHANNELS*4-1:0] u_awid;
    wire [CHANNELS*4-1:0] u_awcache;
    wire [CHANNELS*3-1:0] u_awsize;
    wire [CHANNELS*2-1:0] u_awburst;
    wire [CHANNELS-1:0] u_wvalid;
    wire [CHANNELS-1:0] u_wlast;
    wire [CHANNELS-1:0] u_wready;
    wire [CHANNELS*512-1:0] u_wdata;
    wire [CHANNELS*64-1:0] u_wstrb;
    wire [CHANNELS-1:0] u_bvalid;
    wire [CHANNELS-1:0] u_bready;
    wire [CHANNELS*4-1:0] u_bid;
    wire [CHANNELS*2-1:0] u_bresp;
    ntrpu_peer_multibuffer_elastic #(.PACKED_RAM(PACKED_RAM), .CHANNELS(CHANNELS),
        .RESERVE_BANKS(RESERVE_BANKS), .BANK_BEATS(BANK_BEATS),
        .DESCRIPTORS(DESCRIPTORS), .MAX_OUTSTANDING(MAX_OUTSTANDING)) receiver (
        .m_awvalid(u_awvalid),
        .m_awready(u_awready),
        .m_awaddr(u_awaddr),
        .m_awlen(u_awlen),
        .m_awid(u_awid),
        .m_awcache(u_awcache),
        .m_awsize(u_awsize),
        .m_awburst(u_awburst),
        .m_wvalid(u_wvalid),
        .m_wlast(u_wlast),
        .m_wready(u_wready),
        .m_wdata(u_wdata),
        .m_wstrb(u_wstrb),
        .m_bvalid(u_bvalid),
        .m_bready(u_bready),
        .m_bid(u_bid),
        .m_bresp(u_bresp),
        .*
    );
    genvar g;
    generate for(g=0;g<CHANNELS;g=g+1) begin: memory_port
        ntrpu_peer_memory_port #(.MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH)) port (
            .clk(clk), .rst_n(rst_n), .mem_clk(mem_clk[g]), .mem_rst_n(mem_rst_n[g]),
            .s_awvalid(u_awvalid[g]),
            .s_awready(u_awready[g]),
            .s_awaddr(u_awaddr[g*64 +:64]),
            .s_awlen(u_awlen[g*8 +:8]),
            .s_awid(u_awid[g*4 +:4]),
            .s_awcache(u_awcache[g*4 +:4]),
            .s_awsize(u_awsize[g*3 +:3]),
            .s_awburst(u_awburst[g*2 +:2]),
            .s_wvalid(u_wvalid[g]),
            .s_wlast(u_wlast[g]),
            .s_wready(u_wready[g]),
            .s_wdata(u_wdata[g*512 +:512]),
            .s_wstrb(u_wstrb[g*64 +:64]),
            .s_bvalid(u_bvalid[g]),
            .s_bready(u_bready[g]),
            .s_bid(u_bid[g*4 +:4]),
            .s_bresp(u_bresp[g*2 +:2]),
            .m_awvalid(m_awvalid[g]),
            .m_awready(m_awready[g]),
            .m_awaddr(m_awaddr[g*MEMORY_ADDR_WIDTH +:MEMORY_ADDR_WIDTH]),
            .m_awlen(m_awlen[g*8 +:8]),
            .m_awid(m_awid[g*4 +:4]),
            .m_awcache(m_awcache[g*4 +:4]),
            .m_awsize(m_awsize[g*3 +:3]),
            .m_awburst(m_awburst[g*2 +:2]),
            .m_wvalid(m_wvalid[g]),
            .m_wlast(m_wlast[g]),
            .m_wready(m_wready[g]),
            .m_wdata(m_wdata[g*512 +:512]),
            .m_wstrb(m_wstrb[g*64 +:64]),
            .m_bvalid(m_bvalid[g]),
            .m_bready(m_bready[g]),
            .m_bid(m_bid[g*4 +:4]),
            .m_bresp(m_bresp[g*2 +:2])
        );
    end endgenerate
endmodule
