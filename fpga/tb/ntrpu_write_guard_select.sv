`timescale 1ns/1ps
// Measurement adapter; GUARD_SLOTS=0 preserves the original guard.
module ntrpu_write_guard_select #(parameter integer MEMORY_ADDR_WIDTH=37, GUARD_SLOTS=4) (
input wire clk, rst_n, admit,
    output wire idle,
    input wire [63:0] s_axi_awaddr,
    input wire [7:0] s_axi_awlen,
    input wire [2:0] s_axi_awsize, s_axi_awprot,
    input wire [1:0] s_axi_awburst,
    input wire [3:0] s_axi_awid, s_axi_awcache,
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
    output wire [MEMORY_ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize, m_axi_awprot,
    output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awid, m_axi_awcache,
    output wire m_axi_awlock, m_axi_awvalid,
    input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast, m_axi_wvalid,
    input wire m_axi_wready,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    generate if(GUARD_SLOTS==0) begin: legacy
        ntrpu_axi_capacity_guard #(.MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH)) selected (
            .s_axi_araddr(64'b0), .s_axi_arlen(8'b0), .s_axi_arsize(3'b0), .s_axi_arburst(2'b01),
            .s_axi_arid(4'b0), .s_axi_arcache(4'b0), .s_axi_arprot(3'b0), .s_axi_arlock(1'b0),
            .s_axi_arvalid(1'b0), .s_axi_arready(), .s_axi_rdata(), .s_axi_rid(),
            .s_axi_rresp(), .s_axi_rlast(), .s_axi_rvalid(), .s_axi_rready(1'b1),
            .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(), .m_axi_arid(),
            .m_axi_arcache(), .m_axi_arprot(), .m_axi_arlock(), .m_axi_arvalid(), .m_axi_arready(1'b0),
            .m_axi_rdata(512'b0), .m_axi_rid(4'b0), .m_axi_rresp(2'b0),
            .m_axi_rlast(1'b0), .m_axi_rvalid(1'b0), .m_axi_rready(), .*
        );
    end else begin: queued
        ntrpu_axi_write_capacity_queue #(.MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH), .DEPTH(GUARD_SLOTS)) selected (.*);
    end endgenerate
endmodule
