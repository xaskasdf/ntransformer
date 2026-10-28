`timescale 1ns/1ps
// Isolated same-clock composition. Upstream descriptors are already authorized.
// Stop closes publication, never the adapters needed to drain committed work.
// Malformed B is quarantined by its adapter. Only coordinated cold reset.
module ntrpu_hbm_transport (
    input wire  clk,
    input wire  rst_n,
    input wire  stop,
    input wire  fault_in,
    input wire [1:0] submit_valid,
    output wire [1:0] submit_ready,
    input wire [463:0] submit_desc,
    output wire  fetch_valid,
    output wire  fetch_channel,
    input wire  fetch_ready,
    output wire [231:0] fetch_desc,
    input wire  payload_valid,
    output wire  payload_ready,
    input wire [511:0] payload_data,
    input wire  fence_valid,
    output wire  fence_ready,
    input wire [63:0] fence_id,
    output wire  fence_done_valid,
    input wire  fence_done_ready,
    output wire [63:0] fence_done_id,
    output wire [1:0] cpl_valid,
    input wire [1:0] cpl_ready,
    output wire [127:0] cpl_tag,
    output wire [3:0] cpl_status,
    output wire  fault,
    output wire  drained,
    output wire [7:0] banks_used,
    output wire [7:0] queued,
    output wire [9:0] pending_counts,
    output wire [1:0] h_awvalid,
    output wire [1:0] h_wvalid,
    output wire [1:0] h_wlast,
    output wire [1:0] h_bready,
    input wire [1:0] h_awready,
    input wire [1:0] h_wready,
    input wire [1:0] h_bvalid,
    output wire [67:0] h_awaddr,
    output wire [63:0] h_wstrb,
    output wire [7:0] h_awlen,
    output wire [11:0] h_awid,
    output wire [5:0] h_awsize,
    output wire [3:0] h_awburst,
    output wire [511:0] h_wdata,
    input wire [11:0] h_bid,
    input wire [3:0] h_bresp,
    output wire [1:0] adapter_idle,
    output wire [1:0] adapter_fault
);
    wire transport_fault,transport_drained;
    assign fault=transport_fault || (|adapter_fault);
    assign drained=transport_drained && (&adapter_idle);
    wire [1:0] t_awvalid;
    wire [1:0] t_awready;
    wire [127:0] t_awaddr;
    wire [15:0] t_awlen;
    wire [7:0] t_awid;
    wire [7:0] t_awcache;
    wire [5:0] t_awsize;
    wire [3:0] t_awburst;
    wire [1:0] t_wvalid;
    wire [1:0] t_wready;
    wire [1:0] t_wlast;
    wire [1023:0] t_wdata;
    wire [127:0] t_wstrb;
    wire [1:0] t_bready;
    wire [1:0] t_bvalid;
    wire [7:0] t_bid;
    wire [3:0] t_bresp;
    ntrpu_candidate_transport transport (
        .fault(transport_fault),.drained(transport_drained),.fault_in(fault_in || (|adapter_fault)),
.m_awvalid(t_awvalid),.m_awready(t_awready),.m_awaddr(t_awaddr),.m_awlen(t_awlen),.m_awid(t_awid),.m_awcache(t_awcache),.m_awsize(t_awsize),.m_awburst(t_awburst),.m_wvalid(t_wvalid),.m_wready(t_wready),.m_wlast(t_wlast),.m_wdata(t_wdata),.m_wstrb(t_wstrb),.m_bready(t_bready),.m_bvalid(t_bvalid),.m_bid(t_bid),.m_bresp(t_bresp),.*);
    for(genvar lane=0;lane<2;lane++)begin: memory
        ntrpu_hbm_write_adapter adapter (
            .clk(clk),.rst_n(rst_n),.admit(rst_n),
            .idle(adapter_idle[lane]),.fault(adapter_fault[lane]),
            .s_axi_awprot(3'b0),.s_axi_awlock(1'b0),
            .s_axi_awvalid(t_awvalid[lane]),
            .s_axi_awready(t_awready[lane]),
            .s_axi_awaddr(t_awaddr[lane*64+:64]),
            .s_axi_awlen(t_awlen[lane*8+:8]),
            .s_axi_awid(t_awid[lane*4+:4]),
            .s_axi_awcache(t_awcache[lane*4+:4]),
            .s_axi_awsize(t_awsize[lane*3+:3]),
            .s_axi_awburst(t_awburst[lane*2+:2]),
            .s_axi_wvalid(t_wvalid[lane]),
            .s_axi_wready(t_wready[lane]),
            .s_axi_wlast(t_wlast[lane]),
            .s_axi_wdata(t_wdata[lane*512+:512]),
            .s_axi_wstrb(t_wstrb[lane*64+:64]),
            .s_axi_bready(t_bready[lane]),
            .s_axi_bvalid(t_bvalid[lane]),
            .s_axi_bid(t_bid[lane*4+:4]),
            .s_axi_bresp(t_bresp[lane*2+:2]),
            .m_axi_awvalid(h_awvalid[lane]),
            .m_axi_awready(h_awready[lane]),
            .m_axi_awaddr(h_awaddr[lane*34+:34]),
            .m_axi_awlen(h_awlen[lane*4+:4]),
            .m_axi_awid(h_awid[lane*6+:6]),
            .m_axi_awsize(h_awsize[lane*3+:3]),
            .m_axi_awburst(h_awburst[lane*2+:2]),
            .m_axi_wvalid(h_wvalid[lane]),
            .m_axi_wready(h_wready[lane]),
            .m_axi_wlast(h_wlast[lane]),
            .m_axi_wdata(h_wdata[lane*256+:256]),
            .m_axi_wstrb(h_wstrb[lane*32+:32]),
            .m_axi_bready(h_bready[lane]),
            .m_axi_bvalid(h_bvalid[lane]),
            .m_axi_bid(h_bid[lane*6+:6]),
            .m_axi_bresp(h_bresp[lane*2+:2])
        );
    end
endmodule
