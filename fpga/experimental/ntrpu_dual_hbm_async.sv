`timescale 1ns/1ps
// Experimental dual command/payload receiver + qualified CDC/HBM adapters.
// 16 GiB address limit; no PCIe/registered-region authorization in this wrapper.
// Isolated asynchronous composition. Cold reset only; source and memory clocks.
// Stop closes publication, never the adapters needed to drain committed work.
// Malformed B is quarantined by its adapter. Only coordinated cold reset.
module ntrpu_dual_hbm_async #(parameter integer PREFER_LOCAL=1) (
    input wire  clk,
    input wire  cold_reset_n,
    input wire  mem_clk,
    input wire [1:0] init_complete_async,
    output wire rst_n,mem_rst_n,started,
    output wire [11:0] inflight,
    input wire  stop,
    input wire  fault_in,
    input wire [1:0] cmd_valid,
    output wire [1:0] cmd_ready,reserve_can,
    input wire [127:0] cmd_addr,cmd_tag,
    input wire [15:0] cmd_len,
    input wire [1:0] data_valid,
    output wire [1:0] data_ready,
    input wire [1023:0] data,
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
    wire transport_fault,transport_drained,startup_fault;
    wire [1:0] core_ready,core_reserve;
    wire core_fence_ready,core_fence_done,core_fence_ack;
    reg external_fault,fence_offered;
    wire [1:0] ledger_fault;
    wire bad_address=started && ((cmd_valid[0] && |cmd_addr[63:34]) ||
                                               (cmd_valid[1] && |cmd_addr[127:98]));
    wire publication=started && !fault && !stop;
    ntrpu_hbm_startup startup(.clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_reset_n),
        .init_complete_async(init_complete_async),.adapter_fault_mem(adapter_fault),
        .src_rst_n(rst_n),.mem_rst_n(mem_rst_n),.started(started),.fault(startup_fault));
    always @(posedge clk) begin
        if(!rst_n)external_fault<=0;
        else if(fault_in || bad_address)external_fault<=1;
    end
    // Once advertised, a completed fence remains stable even if a later fault
    // closes the epoch. Faults before completion quarantine the pending fence.
    always @(posedge clk) begin
        if(!rst_n)fence_offered<=0;
        else if(fence_done_valid && fence_done_ready)fence_offered<=0;
        else if(fence_done_valid)fence_offered<=1;
    end
    assign cmd_ready=core_ready & {2{publication}};
    assign reserve_can=core_reserve & {2{publication}};
    assign fence_ready=core_fence_ready && publication;
    assign fault=transport_fault || startup_fault || external_fault || fault_in || bad_address || |ledger_fault;
    // "drained" means quiescent, not successful; consumers must also inspect fault.
    assign drained=started && transport_drained && inflight==0;
    assign fence_done_valid=core_fence_done && inflight==0 && (!fault || fence_offered);
    assign core_fence_ack=fence_done_ready && inflight==0 && (!fault || fence_offered);
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
    ntrpu_dual_ingress_locality #(.PREFER_LOCAL(PREFER_LOCAL)) receiver (
        .clk(clk),.rst_n(rst_n),.stop(stop || startup_fault || external_fault || fault_in || bad_address || |ledger_fault),
        .fault(transport_fault),.drained(transport_drained),
        .cmd_valid(cmd_valid & {2{publication}}),.cmd_ready(core_ready),.reserve_can(core_reserve),
        .cmd_addr(cmd_addr),.cmd_tag(cmd_tag),.cmd_len(cmd_len),
        .data_valid(data_valid),.data_ready(data_ready),.data(data),
        .fence_valid(fence_valid && publication),.fence_ready(core_fence_ready),
        .fence_id(fence_id),.fence_done_valid(core_fence_done),.fence_done_ready(core_fence_ack),.fence_done_id(fence_done_id),
        .cpl_valid(cpl_valid),.cpl_ready(cpl_ready),.cpl_tag(cpl_tag),.cpl_status(cpl_status),
        .banks_used(banks_used),.pending_counts(pending_counts),
        .m_awvalid(t_awvalid),.m_awready(t_awready),.m_awaddr(t_awaddr),.m_awlen(t_awlen),
        .m_awid(t_awid),.m_awcache(t_awcache),.m_awsize(t_awsize),.m_awburst(t_awburst),
        .m_wvalid(t_wvalid),.m_wready(t_wready),.m_wlast(t_wlast),.m_wdata(t_wdata),.m_wstrb(t_wstrb),
        .m_bready(t_bready),.m_bvalid(t_bvalid),.m_bid(t_bid),.m_bresp(t_bresp));
    wire [1:0] c_awvalid;
    wire [1:0] c_awready;
    wire [127:0] c_awaddr;
    wire [15:0] c_awlen;
    wire [7:0] c_awid;
    wire [7:0] c_awcache;
    wire [5:0] c_awsize;
    wire [3:0] c_awburst;
    wire [1:0] c_wvalid;
    wire [1:0] c_wready;
    wire [1:0] c_wlast;
    wire [1023:0] c_wdata;
    wire [127:0] c_wstrb;
    wire [1:0] c_bready;
    wire [1:0] c_bvalid;
    wire [7:0] c_bid;
    wire [3:0] c_bresp;
    for(genvar lane=0;lane<2;lane++)begin: memory
        // Include AW buffered inside CDC, not only the four adapter credits.
        reg [5:0] outstanding;
        reg counter_fault;
        assign ledger_fault[lane]=counter_fault;
        always @(posedge clk) begin
            if(!rst_n)counter_fault<=0;
            else if((replied && outstanding==0 && !accepted) ||
                    (accepted && outstanding==4 && !replied))counter_fault<=1;
        end
        wire accepted=t_awvalid[lane]&&t_awready[lane];
        wire replied=t_bvalid[lane]&&t_bready[lane];
        assign inflight[lane*6+:6]=outstanding;
        always @(posedge clk)begin
            if(!rst_n)outstanding<=0;
            else case({accepted,replied})
                2'b10:outstanding<=outstanding+1'b1;
                2'b01:outstanding<=outstanding-1'b1;
                default:begin end
            endcase
        end
        ntrpu_hbm_write_adapter adapter (
            .clk(mem_clk),.rst_n(mem_rst_n),.admit(mem_rst_n),
            .idle(adapter_idle[lane]),.fault(adapter_fault[lane]),
            .s_axi_awprot(3'b0),.s_axi_awlock(1'b0),
            .s_axi_awvalid(c_awvalid[lane]),
            .s_axi_awready(c_awready[lane]),
            .s_axi_awaddr(c_awaddr[lane*64+:64]),
            .s_axi_awlen(c_awlen[lane*8+:8]),
            .s_axi_awid(c_awid[lane*4+:4]),
            .s_axi_awcache(c_awcache[lane*4+:4]),
            .s_axi_awsize(c_awsize[lane*3+:3]),
            .s_axi_awburst(c_awburst[lane*2+:2]),
            .s_axi_wvalid(c_wvalid[lane]),
            .s_axi_wready(c_wready[lane]),
            .s_axi_wlast(c_wlast[lane]),
            .s_axi_wdata(c_wdata[lane*512+:512]),
            .s_axi_wstrb(c_wstrb[lane*64+:64]),
            .s_axi_bready(c_bready[lane]),
            .s_axi_bvalid(c_bvalid[lane]),
            .s_axi_bid(c_bid[lane*4+:4]),
            .s_axi_bresp(c_bresp[lane*2+:2]),
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
        hbm_cdc converter (
.s_axi_aclk(clk),.s_axi_aresetn(rst_n),.m_axi_aclk(mem_clk),.m_axi_aresetn(mem_rst_n),
.s_axi_awprot(3'b0),.s_axi_awlock(1'b0),.s_axi_awqos(4'b0),.s_axi_awregion(4'b0),
.m_axi_awprot(),.m_axi_awlock(),.m_axi_awqos(),.m_axi_awregion(),
.s_axi_awvalid(t_awvalid[lane]),.m_axi_awvalid(c_awvalid[lane]),
.s_axi_awready(t_awready[lane]),.m_axi_awready(c_awready[lane]),
.s_axi_awaddr(t_awaddr[lane*64+:64]),.m_axi_awaddr(c_awaddr[lane*64+:64]),
.s_axi_awlen(t_awlen[lane*8+:8]),.m_axi_awlen(c_awlen[lane*8+:8]),
.s_axi_awid(t_awid[lane*4+:4]),.m_axi_awid(c_awid[lane*4+:4]),
.s_axi_awcache(t_awcache[lane*4+:4]),.m_axi_awcache(c_awcache[lane*4+:4]),
.s_axi_awsize(t_awsize[lane*3+:3]),.m_axi_awsize(c_awsize[lane*3+:3]),
.s_axi_awburst(t_awburst[lane*2+:2]),.m_axi_awburst(c_awburst[lane*2+:2]),
.s_axi_wvalid(t_wvalid[lane]),.m_axi_wvalid(c_wvalid[lane]),
.s_axi_wready(t_wready[lane]),.m_axi_wready(c_wready[lane]),
.s_axi_wlast(t_wlast[lane]),.m_axi_wlast(c_wlast[lane]),
.s_axi_wdata(t_wdata[lane*512+:512]),.m_axi_wdata(c_wdata[lane*512+:512]),
.s_axi_wstrb(t_wstrb[lane*64+:64]),.m_axi_wstrb(c_wstrb[lane*64+:64]),
.s_axi_bready(t_bready[lane]),.m_axi_bready(c_bready[lane]),
.s_axi_bvalid(t_bvalid[lane]),.m_axi_bvalid(c_bvalid[lane]),
.s_axi_bid(t_bid[lane*4+:4]),.m_axi_bid(c_bid[lane*4+:4]),
.s_axi_bresp(t_bresp[lane*2+:2]),.m_axi_bresp(c_bresp[lane*2+:2])
        );
    end
endmodule
