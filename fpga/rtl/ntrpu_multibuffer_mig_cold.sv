// First-controller bring-up: original two-destination R4 core, destination 0
// connected to MIG. Destination 1 is reserved/inactive, never emulated as RAM.
// Readback is an independent UI-domain test/host interface outside this block.
module ntrpu_multibuffer_mig_cold (
    input wire mgmt_clk, source_clk, ui_clk,
    input wire external_reset_async, rearm, calibrated, ui_rst, stop,
    output wire source_reset_n, memory_reset_n, admit, running, rearm_ready,
    output wire restart_required, peers_reset_req, host_alive, device_alive,
    input wire cmd_valid, data_valid,
    output wire cmd_ready, data_ready,
    input wire [63:0] cmd_addr, cmd_tag,
    input wire [7:0] cmd_len,
    input wire [511:0] data,
    output wire cpl_valid,
    input wire cpl_ready,
    output wire [63:0] cpl_tag,
    output wire [1:0] cpl_status,
    output wire fault, drained, inactive_port_idle,
    output wire [7:0] banks_used,
    output wire [9:0] pending_counts,
    output wire m_axi_awvalid, m_axi_wvalid, m_axi_wlast, m_axi_bready,
    input wire m_axi_awready, m_axi_wready, m_axi_bvalid,
    output wire [36:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [3:0] m_axi_awid, m_axi_awcache,
    output wire [2:0] m_axi_awsize, m_axi_awprot,
    output wire [1:0] m_axi_awburst,
    output wire m_axi_awlock,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_rvalid
);
    wire [1:0] awvalid, wvalid, wlast, bready, completed;
    wire [73:0] awaddr;
    wire [15:0] awlen;
    wire [7:0] awid, awcache;
    wire [5:0] awsize;
    wire [3:0] awburst, status;
    wire [1023:0] wdata;
    wire [127:0] wstrb, tags;
    wire core_cmd_ready, core_drained;
    wire source_clean = banks_used == 0 && pending_counts == 0 && completed == 0;
    ntrpu_mig_cold_start startup (.*,
        .source_clean(source_clean), .memory_bvalid(m_axi_bvalid), .memory_rvalid(m_axi_rvalid));
    assign cmd_ready = admit && core_cmd_ready;
    assign drained = admit && core_drained;
    assign inactive_port_idle = !(awvalid[1] || wvalid[1] || completed[1]) && pending_counts[9:5] == 0;
    assign m_axi_awvalid=awvalid[0], m_axi_awaddr=awaddr[36:0], m_axi_awlen=awlen[7:0];
    assign m_axi_awid=awid[3:0], m_axi_awcache=awcache[3:0], m_axi_awsize=awsize[2:0];
    assign m_axi_awburst=awburst[1:0], m_axi_awlock=1'b0, m_axi_awprot=3'b0;
    assign m_axi_wvalid=wvalid[0], m_axi_wlast=wlast[0], m_axi_wdata=wdata[511:0], m_axi_wstrb=wstrb[63:0];
    assign m_axi_bready=bready[0], cpl_valid=completed[0], cpl_tag=tags[63:0], cpl_status=status[1:0];
    ntrpu_peer_multibuffer_credits #(.GUARD_SLOTS(4)) receiver (
        .clk(source_clk), .rst_n(source_reset_n), .mem_clk({2{ui_clk}}), .mem_rst_n({2{memory_reset_n}}),
        .stop(stop || !admit), .cmd_valid(cmd_valid && admit), .cmd_ready(core_cmd_ready), .cmd_channel(1'b0),
        .cmd_addr(cmd_addr), .cmd_tag(cmd_tag), .cmd_len(cmd_len), .data_valid(data_valid), .data_ready(data_ready), .data(data),
        .m_awvalid(awvalid), .m_awready({1'b0,m_axi_awready}), .m_awaddr(awaddr), .m_awlen(awlen),
        .m_awid(awid), .m_awcache(awcache), .m_awsize(awsize), .m_awburst(awburst),
        .m_wvalid(wvalid), .m_wready({1'b0,m_axi_wready}), .m_wlast(wlast), .m_wdata(wdata), .m_wstrb(wstrb),
        .m_bvalid({1'b0,m_axi_bvalid}), .m_bready(bready), .m_bid({4'b0,m_axi_bid}), .m_bresp({2'b0,m_axi_bresp}),
        .cpl_valid(completed), .cpl_ready({1'b1,cpl_ready}), .cpl_tag(tags), .cpl_status(status),
        .fault(fault), .drained(core_drained), .banks_used(banks_used), .pending_counts(pending_counts)
    );
endmodule
