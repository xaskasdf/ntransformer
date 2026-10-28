// Shared R4 pool, two real memory destinations, one coordinated cold epoch.
module ntrpu_dual_multibuffer_mig (
    input wire mgmt_clk, source_clk, external_reset_async, rearm, stop,
    input wire [1:0] ui_clk, calibrated, ui_rst,
    output wire source_reset_n, admit, running, rearm_ready, restart_required, peers_reset_req, source_alive,
    output wire [1:0] memory_reset_n, memory_alive,
    input wire cmd_valid, cmd_channel, data_valid,
    output wire cmd_ready, data_ready,
    input wire [63:0] cmd_addr, cmd_tag,
    input wire [7:0] cmd_len,
    input wire [511:0] data,
    output wire [1:0] cpl_valid,
    input wire [1:0] cpl_ready,
    output wire [127:0] cpl_tag,
    output wire [3:0] cpl_status,
    output wire fault, drained,
    output wire [7:0] banks_used,
    output wire [9:0] pending_counts,
    output wire [1:0] m_awvalid, m_wvalid, m_wlast, m_bready,
    input wire [1:0] m_awready, m_wready, m_bvalid, m_rvalid,
    output wire [73:0] m_awaddr,
    output wire [15:0] m_awlen,
    output wire [7:0] m_awid, m_awcache,
    output wire [5:0] m_awsize,
    output wire [3:0] m_awburst,
    output wire [1023:0] m_wdata,
    output wire [127:0] m_wstrb,
    input wire [7:0] m_bid,
    input wire [3:0] m_bresp
);
    wire source_clean = banks_used == 0 && pending_counts == 0 && cpl_valid == 0;
    wire core_cmd_ready, core_drained;
    ntrpu_dual_mig_cold_start startup (.*, .memory_bvalid(m_bvalid), .memory_rvalid(m_rvalid));
    assign cmd_ready = admit && core_cmd_ready;
    assign drained = admit && core_drained;
    ntrpu_peer_multibuffer_credits #(.GUARD_SLOTS(4)) receiver (
        .clk(source_clk), .rst_n(source_reset_n), .mem_clk(ui_clk), .mem_rst_n(memory_reset_n),
        .stop(stop || !admit), .cmd_valid(cmd_valid && admit), .cmd_ready(core_cmd_ready),
        .drained(core_drained), .*
    );
endmodule
