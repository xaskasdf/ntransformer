`timescale 1ns/1ps
// One cold configuration, one exclusive line lease, real dual-MIG supervisor.
// Pool runtime reset invalidates but never clears retained bridge ownership.
// Recovery requires complete FPGA reconfiguration; no warm bridge rearm.
module ntrpu_fragment_dual_mig #(
    parameter [63:0] FIRST_COOKIE = 64'd1
) (
    input wire mgmt_clk, source_clk, external_reset_async, rearm, stop,
    input wire [1:0] ui_clk, calibrated, ui_rst,
    input wire [1:0] m_rvalid,
    output wire source_reset_n, admit, running, rearm_ready, restart_required,
    output wire peers_reset_req, source_alive, bridge_reset_n, restart_required_source,
    output wire [1:0] memory_reset_n, memory_alive,
    input wire arm_valid,
    output wire arm_ready,
    input wire [63:0] arm_epoch, arm_tag, arm_destination,
    input wire arm_channel,
    output wire [63:0] epoch,
    output wire lease_held,
    input wire retire_valid,
    output wire retire_ready,
    input wire [63:0] retire_epoch, retire_tag,
    input wire line_valid,
    output wire line_ready,
    input wire [63:0] line_epoch, line_tag, line_address,
    input wire [511:0] line_data,
    output wire done_valid,
    input wire done_ready,
    output wire [63:0] done_epoch, done_tag,
    output wire [3:0] done_status,
    output wire [1:0] done_pool_status,
    output wire protocol_fault, session_lost, exhausted, session_invalid, idle,
    output wire [2:0] phase,
    output wire [63:0] next_cookie,
    output wire [31:0] rejected_completions,
    output wire pool_fault, pool_drained,
    output wire [7:0] banks_used,
    output wire [9:0] pending_counts,
    output wire [1:0] m_awvalid, m_wvalid, m_wlast, m_bready,
    input wire [1:0] m_awready, m_wready, m_bvalid,
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
    wire cmd_valid, cmd_ready, cmd_channel, data_valid, data_ready;
    wire [63:0] cmd_addr, cmd_tag;
    wire [7:0] cmd_len;
    wire [511:0] data;
    wire [1:0] cpl_valid, cpl_ready;
    wire [127:0] cpl_tag;
    wire [3:0] cpl_status;
    // FPGA configuration is the only way to clear this lifetime bit. Runtime
    // pool reset must not erase the bridge's invalid lease/cookie ledger.
    logic bridge_started = 1'b0;
    always_ff @(posedge source_clk)
        if (source_reset_n) bridge_started <= 1'b1;
    assign bridge_reset_n = bridge_started;
    // The supervisor's management-domain status is not a source-domain input.
    // admit already closes locally/asynchronously; this synchronized diagnostic
    // additionally prevents admission after a fault before admit was observed.
    xpm_cdc_single #(.DEST_SYNC_FF(3), .INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1), .SRC_INPUT_REG(1)) restart_cdc (
        .src_clk(mgmt_clk), .src_in(restart_required),
        .dest_clk(source_clk), .dest_out(restart_required_source));
    wire pool_stop = stop && !lease_held;
    ntrpu_fragment_pool_bridge #(.FIRST_COOKIE(FIRST_COOKIE)) bridge (
        .clk(source_clk), .rst_n(bridge_reset_n), .pool_admit(admit),
        .pool_restart_required(restart_required_source), .*
    );
    ntrpu_dual_multibuffer_mig receiver (
        .stop(pool_stop), .fault(pool_fault), .drained(pool_drained), .*
    );
endmodule
