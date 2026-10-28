`timescale 1ns/1ps
// One owner, one line, exclusive two-channel pool. No host ABI is implied.
// rst_n and mem_rst_n must belong to a coordinated cold transport session.
// pool_admit/restart_required come from an external cold-session supervisor.
module ntrpu_fragment_pool_transport #(
    parameter [63:0] FIRST_COOKIE = 64'd1
) (
    input wire clk, rst_n, stop,
    input wire [1:0] mem_clk, mem_rst_n,
    input wire pool_admit, pool_restart_required,
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
    // STOP cannot close the command sink while an irrevocable command is held.
    // ARM is already blocked by the bridge during STOP. Only the pool's new
    // allocations are stopped between leases; accepted data/B/CPL keep draining.
    wire pool_stop = stop && !lease_held;
    ntrpu_fragment_pool_bridge #(.FIRST_COOKIE(FIRST_COOKIE)) bridge (.*);
    ntrpu_peer_multibuffer_credits #(.GUARD_SLOTS(4)) pool (
        .stop(pool_stop), .fault(pool_fault), .drained(pool_drained), .*
    );
endmodule
