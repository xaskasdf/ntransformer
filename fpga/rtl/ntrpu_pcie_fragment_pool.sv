`timescale 1ns / 1ps
// One NTRPUF01 collector + one exclusive dual-pool reservation.
// Trusted descriptor/ACK interface; no MMIO ABI, PCIe IP or CDC is implied.
// rst_n must be the shared cold configuration lifetime, NEVER pool runtime reset.
// Barrier inputs are external trusted proofs, never inferred from local drain.
module ntrpu_pcie_fragment_pool #(parameter [63:0] FIRST_COOKIE=64'd1) (

    input wire clk, rst_n, stop,
    input wire source_stopped, ingress_isolated,
    input wire [63:0] quiesce_epoch,
    input wire arm_valid,
    output wire arm_ready,
    input wire [63:0] arm_epoch, arm_tag, arm_window, arm_destination,
    output wire [63:0] epoch,
    output wire lease_held, failed,
    output wire [3:0] error_code, // Lease errors:1stop,4conflict,5destination,6late write; frame counters also2format,3range,7identity
    output wire transport_idle,
    output wire [63:0] valid_bytes,
    output wire [31:0] rejected_frames, rejected_identity,
    output wire [3:0] last_reject_code,
    output wire [6:0] last_reject_faults,

    input wire [63:0] s_axi_awaddr,
    input wire [3:0] s_axi_awid,
    input wire [7:0] s_axi_awlen,
    input wire [2:0] s_axi_awsize,
    input wire [1:0] s_axi_awburst,
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

    output wire line_valid,
    output wire line_ready,
    output wire [63:0] line_epoch, line_tag, line_address,
    output wire [511:0] line_data,
    output wire done_valid,
    output wire done_ready,
    output wire [63:0] done_epoch, done_tag,
    output wire done_error,
    output wire result_valid,
    output wire result_ready,
    output wire [63:0] result_epoch, result_tag,
    output wire [3:0] result_status,

    input wire arm_channel,
    input wire ack_valid,
    output wire ack_ready,
    input wire [63:0] ack_epoch, ack_tag,
    input wire pool_admit, pool_restart_required, pool_fault, pool_drained,
    output wire cmd_valid,
    input wire cmd_ready,
    output wire cmd_channel,
    output wire [63:0] cmd_addr, cmd_tag,
    output wire [7:0] cmd_len,
    output wire data_valid,
    input wire data_ready,
    output wire [511:0] data,
    input wire [1:0] cpl_valid,
    output wire [1:0] cpl_ready,
    input wire [127:0] cpl_tag,
    input wire [3:0] cpl_status,
    output wire bridge_lease_held,
    output wire [63:0] bridge_epoch,
    output wire bridge_idle, bridge_arm_ready, collector_arm_ready, retire_ready,
    output wire [2:0] bridge_phase,
    output wire [3:0] done_status,
    output wire [1:0] done_pool_status,
    output wire protocol_fault, session_lost, session_invalid, exhausted,
    output wire [63:0] next_cookie,
    output wire [31:0] rejected_completions
);

    // Both modules own one descriptor from the same edge. Neither can take
    // a reservation while the other is blocked (including address validation).
    wire joint_arm = arm_valid && arm_ready;
    assign arm_ready = collector_arm_ready && bridge_arm_ready;
    wire collector_stop = stop || session_invalid;
    // A local conflict can finish without a LINE. Cancel only the active
    // bridge lease: collector.failed stays set until a future joint ARM.
    wire bridge_stop = stop || (failed && bridge_lease_held);
    // A visible result is diagnostic, not proof of a still-valid session.
    // Recheck the external barrier and pending ingress at the ACK edge.
    wire ack_identity = ack_epoch == result_epoch && ack_tag == result_tag;
    wire ack_barrier = source_stopped && ingress_isolated && quiesce_epoch == epoch;
    wire ack_transport = transport_idle && !s_axi_awvalid && !s_axi_wvalid;
    assign ack_ready = rst_n && result_valid && ack_identity && ack_barrier &&
                       ack_transport && retire_ready && !session_invalid;
    assign result_ready = ack_valid && ack_ready;
    // STOPPED is an application cancellation, not a destination error.
    assign done_error = done_status != 0 && done_status != 1;
    ntrpu_pcie_fragment_ingress collector (
        .stop(collector_stop), .arm_valid(joint_arm), .arm_ready(collector_arm_ready), .*);
    ntrpu_fragment_pool_bridge #(.FIRST_COOKIE(FIRST_COOKIE)) bridge (
        .stop(bridge_stop), .arm_valid(joint_arm), .arm_ready(bridge_arm_ready),
        .epoch(bridge_epoch), .lease_held(bridge_lease_held),
        .retire_valid(result_ready), .retire_epoch(ack_epoch), .retire_tag(ack_tag),
        .idle(bridge_idle), .phase(bridge_phase), .*);
endmodule
