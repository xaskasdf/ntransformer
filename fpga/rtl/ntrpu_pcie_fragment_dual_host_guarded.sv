`timescale 1ns / 1ps
// Retained NTRD + automatic host guard + actual shared pool/CDC/cold supervisor.
// Host AXI ports use host_clk. The two memory AXI ports attach to MIG.
// Native endpoint and board-level reset coordination remain external contracts.
// Never connect host_reset_n/transport_reset_n to memory configuration lifetime.
// This top requires FPGA reconfiguration after any loss of runtime admission.
module ntrpu_pcie_fragment_dual_host_guarded #(parameter [63:0] FIRST_COOKIE=64'd1) (

    input wire host_clk, link_up_async, endpoint_reset_async,
    input wire guard_arm, guard_peers_held, guard_peers_ready,
    output wire host_reset_n, transport_reset_n, host_reset_req,
    output wire host_admit, source_admit, host_alive, guard_source_alive,
    output wire guard_invalidate, loss_mgmt, waiting_ack, guard_running,
    output wire [4:0] loss_causes,
    output wire [63:0] first_loss_tick,
    output wire [3:0] guard_state,
    output wire owner_invalid_ack, boundary_invalid,
    output wire [7:0] boundary_fenced,
    output wire [15:0] boundary_pending, boundary_fault,
    output wire [511:0] boundary_accepted, boundary_delivered, boundary_abandoned,

    input wire mgmt_clk, source_clk, external_reset_async, rearm,
    input wire [1:0] ui_clk, calibrated, ui_rst,
    input wire [1:0] m_rvalid,
    output wire source_reset_n, admit, running, rearm_ready, restart_required,
    output wire peers_reset_req, source_alive, lifetime_reset_n, restart_required_source,
    output wire [1:0] memory_reset_n, memory_alive,
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
    input wire [3:0] m_bresp,



    output wire stop,
    input wire source_stopped, ingress_isolated,
    input wire [63:0] quiesce_epoch,
    output wire arm_valid,
    output wire arm_ready,
    output wire [63:0] arm_epoch, arm_tag, arm_window, arm_destination,
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

    output wire arm_channel,
    output wire ack_valid,
    output wire ack_ready,
    output wire [63:0] ack_epoch, ack_tag,
    output wire pool_admit, pool_restart_required, pool_fault, pool_drained,
    output wire cmd_valid,
    output wire cmd_ready,
    output wire cmd_channel,
    output wire [63:0] cmd_addr, cmd_tag,
    output wire [7:0] cmd_len,
    output wire data_valid,
    output wire data_ready,
    output wire [511:0] data,
    output wire [1:0] cpl_valid,
    output wire [1:0] cpl_ready,
    output wire [127:0] cpl_tag,
    output wire [3:0] cpl_status,
    output wire bridge_lease_held,
    output wire [63:0] bridge_epoch,
    output wire bridge_idle, bridge_arm_ready, collector_arm_ready, retire_ready,
    output wire [2:0] bridge_phase,
    output wire [3:0] done_status,
    output wire [1:0] done_pool_status,
    output wire protocol_fault, session_lost, session_invalid, exhausted,
    output wire [63:0] next_cookie,
    output wire [31:0] rejected_completions,
    input wire [31:0] s_axil_awaddr, s_axil_wdata, s_axil_araddr,
    input wire [3:0] s_axil_wstrb,
    input wire s_axil_awvalid, s_axil_wvalid, s_axil_bready,
    input wire s_axil_arvalid, s_axil_rready,
    output wire s_axil_awready, s_axil_wready, s_axil_arready,
    output wire s_axil_bvalid, s_axil_rvalid,
    output wire [1:0] s_axil_bresp, s_axil_rresp,
    output wire [31:0] s_axil_rdata,
    output wire irq, control_fault
);

    // Configuration owns this bit. Runtime pool resets close admission but
    // cannot erase control/collector/bridge state or accepted host AXI traffic.
    logic lifetime_started = 1'b0;
    always_ff @(posedge source_clk)
        if (source_reset_n) lifetime_started <= 1'b1;
    assign lifetime_reset_n = lifetime_started;
    xpm_cdc_single #(.DEST_SYNC_FF(3), .INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1), .SRC_INPUT_REG(1)) restart_cdc (
        .src_clk(mgmt_clk), .src_in(restart_required),
        .dest_clk(source_clk), .dest_out(restart_required_source));
    assign pool_admit = admit;
    assign pool_restart_required = restart_required_source;
    // Collector lease_held is retained history after ACK; only bridge ownership
    // describes an active pool reservation. Never strand an exposed command.
    wire pool_stop = stop && !bridge_lease_held;
    ntrpu_pcie_fragment_guarded_host #(.FIRST_COOKIE(FIRST_COOKIE)) host (
        .cold_rst_n(lifetime_reset_n), .source_alive(guard_source_alive), .*);
    ntrpu_dual_multibuffer_mig receiver (
        .stop(pool_stop), .fault(pool_fault), .drained(pool_drained), .*);
endmodule
