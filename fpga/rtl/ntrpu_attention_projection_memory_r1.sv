// Real 2048x2048 O projection: full parent identity, dedicated matrix engine.
// The caller owns immutable weights/activations until completion is consumed.
// Shared reset must also quiesce the external AXI peer.
`default_nettype none
module ntrpu_attention_projection_memory_r1 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter logic [63:0] INITIAL_CHILD_TAG=64'd0
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_weight_addr,command_weight_region_base,command_weight_region_bytes,
    input wire logic [63:0] command_activation_addr,command_activation_region_base,command_activation_region_bytes,
    input wire logic [63:0] command_output_addr,command_output_region_base,command_output_region_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic result_valid,
    output logic [63:0] result_address,
    output logic [16:0] result_bytes,
    output logic tag_exhausted,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize, m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid, m_axi_arcache,
    output logic m_axi_arlock, m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize, m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid, m_axi_awcache,
    output logic m_axi_awlock, m_axi_awvalid,
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast, m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready
);
    logic child_job_valid;
    logic child_job_ready;
    logic [1:0] child_job_layout;
    logic [13:0] child_job_rows;
    logic child_job_output_fp32;
    logic [8:0] child_job_k_blocks;
    logic [5:0] child_job_tile_k;
    logic [31:0] child_job_sequence,child_job_epoch;
    logic [63:0] child_job_weight_addr,child_job_region_base,child_job_region_bytes;
    logic [63:0] child_job_activation_addr,child_job_activation_region_base,child_job_activation_region_bytes;
    logic [31:0] child_job_activation_epoch;
    logic [63:0] child_job_output_addr,child_job_output_region_base,child_job_output_region_bytes;
    logic [31:0] child_job_output_epoch;
    logic child_cancel_valid;
    logic child_cancel_ready,child_activity;
    logic child_done_valid;
    logic child_done_ready;
    logic [3:0] child_done_status;
    logic [16:0] child_done_bytes;
    logic [13:0] child_done_rows;
    logic [31:0] child_done_sequence,child_done_epoch;
    assign child_activity=(m_axi_arvalid && m_axi_arready) || (m_axi_rvalid && m_axi_rready) ||
        (m_axi_awvalid && m_axi_awready) || (m_axi_wvalid && m_axi_wready) || (m_axi_bvalid && m_axi_bready);
    ntrpu_attention_projection_control_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.INITIAL_CHILD_TAG(INITIAL_CHILD_TAG)) control(.*);
    ntrpu_q8_matrix_memory #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.FP32_CAPABLE(1)) matrix(
        .clk(clk),
        .rst_n(rst_n),
        .job_valid(child_job_valid),
        .job_ready(child_job_ready),
        .job_layout(child_job_layout),
        .job_rows(child_job_rows),
        .job_output_fp32(child_job_output_fp32),
        .job_k_blocks(child_job_k_blocks),
        .job_tile_k(child_job_tile_k),
        .job_sequence(child_job_sequence),
        .job_epoch(child_job_epoch),
        .job_weight_addr(child_job_weight_addr),
        .job_region_base(child_job_region_base),
        .job_region_bytes(child_job_region_bytes),
        .job_activation_addr(child_job_activation_addr),
        .job_activation_region_base(child_job_activation_region_base),
        .job_activation_region_bytes(child_job_activation_region_bytes),
        .job_activation_epoch(child_job_activation_epoch),
        .job_output_addr(child_job_output_addr),
        .job_output_region_base(child_job_output_region_base),
        .job_output_region_bytes(child_job_output_region_bytes),
        .job_output_epoch(child_job_output_epoch),
        .cancel_valid(child_cancel_valid),
        .cancel_ready(child_cancel_ready),
        .done_valid(child_done_valid),
        .done_ready(child_done_ready),
        .done_status(child_done_status),
        .done_bytes(child_done_bytes),
        .done_rows(child_done_rows),
        .done_sequence(child_done_sequence),
        .done_epoch(child_done_epoch),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arid(m_axi_arid),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arlock(m_axi_arlock),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awid(m_axi_awid),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready)
    );
endmodule
`default_nettype wire
