// Full matrix descriptor frontend for the validated memory-to-memory GEMV engine.
// Completion publishes the whole matrix; any failure/reset invalidates its destination.
module ntrpu_q8_matrix_memory #(parameter integer WATCHDOG_CYCLES=65536,parameter bit FP32_CAPABLE=0) (

    input logic clk, rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [1:0] job_layout,
    input logic [13:0] job_rows,
    input logic job_output_fp32,
    input logic [8:0] job_k_blocks,
    input logic [5:0] job_tile_k,
    input logic [31:0] job_sequence, job_epoch,
    input logic [63:0] job_weight_addr, job_region_base, job_region_bytes,
    input logic [63:0] job_activation_addr, job_activation_region_base, job_activation_region_bytes,
    input logic [31:0] job_activation_epoch,
    input logic [63:0] job_output_addr, job_output_region_base, job_output_region_bytes,
    input logic [31:0] job_output_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
    output logic [16:0] done_bytes,
    output logic [13:0] done_rows,
    output logic [31:0] done_sequence, done_epoch,
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
    logic e_job_valid,e_job_output_fp32;
    logic e_job_ready;
    logic [1:0] e_job_layout;
    logic [5:0] e_job_rows;
    logic [8:0] e_job_k_blocks;
    logic [31:0] e_job_sequence, e_job_epoch;
    logic [63:0] e_job_region_base, e_job_region_bytes;
    logic [63:0] e_job_activation_addr, e_job_activation_region_base, e_job_activation_region_bytes;
    logic [31:0] e_job_activation_epoch;
    logic [63:0] e_job_output_addr, e_job_output_region_base, e_job_output_region_bytes;
    logic [31:0] e_job_output_epoch;
    logic e_tile_valid;
    logic e_tile_ready;
    logic [63:0] e_tile_addr;
    logic [8:0] e_tile_k_base;
    logic [5:0] e_tile_k_count;
    logic [31:0] e_tile_tag, e_tile_epoch;
    logic e_cancel_valid;
    logic e_cancel_ready;
    logic e_done_valid;
    logic e_done_ready;
    logic [3:0] e_done_status;
    logic [8:0] e_done_bytes;
    logic [31:0] e_done_sequence, e_done_epoch;
    ntrpu_matrix_sequencer #(.FP32_CAPABLE(FP32_CAPABLE)) controller(.*);
    ntrpu_q8_gemv_store #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.FP32_CAPABLE(FP32_CAPABLE)) engine (
        .clk,
        .rst_n,
        .job_valid(e_job_valid),
        .job_ready(e_job_ready),
        .job_layout(e_job_layout),
        .job_rows(e_job_rows),
        .job_output_fp32(e_job_output_fp32),
        .job_k_blocks(e_job_k_blocks),
        .job_sequence(e_job_sequence),
        .job_epoch(e_job_epoch),
        .job_region_base(e_job_region_base),
        .job_region_bytes(e_job_region_bytes),
        .job_activation_addr(e_job_activation_addr),
        .job_activation_region_base(e_job_activation_region_base),
        .job_activation_region_bytes(e_job_activation_region_bytes),
        .job_activation_epoch(e_job_activation_epoch),
        .job_output_addr(e_job_output_addr),
        .job_output_region_base(e_job_output_region_base),
        .job_output_region_bytes(e_job_output_region_bytes),
        .job_output_epoch(e_job_output_epoch),
        .tile_valid(e_tile_valid),
        .tile_ready(e_tile_ready),
        .tile_addr(e_tile_addr),
        .tile_k_base(e_tile_k_base),
        .tile_k_count(e_tile_k_count),
        .tile_tag(e_tile_tag),
        .tile_epoch(e_tile_epoch),
        .cancel_valid(e_cancel_valid),
        .cancel_ready(e_cancel_ready),
        .done_valid(e_done_valid),
        .done_ready(e_done_ready),
        .done_status(e_done_status),
        .done_bytes(e_done_bytes),
        .done_sequence(e_done_sequence),
        .done_epoch(e_done_epoch),
        .m_axi_araddr,
        .m_axi_arlen,
        .m_axi_arsize,
        .m_axi_arprot,
        .m_axi_arburst,
        .m_axi_arid,
        .m_axi_arcache,
        .m_axi_arlock,
        .m_axi_arvalid,
        .m_axi_arready,
        .m_axi_rdata,
        .m_axi_rresp,
        .m_axi_rid,
        .m_axi_rlast,
        .m_axi_rvalid,
        .m_axi_rready,
        .m_axi_awaddr,
        .m_axi_awlen,
        .m_axi_awsize,
        .m_axi_awprot,
        .m_axi_awburst,
        .m_axi_awid,
        .m_axi_awcache,
        .m_axi_awlock,
        .m_axi_awvalid,
        .m_axi_awready,
        .m_axi_wdata,
        .m_axi_wstrb,
        .m_axi_wlast,
        .m_axi_wvalid,
        .m_axi_wready,
        .m_axi_bid,
        .m_axi_bresp,
        .m_axi_bvalid,
        .m_axi_bready
    );
endmodule
