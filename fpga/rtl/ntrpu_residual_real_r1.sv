// Actual generated residual HLS; caller retains both input regions through ACK.
`default_nettype none
module ntrpu_residual_real_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_source_sequence,command_source_epoch,
    input wire logic command_source_valid,
    input wire logic [63:0] command_input_address,command_input_base,command_input_bytes,
    input wire logic [63:0] command_branch_address,command_branch_base,command_branch_bytes,
    input wire logic [63:0] command_residual_address,command_residual_base,command_residual_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [31:0] completion_hls_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [9:0] completion_read_beats,
    output logic [7:0] completion_residual_beats,
    output logic result_valid,
    output logic [63:0] result_residual_address,
    output logic [13:0] result_residual_bytes,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,m_axi_arcache,
    output logic m_axi_arlock,m_axi_arvalid,
    input wire logic m_axi_arready,
    input wire logic [511:0] m_axi_rdata,
    input wire logic [1:0] m_axi_rresp,
    input wire logic [3:0] m_axi_rid,
    input wire logic m_axi_rlast,m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize,m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid,m_axi_awcache,
    output logic m_axi_awlock,m_axi_awvalid,
    input wire logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,m_axi_wvalid,
    input wire logic m_axi_wready,
    input wire logic [3:0] m_axi_bid,
    input wire logic [1:0] m_axi_bresp,
    input wire logic m_axi_bvalid,
    output logic m_axi_bready
);
    wire hls_start;
    wire hls_done,hls_ready;
    wire [31:0] hls_return;
    wire [31:0] hls_elements;
    wire [1:0] hls_input_ce;
    wire [10:0] hls_input_address [0:1];
    wire [31:0] hls_input_q [0:1];
    wire hls_residual_ce,hls_residual_we;
    wire [10:0] hls_residual_address;
    wire [31:0] hls_residual_d;

    ntrpu_residual_path_r1 #(
        .AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)
    ) path(.*);
    ntrpu_residual_hls_ports_r1 arithmetic(.*);
endmodule
`default_nettype wire
