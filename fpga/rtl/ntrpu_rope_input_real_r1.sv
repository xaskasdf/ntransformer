// Original RoPE HLS over real AXI loading and retained local Q/K output.
`default_nettype none
module ntrpu_rope_input_real_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_source_sequence,command_source_epoch,
    input wire logic command_source_valid,
    input wire logic [31:0] command_position,command_source_position,command_coefficient_position,
    input wire logic [63:0] command_q_address,command_q_base,command_q_bytes,
    input wire logic [63:0] command_k_address,command_k_base,command_k_bytes,
    input wire logic [63:0] command_coefficient_address,command_coefficient_base,command_coefficient_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [9:0] completion_beats,
    output logic [31:0] completion_position,completion_coefficient_position,
    output logic result_valid,
    output logic [31:0] completion_hls_status,
    output logic completion_revoked,
    input wire logic result_ce,
    input wire logic [11:0] result_address,
    output logic [31:0] result_q,
    output logic result_q_valid,
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
    output logic m_axi_rready
);
    wire hls_start;
    wire hls_done,hls_ready;
    wire [31:0] hls_return;
    wire [31:0] hls_heads,hls_position,hls_coefficient_position;
    wire [1:0] hls_input_ce;
    wire [11:0] hls_input_address [0:1];
    wire [31:0] hls_input_q [0:1];
    wire hls_output_ce,hls_output_we;
    wire [11:0] hls_output_address;
    wire [31:0] hls_output_d;
    ntrpu_rope_input_path_r2 #(
        .AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)
    ) path(.*);
    ntrpu_rope_hls_ports_r1 arithmetic(.*);
endmodule
`default_nettype wire
