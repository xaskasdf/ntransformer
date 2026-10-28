// Original generated RMSNorm/Q8 HLS; input allocations retained through public ACK.
`default_nettype none
module ntrpu_input_norm_real_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter logic [63:0] INITIAL_CHILD_TAG=64'd0
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_source_sequence,command_source_epoch,
    input wire logic command_source_valid,
    input wire logic [63:0] command_input_address,command_input_base,command_input_bytes,
    input wire logic [63:0] command_weight_address,command_weight_base,command_weight_bytes,
    input wire logic [63:0] command_output_address,command_output_base,command_output_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,tag_exhausted,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [31:0] completion_hls_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [9:0] completion_read_beats,
    output logic [6:0] completion_records,
    output logic result_valid,completion_revoked,
    output logic [63:0] result_address,
    output logic [12:0] result_bytes,
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
    wire [31:0] hls_elements,hls_epsilon_bits;
    wire [1:0] hls_input_ce;
    wire [10:0] hls_input_address [0:1];
    wire [31:0] hls_input_q [0:1];
    wire hls_output_ce,hls_output_we;
    wire [8:0] hls_output_address;
    wire [63:0] hls_output_d;

    ntrpu_input_norm_path_r1 #(
        .AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES),
        .INITIAL_CHILD_TAG(INITIAL_CHILD_TAG)
    ) path(.*);
    ntrpu_input_norm_hls_ports_r1 arithmetic(.*);
endmodule
`default_nettype wire
