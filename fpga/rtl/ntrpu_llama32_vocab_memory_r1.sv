// Full vocabulary using one exclusive original FP32 matrix datapath.
// Public/source identity stays 64-bit; matrix identity comes from reserved tags.
// A shared reset must quiesce the source and AXI peer as well as this module.
`default_nettype none
module ntrpu_llama32_vocab_memory_r1 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter integer CHILD_TIMEOUT_CYCLES=16777216,
    parameter logic [63:0] INITIAL_CHILD_TAG=64'd0
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire [63:0] command_addr [0:2],
    input wire [63:0] command_base [0:2],
    input wire [63:0] command_bytes [0:2],
    input wire cancel_valid,
    output wire cancel_ready,busy,
    input wire source_completion_valid,
    output wire source_completion_ready,
    input wire [3:0] source_completion_status,
    input wire [63:0] source_sequence,source_epoch,source_result_address,
    input wire source_result_valid,
    input wire [12:0] source_result_bytes,
    output wire completion_valid,
    input wire completion_ready,
    output wire [3:0] completion_status,completion_source_status,completion_child_status,
    output wire [63:0] completion_sequence,completion_epoch,
    output wire [3:0] completion_phase,
    output wire [4:0] completion_matrices,
    output wire [18:0] completion_total_bytes,
    output wire [16:0] completion_child_bytes,
    output wire [13:0] completion_child_rows,
    output wire completion_revoked,result_valid,tags_exhausted,
    output wire [63:0] result_address,
    output wire [18:0] result_bytes,
    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize, m_axi_arprot,
    output wire [1:0] m_axi_arburst,
    output wire [3:0] m_axi_arid, m_axi_arcache,
    output wire m_axi_arlock, m_axi_arvalid,
    input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire [3:0] m_axi_rid,
    input wire m_axi_rlast, m_axi_rvalid,
    output wire m_axi_rready,
    output wire [63:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize, m_axi_awprot,
    output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awid, m_axi_awcache,
    output wire m_axi_awlock, m_axi_awvalid,
    input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast, m_axi_wvalid,
    input wire m_axi_wready,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    wire child_job_valid;
    wire child_job_ready;
    wire [1:0] child_job_layout;
    wire [13:0] child_job_rows;
    wire child_job_output_fp32;
    wire [8:0] child_job_k_blocks;
    wire [5:0] child_job_tile_k;
    wire [31:0] child_job_sequence,child_job_epoch;
    wire [63:0] child_job_weight_addr,child_job_region_base,child_job_region_bytes;
    wire [63:0] child_job_activation_addr,child_job_activation_region_base,child_job_activation_region_bytes;
    wire [31:0] child_job_activation_epoch;
    wire [63:0] child_job_output_addr,child_job_output_region_base,child_job_output_region_bytes;
    wire [31:0] child_job_output_epoch;
    wire child_cancel_valid;
    wire child_cancel_ready;
    wire child_done_valid;
    wire child_done_ready;
    wire [3:0] child_done_status;
    wire [16:0] child_done_bytes;
    wire [13:0] child_done_rows;
    wire [31:0] child_done_sequence,child_done_epoch;
    ntrpu_llama32_vocab_control_r1 #(
        .INITIAL_TAG(INITIAL_CHILD_TAG),.CHILD_TIMEOUT_CYCLES(CHILD_TIMEOUT_CYCLES)
    ) control(.*);
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
