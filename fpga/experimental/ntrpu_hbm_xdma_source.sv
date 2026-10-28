`timescale 1ns/1ps
`default_nettype none
// Isolated physical-lease composition. Connect reader_stop to the reader stop port.
// reader_started/fault are in clk; only coordinated cold reset may drop ownership.
module ntrpu_hbm_xdma_source #(
    parameter integer GUARD_WAIT_LIMIT=65536, MASTER_WAIT_LIMIT=65536
) (
    input wire clk,dma_clk,cold_reset_n,stop,
    output wire started,rst_n,dma_rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input ntrpu_peer_read_pkg::lease_t cfg,
    input wire [63:0] cfg_writeback,
    output wire cfg_error,
    input wire submit_valid,submit_channel,
    output wire submit_ready,
    input ntrpu_peer_read_pkg::request_t submit_context,
    output wire reject_valid,
    input wire reject_ready,
    output wire [3:0] reject_code,
    input wire reader_started,reader_fault,
    output wire reader_stop,source_fault,admission_pending,
    output wire [3:0] source_held,
    output wire [1:0] cmd_valid,
    input wire [1:0] cmd_ready,
    output wire [127:0] cmd_addr,cmd_tag,
    output wire [15:0] cmd_len,
    input wire [1:0] r_data_valid,r_data_last,
    output wire [1:0] r_data_ready,
    input wire [1023:0] r_data,
    input wire [127:0] r_data_tag,
    input wire [1:0] r_cpl_valid,
    output wire [1:0] r_cpl_ready,
    input wire [127:0] r_cpl_tag,
    input wire [3:0] r_cpl_status,
    input wire [25:0] r_cpl_bytes,
    input wire [1:0] ack_valid,result_ready,
    input wire [1159:0] ack_event,
    output wire [1:0] result_valid,
    output wire [2047:0] result_context,
    output wire [7:0] result_status,
    output wire [127:0] result_bytes,
    output wire [3:0] contexts_held,wire_sent,
    output wire [1:0] halted,dma_fault,
    input wire dma_link_up,
    input wire [1:0] dma_run,dma_error,dma_done,dsc_ready,axis_ready,
    output wire [1:0] dsc_load,axis_valid,axis_last,
    output wire [127:0] dsc_src,dsc_dst,axis_keep,
    output wire [55:0] dsc_len,
    output wire [31:0] dsc_ctl,
    output wire [1023:0] axis_data,
    input wire management_core_ready, exclusive_owner, management_fault,
    input wire management_wait_expired, management_reset_request, guard_start,
    input wire [63:0] guard_epoch,
    output wire guard_start_ready, guard_active,
    output wire [15:0] guard_faults,
    output wire [1:0] admit_descriptor, validated_done,
    output wire [63:0] published_count, observed_count, delivered_count,
    output wire [63:0] control_snapshot, status_snapshot, count_snapshot,
    output wire [31:0] m_axil_awaddr,m_axil_wdata,m_axil_araddr,
    output wire [2:0] m_axil_awprot,m_axil_arprot,
    output wire [3:0] m_axil_wstrb,
    output wire m_axil_awvalid,m_axil_wvalid,m_axil_bready,m_axil_arvalid,m_axil_rready,
    input wire m_axil_awready,m_axil_wready,m_axil_bvalid,m_axil_arready,m_axil_rvalid,
    input wire [1:0] m_axil_bresp,m_axil_rresp,
    input wire [31:0] m_axil_rdata
);
    wire b_cfg_valid,b_cfg_ready,b_cfg_error,b_submit_valid,b_submit_ready;
    wire [1:0] data_valid,data_last,data_ready,local_valid,b_result_ready;
    wire [1:0] b_result_valid,result_permit;
    assign result_valid=b_result_valid & result_permit;
    wire [1023:0] data;
    wire [2047:0] data_context;
    wire [1159:0] local_event;
    wire [1:0] raw_cmd_valid,raw_cmd_ready,domain_fault,domain_pending;
    ntrpu_hbm_source_join source_join(
        .bridge_halted(halted),.fault(source_fault),.stop_out(reader_stop),
        .reader_fault(reader_fault || |domain_fault),
        .b_reject_valid(reject_valid),.cmd_valid(raw_cmd_valid),.cmd_ready(raw_cmd_ready),
        .result_valid(b_result_valid),.*);
    for(genvar lane=0;lane<2;lane=lane+1)begin: domains
        ntrpu_peer_read_domain fence(
            .clk(clk),.rst_n(rst_n),.s_valid(raw_cmd_valid[lane]),.s_ready(raw_cmd_ready[lane]),
            .address(cmd_addr[lane*64+:64]),.m_valid(cmd_valid[lane]),.m_ready(cmd_ready[lane]),
            .complete(r_cpl_valid[lane] && r_cpl_ready[lane]),
            .pending(domain_pending[lane]),.fault(domain_fault[lane]));
    end
    ntrpu_hbm_xdma_guarded_held #(.GUARD_WAIT_LIMIT(GUARD_WAIT_LIMIT),.MASTER_WAIT_LIMIT(MASTER_WAIT_LIMIT)) guarded(
        .stop(reader_stop),.cfg_valid(b_cfg_valid),.cfg_ready(b_cfg_ready),.cfg_error(b_cfg_error),
        .submit_valid(b_submit_valid),.submit_ready(b_submit_ready),.result_ready(b_result_ready),
        .result_valid(b_result_valid),.*);
endmodule
`default_nettype wire
