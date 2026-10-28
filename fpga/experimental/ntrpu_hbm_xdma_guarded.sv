`timescale 1ns/1ps
`default_nettype none
// Isolated composition; compile with the qualified rev3 bridge sources.
// All management inputs are synchronous to dma_clk. cold_reset_n is global
// configuration reset only: native reset/link loss must preserve this wrapper.
module ntrpu_hbm_xdma_guarded #(
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
    input wire [1:0] data_valid,data_last,
    output wire [1:0] data_ready,
    input wire [1023:0] data,
    input wire [2047:0] data_context,
    input wire [1:0] local_valid,ack_valid,result_ready,
    input wire [1159:0] local_event,ack_event,
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
    wire [1:0] guard_error;
    wire m_admit,m_req_valid,m_req_ready,m_req_write,m_rsp_valid,m_rsp_ready,m_rsp_timeout,m_timeout_latched;
    wire [31:0] m_req_address,m_req_data,m_rsp_data;
    wire [3:0] m_req_strobe;
    wire [127:0] m_req_identity,m_rsp_identity;
    wire [1:0] m_rsp_code;
    wire management_busy;
    ntrpu_hbm_dma_bridge bridge(
        .dma_done(validated_done),.dma_error(dma_error | guard_error),
        .dsc_ready(dsc_ready & admit_descriptor),.*);
    ntrpu_xdma_completion_guard #(.WAIT_LIMIT(GUARD_WAIT_LIMIT)) guard(
        .clk(dma_clk),.cold_rst_n(dma_rst_n),.core_ready(management_core_ready && dma_link_up),
        .external_fault(management_fault || |dma_error),.wait_expired(management_wait_expired),
        .reset_request(management_reset_request),.start_valid(guard_start),.start_ready(guard_start_ready),
        .start_epoch(guard_epoch),.native_run(dma_run),.native_done(dma_done),.descriptor_load(dsc_load),
        .dma_error(guard_error),.active(guard_active),.faults(guard_faults),.*);
    ntrpu_xdma_local_master #(.WAIT_LIMIT(MASTER_WAIT_LIMIT)) master(
        .clk(dma_clk),.cold_rst_n(dma_rst_n),.admit(m_admit),
        .req_valid(m_req_valid),.req_ready(m_req_ready),.req_write(m_req_write),
        .req_address(m_req_address),.req_data(m_req_data),.req_strobe(m_req_strobe),.req_identity(m_req_identity),
        .rsp_valid(m_rsp_valid),.rsp_ready(m_rsp_ready),.rsp_data(m_rsp_data),.rsp_code(m_rsp_code),
        .rsp_identity(m_rsp_identity),.rsp_timeout(m_rsp_timeout),.timeout_latched(m_timeout_latched),
        .busy(management_busy),.*);
endmodule
`default_nettype wire
