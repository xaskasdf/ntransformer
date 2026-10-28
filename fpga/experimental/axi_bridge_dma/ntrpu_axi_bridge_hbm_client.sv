`timescale 1ns/1ps
`default_nettype none
// Reader-facing signals share clk (the bridge AXI clock). The reader owns HBM CDC.
// A direct attachment at 250 MHz requires new reader timing qualification.
module ntrpu_axi_bridge_hbm_client #(
    parameter integer AXI_ADDR_BITS=64,WAIT_LIMIT=65536,SHARED_NATIVE_RESET=1,
    parameter [63:0] APERTURE_BASE=64'h10000000
) (
    input wire  clk,
    input wire  mgmt_clk,
    input wire  configuration_rst_n,
    input wire  native_link_up,
    input wire  native_axi_reset_n,
    input wire  native_ctl_reset_n,
    input wire  perst_n,
    output wire  client_cold_rst_n,
    output wire  client_link_up,
    output wire  client_stop,
    output wire  guard_clock_alive,
    output wire  guard_armed,
    output wire  guard_fault,
    output wire [1:0] guard_fault_code,
    input wire  cfg_valid,
    output wire  cfg_ready,
    input wire [63:0] cfg_session,
    input wire [63:0] cfg_generation,
    input wire [63:0] cfg_axi_base,
    input wire [63:0] cfg_pcie_base,
    output wire  cfg_rsp_valid,
    input wire  cfg_rsp_ready,
    output wire  cfg_rsp_error,
    input wire  cmd_valid,
    output wire  cmd_ready,
    input wire [127:0] cmd_tag,
    input wire [63:0] cmd_generation,
    input wire [63:0] cmd_offset,
    input wire [12:0] cmd_bytes,
    output wire  reject_valid,
    input wire  reject_ready,
    output wire [3:0] reject_code,
    output wire [127:0] reject_tag,
    output wire [AXI_ADDR_BITS-1:0] m_awaddr,
    output wire [3:0] m_awid,
    output wire [7:0] m_awlen,
    output wire [2:0] m_awsize,
    output wire [1:0] m_awburst,
    output wire  m_awvalid,
    input wire  m_awready,
    output wire [511:0] m_wdata,
    output wire [63:0] m_wstrb,
    output wire  m_wlast,
    output wire  m_wvalid,
    input wire  m_wready,
    input wire  m_bvalid,
    output wire  m_bready,
    input wire [1:0] m_bresp,
    input wire [3:0] m_bid,
    input wire  ack_valid,
    output wire  ack_ready,
    input wire [127:0] ack_tag,
    input wire [63:0] ack_generation,
    input wire [12:0] ack_bytes,
    output wire  result_valid,
    input wire  result_ready,
    output wire [127:0] result_tag,
    output wire [12:0] result_bytes,
    output wire  owned,
    output wire  fault,
    output wire [7:0] fault_code,
    output wire  window_held,
    output wire [31:0] m_axil_awaddr,
    output wire [2:0] m_axil_awprot,
    output wire  m_axil_awvalid,
    input wire  m_axil_awready,
    output wire [31:0] m_axil_wdata,
    output wire [3:0] m_axil_wstrb,
    output wire  m_axil_wvalid,
    input wire  m_axil_wready,
    input wire [1:0] m_axil_bresp,
    input wire  m_axil_bvalid,
    output wire  m_axil_bready,
    output wire [31:0] m_axil_araddr,
    output wire [2:0] m_axil_arprot,
    output wire  m_axil_arvalid,
    input wire  m_axil_arready,
    input wire [31:0] m_axil_rdata,
    input wire [1:0] m_axil_rresp,
    input wire  m_axil_rvalid,
    output wire  m_axil_rready,
    output wire  mapper_fault,
    output wire [7:0] mapper_fault_code,
    output wire  map_verified,
    input wire [63:0] cmd_source,
    input wire  reader_started,
    input wire  reader_fault,
    output wire  reader_stop,
    output wire  source_owned,
    output wire  source_fault,
    output wire [1:0] r_cmd_valid,
    input wire [1:0] r_cmd_ready,
    output wire [127:0] r_cmd_addr,
    output wire [127:0] r_cmd_tag,
    output wire [15:0] r_cmd_len,
    input wire [1:0] r_data_valid,
    input wire [1:0] r_data_last,
    output wire [1:0] r_data_ready,
    input wire [1023:0] r_data,
    input wire [127:0] r_data_tag,
    input wire [1:0] r_cpl_valid,
    output wire [1:0] r_cpl_ready,
    input wire [127:0] r_cpl_tag,
    input wire [3:0] r_cpl_status,
    input wire [25:0] r_cpl_bytes
);
    wire [12:0] src_cpl_bytes;
    wire  src_cpl_ready;
    wire [1:0] src_cpl_status;
    wire [127:0] src_cpl_tag;
    wire  src_cpl_valid;
    wire [511:0] src_data;
    wire  src_last;
    wire  src_ready;
    wire [63:0] src_req_address;
    wire [12:0] src_req_bytes;
    wire  src_req_ready;
    wire [127:0] src_req_tag;
    wire  src_req_valid;
    wire [127:0] src_tag;
    wire  src_valid;
    wire pcie_link,pcie_stop;
    (* KEEP_HIERARCHY="YES" *) ntrpu_axi_bridge_guard_v3 #(.SHARED_NATIVE_RESET(SHARED_NATIVE_RESET)) guard(
        .mgmt_clk(mgmt_clk),.axi_clk(clk),.cold_rst_n(configuration_rst_n),
        .native_link_up(native_link_up),.native_axi_reset_n(native_axi_reset_n),
        .native_ctl_reset_n(native_ctl_reset_n),.perst_n(perst_n),
        .client_cold_rst_n(client_cold_rst_n),.client_link_up(pcie_link),.client_stop(pcie_stop),
        .clock_alive(guard_clock_alive),.armed(guard_armed),.fault(guard_fault),.fault_code(guard_fault_code));
    assign client_link_up=pcie_link&&reader_started&&!source_fault&&!reader_fault;
    assign client_stop=pcie_stop||source_fault||reader_fault;
    (* KEEP_HIERARCHY="YES" *) ntrpu_axi_bridge_endpoint_client #(
        .AXI_ADDR_BITS(AXI_ADDR_BITS),.SOURCE_ADDR_BITS(34),
        .WAIT_LIMIT(WAIT_LIMIT),.APERTURE_BASE(APERTURE_BASE)) client(
        .cold_rst_n(client_cold_rst_n),.link_up(client_link_up),.stop(client_stop),.*);
    (* KEEP_HIERARCHY="YES" *) ntrpu_axi_bridge_hbm_source source_join(
        .cold_rst_n(client_cold_rst_n),.enable(pcie_link),.stop(pcie_stop||fault),
        .retire_valid(result_valid&&result_ready),.retire_tag(result_tag),
        .owned(source_owned),.fault(source_fault),.*);
endmodule
`default_nettype wire
