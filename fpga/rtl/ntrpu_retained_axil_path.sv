`timescale 1ns/1ps
// Retained AXI-Lite path and fresh local-response provider. Cold lifetime only.
module ntrpu_retained_axil_path(
    input wire mgmt_clk, host_clk, device_clk, cold_rst_n,
    input wire req_valid, local_request,
    input wire [63:0] generation, tag,
    output wire section_ack, section_fault, local_idle, request_present, reply_pending,
    output wire local_ack,
    output wire [63:0] local_generation, local_tag,
    output wire [1:0] local_fault,
    output wire host_reset_n, device_reset_n,
    output wire [1:0] source_faults,
    input wire [31:0] s_awaddr,
    input wire [31:0] s_wdata,
    input wire [31:0] s_araddr,
    input wire [2:0] s_awprot,
    input wire [2:0] s_arprot,
    input wire [3:0] s_wstrb,
    input wire s_awvalid,
    input wire s_wvalid,
    input wire s_bready,
    input wire s_arvalid,
    input wire s_rready,
    output wire s_awready,
    output wire s_wready,
    output wire s_bvalid,
    output wire s_arready,
    output wire s_rvalid,
    output wire [1:0] s_bresp,
    output wire [1:0] s_rresp,
    output wire [31:0] s_rdata,
    output wire [31:0] d_awaddr,
    output wire [31:0] d_wdata,
    output wire [31:0] d_araddr,
    output wire [2:0] d_awprot,
    output wire [2:0] d_arprot,
    output wire [3:0] d_wstrb,
    output wire d_awvalid,
    output wire d_wvalid,
    output wire d_bready,
    output wire d_arvalid,
    output wire d_rready,
    input wire d_awready,
    input wire d_wready,
    input wire d_bvalid,
    input wire d_arready,
    input wire d_rvalid,
    input wire [1:0] d_bresp,
    input wire [1:0] d_rresp,
    input wire [31:0] d_rdata
);
    wire [31:0] g_awaddr;
    wire [31:0] g_wdata;
    wire [31:0] g_araddr;
    wire [2:0] g_awprot;
    wire [2:0] g_arprot;
    wire [3:0] g_wstrb;
    wire g_awvalid;
    wire g_wvalid;
    wire g_bready;
    wire g_arvalid;
    wire g_rready;
    wire g_awready;
    wire g_wready;
    wire g_bvalid;
    wire g_arready;
    wire g_rvalid;
    wire [1:0] g_bresp;
    wire [1:0] g_rresp;
    wire [31:0] g_rdata;
    wire [31:0] h_awaddr;
    wire [31:0] h_wdata;
    wire [31:0] h_araddr;
    wire [2:0] h_awprot;
    wire [2:0] h_arprot;
    wire [3:0] h_wstrb;
    wire h_awvalid;
    wire h_wvalid;
    wire h_bready;
    wire h_arvalid;
    wire h_rready;
    wire h_awready;
    wire h_wready;
    wire h_bvalid;
    wire h_arready;
    wire h_rvalid;
    wire [1:0] h_bresp;
    wire [1:0] h_rresp;
    wire [31:0] h_rdata;
    wire c_bvalid,c_bready,c_rvalid,c_rready,shutdown_request,shutdown_active;
    wire shutdown_requested,wr_shutdown,rd_shutdown,irq,wr_irq,rd_irq;
    wire [1:0] c_bresp,c_rresp;
    wire [31:0] c_rdata;
    wire [3:0] section_fault_bits;
    assign section_fault=|section_fault_bits;
    assign request_present=s_awvalid || s_wvalid || s_arvalid;
    assign reply_pending=s_bvalid || s_rvalid;
    // Configuration-only rails: no operational reset can erase an obligation.
    ntrpu_reset_release host_release(.clk(host_clk),.hold_async(!cold_rst_n),.released(host_reset_n));
    ntrpu_reset_release device_release(.clk(device_clk),.hold_async(!cold_rst_n),.released(device_reset_n));
    ntrpu_axil_fence_section_r3  section(
        .mgmt_clk(mgmt_clk),
        .host_clk(host_clk),
        .device_clk(device_clk),
        .cold_rst_n(cold_rst_n),
        .req_valid(req_valid),
        .generation(generation),
        .tag(tag),
        .shutdown_active(shutdown_active),
        .device_bus({d_awvalid,d_awready,d_wvalid,d_wready,d_bvalid,d_bready,d_arvalid,d_arready,d_rvalid,d_rready}),
        .host_requests({h_awvalid,h_awready,h_wvalid,h_wready,h_arvalid,h_arready}),
        .c_bvalid(c_bvalid),
        .c_rvalid(c_rvalid),
        .c_bready(c_bready),
        .c_rready(c_rready),
        .c_bresp(c_bresp),
        .c_rresp(c_rresp),
        .c_rdata(c_rdata),
        .h_bvalid(h_bvalid),
        .h_rvalid(h_rvalid),
        .h_bready(h_bready),
        .h_rready(h_rready),
        .h_bresp(h_bresp),
        .h_rresp(h_rresp),
        .h_rdata(h_rdata),
        .guard_fault_sources(),
        .guard_aw_pending(),
        .guard_w_pending(),
        .guard_ar_pending(),
        .close_host(shutdown_request),
        .host_proof_request(),
        .host_proof_tag(),
        .section_ack(section_ack),
        .upstream_certificate(),
        .pair_certificate(),
        .host_fault(),
        .device_fault(),
        .fault_mgmt(section_fault_bits),
        .host_mailbox_busy(),
        .pair_host_busy(),
        .pair_device_busy(),
        .diagnostics(),
        .s_awaddr(s_awaddr),
        .s_wdata(s_wdata),
        .s_araddr(s_araddr),
        .s_awprot(s_awprot),
        .s_arprot(s_arprot),
        .s_wstrb(s_wstrb),
        .s_awvalid(s_awvalid),
        .s_wvalid(s_wvalid),
        .s_bready(s_bready),
        .s_arvalid(s_arvalid),
        .s_rready(s_rready),
        .s_awready(s_awready),
        .s_wready(s_wready),
        .s_bvalid(s_bvalid),
        .s_arready(s_arready),
        .s_rvalid(s_rvalid),
        .s_bresp(s_bresp),
        .s_rresp(s_rresp),
        .s_rdata(s_rdata),
        .m_awaddr(g_awaddr),
        .m_wdata(g_wdata),
        .m_araddr(g_araddr),
        .m_awprot(g_awprot),
        .m_arprot(g_arprot),
        .m_wstrb(g_wstrb),
        .m_awvalid(g_awvalid),
        .m_wvalid(g_wvalid),
        .m_bready(g_bready),
        .m_arvalid(g_arvalid),
        .m_rready(g_rready),
        .m_awready(g_awready),
        .m_wready(g_wready),
        .m_bvalid(g_bvalid),
        .m_arready(g_arready),
        .m_rvalid(g_rvalid),
        .m_bresp(g_bresp),
        .m_rresp(g_rresp),
        .m_rdata(g_rdata),
        .source_fenced(),
        .drain_ack(),
        .local_idle(local_idle),
        .protocol_fault(),
        .fault_sources(source_faults),
        .drain_tag(),
        .aw_pending(),
        .w_pending(),
        .ar_pending(),
        .aw_left(),
        .w_left(),
        .ar_left(),
        .state_debug()
    );
    ntrpu_host_axil_shutdown shutdown_manager(
        .clk(host_clk),.resetn(host_reset_n),.request_shutdown(shutdown_request),
        .shutdown_requested(shutdown_requested),.in_shutdown(shutdown_active),
        .wr_in_shutdown(wr_shutdown),.rd_in_shutdown(rd_shutdown),.irq(irq),.wr_irq(wr_irq),.rd_irq(rd_irq),
        .s_axi_awaddr(g_awaddr),.s_axi_awprot(g_awprot),.s_axi_awvalid(g_awvalid),.s_axi_awready(g_awready),
        .s_axi_wdata(g_wdata),.s_axi_wstrb(g_wstrb),.s_axi_wvalid(g_wvalid),.s_axi_wready(g_wready),
        .s_axi_bresp(g_bresp),.s_axi_bvalid(g_bvalid),.s_axi_bready(g_bready),
        .s_axi_araddr(g_araddr),.s_axi_arprot(g_arprot),.s_axi_arvalid(g_arvalid),.s_axi_arready(g_arready),
        .s_axi_rdata(g_rdata),.s_axi_rresp(g_rresp),.s_axi_rvalid(g_rvalid),.s_axi_rready(g_rready),
        .m_axi_awaddr(h_awaddr),.m_axi_awprot(h_awprot),.m_axi_awvalid(h_awvalid),.m_axi_awready(h_awready),
        .m_axi_wdata(h_wdata),.m_axi_wstrb(h_wstrb),.m_axi_wvalid(h_wvalid),.m_axi_wready(h_wready),
        .m_axi_bresp(h_bresp),.m_axi_bvalid(h_bvalid),.m_axi_bready(h_bready),
        .m_axi_araddr(h_araddr),.m_axi_arprot(h_arprot),.m_axi_arvalid(h_arvalid),.m_axi_arready(h_arready),
        .m_axi_rdata(h_rdata),.m_axi_rresp(h_rresp),.m_axi_rvalid(h_rvalid),.m_axi_rready(h_rready));
    ntrpu_axil_cdc_recode cdc(
        .s_axi_aclk(host_clk),.s_axi_aresetn(host_reset_n),.m_axi_aclk(device_clk),.m_axi_aresetn(device_reset_n),
        .s_axi_awaddr(h_awaddr),.s_axi_wdata(h_wdata),.s_axi_araddr(h_araddr),
        .s_axi_awprot(h_awprot),.s_axi_arprot(h_arprot),.s_axi_wstrb(h_wstrb),
        .s_axi_awvalid(h_awvalid),.s_axi_wvalid(h_wvalid),.s_axi_bready(c_bready),.s_axi_arvalid(h_arvalid),.s_axi_rready(c_rready),
        .s_axi_awready(h_awready),.s_axi_wready(h_wready),.s_axi_bvalid(c_bvalid),.s_axi_arready(h_arready),.s_axi_rvalid(c_rvalid),
        .s_axi_bresp(c_bresp),.s_axi_rresp(c_rresp),.s_axi_rdata(c_rdata),
        .m_axi_awaddr(d_awaddr),.m_axi_wdata(d_wdata),.m_axi_araddr(d_araddr),
        .m_axi_awprot(d_awprot),.m_axi_arprot(d_arprot),.m_axi_wstrb(d_wstrb),
        .m_axi_awvalid(d_awvalid),.m_axi_wvalid(d_wvalid),.m_axi_bready(d_bready),.m_axi_arvalid(d_arvalid),.m_axi_rready(d_rready),
        .m_axi_awready(d_awready),.m_axi_wready(d_wready),.m_axi_bvalid(d_bvalid),.m_axi_arready(d_arready),.m_axi_rvalid(d_rvalid),
        .m_axi_bresp(d_bresp),.m_axi_rresp(d_rresp),.m_axi_rdata(d_rdata));
    ntrpu_local_reply_certificate_r2 #(.FAULT_WIDTH(2)) local_provider(
        .mgmt_clk(mgmt_clk),
        .host_clk(host_clk),
        .cold_rst_n(cold_rst_n),
        .req_valid(local_request),
        .generation(generation),
        .tag(tag),
        .local_idle(local_idle),
        .request_present(request_present),
        .reply_pending(reply_pending),
        .source_faults(source_faults),
        .section_fault_mgmt(section_fault),
        .ack_valid(local_ack),
        .ack_generation(local_generation),
        .ack_tag(local_tag),
        .busy(),
        .peer_request(),
        .peer_answer(),
        .peer_generation(),
        .peer_tag(),
        .fault_mgmt(local_fault),
        .revoked_source(),
        .obsolete_completions()
    );
endmodule
