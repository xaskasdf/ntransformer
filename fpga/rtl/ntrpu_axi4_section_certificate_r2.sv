`timescale 1ns / 1ps
// Revocable proof of the retained AXI4 admission + external CDC section only.
// Does not include local error replies or protected endpoint queues. No reset
// permission for XDMA, no operational reset and no reopening of admission.
module ntrpu_axi4_section_certificate_r2 (
    input wire mgmt_clk,host_clk,device_clk,cold_rst_n,req_valid,
    input wire [63:0] generation,tag,
    output wire close_host,host_proof_request,
    output wire [63:0] host_proof_tag,
    input wire host_drain_ack,host_source_fenced,
    input wire [63:0] host_drain_tag,
    input wire [6:0] admission_fault_sources,
    input wire host_awvalid,
    input wire host_awready,
    input wire [63:0] host_awaddr,
    input wire [3:0] host_awid,
    input wire [7:0] host_awlen,
    input wire [2:0] host_awsize,
    input wire [2:0] host_awprot,
    input wire [1:0] host_awburst,
    input wire [3:0] host_awcache,
    input wire host_awlock,
    input wire host_wvalid,
    input wire host_wready,
    input wire [511:0] host_wdata,
    input wire [63:0] host_wstrb,
    input wire host_wlast,
    input wire host_bvalid,
    input wire host_bready,
    input wire [3:0] host_bid,
    input wire [1:0] host_bresp,
    input wire host_arvalid,
    input wire host_arready,
    input wire [63:0] host_araddr,
    input wire [3:0] host_arid,
    input wire [7:0] host_arlen,
    input wire [2:0] host_arsize,
    input wire [2:0] host_arprot,
    input wire [1:0] host_arburst,
    input wire [3:0] host_arcache,
    input wire host_arlock,
    input wire host_rvalid,
    input wire host_rready,
    input wire [511:0] host_rdata,
    input wire [3:0] host_rid,
    input wire [1:0] host_rresp,
    input wire host_rlast,
    input wire device_awvalid,
    input wire device_awready,
    input wire [63:0] device_awaddr,
    input wire [3:0] device_awid,
    input wire [7:0] device_awlen,
    input wire [2:0] device_awsize,
    input wire [2:0] device_awprot,
    input wire [1:0] device_awburst,
    input wire [3:0] device_awcache,
    input wire device_awlock,
    input wire device_wvalid,
    input wire device_wready,
    input wire [511:0] device_wdata,
    input wire [63:0] device_wstrb,
    input wire device_wlast,
    input wire device_bvalid,
    input wire device_bready,
    input wire [3:0] device_bid,
    input wire [1:0] device_bresp,
    input wire device_arvalid,
    input wire device_arready,
    input wire [63:0] device_araddr,
    input wire [3:0] device_arid,
    input wire [7:0] device_arlen,
    input wire [2:0] device_arsize,
    input wire [2:0] device_arprot,
    input wire [1:0] device_arburst,
    input wire [3:0] device_arcache,
    input wire device_arlock,
    input wire device_rvalid,
    input wire device_rready,
    input wire [511:0] device_rdata,
    input wire [3:0] device_rid,
    input wire [1:0] device_rresp,
    input wire device_rlast,
    output wire section_ack,upstream_certificate,pair_certificate,
    output wire host_certificate,device_certificate,
    output wire host_mailbox_busy,pair_host_busy,pair_device_busy,
    output wire [5:0] host_fault_sources,device_fault_sources,
    output wire [18:0] fault_mgmt,
    output wire [5:0] host_aw_pending,host_w_pending,device_aw_pending,device_w_pending,
    output wire [7:0] host_ar_pending,device_ar_pending,
    output wire [8:0] host_w_partial,device_w_partial
);
    wire [6:0] synchronized_admission;
    reg [6:0] retained_admission=0;
    wire [11:0] pair_fault_mgmt;
    wire upstream_cached,pair_cached,fault_close_host;
    reg retained_close=0;
    wire any_fault=|fault_mgmt;
    wire accepted_request=cold_rst_n && req_valid && !any_fault;
    wire [2:0] peer_kind;
    wire [63:0] peer_generation;
    // Each source is already a retained host register. Cross individually;
    // an additional host edge must not be required after source clock loss.
    for(genvar i=0;i<7;i=i+1)begin:admission_sources
        xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
            .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) synchronize_fault(
            .src_clk(1'b0),.src_in(admission_fault_sources[i]),
            .dest_clk(mgmt_clk),.dest_out(synchronized_admission[i]));
    end
    always @(posedge mgmt_clk)begin
        if(!cold_rst_n)retained_admission<=0;
        else retained_admission<=retained_admission | synchronized_admission;
    end
    // Pair faults already originate from individual XPM synchronizers and
    // retained registers in this management domain. Do not synchronize again.
    assign fault_mgmt={pair_fault_mgmt,retained_admission | synchronized_admission};
    assign upstream_certificate=upstream_cached && !any_fault;
    assign pair_certificate=pair_cached && !any_fault;
    assign section_ack=upstream_certificate && pair_certificate;
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(1)) fault_close_cdc(
            .src_clk(mgmt_clk),.src_in(any_fault),
            .dest_clk(host_clk),.dest_out(fault_close_host));
    assign close_host=retained_close || host_proof_request ||
        (|admission_fault_sources) || (|host_fault_sources) || fault_close_host;
    always @(posedge host_clk)begin
        if(!cold_rst_n)retained_close<=0;
        else if(host_proof_request || (|admission_fault_sources) ||
                (|host_fault_sources) || fault_close_host)retained_close<=1;
    end
    ntrpu_tagged_reset_mailbox upstream_mailbox(
        .mgmt_clk(mgmt_clk),.peer_clk(host_clk),.req_valid(accepted_request),
        .req_kind(3'd1),.req_generation(generation),.req_tag(tag),
        .ack_valid(upstream_cached),.ack_kind(),.ack_generation(),.ack_tag(),
        .busy(host_mailbox_busy),.obsolete_completions(),
        .peer_req_valid(host_proof_request),.peer_req_kind(peer_kind),
        .peer_req_generation(peer_generation),.peer_req_tag(host_proof_tag),
        .peer_ack_valid(host_drain_ack && host_source_fenced && peer_kind==1),
        .peer_ack_kind(3'd1),.peer_ack_generation(peer_generation),
        .peer_ack_tag(host_drain_tag),.rejected_peer_ack_cycles());
    ntrpu_axi4_pair_drain_r2 pair(
        .mgmt_clk(mgmt_clk),.host_clk(host_clk),.device_clk(device_clk),.cold_rst_n(cold_rst_n),
        .req_valid(accepted_request && upstream_certificate),.generation(generation),.tag(tag),
        .host_source_fenced(host_source_fenced),
        .host_awvalid(host_awvalid),
        .host_awready(host_awready),
        .host_awaddr(host_awaddr),
        .host_awid(host_awid),
        .host_awlen(host_awlen),
        .host_awsize(host_awsize),
        .host_awprot(host_awprot),
        .host_awburst(host_awburst),
        .host_awcache(host_awcache),
        .host_awlock(host_awlock),
        .host_wvalid(host_wvalid),
        .host_wready(host_wready),
        .host_wdata(host_wdata),
        .host_wstrb(host_wstrb),
        .host_wlast(host_wlast),
        .host_bvalid(host_bvalid),
        .host_bready(host_bready),
        .host_bid(host_bid),
        .host_bresp(host_bresp),
        .host_arvalid(host_arvalid),
        .host_arready(host_arready),
        .host_araddr(host_araddr),
        .host_arid(host_arid),
        .host_arlen(host_arlen),
        .host_arsize(host_arsize),
        .host_arprot(host_arprot),
        .host_arburst(host_arburst),
        .host_arcache(host_arcache),
        .host_arlock(host_arlock),
        .host_rvalid(host_rvalid),
        .host_rready(host_rready),
        .host_rdata(host_rdata),
        .host_rid(host_rid),
        .host_rresp(host_rresp),
        .host_rlast(host_rlast),
        .device_awvalid(device_awvalid),
        .device_awready(device_awready),
        .device_awaddr(device_awaddr),
        .device_awid(device_awid),
        .device_awlen(device_awlen),
        .device_awsize(device_awsize),
        .device_awprot(device_awprot),
        .device_awburst(device_awburst),
        .device_awcache(device_awcache),
        .device_awlock(device_awlock),
        .device_wvalid(device_wvalid),
        .device_wready(device_wready),
        .device_wdata(device_wdata),
        .device_wstrb(device_wstrb),
        .device_wlast(device_wlast),
        .device_bvalid(device_bvalid),
        .device_bready(device_bready),
        .device_bid(device_bid),
        .device_bresp(device_bresp),
        .device_arvalid(device_arvalid),
        .device_arready(device_arready),
        .device_araddr(device_araddr),
        .device_arid(device_arid),
        .device_arlen(device_arlen),
        .device_arsize(device_arsize),
        .device_arprot(device_arprot),
        .device_arburst(device_arburst),
        .device_arcache(device_arcache),
        .device_arlock(device_arlock),
        .device_rvalid(device_rvalid),
        .device_rready(device_rready),
        .device_rdata(device_rdata),
        .device_rid(device_rid),
        .device_rresp(device_rresp),
        .device_rlast(device_rlast),
        .external_drain_ack(pair_cached),.host_certificate(host_certificate),.device_certificate(device_certificate),
        .host_mailbox_busy(pair_host_busy),.device_mailbox_busy(pair_device_busy),
        .host_fault_sources(host_fault_sources),.device_fault_sources(device_fault_sources),
        .fault_mgmt(pair_fault_mgmt),.host_aw_pending(host_aw_pending),.host_w_pending(host_w_pending),
        .host_ar_pending(host_ar_pending),.host_w_partial(host_w_partial),
        .device_aw_pending(device_aw_pending),.device_w_pending(device_w_pending),
        .device_ar_pending(device_ar_pending),.device_w_partial(device_w_partial));
endmodule
