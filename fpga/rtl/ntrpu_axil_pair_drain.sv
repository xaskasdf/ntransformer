`timescale 1ns / 1ps
// Sequential EXTERNAL interface drain certificates, not an internal IP-idle
// signal or permission to reset XDMA. Host source_fenced must remain asserted
// for the entire maintenance phase and cover all upstream producers/queues.
// First prove the host transaction ledger, then ask the device ledger afresh.
// No bus signals are gated here; already offered transfers must complete.
module ntrpu_axil_pair_drain (
    input wire mgmt_clk, host_clk, device_clk, cold_rst_n,
    input wire req_valid,
    input wire [63:0] generation, tag,
    input wire host_source_fenced,
    // Each bus packs AWVALID,AWREADY,WVALID,WREADY,BVALID,BREADY,
    // ARVALID,ARREADY,RVALID,RREADY from bit 9 down to bit 0.
    input wire [9:0] host_bus, device_bus,
    output wire external_drain_ack, host_certificate, device_certificate,
    output wire host_mailbox_busy, device_mailbox_busy,
    output wire host_fault, device_fault,
    output wire [3:0] host_aw_pending,host_w_pending,host_ar_pending,
    output wire [3:0] device_aw_pending,device_w_pending,device_ar_pending
);
    wire hreq,dreq,hack,dack;
    wire [2:0] hk,dk;
    wire [63:0] hg,dg,ht,dt,ha,da;
    wire [63:0] hreturn_gen,dreturn_gen,hreturn_tag,dreturn_tag;
    wire [2:0] hreturn_kind,dreturn_kind;
    assign external_drain_ack=host_certificate && device_certificate;
    ntrpu_tagged_reset_mailbox host_mailbox(
        .mgmt_clk(mgmt_clk),.peer_clk(host_clk),.req_valid(req_valid),.req_kind(3'd1),
        .req_generation(generation),.req_tag(tag),.ack_valid(host_certificate),
        .ack_kind(hreturn_kind),.ack_generation(hreturn_gen),.ack_tag(hreturn_tag),
        .busy(host_mailbox_busy),.obsolete_completions(),.peer_req_valid(hreq),
        .peer_req_kind(hk),.peer_req_generation(hg),.peer_req_tag(ht),
        .peer_ack_valid(hack),.peer_ack_kind(3'd1),.peer_ack_generation(hg),.peer_ack_tag(ha),
        .rejected_peer_ack_cycles());
    ntrpu_axil_drain_monitor host_ledger(
        .clk(host_clk),.cold_rst_n(cold_rst_n),.req_valid(hreq && hk==1),.req_tag(ht),
        .source_fenced(host_source_fenced),.ack_valid(hack),.ack_tag(ha),.protocol_fault(host_fault),
        .awvalid(host_bus[9]),.awready(host_bus[8]),.wvalid(host_bus[7]),.wready(host_bus[6]),
        .bvalid(host_bus[5]),.bready(host_bus[4]),.arvalid(host_bus[3]),.arready(host_bus[2]),
        .rvalid(host_bus[1]),.rready(host_bus[0]),
        .aw_pending(host_aw_pending),.w_pending(host_w_pending),.ar_pending(host_ar_pending));
    ntrpu_tagged_reset_mailbox device_mailbox(
        .mgmt_clk(mgmt_clk),.peer_clk(device_clk),.req_valid(req_valid && host_certificate),.req_kind(3'd1),
        .req_generation(generation),.req_tag(tag),.ack_valid(device_certificate),
        .ack_kind(dreturn_kind),.ack_generation(dreturn_gen),.ack_tag(dreturn_tag),
        .busy(device_mailbox_busy),.obsolete_completions(),.peer_req_valid(dreq),
        .peer_req_kind(dk),.peer_req_generation(dg),.peer_req_tag(dt),
        .peer_ack_valid(dack),.peer_ack_kind(3'd1),.peer_ack_generation(dg),.peer_ack_tag(da),
        .rejected_peer_ack_cycles());
    // Causal fence: this request is emitted only after the host certificate.
    // With the host source persistently fenced and its ledger empty, no new
    // request can arrive downstream. Responses already in flight still drain.
    ntrpu_axil_drain_monitor device_ledger(
        .clk(device_clk),.cold_rst_n(cold_rst_n),.req_valid(dreq && dk==1),.req_tag(dt),
        .source_fenced(dreq),.ack_valid(dack),.ack_tag(da),.protocol_fault(device_fault),
        .awvalid(device_bus[9]),.awready(device_bus[8]),.wvalid(device_bus[7]),.wready(device_bus[6]),
        .bvalid(device_bus[5]),.bready(device_bus[4]),.arvalid(device_bus[3]),.arready(device_bus[2]),
        .rvalid(device_bus[1]),.rready(device_bus[0]),
        .aw_pending(device_aw_pending),.w_pending(device_w_pending),.ar_pending(device_ar_pending));
    // Generation is stable with each mailbox request and its monitor request
    // is dropped for a full handshake between deliveries. Never attach a
    // different producer that can carry a stale high ACK across that gap.
endmodule
