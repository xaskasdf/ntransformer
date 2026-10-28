`timescale 1ns / 1ps
// Fresh management certificate for the RETAINED admission producer + shutdown
// manager + both external CDC interfaces. The admission block permanently stops
// its downstream producer; local late-error responses remain outside the reset
// section. This is not complete XDMA drain, endpoint reset permission or NTRD rearm.
module ntrpu_axil_fence_certificate (
    input wire mgmt_clk,host_clk,device_clk,cold_rst_n,
    input wire req_valid,
    input wire [63:0] generation,tag,
    output wire close_host,host_proof_request,
    output wire [63:0] host_proof_tag,
    input wire host_drain_ack,
    input wire [63:0] host_drain_tag,
    input wire shutdown_active,
    input wire [9:0] host_bus,device_bus,
    output wire section_ack,upstream_certificate,pair_certificate,
    output wire host_fault,device_fault,
    output wire host_mailbox_busy,pair_host_busy,pair_device_busy,
    output wire [7:0] diagnostics
);
    wire [2:0] hk,ak;
    wire [63:0] hg,ag,at;
    wire hcert,dcert;
    wire [3:0] ha,hw,hq,da,dw,dq;
    reg retained_close=0;
    assign close_host=retained_close || host_proof_request;
    always @(posedge host_clk)begin
        if(!cold_rst_n)retained_close<=0;
        else if(host_proof_request)retained_close<=1;
    end
    ntrpu_tagged_reset_mailbox upstream_mailbox(
        .mgmt_clk(mgmt_clk),.peer_clk(host_clk),.req_valid(req_valid),.req_kind(3'd1),
        .req_generation(generation),.req_tag(tag),.ack_valid(upstream_certificate),
        .ack_kind(ak),.ack_generation(ag),.ack_tag(at),.busy(host_mailbox_busy),
        .obsolete_completions(),.peer_req_valid(host_proof_request),
        .peer_req_kind(hk),.peer_req_generation(hg),.peer_req_tag(host_proof_tag),
        .peer_ack_valid(host_drain_ack && shutdown_active && hk==1),
        .peer_ack_kind(3'd1),.peer_ack_generation(hg),.peer_ack_tag(host_drain_tag),
        .rejected_peer_ack_cycles());
    ntrpu_axil_pair_drain pair(
        .mgmt_clk(mgmt_clk),.host_clk(host_clk),.device_clk(device_clk),.cold_rst_n(cold_rst_n),
        .req_valid(req_valid && upstream_certificate),.generation(generation),.tag(tag),
        .host_source_fenced(shutdown_active),.host_bus(host_bus),.device_bus(device_bus),
        .external_drain_ack(pair_certificate),.host_certificate(hcert),.device_certificate(dcert),
        .host_mailbox_busy(pair_host_busy),.device_mailbox_busy(pair_device_busy),
        .host_fault(host_fault),.device_fault(device_fault),
        .host_aw_pending(ha),.host_w_pending(hw),.host_ar_pending(hq),
        .device_aw_pending(da),.device_w_pending(dw),.device_ar_pending(dq));
    assign section_ack=upstream_certificate && pair_certificate;
    assign diagnostics={hcert,dcert,|ha,|hw,|hq,|da,|dw,|dq};
endmodule
