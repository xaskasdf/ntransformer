`timescale 1ns / 1ps
// Configuration-lifetime closure and revocable management certificate for the
// AXI-Lite admission/shutdown/CDC section. NOT permission to reset XDMA or rearm
// ownership: local error responses and protected endpoint queues are excluded.
// cold_rst_n is configuration initialization only, never operational reset.
module ntrpu_axil_fence_certificate_r4 (
    input wire mgmt_clk,host_clk,device_clk,cold_rst_n,
    input wire req_valid,
    input wire [63:0] generation,tag,
    output wire close_host,host_proof_request,
    output wire [63:0] host_proof_tag,
    input wire host_drain_ack,
    input wire [63:0] host_drain_tag,
    input wire shutdown_active,
    input wire [1:0] admission_fault_sources,
    input wire [9:0] host_bus,device_bus,
    output wire section_ack,upstream_certificate,pair_certificate,
    output wire host_fault,device_fault,
    output wire [2:0] fault_mgmt,
    output wire host_mailbox_busy,pair_host_busy,pair_device_busy,
    output wire [7:0] diagnostics
);
    wire [2:0] hk,ak;
    wire [63:0] hg,ag,at;
    wire hcert,dcert,upstream_cached,pair_cached;
    wire [3:0] ha,hw,hq,da,dw,dq;
    wire [2:0] synchronized_faults;
    reg [2:0] retained_faults=0;
    reg retained_close=0;
    wire fault_close_host;
    wire any_fault=|fault_mgmt;
    wire accepted_request=req_valid && !any_fault;

    // Sources are retained, monotonic fault levels in their respective domains.
    // No extra source register: the source clock may stop immediately after
    // reporting a fault. Three destination stages continue on management clock.
    wire admission_fault=|admission_fault_sources;
    wire [1:0] admission_faults_mgmt;
    // Each input is a separate retained host-domain register. Combining them
    // only after destination synchronization avoids a combinational CDC source
    // without adding a host edge that might never arrive after clock loss.
    for(genvar bit_index=0;bit_index<2;bit_index=bit_index+1)begin:admission_sources
        xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
            .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) synchronize_fault(
            .src_clk(1'b0),.src_in(admission_fault_sources[bit_index]),
            .dest_clk(mgmt_clk),.dest_out(admission_faults_mgmt[bit_index]));
    end
    assign synchronized_faults[0]=|admission_faults_mgmt;
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) host_fault_cdc(
        .src_clk(1'b0),.src_in(host_fault),
        .dest_clk(mgmt_clk),.dest_out(synchronized_faults[1]));
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) device_fault_cdc(
        .src_clk(1'b0),.src_in(device_fault),
        .dest_clk(mgmt_clk),.dest_out(synchronized_faults[2]));
    always @(posedge mgmt_clk)begin
        if(!cold_rst_n)retained_faults<=0;
        else retained_faults<=retained_faults | synchronized_faults;
    end
    // Revoke at the synchronizer output; retention cannot add another cycle.
    // Faults persist across request cancellation, generation changes and resets
    // of the local datapath. Existing mailbox handshakes are allowed to stall;
    // accepted_request suppresses new work without truncating XPM src_send.
    assign fault_mgmt=retained_faults | synchronized_faults;
    assign upstream_certificate=upstream_cached && !any_fault;
    assign pair_certificate=pair_cached && !any_fault;
    assign section_ack=upstream_certificate && pair_certificate;
    // Management is the continuously running domain. Register its aggregate
    // before the host CDC; this does not delay management ACK revocation.
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(1)) fault_close_cdc(
        .src_clk(mgmt_clk),.src_in(any_fault),
        .dest_clk(host_clk),.dest_out(fault_close_host));
    // A fault can close admission even before a management proof request.
    // The admission producer itself finishes its finite held-VALID prefix.
    assign close_host=retained_close || host_proof_request ||
                      admission_fault || host_fault || fault_close_host;
    always @(posedge host_clk)begin
        if(!cold_rst_n)retained_close<=0;
        else if(host_proof_request || admission_fault || host_fault || fault_close_host)
            retained_close<=1;
    end
    ntrpu_tagged_reset_mailbox upstream_mailbox(
        .mgmt_clk(mgmt_clk),.peer_clk(host_clk),.req_valid(accepted_request),.req_kind(3'd1),
        .req_generation(generation),.req_tag(tag),.ack_valid(upstream_cached),
        .ack_kind(ak),.ack_generation(ag),.ack_tag(at),.busy(host_mailbox_busy),
        .obsolete_completions(),.peer_req_valid(host_proof_request),
        .peer_req_kind(hk),.peer_req_generation(hg),.peer_req_tag(host_proof_tag),
        .peer_ack_valid(host_drain_ack && shutdown_active && hk==1),
        .peer_ack_kind(3'd1),.peer_ack_generation(hg),.peer_ack_tag(host_drain_tag),
        .rejected_peer_ack_cycles());
    ntrpu_axil_pair_drain pair(
        .mgmt_clk(mgmt_clk),.host_clk(host_clk),.device_clk(device_clk),.cold_rst_n(cold_rst_n),
        .req_valid(accepted_request && upstream_certificate),.generation(generation),.tag(tag),
        .host_source_fenced(shutdown_active),.host_bus(host_bus),.device_bus(device_bus),
        .external_drain_ack(pair_cached),.host_certificate(hcert),.device_certificate(dcert),
        .host_mailbox_busy(pair_host_busy),.device_mailbox_busy(pair_device_busy),
        .host_fault(host_fault),.device_fault(device_fault),
        .host_aw_pending(ha),.host_w_pending(hw),.host_ar_pending(hq),
        .device_aw_pending(da),.device_w_pending(dw),.device_ar_pending(dq));
    assign diagnostics={hcert,dcert,|ha,|hw,|hq,|da,|dw,|dq};
endmodule
