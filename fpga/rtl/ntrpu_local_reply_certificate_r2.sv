`timescale 1ns / 1ps
// Revocable proof of local replies after the upstream contracts and forwarded
// section have closed. local_idle includes partial AW/W and held source offers.
// Requests are proof messages, never commands to discard traffic or drive reset.
module ntrpu_local_reply_certificate_r2 #(parameter integer FAULT_WIDTH=7) (
    input wire mgmt_clk,host_clk,cold_rst_n,req_valid,
    input wire [63:0] generation,tag,
    input wire local_idle,request_present,reply_pending,
    input wire [FAULT_WIDTH-1:0] source_faults,
    // Already synchronized and retained by the associated real section.
    // This management-domain signal must include all admission source faults.
    input wire section_fault_mgmt,
    output wire ack_valid,
    output wire [63:0] ack_generation,ack_tag,
    output wire busy,peer_request,peer_answer,
    output wire [63:0] peer_generation,peer_tag,
    output wire [1:0] fault_mgmt,
    output reg revoked_source=0,
    output wire [31:0] obsolete_completions
);
    wire [2:0] peer_kind;
    wire cached;
    reg captured=0,answer=0,ever_certified=0;
    reg [63:0] saved_generation=0,saved_tag=0;
    reg [1:0] quiet=0;
    wire observed_idle=local_idle && !request_present && !reply_pending && !(|source_faults);
    wire peer_current=peer_request && captured && peer_kind==1 &&
        peer_generation==saved_generation && peer_tag==saved_tag;
    wire synchronized_revocation;
    reg [1:0] retained_faults=0;
    wire [1:0] synchronized_faults={synchronized_revocation,section_fault_mgmt};
    // Only the provider's own registered revocation crosses here. Reuse the
    // section's existing crossing for admission faults; duplicating its XPM
    // chains creates reconvergent observations of the same source registers.
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) synchronize_revocation(
        .src_clk(1'b0),.src_in(revoked_source),
        .dest_clk(mgmt_clk),.dest_out(synchronized_revocation));
    always @(posedge mgmt_clk)begin
        if(!cold_rst_n)retained_faults<=0;
        else retained_faults<=retained_faults | synchronized_faults;
    end
    assign fault_mgmt=retained_faults | synchronized_faults;
    assign ack_valid=cold_rst_n && cached && !(|fault_mgmt);
    assign peer_answer=cold_rst_n && answer && peer_current && observed_idle && !revoked_source;
    ntrpu_tagged_reset_mailbox mailbox(
        .mgmt_clk(mgmt_clk),.peer_clk(host_clk),
        .req_valid(cold_rst_n && req_valid && !(|fault_mgmt)),.req_kind(3'd1),
        .req_generation(generation),.req_tag(tag),.ack_valid(cached),.ack_kind(),
        .ack_generation(ack_generation),.ack_tag(ack_tag),.busy(busy),
        .obsolete_completions(obsolete_completions),
        .peer_req_valid(peer_request),.peer_req_kind(peer_kind),
        .peer_req_generation(peer_generation),.peer_req_tag(peer_tag),
        .peer_ack_valid(peer_answer),.peer_ack_kind(3'd1),
        .peer_ack_generation(saved_generation),.peer_ack_tag(saved_tag),.rejected_peer_ack_cycles());
    always @(posedge host_clk)begin
        // Configuration initialization only, never local/endpoint warm reset.
        if(!cold_rst_n)begin
            captured<=0;answer<=0;ever_certified<=0;revoked_source<=0;
            saved_generation<=0;saved_tag<=0;quiet<=0;
        end else begin
            if(ever_certified && !observed_idle)revoked_source<=1;
            if(!peer_request)begin captured<=0;answer<=0;quiet<=0;end
            else if(!peer_current)begin
                captured<=1;answer<=0;quiet<=0;
                saved_generation<=peer_generation;saved_tag<=peer_tag;
            end else if(!observed_idle || revoked_source)begin answer<=0;quiet<=0;end
            else if(quiet<3)quiet<=quiet+1'b1;
            else begin answer<=1;ever_certified<=1;end
        end
    end
endmodule
