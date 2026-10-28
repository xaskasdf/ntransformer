`timescale 1ns / 1ps
// Certificate for an invalidated, previously established NTRPU ownership epoch.
// All observed owner inputs belong to source_clk. generation MUST be that
// actual epoch (nonzero), shared by every provider participating in the round.
// This is not a drain certificate: held CMD/DATA/result/leases remain obligations.
// No warm reset, abort, retirement, or new ownership is commanded here.
module ntrpu_ownership_certificate (
    input wire mgmt_clk,source_clk,cold_rst_n,req_valid,
    input wire [63:0] generation,tag,
    input wire session_lost,session_invalid,
    input wire [63:0] collector_epoch,bridge_epoch,
    input wire collector_lease_held,bridge_lease_held,
    input wire arm_ready,retire_ready,line_ready,
    input wire [1:0] cpl_ready,
    output wire ack_valid,
    output wire [63:0] ack_generation,ack_tag,
    output wire busy,peer_request,peer_answer,
    output wire [63:0] peer_generation,peer_tag,
    output wire fault_mgmt,
    output reg revoked_source=0,
    output wire [31:0] obsolete_completions
);
    wire [2:0] peer_kind;
    wire cached,synchronized_revocation;
    reg captured=0,answer=0,ever_certified=0,retained_fault=0;
    reg [63:0] saved_generation=0,saved_tag=0,bound_epoch=0;
    reg [1:0] quiet=0,bound_leases=0;
    wire [1:0] leases={collector_lease_held,bridge_lease_held};
    wire observed_invalid=session_lost && session_invalid && collector_epoch!=0 &&
        collector_epoch==bridge_epoch && !arm_ready && !retire_ready &&
        !line_ready && !(|cpl_ready);
    wire identity_matches=peer_generation==collector_epoch && peer_tag!=0;
    wire peer_current=peer_request && captured && peer_kind==1 &&
        peer_generation==saved_generation && peer_tag==saved_tag;
    wire ownership_stable=!ever_certified ||
        (collector_epoch==bound_epoch && leases==bound_leases);
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) synchronize_revocation(
        .src_clk(1'b0),.src_in(revoked_source),
        .dest_clk(mgmt_clk),.dest_out(synchronized_revocation));
    always @(posedge mgmt_clk) begin
        if(!cold_rst_n)retained_fault<=0;
        else retained_fault<=retained_fault | synchronized_revocation;
    end
    assign fault_mgmt=retained_fault | synchronized_revocation;
    assign ack_valid=cold_rst_n && cached && !fault_mgmt;
    assign peer_answer=cold_rst_n && answer && peer_current && observed_invalid &&
        identity_matches && ownership_stable && !revoked_source;
    ntrpu_tagged_reset_mailbox mailbox(
        .mgmt_clk(mgmt_clk),.peer_clk(source_clk),
        .req_valid(cold_rst_n && req_valid && !fault_mgmt),.req_kind(3'd1),
        .req_generation(generation),.req_tag(tag),.ack_valid(cached),.ack_kind(),
        .ack_generation(ack_generation),.ack_tag(ack_tag),.busy(busy),
        .obsolete_completions(obsolete_completions),
        .peer_req_valid(peer_request),.peer_req_kind(peer_kind),
        .peer_req_generation(peer_generation),.peer_req_tag(peer_tag),
        .peer_ack_valid(peer_answer),.peer_ack_kind(3'd1),
        .peer_ack_generation(saved_generation),.peer_ack_tag(saved_tag),
        .rejected_peer_ack_cycles());
    always @(posedge source_clk) begin
        // Configuration initialization only. Expected ownership loss is proof,
        // while contradictory ownership observations revoke it permanently.
        if(!cold_rst_n)begin
            captured<=0;answer<=0;ever_certified<=0;revoked_source<=0;
            saved_generation<=0;saved_tag<=0;bound_epoch<=0;bound_leases<=0;quiet<=0;
        end else begin
            if(ever_certified && (!observed_invalid || !ownership_stable))revoked_source<=1;
            if(!peer_request)begin captured<=0;answer<=0;quiet<=0;end
            else if(!peer_current)begin
                captured<=1;answer<=0;quiet<=0;
                saved_generation<=peer_generation;saved_tag<=peer_tag;
            end else if(!observed_invalid || !identity_matches || !ownership_stable || revoked_source)begin
                answer<=0;quiet<=0;
            end else if(quiet<3)quiet<=quiet+1'b1;
            else begin
                answer<=1;ever_certified<=1;
                if(!ever_certified)begin bound_epoch<=collector_epoch;bound_leases<=leases;end
            end
        end
    end
endmodule
