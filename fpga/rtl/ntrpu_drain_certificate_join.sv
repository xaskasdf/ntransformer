`timescale 1ns / 1ps
// Management-domain composition only. Inputs must already be coherent,
// revocable certificates, NOT raw asynchronous status or software MMIO bits.
// Slots: ownership invalid, host closed, protected XDMA inventory,
// AXI-Lite section, AXI4 section, AXI-Lite local replies, AXI4 local replies.
// The last two requests follow the first five certificates. Missing native
// contracts stay absent. This module neither drives reset nor reopens traffic.
module ntrpu_drain_certificate_join (
    input wire mgmt_clk, cold_rst_n, req_valid,
    input wire [63:0] generation, tag,
    input wire [6:0] contract_present, peer_ready, proof_fault, ack_valid,
    input wire [447:0] ack_generation, ack_tag,
    output reg [6:0] proof_request,
    output wire [63:0] proof_generation, proof_tag,
    output reg complete_ack,
    output wire [63:0] complete_generation, complete_tag,
    output wire [6:0] accepted_mask, armed_mask, retained_faults,
    output wire revoked, identity_fault
);
    reg have_identity=0, active=0, poisoned=0, replay_fault=0;
    reg [63:0] saved_generation=0, saved_tag=0;
    reg [6:0] armed=0, seen=0, faults=0;
    reg [6:0] good;
    reg current, new_identity_allowed;
    wire any_fault=(|faults) || (|proof_fault) || replay_fault;
    assign proof_generation=saved_generation;
    assign proof_tag=saved_tag;
    assign complete_generation=saved_generation;
    assign complete_tag=saved_tag;
    assign accepted_mask=good;
    assign armed_mask=armed;
    assign retained_faults=faults | proof_fault;
    assign revoked=poisoned;
    assign identity_fault=replay_fault;

    always @* begin
        current=0;
        new_identity_allowed=0;
        proof_request=0;
        good=0;
        complete_ack=0;
        // If a four-state input is unknown, the procedural condition does not
        // approve it. This is simulation containment, not a physical X sensor.
        if(cold_rst_n && req_valid && active && have_identity &&
           generation==saved_generation && tag==saved_tag && !any_fault && !poisoned)
            current=1;
        if(generation!=0 && tag!=0 && (!have_identity ||
           generation>saved_generation || (generation==saved_generation && tag>saved_tag)))
            new_identity_allowed=1;
        for(integer i=0;i<5;i=i+1) begin
            if(current && contract_present[i] && peer_ready[i]) proof_request[i]=1;
            if(proof_request[i] && armed[i] && ack_valid[i] &&
               ack_generation[i*64+:64]==saved_generation && ack_tag[i*64+:64]==saved_tag)
                good[i]=1;
        end
        for(integer i=5;i<7;i=i+1) begin
            if(current && (&good[4:0]) && contract_present[i] && peer_ready[i]) proof_request[i]=1;
            if(proof_request[i] && armed[i] && ack_valid[i] &&
               ack_generation[i*64+:64]==saved_generation && ack_tag[i*64+:64]==saved_tag)
                good[i]=1;
        end
        if(current && (&good)) complete_ack=1;
    end

    always @(posedge mgmt_clk) begin
        // Configuration initialization only, never endpoint/local runtime reset.
        if(!cold_rst_n) begin
            have_identity<=0;active<=0;poisoned<=0;replay_fault<=0;
            saved_generation<=0;saved_tag<=0;armed<=0;seen<=0;faults<=0;
        end else begin
            faults<=faults | proof_fault;
            if(!req_valid) begin active<=0;armed<=0;seen<=0;poisoned<=0;end
            else if(!active || generation!=saved_generation || tag!=saved_tag) begin
                active<=0;armed<=0;seen<=0;poisoned<=0;
                if(new_identity_allowed && !any_fault) begin
                    have_identity<=1;active<=1;saved_generation<=generation;saved_tag<=tag;
                end else replay_fault<=1;
            end else if(current) begin
                for(integer i=0;i<7;i=i+1)
                    if(proof_request[i] && ack_valid[i]==1'b0) armed[i]<=1;
                seen<=seen | good;
                // A previously accepted certificate is revocable. Restoring
                // its level cannot resurrect this round: use a fresh identity.
                if(|(seen & ~good)) poisoned<=1;
            end
        end
    end
endmodule
