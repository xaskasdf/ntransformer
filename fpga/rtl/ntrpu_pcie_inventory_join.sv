`timescale 1ns / 1ps
// Full public inventory retired one cycle after the canonical producer commit.
// Sequence is diagnostic only. Cold reset cancels the pending publication.
// This passive observer does not certify hidden core queues or posted receipt.
module ntrpu_pcie_inventory_join (
    input wire clk, cold_rst_n,
    input wire [31:0] sequence_in,
    input wire profile_valid, user_reset, link_up,
    input wire [15:0] requester_id,
    input wire [1:0] attribute_mask,
    input wire rq_valid, rc_valid, rc_ready,
    input wire [3:0] rq_ready,
    input wire [511:0] rq_data, rc_data,
    input wire [136:0] rq_user,
    input wire [160:0] rc_user,
    output reg observing=0,
    output reg [255:0] allocated_tags=0,
    output reg [8:0] tag_count=0, outstanding_reads=0,
    output reg [20:0] remaining_bytes=0,
    output reg rq_partial=0, rc_partial=0,
    output reg [31:0] requests=0, completed_reads=0, posted_writes=0,
    output reg retired_valid=0,
    output reg [31:0] retired_sequence=0,
    output wire [15:0] fault_sources,
    output wire protocol_fault, inventory_empty, pipeline_pending
);
    wire p_observing, p_rq_partial, p_rc_partial, p_pending;
    wire [255:0] p_tags;
    wire [8:0] p_tag_count, p_reads;
    wire [20:0] p_bytes;
    wire [31:0] p_requests, p_completed, p_writes;
    wire [15:0] p_faults;
    wire [3:0] metadata_owned;
    wire [119:0] metadata_data;
    wire [51:0] canonical_delivered;
    wire [70:0] p_rc_context;
    wire [42:0] p_rq_context;
    ntrpu_pcie_inventory_scheduler producer (
        .clk(clk),.cold_rst_n(cold_rst_n),.sequence_in(sequence_in),
        .profile_valid(profile_valid),.user_reset(user_reset),.link_up(link_up),
        .requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rq_ready(rq_ready),.rc_valid(rc_valid),.rc_ready(rc_ready),
        .rq_data(rq_data),.rq_user(rq_user),.rc_data(rc_data),.rc_user(rc_user),
        .observing(p_observing),.allocated_tags(p_tags),.tag_count(p_tag_count),
        .outstanding_reads(p_reads),.remaining_bytes(p_bytes),
        .rq_partial(p_rq_partial),.rc_partial(p_rc_partial),.requests(p_requests),
        .completed_reads(p_completed),.posted_writes(p_writes),
        .early_fault_sources(p_faults),.state_pending(p_pending),
        .metadata_owned(metadata_owned),.metadata_data(metadata_data),
        .canonical_delivered(canonical_delivered),
        .retained_rc_context(p_rc_context),.retained_rq_context(p_rq_context)
    );
    reg [15:0] saved_requester=0, captured_requester=0;
    reg [1:0] saved_attribute_mask=0;
    reg snapshot_valid=0, captured_observing=0, capture_effect_pending=0;
    reg [31:0] captured_sequence=0;
    reg captured_rq_partial=0, captured_rc_partial=0;
    reg captured_rq_valid=0, captured_rc_valid=0, captured_rc_ready=0;
    reg [3:0] captured_rq_ready=0;
    reg [511:0] captured_rq_data=0, captured_rc_data=0;
    reg [136:0] captured_rq_user=0;
    reg [160:0] captured_rc_user=0;
    reg [70:0] captured_rc_context=0;
    reg [42:0] captured_rq_context=0;
    reg [15:0] retired_early_faults=0, validation_latched=0;
    wire [15:0] validation_faults;
    ntrpu_pcie_inventory_validation validation (
        .observing(captured_observing),.expected_requester(captured_requester),
        .rq_partial(captured_rq_partial),.rc_partial(captured_rc_partial),
        .rq_valid(captured_rq_valid),.rq_ready(captured_rq_ready),
        .rc_valid(captured_rc_valid),.rc_ready(captured_rc_ready),
        .rq_data(captured_rq_data),.rq_user(captured_rq_user),
        .rc_data(captured_rc_data),.rc_user(captured_rc_user),
        .rc_context(captured_rc_context),.rq_context(captured_rq_context),
        .metadata_owned(metadata_owned),.metadata_data(metadata_data),
        .canonical_delivered(canonical_delivered),.faults(validation_faults)
    );
    assign fault_sources=retired_early_faults | validation_latched;
    assign protocol_fault=|fault_sources;
    assign pipeline_pending=(cold_rst_n===1'b1) &&
        ((p_pending!==1'b0) || capture_effect_pending);
    // Current input activity revokes empty combinationally, before capture.
    // Idle snapshots are diagnostic records and do not keep validation busy.
    assign inventory_empty=(cold_rst_n===1'b1) && retired_valid && observing &&
        (profile_valid===1'b1) && (user_reset===1'b0) && (link_up===1'b1) &&
        (requester_id===saved_requester) && (attribute_mask===saved_attribute_mask) &&
        (rq_valid===1'b0) && (rc_valid===1'b0) &&
        ((rq_ready===4'b0000) || (rq_ready===4'b1111)) &&
        ((rc_ready===1'b0) || (rc_ready===1'b1)) &&
        (fault_sources===16'b0) && (p_faults===16'b0) && !pipeline_pending &&
        (tag_count===9'b0) && (outstanding_reads===9'b0) && (remaining_bytes===21'b0) &&
        (rq_partial===1'b0) && (rc_partial===1'b0);

    always @(posedge clk) begin
        if (!cold_rst_n) begin
            observing<=0; allocated_tags<=0; tag_count<=0; outstanding_reads<=0;
            remaining_bytes<=0; rq_partial<=0; rc_partial<=0;
            requests<=0; completed_reads<=0; posted_writes<=0;
            retired_valid<=0; retired_sequence<=0;
            retired_early_faults<=0; validation_latched<=0;
            saved_requester<=0; saved_attribute_mask<=0; captured_requester<=0;
            snapshot_valid<=0; captured_sequence<=0; captured_observing<=0; capture_effect_pending<=0;
            captured_rq_partial<=0; captured_rc_partial<=0;
            captured_rq_valid<=0; captured_rc_valid<=0; captured_rc_ready<=0; captured_rq_ready<=0;
            captured_rq_data<=0; captured_rc_data<=0; captured_rq_user<=0; captured_rc_user<=0;
            captured_rc_context<=0; captured_rq_context<=0;
        end else begin
            if (!p_observing && profile_valid===1'b1 && user_reset===1'b0 && link_up===1'b1) begin
                saved_requester<=requester_id; saved_attribute_mask<=attribute_mask;
            end
            snapshot_valid<=1; captured_sequence<=sequence_in;
            captured_observing<=p_observing; captured_requester<=saved_requester;
            capture_effect_pending<=p_observing && ((rq_valid!==1'b0) || (rc_valid!==1'b0));
            captured_rq_partial<=p_rq_partial; captured_rc_partial<=p_rc_partial;
            captured_rq_valid<=rq_valid; captured_rq_ready<=rq_ready;
            captured_rc_valid<=rc_valid; captured_rc_ready<=rc_ready;
            captured_rq_data<=rq_data; captured_rq_user<=rq_user;
            captured_rc_data<=rc_data; captured_rc_user<=rc_user;
            captured_rc_context<=p_rc_context; captured_rq_context<=p_rq_context;
            retired_valid<=snapshot_valid;
            retired_sequence<=snapshot_valid ? captured_sequence : 0;
            observing<=p_observing; allocated_tags<=p_tags; tag_count<=p_tag_count;
            outstanding_reads<=p_reads; remaining_bytes<=p_bytes;
            rq_partial<=p_rq_partial; rc_partial<=p_rc_partial;
            requests<=p_requests; completed_reads<=p_completed; posted_writes<=p_writes;
            retired_early_faults<=p_faults;
            validation_latched<=validation_latched | validation_faults;
        end
    end
endmodule
