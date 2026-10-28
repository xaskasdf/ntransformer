`timescale 1ns / 1ps
// Complete public C/E/D/A/B/R composition; five clocks from C to retirement.
// Prototype: the quiet certificate requires its own integrated regression.
// Passive observation does not certify hidden PCIe queues or posted receipt.
module ntrpu_pcie_symbolic_monitor_predecode_r1 (
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
    output wire observing,
    output wire [255:0] allocated_tags,
    output wire [8:0] tag_count, outstanding_reads,
    output wire [20:0] remaining_bytes,
    output wire rq_partial, rc_partial,
    output wire [31:0] requests, completed_reads, posted_writes,
    output wire retired_valid,
    output wire [31:0] retired_sequence,
    output wire retired_effect_pending,
    output wire [15:0] fault_sources,
    output wire protocol_fault, pipeline_pending, inventory_empty
);
    import ntrpu_pcie_symbolic_frontend_pkg::*;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    import ntrpu_pcie_symbolic_numeric_pkg_r1::*;
    wire c_record_t c_record;
    wire e_record_t e_record;
    wire [511:0] admission_matches;
    wire control_t d_control, r_control;
    wire s1_record_t out_record;
    wire a_record_t numeric_a;
    wire b_record_t numeric_b;
    wire c_observing, c_control_pending;
    wire [15:0] c_early_fault_sources;

    ntrpu_pcie_symbolic_control_r3 c_stage (
        .clk(clk),.cold_rst_n(cold_rst_n),.sequence_in(sequence_in),
        .profile_valid(profile_valid),.user_reset(user_reset),.link_up(link_up),
        .requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user),
        .observing(c_observing),.early_fault_sources(c_early_fault_sources),
        .control_pending(c_control_pending),.out_c(c_record));
    ntrpu_pcie_symbolic_effects_predecode_r1 e_stage (
        .clk(clk),.cold_rst_n(cold_rst_n),.in_c(c_record),.out_e(e_record),.out_admission_matches(admission_matches));
    ntrpu_pcie_symbolic_descriptions_predecode_r1 d_stage (
        .clk(clk),.cold_rst_n(cold_rst_n),.in_e(e_record),
        .in_admission_matches(admission_matches),
        .out_control(d_control),.out_record(out_record));
    ntrpu_pcie_symbolic_apply_r1 apply_stage (
        .clk(clk),.cold_rst_n(cold_rst_n),.in_control(d_control),
        .in_record(out_record),.out_a(numeric_a));
    ntrpu_pcie_symbolic_delivery_r1 delivery_stage (
        .clk(clk),.cold_rst_n(cold_rst_n),.in_a(numeric_a),.out_b(numeric_b));
    ntrpu_pcie_symbolic_retirement_r1 retirement_i (
        .clk(clk),.cold_rst_n(cold_rst_n),.in_b(numeric_b),
        .out_valid(retired_valid),.sequence_out(retired_sequence),
        .effect_pending(retired_effect_pending),.out_control(r_control),
        .remaining_bytes(remaining_bytes),.fault_sources(fault_sources),.protocol_fault(protocol_fault));

    assign observing=r_control.observing;
    assign allocated_tags=r_control.allocated_tags;
    assign tag_count=r_control.tag_count;
    assign outstanding_reads=r_control.outstanding_reads;
    assign rq_partial=r_control.rq_partial;
    assign rc_partial=r_control.rc_partial;
    assign requests=r_control.requests;
    assign completed_reads=r_control.completed_reads;
    assign posted_writes=r_control.posted_writes;

    reg [15:0] saved_requester=0;
    reg [1:0] saved_attribute_mask=0;
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            saved_requester<=0;
            saved_attribute_mask<=0;
        end else if (!c_observing && profile_valid===1'b1 && user_reset===1'b0 && link_up===1'b1) begin
            saved_requester<=requester_id;
            saved_attribute_mask<=attribute_mask;
        end
    end

    // Only unretired stages contribute queued traffic. R's effect flag labels
    // the just-retired origin, and idle diagnostic records are not obligations.
    assign pipeline_pending=(cold_rst_n===1'b1) &&
        (1'b0 ||
         (c_record.record.effect_pending!==1'b0) ||
         (e_record.origin.record.effect_pending!==1'b0) ||
         (out_record.effect_pending!==1'b0) ||
         (numeric_a.record.effect_pending!==1'b0) ||
         (numeric_b.origin.record.effect_pending!==1'b0));
    // Current inputs revoke quiet before capture. Unknown values fail closed.
    assign inventory_empty=(cold_rst_n===1'b1) && (retired_valid===1'b1) &&
        (observing===1'b1) && (profile_valid===1'b1) &&
        (user_reset===1'b0) && (link_up===1'b1) &&
        (requester_id===saved_requester) && (attribute_mask===saved_attribute_mask) &&
        1'b1 && (rc_valid===1'b0) &&
        ((rq_ready===4'b0000) || (rq_ready===4'b1111)) &&
        ((rc_ready===1'b0) || (rc_ready===1'b1)) &&
        (c_early_fault_sources===16'b0) && (fault_sources===16'b0) &&
        (pipeline_pending===1'b0) && (allocated_tags===256'b0) &&
        (tag_count===9'b0) && (outstanding_reads===9'b0) &&
        (remaining_bytes===21'b0) && (rq_partial===1'b0) && (rc_partial===1'b0);
endmodule
