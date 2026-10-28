`timescale 1ns / 1ps
// Three registered boundaries; externally aligned S1 diagnostic state is two cycles later.
// Numeric validation, byte retirement and a complete empty certificate are downstream.
module ntrpu_pcie_symbolic_frontend_r3_ced_control_r3_mapped_reference_r1 (
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
    output wire rq_partial, rc_partial,
    output wire [31:0] requests, completed_reads, posted_writes,
    output wire [15:0] early_fault_sources,
    output wire early_protocol_fault, control_pending,
    output wire pipeline_pending,
    output ntrpu_pcie_symbolic_frontend_pkg::s1_record_t out_record
);
    import ntrpu_pcie_symbolic_ced_pkg::*;
    wire c_record_t c_record;
    wire e_record_t e_record;
    wire control_t d_control;
    ntrpu_pcie_symbolic_control_r3_ced_control_r3_mapped_reference_r1 c_stage (
        .clk(clk),.cold_rst_n(cold_rst_n),.sequence_in(sequence_in),
        .profile_valid(profile_valid),.user_reset(user_reset),.link_up(link_up),
        .requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user),.out_c(c_record));
    ntrpu_pcie_symbolic_effects_r2_ced_control_r3_mapped_reference_r1 e_stage(.clk(clk),.cold_rst_n(cold_rst_n),.in_c(c_record),.out_e(e_record));
    ntrpu_pcie_symbolic_descriptions_r2_ced_control_r3_mapped_reference_r1 d_stage(.clk(clk),.cold_rst_n(cold_rst_n),.in_e(e_record),
        .out_control(d_control),.out_record(out_record));
    assign {observing,allocated_tags,tag_count,outstanding_reads,rq_partial,rc_partial,
        requests,completed_reads,posted_writes,early_fault_sources,early_protocol_fault}=d_control[390:1];
    // Cold cancels queued origins, but current raw activity still revokes quiet.
    assign control_pending=!cold_rst_n ? (rq_valid || rc_valid) : d_control.control_pending;
    assign pipeline_pending=rq_valid || rc_valid || c_record.record.effect_pending ||
        e_record.origin.record.effect_pending || out_record.effect_pending;
endmodule
