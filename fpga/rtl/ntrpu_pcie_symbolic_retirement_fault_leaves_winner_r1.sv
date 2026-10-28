`timescale 1ns / 1ps
// R: retire one captured B origin per clock. No lookup of current tag state.
// Public state has five clocks of latency from C. Only cold reset cancels work.
// This module does not yet issue a quiet/ACK certificate.
module ntrpu_pcie_symbolic_retirement_fault_leaves_winner_r1 (
    input wire clk, cold_rst_n,
    input ntrpu_pcie_symbolic_numeric_pkg_r1::b_record_t in_b,
    output reg out_valid=0,
    output reg [31:0] sequence_out=0,
    output reg effect_pending=0,
    output ntrpu_pcie_symbolic_ced_pkg::control_t out_control,
    output reg [20:0] remaining_bytes=0,
    output wire [15:0] fault_sources,
    output wire protocol_fault
);
    import ntrpu_pcie_symbolic_ced_pkg::*;
    control_t saved_control='0;
    assign out_control=saved_control;
    wire [15:0] late_faults;
    ntrpu_pcie_symbolic_late_fault_leaves_winner_r1 late_i(.in_b(in_b),.faults(late_faults));
    // Preserve R7's signed 16-bit RQ sum and balanced unsigned 15-bit RC sum.
    // B emits zero for invalid EOP slots; its checked records have distinct lanes.
    wire signed [15:0] rq_span_sum=$signed(in_b.origin.record.rq_spans[0])+
        $signed(in_b.origin.record.rq_spans[1]);
    wire [14:0] rc_span_sum=
        ({2'b0,in_b.eop_removed[0]}+{2'b0,in_b.eop_removed[1]})+
        ({2'b0,in_b.eop_removed[2]}+{2'b0,in_b.eop_removed[3]});
    wire signed [22:0] candidate=$signed({2'b0,remaining_bytes})+
        $signed(rq_span_sum)-$signed({1'b0,rc_span_sum});
    reg [1:0] byte_range_high=0;
    reg range_checked=0;
    reg [15:0] fault_latched=0;
    wire signed [22:0] checked_bytes=$signed({byte_range_high,remaining_bytes});
    wire byte_range_fault=range_checked &&
        ((checked_bytes<0 || checked_bytes>1052672)===1'b1);
    assign fault_sources=fault_latched | {2'b0,byte_range_fault,13'b0};
    assign protocol_fault=|fault_sources;

    always @(posedge clk) begin
        if (!cold_rst_n) begin
            out_valid<=0;sequence_out<=0;effect_pending<=0;saved_control<='0;
            remaining_bytes<=0;byte_range_high<=0;range_checked<=0;fault_latched<=0;
        end else begin
            out_valid<=in_b.origin.record.valid;
            sequence_out<=0;effect_pending<=0;
            if (in_b.origin.record.valid) begin
                sequence_out<=in_b.origin.record.sequence_number;
                effect_pending<=in_b.origin.record.effect_pending;
                saved_control<=in_b.origin.control;
                // Previous visible byte-range fault must survive a later in-range beat.
                fault_latched<=fault_sources | in_b.origin.control.early_fault_sources | late_faults;
                // Arming publishes control but does not execute a byte transition.
                if (in_b.origin.record.pre_observing) begin
                    remaining_bytes<=candidate[20:0];
                    byte_range_high<=candidate[22:21];
                    range_checked<=1;
                end
            end
        end
    end
endmodule
