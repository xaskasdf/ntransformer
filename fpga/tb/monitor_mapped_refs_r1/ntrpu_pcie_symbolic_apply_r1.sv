`timescale 1ns / 1ps
// A: evaluate all nine descriptions once; only cold reset cancels an origin.
module ntrpu_pcie_symbolic_apply_r1_monitor_mapped_reference_r1 (
    input wire clk, cold_rst_n,
    input ntrpu_pcie_symbolic_ced_pkg::control_t in_control,
    input ntrpu_pcie_symbolic_frontend_pkg::s1_record_t in_record,
    output ntrpu_pcie_symbolic_numeric_pkg_r1::a_record_t out_a
);
    import ntrpu_pcie_symbolic_numeric_pkg_r1::*;
    a_record_t next_a;
    integer slot;
    function automatic numeric_value_t evaluate(input logic [50:0] description);
        logic [29:0] result;
        begin
            result=ntrpu_pcie_metadata_effect_pkg::apply_effect(
                {31'b0,description[25:0]}, {5'b0,description[50:26]});
            evaluate={result[12:0],result[24:13]};
        end
    endfunction
    always @* begin
        next_a='0;
        if (in_record.valid) begin
            next_a.control=in_control;
            next_a.record=in_record;
            for (slot=0;slot<4;slot=slot+1) begin
                if (in_record.queries[slot].valid)
                    next_a.queries[slot]=evaluate(in_record.queries[slot].description);
                if (in_record.eops[slot].valid)
                    next_a.eops[slot]=evaluate(in_record.eops[slot].description);
            end
            next_a.context_value=evaluate(in_record.pre_rc_context.description);
        end
    end
    always @(posedge clk) begin
        if (!cold_rst_n) out_a<='0;
        else out_a<=next_a;
    end
endmodule
