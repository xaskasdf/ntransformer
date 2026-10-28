`timescale 1ns / 1ps
// B: arithmetic on A's copies. Captured enables/terminal bits keep four-state semantics.
module ntrpu_pcie_symbolic_delivery_r1 (
    input wire clk, cold_rst_n,
    input ntrpu_pcie_symbolic_numeric_pkg_r1::a_record_t in_a,
    output ntrpu_pcie_symbolic_numeric_pkg_r1::b_record_t out_b
);
    import ntrpu_pcie_symbolic_numeric_pkg_r1::*;
    b_record_t next_b;
    integer slot;
    function automatic logic [12:0] delivery(input numeric_value_t value,
        input logic [10:0] words, input logic enabled);
        logic signed [14:0] available, delivered;
        begin
            available=0; delivered=0;
            if (enabled) begin
                available=words*4-value[1:0];
                delivered=(available<value[24:12]) ? available : value[24:12];
            end
            delivery=delivered;
        end
    endfunction
    always @* begin
        next_b='0;
        if (in_a.record.valid) begin
            next_b.origin=in_a;
            for (slot=0;slot<4;slot=slot+1) begin
                if (in_a.record.queries[slot].valid)
                    next_b.query_delivered[slot]=delivery(in_a.queries[slot],
                        in_a.record.queries[slot].words,in_a.record.queries[slot].delivery_enabled);
                if (in_a.record.eops[slot].valid) begin
                    next_b.eop_delivered[slot]=delivery(in_a.eops[slot],
                        in_a.record.eops[slot].words,in_a.record.eops[slot].delivery_enabled);
                    next_b.eop_removed[slot]=in_a.record.eops[slot].terminal ?
                        in_a.eops[slot][24:12] : next_b.eop_delivered[slot];
                end
            end
            next_b.context_delivered=delivery(in_a.context_value,
                in_a.record.pre_rc_context.words,in_a.record.pre_rc_context.delivery_enabled);
        end
    end
    always @(posedge clk) begin
        if (!cold_rst_n) out_b<='0;
        else out_b<=next_b;
    end
endmodule
