`timescale 1ns / 1ps
// E: effects and admission row selects share the same capture edge.
// No extra pipeline stage; unknown comparisons remain untaken procedural branches.
module ntrpu_pcie_symbolic_effects_predecode_row_enable_r1 (
    input wire clk, cold_rst_n,
    input ntrpu_pcie_symbolic_ced_pkg::c_record_t in_c,
    output ntrpu_pcie_symbolic_ced_pkg::e_record_t out_e,
    output logic [511:0] out_admission_matches,
    output logic [255:0] out_row_enables, out_attribute_enables
);
    import ntrpu_pcie_metadata_effect_pkg::*;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    e_record_t next_e;
    logic [511:0] next_admission_matches;
    logic [255:0] next_row_enables, next_attribute_enables;
    integer slot, admission, row;
    initial begin out_e='0; out_admission_matches='0; out_row_enables='0; out_attribute_enables='0; end
    function automatic command_t prepare_operation(input operation_t op,
        input append_consume, input enabled, input [10:0] words);
        effect_t e0,e1,e2,e3,prefix_effect,summary;
        command_t result;
        begin
            result='0; result.retained=op.retained; result.tag=op.tag;
            summary='0; prefix_effect='0;
            e0=encode_effect(op.mask[0],1'b0,30'b0,in_c.ops.prefix_counts[0+:11]);
            e1=encode_effect(op.mask[1],1'b0,30'b0,in_c.ops.prefix_counts[11+:11]);
            e2=encode_effect(op.mask[2],1'b0,30'b0,in_c.ops.prefix_counts[22+:11]);
            e3=encode_effect(op.mask[3],1'b0,30'b0,in_c.ops.prefix_counts[33+:11]);
            if (op.prefix) begin
                prefix_effect=compose_effect(compose_effect(e0,e1),compose_effect(e2,e3));
                summary=prefix_effect;
                if (op.retained)
                    summary=compose_effect(encode_effect(in_c.record.pre_rc_context.delivery_enabled,
                        1'b0,30'b0,in_c.record.pre_rc_context.words),prefix_effect);
                result.apply_summary=1;
            end
            if (append_consume) begin
                summary=compose_effect(summary,encode_effect(enabled,1'b0,30'b0,words));
                result.apply_summary=1;
            end
            result.summary=summary[25:0];
            prepare_operation=result;
        end
    endfunction
    always_comb begin
        next_e='0; next_e.origin=in_c;
        next_admission_matches='0;
        next_row_enables='0; next_attribute_enables='0;
        for (admission=0;admission<2;admission=admission+1) begin
            for (row=0;row<256;row=row+1) begin
                if (in_c.ops.admissions[admission].valid &&
                    in_c.ops.admissions[admission].tag==row)
                    next_admission_matches[admission*256+row]=1'b1;
            end
        end
        // Capture row-local write decisions with the original E record.
        // Procedural branches keep unknown conditions untaken, as in D.
        if (in_c.record.valid) begin
            if (in_c.record.pre_observing) begin
                for (row=0;row<256;row=row+1) begin
                    if (1'b0)
                        next_row_enables[row]=1'b1;
                    for (admission=0;admission<2;admission=admission+1) begin
                        if (next_admission_matches[admission*256+row]) begin
                            next_row_enables[row]=1'b1;
                            next_attribute_enables[row]=1'b1;
                        end
                    end
                end
            end
        end
        for (slot=0;slot<4;slot=slot+1) begin
            next_e.queries[slot]=prepare_operation(in_c.ops.queries[slot],1'b0,1'b0,11'b0);
            next_e.eops[slot]=prepare_operation(in_c.ops.eops[slot],1'b0,1'b0,11'b0);
            next_e.updates[slot]=prepare_operation(in_c.ops.updates[slot],1'b1,
                in_c.record.eops[slot].delivery_enabled,in_c.record.eops[slot].words);
        end
        next_e.context_after=prepare_operation(in_c.ops.context_after,1'b0,1'b0,11'b0);
    end
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            out_e<='0;
            out_admission_matches<='0;
            out_row_enables<='0; out_attribute_enables<='0;
        end else begin
            out_e<=next_e;
            out_admission_matches<=next_admission_matches;
            out_row_enables<=next_row_enables; out_attribute_enables<=next_attribute_enables;
        end
    end
endmodule
