`timescale 1ns / 1ps
// E: effects and admission row selects share the same capture edge.
// No extra pipeline stage; unknown comparisons remain untaken procedural branches.
module ntrpu_pcie_symbolic_effects_predecode_sum_r1 (
    input wire clk, cold_rst_n,
    input ntrpu_pcie_symbolic_ced_pkg::c_record_t in_c,
    output ntrpu_pcie_symbolic_ced_pkg::e_record_t out_e,
    output logic [511:0] out_admission_matches
);
    import ntrpu_pcie_metadata_effect_pkg::*;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    e_record_t next_e;
    logic [511:0] next_admission_matches;
    integer slot, admission, row;
    initial begin out_e='0; out_admission_matches='0; end
    function automatic command_t prepare_operation(input operation_t op,
        input append_consume, input enabled, input [10:0] words);
        command_t result;
        logic [25:0] summary;
        logic [5:0] active, unknown_words;
        logic [10:0] raw_words [0:5], enabled_words [0:5];
        logic [2:0] consumes;
        logic [11:0] sum01, sum23, sum45;
        logic [12:0] sum0123;
        logic [13:0] total;
        integer consume;
        begin
            result='0; result.retained=op.retained; result.tag=op.tag;
            summary='0; active='0; unknown_words='0;
            raw_words[0]=in_c.record.pre_rc_context.words;
            raw_words[1]=in_c.ops.prefix_counts[0+:11];
            raw_words[2]=in_c.ops.prefix_counts[11+:11];
            raw_words[3]=in_c.ops.prefix_counts[22+:11];
            raw_words[4]=in_c.ops.prefix_counts[33+:11];
            raw_words[5]=words;
            // Preserve each original procedural-if decision, including X/Z.
            if (op.prefix) begin
                if (op.retained)
                    if (in_c.record.pre_rc_context.delivery_enabled) active[0]=1;
                for (consume=0;consume<4;consume=consume+1)
                    if (op.mask[consume]) active[consume+1]=1;
                result.apply_summary=1;
            end
            if (append_consume) begin
                if (enabled) active[5]=1;
                result.apply_summary=1;
            end
            for (consume=0;consume<6;consume=consume+1) begin
                enabled_words[consume]=0;
                if (active[consume]) begin
                    enabled_words[consume]=raw_words[consume];
                    if ((^raw_words[consume])===1'bx) unknown_words[consume]=1;
                end
            end
            sum01={1'b0,enabled_words[0]}+{1'b0,enabled_words[1]};
            sum23={1'b0,enabled_words[2]}+{1'b0,enabled_words[3]};
            sum45={1'b0,enabled_words[4]}+{1'b0,enabled_words[5]};
            sum0123={1'b0,sum01}+{1'b0,sum23};
            total={1'b0,sum0123}+{2'b0,sum45};
            consumes=({2'b0,active[0]}+{2'b0,active[1]})+
                     ({2'b0,active[2]}+{2'b0,active[3]})+
                     ({2'b0,active[4]}+{2'b0,active[5]});
            summary[25:24]=(consumes>=3'd2) ? 2'd2 : consumes[1:0];
            // Earlier active word wins; preserve its raw X/Z bits unchanged.
            for (consume=0;consume<6;consume=consume+1)
                if (active[consume]) summary[23:13]=raw_words[consume];
            summary[0]=|unknown_words;
            if (!summary[0]) summary[12:1]=(total>14'd2049) ? 12'd2049 : total[11:0];
            result.summary=summary;
            prepare_operation=result;
        end
    endfunction
    always_comb begin
        next_e='0; next_e.origin=in_c;
        next_admission_matches='0;
        for (admission=0;admission<2;admission=admission+1) begin
            for (row=0;row<256;row=row+1) begin
                if (in_c.ops.admissions[admission].valid &&
                    in_c.ops.admissions[admission].tag==row)
                    next_admission_matches[admission*256+row]=1'b1;
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
        end else begin
            out_e<=next_e;
            out_admission_matches<=next_admission_matches;
        end
    end
endmodule
