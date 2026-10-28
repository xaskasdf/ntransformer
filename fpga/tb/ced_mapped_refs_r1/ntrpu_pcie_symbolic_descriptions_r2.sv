`timescale 1ns / 1ps
// D: own ordered table and a retained description by value. One E origin/cycle.
module ntrpu_pcie_symbolic_descriptions_r2_ced_mapped_reference_r1 (
    input wire clk, cold_rst_n,
    input ntrpu_pcie_symbolic_ced_pkg::e_record_t in_e,
    output ntrpu_pcie_symbolic_ced_pkg::control_t out_control,
    output ntrpu_pcie_symbolic_frontend_pkg::s1_record_t out_record
);
    import ntrpu_pcie_metadata_effect_pkg::*;
    import ntrpu_pcie_symbolic_frontend_pkg::*;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    description_t symbolic_rows [0:255], next_rows [0:255];
    logic [2:0] traffic_class [0:255], next_tc [0:255];
    logic [1:0] attributes [0:255], next_attributes [0:255];
    description_t retained_description=0, next_retained;
    description_t [3:0] update_descriptions;
    s1_record_t next_record;
    integer slot, row, admission, reset_index;
    initial begin out_control='0; out_record='0; end
    function automatic description_t resolve_operation(input command_t command);
        description_t base;
        effect_t result;
        begin
            base=symbolic_rows[command.tag];
            if (command.retained) base=retained_description;
            result='0;
            if (command.apply_summary) begin
                result=compose_effect({31'b0,base[25:0]},{31'b0,command.summary});
                base={base[50:26],result[25:0]};
            end
            resolve_operation=base;
        end
    endfunction
    always_comb begin
        next_record=in_e.origin.record;
        next_record.pre_rc_context.description=retained_description;
        update_descriptions='0;
        for (slot=0;slot<4;slot=slot+1) begin
            if (in_e.origin.record.queries[slot].valid) begin
                next_record.queries[slot].description=resolve_operation(in_e.queries[slot]);
                next_record.queries[slot].attributes_tc={attributes[in_e.queries[slot].tag],
                    traffic_class[in_e.queries[slot].tag]};
            end
            if (in_e.origin.record.eops[slot].valid)
                next_record.eops[slot].description=resolve_operation(in_e.eops[slot]);
            if (in_e.origin.ops.update_valid[slot])
                update_descriptions[slot]=resolve_operation(in_e.updates[slot]);
        end
        next_retained=retained_description;
        if (in_e.origin.record.pre_observing)
            next_retained=resolve_operation(in_e.context_after);
        for (row=0;row<256;row=row+1) begin
            next_rows[row]=symbolic_rows[row];
            next_tc[row]=traffic_class[row]; next_attributes[row]=attributes[row];
            if (in_e.origin.record.pre_observing) begin
                if (in_e.origin.ops.commit_valid[row])
                    next_rows[row]=update_descriptions[in_e.origin.ops.commit_slot[row*2+:2]];
                // RC commits precede both RQ admissions, including tag reuse.
                for (admission=0;admission<2;admission=admission+1) begin
                    if (in_e.origin.ops.admissions[admission].valid &&
                        in_e.origin.ops.admissions[admission].tag==row) begin
                        next_rows[row]={in_e.origin.ops.admissions[admission].address,
                            in_e.origin.ops.admissions[admission].remaining,26'b0};
                        next_tc[row]=in_e.origin.ops.admissions[admission].traffic_class;
                        next_attributes[row]=in_e.origin.ops.admissions[admission].attributes;
                    end
                end
            end
        end
    end
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            out_control<='0; out_record<='0; retained_description<=0;
            for (reset_index=0;reset_index<256;reset_index=reset_index+1) begin
                symbolic_rows[reset_index]<=0;
                traffic_class[reset_index]<=0; attributes[reset_index]<=0;
            end
        end else if (in_e.origin.record.valid) begin
            out_control<=in_e.origin.control; out_record<=next_record;
            retained_description<=next_retained;
            for (reset_index=0;reset_index<256;reset_index=reset_index+1) begin
                symbolic_rows[reset_index]<=next_rows[reset_index];
                traffic_class[reset_index]<=next_tc[reset_index];
                attributes[reset_index]<=next_attributes[reset_index];
            end
        end
    end
endmodule
