`timescale 1ns / 1ps
// Drive public C records through both real E/D pairs. Never force internal state.
module tb_ntrpu_pcie_symbolic_row_enable_r1;
    import ntrpu_pcie_symbolic_frontend_pkg::*;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    reg clk=0, cold_rst_n=0;
    c_record_t in_c='0;
    e_record_t expected_e, actual_e, previous_e='0;
    wire [511:0] expected_matches, actual_matches;
    reg [511:0] previous_matches='0;
    wire [255:0] row_enables, attribute_enables;
    reg [255:0] previous_row_enables='0, previous_attribute_enables='0;
    control_t expected_control, actual_control, previous_control='0;
    s1_record_t expected_record, actual_record, previous_record='0;
    description_t previous_retained='0, previous_rows [0:255];
    reg [4:0] previous_attributes [0:255];
    reg [31:0] random_state=32'h7e57c0de;
    integer rows=0, pre=0, post=0, binary_cases=0, four_state_cases=0;
    integer bit_cases=0, random_cases=0;
    integer target, gate, profile, bit_index, value, iteration, slot, index;

    ntrpu_pcie_symbolic_effects_predecode_r1 reference_e(
        .clk(clk),.cold_rst_n(cold_rst_n),.in_c(in_c),
        .out_e(expected_e),.out_admission_matches(expected_matches));
    ntrpu_pcie_symbolic_effects_predecode_row_enable_r1 candidate_e(
        .clk(clk),.cold_rst_n(cold_rst_n),.in_c(in_c),
        .out_e(actual_e),.out_admission_matches(actual_matches),
        .out_row_enables(row_enables),.out_attribute_enables(attribute_enables));
    ntrpu_pcie_symbolic_descriptions_predecode_r1 reference_d(
        .clk(clk),.cold_rst_n(cold_rst_n),.in_e(expected_e),
        .in_admission_matches(expected_matches),.out_control(expected_control),.out_record(expected_record));
    ntrpu_pcie_symbolic_descriptions_predecode_row_enable_r1 candidate_d(
        .clk(clk),.cold_rst_n(cold_rst_n),.in_e(actual_e),
        .in_admission_matches(actual_matches),.in_row_enables(row_enables),
        .in_attribute_enables(attribute_enables),.out_control(actual_control),.out_record(actual_record));

    function automatic logic digit(input integer code);
        case(code) 0:digit=0;1:digit=1;2:digit=1'bx;default:digit=1'bz;endcase
    endfunction
    task compare(input integer phase);
        integer r;
        reg [255:0] wanted_rows, wanted_attributes;
        begin
            if ({actual_e,actual_matches}!=={expected_e,expected_matches})
                $fatal(1,"ROW_ENABLE_REFERENCE_CHANGED row=%0d phase=%0d",rows,phase);
            if ({actual_control,actual_record}!=={expected_control,expected_record})
                $fatal(1,"ROW_ENABLE_OUTPUT_MISMATCH row=%0d phase=%0d",rows,phase);
            if (candidate_d.retained_description!==reference_d.retained_description)
                $fatal(1,"ROW_ENABLE_CONTEXT_MISMATCH row=%0d phase=%0d",rows,phase);
            for(r=0;r<256;r=r+1) begin
                if ({candidate_d.symbolic_rows[r],candidate_d.traffic_class[r],candidate_d.attributes[r]}!==
                    {reference_d.symbolic_rows[r],reference_d.traffic_class[r],reference_d.attributes[r]}) begin
                    $display("ROW_ENABLE_VALUE actual=%b%b%b expected=%b%b%b",
                        candidate_d.symbolic_rows[r],candidate_d.traffic_class[r],candidate_d.attributes[r],
                        reference_d.symbolic_rows[r],reference_d.traffic_class[r],reference_d.attributes[r]);
                    $fatal(1,"ROW_ENABLE_MISMATCH row=%0d phase=%0d field=table index=%0d",rows,phase,r);
                end
            end
            // Independently derive mask truth from the unchanged registered E.
            wanted_rows='0;wanted_attributes='0;
            if (expected_e.origin.record.valid && expected_e.origin.record.pre_observing) begin
                for(r=0;r<256;r=r+1) begin
                    if (expected_e.origin.ops.commit_valid[r] || expected_matches[r] || expected_matches[256+r])
                        wanted_rows[r]=1;
                    if (expected_matches[r] || expected_matches[256+r]) wanted_attributes[r]=1;
                end
            end
            if ({row_enables,attribute_enables}!=={wanted_rows,wanted_attributes}) begin
                $display("ROW_ENABLE_VALUE actual=%b%b expected=%b%b",
                    row_enables,attribute_enables,wanted_rows,wanted_attributes);
                $fatal(1,"ROW_ENABLE_MISMATCH row=%0d phase=%0d field=enables index=0",rows,phase);
            end
        end
    endtask
    task tick;
        integer r;
        begin
            #1;
            compare(0);
            if ({expected_e,expected_matches,expected_control,expected_record,reference_d.retained_description,
                 row_enables,attribute_enables}!==
                {previous_e,previous_matches,previous_control,previous_record,previous_retained,
                 previous_row_enables,previous_attribute_enables})
                $fatal(1,"ROW_ENABLE_EARLY_CHANGE row=%0d",rows);
            for(r=0;r<256;r=r+1) begin
                if ({reference_d.symbolic_rows[r],reference_d.traffic_class[r],reference_d.attributes[r]}!==
                    {previous_rows[r],previous_attributes[r]})
                    $fatal(1,"ROW_ENABLE_TABLE_EARLY_CHANGE row=%0d index=%0d",rows,r);
            end
            pre=pre+1;clk=1;#0.001;compare(1);post=post+1;
            previous_e=expected_e;previous_matches=expected_matches;
            previous_control=expected_control;previous_record=expected_record;
            previous_retained=reference_d.retained_description;
            previous_row_enables=row_enables;previous_attribute_enables=attribute_enables;
            for(r=0;r<256;r=r+1) begin
                previous_rows[r]=reference_d.symbolic_rows[r];
                previous_attributes[r]={reference_d.traffic_class[r],reference_d.attributes[r]};
            end
            clk=0;#0.999;rows=rows+1;
        end
    endtask
    task seed(input integer tag);
        integer s;
        begin
            in_c='0;in_c.record.valid=1;in_c.record.pre_observing=1;
            in_c.record.sequence_number=rows;
            in_c.ops.commit_slot[tag*2+:2]=tag%4;
            in_c.ops.update_valid='1;
            for(s=0;s<4;s=s+1) begin
                in_c.ops.queries[s].tag=tag;
                in_c.ops.eops[s].tag=tag;
                in_c.ops.updates[s].tag=tag;
                in_c.record.queries[s].valid=1;
                in_c.record.eops[s].valid=1;
                in_c.record.eops[s].delivery_enabled=1;
                in_c.record.eops[s].words=s+1;
            end
            in_c.ops.context_after.tag=tag;
            for(s=0;s<2;s=s+1) begin
                in_c.ops.admissions[s].tag=tag;
                in_c.ops.admissions[s].address=tag+17+s*257;
                in_c.ops.admissions[s].remaining=tag+9+s*19;
                in_c.ops.admissions[s].traffic_class=tag+s;
                in_c.ops.admissions[s].attributes=tag+s;
            end
        end
    endtask
    task random_next;
        begin
            random_state=random_state^(random_state<<13);
            random_state=random_state^(random_state>>17);
            random_state=random_state^(random_state<<5);
        end
    endtask
    initial begin
        if ($bits(c_record_t)!=3574 || $bits(e_record_t)!=4042 || $bits(description_t)!=51 ||
            $bits(control_t)!=391 || $bits(s1_record_t)!=2107) $fatal(1,"ROW_ENABLE_SCHEMA");
        tick();cold_rst_n=1;tick();
        seed(7);in_c.ops.admissions[0].valid=1;
        in_c.ops.admissions[0].address=12'h123;in_c.ops.admissions[0].remaining=9;
        in_c.ops.admissions[0].traffic_class=5;in_c.ops.admissions[0].attributes=2;
        tick();in_c='0;tick();
        if ({actual_control,actual_record}!=={expected_control,expected_record} ||
            {reference_d.symbolic_rows[7],reference_d.traffic_class[7],reference_d.attributes[7]}!==
            {12'h123,13'd9,26'b0,3'd5,2'd2}) $fatal(1,"ROW_ENABLE_ADMISSION_ORACLE");
        seed(9);in_c.record.valid=0;in_c.ops.admissions[0].valid=1;tick();in_c='0;tick();
        if (reference_d.symbolic_rows[9]!==51'b0) $fatal(1,"ROW_ENABLE_VALID_ORACLE");
        seed(11);in_c.record.pre_observing=0;in_c.ops.admissions[0].valid=1;tick();in_c='0;tick();
        if (reference_d.symbolic_rows[11]!==51'b0) $fatal(1,"ROW_ENABLE_OBSERVING_ORACLE");
        seed(7);in_c.ops.commit_valid[7]=1;in_c.ops.commit_slot[14+:2]=0;
        in_c.record.eops[0].words=3;tick();in_c='0;tick();
        if (reference_d.symbolic_rows[7][25:0]!=={2'd1,11'd3,12'd3,1'b0}) $fatal(1,"ROW_ENABLE_COMMIT_ORACLE");
        seed(7);in_c.ops.commit_valid[7]=1;
        in_c.ops.admissions[0].valid=1;in_c.ops.admissions[1].valid=1;
        in_c.ops.admissions[1].address=12'h456;in_c.ops.admissions[1].remaining=17;
        in_c.ops.admissions[1].traffic_class=2;in_c.ops.admissions[1].attributes=1;
        tick();in_c='0;tick();
        if ({reference_d.symbolic_rows[7],reference_d.traffic_class[7],reference_d.attributes[7]}!==
            {12'h456,13'd17,26'b0,3'd2,2'd1}) $fatal(1,"ROW_ENABLE_PRIORITY_ORACLE");
        seed(23);in_c.ops.admissions[0].valid=1;tick();cold_rst_n=0;tick();
        cold_rst_n=1;in_c='0;tick();tick();
        if (rows!=16) $fatal(1,"ROW_ENABLE_DIRECTED_COUNT");
        // Every table row and all binary combinations of the six write/reset gates.
        for(target=0;target<256;target=target+1) for(gate=0;gate<64;gate=gate+1) begin
            seed(target);in_c.record.valid=gate[0];in_c.record.pre_observing=gate[1];
            in_c.ops.commit_valid[target]=gate[2];in_c.ops.admissions[0].valid=gate[3];
            in_c.ops.admissions[1].valid=gate[4];cold_rst_n=gate[5];
            tick();binary_cases=binary_cases+1;
        end
        // All four-state combinations, same/different/unknown admission addresses.
        for(gate=0;gate<4096;gate=gate+1) for(profile=0;profile<4;profile=profile+1) begin
            target=gate%256;seed(target);
            cold_rst_n=digit(gate%4);in_c.record.valid=digit((gate/4)%4);
            in_c.record.pre_observing=digit((gate/16)%4);
            in_c.ops.commit_valid[target]=digit((gate/64)%4);
            in_c.ops.admissions[0].valid=digit((gate/256)%4);
            in_c.ops.admissions[1].valid=digit((gate/1024)%4);
            if (profile==1) in_c.ops.admissions[1].tag=target+1;
            if (profile==2) begin
                in_c.ops.admissions[0].tag[0]=1'bx;in_c.ops.commit_slot[target*2+:2]=2'bxz;
            end
            if (profile==3) begin
                in_c.ops.admissions[1].tag[7]=1'bz;in_c.ops.updates[target%4].retained=1;
            end
            tick();four_state_cases=four_state_cases+1;
        end
        // Every C bit with X and Z, including retained commands and data fields.
        cold_rst_n=1;
        for(bit_index=0;bit_index<3574;bit_index=bit_index+1) for(value=2;value<4;value=value+1) begin
            in_c='1;in_c[bit_index]=digit(value);tick();bit_cases=bit_cases+1;
        end
        for(iteration=0;iteration<1024;iteration=iteration+1) begin
            for(index=0;index<3574;index=index+1) begin
                if (index%32==0) random_next();
                in_c[index]=random_state[index%32];
            end
            cold_rst_n=(iteration%64!=0);tick();random_cases=random_cases+1;
        end
        cold_rst_n=1;in_c='0;tick();tick();
        if (rows!=40958 || pre!=rows || post!=rows || binary_cases!=16384 || four_state_cases!=16384 ||
            bit_cases!=7148 || random_cases!=1024) $fatal(1,"ROW_ENABLE_INCOMPLETE");
        $display("ROW_ENABLE_PASS rows=40958 pre=40958 post=40958 binary_cases=16384 four_state_cases=16384 bit_cases=7148 random_cases=1024 table_bits=14336 retained_bits=51 d_output_bits=2498 e_output_bits=4554 enable_bits=512");
        $finish;
    end
    initial begin #1000000;$fatal(1,"ROW_ENABLE_WATCHDOG");end
endmodule
