`timescale 1ns / 1ps
// Whole actual-E comparison: public C input only; unchanged original E reference.
module tb_ntrpu_pcie_symbolic_effects_predecode_sum_r1;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    reg clk=0,cold_rst_n=0;
    c_record_t in_c='0;
    e_record_t actual_e,expected_e,previous_e='0;
    wire [511:0] actual_matches,expected_matches;
    reg [511:0] previous_matches='0;
    operation_t operation;
    reg [10:0] words [0:5];
    reg [31:0] random_state=32'h9e3779b9;
    reg [7:0] tag_bits;
    integer rows=0,pre=0,post=0,control_cases=0,word_bit_cases=0,random_cases=0,cold_cases=0;
    integer gate,profile,index,slot,mask,word_index,bit_index,value,iteration;
    ntrpu_pcie_symbolic_effects_predecode_r1 reference_i(
        .clk(clk),.cold_rst_n(cold_rst_n),.in_c(in_c),.out_e(expected_e),.out_admission_matches(expected_matches));
    ntrpu_pcie_symbolic_effects_predecode_sum_r1 candidate_i(
        .clk(clk),.cold_rst_n(cold_rst_n),.in_c(in_c),.out_e(actual_e),.out_admission_matches(actual_matches));
    function automatic logic digit(input integer code);
        case(code) 0:digit=0;1:digit=1;2:digit=1'bx;default:digit=1'bz;endcase
    endfunction
    task tick;
        begin
            #1;
            if ({actual_e,actual_matches}!=={previous_e,previous_matches} ||
                {expected_e,expected_matches}!=={previous_e,previous_matches})
                $fatal(1,"E_SUM_EARLY_CHANGE row=%0d",rows);
            pre=pre+1;
            clk=1; #0.001;
            if ({actual_e,actual_matches}!=={expected_e,expected_matches}) begin
                $display("E_SUM_ACTUAL %b%b",actual_e,actual_matches);
                $display("E_SUM_EXPECTED %b%b",expected_e,expected_matches);
                $display("E_SUM_ACTUAL_QUERY %b",actual_e.queries[0]);
                $display("E_SUM_EXPECTED_QUERY %b",expected_e.queries[0]);
                $display("E_SUM_ACTUAL_UPDATE %b",actual_e.updates[0]);
                $display("E_SUM_EXPECTED_UPDATE %b",expected_e.updates[0]);
                $fatal(1,"E_SUM_MISMATCH row=%0d phase=1",rows);
            end
            if (cold_rst_n===1'b0 && ({actual_e,actual_matches}!=='0 || {expected_e,expected_matches}!=='0))
                $fatal(1,"E_SUM_COLD_ORACLE row=%0d",rows);
            previous_e=expected_e;previous_matches=expected_matches;post=post+1;
            clk=0;#0.999;rows=rows+1;
        end
    endtask
    task set_words;
        begin
            in_c.record.pre_rc_context.words=words[0];
            for(slot=0;slot<4;slot=slot+1) begin
                in_c.ops.prefix_counts[slot*11+:11]=words[slot+1];
                in_c.record.eops[slot].words=words[5];
            end
        end
    endtask
    task set_operations;
        begin
            for(slot=0;slot<4;slot=slot+1) begin
                in_c.ops.queries[slot]=operation;
                in_c.ops.eops[slot]=operation;
                in_c.ops.updates[slot]=operation;
            end
            in_c.ops.context_after=operation;
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
        if ($bits(e_record_t)!=4042 || $bits(c_record_t)!=3574 || $bits(command_t)!=36)
            $fatal(1,"E_SUM_SCHEMA");
        // Independent first witnesses; all controls fail through real output ports.
        tick(); cold_rst_n=1; tick(); // updates always request a summary, even without active consumes
        in_c='0; in_c.ops.queries[0].prefix=1; in_c.ops.queries[0].mask=3;
        in_c.ops.prefix_counts[0+:11]=2047;in_c.ops.prefix_counts[11+:11]=2;tick();
        if (actual_e.queries[0].summary!=={2'd2,11'd2047,12'd2049,1'b0}) $fatal(1,"E_SUM_CAP_ORACLE");
        in_c='0;in_c.record.eops[0].delivery_enabled=1;in_c.record.eops[0].words=7;tick();
        if (actual_e.updates[0].summary!=={2'd1,11'd7,12'd7,1'b0}) $fatal(1,"E_SUM_APPEND_ORACLE");
        in_c='0;in_c.ops.queries[0].prefix=1;in_c.ops.queries[0].retained=1;in_c.ops.queries[0].mask=1;
        in_c.record.pre_rc_context.delivery_enabled=1;in_c.record.pre_rc_context.words=3;
        in_c.ops.prefix_counts[0+:11]=9;tick();
        if (actual_e.queries[0].summary!=={2'd2,11'd3,12'd12,1'b0}) $fatal(1,"E_SUM_FIRST_ORACLE");
        in_c='0;in_c.ops.queries[0].prefix=1;in_c.ops.queries[0].mask=3;
        in_c.ops.prefix_counts[0+:11]=3;in_c.ops.prefix_counts[11+:11]=11'b0000000000x;tick();
        if (actual_e.queries[0].summary!=={2'd2,11'd3,12'd0,1'b1}) $fatal(1,"E_SUM_UNKNOWN_ORACLE");
        in_c='0;in_c.ops.queries[0].prefix=1'bx;in_c.ops.queries[0].mask=1;in_c.ops.prefix_counts[0+:11]=5;tick();
        if (actual_e.queries[0]!==36'b0) $fatal(1,"E_SUM_UNKNOWN_PREFIX_ORACLE");
        // All4-state controls: prefix,retained,retained-enable,mask4,appended-enable.
        for(gate=0;gate<65536;gate=gate+1) for(profile=0;profile<4;profile=profile+1) begin
            in_c='0;in_c.record.sequence_number=rows;operation='0;
            operation.prefix=digit(gate%4);operation.retained=digit((gate/4)%4);
            in_c.record.pre_rc_context.delivery_enabled=digit((gate/16)%4);
            for(index=0;index<4;index=index+1) operation.mask[index]=digit((gate>>(6+2*index))%4);
            for(slot=0;slot<4;slot=slot+1) in_c.record.eops[slot].delivery_enabled=digit((gate/16384)%4);
            for(index=0;index<6;index=index+1) begin
                case(profile)
                    0:words[index]=(index%2) ? 2047 : 0;
                    1:words[index]=11'bxxxxxxxxxxx;
                    2:words[index]=11'bzzzzzzzzzzz;
                    3:words[index]=(index%2) ? 11'b10x10101010 : 11'b01z01010101;
                endcase
            end
            operation.tag=(gate^profile);
            set_words();set_operations();tick();control_cases=control_cases+1;
        end
        // Every individual bit in each active or masked word: 0,1,X,Z.
        for(mask=0;mask<64;mask=mask+1) for(word_index=0;word_index<6;word_index=word_index+1)
            for(bit_index=0;bit_index<11;bit_index=bit_index+1) for(value=0;value<4;value=value+1) begin
                in_c='0;in_c.record.sequence_number=rows;operation='0;
                operation.prefix=1;operation.retained=1;operation.mask=mask[4:1];
                in_c.record.pre_rc_context.delivery_enabled=mask[0];
                for(slot=0;slot<4;slot=slot+1) in_c.record.eops[slot].delivery_enabled=mask[5];
                for(index=0;index<6;index=index+1) words[index]=index*341;
                words[word_index][bit_index]=digit(value);
                set_words();set_operations();tick();word_bit_cases=word_bit_cases+1;
            end
        // Known binary arithmetic and admission matching with deterministic inputs.
        for(iteration=0;iteration<4096;iteration=iteration+1) begin
            in_c='0;in_c.record.sequence_number=rows;operation='0;
            random_next();operation=random_state[13:0];
            in_c.record.pre_rc_context.delivery_enabled=random_state[14];
            for(slot=0;slot<4;slot=slot+1) in_c.record.eops[slot].delivery_enabled=random_state[15+slot];
            for(index=0;index<6;index=index+1) begin random_next();words[index]=random_state[10:0];end
            set_words();set_operations();
            for(index=0;index<2;index=index+1) begin
                random_next();in_c.ops.admissions[index].valid=digit(random_state%4);
                tag_bits=random_state[15:8];tag_bits[iteration%8]=digit((random_state/4)%4);
                in_c.ops.admissions[index].tag=tag_bits;
            end
            tick();random_cases=random_cases+1;
        end
        for(value=0;value<4;value=value+1) begin cold_rst_n=digit(value);tick();cold_cases=cold_cases+1;end
        cold_rst_n=0;tick();
        if(rows!=283148 || pre!=rows || post!=rows || control_cases!=262144 || word_bit_cases!=16896 || random_cases!=4096 || cold_cases!=4)
            $fatal(1,"E_SUM_INCOMPLETE");
        $display("E_SUM_PASS rows=%0d pre=%0d post=%0d control_cases=%0d word_bit_cases=%0d random_cases=%0d cold_cases=%0d output_bits=4554",
            rows,pre,post,control_cases,word_bit_cases,random_cases,cold_cases);
        $finish;
    end
    initial begin #1000000;$fatal(1,"E_SUM_WATCHDOG");end
endmodule
