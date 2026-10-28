`timescale 1ns / 1ps
// Actual registered B and five bounds; public A input only, no internal writes.
module tb_ntrpu_pcie_symbolic_delivery_predecode_bound_r1;
    import ntrpu_pcie_symbolic_numeric_pkg_r1::*;
    reg clk=0,cold_rst_n=0;
    a_record_t in_a='0;
    b_record_t actual_b,expected_b,previous_b='x;
    wire [59:0] actual_query_limits;
    wire [14:0] actual_context_limit;
    reg [74:0] previous_limits='x;
    reg [59:0] expected_query_limits;
    reg [14:0] expected_context_limit;
    reg [31:0] random_state=32'h6a09e667;
    integer rows=0,pre=0,post=0,arithmetic_cases=0,control_cases=0,bit_cases=0,random_cases=0,cold_cases=0;
    integer words,offset,boundary,gate,profile,bit_at,value,iteration,chunk,slot,check_slot;
    ntrpu_pcie_symbolic_delivery_r1 reference_i(
        .clk(clk),.cold_rst_n(cold_rst_n),.in_a(in_a),.out_b(expected_b));
    ntrpu_pcie_symbolic_delivery_predecode_bound_r1 candidate_i(
        .clk(clk),.cold_rst_n(cold_rst_n),.in_a(in_a),.out_b(actual_b),
        .query_payload_limit(actual_query_limits),.context_payload_limit(actual_context_limit));

    function automatic logic digit(input integer code);
        case(code) 0:digit=0;1:digit=1;2:digit=1'bx;default:digit=1'bz;endcase
    endfunction
    function automatic [12:0] remaining(input integer at);
        case(at)
            0:remaining=0;1:remaining=1;2:remaining=2;3:remaining=3;
            4:remaining=4;5:remaining=5;6:remaining=7;7:remaining=8;
            8:remaining=15;9:remaining=16;10:remaining=31;11:remaining=32;
            12:remaining=4095;13:remaining=4096;14:remaining=8190;default:remaining=8191;
        endcase
    endfunction
    // Independent wide arithmetic on the unchanged reference's captured output.
    function automatic [14:0] exact_bound(input numeric_value_t numeric,
        input [12:0] delivered);
        reg [31:0] wide;
        begin
            wide=0;wide[1:0]=numeric[1:0];
            wide=wide+{19'b0,delivered};
            if ((^wide)!==1'bx && wide>8194) $fatal(1,"B_BOUND_MODEL_RANGE");
            exact_bound=wide[14:0];
        end
    endfunction
    task base_record;
        begin
            in_a='0;in_a.record.valid=1;in_a.record.sequence_number=rows;
            for(slot=0;slot<4;slot=slot+1) begin
                in_a.record.queries[slot].valid=1;in_a.record.queries[slot].delivery_enabled=1;
                in_a.record.queries[slot].words=slot+2;
                in_a.queries[slot]={13'd100,10'b0,2'((slot+1)%4)};
                in_a.record.eops[slot].valid=1;in_a.record.eops[slot].delivery_enabled=1;
                in_a.record.eops[slot].terminal=slot%2;in_a.record.eops[slot].words=slot+3;
                in_a.eops[slot]={13'd200,10'b0,2'(slot)};
            end
            in_a.record.pre_rc_context.delivery_enabled=1;
            in_a.record.pre_rc_context.words=7;in_a.context_value={13'd100,12'd2};
        end
    endtask
    task random_next;
        begin
            random_state=random_state^(random_state<<13);
            random_state=random_state^(random_state>>17);
            random_state=random_state^(random_state<<5);
        end
    endtask
    task tick;
        begin
            #1;
            if (actual_b!==previous_b || expected_b!==previous_b ||
                {actual_query_limits,actual_context_limit}!==previous_limits)
                $fatal(1,"B_BOUND_EARLY_CHANGE row=%0d",rows);
            pre=pre+1;clk=1;#0.001;
            for(check_slot=0;check_slot<4;check_slot=check_slot+1)
                expected_query_limits[check_slot*15+:15]=exact_bound(
                    expected_b.origin.queries[check_slot],expected_b.query_delivered[check_slot]);
            expected_context_limit=exact_bound(expected_b.origin.context_value,expected_b.context_delivered);
            if ({actual_b,actual_query_limits,actual_context_limit}!==
                {expected_b,expected_query_limits,expected_context_limit}) begin
                $display("B_BOUND_ACTUAL %b%b%b",actual_b,actual_query_limits,actual_context_limit);
                $display("B_BOUND_EXPECTED %b%b%b",expected_b,expected_query_limits,expected_context_limit);
                $display("B_BOUND_ACTUAL_Q0 %b",actual_query_limits[0+:15]);
                $display("B_BOUND_EXPECTED_Q0 %b",expected_query_limits[0+:15]);
                $fatal(1,"B_BOUND_MISMATCH row=%0d phase=1",rows);
            end
            if(cold_rst_n===1'b0 && {actual_b,actual_query_limits,actual_context_limit}!=='0)
                $fatal(1,"B_BOUND_COLD_ORACLE row=%0d",rows);
            previous_b=expected_b;previous_limits={expected_query_limits,expected_context_limit};
            post=post+1;clk=0;#0.999;rows=rows+1;
        end
    endtask
    initial begin
        if($bits(a_record_t)!=2723 || $bits(b_record_t)!=2892) $fatal(1,"B_BOUND_SCHEMA");
        tick();cold_rst_n=1;base_record();tick();
        if(actual_query_limits!=={15'd20,15'd16,15'd12,15'd8} || actual_context_limit!==15'd28)
            $fatal(1,"B_BOUND_ALIGNMENT_ORACLE");
        // Keep malformed encodings: an underflow can produce delivered=8191.
        base_record();in_a.record.queries[0].words=0;in_a.queries[0]={13'd8191,12'd3};tick();
        if(actual_b.query_delivered[0]!==13'd8191 || actual_query_limits[0+:15]!==15'd8194)
            $fatal(1,"B_BOUND_WIDTH_ORACLE");
        in_a='0;in_a.record.valid=1;in_a.queries[0][0]=1'bx;tick();
        if(actual_query_limits[0+:15]!==15'bxxxxxxxxxxxxxxx) $fatal(1,"B_BOUND_UNKNOWN_ORACLE");
        cold_rst_n=0;base_record();tick();cold_rst_n=1;
        // Every encoded word count and byte alignment, sixteen remainder edges.
        for(words=0;words<2048;words=words+1) for(offset=0;offset<4;offset=offset+1)
            for(boundary=0;boundary<16;boundary=boundary+1) begin
                base_record();
                for(slot=0;slot<4;slot=slot+1) begin
                    in_a.record.queries[slot].words=(words+slot)%2048;
                    in_a.queries[slot]={remaining((boundary+slot)%16),10'b0,2'((offset+slot)%4)};
                    in_a.record.eops[slot].words=(words+2*slot)%2048;
                    in_a.eops[slot]={remaining((boundary+2*slot)%16),10'b0,2'((offset+slot)%4)};
                end
                in_a.record.pre_rc_context.words=words;
                in_a.context_value={remaining(boundary),10'b0,2'(offset)};
                tick();arithmetic_cases=arithmetic_cases+1;
            end
        // All four-state gate combinations, with binary and unknown operands.
        for(gate=0;gate<16384;gate=gate+1) for(profile=0;profile<4;profile=profile+1) begin
            base_record();in_a.record.valid=digit(gate%4);
            for(slot=0;slot<4;slot=slot+1) begin
                in_a.record.queries[slot].valid=digit((gate/4)%4);
                in_a.record.queries[slot].delivery_enabled=digit((gate/16)%4);
                in_a.record.eops[slot].valid=digit((gate/64)%4);
                in_a.record.eops[slot].delivery_enabled=digit((gate/256)%4);
                in_a.record.eops[slot].terminal=digit((gate/1024)%4);
                case(profile)
                    1:begin in_a.queries[slot]='x;in_a.eops[slot]='x;in_a.record.queries[slot].words='x;in_a.record.eops[slot].words='x;end
                    2:begin in_a.queries[slot]='z;in_a.eops[slot]='z;in_a.record.queries[slot].words='z;in_a.record.eops[slot].words='z;end
                    3:begin in_a.queries[slot][slot]=1'bx;in_a.eops[slot][slot+12]=1'bz;in_a.record.queries[slot].words[slot]=1'bz;end
                endcase
            end
            in_a.record.pre_rc_context.delivery_enabled=digit((gate/4096)%4);
            case(profile)
                1:begin in_a.context_value='x;in_a.record.pre_rc_context.words='x;end
                2:begin in_a.context_value='z;in_a.record.pre_rc_context.words='z;end
                3:begin in_a.context_value[0]=1'bz;in_a.record.pre_rc_context.words[10]=1'bx;end
            endcase
            tick();control_cases=control_cases+1;
        end
        // Every bit of the public A record: 0,1,X,Z, including all origin fields.
        for(bit_at=0;bit_at<2723;bit_at=bit_at+1) for(value=0;value<4;value=value+1) begin
            base_record();in_a[bit_at]=digit(value);tick();bit_cases=bit_cases+1;
        end
        for(iteration=0;iteration<4096;iteration=iteration+1) begin
            in_a='0;
            for(chunk=0;chunk<86;chunk=chunk+1) begin random_next();in_a=(in_a<<32)|random_state;end
            in_a.record.valid=digit(iteration%4);
            for(slot=0;slot<4;slot=slot+1) begin
                in_a.record.queries[slot].valid=digit((iteration>>(2*slot))%4);
                in_a.record.eops[slot].terminal=digit((iteration>>(2*slot+2))%4);
            end
            tick();random_cases=random_cases+1;
        end
        for(value=0;value<4;value=value+1) begin
            base_record();cold_rst_n=digit(value);tick();cold_cases=cold_cases+1;
        end
        cold_rst_n=0;tick();
        if(rows!=211606 || pre!=rows || post!=rows || arithmetic_cases!=131072 || control_cases!=65536 ||
            bit_cases!=10892 || random_cases!=4096 || cold_cases!=4) $fatal(1,"B_BOUND_INCOMPLETE");
        $display("B_BOUND_PASS rows=%0d pre=%0d post=%0d arithmetic_cases=%0d control_cases=%0d bit_cases=%0d random_cases=%0d cold_cases=%0d output_bits=2967",
            rows,pre,post,arithmetic_cases,control_cases,bit_cases,random_cases,cold_cases);
        $finish;
    end
    initial begin #1000000;$fatal(1,"B_BOUND_WATCHDOG");end
endmodule
