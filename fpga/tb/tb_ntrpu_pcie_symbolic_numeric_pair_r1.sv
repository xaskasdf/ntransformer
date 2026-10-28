`timescale 1ns / 1ps
// Whole registered A/B modules. Reference functions are copied from the sealed model.
module tb_ntrpu_pcie_symbolic_numeric_pair_r1;
    import ntrpu_pcie_symbolic_frontend_pkg::*;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    import ntrpu_pcie_symbolic_numeric_pkg_r1::*;
    reg clk=0; always #2 clk=~clk;
    reg cold_rst_n=0;
    control_t in_control='0;
    s1_record_t in_record='0;
    wire a_record_t actual_a;
    wire b_record_t actual_b;
    a_record_t expected_a, held_a='0;
    b_record_t expected_b, previous_b='0, held_b='0;
    ntrpu_pcie_symbolic_numeric_pair_r1 pair(.clk(clk),.cold_rst_n(cold_rst_n),
        .in_control(in_control),.in_record(in_record),.out_b(actual_b));
    assign actual_a=pair.a_value;
    integer steps=0, captured=0, retired=0, cancelled=0, cold_rows=0;
    integer binary_cases=0, description_cases=0, word_cases=0, validity_cases=0;
    integer r, low, count, en, base, target, bit_index, unknown_kind, vr, vq, ve, slot;
    reg [12:0] remaining;
    reg [50:0] description, altered_description;
    reg [10:0] altered_words;
    reg unknown_bit;
    string mode;

    function automatic [50:0] reference_symbolic_seed(input [12:0] r,input [11:0] a);
        reference_symbolic_seed={a,r,26'b0};
    endfunction

    function automatic [24:0] reference_symbolic_value(input [50:0] description);
        reg [29:0] result;
        begin
            result=ntrpu_pcie_metadata_effect_pkg::apply_effect(
                {31'b0,description[25:0]},{5'b0,description[50:26]});
            reference_symbolic_value={result[12:0],result[24:13]};
        end
    endfunction

    function automatic [12:0] reference_symbolic_delivery(input [50:0] description,
        input [10:0] words,input enabled);
        reg [24:0] value;
        reg signed [14:0] available,delivered;
        begin
            value=reference_symbolic_value(description); available=0; delivered=0;
            if (enabled) begin
                available=words*4-value[1:0];
                delivered=(available<value[24:12]) ? available : value[24:12];
            end
            reference_symbolic_delivery=delivered;
        end
    endfunction

    function automatic reg four_state(input integer code);
        case (code)
            0: four_state=0;
            1: four_state=1;
            2: four_state=1'bx;
            3: four_state=1'bz;
        endcase
    endfunction
    function automatic [12:0] boundary_remaining(input integer which);
        case (which)
            0: boundary_remaining=0;
            1: boundary_remaining=1;
            2: boundary_remaining=2;
            3: boundary_remaining=3;
            4: boundary_remaining=4095;
            5: boundary_remaining=4096;
            6: boundary_remaining=8190;
            7: boundary_remaining=8191;
        endcase
    endfunction
    function automatic [50:0] baseline_description(input integer which);
        reg [50:0] result;
        begin
            result=reference_symbolic_seed(13'd4095,12'habc);
            case (which)
                1: begin result[25:24]=1; result[23:13]=3; result[12:1]=3; end
                2: begin result[25:24]=2; result[23:13]=1; result[12:1]=8; end
                3: begin result[25:24]=2; result[23:13]=1; result[12:1]=8; result[0]=1; end
            endcase
            baseline_description=result;
        end
    endfunction
    task load_case(input [50:0] desc, input [10:0] words, input enabled);
        integer k;
        begin
            in_control='0; in_control.observing=1;
            in_control.allocated_tags=256'b1<<(steps%256);
            in_control.requests=steps; in_control.completed_reads=~steps;
            in_control.early_fault_sources=steps[15:0]; in_control.control_pending=steps[0];
            in_record='0; in_record.valid=1;
            in_record.sequence_number=32'hfffffff0+steps;
            in_record.effect_pending=steps[0]; in_record.pre_observing=1;
            in_record.expected_requester=steps[15:0];
            in_record.rq_data={16{32'h31415926^steps}};
            in_record.rc_data={16{32'ha5a55a5a^steps}};
            in_record.rq_spans[0]=steps[14:0]; in_record.rq_spans[1]=~steps[14:0];
            in_record.pre_rc_context.description=desc;
            in_record.pre_rc_context.words=words;
            in_record.pre_rc_context.delivery_enabled=enabled;
            in_record.pre_rc_context.owned=0; // Retention must not depend on ownership.
            for (k=0;k<4;k=k+1) begin
                in_record.queries[k].valid=1; in_record.queries[k].owned=k[0];
                in_record.queries[k].attributes_tc=k;
                in_record.queries[k].description=desc; in_record.queries[k].words=words;
                in_record.queries[k].delivery_enabled=enabled;
                in_record.eops[k].valid=1; in_record.eops[k].terminal=four_state(k);
                in_record.eops[k].description=desc; in_record.eops[k].words=words;
                in_record.eops[k].delivery_enabled=enabled; in_record.eops[k].lane=k*4+3;
            end
        end
    endtask
    task build_expected;
        integer k;
        reg [24:0] value;
        begin
            expected_a='0; expected_b='0;
            if (in_record.valid) begin
                expected_a.control=in_control; expected_a.record=in_record;
                for (k=0;k<4;k=k+1) begin
                    if (in_record.queries[k].valid) begin
                        expected_a.queries[k]=reference_symbolic_value(in_record.queries[k].description);
                        expected_b.query_delivered[k]=reference_symbolic_delivery(in_record.queries[k].description,
                            in_record.queries[k].words,in_record.queries[k].delivery_enabled);
                    end
                    if (in_record.eops[k].valid) begin
                        value=reference_symbolic_value(in_record.eops[k].description);
                        expected_a.eops[k]=value;
                        expected_b.eop_delivered[k]=reference_symbolic_delivery(in_record.eops[k].description,
                            in_record.eops[k].words,in_record.eops[k].delivery_enabled);
                        expected_b.eop_removed[k]=in_record.eops[k].terminal ? value[24:12] : expected_b.eop_delivered[k];
                    end
                end
                expected_a.context_value=reference_symbolic_value(in_record.pre_rc_context.description);
                expected_b.context_delivered=reference_symbolic_delivery(in_record.pre_rc_context.description,
                    in_record.pre_rc_context.words,in_record.pre_rc_context.delivery_enabled);
                expected_b.origin=expected_a;
            end
        end
    endtask
    task step;
        begin
            @(negedge clk); #1;
            if (steps!=0 && (actual_a!==held_a || actual_b!==held_b))
                $fatal(1,"NUMERIC_FOCUS_HOLD_MISMATCH step=%0d",steps);
            build_expected();
            @(posedge clk); #0.001;
            if (!cold_rst_n) begin
                if (actual_a!=='0 || actual_b!=='0) $fatal(1,"NUMERIC_FOCUS_COLD_MISMATCH step=%0d",steps);
                if (previous_b.origin.record.valid) cancelled=cancelled+1;
                previous_b='0; cold_rows=cold_rows+1;
            end else begin
                if (actual_a!==expected_a) $fatal(1,"NUMERIC_FOCUS_A_MISMATCH step=%0d",steps);
                if (actual_b!==previous_b) $fatal(1,"NUMERIC_FOCUS_B_MISMATCH step=%0d",steps);
                if (previous_b.origin.record.valid) retired=retired+1;
                if (expected_a.record.valid) captured=captured+1;
                previous_b=expected_b;
            end
            held_a=actual_a; held_b=actual_b;
            steps=steps+1;
            if (steps%8192==0) $display("NUMERIC_FOCUS_PROGRESS steps=%0d captured=%0d retired=%0d cancelled=%0d",steps,captured,retired,cancelled);
        end
    endtask
    task cold_boundary;
        begin cold_rst_n=0; step(); cold_rst_n=1; end
    endtask
    initial begin
        if (!$value$plusargs("MODE=%s",mode)) $fatal(1,"missing focused mode");
        $display("NUMERIC_FOCUS_SCHEMA mode=%s a_bits=%0d b_bits=%0d",mode,$bits(a_record_t),$bits(b_record_t));
        step(); cold_rst_n=1;
        // Fill both stages before the first cancellation; controls fail promptly.
        load_case(reference_symbolic_seed(13'd31,12'habc),11'd7,1'b0); step();
        load_case(reference_symbolic_seed(13'd17,12'habf),11'd0,1'b1); step();
        cold_boundary();
        for (r=0;r<8;r=r+1)
            for (low=0;low<4;low=low+1)
                for (count=0;count<2048;count=count+1)
                    for (en=0;en<4;en=en+1) begin
                        load_case(reference_symbolic_seed(boundary_remaining(r),12'hab0+low),count[10:0],four_state(en));
                        step(); binary_cases=binary_cases+1;
                    end
        cold_boundary();
        for (base=0;base<4;base=base+1)
            for (target=0;target<9;target=target+1)
                for (bit_index=0;bit_index<51;bit_index=bit_index+1)
                    for (unknown_kind=0;unknown_kind<2;unknown_kind=unknown_kind+1) begin
                        description=baseline_description(base); altered_description=description;
                        altered_description[bit_index]=unknown_kind ? 1'bz : 1'bx;
                        load_case(description,11'd7,1'b1);
                        if (target<4) in_record.queries[target].description=altered_description;
                        else if (target<8) in_record.eops[target-4].description=altered_description;
                        else in_record.pre_rc_context.description=altered_description;
                        step(); description_cases=description_cases+1;
                    end
        cold_boundary();
        for (base=0;base<4;base=base+1)
            for (target=0;target<9;target=target+1)
                for (bit_index=0;bit_index<11;bit_index=bit_index+1)
                    for (unknown_kind=0;unknown_kind<2;unknown_kind=unknown_kind+1) begin
                        load_case(baseline_description(base),11'd7,1'b1);
                        altered_words=7; altered_words[bit_index]=unknown_kind ? 1'bz : 1'bx;
                        if (target<4) in_record.queries[target].words=altered_words;
                        else if (target<8) in_record.eops[target-4].words=altered_words;
                        else in_record.pre_rc_context.words=altered_words;
                        step(); word_cases=word_cases+1;
                    end
        cold_boundary();
        for (vr=0;vr<4;vr=vr+1)
            for (vq=0;vq<4;vq=vq+1)
                for (ve=0;ve<4;ve=ve+1) begin
                    load_case(baseline_description(1),11'd7,1'b1);
                    in_record.valid=four_state(vr);
                    for (slot=0;slot<4;slot=slot+1) begin
                        in_record.queries[slot].valid=four_state(vq);
                        in_record.eops[slot].valid=four_state(ve);
                    end
                    step(); validity_cases=validity_cases+1;
                end
        in_record='0; in_control='0; step();
        cold_boundary();
        if (binary_cases!=262144 || description_cases!=3672 || word_cases!=792 || validity_cases!=64 ||
            steps!=266681 || cold_rows!=6 || captured!=266626 || cancelled!=4 || retired!=266622 ||
            captured!=retired+cancelled || previous_b!=='0) $fatal(1,"NUMERIC_FOCUS_INCOMPLETE");
        $display("NUMERIC_FOCUS_PASS mode=%s steps=%0d binary=%0d description_xz=%0d words_xz=%0d validity=%0d cold=%0d captured=%0d retired=%0d cancelled=%0d a_bits=%0d b_bits=%0d directed=2",
            mode,steps,binary_cases,description_cases,word_cases,validity_cases,cold_rows,captured,retired,cancelled,$bits(a_record_t),$bits(b_record_t));
        $finish;
    end
    initial begin #2000000; $fatal(1,"NUMERIC_FOCUS_WATCHDOG"); end
endmodule
