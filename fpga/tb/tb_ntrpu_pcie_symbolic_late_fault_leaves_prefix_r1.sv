`timescale 1ns / 1ps
// Direct comparison on arbitrary B records, including malformed and X/Z inputs.
module tb_ntrpu_pcie_symbolic_late_fault_leaves_prefix_r1;
    import ntrpu_pcie_symbolic_numeric_pkg_r1::*;
    localparam integer BITS=$bits(b_record_t);
    b_record_t stimulus='0;
    wire [15:0] original_faults, candidate_faults;
    ntrpu_pcie_symbolic_late_validation_r1 original_i(.in_b(stimulus),.faults(original_faults));
    ntrpu_pcie_symbolic_late_fault_leaves_prefix_r1 candidate_i(.in_b(stimulus),.faults(candidate_faults));
    wire [15:0] leaf_reference_faults;
    ntrpu_pcie_symbolic_late_fault_leaves_r1 leaf_reference_i(.in_b(stimulus),.faults(leaf_reference_faults));
    integer direction, sop_lane, end_lane, partial_mode, seen_case, end_mode, initial_seen;
    integer prefix_cases=0;
    reg [3:0] prefix_eop_count;
    reg prefix_partial;
    reg [31:0] rng=32'h8a719c43;
    reg [BITS+31:0] random_bits;
    integer samples=0, zero_samples=0, nonzero_samples=0;
    integer hits[0:15];
    integer direct_lane_cases=0, unknown_cases=0, random_cases=0;
    integer lane, bit_at, mode, value, batch, k;

    task compare(input string label, input integer expected_mask);
        begin
            #1;
            if (original_faults!==candidate_faults)
                $fatal(1,"FAULT_LEAVES_PREFIX_MISMATCH sample=%0d label=%s original=%h candidate=%h",samples,label,original_faults,candidate_faults);
            if (leaf_reference_faults!==original_faults)
                $fatal(1,"FAULT_LEAVES_PREFIX_PARENT_MISMATCH sample=%0d label=%s",samples,label);
            // Compare before the OR so one lane cannot hide another lane's error.
            for (integer check_lane=0;check_lane<16;check_lane=check_lane+1) begin
                if (leaf_reference_i.rc_faults[check_lane]!==candidate_i.rc_faults[check_lane])
                    $fatal(1,"FAULT_LEAVES_PREFIX_LANE_MISMATCH sample=%0d label=%s direction=rc lane=%0d original=%h candidate=%h",samples,label,check_lane,leaf_reference_i.rc_faults[check_lane],candidate_i.rc_faults[check_lane]);
                if (leaf_reference_i.rq_faults[check_lane]!==candidate_i.rq_faults[check_lane])
                    $fatal(1,"FAULT_LEAVES_PREFIX_LANE_MISMATCH sample=%0d label=%s direction=rq lane=%0d original=%h candidate=%h",samples,label,check_lane,leaf_reference_i.rq_faults[check_lane],candidate_i.rq_faults[check_lane]);
            end
            if ((^original_faults)===1'bx)
                $fatal(1,"FAULT_LEAVES_PREFIX_UNKNOWN_ORACLE sample=%0d label=%s",samples,label);
            if (expected_mask>=0 && original_faults!==expected_mask[15:0])
                $fatal(1,"FAULT_LEAVES_PREFIX_DIRECT_ORACLE_MISMATCH sample=%0d label=%s expected=%h original=%h",samples,label,expected_mask[15:0],original_faults);
            samples=samples+1;
            if (original_faults==0) zero_samples=zero_samples+1;
            else nonzero_samples=nonzero_samples+1;
            for (integer n=0;n<16;n=n+1) if (original_faults[n]) hits[n]=hits[n]+1;
        end
    endtask

    task empty_record;
        begin stimulus='0; end
    endtask

    task rc_payload;
        begin
            stimulus='0;
            stimulus.origin.record.valid=1;
            stimulus.origin.record.pre_observing=1;
            stimulus.origin.record.rc_valid=1;
            stimulus.origin.record.rc_ready=1;
            stimulus.origin.record.pre_rc_partial=1;
            stimulus.origin.record.pre_rc_context.normal=1;
            stimulus.origin.record.pre_rc_context.seen=3;
            stimulus.origin.record.pre_rc_context.expected=19;
            stimulus.origin.record.rc_user[63:0]={16{4'hf}};
            stimulus.context_delivered=64;
        end
    endtask

    task rq_payload;
        begin
            stimulus='0;
            stimulus.origin.record.valid=1;
            stimulus.origin.record.pre_observing=1;
            stimulus.origin.record.rq_valid=1;
            stimulus.origin.record.rq_ready=15;
            stimulus.origin.record.pre_rq_partial=1;
            stimulus.origin.record.pre_rq_context={11'd4,11'd20,4'hf,4'hf,11'd16,1'b1,1'b0};
        end
    endtask

    task valid_header;
        begin
            stimulus='0;
            stimulus.origin.record.valid=1;
            stimulus.origin.record.pre_observing=1;
            stimulus.origin.record.expected_requester=16'h0123;
            stimulus.origin.record.rc_valid=1;
            stimulus.origin.record.rc_ready=1;
            stimulus.origin.record.rc_user[67:64]=1;
            stimulus.origin.record.rc_user[79:76]=1;
            stimulus.origin.record.rc_user[83:80]=3;
            stimulus.origin.record.rc_user[15:12]=15;
            stimulus.origin.record.rc_data[42:32]=1;
            stimulus.origin.record.rc_data[30]=1;
            stimulus.origin.record.rc_data[63:48]=16'h0123;
            stimulus.origin.record.rc_data[28:16]=4;
            stimulus.origin.record.queries[0].valid=1;
            stimulus.origin.record.queries[0].owned=1;
            stimulus.origin.queries[0]={13'd4,12'd0};
            stimulus.query_delivered[0]=4;
        end
    endtask

    task random_record;
        begin
            for (integer at=0;at<BITS;at=at+32) begin
                rng=rng^(rng<<13); rng=rng^(rng>>17); rng=rng^(rng<<5);
                random_bits[at+:32]=rng;
            end
            stimulus=random_bits[BITS-1:0];
        end
    endtask

    initial begin
        for(k=0;k<16;k=k+1) hits[k]=0;
        if(BITS!=2892) $fatal(1,"FAULT_LEAVES_PREFIX_INPUT_WIDTH_CHANGED bits=%0d",BITS);
        compare("initial_zero",0);
        // Each of the 32 leaves gets its own externally visible, hand-checked fault.
        for(mode=0;mode<2;mode=mode+1) begin
            for(lane=0;lane<16;lane=lane+1) begin
                for(value=0;value<2;value=value+1) begin
                    if(mode==0) rc_payload(); else rq_payload();
                    if(mode==0) stimulus.origin.record.rc_data[lane*32+:8]=(value==0) ? 8'hxx : 8'hzz;
                    else stimulus.origin.record.rq_data[lane*32+:8]=(value==0) ? 8'hxx : 8'hzz;
                    compare($sformatf("payload_%0d_lane_%0d_xz_%0d",mode,lane,value),16'h0010);
                    direct_lane_cases=direct_lane_cases+1;
                    empty_record(); compare("clear_after_leaf",0);
                end
            end
        end
        valid_header(); compare("valid_header",0);
        stimulus.origin.record.rc_data[48]=~stimulus.origin.record.rc_data[48];
        compare("header_requester_fault",16'h0080);
        valid_header(); stimulus.origin.record.rc_data[16]=~stimulus.origin.record.rc_data[16];
        compare("header_value_fault",16'h0100);
        valid_header(); stimulus.origin.record.rc_data[30]=0;
        compare("header_terminal_fault",16'h0200);
        valid_header(); stimulus.origin.record.rc_user[15:12]=7;
        compare("payload_enable_fault",16'h0400);
        empty_record(); compare("clear_after_header",0);
        // Inactive and unknown control must preserve procedural-if semantics.
        for(value=0;value<4;value=value+1) begin
            rc_payload(); stimulus.origin.record.rc_data[7:0]='x;
            case(value)
                0: stimulus.origin.record.valid=0;
                1: stimulus.origin.record.valid=1'bx;
                2: stimulus.origin.record.valid=1'bz;
                3: stimulus.origin.record.pre_observing=1'bx;
            endcase
            compare("inactive_or_unknown_observation",0);
        end
        rq_payload(); stimulus.origin.record.pre_rq_context[0]=1;
        stimulus.origin.record.rq_data='x;
        compare("aborted_payload_unknown",0);
        rc_payload(); stimulus.origin.record.rc_user[96]=1;
        stimulus.origin.record.rc_user[63:0]=0;
        stimulus.origin.record.rc_data='x;
        compare("poisoned_payload_unknown",0);
        // Cover every aligned SOP and EOP position, including end-before-start,
        // same-lane SOP/EOP, inactive/X/Z inherited state and eleven-bit wrap.
        for(direction=0;direction<2;direction=direction+1)
            for(sop_lane=0;sop_lane<16;sop_lane=sop_lane+4)
                for(end_lane=0;end_lane<16;end_lane=end_lane+1)
                    for(partial_mode=0;partial_mode<4;partial_mode=partial_mode+1)
                        for(seen_case=0;seen_case<10;seen_case=seen_case+1)
                            for(end_mode=0;end_mode<2;end_mode=end_mode+1) begin
                                case(seen_case)
                                    0: initial_seen=0;
                                    1: initial_seen=1;
                                    2: initial_seen=2;
                                    3: initial_seen=3;
                                    4: initial_seen=1023;
                                    5: initial_seen=1024;
                                    6: initial_seen=1027;
                                    7: initial_seen=1028;
                                    8: initial_seen=2046;
                                    9: initial_seen=2047;
                                endcase
                                case(partial_mode)
                                    0: prefix_partial=0;
                                    1: prefix_partial=1;
                                    2: prefix_partial=1'bx;
                                    3: prefix_partial=1'bz;
                                endcase
                                prefix_eop_count=(end_mode==0) ? 1 : 3;
                                if(direction==0) begin
                                    rc_payload();
                                    stimulus.origin.record.pre_rc_partial=prefix_partial;
                                    stimulus.origin.record.pre_rc_context.seen=initial_seen;
                                    stimulus.origin.record.expected_requester=16'h0123;
                                    stimulus.origin.record.rc_data='x;
                                    stimulus.origin.record.rc_data[sop_lane*32+:128]=0;
                                    stimulus.origin.record.rc_data[sop_lane*32+42-:11]=1;
                                    stimulus.origin.record.rc_data[sop_lane*32+30]=1;
                                    stimulus.origin.record.rc_data[sop_lane*32+63-:16]=16'h0123;
                                    stimulus.origin.record.rc_data[sop_lane*32+28-:13]=4;
                                    stimulus.origin.record.queries[sop_lane/4].owned=1;
                                    stimulus.origin.queries[sop_lane/4]={13'd4,12'd0};
                                    stimulus.query_delivered[sop_lane/4]=4;
                                    stimulus.origin.record.rc_user[67:64]=1;
                                    stimulus.origin.record.rc_user[69:68]=sop_lane/4;
                                    stimulus.origin.record.rc_user[79:76]=prefix_eop_count;
                                    stimulus.origin.record.rc_user[83:80]=end_lane;
                                    stimulus.origin.record.rc_user[87:84]=(end_lane+1)%16;
                                end else begin
                                    rq_payload();
                                    stimulus.origin.record.pre_rq_partial=prefix_partial;
                                    stimulus.origin.record.pre_rq_context={initial_seen[10:0],11'd20,4'hf,4'hf,11'd16,1'b1,1'b0};
                                    stimulus.origin.record.rq_data='x;
                                    stimulus.origin.record.rq_data[sop_lane*32+:128]=0;
                                    stimulus.origin.record.rq_data[sop_lane*32+74-:11]=1;
                                    stimulus.origin.record.rq_data[sop_lane*32+78-:4]=1;
                                    stimulus.origin.record.rq_user[3:0]=15;
                                    stimulus.origin.record.rq_user[11:8]=15;
                                    stimulus.origin.record.rq_user[21:20]=1;
                                    stimulus.origin.record.rq_user[23:22]=sop_lane/4;
                                    stimulus.origin.record.rq_user[27:26]=prefix_eop_count[1:0];
                                    stimulus.origin.record.rq_user[31:28]=end_lane;
                                    stimulus.origin.record.rq_user[35:32]=(end_lane+1)%16;
                                end
                                compare($sformatf("prefix_d%0d_s%0d_e%0d_p%0d_c%0d_m%0d",direction,sop_lane,end_lane,partial_mode,initial_seen,end_mode),-1);
                                prefix_cases=prefix_cases+1;
                            end
        // All B-record bits independently become X and Z in three active contexts.
        for(mode=0;mode<3;mode=mode+1) begin
            for(bit_at=0;bit_at<BITS;bit_at=bit_at+1) begin
                for(value=0;value<2;value=value+1) begin
                    if(mode==0) rc_payload(); else if(mode==1) rq_payload(); else valid_header();
                    stimulus[bit_at]=(value==0) ? 1'bx : 1'bz;
                    compare("walking_unknown",-1);
                    unknown_cases=unknown_cases+1;
                end
            end
        end
        // Include malformed pointers, counter wrap, both directions and ungated data.
        for(batch=0;batch<20000;batch=batch+1) begin
            random_record();
            if(batch%4!=0) begin
                stimulus.origin.record.valid=1;
                stimulus.origin.record.pre_observing=1;
                stimulus.origin.record.rq_valid=1;
                stimulus.origin.record.rq_ready=15;
                stimulus.origin.record.rc_valid=1;
                stimulus.origin.record.rc_ready=1;
            end
            if(batch%5==0) stimulus.origin.record.rc_data[batch%512]=1'bx;
            if(batch%7==0) stimulus.origin.record.rq_data[batch%512]=1'bz;
            if(batch%11==0) stimulus.origin.record.rc_user[batch%161]=1'bx;
            if(batch%13==0) stimulus.origin.record.rq_user[batch%137]=1'bz;
            compare("random_record",-1);
            random_cases=random_cases+1;
        end
        empty_record(); compare("final_clear",0);
        if(direct_lane_cases!=64 || unknown_cases!=17352 || random_cases!=20000 || prefix_cases!=10240 || samples!=47734)
            $fatal(1,"FAULT_LEAVES_PREFIX_COUNTS samples=%0d lane=%0d unknown=%0d random=%0d",samples,direct_lane_cases,unknown_cases,random_cases);
        for(k=0;k<16;k=k+1) begin
            if((k==4 || k==7 || k==8 || k==9 || k==10) && hits[k]==0)
                $fatal(1,"FAULT_LEAVES_PREFIX_MISSING_BIT_COVERAGE bit=%0d",k);
            $display("FAULT_LEAVES_PREFIX_COVERAGE bit=%0d hits=%0d",k,hits[k]);
        end
        $display("FAULT_LEAVES_PREFIX_PREFIX_CASES count=%0d lane_vectors_per_sample=32 bits_per_lane=16",prefix_cases);
        $display("FAULT_LEAVES_PREFIX_PASS samples=%0d bits=16 input_bits=%0d lane_cases=%0d unknown_cases=%0d random_cases=%0d zero=%0d nonzero=%0d rng=%h",samples,BITS,direct_lane_cases,unknown_cases,random_cases,zero_samples,nonzero_samples,rng);
        $finish;
    end
endmodule
