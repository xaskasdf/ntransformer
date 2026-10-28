`timescale 1ns / 1ps
// Focused raw-input comparison against the unchanged, qualified parent C.
// No internal state writes, force/release, filtered fault rows, or oracle feedback.
module tb_ntrpu_pcie_symbolic_control_predecode_rq_bitmap_r1;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    reg clk=0, cold_rst_n=0, profile_valid=1, user_reset=0, link_up=1;
    reg [31:0] sequence_in=0;
    reg [15:0] requester_id=16'h01a0;
    reg [1:0] attribute_mask=3;
    reg rq_valid=0, rc_valid=0, rc_ready=1;
    reg [3:0] rq_ready=15;
    reg [511:0] rq_data=0, rc_data=0;
    reg [136:0] rq_user=0;
    reg [160:0] rc_user=0;
    c_record_t reference_c, candidate_c;
    wire reference_pending, candidate_pending;
    ntrpu_pcie_symbolic_control_predecode_commit_priority_r1 reference_i(
        .clk(clk),.cold_rst_n(cold_rst_n),.sequence_in(sequence_in),.profile_valid(profile_valid),
        .user_reset(user_reset),.link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user),
        .out_c(reference_c),.control_pending(reference_pending));
    ntrpu_pcie_symbolic_control_predecode_rq_bitmap_r1 candidate_i(
        .clk(clk),.cold_rst_n(cold_rst_n),.sequence_in(sequence_in),.profile_valid(profile_valid),
        .user_reset(user_reset),.link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user),
        .out_c(candidate_c),.control_pending(candidate_pending));
    integer rows=0, pre=0, post=0, control_cases=0, kind_cases=0, handshake_cases=0;
    integer c,p0,p1,pattern,lane,kind_mask,ready_bits,valid_bits;
    integer fd,fields,vector_rows=0,scenario,native_ps;
    integer expected_observing,expected_fault,check_state,etags,ereads,ebytes;
    integer erp,ecp,erequests,ecompleted,ewrites;
    reg [255:0] expected_bitmap;
    string vectors,trailing;
    function automatic logic digit(input integer value);
        case(value) 0:digit=0; 1:digit=1; 2:digit=1'bx; default:digit=1'bz; endcase
    endfunction
    function automatic [1:0] pair(input integer value);
        pair={digit(value/4),digit(value%4)};
    endfunction
    task check(input integer phase);
        begin
            if ({candidate_c,candidate_pending,candidate_i.next_tags} !==
                {reference_c,reference_pending,reference_i.next_tags}) begin
                $display("RQ_BITMAP_ACTUAL_NEXT %h",candidate_i.next_tags);
                $display("RQ_BITMAP_EXPECTED_NEXT %h",reference_i.next_tags);
                $display("RQ_BITMAP_ACTUAL_C %h",candidate_c);
                $display("RQ_BITMAP_EXPECTED_C %h",reference_c);
                $fatal(1,"RQ_BITMAP_MISMATCH row=%0d phase=%0d",rows,phase);
            end
            if (phase==0) pre=pre+1; else post=post+1;
        end
    endtask
    task tick;
        begin
            #1; check(0); clk=1; #0.001; check(1); clk=0; #0.999;
            sequence_in=sequence_in+1; rows=rows+1;
        end
    endtask
    task reset_arm;
        begin
            cold_rst_n=0; profile_valid=1; user_reset=0; link_up=1;
            rq_valid=0; rc_valid=0; rq_ready=15; rc_ready=1;
            rq_data=0; rc_data=0; rq_user=0; rc_user=0;
            tick(); cold_rst_n=1; tick();
        end
    endtask
    task headers(input integer kinds, input integer unknown_tags);
        begin
            rq_data=0; rq_user=0; rq_user[7:0]=8'hff;
            for (lane=0;lane<4;lane=lane+1) begin
                rq_data[lane*128+:64]=64'h100+4*lane;
                rq_data[lane*128+64+:11]=1;
                rq_data[lane*128+75+:4]=kinds[lane] ? 1 : 0;
                rq_data[lane*128+80+:16]=requester_id;
                rq_data[lane*128+96+:8]=53+37*lane;
                if (unknown_tags) begin
                    case(lane)
                        0:rq_data[lane*128+96+:8]=8'hxx;
                        1:rq_data[lane*128+96+:8]=8'hzz;
                        2:rq_data[lane*128+96+:8]=8'b0x010101;
                        3:rq_data[lane*128+96+:8]=8'b1z010101;
                    endcase
                end
            end
        end
    endtask
    initial begin
        if (!$value$plusargs("VECTORS=%s",vectors)) $fatal(1,"RQ_BITMAP_ARGUMENTS");
        // Four mutation witnesses: idle, malformed lane3, tagX, tagZ.
        reset_arm(); tick();
        if (candidate_i.allocated_tags!==256'b0) $fatal(1,"RQ_BITMAP_IDLE_ORACLE");
        reset_arm(); headers(0,0); rq_valid=1; rq_user[21:20]=1; rq_user[23:22]=3;
        rq_data[480+:8]=23; tick();
        if (candidate_i.allocated_tags!==(256'b1<<23)) $fatal(1,"RQ_BITMAP_LANE3_ORACLE");
        reset_arm(); headers(0,0); rq_valid=1; rq_user[21:20]=1; rq_data[96+:8]=8'hxx; tick();
        if (candidate_i.allocated_tags!==256'b0) $fatal(1,"RQ_BITMAP_X_ORACLE");
        reset_arm(); headers(0,0); rq_valid=1; rq_user[21:20]=1; rq_data[96+:8]=8'hzz; tick();
        if (candidate_i.allocated_tags!==256'b0) $fatal(1,"RQ_BITMAP_Z_ORACLE");
        for(c=0;c<16;c=c+1) for(p0=0;p0<16;p0=p0+1) for(p1=0;p1<16;p1=p1+1)
            for(pattern=0;pattern<2;pattern=pattern+1) begin
                reset_arm(); headers(0,pattern); rq_valid=1;
                rq_user[21:20]=pair(c); rq_user[23:22]=pair(p0); rq_user[25:24]=pair(p1);
                tick(); tick(); control_cases=control_cases+1;
            end
        for(p0=0;p0<4;p0=p0+1) for(p1=0;p1<4;p1=p1+1) for(kind_mask=0;kind_mask<16;kind_mask=kind_mask+1) begin
            reset_arm(); headers(kind_mask,0); rq_valid=1;
            rq_user[21:20]=3; rq_user[23:22]=p0; rq_user[25:24]=p1;
            tick(); tick(); kind_cases=kind_cases+1;
        end
        for(ready_bits=0;ready_bits<16;ready_bits=ready_bits+1) for(valid_bits=0;valid_bits<4;valid_bits=valid_bits+1) begin
            reset_arm(); headers(0,0); rq_user[21:20]=3; rq_user[25:24]=2;
            rq_ready=ready_bits; rq_valid=digit(valid_bits); tick(); tick();
            handshake_cases=handshake_cases+1;
        end
        // Reviewed raw RQ supplement includes capacity and actual RC-before-RQ reuse.
        fd=$fopen(vectors,"r"); if(!fd) $fatal(1,"RQ_BITMAP_OPEN");
        while(vector_rows<8850) begin
            fields=$fscanf(fd,"%d %d %d %d %d %d %h %h %d %h %b %h %d %d %b %h %d %d %d %d %d %d %d %d %d %d %d %h\n",
                scenario,native_ps,cold_rst_n,profile_valid,user_reset,link_up,requester_id,attribute_mask,
                rq_valid,rq_ready,rq_data,rq_user,rc_valid,rc_ready,rc_data,rc_user,
                expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites,expected_bitmap);
            if(fields!=28) $fatal(1,"RQ_BITMAP_VECTOR_FIELDS row=%0d fields=%0d",vector_rows,fields);
            tick(); vector_rows=vector_rows+1;
        end
        fields=$fscanf(fd,"%s",trailing); if(fields!=-1) $fatal(1,"RQ_BITMAP_EXTRA_VECTOR"); $fclose(fd);
        if(rows!=42910 || pre!=rows || post!=rows || control_cases!=8192 || kind_cases!=256 || handshake_cases!=64)
            $fatal(1,"RQ_BITMAP_INCOMPLETE");
        $display("RQ_BITMAP_PASS rows=%0d pre=%0d post=%0d control_cases=%0d kind_cases=%0d handshake_cases=%0d vector_rows=%0d record_bits=%0d",
            rows,pre,post,control_cases,kind_cases,handshake_cases,vector_rows,$bits(c_record_t));
        $finish;
    end
    initial begin #200000; $fatal(1,"RQ_BITMAP_WATCHDOG"); end
endmodule
