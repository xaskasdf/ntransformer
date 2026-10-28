`timescale 1ns / 1ps
// Source-derived functions; compare against signed arithmetic and unchanged R2.
module tb_ntrpu_pcie_symbolic_capacity_r3_omit_second;
    function automatic integer original_thermometer(input [3:0] mask);
        case (mask)
            0: original_thermometer=0;
            1: original_thermometer=1;
            3: original_thermometer=2;
            7: original_thermometer=3;
            15: original_thermometer=4;
            default: original_thermometer=-1;
        endcase
    endfunction

    function automatic [18:0] original_rq_admission_flags(
        input [511:0] data,
        input [136:0] user_bits,
        input [255:0] allocated,
        input [3:0] released_valid,
        input [31:0] released_tags,
        input signed [10:0] base_count
    );
        reg [1:0] pointer0,pointer1,position [0:1];
        reg valid0,valid1;
        reg [1:0] event_valid;
        reg [3:0] admitted,blocked;
        reg previous_valid,busy;
        reg [7:0] tag,previous_tag;
        reg [127:0] header;
        reg signed [10:0] current_count;
        integer starts,event_index,scan;
        begin
            starts=original_thermometer({2'b0,user_bits[21:20]});
            pointer0=user_bits[23:22];pointer1=user_bits[25:24];
            // An unknown indexed procedural write to sop_map changes no bit.
            valid0=(starts>0 && (^pointer0)!==1'bx);
            valid1=(starts>1 && (^pointer1)!==1'bx);
            event_valid=0;position[0]=0;position[1]=0;
            if (valid1 && (!valid0 || pointer1<=pointer0)) begin
                event_valid[0]=1;position[0]=pointer1;
                if (valid0 && pointer0!=pointer1) begin
                    event_valid[1]=1;position[1]=pointer0;
                end
            end else if (valid0) begin
                event_valid[0]=1;position[0]=pointer0;
                if (valid1) begin event_valid[1]=1;position[1]=pointer1;end
            end
            admitted=0;blocked=0;previous_valid=0;previous_tag=0;
            current_count=base_count;header=0;tag=0;busy=0;
            for (event_index=0;event_index<2;event_index=event_index+1) begin
                if (event_valid[event_index]) begin
                    header=data >> (position[event_index]*128);
                    tag=header[103:96];
                    if (header[78:75]==0) begin
                        busy=allocated[tag];
                        for (scan=0;scan<4;scan=scan+1)
                            if (released_valid[scan] && released_tags[scan*8+:8]==tag) busy=0;
                        if (previous_valid && previous_tag==tag) busy=1;
                        // Preserve procedural-if X behavior and signed count.
                        if (busy || current_count>=256) blocked[position[event_index]]=1;
                        else begin
                            admitted[position[event_index]]=1;
                            current_count=current_count+1;
                            previous_valid=1;previous_tag=tag;
                        end
                    end
                end
            end
            original_rq_admission_flags={current_count,blocked,admitted};
        end
    endfunction

    function automatic integer candidate_thermometer(input [3:0] mask);
        case (mask)
            0: candidate_thermometer=0;
            1: candidate_thermometer=1;
            3: candidate_thermometer=2;
            7: candidate_thermometer=3;
            15: candidate_thermometer=4;
            default: candidate_thermometer=-1;
        endcase
    endfunction

    function automatic [1:0] candidate_rq_capacity_flags(input [8:0] stored_count, input [2:0] releases);
        reg [7:0] before_first, after_first;
        begin
            before_first[0]=(stored_count>=9'd256);
            after_first[0]=(stored_count>=9'd255);
            before_first[1]=(stored_count>=9'd257);
            after_first[1]=(stored_count>=9'd256);
            before_first[2]=(stored_count>=9'd258);
            after_first[2]=(stored_count>=9'd257);
            before_first[3]=(stored_count>=9'd259);
            after_first[3]=(stored_count>=9'd258);
            before_first[4]=(stored_count>=9'd260);
            after_first[4]=(stored_count>=9'd259);
            before_first[5]=(stored_count>=9'd261);
            after_first[5]=(stored_count>=9'd260);
            before_first[6]=(stored_count>=9'd262);
            after_first[6]=(stored_count>=9'd261);
            before_first[7]=(stored_count>=9'd263);
            after_first[7]=(stored_count>=9'd262);
            candidate_rq_capacity_flags={before_first[releases],before_first[releases]};
        end
    endfunction

    function automatic [18:0] candidate_rq_admission_flags(
        input [511:0] data,
        input [136:0] user_bits,
        input [255:0] allocated,
        input [3:0] released_valid,
        input [31:0] released_tags,
        input signed [10:0] base_count,
        input [1:0] capacity_flags
    );
        reg [1:0] pointer0,pointer1,position [0:1];
        reg valid0,valid1;
        reg [1:0] event_valid;
        reg [3:0] admitted,blocked;
        reg previous_valid,busy;
        reg [7:0] tag,previous_tag;
        reg [127:0] header;
        reg signed [10:0] current_count;
        integer starts,event_index,scan;
        begin
            starts=candidate_thermometer({2'b0,user_bits[21:20]});
            pointer0=user_bits[23:22];pointer1=user_bits[25:24];
            // An unknown indexed procedural write to sop_map changes no bit.
            valid0=(starts>0 && (^pointer0)!==1'bx);
            valid1=(starts>1 && (^pointer1)!==1'bx);
            event_valid=0;position[0]=0;position[1]=0;
            if (valid1 && (!valid0 || pointer1<=pointer0)) begin
                event_valid[0]=1;position[0]=pointer1;
                if (valid0 && pointer0!=pointer1) begin
                    event_valid[1]=1;position[1]=pointer0;
                end
            end else if (valid0) begin
                event_valid[0]=1;position[0]=pointer0;
                if (valid1) begin event_valid[1]=1;position[1]=pointer1;end
            end
            admitted=0;blocked=0;previous_valid=0;previous_tag=0;
            current_count=base_count;header=0;tag=0;busy=0;
            for (event_index=0;event_index<2;event_index=event_index+1) begin
                if (event_valid[event_index]) begin
                    header=data >> (position[event_index]*128);
                    tag=header[103:96];
                    if (header[78:75]==0) begin
                        busy=allocated[tag];
                        for (scan=0;scan<4;scan=scan+1)
                            if (released_valid[scan] && released_tags[scan*8+:8]==tag) busy=0;
                        if (previous_valid && previous_tag==tag) busy=1;
                        // Preserve procedural-if X behavior and signed count.
                        if (busy || (previous_valid ? capacity_flags[1] : capacity_flags[0])) blocked[position[event_index]]=1;
                        else begin
                            admitted[position[event_index]]=1;
                            current_count=current_count+1;
                            previous_valid=1;previous_tag=tag;
                        end
                    end
                end
            end
            candidate_rq_admission_flags={current_count,blocked,admitted};
        end
    endfunction

    reg [8:0] stored_count;
    reg [2:0] releases;
    reg signed [10:0] base_count, plus_one;
    reg [1:0] flags;
    reg [511:0] data;
    reg [136:0] user_bits;
    reg [255:0] allocated;
    reg [3:0] released_valid;
    reg [31:0] released_tags;
    reg [18:0] expected, actual;
    integer code, relcode, bit_index, n, mask, starts, p0, p1, pattern, lane;
    integer predicate_cases=0, admission_cases=0;
    integer supplementary=0, count_pattern, unknown_bit, value_kind;

    task check_predicate;
        begin
            base_count=$signed({2'b0,stored_count})-$signed({8'b0,releases});
            plus_one=base_count+11'sd1;
            flags=candidate_rq_capacity_flags(stored_count,releases);
            if (flags !== {(plus_one>=11'sd256),(base_count>=11'sd256)})
                $fatal(1,"CAPACITY_PREDICATE_MISMATCH case=%0d count=%b releases=%b flags=%b base=%b",predicate_cases,stored_count,releases,flags,base_count);
            predicate_cases=predicate_cases+1;
        end
    endtask

    task check_admission;
        begin
            releases=({2'b0,released_valid[0]}+{2'b0,released_valid[1]})+
                     ({2'b0,released_valid[2]}+{2'b0,released_valid[3]});
            base_count=$signed({2'b0,stored_count})-$signed({8'b0,releases});
            expected=original_rq_admission_flags(data,user_bits,allocated,released_valid,released_tags,base_count);
            actual=candidate_rq_admission_flags(data,user_bits,allocated,released_valid,released_tags,base_count,
                                               candidate_rq_capacity_flags(stored_count,releases));
            if (actual !== expected)
                $fatal(1,"CAPACITY_ADMISSION_MISMATCH case=%0d count=%b release=%b user=%h expected=%h actual=%h",admission_cases,stored_count,released_valid,user_bits,expected,actual);
            admission_cases=admission_cases+1;
        end
    endtask

    initial begin
        // All four-state combinations of nine counter bits and three release bits.
        for (code=0;code<262144;code=code+1) begin
            for (bit_index=0;bit_index<9;bit_index=bit_index+1)
                case ((code>>(2*bit_index))&3)
                    0:stored_count[bit_index]=0;
                    1:stored_count[bit_index]=1;
                    2:stored_count[bit_index]=1'bx;
                    3:stored_count[bit_index]=1'bz;
                endcase
            for (relcode=0;relcode<64;relcode=relcode+1) begin
                for (bit_index=0;bit_index<3;bit_index=bit_index+1)
                    case ((relcode>>(2*bit_index))&3)
                        0:releases[bit_index]=0;
                        1:releases[bit_index]=1;
                        2:releases[bit_index]=1'bx;
                        3:releases[bit_index]=1'bz;
                    endcase
                check_predicate();
            end
            if ((code%65536)==0) $display("CAPACITY_PREDICATE_PROGRESS codes=%0d cases=%0d",code+1,predicate_cases);
        end
        $display("CAPACITY_PREDICATE_PASS cases=%0d",predicate_cases);
        // All stored counts, release masks, SOP thermometers and both pointers.
        // Four ownership/kind patterns: free, busy, duplicate tag, posted mix.
        for (n=0;n<512;n=n+1)
            for (mask=0;mask<16;mask=mask+1)
                for (starts=0;starts<4;starts=starts+1)
                    for (p0=0;p0<4;p0=p0+1)
                        for (p1=0;p1<4;p1=p1+1)
                            for (pattern=0;pattern<4;pattern=pattern+1) begin
                                stored_count=n;released_valid=mask;
                                released_tags={8'd9,8'd7,8'd8,8'd7};
                                data=0;user_bits=0;allocated=0;
                                if (pattern==1) allocated='1;
                                for (lane=0;lane<4;lane=lane+1) begin
                                    data[lane*128+96+:8]=(pattern==2) ? 7 : 7+lane;
                                    data[lane*128+75+:4]=(pattern==3 && (lane%2)==1) ? 1 : 0;
                                end
                                user_bits[21:20]=starts;user_bits[23:22]=p0;user_bits[25:24]=p1;
                                check_admission();
                            end
        $display("CAPACITY_ADMISSION_BINARY_PASS cases=%0d",admission_cases);
        // Unknowns at every relevant operand bit, with multiple capacity states.
        for (count_pattern=0;count_pattern<6;count_pattern=count_pattern+1)
            for (unknown_bit=0;unknown_bit<81;unknown_bit=unknown_bit+1)
                for (value_kind=0;value_kind<2;value_kind=value_kind+1) begin
                    case(count_pattern)
                        0:stored_count=0;1:stored_count=254;2:stored_count=255;
                        3:stored_count=256;4:stored_count=259;5:stored_count=511;
                    endcase
                    released_valid=4'b0111;released_tags={8'd9,8'd7,8'd8,8'd7};
                    allocated=0;allocated[7]=1;allocated[8]=1;
                    data=0;data[96+:8]=7;data[256+96+:8]=8;
                    user_bits=0;user_bits[21:20]=3;user_bits[25:24]=2;
                    if (unknown_bit<9) stored_count[unknown_bit]=value_kind ? 1'bz : 1'bx;
                    else if (unknown_bit<13) released_valid[unknown_bit-9]=value_kind ? 1'bz : 1'bx;
                    else if (unknown_bit<45) released_tags[unknown_bit-13]=value_kind ? 1'bz : 1'bx;
                    else if (unknown_bit<51) user_bits[20+unknown_bit-45]=value_kind ? 1'bz : 1'bx;
                    else if (unknown_bit<59) data[96+unknown_bit-51]=value_kind ? 1'bz : 1'bx;
                    else if (unknown_bit<67) data[256+96+unknown_bit-59]=value_kind ? 1'bz : 1'bx;
                    else if (unknown_bit<71) data[75+unknown_bit-67]=value_kind ? 1'bz : 1'bx;
                    else if (unknown_bit<75) data[256+75+unknown_bit-71]=value_kind ? 1'bz : 1'bx;
                    else allocated[7+unknown_bit-75]=value_kind ? 1'bz : 1'bx;
                    check_admission();supplementary=supplementary+1;
                end
        if (predicate_cases!=16777216 || admission_cases!=2098124 || supplementary!=972)
            $fatal(1,"CAPACITY_CASE_COUNT_MISMATCH predicates=%0d admissions=%0d supplementary=%0d",predicate_cases,admission_cases,supplementary);
        $display("CAPACITY_COMPLETE_PASS predicates=%0d admissions=%0d supplementary=%0d",predicate_cases,admission_cases,supplementary);
        $finish;
    end
endmodule
