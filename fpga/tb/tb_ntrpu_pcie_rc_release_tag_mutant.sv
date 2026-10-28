`timescale 1ns / 1ps
// Isolated combinational guard test. Canonical bitmap/count are deliberately
// injected by force; full integration benches retain public-input-only state.
module tb_ntrpu_pcie_rc_release_tag_mutant;
    reg [2:0] flags=7;
    reg [255:0] allocated=0;
    reg [8:0] tag_count=0;
    reg [511:0] rc_data=0, rq_data=0;
    reg [160:0] rc_user=0;
    reg [136:0] rq_user=0;
    reg [931:0] seed,poisoned;
    integer checks=0,unknown_checks=0,masked_differences=0,valid_checks=0;
    integer lane,tag_value,context_id,bit_id,polarity,combo,mask,pointers;
    ntrpu_pcie_inventory_scheduler_bad_release_tag dut (
        .clk(1'b0),.cold_rst_n(1'b1),.sequence_in(32'b0),
        .profile_valid(1'b1),.user_reset(1'b0),.link_up(1'b1),
        .requester_id(16'b0),.attribute_mask(2'b0),
        .rq_valid(1'b1),.rq_ready(4'b1111),.rq_data(rq_data),.rq_user(rq_user),
        .rc_valid(flags[1]),.rc_ready(flags[2]),.rc_data(rc_data),.rc_user(rc_user)
    );
    function automatic bit_value(input integer code);
        case(code) 0:bit_value=0;1:bit_value=1;2:bit_value=1'bx;default:bit_value=1'bz;endcase
    endfunction
    function automatic integer thermometer(input [3:0] bits);
        case(bits) 0:thermometer=0;1:thermometer=1;3:thermometer=2;
            7:thermometer=3;15:thermometer=4;default:thermometer=-1;endcase
    endfunction
    // Sequential physical-lane recurrence, independent of selector R3's
    // parallel ownership computation; preserves procedural-if four-state rules.
    function automatic [39:0] sequential_ownership(
        input [2:0] control,input [255:0] bitmap,
        input [511:0] data,input [160:0] user_bits
    );
        reg [15:0] sop;
        reg [3:0] owned_flags,releases;
        reg [31:0] tags;
        reg [7:0] tag;
        reg owned;
        integer starts,slot,ptr,at,scan;
        begin
            sop=0;owned_flags=0;releases=0;tags=0;tag=0;owned=0;
            if(control[0] && control[1]===1'b1 && control[2]===1'b1) begin
                starts=thermometer(user_bits[67:64]);
                for(slot=0;slot<4;slot=slot+1) if(slot<starts) begin
                    ptr=4*user_bits[68+slot*2+:2];sop[ptr]=1;
                end
                for(at=0;at<16;at=at+1) if(at%4==0 && sop[at]) begin
                    tag=data[at*32+64+:8];owned=bitmap[tag];
                    for(scan=0;scan<4;scan=scan+1)
                        if(releases[scan] && tags[scan*8+:8]==tag) owned=0;
                    owned_flags[at/4]=owned;
                    if(data[at*32+30] && owned) begin
                        releases[at/4]=1;tags[(at/4)*8+:8]=tag;
                    end
                end
            end
            sequential_ownership={tags,releases,owned_flags};
        end
    endfunction
    task automatic baseline(input integer kind);
        integer at;
        begin
            flags=7;allocated={256{1'b1}};tag_count=256;
            rc_data=0;rc_user=0;rq_data=0;rq_user=0;
            rc_user[67:64]=15;rc_user[79:76]=15;
            for(at=0;at<4;at=at+1) begin
                rc_data[at*128+64+:8]=(kind==1) ? 7 : (7+17*at);
                rc_data[at*128+30]=(kind==2) ? (at%2) : 1;
                rc_data[at*128+32+:11]=1;
                rc_user[68+at*2+:2]=at;rc_user[80+at*4+:4]=3+4*at;
            end
            if(kind==3) begin allocated=0;rc_user[75:68]=8'h1b;end
            rq_user[21:20]=3;rq_user[23:22]=0;rq_user[25:24]=2;
            rq_user[27:26]=3;rq_user[31:28]=3;rq_user[35:32]=11;
            rq_user[7:0]=8'hff;rq_data[64+:11]=1;rq_data[320+:11]=1;
            rq_data[96+:8]=7;rq_data[352+:8]=24;
        end
    endtask
    task automatic check(input integer unknown_case);
        reg [39:0] expected;
        reg [18:0] admission;
        reg [255:0] bitmap;
        reg signed [10:0] base_count;
        integer at,index,releases;
        begin
            #1;
            expected=sequential_ownership(flags,allocated,rc_data,rc_user);
            if(dut.rc_release_valid!==expected[7:4] || dut.selected_release_tags!==expected[39:8])
                $fatal(1,"RC_RELEASE_GUARD_SELECTOR_MISMATCH check=%0d",checks);
            releases=0;
            for(at=0;at<4;at=at+1) begin
                if(dut.rc_release_valid[at]!==1'b0 && dut.rc_release_valid[at]!==1'b1)
                    $fatal(1,"RC_RELEASE_GUARD_UNKNOWN_VALID check=%0d lane=%0d",checks,at);
                if(dut.rc_release_valid[at]) begin
                    if(dut.rc_release_tag[at]!==expected[8+at*8+:8])
                        $fatal(1,"RC_RELEASE_GUARD_VALID_TAG_MISMATCH check=%0d lane=%0d valid=1 actual=%h expected=%h",checks,at,dut.rc_release_tag[at],expected[8+at*8+:8]);
                    valid_checks=valid_checks+1;releases=releases+1;
                end else if(dut.rc_release_tag[at]!==expected[8+at*8+:8]) masked_differences=masked_differences+1;
                if(dut.rc_release_tag[at]!==rc_data[at*128+64+:8])
                    $fatal(1,"RC_RELEASE_GUARD_RAW_TAG_MISMATCH check=%0d lane=%0d",checks,at);
            end
            base_count=$signed({2'b0,tag_count})-releases;
            admission=0;
            if(flags[0]) admission=dut.rq_admission_flags(rq_data,rq_user,allocated,
                expected[7:4],expected[39:8],base_count);
            if(dut.rq_admission!==admission)
                $fatal(1,"RC_RELEASE_GUARD_ADMISSION_MISMATCH check=%0d actual=%h expected=%h",checks,dut.rq_admission,admission);
            // Reconstruct the bitmap using the old masked tags and old admission.
            bitmap=allocated;
            for(index=0;index<256;index=index+1) begin
                for(at=0;at<4;at=at+1)
                    if(expected[4+at] && expected[8+at*8+:8]==index) bitmap[index]=0;
                if(admission[0] && rq_data[96+:8]==index) bitmap[index]=1;
                if(admission[2] && rq_data[352+:8]==index) bitmap[index]=1;
            end
            if(dut.next_tags!==bitmap) $fatal(1,"RC_RELEASE_GUARD_BITMAP_MISMATCH check=%0d",checks);
            checks=checks+1;unknown_checks=unknown_checks+unknown_case;
        end
    endtask
    initial begin
        force dut.observing=flags[0];
        force dut.allocated_tags=allocated;
        force dut.tag_count=tag_count;
        $display("RC_RELEASE_GUARD_MODE mutant=1 canonical_state_injected=1");
        for(lane=0;lane<4;lane=lane+1)
            for(tag_value=0;tag_value<256;tag_value=tag_value+1) begin
                baseline(0);rc_data[lane*128+64+:8]=tag_value;
                case(tag_value%4) 0:tag_count=0;1:tag_count=255;2:tag_count=256;3:tag_count=511;endcase
                check(0);
            end
        for(mask=0;mask<16;mask=mask+1)
            for(pointers=0;pointers<256;pointers=pointers+1) begin
                baseline(1);rc_user[67:64]=mask;rc_user[75:68]=pointers;check(0);
            end
        for(context_id=0;context_id<4;context_id=context_id+1)
            for(combo=0;combo<64;combo=combo+1) begin
                baseline(context_id);
                flags={bit_value((combo>>4)&3),bit_value((combo>>2)&3),bit_value(combo&3)};
                check((combo&42)!=0);
            end
        for(context_id=0;context_id<4;context_id=context_id+1) begin
            baseline(context_id);seed={rc_user,rc_data,allocated,flags};
            for(bit_id=0;bit_id<932;bit_id=bit_id+1)
                for(polarity=2;polarity<4;polarity=polarity+1) begin
                    poisoned=seed;poisoned[bit_id]=bit_value(polarity);
                    {rc_user,rc_data,allocated,flags}=poisoned;check(1);
                end
        end
        if(checks!=12832 || unknown_checks!=7680 || valid_checks==0 || masked_differences==0)
            $fatal(1,"Incomplete release guard coverage checks=%0d unknown=%0d",checks,unknown_checks);
        $display("RC_RELEASE_GUARD_PASS checks=%0d unknown_checks=%0d valid_checks=%0d masked_differences=%0d",checks,unknown_checks,valid_checks,masked_differences);
        $finish;
    end
endmodule
