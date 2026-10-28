`timescale 1ns / 1ps
module tb_ntrpu_pcie_rc_event_ancestry;
    reg [2:0] flags;
    reg [255:0] allocated_tags;
    reg [10:0] retained;
    reg [511:0] rc_data;
    reg [160:0] rc_user;
    wire [3:0] header_previous_valid, header_seed_retained, header_owned, header_release_valid;
    wire [7:0] header_previous_slot;
    wire [255:0] commit_valid;
    wire [511:0] commit_slot;
    wire [15:0] header_prefix_mask;
    wire [31:0] header_release_tags;
    wire [839:0] actual={header_release_tags,header_release_valid,header_owned,
        header_seed_retained,header_prefix_mask,commit_slot,commit_valid,header_previous_slot,header_previous_valid};
    wire [3:0] old_previous, old_seed;
    wire [7:0] old_slot;
    wire [255:0] old_commit;
    wire [511:0] old_commit_slot;
    wire [15:0] old_mask;
    wire [799:0] old_actual={old_seed,old_mask,old_commit_slot,old_commit,old_slot,old_previous};
    reg [839:0] vector_expected;
    reg [942:0] seed, poisoned;
    reg [7:0] partitions [0:14];
    integer file,fields,rows=0,checks=0,unknown_checks=0,part,combo,at,context_id,bit_id,poison;
    string path;
    ntrpu_pcie_rc_update_select dut (
        .observing(flags[0]),.rc_valid(flags[1]),.rc_ready(flags[2]),
        .allocated_tags(allocated_tags),.rc_data(rc_data),.rc_user(rc_user),
        .retained_tag(retained[7:0]),.retained_owned(retained[8]),
        .retained_terminal(retained[9]),.retained_normal(retained[10]),
        .header_previous_valid(header_previous_valid),.header_previous_slot(header_previous_slot),
        .commit_valid(commit_valid),.commit_slot(commit_slot),
        .header_seed_retained(header_seed_retained),.header_prefix_mask(header_prefix_mask),
        .header_owned(header_owned),.header_release_valid(header_release_valid),.header_release_tags(header_release_tags)
    );
    ntrpu_pcie_rc_update_select_reference_r1 reference (
        .observing(flags[0]),.rc_valid(flags[1]),.rc_ready(flags[2]),
        .allocated_tags(allocated_tags),.rc_data(rc_data),.rc_user(rc_user),
        .retained_tag(retained[7:0]),.retained_owned(retained[8]),
        .retained_terminal(retained[9]),.retained_normal(retained[10]),
        .header_previous_valid(old_previous),.header_previous_slot(old_slot),
        .commit_valid(old_commit),.commit_slot(old_commit_slot),
        .header_seed_retained(old_seed),.header_prefix_mask(old_mask)
    );
    function automatic bit_value(input integer code);
        case(code) 0:bit_value=0; 1:bit_value=1; 2:bit_value=1'bx; default:bit_value=1'bz; endcase
    endfunction
    function automatic integer thermometer(input [3:0] mask);
        case(mask) 0:thermometer=0; 1:thermometer=1; 3:thermometer=2;
            7:thermometer=3; 15:thermometer=4; default:thermometer=-1; endcase
    endfunction
    // Independent original recurrence: scan physical DWORDs, consult only
    // releases actually accepted earlier, and preserve the raw lookup value.
    function automatic [39:0] sequential_ownership(
        input [2:0] control, input [255:0] allocated,
        input [511:0] data, input [160:0] user_bits
    );
        reg [15:0] sop;
        reg [3:0] owned_flags, releases;
        reg [31:0] tags;
        reg [7:0] tag;
        reg owned;
        integer starts,slot,ptr,lane,scan;
        begin
            sop=0; owned_flags=0; releases=0; tags=0; tag=0; owned=0;
            if(control[0] && control[1]===1'b1 && control[2]===1'b1) begin
                starts=thermometer(user_bits[67:64]);
                for(slot=0;slot<4;slot=slot+1) if(slot<starts) begin
                    ptr=4*user_bits[68+slot*2+:2]; sop[ptr]=1;
                end
                for(lane=0;lane<16;lane=lane+1) if(lane%4==0 && sop[lane]) begin
                    tag=data[lane*32+64+:8]; owned=allocated[tag];
                    for(scan=0;scan<4;scan=scan+1)
                        if(releases[scan] && tags[scan*8+:8]==tag) owned=0;
                    owned_flags[lane/4]=owned;
                    if(data[lane*32+30] && owned) begin
                        releases[lane/4]=1; tags[(lane/4)*8+:8]=tag;
                    end
                end
            end
            sequential_ownership={tags,releases,owned_flags};
        end
    endfunction
    task automatic check(input integer unknown_case);
        reg [839:0] expected;
        begin
            #1;
            expected={sequential_ownership(flags,allocated_tags,rc_data,rc_user),old_actual};
            if(actual!==expected) $fatal(1,"RC_OWNERSHIP_MISMATCH check=%0d actual=%h expected=%h flags=%h user=%h data=%h allocated=%h",checks,actual,expected,flags,rc_user,rc_data,allocated_tags);
            checks=checks+1; unknown_checks=unknown_checks+unknown_case;
        end
    endtask
    task automatic baseline(input integer kind);
        integer i;
        begin
            flags=7; allocated_tags=0; retained=11'h507; rc_data=0; rc_user=0;
            rc_user[67:64]=15; rc_user[79:76]=15;
            for(i=0;i<4;i=i+1) begin
                allocated_tags[7+i]=1;
                rc_data[i*128+64+:8]=7+((kind==0) ? 0 : (i%2));
                rc_data[i*128+30]=(kind==1 || kind==2) && i%2==0;
                rc_user[68+i*2+:2]=i; rc_user[80+i*4+:4]=3+4*i;
            end
            if(kind==2) retained[9]=1;
            if(kind==3) rc_user[75:68]=8'h55;
            if(kind==4) rc_user[75:68]=8'h1b;
            if(kind==5) rc_user[67:64]=4'b1010;
            if(kind==6) begin
                rc_user[67:64]=7; rc_user[73:68]=6'b111001; rc_user[83:80]=0;
            end
            if(kind==7) begin rc_user[96]=1; rc_data[15:12]=9; end
        end
    endtask
    initial begin
        partitions[0]=8'h00;
        partitions[1]=8'h40;
        partitions[2]=8'h10;
        partitions[3]=8'h50;
        partitions[4]=8'h90;
        partitions[5]=8'h04;
        partitions[6]=8'h44;
        partitions[7]=8'h84;
        partitions[8]=8'h14;
        partitions[9]=8'h54;
        partitions[10]=8'h94;
        partitions[11]=8'h24;
        partitions[12]=8'h64;
        partitions[13]=8'ha4;
        partitions[14]=8'he4;
        if(!$value$plusargs("OWNERSHIP_VECTORS=%s",path)) $fatal(1,"Missing OWNERSHIP_VECTORS");
        file=$fopen(path,"r"); if(!file) $fatal(1,"Cannot open ownership vectors");
        while(!$feof(file)) begin
            fields=$fscanf(file,"%h %h %h %h %h %h\n",flags,allocated_tags,retained,rc_data,rc_user,vector_expected);
            if(fields!=6) $fatal(1,"Malformed vector row=%0d",rows);
            check(0);
            if(actual!==vector_expected) $fatal(1,"RC_OWNERSHIP_MODEL_MISMATCH row=%0d",rows);
            rows=rows+1;
        end
        $fclose(file);
        if(rows!=155168) $fatal(1,"Incomplete binary ownership rows=%0d",rows);
        for(part=0;part<15;part=part+1) begin
            baseline(0);
            for(at=0;at<4;at=at+1) rc_data[at*128+64+:8]=7+partitions[part][at*2+:2];
            for(combo=0;combo<65536;combo=combo+1) begin
                for(at=0;at<4;at=at+1) begin
                    allocated_tags[7+at]=bit_value((combo>>(2*at))&3);
                    rc_data[at*128+30]=bit_value((combo>>(8+2*at))&3);
                end
                check((combo&16'haaaa)!=0);
            end
        end
        for(context_id=0;context_id<8;context_id=context_id+1) begin
            baseline(context_id); seed={rc_user,rc_data,retained,allocated_tags,flags};
            for(bit_id=0;bit_id<943;bit_id=bit_id+1) begin
                for(poison=2;poison<4;poison=poison+1) begin
                    poisoned=seed; poisoned[bit_id]=bit_value(poison);
                    {rc_user,rc_data,retained,allocated_tags,flags}=poisoned;
                    check(1);
                end
            end
        end
        if(checks!=1153296 || unknown_checks!=994288) $fatal(1,"Incomplete ownership checks");
        $display("RC_OWNERSHIP_PASS rows=%0d checks=%0d unknown_checks=%0d",rows,checks,unknown_checks);
        // Preserve the complete inherited bank, then jointly vary six controls.
        checks=0; unknown_checks=0;
        for(context_id=0;context_id<8;context_id=context_id+1) begin
            for(combo=0;combo<4096;combo=combo+1) begin
                baseline(0);
                if(context_id&1) begin
                    rc_user[67:64]=7; rc_user[73:68]=6'b111001;
                    rc_user[83:80]=0;
                end
                if(context_id&2) begin
                    for(at=0;at<4;at=at+1) rc_user[80+at*4+:4]=at;
                end
                if(context_id&4) begin
                    rc_user[95:80]=16'h7733; rc_user[96]=1;
                end
                retained[8]=bit_value(combo&3);
                retained[9]=bit_value((combo>>2)&3);
                retained[10]=bit_value((combo>>4)&3);
                allocated_tags[7]=bit_value((combo>>6)&3);
                rc_data[30]=bit_value((combo>>8)&3);
                rc_data[46]=bit_value((combo>>10)&3);
                check((combo&12'haaa)!=0);
            end
        end
        if(checks!=32768 || unknown_checks!=32256) $fatal(1,"Incomplete event ancestry controls");
        $display("RC_EVENT_ANCESTRY_PASS checks=%0d unknown_checks=%0d",checks,unknown_checks);
        $finish;
    end
endmodule
