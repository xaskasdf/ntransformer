`timescale 1ns / 1ps
// Passive PG213 observer: 512-bit DWORD-aligned RQ2/RC4, client tags, one PF.
// R7: preserve range-guard high bits alongside counters; evaluate after registers.
// Sticky range fault is visible on the same edge; no throughput/capacity change.
// Only configuration-cold reset clears state. There is deliberately no ACK.
// This inventory does not certify posted-write receipt or hidden core queues.
module ntrpu_pcie_request_inventory_symbolic_probe_numeric_ab_mapped_reference_r1 (
    input wire clk, cold_rst_n,
    input wire profile_valid, user_reset, link_up,
    input wire [15:0] requester_id,
    input wire [1:0] attribute_mask,
    input wire rq_valid, rc_valid, rc_ready,
    input wire [3:0] rq_ready,
    input wire [511:0] rq_data, rc_data,
    input wire [136:0] rq_user,
    input wire [160:0] rc_user,
    output reg observing=0,
    output reg [255:0] allocated_tags=0,
    output reg [8:0] tag_count=0, outstanding_reads=0,
    output reg [20:0] remaining_bytes=0,
    output reg rq_partial=0, rc_partial=0,
    output reg [31:0] requests=0, completed_reads=0, posted_writes=0,
    output wire [15:0] fault_sources,
    output wire protocol_fault, inventory_empty
);
    // faults: profile/link, handshake/stall, framing, descriptor, unknown,
    // duplicate/capacity, orphan, identity, byte-count/address, terminal,
    // payload enables/length, completion error, discontinue, counter range.
    reg [3:0] observed_query_valid,observed_query_owned;
    reg [119:0] observed_query_data;
    reg [51:0] observed_query_delivered;
    reg [12:0] remaining [0:255], next_remaining [0:255];
    reg [11:0] address_next [0:255], next_address [0:255];
    reg [2:0] traffic_class [0:255], next_tc [0:255];
    reg [1:0] attributes [0:255], next_attributes [0:255];
    reg [15:0] saved_requester=0;
    reg [1:0] saved_attribute_mask=0;
    reg [10:0] rq_seen=0, rq_expected=0;
    reg [3:0] rq_first=0, rq_last=0;
    reg [10:0] rq_count=0;
    reg rq_write=0, rq_aborted=0;
    reg [10:0] rc_seen=0, rc_expected=0;
    reg [7:0] rc_tag=0;
    reg rc_terminal=0, rc_normal=0, rc_owned=0;
    reg [12:0] rc_original=0, rc_delivered=0;
    reg [11:0] rc_address=0;
    reg rq_held=0, rc_held=0;
    reg [511:0] held_rq_data=0, held_rc_data=0;
    reg [136:0] held_rq_user=0;
    reg [160:0] held_rc_user=0;

    // Low counter bits retain the exact truncation/wrap behavior of R6.
    // Register only the six discarded high bits of the signed candidates.
    reg [1:0] tag_range_high=0, read_range_high=0, byte_range_high=0;
    reg range_checked=0;
    reg [15:0] fault_latched=0;
    wire signed [10:0] checked_tags=$signed({tag_range_high,tag_count});
    wire signed [10:0] checked_reads=$signed({read_range_high,outstanding_reads});
    wire signed [22:0] checked_bytes=$signed({byte_range_high,remaining_bytes});
    // Case equality preserves procedural-if behavior when an operand contains X.
    wire range_fault_now=range_checked && ((checked_tags<0 || checked_tags>256 ||
        checked_reads<0 || checked_reads>257 || checked_bytes<0 || checked_bytes>1052672)===1'b1);
    assign fault_sources=fault_latched | {2'b0,range_fault_now,13'b0};

    reg [255:0] next_tags;
    reg [15:0] faults;
    reg rp, cp, rw, ra, ct, cn, co;
    reg [10:0] rs, rexp, rcount, cs, cexp;
    reg [3:0] rfirst, rlast;
    reg [7:0] ctag;
    reg [12:0] coriginal, cdelivered;
    reg [11:0] caddress;
    // Signed guard widths cover the full stored range and every possible beat delta.
    // Counts change once per cycle, not through a serial 32-bit adder per DWORD.
    reg signed [10:0] ntags, nreads;
    reg signed [22:0] nbytes;
    reg [31:0] nrequests, ncompleted, nwrites;
    integer i, lane, slot, j;
    // Thermometers produce -1..4; pointers and newest EOP lane are -1..15.
    // Both pointer operands stay signed so the initial -1 sorts before lane0.
    reg signed [3:0] starts, ends;
    reg signed [4:0] ptr, prev_ptr;
    reg [1:0] first_index, last_index;
    // Even malformed eleven-bit counts fit: span -6..8188, available -3..8188.
    // Keep signed guards and the original mixed-sign comparison semantics.
    reg signed [14:0] span, available, delivered, payload_index;
    reg [12:0] first_address;
    reg [3:0] expected_enable, word_mask;
    reg [15:0] sop_map, eop_map;
    reg [1:0] rq_start_slot [0:15];
    // RC can end at most four packets per beat; RQ can start at most two.
    // Keep updates as records and forward the newest preceding RC write.
    // Commit once per tag after parsing RC and then RQ, preserving reuse order.
    reg [1:0] rc_end_slot [0:15];
    reg rq_end_slot [0:15];
    reg [3:0] rc_complete_events;
    reg [3:0] rc_release_valid;
    reg [7:0] rc_release_tag [0:3];
    reg [2:0] release_count;
    reg rq_tag_busy;
    reg [1:0] rq_complete_events, rq_posted_events;
    reg [12:0] rc_removed_span [0:3];
    reg signed [14:0] rq_added_span [0:1];
    reg [14:0] rc_span_sum;
    reg signed [15:0] rq_span_sum;
    reg [2:0] complete_count;
    reg [1:0] allocate_count, request_count, posted_count;
    reg [3:0] rc_update_valid;
    reg [1:0] rq_update_valid;
    // Acyclic prefix chain over the four possible DWORD header positions.
    // RQ still admits at most two packets; malformed pointers retain fault behavior.
    // No allocation decision reads the record written by that same header.
    reg [1:0] rq_prefix_valid [0:4];
    reg [7:0] rq_prefix_tag [0:4][0:1];
    reg signed [10:0] rq_prefix_count [0:4];
    reg [7:0] rc_update_tag [0:3], rq_update_tag [0:1];
    reg [12:0] rc_update_remaining [0:3], rq_update_remaining [0:1];
    reg [11:0] rc_update_address [0:3], rq_update_address [0:1];
    reg [3:0] rc_update_lane [0:3];
    reg [2:0] rq_update_tc [0:1];
    reg [1:0] rq_update_attributes [0:1];
    integer update_slot;
    reg signed [4:0] latest_rc_lane;
    reg [127:0] h;
    reg [10:0] count;
    reg [7:0] tag;
    reg [3:0] kind, error_code;
    reg terminal, normal;

    assign protocol_fault=|fault_sources;
    assign inventory_empty=observing && profile_valid && !user_reset && link_up &&
        !protocol_fault && tag_count==0 && outstanding_reads==0 &&
        remaining_bytes==0 && !rq_partial && !rc_partial && !rq_held && !rc_held &&
        !rq_valid && !rc_valid;

    function automatic integer thermometer(input [3:0] mask);
        case (mask)
            0: thermometer=0;
            1: thermometer=1;
            3: thermometer=2;
            7: thermometer=3;
            15: thermometer=4;
            default: thermometer=-1;
        endcase
    endfunction

    always @* begin
        observed_query_valid=0; observed_query_owned=0;
        observed_query_data=0; observed_query_delivered=0;
        next_tags=allocated_tags; faults=fault_sources;
        ntags=tag_count; nreads=outstanding_reads; nbytes=remaining_bytes;
        nrequests=requests; ncompleted=completed_reads; nwrites=posted_writes;
        rp=rq_partial; rs=rq_seen; rexp=rq_expected; rfirst=rq_first; rlast=rq_last;
        rcount=rq_count; rw=rq_write; ra=rq_aborted;
        cp=rc_partial; cs=rc_seen; cexp=rc_expected; ctag=rc_tag;
        ct=rc_terminal; cn=rc_normal; co=rc_owned;
        coriginal=rc_original; cdelivered=rc_delivered; caddress=rc_address;
        rc_update_valid=0; rq_update_valid=0; latest_rc_lane=-1;
        rc_complete_events=0; rq_complete_events=0; rq_posted_events=0;
        rc_release_valid=0; release_count=0; rq_tag_busy=0;
        rc_span_sum=0; rq_span_sum=0; complete_count=0;
        allocate_count=0; request_count=0; posted_count=0;
        for (i=0;i<4;i=i+1) begin
            rc_release_tag[i]=0;
            rc_update_tag[i]=0; rc_update_remaining[i]=0;
            rc_update_address[i]=0; rc_update_lane[i]=0; rc_removed_span[i]=0;
        end
        for (i=0;i<2;i=i+1) begin
            rq_update_tag[i]=0; rq_update_remaining[i]=0;
            rq_update_address[i]=0; rq_update_tc[i]=0; rq_update_attributes[i]=0; rq_added_span[i]=0;
        end
        for (i=0;i<5;i=i+1) begin
            rq_prefix_valid[i]=0; rq_prefix_count[i]=0;
            for (j=0;j<2;j=j+1) rq_prefix_tag[i][j]=0;
        end
        sop_map=0; eop_map=0; starts=0; ends=0; ptr=0; prev_ptr=-1;
        h=0; count=0; tag=0; kind=0; error_code=0; terminal=0; normal=0;
        first_index=0; last_index=0; span=0; first_address=0;
        available=0; delivered=0; payload_index=0; expected_enable=0; word_mask=0;
        for (i=0;i<16;i=i+1) begin rq_start_slot[i]=0; rc_end_slot[i]=0; rq_end_slot[i]=0; end
        if (observing) begin
            if (profile_valid!==1'b1 || user_reset!==1'b0 || link_up!==1'b1 ||
                requester_id!==saved_requester || attribute_mask!==saved_attribute_mask)
                faults[0]=1;
            if ((rq_valid!==1'b0 && rq_valid!==1'b1) ||
                (rc_valid!==1'b0 && rc_valid!==1'b1) ||
                (rq_ready!==4'b0000 && rq_ready!==4'b1111) ||
                (rc_ready!==1'b0 && rc_ready!==1'b1)) faults[1]=1;
            if (rq_held && (rq_valid!==1'b1 || rq_data!==held_rq_data || rq_user!==held_rq_user)) faults[1]=1;
            if (rc_held && (rc_valid!==1'b1 || rc_data!==held_rc_data || rc_user!==held_rc_user)) faults[1]=1;

            // RC first: a terminal descriptor permits same-edge tag reuse.
            // Its payload context stays separate until EOP, including across cycles.
            if (rc_valid===1'b1 && rc_ready===1'b1) begin
                starts=thermometer(rc_user[67:64]); ends=thermometer(rc_user[79:76]);
                if (starts<0 || ends<0 || (^rc_user[96:0])===1'bx) faults[2]=1;
                sop_map=0; eop_map=0; prev_ptr=-1;
                for (slot=0;slot<4;slot=slot+1) begin
                    if (slot<starts) begin
                        ptr=4*rc_user[68+slot*2+:2];
                        if (ptr<=prev_ptr) faults[2]=1;
                        sop_map[ptr]=1; prev_ptr=ptr;
                    end
                end
                prev_ptr=-1;
                for (slot=0;slot<4;slot=slot+1) begin
                    if (slot<ends) begin
                        ptr=rc_user[80+slot*4+:4];
                        if (ptr<=prev_ptr) faults[2]=1;
                        eop_map[ptr]=1; rc_end_slot[ptr]=slot; prev_ptr=ptr;
                    end
                end
                if ((!cp && starts==0) || (rc_user[96] && (starts!=0 || ends==0))) faults[2]=1;
                if (rc_user[96]) begin faults[12]=1; cn=0; end
                for (lane=0;lane<16;lane=lane+1) begin
                    if ((lane%4)==0 && sop_map[lane]) begin
                        if (cp) faults[2]=1;
                        h=rc_data >> (lane*32); error_code=h[15:12];
                        tag=h[71:64]; terminal=h[30]; count=h[42:32];
                        normal=(error_code==0 && h[45:43]==0 && h[46]==0);
                        if ((^error_code)===1'bx) faults[4]=1;
                        if (error_code==9) begin
                            if ((^{h[71:64],h[55:48],h[30]})===1'bx) faults[4]=1;
                        end else if ((^h[95:0])===1'bx) faults[4]=1;
                        cp=1; cs=0; cexp=3+count; ctag=tag; ct=terminal; cn=normal;
                        // A preceding terminal header releases its tag even if its
                        // payload is unfinished. Later RC cannot claim that tag.
                        co=allocated_tags[tag];
                        for (update_slot=0;update_slot<4;update_slot=update_slot+1)
                            if (rc_release_valid[update_slot] && rc_release_tag[update_slot]==tag) co=0;
                        coriginal=remaining[tag];
                        caddress=address_next[tag]; cdelivered=0;
                        latest_rc_lane=-1;
                        for (update_slot=0;update_slot<4;update_slot=update_slot+1) begin
                            if (rc_update_valid[update_slot] && rc_update_tag[update_slot]==tag &&
                                (latest_rc_lane<0 || rc_update_lane[update_slot]>latest_rc_lane)) begin
                                coriginal=rc_update_remaining[update_slot];
                                caddress=rc_update_address[update_slot];
                                latest_rc_lane=rc_update_lane[update_slot];
                            end
                        end
                        if (!co) faults[6]=1;
                        if (h[50:48]!==saved_requester[2:0]) faults[7]=1;
                        if (!normal) faults[11]=1;
                        if (normal && co) begin
                            if (h[63:48]!==saved_requester || h[91:89]!==traffic_class[tag] ||
                                h[94:92]!=={1'b0,attributes[tag]}) faults[7]=1;
                            if (h[28:16]!==coriginal || h[11:0]!==caddress) faults[8]=1;
                            available=count*4-caddress[1:0];
                            delivered=(available<coriginal) ? available : coriginal;
                            if (count<1 || count>1024 || delivered<1 ||
                                count!=((caddress[1:0]+delivered+3)/4)) faults[10]=1;
                            if (terminal!==(delivered==coriginal)) faults[9]=1;
                            cdelivered=delivered;
                        end
                        observed_query_valid[lane/4]=1;
                        observed_query_owned[lane/4]=co;
                        observed_query_data[(lane/4)*30+:30]={attributes[tag],traffic_class[tag],caddress,coriginal};
                        observed_query_delivered[(lane/4)*13+:13]=cdelivered;
                        if (terminal && co) begin
                            rc_release_valid[lane/4]=1;
                            rc_release_tag[lane/4]=tag;
                        end
                    end
                    if (cp) begin
                        expected_enable=0;
                        if (cs>=3 && cn) begin
                            for (j=0;j<4;j=j+1) begin
                                payload_index=(cs-3)*4+j;
                                if (payload_index>=caddress[1:0] && payload_index<caddress[1:0]+cdelivered)
                                    expected_enable=expected_enable | (1<<j);
                            end
                        end
                        if (cs<3 && rc_user[lane*4+:4]!==4'b0) faults[10]=1;
                        if (cs>=3 && cn) begin
                            if (rc_user[lane*4+:4]!==expected_enable[3:0]) faults[10]=1;
                            for (j=0;j<4;j=j+1)
                                if (expected_enable[j] && (^rc_data[lane*32+j*8+:8])===1'bx) faults[4]=1;
                        end
                        cs=cs+1;
                        if (cs>1027) faults[10]=1;
                    end else if (rc_user[lane*4+:4]!==4'b0) faults[10]=1;
                    if (eop_map[lane]) begin
                        if (!cp || cs<3) faults[2]=1;
                        if (cn && cs!=cexp) faults[10]=1;
                        if (co) begin
                            if (ct) begin
                                rc_complete_events[rc_end_slot[lane]]=1;
                                rc_removed_span[rc_end_slot[lane]]=coriginal;
                            end else if (cn) begin
                                rc_update_valid[rc_end_slot[lane]]=1;
                                rc_update_tag[rc_end_slot[lane]]=ctag;
                                rc_update_remaining[rc_end_slot[lane]]=coriginal-cdelivered;
                                rc_update_address[rc_end_slot[lane]]=caddress+cdelivered;
                                rc_update_lane[rc_end_slot[lane]]=lane;
                                rc_removed_span[rc_end_slot[lane]]=cdelivered;
                            end
                        end
                        cp=0; cs=0; co=0;
                    end
                end
            end

            release_count=({2'b0,rc_release_valid[0]}+{2'b0,rc_release_valid[1]})+
                          ({2'b0,rc_release_valid[2]}+{2'b0,rc_release_valid[3]});
            ntags=$signed({2'b0,tag_count})-$signed({1'b0,release_count});
            rq_prefix_count[0]=ntags;
            if (rq_valid===1'b1 && rq_ready===4'b1111) begin
                starts=thermometer({2'b0,rq_user[21:20]}); ends=thermometer({2'b0,rq_user[27:26]});
                if (starts<0 || ends<0 || (^rq_user[72:0])===1'bx || rq_user[19:16]!=0) faults[2]=1;
                sop_map=0; eop_map=0; prev_ptr=-1;
                for (slot=0;slot<2;slot=slot+1) begin
                    if (slot<starts) begin
                        ptr=4*rq_user[22+slot*2+:2];
                        if ((ptr!=0 && ptr!=8) || ptr<=prev_ptr || (!rp && slot==0 && ptr!=0)) faults[2]=1;
                        sop_map[ptr]=1; rq_start_slot[ptr]=slot; prev_ptr=ptr;
                    end
                end
                prev_ptr=-1;
                for (slot=0;slot<2;slot=slot+1) begin
                    if (slot<ends) begin
                        ptr=rq_user[28+slot*4+:4];
                        if (ptr<=prev_ptr) faults[2]=1;
                        eop_map[ptr]=1; rq_end_slot[ptr]=slot; prev_ptr=ptr;
                    end
                end
                if ((!rp && starts==0) || (rq_user[36] && starts!=0)) faults[2]=1;
                if (rq_user[36]) begin faults[12]=1; ra=1; end
                for (lane=0;lane<16;lane=lane+1) begin
                    if ((lane%4)==0) begin
                        rq_prefix_valid[lane/4+1]=rq_prefix_valid[lane/4];
                        rq_prefix_count[lane/4+1]=rq_prefix_count[lane/4];
                        for (update_slot=0;update_slot<2;update_slot=update_slot+1)
                            rq_prefix_tag[lane/4+1][update_slot]=rq_prefix_tag[lane/4][update_slot];
                    end
                    if ((lane%4)==0 && sop_map[lane]) begin
                        if (rp) faults[2]=1;
                        h=rq_data >> (lane*32); count=h[74:64]; kind=h[78:75]; tag=h[103:96];
                        if ((^h)===1'bx) faults[4]=1;
                        if ((kind!=0 && kind!=1) || count<1 || count>1024 || h[1:0]!=0 || h[79]) faults[3]=1;
                        if (h[82:80]!==saved_requester[2:0] || (h[120] && h[95:80]!==saved_requester)) faults[7]=1;
                        rfirst=rq_user[4*rq_start_slot[lane]+:4];
                        rlast=rq_user[8+4*rq_start_slot[lane]+:4];
                        if ((count==1 && rlast!=0) || (count>1 && (rfirst==0 || rlast==0))) faults[3]=1;
                        if (h[11:0]+count*4>4096) faults[3]=1;
                        first_index=0; last_index=0;
                        for (j=3;j>=0;j=j-1) if (rfirst[j]) first_index=j;
                        for (j=0;j<4;j=j+1) if ((count==1 ? rfirst[j] : rlast[j])) last_index=j;
                        span=(count==1 && rfirst==0) ? 1 : (count-1)*4+last_index-first_index+1;
                        first_address=h[11:0]+first_index;
                        rp=1; rs=0; rw=(kind==1); ra=0; rcount=count; rexp=4+(rw ? count : 0);
                        if (kind==0) begin
                            rq_tag_busy=allocated_tags[tag];
                            for (update_slot=0;update_slot<4;update_slot=update_slot+1)
                                if (rc_release_valid[update_slot] && rc_release_tag[update_slot]==tag) rq_tag_busy=0;
                            for (update_slot=0;update_slot<2;update_slot=update_slot+1)
                                if (rq_prefix_valid[lane/4][update_slot] && rq_prefix_tag[lane/4][update_slot]==tag) rq_tag_busy=1;
                            if (rq_tag_busy || rq_prefix_count[lane/4]>=256) faults[5]=1;
                            else begin
                                rq_prefix_count[lane/4+1]=rq_prefix_count[lane/4]+1;
                                rq_added_span[rq_start_slot[lane]]=span;
                                rq_prefix_valid[lane/4+1][rq_start_slot[lane]]=1;
                                rq_prefix_tag[lane/4+1][rq_start_slot[lane]]=tag;
                                rq_update_remaining[rq_start_slot[lane]]=span;
                                rq_update_address[rq_start_slot[lane]]=first_address;
                                rq_update_tc[rq_start_slot[lane]]=h[123:121];
                                rq_update_attributes[rq_start_slot[lane]]=h[125:124]&saved_attribute_mask;
                            end
                        end
                    end
                    if (rp) begin
                        if (rw && rs>=4 && !ra) begin
                            payload_index=rs-4;
                            word_mask=(payload_index==0) ? rfirst : ((payload_index==rcount-1) ? rlast : 15);
                            for (j=0;j<4;j=j+1)
                                if (word_mask[j] && (^rq_data[lane*32+j*8+:8])===1'bx) faults[4]=1;
                        end
                        rs=rs+1;
                        if (rs>1028) faults[10]=1;
                    end
                    if (eop_map[lane]) begin
                        if (!rp || rs<4) faults[2]=1;
                        if (!ra && rs!=rexp) faults[10]=1;
                        rq_complete_events[rq_end_slot[lane]]=1;
                        if (rw) rq_posted_events[rq_end_slot[lane]]=1;
                        rp=0; rs=0;
                    end
                end
                ntags=rq_prefix_count[4];
                rq_update_valid=rq_prefix_valid[4];
                for (update_slot=0;update_slot<2;update_slot=update_slot+1)
                    rq_update_tag[update_slot]=rq_prefix_tag[4][update_slot];
            end
        end
        allocate_count={1'b0,rq_update_valid[0]}+{1'b0,rq_update_valid[1]};
        request_count={1'b0,rq_complete_events[0]}+{1'b0,rq_complete_events[1]};
        posted_count={1'b0,rq_posted_events[0]}+{1'b0,rq_posted_events[1]};
        complete_count=({2'b0,rc_complete_events[0]}+{2'b0,rc_complete_events[1]})+
                       ({2'b0,rc_complete_events[2]}+{2'b0,rc_complete_events[3]});
        rc_span_sum=({2'b0,rc_removed_span[0]}+{2'b0,rc_removed_span[1]})+
                    ({2'b0,rc_removed_span[2]}+{2'b0,rc_removed_span[3]});
        rq_span_sum=rq_added_span[0]+rq_added_span[1];
        nreads=$signed({2'b0,outstanding_reads})+$signed({1'b0,allocate_count})-$signed({1'b0,complete_count});
        nbytes=$signed({2'b0,remaining_bytes})+$signed(rq_span_sum)-$signed({1'b0,rc_span_sum});
        nrequests=requests+request_count; ncompleted=completed_reads+complete_count;
        nwrites=posted_writes+posted_count;
        // Range fault is evaluated from the full registered candidate below.
        for (i=0;i<256;i=i+1) begin
            // Bitmap commit follows the same RC-before-RQ order as metadata.
            next_tags[i]=allocated_tags[i];
            for (update_slot=0;update_slot<4;update_slot=update_slot+1)
                if (rc_release_valid[update_slot] && rc_release_tag[update_slot]==i) next_tags[i]=0;
            for (update_slot=0;update_slot<2;update_slot=update_slot+1)
                if (rq_update_valid[update_slot] && rq_update_tag[update_slot]==i) next_tags[i]=1;
            next_remaining[i]=remaining[i]; next_address[i]=address_next[i];
            next_tc[i]=traffic_class[i]; next_attributes[i]=attributes[i];
            latest_rc_lane=-1;
            for (update_slot=0;update_slot<4;update_slot=update_slot+1) begin
                if (rc_update_valid[update_slot] && rc_update_tag[update_slot]==i &&
                    (latest_rc_lane<0 || rc_update_lane[update_slot]>latest_rc_lane)) begin
                    next_remaining[i]=rc_update_remaining[update_slot];
                    next_address[i]=rc_update_address[update_slot];
                    latest_rc_lane=rc_update_lane[update_slot];
                end
            end
            for (update_slot=0;update_slot<2;update_slot=update_slot+1) begin
                if (rq_update_valid[update_slot] && rq_update_tag[update_slot]==i) begin
                    next_remaining[i]=rq_update_remaining[update_slot];
                    next_address[i]=rq_update_address[update_slot];
                    next_tc[i]=rq_update_tc[update_slot];
                    next_attributes[i]=rq_update_attributes[update_slot];
                end
            end
        end
    end

    integer reset_index;
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            observing<=0; allocated_tags<=0; tag_count<=0; outstanding_reads<=0;
            remaining_bytes<=0; rq_partial<=0; rc_partial<=0; fault_latched<=0;
            tag_range_high<=0; read_range_high<=0; byte_range_high<=0; range_checked<=0;
            requests<=0; completed_reads<=0; posted_writes<=0;
            rq_seen<=0; rc_seen<=0; rq_held<=0; rc_held<=0;
            rq_expected<=0; rq_first<=0; rq_last<=0; rq_count<=0; rq_write<=0; rq_aborted<=0;
            rc_expected<=0; rc_tag<=0; rc_terminal<=0; rc_normal<=0; rc_owned<=0;
            rc_original<=0; rc_delivered<=0; rc_address<=0;
            saved_requester<=0; saved_attribute_mask<=0;
            for (reset_index=0;reset_index<256;reset_index=reset_index+1) begin
                remaining[reset_index]<=0; address_next[reset_index]<=0;
                traffic_class[reset_index]<=0; attributes[reset_index]<=0;
            end
        end else if (!observing) begin
            if (profile_valid===1'b1 && user_reset===1'b0 && link_up===1'b1) begin
                observing<=1; saved_requester<=requester_id; saved_attribute_mask<=attribute_mask;
                // Arming is allowed only at a quiet public boundary.
                if (rq_valid!==1'b0 || rc_valid!==1'b0 || (^{requester_id,attribute_mask})===1'bx) fault_latched[0]<=1;
            end
        end else begin
            allocated_tags<=next_tags; tag_count<=ntags; outstanding_reads<=nreads; remaining_bytes<=nbytes;
            requests<=nrequests; completed_reads<=ncompleted; posted_writes<=nwrites; fault_latched<=faults;
            tag_range_high<=ntags[10:9]; read_range_high<=nreads[10:9]; byte_range_high<=nbytes[22:21];
            range_checked<=1;
            rq_partial<=rp; rq_seen<=rs; rq_expected<=rexp; rq_first<=rfirst; rq_last<=rlast;
            rq_count<=rcount; rq_write<=rw; rq_aborted<=ra;
            rc_partial<=cp; rc_seen<=cs; rc_expected<=cexp; rc_tag<=ctag;
            rc_terminal<=ct; rc_normal<=cn; rc_owned<=co;
            rc_original<=coriginal; rc_delivered<=cdelivered; rc_address<=caddress;
            for (reset_index=0;reset_index<256;reset_index=reset_index+1) begin
                remaining[reset_index]<=next_remaining[reset_index]; address_next[reset_index]<=next_address[reset_index];
                traffic_class[reset_index]<=next_tc[reset_index]; attributes[reset_index]<=next_attributes[reset_index];
            end
            rq_held<=rq_valid && rq_ready==0; rc_held<=rc_valid && !rc_ready;
            if (rq_valid && rq_ready==0) begin held_rq_data<=rq_data; held_rq_user<=rq_user; end
            if (rc_valid && !rc_ready) begin held_rc_data<=rc_data; held_rc_user<=rc_user; end
        end
    end
endmodule
