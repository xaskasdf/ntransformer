`timescale 1ns / 1ps
// Decode RC write selection without remaining-byte/address arithmetic.
// Inputs are raw beats and the producer's own registered canonical context.
module ntrpu_pcie_rc_update_select (
    input wire observing, rc_valid, rc_ready,
    input wire [511:0] rc_data,
    input wire [160:0] rc_user,
    input wire [255:0] allocated_tags,
    input wire [7:0] retained_tag,
    input wire retained_owned, retained_terminal, retained_normal,
    output reg [3:0] header_previous_valid,
    output reg [7:0] header_previous_slot,
    output reg [255:0] commit_valid,
    output reg [511:0] commit_slot,
    output reg [3:0] header_seed_retained,
    output reg [15:0] header_prefix_mask,
    output reg [3:0] header_owned, header_release_valid,
    output reg [31:0] header_release_tags
);
    reg [15:0] sop, eop;
    reg [1:0] end_slot [0:15];
    reg [39:0] ownership;
    reg [3:0] event_present, event_write;
    reg [3:0] event_lane [0:3];
    reg signed [3:0] event_origin [0:3];
    reg [7:0] event_tag [0:3];
    reg [3:0] event_owned, event_terminal, event_normal, event_first;
    reg [7:0] header_tag [0:3];
    // Combinational table: priority depends only on decoded event lanes.
    reg [31:0] commit_priority;
    reg [3:0] commit_matches;
    integer starts, ends, pointer, newest;
    integer slot, lane, source_header, scan, header_index, tag_index, prefix_origin;

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

    // Earlier terminal candidates use only registered ownership, never another
    // header's release result. Procedural if retains original X/Z behavior.
    function automatic [39:0] rc_ownership(
        input observe, valid, ready,
        input [511:0] data,
        input [160:0] user_bits,
        input [255:0] allocated
    );
        reg [15:0] present;
        reg [3:0] base_owned, owned_flags, release_flags;
        reg [31:0] release_tags;
        reg [7:0] tags [0:3];
        reg [3:0] terminals;
        integer starts, pointer, at, prior;
        begin
            present=0; base_owned=0; owned_flags=0; release_flags=0;
            release_tags=0; terminals=0; starts=0; pointer=0;
            for (at=0;at<4;at=at+1) tags[at]=0;
            if (observe && valid===1'b1 && ready===1'b1) begin
                starts=thermometer(user_bits[67:64]);
                for (at=0;at<4;at=at+1) begin
                    if (at<starts) begin
                        pointer=4*user_bits[68+at*2+:2];
                        present[pointer]=1;
                    end
                    tags[at]=data[at*128+64+:8];
                    terminals[at]=data[at*128+30];
                    base_owned[at]=allocated[tags[at]];
                end
                for (at=0;at<4;at=at+1) begin
                    if (present[at*4]) begin
                        owned_flags[at]=base_owned[at];
                        for (prior=0;prior<4;prior=prior+1) begin
                            if (prior<at && present[prior*4] && terminals[prior] &&
                                base_owned[prior] && tags[prior]==tags[at]) owned_flags[at]=0;
                        end
                        if (terminals[at] && owned_flags[at]) begin
                            release_flags[at]=1; release_tags[at*8+:8]=tags[at];
                        end
                    end
                end
            end
            rc_ownership={release_tags,release_flags,owned_flags};
        end
    endfunction


    function automatic [31:0] commit_priority_table(input [15:0] positions);
        reg [15:0] later;
        reg [1:0] selected;
        reg found;
        integer mask, at, prior;
        begin
            later=0; selected=0; found=0; commit_priority_table=0;
            for(at=0;at<4;at=at+1)
                for(prior=0;prior<4;prior=prior+1)
                    if(positions[at*4+:4]>positions[prior*4+:4]) later[at*4+prior]=1;
            for(mask=0;mask<16;mask=mask+1) begin
                selected=0; found=0;
                for(at=0;at<4;at=at+1) begin
                    if(mask[at] && (!found || later[at*4+selected])) begin
                        selected=at; found=1;
                    end
                end
                commit_priority_table[mask*2+:2]=selected;
            end
        end
    endfunction

    always_comb begin
        ownership=rc_ownership(observing,rc_valid,rc_ready,rc_data,rc_user,allocated_tags);
        header_owned=ownership[3:0]; header_release_valid=ownership[7:4];
        header_release_tags=ownership[39:8];
        header_previous_valid=0; header_previous_slot=0;
        header_seed_retained=0; header_prefix_mask=0;
        commit_valid=0; commit_slot=0; commit_priority=0; commit_matches=0;
        sop=0; eop=0; event_present=0; event_write=0;
        event_owned=0; event_terminal=0; event_normal=0; event_first=0;
        starts=0; ends=0; pointer=0; newest=-1;
        for(lane=0;lane<16;lane=lane+1) end_slot[lane]=0;
        for(slot=0;slot<4;slot=slot+1) begin
            event_lane[slot]=0; event_origin[slot]=-1; event_tag[slot]=0;
            header_tag[slot]=rc_data[slot*128+64+:8];
        end
        if(observing && rc_valid===1'b1 && rc_ready===1'b1) begin
            starts=thermometer(rc_user[67:64]); ends=thermometer(rc_user[79:76]);
            for(slot=0;slot<4;slot=slot+1) begin
                if(slot<starts) begin
                    pointer=4*rc_user[68+slot*2+:2]; sop[pointer]=1;
                end
                if(slot<ends) begin
                    pointer=rc_user[80+slot*4+:4]; eop[pointer]=1; end_slot[pointer]=slot;
                end
            end
            // Duplicate EOP pointers keep the original last-declared slot.
            for(lane=0;lane<16;lane=lane+1) begin
                if(eop[lane]) begin
                    event_present[end_slot[lane]]=1;
                    event_lane[end_slot[lane]]=lane;
                end
            end
            for(slot=0;slot<4;slot=slot+1) begin
                event_tag[slot]=retained_tag;
                event_owned[slot]=retained_owned;
                event_terminal[slot]=retained_terminal;
                event_normal[slot]=retained_normal;
                if(rc_user[96]) event_normal[slot]=0;
                // Each context is selected solely from the decoded SOP map.
                for(source_header=0;source_header<4;source_header=source_header+1) begin
                    if(sop[source_header*4] && source_header*4<=event_lane[slot]) begin
                        event_origin[slot]=source_header;
                        event_tag[slot]=header_tag[source_header];
                        event_owned[slot]=header_owned[source_header];
                        event_terminal[slot]=rc_data[source_header*128+30];
                        event_normal[slot]=(rc_data[source_header*128+12+:4]==0 &&
                            rc_data[source_header*128+43+:3]==0 && rc_data[source_header*128+46]==0);
                    end
                end
                event_first[slot]=1;
                for(lane=0;lane<16;lane=lane+1) begin
                    if(eop[lane] && lane<event_lane[slot] &&
                        (event_origin[slot]<0 || lane>=event_origin[slot]*4)) event_first[slot]=0;
                end
                if(event_present[slot] && event_first[slot] && event_owned[slot]) begin
                    // Terminal X/Z follows the original procedural else branch.
                    if(event_terminal[slot]) begin end
                    else if(event_normal[slot]) event_write[slot]=1;
                end
            end
            for(header_index=0;header_index<4;header_index=header_index+1) begin
                newest=-1;
                if(sop[header_index*4]) begin
                    for(scan=0;scan<4;scan=scan+1) begin
                        if(event_write[scan] && event_lane[scan]<header_index*4 &&
                            event_tag[scan]==header_tag[header_index]) begin
                            if(newest<0 || event_lane[scan]>newest) begin
                                header_previous_valid[header_index]=1;
                                header_previous_slot[header_index*2+:2]=scan;
                                newest=event_lane[scan];
                            end
                            if(event_origin[scan]<0) header_seed_retained[header_index]=1;
                        end
                    end
                end
            end
            // Independent constant-index bits. A procedural if contributes only
            // on a known-true match; unknown tags retain the original behavior.
            for(header_index=0;header_index<4;header_index=header_index+1) begin
                for(prefix_origin=0;prefix_origin<4;prefix_origin=prefix_origin+1) begin
                    for(scan=0;scan<4;scan=scan+1) begin
                        if(sop[header_index*4] && event_write[scan] &&
                            event_lane[scan]<header_index*4 &&
                            event_tag[scan]==header_tag[header_index] &&
                            event_origin[scan]==prefix_origin && header_index!=3)
                            header_prefix_mask[header_index*4+prefix_origin]=1;
                    end
                end
            end
            commit_priority=commit_priority_table({event_lane[3],event_lane[2],event_lane[1],event_lane[0]});
            for(tag_index=0;tag_index<256;tag_index=tag_index+1) begin
                commit_matches=0;
                for(scan=0;scan<4;scan=scan+1) begin
                    // Only a known-true procedural condition makes a candidate.
                    if(event_write[scan] && event_tag[scan]==tag_index) commit_matches[scan]=1;
                end
                commit_valid[tag_index]=|commit_matches;
                commit_slot[tag_index*2+:2]=commit_priority[commit_matches*2+:2];
            end
        end
    end
endmodule
