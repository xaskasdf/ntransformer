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
    reg [3:0] updates;
    reg [39:0] ownership;
    reg [7:0] update_tag [0:3];
    reg [3:0] update_lane [0:3];
    reg [7:0] tag, active_tag;
    reg active_seed_retained;
    reg [3:0] active_mask;
    reg [3:0] update_seed_retained;
    reg [3:0] update_mask [0:3];
    reg owned, terminal, normal;
    reg [127:0] header;
    reg signed [3:0] starts, ends;
    reg signed [4:0] ptr, newest;
    integer slot, lane, scan, index;

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

    always_comb begin
        ownership=rc_ownership(observing,rc_valid,rc_ready,rc_data,rc_user,allocated_tags);
        header_owned=ownership[3:0]; header_release_valid=ownership[7:4];
        header_release_tags=ownership[39:8];
        header_previous_valid=0; header_previous_slot=0;
        header_seed_retained=0; header_prefix_mask=0;
        active_seed_retained=1; active_mask=0; update_seed_retained=0;
        commit_valid=0; commit_slot=0;
        sop=0; eop=0; updates=0;
        starts=0; ends=0; ptr=0; newest=-1; tag=0; header=0;
        active_tag=retained_tag; owned=retained_owned;
        terminal=retained_terminal; normal=retained_normal;
        for (slot=0;slot<16;slot=slot+1) end_slot[slot]=0;
        for (slot=0;slot<4;slot=slot+1) begin
            update_tag[slot]=0; update_lane[slot]=0; update_mask[slot]=0;
        end
        if (observing && rc_valid===1'b1 && rc_ready===1'b1) begin
            starts=thermometer(rc_user[67:64]); ends=thermometer(rc_user[79:76]);
            for (slot=0;slot<4;slot=slot+1) begin
                if (slot<starts) begin
                    ptr=4*rc_user[68+slot*2+:2];
                    sop[ptr]=1;
                end
                if (slot<ends) begin
                    ptr=rc_user[80+slot*4+:4];
                    eop[ptr]=1; end_slot[ptr]=slot;
                end
            end
            if (rc_user[96]) normal=0;
            for (lane=0;lane<16;lane=lane+1) begin
                if ((lane%4)==0 && sop[lane]) begin
                    header=rc_data >> (lane*32); tag=header[71:64];
                    active_tag=tag; terminal=header[30];
                    normal=(header[15:12]==0 && header[45:43]==0 && header[46]==0);
                    owned=header_owned[lane/4];
                    active_seed_retained=0; active_mask=0;
                    newest=-1;
                    for (scan=0;scan<4;scan=scan+1) begin
                        if (updates[scan] && update_tag[scan]==tag &&
                            (newest<0 || update_lane[scan]>newest)) begin
                            header_previous_valid[lane/4]=1;
                            header_previous_slot[(lane/4)*2+:2]=scan;
                            active_seed_retained=update_seed_retained[scan];
                            active_mask=update_mask[scan];
                            newest=update_lane[scan];
                        end
                    end
                    header_seed_retained[lane/4]=active_seed_retained;
                    header_prefix_mask[(lane/4)*4+:4]=active_mask;
                    active_mask[lane/4]=1;
                end
                if (eop[lane]) begin
                    if (owned) begin
                        if (terminal) begin end
                        else if (normal) begin
                            updates[end_slot[lane]]=1;
                            update_tag[end_slot[lane]]=active_tag;
                            update_lane[end_slot[lane]]=lane;
                            update_seed_retained[end_slot[lane]]=active_seed_retained;
                            update_mask[end_slot[lane]]=active_mask;
                        end
                    end
                    owned=0;
                end
            end
        end
        for (index=0;index<256;index=index+1) begin
            newest=-1;
            for (scan=0;scan<4;scan=scan+1) begin
                if (updates[scan] && update_tag[scan]==index &&
                    (newest<0 || update_lane[scan]>newest)) begin
                    commit_valid[index]=1; commit_slot[index*2+:2]=scan;
                    newest=update_lane[scan];
                end
            end
        end
    end
endmodule
