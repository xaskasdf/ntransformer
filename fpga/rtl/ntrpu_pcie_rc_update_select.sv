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
    output reg [511:0] commit_slot
);
    reg [15:0] sop, eop;
    reg [1:0] end_slot [0:15];
    reg [3:0] releases, updates;
    reg [7:0] release_tag [0:3], update_tag [0:3];
    reg [3:0] update_lane [0:3];
    reg [7:0] tag, active_tag;
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

    always_comb begin
        header_previous_valid=0; header_previous_slot=0;
        commit_valid=0; commit_slot=0;
        sop=0; eop=0; releases=0; updates=0;
        starts=0; ends=0; ptr=0; newest=-1; tag=0; header=0;
        active_tag=retained_tag; owned=retained_owned;
        terminal=retained_terminal; normal=retained_normal;
        for (slot=0;slot<16;slot=slot+1) end_slot[slot]=0;
        for (slot=0;slot<4;slot=slot+1) begin
            release_tag[slot]=0; update_tag[slot]=0; update_lane[slot]=0;
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
                    owned=allocated_tags[tag];
                    for (scan=0;scan<4;scan=scan+1)
                        if (releases[scan] && release_tag[scan]==tag) owned=0;
                    newest=-1;
                    for (scan=0;scan<4;scan=scan+1) begin
                        if (updates[scan] && update_tag[scan]==tag &&
                            (newest<0 || update_lane[scan]>newest)) begin
                            header_previous_valid[lane/4]=1;
                            header_previous_slot[(lane/4)*2+:2]=scan;
                            newest=update_lane[scan];
                        end
                    end
                    if (terminal && owned) begin
                        releases[lane/4]=1; release_tag[lane/4]=tag;
                    end
                end
                if (eop[lane]) begin
                    if (owned) begin
                        if (terminal) begin end
                        else if (normal) begin
                            updates[end_slot[lane]]=1;
                            update_tag[end_slot[lane]]=active_tag;
                            update_lane[end_slot[lane]]=lane;
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
