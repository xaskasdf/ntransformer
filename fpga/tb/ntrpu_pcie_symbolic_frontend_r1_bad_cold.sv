// S1 R1: registered symbolic queries/events and live control. Numeric application/bytes live downstream.
// Behavioral symbolic-value model R1. Not a pipelined or qualified implementation.
// R15: R12 raw-selected prefix queries use the reviewed metadata-effect algebra.
// R12: R8 producer with lane-local RC byte contributions and balanced reduction.
// R8: R7 producer unchanged; pair with selector R3 direct EOP-event ancestry.
`timescale 1ns / 1ps
// Causal state producer: raw RQ/RC only, one commit edge, no external oracle.
// Registered canonical metadata accompanies the same captured input beat.
// Early faults are incomplete: payload/metadata validation is a later consumer.
// No empty/ACK certificate. Only cold reset clears canonical obligations.
module ntrpu_pcie_symbolic_frontend (
    input wire clk, cold_rst_n,
    input wire [31:0] sequence_in,
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
    output reg rq_partial=0, rc_partial=0,
    output reg [31:0] requests=0, completed_reads=0, posted_writes=0,
    output wire [15:0] early_fault_sources,
    output wire early_protocol_fault, control_pending,
    output ntrpu_pcie_symbolic_frontend_pkg::s1_record_t out_record
);
    // faults: profile/link, handshake/stall, framing, descriptor, unknown,
    // duplicate/capacity, orphan, identity, byte-count/address, terminal,
    // payload enables/length, completion error, discontinue, counter range.
    import ntrpu_pcie_symbolic_frontend_pkg::*;
    reg [50:0] symbolic_rows [0:255], next_symbolic_rows [0:255];
    query_t [3:0] pending_queries;
    eop_t [3:0] pending_eops;
    initial out_record='0;
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
    reg [50:0] rc_description=0;
    reg [10:0] rc_description_count=0;
    reg rc_delivery_enabled=0;
    reg [50:0] cdescription;
    reg [10:0] cdescription_count;
    reg cdelivery_enabled;
    reg rq_held=0, rc_held=0;
    reg [511:0] held_rq_data=0, held_rc_data=0;
    reg [136:0] held_rq_user=0;
    reg [160:0] held_rc_user=0;

    // Low counter bits retain the exact truncation/wrap behavior of R6.
    // Register only the six discarded high bits of the signed candidates.
    reg [1:0] tag_range_high=0, read_range_high=0;
    reg range_checked=0;
    reg [15:0] fault_latched=0;
    wire signed [10:0] checked_tags=$signed({tag_range_high,tag_count});
    wire signed [10:0] checked_reads=$signed({read_range_high,outstanding_reads});
    // Case equality preserves procedural-if behavior when an operand contains X.
    wire range_fault_now=range_checked && ((checked_tags<0 || checked_tags>256 ||
        checked_reads<0 || checked_reads>257)===1'b1);
    assign early_fault_sources=fault_latched | {2'b0,range_fault_now,13'b0};

    reg [255:0] next_tags;
    reg [15:0] faults;
    reg rp, cp, rw, ra, ct, cn, co;
    reg [10:0] rs, rexp, rcount, cs, cexp;
    reg [3:0] rfirst, rlast;
    reg [7:0] ctag;
    // Signed guard widths cover the full stored range and every possible beat delta.
    // Counts change once per cycle, not through a serial 32-bit adder per DWORD.
    reg signed [10:0] ntags, nreads;
    reg [31:0] nrequests, ncompleted, nwrites;
    integer i, lane, slot, j;
    // Thermometers produce -1..4; pointers and newest EOP lane are -1..15.
    // Both pointer operands stay signed so the initial -1 sorts before lane0.
    reg signed [3:0] starts, ends;
    reg signed [4:0] ptr, prev_ptr;
    reg [1:0] first_index, last_index;
    // Even malformed eleven-bit counts fit: span -6..8188, available -3..8188.
    // Keep signed guards and the original mixed-sign comparison semantics.
    reg signed [14:0] span;
    reg [12:0] first_address;
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
    reg [18:0] rq_admission;
    reg [1:0] rq_complete_events, rq_posted_events;
    reg signed [14:0] rq_added_span [0:1];
    reg [2:0] complete_count;
    reg [1:0] allocate_count, request_count, posted_count;
    reg [3:0] rc_update_valid;
    reg [1:0] rq_update_valid;
    reg [7:0] rc_update_tag [0:3], rq_update_tag [0:1];
    reg [50:0] rc_update_description [0:3];
    reg [12:0] rq_update_remaining [0:1];
    reg [11:0] rq_update_address [0:1];
    reg [3:0] rc_update_lane [0:3];
    reg [2:0] rq_update_tc [0:1];
    reg [1:0] rq_update_attributes [0:1];
    integer update_slot;
    reg [127:0] h;
    reg [10:0] count;
    reg [7:0] tag;
    reg [3:0] kind, error_code;
    reg terminal, normal;

    wire [3:0] selected_header_owned, selected_release_valid;
    wire [31:0] selected_release_tags;
    wire [3:0] selected_header_valid;
    wire [7:0] selected_header_slot;
    wire [255:0] selected_commit_valid;
    wire [511:0] selected_commit_slot;
    wire [3:0] selected_seed_retained;
    wire [15:0] selected_prefix_mask;
    wire [43:0] prefix_counts={rc_data[416+:11],rc_data[288+:11],rc_data[160+:11],rc_data[32+:11]};
    wire [50:0] retained_after_description=symbolic_consume(
        rc_description,rc_delivery_enabled,rc_description_count);
    reg [50:0] prefix_seed_description;
    ntrpu_pcie_rc_update_select rc_selector (
        .observing(observing),.rc_valid(rc_valid),.rc_ready(rc_ready),
        .rc_data(rc_data),.rc_user(rc_user),.allocated_tags(allocated_tags),
        .retained_tag(rc_tag),.retained_owned(rc_owned),
        .retained_terminal(rc_terminal),.retained_normal(rc_normal),
        .header_previous_valid(selected_header_valid),.header_previous_slot(selected_header_slot),
        .commit_valid(selected_commit_valid),.commit_slot(selected_commit_slot),
        .header_seed_retained(selected_seed_retained),.header_prefix_mask(selected_prefix_mask),
        .header_owned(selected_header_owned),.header_release_valid(selected_release_valid),
        .header_release_tags(selected_release_tags)
    );

    assign early_protocol_fault=|early_fault_sources;
    // This is only the live control obligation. Byte and downstream pending
    // state must be combined by the complete pipeline; no empty certificate here.
    assign control_pending=(tag_count!=0 || outstanding_reads!=0 ||
        rq_partial || rc_partial || rq_held || rc_held || rq_valid || rc_valid);

    // Descriptor: seed address/remaining[50:26], canonical consume summary[25:0].
    // TC/attributes deliberately remain in their separate current-row arrays.
    function automatic [50:0] symbolic_seed(input [12:0] r,input [11:0] a);
        symbolic_seed={a,r,26'b0};
    endfunction

    function automatic [50:0] symbolic_consume(input [50:0] description,
        input enabled,input [10:0] words);
        ntrpu_pcie_metadata_effect_pkg::effect_t result;
        begin
            result=ntrpu_pcie_metadata_effect_pkg::compose_effect(
                {31'b0,description[25:0]},
                ntrpu_pcie_metadata_effect_pkg::encode_effect(enabled,1'b0,30'b0,words));
            symbolic_consume={description[50:26],result[25:0]};
        end
    endfunction

    function automatic [50:0] symbolic_prefix(input [50:0] description,
        input [3:0] mask,input [43:0] counts);
        ntrpu_pcie_metadata_effect_pkg::effect_t e0,e1,e2,e3,summary,result;
        begin
            e0=ntrpu_pcie_metadata_effect_pkg::encode_effect(mask[0],1'b0,30'b0,counts[0+:11]);
            e1=ntrpu_pcie_metadata_effect_pkg::encode_effect(mask[1],1'b0,30'b0,counts[11+:11]);
            e2=ntrpu_pcie_metadata_effect_pkg::encode_effect(mask[2],1'b0,30'b0,counts[22+:11]);
            e3=ntrpu_pcie_metadata_effect_pkg::encode_effect(mask[3],1'b0,30'b0,counts[33+:11]);
            summary=ntrpu_pcie_metadata_effect_pkg::compose_effect(
                ntrpu_pcie_metadata_effect_pkg::compose_effect(e0,e1),
                ntrpu_pcie_metadata_effect_pkg::compose_effect(e2,e3));
            result=ntrpu_pcie_metadata_effect_pkg::compose_effect({31'b0,description[25:0]},summary);
            symbolic_prefix={description[50:26],result[25:0]};
        end
    endfunction

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

    // At most two declared SOPs survive pointer decoding. Sort those events
    // before admission; malformed positions and duplicate-slot priority remain.
    // Bits18:8=count,7:4=blocked by lane,3:0=admitted by lane.
    function automatic [18:0] rq_admission_flags(
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
            starts=thermometer({2'b0,user_bits[21:20]});
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
            rq_admission_flags={current_count,blocked,admitted};
        end
    endfunction

    always_comb begin
        next_tags=allocated_tags; faults=early_fault_sources;
        prefix_seed_description=0;
        pending_queries='0; pending_eops='0;
        ntags=tag_count; nreads=outstanding_reads;
        nrequests=requests; ncompleted=completed_reads; nwrites=posted_writes;
        rp=rq_partial; rs=rq_seen; rexp=rq_expected; rfirst=rq_first; rlast=rq_last;
        rcount=rq_count; rw=rq_write; ra=rq_aborted;
        cp=rc_partial; cs=rc_seen; cexp=rc_expected; ctag=rc_tag;
        ct=rc_terminal; cn=rc_normal; co=rc_owned;
        cdescription=rc_description; cdescription_count=rc_description_count;
        cdelivery_enabled=rc_delivery_enabled;
        rc_update_valid=0; rq_update_valid=0;
        rc_complete_events=0; rq_complete_events=0; rq_posted_events=0;
        rc_release_valid=selected_release_valid; release_count=0; rq_admission=0;
        complete_count=0;
        allocate_count=0; request_count=0; posted_count=0;
        for (i=0;i<4;i=i+1) begin
            rc_release_tag[i]=selected_release_tags[i*8+:8];
            rc_update_tag[i]=0; rc_update_description[i]=0; rc_update_lane[i]=0;
        end
        for (i=0;i<2;i=i+1) begin
            rq_update_tag[i]=0; rq_update_remaining[i]=0;
            rq_update_address[i]=0; rq_update_tc[i]=0; rq_update_attributes[i]=0; rq_added_span[i]=0;
        end
        sop_map=0; eop_map=0; starts=0; ends=0; ptr=0; prev_ptr=-1;
        h=0; count=0; tag=0; kind=0; error_code=0; terminal=0; normal=0;
        first_index=0; last_index=0; span=0; first_address=0;
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
                        co=selected_header_owned[lane/4];
                        prefix_seed_description=symbolic_rows[tag];
                        if (selected_seed_retained[lane/4])
                            prefix_seed_description=retained_after_description;
                        cdescription=symbolic_prefix(prefix_seed_description,
                            selected_prefix_mask[(lane/4)*4+:4],prefix_counts);
                        cdescription_count=count;
                        cdelivery_enabled=((normal && co)===1'b1);
                        if (!co) faults[6]=1;
                        if (h[50:48]!==saved_requester[2:0]) faults[7]=1;
                        if (!normal) faults[11]=1;
                        pending_queries[lane/4].valid=1;
                        pending_queries[lane/4].owned=co;
                        pending_queries[lane/4].attributes_tc={attributes[tag],traffic_class[tag]};
                        pending_queries[lane/4].description=cdescription;
                        pending_queries[lane/4].words=cdescription_count;
                        pending_queries[lane/4].delivery_enabled=cdelivery_enabled;
                    end
                    if (cp) cs=cs+1;
                    if (eop_map[lane]) begin
                        if (!cp || cs<3) faults[2]=1;
                        if (co) begin
                            if (ct) begin
                                rc_complete_events[rc_end_slot[lane]]=1;
                                pending_eops[rc_end_slot[lane]].valid=1;
                                pending_eops[rc_end_slot[lane]].terminal=1'b1;
                                pending_eops[rc_end_slot[lane]].delivery_enabled=cdelivery_enabled;
                                pending_eops[rc_end_slot[lane]].description=cdescription;
                                pending_eops[rc_end_slot[lane]].words=cdescription_count;
                                pending_eops[rc_end_slot[lane]].lane=lane;
                            end else if (cn) begin
                                rc_update_valid[rc_end_slot[lane]]=1;
                                rc_update_tag[rc_end_slot[lane]]=ctag;
                                rc_update_description[rc_end_slot[lane]]=symbolic_consume(
                                    cdescription,cdelivery_enabled,cdescription_count);
                                rc_update_lane[rc_end_slot[lane]]=lane;
                                pending_eops[rc_end_slot[lane]].valid=1;
                                pending_eops[rc_end_slot[lane]].terminal=1'b0;
                                pending_eops[rc_end_slot[lane]].delivery_enabled=cdelivery_enabled;
                                pending_eops[rc_end_slot[lane]].description=cdescription;
                                pending_eops[rc_end_slot[lane]].words=cdescription_count;
                                pending_eops[rc_end_slot[lane]].lane=lane;
                            end
                        end
                        cp=0; cs=0; co=0;
                    end
                end
            end

            release_count=({2'b0,rc_release_valid[0]}+{2'b0,rc_release_valid[1]})+
                          ({2'b0,rc_release_valid[2]}+{2'b0,rc_release_valid[3]});
            ntags=$signed({2'b0,tag_count})-$signed({1'b0,release_count});
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
                rq_admission=rq_admission_flags(rq_data,rq_user,allocated_tags,rc_release_valid,
                    {rc_release_tag[3],rc_release_tag[2],rc_release_tag[1],rc_release_tag[0]},ntags);
                for (lane=0;lane<16;lane=lane+1) begin
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
                            if (rq_admission[4+lane/4]) faults[5]=1;
                            if (rq_admission[lane/4]) begin
                                rq_added_span[rq_start_slot[lane]]=span;
                                rq_update_valid[rq_start_slot[lane]]=1;
                                rq_update_tag[rq_start_slot[lane]]=tag;
                                rq_update_remaining[rq_start_slot[lane]]=span;
                                rq_update_address[rq_start_slot[lane]]=first_address;
                                rq_update_tc[rq_start_slot[lane]]=h[123:121];
                                rq_update_attributes[rq_start_slot[lane]]=h[125:124]&saved_attribute_mask;
                            end
                        end
                    end
                    if (rp) rs=rs+1;
                    if (eop_map[lane]) begin
                        if (!rp || rs<4) faults[2]=1;
                        rq_complete_events[rq_end_slot[lane]]=1;
                        if (rw) rq_posted_events[rq_end_slot[lane]]=1;
                        rp=0; rs=0;
                    end
                end
                ntags=$signed(rq_admission[18:8]);
            end
        end
        allocate_count={1'b0,rq_update_valid[0]}+{1'b0,rq_update_valid[1]};
        request_count={1'b0,rq_complete_events[0]}+{1'b0,rq_complete_events[1]};
        posted_count={1'b0,rq_posted_events[0]}+{1'b0,rq_posted_events[1]};
        complete_count=({2'b0,rc_complete_events[0]}+{2'b0,rc_complete_events[1]})+
                       ({2'b0,rc_complete_events[2]}+{2'b0,rc_complete_events[3]});
        nreads=$signed({2'b0,outstanding_reads})+$signed({1'b0,allocate_count})-$signed({1'b0,complete_count});
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
            next_symbolic_rows[i]=symbolic_rows[i];
            next_tc[i]=traffic_class[i]; next_attributes[i]=attributes[i];
            if (selected_commit_valid[i]) begin
                next_symbolic_rows[i]=rc_update_description[selected_commit_slot[i*2+:2]];
            end
            for (update_slot=0;update_slot<2;update_slot=update_slot+1) begin
                if (rq_update_valid[update_slot] && rq_update_tag[update_slot]==i) begin
                    next_symbolic_rows[i]=symbolic_seed(
                        rq_update_remaining[update_slot],rq_update_address[update_slot]);
                    next_tc[i]=rq_update_tc[update_slot];
                    next_attributes[i]=rq_update_attributes[update_slot];
                end
            end
        end
    end

    // Capture a complete value record on every non-cold edge, including idle
    // diagnostics. Only effect_pending marks downstream traffic obligations.
    always @(posedge clk) begin
        if (!cold_rst_n) out_record<=out_record;
        else begin
            out_record.valid<=1;
            out_record.sequence_number<=sequence_in;
            out_record.effect_pending<=observing && ((rq_valid!==1'b0) || (rc_valid!==1'b0));
            out_record.pre_observing<=observing;
            out_record.expected_requester<=saved_requester;
            out_record.pre_rq_partial<=rq_partial;
            out_record.pre_rc_partial<=rc_partial;
            out_record.pre_rc_context<={rc_seen,rc_expected,rc_tag,rc_terminal,rc_normal,
                rc_owned,rc_description,rc_description_count,rc_delivery_enabled};
            out_record.pre_rq_context<={rq_seen,rq_expected,rq_first,rq_last,rq_count,rq_write,rq_aborted};
            out_record.rq_valid<=rq_valid; out_record.rq_ready<=rq_ready;
            out_record.rc_valid<=rc_valid; out_record.rc_ready<=rc_ready;
            out_record.rq_data<=rq_data; out_record.rq_user<=rq_user;
            out_record.rc_data<=rc_data; out_record.rc_user<=rc_user;
            out_record.queries<=pending_queries; out_record.eops<=pending_eops;
            out_record.rq_spans<={rq_added_span[1],rq_added_span[0]};
        end
    end
    integer reset_index;
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            observing<=0; allocated_tags<=0; tag_count<=0; outstanding_reads<=0;
            rq_partial<=0; rc_partial<=0; fault_latched<=0;
            tag_range_high<=0; read_range_high<=0; range_checked<=0;
            requests<=0; completed_reads<=0; posted_writes<=0;
            rq_seen<=0; rc_seen<=0; rq_held<=0; rc_held<=0;
            rq_expected<=0; rq_first<=0; rq_last<=0; rq_count<=0; rq_write<=0; rq_aborted<=0;
            rc_expected<=0; rc_tag<=0; rc_terminal<=0; rc_normal<=0; rc_owned<=0;
            rc_description<=0; rc_description_count<=0; rc_delivery_enabled<=0;
            saved_requester<=0; saved_attribute_mask<=0;
            for (reset_index=0;reset_index<256;reset_index=reset_index+1) begin
                symbolic_rows[reset_index]<=0;
                traffic_class[reset_index]<=0; attributes[reset_index]<=0;
            end
        end else if (!observing) begin
            if (profile_valid===1'b1 && user_reset===1'b0 && link_up===1'b1) begin
                observing<=1; saved_requester<=requester_id; saved_attribute_mask<=attribute_mask;
                // Arming is allowed only at a quiet public boundary.
                if (rq_valid!==1'b0 || rc_valid!==1'b0 || (^{requester_id,attribute_mask})===1'bx) fault_latched[0]<=1;
            end
        end else begin
            allocated_tags<=next_tags; tag_count<=ntags; outstanding_reads<=nreads;
            requests<=nrequests; completed_reads<=ncompleted; posted_writes<=nwrites; fault_latched<=faults;
            tag_range_high<=ntags[10:9]; read_range_high<=nreads[10:9];
            range_checked<=1;
            rq_partial<=rp; rq_seen<=rs; rq_expected<=rexp; rq_first<=rfirst; rq_last<=rlast;
            rq_count<=rcount; rq_write<=rw; rq_aborted<=ra;
            rc_partial<=cp; rc_seen<=cs; rc_expected<=cexp; rc_tag<=ctag;
            rc_terminal<=ct; rc_normal<=cn; rc_owned<=co;
            rc_description<=cdescription; rc_description_count<=cdescription_count;
            rc_delivery_enabled<=cdelivery_enabled;
            for (reset_index=0;reset_index<256;reset_index=reset_index+1) begin
                symbolic_rows[reset_index]<=next_symbolic_rows[reset_index];
                traffic_class[reset_index]<=next_tc[reset_index]; attributes[reset_index]<=next_attributes[reset_index];
            end
            rq_held<=rq_valid && rq_ready==0; rc_held<=rc_valid && !rc_ready;
            if (rq_valid && rq_ready==0) begin held_rq_data<=rq_data; held_rq_user<=rq_user; end
            if (rc_valid && !rc_ready) begin held_rc_data<=rc_data; held_rc_user<=rc_user; end
        end
    end
endmodule
