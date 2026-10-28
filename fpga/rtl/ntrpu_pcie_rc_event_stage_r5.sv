`timescale 1ns / 1ps
// R5 candidate: bounded DWORD/byte predicates before fragment selection; latency2 and RC4.
// header_owned is the upstream allocation bitmap's per-header prefix decision.
// expected_* is the registered metadata response for the captured input beat:
// a raw beat sampled at edge N consumes expected_* at edge N+1.
// No expected identity is learned from RC. Only cold reset discards state.
// Outputs are CANDIDATES. packet_checked is a local check, never a drain/ACK.
// Diagnostic sequence/origin wraps modulo 2^32 and is not an ownership key.
module ntrpu_pcie_rc_event_stage (
    input wire clk, cold_rst_n, observing,
    input wire rc_valid, rc_ready,
    input wire [511:0] rc_data,
    input wire [160:0] rc_user,
    input wire [3:0] header_owned,
    input wire [15:0] expected_requester,
    input wire [31:0] sequence_in,
    input wire [3:0] expected_valid,
    input wire [119:0] expected_metadata,
    output reg result_valid=0,
    output reg [31:0] sequence_out=0,
    output reg [3:0] result_headers=0,
    output reg [63:0] header_faults=0,
    output reg [15:0] beat_faults=0, fault_latched=0,
    output reg [3:0] update_valid=0, complete=0, unresolved=0, packet_checked=0,
    output reg [31:0] event_tags=0,
    output reg [51:0] event_remaining=0, event_removed=0,
    output reg [47:0] event_address=0,
    output reg [15:0] event_lanes=0,
    output reg [127:0] event_origins=0,
    output reg context_active=0, context_owned=0, unresolved_latched=0,
    output wire pipeline_pending
);
    reg [3:0] capture_headers=0;
    integer raw_starts, raw_ends, raw_slot, raw_position, raw_prior;
    reg [15:0] capture_end_map=0, capture_framing_faults=0;
    reg [31:0] capture_end_slots=0;
    reg capture_no_starts=0;
    function automatic integer thermometer(input [3:0] mask);
        case (mask)
            0:thermometer=0;1:thermometer=1;3:thermometer=2;
            7:thermometer=3;15:thermometer=4;default:thermometer=-1;
        endcase
    endfunction
    wire accepted=cold_rst_n && observing && rc_valid===1'b1 && rc_ready===1'b1;
    always @* begin
        capture_headers=0;capture_end_map=0;capture_end_slots=0;capture_framing_faults=0;
        raw_starts=thermometer(rc_user[67:64]);raw_ends=thermometer(rc_user[79:76]);
        capture_no_starts=(raw_starts==0);raw_position=0;raw_prior=-1;
        if (raw_starts<0 || raw_ends<0 || (^rc_user[96:0])===1'bx) capture_framing_faults[2]=1;
        for (raw_slot=0;raw_slot<4;raw_slot=raw_slot+1)
            if (raw_slot<raw_starts) begin
                raw_position=rc_user[68+raw_slot*2+:2];
                if (accepted) capture_headers[raw_position]=1;
                if (raw_position<=raw_prior) capture_framing_faults[2]=1;
                raw_prior=raw_position;
            end
        raw_prior=-1;
        for (raw_slot=0;raw_slot<4;raw_slot=raw_slot+1)
            if (raw_slot<raw_ends) begin
                raw_position=rc_user[80+raw_slot*4+:4];
                if (raw_position<=raw_prior) capture_framing_faults[2]=1;
                raw_prior=raw_position;capture_end_map[raw_position]=1;
                capture_end_slots[raw_position*2+:2]=raw_slot;
            end
        if (rc_user[96]) begin
            capture_framing_faults[12]=1;
            if (raw_starts!=0 || raw_ends==0) capture_framing_faults[2]=1;
        end
    end

    wire [3:0] headers, normal_proposal, length_error, terminal_error;
    wire [383:0] descriptors;
    wire [51:0] delivered, proposed_remaining;
    wire [47:0] proposed_address;
    ntrpu_pcie_rc_proposal arithmetic (
        .clk(clk),.cold_rst_n(cold_rst_n),.header_valid(capture_headers),.header_data(rc_data),
        .header_seen(headers),.proposal_valid(normal_proposal),.captured_headers(descriptors),
        .delivered_bytes(delivered),.proposed_remaining(proposed_remaining),
        .proposed_address(proposed_address),.length_error(length_error),.terminal_error(terminal_error)
    );
    reg stage_valid=0;
    reg [15:0] end_map_stage=0, framing_faults_stage=0;
    reg [31:0] end_slots_stage=0;
    reg no_starts_stage=0;
    reg [511:0] data_stage=0;
    reg [160:0] user_stage=0;
    reg [3:0] owned_stage=0;
    reg [15:0] requester_stage=0;
    reg [31:0] sequence_stage=0;
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            stage_valid<=0;data_stage<=0;user_stage<=0;owned_stage<=0;
            requester_stage<=0;sequence_stage<=0;
            end_map_stage<=0;end_slots_stage<=0;framing_faults_stage<=0;no_starts_stage<=0;
        end else begin
            stage_valid<=accepted;
            if (accepted) begin
                data_stage<=rc_data;user_stage<=rc_user;owned_stage<=header_owned;
                requester_stage<=expected_requester;sequence_stage<=sequence_in;
                end_map_stage<=capture_end_map;end_slots_stage<=capture_end_slots;
                framing_faults_stage<=capture_framing_faults;no_starts_stage<=capture_no_starts;
            end
        end
    end

    reg context_terminal=0, context_normal=0, context_ready=0;
    reg [7:0] context_tag=0;
    reg [12:0] context_original=0, context_delivered=0, context_remaining=0;
    reg [11:0] context_address=0, context_lower_address=0;
    reg [10:0] context_seen=0, context_expected=0;
    reg [15:0] context_faults=0;
    reg [31:0] context_origin=0;
    reg active_next,owned_next,terminal_next,normal_next,ready_next,lost_next;
    reg [7:0] tag_next;
    reg [12:0] original_next,delivered_next,remaining_next;
    reg [11:0] address_next,lower_address_next;
    reg [10:0] seen_next,expected_next;
    reg [15:0] context_faults_next,faults_next,header_error,framing_faults;
    reg [31:0] origin_next;
    reg [3:0] updates_next,complete_next,unresolved_next,checked_next;
    reg [31:0] tags_next;
    reg [51:0] remain_next,removed_next;
    reg [47:0] addresses_next;
    reg [15:0] lanes_next;
    reg [127:0] origins_next;
    reg [63:0] header_faults_next;
    reg [95:0] header;
    reg [29:0] metadata;
    // End is exclusive and needs fourteen bits: low(0..3)+delivered(0..8191).
    wire [13:0] context_payload_end={1'b0,context_delivered}+{12'b0,context_lower_address[1:0]};
    wire [55:0] header_payload_ends;
    genvar bound_header;
    generate for (bound_header=0;bound_header<4;bound_header=bound_header+1) begin : payload_bounds
        assign header_payload_ends[bound_header*14+:14]=
            {1'b0,delivered[bound_header*13+:13]}+{12'b0,descriptors[bound_header*96+:2]};
    end endgenerate
    function automatic [3:0] payload_byte_mask(
        input [10:0] seen,
        input [1:0] low_byte,
        input [13:0] exclusive_end
    );
        reg [10:0] payload_dword;
        reg below_end, at_end;
        integer b;
        begin
            payload_byte_mask=0;payload_dword=0;below_end=0;at_end=0;
            if (seen>=11'd3) begin
                payload_dword=seen-11'd3;
                // Keep the twelfth bound bit: an end DWORD of2048 must not wrap.
                below_end=({1'b0,payload_dword}<exclusive_end[13:2]);
                at_end=({1'b0,payload_dword}==exclusive_end[13:2]);
                for (b=0;b<4;b=b+1)
                    if ((payload_dword!=0 || b>=low_byte) &&
                        (below_end || (at_end && b<exclusive_end[1:0])))
                        payload_byte_mask[b]=1;
            end
        end
    endfunction
    // Each lane selects its fragment independently. No seen-counter recurrence
    // or payload predicate is carried through the preceding fifteen DWORDs.
    wire [15:0] lane_faults [0:15], lane_context_faults [0:15];
    wire [10:0] lane_after_seen [0:15];
    genvar g;
    generate for (g=0;g<16;g=g+1) begin : payload_lane
        reg fragment_active, fragment_normal;
        reg [10:0] seen_before, seen_after;
        reg [3:0] enable_expected, selected_enable;
        reg [15:0] errors, retained_errors;
        integer h, p, byte_index, latest_start, latest_end;
        always_comb begin
            fragment_normal=context_normal;
            if (user_stage[96]) fragment_normal=0;
            selected_enable=payload_byte_mask(context_seen+(context_active ? g : 0),
                context_lower_address[1:0],context_payload_end);
            latest_start=-1;latest_end=-1;
            for (h=0;h<4;h=h+1)
                if (4*h<=g && headers[h]) begin
                    latest_start=4*h;fragment_normal=normal_proposal[h];
                    selected_enable=payload_byte_mask(g-4*h,descriptors[h*96+:2],header_payload_ends[h*14+:14]);
                end
            for (p=0;p<g;p=p+1)
                if (end_map_stage[p]) latest_end=p;
            if (latest_start>=0) begin
                fragment_active=(latest_end<latest_start);
                seen_before=fragment_active ? g-latest_start : 0;
            end else begin
                fragment_active=context_active && latest_end<0;
                seen_before=latest_end>=0 ? 0 : context_seen+(context_active ? g : 0);
            end
            seen_after=seen_before;
            enable_expected=0;errors=0;retained_errors=0;
            if (fragment_active) begin
                if (seen_before>=3 && fragment_normal) enable_expected=selected_enable;
                if ((seen_before<3 && user_stage[g*4+:4]!==4'b0) ||
                    (seen_before>=3 && fragment_normal && user_stage[g*4+:4]!==enable_expected)) errors[10]=1;
                if (seen_before>=3 && fragment_normal)
                    for (byte_index=0;byte_index<4;byte_index=byte_index+1)
                        if (enable_expected[byte_index] && (^data_stage[g*32+byte_index*8+:8])===1'bx) errors[4]=1;
                seen_after=seen_before+1;
                if (seen_after>1027) errors[10]=1;
                retained_errors=errors;
            end else if (user_stage[g*4+:4]!==4'b0) errors[10]=1;
        end
        assign lane_faults[g]=errors;
        assign lane_context_faults[g]=retained_errors;
        assign lane_after_seen[g]=seen_after;
    end endgenerate
    integer slot,lane,q;
    always @* begin
        active_next=context_active;owned_next=context_owned;
        terminal_next=context_terminal;normal_next=context_normal;ready_next=context_ready;
        tag_next=context_tag;original_next=context_original;delivered_next=context_delivered;
        remaining_next=context_remaining;address_next=context_address;lower_address_next=context_lower_address;
        seen_next=context_seen;expected_next=context_expected;context_faults_next=context_faults;
        origin_next=context_origin;lost_next=unresolved_latched;
        faults_next=0;framing_faults=0;header_faults_next=0;
        updates_next=0;complete_next=0;unresolved_next=0;checked_next=0;
        tags_next=0;remain_next=0;removed_next=0;addresses_next=0;lanes_next=0;origins_next=0;
        header=0;metadata=0;header_error=0;q=0;
        if (stage_valid) begin
            framing_faults=framing_faults_stage;
            if (!active_next && no_starts_stage) framing_faults[2]=1;
            if (user_stage[96]) normal_next=0;
            faults_next=framing_faults;
            context_faults_next=context_faults_next|framing_faults;
            for (lane=0;lane<16;lane=lane+1) begin
                if (lane%4==0 && headers[lane/4]) begin
                    q=lane/4;header=descriptors[q*96+:96];metadata=expected_metadata[q*30+:30];
                    header_error=0;
                    if (active_next) begin faults_next[2]=1;header_error[2]=1;
                        if (owned_next) lost_next=1;
                    end
                    if ((^header[15:12])===1'bx) header_error[4]=1;
                    if (header[15:12]==9) begin
                        if ((^{header[71:64],header[55:48],header[30]})===1'bx) header_error[4]=1;
                    end else if ((^header)===1'bx) header_error[4]=1;
                    if (owned_stage[q]!==1'b1) header_error[6]=1;
                    if (header[50:48]!==requester_stage[2:0]) header_error[7]=1;
                    if (!normal_proposal[q]) header_error[11]=1;
                    if (normal_proposal[q] && owned_stage[q]) begin
                        if (expected_valid[q]!==1'b1) header_error[14]=1;
                        if (header[63:48]!==requester_stage || header[91:89]!==metadata[27:25] ||
                            header[94:92]!=={1'b0,metadata[29:28]}) header_error[7]=1;
                        if (header[28:16]!==metadata[12:0] || header[11:0]!==metadata[24:13]) header_error[8]=1;
                        if (length_error[q]!==1'b0) header_error[10]=1;
                        if (terminal_error[q]!==1'b0) header_error[9]=1;
                    end
                    header_faults_next[q*16+:16]=header_error;
                    faults_next=faults_next|header_error;
                    active_next=1;owned_next=(owned_stage[q]===1'b1);
                    tag_next=header[71:64];terminal_next=(header[30]===1'b1);
                    normal_next=normal_proposal[q];ready_next=normal_proposal[q];
                    original_next=header[28:16];delivered_next=delivered[q*13+:13];
                    remaining_next=proposed_remaining[q*13+:13];address_next=proposed_address[q*12+:12];
                    lower_address_next=header[11:0];seen_next=0;expected_next=3+header[42:32];
                    context_faults_next=header_error|framing_faults;origin_next=sequence_stage;
                end
                faults_next=faults_next|lane_faults[lane];
                context_faults_next=context_faults_next|lane_context_faults[lane];
                seen_next=lane_after_seen[lane];
                if (end_map_stage[lane]) begin
                    if (!active_next || seen_next<3) begin faults_next[2]=1;context_faults_next[2]=1;end
                    if (normal_next && seen_next!=expected_next) begin faults_next[10]=1;context_faults_next[10]=1;end
                    slot=end_slots_stage[lane*2+:2];
                    if (active_next && owned_next) begin
                        tags_next[slot*8+:8]=tag_next;
                        lanes_next[slot*4+:4]=lane;origins_next[slot*32+:32]=origin_next;
                        if (!ready_next) begin unresolved_next[slot]=1;lost_next=1;end
                        else if (terminal_next) begin
                            complete_next[slot]=1;removed_next[slot*13+:13]=original_next;
                        end else if (normal_next) begin
                            updates_next[slot]=1;remain_next[slot*13+:13]=remaining_next;
                            addresses_next[slot*12+:12]=address_next;removed_next[slot*13+:13]=delivered_next;
                        end
                        checked_next[slot]=ready_next && context_faults_next==0 && (terminal_next || normal_next);
                    end
                    active_next=0;owned_next=0;seen_next=0;
                end
            end
            // A malformed packet may lose its context; never erase that fact.
            if (faults_next[2]) lost_next=1;
        end
    end
    assign pipeline_pending=stage_valid || result_valid || context_active || unresolved_latched || (observing && rc_valid);
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            result_valid<=0;sequence_out<=0;result_headers<=0;header_faults<=0;beat_faults<=0;fault_latched<=0;
            update_valid<=0;complete<=0;unresolved<=0;packet_checked<=0;
            event_tags<=0;event_remaining<=0;event_removed<=0;event_address<=0;event_lanes<=0;event_origins<=0;
            context_active<=0;context_owned<=0;unresolved_latched<=0;
            context_terminal<=0;context_normal<=0;context_ready<=0;context_tag<=0;
            context_original<=0;context_delivered<=0;context_remaining<=0;
            context_address<=0;context_lower_address<=0;context_seen<=0;context_expected<=0;
            context_faults<=0;context_origin<=0;
        end else begin
            result_valid<=stage_valid;sequence_out<=stage_valid ? sequence_stage : 0;
            result_headers<=headers;header_faults<=header_faults_next;
            beat_faults<=faults_next;fault_latched<=fault_latched|faults_next;
            update_valid<=updates_next;complete<=complete_next;unresolved<=unresolved_next;packet_checked<=checked_next;
            event_tags<=tags_next;event_remaining<=remain_next;event_removed<=removed_next;
            event_address<=addresses_next;event_lanes<=lanes_next;event_origins<=origins_next;
            context_active<=active_next;context_owned<=owned_next;unresolved_latched<=lost_next;
            context_terminal<=terminal_next;context_normal<=normal_next;context_ready<=ready_next;context_tag<=tag_next;
            context_original<=original_next;context_delivered<=delivered_next;context_remaining<=remaining_next;
            context_address<=address_next;context_lower_address<=lower_address_next;context_seen<=seen_next;context_expected<=expected_next;
            context_faults<=context_faults_next;context_origin<=origin_next;
        end
    end
endmodule
