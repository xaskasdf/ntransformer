`timescale 1ns / 1ps
// Missing checks over one captured beat and the producer's own canonical context.
// No tag table, state transition feedback, learned requester ID, or reference input.
module ntrpu_pcie_symbolic_late_fault_leaves_r1 (
    input ntrpu_pcie_symbolic_numeric_pkg_r1::b_record_t in_b,
    output wire [15:0] faults
);
    wire ntrpu_pcie_symbolic_frontend_pkg::s1_record_t record;
    assign record=in_b.origin.record;
    wire observing=record.valid && record.pre_observing;
    wire rq_partial=record.pre_rq_partial, rc_partial=record.pre_rc_partial;
    wire [15:0] expected_requester=record.expected_requester;
    wire rq_valid=record.rq_valid, rc_valid=record.rc_valid, rc_ready=record.rc_ready;
    wire [3:0] rq_ready=record.rq_ready;
    wire [511:0] rq_data=record.rq_data, rc_data=record.rc_data;
    wire [136:0] rq_user=record.rq_user;
    wire [160:0] rc_user=record.rc_user;
    wire [42:0] rq_context=record.pre_rq_context;
    wire [70:0] rc_context={record.pre_rc_context.seen,record.pre_rc_context.expected,
        record.pre_rc_context.tag,record.pre_rc_context.terminal,record.pre_rc_context.normal,
        record.pre_rc_context.owned,in_b.origin.context_value[24:12],
        in_b.context_delivered,in_b.origin.context_value[11:0]};
    wire [3:0] metadata_owned,guard_length,guard_terminal;
    wire [119:0] metadata_data;
    wire [51:0] canonical_delivered;
    wire [59:0] guard_delivered;
    genvar at;
    generate for(at=0;at<4;at=at+1) begin:header_guard
        assign metadata_owned[at]=record.queries[at].owned;
        assign metadata_data[at*30+:30]={record.queries[at].attributes_tc,
            in_b.origin.queries[at][11:0],in_b.origin.queries[at][24:12]};
        assign canonical_delivered[at*13+:13]=in_b.query_delivered[at];
        ntrpu_pcie_symbolic_header_guard_r1 guard_i(
            .coriginal(in_b.origin.queries[at][24:12]),.caddress(in_b.origin.queries[at][11:0]),
            .count(rc_data[at*128+32+:11]),
            .normal(rc_data[at*128+12+:4]==0 && rc_data[at*128+43+:3]==0 && rc_data[at*128+46]==0),
            .co(record.queries[at].owned),.terminal(rc_data[at*128+30]),
            .delivered(guard_delivered[at*15+:15]),.length_fault(guard_length[at]),.terminal_fault(guard_terminal[at]));
    end endgenerate
    reg rp, cp, rw, ra, cn, co, normal, terminal;
    reg [10:0] rs, rexp, rcount, cs, cexp, count;
    reg [3:0] rfirst, rlast, expected_enable, word_mask, kind;
    reg [12:0] coriginal, cdelivered;
    reg [11:0] caddress;
    reg signed [14:0] delivered, payload_index;
    reg signed [3:0] starts, ends;
    reg signed [4:0] ptr;
    reg [15:0] sop_map, eop_map;
    reg [1:0] rq_start_slot [0:15];
    reg [127:0] h;
    reg [29:0] metadata;
    integer lane, slot, j;

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

    // Each leaf starts at zero and is only set to one by original predicates.
    // Context assignments and their order remain identical to late_validation_r1.
    reg [15:0] rc_faults [0:15], rq_faults [0:15];
    wire [15:0] fault_tree [1:63];
    integer fault_lane;
    genvar fault_at;
    generate
        for (fault_at=0;fault_at<16;fault_at=fault_at+1) begin:fault_leaves
            assign fault_tree[32+fault_at]=(fault_at==15) ? 16'b0 : rc_faults[fault_at];
            assign fault_tree[48+fault_at]=rq_faults[fault_at];
        end
        for (fault_at=1;fault_at<32;fault_at=fault_at+1) begin:fault_reduce
            assign fault_tree[fault_at]=fault_tree[2*fault_at] | fault_tree[2*fault_at+1];
        end
    endgenerate
    assign faults=fault_tree[1];

    always @* begin
        for (fault_lane=0;fault_lane<16;fault_lane=fault_lane+1) begin
            rc_faults[fault_lane]=0;
            rq_faults[fault_lane]=0;
        end
        rp=rq_partial; cp=rc_partial;
        {rs,rexp,rfirst,rlast,rcount,rw,ra}=rq_context;
        cs=rc_context[70:60]; cexp=rc_context[59:49];
        cn=rc_context[39]; co=rc_context[38];
        coriginal=rc_context[37:25]; cdelivered=rc_context[24:12];
        caddress=rc_context[11:0];
        normal=0; terminal=0; count=0; kind=0;
        expected_enable=0; word_mask=0; delivered=0; payload_index=0;
        starts=0; ends=0; ptr=0; sop_map=0; eop_map=0; h=0; metadata=0;
        for (lane=0;lane<16;lane=lane+1) rq_start_slot[lane]=0;
        if (observing) begin
            if (rc_valid===1'b1 && rc_ready===1'b1) begin
                starts=thermometer(rc_user[67:64]); ends=thermometer(rc_user[79:76]);
                for (slot=0;slot<4;slot=slot+1) begin
                    if (slot<starts) begin ptr=4*rc_user[68+slot*2+:2]; sop_map[ptr]=1; end
                    if (slot<ends) begin ptr=rc_user[80+slot*4+:4]; eop_map[ptr]=1; end
                end
                if (rc_user[96]) cn=0;
                for (lane=0;lane<16;lane=lane+1) begin
                    if ((lane%4)==0 && sop_map[lane]) begin
                        h=rc_data >> (lane*32); count=h[42:32]; terminal=h[30];
                        normal=(h[15:12]==0 && h[45:43]==0 && h[46]==0);
                        cp=1; cs=0; cexp=3+count; cn=normal;
                        metadata=metadata_data[(lane/4)*30+:30];
                        co=metadata_owned[lane/4];
                        coriginal=metadata[12:0]; caddress=metadata[24:13];
                        cdelivered=canonical_delivered[(lane/4)*13+:13];
                        delivered=guard_delivered[(lane/4)*15+:15];
                        if (normal && co) begin
                            if (h[63:48]!==expected_requester || h[91:89]!==metadata[27:25] ||
                                h[94:92]!=={1'b0,metadata[29:28]}) rc_faults[lane][7]=1;
                            if (h[28:16]!==coriginal || h[11:0]!==caddress) rc_faults[lane][8]=1;
                            if (guard_length[lane/4]) rc_faults[lane][10]=1;
                            if (guard_terminal[lane/4]) rc_faults[lane][9]=1;
                        end
                    end
                    if (cp) begin
                        expected_enable=0;
                        if (cs>=3 && cn)
                            for (j=0;j<4;j=j+1) begin
                                payload_index=(cs-3)*4+j;
                                if (payload_index>=caddress[1:0] && payload_index<caddress[1:0]+cdelivered)
                                    expected_enable=expected_enable | (1<<j);
                            end
                        if (cs<3 && rc_user[lane*4+:4]!==4'b0) rc_faults[lane][10]=1;
                        if (cs>=3 && cn) begin
                            if (rc_user[lane*4+:4]!==expected_enable) rc_faults[lane][10]=1;
                            for (j=0;j<4;j=j+1)
                                if (expected_enable[j] && (^rc_data[lane*32+j*8+:8])===1'bx) rc_faults[lane][4]=1;
                        end
                        cs=cs+1;
                        if (cs>1027) rc_faults[lane][10]=1;
                    end else if (rc_user[lane*4+:4]!==4'b0) rc_faults[lane][10]=1;
                    if (eop_map[lane]) begin
                        if (cn && cs!=cexp) rc_faults[lane][10]=1;
                        cp=0; cs=0; co=0;
                    end
                end
            end
            if (rq_valid===1'b1 && rq_ready===4'b1111) begin
                starts=thermometer({2'b0,rq_user[21:20]}); ends=thermometer({2'b0,rq_user[27:26]});
                sop_map=0; eop_map=0;
                for (slot=0;slot<2;slot=slot+1) begin
                    if (slot<starts) begin
                        ptr=4*rq_user[22+slot*2+:2]; sop_map[ptr]=1; rq_start_slot[ptr]=slot;
                    end
                    if (slot<ends) begin ptr=rq_user[28+slot*4+:4]; eop_map[ptr]=1; end
                end
                if (rq_user[36]) ra=1;
                for (lane=0;lane<16;lane=lane+1) begin
                    if ((lane%4)==0 && sop_map[lane]) begin
                        h=rq_data >> (lane*32); count=h[74:64]; kind=h[78:75];
                        rfirst=rq_user[4*rq_start_slot[lane]+:4];
                        rlast=rq_user[8+4*rq_start_slot[lane]+:4];
                        rp=1; rs=0; rw=(kind==1); ra=0; rcount=count; rexp=4+(rw ? count : 0);
                    end
                    if (rp) begin
                        if (rw && rs>=4 && !ra) begin
                            payload_index=rs-4;
                            word_mask=(payload_index==0) ? rfirst : ((payload_index==rcount-1) ? rlast : 15);
                            for (j=0;j<4;j=j+1)
                                if (word_mask[j] && (^rq_data[lane*32+j*8+:8])===1'bx) rq_faults[lane][4]=1;
                        end
                        rs=rs+1;
                        if (rs>1028) rq_faults[lane][10]=1;
                    end
                    if (eop_map[lane]) begin
                        if (!ra && rs!=rexp) rq_faults[lane][10]=1;
                        rp=0; rs=0;
                    end
                end
            end
        end
    end
endmodule
