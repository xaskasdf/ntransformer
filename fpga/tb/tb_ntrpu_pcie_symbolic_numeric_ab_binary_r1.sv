`timescale 1ns / 1ps
// S1 boundary proof: original numeric R7, observational R7, reviewed value model.
// No source row is filtered after fault. Check all 256 rows on every clock.
module tb_ntrpu_pcie_symbolic_numeric_ab_binary_r1;
    import ntrpu_pcie_symbolic_frontend_pkg::*;
    reg clk=0; always #2 clk=~clk;
    reg cold_rst_n=0, profile_valid=1, user_reset=0, link_up=1;
    reg [15:0] requester_id=16'h01a0;
    reg [1:0] attribute_mask=3;
    reg rq_valid=0, rc_valid=0, rc_ready=1;
    reg [3:0] rq_ready=15;
    reg [511:0] rq_data=0, rc_data=0;
    reg [136:0] rq_user=0;
    reg [160:0] rc_user=0;
    integer index=0;
    wire [31:0] sequence_in=32'hffffff80+index;
    ntrpu_pcie_request_inventory_numeric_ab_mapped_reference_r1 reference_i(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user));
    ntrpu_pcie_request_inventory_symbolic_probe_numeric_ab_mapped_reference_r1 probe(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user));
    ntrpu_pcie_inventory_scheduler_numeric_ab_mapped_reference_r1 model(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user));
    ntrpu_pcie_symbolic_frontend_numeric_ab_mapped_reference_r1 s1(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user),.sequence_in(sequence_in));

    // Outputs deliberately left unconnected: hierarchical observations cannot
    // feed state into any of the four independently instantiated input consumers.
    wire [372:0] control={s1.observing,s1.allocated_tags,s1.tag_count,s1.outstanding_reads,
        s1.rq_partial,s1.rc_partial,s1.requests,s1.completed_reads,s1.posted_writes};
    wire [372:0] reference_control={reference_i.observing,reference_i.allocated_tags,
        reference_i.tag_count,reference_i.outstanding_reads,reference_i.rq_partial,
        reference_i.rc_partial,reference_i.requests,reference_i.completed_reads,reference_i.posted_writes};
    wire [409:0] reference_state={reference_control,reference_i.remaining_bytes,reference_i.fault_sources};
    wire [409:0] probe_state={probe.observing,probe.allocated_tags,probe.tag_count,probe.outstanding_reads,
        probe.rq_partial,probe.rc_partial,probe.requests,probe.completed_reads,probe.posted_writes,
        probe.remaining_bytes,probe.fault_sources};
    wire [70:0] reference_rc={reference_i.rc_seen,reference_i.rc_expected,reference_i.rc_tag,
        reference_i.rc_terminal,reference_i.rc_normal,reference_i.rc_owned,
        reference_i.rc_original,reference_i.rc_delivered,reference_i.rc_address};
    wire [42:0] reference_rq={reference_i.rq_seen,reference_i.rq_expected,reference_i.rq_first,
        reference_i.rq_last,reference_i.rq_count,reference_i.rq_write,reference_i.rq_aborted};

    s1_record_t expected_record, checked_record, previous_record='0;
    reg [179:0] saved_query, actual_query;
    reg [207:0] saved_lanes, actual_lanes;
    reg [70:0] saved_rc, actual_rc;
    reg [3:0] saved_eop_valid, saved_eop_terminal;
    reg signed [22:0] saved_byte_candidate, actual_byte_candidate;
    reg [20:0] saved_bytes;
    reg range_sticky=0;
    reg [24:0] value;
    reg [12:0] delivered;
    reg [16:0] rc_sum;
    reg [15:0] lanes_used;
    reg expected_pending;
    integer slot, lane, tag_q;
    integer tables=0, queries=0, eops=0, cold_rows=0, fault_rows=0, wraps=0;
    integer fd, fields, scenario, native_ps, last_scenario=-1, cases=0, expected_rows;
    integer expected_observing, expected_fault, check_state, etags, ereads, ebytes;
    integer erp, ecp, erequests, ecompleted, ewrites;
    reg [255:0] expected_bitmap;
    string vector_file, bank, trailing_token;

    ntrpu_pcie_symbolic_frontend_r3_numeric_ab_mapped_reference_r1 ced(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user),.sequence_in(sequence_in));
    wire [2497:0] s1_ports={s1.observing,s1.allocated_tags,s1.tag_count,s1.outstanding_reads,
        s1.rq_partial,s1.rc_partial,s1.requests,s1.completed_reads,s1.posted_writes,
        s1.early_fault_sources,s1.early_protocol_fault,s1.control_pending,s1.out_record};
    wire [2497:0] ced_ports={ced.observing,ced.allocated_tags,ced.tag_count,ced.outstanding_reads,
        ced.rq_partial,ced.rc_partial,ced.requests,ced.completed_reads,ced.posted_writes,
        ced.early_fault_sources,ced.early_protocol_fault,ced.control_pending,ced.out_record};
    reg [2497:0] history_ports [0:1];
    reg [55:0] history_table [0:1][0:255];
    reg [50:0] history_context [0:1];
    integer history_origin [0:1];
    reg [1:0] history_valid=0;
    integer ced_retired=0, ced_captured=0, ced_cancelled=0, ced_tables=0, ced_busy_cold=0;
    integer ced_tag, hist, flush_index;
    s1_record_t history_record0, history_record1;
    reg expected_pipeline_pending;
    s1_record_t captured_expected;
    integer capture_slot;
    task check_ced;
        begin
            if (!cold_rst_n) begin
                if (ced_ports!==s1_ports || ced.c_record.record.valid!==0 || ced.e_record.origin.record.valid!==0)
                    $fatal(1,"CED_COLD_MISMATCH row=%0d",index);
                if (history_valid!=0) ced_busy_cold=ced_busy_cold+1;
                for (hist=0;hist<2;hist=hist+1)
                    if (history_valid[hist] && history_origin[hist]<expected_rows) ced_cancelled=ced_cancelled+1;
                history_valid=0;
                for (ced_tag=0;ced_tag<256;ced_tag=ced_tag+1)
                    if ({ced.d_stage.attributes[ced_tag],ced.d_stage.traffic_class[ced_tag],
                        ced.d_stage.symbolic_rows[ced_tag]}!==56'b0)
                        $fatal(1,"CED_COLD_TABLE_MISMATCH row=%0d tag=%0d",index,ced_tag);
                if (ced.d_stage.retained_description!==0) $fatal(1,"CED_COLD_CONTEXT_MISMATCH row=%0d",index);
                expected_pipeline_pending=rq_valid || rc_valid;
            end else begin
                if (ced.c_record.control!==s1_ports[2497:2107])
                    $fatal(1,"CED_C_CONTROL_MISMATCH row=%0d",index);
                captured_expected=s1.out_record;
                captured_expected.pre_rc_context.description=0;
                for (capture_slot=0;capture_slot<4;capture_slot=capture_slot+1) begin
                    captured_expected.queries[capture_slot].description=0;
                    captured_expected.queries[capture_slot].attributes_tc=0;
                    captured_expected.eops[capture_slot].description=0;
                end
                if (ced.c_record.record!==captured_expected)
                    $fatal(1,"CED_C_CAPTURE_MISMATCH row=%0d",index);
                if (history_valid[1]) begin
                    if (ced_ports!==history_ports[1]) begin
                        $display("CED_ACTUAL %h",ced_ports);
                        $display("CED_EXPECTED %h",history_ports[1]);
                        $fatal(1,"CED_PORT_MISMATCH row=%0d origin=%0d",index,history_origin[1]);
                    end
                    for (ced_tag=0;ced_tag<256;ced_tag=ced_tag+1) begin
                        if ({ced.d_stage.attributes[ced_tag],ced.d_stage.traffic_class[ced_tag],
                            ced.d_stage.symbolic_rows[ced_tag]}!==history_table[1][ced_tag])
                            $fatal(1,"CED_TABLE_MISMATCH row=%0d origin=%0d tag=%0d actual=%h expected=%h",
                                index,history_origin[1],ced_tag,ced.d_stage.symbolic_rows[ced_tag],history_table[1][ced_tag]);
                    end
                    if (ced.d_stage.retained_description!==history_context[1])
                        $fatal(1,"CED_RETAINED_CONTEXT_MISMATCH row=%0d origin=%0d",index,history_origin[1]);
                    if (history_origin[1]<expected_rows) begin
                        ced_retired=ced_retired+1; ced_tables=ced_tables+256;
                    end
                end else if (ced_ports!=='0) $fatal(1,"CED_FILL_MISMATCH row=%0d",index);
                history_record0=history_ports[0][2106:0];
                history_record1=history_ports[1][2106:0];
                expected_pipeline_pending=rq_valid || rc_valid || s1.out_record.effect_pending ||
                    (history_valid[0] && history_record0.effect_pending) ||
                    (history_valid[1] && history_record1.effect_pending);
                history_ports[1]=history_ports[0]; history_context[1]=history_context[0];
                history_origin[1]=history_origin[0]; history_valid[1]=history_valid[0];
                for (ced_tag=0;ced_tag<256;ced_tag=ced_tag+1) begin
                    history_table[1][ced_tag]=history_table[0][ced_tag];
                    history_table[0][ced_tag]={s1.attributes[ced_tag],s1.traffic_class[ced_tag],s1.symbolic_rows[ced_tag]};
                end
                history_ports[0]=s1_ports; history_context[0]=s1.rc_description;
                history_origin[0]=index; history_valid[0]=1;
                if (index<expected_rows) ced_captured=ced_captured+1;
            end
            if (ced.pipeline_pending!==expected_pipeline_pending)
                $fatal(1,"CED_PENDING_MISMATCH row=%0d",index);
            if (index%2048==0) $display("CED_PROGRESS bank=%s row=%0d retired=%0d cancelled=%0d tables=%0d",
                bank,index,ced_retired,ced_cancelled,ced_tables);
        end
    endtask

    // Expected records come from the independent S1/R7/probe/model consumers.
    // Candidate A/B outputs are only observed, never used to build expectations.
    import ntrpu_pcie_symbolic_numeric_pkg_r1::*;
    wire a_record_t numeric_a;
    wire b_record_t numeric_b;
    wire b_record_t reference_numeric_b;
    integer mapped_mode=0, mapped_rows=0;
    ntrpu_pcie_symbolic_ced_pkg::control_t delayed_control='0, mapped_control;
    s1_record_t mapped_record;
    integer mapped_slot;
    initial begin
        if ($test$plusargs("OMIT_CONTROL")) mapped_mode=1;
        if ($test$plusargs("OMIT_RECORD")) mapped_mode=2;
        if ($test$plusargs("OMIT_COLD")) mapped_mode=3;
        if ($test$plusargs("WRONG_ORIGIN")) mapped_mode=4;
        if ($test$plusargs("IGNORE_ENABLE")) mapped_mode=5;
        $display("NUMERIC_MAPPED_MODE mode=%0d",mapped_mode);
    end
    always @(posedge clk) begin
        if (!cold_rst_n) delayed_control<='0;
        else delayed_control<=ced.d_control;
    end
    always @* begin
        mapped_control=ced.d_control;
        mapped_record=ced.out_record;
        if (mapped_mode==1) mapped_control='0;
        if (mapped_mode==2) mapped_record='0;
        if (mapped_mode==4) mapped_control=delayed_control;
        if (mapped_mode==5) begin
            for (mapped_slot=0;mapped_slot<4;mapped_slot=mapped_slot+1) begin
                mapped_record.queries[mapped_slot].delivery_enabled=1'b1;
                mapped_record.eops[mapped_slot].delivery_enabled=1'b1;
            end
            mapped_record.pre_rc_context.delivery_enabled=1'b1;
        end
    end
    ntrpu_pcie_symbolic_numeric_pair_r1 mapped(
        .clk(clk),
        .cold_rst_n(mapped_mode==3 && index>6 ? 1'b1 : cold_rst_n),
        .\in_control[observing] (mapped_control.observing),
        .\in_control[allocated_tags] (mapped_control.allocated_tags),
        .\in_control[tag_count] (mapped_control.tag_count),
        .\in_control[outstanding_reads] (mapped_control.outstanding_reads),
        .\in_control[rq_partial] (mapped_control.rq_partial),
        .\in_control[rc_partial] (mapped_control.rc_partial),
        .\in_control[requests] (mapped_control.requests),
        .\in_control[completed_reads] (mapped_control.completed_reads),
        .\in_control[posted_writes] (mapped_control.posted_writes),
        .\in_control[early_fault_sources] (mapped_control.early_fault_sources),
        .\in_control[early_protocol_fault] (mapped_control.early_protocol_fault),
        .\in_control[control_pending] (mapped_control.control_pending),
        .\in_record[valid] (mapped_record.valid),
        .\in_record[sequence_number] (mapped_record.sequence_number),
        .\in_record[effect_pending] (mapped_record.effect_pending),
        .\in_record[pre_observing] (mapped_record.pre_observing),
        .\in_record[expected_requester] (mapped_record.expected_requester),
        .\in_record[pre_rq_partial] (mapped_record.pre_rq_partial),
        .\in_record[pre_rc_partial] (mapped_record.pre_rc_partial),
        .\in_record[pre_rc_context][seen] (mapped_record.pre_rc_context.seen),
        .\in_record[pre_rc_context][expected] (mapped_record.pre_rc_context.expected),
        .\in_record[pre_rc_context][tag] (mapped_record.pre_rc_context.tag),
        .\in_record[pre_rc_context][terminal] (mapped_record.pre_rc_context.terminal),
        .\in_record[pre_rc_context][normal] (mapped_record.pre_rc_context.normal),
        .\in_record[pre_rc_context][owned] (mapped_record.pre_rc_context.owned),
        .\in_record[pre_rc_context][description] (mapped_record.pre_rc_context.description),
        .\in_record[pre_rc_context][words] (mapped_record.pre_rc_context.words),
        .\in_record[pre_rc_context][delivery_enabled] (mapped_record.pre_rc_context.delivery_enabled),
        .\in_record[pre_rq_context] (mapped_record.pre_rq_context),
        .\in_record[rq_valid] (mapped_record.rq_valid),
        .\in_record[rc_valid] (mapped_record.rc_valid),
        .\in_record[rc_ready] (mapped_record.rc_ready),
        .\in_record[rq_ready] (mapped_record.rq_ready),
        .\in_record[rq_data] (mapped_record.rq_data),
        .\in_record[rc_data] (mapped_record.rc_data),
        .\in_record[rq_user] (mapped_record.rq_user),
        .\in_record[rc_user] (mapped_record.rc_user),
        .\in_record[queries][3][valid] (mapped_record.queries[3].valid),
        .\in_record[queries][3][owned] (mapped_record.queries[3].owned),
        .\in_record[queries][3][attributes_tc] (mapped_record.queries[3].attributes_tc),
        .\in_record[queries][3][description] (mapped_record.queries[3].description),
        .\in_record[queries][3][words] (mapped_record.queries[3].words),
        .\in_record[queries][3][delivery_enabled] (mapped_record.queries[3].delivery_enabled),
        .\in_record[queries][2][valid] (mapped_record.queries[2].valid),
        .\in_record[queries][2][owned] (mapped_record.queries[2].owned),
        .\in_record[queries][2][attributes_tc] (mapped_record.queries[2].attributes_tc),
        .\in_record[queries][2][description] (mapped_record.queries[2].description),
        .\in_record[queries][2][words] (mapped_record.queries[2].words),
        .\in_record[queries][2][delivery_enabled] (mapped_record.queries[2].delivery_enabled),
        .\in_record[queries][1][valid] (mapped_record.queries[1].valid),
        .\in_record[queries][1][owned] (mapped_record.queries[1].owned),
        .\in_record[queries][1][attributes_tc] (mapped_record.queries[1].attributes_tc),
        .\in_record[queries][1][description] (mapped_record.queries[1].description),
        .\in_record[queries][1][words] (mapped_record.queries[1].words),
        .\in_record[queries][1][delivery_enabled] (mapped_record.queries[1].delivery_enabled),
        .\in_record[queries][0][valid] (mapped_record.queries[0].valid),
        .\in_record[queries][0][owned] (mapped_record.queries[0].owned),
        .\in_record[queries][0][attributes_tc] (mapped_record.queries[0].attributes_tc),
        .\in_record[queries][0][description] (mapped_record.queries[0].description),
        .\in_record[queries][0][words] (mapped_record.queries[0].words),
        .\in_record[queries][0][delivery_enabled] (mapped_record.queries[0].delivery_enabled),
        .\in_record[eops][3][valid] (mapped_record.eops[3].valid),
        .\in_record[eops][3][terminal] (mapped_record.eops[3].terminal),
        .\in_record[eops][3][delivery_enabled] (mapped_record.eops[3].delivery_enabled),
        .\in_record[eops][3][description] (mapped_record.eops[3].description),
        .\in_record[eops][3][words] (mapped_record.eops[3].words),
        .\in_record[eops][3][lane] (mapped_record.eops[3].lane),
        .\in_record[eops][2][valid] (mapped_record.eops[2].valid),
        .\in_record[eops][2][terminal] (mapped_record.eops[2].terminal),
        .\in_record[eops][2][delivery_enabled] (mapped_record.eops[2].delivery_enabled),
        .\in_record[eops][2][description] (mapped_record.eops[2].description),
        .\in_record[eops][2][words] (mapped_record.eops[2].words),
        .\in_record[eops][2][lane] (mapped_record.eops[2].lane),
        .\in_record[eops][1][valid] (mapped_record.eops[1].valid),
        .\in_record[eops][1][terminal] (mapped_record.eops[1].terminal),
        .\in_record[eops][1][delivery_enabled] (mapped_record.eops[1].delivery_enabled),
        .\in_record[eops][1][description] (mapped_record.eops[1].description),
        .\in_record[eops][1][words] (mapped_record.eops[1].words),
        .\in_record[eops][1][lane] (mapped_record.eops[1].lane),
        .\in_record[eops][0][valid] (mapped_record.eops[0].valid),
        .\in_record[eops][0][terminal] (mapped_record.eops[0].terminal),
        .\in_record[eops][0][delivery_enabled] (mapped_record.eops[0].delivery_enabled),
        .\in_record[eops][0][description] (mapped_record.eops[0].description),
        .\in_record[eops][0][words] (mapped_record.eops[0].words),
        .\in_record[eops][0][lane] (mapped_record.eops[0].lane),
        .\in_record[rq_spans][1] (mapped_record.rq_spans[1]),
        .\in_record[rq_spans][0] (mapped_record.rq_spans[0]),
        .\out_b[origin][control][observing] (numeric_b.origin.control.observing),
        .\out_b[origin][control][allocated_tags] (numeric_b.origin.control.allocated_tags),
        .\out_b[origin][control][tag_count] (numeric_b.origin.control.tag_count),
        .\out_b[origin][control][outstanding_reads] (numeric_b.origin.control.outstanding_reads),
        .\out_b[origin][control][rq_partial] (numeric_b.origin.control.rq_partial),
        .\out_b[origin][control][rc_partial] (numeric_b.origin.control.rc_partial),
        .\out_b[origin][control][requests] (numeric_b.origin.control.requests),
        .\out_b[origin][control][completed_reads] (numeric_b.origin.control.completed_reads),
        .\out_b[origin][control][posted_writes] (numeric_b.origin.control.posted_writes),
        .\out_b[origin][control][early_fault_sources] (numeric_b.origin.control.early_fault_sources),
        .\out_b[origin][control][early_protocol_fault] (numeric_b.origin.control.early_protocol_fault),
        .\out_b[origin][control][control_pending] (numeric_b.origin.control.control_pending),
        .\out_b[origin][record][valid] (numeric_b.origin.record.valid),
        .\out_b[origin][record][sequence_number] (numeric_b.origin.record.sequence_number),
        .\out_b[origin][record][effect_pending] (numeric_b.origin.record.effect_pending),
        .\out_b[origin][record][pre_observing] (numeric_b.origin.record.pre_observing),
        .\out_b[origin][record][expected_requester] (numeric_b.origin.record.expected_requester),
        .\out_b[origin][record][pre_rq_partial] (numeric_b.origin.record.pre_rq_partial),
        .\out_b[origin][record][pre_rc_partial] (numeric_b.origin.record.pre_rc_partial),
        .\out_b[origin][record][pre_rc_context][seen] (numeric_b.origin.record.pre_rc_context.seen),
        .\out_b[origin][record][pre_rc_context][expected] (numeric_b.origin.record.pre_rc_context.expected),
        .\out_b[origin][record][pre_rc_context][tag] (numeric_b.origin.record.pre_rc_context.tag),
        .\out_b[origin][record][pre_rc_context][terminal] (numeric_b.origin.record.pre_rc_context.terminal),
        .\out_b[origin][record][pre_rc_context][normal] (numeric_b.origin.record.pre_rc_context.normal),
        .\out_b[origin][record][pre_rc_context][owned] (numeric_b.origin.record.pre_rc_context.owned),
        .\out_b[origin][record][pre_rc_context][description] (numeric_b.origin.record.pre_rc_context.description),
        .\out_b[origin][record][pre_rc_context][words] (numeric_b.origin.record.pre_rc_context.words),
        .\out_b[origin][record][pre_rc_context][delivery_enabled] (numeric_b.origin.record.pre_rc_context.delivery_enabled),
        .\out_b[origin][record][pre_rq_context] (numeric_b.origin.record.pre_rq_context),
        .\out_b[origin][record][rq_valid] (numeric_b.origin.record.rq_valid),
        .\out_b[origin][record][rc_valid] (numeric_b.origin.record.rc_valid),
        .\out_b[origin][record][rc_ready] (numeric_b.origin.record.rc_ready),
        .\out_b[origin][record][rq_ready] (numeric_b.origin.record.rq_ready),
        .\out_b[origin][record][rq_data] (numeric_b.origin.record.rq_data),
        .\out_b[origin][record][rc_data] (numeric_b.origin.record.rc_data),
        .\out_b[origin][record][rq_user] (numeric_b.origin.record.rq_user),
        .\out_b[origin][record][rc_user] (numeric_b.origin.record.rc_user),
        .\out_b[origin][record][queries][3][valid] (numeric_b.origin.record.queries[3].valid),
        .\out_b[origin][record][queries][3][owned] (numeric_b.origin.record.queries[3].owned),
        .\out_b[origin][record][queries][3][attributes_tc] (numeric_b.origin.record.queries[3].attributes_tc),
        .\out_b[origin][record][queries][3][description] (numeric_b.origin.record.queries[3].description),
        .\out_b[origin][record][queries][3][words] (numeric_b.origin.record.queries[3].words),
        .\out_b[origin][record][queries][3][delivery_enabled] (numeric_b.origin.record.queries[3].delivery_enabled),
        .\out_b[origin][record][queries][2][valid] (numeric_b.origin.record.queries[2].valid),
        .\out_b[origin][record][queries][2][owned] (numeric_b.origin.record.queries[2].owned),
        .\out_b[origin][record][queries][2][attributes_tc] (numeric_b.origin.record.queries[2].attributes_tc),
        .\out_b[origin][record][queries][2][description] (numeric_b.origin.record.queries[2].description),
        .\out_b[origin][record][queries][2][words] (numeric_b.origin.record.queries[2].words),
        .\out_b[origin][record][queries][2][delivery_enabled] (numeric_b.origin.record.queries[2].delivery_enabled),
        .\out_b[origin][record][queries][1][valid] (numeric_b.origin.record.queries[1].valid),
        .\out_b[origin][record][queries][1][owned] (numeric_b.origin.record.queries[1].owned),
        .\out_b[origin][record][queries][1][attributes_tc] (numeric_b.origin.record.queries[1].attributes_tc),
        .\out_b[origin][record][queries][1][description] (numeric_b.origin.record.queries[1].description),
        .\out_b[origin][record][queries][1][words] (numeric_b.origin.record.queries[1].words),
        .\out_b[origin][record][queries][1][delivery_enabled] (numeric_b.origin.record.queries[1].delivery_enabled),
        .\out_b[origin][record][queries][0][valid] (numeric_b.origin.record.queries[0].valid),
        .\out_b[origin][record][queries][0][owned] (numeric_b.origin.record.queries[0].owned),
        .\out_b[origin][record][queries][0][attributes_tc] (numeric_b.origin.record.queries[0].attributes_tc),
        .\out_b[origin][record][queries][0][description] (numeric_b.origin.record.queries[0].description),
        .\out_b[origin][record][queries][0][words] (numeric_b.origin.record.queries[0].words),
        .\out_b[origin][record][queries][0][delivery_enabled] (numeric_b.origin.record.queries[0].delivery_enabled),
        .\out_b[origin][record][eops][3][valid] (numeric_b.origin.record.eops[3].valid),
        .\out_b[origin][record][eops][3][terminal] (numeric_b.origin.record.eops[3].terminal),
        .\out_b[origin][record][eops][3][delivery_enabled] (numeric_b.origin.record.eops[3].delivery_enabled),
        .\out_b[origin][record][eops][3][description] (numeric_b.origin.record.eops[3].description),
        .\out_b[origin][record][eops][3][words] (numeric_b.origin.record.eops[3].words),
        .\out_b[origin][record][eops][3][lane] (numeric_b.origin.record.eops[3].lane),
        .\out_b[origin][record][eops][2][valid] (numeric_b.origin.record.eops[2].valid),
        .\out_b[origin][record][eops][2][terminal] (numeric_b.origin.record.eops[2].terminal),
        .\out_b[origin][record][eops][2][delivery_enabled] (numeric_b.origin.record.eops[2].delivery_enabled),
        .\out_b[origin][record][eops][2][description] (numeric_b.origin.record.eops[2].description),
        .\out_b[origin][record][eops][2][words] (numeric_b.origin.record.eops[2].words),
        .\out_b[origin][record][eops][2][lane] (numeric_b.origin.record.eops[2].lane),
        .\out_b[origin][record][eops][1][valid] (numeric_b.origin.record.eops[1].valid),
        .\out_b[origin][record][eops][1][terminal] (numeric_b.origin.record.eops[1].terminal),
        .\out_b[origin][record][eops][1][delivery_enabled] (numeric_b.origin.record.eops[1].delivery_enabled),
        .\out_b[origin][record][eops][1][description] (numeric_b.origin.record.eops[1].description),
        .\out_b[origin][record][eops][1][words] (numeric_b.origin.record.eops[1].words),
        .\out_b[origin][record][eops][1][lane] (numeric_b.origin.record.eops[1].lane),
        .\out_b[origin][record][eops][0][valid] (numeric_b.origin.record.eops[0].valid),
        .\out_b[origin][record][eops][0][terminal] (numeric_b.origin.record.eops[0].terminal),
        .\out_b[origin][record][eops][0][delivery_enabled] (numeric_b.origin.record.eops[0].delivery_enabled),
        .\out_b[origin][record][eops][0][description] (numeric_b.origin.record.eops[0].description),
        .\out_b[origin][record][eops][0][words] (numeric_b.origin.record.eops[0].words),
        .\out_b[origin][record][eops][0][lane] (numeric_b.origin.record.eops[0].lane),
        .\out_b[origin][record][rq_spans][1] (numeric_b.origin.record.rq_spans[1]),
        .\out_b[origin][record][rq_spans][0] (numeric_b.origin.record.rq_spans[0]),
        .\out_b[origin][queries][3] (numeric_b.origin.queries[3]),
        .\out_b[origin][queries][2] (numeric_b.origin.queries[2]),
        .\out_b[origin][queries][1] (numeric_b.origin.queries[1]),
        .\out_b[origin][queries][0] (numeric_b.origin.queries[0]),
        .\out_b[origin][eops][3] (numeric_b.origin.eops[3]),
        .\out_b[origin][eops][2] (numeric_b.origin.eops[2]),
        .\out_b[origin][eops][1] (numeric_b.origin.eops[1]),
        .\out_b[origin][eops][0] (numeric_b.origin.eops[0]),
        .\out_b[origin][context_value] (numeric_b.origin.context_value),
        .\out_b[query_delivered][3] (numeric_b.query_delivered[3]),
        .\out_b[query_delivered][2] (numeric_b.query_delivered[2]),
        .\out_b[query_delivered][1] (numeric_b.query_delivered[1]),
        .\out_b[query_delivered][0] (numeric_b.query_delivered[0]),
        .\out_b[eop_delivered][3] (numeric_b.eop_delivered[3]),
        .\out_b[eop_delivered][2] (numeric_b.eop_delivered[2]),
        .\out_b[eop_delivered][1] (numeric_b.eop_delivered[1]),
        .\out_b[eop_delivered][0] (numeric_b.eop_delivered[0]),
        .\out_b[eop_removed][3] (numeric_b.eop_removed[3]),
        .\out_b[eop_removed][2] (numeric_b.eop_removed[2]),
        .\out_b[eop_removed][1] (numeric_b.eop_removed[1]),
        .\out_b[eop_removed][0] (numeric_b.eop_removed[0]),
        .\out_b[context_delivered] (numeric_b.context_delivered));
    task check_mapped;
        begin
            if (numeric_b!==reference_numeric_b) begin
                $display("NUMERIC_MAPPED_ACTUAL %h",numeric_b);
                $display("NUMERIC_MAPPED_EXPECTED %h",reference_numeric_b);
                $fatal(1,"NUMERIC_MAPPED_PORT_MISMATCH row=%0d mode=%0d",index,mapped_mode);
            end
            mapped_rows=mapped_rows+1;
        end
    endtask

    ntrpu_pcie_symbolic_apply_r1_numeric_ab_mapped_reference_r1 apply_stage(.clk(clk),.cold_rst_n(cold_rst_n),
        .in_control(ced.d_control),.in_record(ced.out_record),.out_a(numeric_a));
    ntrpu_pcie_symbolic_delivery_r1_numeric_ab_mapped_reference_r1 delivery_stage(.clk(clk),.cold_rst_n(cold_rst_n),
        .in_a(numeric_a),.out_b(reference_numeric_b));
    a_record_t expected_a, numeric_history_a [0:3];
    b_record_t expected_b, numeric_history_b [0:3];
    reg [179:0] numeric_history_query [0:3], numeric_actual_query;
    reg [70:0] numeric_history_context [0:3], numeric_actual_context;
    reg [207:0] numeric_history_lanes [0:3], numeric_actual_lanes;
    reg [20:0] numeric_history_bytes [0:3];
    reg signed [22:0] numeric_history_candidate [0:3], numeric_actual_candidate;
    integer numeric_history_origin [0:3];
    reg [3:0] numeric_history_valid=0;
    integer numeric_captured=0, numeric_a_retired=0, numeric_b_retired=0;
    integer numeric_cancelled_a=0, numeric_cancelled_b=0, numeric_busy_cold=0;
    integer numeric_slot, numeric_hist, numeric_lane;
    reg [16:0] numeric_sum;
    reg [15:0] numeric_lanes_used;
    reg [24:0] numeric_expected_value;
    task check_numeric;
        begin
            if (!cold_rst_n) begin
                if (numeric_a!=='0 || numeric_b!=='0)
                    $fatal(1,"NUMERIC_COLD_MISMATCH row=%0d",index);
                if (numeric_history_valid!=0) numeric_busy_cold=numeric_busy_cold+1;
                for (numeric_hist=0;numeric_hist<4;numeric_hist=numeric_hist+1)
                    if (numeric_history_valid[numeric_hist] && numeric_history_origin[numeric_hist]<expected_rows) begin
                        numeric_cancelled_b=numeric_cancelled_b+1;
                        if (numeric_hist<3) numeric_cancelled_a=numeric_cancelled_a+1;
                    end
                numeric_history_valid=0;
            end else begin
                if (numeric_history_valid[2]) begin
                    if (numeric_a!==numeric_history_a[2])
                        $fatal(1,"NUMERIC_A_MISMATCH row=%0d origin=%0d",index,numeric_history_origin[2]);
                    if (numeric_history_origin[2]<expected_rows) numeric_a_retired=numeric_a_retired+1;
                end else if (numeric_a!=='0) $fatal(1,"NUMERIC_A_FILL_MISMATCH row=%0d",index);
                if (numeric_history_valid[3]) begin
                    if (numeric_b!==numeric_history_b[3])
                        $fatal(1,"NUMERIC_B_MISMATCH row=%0d origin=%0d",index,numeric_history_origin[3]);
                    numeric_actual_query=0; numeric_actual_lanes=0; numeric_lanes_used=0;
                    for (numeric_slot=0;numeric_slot<4;numeric_slot=numeric_slot+1) begin
                        if (numeric_b.origin.record.queries[numeric_slot].valid) begin
                            numeric_actual_query[176+numeric_slot]=1;
                            numeric_actual_query[172+numeric_slot]=numeric_b.origin.record.queries[numeric_slot].owned;
                            numeric_actual_query[52+numeric_slot*30+:30]={
                                numeric_b.origin.record.queries[numeric_slot].attributes_tc,
                                numeric_b.origin.queries[numeric_slot][11:0],numeric_b.origin.queries[numeric_slot][24:12]};
                            numeric_actual_query[numeric_slot*13+:13]=numeric_b.query_delivered[numeric_slot];
                        end
                        if (numeric_b.origin.record.eops[numeric_slot].valid) begin
                            numeric_lane=numeric_b.origin.record.eops[numeric_slot].lane;
                            if ((^numeric_b.origin.record.eops[numeric_slot].lane)===1'bx || numeric_lanes_used[numeric_lane])
                                $fatal(1,"NUMERIC_EOP_ALIAS_MISMATCH row=%0d",index);
                            numeric_lanes_used[numeric_lane]=1;
                            numeric_actual_lanes[numeric_lane*13+:13]=numeric_b.eop_removed[numeric_slot];
                        end
                    end
                    numeric_actual_context={numeric_b.origin.record.pre_rc_context.seen,
                        numeric_b.origin.record.pre_rc_context.expected,numeric_b.origin.record.pre_rc_context.tag,
                        numeric_b.origin.record.pre_rc_context.terminal,numeric_b.origin.record.pre_rc_context.normal,
                        numeric_b.origin.record.pre_rc_context.owned,numeric_b.origin.context_value[24:12],
                        numeric_b.context_delivered,numeric_b.origin.context_value[11:0]};
                    if (numeric_actual_query!==numeric_history_query[3]) $fatal(1,"NUMERIC_R7_QUERY_MISMATCH row=%0d",index);
                    if (numeric_actual_context!==numeric_history_context[3]) $fatal(1,"NUMERIC_R7_CONTEXT_MISMATCH row=%0d",index);
                    if (numeric_actual_lanes!==numeric_history_lanes[3]) $fatal(1,"NUMERIC_LANES_MISMATCH row=%0d",index);
                    numeric_sum=0;
                    for (numeric_lane=0;numeric_lane<16;numeric_lane=numeric_lane+1)
                        numeric_sum=numeric_sum+numeric_actual_lanes[numeric_lane*13+:13];
                    numeric_actual_candidate=$signed({2'b0,numeric_history_bytes[3]})+
                        $signed({{8{numeric_b.origin.record.rq_spans[0][14]}},numeric_b.origin.record.rq_spans[0]})+
                        $signed({{8{numeric_b.origin.record.rq_spans[1][14]}},numeric_b.origin.record.rq_spans[1]})-
                        $signed({6'b0,numeric_sum});
                    if (numeric_actual_candidate!==numeric_history_candidate[3]) $fatal(1,"NUMERIC_BYTES_MISMATCH row=%0d",index);
                    if (numeric_history_origin[3]<expected_rows) numeric_b_retired=numeric_b_retired+1;
                end else if (numeric_b!=='0) $fatal(1,"NUMERIC_B_FILL_MISMATCH row=%0d",index);
                for (numeric_hist=3;numeric_hist>0;numeric_hist=numeric_hist-1) begin
                    numeric_history_a[numeric_hist]=numeric_history_a[numeric_hist-1];
                    numeric_history_b[numeric_hist]=numeric_history_b[numeric_hist-1];
                    numeric_history_query[numeric_hist]=numeric_history_query[numeric_hist-1];
                    numeric_history_context[numeric_hist]=numeric_history_context[numeric_hist-1];
                    numeric_history_lanes[numeric_hist]=numeric_history_lanes[numeric_hist-1];
                    numeric_history_bytes[numeric_hist]=numeric_history_bytes[numeric_hist-1];
                    numeric_history_candidate[numeric_hist]=numeric_history_candidate[numeric_hist-1];
                    numeric_history_origin[numeric_hist]=numeric_history_origin[numeric_hist-1];
                    numeric_history_valid[numeric_hist]=numeric_history_valid[numeric_hist-1];
                end
                expected_a='0; expected_b='0;
                expected_a.control=s1_ports[2497:2107]; expected_a.record=s1.out_record;
                for (numeric_slot=0;numeric_slot<4;numeric_slot=numeric_slot+1) begin
                    if (s1.out_record.queries[numeric_slot].valid) begin
                        expected_a.queries[numeric_slot]=model.symbolic_value(s1.out_record.queries[numeric_slot].description);
                        expected_b.query_delivered[numeric_slot]=model.symbolic_delivery(s1.out_record.queries[numeric_slot].description,
                            s1.out_record.queries[numeric_slot].words,s1.out_record.queries[numeric_slot].delivery_enabled);
                    end
                    if (s1.out_record.eops[numeric_slot].valid) begin
                        numeric_expected_value=model.symbolic_value(s1.out_record.eops[numeric_slot].description);
                        expected_a.eops[numeric_slot]=numeric_expected_value;
                        expected_b.eop_delivered[numeric_slot]=model.symbolic_delivery(s1.out_record.eops[numeric_slot].description,
                            s1.out_record.eops[numeric_slot].words,s1.out_record.eops[numeric_slot].delivery_enabled);
                        expected_b.eop_removed[numeric_slot]=s1.out_record.eops[numeric_slot].terminal ?
                            numeric_expected_value[24:12] : expected_b.eop_delivered[numeric_slot];
                    end
                end
                expected_a.context_value=model.symbolic_value(s1.out_record.pre_rc_context.description);
                expected_b.context_delivered=model.symbolic_delivery(s1.out_record.pre_rc_context.description,
                    s1.out_record.pre_rc_context.words,s1.out_record.pre_rc_context.delivery_enabled);
                expected_b.origin=expected_a;
                numeric_history_a[0]=expected_a; numeric_history_b[0]=expected_b;
                numeric_history_query[0]=saved_query; numeric_history_context[0]=saved_rc;
                numeric_history_lanes[0]=saved_lanes; numeric_history_bytes[0]=saved_bytes;
                numeric_history_candidate[0]=saved_byte_candidate;
                numeric_history_origin[0]=index; numeric_history_valid[0]=1;
                if (index<expected_rows) numeric_captured=numeric_captured+1;
            end
            if (index%2048==0) $display("NUMERIC_PROGRESS bank=%s row=%0d a_retired=%0d b_retired=%0d cancelled_a=%0d cancelled_b=%0d",
                bank,index,numeric_a_retired,numeric_b_retired,numeric_cancelled_a,numeric_cancelled_b);
        end
    endtask

    task precheck;
        begin
            if (index!=0 && s1.out_record!==previous_record)
                $fatal(1,"S1_COMBINATIONAL_CAPTURE row=%0d",index);
            expected_record='0;
            expected_record.valid=1;
            expected_record.sequence_number=sequence_in;
            expected_record.effect_pending=reference_i.observing && ((rq_valid!==1'b0)||(rc_valid!==1'b0));
            expected_record.pre_observing=reference_i.observing;
            expected_record.expected_requester=reference_i.saved_requester;
            expected_record.pre_rq_partial=reference_i.rq_partial;
            expected_record.pre_rc_partial=reference_i.rc_partial;
            expected_record.pre_rc_context={model.rc_seen,model.rc_expected,model.rc_tag,
                model.rc_terminal,model.rc_normal,model.rc_owned,model.rc_description,
                model.rc_description_count,model.rc_delivery_enabled};
            expected_record.pre_rq_context=reference_rq;
            expected_record.rq_valid=rq_valid; expected_record.rq_ready=rq_ready;
            expected_record.rc_valid=rc_valid; expected_record.rc_ready=rc_ready;
            expected_record.rq_data=rq_data; expected_record.rc_data=rc_data;
            expected_record.rq_user=rq_user; expected_record.rc_user=rc_user;
            expected_record.rq_spans={reference_i.rq_added_span[1],reference_i.rq_added_span[0]};
            saved_query={probe.observed_query_valid,probe.observed_query_owned,
                probe.observed_query_data,probe.observed_query_delivered};
            saved_rc=reference_rc;
            for (lane=0;lane<16;lane=lane+1) saved_lanes[lane*13+:13]=model.rc_removed_by_lane[lane];
            saved_eop_valid=model.rc_complete_events | model.rc_update_valid;
            saved_eop_terminal=model.rc_complete_events;
            saved_bytes=reference_i.remaining_bytes;
            saved_byte_candidate=reference_i.nbytes;
            if (!cold_rst_n) range_sticky=0;
            else if (reference_i.observing && ((reference_i.ntags<0 || reference_i.ntags>256 ||
                reference_i.nreads<0 || reference_i.nreads>257)===1'b1)) range_sticky=1;
        end
    endtask

    task check_boundary;
        begin
            if (control!==reference_control) $fatal(1,"S1_CONTROL_MISMATCH row=%0d",index);
            if (probe_state!==reference_state) $fatal(1,"S1_PROBE_STATE_MISMATCH row=%0d",index);
            if ((s1.early_fault_sources & 16'hdfff)!==(model.early_fault_sources & 16'hdfff) ||
                s1.early_fault_sources[13]!==range_sticky ||
                s1.early_protocol_fault!==( |s1.early_fault_sources))
                $fatal(1,"S1_FAULT_MISMATCH row=%0d",index);
            expected_pending=(reference_i.tag_count!=0 || reference_i.outstanding_reads!=0 ||
                reference_i.rq_partial || reference_i.rc_partial || reference_i.rq_held ||
                reference_i.rc_held || rq_valid || rc_valid);
            if (s1.control_pending!==expected_pending) $fatal(1,"S1_PENDING_MISMATCH row=%0d",index);
            for (tag_q=0;tag_q<256;tag_q=tag_q+1) begin
                value=model.symbolic_value(s1.symbolic_rows[tag_q]);
                if ({s1.attributes[tag_q],s1.traffic_class[tag_q],value[11:0],value[24:12]} !==
                    {reference_i.attributes[tag_q],reference_i.traffic_class[tag_q],
                    reference_i.address_next[tag_q],reference_i.remaining[tag_q]})
                    $fatal(1,"S1_TABLE_MISMATCH row=%0d tag=%0d",index,tag_q);
                if (s1.symbolic_rows[tag_q]!==model.symbolic_rows[tag_q])
                    $fatal(1,"S1_SYMBOLIC_COMMIT_MISMATCH row=%0d tag=%0d",index,tag_q);
                if ({probe.attributes[tag_q],probe.traffic_class[tag_q],probe.address_next[tag_q],probe.remaining[tag_q]} !==
                    {reference_i.attributes[tag_q],reference_i.traffic_class[tag_q],reference_i.address_next[tag_q],reference_i.remaining[tag_q]})
                    $fatal(1,"S1_PROBE_TABLE_MISMATCH row=%0d tag=%0d",index,tag_q);
                tables=tables+1;
            end
            if (!cold_rst_n) begin
                if (s1.out_record!=='0) $fatal(1,"S1_COLD_CAPTURE_MISMATCH row=%0d",index);
                cold_rows=cold_rows+1;
            end else begin
                checked_record=s1.out_record;
                checked_record.queries='0; checked_record.eops='0;
                if (checked_record!==expected_record) $fatal(1,"S1_RECORD_MISMATCH row=%0d",index);
                actual_query=0;
                for (slot=0;slot<4;slot=slot+1) begin
                    if (s1.out_record.queries[slot].valid) begin
                        value=model.symbolic_value(s1.out_record.queries[slot].description);
                        delivered=model.symbolic_delivery(s1.out_record.queries[slot].description,
                            s1.out_record.queries[slot].words,s1.out_record.queries[slot].delivery_enabled);
                        actual_query[176+slot]=1;
                        actual_query[172+slot]=s1.out_record.queries[slot].owned;
                        actual_query[52+slot*30+:30]={s1.out_record.queries[slot].attributes_tc,value[11:0],value[24:12]};
                        actual_query[slot*13+:13]=delivered;
                        queries=queries+1;
                    end else if (s1.out_record.queries[slot]!=='0)
                        $fatal(1,"S1_UNUSED_QUERY_MISMATCH row=%0d",index);
                end
                if (actual_query!==saved_query) $fatal(1,"S1_QUERY_MISMATCH row=%0d",index);
                value=model.symbolic_value(s1.out_record.pre_rc_context.description);
                delivered=model.symbolic_delivery(s1.out_record.pre_rc_context.description,
                    s1.out_record.pre_rc_context.words,s1.out_record.pre_rc_context.delivery_enabled);
                actual_rc={s1.out_record.pre_rc_context.seen,s1.out_record.pre_rc_context.expected,
                    s1.out_record.pre_rc_context.tag,s1.out_record.pre_rc_context.terminal,
                    s1.out_record.pre_rc_context.normal,s1.out_record.pre_rc_context.owned,
                    value[24:12],delivered,value[11:0]};
                if (actual_rc!==saved_rc) $fatal(1,"S1_CONTEXT_MISMATCH row=%0d",index);
                actual_lanes=0; lanes_used=0; rc_sum=0;
                for (slot=0;slot<4;slot=slot+1) begin
                    if (s1.out_record.eops[slot].valid!==saved_eop_valid[slot])
                        $fatal(1,"S1_EOP_ELIGIBILITY_MISMATCH row=%0d slot=%0d",index,slot);
                    if (s1.out_record.eops[slot].valid) begin
                        if (s1.out_record.eops[slot].terminal!==saved_eop_terminal[slot])
                            $fatal(1,"S1_EOP_KIND_MISMATCH row=%0d",index);
                        lane=s1.out_record.eops[slot].lane;
                        if ((^s1.out_record.eops[slot].lane)===1'bx || lanes_used[lane])
                            $fatal(1,"S1_EOP_ALIAS_MISMATCH row=%0d",index);
                        lanes_used[lane]=1;
                        value=model.symbolic_value(s1.out_record.eops[slot].description);
                        delivered=model.symbolic_delivery(s1.out_record.eops[slot].description,
                            s1.out_record.eops[slot].words,s1.out_record.eops[slot].delivery_enabled);
                        actual_lanes[lane*13+:13]=s1.out_record.eops[slot].terminal ? value[24:12] : delivered;
                        eops=eops+1;
                    end else if (s1.out_record.eops[slot]!=='0)
                        $fatal(1,"S1_UNUSED_EOP_MISMATCH row=%0d",index);
                end
                if (actual_lanes!==saved_lanes) $fatal(1,"S1_EOP_LANES_MISMATCH row=%0d",index);
                for (lane=0;lane<16;lane=lane+1) rc_sum=rc_sum+actual_lanes[lane*13+:13];
                actual_byte_candidate=$signed({2'b0,saved_bytes})+
                    $signed({{8{s1.out_record.rq_spans[0][14]}},s1.out_record.rq_spans[0]})+
                    $signed({{8{s1.out_record.rq_spans[1][14]}},s1.out_record.rq_spans[1]})-$signed({6'b0,rc_sum});
                if (actual_byte_candidate!==saved_byte_candidate) $fatal(1,"S1_BYTE_CANDIDATE_MISMATCH row=%0d",index);
            end
            previous_record=s1.out_record;
            if (reference_i.protocol_fault) fault_rows=fault_rows+1;
            if (index==128) wraps=wraps+1;
            if ((index%1024)==0 || scenario!=last_scenario)
                $display("S1_PROGRESS bank=%s index=%0d scenario=%0d tables=%0d queries=%0d eops=%0d cold=%0d faults=%0d",
                    bank,index,scenario,tables,queries,eops,cold_rows,fault_rows);
        end
    endtask

    initial begin
        #200; // Vendor GSR startup, identical for positive and negative cases.
        if (!$value$plusargs("VECTORS=%s",vector_file) || !$value$plusargs("BANK=%s",bank) ||
            !$value$plusargs("ROWS=%d",expected_rows)) $fatal(1,"missing S1 arguments");
        $display("S1_SCHEMA record_bits=%0d query_bits=%0d eop_bits=%0d context_bits=%0d",
            $bits(s1_record_t),$bits(query_t),$bits(eop_t),$bits(rc_context_t));
        fd=$fopen(vector_file,"r"); if (!fd) $fatal(1,"cannot open S1 vectors");
        while (index<expected_rows) begin
            @(negedge clk);
            fields=$fscanf(fd,"%d %d %d %d %d %d %h %h %d %h %b %h %d %d %b %h %d %d %d %d %d %d %d %d %d %d %d %h\n",
                scenario,native_ps,cold_rst_n,profile_valid,user_reset,link_up,requester_id,attribute_mask,
                rq_valid,rq_ready,rq_data,rq_user,rc_valid,rc_ready,rc_data,rc_user,
                expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites,expected_bitmap);
            if (fields==-1 && $feof(fd)) break;
            if (fields!=28) $fatal(1,"bad S1 vector fields=%0d row=%0d",fields,index);
            #1; precheck();
            @(posedge clk); #1; check_mapped(); check_boundary(); check_ced(); check_numeric();
            if (reference_i.observing!==expected_observing[0] || reference_i.protocol_fault!==expected_fault[0])
                $fatal(1,"S1_REFERENCE_ORACLE_MISMATCH row=%0d",index);
            if (check_state && ({reference_i.tag_count,reference_i.outstanding_reads,reference_i.remaining_bytes,
                reference_i.rq_partial,reference_i.rc_partial,reference_i.requests,reference_i.completed_reads,
                reference_i.posted_writes,reference_i.allocated_tags} !==
                {etags[8:0],ereads[8:0],ebytes[20:0],erp[0],ecp[0],erequests[31:0],ecompleted[31:0],ewrites[31:0],expected_bitmap}))
                $fatal(1,"S1_REFERENCE_COUNTER_MISMATCH row=%0d",index);
            if (scenario!=last_scenario) begin cases=cases+1; last_scenario=scenario; end
            index=index+1;
        end
        fields=$fscanf(fd,"%s",trailing_token);
        if (fields!=-1) $fatal(1,"CED_EXTRA_VECTOR_DATA");
        $fclose(fd);
        if (index!=expected_rows || tables!=256*index) $fatal(1,"S1_INCOMPLETE_ROWS");
        $display("S1_FRONTEND_PASS bank=%s rows=%0d scenarios=%0d tables=%0d queries=%0d eops=%0d cold=%0d faults=%0d wraps=%0d",
            bank,index,cases,tables,queries,eops,cold_rows,fault_rows,wraps);
        // Four fixed drain edges retire all surviving input origins. Diagnostic idle
        // records continue, but are excluded from the raw-row accounting below.
        for (flush_index=0;flush_index<4;flush_index=flush_index+1) begin
            @(negedge clk); cold_rst_n=1; rq_valid=0; rc_valid=0; rq_ready=15; rc_ready=1;
            #1; precheck();
            @(posedge clk); #1; check_mapped(); check_ced(); check_numeric();
            previous_record=s1.out_record; index=index+1;
        end
        if (ced_captured!=ced_retired+ced_cancelled || ced_tables!=256*ced_retired)
            $fatal(1,"CED_INCOMPLETE_RETIREMENT captured=%0d retired=%0d cancelled=%0d",ced_captured,ced_retired,ced_cancelled);
        $display("CED_FRONTEND_PASS bank=%s rows=%0d captured=%0d retired=%0d cancelled=%0d tables=%0d busy_cold=%0d",
            bank,expected_rows,ced_captured,ced_retired,ced_cancelled,ced_tables,ced_busy_cold);
        if (numeric_captured!=numeric_a_retired+numeric_cancelled_a ||
            numeric_captured!=numeric_b_retired+numeric_cancelled_b)
            $fatal(1,"NUMERIC_INCOMPLETE_RETIREMENT");
        $display("NUMERIC_AB_PASS bank=%s rows=%0d captured=%0d a_retired=%0d b_retired=%0d cancelled_a=%0d cancelled_b=%0d busy_cold=%0d a_bits=%0d b_bits=%0d",
            bank,expected_rows,numeric_captured,numeric_a_retired,numeric_b_retired,numeric_cancelled_a,
            numeric_cancelled_b,numeric_busy_cold,$bits(a_record_t),$bits(b_record_t));
        if (mapped_rows!=expected_rows+4 || $bits(numeric_b)!=2892)
            $fatal(1,"NUMERIC_MAPPED_INCOMPLETE");
        $display("NUMERIC_MAPPED_PASS bank=%s rows=%0d samples=%0d mode=%0d ports=2892",
            bank,expected_rows,mapped_rows,mapped_mode);
        $finish;
    end
    initial begin #1000000; $fatal(1,"S1 watchdog"); end
endmodule
