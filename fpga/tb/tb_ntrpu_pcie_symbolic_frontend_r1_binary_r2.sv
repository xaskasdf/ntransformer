`timescale 1ns / 1ps
// S1 boundary proof: original numeric R7, observational R7, reviewed value model.
// No source row is filtered after fault. Check all 256 rows on every clock.
module tb_ntrpu_pcie_symbolic_frontend_r1_binary_r2;
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
    ntrpu_pcie_request_inventory reference_i(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user));
    ntrpu_pcie_request_inventory_symbolic_probe probe(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user));
    ntrpu_pcie_inventory_scheduler model(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user));
    ntrpu_pcie_symbolic_frontend_rtl_reference s1(.clk(clk),.cold_rst_n(cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(rq_valid),.rc_valid(rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user),.sequence_in(sequence_in));

    integer mode=0;
    initial begin
        if ($test$plusargs("OMIT_RQ")) mode=1;
        if ($test$plusargs("OMIT_RC")) mode=2;
        if ($test$plusargs("OMIT_COLD")) mode=3;
        if ($test$plusargs("WRONG_SEQUENCE")) mode=4;
        $display("S1_MAPPED_MODE mode=%0d",mode);
    end
    wire s1_record_t mapped_record;
    ntrpu_pcie_symbolic_frontend mapped(.clk(clk),.cold_rst_n(mode==3 && index>16 ? 1'b1 : cold_rst_n),.profile_valid(profile_valid),.user_reset(user_reset),
        .link_up(link_up),.requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(mode==1 ? 1'b0 : rq_valid),.rc_valid(mode==2 ? 1'b0 : rc_valid),.rc_ready(rc_ready),.rq_ready(rq_ready),
        .rq_data(rq_data),.rc_data(rc_data),.rq_user(rq_user),.rc_user(rc_user),.sequence_in(mode==4 ? sequence_in+1 : sequence_in),
        .\out_record[valid] (mapped_record.valid),
        .\out_record[sequence_number] (mapped_record.sequence_number),
        .\out_record[effect_pending] (mapped_record.effect_pending),
        .\out_record[pre_observing] (mapped_record.pre_observing),
        .\out_record[expected_requester] (mapped_record.expected_requester),
        .\out_record[pre_rq_partial] (mapped_record.pre_rq_partial),
        .\out_record[pre_rc_partial] (mapped_record.pre_rc_partial),
        .\out_record[pre_rc_context][seen] (mapped_record.pre_rc_context.seen),
        .\out_record[pre_rc_context][expected] (mapped_record.pre_rc_context.expected),
        .\out_record[pre_rc_context][tag] (mapped_record.pre_rc_context.tag),
        .\out_record[pre_rc_context][terminal] (mapped_record.pre_rc_context.terminal),
        .\out_record[pre_rc_context][normal] (mapped_record.pre_rc_context.normal),
        .\out_record[pre_rc_context][owned] (mapped_record.pre_rc_context.owned),
        .\out_record[pre_rc_context][description] (mapped_record.pre_rc_context.description),
        .\out_record[pre_rc_context][words] (mapped_record.pre_rc_context.words),
        .\out_record[pre_rc_context][delivery_enabled] (mapped_record.pre_rc_context.delivery_enabled),
        .\out_record[pre_rq_context] (mapped_record.pre_rq_context),
        .\out_record[rq_valid] (mapped_record.rq_valid),
        .\out_record[rc_valid] (mapped_record.rc_valid),
        .\out_record[rc_ready] (mapped_record.rc_ready),
        .\out_record[rq_ready] (mapped_record.rq_ready),
        .\out_record[rq_data] (mapped_record.rq_data),
        .\out_record[rc_data] (mapped_record.rc_data),
        .\out_record[rq_user] (mapped_record.rq_user),
        .\out_record[rc_user] (mapped_record.rc_user),
        .\out_record[queries][3][valid] (mapped_record.queries[3].valid),
        .\out_record[queries][3][owned] (mapped_record.queries[3].owned),
        .\out_record[queries][3][attributes_tc] (mapped_record.queries[3].attributes_tc),
        .\out_record[queries][3][description] (mapped_record.queries[3].description),
        .\out_record[queries][3][words] (mapped_record.queries[3].words),
        .\out_record[queries][3][delivery_enabled] (mapped_record.queries[3].delivery_enabled),
        .\out_record[queries][2][valid] (mapped_record.queries[2].valid),
        .\out_record[queries][2][owned] (mapped_record.queries[2].owned),
        .\out_record[queries][2][attributes_tc] (mapped_record.queries[2].attributes_tc),
        .\out_record[queries][2][description] (mapped_record.queries[2].description),
        .\out_record[queries][2][words] (mapped_record.queries[2].words),
        .\out_record[queries][2][delivery_enabled] (mapped_record.queries[2].delivery_enabled),
        .\out_record[queries][1][valid] (mapped_record.queries[1].valid),
        .\out_record[queries][1][owned] (mapped_record.queries[1].owned),
        .\out_record[queries][1][attributes_tc] (mapped_record.queries[1].attributes_tc),
        .\out_record[queries][1][description] (mapped_record.queries[1].description),
        .\out_record[queries][1][words] (mapped_record.queries[1].words),
        .\out_record[queries][1][delivery_enabled] (mapped_record.queries[1].delivery_enabled),
        .\out_record[queries][0][valid] (mapped_record.queries[0].valid),
        .\out_record[queries][0][owned] (mapped_record.queries[0].owned),
        .\out_record[queries][0][attributes_tc] (mapped_record.queries[0].attributes_tc),
        .\out_record[queries][0][description] (mapped_record.queries[0].description),
        .\out_record[queries][0][words] (mapped_record.queries[0].words),
        .\out_record[queries][0][delivery_enabled] (mapped_record.queries[0].delivery_enabled),
        .\out_record[eops][3][valid] (mapped_record.eops[3].valid),
        .\out_record[eops][3][terminal] (mapped_record.eops[3].terminal),
        .\out_record[eops][3][delivery_enabled] (mapped_record.eops[3].delivery_enabled),
        .\out_record[eops][3][description] (mapped_record.eops[3].description),
        .\out_record[eops][3][words] (mapped_record.eops[3].words),
        .\out_record[eops][3][lane] (mapped_record.eops[3].lane),
        .\out_record[eops][2][valid] (mapped_record.eops[2].valid),
        .\out_record[eops][2][terminal] (mapped_record.eops[2].terminal),
        .\out_record[eops][2][delivery_enabled] (mapped_record.eops[2].delivery_enabled),
        .\out_record[eops][2][description] (mapped_record.eops[2].description),
        .\out_record[eops][2][words] (mapped_record.eops[2].words),
        .\out_record[eops][2][lane] (mapped_record.eops[2].lane),
        .\out_record[eops][1][valid] (mapped_record.eops[1].valid),
        .\out_record[eops][1][terminal] (mapped_record.eops[1].terminal),
        .\out_record[eops][1][delivery_enabled] (mapped_record.eops[1].delivery_enabled),
        .\out_record[eops][1][description] (mapped_record.eops[1].description),
        .\out_record[eops][1][words] (mapped_record.eops[1].words),
        .\out_record[eops][1][lane] (mapped_record.eops[1].lane),
        .\out_record[eops][0][valid] (mapped_record.eops[0].valid),
        .\out_record[eops][0][terminal] (mapped_record.eops[0].terminal),
        .\out_record[eops][0][delivery_enabled] (mapped_record.eops[0].delivery_enabled),
        .\out_record[eops][0][description] (mapped_record.eops[0].description),
        .\out_record[eops][0][words] (mapped_record.eops[0].words),
        .\out_record[eops][0][lane] (mapped_record.eops[0].lane),
        .\out_record[rq_spans][1] (mapped_record.rq_spans[1]),
        .\out_record[rq_spans][0] (mapped_record.rq_spans[0]));
    wire [2497:0] rtl_ports={s1.observing,s1.allocated_tags,s1.tag_count,s1.outstanding_reads,s1.rq_partial,s1.rc_partial,s1.requests,s1.completed_reads,s1.posted_writes,s1.early_fault_sources,s1.early_protocol_fault,s1.control_pending,s1.out_record};
    wire [2497:0] mapped_ports={mapped.observing,mapped.allocated_tags,mapped.tag_count,mapped.outstanding_reads,mapped.rq_partial,mapped.rc_partial,mapped.requests,mapped.completed_reads,mapped.posted_writes,mapped.early_fault_sources,mapped.early_protocol_fault,mapped.control_pending,mapped_record};

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
    string vector_file, bank;

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
            if (mapped_ports!==rtl_ports) begin
                $display("S1_MAPPED_DIAGNOSTIC row=%0d actual=%h reference=%h",index,mapped_ports,rtl_ports);
                $fatal(1,"S1_MAPPED_PORT_MISMATCH row=%0d mode=%0d",index,mode);
            end
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
        if (!$value$plusargs("VECTORS=%s",vector_file) || !$value$plusargs("BANK=%s",bank) ||
            !$value$plusargs("ROWS=%d",expected_rows)) $fatal(1,"missing S1 arguments");
        $display("S1_SCHEMA record_bits=%0d query_bits=%0d eop_bits=%0d context_bits=%0d",
            $bits(s1_record_t),$bits(query_t),$bits(eop_t),$bits(rc_context_t));
        #200; // Allow the vendor functional netlist GSR startup to finish.
        fd=$fopen(vector_file,"r"); if (!fd) $fatal(1,"cannot open S1 vectors");
        while (!$feof(fd)) begin
            @(negedge clk);
            fields=$fscanf(fd,"%d %d %d %d %d %d %h %h %d %h %b %h %d %d %b %h %d %d %d %d %d %d %d %d %d %d %d %h\n",
                scenario,native_ps,cold_rst_n,profile_valid,user_reset,link_up,requester_id,attribute_mask,
                rq_valid,rq_ready,rq_data,rq_user,rc_valid,rc_ready,rc_data,rc_user,
                expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites,expected_bitmap);
            if (fields==-1 && $feof(fd)) break;
            if (fields!=28) $fatal(1,"bad S1 vector fields=%0d row=%0d",fields,index);
            #1; precheck();
            @(posedge clk); #1; check_boundary();
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
        $fclose(fd);
        if (index!=expected_rows || tables!=256*index) $fatal(1,"S1_INCOMPLETE_ROWS");
        $display("S1_FRONTEND_PASS bank=%s rows=%0d scenarios=%0d tables=%0d queries=%0d eops=%0d cold=%0d faults=%0d wraps=%0d",
            bank,index,cases,tables,queries,eops,cold_rows,fault_rows,wraps);
        $display("S1_MAPPED_PASS bank=%s rows=%0d mode=%0d ports=2498",bank,index,mode);
        $finish;
    end
    initial begin #1000000; $fatal(1,"S1 watchdog"); end
endmodule
