`timescale 1ns / 1ps
module tb_ntrpu_pcie_inventory_scheduler;
    reg clk=0; always #2 clk=~clk;
    reg cold_rst_n=0, profile_valid=1, user_reset=0, link_up=1;
    reg [15:0] requester_id=16'h01a0;
    reg [1:0] attribute_mask=3;
    reg rq_valid=0, rc_valid=0, rc_ready=1;
    reg [3:0] rq_ready=15;
    reg [511:0] rq_data=0, rc_data=0;
    reg [136:0] rq_user=0;
    reg [160:0] rc_user=0;
    wire observing, protocol_fault, inventory_empty, rq_partial, rc_partial;
    wire [255:0] allocated_tags;
    wire [8:0] tag_count, outstanding_reads;
    wire [20:0] remaining_bytes;
    wire [31:0] requests, completed_reads, posted_writes;
    wire [15:0] fault_sources;
    ntrpu_pcie_request_inventory dut(.*);


    integer fd,fields,index=0,scenario,native_ps,last_scenario=-1,cases=0;
    integer scheduler_mode=0;
    initial begin
        if ($test$plusargs("OMIT_RQ")) scheduler_mode=1;
        if ($test$plusargs("OMIT_RC")) scheduler_mode=2;
        if ($test$plusargs("CORRUPT_TABLE")) scheduler_mode=3;
        if ($test$plusargs("CORRUPT_CONTEXT")) scheduler_mode=4;
        $display("SCHED_MODE mode=%0d",scheduler_mode);
    end
    wire observing_s, rq_partial_s, rc_partial_s, early_protocol_fault_s, state_pending_s;
    wire [255:0] allocated_tags_s;
    wire [8:0] tag_count_s, outstanding_reads_s;
    wire [20:0] remaining_bytes_s;
    wire [31:0] requests_s, completed_reads_s, posted_writes_s;
    wire [15:0] early_fault_sources_s;
    wire [3:0] header_owned_prefix, metadata_valid, metadata_owned;
    wire [119:0] metadata_data;
    wire [51:0] canonical_delivered;
    wire [70:0] retained_rc_context;
    wire [42:0] retained_rq_context;
    wire record_valid;
    wire [31:0] record_sequence;
    wire [31:0] sequence_in=32'hffffff80+index;
    ntrpu_pcie_inventory_scheduler sched (
        .clk(clk),.cold_rst_n(cold_rst_n),.sequence_in(sequence_in),
        .profile_valid(profile_valid),.user_reset(user_reset),.link_up(link_up),
        .requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(scheduler_mode==1 ? 1'b0 : rq_valid),.rq_ready(rq_ready),
        .rc_valid(scheduler_mode==2 ? 1'b0 : rc_valid),.rc_ready(rc_ready),
        .rq_data(rq_data),.rq_user(rq_user),.rc_data(rc_data),.rc_user(rc_user),
        .observing(observing_s),.allocated_tags(allocated_tags_s),.tag_count(tag_count_s),
        .outstanding_reads(outstanding_reads_s),.remaining_bytes(remaining_bytes_s),
        .rq_partial(rq_partial_s),.rc_partial(rc_partial_s),.requests(requests_s),
        .completed_reads(completed_reads_s),.posted_writes(posted_writes_s),
        .early_fault_sources(early_fault_sources_s),.early_protocol_fault(early_protocol_fault_s),
        .state_pending(state_pending_s),.header_owned_prefix(header_owned_prefix),
        .record_valid(record_valid),.record_sequence(record_sequence),
        .metadata_valid(metadata_valid),.metadata_owned(metadata_owned),.metadata_data(metadata_data),
        .canonical_delivered(canonical_delivered),.retained_rc_context(retained_rc_context),
        .retained_rq_context(retained_rq_context)
    );
    reg expected_record_valid, expected_pending;
    reg [31:0] expected_sequence;
    reg [3:0] expected_valid, expected_owned, captured_prefix;
    reg [119:0] expected_data;
    reg [51:0] expected_delivered;
    integer starts_q, slot_q, query_q, write_q, lane_q, latest_q, table_compares, tag_q;
    reg [7:0] query_tag;
    reg [95:0] query_header;
    reg signed [14:0] query_available, query_amount;
    reg [12:0] query_original;
    reg [11:0] query_address;
    task capture_scheduler_expectation;
        begin
            expected_record_valid=cold_rst_n && observing;
            expected_sequence=expected_record_valid ? sequence_in : 0;
            expected_valid=0; expected_owned=0; expected_data=0; expected_delivered=0;
            captured_prefix=expected_record_valid ? header_owned_prefix : 0;
            if (expected_record_valid && rc_valid===1'b1 && rc_ready===1'b1) begin
                starts_q=dut.thermometer(rc_user[67:64]);
                for (slot_q=0;slot_q<4;slot_q=slot_q+1)
                    if (slot_q<starts_q) expected_valid[rc_user[68+slot_q*2+:2]]=1;
                for (query_q=0;query_q<4;query_q=query_q+1) begin
                    if (expected_valid[query_q]) begin
                        query_header=rc_data[query_q*128+:96]; query_tag=query_header[71:64];
                        expected_owned[query_q]=allocated_tags[query_tag];
                        for (write_q=0;write_q<query_q;write_q=write_q+1)
                            if (dut.rc_release_valid[write_q] && dut.rc_release_tag[write_q]==query_tag)
                                expected_owned[query_q]=0;
                        query_original=dut.remaining[query_tag]; query_address=dut.address_next[query_tag];
                        latest_q=-1;
                        for (write_q=0;write_q<4;write_q=write_q+1)
                            if (dut.rc_update_valid[write_q] && dut.rc_update_tag[write_q]==query_tag &&
                                dut.rc_update_lane[write_q]<query_q*4 && (latest_q<0 || dut.rc_update_lane[write_q]>latest_q)) begin
                                query_original=dut.rc_update_remaining[write_q];
                                query_address=dut.rc_update_address[write_q];
                                latest_q=dut.rc_update_lane[write_q];
                            end
                        expected_data[query_q*30+:30]={dut.attributes[query_tag],dut.traffic_class[query_tag],query_address,query_original};
                        if (query_header[15:12]==0 && query_header[45:43]==0 && query_header[46]==0 && expected_owned[query_q]) begin
                            query_available=query_header[42:32]*4-query_address[1:0];
                            query_amount=(query_available<query_original) ? query_available : query_original;
                            expected_delivered[query_q*13+:13]=query_amount;
                        end
                    end
                end
            end
        end
    endtask
    task check_scheduler;
        begin
            table_compares=0;
`ifdef SCHED_RTL_INTERNAL_CHECKS
            for (tag_q=0;tag_q<256;tag_q=tag_q+1) begin
                if ({sched.attributes[tag_q],sched.traffic_class[tag_q],sched.address_next[tag_q],sched.remaining[tag_q]} !==
                    {dut.attributes[tag_q],dut.traffic_class[tag_q],dut.address_next[tag_q],dut.remaining[tag_q]})
                    $fatal(1,"SCHED_TABLE_MISMATCH row=%0d tag=%0d mode=%0d",index,tag_q,scheduler_mode);
                table_compares=table_compares+1;
            end
`endif
            if ({record_valid,record_sequence,metadata_valid,metadata_owned,metadata_data,canonical_delivered} !==
                {expected_record_valid,expected_sequence,expected_valid,expected_owned,expected_data,expected_delivered} ||
                captured_prefix!==expected_owned)
                $fatal(1,"SCHED_QUERY_MISMATCH row=%0d mode=%0d",index,scheduler_mode);
            if ({observing_s,allocated_tags_s,tag_count_s,outstanding_reads_s,remaining_bytes_s,rq_partial_s,rc_partial_s,
                 requests_s,completed_reads_s,posted_writes_s} !==
                {observing,allocated_tags,tag_count,outstanding_reads,remaining_bytes,rq_partial,rc_partial,
                 requests,completed_reads,posted_writes})
                $fatal(1,"SCHED_STATE_MISMATCH row=%0d mode=%0d",index,scheduler_mode);
            if (retained_rc_context !== {dut.rc_seen,dut.rc_expected,dut.rc_tag,dut.rc_terminal,dut.rc_normal,
                dut.rc_owned,dut.rc_original,dut.rc_delivered,dut.rc_address} ||
                retained_rq_context !== {dut.rq_seen,dut.rq_expected,dut.rq_first,dut.rq_last,dut.rq_count,dut.rq_write,dut.rq_aborted})
                $fatal(1,"SCHED_CONTEXT_MISMATCH row=%0d mode=%0d",index,scheduler_mode);
            expected_pending=(tag_count!=0 || outstanding_reads!=0 || remaining_bytes!=0 ||
                rq_partial || rc_partial || dut.rq_held || dut.rc_held || (scheduler_mode==1 ? 1'b0 : rq_valid) || (scheduler_mode==2 ? 1'b0 : rc_valid));
            if (state_pending_s !== expected_pending)
                $fatal(1,"SCHED_PENDING_MISMATCH row=%0d mode=%0d",index,scheduler_mode);
            $display("SCHED_ROW index=%0d scenario=%0d mode=%0d words=%0d observing=%0d tags=%0d reads=%0d bytes=%0d rq_partial=%0d rc_partial=%0d requests=%0d completed=%0d writes=%0d bitmap=%064h record=%0d seq=%08h valid=%h owned=%h metadata=%030h delivered=%013h rc_context=%018h rq_context=%011h pending=%0d early_faults=%04h",
                index,scenario,scheduler_mode,table_compares,observing_s,tag_count_s,outstanding_reads_s,remaining_bytes_s,
                rq_partial_s,rc_partial_s,requests_s,completed_reads_s,posted_writes_s,allocated_tags_s,record_valid,record_sequence,
                metadata_valid,metadata_owned,metadata_data,canonical_delivered,retained_rc_context,retained_rq_context,state_pending_s,early_fault_sources_s);
            $display("SCHED_EXPECT index=%0d scenario=%0d observing=%0d tags=%0d reads=%0d bytes=%0d rq_partial=%0d rc_partial=%0d requests=%0d completed=%0d writes=%0d bitmap=%064h record=%0d seq=%08h valid=%h owned=%h metadata=%030h delivered=%013h rc_context=%018h rq_context=%011h pending=%0d",
                index,scenario,observing,tag_count,outstanding_reads,remaining_bytes,rq_partial,rc_partial,requests,completed_reads,posted_writes,allocated_tags,
                expected_record_valid,expected_sequence,expected_valid,expected_owned,expected_data,expected_delivered,
                {dut.rc_seen,dut.rc_expected,dut.rc_tag,dut.rc_terminal,dut.rc_normal,dut.rc_owned,dut.rc_original,dut.rc_delivered,dut.rc_address},
                {dut.rq_seen,dut.rq_expected,dut.rq_first,dut.rq_last,dut.rq_count,dut.rq_write,dut.rq_aborted},expected_pending);
        end
    endtask

    integer expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites;
    reg [255:0] expected_bitmap;
    string vector_file;
    initial begin
        if (!$value$plusargs("VECTORS=%s",vector_file)) $fatal(1,"missing VECTORS");
        fd=$fopen(vector_file,"r"); if (!fd) $fatal(1,"cannot open vectors");
        #200;
        $display("SCHED_BINARY_SCOPE rows=22330 scenarios=67 startup_ns=200 sample_ns=1 native=0");
        while (!$feof(fd)) begin
            @(negedge clk);
            fields=$fscanf(fd,"%d %d %d %d %d %d %h %h %d %h %b %h %d %d %b %h %d %d %d %d %d %d %d %d %d %d %d %h\n",
                scenario,native_ps,cold_rst_n,profile_valid,user_reset,link_up,requester_id,attribute_mask,
                rq_valid,rq_ready,rq_data,rq_user,rc_valid,rc_ready,rc_data,rc_user,
                expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites,expected_bitmap);
            if (fields==-1 && $feof(fd)) break;
            if (fields!=28) $fatal(1,"bad vector fields=%0d row=%0d",fields,index);
            #1; capture_scheduler_expectation();
            @(posedge clk); #1;
            check_scheduler();
            if (scenario!=last_scenario) begin cases=cases+1;last_scenario=scenario;end
            $display("PCIE_INV_ROW index=%0d scenario=%0d native_ps=%0d observing=%0d fault=%04h tags=%0d reads=%0d bytes=%0d rq_partial=%0d rc_partial=%0d requests=%0d completed=%0d writes=%0d bitmap=%064h empty=%0d",
                index,scenario,native_ps,observing,fault_sources,tag_count,outstanding_reads,remaining_bytes,rq_partial,rc_partial,requests,completed_reads,posted_writes,allocated_tags,inventory_empty);
            if (observing!==expected_observing[0] || protocol_fault!==expected_fault[0])
                $fatal(1,"state/fault mismatch row=%0d expected_fault=%0d",index,expected_fault);
            if (protocol_fault && inventory_empty) $fatal(1,"faulty inventory presented empty");
            if (check_state && (tag_count!==etags[8:0] || outstanding_reads!==ereads[8:0] || remaining_bytes!==ebytes[20:0] ||
                rq_partial!==erp[0] || rc_partial!==ecp[0] || requests!==erequests[31:0] ||
                completed_reads!==ecompleted[31:0] || posted_writes!==ewrites[31:0] || allocated_tags!==expected_bitmap))
                $fatal(1,"inventory mismatch row=%0d expected_tags=%0d reads=%0d bytes=%0d requests=%0d completed=%0d writes=%0d",index,etags,ereads,ebytes,erequests,ecompleted,ewrites);
            index=index+1;
        end
        $fclose(fd);
        $display("PCIE_INV_SUITE_PASS rows=%0d scenarios=%0d native_endpoint=0 native_trace_replayed=1 global_drain=0",index,cases);
        $display("SCHED_SUITE_PASS rows=%0d scenarios=%0d mode=%0d",index,cases,scheduler_mode);
        $finish;
    end
    initial begin #200000; $fatal(1,"vector bench watchdog"); end
endmodule
