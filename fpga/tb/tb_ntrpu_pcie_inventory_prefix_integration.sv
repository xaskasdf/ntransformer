`timescale 1ns / 1ps
module tb_ntrpu_pcie_inventory_prefix_integration;
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
    integer join_mode=0,tag_q,words=0,retired_rows=0,canceled_rows=0,revocations=0,empty_rows=0;
    initial begin
        if ($test$plusargs("OMIT_RQ")) join_mode=1;
        if ($test$plusargs("OMIT_RC")) join_mode=2;
        if ($test$plusargs("OMIT_VALIDATION")) join_mode=3;
        if ($test$plusargs("FALSE_EMPTY")) join_mode=4;
        if ($test$plusargs("WRONG_SEQUENCE")) join_mode=5;
        if ($test$plusargs("OMIT_COLD_RESET")) join_mode=6;
        if ($test$plusargs("CORRUPT_METADATA")) join_mode=7;
        $display("JOIN_MODE mode=%0d",join_mode);
    end
    wire j_observing,j_rq_partial,j_rc_partial,j_fault,j_empty,j_pending,j_valid;
    wire [255:0] j_tags;
    wire [8:0] j_tag_count,j_reads;
    wire [20:0] j_bytes;
    wire [31:0] j_requests,j_completed,j_writes,j_sequence;
    wire [15:0] j_faults;
    wire [31:0] sequence_in=32'hffffff80+index;
    ntrpu_pcie_inventory_join joined (
        .clk(clk),.cold_rst_n(join_mode==6 && index>16 ? 1'b1 : cold_rst_n),
        .sequence_in(join_mode==5 && index==128 ? sequence_in-1 : sequence_in),
        .profile_valid(profile_valid),.user_reset(user_reset),.link_up(link_up),
        .requester_id(requester_id),.attribute_mask(attribute_mask),
        .rq_valid(join_mode==1 ? 1'b0 : rq_valid),.rq_ready(rq_ready),
        .rc_valid(join_mode==2 ? 1'b0 : rc_valid),.rc_ready(rc_ready),
        .rq_data(rq_data),.rq_user(rq_user),.rc_data(rc_data),.rc_user(rc_user),
        .observing(j_observing),.allocated_tags(j_tags),.tag_count(j_tag_count),
        .outstanding_reads(j_reads),.remaining_bytes(j_bytes),
        .rq_partial(j_rq_partial),.rc_partial(j_rc_partial),.requests(j_requests),
        .completed_reads(j_completed),.posted_writes(j_writes),
        .fault_sources(j_faults),.protocol_fault(j_fault),.inventory_empty(j_empty),
        .pipeline_pending(j_pending),.retired_valid(j_valid),.retired_sequence(j_sequence)
    );
`ifdef JOIN_RTL_INTERNAL_CHECKS
    initial begin
        #1;
        if (join_mode==3) begin
            $display("JOIN_VALIDATION_OMISSION_INJECTED"); force joined.validation_faults=16'b0;
        end
        if (join_mode==4) begin
            wait(rq_valid===1'b1); #0.1;
            $display("JOIN_FALSE_EMPTY_INJECTED"); force joined.inventory_empty=1'b1;
        end
        if (join_mode==7) begin
            $display("JOIN_METADATA_CORRUPTION_INJECTED"); force joined.metadata_data=120'b0;
        end
    end
`endif
    wire [409:0] reference_state={observing,allocated_tags,tag_count,outstanding_reads,remaining_bytes,
        rq_partial,rc_partial,requests,completed_reads,posted_writes,fault_sources};
    wire [409:0] joined_state={j_observing,j_tags,j_tag_count,j_reads,j_bytes,
        j_rq_partial,j_rc_partial,j_requests,j_completed,j_writes,j_faults};
    reg [409:0] prior_state=0,expected_state;
    reg previous_cold=0,previous_empty=0,pre_observing=0,expected_valid,expected_empty,expected_pending;
    reg [31:0] expected_sequence;
    integer source_index;
    task precheck;
        begin
            pre_observing=observing;
            if (rq_valid!==1'b0 || rc_valid!==1'b0 || cold_rst_n!==1'b1 ||
                profile_valid!==1'b1 || user_reset!==1'b0 || link_up!==1'b1) begin
                if (j_empty!==1'b0) $fatal(1,"JOIN_REVOKE_MISMATCH row=%0d mode=%0d",index,join_mode);
                if (previous_empty && (rq_valid || rc_valid)) revocations=revocations+1;
            end
        end
    endtask
    task check_join;
        begin
            expected_valid=cold_rst_n && previous_cold;
            source_index=expected_valid ? index-1 : -1;
            expected_sequence=expected_valid ? 32'hffffff80+index-1 : 0;
            expected_state=expected_valid ? prior_state : 0;
            if ({j_valid,j_sequence}!=={expected_valid,expected_sequence})
                $fatal(1,"JOIN_SEQUENCE_MISMATCH row=%0d mode=%0d",index,join_mode);
            if (joined_state!==expected_state) begin
                $display("JOIN_STATE_DIAGNOSTIC actual=%h expected=%h",joined_state,expected_state);
                $fatal(1,"JOIN_STATE_MISMATCH row=%0d mode=%0d",index,join_mode);
            end
            if (j_fault!==( |j_faults)) $fatal(1,"JOIN_FAULT_MISMATCH row=%0d mode=%0d",index,join_mode);
            expected_pending=cold_rst_n && ((tag_count!=0 || outstanding_reads!=0 || remaining_bytes!=0 ||
                rq_partial || rc_partial || dut.rq_held || dut.rc_held || rq_valid || rc_valid)!==1'b0 ||
                (pre_observing && (rq_valid!==1'b0 || rc_valid!==1'b0)));
            if (j_pending!==expected_pending) $fatal(1,"JOIN_PENDING_MISMATCH row=%0d mode=%0d",index,join_mode);
            // Named fields avoid relying on packed offsets for public obligations.
            expected_empty=expected_valid && (inventory_empty===1'b1) &&
                (requester_id===dut.saved_requester) && (attribute_mask===dut.saved_attribute_mask) &&
                ((rq_ready===4'b0000) || (rq_ready===4'b1111)) &&
                ((rc_ready===1'b0) || (rc_ready===1'b1)) && !expected_pending &&
                j_observing && j_tag_count===9'b0 && j_reads===9'b0 && j_bytes===21'b0 &&
                j_rq_partial===1'b0 && j_rc_partial===1'b0 && j_faults===16'b0;
            if (j_empty!==expected_empty) $fatal(1,"JOIN_EMPTY_MISMATCH row=%0d mode=%0d",index,join_mode);
            words=0;
`ifdef JOIN_RTL_INTERNAL_CHECKS
            if (join_mode==0 && index<544) begin
                for (tag_q=0;tag_q<256;tag_q=tag_q+1) begin
                    if ({joined.producer.attributes[tag_q],joined.producer.traffic_class[tag_q],
                        joined.producer.address_next[tag_q],joined.producer.remaining[tag_q]} !==
                        {dut.attributes[tag_q],dut.traffic_class[tag_q],dut.address_next[tag_q],dut.remaining[tag_q]})
                        $fatal(1,"JOIN_TABLE_MISMATCH row=%0d mode=%0d",index,join_mode);
                    words=words+1;
                end
            end
`endif
            if (expected_valid) retired_rows=retired_rows+1;
            if (!cold_rst_n && previous_cold) begin
                canceled_rows=canceled_rows+1;
                $display("JOIN_CANCELED index=%0d source=%0d seq=%08h",index,index-1,32'hffffff80+index-1);
            end
            if (j_empty) empty_rows=empty_rows+1;
            $display("JOIN_ROW index=%0d source=%0d mode=%0d words=%0d valid=%0d seq=%08h observing=%0d fault=%04h tags=%0d reads=%0d bytes=%0d rq_partial=%0d rc_partial=%0d requests=%0d completed=%0d writes=%0d bitmap=%064h empty=%0d pending=%0d",
                index,source_index,join_mode,words,j_valid,j_sequence,j_observing,j_faults,j_tag_count,j_reads,j_bytes,
                j_rq_partial,j_rc_partial,j_requests,j_completed,j_writes,j_tags,j_empty,j_pending);
            prior_state=reference_state; previous_cold=cold_rst_n; previous_empty=j_empty;
        end
    endtask
    integer expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites;
    reg [255:0] expected_bitmap;
    string vector_file;
    initial begin
        if (!$value$plusargs("VECTORS=%s",vector_file)) $fatal(1,"missing VECTORS");
        fd=$fopen(vector_file,"r"); if (!fd) $fatal(1,"cannot open vectors");
        while (!$feof(fd)) begin
            @(negedge clk);
            fields=$fscanf(fd,"%d %d %d %d %d %d %h %h %d %h %b %h %d %d %b %h %d %d %d %d %d %d %d %d %d %d %d %h\n",
                scenario,native_ps,cold_rst_n,profile_valid,user_reset,link_up,requester_id,attribute_mask,
                rq_valid,rq_ready,rq_data,rq_user,rc_valid,rc_ready,rc_data,rc_user,
                expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites,expected_bitmap);
            if (fields==-1 && $feof(fd)) break;
            if (fields!=28) $fatal(1,"bad vector fields=%0d row=%0d",fields,index);
            #1; precheck();
            @(posedge clk); #0.001;
            check_join();
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
        $display("PCIE_INV_SUITE_PASS rows=%0d scenarios=%0d native_endpoint=0 native_trace_replayed=0 global_drain=0",index,cases);

        // Complete publication of the last source row without adding protocol rows.
        cold_rst_n=1; rq_valid=0; rc_valid=0;
        #1; precheck();
        @(posedge clk); #0.001; check_join();
        $display("JOIN_SUITE_PASS rows=%0d scenarios=%0d mode=%0d retired=%0d canceled=%0d revocations=%0d empty=%0d",index,cases,join_mode,retired_rows,canceled_rows,revocations,empty_rows);
        $finish;
    end
    initial begin #200000; $fatal(1,"vector bench watchdog"); end
endmodule
