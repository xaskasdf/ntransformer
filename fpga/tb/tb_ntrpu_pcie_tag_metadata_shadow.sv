`timescale 1ns / 1ps
module tb_ntrpu_pcie_tag_metadata_shadow;
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

    // Shadow only: R7 remains the protocol reference and does not consume this result.
    reg [5:0] metadata_write_valid, metadata_write_full;
    reg [47:0] metadata_write_tags;
    reg [179:0] metadata_write_data;
    reg [3:0] metadata_lookup_valid;
    reg [31:0] metadata_lookup_tags;
    reg [11:0] metadata_lookup_cutoffs;
    wire [3:0] metadata_result_valid, metadata_result_error;
    wire [119:0] metadata_result_data;
    reg [5:0] applied_valid, applied_full;
    reg [47:0] applied_tags;
    reg [179:0] applied_data;
    integer shadow_mode=0;
    initial begin
        if ($test$plusargs("OMIT_RQ")) shadow_mode=1;
        if ($test$plusargs("REVERSE_WRITES")) shadow_mode=2;
        $display("META_SHADOW_MODE mode=%0d",shadow_mode);
    end
    ntrpu_pcie_tag_metadata metadata_bank (
        .clk(clk),.cold_rst_n(cold_rst_n),
        .lookup_valid(metadata_lookup_valid),.lookup_tags(metadata_lookup_tags),
        .lookup_cutoffs(metadata_lookup_cutoffs),.write_valid(applied_valid),
        .write_full(applied_full),.write_tags(applied_tags),.write_data(applied_data),
        .result_valid(metadata_result_valid),.result_error(metadata_result_error),
        .result_data(metadata_result_data)
    );
    integer position, candidate, best, query, preceding, reverse_slot;
    reg [3:0] used_rc;
    always @* begin
        metadata_write_valid=0; metadata_write_full=6'b110000;
        metadata_write_tags=0; metadata_write_data=0; used_rc=0; best=-1;
        // R7 resolves malformed EOP order by lane, not by record slot number.
        for (position=0;position<4;position=position+1) begin
            best=-1;
            for (candidate=0;candidate<4;candidate=candidate+1)
                if (dut.rc_update_valid[candidate] && !used_rc[candidate] &&
                    (best<0 || dut.rc_update_lane[candidate]<dut.rc_update_lane[best])) best=candidate;
            if (best>=0) begin
                used_rc[best]=1;
                metadata_write_valid[position]=1;
                metadata_write_tags[position*8+:8]=dut.rc_update_tag[best];
                metadata_write_data[position*30+:30]={5'b0,dut.rc_update_address[best],dut.rc_update_remaining[best]};
            end
        end
        for (position=0;position<2;position=position+1) begin
            metadata_write_valid[4+position]=dut.rq_update_valid[position];
            metadata_write_tags[(4+position)*8+:8]=dut.rq_update_tag[position];
            metadata_write_data[(4+position)*30+:30]={dut.rq_update_attributes[position],dut.rq_update_tc[position],dut.rq_update_address[position],dut.rq_update_remaining[position]};
        end
        metadata_lookup_valid=0; metadata_lookup_tags=0; metadata_lookup_cutoffs=0;
        for (query=0;query<4;query=query+1) begin
            metadata_lookup_tags[query*8+:8]=rc_data[query*128+64+:8];
            metadata_lookup_valid[query]=((^rc_data[query*128+64+:8])!==1'bx);
            preceding=0;
            for (candidate=0;candidate<4;candidate=candidate+1)
                if (dut.rc_update_valid[candidate] && dut.rc_update_lane[candidate]<query*4) preceding=preceding+1;
            metadata_lookup_cutoffs[query*3+:3]=preceding;
        end
        applied_valid=metadata_write_valid; applied_full=metadata_write_full;
        applied_tags=metadata_write_tags; applied_data=metadata_write_data;
        if (shadow_mode==1) applied_valid[4]=0;
        if (shadow_mode==2) begin
            for (reverse_slot=0;reverse_slot<6;reverse_slot=reverse_slot+1) begin
                applied_valid[reverse_slot]=metadata_write_valid[5-reverse_slot];
                applied_full[reverse_slot]=metadata_write_full[5-reverse_slot];
                applied_tags[reverse_slot*8+:8]=metadata_write_tags[(5-reverse_slot)*8+:8];
                applied_data[reverse_slot*30+:30]=metadata_write_data[(5-reverse_slot)*30+:30];
            end
        end
    end
    integer checked_tag, check_query, check_write;
    reg [7:0] query_tag;
    reg [119:0] expected_lookup_data;
    reg [3:0] expected_lookup_valid;
    // Capture before the edge; compare after NBA updates in the vector loop below.
    task capture_metadata_expectation;
        begin
            expected_lookup_data=0; expected_lookup_valid=0;
            if (cold_rst_n) begin
                expected_lookup_valid=metadata_lookup_valid;
                for (check_query=0;check_query<4;check_query=check_query+1) begin
                    if (metadata_lookup_valid[check_query]) begin
                        query_tag=metadata_lookup_tags[check_query*8+:8];
                        expected_lookup_data[check_query*30+:30]={dut.attributes[query_tag],dut.traffic_class[query_tag],dut.address_next[query_tag],dut.remaining[query_tag]};
                        for (check_write=0;check_write<4;check_write=check_write+1)
                            if (check_write<metadata_lookup_cutoffs[check_query*3+:3] && metadata_write_valid[check_write] && metadata_write_tags[check_write*8+:8]==query_tag)
                                expected_lookup_data[check_query*30+:25]=metadata_write_data[check_write*30+:25];
                    end
                end
            end
        end
    endtask
    task check_metadata_shadow;
        begin
            for (checked_tag=0;checked_tag<256;checked_tag=checked_tag+1)
                if (metadata_bank.metadata[checked_tag]!=={dut.attributes[checked_tag],dut.traffic_class[checked_tag],dut.address_next[checked_tag],dut.remaining[checked_tag]})
                    $fatal(1,"META_SHADOW_MISMATCH row=%0d tag=%0d mode=%0d",index,checked_tag,shadow_mode);
            if (metadata_result_valid!==expected_lookup_valid || metadata_result_error!==4'b0 || metadata_result_data!==expected_lookup_data)
                $fatal(1,"META_SHADOW_LOOKUP_MISMATCH row=%0d mode=%0d",index,shadow_mode);
            $display("META_SHADOW_ROW index=%0d scenario=%0d words=256 queries=4 mode=%0d",index,scenario,shadow_mode);
        end
    endtask

    integer fd,fields,index=0,scenario,native_ps,last_scenario=-1,cases=0;
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
            #1; capture_metadata_expectation();
            @(posedge clk); #0.001;
            check_metadata_shadow();
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
        $display("META_SHADOW_SUITE_PASS rows=%0d scenarios=%0d mode=%0d",index,cases,shadow_mode);
        $finish;
    end
    initial begin #200000; $fatal(1,"vector bench watchdog"); end
endmodule
