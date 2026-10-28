`timescale 1ns / 1ps
module tb_ntrpu_pcie_rc_event_stage_binary;
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
    integer mode=0;
    integer expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites;
    reg [255:0] expected_bitmap;
    string vector_file;

    reg [3:0] accepted_headers=0;
    wire [3:0] header_seen, proposal_valid, length_error, terminal_error;
    wire [383:0] captured_headers;
    wire [51:0] delivered_bytes, proposed_remaining;
    wire [47:0] proposed_address;
    ntrpu_pcie_rc_proposal_reference proposal_unit (
        .clk(clk),.cold_rst_n(cold_rst_n),.header_valid(accepted_headers),.header_data(rc_data),
        .header_seen(header_seen),.proposal_valid(proposal_valid),.captured_headers(captured_headers),
        .delivered_bytes(delivered_bytes),.proposed_remaining(proposed_remaining),
        .proposed_address(proposed_address),.length_error(length_error),.terminal_error(terminal_error)
    );
    integer starts,slot,position;
    always @* begin
        accepted_headers=0;starts=-1;position=0;
        case (rc_user[67:64])
            0:starts=0;1:starts=1;3:starts=2;7:starts=3;15:starts=4;
            default:starts=-1;
        endcase
        if (cold_rst_n && observing && rc_valid===1'b1 && rc_ready===1'b1)
            for (slot=0;slot<4;slot=slot+1)
                if (slot<starts) begin position=rc_user[68+slot*2+:2];accepted_headers[position]=1;end
    end
    reg cap_accept;
    reg [3:0] cap_headers, cap_owned, reference_valid, reference_complete;
    reg [51:0] expected_original, reference_remaining, reference_removed;
    reg [47:0] expected_address, reference_address;
    reg [31:0] reference_tags;
    reg [119:0] cap_metadata=0;
    integer q,k,newest_lane;
    reg [7:0] lookup_tag;
    task capture_reference;
        begin
            cap_accept=cold_rst_n && observing && rc_valid===1'b1 && rc_ready===1'b1;
            cap_headers=accepted_headers;cap_owned=0;expected_original=0;expected_address=0;cap_metadata=0;
            reference_valid=0;reference_complete=0;reference_remaining=0;
            reference_removed=0;reference_address=0;reference_tags=0;
            if (cold_rst_n) begin
                reference_valid=dut.rc_update_valid;reference_complete=dut.rc_complete_events;
                for (k=0;k<4;k=k+1) begin
                    reference_tags[k*8+:8]=dut.rc_update_tag[k];
                    reference_remaining[k*13+:13]=dut.rc_update_remaining[k];
                    reference_address[k*12+:12]=dut.rc_update_address[k];
                    reference_removed[k*13+:13]=dut.rc_removed_span[k];
                end
            end
            for (q=0;q<4;q=q+1) begin
                if (cap_headers[q]) begin
                    lookup_tag=rc_data[q*128+64+:8];
                    cap_owned[q]=dut.allocated_tags[lookup_tag];
                    expected_original[q*13+:13]=dut.remaining[lookup_tag];
                    expected_address[q*12+:12]=dut.address_next[lookup_tag];
                    newest_lane=-1;
                    for (k=0;k<4;k=k+1) begin
                        if (k<q && dut.rc_release_valid[k] && dut.rc_release_tag[k]==lookup_tag) cap_owned[q]=0;
                        if (dut.rc_update_valid[k] && dut.rc_update_tag[k]==lookup_tag && dut.rc_update_lane[k]<q*4 && (newest_lane<0 || dut.rc_update_lane[k]>newest_lane)) begin
                            newest_lane=dut.rc_update_lane[k];
                            expected_original[q*13+:13]=dut.rc_update_remaining[k];
                            expected_address[q*12+:12]=dut.rc_update_address[k];
                        end
                    end
                    cap_metadata[q*30+:30]={dut.attributes[lookup_tag],dut.traffic_class[lookup_tag],expected_address[q*12+:12],expected_original[q*13+:13]};
                end
            end
        end
    endtask
    // Simulation-only retained proposal context, independent of future tag reuse.
    reg context_active=0,context_owned=0,context_terminal=0,context_normal=0,context_ready=0;
    reg [7:0] context_tag=0;
    reg [12:0] context_original=0,context_delivered=0,context_remaining=0;
    reg [11:0] context_address=0;
    reg [3:0] candidate_valid,candidate_complete,unresolved;
    reg [31:0] candidate_tags;
    reg [51:0] candidate_remaining,candidate_removed;
    reg [47:0] candidate_address;
    reg [15:0] end_map;
    integer end_slot[0:15];
    integer ends,lane,n;
    task check_proposal_events;
        begin
            candidate_valid=0;candidate_complete=0;unresolved=0;
            candidate_tags=0;candidate_remaining=0;candidate_removed=0;candidate_address=0;
            if (!cold_rst_n) begin
                context_active=0;context_owned=0;context_terminal=0;context_normal=0;context_ready=0;
                context_tag=0;context_original=0;context_delivered=0;context_remaining=0;context_address=0;
            end else if (cap_accept) begin
                end_map=0;ends=-1;
                for (n=0;n<16;n=n+1) end_slot[n]=0;
                case (rc_user[79:76])
                    0:ends=0;1:ends=1;3:ends=2;7:ends=3;15:ends=4;
                    default:ends=-1;
                endcase
                for (n=0;n<4;n=n+1)
                    if (n<ends) begin end_map[rc_user[80+n*4+:4]]=1;end_slot[rc_user[80+n*4+:4]]=n;end
                if (rc_user[96]) context_normal=0;
                for (lane=0;lane<16;lane=lane+1) begin
                    if (lane%4==0 && header_seen[lane/4]) begin
                        q=lane/4;context_active=1;context_owned=cap_owned[q];
                        context_tag=captured_headers[q*96+64+:8];
                        context_original=captured_headers[q*96+16+:13];
                        context_terminal=captured_headers[q*96+30];
                        context_normal=proposal_valid[q];context_ready=proposal_valid[q];
                        context_delivered=delivered_bytes[q*13+:13];
                        context_remaining=proposed_remaining[q*13+:13];
                        context_address=proposed_address[q*12+:12];
                    end
                    if (end_map[lane]) begin
                        n=end_slot[lane];
                        if (context_active && context_owned) begin
                            if (!context_ready) unresolved[n]=1;
                            else if (context_terminal) begin
                                candidate_complete[n]=1;
                                candidate_removed[n*13+:13]=context_original;
                            end else if (context_normal) begin
                                candidate_valid[n]=1;candidate_tags[n*8+:8]=context_tag;
                                candidate_remaining[n*13+:13]=context_remaining;
                                candidate_address[n*12+:12]=context_address;
                                candidate_removed[n*13+:13]=context_delivered;
                            end
                        end
                        context_active=0;context_owned=0;
                    end
                end
            end
            $display("RC_PROP_ROW index=%0d scenario=%0d seen=%h valid=%h headers=%h delivered=%h remaining=%h address=%h length=%h terminal=%h",index,scenario,header_seen,proposal_valid,captured_headers,delivered_bytes,proposed_remaining,proposed_address,length_error,terminal_error);
            $display("RC_PROP_CONTEXT index=%0d owned=%h original=%h address=%h",index,cap_owned,expected_original,expected_address);
            $display("RC_PROP_EVENTS index=%0d valid=%h complete=%h tags=%h remaining=%h address=%h removed=%h unresolved=%h ref_valid=%h ref_complete=%h ref_tags=%h ref_remaining=%h ref_address=%h ref_removed=%h",index,candidate_valid,candidate_complete,candidate_tags,candidate_remaining,candidate_address,candidate_removed,unresolved,reference_valid,reference_complete,reference_tags,reference_remaining,reference_address,reference_removed);
            if (header_seen!==cap_headers) $fatal(1,"Proposal capture mask differs row=%0d seen=%h expected=%h reset=%b accept=%b",index,header_seen,cap_headers,cold_rst_n,cap_accept);
            if (!protocol_fault && ({candidate_valid,candidate_complete,candidate_tags,candidate_remaining,candidate_address,candidate_removed}!=={reference_valid,reference_complete,reference_tags,reference_remaining,reference_address,reference_removed} || unresolved!=0))
                $fatal(1,"Healthy proposal events diverged row=%0d",index);
        end
    endtask

    reg [3:0] expected_valid_delayed=0;
    reg [119:0] expected_metadata_delayed=0;
    always @(posedge clk) begin
        if (!cold_rst_n) begin expected_valid_delayed<=0;expected_metadata_delayed<=0;end
        else begin expected_valid_delayed<=cap_headers;expected_metadata_delayed<=cap_metadata;end
    end
    wire stage_result_valid,stage_active,stage_owned,stage_loss,stage_pending;
    wire [31:0] stage_sequence,stage_tags;
    wire [3:0] stage_headers,stage_updates,stage_complete,stage_unresolved,stage_checked;
    wire [63:0] stage_header_faults;
    wire [15:0] stage_beat_faults,stage_faults,stage_lanes;
    wire [51:0] stage_remaining,stage_removed;
    wire [47:0] stage_address;
    wire [127:0] stage_origins;
    wire [31:0] input_sequence=32'hffffff00+index;
    ntrpu_pcie_rc_event_stage stage_dut (
        .clk(clk),.cold_rst_n(cold_rst_n),.observing(observing),
        .rc_valid(rc_valid && !(mode==1 && cap_headers!=0)),.rc_ready(rc_ready),
        .rc_data(rc_data),.rc_user(rc_user),.header_owned(cap_owned),
        .expected_requester(dut.saved_requester),.sequence_in(input_sequence),
        .expected_valid(mode==2 ? 4'b0 : expected_valid_delayed),.expected_metadata(expected_metadata_delayed),
        .result_valid(stage_result_valid),.sequence_out(stage_sequence),.result_headers(stage_headers),
        .header_faults(stage_header_faults),.beat_faults(stage_beat_faults),.fault_latched(stage_faults),
        .update_valid(stage_updates),.complete(stage_complete),.unresolved(stage_unresolved),.packet_checked(stage_checked),
        .event_tags(stage_tags),.event_remaining(stage_remaining),.event_removed(stage_removed),
        .event_address(stage_address),.event_lanes(stage_lanes),.event_origins(stage_origins),
        .context_active(stage_active),.context_owned(stage_owned),.unresolved_latched(stage_loss),.pipeline_pending(stage_pending)
    );
    integer previous_index=-1;
    reg previous_accept=0,previous_fault=0,previous_active=0,previous_owned=0;
    reg [3:0] previous_headers=0,previous_updates=0,previous_complete=0,previous_unresolved=0;
    reg [31:0] previous_sequence=0,previous_tags=0;
    reg [51:0] previous_remaining=0,previous_removed=0;
    reg [47:0] previous_address=0;
    reg [31:0] masked_tags;
    integer check_slot;
    task check_stage_result;
        begin
            $display("RC_STAGE_ROW tick=%0d source=%0d sequence=%h accepted=%b headers=%h update=%h complete=%h unresolved=%h checked=%h tags=%h remaining=%h address=%h removed=%h lanes=%h origins=%h active=%b owned=%b loss=%b beat_faults=%h faults=%h pending=%b h_faults=%h",index,previous_index,stage_sequence,stage_result_valid,stage_headers,stage_updates,stage_complete,stage_unresolved,stage_checked,stage_tags,stage_remaining,stage_address,stage_removed,stage_lanes,stage_origins,stage_active,stage_owned,stage_loss,stage_beat_faults,stage_faults,stage_pending,stage_header_faults);
            if (!cold_rst_n) begin
                if ({stage_result_valid,stage_sequence,stage_headers,stage_updates,stage_complete,stage_unresolved,stage_checked,stage_tags,stage_remaining,stage_address,stage_removed,stage_lanes,stage_origins,stage_active,stage_owned,stage_loss,stage_beat_faults,stage_faults,stage_header_faults}!==0)
                    $fatal(1,"RC_STAGE_COLD_RESET_MISMATCH tick=%0d",index);
            end else begin
                masked_tags=0;
                for (check_slot=0;check_slot<4;check_slot=check_slot+1)
                    if (stage_updates[check_slot]) masked_tags[check_slot*8+:8]=stage_tags[check_slot*8+:8];
                if (stage_result_valid!==previous_accept || stage_headers!==previous_headers || stage_sequence!==(previous_accept ? previous_sequence : 32'b0) ||
                    {stage_updates,stage_complete,stage_unresolved,masked_tags,stage_remaining,stage_address,stage_removed}!==
                    {previous_updates,previous_complete,previous_unresolved,previous_tags,previous_remaining,previous_address,previous_removed})
                    $fatal(1,"RC_STAGE_EVENT_MISMATCH tick=%0d source=%0d mode=%0d",index,previous_index,mode);
                if (stage_active!==previous_active || stage_owned!==previous_owned)
                    $fatal(1,"RC_STAGE_CONTEXT_MISMATCH tick=%0d source=%0d mode=%0d",index,previous_index,mode);
                if (!previous_fault && (stage_faults!=0 || stage_loss || stage_checked!==(previous_updates|previous_complete)))
                    $fatal(1,"RC_STAGE_VALIDATION_MISMATCH tick=%0d source=%0d mode=%0d",index,previous_index,mode);
            end
            if ((stage_active || stage_loss || stage_result_valid || cap_accept) && !stage_pending)
                $fatal(1,"RC_STAGE_FALSE_IDLE tick=%0d",index);
        end
    endtask
    task save_stage_expected;
        begin
            previous_index=index;previous_accept=cap_accept;previous_headers=cap_headers;
            previous_sequence=input_sequence;previous_fault=protocol_fault;
            previous_active=dut.rc_partial;previous_owned=dut.rc_owned;
            previous_updates=candidate_valid;previous_complete=candidate_complete;previous_unresolved=unresolved;
            previous_tags=candidate_tags;previous_remaining=candidate_remaining;
            previous_address=candidate_address;previous_removed=candidate_removed;
            $display("RC_STAGE_EXPECT index=%0d accepted=%b headers=%h owned=%h metadata=%h requester=%h sequence=%h",index,cap_accept,cap_headers,cap_owned,cap_metadata,dut.saved_requester,input_sequence);
        end
    endtask

    initial begin
        if ($test$plusargs("OMIT_RC")) mode=1;
        if ($test$plusargs("MISSING_METADATA")) mode=2;
        $display("RC_STAGE_MODE mode=%0d",mode);
        $display("RC_STAGE_BINARY_SCOPE rows=22330 scenarios=67 startup_ns=200 sample_ns=1 native=0");
        if (!$value$plusargs("VECTORS=%s",vector_file)) $fatal(1,"missing VECTORS");
        fd=$fopen(vector_file,"r"); if (!fd) $fatal(1,"cannot open vectors");
        #200;
        while (!$feof(fd)) begin
            @(negedge clk);
            fields=$fscanf(fd,"%d %d %d %d %d %d %h %h %d %h %b %h %d %d %b %h %d %d %d %d %d %d %d %d %d %d %d %h\n",
                scenario,native_ps,cold_rst_n,profile_valid,user_reset,link_up,requester_id,attribute_mask,
                rq_valid,rq_ready,rq_data,rq_user,rc_valid,rc_ready,rc_data,rc_user,
                expected_observing,expected_fault,check_state,etags,ereads,ebytes,erp,ecp,erequests,ecompleted,ewrites,expected_bitmap);
            if (fields==-1 && $feof(fd)) break;
            if (fields!=28) $fatal(1,"bad vector fields=%0d row=%0d",fields,index);
            #1;capture_reference();
            @(posedge clk); #1;
            check_stage_result();
            check_proposal_events();
            save_stage_expected();
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
        $display("RC_PROP_SHADOW_SUITE_PASS rows=%0d scenarios=%0d",index,cases);
        $display("RC_STAGE_SUITE_PASS rows=%0d scenarios=%0d mode=%0d latency=2",index,cases,mode);
        $finish;
    end
    initial begin #200000; $fatal(1,"vector bench watchdog"); end
endmodule
