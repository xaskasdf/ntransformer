`timescale 1ns / 1ps
module tb_ntrpu_pcie_rc_fragment_matrix;
    reg clk=0;always #2 clk=~clk;
    reg cold_rst_n=0,observing=0,rc_valid=0,rc_ready=1;
    reg [511:0] rc_data=0;
    reg [160:0] rc_user=0;
    reg [3:0] header_owned=0,query_valid=0,expected_valid=0;
    reg [119:0] query_metadata=0,expected_metadata=0;
    reg [15:0] expected_requester=0;
    reg [31:0] sequence_in=0;
    wire result_valid,context_active,context_owned,unresolved_latched,pipeline_pending;
    wire [31:0] sequence_out,event_tags;
    wire [3:0] result_headers,update_valid,complete,unresolved,packet_checked;
    wire [63:0] header_faults;
    wire [15:0] beat_faults,fault_latched,event_lanes;
    wire [51:0] event_remaining,event_removed;
    wire [47:0] event_address;
    wire [127:0] event_origins;
    ntrpu_pcie_rc_event_stage dut(.*);
    wire ref_result_valid;
    wire ref_context_active;
    wire ref_context_owned;
    wire ref_unresolved_latched;
    wire ref_pipeline_pending;
    wire [31:0] ref_sequence_out;
    wire [31:0] ref_event_tags;
    wire [3:0] ref_result_headers;
    wire [3:0] ref_update_valid;
    wire [3:0] ref_complete;
    wire [3:0] ref_unresolved;
    wire [3:0] ref_packet_checked;
    wire [63:0] ref_header_faults;
    wire [15:0] ref_beat_faults;
    wire [15:0] ref_fault_latched;
    wire [15:0] ref_event_lanes;
    wire [51:0] ref_event_remaining;
    wire [51:0] ref_event_removed;
    wire [47:0] ref_event_address;
    wire [127:0] ref_event_origins;
    ntrpu_pcie_rc_event_stage_reference reference_dut (
        .clk(clk),
        .cold_rst_n(cold_rst_n),
        .observing(observing),
        .rc_valid(rc_valid),
        .rc_ready(rc_ready),
        .rc_data(rc_data),
        .rc_user(rc_user),
        .header_owned(header_owned),
        .expected_requester(expected_requester),
        .sequence_in(sequence_in),
        .expected_valid(expected_valid),
        .expected_metadata(expected_metadata),
        .result_valid(ref_result_valid),
        .context_active(ref_context_active),
        .context_owned(ref_context_owned),
        .unresolved_latched(ref_unresolved_latched),
        .pipeline_pending(ref_pipeline_pending),
        .sequence_out(ref_sequence_out),
        .event_tags(ref_event_tags),
        .result_headers(ref_result_headers),
        .update_valid(ref_update_valid),
        .complete(ref_complete),
        .unresolved(ref_unresolved),
        .packet_checked(ref_packet_checked),
        .header_faults(ref_header_faults),
        .beat_faults(ref_beat_faults),
        .fault_latched(ref_fault_latched),
        .event_lanes(ref_event_lanes),
        .event_remaining(ref_event_remaining),
        .event_removed(ref_event_removed),
        .event_address(ref_event_address),
        .event_origins(ref_event_origins)
    );
    always @(posedge clk) begin
        if (!cold_rst_n) begin expected_valid<=0;expected_metadata<=0;end
        else begin expected_valid<=query_valid;expected_metadata<=query_metadata;end
    end
    integer fd,fields,index=0,phase,last_phase=-1,cases=0;
    string path;
    initial begin
        if (!$value$plusargs("VECTORS=%s",path)) $fatal(1,"missing VECTORS");
        fd=$fopen(path,"r");if (!fd) $fatal(1,"cannot open vectors");
        #200;
        while (!$feof(fd)) begin
            @(negedge clk);
            fields=$fscanf(fd,"%d %d %d %d %d %h %h %h %h %h %h %h\n",phase,cold_rst_n,observing,rc_valid,rc_ready,header_owned,expected_requester,sequence_in,query_valid,query_metadata,rc_data,rc_user);
            if (fields==-1 && $feof(fd)) break;
            if (fields!=12) $fatal(1,"bad vector row=%0d fields=%0d",index,fields);
            @(posedge clk);#1;
            $display("RC_BOUNDARY_ROW tick=%0d phase=%0d sequence=%h accepted=%b headers=%h update=%h complete=%h unresolved=%h checked=%h tags=%h remaining=%h address=%h removed=%h lanes=%h origins=%h active=%b owned=%b loss=%b beat_faults=%h faults=%h pending=%b h_faults=%h",index,phase,sequence_out,result_valid,result_headers,update_valid,complete,unresolved,packet_checked,event_tags,event_remaining,event_address,event_removed,event_lanes,event_origins,context_active,context_owned,unresolved_latched,beat_faults,fault_latched,pipeline_pending,header_faults);
            if ({result_valid,context_active,context_owned,unresolved_latched,pipeline_pending,sequence_out,event_tags,result_headers,update_valid,complete,unresolved,packet_checked,header_faults,beat_faults,fault_latched,event_lanes,event_remaining,event_removed,event_address,event_origins}!=={ref_result_valid,ref_context_active,ref_context_owned,ref_unresolved_latched,ref_pipeline_pending,ref_sequence_out,ref_event_tags,ref_result_headers,ref_update_valid,ref_complete,ref_unresolved,ref_packet_checked,ref_header_faults,ref_beat_faults,ref_fault_latched,ref_event_lanes,ref_event_remaining,ref_event_removed,ref_event_address,ref_event_origins}) $fatal(1,"RC_FRAGMENT_REFERENCE_MISMATCH row=%0d",index);
            if (phase!=last_phase) begin cases=cases+1;last_phase=phase;end
            index=index+1;
        end
        $fclose(fd);
        $display("RC_BOUNDARY_SUITE_COMPLETE rows=%0d phases=%0d latency=2 native=0",index,cases);
        $display("RC_FRAGMENT_MATRIX_REFERENCE_PASS rows=%0d native=0",index);
        $finish;
    end
    initial begin #2000000;$fatal(1,"boundary watchdog");end
endmodule
