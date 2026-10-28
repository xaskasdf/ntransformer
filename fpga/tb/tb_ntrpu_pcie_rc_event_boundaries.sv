`timescale 1ns / 1ps
module tb_ntrpu_pcie_rc_event_boundaries;
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
            if (phase!=last_phase) begin cases=cases+1;last_phase=phase;end
            index=index+1;
        end
        $fclose(fd);
        $display("RC_BOUNDARY_SUITE_COMPLETE rows=%0d phases=%0d latency=2 native=0",index,cases);
        $finish;
    end
    initial begin #20000;$fatal(1,"boundary watchdog");end
endmodule
