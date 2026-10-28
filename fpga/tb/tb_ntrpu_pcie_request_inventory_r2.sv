`timescale 1ns / 1ps
module tb_ntrpu_pcie_request_inventory;
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
            @(posedge clk); #0.001;
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
        $finish;
    end
    initial begin #200000; $fatal(1,"vector bench watchdog"); end
endmodule
