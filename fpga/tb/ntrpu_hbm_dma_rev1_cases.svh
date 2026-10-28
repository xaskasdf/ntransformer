    // Inserted into the frozen donor bench by prepare_ntrpu_hbm_dma_rev1.py.
    // Source-synchronous stop; cold reset is externally coordinated in this bench.
    task automatic check_stop_crossing;
        begin
            repeat(2)@(posedge dma_clk);
            #0.001;
            if(dut.lanes[0].bridge.stop_d2!==1'b1 || dut.lanes[1].bridge.stop_d2!==1'b1)
                $fatal(1,"REV1 STOP_LATENCY");
            repeat(6)begin
                @(posedge dma_clk);#0.001;
                if(dsc_load!==0)$fatal(1,"REV1 DESCRIPTOR_AFTER_STOP");
            end
        end
    endtask
    task automatic pulse_source_stop;
        begin
            @(negedge clk);stop=1;
            @(posedge clk);#0.001;
            if(dut.lanes[0].bridge.stop_source!==1'b1 || dut.lanes[1].bridge.stop_source!==1'b1)
                $fatal(1,"REV1 SOURCE_STOP_NOT_CAPTURED");
            fork
                begin @(negedge clk);stop=0;end
                check_stop_crossing();
            join
        end
    endtask
    task automatic revision_tests;
        request_t x,y;
        reg [511:0] offered;
        begin
            reset_case(25);
            x=request_for(0,41,64,0);y=request_for(1,42,64,0);
            reserve(0,x,0);reserve(1,y,0);packet(0,x);packet(1,y);
            pulse_source_stop();
            @(negedge dma_clk);dma_run=3;dsc_allow=3;axis_allow=3;
            check_quarantine(2);
            if(descriptors[0] || descriptors[1] || packets[0] || packets[1])
                $fatal(1,"REV1 PENDING_DESCRIPTOR_ESCAPED");
            passed("source_stop_pending_both_lanes");

            reset_case(26);x=request_for(0,51,128,0);reserve(0,x,0);
            dma_run=3;dsc_allow=3;axis_allow=0;beat(0,x,0,0);
            wait(axis_valid[0]);offered=axis_data[511:0];
            pulse_source_stop();
            if(!axis_valid[0] || axis_data[511:0]!==offered)$fatal(1,"REV1 STOP_LOST_HELD_BEAT");
            @(negedge dma_clk);axis_allow=1;
            wait(beat_index[0]==1);beat(0,x,64,1);wait(packets[0]==1);
            done_pulses(1,1);check_quarantine(1);
            if(descriptors[0]!=1 || retired[0]!=1)$fatal(1,"REV1 STOP_OWNERSHIP");
            passed("source_stop_preserves_and_drains_offer");

            reset_case(27);x=request_for(1,61,128,0);reserve(1,x,0);
            @(negedge clk);data_context[2047:1024]=x;data_valid[1]=1;data_last[1]=1;
            @(posedge clk);#0.001;
            if(halted[1]!==1'b1 || dut.lanes[1].bridge.stop_source!==1'b0)
                $fatal(1,"REV1 LOCAL_FAULT_CAPTURE");
            @(negedge clk);data_valid=0;
            @(posedge clk);#0.001;
            if(dut.lanes[0].bridge.stop_source!==1'b1 || dut.lanes[1].bridge.stop_source!==1'b1)
                $fatal(1,"REV1 LOCAL_FAULT_NOT_REGISTERED");
            check_stop_crossing();
            dma_run=3;dsc_allow=3;check_quarantine(1);
            if(descriptors[0] || descriptors[1])$fatal(1,"REV1 FAULT_DESCRIPTOR_ESCAPED");
            passed("local_fault_registered_before_crossing");

            // Old RAM bits from cases 25/26 survive; reset validity must hide them.
            reset_case(28);dma_run=3;dsc_allow=3;axis_allow=3;
            repeat(12)@(negedge clk);
            if(axis_valid || result_valid || descriptors[0] || descriptors[1] || contexts_held)
                $fatal(1,"REV1 STALE_RAM_VISIBLE");
            x=request_for(0,73,64,0);reserve(0,x,0);packet(0,x);local_finish(0,x);
            wait(wire_sent[0]);done_pulses(1,1);acknowledge(0,x);
            wait(result_valid[0]);
            if(result_status[3:0]!=0 || result_bytes[63:0]!=64)$fatal(1,"REV1 RESET_RESULT");
            @(negedge clk);result_ready=1;wait(retired[0]==1);repeat(4)@(negedge clk);
            if(contexts_held || halted || dma_fault || packets[0]!=1 || packets[1])
                $fatal(1,"REV1 COLD_RESTART_FAILED");
            passed("cold_reset_hides_stale_ram_and_restarts");
            $display("BR REV1_ALL_PASS 4");
        end
    endtask
