    // Included only in the frozen dual-instance revision-3 bench.
    task automatic revision3_bounds;
        integer fd,fields,count,wanted;
        reg [2047:0] filename;
        begin
            reset_case(30);count=0;
            if(!$value$plusargs("vectors=%s",filename))$fatal(1,"REV3 MISSING_VECTORS");
            fd=$fopen(filename,"r");if(!fd)$fatal(1,"REV3 VECTOR_OPEN");
            while(!$feof(fd))begin
                @(negedge clk);
                fields=$fscanf(fd,"%h %h %h %d\n",cfg,submit_context,cfg_writeback,wanted);
                if(fields!=4)$fatal(1,"REV3 MALFORMED_VECTOR count=%0d fields=%0d",count,fields);
                cfg_valid=1;
                @(posedge clk);if(!cfg_ready)$fatal(1,"REV3 CONFIG_STALLED");
                @(negedge clk);cfg_valid=0;
                #0.01;
                if(cfg_error || dut.admission_error!==4'(wanted) ||
                   reference_dut.admission_error!==4'(wanted))
                    $fatal(1,"REV3 NUMERIC_ORACLE row=%0d got=%0d reference=%0d expected=%0d",
                        count,dut.admission_error,reference_dut.admission_error,wanted);
                if(contexts_held || dut.watermark || reject_valid || dsc_load)
                    $fatal(1,"REV3 PROBE_ACQUIRED_OWNERSHIP");
                count=count+1;
            end
            $fclose(fd);$display("REV3 BOUNDS_PASS %0d",count);
        end
    endtask

    task automatic revision3_transactions;
        request_t x,y,z;
        reg [259:0] bounds;
        reg [3:0] rejection;
        begin
            // Sustained VALID: consecutive handshakes reserve atomically.
            reset_case(31);x=request_for(0,11,64,0);y=request_for(1,12,64,0);
            @(negedge clk);submit_valid=1;submit_channel=0;submit_context=x;
            expected_seq[0][0]=11;expected_len[0][0]=64;expected_dst[0][0]=cfg.target_base;admitted[0]=1;
            @(posedge clk);if(!submit_ready)$fatal(1,"REV3 FIRST_HANDSHAKE");
            @(negedge clk);submit_channel=1;submit_context=y;
            expected_seq[1][0]=12;expected_len[1][0]=64;expected_dst[1][0]=cfg.target_base+y.target_offset;admitted[1]=1;
            @(posedge clk);if(!submit_ready)$fatal(1,"REV3 BACK_TO_BACK");
            @(negedge clk);submit_valid=0;
            if(contexts_held!=5 || dut.watermark!=12)$fatal(1,"REV3 ATOMIC_OWNERSHIP");
            reserve(0,x,REPLAY);
            z=x;z.identity.sequence_id=13;reserve(1,z,OVERLAP);
            passed("rev3_back_to_back_replay_overlap");

            // Rejected requests consume neither sequence number nor context.
            reset_case(32);x=request_for(0,21,64,0);x.bytes=0;
            @(negedge clk);submit_context=x;submit_channel=0;submit_valid=1;
            @(posedge clk);while(!submit_ready)@(posedge clk);
            @(negedge clk);submit_valid=0;rejection=reject_code;
            if(!reject_valid || rejection!=RANGE)$fatal(1,"REV3 EXPECTED_RANGE");
            repeat(12)begin
                @(negedge clk);submit_context.source_offset=submit_context.source_offset+64;
                if(!reject_valid || reject_code!=rejection || submit_ready || contexts_held || dut.watermark)
                    $fatal(1,"REV3 REJECT_BACKPRESSURE");
            end
            reject_ready=1;@(negedge clk);reject_ready=0;
            x=request_for(0,21,64,0);reserve(0,x,0);
            // A full lane backpressures its third request, without reserving it.
            y=request_for(0,22,64,1);reserve(0,y,0);
            @(negedge clk);submit_context=request_for(0,23,64,2);submit_valid=1;
            repeat(8)begin @(negedge clk);
                if(submit_ready || contexts_held!=2 || dut.watermark!=22)$fatal(1,"REV3 CREDIT_LEAK");
            end
            stop=1;@(negedge clk);submit_valid=0;stop=0;
            check_quarantine(2);passed("rev3_reject_and_credit_backpressure_stop");

            // Cache replacement is atomic; invalid/busy configuration preserves it.
            reset_case(33);bounds={dut.window_end,dut.published_end,dut.source_capacity,dut.target_capacity};
            @(negedge clk);cfg.source.session=0;cfg.window_generation=2;cfg.window_offset=64'hffffffffffffffff;
            cfg_valid=1;
            @(negedge clk);cfg_valid=0;
            if(!cfg_error || {dut.window_end,dut.published_end,dut.source_capacity,dut.target_capacity}!==bounds)
                $fatal(1,"REV3 INVALID_CONFIG_CHANGED_BOUNDS");
            @(negedge clk);cfg.source.session=1;cfg.window_offset=64;cfg.window_bytes=64;
            cfg.published_offset=64;cfg.published_bytes=64;cfg.source_base=64;cfg_valid=1;
            @(negedge clk);cfg_valid=0;
            if(cfg_error || dut.window_end!=128 || dut.published_end!=128 || dut.source_capacity!=65'hffffffffffffffc0)
                $fatal(1,"REV3 VALID_CONFIG_CACHE");
            x=request_for(0,31,64,0);x.window_generation=2;x.source_offset=64;x.target_offset=64;
            reserve(0,x,0);bounds={dut.window_end,dut.published_end,dut.source_capacity,dut.target_capacity};
            @(negedge clk);cfg.window_generation=3;cfg.window_bytes=4096;cfg_valid=1;
            repeat(8)begin @(negedge clk);
                if(cfg_ready || cfg_error || {dut.window_end,dut.published_end,dut.source_capacity,dut.target_capacity}!==bounds)
                    $fatal(1,"REV3 BUSY_CONFIG_CHANGED_BOUNDS");
            end
            cfg_valid=0;passed("rev3_atomic_config_cache");
            // Cold restart must clear cached state before reconfiguration.
            @(negedge clk);cold_reset_n=0;repeat(4)@(negedge clk);
            if({dut.window_end,dut.published_end,dut.source_capacity,dut.target_capacity}!==0)
                $fatal(1,"REV3 COLD_CACHE");
            $display("REV3 TRANSACTIONS_PASS 3");
        end
    endtask
