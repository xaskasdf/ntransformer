    // Observe actual allocator commit/return and internal W, not only occupancy.
    `define RX dut.transport.core.receiver.core
    `define POOL dut.transport.core.receiver.core.allocator.core
    reg [1:0] held_native_w=0;
    reg [577:0] saved_native_w=0;
    integer saturation_windows=0;
    always @(posedge clk) if(rst_n) begin
        for(integer lane=0;lane<2;lane++) begin
            if(`RX.accept_cmd && `RX.cmd_channel==lane)
                $display("NR ALLOC %0d %0d %0d %0d %0d",stamp(),lane,`RX.cmd_tag,`RX.alloc_bank,`RX.alloc_generation);
            if(`RX.finish_w[lane])
                $display("NR LAST %0d %0d %0d %0d",stamp(),lane,`RX.return_bank[lane*3+:3],`RX.return_generation[lane*64+:64]);
            if(`POOL.return_valid[lane]) begin
                need(!`POOL.return_error[lane],"VALID_LEASE_RETURN");
                $display("NR RETURN %0d %0d %0d %0d",stamp(),lane,`POOL.return_bank[lane*3+:3],`POOL.return_generation[lane*64+:64]);
            end
        end
    end
    always @(posedge mem_clk) begin
        if(!mem_rst_n) held_native_w=0;
        else for(integer lane=0;lane<2;lane++) begin
            need(!held_native_w[lane] || (h_wvalid[lane] &&
                {h_wdata[lane*256+:256],h_wstrb[lane*32+:32],h_wlast[lane]}===saved_native_w[lane*289+:289]),"GATED_W_STABLE");
            held_native_w[lane]=h_wvalid[lane]&&!h_wready[lane];
            saved_native_w[lane*289+:289]={h_wdata[lane*256+:256],h_wstrb[lane*32+:32],h_wlast[lane]};
        end
    end
    task automatic pressure_wave(input integer first);
        integer previous0,previous1;
        previous0=cpls[0];previous1=cpls[1];
        need(write_open==0&&banks_used==0,"EMPTY_BEFORE_PRESSURE");
        $display("NR CLOSED %0d %0d",stamp(),scenario);
        fork
            for(integer j=0;j<12;j++)push_one(0,first+2*j,FRAME_BEATS);
            for(integer j=0;j<12;j++)push_one(1,first+2*j+1,FRAME_BEATS);
            begin
                wait(banks_used==8'hff);
                // Complete the currently granted fill; allocation is serialized.
                wait(!`RX.fill_active);
                $display("NR FULL %0d %0d %h",stamp(),scenario,banks_used);
                repeat(100)begin
                    step;need(banks_used==8'hff&&!drained&&!fault,"SATURATED_RETAINED");
                    need(cpls[0]==previous0&&cpls[1]==previous1,"NO_EARLY_COMPLETION");
                end
                saturation_windows++;
                @(negedge mem_clk);write_open=1;
                $display("NR OPEN0 %0d %0d",stamp(),scenario);
                wait(cpls[0]>previous0);repeat(40)step;
                need(cpls[1]==previous1&&!drained,"OTHER_LANE_RETAINED");
                $display("NR ISOLATED %0d %0d %0d %0d",stamp(),scenario,cpls[0]-previous0,cpls[1]-previous1);
                @(negedge mem_clk);write_open=3;
                $display("NR OPEN1 %0d %0d",stamp(),scenario);
            end
        join
        drain_all;
        need(banks_used==0&&pending_counts==0&&inflight==0&&queued==0,"ALL_OWNERS_RETURNED");
        verify_pair(first,12);pass_case;
        $display("NR EMPTY %0d %0d",stamp(),scenario);
    endtask
    initial begin
        #4700;@(negedge apb_clk);apb_rst_n=1;
        step;cold_reset_n=1;$display("NC RELEASE %0d",stamp());
        wait(started);step;need(prestart_samples>0,"QUALIFIED_COLD_START");
        booted=1;$display("NC BOOT %0d %b %b",stamp(),apb_complete_0,apb_complete_1);
        $display("NC THERMAL %0d %b %b",stamp(),DRAM_0_STAT_CATTRIP,DRAM_1_STAT_CATTRIP);
        scenario=1;pressure_wave(100);
        // No reset: second saturation must reuse each bank with a new generation.
        need(drained&&h_wvalid==0,"CLOSE_ONLY_QUIESCENT");
        @(negedge mem_clk);write_open=0;
        scenario=2;pressure_wave(200);
        repeat(20)step;
        need(saturation_windows==2&&peak_banks==8&&used_union==8'hff,"ALL_RESERVES_EXERCISED");
        need(submitted[0]+submitted[1]==48&&cpls[0]+cpls[1]==48&&native_w==48*FRAME_BEATS*2&&native_r==48*FRAME_BEATS*2&&native_aw==48*FRAME_BEATS/8&&native_b==native_aw&&native_ar==native_aw,"FINAL_COUNTS");
        $display("NC COUNTS sub=%0d cpl=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d r_stalls=%0d peak_banks=%0d used_union=%0d prestart=%0d thermal_unknown=%0d",submitted[0]+submitted[1],cpls[0]+cpls[1],native_aw,native_w,native_b,native_ar,native_r,native_r_stalls,peak_banks,used_union,prestart_samples,thermal_unknown);
        $display("NR ALL_PASS cases=2 time_ps=%0d",stamp());$finish;
    end
    initial begin #100000;$fatal(1,"RESERVE_TRANSPORT_WATCHDOG");end
    `undef POOL
    `undef RX
