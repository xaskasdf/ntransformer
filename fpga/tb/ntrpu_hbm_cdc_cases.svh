        reset_case(18);
        submit_valid=1;submit_desc[231:0]=descriptor(180,9);
        repeat(20)begin step;need(!started&&submit_ready==0&&!fetch_valid&&ha[0]==0,"BOOT_CLOSED");end
        init_complete_async=1;
        repeat(20)begin step;need(!started&&submitted[0]==0,"BOOT_NEEDS_BOTH_STACKS");end
        init_complete_async=2;
        repeat(20)begin step;need(!started&&submitted[0]==0,"BOOT_NEEDS_SAME_EPOCH");end
        $display("CD BEFORE_INIT %0d",stamp());init_complete_async=3;
        begin: startup_wait
            for(integer j=0;j<100;j++)begin
                step;if(submitted[0]==1)begin submit_valid=0;disable startup_wait;end
            end
            need(0,"BOOT_ADMISSION_TIMEOUT");
        end
        $display("CD STARTED %0d",stamp());drain_all;pass_case;

        reset_case(19);hold_b=3;
        push_one(0,190,64);push_one(1,191,9);
        repeat(300)step;need(inflight!=0&&cpls[0]==0&&cpls[1]==0,"PAUSE_HAS_WORK");
        @(negedge mem_clk);mem_pause=1;step;stop=1;step;stop=0;
        $display("CD PAUSE %0d %0d %0d",stamp(),ha[0],ha[1]);
        repeat(60)begin step;need(!drained&&submit_ready==0&&cpls[0]==0&&cpls[1]==0,"PAUSE_RETAINS_WORK");end
        $display("CD RESUME %0d %0d %0d",stamp(),ha[0],ha[1]);
        mem_pause=0;hold_b=0;drain_all;pass_case;

        reset_case(20);hold_b=1;push_one(0,200,8);repeat(200)step;
        need(inflight!=0&&!drained,"LOSS_HAS_WORK");
        init_complete_async=1;repeat(20)step;
        need(started&&fault&&submit_ready==0&&!drained&&mem_rst_n,"LOSS_LATCHES_FAULT_WITHOUT_RESET");
        $display("CD LOST %0d",stamp());init_complete_async=3;hold_b=0;
        wait_cpl(0,1);repeat(20)step;
        need(fault&&submit_ready==0&&drained&&inflight==0&&mem_rst_n,"LOSS_DRAINS_BUT_STAYS_CLOSED");
        $display("CD LOSS_DRAINED %0d",stamp());pass_case;

        mem_half=4.013;
        reset_case(21);
        for(integer j=0;j<16;j++)push_one(j%2,210+j,j%4==0?1:j%4==1?8:j%4==2?9:64);
        drain_all;$display("CD SLOW_MEMORY %0d",stamp());pass_case;
        $display("CD ALL_PASS cases=21 time_ps=%0d",stamp());$finish;
