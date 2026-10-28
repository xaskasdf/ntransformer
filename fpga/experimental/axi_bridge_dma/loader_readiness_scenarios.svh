// Included inside the existing sequential loader campaign, before its PASS.
        // Exercise both incomplete APB combinations, then full availability.
        reset_all;
        @(negedge clk);cold_rst_n=0;mem_cold_rst_n=0;bus_rst_n=0;
        fabric_locked=0;apb_complete=0;ticks(8);
        cold_rst_n=1;mem_cold_rst_n=1;bus_rst_n=1;fabric_locked=1;apb_complete=1;
        ticks(40);require(ready_mgmt==1&&!memory_ready_async&&!dut.available,"partial stack zero unavailable");
        @(negedge mgmt_clk);apb_complete=2;
        ticks(40);require(ready_mgmt==2&&!memory_ready_async&&!dut.available,"partial stack one unavailable");
        @(negedge mgmt_clk);apb_complete=3;
        ticks(40);require(memory_ready_async&&dut.available,"complete startup propagated");
        group_pass("registered_staggered_startup");

        // A stopped management clock holds the source; cold reset still clears
        // it asynchronously. Availability detection requires management clocks.
        @(negedge mgmt_clk);mgmt_enable=0;fabric_locked=0;ticks(20);
        require(memory_ready_async&&dut.available&&!availability_fault,"management pause holds sampled source");
        mgmt_enable=1;ticks(40);
        require(!memory_ready_async&&!dut.available&&availability_fault,"loss recognized after management resumes");
        group_pass("management_clock_pause");

        reset_all;stage('h2000,64'haaaa,9);hold_b=1;write_reg('h80c,1,15,0);ticks(20);
        require(dut.busy&&!dut.done,"accepted line before memory halt");
        @(negedge mem_clk);mem_enable=0;fabric_locked=0;ticks(40);
        require(availability_fault&&!memory_ready_async&&dut.busy&&!dut.done,"source records loss while memory halted");
        bus_rst_n=0;ticks(10);bus_rst_n=1;ticks(10);
        require(dut.busy&&!dut.done&&!sealed,"transport reset cannot retire stalled load");
        mem_enable=1;ticks(30);
        require(!dut.available&&dut.lost_ready&&dut.busy&&!dut.done,"memory observes retained loss after restart");
        hold_b=0;completion_wait(64'haaaa,8);require(dut.failed,"late B keeps loss attribution");
        write_reg('h810,1,15,0);write_reg('h818,1,15,0);write_reg('h814,1,15,2);
        require(!sealed,"failed held load cannot seal");
        group_pass("memory_halt_loss_and_late_b");

        fabric_locked=1;apb_complete=3;ticks(50);
        require(availability_fault&&!memory_ready_async&&!dut.available&&dut.failed,"sticky fault cannot rearm after relock");
        group_pass("no_rearm_after_relock");

        reset_all;stage('h2020,64'hbbbb,10);hold_b=1;write_reg('h80c,1,15,0);ticks(20);
        require(dut.busy,"cold reset has owned work");
        @(negedge mem_clk);mem_enable=0;
        @(negedge mgmt_clk);mgmt_enable=0;
        @(negedge clk);cold_rst_n=0;mem_cold_rst_n=0;bus_rst_n=0;
        #0.001;
        require(!memory_ready_async&&!dut.busy&&!dut.done&&!dut.failed&&!sealed,"cold reset works with both clocks stopped");
        ticks(8);require(!memory_ready_async,"registered source remains reset without edges");
        group_pass("cold_reset_with_stopped_clocks");

        reset_all;bus_rst_n=0;ticks(10);
        require(memory_ready_async&&dut.available,"transport reset does not clear source");
        bus_rst_n=1;ticks(10);
        stage('h2040,64'hcccc,11);write_reg('h80c,1,15,0);completion_wait(64'hcccc,0);
        write_reg('h810,1,15,0);write_reg('h814,1,15,0);
        require(sealed&&!dut.failed&&memory_ready_async,"cold recovery permits successful sealed load");
        group_pass("registered_source_recovery_and_seal");
