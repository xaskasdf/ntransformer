// Simulation-only scenario included inside ntrpu_axi_bridge_clock_loss.
// Force only the public AXI clock export. This is an injected boundary fault,
// not a physical GT/refclk failure or a claim about every internal IP clock.
    integer stop_level=0;
    reg pause_active=0,pause_complete=0,clock_case_done=0;
    initial begin
        if($value$plusargs("clock_stop_level=%d",stop_level))begin end
        if(stop_level!=0 && stop_level!=1)$fatal(1,"CL invalid stop level");
    end
    always @(clk)if(pause_active)$fatal(1,"CL AXI clock changed during pause");

    task retained_clock_context;
        require(configuration_rst_n && cold_rst_n && board.sys_rst_n,"CL cold reset and PERST remain released");
        require(native_available,"CL native status must remain available for pure clock-loss case");
        require(owned && src_req_valid && src_req_tag=={64'd9,64'd3} &&
                src_req_address==64'h1000 && src_req_bytes==64,"CL retained source identity");
        require(!src_req_ready && !src_valid && !src_cpl_valid && !ack_valid && !result_valid,
                "CL pending source has no synthetic completion");
    endtask

    task run_clock_loss;
        retained_clock_context;
        require(guard_armed && guard_alive && !guard_fault && !fault && !mapper_fault &&
                link_up && !stop,"CL healthy before injection");
        if(stop_level==0)@(negedge clk);else @(posedge clk);
        #1; // Avoid changing a level at an active clock edge.
        if(stop_level==0)force board.AXI_PCIE_EP.axi_aclk_out=1'b0;
        else force board.AXI_PCIE_EP.axi_aclk_out=1'b1;
        pause_active=1;
        $display("CL PAUSE level=%0d t=%0t",stop_level,$time);
        // 128 management cycles exceed the 64-cycle watchdog plus CDC margin.
        repeat(128)begin
            @(negedge mgmt_clk);retained_clock_context;
            require(!fault && !mapper_fault,"CL AXI-domain fault registers cannot advance without clock");
        end
        require(guard_fault && guard_code==2 && stop && !link_up && !cmd_ready && !cfg_ready,
                "CL watchdog closed admission without an AXI edge");
        $display("CL HELD level=%0d t=%0t code=%0d",stop_level,$time,guard_code);
        #1;pause_active=0;pause_complete=1;
        release board.AXI_PCIE_EP.axi_aclk_out;
        $display("CL RELEASE level=%0d t=%0t",stop_level,$time);
        repeat(64)@(negedge mgmt_clk);
        retained_clock_context;
        require(guard_alive && guard_fault && guard_code==2 && stop && !link_up &&
                fault && mapper_fault && !cmd_ready && !cfg_ready,"CL recovery cannot rearm or retire context");
        clock_case_done=1;
        $display("CL RECOVERED level=%0d t=%0t",stop_level,$time);
        $display("BRIDGE_CLOCK_LOSS_PASS level=%0d timeout=64 hold_cycles=128 recovery_cycles=64",stop_level);
    endtask
