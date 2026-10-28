// Diagnostic observation only. Keep the optional reset asserted at $finish.
`timescale 1ps/1ps
module ntrpu_xdma_reset_assert;
    reg reset_n = 1'b1;
    integer phase = 0;
    integer endpoint_edges = 0;
    integer root_edges = 0;
    integer reset_samples = 0;
    integer link_down_samples = 0;
    integer i;

    wire [14:0] axi_valids = {
        board.EP.m_axi_awvalid, board.EP.m_axi_wvalid, board.EP.m_axi_bvalid,
        board.EP.m_axi_arvalid, board.EP.m_axi_rvalid,
        board.EP.m_axib_awvalid, board.EP.m_axib_wvalid, board.EP.m_axib_bvalid,
        board.EP.m_axib_arvalid, board.EP.m_axib_rvalid,
        board.EP.m_axil_awvalid, board.EP.m_axil_wvalid, board.EP.m_axil_bvalid,
        board.EP.m_axil_arvalid, board.EP.m_axil_rvalid
    };

    always @(posedge board.EP.user_clk) begin
        if (phase != 0) begin
            endpoint_edges = endpoint_edges + 1;
            if (axi_valids !== 15'b0) $fatal(1, "RESET_ASSERT_AXI_NOT_QUIET");
        end
    end
    always @(posedge board.RP.user_clk) begin
        if (phase != 0) begin
            root_edges = root_edges + 1;
            if ({board.RP.s_axis_rq_tvalid, board.RP.m_axis_rc_tvalid} !== 2'b0)
                $fatal(1, "RESET_ASSERT_ROOT_NOT_QUIET");
        end
    end

    task run_probe;
        begin
            $display("XDMA_RESET_ENUM ps=%0t", $time);
            repeat (256) @(posedge board.ep_sys_clk_p);
            @(negedge board.ep_sys_clk_p);
            #1;
            if ({board.sys_rst_n, board.EP.user_resetn, board.EP.user_lnk_up,
                 board.RP.user_lnk_up, reset_n} !== 5'b11111)
                $fatal(1, "RESET_ASSERT_UNHEALTHY_START");
            phase = 1;
            $display("XDMA_RESET_QUIET ps=%0t", $time);
            repeat (256) @(posedge board.ep_sys_clk_p);
            @(negedge board.ep_sys_clk_p);
            #1;
            phase = 2;
            reset_n = 1'b0;
            $display("XDMA_RESET_ASSERT ps=%0t", $time);
            for (i=0; i<256; i=i+1) begin
                @(posedge board.ep_sys_clk_p);
                #1;
                if ($isunknown({board.sys_rst_n, board.EP.user_resetn,
                                board.EP.user_lnk_up, board.RP.user_lnk_up, reset_n}))
                    $fatal(1, "RESET_ASSERT_UNKNOWN_CONTROL");
                if (board.sys_rst_n !== 1'b1 || reset_n !== 1'b0)
                    $fatal(1, "RESET_ASSERT_STIMULUS_CHANGED");
                if (!board.EP.user_resetn) reset_samples = reset_samples + 1;
                if (!board.EP.user_lnk_up) link_down_samples = link_down_samples + 1;
                $display("XDMA_RESET_SAMPLE index=%0d ps=%0t reset=%0b link=%0b rp_link=%0b", i,
                    $time, board.EP.user_resetn, board.EP.user_lnk_up, board.RP.user_lnk_up);
            end
            $display("XDMA_RESET_ASSERT_OBSERVED ps=%0t samples=256 reset_samples=%0d link_down_samples=%0d ep_edges=%0d rp_edges=%0d held=1 release_tested=0",
                $time, reset_samples, link_down_samples, endpoint_edges, root_edges);
        end
    endtask

    initial begin
        #200000000;
        $fatal(1, "RESET_ASSERT_BOUNDED_TIMEOUT");
    end
endmodule
