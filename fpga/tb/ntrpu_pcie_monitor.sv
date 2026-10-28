// Instrumentation for the isolated AMD XDMA AXI-MM example (not NTRPU DDR).
`timescale 1ps/1ps
module ntrpu_pcie_monitor;
    longint unsigned cycles = 0, wbytes = 0, wbeats = 0, rbeats = 0;
    longint unsigned aw_count = 0, ar_count = 0, b_count = 0;
    longint unsigned w_stall = 0, r_stall = 0;
    time first_write = 0, last_write = 0, first_read = 0, last_read = 0;
    time previous_clock = 0, clock_period = 0, link_up_time = 0;
    integer lane;

    // Decimal time values use this module's 1 ps unit. %t would instead use
    // the simulator-wide display precision (1 fs in the generated PCIe model).
    initial $display("NTRPU_PCIE_MONITOR_FORMAT version=2 time_unit=ps");

    always @(posedge board.EP.user_lnk_up) begin
        link_up_time = $time;
        $display("NTRPU_PCIE_LINK_UP time_ps=%0d", $time);
    end

    always @(posedge board.EP.user_clk) begin
        clock_period = $time - previous_clock;
        previous_clock = $time;
        if (board.EP.user_resetn === 1'b1) begin
            cycles++;
            if (board.EP.m_axi_awvalid && board.EP.m_axi_awready) begin
                aw_count++;
                $display("NTRPU_PCIE_AW time_ps=%0d addr=%h len=%0d size=%0d", $time,
                    board.EP.m_axi_awaddr, board.EP.m_axi_awlen, board.EP.m_axi_awsize);
            end
            if (board.EP.m_axi_arvalid && board.EP.m_axi_arready) begin
                ar_count++;
                $display("NTRPU_PCIE_AR time_ps=%0d addr=%h len=%0d size=%0d", $time,
                    board.EP.m_axi_araddr, board.EP.m_axi_arlen, board.EP.m_axi_arsize);
            end
            if (board.EP.m_axi_wvalid && !board.EP.m_axi_wready) w_stall++;
            if (board.EP.m_axi_rvalid && !board.EP.m_axi_rready) r_stall++;
            if (board.EP.m_axi_wvalid && board.EP.m_axi_wready) begin
                if (wbeats == 0) first_write = $time;
                last_write = $time;
                wbeats++;
                if ($isunknown(board.EP.m_axi_wstrb)) $fatal(1, "Unknown AXI WSTRB");
                for (lane = 0; lane < 64; lane++) begin
                    if (board.EP.m_axi_wstrb[lane]) begin
                        if (board.EP.m_axi_wdata[lane*8 +: 8] !== (8'(wbytes)))
                            $fatal(1, "H2C byte mismatch at byte %0d", wbytes);
                        wbytes++;
                    end
                end
            end
            if (board.EP.m_axi_bvalid && board.EP.m_axi_bready) begin
                b_count++;
                if (board.EP.m_axi_bresp !== 2'b00) $fatal(1, "AXI BRESP error");
            end
            if (board.EP.m_axi_rvalid && board.EP.m_axi_rready) begin
                if (rbeats == 0) first_read = $time;
                last_read = $time;
                rbeats++;
                if (board.EP.m_axi_rresp !== 2'b00) $fatal(1, "AXI RRESP error");
            end
        end
    end

    // Separate wall/simulation limits also apply in the launcher.
    initial begin
        #2000000000;
        $fatal(1, "NTRPU PCIe simulation exceeded 2 ms without completing");
    end

    final begin
        $display("NTRPU_PCIE_METRICS cycles=%0d clock_ps=%0d link_up_ps=%0d wbytes=%0d wbeats=%0d rbeats=%0d aw=%0d ar=%0d b=%0d w_stall=%0d r_stall=%0d first_write_ps=%0d last_write_ps=%0d first_read_ps=%0d last_read_ps=%0d rp_width_raw=%0d rp_speed_raw=%0d",
            cycles, clock_period, link_up_time, wbytes, wbeats, rbeats, aw_count,
            ar_count, b_count, w_stall, r_stall, first_write, last_write,
            first_read, last_read, board.RP.cfg_negotiated_width, board.RP.cfg_current_speed);
        if (wbytes != board.RP.tx_usrapp.DMA_BYTE_CNT || !rbeats ||
            !aw_count || !ar_count || b_count != aw_count ||
            board.RP.tx_usrapp.h2c_status !== 32'h6 ||
            board.RP.tx_usrapp.c2h_status !== 32'h6 ||
            board.EP.user_lnk_up !== 1'b1)
            $error("NTRPU_PCIE_MONITOR_INCOMPLETE");
        else
            $display("NTRPU_PCIE_MONITOR_COMPLETE");
    end
endmodule
