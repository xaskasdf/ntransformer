// Isolated XDMA AXI-MM scoreboard; 768/1536 aligned bytes, no DDR attached.
`timescale 1ps/1ps
module ntrpu_pcie_multi_monitor #(parameter bit HOST64 = 0);
    localparam integer BYTES = HOST64 ? 1536 : 768;
    integer wbytes = 0, rbytes = 0, wbeats = 0, rbeats = 0;
    integer aw = 0, ar = 0, b = 0, wl = 0, rl = 0, lane;
    integer aw_bytes = 0, ar_bytes = 0;
    integer aw_end [0:31], ar_end [0:31], w_end [0:31], r_end [0:31];
    logic [3:0] aw_id [0:31], ar_id [0:31];
    bit b_received [0:31];
    integer response_index;
    integer w_stalls = 0, r_stalls = 0;
    integer aw_stalls = 0, ar_stalls = 0, b_stalls = 0;
    time first_w = 0, last_w = 0, first_r = 0, last_r = 0;
    time previous_clock = 0, clock_ps = 0, link_ps = 0;
    bit complete = 0;

    ntrpu_pcie_cq_check #(.PAYLOAD_BYTES(BYTES), .HOST64(HOST64)) cq (
        .clk(board.RP.user_clk), .reset(board.RP.user_reset),
        .valid(board.RP.m_axis_cq_tvalid), .ready(board.RP.m_axis_cq_tready),
        .last(board.RP.m_axis_cq_tlast), .data(board.RP.m_axis_cq_tdata),
        .keep(board.RP.m_axis_cq_tkeep), .user(board.RP.m_axis_cq_tuser)
    );
    function automatic [7:0] pattern(input integer offset);
        pattern = 8'(offset ^ (offset >> 8) ^ 'ha5);
    endfunction
    initial begin
        if (HOST64) $display("NTRPU_PCIE_HOST64_FORMAT version=1 time_unit=ps payload=1536");
        else $display("NTRPU_PCIE_MULTI_FORMAT version=1 time_unit=ps payload=768");
    end
    always @(posedge board.EP.user_lnk_up) link_ps = $time;
    always @(posedge board.EP.user_clk) begin
        clock_ps = $time - previous_clock;
        previous_clock = $time;
        if (board.EP.user_resetn === 1'b1) begin
            if (board.EP.m_axi_awvalid && board.EP.m_axi_awready) begin
                if (aw >= 32 || $isunknown({board.EP.m_axi_awaddr, board.EP.m_axi_awlen}) ||
                    board.EP.m_axi_awaddr != aw_bytes || board.EP.m_axi_awsize !== 3'd6 ||
                    board.EP.m_axi_awburst !== 2'b01) $fatal(1, "MULTI_AW_RANGE");
                aw_bytes += (int'(board.EP.m_axi_awlen)+1)*64;
                if (aw_bytes > BYTES) $fatal(1, "MULTI_AW_OVERFLOW");
                if ($isunknown(board.EP.m_axi_awid)) $fatal(1, "MULTI_AW_ID");
                aw_id[aw] = board.EP.m_axi_awid;
                aw_end[aw++] = aw_bytes;
            end
            if (board.EP.m_axi_arvalid && board.EP.m_axi_arready) begin
                if (ar >= 32 || $isunknown({board.EP.m_axi_araddr, board.EP.m_axi_arlen}) ||
                    board.EP.m_axi_araddr != ar_bytes || board.EP.m_axi_arsize !== 3'd6 ||
                    board.EP.m_axi_arburst !== 2'b01) $fatal(1, "MULTI_AR_RANGE");
                ar_bytes += (int'(board.EP.m_axi_arlen)+1)*64;
                if (ar_bytes > BYTES) $fatal(1, "MULTI_AR_OVERFLOW");
                if ($isunknown(board.EP.m_axi_arid)) $fatal(1, "MULTI_AR_ID");
                ar_id[ar] = board.EP.m_axi_arid;
                ar_end[ar++] = ar_bytes;
            end
            if (board.EP.m_axi_awvalid && !board.EP.m_axi_awready) aw_stalls++;
            if (board.EP.m_axi_arvalid && !board.EP.m_axi_arready) ar_stalls++;
            if (board.EP.m_axi_bvalid && !board.EP.m_axi_bready) b_stalls++;
            if (board.EP.m_axi_wvalid && !board.EP.m_axi_wready) w_stalls++;
            if (board.EP.m_axi_rvalid && !board.EP.m_axi_rready) r_stalls++;
            if (board.EP.m_axi_wvalid && board.EP.m_axi_wready) begin
                if (wbytes == 0) first_w = $time;
                last_w = $time; wbeats++;
                if (board.EP.m_axi_wstrb !== 64'hffffffffffffffff ||
                    $isunknown(board.EP.m_axi_wlast)) $fatal(1, "MULTI_W_CONTROL");
                for (lane = 0; lane < 64; lane++) begin
                    if (wbytes >= BYTES || board.EP.m_axi_wdata[lane*8 +: 8] !== pattern(wbytes))
                        $fatal(1, "MULTI_W_DATA offset=%0d", wbytes);
                    wbytes++;
                end
                if (board.EP.m_axi_wlast) begin
                    if (wl >= 32) $fatal(1, "MULTI_WLAST_OVERFLOW");
                    w_end[wl++] = wbytes;
                end
            end
            if (board.EP.m_axi_bvalid && board.EP.m_axi_bready) begin
                response_index = -1;
                for (integer i = 0; i < aw; i++)
                    if (response_index == -1 && !b_received[i] && aw_id[i] === board.EP.m_axi_bid)
                        response_index = i;
                if (response_index == -1 || response_index >= wl) $fatal(1, "MULTI_B_ID_OR_EARLY");
                b_received[response_index] = 1;
                b++;
                if (board.EP.m_axi_bresp !== 2'b00 || b > aw) $fatal(1, "MULTI_B_RESPONSE");
            end
            if (board.EP.m_axi_rvalid && board.EP.m_axi_rready) begin
                // This bounded fixture expects sequential reads, with no ID
                // reordering; correlate each response burst with its AR.
                if (rl >= ar || board.EP.m_axi_rid !== ar_id[rl]) $fatal(1, "MULTI_R_ID_OR_EARLY");
                if (rbytes == 0) first_r = $time;
                last_r = $time; rbeats++;
                if (board.EP.m_axi_rresp !== 2'b00 || $isunknown(board.EP.m_axi_rlast))
                    $fatal(1, "MULTI_R_RESPONSE");
                for (lane = 0; lane < 64; lane++) begin
                    if (rbytes >= BYTES || board.EP.m_axi_rdata[lane*8 +: 8] !== pattern(rbytes))
                        $fatal(1, "MULTI_R_DATA offset=%0d", rbytes);
                    rbytes++;
                end
                if (board.EP.m_axi_rlast) begin
                    if (rl >= 32) $fatal(1, "MULTI_RLAST_OVERFLOW");
                    r_end[rl++] = rbytes;
                end
            end
        end
    end
    task verify_complete;
        integer i;
        begin
            cq.verify_complete();
`ifdef NTRPU_PCIE_STALLS
            board.EP.ntrpu_w_gate.verify_complete(BYTES/64);
            board.EP.ntrpu_r_gate.verify_complete(BYTES/64);
`endif
`ifdef NTRPU_PCIE_ALL_STALLS
            board.EP.ntrpu_aw_gate.verify_complete(aw);
            board.EP.ntrpu_ar_gate.verify_complete(ar);
            board.EP.ntrpu_b_gate.verify_complete(b);
            if (board.EP.ntrpu_aw_gate.source_stalls != aw_stalls ||
                board.EP.ntrpu_ar_gate.source_stalls != ar_stalls ||
                board.EP.ntrpu_b_gate.sink_stalls != b_stalls)
                $fatal(1, "MULTI_CONTROL_STALL_COUNTS");
            $display("NTRPU_AXI_CONTROL_COMPLETE aw_stalls=%0d ar_stalls=%0d b_sink_stalls=%0d bid_checked=%0d rid_beats_checked=%0d",
                aw_stalls, ar_stalls, b_stalls, b, rbeats);
`endif
`ifdef NTRPU_PCIE_HOST64
            board.ntrpu_host.verify_complete();
`endif
            if (complete || wbytes != BYTES || rbytes != BYTES || aw_bytes != BYTES ||
                ar_bytes != BYTES || aw != wl || ar != rl || b != aw ||
                board.RP.tx_usrapp.h2c_status !== 32'h6 || board.RP.tx_usrapp.c2h_status !== 32'h6 ||
                clock_ps != 4000 || link_ps == 0 || board.EP.user_lnk_up !== 1'b1 ||
                board.RP.cfg_negotiated_width !== 3'd4 || board.RP.cfg_current_speed !== 3'd2)
                $fatal(1, "MULTI_INCOMPLETE");
            for (i = 0; i < aw; i++) if (aw_end[i] != w_end[i]) $fatal(1, "MULTI_WLAST_LENGTH");
            for (i = 0; i < ar; i++) if (ar_end[i] != r_end[i]) $fatal(1, "MULTI_RLAST_LENGTH");
            complete = 1;
            $display("NTRPU_PCIE_MULTI_COMPLETE wbytes=%0d rbytes=%0d wbeats=%0d rbeats=%0d aw=%0d ar=%0d b=%0d w_stalls=%0d r_stalls=%0d clock_ps=%0d link_ps=%0d first_w_ps=%0d last_w_ps=%0d first_r_ps=%0d last_r_ps=%0d rp_mps_raw=%0d rp_mrrs_raw=%0d",
                wbytes, rbytes, wbeats, rbeats, aw, ar, b, w_stalls, r_stalls,
                clock_ps, link_ps, first_w, last_w, first_r, last_r,
                board.RP.cfg_max_payload, board.RP.cfg_max_read_req);
        end
    endtask
    initial begin #200000000; $fatal(1, "MULTI_TIMEOUT_200US"); end
    final if (!complete) $error("NTRPU_PCIE_MULTI_INCOMPLETE");
endmodule
