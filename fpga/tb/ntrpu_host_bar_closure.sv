`timescale 1ps/1ps
// Host submission/receipt model against native BAR traffic. Not a driver or global fence.
module ntrpu_xdma_stop_inventory;
    reg hold_dma_b = 1'b0;
    reg hold_bypass_b = 1'b1;
    integer phase = 0;
    integer generation = 0, admitted = 0, retired = 0, outstanding = 0;
    integer close_target = 0, rejects = 0, certificate_rejects = 0;
    reg producer_closed = 1'b1, certificate_valid = 1'b0;
    integer certificates = 0;
    integer bypass_aw = 0, bypass_w = 0, bypass_b = 0, bypass_ar = 0, bypass_r = 0;
    reg [63:0] base_address;
    reg [31:0] value;

    task require(input bit condition, input string reason);
        if (!condition) $fatal(1, "HOST_CLOSE_ASSERT phase=%0d generation=%0d reason=%s", phase, generation, reason);
    endtask

    task read_reg(input [31:0] address);
        begin
            board.RP.tx_usrapp.TSK_XDMA_REG_READ(address);
            value = board.RP.tx_usrapp.P_READ_DATA;
            require(value === 32'b0, "both SGDMA channels remain unused");
            $display("HOST_CLOSE_REG t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, value);
        end
    endtask

    task idle_channels;
        begin
            read_reg('h0004); read_reg('h0040); read_reg('h0048);
            read_reg('h1004); read_reg('h1040); read_reg('h1048);
        end
    endtask

    task begin_generation(input integer next_generation);
        begin
            require(generation == 0 || (producer_closed && certificate_valid && outstanding == 0),
                    "prior generation fully observed before reuse");
            require(next_generation > generation && next_generation < 32'h7fffffff, "no epoch wrap/reuse");
            generation = next_generation;
            admitted = 0; retired = 0; outstanding = 0; close_target = 0;
            producer_closed = 0; certificate_valid = 0;
            hold_bypass_b = 1;
            $display("HOST_CLOSE_OPEN t=%0t generation=%0d", $time, generation);
        end
    endtask

    // Atomic admission decision has no event control. Only admitted work calls TX.
    // The native sender may consume cycles; close never cancels its obligation.
    task submit_write(input integer requested_generation, input [63:0] address,
                      input [31:0] data, input bit expect_admitted);
        bit accepted;
        integer j;
        begin
            accepted = !producer_closed && requested_generation == generation;
            require(accepted == expect_admitted, "submission result");
            if (!accepted) begin
                rejects = rejects+1;
                $display("HOST_CLOSE_SUBMIT t=%0t generation=%0d requested=%0d accepted=0 address=%h data=%h closed=%b admitted=%0d outstanding=%0d",
                    $time, generation, requested_generation, address, data, producer_closed, admitted, outstanding);
            end else begin
                require(admitted == 0 && outstanding == 0, "bounded one-write epoch");
                admitted = admitted+1; outstanding = outstanding+1;
                $display("HOST_CLOSE_SUBMIT t=%0t generation=%0d requested=%0d accepted=1 address=%h data=%h closed=%b admitted=%0d outstanding=%0d",
                    $time, generation, requested_generation, address, data, producer_closed, admitted, outstanding);
                for (j=0; j<4; j=j+1) board.RP.tx_usrapp.DATA_STORE[j] = data[j*8 +: 8];
                board.RP.tx_usrapp.TSK_TX_MEMORY_WRITE_64(board.RP.tx_usrapp.DEFAULT_TAG,
                    3'b000, 11'd1, address, 4'h0, 4'hf, 1'b0);
                board.RP.tx_usrapp.DEFAULT_TAG = board.RP.tx_usrapp.DEFAULT_TAG+1;
                $display("HOST_CLOSE_SENT t=%0t generation=%0d address=%h outstanding=%0d", $time, generation, address, outstanding);
            end
        end
    endtask

    task close_generation;
        begin
            require(!producer_closed && admitted == 1 && outstanding == 1, "close retains admitted operation");
            producer_closed = 1; close_target = admitted;
            $display("HOST_CLOSE_CLOSED t=%0t generation=%0d target=%0d outstanding=%0d", $time, generation, close_target, outstanding);
        end
    endtask

    task request_certificate(input integer requested_generation, input integer requested_target,
                             input bit expect_accepted);
        bit accepted;
        begin
            accepted = producer_closed && requested_generation == generation && requested_target == close_target &&
                       close_target == admitted && retired == admitted && outstanding == 0;
            require(accepted == expect_accepted, "certificate admission");
            if (accepted) begin
                require(!certificate_valid, "one certificate per generation");
                certificate_valid = 1; certificates = certificates+1;
            end else certificate_rejects = certificate_rejects+1;
            $display("HOST_CLOSE_CERT t=%0t generation=%0d requested=%0d target=%0d accepted=%b outstanding=%0d retired=%0d",
                     $time, generation, requested_generation, requested_target, accepted, outstanding, retired);
        end
    endtask

    task readback(input [63:0] address, input [31:0] expected_data);
        begin
            require(producer_closed && outstanding == 1 && retired == 0, "readback belongs to closed obligation");
            board.RP.tx_usrapp.P_READ_DATA = 32'hffffffff;
            $display("HOST_CLOSE_READ_BEGIN t=%0t generation=%0d address=%h", $time, generation, address);
            fork
                board.RP.tx_usrapp.TSK_TX_MEMORY_READ_64(board.RP.tx_usrapp.DEFAULT_TAG, 3'b000,
                    11'd1, address, 4'h0, 4'hf);
                board.RP.tx_usrapp.TSK_WAIT_FOR_READ_DATA;
            join
            board.RP.tx_usrapp.DEFAULT_TAG = board.RP.tx_usrapp.DEFAULT_TAG+1;
            value = board.RP.tx_usrapp.P_READ_DATA;
            require(value === expected_data, "native same-address readback");
            retired = retired+1; outstanding = outstanding-1;
            $display("HOST_CLOSE_RECEIPT t=%0t generation=%0d address=%h data=%h outstanding=%0d retired=%0d",
                     $time, generation, address, value, outstanding, retired);
        end
    endtask

    task drain_epoch(input [63:0] address, input [31:0] expected_data);
        begin
            // The read may wait behind the held B. Its completion cannot be the
            // event that releases B. No speculative certificate during the wait.
            fork
                readback(address, expected_data);
                begin
                    repeat (64) @(negedge board.EP.user_clk);
                    request_certificate(generation, close_target, 1'b0);
                    repeat (960) @(negedge board.EP.user_clk);
                    require(outstanding == 1 && board.EP.ntrpu_app_bypass_bvalid === 1'b1,
                            "response independently retained");
                    hold_bypass_b = 0;
                    $display("HOST_CLOSE_RELEASE t=%0t generation=%0d", $time, generation);
                end
            join
            @(negedge board.EP.user_clk);
            request_certificate(generation-1, close_target, 1'b0);
            request_certificate(generation, close_target+1, 1'b0);
            request_certificate(generation, close_target, 1'b1);
            repeat (128) @(negedge board.EP.user_clk);
        end
    endtask

    task run_probe;
        begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1, "native endpoint ready");
            require(board.RP.tx_usrapp.xdma_bar == 1 && board.RP.tx_usrapp.BAR_INIT_P_BAR_ENABLED[2] == 2'b11,
                    "native DMA and 64-bit bypass BARs");
            base_address = {board.RP.tx_usrapp.BAR_INIT_P_BAR[3][31:0], board.RP.tx_usrapp.BAR_INIT_P_BAR[2][31:0]};
            phase = 1;
            $display("HOST_CLOSE_BEGIN t=%0t dma_bar=1 bypass_base=%h", $time, base_address);
            idle_channels;
            @(negedge board.EP.user_clk);
            phase = 2; begin_generation(7);
            submit_write(7, base_address+'h2000, 32'hd3c2b1a0, 1'b1);
            wait (board.EP.ntrpu_app_bypass_bvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(bypass_aw == 1 && bypass_w == 1 && bypass_b == 0, "epoch7 retained write");
            $display("HOST_CLOSE_PENDING t=%0t generation=7 aw=%0d w=%0d b=%0d", $time, bypass_aw, bypass_w, bypass_b);
            close_generation;
            // Deterministic close-before-submit ordering at adjacent clock edges.
            @(posedge board.EP.user_clk);
            submit_write(7, base_address+'h2040, 32'h01234567, 1'b0);
            @(negedge board.EP.user_clk);
            submit_write(7, base_address+'h2080, 32'h89abcdef, 1'b0);
            request_certificate(7, 1, 1'b0);
            request_certificate(8, 1, 1'b0);
            drain_epoch(base_address+'h2000, 32'hd3c2b1a0);
            phase = 3;
            idle_channels;

            @(negedge board.EP.user_clk);
            phase = 4; begin_generation(8);
            request_certificate(7, 1, 1'b0); // Previous certificate is stale here.
            submit_write(7, base_address+'h20c0, 32'h13579bdf, 1'b0);
            // Admission before close survives even when native B is outstanding.
            submit_write(8, base_address+'h2100, 32'h4c3d2e1f, 1'b1);
            @(negedge board.EP.user_clk);
            close_generation;
            @(posedge board.EP.user_clk);
            submit_write(8, base_address+'h2140, 32'h2468ace0, 1'b0);
            wait (board.EP.ntrpu_app_bypass_bvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(bypass_aw == 2 && bypass_w == 2 && bypass_b == 1, "epoch8 retained write");
            $display("HOST_CLOSE_PENDING t=%0t generation=8 aw=%0d w=%0d b=%0d", $time, bypass_aw, bypass_w, bypass_b);
            request_certificate(8, 1, 1'b0);
            drain_epoch(base_address+'h2100, 32'h4c3d2e1f);
            phase = 5;
            idle_channels;
            repeat (1024) @(negedge board.EP.user_clk);
            require(bypass_aw == 2 && bypass_w == 2 && bypass_b == 2 && bypass_ar == 2 && bypass_r == 2,
                    "no untracked native BAR operations");
            require(outstanding == 0 && certificate_valid && certificates == 2 && rejects == 4 && certificate_rejects == 10,
                    "complete bounded producer ledger");
            phase = 6;
            $display("HOST_CLOSE_PASS t=%0t generations=2 writes=2 receipts=2 rejects=%0d certificate_rejects=%0d driver_tested=0 global_drain_tested=0 reset_tested=0",
                     $time, rejects, certificate_rejects);
        end
    endtask

    always @(posedge board.EP.user_clk) begin
        if (phase != 0) begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1, "native link/reset unchanged");
            require(!(board.EP.m_axi_awvalid || board.EP.m_axi_arvalid), "SGDMA data paths idle");
            if (board.EP.m_axib_awvalid && board.EP.m_axib_awready) begin
                bypass_aw = bypass_aw+1;
                $display("HOST_CLOSE_AW t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axib_awid, board.EP.m_axib_awaddr, board.EP.m_axib_awlen, board.EP.m_axib_awsize);
            end
            if (board.EP.m_axib_wvalid && board.EP.m_axib_wready) begin
                bypass_w = bypass_w+1;
                $display("HOST_CLOSE_W t=%0t data=%h strb=%h last=%b", $time, board.EP.m_axib_wdata, board.EP.m_axib_wstrb, board.EP.m_axib_wlast);
            end
            if (board.EP.m_axib_bvalid && board.EP.m_axib_bready) begin
                bypass_b = bypass_b+1;
                require(board.EP.m_axib_bresp === 2'b00, "native BAR write response");
                $display("HOST_CLOSE_B t=%0t id=%h resp=%h", $time, board.EP.m_axib_bid, board.EP.m_axib_bresp);
            end
            if (board.EP.m_axib_arvalid && board.EP.m_axib_arready) begin
                bypass_ar = bypass_ar+1;
                $display("HOST_CLOSE_AR t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axib_arid, board.EP.m_axib_araddr, board.EP.m_axib_arlen, board.EP.m_axib_arsize);
            end
            if (board.EP.m_axib_rvalid && board.EP.m_axib_rready) begin
                bypass_r = bypass_r+1;
                require(board.EP.m_axib_rresp === 2'b00 && board.EP.m_axib_rlast === 1'b1, "native BAR read response");
                $display("HOST_CLOSE_R t=%0t id=%h data=%h resp=%h last=%b", $time, board.EP.m_axib_rid, board.EP.m_axib_rdata, board.EP.m_axib_rresp, board.EP.m_axib_rlast);
            end
        end
    end
    initial begin
        #200000000;
        $fatal(1, "HOST_CLOSE_TIMEOUT phase=%0d generation=%0d outstanding=%0d", phase, generation, outstanding);
    end
endmodule
