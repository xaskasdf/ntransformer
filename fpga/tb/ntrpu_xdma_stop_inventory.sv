`timescale 1ps/1ps
// Public native XDMA experiment. Delay application B responses; never force IP state.
module ntrpu_xdma_stop_inventory;
    reg hold_dma_b = 1'b1;
    reg hold_bypass_b = 1'b1;
    integer phase = 0;
    integer dma_aw = 0, dma_w = 0, dma_b = 0;
    integer bypass_aw = 0, bypass_w = 0, bypass_b = 0;
    reg [31:0] value;
    integer i;

    task require(input bit condition, input string reason);
        if (!condition) $fatal(1, "XDMA_STOP_ASSERT phase=%0d reason=%s", phase, reason);
    endtask

    task read_reg(input [31:0] address);
        begin
            board.RP.tx_usrapp.TSK_XDMA_REG_READ(address);
            value = board.RP.tx_usrapp.P_READ_DATA;
            require(!$isunknown(value), "unknown register response");
            $display("XDMA_STOP_REG t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, value);
        end
    endtask

    task write_reg(input [31:0] address, input [31:0] data);
        begin
            $display("XDMA_STOP_WRITE t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, data);
            board.RP.tx_usrapp.TSK_XDMA_REG_WRITE(address, data, 4'hf);
        end
    endtask

    task stopped_registers(input bit busy_expected);
        begin
            read_reg(32'h0004);
            require(value == 32'h00fffe7e, "Run cleared; logging controls retained");
            read_reg(32'h0040);
            require(value[0] == busy_expected && (value & 32'h00ffff00) == 0,
                    "H2C Busy or error status");
            read_reg(32'h0048);
            require(value == (busy_expected ? 0 : 1), "completed descriptor count");
            read_reg(32'h1004);
            require(value[0] == 0, "C2H Run remains zero");
            read_reg(32'h1040);
            require(value == 0, "C2H idle and error-free");
        end
    endtask

    task run_probe;
        reg [63:0] bypass_address;
        begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1,
                    "native endpoint ready after enumeration");
            require(board.RP.tx_usrapp.xdma_bar == 1, "expected DMA BAR1");
            require(board.RP.tx_usrapp.BAR_INIT_P_BAR_ENABLED[2] == 2'b11,
                    "expected 64-bit bypass BAR2");
            phase = 1;
            $display("XDMA_STOP_BEGIN t=%0t dma_bar=%0d", $time, board.RP.tx_usrapp.xdma_bar);
            // One terminal descriptor: 64 host bytes at 0x400 -> application 0x1000.
            // A single terminal descriptor does not test prefetched-chain cancellation.
            for (i=0; i<32; i=i+1) board.RP.tx_usrapp.DATA_STORE[256+i] = 0;
            board.RP.tx_usrapp.DATA_STORE[256] = 8'h13;
            board.RP.tx_usrapp.DATA_STORE[258] = 8'h4b;
            board.RP.tx_usrapp.DATA_STORE[259] = 8'had;
            board.RP.tx_usrapp.DATA_STORE[260] = 8'h40;
            board.RP.tx_usrapp.DATA_STORE[265] = 8'h04;
            board.RP.tx_usrapp.DATA_STORE[273] = 8'h10;
            for (i=0; i<64; i=i+1)
                board.RP.tx_usrapp.DATA_STORE[1024+i] = (i*13) ^ 8'ha5;
            read_reg(32'h0040); require(value == 0, "initial H2C idle");
            read_reg(32'h0048); require(value == 0, "initial count zero");
            write_reg(32'h4080, 32'h100);
            write_reg(32'h4084, 0);
            write_reg(32'h4088, 0);
            write_reg(32'h0004, 32'h00fffe7f);
            wait (board.EP.ntrpu_app_dma_bvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(dma_aw == 1 && dma_w == 1 && dma_b == 0, "one pending DMA write");
            $display("XDMA_STOP_PENDING t=%0t phase=%0d aw=%0d w=%0d b=%0d", $time, phase, dma_aw, dma_w, dma_b);
            read_reg(32'h0040); require(value[0] == 1, "Busy before stop");
            read_reg(32'h0048); require(value == 0, "B delay prevents descriptor completion");
            phase = 2;
            write_reg(32'h000c, 1); // W1C Run only, through real PCIe register request.
            stopped_registers(1'b1);
            repeat (64) @(negedge board.EP.user_clk);
            stopped_registers(1'b1);
            require(dma_b == 0 && board.EP.ntrpu_app_dma_bvalid === 1'b1,
                    "B remains outstanding after Run readback zero");
            @(negedge board.EP.user_clk);
            phase = 3;
            hold_dma_b = 0;
            $display("XDMA_STOP_RELEASE t=%0t phase=%0d", $time, phase);
            wait (dma_b == 1);
            repeat (64) @(negedge board.EP.user_clk);
            stopped_registers(1'b0);

            // A new host BAR write is independent of both stopped SGDMA channels.
            phase = 4;
            bypass_address = {board.RP.tx_usrapp.BAR_INIT_P_BAR[3][31:0],
                              board.RP.tx_usrapp.BAR_INIT_P_BAR[2][31:0]} + 64'h2000;
            board.RP.tx_usrapp.DATA_STORE[0] = 8'ha0;
            board.RP.tx_usrapp.DATA_STORE[1] = 8'hb1;
            board.RP.tx_usrapp.DATA_STORE[2] = 8'hc2;
            board.RP.tx_usrapp.DATA_STORE[3] = 8'hd3;
            $display("XDMA_STOP_BAR_WRITE t=%0t address=%016h value=d3c2b1a0", $time, bypass_address);
            board.RP.tx_usrapp.TSK_TX_MEMORY_WRITE_64(board.RP.tx_usrapp.DEFAULT_TAG,
                board.RP.tx_usrapp.DEFAULT_TC, 11'd1, bypass_address, 4'h0, 4'hf, 1'b0);
            board.RP.tx_usrapp.DEFAULT_TAG = board.RP.tx_usrapp.DEFAULT_TAG + 1;
            wait (board.EP.ntrpu_app_bypass_bvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(bypass_aw == 1 && bypass_w == 1 && bypass_b == 0, "one pending BAR write");
            $display("XDMA_STOP_PENDING t=%0t phase=%0d aw=%0d w=%0d b=%0d", $time, phase, bypass_aw, bypass_w, bypass_b);
            stopped_registers(1'b0);
            repeat (64) @(negedge board.EP.user_clk);
            require(bypass_b == 0 && board.EP.ntrpu_app_bypass_bvalid === 1'b1,
                    "BAR obligation retained while SGDMA idle");
            phase = 5;
            hold_bypass_b = 0;
            $display("XDMA_STOP_RELEASE t=%0t phase=%0d", $time, phase);
            wait (bypass_b == 1);
            board.RP.tx_usrapp.P_READ_DATA = 32'hffffffff;
            fork
                board.RP.tx_usrapp.TSK_TX_MEMORY_READ_64(board.RP.tx_usrapp.DEFAULT_TAG,
                    board.RP.tx_usrapp.DEFAULT_TC, 11'd1, bypass_address, 4'h0, 4'hf);
                board.RP.tx_usrapp.TSK_WAIT_FOR_READ_DATA;
            join
            value = board.RP.tx_usrapp.P_READ_DATA;
            require(value === 32'hd3c2b1a0, "native BAR readback");
            $display("XDMA_STOP_BAR_READ t=%0t value=%08h", $time, value);
            repeat (64) @(negedge board.EP.user_clk);
            require(dma_aw == 1 && dma_w == 1 && dma_b == 1 &&
                    bypass_aw == 1 && bypass_w == 1 && bypass_b == 1,
                    "no extra writes after drain");
            phase = 6;
            $display("XDMA_STOP_PASS t=%0t dma_bytes=64 bypass_bytes=4 reset_tested=0 chain_stop_tested=0", $time);
        end
    endtask

    // Accepted transfers, not level counts. A second reviewer reconstructs them from VCD.
    always @(posedge board.EP.user_clk) begin
        if (phase != 0) begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1,
                    "unexpected native reset or link loss");
            if (board.EP.m_axi_awvalid && board.EP.m_axi_awready) begin
                dma_aw = dma_aw+1;
                $display("XDMA_STOP_AW t=%0t port=DMA id=%h addr=%h len=%h size=%h", $time, board.EP.m_axi_awid, board.EP.m_axi_awaddr, board.EP.m_axi_awlen, board.EP.m_axi_awsize);
            end
            if (board.EP.m_axi_wvalid && board.EP.m_axi_wready) begin
                dma_w = dma_w+1;
                $display("XDMA_STOP_W t=%0t port=DMA data=%h strb=%h last=%b", $time, board.EP.m_axi_wdata, board.EP.m_axi_wstrb, board.EP.m_axi_wlast);
            end
            if (board.EP.m_axi_bvalid && board.EP.m_axi_bready) begin
                dma_b = dma_b+1;
                $display("XDMA_STOP_B t=%0t port=DMA id=%h resp=%h", $time, board.EP.m_axi_bid, board.EP.m_axi_bresp);
            end
            if (board.EP.m_axib_awvalid && board.EP.m_axib_awready) begin
                bypass_aw = bypass_aw+1;
                $display("XDMA_STOP_AW t=%0t port=BAR id=%h addr=%h len=%h size=%h", $time, board.EP.m_axib_awid, board.EP.m_axib_awaddr, board.EP.m_axib_awlen, board.EP.m_axib_awsize);
            end
            if (board.EP.m_axib_wvalid && board.EP.m_axib_wready) begin
                bypass_w = bypass_w+1;
                $display("XDMA_STOP_W t=%0t port=BAR data=%h strb=%h last=%b", $time, board.EP.m_axib_wdata, board.EP.m_axib_wstrb, board.EP.m_axib_wlast);
            end
            if (board.EP.m_axib_bvalid && board.EP.m_axib_bready) begin
                bypass_b = bypass_b+1;
                $display("XDMA_STOP_B t=%0t port=BAR id=%h resp=%h", $time, board.EP.m_axib_bid, board.EP.m_axib_bresp);
            end
        end
    end
    initial begin
        #180000000;
        $fatal(1, "XDMA_STOP_TIMEOUT phase=%0d", phase);
    end
endmodule
