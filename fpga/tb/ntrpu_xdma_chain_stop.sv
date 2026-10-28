`timescale 1ps/1ps
// Native two-descriptor observation. Only public application B is delayed.
module ntrpu_xdma_stop_inventory;
    reg hold_dma_b = 1'b1;
    reg hold_bypass_b = 1'b1;
    integer phase = 0;
    integer dma_aw = 0, dma_w = 0, dma_b = 0;
    integer settled_count = 0;
    reg [31:0] value;
    integer i;

    task require(input bit condition, input string reason);
        if (!condition) $fatal(1, "XDMA_CHAIN_ASSERT phase=%0d reason=%s", phase, reason);
    endtask

    task store_word(input integer address, input [31:0] data);
        integer j;
        for (j=0; j<4; j=j+1)
            board.RP.tx_usrapp.DATA_STORE[address+j] = data[j*8 +: 8];
    endtask

    task read_reg(input [31:0] address);
        begin
            board.RP.tx_usrapp.TSK_XDMA_REG_READ(address);
            value = board.RP.tx_usrapp.P_READ_DATA;
            require(!$isunknown(value), "unknown register response");
            $display("XDMA_CHAIN_REG t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, value);
        end
    endtask

    task write_reg(input [31:0] address, input [31:0] data);
        begin
            $display("XDMA_CHAIN_WRITE t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, data);
            board.RP.tx_usrapp.TSK_XDMA_REG_WRITE(address, data, 4'hf);
        end
    endtask

    task stopped_registers(input bit busy_expected, input integer completed);
        begin
            read_reg(32'h0004); require(value == 32'h00fffe7e, "Run cleared, notification controls retained");
            read_reg(32'h0040);
            require(value[0] == busy_expected && (value & ~32'h47) == 0, "H2C Busy/error status");
            read_reg(32'h0048); require(value == completed, "completed descriptor count");
            read_reg(32'h1004); require(value == 0, "C2H Run unchanged");
            read_reg(32'h1040); require(value == 0, "C2H idle/error-free");
        end
    endtask

    task run_probe;
        begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1, "native endpoint ready");
            require(board.RP.tx_usrapp.xdma_bar == 1, "DMA BAR1");
            phase = 1;
            $display("XDMA_CHAIN_BEGIN t=%0t dma_bar=1 descriptors=2 adjacent=1", $time);
            // One 64-byte adjacent block. The first points to the terminal second.
            for (i=0; i<64; i=i+1) board.RP.tx_usrapp.DATA_STORE[256+i] = 0;
            store_word('h100, 32'had4b0012);
            store_word('h104, 64); store_word('h108, 'h400);
            store_word('h110, 'h1000); store_word('h118, 'h120);
            store_word('h120, 32'had4b0013);
            store_word('h124, 64); store_word('h128, 'h500);
            store_word('h130, 'h1100);
            for (i=0; i<64; i=i+1) begin
                board.RP.tx_usrapp.DATA_STORE[1024+i] = (i*13) ^ 8'ha5;
                board.RP.tx_usrapp.DATA_STORE[1280+i] = (i*7) ^ 8'h3c;
            end
            read_reg('h40); require(value == 0, "initial H2C idle");
            read_reg('h48); require(value == 0, "initial count zero");
            write_reg('h4080, 'h100); write_reg('h4084, 0);
            write_reg('h4088, 1); write_reg('h0004, 'h00fffe7f);
            wait (board.EP.ntrpu_app_dma_bvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(dma_aw >= 1 && dma_aw <= 2 && dma_w >= 1 && dma_w <= 2 && dma_b == 0,
                    "first response pending, at most two descriptors issued");
            $display("XDMA_CHAIN_PENDING t=%0t phase=%0d aw=%0d w=%0d b=%0d", $time, phase, dma_aw, dma_w, dma_b);
            read_reg('h40); require(value[0] == 1, "Busy before stop");
            read_reg('h48); require(value == 0, "no completion before first B");
            phase = 2;
            write_reg('h000c, 1);
            stopped_registers(1'b1, 0);
            repeat (64) @(negedge board.EP.user_clk);
            stopped_registers(1'b1, 0);
            require(dma_b == 0 && board.EP.ntrpu_app_dma_bvalid === 1'b1,
                    "response retained after Run readback zero");
            @(negedge board.EP.user_clk);
            phase = 3; hold_dma_b = 0;
            $display("XDMA_CHAIN_RELEASE t=%0t phase=%0d aw=%0d w=%0d b=%0d", $time, phase, dma_aw, dma_w, dma_b);
            wait (dma_b >= 1);
            repeat (1024) @(negedge board.EP.user_clk);
            // Classification is based on observed completions, not assumed
            // cancellation of prefetched descriptors. The reviewer checks all bytes.
            settled_count = dma_b;
            require(settled_count >= 1 && settled_count <= 2 && dma_aw == settled_count && dma_w == settled_count,
                    "bounded drain of accepted AXI writes");
            $display("XDMA_CHAIN_SETTLED t=%0t completed=%0d", $time, settled_count);
            stopped_registers(1'b0, settled_count);
            repeat (1024) @(negedge board.EP.user_clk);
            stopped_registers(1'b0, settled_count);
            require(dma_aw == settled_count && dma_w == settled_count && dma_b == settled_count,
                    "no further writes during stopped observation interval");
            phase = 4;
            $display("XDMA_CHAIN_PASS t=%0t descriptors=2 completed=%0d payload_bytes=%0d reset_tested=0 c2h_tested=0", $time, settled_count, settled_count*64);
        end
    endtask

    always @(posedge board.EP.user_clk) begin
        if (phase != 0) begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1, "native reset/link unchanged");
            require(!(board.EP.m_axi_arvalid && board.EP.m_axi_arready), "C2H not exercised");
            require(!(board.EP.m_axib_awvalid || board.EP.m_axib_arvalid), "BAR data path not exercised");
            if (board.EP.m_axi_awvalid && board.EP.m_axi_awready) begin
                dma_aw = dma_aw+1;
                require(dma_aw <= 2, "unexpected extra DMA address");
                $display("XDMA_CHAIN_AW t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axi_awid, board.EP.m_axi_awaddr, board.EP.m_axi_awlen, board.EP.m_axi_awsize);
            end
            if (board.EP.m_axi_wvalid && board.EP.m_axi_wready) begin
                dma_w = dma_w+1;
                require(dma_w <= 2 && board.EP.m_axi_wlast === 1'b1, "one beat per descriptor");
                $display("XDMA_CHAIN_W t=%0t data=%h strb=%h last=%b", $time, board.EP.m_axi_wdata, board.EP.m_axi_wstrb, board.EP.m_axi_wlast);
            end
            if (board.EP.m_axi_bvalid && board.EP.m_axi_bready) begin
                dma_b = dma_b+1;
                require(dma_b <= 2 && board.EP.m_axi_bresp === 2'b00, "successful bounded response");
                $display("XDMA_CHAIN_B t=%0t id=%h resp=%h", $time, board.EP.m_axi_bid, board.EP.m_axi_bresp);
            end
        end
    end
    initial begin
        #180000000;
        $fatal(1, "XDMA_CHAIN_TIMEOUT phase=%0d aw=%0d w=%0d b=%0d", phase, dma_aw, dma_w, dma_b);
    end
endmodule
