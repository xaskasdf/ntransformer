`timescale 1ps/1ps
// Native H2C seeds application BRAM; native C2H returns it to a passive CQ sink.
module ntrpu_xdma_stop_inventory;
    reg hold_dma_b = 1'b0;
    reg hold_bypass_b = 1'b0;
    reg hold_dma_r = 1'b1;
    integer phase = 0;
    integer dma_aw = 0, dma_w = 0, dma_b = 0, dma_ar = 0, dma_r = 0;
    reg [511:0] host_memory = 0;
    reg [63:0] host_written = 0;
    integer host_bytes = 0, host_packets = 0;
    reg host_active = 0;
    integer i, lane, first_lane;
    reg [31:0] value;

    task require(input bit condition, input string reason);
        if (!condition) $fatal(1, "XDMA_C2H_ASSERT phase=%0d reason=%s", phase, reason);
    endtask

    task store_word(input integer address, input [31:0] data);
        integer j;
        for (j=0; j<4; j=j+1)
            board.RP.tx_usrapp.DATA_STORE[address+j] = data[j*8 +: 8];
    endtask

    task descriptor(input integer address, input [31:0] source, input [31:0] destination);
        integer j;
        begin
            for (j=0; j<32; j=j+1) board.RP.tx_usrapp.DATA_STORE[address+j] = 0;
            store_word(address, 32'had4b0013);
            store_word(address+4, 64);
            store_word(address+8, source);
            store_word(address+16, destination);
        end
    endtask

    task read_reg(input [31:0] address);
        begin
            board.RP.tx_usrapp.TSK_XDMA_REG_READ(address);
            value = board.RP.tx_usrapp.P_READ_DATA;
            require(!$isunknown(value), "known native register response");
            $display("XDMA_C2H_REG t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, value);
        end
    endtask

    task write_reg(input [31:0] address, input [31:0] data);
        begin
            $display("XDMA_C2H_WRITE t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, data);
            board.RP.tx_usrapp.TSK_XDMA_REG_WRITE(address, data, 4'hf);
        end
    endtask

    task stopped_registers(input bit busy_expected);
        begin
            read_reg(32'h1004); require(value == 32'h00fffe7e, "C2H Run cleared");
            read_reg(32'h1040);
            require(value[0] == busy_expected && (value & ~32'h47) == 0, "C2H Busy and error status");
            read_reg(32'h1048); require(value == (busy_expected ? 0 : 1), "C2H descriptor count");
            read_reg(32'h0004); require(value == 32'h00fffe7e, "H2C remains stopped");
            read_reg(32'h0040); require(value[0] == 0 && (value & ~32'h47) == 0, "H2C remains idle");
            read_reg(32'h0048); require(value == 1, "H2C count retained");
        end
    endtask

    task run_probe;
        begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1,
                    "native endpoint ready after enumeration");
            require(board.RP.tx_usrapp.xdma_bar == 1, "DMA BAR1");
            phase = 1;
            $display("XDMA_C2H_BEGIN t=%0t dma_bar=1 payload_bytes=64", $time);
            descriptor('h100, 'h400, 'h1000);
            descriptor('h200, 'h1000, 'h600);
            for (i=0; i<64; i=i+1)
                board.RP.tx_usrapp.DATA_STORE[1024+i] = (i*13) ^ 8'ha5;
            read_reg('h0040); require(value == 0, "initial H2C idle");
            read_reg('h1040); require(value == 0, "initial C2H idle");
            write_reg('h4080, 'h100); write_reg('h4084, 0); write_reg('h4088, 0);
            write_reg('h0004, 'h00fffe7f);
            wait (dma_b == 1);
            repeat (64) @(negedge board.EP.user_clk);
            read_reg('h0040); require(value[0] == 0 && (value & ~32'h47) == 0, "seed H2C idle");
            read_reg('h0048); require(value == 1, "seed H2C complete");
            write_reg('h000c, 1);
            read_reg('h0004); require(value == 'h00fffe7e, "seed H2C stopped");
            phase = 2;
            $display("XDMA_C2H_SEEDED t=%0t", $time);
            write_reg('h5080, 'h200); write_reg('h5084, 0); write_reg('h5088, 0);
            write_reg('h1004, 'h00fffe7f);
            wait (board.EP.ntrpu_app_dma_rvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(dma_ar == 1 && dma_r == 0 && host_bytes == 0, "one retained AXI read, no host receipt");
            $display("XDMA_C2H_PENDING t=%0t ar=%0d r=%0d host_bytes=%0d", $time, dma_ar, dma_r, host_bytes);
            read_reg('h1040); require(value[0] == 1 && (value & ~32'h47) == 0, "C2H Busy before stop");
            read_reg('h1048); require(value == 0, "C2H not complete before R");
            phase = 3;
            write_reg('h100c, 1);
            stopped_registers(1'b1);
            repeat (64) @(negedge board.EP.user_clk);
            stopped_registers(1'b1);
            require(dma_r == 0 && host_bytes == 0 && board.EP.ntrpu_app_dma_rvalid === 1'b1,
                    "R retained after Run readback zero");
            @(negedge board.EP.user_clk);
            phase = 4;
            hold_dma_r = 0;
            $display("XDMA_C2H_RELEASE t=%0t", $time);
            wait (dma_r == 1 && host_packets == 1);
            repeat (64) @(negedge board.EP.user_clk);
            stopped_registers(1'b0);
            repeat (1024) @(negedge board.EP.user_clk);
            stopped_registers(1'b0);
            require(dma_aw == 1 && dma_w == 1 && dma_b == 1 && dma_ar == 1 && dma_r == 1,
                    "exact DMA inventory after observation interval");
            require(host_packets == 1 && host_bytes == 64 && host_written == 64'hffffffffffffffff,
                    "complete host receipt");
            for (i=0; i<64; i=i+1)
                require(host_memory[i*8 +: 8] == (((i*13)^'ha5)&255), "host memory payload");
            phase = 5;
            $display("XDMA_C2H_PASS t=%0t payload_bytes=64 host_packets=%0d memory=%h reset_tested=0 global_drain_tested=0",
                     $time, host_packets, host_memory);
        end
    endtask

    always @(posedge board.EP.user_clk) begin
        if (phase != 0) begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1, "native readiness retained");
            require(!(board.EP.m_axib_awvalid || board.EP.m_axib_arvalid), "no BAR data traffic");
            if (board.EP.m_axi_awvalid && board.EP.m_axi_awready) begin
                dma_aw = dma_aw+1;
                require(dma_aw == 1, "one H2C address");
                $display("XDMA_C2H_AW t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axi_awid, board.EP.m_axi_awaddr, board.EP.m_axi_awlen, board.EP.m_axi_awsize);
            end
            if (board.EP.m_axi_wvalid && board.EP.m_axi_wready) begin
                dma_w = dma_w+1;
                require(dma_w == 1 && board.EP.m_axi_wlast === 1'b1, "one H2C beat");
                $display("XDMA_C2H_W t=%0t data=%h strb=%h last=%b", $time, board.EP.m_axi_wdata, board.EP.m_axi_wstrb, board.EP.m_axi_wlast);
            end
            if (board.EP.m_axi_bvalid && board.EP.m_axi_bready) begin
                dma_b = dma_b+1;
                require(dma_b == 1 && board.EP.m_axi_bresp === 2'b00, "one successful H2C response");
                $display("XDMA_C2H_B t=%0t id=%h resp=%h", $time, board.EP.m_axi_bid, board.EP.m_axi_bresp);
            end
            if (board.EP.m_axi_arvalid && board.EP.m_axi_arready) begin
                dma_ar = dma_ar+1;
                require(dma_ar == 1, "one C2H address");
                $display("XDMA_C2H_AR t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axi_arid, board.EP.m_axi_araddr, board.EP.m_axi_arlen, board.EP.m_axi_arsize);
            end
            if (board.EP.m_axi_rvalid && board.EP.m_axi_rready) begin
                dma_r = dma_r+1;
                require(dma_r == 1 && board.EP.m_axi_rresp === 2'b00 && board.EP.m_axi_rlast === 1'b1,
                        "one successful C2H read beat");
                $display("XDMA_C2H_R t=%0t id=%h data=%h resp=%h last=%b", $time, board.EP.m_axi_rid, board.EP.m_axi_rdata, board.EP.m_axi_rresp, board.EP.m_axi_rlast);
            end
        end
    end

    // Bounded, passive memory at host 0x600. Only accepted CQ payload bytes write it.
    // No endpoint state is read or driven; original AMD RX/logger remains unchanged.
    always @(posedge board.RP.user_clk) begin
        if (phase != 0 && board.RP.m_axis_cq_tvalid && board.RP.m_axis_cq_tready) begin
            first_lane = 0;
            if (!host_active) begin
                require(board.RP.m_axis_cq_tuser[81:80] === 2'b01, "CQ packet SOP");
                if (board.RP.m_axis_cq_tdata[78:75] == 4'h1) begin
                    require(phase == 4 && host_packets == 0 && host_bytes == 0, "bounded C2H host packet");
                    require(board.RP.m_axis_cq_tdata[63:0] == 64'h600 && board.RP.m_axis_cq_tdata[74:64] == 16,
                            "host destination and length");
                    require(board.RP.m_axis_cq_tuser[15:0] == 16'h0f0f, "full first/last byte enables");
                    host_active = 1;
                    first_lane = 16;
                    $display("XDMA_C2H_HOST_BEGIN t=%0t address=0000000000000600 bytes=64", $time);
                end else require(board.RP.m_axis_cq_tdata[78:75] == 0, "only expected host reads or writes");
            end else require(board.RP.m_axis_cq_tuser[81:80] === 2'b00, "no nested host packet");
            if (host_active) begin
                require(board.RP.m_axis_cq_tuser[96] === 1'b0, "no CQ discontinue");
                for (lane=0; lane<64; lane=lane+1) begin
                    if (board.RP.m_axis_cq_tuser[16+lane] === 1'b1) begin
                        require(lane >= first_lane && board.RP.m_axis_cq_tkeep[lane/4] === 1'b1 && host_bytes < 64,
                                "accepted enabled payload byte");
                        host_memory[host_bytes*8 +: 8] = board.RP.m_axis_cq_tdata[lane*8 +: 8];
                        host_written[host_bytes] = 1;
                        host_bytes = host_bytes+1;
                    end else require(board.RP.m_axis_cq_tuser[16+lane] === 1'b0, "known byte enable");
                end
                $display("XDMA_C2H_HOST_BEAT t=%0t bytes=%0d data=%h keep=%h last=%b user=%h", $time,
                         host_bytes, board.RP.m_axis_cq_tdata, board.RP.m_axis_cq_tkeep, board.RP.m_axis_cq_tlast, board.RP.m_axis_cq_tuser);
                if (board.RP.m_axis_cq_tlast) begin
                    require(host_bytes == 64, "host packet has all bytes");
                    host_active = 0;
                    host_packets = host_packets+1;
                    $display("XDMA_C2H_HOST_END t=%0t memory=%h", $time, host_memory);
                end
            end
        end
    end
    initial begin
        #200000000;
        $fatal(1, "XDMA_C2H_TIMEOUT phase=%0d ar=%0d r=%0d host_bytes=%0d", phase, dma_ar, dma_r, host_bytes);
    end
endmodule
