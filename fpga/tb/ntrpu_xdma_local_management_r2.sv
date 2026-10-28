`timescale 1ps/1ps
// Local XDMA register traffic under held BAR B, H2C B and C2H R. No reset or drain certificate.
module ntrpu_xdma_stop_inventory;
    reg hold_dma_b = 1'b1;
    reg hold_bypass_b = 1'b1;
    reg hold_dma_r = 1'b1;
    integer phase = 0;
    reg init_rvalid=0;
    integer init_ar_count=0, init_r_count=0;
    wire init_arready = board.EP.user_resetn && !init_rvalid && phase==0;

    // AMD TSK_XDMA_FIND_BAR probes BAR0 address0 before finding native BAR1.
    // A bounded public responder preserves the original zero-initialized RAM read.
    always @(posedge board.EP.user_clk) begin
        if (board.EP.user_resetn !== 1'b1) begin
            init_rvalid <= 0;
            init_ar_count <= 0;
            init_r_count <= 0;
        end else begin
            require(board.EP.m_axil_awvalid === 1'b0, "no BAR0 writes in discovery fixture");
            if (board.EP.m_axil_arvalid && init_arready) begin
                require(phase==0 && init_ar_count==0 && board.EP.m_axil_araddr===32'b0,
                        "one BAR0 address-zero discovery read only");
                init_rvalid <= 1;
                init_ar_count <= init_ar_count+1;
                $display("XDMA_LOCAL_DISCOVERY_AR t=%0t addr=%h",$time,board.EP.m_axil_araddr);
            end
            if (init_rvalid && board.EP.m_axil_rready) begin
                require(phase==0 && init_r_count==0,"one BAR0 discovery response only");
                init_rvalid <= 0;
                init_r_count <= init_r_count+1;
                $display("XDMA_LOCAL_DISCOVERY_R t=%0t data=00000000",$time);
            end
        end
    end
    integer dma_aw = 0, dma_w = 0, dma_b = 0, dma_ar = 0, dma_r = 0;
    reg [511:0] host_memory = 0;
    reg [63:0] host_written = 0;
    integer host_bytes = 0, host_packets = 0;
    reg host_active = 0;
    integer i, lane, first_lane;
    reg [31:0] value;


    reg [31:0] local_awaddr=0, local_wdata=0, local_araddr=0;
    reg [2:0] local_awprot=0, local_arprot=0;
    reg [3:0] local_wstrb=0;
    reg local_awvalid=0, local_wvalid=0, local_bready=0, local_arvalid=0, local_rready=0;
    wire local_awready, local_wready, local_bvalid, local_arready, local_rvalid;
    wire [1:0] local_bresp, local_rresp;
    wire [31:0] local_rdata;
    wire [7:0] h2c_status, c2h_status;
    reg [31:0] local_value;
    reg [63:0] bypass_base;
    integer bypass_aw=0, bypass_w=0, bypass_b=0, bypass_ar=0, bypass_r=0;
    integer local_reads=0, local_writes=0, h2c_done_events=0, c2h_done_events=0;
    reg [15:0] previous_status=16'hffff;

    task require(input bit condition, input string reason);
        if (!condition) $fatal(1, "XDMA_LOCAL_ASSERT phase=%0d reason=%s", phase, reason);
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
            $display("XDMA_LOCAL_REG t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, value);
        end
    endtask

    task write_reg(input [31:0] address, input [31:0] data);
        begin
            $display("XDMA_LOCAL_WRITE t=%0t phase=%0d address=%08h value=%08h", $time, phase, address, data);
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


    task local_read(input [31:0] address);
        integer cycles;
        begin
            @(negedge board.EP.user_clk);
            local_araddr=address; local_arvalid=1; cycles=0;
            $display("XDMA_LOCAL_READ_BEGIN t=%0t phase=%0d address=%h", $time, phase, address);
            do begin
                @(posedge board.EP.user_clk); cycles=cycles+1;
                require(cycles<2048, "local AR handshake timeout");
            end while (local_arready !== 1'b1);
            @(negedge board.EP.user_clk); local_arvalid=0;
            // Hold READY low until the native response is actually visible.
            cycles=0;
            while(local_rvalid !== 1'b1) begin
                @(negedge board.EP.user_clk);cycles=cycles+1;
                require(cycles<2048,"local RVALID arrival timeout");
            end
            repeat (3) @(negedge board.EP.user_clk);
            local_rready=1; cycles=0;
            do begin
                @(posedge board.EP.user_clk); cycles=cycles+1;
                require(cycles<2048, "local R response timeout");
            end while (local_rvalid !== 1'b1);
            require(local_rresp === 2'b00 && !$isunknown(local_rdata), "known successful local register read");
            local_value=local_rdata; local_reads=local_reads+1;
            $display("XDMA_LOCAL_READ_END t=%0t phase=%0d address=%h value=%h", $time, phase, address, local_value);
            @(negedge board.EP.user_clk); local_rready=0;
        end
    endtask

    task local_write(input [31:0] address, input [31:0] data);
        integer cycles;
        bit aw_done, w_done;
        begin
            @(negedge board.EP.user_clk);
            local_awaddr=address; local_awvalid=1; local_wdata=data;
            local_wstrb=4'hf; local_wvalid=1; aw_done=0; w_done=0; cycles=0;
            $display("XDMA_LOCAL_WRITE_BEGIN t=%0t phase=%0d address=%h value=%h", $time, phase, address, data);
            while (!aw_done || !w_done) begin
                @(posedge board.EP.user_clk); cycles=cycles+1;
                require(cycles<2048, "local AW/W handshake timeout");
                if (local_awvalid && local_awready === 1'b1) aw_done=1;
                if (local_wvalid && local_wready === 1'b1) w_done=1;
                @(negedge board.EP.user_clk);
                if (aw_done) local_awvalid=0;
                if (w_done) local_wvalid=0;
            end
            cycles=0;
            while(local_bvalid !== 1'b1) begin
                @(negedge board.EP.user_clk);cycles=cycles+1;
                require(cycles<2048,"local BVALID arrival timeout");
            end
            repeat (3) @(negedge board.EP.user_clk);
            local_bready=1; cycles=0;
            do begin
                @(posedge board.EP.user_clk); cycles=cycles+1;
                require(cycles<2048, "local B response timeout");
            end while (local_bvalid !== 1'b1);
            require(local_bresp === 2'b00, "successful local register write");
            local_writes=local_writes+1;
            $display("XDMA_LOCAL_WRITE_END t=%0t phase=%0d address=%h value=%h", $time, phase, address, data);
            @(negedge board.EP.user_clk); local_bready=0;
        end
    endtask

    task local_channel(input [31:0] base, input bit busy_expected, input integer count_expected);
        begin
            local_read(base+'h4); require(local_value == 32'h00fffe7e, "local Run cleared with configuration retained");
            local_read(base+'h40);
            require(local_value[0] == busy_expected && (local_value & ~32'h47) == 0, "local Busy and no channel errors");
            local_read(base+'h48); require(local_value == count_expected, "local completed descriptor count");
        end
    endtask

    task bar_readback;
        begin
            fork
                board.RP.tx_usrapp.TSK_TX_MEMORY_READ_64(board.RP.tx_usrapp.DEFAULT_TAG, 3'b000,
                    11'd1, bypass_base+'h2000, 4'h0, 4'hf);
                board.RP.tx_usrapp.TSK_WAIT_FOR_READ_DATA;
            join
            board.RP.tx_usrapp.DEFAULT_TAG=board.RP.tx_usrapp.DEFAULT_TAG+1;
            require(board.RP.tx_usrapp.P_READ_DATA === 32'hd3c2b1a0, "BAR destination readback");
            $display("XDMA_LOCAL_BAR_RECEIPT t=%0t data=%h", $time, board.RP.tx_usrapp.P_READ_DATA);
        end
    endtask

    task run_probe;
        begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1, "native endpoint ready");
            require(board.RP.tx_usrapp.xdma_bar == 1 && board.RP.tx_usrapp.BAR_INIT_P_BAR_ENABLED[2] == 2'b11, "native BAR layout");
            bypass_base={board.RP.tx_usrapp.BAR_INIT_P_BAR[3][31:0],board.RP.tx_usrapp.BAR_INIT_P_BAR[2][31:0]};
            require(init_ar_count==1 && init_r_count==1 && !init_rvalid,"BAR0 discovery completed before probe");
            phase=1;
            $display("XDMA_LOCAL_BEGIN t=%0t bypass_base=%h", $time, bypass_base);
            local_read('h0004); require(local_value == 0, "initial local H2C control");
            local_read('h0040); require(local_value == 0, "initial local H2C status");
            local_read('h0048); require(local_value == 0, "initial local H2C count");
            local_read('h1004); require(local_value == 0, "initial local C2H control");
            local_read('h1040); require(local_value == 0, "initial local C2H status");
            local_read('h1048); require(local_value == 0, "initial local C2H count");
            write_reg('h0004,'h00fffe7e);
            read_reg('h0004); require(value == 'h00fffe7e, "PCIe setup readback before local comparison");
            local_read('h0004); require(local_value == value, "PCIe write observed locally");
            local_write('h1004,'h00fffe7e);
            read_reg('h1004); require(value == 'h00fffe7e, "local write observed through PCIe");

            phase=2;
            for (i=0;i<4;i=i+1) board.RP.tx_usrapp.DATA_STORE[i]=32'hd3c2b1a0 >> (i*8);
            board.RP.tx_usrapp.TSK_TX_MEMORY_WRITE_64(board.RP.tx_usrapp.DEFAULT_TAG,3'b000,
                11'd1,bypass_base+'h2000,4'h0,4'hf,1'b0);
            board.RP.tx_usrapp.DEFAULT_TAG=board.RP.tx_usrapp.DEFAULT_TAG+1;
            wait(board.EP.ntrpu_app_bypass_bvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(bypass_aw==1 && bypass_w==1 && bypass_b==0, "actual BAR B retained before local access");
            $display("XDMA_LOCAL_BAR_PENDING t=%0t",$time);
            local_read('h0040); require(local_value==0, "H2C status during held BAR B");
            local_read('h1040); require(local_value==0, "C2H status during held BAR B");
            local_write('h000c,2); local_write('h100c,2);
            local_read('h0004); require(local_value=='h00fffe7c, "H2C W1C while BAR B held");
            local_read('h1004); require(local_value=='h00fffe7c, "C2H W1C while BAR B held");
            require(bypass_b==0 && board.EP.ntrpu_app_bypass_bvalid === 1'b1, "local operations completed before release");
            @(negedge board.EP.user_clk); hold_bypass_b=0;
            $display("XDMA_LOCAL_BAR_RELEASE t=%0t",$time);
            wait(bypass_b==1); bar_readback;
            read_reg('h0004); require(value=='h00fffe7c, "H2C local W1C PCIe confirmation");
            read_reg('h1004); require(value=='h00fffe7c, "C2H local W1C PCIe confirmation");

            phase=3;
            descriptor('h100,'h400,'h1000); descriptor('h200,'h1000,'h600);
            for(i=0;i<64;i=i+1) board.RP.tx_usrapp.DATA_STORE[1024+i]=(i*13)^8'ha5;
            local_write('h4080,'h100); local_write('h4084,0); local_write('h4088,0);
            local_write('h0004,'h00fffe7f);
            wait(board.EP.ntrpu_app_dma_bvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(dma_aw==1 && dma_w==1 && dma_b==0, "H2C B actually pending");
            $display("XDMA_LOCAL_H2C_PENDING t=%0t",$time);
            local_read('h0040); require(local_value[0]==1 && (local_value&~32'h47)==0,"H2C busy before stop");
            local_write('h000c,1); local_channel(0,1,0);
            require(h2c_status[6]===1'b0 && h2c_status[0]===1'b1 && dma_b==0,"live stopped H2C remains busy");
            phase=4; @(negedge board.EP.user_clk); hold_dma_b=0;
            $display("XDMA_LOCAL_H2C_RELEASE t=%0t",$time);
            wait(dma_b==1); repeat(64) @(negedge board.EP.user_clk);
            local_channel(0,0,1);
            read_reg('h0040); require(value[0]==0 && (value&~32'h47)==0,"native H2C idle confirmation");
            read_reg('h0048); require(value==1,"native H2C count confirmation");

            phase=5;
            local_write('h5080,'h200); local_write('h5084,0); local_write('h5088,0);
            local_write('h1004,'h00fffe7f);
            wait(board.EP.ntrpu_app_dma_rvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(dma_ar==1 && dma_r==0 && host_bytes==0,"C2H R actually pending without host receipt");
            $display("XDMA_LOCAL_C2H_PENDING t=%0t",$time);
            local_read('h1040); require(local_value[0]==1 && (local_value&~32'h47)==0,"C2H busy before stop");
            local_write('h100c,1); local_channel('h1000,1,0); local_channel(0,0,1);
            require(c2h_status[6]===1'b0 && c2h_status[0]===1'b1 && dma_r==0 && host_bytes==0,"stopped C2H still retains R");
            phase=6; @(negedge board.EP.user_clk); hold_dma_r=0;
            $display("XDMA_LOCAL_C2H_RELEASE t=%0t",$time);
            wait(dma_r==1 && host_packets==1);
            repeat(64) @(negedge board.EP.user_clk);
            local_channel(0,0,1); local_channel('h1000,0,1); stopped_registers(0);
            repeat(128) @(negedge board.EP.user_clk);
            require(dma_aw==1 && dma_w==1 && dma_b==1 && dma_ar==1 && dma_r==1,"complete DMA interface inventory");
            require(bypass_aw==1 && bypass_w==1 && bypass_b==1 && bypass_ar==1 && bypass_r==1,"complete BAR inventory");
            require(host_packets==1 && host_bytes==64 && host_written==64'hffffffffffffffff,"all C2H bytes received");
            require(h2c_done_events==1 && c2h_done_events==1,"one native descriptor-done pulse per channel");
            for(i=0;i<64;i=i+1) require(host_memory[i*8 +: 8]==(((i*13)^'ha5)&255),"host payload compared");
            phase=7;
            $display("XDMA_LOCAL_PASS t=%0t local_reads=%0d local_writes=%0d host_bytes=%0d dma_bytes=64 bar_bytes=4 reset_tested=0 global_drain_tested=0",
                     $time,local_reads,local_writes,host_bytes);
        end
    endtask

    always @(posedge board.EP.user_clk) begin
        if (phase != 0) begin
            require(board.EP.user_resetn === 1'b1 && board.EP.user_lnk_up === 1'b1, "native readiness retained");
            require(board.EP.m_axil_awvalid === 1'b0 && board.EP.m_axil_arvalid === 1'b0,
                    "BAR0 receives no request after discovery");

            require(!$isunknown({h2c_status,c2h_status}), "known live native status ports");
            if(h2c_status[3]) h2c_done_events=h2c_done_events+1;
            if(c2h_status[3]) c2h_done_events=c2h_done_events+1;
            if({h2c_status,c2h_status} != previous_status) begin
                $display("XDMA_LOCAL_STATUS t=%0t phase=%0d h2c=%h c2h=%h h2c_done=%0d c2h_done=%0d",
                         $time,phase,h2c_status,c2h_status,h2c_done_events,c2h_done_events);
                previous_status={h2c_status,c2h_status};
            end
            if(board.EP.m_axib_awvalid && board.EP.m_axib_awready) bypass_aw=bypass_aw+1;
            if(board.EP.m_axib_wvalid && board.EP.m_axib_wready) bypass_w=bypass_w+1;
            if(board.EP.m_axib_bvalid && board.EP.m_axib_bready) begin
                bypass_b=bypass_b+1; require(board.EP.m_axib_bresp===2'b00,"BAR B okay");
            end
            if(board.EP.m_axib_arvalid && board.EP.m_axib_arready) bypass_ar=bypass_ar+1;
            if(board.EP.m_axib_rvalid && board.EP.m_axib_rready) begin
                bypass_r=bypass_r+1; require(board.EP.m_axib_rresp===2'b00 && board.EP.m_axib_rlast===1'b1,"BAR R okay");
            end
            if (board.EP.m_axi_awvalid && board.EP.m_axi_awready) begin
                dma_aw = dma_aw+1;
                require(dma_aw == 1, "one H2C address");
                $display("XDMA_LOCAL_AW t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axi_awid, board.EP.m_axi_awaddr, board.EP.m_axi_awlen, board.EP.m_axi_awsize);
            end
            if (board.EP.m_axi_wvalid && board.EP.m_axi_wready) begin
                dma_w = dma_w+1;
                require(dma_w == 1 && board.EP.m_axi_wlast === 1'b1, "one H2C beat");
                $display("XDMA_LOCAL_W t=%0t data=%h strb=%h last=%b", $time, board.EP.m_axi_wdata, board.EP.m_axi_wstrb, board.EP.m_axi_wlast);
            end
            if (board.EP.m_axi_bvalid && board.EP.m_axi_bready) begin
                dma_b = dma_b+1;
                require(dma_b == 1 && board.EP.m_axi_bresp === 2'b00, "one successful H2C response");
                $display("XDMA_LOCAL_B t=%0t id=%h resp=%h", $time, board.EP.m_axi_bid, board.EP.m_axi_bresp);
            end
            if (board.EP.m_axi_arvalid && board.EP.m_axi_arready) begin
                dma_ar = dma_ar+1;
                require(dma_ar == 1, "one C2H address");
                $display("XDMA_LOCAL_AR t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axi_arid, board.EP.m_axi_araddr, board.EP.m_axi_arlen, board.EP.m_axi_arsize);
            end
            if (board.EP.m_axi_rvalid && board.EP.m_axi_rready) begin
                dma_r = dma_r+1;
                require(dma_r == 1 && board.EP.m_axi_rresp === 2'b00 && board.EP.m_axi_rlast === 1'b1,
                        "one successful C2H read beat");
                $display("XDMA_LOCAL_R t=%0t id=%h data=%h resp=%h last=%b", $time, board.EP.m_axi_rid, board.EP.m_axi_rdata, board.EP.m_axi_rresp, board.EP.m_axi_rlast);
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
                    require(phase == 6 && host_packets == 0 && host_bytes == 0, "bounded C2H host packet");
                    require(board.RP.m_axis_cq_tdata[63:0] == 64'h600 && board.RP.m_axis_cq_tdata[74:64] == 16,
                            "host destination and length");
                    require(board.RP.m_axis_cq_tuser[15:0] == 16'h0f0f, "full first/last byte enables");
                    host_active = 1;
                    first_lane = 16;
                    $display("XDMA_LOCAL_HOST_BEGIN t=%0t address=0000000000000600 bytes=64", $time);
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
                $display("XDMA_LOCAL_HOST_BEAT t=%0t bytes=%0d data=%h keep=%h last=%b user=%h", $time,
                         host_bytes, board.RP.m_axis_cq_tdata, board.RP.m_axis_cq_tkeep, board.RP.m_axis_cq_tlast, board.RP.m_axis_cq_tuser);
                if (board.RP.m_axis_cq_tlast) begin
                    require(host_bytes == 64, "host packet has all bytes");
                    host_active = 0;
                    host_packets = host_packets+1;
                    $display("XDMA_LOCAL_HOST_END t=%0t memory=%h", $time, host_memory);
                end
            end
        end
    end
    initial begin
        #240000000;
        $fatal(1, "XDMA_LOCAL_TIMEOUT phase=%0d ar=%0d r=%0d host_bytes=%0d", phase, dma_ar, dma_r, host_bytes);
    end
endmodule
