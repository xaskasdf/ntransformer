`timescale 1ps/1ps
// Native prefetched H2C chain: two publications, independently held B responses. No global certificate.
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


    wire [31:0] local_awaddr, local_wdata, local_araddr;
    wire [2:0] local_awprot, local_arprot;
    wire [3:0] local_wstrb;
    wire local_awvalid, local_wvalid, local_bready, local_arvalid, local_rready;
    wire local_awready, local_wready, local_bvalid, local_arready, local_rvalid;
    wire [1:0] local_bresp, local_rresp;
    wire [31:0] local_rdata;
    wire [7:0] h2c_status, c2h_status;
    reg [31:0] local_value;
    reg [63:0] bypass_base;
    integer bypass_aw=0, bypass_w=0, bypass_b=0, bypass_ar=0, bypass_r=0;
    integer local_reads=0, local_writes=0, h2c_done_events=0, c2h_done_events=0;
    reg [15:0] previous_status=16'hffff;

    // Simulation host producer: complete each descriptor in public DATA_STORE,
    // write its control word last, then notify the endpoint-domain ledger.
    // The engines remain stopped during publication. This fixture is not a
    // cache-coherent driver, a CDC bridge, or a global retirement certificate.
    reg ledger_arm_valid=0,ledger_publish_valid=0,ledger_publish_channel=0;
    reg ledger_close_request=0,ledger_host_closed=0;
    reg [63:0] ledger_host_generation=0,ledger_host_tag=0;
    reg [31:0] ledger_publish_index=0,ledger_publish_address=0;
    reg [255:0] ledger_publish_image=0;
    wire [255:0] ledger_first_image,ledger_second_image;
    wire ledger_arm_ready,ledger_active,ledger_closed,ledger_accounted;
    wire [63:0] ledger_generation,ledger_tag,ledger_published,ledger_completed,ledger_outstanding;
    wire [1:0] ledger_started;
    wire [8:0] ledger_faults;
    wire ledger_ready=board.EP.user_resetn && board.EP.user_lnk_up;
    for(genvar descriptor_byte=0;descriptor_byte<32;descriptor_byte=descriptor_byte+1) begin
        assign ledger_first_image[descriptor_byte*8 +: 8]=board.RP.tx_usrapp.DATA_STORE[256+descriptor_byte];
        assign ledger_second_image[descriptor_byte*8 +: 8]=board.RP.tx_usrapp.DATA_STORE[288+descriptor_byte];
    end
    ntrpu_xdma_descriptor_ledger #(.COUNTER_BITS(32)) descriptor_ledger (
        .clk(board.EP.user_clk),.cold_rst_n(board.EP.user_resetn),.core_ready(ledger_ready),
        .arm_valid(ledger_arm_valid),.arm_ready(ledger_arm_ready),
        .arm_generation(64'd31),.arm_tag(64'd301),
        .publish_valid(ledger_publish_valid),.publish_channel(ledger_publish_channel),
        .publish_generation(64'd31),.publish_tag(64'd301),.publish_index(ledger_publish_index),
        .h2c_status(h2c_status),.c2h_status(c2h_status),
        .close_request(ledger_close_request),.host_closed(ledger_host_closed),
        .host_generation(ledger_host_generation),.host_tag(ledger_host_tag),.release_valid(1'b0),
        .active(ledger_active),.closed(ledger_closed),.generation(ledger_generation),.tag(ledger_tag),
        .started(ledger_started),.faults(ledger_faults),.published(ledger_published),
        .completed(ledger_completed),.outstanding(ledger_outstanding),.accounted(ledger_accounted)
    );
    always @(posedge board.EP.user_clk) if(ledger_active) begin
        if(ledger_publish_valid) begin
            require(ledger_publish_image===(ledger_publish_index==1?ledger_second_image:ledger_first_image),
                    "notification follows complete visible descriptor bytes");
            $display("XDMA_LEDGER_PUBLISH t=%0t channel=%b index=%h address=%h image=%h generation=%h tag=%h",
                $time,ledger_publish_channel,ledger_publish_index,ledger_publish_address,ledger_publish_image,
                ledger_generation,ledger_tag);
        end
        if(ledger_published[31:0]!=0)
            require(ledger_first_image===256'h00000000000001200000000000001000000000000000040000000040ad4b0012,
                    "published H2C descriptor remains immutable");
        if(ledger_published[31:0]>=2)
            require(ledger_second_image===256'h00000000000000000000000000001100000000000000050000000040ad4b0013,
                    "second published H2C descriptor remains immutable");
    end

    reg fixture_admit=1, fixture_req_valid=0, fixture_req_write=0, fixture_rsp_ready=0;
    reg [31:0] fixture_req_address=0, fixture_req_data=0;
    reg [3:0] fixture_req_strobe=0;
    reg [127:0] fixture_req_identity=0;
    wire master_req_ready,master_rsp_valid,master_rsp_timeout,master_timeout_latched,master_busy;
    wire [31:0] master_rsp_data;
    wire [1:0] master_rsp_code;
    wire [127:0] master_rsp_identity;
    integer master_serial=0,master_requests=0,master_reports=0;
    reg [31:0] active_local_address=0,active_local_data=0;
    // Fixture owns setup only. Handoff occurs at idle and never returns.
    // The sequencer, production master and ledger then control public ports.
    reg sequencer_owner=0,seq_start_valid=0;
    wire seq_start_ready,seq_active,seq_snapshot_valid,seq_pending;
    wire [63:0] seq_generation,seq_tag,seq_expected,seq_control,seq_status,seq_count;
    wire [13:0] seq_faults;
    wire [1:0] seq_stops;
    wire [5:0] seq_reads;
    wire seq_admit,seq_req_valid,seq_req_write,seq_rsp_ready;
    wire [31:0] seq_req_address,seq_req_data;
    wire [3:0] seq_req_strobe;
    wire [127:0] seq_req_identity;
    wire master_admit=sequencer_owner ? seq_admit : fixture_admit;
    wire master_req_valid=sequencer_owner ? seq_req_valid : fixture_req_valid;
    wire master_req_write=sequencer_owner ? seq_req_write : fixture_req_write;
    wire master_rsp_ready=sequencer_owner ? seq_rsp_ready : fixture_rsp_ready;
    wire [31:0] master_req_address=sequencer_owner ? seq_req_address : fixture_req_address;
    wire [31:0] master_req_data=sequencer_owner ? seq_req_data : fixture_req_data;
    wire [3:0] master_req_strobe=sequencer_owner ? seq_req_strobe : fixture_req_strobe;
    wire [127:0] master_req_identity=sequencer_owner ? seq_req_identity : fixture_req_identity;
    integer seq_reports=0,seq_requests=0;
    reg seq_previous_snapshot=0;
    ntrpu_xdma_stop_registers stop_registers (
        .clk(board.EP.user_clk),.cold_rst_n(board.EP.user_resetn),.core_ready(ledger_ready),
        .exclusive_owner(sequencer_owner),.external_fault(1'b0),.wait_expired(1'b0),
        .start_valid(seq_start_valid),.start_ready(seq_start_ready),.start_generation(64'd31),.start_tag(64'd301),
        .close_request(ledger_close_request),.host_closed(ledger_host_closed),
        .host_generation(ledger_host_generation),.host_tag(ledger_host_tag),
        .ledger_active(ledger_active),.ledger_closed(ledger_closed),.ledger_accounted(ledger_accounted),
        .ledger_generation(ledger_generation),.ledger_tag(ledger_tag),.ledger_published(ledger_published),
        .ledger_completed(ledger_completed),.ledger_faults(ledger_faults),.release_valid(1'b0),
        .active(seq_active),.generation(seq_generation),.tag(seq_tag),.expected_published(seq_expected),
        .faults(seq_faults),.snapshot_valid(seq_snapshot_valid),.control_snapshot(seq_control),
        .status_snapshot(seq_status),.count_snapshot(seq_count),.stop_responses(seq_stops),.read_responses(seq_reads),
        .transaction_pending(seq_pending),.m_admit(seq_admit),.m_req_valid(seq_req_valid),
        .m_req_ready(master_req_ready && sequencer_owner),.m_req_write(seq_req_write),.m_req_address(seq_req_address),
        .m_req_data(seq_req_data),.m_req_strobe(seq_req_strobe),.m_req_identity(seq_req_identity),
        .m_rsp_valid(master_rsp_valid && sequencer_owner),.m_rsp_ready(seq_rsp_ready),.m_rsp_data(master_rsp_data),
        .m_rsp_code(master_rsp_code),.m_rsp_identity(master_rsp_identity),.m_rsp_timeout(master_rsp_timeout),
        .m_timeout_latched(master_timeout_latched)
    );
    always @(posedge board.EP.user_clk) if(sequencer_owner) begin
        if(master_req_valid && master_req_ready) seq_requests<=seq_requests+1;
        if(master_rsp_valid && master_rsp_ready) seq_reports<=seq_reports+1;
        if(phase<7) require(seq_faults===14'd0,"sequencer has no error before deliberate revocation");
        if(seq_snapshot_valid && !seq_previous_snapshot)
            $display("XDMA_STOP_REG_SNAPSHOT t=%0t reports=%0d control=%h status=%h count=%h host_bytes=%0d global_drain=0",
                     $time,seq_reports,seq_control,seq_status,seq_count,host_bytes);
        seq_previous_snapshot<=seq_snapshot_valid;
    end
    ntrpu_xdma_local_master #(.WAIT_LIMIT(65536)) management_master (
        .clk(board.EP.user_clk),.cold_rst_n(board.EP.user_resetn),.admit(master_admit),
        .req_valid(master_req_valid),.req_ready(master_req_ready),.req_write(master_req_write),
        .req_address(master_req_address),.req_data(master_req_data),.req_strobe(master_req_strobe),
        .req_identity(master_req_identity),.rsp_valid(master_rsp_valid),.rsp_ready(master_rsp_ready),
        .rsp_data(master_rsp_data),.rsp_code(master_rsp_code),.rsp_identity(master_rsp_identity),
        .rsp_timeout(master_rsp_timeout),.timeout_latched(master_timeout_latched),.busy(master_busy),
        .m_axil_awaddr(local_awaddr),
        .m_axil_awprot(local_awprot),
        .m_axil_awvalid(local_awvalid),
        .m_axil_awready(local_awready),
        .m_axil_wdata(local_wdata),
        .m_axil_wstrb(local_wstrb),
        .m_axil_wvalid(local_wvalid),
        .m_axil_wready(local_wready),
        .m_axil_bresp(local_bresp),
        .m_axil_bvalid(local_bvalid),
        .m_axil_bready(local_bready),
        .m_axil_araddr(local_araddr),
        .m_axil_arprot(local_arprot),
        .m_axil_arvalid(local_arvalid),
        .m_axil_arready(local_arready),
        .m_axil_rdata(local_rdata),
        .m_axil_rresp(local_rresp),
        .m_axil_rvalid(local_rvalid),
        .m_axil_rready(local_rready)
    );
    always @(posedge board.EP.user_clk) if(board.EP.user_resetn && phase>0) begin
        require(!master_timeout_latched,"native management must not time out");
        if(master_req_valid && master_req_ready) begin
            master_requests<=master_requests+1;
            active_local_address<=master_req_address;active_local_data<=master_req_data;
            $display("XDMA_MASTER_REQ t=%0t phase=%0d write=%b address=%h data=%h strobe=%h identity=%h",$time,phase,master_req_write,master_req_address,master_req_data,master_req_strobe,master_req_identity);
        end
        if(local_rvalid && local_rready) begin
            local_reads<=local_reads+1;
            $display("XDMA_LOCAL_READ_END t=%0t phase=%0d address=%h value=%h",$time,phase,active_local_address,local_rdata);
        end
        if(local_bvalid && local_bready) begin
            local_writes<=local_writes+1;
            $display("XDMA_LOCAL_WRITE_END t=%0t phase=%0d address=%h value=%h",$time,phase,active_local_address,active_local_data);
        end
        if(master_rsp_valid && master_rsp_ready) begin
            master_reports<=master_reports+1;
            $display("XDMA_MASTER_REPORT t=%0t phase=%0d data=%h code=%h identity=%h timeout=%b",$time,phase,master_rsp_data,master_rsp_code,master_rsp_identity,master_rsp_timeout);
        end
    end

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
            @(negedge board.EP.user_clk);
            require(ledger_active && ledger_faults==0 && h2c_status[6]==0 && c2h_status[6]==0,
                    "descriptors published only with armed stopped engines");
            for (j=0; j<32; j=j+1) board.RP.tx_usrapp.DATA_STORE[address+j] = 0;
            store_word(address+4, 64);
            store_word(address+8, source);
            store_word(address+16, destination);
            store_word(address+24,(address == 'h100) ? 'h120 : 0);
            store_word(address,(address == 'h100) ? 32'had4b0012 : 32'had4b0013);
            ledger_publish_channel=0;ledger_publish_index=((address == 'h120) ? 1 : 0);
            ledger_publish_address=address;
            for(j=0;j<32;j=j+1) ledger_publish_image[j*8 +: 8]=board.RP.tx_usrapp.DATA_STORE[address+j];
            ledger_publish_valid=1;
            @(negedge board.EP.user_clk);ledger_publish_valid=0;
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


    task master_transaction(input bit write_request,input [31:0] address,input [31:0] data);
        integer cycles;
        reg [127:0] identity;
        begin
            @(negedge board.EP.user_clk);
            master_serial=master_serial+1;
            identity={64'h4e545250554d4754,32'(phase),32'(master_serial)};

            fixture_req_write=write_request;fixture_req_address=address;fixture_req_data=data;
            fixture_req_strobe=write_request?4'hf:4'h0;fixture_req_identity=identity;fixture_req_valid=1;
            if(write_request) $display("XDMA_LOCAL_WRITE_BEGIN t=%0t phase=%0d address=%h value=%h",$time,phase,address,data);
            else $display("XDMA_LOCAL_READ_BEGIN t=%0t phase=%0d address=%h",$time,phase,address);
            cycles=0;
            do begin
                @(posedge board.EP.user_clk);cycles=cycles+1;
                require(cycles<2048,"master request timeout");
            end while(master_req_ready!==1'b1);
            @(negedge board.EP.user_clk);fixture_req_valid=0;fixture_admit=0;
            fixture_req_address='hffffffff;fixture_req_data='hbad0bad0;
            fixture_req_strobe=0;fixture_req_identity=0;
            cycles=0;
            while(master_rsp_valid!==1'b1) begin
                @(negedge board.EP.user_clk);cycles=cycles+1;
                require(cycles<2048,"master report arrival timeout");
            end
            repeat(3) begin
                require(master_rsp_valid && master_rsp_identity===identity && master_rsp_code===0 &&
                        !master_rsp_timeout && !master_timeout_latched && master_busy && !master_req_ready,
                        "actual retained native master report");
                @(negedge board.EP.user_clk);
            end
            fixture_rsp_ready=1;
            @(posedge board.EP.user_clk);
            require(master_rsp_valid && master_rsp_identity===identity && master_rsp_code===0 &&
                    !master_rsp_timeout && !$isunknown(master_rsp_data),"native master response identity");
            local_value=master_rsp_data;
            @(negedge board.EP.user_clk);fixture_rsp_ready=0;fixture_admit=1;
        end
    endtask
    task local_read(input [31:0] address);
        master_transaction(0,address,0);
    endtask
    task local_write(input [31:0] address,input [31:0] data);
        master_transaction(1,address,data);
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
            @(negedge board.EP.user_clk);
            require(ledger_arm_ready,"native idle channels admit fresh ledger session");
            ledger_arm_valid=1;
            @(negedge board.EP.user_clk);ledger_arm_valid=0;
            require(ledger_active && ledger_generation==31 && ledger_tag==301,"native ledger armed");
            $display("XDMA_LEDGER_ARMED t=%0t generation=%h tag=%h",$time,ledger_generation,ledger_tag);
            for(i=0;i<64;i=i+1) board.RP.tx_usrapp.DATA_STORE[1024+i]=(i*13)^8'ha5;
            for(i=0;i<64;i=i+1) board.RP.tx_usrapp.DATA_STORE[1280+i]=(i*7)^8'h3c;
            descriptor('h100,'h400,'h1000); descriptor('h120,'h500,'h1100);
            require(ledger_published==64'd2 && ledger_completed==0,"two actual descriptor publications");
            local_write('h4080,'h100); local_write('h4084,0); local_write('h4088,1);
            local_write('h0004,'h00fffe7f);
            wait(board.EP.ntrpu_app_dma_bvalid === 1'b1);
            @(negedge board.EP.user_clk);
            require(dma_aw>=1 && dma_aw<=2 && dma_w>=1 && dma_w<=2 && dma_b==0, "first H2C B actually pending");
            $display("XDMA_LOCAL_H2C_PENDING t=%0t",$time);
            // The adjacent descriptor has already been published; stopping
            // Run cannot cancel its outstanding obligation or prefetched bytes.
            ledger_close_request=1;ledger_host_closed=1;ledger_host_generation=31;ledger_host_tag=301;
            @(negedge board.EP.user_clk);
            require(ledger_closed && !ledger_accounted && ledger_outstanding==64'd2,
                    "closure preserves both pending descriptor obligations");
            require(!master_busy && !master_rsp_valid && !fixture_req_valid,"exclusive ownership handoff at idle");
            sequencer_owner=1;phase=4;
            @(negedge board.EP.user_clk);require(seq_start_ready,"closed coherent session admits RTL sequencer");
            seq_start_valid=1;
            @(negedge board.EP.user_clk);seq_start_valid=0;
            $display("XDMA_STOP_REG_START t=%0t published=%h outstanding=%h",$time,ledger_published,ledger_outstanding);
            wait(seq_reports>=8);
            @(negedge board.EP.user_clk);
            require(seq_stops===2'b11 && !seq_snapshot_valid && !ledger_accounted && dma_b==0 && dma_r==0,
                    "both real W1C responses and first complete poll precede releasing either DMA response");
            require(h2c_status[6]===0 && c2h_status[6]===0 && h2c_status[0]===1 && c2h_status[0]===0,
                    "stopped H2C remains busy; unused C2H remains idle");
            require(seq_count===0 && ledger_outstanding===2,"zero completion cannot qualify two publications");
            $display("XDMA_STOP_CHAIN_ZERO t=%0t reports=%0d count=%h outstanding=%h",$time,seq_reports,seq_count,ledger_outstanding);
            phase=5;hold_dma_b=0;
            $display("XDMA_LOCAL_H2C_RELEASE t=%0t",$time);
            wait(dma_b==1);
            @(negedge board.EP.user_clk);hold_dma_b=1;
            $display("XDMA_STOP_CHAIN_REHOLD t=%0t",$time);
            wait(board.EP.ntrpu_app_dma_bvalid===1'b1);
            @(negedge board.EP.user_clk);
            require(dma_aw==2 && dma_w==2 && dma_b==1,"second application B actually pending");
            $display("XDMA_STOP_CHAIN_SECOND_PENDING t=%0t",$time);
            wait(seq_reports>=20);
            @(negedge board.EP.user_clk);
            require(!seq_snapshot_valid && !ledger_accounted && seq_count===1 && ledger_completed===1 && ledger_outstanding===1,
                    "one native Done/count cannot qualify two published obligations");
            require(dma_b==1 && dma_ar==0 && dma_r==0 && host_bytes==0,"second response remains retained");
            $display("XDMA_STOP_CHAIN_ONE t=%0t reports=%0d count=%h completed=%h outstanding=%h",$time,seq_reports,seq_count,ledger_completed,ledger_outstanding);
            phase=6;hold_dma_b=0;
            $display("XDMA_STOP_CHAIN_SECOND_RELEASE t=%0t",$time);
            wait(seq_snapshot_valid === 1'b1);
            repeat(8) @(negedge board.EP.user_clk);
            require(seq_control===64'h00fffe7c00fffe7e && seq_count===64'd2 &&
                    seq_stops===3 && seq_reads===63 && seq_snapshot_valid && seq_faults===0,"native chain CSR snapshot retained");
            require(dma_aw==2 && dma_w==2 && dma_b==2 && dma_ar==0 && dma_r==0,"complete actual chain DMA inventory");
            require(bypass_aw==1 && bypass_w==1 && bypass_b==1 && bypass_ar==1 && bypass_r==1,"complete BAR inventory");
            require(host_bytes==0 && host_packets==0 && h2c_done_events==2 && c2h_done_events==0,"two actual H2C Done events; C2H is outside this test");
            require(ledger_completed===ledger_published && ledger_faults===0 && ledger_accounted,"ledger agrees with native CSR snapshot");
            require(master_requests==master_reports && seq_requests==seq_reports && !master_busy && !seq_pending,
                    "all actual management responses retired before snapshot");
            $display("XDMA_STOP_CHAIN_ACCOUNTED t=%0t host_bytes=%0d reports=%0d",$time,host_bytes,seq_reports);
            phase=7;ledger_host_closed=0;
            #1;require(!seq_snapshot_valid && !ledger_accounted,"closure revocation gates local observations immediately");
            @(negedge board.EP.user_clk);ledger_host_closed=1;
            repeat(3) @(negedge board.EP.user_clk);
            require(seq_faults===14'h200c && ledger_faults===9'h100 && !seq_snapshot_valid && seq_active,
                    "both retained certificates remain faulted after host level restores");
            $display("XDMA_STOP_CHAIN_NATIVE_PASS t=%0t seq_requests=%0d seq_reports=%0d local_reads=%0d local_writes=%0d host_bytes=%0d faults=%h dma_bytes=128 done_events=2 completed=2 native_xdma=1 c2h_tested=0 driver_closure=0 global_drain=0 reset_tested=0",
                     $time,seq_requests,seq_reports,local_reads,local_writes,host_bytes,seq_faults);

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
                require(dma_aw<=2 && board.EP.m_axi_awaddr===(dma_aw==1?64'h1000:64'h1100) &&
                        board.EP.m_axi_awlen===0 && board.EP.m_axi_awsize===6,"two distinct H2C destinations");
                $display("XDMA_LOCAL_AW t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axi_awid, board.EP.m_axi_awaddr, board.EP.m_axi_awlen, board.EP.m_axi_awsize);
            end
            if (board.EP.m_axi_wvalid && board.EP.m_axi_wready) begin
                dma_w = dma_w+1;
                require(dma_w<=2 && board.EP.m_axi_wlast===1'b1 && board.EP.m_axi_wstrb===64'hffffffffffffffff,"two complete H2C payloads");
                for(integer j=0;j<64;j=j+1)
                    require(board.EP.m_axi_wdata[j*8 +: 8]===((dma_w==1?((j*13)^'ha5):((j*7)^'h3c))&255),"each actual H2C byte matches its descriptor");
                $display("XDMA_LOCAL_W t=%0t data=%h strb=%h last=%b", $time, board.EP.m_axi_wdata, board.EP.m_axi_wstrb, board.EP.m_axi_wlast);
            end
            if (board.EP.m_axi_bvalid && board.EP.m_axi_bready) begin
                dma_b = dma_b+1;
                require(dma_b<=2 && board.EP.m_axi_bresp===2'b00,"two successful H2C responses");
                $display("XDMA_LOCAL_B t=%0t id=%h resp=%h", $time, board.EP.m_axi_bid, board.EP.m_axi_bresp);
            end
            if (board.EP.m_axi_arvalid && board.EP.m_axi_arready) begin
                dma_ar = dma_ar+1;
                require(0,"no C2H address in H2C chain test");
                $display("XDMA_LOCAL_AR t=%0t id=%h addr=%h len=%h size=%h", $time, board.EP.m_axi_arid, board.EP.m_axi_araddr, board.EP.m_axi_arlen, board.EP.m_axi_arsize);
            end
            if (board.EP.m_axi_rvalid && board.EP.m_axi_rready) begin
                dma_r = dma_r+1;
                require(0,"no C2H response in H2C chain test");
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
                    require(0,"no C2H host write in H2C chain test");
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
