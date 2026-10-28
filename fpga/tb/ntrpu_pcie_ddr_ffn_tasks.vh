// Included inside AMD's copied root-complex TX test driver. Testbench only.
integer ntrpu_csr_bar=-1;
integer ntrpu_concurrent_dma=0;
task NTRPU_NEED;
    input ok;
    input [1023:0] why;
    begin if(ok!==1'b1) $fatal(1,"NTRPU_PCIE_FFN_CHECK %0s",why);end
endtask
task NTRPU_READ_EXPECT;
    input [31:0] address,value;
    reg [31:0] got;
    begin
        NTRPU_BAR_READ(ntrpu_csr_bar,address,got);
        if(got!==value) $fatal(1,"NTRPU_CSR_EXPECT addr=%h got=%h expected=%h",address,got,value);
    end
endtask
task NTRPU_WRITE;
    input [31:0] address,value;
    begin NTRPU_BAR_WRITE(ntrpu_csr_bar,address,value,4'hf);end
endtask
task NTRPU_WAIT_STATE;
    input [31:0] mask,value;
    reg [31:0] got;
    integer polls;
    begin
        got=~value;polls=0;
        while((got&mask)!==value && polls<1000) begin
            NTRPU_BAR_READ(ntrpu_csr_bar,'hc,got);polls=polls+1;
            NTRPU_NEED(!got[7],"sticky notifier fault");
        end
        NTRPU_NEED((got&mask)===value,"state poll budget");
    end
endtask
task NTRPU_WAIT_DMA;
    input c2h;
    input integer expected_count;
    integer polls;
    reg [31:0] got;
    begin
        got=0;polls=0;
        while(got!=expected_count && polls<100) begin
            TSK_XDMA_REG_READ(c2h ? 'h1048:'h0048);got=P_READ_DATA;polls=polls+1;
        end
        NTRPU_NEED(got==expected_count,"DMA descriptor completion count");
        // Count and STOP status are separate observations. Keep polling the
        // non-clearing status register, failing immediately on any error.
        polls=0;P_READ_DATA=0;
        while(P_READ_DATA!==32'h6 && polls<100) begin
            TSK_XDMA_REG_READ(c2h ? 'h1040:'h0040);polls=polls+1;
            $display("NTRPU_PCIE_FFN_DMA_STATUS direction=%0d value=%08h time_ps=%0d",c2h,P_READ_DATA,$time);
            NTRPU_NEED((P_READ_DATA&32'hfffffff8)===0,"DMA error or unknown status while waiting for STOP");
        end
        NTRPU_NEED(P_READ_DATA===32'h6,"DMA stopped/completed without engine errors");
        $display("NTRPU_PCIE_FFN_DMA direction=%0d count=%0d status=%08h time_ps=%0d",c2h,got,P_READ_DATA,$time);
    end
endtask
task NTRPU_PROGRAM;
    input integer job_id;
    input valid_compute,enable_irq;
    reg [63:0] addr,base;
    integer b,span;
    begin
        NTRPU_WAIT_STATE('h187,'h100);
        NTRPU_WRITE('h14,enable_irq);
        NTRPU_WRITE('h20,job_id);NTRPU_WRITE('h24,600+job_id);NTRPU_WRITE('h28,600+job_id);
        NTRPU_WRITE('h2c,valid_compute ? 'h00200020:'h00200000);
        NTRPU_WRITE('h30,2);NTRPU_WRITE('h34,'h020202);
        for(b=0;b<8;b=b+1) begin
            base=64'h100000000+b*64'h20000000;addr=base+4032;
            if(b==0 || b==6) span=64;
            else if(b<=3) span=1088;
            else if(b==7) span=128;
            else span=256;
            NTRPU_WRITE('h100+b*32,addr[31:0]);NTRPU_WRITE('h104+b*32,addr[63:32]);
            NTRPU_WRITE('h108+b*32,base[31:0]);NTRPU_WRITE('h10c+b*32,base[63:32]);
            NTRPU_WRITE('h110+b*32,4032+span);NTRPU_WRITE('h114+b*32,0);
            NTRPU_WRITE('h118+b*32,600+job_id);
        end
        $display("NTRPU_PCIE_FFN_CASE id=%0d compute=%0d irq=%0d time_ps=%0d",job_id,valid_compute,enable_irq,$time);
        NTRPU_WRITE('h10,1);
        NTRPU_WAIT_STATE(4,4);
        NTRPU_READ_EXPECT('h18,job_id);NTRPU_READ_EXPECT('h1c,job_id);
        NTRPU_READ_EXPECT('h64,job_id);NTRPU_READ_EXPECT('h68,600+job_id);
        NTRPU_READ_EXPECT('h4c,job_id);NTRPU_READ_EXPECT('h50,600+job_id);
        NTRPU_READ_EXPECT('h44,valid_compute ? 128:0);
        NTRPU_READ_EXPECT('h48,valid_compute ? 704:0);
        NTRPU_READ_EXPECT('h40,valid_compute ? 'h1c00:1);
        NTRPU_READ_EXPECT('h54,0);NTRPU_READ_EXPECT('h60,0);
    end
endtask
task NTRPU_SERVICE;
    input integer job_id;
    begin
        NTRPU_WRITE('h38,job_id);NTRPU_WRITE('h3c,600+job_id);NTRPU_WRITE('h10,4);
        $display("NTRPU_PCIE_FFN_SERVICE id=%0d time_ps=%0d",job_id,$time);
    end
endtask
task NTRPU_FFN_RUN;
    integer b;
    reg [31:0] got;
    begin
        ntrpu_concurrent_dma=$test$plusargs("CONCURRENT_DMA");
        $display("NTRPU_PCIE_FFN_MODE concurrent=%0d",ntrpu_concurrent_dma);
        board.ntrpu_host.initialize();
        for(b=0;b<10000 && !(board.EP.ntrpu_app.host_admit && board.EP.ntrpu_app.device_admit);b=b+1)
            TSK_TX_CLK_EAT(10);
        NTRPU_NEED(board.EP.ntrpu_app.host_admit && board.EP.ntrpu_app.device_admit,"cold DDR admission before MMIO/DMA");
        $display("NTRPU_PCIE_DDR_ADMITTED time_ps=%0d",$time);
        for(b=0;b<6;b=b+1) begin
            if((BAR_INIT_P_BAR_ENABLED[b]==2 || BAR_INIT_P_BAR_ENABLED[b]==3) && b!=xdma_bar) begin
                NTRPU_BAR_READ(b,0,got);
                if(got=='h4652544e) ntrpu_csr_bar=b;
            end
        end
        NTRPU_NEED(ntrpu_csr_bar>=0,"FFN BAR discovery");
        $display("NTRPU_PCIE_FFN_BARS xdma=%0d csr=%0d",xdma_bar,ntrpu_csr_bar);
        NTRPU_READ_EXPECT(4,'h20000);NTRPU_READ_EXPECT(8,15);
        TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG,DEV_CTRL_REG_ADDR,4'hf);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_WAIT_FOR_READ_DATA;
        TSK_TX_TYPE0_CONFIGURATION_WRITE(DEFAULT_TAG,DEV_CTRL_REG_ADDR,
            (P_READ_DATA&32'h00008f1f)|32'h00001020,4'h3);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_TX_CLK_EAT(100);
        TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG,DEV_CTRL_REG_ADDR,4'hf);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_WAIT_FOR_READ_DATA;
        NTRPU_NEED(P_READ_DATA[7:5]===3'd1 && P_READ_DATA[14:12]===3'd1,"MPS/MRRS 256");
        $display("NTRPU_PCIE_FFN_DEVCTL value=%08h",P_READ_DATA);

        TSK_MSIX_EN;
        TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG,12'h060,4'hf);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_WAIT_FOR_READ_DATA;
        $display("NTRPU_PCIE_FFN_MSIX_CAP register=060 value=%08h",P_READ_DATA);
        NTRPU_NEED(P_READ_DATA[31] && P_READ_DATA[26:16]==31 && P_READ_DATA[7:0]==8'h11,"MSI-X capability and 32 vectors");
        TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG,12'h064,4'hf);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_WAIT_FOR_READ_DATA;
        $display("NTRPU_PCIE_FFN_MSIX_CAP register=064 value=%08h",P_READ_DATA);
        NTRPU_NEED(P_READ_DATA===32'h00008001,"MSI-X advertised table offset/BIR");
        TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG,12'h068,4'hf);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_WAIT_FOR_READ_DATA;
        $display("NTRPU_PCIE_FFN_MSIX_CAP register=068 value=%08h",P_READ_DATA);
        NTRPU_NEED(P_READ_DATA===32'h00008fe1,"MSI-X advertised PBA offset/BIR");
        TSK_XDMA_REG_WRITE('h8040,'h800,4'hf);TSK_XDMA_REG_WRITE('h8044,5,4'hf);
        TSK_XDMA_REG_WRITE('h8048,'h8badc0de,4'hf);TSK_XDMA_REG_WRITE('h804c,0,4'hf);
        TSK_XDMA_REG_WRITE('h2080,4,4'hf);
        TSK_XDMA_REG_WRITE('h2010,0,4'hf); // DMA channel IRQs disabled, user vector only.
        TSK_XDMA_REG_WRITE('h2004,1,4'hf);
        TSK_XDMA_REG_READ('h3014);NTRPU_NEED(P_READ_DATA[1:0]===2'b10,"MSI-X enabled, MSI disabled");
        TSK_XDMA_REG_WRITE('h4080,'h100,4'hf);TSK_XDMA_REG_WRITE('h4084,1,4'hf);
        // PG195 control bit 25 selects non-incrementing addresses. Keep it zero.
        TSK_XDMA_REG_WRITE('h0004,'h00fffe7e,4'hf);
        TSK_XDMA_REG_READ('h0004);NTRPU_NEED(!P_READ_DATA[25],"H2C incrementing address mode");
        $display("NTRPU_PCIE_FFN_DMA_CONTROL direction=0 value=%08h",P_READ_DATA);
        TSK_XDMA_REG_WRITE('h0004,'h00fffe7f,4'hf);
        // Retain the former serialized test as a selectable regression. The
        // concurrent mode waits exclusively on real DMA register responses.
        if(!ntrpu_concurrent_dma) begin
            wait(board.EP.ntrpu_app.input_bytes==3328);TSK_TX_CLK_EAT(256);
            $display("NTRPU_PCIE_FFN_DMA_DRAIN direction=0 time_ps=%0d",$time);
        end
        NTRPU_WAIT_DMA(0,4);
        if(ntrpu_concurrent_dma)$display("NTRPU_PCIE_FFN_DMA_DRAIN direction=0 time_ps=%0d",$time);
        NTRPU_NEED(board.EP.ntrpu_app.input_bytes==3328,"all inputs DMA loaded");

        // Valid compute: transport acknowledgement precedes software service.
        NTRPU_PROGRAM(1,1,1);
        wait(board.ntrpu_host.msix_packets==1 && board.EP.ntrpu_app.acknowledgements==1);
        NTRPU_NEED(board.EP.usr_irq_req[0],"IRQ held after transport ACK");
        TSK_XDMA_REG_READ('h2048);NTRPU_NEED(P_READ_DATA[0],"XDMA user pending");
        TSK_XDMA_REG_WRITE('h5080,'h300,4'hf);TSK_XDMA_REG_WRITE('h5084,3,4'hf);
        TSK_XDMA_REG_WRITE('h1004,'h00fffe7e,4'hf);
        TSK_XDMA_REG_READ('h1004);NTRPU_NEED(!P_READ_DATA[25],"C2H incrementing address mode");
        $display("NTRPU_PCIE_FFN_DMA_CONTROL direction=1 value=%08h",P_READ_DATA);
        TSK_XDMA_REG_WRITE('h1004,'h00fffe7f,4'hf);
        if(!ntrpu_concurrent_dma) begin
            wait(board.ntrpu_host.c2h_bytes==128);TSK_TX_CLK_EAT(256);
            $display("NTRPU_PCIE_FFN_DMA_DRAIN direction=1 time_ps=%0d",$time);
        end
        NTRPU_WAIT_DMA(1,1);
        if(ntrpu_concurrent_dma)$display("NTRPU_PCIE_FFN_DMA_DRAIN direction=1 time_ps=%0d",$time);
        wait(board.ntrpu_host.c2h_bytes==128);
        NTRPU_WRITE('h38,2);NTRPU_WRITE('h3c,601);NTRPU_WRITE('h10,4);
        NTRPU_WAIT_STATE('ha4,4); // Wrong software identity must not service or retire.
        NTRPU_NEED(board.EP.usr_irq_req[0],"wrong identity retained IRQ");
        NTRPU_SERVICE(1);NTRPU_WAIT_STATE('h187,'h100);
        TSK_XDMA_REG_READ('h2048);NTRPU_NEED(!P_READ_DATA[0],"XDMA pending cleared after service");

        // Invalid descriptor, actual IRQ masked at XDMA: software services first.
        TSK_XDMA_REG_WRITE('h2004,0,4'hf);
        NTRPU_PROGRAM(2,0,1);
        NTRPU_NEED(board.EP.usr_irq_req[0],"masked request visible");
        NTRPU_SERVICE(2);NTRPU_WAIT_STATE('ha4,'h24);
        TSK_TX_CLK_EAT(256);
        NTRPU_NEED(board.ntrpu_host.msix_packets==1 && board.EP.ntrpu_app.acknowledgements==1 &&
            board.EP.usr_irq_req[0],"service alone cannot retire masked transport");
        TSK_XDMA_REG_WRITE('h2004,1,4'hf);
        wait(board.ntrpu_host.msix_packets==2 && board.EP.ntrpu_app.acknowledgements==2);
        NTRPU_WAIT_STATE('h187,'h100);

        // Polling completion: no MSI-X packet or transport ACK is required.
        NTRPU_PROGRAM(3,0,0);
        NTRPU_NEED(!board.EP.usr_irq_req[0],"polling does not assert IRQ");
        NTRPU_SERVICE(3);NTRPU_WAIT_STATE('h187,'h100);TSK_TX_CLK_EAT(256);
        board.ntrpu_host.verify_complete();
        board.EP.ntrpu_app.verify_ddr_complete();
    end
endtask
