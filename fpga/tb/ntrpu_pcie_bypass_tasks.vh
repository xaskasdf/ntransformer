// Public root-port tasks; actual PCIe requests, no H2C/C2H descriptors.
task NTRPU_BYPASS_RUN;
    integer i,j;
    reg [63:0] base,address;
    reg [31:0] offsets[0:2],words[0:2];
    begin
        offsets[0]=0; offsets[1]=32'h80000; offsets[2]=32'hfffc0;
        words[0]=32'h10234567; words[1]=32'h89abcdef; words[2]=32'hfedcba98;
        base={BAR_INIT_P_BAR[3][31:0],BAR_INIT_P_BAR[2][31:0]};
        $display("NTRPU_BYPASS_ENUM bar0=%h bar1=%h bar2=%h bar3=%h type0=%b type1=%b type2=%b range2=%h dma_bar=%0d ps=%0.0f",
            BAR_INIT_P_BAR[0],BAR_INIT_P_BAR[1],BAR_INIT_P_BAR[2],BAR_INIT_P_BAR[3],
            BAR_INIT_P_BAR_ENABLED[0],BAR_INIT_P_BAR_ENABLED[1],BAR_INIT_P_BAR_ENABLED[2],
            BAR_INIT_P_BAR_RANGE[2],xdma_bar,$realtime/1ps);
        if(base!==64'h0000000900000000 || xdma_bar!=1 ||
           BAR_INIT_P_BAR_ENABLED[0]!==2'b10 || BAR_INIT_P_BAR_ENABLED[1]!==2'b10 ||
           BAR_INIT_P_BAR_ENABLED[2]!==2'b11)
            $fatal(1,"NTRPU_BYPASS_ENUMERATION");
        for(i=0;i<3;i=i+1) begin
            address=base+offsets[i];
            for(j=0;j<4;j=j+1) DATA_STORE[j]=words[i][j*8+:8];
            $display("NTRPU_BYPASS_HOST_WRITE case=%0d addr=%h data=%h tag=%h ps=%0.0f",i,address,words[i],DEFAULT_TAG,$realtime/1ps);
            TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,4'hf,1'b0);
            TSK_TX_CLK_EAT(100);
            DEFAULT_TAG=DEFAULT_TAG+1;
        end
        for(i=0;i<3;i=i+1) begin
            address=base+offsets[i]; P_READ_DATA=32'hffffffff;
            $display("NTRPU_BYPASS_HOST_READ case=%0d addr=%h tag=%h ps=%0.0f",i,address,DEFAULT_TAG,$realtime/1ps);
            fork
                TSK_TX_MEMORY_READ_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,4'hf);
                TSK_WAIT_FOR_READ_DATA;
            join
            $display("NTRPU_BYPASS_HOST_VALUE case=%0d data=%h ps=%0.0f",i,P_READ_DATA,$realtime/1ps);
            if(P_READ_DATA!==words[i]) $fatal(1,"NTRPU_BYPASS_HOST_MISMATCH");
            TSK_TX_CLK_EAT(100); DEFAULT_TAG=DEFAULT_TAG+1;
        end
        board.ntrpu_bypass.check_complete;
        $display("NTRPU_BYPASS_PASS writes=3 reads=3 host_bytes=12 ps=%0.0f",$realtime/1ps);
    end
endtask
