// Public root-port tasks; actual PCIe requests, no H2C/C2H descriptors.
task NTRPU_BYPASS_RUN;
    integer i,j;
    reg [63:0] base,address;
    reg [31:0] offsets[0:8],words[0:8],expected_words[0:8];
    reg [3:0] masks[0:8];
    begin
        offsets[0]=0; offsets[1]=4; offsets[2]=60;
        offsets[3]=32'h80000; offsets[4]=32'h80020; offsets[5]=32'h8003c;
        offsets[6]=32'hfffc0; offsets[7]=32'hfffe0; offsets[8]=32'hffffc;
        masks[0]=4'hf; masks[1]=4'h5; masks[2]=4'ha;
        masks[3]=4'hf; masks[4]=4'h3; masks[5]=4'hc;
        masks[6]=4'hf; masks[7]=4'h1; masks[8]=4'he;
        DEFAULT_TAG=10'h040;
        for(i=0;i<9;i=i+1) begin
            words[i]=32'h10203040 ^ (i*32'h11111111);
            expected_words[i]=32'ha5a5a5a5;
            for(j=0;j<4;j=j+1)
                if(masks[i][j]) expected_words[i][j*8+:8]=words[i][j*8+:8];
        end
        base={BAR_INIT_P_BAR[3][31:0],BAR_INIT_P_BAR[2][31:0]};
        $display("NTRPU_BYTES_ENUM bar0=%h bar1=%h bar2=%h bar3=%h type0=%b type1=%b type2=%b range2=%h dma_bar=%0d ps=%0.0f",
            BAR_INIT_P_BAR[0],BAR_INIT_P_BAR[1],BAR_INIT_P_BAR[2],BAR_INIT_P_BAR[3],
            BAR_INIT_P_BAR_ENABLED[0],BAR_INIT_P_BAR_ENABLED[1],BAR_INIT_P_BAR_ENABLED[2],
            BAR_INIT_P_BAR_RANGE[2],xdma_bar,$realtime/1ps);
        if(base!==64'h0000000900000000 || xdma_bar!=1 ||
           BAR_INIT_P_BAR_ENABLED[0]!==2'b10 || BAR_INIT_P_BAR_ENABLED[1]!==2'b10 ||
           BAR_INIT_P_BAR_ENABLED[2]!==2'b11)
            $fatal(1,"NTRPU_BYTES_ENUMERATION");
        for(i=0;i<9;i=i+1) begin
            address=base+offsets[i];
            for(j=0;j<4;j=j+1) DATA_STORE[j]=words[i][j*8+:8];
            $display("NTRPU_BYTES_HOST_WRITE case=%0d addr=%h data=%h mask=%h tag=%h ps=%0.0f",i,address,words[i],masks[i],DEFAULT_TAG,$realtime/1ps);
            TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,masks[i],1'b0);
            TSK_TX_CLK_EAT(100);
            DEFAULT_TAG=DEFAULT_TAG+1;
        end
        for(i=0;i<9;i=i+1) begin
            address=base+offsets[i]; P_READ_DATA=32'hffffffff;
            $display("NTRPU_BYTES_HOST_READ case=%0d addr=%h tag=%h ps=%0.0f",i,address,DEFAULT_TAG,$realtime/1ps);
            fork
                TSK_TX_MEMORY_READ_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,4'hf);
                TSK_WAIT_FOR_READ_DATA;
            join
            $display("NTRPU_BYTES_HOST_VALUE case=%0d data=%h ps=%0.0f",i,P_READ_DATA,$realtime/1ps);
            if(P_READ_DATA!==expected_words[i]) $fatal(1,"NTRPU_BYTES_HOST_MISMATCH");
            TSK_TX_CLK_EAT(100); DEFAULT_TAG=DEFAULT_TAG+1;
        end
        board.ntrpu_bypass.check_complete;
        $display("NTRPU_BYTES_PASS writes=9 reads=9 write_bytes=24 read_bytes=36 ps=%0.0f",$realtime/1ps);
    end
endtask
