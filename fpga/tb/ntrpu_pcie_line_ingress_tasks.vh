// Public root-port producer; sends actual one-DWORD PCIe Memory Writes.
task NTRPU_LINK_WRITE;
    input integer job,word_index;
    input [31:0] offset;
    input [3:0] mask;
    input integer corrupt;
    integer j;
    reg [31:0] word;
    reg [63:0] address;
    begin
        for(j=0;j<4;j=j+1) word[j*8+:8]=(job*53+(word_index*4+j)*29+23)&255;
        if(corrupt) word=word^32'h1;
        address=64'h900000000+offset+word_index*4;
        for(j=0;j<4;j=j+1) DATA_STORE[j]=word[j*8+:8];
        $display("NTRPU_LINK_HOST_WRITE job=%0d word=%0d addr=%h data=%h mask=%h tag=%h ps=%0.0f",job,word_index,address,word,mask,DEFAULT_TAG,$realtime/1ps);
        TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,mask,1'b0);
        TSK_TX_CLK_EAT(4);
        DEFAULT_TAG=DEFAULT_TAG+1;
    end
endtask
task NTRPU_BYPASS_RUN;
    integer job,i;
    reg [31:0] offset;
    begin
        DEFAULT_TAG=10'h040;
        $display("NTRPU_LINK_ENUM bar0=%h bar1=%h bar2=%h bar3=%h type0=%b type1=%b type2=%b range2=%h dma_bar=%0d ps=%0.0f",
            BAR_INIT_P_BAR[0],BAR_INIT_P_BAR[1],BAR_INIT_P_BAR[2],BAR_INIT_P_BAR[3],
            BAR_INIT_P_BAR_ENABLED[0],BAR_INIT_P_BAR_ENABLED[1],BAR_INIT_P_BAR_ENABLED[2],BAR_INIT_P_BAR_RANGE[2],xdma_bar,$realtime/1ps);
        if({BAR_INIT_P_BAR[3][31:0],BAR_INIT_P_BAR[2][31:0]}!==64'h900000000 || xdma_bar!=1 ||
           BAR_INIT_P_BAR_ENABLED[0]!==2'b10 || BAR_INIT_P_BAR_ENABLED[1]!==2'b10 || BAR_INIT_P_BAR_ENABLED[2]!==2'b11)
            $fatal(1,"NTRPU_LINK_ENUMERATION");
        for(job=0;job<4;job=job+1) begin
            case(job) 0:offset=0;1,2:offset=32'h80000;3:offset=32'hfffc0;endcase
            board.ntrpu_bypass.arm_job(job);
            if(job==2) begin
                NTRPU_LINK_WRITE(job,0,offset,4'hf,0);
                NTRPU_LINK_WRITE(job,0,offset,4'hf,1);
            end else for(i=0;i<16;i=i+1) begin
                if(job==0) NTRPU_LINK_WRITE(job,i,offset,4'hf,0);
                else begin
                    NTRPU_LINK_WRITE(job,i,offset,job==1 ? 4'h5 : 4'h1,0);
                    if(job==1 && i==0) NTRPU_LINK_WRITE(job,i,offset,4'h5,0);
                    NTRPU_LINK_WRITE(job,i,offset,job==1 ? 4'ha : 4'he,0);
                end
            end
            board.ntrpu_bypass.finish_job(job);
        end
        board.ntrpu_bypass.check_complete;
        $display("NTRPU_LINK_PASS jobs=4 writes=83 lines=3 published_bytes=192 ps=%0.0f",$realtime/1ps);
    end
endtask
