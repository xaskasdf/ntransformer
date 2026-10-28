// Public root-port PCIe traffic. Isolation/readback remain explicit test helpers.
integer ntrpu_control_writes=0,ntrpu_control_reads=0,ntrpu_data_writes=0;
task NTRPU_CTRL_WRITE(input [31:0] offset,input [31:0] value);
    reg [63:0] address;
    begin
        address={32'b0,BAR_INIT_P_BAR[0][31:0]}+offset;
        DEFAULT_TAG=10'h020;
        for(integer j=0;j<4;j++) DATA_STORE[j]=value[j*8+:8];
        ntrpu_control_writes++;
        $display("NTRPU_NATIVE_CONTROL_WRITE n=%0d addr=%h data=%h tag=%h ps=%0.0f",ntrpu_control_writes,address,value,DEFAULT_TAG,$realtime/1ps);
        TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,4'hf,1'b0);
        TSK_TX_CLK_EAT(4);
    end
endtask
task NTRPU_CTRL_READ(input [31:0] offset,input [31:0] expected,input [31:0] mask);
    reg [63:0] address;
    begin
        address={32'b0,BAR_INIT_P_BAR[0][31:0]}+offset;
        DEFAULT_TAG=10'h021;P_READ_DATA=32'hffffffff;
        ntrpu_control_reads++;
        $display("NTRPU_NATIVE_CONTROL_READ n=%0d addr=%h expected=%h mask=%h tag=%h ps=%0.0f",ntrpu_control_reads,address,expected,mask,DEFAULT_TAG,$realtime/1ps);
        fork
            TSK_TX_MEMORY_READ_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,4'hf);
            TSK_WAIT_FOR_READ_DATA;
        join
        $display("NTRPU_NATIVE_CONTROL_VALUE n=%0d data=%h ps=%0.0f",ntrpu_control_reads,P_READ_DATA,$realtime/1ps);
        if((P_READ_DATA & mask)!==expected) $fatal(1,"NTRPU_NATIVE_CONTROL_READBACK");
        TSK_TX_CLK_EAT(4);
    end
endtask
task NTRPU_LINK_WRITE(input integer job,word_index,input [31:0] offset,input [3:0] mask,input integer corrupt);
    reg [31:0] word;
    reg [63:0] address;
    begin
        for(integer j=0;j<4;j++) word[j*8+:8]=(job*53+(word_index*4+j)*29+23)&255;
        if(corrupt) word=word^32'h1;
        address=64'h900000000+offset+word_index*4;
        DEFAULT_TAG=10'h040+ntrpu_data_writes;ntrpu_data_writes++;
        for(integer j=0;j<4;j++) DATA_STORE[j]=word[j*8+:8];
        $display("NTRPU_LINK_HOST_WRITE job=%0d word=%0d addr=%h data=%h mask=%h tag=%h ps=%0.0f",job,word_index,address,word,mask,DEFAULT_TAG,$realtime/1ps);
        TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,mask,1'b0);
        TSK_TX_CLK_EAT(4);
    end
endtask
task NTRPU_BYPASS_RUN;
    reg [31:0] offset;
    reg [63:0] destination,window;
    begin
        $display("NTRPU_LINK_ENUM bar0=%h bar1=%h bar2=%h bar3=%h type0=%b type1=%b type2=%b range2=%h dma_bar=%0d ps=%0.0f",
            BAR_INIT_P_BAR[0],BAR_INIT_P_BAR[1],BAR_INIT_P_BAR[2],BAR_INIT_P_BAR[3],
            BAR_INIT_P_BAR_ENABLED[0],BAR_INIT_P_BAR_ENABLED[1],BAR_INIT_P_BAR_ENABLED[2],BAR_INIT_P_BAR_RANGE[2],xdma_bar,$realtime/1ps);
        if({BAR_INIT_P_BAR[3][31:0],BAR_INIT_P_BAR[2][31:0]}!==64'h900000000 || xdma_bar!=1 ||
           BAR_INIT_P_BAR_ENABLED[0]!==2'b10 || BAR_INIT_P_BAR_ENABLED[1]!==2'b10 || BAR_INIT_P_BAR_ENABLED[2]!==2'b11)
            $fatal(1,"NTRPU_LINK_ENUMERATION");
        board.ntrpu_bypass.wait_cold_ready();
        NTRPU_CTRL_READ('h00,'h4952544e,'1);
        NTRPU_CTRL_READ('h04,'h00010000,'1);
        NTRPU_CTRL_READ('h08,1,'1);
        NTRPU_CTRL_WRITE('h14,1);
        for(integer job=0;job<4;job++) begin
            case(job) 0:begin offset=0;destination=0;end
                1,2:begin offset='h80000;destination=64'h100000040;end
                3:begin offset='hfffc0;destination=64'h1fffffffc0;end endcase
            window=64'h100000000+offset;
            board.ntrpu_bypass.prepare_job(job);
            NTRPU_CTRL_WRITE('h20,job+1);NTRPU_CTRL_WRITE('h24,0);
            NTRPU_CTRL_WRITE('h28,'h2000+job);NTRPU_CTRL_WRITE('h2c,0);
            NTRPU_CTRL_WRITE('h30,window[31:0]);NTRPU_CTRL_WRITE('h34,window[63:32]);
            NTRPU_CTRL_WRITE('h38,destination[31:0]);NTRPU_CTRL_WRITE('h3c,destination[63:32]);
            NTRPU_CTRL_WRITE('h10,1);
            NTRPU_CTRL_READ('h18,job,'1); // queued ARM cannot bypass missing proof
            board.ntrpu_bypass.allow_arm(job);
            NTRPU_CTRL_READ('h18,job+1,'1);
            board.ntrpu_bypass.after_arm(job);
            NTRPU_CTRL_WRITE('h20,0);NTRPU_CTRL_WRITE('h10,4); // locked descriptor, early ACK
            if(job==2) begin
                NTRPU_LINK_WRITE(job,0,offset,4'hf,0);NTRPU_LINK_WRITE(job,0,offset,4'hf,1);
            end else for(integer i=0;i<16;i++) begin
                if(job==0) NTRPU_LINK_WRITE(job,i,offset,4'hf,0);
                else begin
                    NTRPU_LINK_WRITE(job,i,offset,job==1 ? 4'h3 : 4'h1,0);
                    if(job==1 && i==0) NTRPU_LINK_WRITE(job,i,offset,4'h3,0);
                    NTRPU_LINK_WRITE(job,i,offset,job==1 ? 4'hc : 4'he,0);
                end
            end
            board.ntrpu_bypass.drain_job(job);
            if(job==2) begin NTRPU_CTRL_WRITE('h10,2);NTRPU_CTRL_READ('h0c,'h10,'h10);end
            board.ntrpu_bypass.publish_job(job);
            NTRPU_CTRL_READ('h60,job+1,'1);NTRPU_CTRL_READ('h64,0,'1);
            NTRPU_CTRL_READ('h68,'h2000+job,'1);NTRPU_CTRL_READ('h6c,0,'1);
            NTRPU_CTRL_READ('h70,job==2 ? 1 : 0,'1);
            NTRPU_CTRL_READ('h0c,1,1);
            NTRPU_CTRL_WRITE('h40,job+1);NTRPU_CTRL_WRITE('h44,0);
            NTRPU_CTRL_WRITE('h48,'h2000+job);NTRPU_CTRL_WRITE('h4c,1);
            NTRPU_CTRL_WRITE('h10,4); // wrong high tag cannot retire
            NTRPU_CTRL_READ('h1c,job,'1);
            NTRPU_CTRL_WRITE('h4c,0);NTRPU_CTRL_WRITE('h10,4);
            NTRPU_CTRL_READ('h1c,job+1,'1);NTRPU_CTRL_READ('h0c,0,1);
            NTRPU_CTRL_WRITE('h10,1); // stale epoch cannot acquire new ownership
            NTRPU_CTRL_READ('h18,job+1,'1);
            board.ntrpu_bypass.after_retire(job);
        end
        board.ntrpu_bypass.check_complete;
        $display("NTRPU_NATIVE_CONTROL_PASS writes=%0d reads=%0d data_writes=%0d ps=%0.0f",ntrpu_control_writes,ntrpu_control_reads,ntrpu_data_writes,$realtime/1ps);
        $display("NTRPU_LINK_PASS jobs=4 writes=83 lines=3 published_bytes=192 ps=%0.0f",$realtime/1ps);
    end
endtask
