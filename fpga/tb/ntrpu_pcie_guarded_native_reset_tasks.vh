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
task NTRPU_FRAME_WRITE(input [63:0] destination,input [511:0] payload);
    reg [63:0] address;reg [511:0] frame;reg [3:0] last_be;
    begin
        address=destination;frame=payload;last_be=4'hf;
        DEFAULT_TAG=10'h040+ntrpu_data_writes;ntrpu_data_writes++;
        $display("NTRD_NATIVE_FRAME n=%0d job=%0d addr=%h data=%h tag=%h ps=%0.0f",ntrpu_data_writes,board.ntrpu_bypass.job,address,frame,DEFAULT_TAG,$realtime/1ps);
        TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd16,address,last_be,4'hf,1'b0);
        TSK_TX_CLK_EAT(4);
    end
endtask
task NTRPU_BYPASS_RUN;
    begin
        $display("NTRPU_LINK_ENUM bar0=%h bar1=%h bar2=%h bar3=%h type0=%b type1=%b type2=%b range2=%h dma_bar=%0d ps=%0.0f",
            BAR_INIT_P_BAR[0],BAR_INIT_P_BAR[1],BAR_INIT_P_BAR[2],BAR_INIT_P_BAR[3],
            BAR_INIT_P_BAR_ENABLED[0],BAR_INIT_P_BAR_ENABLED[1],BAR_INIT_P_BAR_ENABLED[2],BAR_INIT_P_BAR_RANGE[2],xdma_bar,$realtime/1ps);
        if({BAR_INIT_P_BAR[3][31:0],BAR_INIT_P_BAR[2][31:0]}!==64'h900000000 || xdma_bar!=1 ||
           BAR_INIT_P_BAR_ENABLED[0]!==2'b10 || BAR_INIT_P_BAR_ENABLED[1]!==2'b10 || BAR_INIT_P_BAR_ENABLED[2]!==2'b11)
            $fatal(1,"NTRD_NATIVE_ENUMERATION");
        board.ntrpu_bypass.run_suite();
        if(ntrpu_control_writes!=294 || ntrpu_control_reads!=8 || ntrpu_data_writes!=35)
            $fatal(1,"NTRD_NATIVE_STIMULUS_COUNT");
        $display("NTRD_NATIVE_PASS writes=%0d reads=%0d frames=%0d ps=%0.0f",ntrpu_control_writes,ntrpu_control_reads,ntrpu_data_writes,$realtime/1ps);
        board.ntrpu_bypass.run_native_reset();
        if(ntrpu_control_writes!=304 || ntrpu_control_reads!=8 || ntrpu_data_writes!=37)
            $fatal(1,"NTRD_NATIVE_RESET_STIMULUS_COUNT");
        $display("NTRD_NATIVE_RESET_ROOT_PASS writes=%0d reads=%0d frames=%0d ps=%0.0f",ntrpu_control_writes,ntrpu_control_reads,ntrpu_data_writes,$realtime/1ps);
    end
endtask
