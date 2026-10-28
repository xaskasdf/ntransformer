task NTRPU_BYPASS_RUN;
    reg [511:0] frame;
    reg [63:0] address,epoch,tag,fragment_offset;
    reg [31:0] offset,expected;
    reg [3:0] last_be;
    integer job,payload_job,lane;
    begin
        $display("NTRPU_FRAME_ENUM bar0=%h bar1=%h bar2=%h bar3=%h type0=%b type1=%b type2=%b range2=%h dma_bar=%0d ps=%0.0f",BAR_INIT_P_BAR[0],BAR_INIT_P_BAR[1],BAR_INIT_P_BAR[2],BAR_INIT_P_BAR[3],BAR_INIT_P_BAR_ENABLED[0],BAR_INIT_P_BAR_ENABLED[1],BAR_INIT_P_BAR_ENABLED[2],BAR_INIT_P_BAR_RANGE[2],xdma_bar,$realtime/1ps);
        if({BAR_INIT_P_BAR[3][31:0],BAR_INIT_P_BAR[2][31:0]}!==64'h900000000 || xdma_bar!=1) $fatal(1,"NTRPU_FRAME_ENUMERATION");
        for(integer n=0;n<16;n++) begin
            job=n<6 ? n/2 : 3;payload_job=n==6 ? 1 : job;
            board.ntrpu_bypass.prepare(job);
            offset=board.ntrpu_bypass.base(job);
            if(n==15) offset='hfffc0;
            address=64'h900000000+offset;
            epoch=64'h100000001+64'(job);tag=64'h8000000100200300+64'(job);
            fragment_offset=n<6 ? 64'((n%2)*32) : (n==12 || n==13 ? 64'd32 : 64'd0);
            if(n==6) begin epoch=64'h100000002;tag=64'h8000000100200301;end
            if(n==8) tag=tag^(64'h1<<40);
            if(n==9) fragment_offset=64;
            if(n==11) epoch=64'h100000002;
            frame=0;frame[63:0]=64'h313046555052544e;
            if(n==7) frame[0]=~frame[0];
            frame[127:64]=epoch;frame[191:128]=tag;frame[255:192]=fragment_offset;
            for(integer j=0;j<32;j++) frame[256+j*8+:8]=8'(payload_job*53+(int'(fragment_offset)+j)*29+23);
            for(integer j=0;j<64;j++) DATA_STORE[j]=frame[j*8+:8];
            DEFAULT_TAG=10'h40+10'(n);last_be=n==10 ? 4'h7 : 4'hf;
            $display("NTRPU_FRAME_WRITE n=%0d job=%0d addr=%h data=%h first=f last=%h tag=%h ps=%0.0f",n,job,address,frame,last_be,DEFAULT_TAG,$realtime/1ps);
            TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd16,address,last_be,4'hf,1'b0);
            board.ntrpu_bypass.wait_write(n);
        end
        for(integer n=0;n<12;n++) begin
            job=n/4==1 ? 3 : n/4;
            case(n%4) 0:lane=0;1:lane=28;2:lane=32;3:lane=60;endcase
            address=64'h900000000+board.ntrpu_bypass.base(job)+lane;
            for(integer j=0;j<4;j++) expected[j*8+:8]=8'(job*53+(lane+j)*29+23);
            DEFAULT_TAG=10'h60+10'(n);P_READ_DATA='1;
            $display("NTRPU_FRAME_READ n=%0d addr=%h expected=%h tag=%h ps=%0.0f",n,address,expected,DEFAULT_TAG,$realtime/1ps);
            fork
                TSK_TX_MEMORY_READ_64(DEFAULT_TAG,DEFAULT_TC,10'd1,address,4'h0,4'hf);
                TSK_WAIT_FOR_READ_DATA;
            join
            $display("NTRPU_FRAME_VALUE n=%0d data=%h ps=%0.0f",n,P_READ_DATA,$realtime/1ps);
            if(P_READ_DATA!==expected) $fatal(1,"NTRPU_FRAME_ROOT_READBACK");
            TSK_TX_CLK_EAT(4);
        end
        board.ntrpu_bypass.check_complete;
        $display("NTRPU_FRAME_PASS writes=16 reads=12 accepted=9 rejected=7 assembled_lines=4 ps=%0.0f",$realtime/1ps);
    end
endtask
