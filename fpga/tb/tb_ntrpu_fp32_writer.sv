`timescale 1ns/1ps
module tb_ntrpu_fp32_writer;
    logic clk=0,rst_n=0;
    logic cmd_output_fp32=0;
    always #2.5 clk=~clk;
    logic cmd_valid=0,cmd_ready,prepared,result_valid=0,result_ready,commit_valid=0,commit_ready;
    logic cancel_valid=0,cancel_ready,done_valid,done_ready=0;
    logic [63:0] cmd_addr=0,cmd_region_base=0,cmd_region_bytes=0,result_data=0;
    logic [5:0] cmd_rows=0;
    logic [31:0] cmd_sequence=0,cmd_epoch=0,expected_epoch=0,result_sequence=0,result_epoch=0;
    logic [31:0] commit_sequence=0,commit_epoch=0,done_sequence,done_epoch;
    logic [4:0] result_row=0;
    logic result_last=0;
    logic [3:0] commit_status=0,done_status;
    logic [8:0] done_bytes;
    logic [63:0] m_axi_awaddr,m_axi_wstrb;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize,m_axi_awprot;
    logic [1:0] m_axi_awburst,m_axi_bresp=0;
    logic [3:0] m_axi_awid,m_axi_awcache,m_axi_bid=0;
    logic m_axi_awlock,m_axi_awvalid,m_axi_awready=0,m_axi_wlast,m_axi_wvalid,m_axi_wready=0;
    logic m_axi_bvalid=0,m_axi_bready;
    logic [511:0] m_axi_wdata;
`ifdef NTRPU_NETLIST
    // The checkpoint has already fixed the production watchdog parameter.
    localparam int WATCHDOG=65536,CASE_LIMIT=80000;
    ntrpu_result_writer dut(.*);
`else
    localparam int WATCHDOG=32,CASE_LIMIT=2000;
    ntrpu_result_writer #(.WATCHDOG_CYCLES(WATCHDOG),.FP32_CAPABLE(1)) dut(.*);
`endif
    localparam int TIMEOUT_DELAY=WATCHDOG+48;
    localparam logic [63:0] BASE=64'ha000000000;
    int seed=1,random_state,case_id=0,cycles=0;
    int aw_stalls=0,w_stalls=0,done_stalls=0,resets=0,total_aw=0,total_w=0,total_bytes=0;
    int w_before_aw=0,aw_before_w=0;
    function automatic void require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL WRITER case=%0d cycle=%0d: %s",case_id,cycles,why);
    endfunction
    function automatic logic [63:0] ieee_input(input int row);
        case(row%16)
            0:return 64'h0000000000000000;
            1:return 64'h8000000000000000;
            2:return 64'h36a0000000000000;
            3:return 64'hb6a0000000000000;
            4:return 64'h3690000000000000;
            5:return 64'h3690000000000001;
            6:return 64'h380fffffc0000000;
            7:return 64'h380fffffe0000000;
            8:return 64'h3810000000000000;
            9:return 64'h3ff0000010000000;
            10:return 64'h3ff0000030000000;
            11:return 64'h47efffffe0000000;
            12:return 64'hc7efffffe0000000;
            13:return 64'h0000000000000001;
            14:return 64'h8000000000000001;
            15:return 64'h3ff4000000000000;
            default:return 0;
        endcase
    endfunction
    function automatic logic [31:0] ieee_output(input int row);
        case(row%16)
            0:return 32'h00000000;
            1:return 32'h80000000;
            2:return 32'h00000001;
            3:return 32'h80000001;
            4:return 32'h00000000;
            5:return 32'h00000001;
            6:return 32'h007fffff;
            7:return 32'h00800000;
            8:return 32'h00800000;
            9:return 32'h3f800000;
            10:return 32'h3f800002;
            11:return 32'h7f7fffff;
            12:return 32'hff7fffff;
            13:return 32'h00000000;
            14:return 32'h80000000;
            15:return 32'h3fa00000;
            default:return 0;
        endcase
    endfunction
    function automatic logic [63:0] row_value(input int row,input int mode);
        if(mode==35) return ieee_input(row);
        if(row==8) case(mode)
            36,43:return 64'h47efffffe0000001;
            37:return 64'hc7efffffe0000001;
            38:return 64'h7ff0000000000001;
            39:return 64'hfff0000000000000;
            40:return 64'h7fefffffffffffff;
            41:return 64'h47efffffe0000001;
            default:begin end
        endcase
        if(mode==33 && row==0) return 64'h8000000000000000;
        return $realtobits(real'(case_id*100+row)*0.25);
    endfunction
    function automatic logic [31:0] row_fp32(input int row,input int mode);
        if(mode==35) return ieee_output(row);
        if(mode==33 && row==0) return 32'h80000000;
        return $shortrealtobits(shortreal'(real'(case_id*100+row)*0.25));
    endfunction
    task automatic run_case(input int n,input int offset,input int mode,input int status,input bit fp32=1);
        bit accepted,committed,finished,cancelled,aw_seen,aw_hold,w_hold,done_hold;
        bit cmd_hs,result_hs,commit_hs,cancel_hs,aw_hs,w_hs,b_hs,reset_done;
        logic [71:0] saved_aw;
        logic [576:0] saved_w;
        logic [76:0] saved_done;
        logic [63:0] address,expected_value,expected_strobe;
        logic [7:0] memory [0:319];
        int sent,input_count,aw_count,w_count,b_count,w_in_burst,burst_beats,beat_cursor,wbytes,done_wait,send_start;
        int expected_beats,valid_bytes,index,item_bytes,lanes,last_input_cycle;
        item_bytes=fp32 ? 4:8;lanes=fp32 ? 16:8;last_input_cycle=0;
        case_id++;cycles=0;sent=0;input_count=n;aw_count=0;w_count=0;b_count=0;
        w_in_burst=0;beat_cursor=0;wbytes=0;done_wait=0;send_start=0;reset_done=0;
        accepted=0;committed=0;finished=0;cancelled=0;aw_seen=0;aw_hold=0;w_hold=0;done_hold=0;
        foreach(memory[i]) memory[i]=8'ha5;
        if(mode==19) input_count=n-1;
        if(mode==20) input_count=n+1;
        if(mode==42) input_count=70;
        address=BASE+64'(offset);expected_beats=(n+lanes-1)/lanes;
        burst_beats=expected_beats<(4096-offset)/64 ? expected_beats:(4096-offset)/64;
        @(negedge clk);require(cmd_ready,"previous command not released");
        cmd_addr=address;cmd_region_base=BASE;cmd_region_bytes=64'h100000;
        cmd_output_fp32=fp32;cmd_rows=6'(n);cmd_sequence=32'(case_id);cmd_epoch=32'(case_id+100);expected_epoch=cmd_epoch;
        if(mode==2) cmd_addr++;
        if(mode==3) cmd_region_bytes=64'(offset+n*item_bytes-1);
        if(mode==4) expected_epoch++;
        if(mode==5) cmd_region_base=cmd_addr+64;
        if(mode==6) begin cmd_addr=64'hffffffffffffffc0;cmd_region_base=cmd_addr;cmd_region_bytes=4096;end
        cmd_valid=1;result_valid=0;commit_valid=0;cancel_valid=0;m_axi_bvalid=0;
        commit_status=mode==21 ? 6:0;commit_sequence=32'(case_id);commit_epoch=32'(case_id+100);
        if(mode==22) commit_sequence++;
        while(!finished) begin
            m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if(send_start>0) begin
                if((mode==11 && cycles<send_start+TIMEOUT_DELAY) || (mode==26 && cycles<send_start+12)) m_axi_awready=0;
                if((mode==12 && cycles<send_start+TIMEOUT_DELAY) || (mode==25 && cycles<send_start+12)) m_axi_wready=0;
            end else if(mode==11 || mode==26) m_axi_awready=0;
            else if(mode==12 || mode==25) m_axi_wready=0;
            done_ready=done_wait>=5 && $urandom_range(0,2)!=0;
            if(!cancelled && ((mode==23 && prepared && sent>=2) || (mode==24 && accepted && !prepared) || (mode==43 && sent==n && !committed))) cancel_valid=1;
            if(mode==32 && send_start>0) cancel_valid=1;
            if(prepared && !result_valid && sent<input_count && $urandom_range(0,3)!=0) begin
                result_valid=1;result_row=5'(sent);result_data=row_value(sent,mode);
                result_sequence=32'(case_id);result_epoch=32'(case_id+100);result_last=sent==n-1;
                if(mode==14 && sent==n-1) result_row=0;
                if(mode==15 && sent==n-1) result_sequence++;
                if(mode==16 && sent==n-1) result_epoch++;
                if(mode==17 && sent==0) result_last=1;
                if(mode==18 && sent==n-1) result_data=64'h7ff8000000000000;
                if(mode==34 && sent==n-1) result_data=64'h7ff0000000000000;
            end
            if(prepared && !committed && (sent==input_count || (mode==31 && sent==n-1 && result_valid))) commit_valid=1;
            if(aw_seen && w_in_burst==burst_beats && !m_axi_bvalid &&
               !(mode==13 && cycles<send_start+TIMEOUT_DELAY) && !(mode==27 && cycles<send_start+24)) begin
                m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;
                if((mode==7 && b_count==0) || (mode==8 && b_count==1)) m_axi_bresp=2;
                if((mode==9 && b_count==0) || (mode==10 && b_count==1)) m_axi_bid=1;
            end
            @(posedge clk);cycles++;require(cycles<CASE_LIMIT,"bank timeout");
            cmd_hs=cmd_valid&&cmd_ready;result_hs=result_valid&&result_ready;commit_hs=commit_valid&&commit_ready;
            cancel_hs=cancel_valid&&cancel_ready;aw_hs=m_axi_awvalid&&m_axi_awready;
            w_hs=m_axi_wvalid&&m_axi_wready;b_hs=m_axi_bvalid&&m_axi_bready;
            if(aw_hold) require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen}===saved_aw,"AW stability");
            if(w_hold) require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"W stability through timeout");
            if(done_hold) require(done_valid && {done_status,done_bytes,done_sequence,done_epoch}===saved_done,"completion stability");
            aw_hold=m_axi_awvalid&&!m_axi_awready;w_hold=m_axi_wvalid&&!m_axi_wready;done_hold=done_valid&&!done_ready;
            saved_aw={m_axi_awaddr,m_axi_awlen};saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};saved_done={done_status,done_bytes,done_sequence,done_epoch};
            aw_stalls+=int'(aw_hold);w_stalls+=int'(w_hold);done_stalls+=int'(done_hold);
            if(cmd_hs) begin require(!accepted,"duplicate command");accepted=1;end
            if(result_hs) begin sent++;last_input_cycle=cycles;end
            if(commit_hs) begin
                require(!committed,"duplicate commit");
                if(fp32 && sent>0) require(cycles>=last_input_cycle+3,"commit before conversion drain");
                committed=1;end
            if(cancel_hs) cancelled=1;
            if(m_axi_awvalid || m_axi_wvalid) begin
                require(committed && sent==n && commit_status==0,"write before complete successful commit");
                require(!cancel_ready,"cancellation after commit accepted");
                if(send_start==0) send_start=cycles;
            end
            if(aw_hs) begin
                require(!aw_seen && aw_count==b_count,"multiple outstanding AW");
                require(m_axi_awaddr==address && int'(m_axi_awlen)+1==burst_beats,"AW address/high bits/page");
                require(int'(m_axi_awaddr[11:0])+burst_beats*64<=4096,"4KiB crossing");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                if(w_in_burst==0) aw_before_w++;aw_seen=1;aw_count++;
            end
            if(w_hs) begin
                require(w_in_burst<burst_beats && beat_cursor<expected_beats,"extra W beat");
                if(!aw_seen) w_before_aw++;
                expected_strobe=0;valid_bytes=0;
                for(int lane=0;lane<lanes;lane++) begin
                    index=beat_cursor*lanes+lane;expected_value=index<n ? (fp32 ? {32'b0,row_fp32(index,mode)}:row_value(index,mode)):64'd0;
                    if(fp32) require(m_axi_wdata[lane*32+:32]===expected_value[31:0],"FP32 W data byte order/padding");
                    else require(m_axi_wdata[lane*64+:64]===expected_value,"FP64 W data byte order/padding");
                    if(index<n) begin
                        if(fp32) expected_strobe[lane*4+:4]=4'hf;else expected_strobe[lane*8+:8]=8'hff;
                        valid_bytes+=item_bytes;end
                end
                require(m_axi_wstrb==expected_strobe && m_axi_wlast==(w_in_burst==burst_beats-1),"WSTRB/WLAST");
                for(int byte_lane=0;byte_lane<64;byte_lane++)
                    if(m_axi_wstrb[byte_lane]) memory[beat_cursor*64+byte_lane]=m_axi_wdata[byte_lane*8+:8];
                $display("NTRPU_FPWRITER_W id=%0d beat=%0d data=%0128h strb=%016h",case_id,beat_cursor,m_axi_wdata,m_axi_wstrb);
                w_count++;w_in_burst++;beat_cursor++;wbytes+=valid_bytes;
            end
            if(b_hs) begin
                require(aw_seen && w_in_burst==burst_beats,"B before AW/W complete");
                b_count++;aw_seen=0;w_in_burst=0;address+=64'(burst_beats*64);
                burst_beats=expected_beats-beat_cursor;
                if(burst_beats>(4096-int'(address[11:0]))/64) burst_beats=(4096-int'(address[11:0]))/64;
            end
            if(done_valid) begin
                done_wait++;require(!aw_seen && !m_axi_awvalid && !m_axi_wvalid && aw_count==b_count,"done before AXI drained");
                require(done_status==4'(status),$sformatf("status=%0d expected=%0d",done_status,status));
                require(done_sequence==32'(case_id) && done_epoch==32'(case_id+100) && done_bytes==9'(wbytes),"completion identity/bytes");
                if(status==0) require(wbytes==n*item_bytes && beat_cursor==expected_beats,"incomplete successful vector");
                if(status==1 || status==2 || status==5 || status==6 || status==7) require(aw_count==0 && wbytes==0,"rejected operation wrote memory");
                if(status==8 || mode==7 || mode==9) require(aw_count==1,"new burst after write failure");
                if(done_ready) finished=1;
            end
            #1;
            if(cmd_hs) begin cmd_output_fp32=!fp32;cmd_valid=0;cmd_addr=0;cmd_rows=0;cmd_region_base=0;cmd_region_bytes=0;cmd_sequence=0;cmd_epoch=0;expected_epoch=0;end
            if(result_hs) result_valid=0;
            if(commit_hs) commit_valid=0;
            if(cancel_hs) cancel_valid=0;
            if(b_hs) m_axi_bvalid=0;
            if((mode==28 && w_count==1) || (mode==29 && m_axi_bready) || (mode==44 && sent==2)) begin
                @(negedge clk);rst_n=0;cmd_valid=0;result_valid=0;commit_valid=0;cancel_valid=0;m_axi_bvalid=0;
                repeat(4) @(negedge clk);rst_n=1;repeat(8) @(negedge clk);
                require(cmd_ready && !done_valid && !m_axi_awvalid && !m_axi_wvalid && !m_axi_bready,"stale output after coordinated reset");
                reset_done=1;resets++;finished=1;
            end
            if(!finished) @(negedge clk);
        end
        for(int byte_index=0;byte_index<320;byte_index++) begin
            if(byte_index<wbytes) begin
                expected_value=fp32 ? {32'b0,row_fp32(byte_index/item_bytes,mode)}:row_value(byte_index/item_bytes,mode);
                require(memory[byte_index]==expected_value[(byte_index%item_bytes)*8+:8],"memory data");
            end else require(memory[byte_index]==8'ha5,"tail/guard modified");
        end
        require(mode!=25 || w_count>0,"AW-first coverage");require(mode!=26 || w_count>0,"W-first coverage");
        $display("NTRPU_FPWRITER_CASE id=%0d rows=%0d offset=%0d scenario=%0d fp32=%0d status=%0d reset=%0d cycles=%0d inputs=%0d aw=%0d w=%0d b=%0d bytes=%0d",case_id,n,offset,mode,fp32,status,reset_done,cycles,sent,aw_count,w_count,b_count,wbytes);
        total_aw+=aw_count;total_w+=w_count;total_bytes+=wbytes;
        cancel_valid=0;result_valid=0;commit_valid=0;repeat(3) @(negedge clk);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_FPWRITER_SEED seed=%0d",seed);random_state=$urandom(seed);
        $display("NTRPU_FPWRITER_CONFIG watchdog=%0d",WATCHDOG);
`ifdef NTRPU_NETLIST
        repeat(40) @(negedge clk); // Wait past the primitive global startup reset.
`else
        repeat(4) @(negedge clk);
`endif
        rst_n=1;@(posedge clk);#1;
        run_case(1,0,0,0);run_case(7,4032,0,0);run_case(8,0,0,0);run_case(9,4032,0,0);
        run_case(17,4032,0,0);run_case(31,0,0,0);run_case(32,4032,0,0);
        run_case(0,0,1,1);run_case(33,0,1,1);run_case(2,0,2,1);run_case(9,4032,3,1);
        run_case(2,0,4,2);run_case(2,0,5,1);run_case(2,0,6,1);
        run_case(32,4032,7,3);run_case(32,4032,8,3);run_case(32,4032,9,3);run_case(32,4032,10,3);
        run_case(32,4032,11,8);run_case(32,4032,12,8);run_case(32,4032,13,8);
        run_case(9,0,14,5);run_case(9,0,15,5);run_case(9,0,16,5);run_case(9,0,17,5);
        run_case(9,0,18,5);run_case(9,0,19,5);run_case(9,0,20,5);run_case(9,0,21,6);
        run_case(9,0,22,5);run_case(9,0,23,7);run_case(9,0,24,7);
        run_case(32,0,25,0);run_case(32,0,26,0);run_case(32,0,27,0);
        run_case(32,0,28,0);run_case(32,0,29,0);run_case(9,4032,31,0);
        run_case(32,4032,32,0);run_case(9,4032,33,0);run_case(9,0,34,5);run_case(17,4032,0,0);
        run_case(16,4032,35,0);run_case(17,4032,35,0);run_case(31,4032,35,0);
        run_case(9,0,36,5);run_case(9,0,37,5);run_case(9,0,38,5);run_case(9,0,39,5);run_case(9,0,40,5);
        run_case(9,0,41,0,0); // FP64 mode accepts finite values beyond FP32 range.
        run_case(9,0,42,5);run_case(9,0,43,7);run_case(9,0,44,0);
        run_case(17,4032,35,0); // Recovery after pipeline reset.
        run_case(31,4032,0,0,0);run_case(31,4032,0,0,1); // Format changes per descriptor.
        require(aw_stalls>0 && w_stalls>0 && done_stalls>0 && aw_before_w>0 && w_before_aw>0 && resets==3,"coverage counters");
        $display("NTRPU_FPWRITER_PASS cases=%0d aw=%0d w=%0d bytes=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d aw_before_w=%0d w_before_aw=%0d resets=%0d",case_id,total_aw,total_w,total_bytes,aw_stalls,w_stalls,done_stalls,aw_before_w,w_before_aw,resets);
        $finish;
    end
endmodule
