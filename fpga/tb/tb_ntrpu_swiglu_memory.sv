`timescale 1ns/1ps
module tb_ntrpu_swiglu_memory;
    logic clk=0,rst_n=0;
    always #2.5 clk=~clk;
    logic job_valid=0,job_ready,cancel_valid=0,cancel_ready,done_valid,done_ready=0;
    logic [13:0] job_elements=0;
    logic [31:0] job_sequence=0,job_epoch=0,expected_epoch=0;
    logic [63:0] job_gate_addr=0,job_gate_base=0,job_gate_bytes=0;
    logic [63:0] job_up_addr=0,job_up_base=0,job_up_bytes=0;
    logic [63:0] job_output_addr=0,job_output_base=0,job_output_bytes=0;
    logic [31:0] job_gate_epoch=0,job_up_epoch=0,job_output_epoch=0;
    logic [3:0] done_status;
    logic [8:0] done_blocks;
    logic [14:0] done_bytes;
    logic [31:0] done_sequence,done_epoch,done_hls_status;
    logic [63:0] m_axi_araddr,m_axi_awaddr,m_axi_wstrb;
    logic [7:0] m_axi_arlen,m_axi_awlen;
    logic [2:0] m_axi_arsize,m_axi_arprot,m_axi_awsize,m_axi_awprot;
    logic [1:0] m_axi_arburst,m_axi_awburst,m_axi_rresp=0,m_axi_bresp=0;
    logic [3:0] m_axi_arid,m_axi_arcache,m_axi_awid,m_axi_awcache,m_axi_rid=0,m_axi_bid=0;
    logic m_axi_arlock,m_axi_arvalid,m_axi_arready=0,m_axi_rlast=0,m_axi_rvalid=0,m_axi_rready;
    logic m_axi_awlock,m_axi_awvalid,m_axi_awready=0,m_axi_wlast,m_axi_wvalid,m_axi_wready=0;
    logic m_axi_bvalid=0,m_axi_bready;
    logic [511:0] m_axi_rdata=0,m_axi_wdata;
    localparam int WATCHDOG=4096;
    localparam logic [63:0] GATE=64'h100000fc0,UP=64'h8000000fc0,OUTPUT=64'ha000000fc0;
    ntrpu_swiglu_memory #(.WATCHDOG_CYCLES(WATCHDOG)) dut(.*);
    int seed=1,random_state,case_id=0,cycles=0,total_cases=35;
    int ar_stalls=0,aw_stalls=0,w_stalls=0,done_stalls=0,total_resets=0,total_cancel=0;
    int total_ar=0,total_r=0,total_w=0,total_b=0;
    function automatic void require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL SWIGLU_MEMORY case=%0d cycle=%0d: %s",case_id,cycles,why);
    endfunction
    function automatic int up_value(input int index);
        if(index%32==0) return 127;
        if(case_id%3==0) return -1;
        return ((index%32)*11+(index/32)*37)%255-127;
    endfunction
    function automatic logic [511:0] input_beat(input logic [63:0] address,input int n,input int mode);
        bit is_up;
        int start,index;
        logic [511:0] value;
        is_up=address>=UP;start=int'((address-(is_up ? UP:GATE))/8);value=0;
        for(int lane=0;lane<8;lane++) begin
            index=start+lane;
            if(index>=n || (mode==10 && !is_up && index==31) || (mode==11 && is_up && index==63))
                value[lane*64+:64]=64'h7ff8000000000000;
            else value[lane*64+:64]=$realtobits(is_up ? real'(up_value(index)):real'(257+2*(index/32)));
        end
        return value;
    endfunction
    function automatic logic [511:0] reference_record(input int block,input int n);
        logic [511:0] value;
        value=0;value[319:256]=$realtobits(real'(257+2*block));
        for(int lane=0;lane<32;lane++)
            if(block*32+lane<n) value[lane*8+:8]=8'(up_value(block*32+lane));
        return value;
    endfunction
    function automatic int status_for(input int mode);
        if((mode>=1 && mode<=5) || (mode>=27 && mode<=29)) return 1;
        if(mode==6) return 2;
        if(mode>=7 && mode<=9) return 3;
        if(mode==10 || mode==11) return 5;
        if(mode==12 || mode==13) return 4;
        if(mode>=14 && mode<=18) return 7;
        if(mode==19 || mode==20) return 8;
        return 0;
    endfunction
    task automatic run_case(input int n,input int mode);
        bit accepted,finished,cancelled,reading,stored,aw_hold,w_hold,ar_hold,done_hold,reset_done;
        bit job_hs,cancel_hs,ar_hs,r_hs,aw_hs,w_hs,b_hs,hls_hs;
        logic [88:0] saved_aw,saved_ar;
        logic [576:0] saved_w;
        logic [123:0] saved_done;
        logic [63:0] read_address,expected_aw;
        logic [511:0] captured,expected_record;
        logic [7:0] memory [0:16511];
        int ar_count,r_count,aw_count,w_count,b_count,beats_left,read_bursts_left;
        int done_wait,write_start,hls_start,hls_count,status,stored_count,blocks,expected_blocks;
        case_id++;cycles=0;ar_count=0;r_count=0;aw_count=0;w_count=0;b_count=0;beats_left=0;
        done_wait=0;write_start=0;hls_start=0;hls_count=0;stored_count=0;blocks=(n+31)/32;
        accepted=0;finished=0;cancelled=0;reading=0;stored=0;aw_hold=0;w_hold=0;ar_hold=0;done_hold=0;reset_done=0;
        status=status_for(mode);read_address=0;captured=0;expected_aw=OUTPUT;
        foreach(memory[i]) memory[i]=8'ha5;
        @(negedge clk);require(job_ready,"previous job not released");
        job_elements=14'(n);job_sequence=32'(case_id);job_epoch=32'(case_id+100);expected_epoch=job_epoch;
        job_gate_addr=GATE;job_gate_base=GATE;job_gate_bytes=64'(((n+7)/8)*64);job_gate_epoch=job_epoch;
        job_up_addr=UP;job_up_base=UP;job_up_bytes=job_gate_bytes;job_up_epoch=job_epoch;
        job_output_addr=OUTPUT;job_output_base=OUTPUT;job_output_bytes=64'(blocks*64);job_output_epoch=job_epoch;
        if(mode==2) job_gate_bytes--;
        if(mode==3) job_up_bytes--;
        if(mode==4) job_output_bytes--;
        if(mode==5 || mode==27) begin
            job_output_addr=mode==5 ? GATE:UP;job_output_base=job_output_addr;
        end
        if(mode==6) job_up_epoch++;
        if(mode==28) begin job_gate_base=64'hffffffffffffffc0;job_gate_bytes=4096;end
        if(mode==29) job_up_addr++;
        job_valid=1;cancel_valid=0;m_axi_rvalid=0;m_axi_bvalid=0;
        while(!finished) begin
            m_axi_arready=!reading && $urandom_range(0,3)!=0;
            m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if(mode==14 && cycles<30) m_axi_arready=0;
            if(mode==19 && cycles<WATCHDOG+48) m_axi_arready=0;
            if(mode==18 && b_count==1 && !cancelled) m_axi_arready=0;
            if(mode==17 && (write_start==0 || cycles<write_start+16)) m_axi_wready=0;
            if(mode==23) m_axi_awready=0;
            done_ready=done_wait>=5 && $urandom_range(0,2)!=0;
            if(!cancelled && accepted &&
               ((mode==14 && m_axi_arvalid) || (mode==15 && ar_count>0) ||
                (mode==16 && hls_start>0 && cycles>=hls_start+10) ||
                (mode==17 && aw_count>0) || (mode==18 && b_count==1 && m_axi_arvalid))) cancel_valid=1;
            if(reading && !m_axi_rvalid && $urandom_range(0,3)!=0 && !(mode==15 && cycles<30)) begin
                m_axi_rvalid=1;m_axi_rdata=input_beat(read_address,n,mode);m_axi_rid=0;m_axi_rresp=0;
                m_axi_rlast=beats_left==1;
                if((mode==7 && read_address==GATE) || (mode==8 && read_address==UP) ||
                   (mode==9 && read_address==GATE+256)) m_axi_rresp=2;
            end
            if(aw_count==w_count && aw_count>b_count && !m_axi_bvalid &&
               !(mode==20 && cycles<write_start+WATCHDOG+48) && !(mode==24 && cycles<write_start+30)) begin
                m_axi_bvalid=1;m_axi_bresp=(mode==12 && b_count==1) ? 2:0;m_axi_bid=mode==13 ? 1:0;
            end
            @(posedge clk);cycles++;require(cycles<2000000,"bank timeout");
            job_hs=job_valid&&job_ready;cancel_hs=cancel_valid&&cancel_ready;
            ar_hs=m_axi_arvalid&&m_axi_arready;r_hs=m_axi_rvalid&&m_axi_rready;
            aw_hs=m_axi_awvalid&&m_axi_awready;w_hs=m_axi_wvalid&&m_axi_wready;b_hs=m_axi_bvalid&&m_axi_bready;
            hls_hs=dut.activation.ap_start;
            if(ar_hold) require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock}===saved_ar,"AR stability");
            if(aw_hold) require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"AW stability");
            if(w_hold) require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"W stability");
            if(done_hold) require(done_valid && {done_status,done_blocks,done_bytes,done_sequence,done_epoch,done_hls_status}===saved_done,"completion stability");
            ar_hold=m_axi_arvalid&&!m_axi_arready;aw_hold=m_axi_awvalid&&!m_axi_awready;
            w_hold=m_axi_wvalid&&!m_axi_wready;done_hold=done_valid&&!done_ready;
            saved_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};saved_done={done_status,done_blocks,done_bytes,done_sequence,done_epoch,done_hls_status};
            ar_stalls+=int'(ar_hold);aw_stalls+=int'(aw_hold);w_stalls+=int'(w_hold);done_stalls+=int'(done_hold);
            if(job_hs) begin require(!accepted,"duplicate job");accepted=1;end
            if(cancel_hs) begin require(!cancelled,"duplicate cancel");cancelled=1;total_cancel++;end
            if(ar_hs) begin
                require(accepted && !reading,"multiple read bursts");
                require((m_axi_araddr>=GATE && m_axi_araddr+64'(int'(m_axi_arlen)+1)*64<=GATE+64'(((n+7)/8)*64)) ||
                        (m_axi_araddr>=UP && m_axi_araddr+64'(int'(m_axi_arlen)+1)*64<=UP+64'(((n+7)/8)*64)),"AR registered ranges");
                require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR 4KiB crossing");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR attributes");
                ar_count++;reading=1;read_address=m_axi_araddr;beats_left=int'(m_axi_arlen)+1;
                $display("NTRPU_SWIGLU_AR id=%0d address=%016h beats=%0d",case_id,m_axi_araddr,beats_left);
            end
            if(r_hs) begin require(reading && beats_left>0,"unexpected R");r_count++;beats_left--;read_address+=64;if(beats_left==0) reading=0;end
            if(hls_hs) begin hls_count++;hls_start=cycles;end
            if(m_axi_awvalid || m_axi_wvalid) begin
                require(hls_count>b_count && !reading,"write before computation/read drain");
                if(write_start==0) write_start=cycles;
            end
            if(aw_hs) begin
                require(aw_count==b_count && m_axi_awaddr==OUTPUT+64'(aw_count*64),"AW address/order");
                require(m_axi_awlen==0 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                aw_count++;
            end
            if(w_hs) begin
                require(w_count==b_count,"duplicate/pipelined W");expected_record=reference_record(w_count,n);
                require(m_axi_wdata===expected_record && m_axi_wstrb=='1 && m_axi_wlast,"activation bytes/scale/padding");
                captured=m_axi_wdata;
                $display("NTRPU_SWIGLU_W id=%0d block=%0d data=%0128h",case_id,w_count,m_axi_wdata);
                w_count++;
            end
            if(aw_count==w_count && aw_count>stored_count) begin
                for(int lane=0;lane<64;lane++) memory[64+stored_count*64+lane]=captured[lane*8+:8];
                stored_count++;
            end
            if(b_hs) begin require(aw_count==w_count && aw_count==b_count+1,"B before AW/W");b_count++;end
            if(done_valid) begin
                done_wait++;
                require(!reading && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && aw_count==b_count && w_count==b_count,"completion before AXI drain");
                require(done_status==4'(status),$sformatf("status=%0d expected=%0d hls=%0d",done_status,status,done_hls_status));
                require(done_sequence==32'(case_id) && done_epoch==32'(case_id+100) && done_bytes==15'(w_count*64),"completion identity/bytes");
                expected_blocks=status==0 ? blocks:((mode==9 || mode==11 || mode==12 || mode==18) ? 1:0);
                require(done_blocks==9'(expected_blocks),"private completed blocks");
                if(mode==10 || mode==11) require(done_hls_status==2,"nonfinite HLS error propagation");
                if(status==1 || status==2) require(ar_count==0 && w_count==0,"invalid descriptor bus activity");
                if(status==0) require(w_count==blocks && hls_count==blocks && r_count==((n+7)/8)*2,"complete vector traffic");
                if(done_ready) finished=1;
            end
            #1;
            if(job_hs) begin
                job_valid=0;job_elements=0;job_sequence=0;job_epoch=0;expected_epoch=0;
                job_gate_addr=0;job_gate_base=0;job_gate_bytes=0;job_gate_epoch=0;
                job_up_addr=0;job_up_base=0;job_up_bytes=0;job_up_epoch=0;
                job_output_addr=0;job_output_base=0;job_output_bytes=0;job_output_epoch=0;
            end
            if(cancel_hs) cancel_valid=0;
            if(r_hs) m_axi_rvalid=0;
            if(b_hs) m_axi_bvalid=0;
            if((mode==21 && r_count==1) || (mode==22 && hls_start>0 && cycles>=hls_start+10) ||
               (mode==23 && w_count==1) || (mode==24 && aw_count==1 && w_count==1) || (mode==25 && b_count==1)) begin
                @(negedge clk);rst_n=0;job_valid=0;cancel_valid=0;m_axi_rvalid=0;m_axi_bvalid=0;
                repeat(4) @(negedge clk);rst_n=1;repeat(8) @(negedge clk);
                require(job_ready && !done_valid && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && !m_axi_bready,"reset recovery");
                reset_done=1;total_resets++;finished=1;
            end
            if(!finished) @(negedge clk);
        end
        for(int i=0;i<16512;i++) begin
            if(i>=64 && i<64+stored_count*64) begin
                expected_record=reference_record((i-64)/64,n);
                require(memory[i]==expected_record[((i-64)%64)*8+:8],"physical record memory");
            end else require(memory[i]==8'ha5,"destination guard changed");
        end
        require(cancelled==(status==7),"cancel coverage");
        $display("NTRPU_SWIGLU_MEMORY_CASE id=%0d elements=%0d scenario=%0d status=%0d reset=%0d cycles=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d stored=%0d hls=%0d cancel=%0d",case_id,n,mode,status,reset_done,cycles,ar_count,r_count,aw_count,w_count,b_count,stored_count,hls_count,cancelled);
        total_ar+=ar_count;total_r+=r_count;total_w+=w_count;total_b+=b_count;
        cancel_valid=0;repeat(3) @(negedge clk);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_SWIGLU_MEMORY_SEED seed=%0d",seed);random_state=$urandom(seed);
        $display("NTRPU_SWIGLU_MEMORY_PATTERN gate=257_plus_2_block");
        repeat(4) @(negedge clk);rst_n=1;@(posedge clk);#1;
`ifdef NTRPU_SWIGLU_LARGE
        total_cases=1;run_case(8192,0);
`else
        run_case(1,0);run_case(31,0);run_case(32,0);run_case(33,0);run_case(720,0);
        run_case(0,1);run_case(8193,1);
        for(int mode=2;mode<=29;mode++) run_case(mode==26 ? 33:64,mode);
        require(total_resets==5 && total_cancel==5,"fault/reset coverage");
`endif
        require(case_id==total_cases && ar_stalls>0 && aw_stalls>0 && w_stalls>0 && done_stalls>0,"suite coverage");
        $display("NTRPU_SWIGLU_MEMORY_PASS cases=%0d ar=%0d r=%0d w=%0d b=%0d ar_stalls=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d resets=%0d cancels=%0d",case_id,total_ar,total_r,total_w,total_b,ar_stalls,aw_stalls,w_stalls,done_stalls,total_resets,total_cancel);
        $finish;
    end
endmodule
