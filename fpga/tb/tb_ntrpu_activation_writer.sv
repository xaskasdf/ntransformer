`timescale 1ns/1ps
module tb_ntrpu_activation_writer;
    logic clk=0,rst_n=0;
    always #2.5 clk=~clk;
    logic cmd_valid=0,cmd_ready,cancel_valid=0,cancel_ready,done_valid,done_ready=0;
    logic [63:0] cmd_addr=0,cmd_region_base=0,cmd_region_bytes=0;
    logic [511:0] cmd_record=0;
    logic [5:0] cmd_elements=0;
    logic [31:0] cmd_sequence=0,cmd_epoch=0,expected_epoch=0,done_sequence,done_epoch;
    logic [3:0] done_status;
    logic [6:0] done_bytes;
    logic [63:0] m_axi_awaddr,m_axi_wstrb;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize,m_axi_awprot;
    logic [1:0] m_axi_awburst,m_axi_bresp=0;
    logic [3:0] m_axi_awid,m_axi_awcache,m_axi_bid=0;
    logic m_axi_awlock,m_axi_awvalid,m_axi_awready=0,m_axi_wlast,m_axi_wvalid,m_axi_wready=0;
    logic m_axi_bvalid=0,m_axi_bready;
    logic [511:0] m_axi_wdata;
`ifdef NTRPU_NETLIST
    localparam int WATCHDOG=65536,CASE_LIMIT=80000;
    ntrpu_activation_writer dut(.*);
`else
    localparam int WATCHDOG=32,CASE_LIMIT=2000;
    ntrpu_activation_writer #(.WATCHDOG_CYCLES(WATCHDOG)) dut(.*);
`endif
    localparam logic [63:0] BASE=64'ha000000000;
    int seed=1,random_state,case_id=0,cycles=0;
    int aw_stalls=0,w_stalls=0,done_stalls=0,resets=0,total_aw=0,total_w=0,total_b=0,total_stored=0;
    int w_before_aw=0,aw_before_w=0,total_cancel=0;
    function automatic void require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL ACTIVATION_WRITER case=%0d cycle=%0d: %s",case_id,cycles,why);
    endfunction
    function automatic logic [511:0] make_record(input int n,input int mode);
        logic [511:0] value;
        value=0;
        for(int i=0;i<n && i<32;i++) value[i*8+:8]=8'(case_id*17+i*31);
        value[319:256]=64'h3fe0000000000000;
        case(mode)
            9: value[319:256]=64'h7ff8000000000000;
            10: value[319:256]=64'h7ff0000000000000;
            11: value[319:256]=64'hbfe0000000000000;
            12: value[319:256]=64'h4780204070e1c388;
            13: value[320]=1;
            14: value[511]=1;
            15: value[248+:8]=1;
            16: value[319:256]=64'h4780204070e1c387;
            17: value[319:256]=64'h8000000000000000;
            18: value[319:256]=1;
            19: begin value[255:0]='1;value[7:0]=8'h7f;value[319:256]=64'h4070100000000000;end
            46: value[319:256]=64'hfff0000000000000;
        endcase
        return value;
    endfunction
    function automatic int status_for(input int mode);
        if(mode==1 || mode==2 || mode==3 || mode==5 || mode==6 || mode==7 || mode==8) return 1;
        if(mode==4) return 2;
        if((mode>=9 && mode<=15) || mode==46) return 4;
        if(mode>=20 && mode<=22) return 3;
        if((mode>=23 && mode<=25) || mode==42) return 8;
        if((mode>=29 && mode<=36) || mode==43) return 7;
        return 0;
    endfunction
    task automatic run_case(input int n,input int offset,input int mode);
        bit accepted,finished,cancelled,aw_hold,w_hold,done_hold,reset_done,stored;
        bit cmd_hs,cancel_hs,aw_hs,w_hs,b_hs;
        logic [88:0] saved_aw;
        logic [576:0] saved_w;
        logic [74:0] saved_done;
        logic [63:0] address;
        logic [511:0] wanted,captured;
        logic [7:0] memory [0:191];
        int aw_count,w_count,b_count,done_wait,send_start,accept_cycle,status;
        case_id++;cycles=0;aw_count=0;w_count=0;b_count=0;done_wait=0;send_start=0;accept_cycle=0;
        accepted=0;finished=0;cancelled=0;aw_hold=0;w_hold=0;done_hold=0;reset_done=0;stored=0;captured=0;
        status=status_for(mode);wanted=make_record(n,mode);address=BASE+64'(offset);
        foreach(memory[i]) memory[i]=8'ha5;
        @(negedge clk);require(cmd_ready,"previous command not released");
        cmd_addr=address;cmd_region_base=address;cmd_region_bytes=64;
        cmd_elements=6'(n);cmd_record=wanted;cmd_sequence=32'(case_id);cmd_epoch=32'(case_id+100);expected_epoch=cmd_epoch;
        if(mode==2) cmd_addr++;
        if(mode==3) cmd_region_bytes=63;
        if(mode==4) expected_epoch++;
        if(mode==5) cmd_region_base=cmd_addr+64;
        if(mode==6) begin cmd_region_base=64'hffffffffffffffc0;cmd_region_bytes=128;end
        if(mode==7) begin cmd_addr=64'hffffffffffffffc0;cmd_region_base=0;cmd_region_bytes='1;end
        if(mode==8) cmd_region_bytes=0;
        if(mode==45) begin address=64'hffffffffffffff80;cmd_addr=address;cmd_region_base=address;end
        cmd_valid=1;cancel_valid=0;m_axi_bvalid=0;
        while(!finished) begin
            m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if((mode==23 || mode==42) && (send_start==0 || cycles<send_start+WATCHDOG+48)) m_axi_awready=0;
            if(mode==24 && (send_start==0 || cycles<send_start+WATCHDOG+48)) m_axi_wready=0;
            if((mode==27 || mode==34 || mode==38) && (send_start==0 || cycles<send_start+12)) m_axi_awready=0;
            if((mode==26 || mode==35 || mode==39) && (send_start==0 || cycles<send_start+12)) m_axi_wready=0;
            if(mode==33 && (send_start==0 || cycles<send_start+12)) begin m_axi_awready=0;m_axi_wready=0;end
            done_ready=done_wait>=5 && $urandom_range(0,2)!=0;
            if(!cancelled && accepted) begin
                if((mode>=29 && mode<=32 && cycles==accept_cycle+mode-29) ||
                   (mode==33 && send_start>0) || (mode==34 && w_count>0) ||
                   (mode==35 && aw_count>0) || ((mode==36 || mode==43) && aw_count>0 && w_count>0)) cancel_valid=1;
            end
            if(aw_count==1 && w_count==1 && b_count==0 && !m_axi_bvalid &&
               !(mode==25 && cycles<send_start+WATCHDOG+48) &&
               !((mode==28 || mode==36 || mode==40) && cycles<send_start+20)) begin
                m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;
                if(mode==20 || mode==42 || mode==43) m_axi_bresp=2;
                if(mode==21) m_axi_bresp=3;
                if(mode==22) m_axi_bid=15;
            end
            @(posedge clk);cycles++;require(cycles<CASE_LIMIT,"bank timeout");
            cmd_hs=cmd_valid&&cmd_ready;cancel_hs=cancel_valid&&cancel_ready;
            aw_hs=m_axi_awvalid&&m_axi_awready;w_hs=m_axi_wvalid&&m_axi_wready;b_hs=m_axi_bvalid&&m_axi_bready;
            if(aw_hold) require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"AW stability across cancel/timeout");
            if(w_hold) require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"W stability across cancel/timeout");
            if(done_hold) require(done_valid && {done_status,done_bytes,done_sequence,done_epoch}===saved_done,"completion stability");
            aw_hold=m_axi_awvalid&&!m_axi_awready;w_hold=m_axi_wvalid&&!m_axi_wready;done_hold=done_valid&&!done_ready;
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};saved_done={done_status,done_bytes,done_sequence,done_epoch};
            aw_stalls+=int'(aw_hold);w_stalls+=int'(w_hold);done_stalls+=int'(done_hold);
            if(cmd_hs) begin require(!accepted,"duplicate command");accepted=1;accept_cycle=cycles;end
            if(cancel_hs) begin require(!cancelled,"duplicate cancellation");cancelled=1;total_cancel++;end
            if(m_axi_awvalid || m_axi_wvalid) begin
                require(accepted,"write before command");
                require(status!=1 && status!=2 && status!=4 && !(mode>=29 && mode<=32),"rejected command emitted AXI");
                if(send_start==0) send_start=cycles;
            end
            if(aw_hs) begin
                require(aw_count==0,"duplicate AW");
                require(m_axi_awaddr==address && m_axi_awlen==0,"AW address/length/high bits");
                require(int'(m_axi_awaddr[11:0])+64<=4096,"4KiB crossing");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                if(w_count==0 && !w_hs) aw_before_w++;
                aw_count++;
            end
            if(w_hs) begin
                require(w_count==0,"duplicate W");
                require(m_axi_wdata===wanted && m_axi_wstrb=='1 && m_axi_wlast,"raw record/byte order/strobes");
                if(aw_count==0) w_before_aw++;
                captured=m_axi_wdata;w_count++;
                $display("NTRPU_ACT_WRITE id=%0d data=%0128h",case_id,m_axi_wdata);
            end
            if(!stored && aw_count==1 && w_count==1) begin
                for(int lane=0;lane<64;lane++) memory[64+lane]=captured[lane*8+:8];
                stored=1;
            end
            if(b_hs) begin require(aw_count==1 && w_count==1 && b_count==0,"B before AW/W or duplicate");b_count++;end
            if(done_valid) begin
                done_wait++;
                require(!m_axi_awvalid && !m_axi_wvalid && !m_axi_bready && aw_count==b_count && w_count==b_count,"done before AXI drained");
                require(done_status==4'(status),$sformatf("status=%0d expected=%0d",done_status,status));
                require(done_sequence==32'(case_id) && done_epoch==32'(case_id+100) && done_bytes==7'(w_count*64),"completion identity/bytes");
                if(status==0) require(stored && b_count==1,"empty successful write");
                if(done_ready) finished=1;
            end
            #1;
            if(cmd_hs) begin cmd_valid=0;cmd_addr=0;cmd_record='1;cmd_elements=0;cmd_region_base=0;cmd_region_bytes=0;cmd_sequence=0;cmd_epoch=0;expected_epoch=0;end
            if(cancel_hs) cancel_valid=0;
            if(b_hs) m_axi_bvalid=0;
            if((mode==37 && accepted) || (mode==38 && w_count==1) ||
               (mode==39 && aw_count==1) || (mode==40 && aw_count==1 && w_count==1) || (mode==41 && done_valid)) begin
                @(negedge clk);rst_n=0;cmd_valid=0;cancel_valid=0;m_axi_bvalid=0;
                repeat(4) @(negedge clk);rst_n=1;repeat(8) @(negedge clk);
                require(cmd_ready && !done_valid && !m_axi_awvalid && !m_axi_wvalid && !m_axi_bready,"stale output after coordinated reset");
                reset_done=1;resets++;finished=1;
            end
            if(!finished) @(negedge clk);
        end
        for(int i=0;i<192;i++) begin
            if(stored && i>=64 && i<128) require(memory[i]==wanted[(i-64)*8+:8],"physical record bytes");
            else require(memory[i]==8'ha5,"guard or private bytes changed");
        end
        require((status==7)==cancelled,"cancellation coverage");
        $display("NTRPU_ACT_WRITER_CASE id=%0d elements=%0d offset=%0d scenario=%0d status=%0d reset=%0d cycles=%0d aw=%0d w=%0d b=%0d bytes=%0d stored=%0d cancel=%0d",case_id,n,offset,mode,status,reset_done,cycles,aw_count,w_count,b_count,w_count*64,int'(stored)*64,cancelled);
        total_aw+=aw_count;total_w+=w_count;total_b+=b_count;total_stored+=int'(stored)*64;
        cancel_valid=0;repeat(3) @(negedge clk);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_ACT_WRITER_SEED seed=%0d",seed);random_state=$urandom(seed);
        $display("NTRPU_ACT_WRITER_CONFIG watchdog=%0d",WATCHDOG);
`ifdef NTRPU_NETLIST
        repeat(40) @(negedge clk);
`else
        repeat(4) @(negedge clk);
`endif
        rst_n=1;@(posedge clk);#1;
        run_case(1,0,0);run_case(7,4032,0);run_case(8,0,0);run_case(16,4032,0);run_case(31,0,0);run_case(32,4032,0);
        run_case(0,0,1);run_case(33,0,1);
        for(int mode=2;mode<=46;mode++) run_case(mode==15 ? 31:32,4032,mode);
        require(case_id==53 && aw_stalls>0 && w_stalls>0 && done_stalls>0 && aw_before_w>0 && w_before_aw>0 && resets==5 && total_cancel==9,"coverage counters");
        $display("NTRPU_ACT_WRITER_PASS cases=%0d aw=%0d w=%0d b=%0d stored=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d aw_before_w=%0d w_before_aw=%0d resets=%0d cancels=%0d",case_id,total_aw,total_w,total_b,total_stored,aw_stalls,w_stalls,done_stalls,aw_before_w,w_before_aw,resets,total_cancel);
        $finish;
    end
endmodule
