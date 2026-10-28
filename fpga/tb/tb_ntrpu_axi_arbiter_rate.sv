`timescale 1ns/1ps
module tb_ntrpu_axi_arbiter_rate;
    logic clk, rst_n;
    logic [1:0][63:0] s_axi_araddr, s_axi_awaddr;
    logic [1:0][7:0] s_axi_arlen, s_axi_awlen;
    logic [1:0][2:0] s_axi_arsize, s_axi_arprot, s_axi_awsize, s_axi_awprot;
    logic [1:0][1:0] s_axi_arburst, s_axi_awburst;
    logic [1:0][3:0] s_axi_arid, s_axi_arcache, s_axi_awid, s_axi_awcache;
    logic [1:0] s_axi_arlock, s_axi_arvalid, s_axi_awlock, s_axi_awvalid;
    logic [1:0] s_axi_arready, s_axi_awready;
    logic [1:0][511:0] s_axi_rdata;
    logic [1:0][1:0] s_axi_rresp;
    logic [1:0][3:0] s_axi_rid;
    logic [1:0] s_axi_rlast, s_axi_rvalid;
    logic [1:0] s_axi_rready;
    logic [1:0][511:0] s_axi_wdata;
    logic [1:0][63:0] s_axi_wstrb;
    logic [1:0] s_axi_wlast, s_axi_wvalid;
    logic [1:0] s_axi_wready;
    logic [1:0][3:0] s_axi_bid;
    logic [1:0][1:0] s_axi_bresp;
    logic [1:0] s_axi_bvalid;
    logic [1:0] s_axi_bready;
    logic [63:0] m_axi_araddr, m_axi_awaddr;
    logic [7:0] m_axi_arlen, m_axi_awlen;
    logic [2:0] m_axi_arsize, m_axi_arprot, m_axi_awsize, m_axi_awprot;
    logic [1:0] m_axi_arburst, m_axi_awburst;
    logic [3:0] m_axi_arid, m_axi_arcache, m_axi_awid, m_axi_awcache;
    logic m_axi_arlock, m_axi_arvalid, m_axi_awlock, m_axi_awvalid;
    logic m_axi_arready, m_axi_awready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast, m_axi_rvalid;
    logic m_axi_rready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast, m_axi_wvalid;
    logic m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic m_axi_bvalid;
    logic m_axi_bready;

    ntrpu_axi_arbiter dut(.*);
    always #2.5 clk=~clk;
    int ident=0,clock_count=0,expected_interval=1;
    function automatic void need(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL ARBITER RATE case=%0d cycle=%0d: %s",ident,clock_count,why);
    endfunction
    function automatic logic [511:0] payload(input int owner,beat,input bit write_op);
        logic [511:0] word_value;
        for(int lane=0;lane<64;lane++) word_value[lane*8+:8]=8'(owner*97+beat*7+lane*11+int'(write_op)*31);
        return word_value;
    endfunction
    function automatic logic [63:0] mask(input int beat);
        return beat%3==0 ? '1:beat%3==1 ? 64'h5555555555555555:64'haaaaaaaaaaaaaaaa;
    endfunction
    task automatic clear_bus;
        s_axi_araddr=0;s_axi_awaddr=0;s_axi_arlen=0;s_axi_awlen=0;
        s_axi_arsize=0;s_axi_awsize=0;s_axi_arburst=0;s_axi_awburst=0;
        s_axi_arid=0;s_axi_awid=0;s_axi_arcache=0;s_axi_awcache=0;
        s_axi_arprot=0;s_axi_awprot=0;s_axi_arlock=0;s_axi_awlock=0;
        s_axi_arvalid=0;s_axi_awvalid=0;s_axi_rready=0;s_axi_bready=0;
        s_axi_wdata=0;s_axi_wstrb=0;s_axi_wlast=0;s_axi_wvalid=0;
        m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;
        m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;m_axi_rvalid=0;
        m_axi_bid=0;m_axi_bresp=0;m_axi_bvalid=0;
    endtask
    task automatic reset_bus;
        @(negedge clk);rst_n=0;clear_bus();
        repeat(4) @(negedge clk);
        rst_n=1;
    endtask
    task automatic exercise(input int owner,input bit pause);
        int sw,mw,mr,sr,first_mw,last_mw,first_sr,last_sr,overlap_w,overlap_r;
        int max_w,max_r,w_pause_age,r_pause_age,b_age,start_cycle;
        bit saw,sar,maw,mar,sb,mb,whs,wout,rhs,rout,held_w,held_r;
        logic [576:0] previous_w;
        logic [518:0] previous_r;
        logic [63:0] araddr,awaddr;
        sw=0;mw=0;mr=0;sr=0;saw=0;sar=0;maw=0;mar=0;sb=0;mb=0;
        first_mw=-1;last_mw=-1;first_sr=-1;last_sr=-1;overlap_w=0;overlap_r=0;
        max_w=0;max_r=0;w_pause_age=0;r_pause_age=0;b_age=0;held_w=0;held_r=0;
        ident++;reset_bus();start_cycle=clock_count;
        araddr=64'h8000000000+64'(owner)*64'h4000000+64'(ident)*4096;
        awaddr=araddr+64'h2000000;
        $display("NTRPU_RATE_CASE id=%0d owner=%0d pause=%0d beats=64 interval=%0d",ident,owner,pause,expected_interval);
        while(!sb || sr<64) begin
            @(negedge clk);
            s_axi_arvalid[owner]=!sar;s_axi_araddr[owner]=araddr;
            s_axi_arlen[owner]=63;s_axi_arsize[owner]=6;s_axi_arburst[owner]=1;
            s_axi_arid[owner]=7;s_axi_arcache[owner]=2;
            s_axi_awvalid[owner]=!saw;s_axi_awaddr[owner]=awaddr;
            s_axi_awlen[owner]=63;s_axi_awsize[owner]=6;s_axi_awburst[owner]=1;
            s_axi_awid[owner]=7;s_axi_awcache[owner]=2;
            s_axi_wvalid[owner]=sw<64;s_axi_wdata[owner]=payload(owner,sw,1);
            s_axi_wstrb[owner]=mask(sw);s_axi_wlast[owner]=sw==63;
            // Hold each output midway long enough to fill every available slot.
            m_axi_wready=!(pause && mw==32 && w_pause_age<8);
            s_axi_rready[owner]=!(pause && sr==32 && r_pause_age<8);
            if(pause && mw==32 && w_pause_age<8) w_pause_age++;
            if(pause && sr==32 && r_pause_age<8) r_pause_age++;
            s_axi_bready[owner]=1;
            m_axi_arready=!mar;
            m_axi_awready=!maw && (m_axi_wvalid || mw>0);
            m_axi_rvalid=mar && mr<64;m_axi_rdata=payload(owner,mr,0);m_axi_rid=7;
            m_axi_rlast=mr==63;m_axi_rresp=mr%3==0 ? 2:mr%3==1 ? 3:0;
            if(maw && mw==64 && !mb) b_age++;
            m_axi_bvalid=maw && mw==64 && !mb && b_age>=3;
            m_axi_bid=7;m_axi_bresp=pause ? 3:0;
            @(posedge clk);clock_count++;
            need(clock_count-start_cycle<1000,"bounded rate bank");
            if(held_w) need(m_axi_wvalid && previous_w==={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"W remains stable while stalled");
            if(held_r) need(s_axi_rvalid[owner] && previous_r==={s_axi_rdata[owner],s_axi_rresp[owner],s_axi_rid[owner],s_axi_rlast[owner]},"R remains stable while stalled");
            held_w=m_axi_wvalid&&!m_axi_wready;previous_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            held_r=s_axi_rvalid[owner]&&!s_axi_rready[owner];previous_r={s_axi_rdata[owner],s_axi_rresp[owner],s_axi_rid[owner],s_axi_rlast[owner]};
            need(!s_axi_arready[!owner] && !s_axi_awready[!owner] && !s_axi_wready[!owner] &&
                 !s_axi_rvalid[!owner] && !s_axi_bvalid[!owner],"inactive initiator isolated");
            whs=s_axi_wvalid[owner]&&s_axi_wready[owner];wout=m_axi_wvalid&&m_axi_wready;
            rhs=m_axi_rvalid&&m_axi_rready;rout=s_axi_rvalid[owner]&&s_axi_rready[owner];
            if(whs && wout) overlap_w++;
            if(rhs && rout) overlap_r++;
            if(s_axi_arvalid[owner]&&s_axi_arready[owner]) begin need(!sar,"one source AR");sar=1;end
            if(s_axi_awvalid[owner]&&s_axi_awready[owner]) begin need(!saw,"one source AW");saw=1;end
            if(m_axi_arvalid&&m_axi_arready) begin
                need(sar && !mar && m_axi_araddr===araddr && m_axi_arlen==63 && m_axi_arsize==6 &&
                     m_axi_arburst==1 && m_axi_arid==7 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"memory AR preserved");
                mar=1;
            end
            if(m_axi_awvalid&&m_axi_awready) begin
                need(saw && !maw && m_axi_awaddr===awaddr && m_axi_awlen==63 && m_axi_awsize==6 &&
                     m_axi_awburst==1 && m_axi_awid==7 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"memory AW preserved");
                maw=1;
            end
            if(whs) begin
                $display("NTRPU_RATE_BEAT id=%0d channel=SW cycle=%0d beat=%0d last=%0d strb=%016h data=%0128h",ident,clock_count,sw,s_axi_wlast[owner],s_axi_wstrb[owner],s_axi_wdata[owner]);
                sw++;
            end
            if(wout) begin
                need(mw<sw && m_axi_wdata===payload(owner,mw,1) && m_axi_wstrb===mask(mw) && m_axi_wlast===(mw==63),"W data and order");
                $display("NTRPU_RATE_BEAT id=%0d channel=MW cycle=%0d beat=%0d last=%0d strb=%016h data=%0128h",ident,clock_count,mw,m_axi_wlast,m_axi_wstrb,m_axi_wdata);
                if(first_mw<0) first_mw=clock_count;
                if(!pause && last_mw>=0) need(clock_count-last_mw==expected_interval,"continuous W beat interval");
                last_mw=clock_count;mw++;
            end
            if(rhs) begin
                $display("NTRPU_RATE_BEAT id=%0d channel=MR cycle=%0d beat=%0d last=%0d resp=%0d data=%0128h",ident,clock_count,mr,m_axi_rlast,m_axi_rresp,m_axi_rdata);
                mr++;
            end
            if(rout) begin
                need(sr<mr && s_axi_rdata[owner]===payload(owner,sr,0) && s_axi_rlast[owner]===(sr==63) &&
                     s_axi_rid[owner]==7 && s_axi_rresp[owner]===(sr%3==0 ? 2:sr%3==1 ? 3:0),"R data and order");
                $display("NTRPU_RATE_BEAT id=%0d channel=SR cycle=%0d beat=%0d last=%0d resp=%0d data=%0128h",ident,clock_count,sr,s_axi_rlast[owner],s_axi_rresp[owner],s_axi_rdata[owner]);
                if(first_sr<0) first_sr=clock_count;
                if(!pause && last_sr>=0) need(clock_count-last_sr==expected_interval,"continuous R beat interval");
                last_sr=clock_count;sr++;
            end
            if(sw-mw>max_w) max_w=sw-mw;
            if(mr-sr>max_r) max_r=mr-sr;
            need(sw-mw<=2 && mr-sr<=2,"bounded data queues");
            if(m_axi_bvalid&&m_axi_bready) begin need(maw && mw==64 && !mb,"B after full memory W");mb=1;end
            if(s_axi_bvalid[owner]&&s_axi_bready[owner]) begin
                need(mb && !sb && s_axi_bid[owner]==7 && s_axi_bresp[owner]===(pause ? 3:0),"retained B response");
                sb=1;
            end
            #1;
        end
        need(sw==64 && mw==64 && mr==64 && sr==64,"full duplex beat counts");
        if(expected_interval==1) begin
            need(overlap_w>0 && overlap_r>0,"simultaneous enqueue/dequeue");
            if(pause) need(max_w==2 && max_r==2,"second slots exercised");
        end
        $display("NTRPU_RATE_DONE id=%0d w_span=%0d r_span=%0d w_overlap=%0d r_overlap=%0d max_w=%0d max_r=%0d cycles=%0d",
            ident,last_mw-first_mw+1,last_sr-first_sr+1,overlap_w,overlap_r,max_w,max_r,clock_count-start_cycle);
    endtask
    initial begin
        clk=0;rst_n=0;clear_bus();
        if($value$plusargs("EXPECTED_INTERVAL=%d",expected_interval)) begin end
        need(expected_interval==1 || expected_interval==2,"rate reference interval");
        for(int owner=0;owner<2;owner++) begin exercise(owner,0);exercise(owner,1);end
        $display("NTRPU_RATE_PASS cases=4 interval=%0d beats=1024",expected_interval);
        $finish;
    end
endmodule

