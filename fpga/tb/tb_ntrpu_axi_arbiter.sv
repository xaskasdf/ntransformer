`timescale 1ns/1ps
module tb_ntrpu_axi_arbiter;
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
    int seed=1,rng,bank=0,clock_count=0,pipelined=0;
    int total_reads=0,total_writes=0,total_read_beats=0,total_write_beats=0;
    int reset_count=0,overlap_cycles=0,aw_wait_w=0,w_before_aw=0;
    int stalls[0:6];
    function automatic void need(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL ARBITER bank=%0d cycle=%0d: %s",bank,clock_count,why);
    endfunction
    function automatic int length(input int tx);
        case(tx%6) 0:return 1;1:return 2;2:return 3;3:return 4;4:return 16;default:return 64;endcase
    endfunction
    function automatic logic [63:0] address(input int owner,tx,input bit write_op);
        return 64'h100000000+64'(owner)*64'h1000000000+64'(write_op)*64'h20000000+
               64'(tx)*4096+(length(tx)==1 ? 4032:0);
    endfunction
    function automatic logic [1:0] response(input int tx);
        return tx%5==3 ? 2:tx%5==4 ? 3:0;
    endfunction
    function automatic logic [511:0] payload(input int owner,tx,beat,input bit write_op);
        logic [511:0] v;
        for(int k=0;k<64;k++) v[k*8+:8]=8'(owner*71+tx*13+beat*17+k*3+int'(write_op)*29);
        return v;
    endfunction
    function automatic logic [63:0] mask(input int tx,beat);
        if(tx%4==0) return '1;
        if(tx%4==1 && beat==0) return 0;
        return 64'h0123456789abcdef ^ (64'h1111111111111111*64'(beat%16));
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
        repeat(4) begin
            @(posedge clk);#1;
            need({m_axi_arvalid,m_axi_awvalid,m_axi_wvalid,m_axi_rready,m_axi_bready,
                  s_axi_arready,s_axi_awready,s_axi_wready,s_axi_rvalid,s_axi_bvalid}==='0,"reset interface quiescence");
        end
        @(negedge clk);rst_n=1;
    endtask
    task automatic run_bank(input int count,abort_kind=0);
        int rt[0:1],wt[0:1],rb[0:1],wb[0:1],write_age[0:1],rhold_age[0:1],bhold_age[0:1];
        bit ractive[0:1],wactive[0:1],saw[0:1];
        bit mem_read,mem_write,mem_aw,arhs,awhs,whs,rhs,bhs,aborted;
        int mr_owner,mr_tx,mr_beat,mw_owner,mw_tx,received,checked,bdelay;
        logic [511:0] wqueue[0:63];
        logic [63:0] qmask[0:63];bit qlast[0:63];
        int read_grants,write_grants,read_completed,write_completed,last_read_owner,last_write_owner;
        int grant_owner,grant_tx,limit_cycle,local_w_before,local_wait_w,local_overlap;
        int request_r,request_aw,request_w,request_beat;
        logic [88:0] hold_ar,hold_aw;
        logic [576:0] hold_w;
        logic [518:0] hold_r[0:1];
        logic [5:0] hold_b[0:1];
        bit ar_held,aw_held,w_held,r_held[0:1],b_held[0:1];
        bank++;mem_read=0;mem_write=0;mem_aw=0;received=0;checked=0;bdelay=0;
        aborted=0;read_grants=0;write_grants=0;read_completed=0;write_completed=0;
        last_read_owner=-1;last_write_owner=-1;limit_cycle=clock_count+50000;
        local_w_before=0;local_wait_w=0;local_overlap=0;
        ar_held=0;aw_held=0;w_held=0;
        for(int o=0;o<2;o++) begin
            rt[o]=0;wt[o]=0;rb[o]=0;wb[o]=0;write_age[o]=0;rhold_age[o]=0;bhold_age[o]=0;
            ractive[o]=0;wactive[o]=0;saw[o]=0;r_held[o]=0;b_held[o]=0;
        end
        $display("NTRPU_ARB_BANK id=%0d count=%0d abort=%0d",bank,count,abort_kind);
        while(read_completed<count*2 || write_completed<count*2) begin
            @(negedge clk);
            for(int o=0;o<2;o++) begin
                // Requests remain continuously pending from both initiators.
                request_r=rt[o]+int'(pipelined && ractive[o]);
                request_aw=wt[o]+int'(pipelined && saw[o]);
                request_w=wt[o]+int'(pipelined && wb[o]==length(wt[o]));
                request_beat=request_w!=wt[o] ? 0:wb[o];
                s_axi_arvalid[o]=request_r<count && (pipelined || !ractive[o]);
                s_axi_araddr[o]=!pipelined && ractive[o] ? 'hbadbad01:address(o,request_r,0);
                s_axi_arlen[o]=8'(length(request_r)-1);s_axi_arsize[o]=6;s_axi_arburst[o]=1;
                s_axi_arid[o]=4'(request_r);s_axi_arcache[o]=4'(request_r);s_axi_arprot[o]=3'(request_r);
                s_axi_awvalid[o]=request_aw<count && (pipelined || !saw[o]) && (pipelined || wt[o]%3!=2 || write_age[o]>=4);
                s_axi_awaddr[o]=!pipelined && saw[o] ? 'hbadbad02:address(o,request_aw,1);
                s_axi_awlen[o]=8'(length(request_aw)-1);s_axi_awsize[o]=6;s_axi_awburst[o]=1;
                s_axi_awid[o]=4'(request_aw);s_axi_awcache[o]=4'(request_aw);s_axi_awprot[o]=3'(request_aw);
                s_axi_wvalid[o]=request_w<count && request_beat<length(request_w) && (pipelined || wt[o]%3!=1 || write_age[o]>=8);
                s_axi_wdata[o]=payload(o,request_w,request_beat,1);s_axi_wstrb[o]=mask(request_w,request_beat);
                s_axi_wlast[o]=request_beat==length(request_w)-1;
                s_axi_rready[o]=abort_kind!=2 && rhold_age[o]>=3 && ($random(rng)&3)!=0;
                s_axi_bready[o]=abort_kind!=4 && bhold_age[o]>=4 && ($random(rng)&3)!=0;
            end
            m_axi_arready=abort_kind!=1 && !mem_read && ($random(rng)&3)!=0;
            // A legal slave may wait for WVALID before accepting AW. Keep
            // readiness latched by observed W beats if W already transferred.
            m_axi_awready=abort_kind!=3 && !mem_aw &&
                (m_axi_wvalid || received>0) && ($random(rng)&3)!=0;
            m_axi_wready=received<64 && !m_axi_bvalid && ($random(rng)&3)!=0;
            if(mem_read && !m_axi_rvalid && ($random(rng)&3)!=0) begin
                m_axi_rvalid=1;m_axi_rdata=payload(mr_owner,mr_tx,mr_beat,0);
                m_axi_rid=4'(mr_tx);m_axi_rresp=response(mr_tx);m_axi_rlast=mr_beat==length(mr_tx)-1;
            end
            if(mem_aw && checked==length(mw_tx) && !m_axi_bvalid) begin
                bdelay++;
                if(bdelay>=4) begin m_axi_bvalid=1;m_axi_bid=4'(mw_tx);m_axi_bresp=response(mw_tx);end
            end
            @(posedge clk);clock_count++;
            need(clock_count<limit_cycle,"bounded progress");
            arhs=m_axi_arvalid&&m_axi_arready;awhs=m_axi_awvalid&&m_axi_awready;
            whs=m_axi_wvalid&&m_axi_wready;rhs=m_axi_rvalid&&m_axi_rready;bhs=m_axi_bvalid&&m_axi_bready;
            if(ar_held) need(m_axi_arvalid && hold_ar==={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock},"AR held stable");
            if(aw_held) need(m_axi_awvalid && hold_aw==={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock},"AW held stable");
            if(w_held) need(m_axi_wvalid && hold_w==={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"W held stable");
            ar_held=m_axi_arvalid&&!m_axi_arready;aw_held=m_axi_awvalid&&!m_axi_awready;w_held=m_axi_wvalid&&!m_axi_wready;
            hold_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            hold_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            hold_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            stalls[0]+=int'(ar_held);stalls[1]+=int'(aw_held);stalls[2]+=int'(w_held);
            need(!(s_axi_arready[0]&&s_axi_arready[1]) && !(s_axi_awready[0]&&s_axi_awready[1]) &&
                 !(s_axi_wready[0]&&s_axi_wready[1]) && !(s_axi_rvalid[0]&&s_axi_rvalid[1]) &&
                 !(s_axi_bvalid[0]&&s_axi_bvalid[1]),"exclusive per-channel owner");
            for(int o=0;o<2;o++) begin
                if(r_held[o]) need(s_axi_rvalid[o] && hold_r[o]==={s_axi_rdata[o],s_axi_rresp[o],s_axi_rid[o],s_axi_rlast[o]},"R held stable");
                if(b_held[o]) need(s_axi_bvalid[o] && hold_b[o]==={s_axi_bid[o],s_axi_bresp[o]},"B held stable");
                r_held[o]=s_axi_rvalid[o]&&!s_axi_rready[o];b_held[o]=s_axi_bvalid[o]&&!s_axi_bready[o];
                hold_r[o]={s_axi_rdata[o],s_axi_rresp[o],s_axi_rid[o],s_axi_rlast[o]};
                hold_b[o]={s_axi_bid[o],s_axi_bresp[o]};
                stalls[3+o]+=int'(r_held[o]);stalls[5+o]+=int'(b_held[o]);
                write_age[o]++;
                if(s_axi_rvalid[o]) rhold_age[o]++;
                if(s_axi_bvalid[o]) bhold_age[o]++;
                if(s_axi_arvalid[o]&&s_axi_arready[o]) begin
                    need(!ractive[o],"read accepted twice");ractive[o]=1;
                    $display("NTRPU_ARB_S_AR bank=%0d owner=%0d tx=%0d addr=%016h",bank,o,rt[o],s_axi_araddr[o]);
                end
                if(s_axi_awvalid[o]&&s_axi_awready[o]) begin
                    need(!saw[o],"write address accepted twice");saw[o]=1;wactive[o]=1;
                    $display("NTRPU_ARB_S_AW bank=%0d owner=%0d tx=%0d addr=%016h",bank,o,wt[o],s_axi_awaddr[o]);
                end
                if(s_axi_wvalid[o]&&s_axi_wready[o]) begin
                    need(saw[o],"W routing requires selected source AW");
                    need(wb[o]<length(wt[o]),"next burst W accepted before its own AW/grant");
                    $display("NTRPU_ARB_S_W bank=%0d owner=%0d tx=%0d beat=%0d last=%0d strb=%016h data=%0128h",
                        bank,o,wt[o],wb[o],s_axi_wlast[o],s_axi_wstrb[o],s_axi_wdata[o]);
                    wb[o]++;
                end
                if(s_axi_rvalid[o]&&s_axi_rready[o]) begin
                    need(ractive[o],"unsolicited R recipient");
                    need(s_axi_rdata[o]===payload(o,rt[o],rb[o],0) && s_axi_rid[o]===4'(rt[o]) &&
                        s_axi_rresp[o]===response(rt[o]) && s_axi_rlast[o]===(rb[o]==length(rt[o])-1),"R payload/ID/status/last recipient");
                    $display("NTRPU_ARB_S_R bank=%0d owner=%0d tx=%0d beat=%0d rid=%0d resp=%0d last=%0d data=%0128h",
                        bank,o,rt[o],rb[o],s_axi_rid[o],s_axi_rresp[o],s_axi_rlast[o],s_axi_rdata[o]);
                    rb[o]++;total_read_beats++;
                    if(s_axi_rlast[o]) begin rt[o]++;rb[o]=0;ractive[o]=0;read_completed++;total_reads++;end
                    rhold_age[o]=0;
                end
                if(s_axi_bvalid[o]&&s_axi_bready[o]) begin
                    need(wactive[o] && saw[o] && wb[o]==length(wt[o]),"B before complete source W");
                    need(s_axi_bid[o]===4'(wt[o]) && s_axi_bresp[o]===response(wt[o]),"B recipient ID/status");
                    $display("NTRPU_ARB_S_B bank=%0d owner=%0d tx=%0d bid=%0d resp=%0d",bank,o,wt[o],s_axi_bid[o],s_axi_bresp[o]);
                    wt[o]++;wb[o]=0;saw[o]=0;wactive[o]=0;write_age[o]=0;bhold_age[o]=0;write_completed++;total_writes++;
                end
            end
            if(arhs) begin
                need(!mem_read,"overlapping memory AR");
                grant_owner=m_axi_araddr[36];grant_tx=int'((m_axi_araddr&64'h1fffffff)/4096);
                need(m_axi_araddr===address(grant_owner,grant_tx,0) && m_axi_arlen===8'(length(grant_tx)-1) &&
                    m_axi_arsize===3'd6 && m_axi_arburst===2'd1 && m_axi_arid===4'(grant_tx) &&
                    m_axi_arcache===4'(grant_tx) && m_axi_arprot===3'(grant_tx) && !m_axi_arlock,"AR preserved");
                // Both initiators re-assert before the other outstanding burst completes.
                need(last_read_owner!=grant_owner,"round-robin read starvation");
                last_read_owner=grant_owner;read_grants++;mr_owner=grant_owner;mr_tx=grant_tx;mr_beat=0;mem_read=1;
                $display("NTRPU_ARB_M_AR bank=%0d owner=%0d tx=%0d addr=%016h beats=%0d id=%0d cache=%0d prot=%0d",
                    bank,mr_owner,mr_tx,m_axi_araddr,int'(m_axi_arlen)+1,m_axi_arid,m_axi_arcache,m_axi_arprot);
            end
            if(rhs) begin
                $display("NTRPU_ARB_M_R bank=%0d owner=%0d tx=%0d beat=%0d rid=%0d resp=%0d last=%0d data=%0128h",
                    bank,mr_owner,mr_tx,mr_beat,m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata);
                mr_beat++;if(m_axi_rlast) mem_read=0;
            end
            if(awhs) begin
                need(!mem_aw,"overlapping memory AW");
                grant_owner=m_axi_awaddr[36];grant_tx=int'((m_axi_awaddr&64'h1fffffff)/4096);
                need(m_axi_awaddr===address(grant_owner,grant_tx,1) && m_axi_awlen===8'(length(grant_tx)-1) &&
                    m_axi_awsize===3'd6 && m_axi_awburst===2'd1 && m_axi_awid===4'(grant_tx) &&
                    m_axi_awcache===4'(grant_tx) && m_axi_awprot===3'(grant_tx) && !m_axi_awlock,"AW preserved");
                need(last_write_owner!=grant_owner,"round-robin write starvation");
                last_write_owner=grant_owner;write_grants++;mw_owner=grant_owner;mw_tx=grant_tx;mem_aw=1;mem_write=1;
                $display("NTRPU_ARB_M_AW bank=%0d owner=%0d tx=%0d addr=%016h beats=%0d id=%0d cache=%0d prot=%0d",
                    bank,mw_owner,mw_tx,m_axi_awaddr,int'(m_axi_awlen)+1,m_axi_awid,m_axi_awcache,m_axi_awprot);
            end
            if(m_axi_awvalid && !m_axi_wvalid && received==0) begin aw_wait_w++;local_wait_w++;end
            if(whs) begin
                need(received<64,"W queue overflow");
                $display("NTRPU_ARB_RAW_W bank=%0d last=%0d strb=%016h data=%0128h",bank,m_axi_wlast,m_axi_wstrb,m_axi_wdata);
                if(!mem_aw) begin w_before_aw++;local_w_before++;end
                wqueue[received]=m_axi_wdata;qmask[received]=m_axi_wstrb;qlast[received]=m_axi_wlast;
                received++;total_write_beats++;
            end
            if(mem_aw) while(checked<received) begin
                need(checked<length(mw_tx) && wqueue[checked]===payload(mw_owner,mw_tx,checked,1) &&
                    qmask[checked]===mask(mw_tx,checked) && qlast[checked]===(checked==length(mw_tx)-1),"W address/data association");
                $display("NTRPU_ARB_M_W bank=%0d owner=%0d tx=%0d beat=%0d last=%0d strb=%016h data=%0128h",
                    bank,mw_owner,mw_tx,checked,qlast[checked],qmask[checked],wqueue[checked]);
                checked++;
            end
            if(mem_read && (mem_write || received>0 || m_axi_awvalid)) begin overlap_cycles++;local_overlap++;end
            if(bhs) begin
                need(mem_aw && checked==length(mw_tx),"premature B");
                $display("NTRPU_ARB_M_B bank=%0d owner=%0d tx=%0d bid=%0d resp=%0d",bank,mw_owner,mw_tx,m_axi_bid,m_axi_bresp);
                mem_aw=0;mem_write=0;received=0;checked=0;bdelay=0;
            end
            // Abort with requests/data/responses in flight, then reset every peer.
            aborted=(abort_kind==1 && m_axi_arvalid && ar_held) ||
                (abort_kind==2 && (|s_axi_rvalid)) ||
                (abort_kind==3 && whs && !mem_aw) ||
                (abort_kind==4 && (|s_axi_bvalid));
            #1;if(rhs) m_axi_rvalid=0;if(bhs) m_axi_bvalid=0;
            if(aborted) break;
        end
        if(abort_kind!=0) begin
            need(aborted,"reset injection reached");reset_count++;
            $display("NTRPU_ARB_RESET bank=%0d kind=%0d reads=%0d writes=%0d",bank,abort_kind,read_completed,write_completed);
            reset_bus();
        end else begin
            need(!mem_read && !mem_aw && received==0 && !m_axi_rvalid && !m_axi_bvalid,"bank drained");
            need(read_grants==count*2 && write_grants==count*2 && local_wait_w>0 && local_overlap>0,"bank coverage");
            $display("NTRPU_ARB_DONE bank=%0d reads=%0d writes=%0d overlap=%0d aw_wait_w=%0d w_before_aw=%0d",
                bank,read_completed,write_completed,local_overlap,local_wait_w,local_w_before);
        end
    endtask
    initial begin
        clk=0;rst_n=0;clear_bus();
        for(int i=0;i<7;i++) stalls[i]=0;
        if($value$plusargs("SEED=%d",seed)) begin end
        if($value$plusargs("PIPELINED=%d",pipelined)) begin end
        rng=seed;$display("NTRPU_ARB_SEED seed=%0d",seed);
        if(pipelined) $display("NTRPU_ARB_PIPELINED value=1");
        reset_bus();run_bank(24);
        for(int kind=1;kind<=4;kind++) begin run_bank(4,kind);run_bank(4);end
        for(int i=0;i<7;i++) need(stalls[i]>0,"every output channel stalled");
        need(reset_count==4 && overlap_cycles>0 && aw_wait_w>0 && w_before_aw>0,"stress coverage");
        $display("NTRPU_ARB_PASS banks=%0d resets=%0d reads=%0d writes=%0d rbeats=%0d wbeats=%0d overlap=%0d aw_wait_w=%0d w_before_aw=%0d stalls=%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            bank,reset_count,total_reads,total_writes,total_read_beats,total_write_beats,overlap_cycles,aw_wait_w,w_before_aw,
            stalls[0],stalls[1],stalls[2],stalls[3],stalls[4],stalls[5],stalls[6]);
        $finish;
    end
endmodule
