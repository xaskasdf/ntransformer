`timescale 1ns/1ps
module tb_ntrpu_axi_capacity_guard;
    logic clk, rst_n, admit;
    logic idle;
    logic [63:0] s_axi_araddr, s_axi_awaddr;
    logic [7:0] s_axi_arlen, s_axi_awlen;
    logic [2:0] s_axi_arsize, s_axi_arprot, s_axi_awsize, s_axi_awprot;
    logic [1:0] s_axi_arburst, s_axi_awburst;
    logic [3:0] s_axi_arid, s_axi_arcache, s_axi_awid, s_axi_awcache;
    logic s_axi_arlock, s_axi_arvalid, s_axi_awlock, s_axi_awvalid;
    logic s_axi_arready, s_axi_awready;
    logic [511:0] s_axi_rdata;
    logic [1:0] s_axi_rresp;
    logic [3:0] s_axi_rid;
    logic s_axi_rlast, s_axi_rvalid;
    logic s_axi_rready;
    logic [511:0] s_axi_wdata;
    logic [63:0] s_axi_wstrb;
    logic s_axi_wlast, s_axi_wvalid;
    logic s_axi_wready;
    logic [3:0] s_axi_bid;
    logic [1:0] s_axi_bresp;
    logic s_axi_bvalid;
    logic s_axi_bready;
    logic [37-1:0] m_axi_araddr, m_axi_awaddr;
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
    ntrpu_axi_capacity_guard dut(.*);
    integer cycle=0,seed=1,ident=0,beats=1,size=6,burst=1;
    bit locked=0,allow=1,target_enable=1,complete=0;
    logic [63:0] address;
    integer sw=0,mw=0,sr=0,mr=0,saw=0,maw=0,sar=0,mar=0,sb=0,mb=0;
    integer total_sw=0,total_mw=0,total_sr=0,total_mr=0,closed_cycles=0;
    integer stalls[0:9]='{default:0};
    logic [9:0] valid_bus,ready_bus,was_stalled=0;
    logic [1023:0] sample_bus[0:9],held_bus[0:9];
    initial begin clk=0;forever #2.5 clk=~clk;end
    always @(posedge clk) cycle<=cycle+1;
    task automatic tick;@(posedge clk);#0.001;endtask
    task automatic need(input logic good,input string why);
        if(good!==1'b1) $fatal(1,"FAIL CAP case=%0d cycle=%0d: %s",ident,cycle,why);
    endtask
    function automatic logic [511:0] data_word(input logic[63:0] addr,input integer tag);
        logic [511:0] value;
        for(integer lane=0;lane<64;lane++) value[lane*8+:8]=8'((addr>>6)+64'(lane*13+tag*29+seed*11));
        return value;
    endfunction
    function automatic logic [63:0] write_mask(input integer beat);
        logic[63:0] value;
        case((ident+beat)%4)
            0:value='1;
            1:value=0;
            2:value=64'h5555555555555555;
            default:value=64'h8000000000000001;
        endcase
        return value;
    endfunction
    function automatic logic[1:0] read_resp(input integer beat);
        return ident==19 && beat==2 ? 2'b10:2'b00;
    endfunction

    // Independent target fixture: address-derived data, arbitrary strobes,
    // two independent channels and legal downstream SLVERR injection.
    // It is not a DRAM model and never supplies responses for rejected work.
    bit target_read=0,target_aw=0;
    logic[36:0] target_raddr;
    logic[3:0] target_rid,target_wid;
    integer target_rbeats=0,target_rindex=0,target_wbeats=0,target_wcount=0;
    always_comb begin
        m_axi_arready=rst_n && target_enable && !target_read && (cycle+seed)%5!=0;
        // Case2 requires two accepted W beats before accepting AW, proving
        // that the guard does not wait for AWREADY to expose buffered W.
        m_axi_awready=rst_n && target_enable && !target_aw && (cycle+seed)%4!=0 &&
                      (ident!=2 || target_wcount>=2);
        m_axi_wready=rst_n && target_enable && (cycle+seed)%3!=0 && !m_axi_bvalid;
    end
    always @(posedge clk) begin
        if(!rst_n) begin
            target_read<=0;target_aw<=0;target_wcount<=0;
            m_axi_rvalid<=0;m_axi_rdata<=0;m_axi_rresp<=0;m_axi_rid<=0;m_axi_rlast<=0;
            m_axi_bvalid<=0;m_axi_bresp<=0;m_axi_bid<=0;
        end else begin
            if(m_axi_arvalid && m_axi_arready) begin
                target_read<=1;target_raddr<=m_axi_araddr;target_rid<=m_axi_arid;
                target_rbeats<=int'(m_axi_arlen)+1;target_rindex<=0;
            end
            if(target_read && !m_axi_rvalid && (cycle+seed)%3!=0) begin
                m_axi_rvalid<=1;m_axi_rdata<=data_word(64'(target_raddr)+64'(target_rindex*64),ident);
                m_axi_rid<=target_rid;m_axi_rresp<=read_resp(target_rindex);
                m_axi_rlast<=target_rindex==target_rbeats-1;
            end
            if(m_axi_rvalid && m_axi_rready) begin
                if(m_axi_rlast) begin target_read<=0;m_axi_rvalid<=0;end
                else begin
                    target_rindex<=target_rindex+1;
                    m_axi_rdata<=data_word(64'(target_raddr)+64'((target_rindex+1)*64),ident);
                    m_axi_rresp<=read_resp(target_rindex+1);
                    m_axi_rlast<=target_rindex+1==target_rbeats-1;
                end
            end
            if(m_axi_awvalid && m_axi_awready) begin
                target_aw<=1;target_wbeats<=int'(m_axi_awlen)+1;target_wid<=m_axi_awid;
            end
            if(m_axi_wvalid && m_axi_wready) target_wcount<=target_wcount+1;
            if(target_aw && target_wcount==target_wbeats && !m_axi_bvalid) begin
                m_axi_bvalid<=1;m_axi_bid<=target_wid;m_axi_bresp<=ident==19 ? 2'b10:2'b00;
            end
            if(m_axi_bvalid && m_axi_bready) begin
                target_aw<=0;target_wcount<=0;m_axi_bvalid<=0;
            end
        end
    end

    // Stability is observed on all ten boundary channels, including VALID.
    always_comb begin
        valid_bus={s_axi_bvalid,s_axi_rvalid,m_axi_bvalid,m_axi_rvalid,m_axi_wvalid,m_axi_awvalid,m_axi_arvalid,s_axi_wvalid,s_axi_awvalid,s_axi_arvalid};
        ready_bus={s_axi_bready,s_axi_rready,m_axi_bready,m_axi_rready,m_axi_wready,m_axi_awready,m_axi_arready,s_axi_wready,s_axi_awready,s_axi_arready};
        sample_bus[0]={s_axi_araddr,s_axi_arlen,s_axi_arsize,s_axi_arburst,s_axi_arid,s_axi_arcache,s_axi_arprot,s_axi_arlock};
        sample_bus[1]={s_axi_awaddr,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awid,s_axi_awcache,s_axi_awprot,s_axi_awlock};
        sample_bus[2]={s_axi_wdata,s_axi_wstrb,s_axi_wlast};
        sample_bus[3]={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
        sample_bus[4]={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
        sample_bus[5]={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
        sample_bus[6]={m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast};
        sample_bus[7]={m_axi_bresp,m_axi_bid};
        sample_bus[8]={s_axi_rdata,s_axi_rresp,s_axi_rid,s_axi_rlast};
        sample_bus[9]={s_axi_bresp,s_axi_bid};
    end
    always @(posedge clk) begin
        if(!rst_n) was_stalled=0;
        else begin
            for(integer ch=0;ch<10;ch++) begin
                if(was_stalled[ch]) need(valid_bus[ch] && sample_bus[ch]===held_bus[ch],$sformatf("CAP_STABILITY channel=%0d",ch));
                was_stalled[ch]=valid_bus[ch] && !ready_bus[ch];
                if(was_stalled[ch]) begin
                    held_bus[ch]=sample_bus[ch];stalls[ch]++;
                    $display("NTRPU_CAP_HOLD case=%0d channel=%0d cycle=%0d sample=%0256h",ident,ch,cycle,sample_bus[ch]);
                end
            end
            if(!allow) need(!m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid,"CAP_REJECT_EXPOSED");
            if(!admit) begin
                closed_cycles++;
                need(!s_axi_arready && !s_axi_awready,"CAP_CLOSED_ADMISSION");
            end
            if(ident>0 && ((sar>0 && sr<beats) || (saw>sb))) need(!idle,"CAP_EARLY_IDLE");
            if(s_axi_arvalid && s_axi_arready) begin
                sar++;
                need(sar==1,"CAP_DUPLICATE_ADDRESS");
                $display("NTRPU_CAP_ADDR case=%0d side=0 write=0 cycle=%0d ps=%0d addr=%016h beats=%0d size=%0d burst=%0d lock=%0d aid=%0d cache=%0d prot=%0d",ident,cycle,longint'($realtime*1000),64'(s_axi_araddr),int'(s_axi_arlen)+1,s_axi_arsize,s_axi_arburst,s_axi_arlock,s_axi_arid,s_axi_arcache,s_axi_arprot);
            end
            if(s_axi_awvalid && s_axi_awready) begin
                saw++;
                need(saw==1,"CAP_DUPLICATE_ADDRESS");
                $display("NTRPU_CAP_ADDR case=%0d side=0 write=1 cycle=%0d ps=%0d addr=%016h beats=%0d size=%0d burst=%0d lock=%0d aid=%0d cache=%0d prot=%0d",ident,cycle,longint'($realtime*1000),64'(s_axi_awaddr),int'(s_axi_awlen)+1,s_axi_awsize,s_axi_awburst,s_axi_awlock,s_axi_awid,s_axi_awcache,s_axi_awprot);
            end
            if(s_axi_wvalid && s_axi_wready) begin
                need(s_axi_wdata===data_word(address+64'(64*sw),ident+32) && s_axi_wstrb===write_mask(sw) && s_axi_wlast===(sw==beats-1),"CAP_WRITE_PAYLOAD");
                $display("NTRPU_CAP_W case=%0d side=0 cycle=%0d ps=%0d beat=%0d last=%0d data=%0128h,strb=%016h",ident,cycle,longint'($realtime*1000),sw,s_axi_wlast,s_axi_wdata,s_axi_wstrb);
                sw++;total_sw++;
            end
            if(s_axi_rvalid && s_axi_rready) begin
                need(s_axi_rdata===(allow ? data_word(address+64'(64*sr),ident):512'b0) && s_axi_rid==4'(ident*3) && s_axi_rresp===(allow ? read_resp(sr):2'b11) && s_axi_rlast===(sr==beats-1),"CAP_READ_PAYLOAD");
                $display("NTRPU_CAP_R case=%0d side=0 cycle=%0d ps=%0d beat=%0d last=%0d data=%0128h aid=%0d resp=%0d",ident,cycle,longint'($realtime*1000),sr,s_axi_rlast,s_axi_rdata,s_axi_rid,s_axi_rresp);
                sr++;total_sr++;
            end
            if(s_axi_bvalid && s_axi_bready) begin
                need(s_axi_bid==4'(ident*5) && s_axi_bresp==(allow ? (ident==19 ? 2:0):3),"CAP_B_RESPONSE");
                sb++;
                $display("NTRPU_CAP_B case=%0d side=0 cycle=%0d ps=%0d aid=%0d resp=%0d",ident,cycle,longint'($realtime*1000),s_axi_bid,s_axi_bresp);
            end
            if(m_axi_arvalid && m_axi_arready) begin
                mar++;
                need(mar==1,"CAP_DUPLICATE_ADDRESS");
                $display("NTRPU_CAP_ADDR case=%0d side=1 write=0 cycle=%0d ps=%0d addr=%016h beats=%0d size=%0d burst=%0d lock=%0d aid=%0d cache=%0d prot=%0d",ident,cycle,longint'($realtime*1000),64'(m_axi_araddr),int'(m_axi_arlen)+1,m_axi_arsize,m_axi_arburst,m_axi_arlock,m_axi_arid,m_axi_arcache,m_axi_arprot);
            end
            if(m_axi_awvalid && m_axi_awready) begin
                maw++;
                need(maw==1,"CAP_DUPLICATE_ADDRESS");
                $display("NTRPU_CAP_ADDR case=%0d side=1 write=1 cycle=%0d ps=%0d addr=%016h beats=%0d size=%0d burst=%0d lock=%0d aid=%0d cache=%0d prot=%0d",ident,cycle,longint'($realtime*1000),64'(m_axi_awaddr),int'(m_axi_awlen)+1,m_axi_awsize,m_axi_awburst,m_axi_awlock,m_axi_awid,m_axi_awcache,m_axi_awprot);
            end
            if(m_axi_wvalid && m_axi_wready) begin
                need(m_axi_wdata===data_word(address+64'(64*mw),ident+32) && m_axi_wstrb===write_mask(mw) && m_axi_wlast===(mw==beats-1),"CAP_WRITE_PAYLOAD");
                $display("NTRPU_CAP_W case=%0d side=1 cycle=%0d ps=%0d beat=%0d last=%0d data=%0128h,strb=%016h",ident,cycle,longint'($realtime*1000),mw,m_axi_wlast,m_axi_wdata,m_axi_wstrb);
                mw++;total_mw++;
            end
            if(m_axi_rvalid && m_axi_rready) begin
                need(m_axi_rdata===data_word(address+64'(64*mr),ident) && m_axi_rid==4'(ident*3) && m_axi_rresp===read_resp(mr) && m_axi_rlast===(mr==beats-1),"CAP_READ_PAYLOAD");
                $display("NTRPU_CAP_R case=%0d side=1 cycle=%0d ps=%0d beat=%0d last=%0d data=%0128h aid=%0d resp=%0d",ident,cycle,longint'($realtime*1000),mr,m_axi_rlast,m_axi_rdata,m_axi_rid,m_axi_rresp);
                mr++;total_mr++;
            end
            if(m_axi_bvalid && m_axi_bready) begin
                need(m_axi_bid==4'(ident*5) && m_axi_bresp==(allow ? (ident==19 ? 2:0):3),"CAP_B_RESPONSE");
                mb++;
                $display("NTRPU_CAP_B case=%0d side=1 cycle=%0d ps=%0d aid=%0d resp=%0d",ident,cycle,longint'($realtime*1000),m_axi_bid,m_axi_bresp);
            end
        end
    end
    task automatic send_read();
        repeat(ident%2) tick();
        @(negedge clk);s_axi_araddr=address;s_axi_arlen=8'(beats-1);
        s_axi_arsize=3'(size);s_axi_arburst=2'(burst);s_axi_arlock=locked;
        s_axi_arid=4'(ident*3);s_axi_arcache=4'(ident);s_axi_arprot=3'(ident);s_axi_arvalid=1;
        do begin @(posedge clk);end while(!s_axi_arready);
        @(negedge clk);s_axi_arvalid=0;
        for(integer beat=0;beat<beats;beat++) begin
            while(!s_axi_rvalid) tick();
            repeat(beat==0 ? 7:(beat+ident)%3) tick();
            @(negedge clk);s_axi_rready=1;
            do begin @(posedge clk);end while(!s_axi_rvalid);
            @(negedge clk);s_axi_rready=0;
        end
    endtask
    task automatic send_aw();
        repeat(ident%3==1 ? 6:0) tick();
        @(negedge clk);s_axi_awaddr=address;s_axi_awlen=8'(beats-1);
        s_axi_awsize=3'(size);s_axi_awburst=2'(burst);s_axi_awlock=locked;
        s_axi_awid=4'(ident*5);s_axi_awcache=4'(ident+1);s_axi_awprot=3'(ident+1);s_axi_awvalid=1;
        do begin @(posedge clk);end while(!s_axi_awready);
        @(negedge clk);s_axi_awvalid=0;
    endtask
    task automatic send_w();
        repeat(ident%3==0 ? 4:0) tick();
        @(negedge clk);
        for(integer beat=0;beat<beats;beat++) begin
            s_axi_wdata=data_word(address+64'(beat*64),ident+32);s_axi_wstrb=write_mask(beat);
            s_axi_wlast=beat==beats-1;s_axi_wvalid=1;
            do begin @(posedge clk);end while(!s_axi_wready);
            @(negedge clk);
        end
        s_axi_wvalid=0;
    endtask
    task automatic send_write();
        fork send_aw();send_w();join
        while(!s_axi_bvalid) tick();
        repeat(7) tick();
        @(negedge clk);s_axi_bready=1;
        @(posedge clk);need(s_axi_bvalid,"CAP_B_LOST");
        @(negedge clk);s_axi_bready=0;
    endtask
    task automatic admission_schedule();
        if(ident==1) begin
            repeat(15) tick();
            @(negedge clk);admit=1;
            $display("NTRPU_CAP_ADMIT case=%0d cycle=%0d ps=%0d value=1",ident,cycle,longint'($realtime*1000));
        end else if(ident==20) begin
            while(!(sar==1 && saw==1)) tick();
            repeat(3) tick();
            @(negedge clk);admit=0;
            need(!idle && m_axi_arvalid && m_axi_awvalid,"CAP_PENDING_BEFORE_CLOSE");
            $display("NTRPU_CAP_ADMIT case=%0d cycle=%0d ps=%0d value=0",ident,cycle,longint'($realtime*1000));
            repeat(8) tick();
            need(!idle && m_axi_arvalid && m_axi_awvalid,"CAP_PENDING_AFTER_CLOSE");
            @(negedge clk);target_enable=1;
            $display("NTRPU_CAP_TARGET_RELEASE case=%0d cycle=%0d ps=%0d",ident,cycle,longint'($realtime*1000));
        end
    endtask
    initial begin
        rst_n=0;admit=0;
        s_axi_arvalid=0;s_axi_awvalid=0;s_axi_wvalid=0;s_axi_rready=0;s_axi_bready=0;
        s_axi_araddr=0;s_axi_awaddr=0;s_axi_arlen=0;s_axi_awlen=0;
        s_axi_arsize=6;s_axi_awsize=6;s_axi_arburst=1;s_axi_awburst=1;
        s_axi_arlock=0;s_axi_awlock=0;s_axi_arid=0;s_axi_awid=0;
        s_axi_arcache=0;s_axi_awcache=0;s_axi_arprot=0;s_axi_awprot=0;
        s_axi_wdata=0;s_axi_wstrb=0;s_axi_wlast=0;
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_CAP_CONFIG seed=%0d period_ps=5000 memory_addr_width=37",seed);
`ifdef NTRPU_NETLIST
        // Keep the external reset asserted beyond the UNISIM global reset.
        repeat(64) tick();
`else
        repeat(8) tick();
`endif
        @(negedge clk);rst_n=1;
        for(integer test=1;test<=20;test++) begin
            @(negedge clk);ident=test;beats=4;size=6;burst=1;locked=0;allow=1;
            case(test)
                1:begin address=0;beats=1;end
                2:address=64'h100000000;
                3:begin address=64'h1fffffffc0;beats=1;end
                4:begin address=64'h1ffffff000;beats=64;end
                5:begin address=64'h2000000000;beats=1;allow=0;end
                6:begin address=64'h1000000010000;allow=0;end
                7:begin address=64'hffffffffffffffc0;beats=2;allow=0;end
                8:begin address=64'hfc0;beats=2;allow=0;end
                9:begin address=64'h41;beats=1;allow=0;end
                10:begin address=64'h20000;size=5;allow=0;end
                11:begin address=64'h30000;burst=0;allow=0;end
                12:begin address=64'h40000;burst=2;allow=0;end
                13:begin address=64'h50000;beats=1;locked=1;allow=0;end
                14:begin address=64'h60000;beats=256;allow=0;end
                15:begin address=64'h1ffffff000;beats=1;end
                16:begin address=64'h100001000;beats=64;end
                17:begin address=64'h100000fc0;beats=1;end
                18:begin address=64'h1fffffff80;beats=2;end
                19:address=64'h70000;
                20:begin address=64'h80000;beats=8;target_enable=0;end
            endcase
            need(idle,"CAP_START_NOT_IDLE");
            sw=0;mw=0;sr=0;mr=0;saw=0;maw=0;sar=0;mar=0;sb=0;mb=0;
            $display("NTRPU_CAP_BEGIN case=%0d cycle=%0d ps=%0d addr=%016h beats=%0d size=%0d burst=%0d lock=%0d allow=%0d admit=%0d",ident,cycle,longint'($realtime*1000),address,beats,size,burst,locked,allow,admit);
            fork send_read();send_write();admission_schedule();join
            repeat(3) tick();
            need(idle && saw==1 && sar==1 && sw==beats && sr==beats && sb==1,"CAP_SOURCE_DRAIN");
            need(maw==int'(allow) && mar==int'(allow) && mb==int'(allow) && mw==(allow ? beats:0) && mr==(allow ? beats:0),"CAP_TARGET_COUNTS");
            need(!target_read && !target_aw && !m_axi_rvalid && !m_axi_bvalid,"CAP_TARGET_DRAIN");
            $display("NTRPU_CAP_END case=%0d cycle=%0d ps=%0d idle=%0d admit=%0d sw=%0d mw=%0d sr=%0d mr=%0d",ident,cycle,longint'($realtime*1000),idle,admit,sw,mw,sr,mr);
        end
        need(closed_cycles>20,"CAP_ADMISSION_COVERAGE");
        for(integer ch=0;ch<10;ch++) begin
            $display("NTRPU_CAP_STALL channel=%0d cycles=%0d",ch,stalls[ch]);
            // A registered B sink can accept immediately; all other boundaries stall.
            if(ch!=7) need(stalls[ch]>0,"CAP_STALL_COVERAGE");
        end
        need(total_sw==429 && total_sr==429 && total_mw==150 && total_mr==150,"CAP_TOTAL_BEATS");
        $display("NTRPU_CAP_PASS cases=20 allowed=10 rejected=10 sw=%0d mw=%0d sr=%0d mr=%0d closed_cycles=%0d",total_sw,total_mw,total_sr,total_mr,closed_cycles);
        complete=1;$finish;
    end
    initial begin #1000000;$fatal(1,"FAIL CAP_GLOBAL_WATCHDOG");end
    final if(!complete) $error("NTRPU_CAP_INCOMPLETE");
endmodule
