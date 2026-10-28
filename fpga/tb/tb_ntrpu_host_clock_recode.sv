// Protocol peers for the generated AMD AXI4 and AXI4-Lite clock converters.
// Testbench cross-domain counters are scoreboards, not synthesizable CDC logic.
`timescale 1ns/1ps
module ntrpu_host_clock_recode_bank #(parameter bit LITE=0)(output bit finished);
    localparam int DW=LITE ? 32:512, AW=LITE ? 32:64, BYTES=DW/8, TX=12;
    logic s_axi_aclk=0,m_axi_aclk=0,s_axi_aresetn=0,m_axi_aresetn=0;
    logic [AW-1:0] s_axi_awaddr,s_axi_araddr,m_axi_awaddr,m_axi_araddr;
    logic [DW-1:0] s_axi_wdata,s_axi_rdata,m_axi_wdata,m_axi_rdata;
    logic [BYTES-1:0] s_axi_wstrb,m_axi_wstrb;
    logic [3:0] s_axi_awid,s_axi_arid,s_axi_bid,s_axi_rid,m_axi_awid,m_axi_arid,m_axi_bid,m_axi_rid;
    logic [7:0] s_axi_awlen,s_axi_arlen,m_axi_awlen,m_axi_arlen;
    logic [2:0] s_axi_awsize,s_axi_arsize,s_axi_awprot,s_axi_arprot,m_axi_awsize,m_axi_arsize,m_axi_awprot,m_axi_arprot;
    logic [1:0] s_axi_awburst,s_axi_arburst,m_axi_awburst,m_axi_arburst;
    logic [1:0] s_axi_bresp,s_axi_rresp,m_axi_bresp,m_axi_rresp;
    logic [3:0] s_axi_awcache,s_axi_arcache,s_axi_awqos,s_axi_arqos,s_axi_awregion,s_axi_arregion;
    logic [3:0] m_axi_awcache,m_axi_arcache,m_axi_awqos,m_axi_arqos,m_axi_awregion,m_axi_arregion;
    logic s_axi_awlock,s_axi_arlock,m_axi_awlock,m_axi_arlock;
    logic s_axi_awvalid,s_axi_awready,s_axi_wvalid,s_axi_wready,s_axi_wlast,s_axi_bvalid,s_axi_bready;
    logic s_axi_arvalid,s_axi_arready,s_axi_rvalid,s_axi_rready,s_axi_rlast;
    logic m_axi_awvalid,m_axi_awready,m_axi_wvalid,m_axi_wready,m_axi_wlast,m_axi_bvalid,m_axi_bready;
    logic m_axi_arvalid,m_axi_arready,m_axi_rvalid,m_axi_rready,m_axi_rlast;
    if(LITE) begin: lite
        logic [1:0] recode_rresp;
        ntrpu_axil_cdc_recode dut(.s_axi_rresp(recode_rresp),.*);
        // Test-only corruption after the production decoder, no RTL fault flag.
        assign s_axi_rresp=recode_rresp ^ ((inject_rresp && ident==3) ? 2'b01:2'b00);
        assign m_axi_awlen=0,m_axi_arlen=0,m_axi_wlast=1,s_axi_rlast=1;
        assign m_axi_awid=0,m_axi_arid=0,s_axi_bid=0,s_axi_rid=0;
    end else begin: full
        ntrpu_host_axi_cdc dut(.*);
    end
    int s_half_ps=2000,m_half_ps=2500,phase_ps=700,profile=0;
    bit run_s=1,go=0,inject_stale=0,inject_rresp=0,clocks_ready=0;
    int ident=0,abort_mode=0,sc=0,mc=0,saw=0,sar=0,sw_tx=0,sw_beat=0,sb=0,sr_tx=0,sr_beat=0;
    int maw=0,mar=0,mw_tx=0,mw_beat=0,mb=0,mr_tx=0,mr_beat=0;
    int sw_total=0,sr_total=0,mw_total=0,mr_total=0;
    int offer_aw=0,offer_ar=0,offer_w=0,offer_b=0,offer_r=0;
    longint s_ticks=0,m_ticks=0;
    int s_stalls[0:4],m_stalls[0:4];
    bit held_saw=0,held_sar=0,held_sw=0,held_mb=0,held_mr=0;
    bit held_maw=0,held_mar=0,held_mw=0,held_sb=0,held_sr=0;
    logic [AW+36:0] prev_saw,prev_sar,prev_maw,prev_mar;
    logic [DW+BYTES:0] prev_sw,prev_mw;
    logic [DW+6:0] prev_mr,prev_sr;
    logic [5:0] prev_mb,prev_sb;
    initial begin wait(clocks_ready);forever begin #(s_half_ps/1000.0);if(run_s) s_axi_aclk=~s_axi_aclk;end end
    initial begin wait(clocks_ready);#(phase_ps/1000.0);forever #(m_half_ps/1000.0) m_axi_aclk=~m_axi_aclk;end
    function automatic void need(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL HOST_CLOCK lite=%0d case=%0d sc=%0d mc=%0d: %s",LITE,ident,sc,mc,why);
    endfunction
    function automatic int length(input int tx);
        if(LITE) return 1;
        case(tx%6) 0:return 1;1:return 64;2:return 2;3:return 16;4:return 3;default:return 4;endcase
    endfunction
    function automatic logic [AW-1:0] addr(input int tx,input bit write_op);
        return AW'((LITE ? 64'h80000000:64'h8000000000)+64'(ident)*64'h100000+
                   64'(write_op)*64'h40000+64'(tx)*4096+(tx%6==0 ? 4096-BYTES:0));
    endfunction
    function automatic logic [3:0] tag(input int tx);return LITE ? 0:4'(tx+ident*3);endfunction
    function automatic logic [DW-1:0] payload(input int epoch,tx,beat,input bit write_op);
        logic [DW-1:0] v;
        for(int lane=0;lane<BYTES;lane++) v[lane*8+:8]=8'(epoch*53+tx*17+beat*7+lane*11+int'(write_op)*31);
        return v;
    endfunction
    function automatic logic [BYTES-1:0] mask(input int tx,beat);
        logic [BYTES-1:0] v;
        for(int lane=0;lane<BYTES;lane++) v[lane]=(tx+beat)%4==0 ? 1:(tx+beat)%4==1 ? lane%2==0:(tx+beat)%4==2 ? lane%2==1:0;
        return v;
    endfunction
    function automatic logic [1:0] response(input int tx,beat);
        return (tx+beat)%3==0 ? 0:(tx+beat)%3==1 ? 2:3;
    endfunction
    function automatic longint ps;return longint'($realtime*1000.0);endfunction
    task automatic address_trace(input string channel,input int tx,input logic [AW-1:0] a,input logic [32:0] meta);
        $display("NTRPU_CDC_ADDR lite=%0d case=%0d ch=%s tx=%0d time_ps=%0d addr=%016h meta=%09h",LITE,ident,channel,tx,ps(),a,meta);
    endtask
    task automatic data_trace(input string channel,input int tx,beat,input logic [DW-1:0] data,input logic [BYTES-1:0] strobe,input bit last,input logic [3:0] id,input logic [1:0] resp);
        $display("NTRPU_CDC_DATA lite=%0d case=%0d ch=%s tx=%0d beat=%0d time_ps=%0d data=%h strb=%h last=%0d tag=%0d resp=%0d",LITE,ident,channel,tx,beat,ps(),data,strobe,last,id,resp);
    endtask
    // Source drives at falling edges; stalled VALID and payload remain unchanged.
    always @(negedge s_axi_aclk) begin
        s_axi_awvalid=s_axi_aresetn && go && saw<TX && (s_axi_awvalid || sc%7>=2);
        s_axi_arvalid=s_axi_aresetn && go && sar<TX && (s_axi_arvalid || sc%5!=0);
        s_axi_wvalid=s_axi_aresetn && go && sw_tx<TX && (s_axi_wvalid || sc%6!=1);
        s_axi_awaddr=addr(saw,1);s_axi_araddr=addr(sar,0);
        s_axi_awlen=8'(length(saw)-1);s_axi_arlen=8'(length(sar)-1);
        s_axi_awsize=$clog2(BYTES);s_axi_arsize=$clog2(BYTES);
        s_axi_awburst=1;s_axi_arburst=1;s_axi_awid=tag(saw);s_axi_arid=tag(sar);
        s_axi_awcache=4'(saw);s_axi_arcache=4'(sar);s_axi_awprot=3'(saw);s_axi_arprot=3'(sar);
        s_axi_awqos=4'(saw^5);s_axi_arqos=4'(sar^5);s_axi_awregion=4'(saw^10);s_axi_arregion=4'(sar^10);
        s_axi_awlock=0;s_axi_arlock=0;
        s_axi_wdata=payload(ident,sw_tx,sw_beat,1);s_axi_wstrb=mask(sw_tx,sw_beat);s_axi_wlast=sw_beat==length(sw_tx)-1;
        s_axi_bready=s_axi_aresetn && go && abort_mode!=4 && sc%9>=3;
        s_axi_rready=s_axi_aresetn && go && abort_mode!=2 && sc%11>=4;
    end
    always @(negedge m_axi_aclk) begin
        m_axi_arready=m_axi_aresetn && go && abort_mode!=1 && mc%7>=2;
        // Deliberately permit W before accepting AW, including long AW stalls.
        m_axi_awready=m_axi_aresetn && go && mc%9>=4 && (m_axi_wvalid || mw_total>0);
        m_axi_wready=m_axi_aresetn && go && !(abort_mode==3 && mw_total>=1) && mc%8>=3;
        m_axi_bvalid=m_axi_aresetn && go && mb<maw && mb<mw_tx && (m_axi_bvalid || mc%5!=0);
        m_axi_bid=tag(mb);m_axi_bresp=response(mb,0);
        m_axi_rvalid=m_axi_aresetn && go && mr_tx<mar && (m_axi_rvalid || mc%6!=1);
        m_axi_rid=tag(mr_tx);m_axi_rresp=response(mr_tx,mr_beat);m_axi_rlast=mr_beat==length(mr_tx)-1;
        m_axi_rdata=payload(ident-((inject_stale && !LITE && ident==3) ? 1:0),mr_tx,mr_beat,0);
    end
    always @(posedge s_axi_aclk) begin
        s_ticks++;
        if(!s_axi_aresetn) begin
            sc=0;saw=0;sar=0;sw_tx=0;sw_beat=0;sb=0;sr_tx=0;sr_beat=0;sw_total=0;sr_total=0;
            held_saw=0;held_sar=0;held_sw=0;held_sb=0;held_sr=0;
            offer_aw=0;offer_ar=0;offer_w=0;
            for(int i=0;i<5;i++) s_stalls[i]=0;
        end else if(go) begin
            sc++;
            need(sc<50000,"bounded source progress");
            if(held_saw) need(s_axi_awvalid && prev_saw==={s_axi_awaddr,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awid,s_axi_awcache,s_axi_awprot,s_axi_awlock,s_axi_awqos,s_axi_awregion},"source AW stability");
            if(held_sar) need(s_axi_arvalid && prev_sar==={s_axi_araddr,s_axi_arlen,s_axi_arsize,s_axi_arburst,s_axi_arid,s_axi_arcache,s_axi_arprot,s_axi_arlock,s_axi_arqos,s_axi_arregion},"source AR stability");
            if(held_sw) need(s_axi_wvalid && prev_sw==={s_axi_wdata,s_axi_wstrb,s_axi_wlast},"source W stability");
            if(held_sb) need(s_axi_bvalid && prev_sb==={s_axi_bid,s_axi_bresp},"source B stability");
            if(held_sr) need(s_axi_rvalid && prev_sr==={s_axi_rdata,s_axi_rid,s_axi_rresp,s_axi_rlast},"source R stability");
            held_saw=s_axi_awvalid&&!s_axi_awready;prev_saw={s_axi_awaddr,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awid,s_axi_awcache,s_axi_awprot,s_axi_awlock,s_axi_awqos,s_axi_awregion};
            held_sar=s_axi_arvalid&&!s_axi_arready;prev_sar={s_axi_araddr,s_axi_arlen,s_axi_arsize,s_axi_arburst,s_axi_arid,s_axi_arcache,s_axi_arprot,s_axi_arlock,s_axi_arqos,s_axi_arregion};
            held_sw=s_axi_wvalid&&!s_axi_wready;prev_sw={s_axi_wdata,s_axi_wstrb,s_axi_wlast};
            held_sb=s_axi_bvalid&&!s_axi_bready;prev_sb={s_axi_bid,s_axi_bresp};
            held_sr=s_axi_rvalid&&!s_axi_rready;prev_sr={s_axi_rdata,s_axi_rid,s_axi_rresp,s_axi_rlast};
            s_stalls[0]+=int'(held_saw);s_stalls[1]+=int'(held_sw);s_stalls[2]+=int'(held_sb);s_stalls[3]+=int'(held_sar);s_stalls[4]+=int'(held_sr);
            if(s_axi_awvalid && offer_aw==saw) begin address_trace("SAWO",saw,s_axi_awaddr,LITE ? {30'b0,s_axi_awprot}:{s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awid,s_axi_awcache,s_axi_awprot,s_axi_awlock,s_axi_awqos,s_axi_awregion});offer_aw++;end
            if(s_axi_arvalid && offer_ar==sar) begin address_trace("SARO",sar,s_axi_araddr,LITE ? {30'b0,s_axi_arprot}:{s_axi_arlen,s_axi_arsize,s_axi_arburst,s_axi_arid,s_axi_arcache,s_axi_arprot,s_axi_arlock,s_axi_arqos,s_axi_arregion});offer_ar++;end
            if(s_axi_wvalid && offer_w==sw_total) begin data_trace("SWO",sw_tx,sw_beat,s_axi_wdata,s_axi_wstrb,s_axi_wlast,0,0);offer_w++;end
            if(s_axi_awvalid&&s_axi_awready) begin address_trace("SAW",saw,s_axi_awaddr,LITE ? {30'b0,s_axi_awprot}:{s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awid,s_axi_awcache,s_axi_awprot,s_axi_awlock,s_axi_awqos,s_axi_awregion});saw++;end
            if(s_axi_arvalid&&s_axi_arready) begin address_trace("SAR",sar,s_axi_araddr,LITE ? {30'b0,s_axi_arprot}:{s_axi_arlen,s_axi_arsize,s_axi_arburst,s_axi_arid,s_axi_arcache,s_axi_arprot,s_axi_arlock,s_axi_arqos,s_axi_arregion});sar++;end
            if(s_axi_wvalid&&s_axi_wready) begin
                data_trace("SW",sw_tx,sw_beat,s_axi_wdata,s_axi_wstrb,s_axi_wlast,0,0);sw_total++;
                if(sw_beat==length(sw_tx)-1) begin sw_tx++;sw_beat=0;end else sw_beat++;
            end
            if(s_axi_bvalid&&s_axi_bready) begin
                need(sb<offer_b && s_axi_bid===tag(sb) && s_axi_bresp===response(sb,0),"B after memory offer, ID/response");
                data_trace("SB",sb,0,0,0,1,s_axi_bid,s_axi_bresp);sb++;
            end
            if(s_axi_rvalid&&s_axi_rready) begin
                need(sr_total<offer_r && s_axi_rdata===payload(ident,sr_tx,sr_beat,0) && s_axi_rid===tag(sr_tx) &&
                     s_axi_rresp===response(sr_tx,sr_beat) && s_axi_rlast===(sr_beat==length(sr_tx)-1),"R data/epoch/order/ID/response/last");
                data_trace("SR",sr_tx,sr_beat,s_axi_rdata,0,s_axi_rlast,s_axi_rid,s_axi_rresp);sr_total++;
                if(sr_beat==length(sr_tx)-1) begin sr_tx++;sr_beat=0;end else sr_beat++;
            end
        end
    end
    always @(posedge m_axi_aclk) begin
        m_ticks++;
        if(!m_axi_aresetn) begin
            mc=0;maw=0;mar=0;mw_tx=0;mw_beat=0;mb=0;mr_tx=0;mr_beat=0;mw_total=0;mr_total=0;
            held_maw=0;held_mar=0;held_mw=0;held_mb=0;held_mr=0;
            offer_b=0;offer_r=0;
            for(int i=0;i<5;i++) m_stalls[i]=0;
        end else if(go) begin
            mc++;
            need(mc<50000,"bounded target progress");
            if(held_maw) need(m_axi_awvalid && prev_maw==={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock,m_axi_awqos,m_axi_awregion},"memory AW stability");
            if(held_mar) need(m_axi_arvalid && prev_mar==={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock,m_axi_arqos,m_axi_arregion},"memory AR stability");
            if(held_mw) need(m_axi_wvalid && prev_mw==={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"memory W stability");
            if(held_mb) need(m_axi_bvalid && prev_mb==={m_axi_bid,m_axi_bresp},"memory B stability");
            if(held_mr) need(m_axi_rvalid && prev_mr==={m_axi_rdata,m_axi_rid,m_axi_rresp,m_axi_rlast},"memory R stability");
            held_maw=m_axi_awvalid&&!m_axi_awready;prev_maw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock,m_axi_awqos,m_axi_awregion};
            held_mar=m_axi_arvalid&&!m_axi_arready;prev_mar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock,m_axi_arqos,m_axi_arregion};
            held_mw=m_axi_wvalid&&!m_axi_wready;prev_mw={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            held_mb=m_axi_bvalid&&!m_axi_bready;prev_mb={m_axi_bid,m_axi_bresp};
            held_mr=m_axi_rvalid&&!m_axi_rready;prev_mr={m_axi_rdata,m_axi_rid,m_axi_rresp,m_axi_rlast};
            m_stalls[0]+=int'(held_maw);m_stalls[1]+=int'(held_mw);m_stalls[2]+=int'(held_mb);m_stalls[3]+=int'(held_mar);m_stalls[4]+=int'(held_mr);
            if(m_axi_bvalid && offer_b==mb) begin data_trace("MBO",mb,0,0,0,1,m_axi_bid,m_axi_bresp);offer_b++;end
            if(m_axi_rvalid && offer_r==mr_total) begin data_trace("MRO",mr_tx,mr_beat,m_axi_rdata,0,m_axi_rlast,m_axi_rid,m_axi_rresp);offer_r++;end
            if(m_axi_awvalid&&m_axi_awready) begin
                need(maw<offer_aw && m_axi_awaddr===addr(maw,1) && m_axi_awprot===3'(maw),"AW address/protection/order");
                if(!LITE) need(m_axi_awlen==length(maw)-1 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid===tag(maw) && m_axi_awcache===4'(maw) && !m_axi_awlock && m_axi_awqos===4'(maw^5) && m_axi_awregion===4'(maw^10),"AW metadata");
                address_trace("MAW",maw,m_axi_awaddr,LITE ? {30'b0,m_axi_awprot}:{m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock,m_axi_awqos,m_axi_awregion});maw++;
            end
            if(m_axi_arvalid&&m_axi_arready) begin
                need(mar<offer_ar && m_axi_araddr===addr(mar,0) && m_axi_arprot===3'(mar),"AR address/protection/order");
                if(!LITE) need(m_axi_arlen==length(mar)-1 && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid===tag(mar) && m_axi_arcache===4'(mar) && !m_axi_arlock && m_axi_arqos===4'(mar^5) && m_axi_arregion===4'(mar^10),"AR metadata");
                address_trace("MAR",mar,m_axi_araddr,LITE ? {30'b0,m_axi_arprot}:{m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock,m_axi_arqos,m_axi_arregion});mar++;
            end
            if(m_axi_wvalid&&m_axi_wready) begin
                need(mw_total<offer_w && m_axi_wdata===payload(ident,mw_tx,mw_beat,1) && m_axi_wstrb===mask(mw_tx,mw_beat) && m_axi_wlast===(mw_beat==length(mw_tx)-1),"W payload/mask/order/last");
                data_trace("MW",mw_tx,mw_beat,m_axi_wdata,m_axi_wstrb,m_axi_wlast,0,0);mw_total++;
                if(mw_beat==length(mw_tx)-1) begin mw_tx++;mw_beat=0;end else mw_beat++;
            end
            if(m_axi_bvalid&&m_axi_bready) begin data_trace("MB",mb,0,0,0,1,m_axi_bid,m_axi_bresp);mb++;end
            if(m_axi_rvalid&&m_axi_rready) begin
                data_trace("MR",mr_tx,mr_beat,m_axi_rdata,0,m_axi_rlast,m_axi_rid,m_axi_rresp);mr_total++;
                if(mr_beat==length(mr_tx)-1) begin mr_tx++;mr_beat=0;end else mr_beat++;
            end
        end
    end
    task automatic reset_pair(input bit stopped);
        longint before_ticks,stop_time;
        // The peer clocks may stop, but reset is held until BOTH have sampled it.
        if(stopped) begin @(negedge s_axi_aclk);run_s=0;end
        // Log the final prefix at the reset boundary, after any last source
        // edge needed to park its clock low. No traffic may follow this marker.
        if(go && abort_mode!=0)
            $display("NTRPU_CDC_ABORT lite=%0d case=%0d mode=%0d saw=%0d sar=%0d sw=%0d sb=%0d sr=%0d maw=%0d mar=%0d mw=%0d mb=%0d mr=%0d stopped=%0d time_ps=%0d",LITE,ident,abort_mode,saw,sar,sw_total,sb,sr_total,maw,mar,mw_total,mb,mr_total,stopped,ps());
        before_ticks=s_ticks;stop_time=ps();
        go=0;s_axi_aresetn=0;m_axi_aresetn=0;
        repeat(32) @(negedge m_axi_aclk);
        if(stopped) begin
            need(!s_axi_aclk && s_ticks==before_ticks,"no source edge during stopped-clock reset");
            $display("NTRPU_CDC_STOP lite=%0d case=%0d begin_ps=%0d end_ps=%0d source_edges=0 target_falling_edges=32",LITE,ident,stop_time,ps());
            run_s=1;
        end
        fork
            begin repeat(32) @(negedge s_axi_aclk);s_axi_aresetn=1;end
            begin repeat(32) @(negedge m_axi_aclk);m_axi_aresetn=1;end
        join
        repeat(20) @(negedge s_axi_aclk);
        need(!s_axi_bvalid && !s_axi_rvalid && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid,"no stale transaction after coordinated reset");
    endtask
    task automatic exercise(input int mode);
        reset_pair(0);ident++;abort_mode=mode;
        $display("NTRPU_CDC_CASE lite=%0d case=%0d abort=%0d profile=%0d source_half_ps=%0d target_half_ps=%0d phase_ps=%0d tx=%0d time_ps=%0d",LITE,ident,mode,profile,s_half_ps,m_half_ps,phase_ps,TX,ps());
        go=1;
        if(mode==0) begin
            wait(sb==TX && sr_tx==TX && mb==TX && mr_tx==TX);@(negedge s_axi_aclk);
            need(saw==TX && sar==TX && sw_tx==TX && maw==TX && mar==TX && mw_tx==TX && mb==TX && mr_tx==TX,"complete source and target counts");
            $display("NTRPU_CDC_DONE lite=%0d case=%0d sc=%0d mc=%0d wbeats=%0d rbeats=%0d time_ps=%0d",LITE,ident,sc,mc,sw_total,sr_total,ps());
            $display("NTRPU_CDC_STALLS lite=%0d case=%0d saw=%0d sw=%0d sb=%0d sar=%0d sr=%0d maw=%0d mw=%0d mb=%0d mar=%0d mr=%0d",LITE,ident,s_stalls[0],s_stalls[1],s_stalls[2],s_stalls[3],s_stalls[4],m_stalls[0],m_stalls[1],m_stalls[2],m_stalls[3],m_stalls[4]);
            go=0;
        end else begin
            case(mode)
                1:wait(m_axi_arvalid && !m_axi_arready && offer_ar>0);
                2:wait(s_axi_rvalid && !s_axi_rready && offer_r>0);
                3:wait(m_axi_wvalid && !m_axi_wready && mw_total==1);
                4:wait(s_axi_bvalid && !s_axi_bready && offer_b>0);
            endcase
            repeat(8) @(negedge m_axi_aclk);
            reset_pair(mode==4);
        end
    endtask
    initial begin
        finished=0;
        if($value$plusargs("PROFILE=%d",profile)) begin end
        need(profile>=0 && profile<=2,"clock profile");
        case(profile)
            0:begin s_half_ps=2000;m_half_ps=2500;phase_ps=700;end
            1:begin s_half_ps=3100;m_half_ps=2300;phase_ps=900;end
            2:begin s_half_ps=1700;m_half_ps=2900;phase_ps=300;end
        endcase
        clocks_ready=1;
        inject_stale=$test$plusargs("INJECT_STALE");
        inject_rresp=$test$plusargs("INJECT_RRESP");
        if(LITE) $display("NTRPU_RRESP_RECODE enabled=1 fault=%0d profile=%0d",inject_rresp,profile);
        exercise(0);
        for(int mode=1;mode<=4;mode++) begin exercise(mode);exercise(0);end
        $display("NTRPU_CDC_PASS lite=%0d profile=%0d cases=9 resets=4",LITE,profile);
        finished=1;
    end
endmodule
module tb_ntrpu_host_clock_recode;
    wire full_done,lite_done;
    ntrpu_host_clock_recode_bank #(.LITE(0)) full(full_done);
    ntrpu_host_clock_recode_bank #(.LITE(1)) lite(lite_done);
    initial begin wait(full_done && lite_done);$display("NTRPU_HOST_CLOCK_PASS banks=18 resets=8");$finish;end
    initial begin #1000000;$fatal(1,"FAIL HOST_CLOCK global watchdog");end
endmodule
