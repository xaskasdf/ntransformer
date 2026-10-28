`timescale 1ps / 1ps
module tb_ntrpu_axi4_pair_drain;
    parameter integer DEVICE_PERIOD_PS=5000;
    localparam integer N=24;
    reg mgmt_clk=0,host_clk=0,device_clk=0,run_host=1,run_device=1;
    always #5000 mgmt_clk=~mgmt_clk;
    always #3500 if(run_host)host_clk=~host_clk;
    initial begin #600;forever begin #(DEVICE_PERIOD_PS/2);if(run_device)device_clk=~device_clk;end end
    reg cold_rst_n=0,host_rst_n=0,device_rst_n=0,active=0;
    reg req_valid=0,host_source_fenced=0,release_responses=0,inject_orphan=0;
    reg [63:0] generation=64'h100000001,tag=64'h200000001;
    integer phase=0,hcycle=0,dcycle=0;
    wire external_drain_ack,host_certificate,device_certificate,host_mailbox_busy,device_mailbox_busy;
    wire [5:0] host_fault_sources,device_fault_sources;
    wire [11:0] fault_mgmt;
    wire [5:0] host_aw_pending,host_w_pending,device_aw_pending,device_w_pending;
    wire [7:0] host_ar_pending,device_ar_pending;
    wire [8:0] host_w_partial,device_w_partial;
    reg h_awvalid=0;
    wire h_awready;
    reg [63:0] h_awaddr=0;
    reg [3:0] h_awid=0;
    reg [7:0] h_awlen=0;
    reg [2:0] h_awsize=0;
    reg [2:0] h_awprot=0;
    reg [1:0] h_awburst=0;
    reg [3:0] h_awcache=0;
    reg h_awlock=0;
    reg h_wvalid=0;
    wire h_wready;
    reg [511:0] h_wdata=0;
    reg [63:0] h_wstrb=0;
    reg h_wlast=0;
    wire h_bvalid;
    reg h_bready=0;
    wire [3:0] h_bid;
    wire [1:0] h_bresp;
    reg h_arvalid=0;
    wire h_arready;
    reg [63:0] h_araddr=0;
    reg [3:0] h_arid=0;
    reg [7:0] h_arlen=0;
    reg [2:0] h_arsize=0;
    reg [2:0] h_arprot=0;
    reg [1:0] h_arburst=0;
    reg [3:0] h_arcache=0;
    reg h_arlock=0;
    wire h_rvalid;
    reg h_rready=0;
    wire [511:0] h_rdata;
    wire [3:0] h_rid;
    wire [1:0] h_rresp;
    wire h_rlast;
    wire d_awvalid;
    reg d_awready=0;
    wire [63:0] d_awaddr;
    wire [3:0] d_awid;
    wire [7:0] d_awlen;
    wire [2:0] d_awsize;
    wire [2:0] d_awprot;
    wire [1:0] d_awburst;
    wire [3:0] d_awcache;
    wire d_awlock;
    wire d_wvalid;
    reg d_wready=0;
    wire [511:0] d_wdata;
    wire [63:0] d_wstrb;
    wire d_wlast;
    reg d_bvalid=0;
    wire d_bready;
    reg [3:0] d_bid=0;
    reg [1:0] d_bresp=0;
    wire d_arvalid;
    reg d_arready=0;
    wire [63:0] d_araddr;
    wire [3:0] d_arid;
    wire [7:0] d_arlen;
    wire [2:0] d_arsize;
    wire [2:0] d_arprot;
    wire [1:0] d_arburst;
    wire [3:0] d_arcache;
    wire d_arlock;
    reg d_rvalid=0;
    wire d_rready;
    reg [511:0] d_rdata=0;
    reg [3:0] d_rid=0;
    reg [1:0] d_rresp=0;
    reg d_rlast=0;
    ntrpu_axi4_pair_drain dut(
        .host_awvalid(h_awvalid),
        .host_awready(h_awready),
        .host_awaddr(h_awaddr),
        .host_awid(h_awid),
        .host_awlen(h_awlen),
        .host_awsize(h_awsize),
        .host_awprot(h_awprot),
        .host_awburst(h_awburst),
        .host_awcache(h_awcache),
        .host_awlock(h_awlock),
        .host_wvalid(h_wvalid),
        .host_wready(h_wready),
        .host_wdata(h_wdata),
        .host_wstrb(h_wstrb),
        .host_wlast(h_wlast),
        .host_bvalid(h_bvalid),
        .host_bready(h_bready),
        .host_bid(h_bid),
        .host_bresp(h_bresp),
        .host_arvalid(h_arvalid),
        .host_arready(h_arready),
        .host_araddr(h_araddr),
        .host_arid(h_arid),
        .host_arlen(h_arlen),
        .host_arsize(h_arsize),
        .host_arprot(h_arprot),
        .host_arburst(h_arburst),
        .host_arcache(h_arcache),
        .host_arlock(h_arlock),
        .host_rvalid(h_rvalid),
        .host_rready(h_rready),
        .host_rdata(h_rdata),
        .host_rid(h_rid),
        .host_rresp(h_rresp),
        .host_rlast(h_rlast),
        .device_awvalid(d_awvalid),
        .device_awready(d_awready),
        .device_awaddr(d_awaddr),
        .device_awid(d_awid),
        .device_awlen(d_awlen),
        .device_awsize(d_awsize),
        .device_awprot(d_awprot),
        .device_awburst(d_awburst),
        .device_awcache(d_awcache),
        .device_awlock(d_awlock),
        .device_wvalid(d_wvalid),
        .device_wready(d_wready),
        .device_wdata(d_wdata),
        .device_wstrb(d_wstrb),
        .device_wlast(d_wlast),
        .device_bvalid(d_bvalid),
        .device_bready(d_bready),
        .device_bid(d_bid),
        .device_bresp(d_bresp),
        .device_arvalid(d_arvalid),
        .device_arready(d_arready),
        .device_araddr(d_araddr),
        .device_arid(d_arid),
        .device_arlen(d_arlen),
        .device_arsize(d_arsize),
        .device_arprot(d_arprot),
        .device_arburst(d_arburst),
        .device_arcache(d_arcache),
        .device_arlock(d_arlock),
        .device_rvalid(d_rvalid),
        .device_rready(d_rready),
        .device_rdata(d_rdata),
        .device_rid(d_rid),
        .device_rresp(d_rresp),
        .device_rlast(d_rlast),
        .*);
    ntrpu_axi4_cdc_probe cdc(
        .s_axi_aclk(host_clk),.s_axi_aresetn(host_rst_n),.m_axi_aclk(device_clk),.m_axi_aresetn(device_rst_n),
        .s_axi_awqos(4'b0),.m_axi_awqos(),.s_axi_arqos(4'b0),.m_axi_arqos(),
        .s_axi_awregion(4'b0),.m_axi_awregion(),.s_axi_arregion(4'b0),.m_axi_arregion(),
        .s_axi_awvalid(h_awvalid),
        .s_axi_awready(h_awready),
        .s_axi_awaddr(h_awaddr),
        .s_axi_awid(h_awid),
        .s_axi_awlen(h_awlen),
        .s_axi_awsize(h_awsize),
        .s_axi_awprot(h_awprot),
        .s_axi_awburst(h_awburst),
        .s_axi_awcache(h_awcache),
        .s_axi_awlock(h_awlock),
        .s_axi_wvalid(h_wvalid),
        .s_axi_wready(h_wready),
        .s_axi_wdata(h_wdata),
        .s_axi_wstrb(h_wstrb),
        .s_axi_wlast(h_wlast),
        .s_axi_bvalid(h_bvalid),
        .s_axi_bready(h_bready),
        .s_axi_bid(h_bid),
        .s_axi_bresp(h_bresp),
        .s_axi_arvalid(h_arvalid),
        .s_axi_arready(h_arready),
        .s_axi_araddr(h_araddr),
        .s_axi_arid(h_arid),
        .s_axi_arlen(h_arlen),
        .s_axi_arsize(h_arsize),
        .s_axi_arprot(h_arprot),
        .s_axi_arburst(h_arburst),
        .s_axi_arcache(h_arcache),
        .s_axi_arlock(h_arlock),
        .s_axi_rvalid(h_rvalid),
        .s_axi_rready(h_rready),
        .s_axi_rdata(h_rdata),
        .s_axi_rid(h_rid),
        .s_axi_rresp(h_rresp),
        .s_axi_rlast(h_rlast),
        .m_axi_awvalid(d_awvalid),
        .m_axi_awready(d_awready),
        .m_axi_awaddr(d_awaddr),
        .m_axi_awid(d_awid),
        .m_axi_awlen(d_awlen),
        .m_axi_awsize(d_awsize),
        .m_axi_awprot(d_awprot),
        .m_axi_awburst(d_awburst),
        .m_axi_awcache(d_awcache),
        .m_axi_awlock(d_awlock),
        .m_axi_wvalid(d_wvalid),
        .m_axi_wready(d_wready),
        .m_axi_wdata(d_wdata),
        .m_axi_wstrb(d_wstrb),
        .m_axi_wlast(d_wlast),
        .m_axi_bvalid(d_bvalid),
        .m_axi_bready(d_bready),
        .m_axi_bid(d_bid),
        .m_axi_bresp(d_bresp),
        .m_axi_arvalid(d_arvalid),
        .m_axi_arready(d_arready),
        .m_axi_araddr(d_araddr),
        .m_axi_arid(d_arid),
        .m_axi_arlen(d_arlen),
        .m_axi_arsize(d_arsize),
        .m_axi_arprot(d_arprot),
        .m_axi_arburst(d_arburst),
        .m_axi_arcache(d_arcache),
        .m_axi_arlock(d_arlock),
        .m_axi_rvalid(d_rvalid),
        .m_axi_rready(d_rready),
        .m_axi_rdata(d_rdata),
        .m_axi_rid(d_rid),
        .m_axi_rresp(d_rresp),
        .m_axi_rlast(d_rlast));
    integer lens[0:6]='{1,2,4,16,64,256,3};
    integer aw_count=0,w_count=0,w_beat=0,ar_count=0,paired=0,b_sent=0,r_sent=0;
    integer h_bcount=0,h_rcount=0,h_awcount=0,h_wcount=0,h_arcount=0;
    integer aw_len[0:N-1],aw_size[0:N-1],aw_id[0:N-1],w_len[0:N-1];
    reg [63:0] aw_addr[0:N-1],ar_addr[0:N-1];
    integer ar_len[0:N-1],ar_id[0:N-1],r_index[0:15],r_beat[0:15];
    reg [511:0] saved_w[0:N-1][0:255];
    reg [63:0] saved_strb[0:N-1][0:255];
    integer r_cursor=15;
    reg orphan_offer=0,orphan_sent=0;
    function [511:0] pattern(input [31:0] seed);
        for(integer j=0;j<16;j=j+1) pattern[j*32+:32]=seed^(32'h9e3779b9*j);
    endfunction
    function [63:0] mask(input integer beat,input integer size);
        mask=(size==4)?(64'ha5f3<<((beat%4)*16)):
             (64'hf0ff55aa12345678^(64'h0101010101010101*(beat%16)));
    endfunction
    task must(input bit ok,input string text);
        if(!ok)$fatal(1,"AXI4_PAIR_ASSERT phase=%0d %s",phase,text);
    endtask
    task event_log(input string name);
        $display("AXI4_PAIR_EVENT name=%s t=%0d phase=%0d gen=%h tag=%h ack=%0d hc=%0d dc=%0d hb=%0d db=%0d fm=%0d hf=%0d df=%0d hap=%0d hwp=%0d har=%0d dap=%0d dwp=%0d dar=%0d haw=%0d hw=%0d haddr=%0d hbs=%0d hrs=%0d paired=%0d bs=%0d rs=%0d",
            name,$time,phase,generation,tag,external_drain_ack,host_certificate,device_certificate,
            host_mailbox_busy,device_mailbox_busy,fault_mgmt,host_fault_sources,device_fault_sources,
            host_aw_pending,host_w_pending,host_ar_pending,device_aw_pending,device_w_pending,device_ar_pending,
            h_awcount,h_wcount,h_arcount,h_bcount,h_rcount,paired,b_sent,r_sent);
    endtask
    task host_aw(input integer n);
        @(negedge host_clk);
        h_awid=n%16;h_awaddr=64'h100000000+n*4096;h_awlen=lens[n%7]-1;
        h_awsize=(n%7==5)?4:6;h_awprot=n%8;h_awcache=3+4*(n%4);h_awburst=1;h_awlock=0;h_awvalid=1;
        do @(posedge host_clk);while(!h_awready);
        @(negedge host_clk);h_awvalid=0;
    endtask
    task host_w(input integer n,input integer beat);
        @(negedge host_clk);
        h_wdata=pattern(n*4096^(beat<<8));h_wstrb=mask(beat,(n%7==5)?4:6);
        h_wlast=beat==lens[n%7]-1;h_wvalid=1;
        do @(posedge host_clk);while(!h_wready);
        @(negedge host_clk);h_wvalid=0;
    endtask
    task host_ar(input integer n);
        @(negedge host_clk);
        h_arid=n%16;h_araddr=64'h200000000+n*4096;h_arlen=lens[n%7]-1;
        h_arsize=(n%7==5)?4:6;h_arprot=n%8;h_arcache=3+4*(n%4);h_arburst=1;h_arlock=0;h_arvalid=1;
        do @(posedge host_clk);while(!h_arready);
        @(negedge host_clk);h_arvalid=0;
    endtask
    always @(negedge host_clk) if(active)begin
        h_bready=release_responses && (phase>=3 || hcycle%5!=1);
        h_rready=release_responses && (phase>=3 || hcycle%7!=2);
    end
    always @(negedge device_clk) if(active)begin
        d_awready=device_rst_n && aw_count<N && dcycle%5!=2;
        d_wready=device_rst_n && w_count<N && dcycle%7!=3;
        d_arready=device_rst_n && ar_count<N && dcycle%4!=1;
    end
    always @(posedge host_clk)if(active && host_rst_n)begin
        hcycle=hcycle+1;
        if(h_awvalid && h_awready)h_awcount=h_awcount+1;
        if(h_wvalid && h_wready && h_wlast)h_wcount=h_wcount+1;
        if(h_arvalid && h_arready)h_arcount=h_arcount+1;
        if(h_bvalid && h_bready)h_bcount=h_bcount+1;
        if(h_rvalid && h_rready)h_rcount=h_rcount+1;
        if(phase<3)must(host_fault_sources==0,"host observer healthy");
    end
    always @(posedge device_clk)if(active && device_rst_n)begin:backend
        integer selected,candidate,transaction;
        dcycle=dcycle+1;
        if(phase<3)must(device_fault_sources==0,"device observer healthy");
        if(d_awvalid && d_awready)begin
            aw_len[aw_count]=d_awlen+1;aw_size[aw_count]=d_awsize;
            aw_id[aw_count]=d_awid;aw_addr[aw_count]=d_awaddr;aw_count=aw_count+1;
        end
        if(d_wvalid && d_wready)begin
            must(w_beat<256,"bounded W burst");saved_w[w_count][w_beat]=d_wdata;saved_strb[w_count][w_beat]=d_wstrb;
            w_beat=w_beat+1;
            if(d_wlast)begin w_len[w_count]=w_beat;w_count=w_count+1;w_beat=0;end
        end
        if(paired<aw_count && paired<w_count)begin
            must(aw_len[paired]==w_len[paired],"device LEN LAST association");
            must(aw_id[paired]==paired%16 && aw_addr[paired]==64'h100000000+paired*4096,"full write address and ID");
            for(integer beat=0;beat<w_len[paired];beat=beat+1)begin
                must(saved_w[paired][beat]===pattern(paired*4096^(beat<<8)),"device W payload");
                must(saved_strb[paired][beat]===mask(beat,aw_size[paired]),"device W strobe");
            end
            paired=paired+1;
        end
        if(d_arvalid && d_arready)begin
            must(d_arid==ar_count%16 && d_araddr==64'h200000000+ar_count*4096,"full read address and ID");
            ar_len[ar_count]=d_arlen+1;ar_id[ar_count]=d_arid;ar_addr[ar_count]=d_araddr;ar_count=ar_count+1;
        end
        if(d_bvalid && d_bready)b_sent=b_sent+1;
        if(!d_bvalid || d_bready)begin
            if(paired==N && b_sent<N)begin
                transaction=(b_sent<16)?15-b_sent:39-b_sent;
                d_bid<=transaction%16;d_bresp<=transaction%4;d_bvalid<=1;
            end else d_bvalid<=0;
        end
        if(d_rvalid && d_rready)begin
            if(orphan_offer)orphan_sent=1;
            else begin
                r_sent=r_sent+1;
                if(d_rlast)begin r_index[d_rid]=r_index[d_rid]+16;r_beat[d_rid]=0;end
                else r_beat[d_rid]=r_beat[d_rid]+1;
                r_cursor=(d_rid==0)?15:d_rid-1;
            end
        end
        if(!d_rvalid || d_rready)begin
            selected=-1;
            if(ar_count==N)for(integer offset=0;offset<16;offset=offset+1)begin
                candidate=(r_cursor-offset+16)%16;
                if(selected<0 && r_index[candidate]<N)selected=candidate;
            end
            if(selected>=0)begin
                transaction=r_index[selected];
                d_rid<=selected;d_rresp<=r_beat[selected]%4;
                d_rdata<=pattern(transaction*4096^(r_beat[selected]<<8)^selected);
                d_rlast<=r_beat[selected]==ar_len[transaction]-1;d_rvalid<=1;orphan_offer<=0;
            end else if(inject_orphan && !orphan_sent)begin
                d_rid<=3;d_rresp<=2;d_rdata<=pattern(32'hbad4c0de);d_rlast<=1;d_rvalid<=1;orphan_offer<=1;
            end else begin d_rvalid<=0;orphan_offer<=0;end
        end
    end
    always @(posedge host_clk)if(active && host_rst_n)begin
        $display("AXI4_CDC_BUS side=H t=%0d phase=%0d awvalid=%0d awready=%0d awaddr=%h awid=%0d awlen=%0d awsize=%0d awprot=%0d awburst=%0d awcache=%0d awlock=%0d wvalid=%0d wready=%0d wdata=%h wstrb=%h wlast=%0d bvalid=%0d bready=%0d bid=%0d bresp=%0d arvalid=%0d arready=%0d araddr=%h arid=%0d arlen=%0d arsize=%0d arprot=%0d arburst=%0d arcache=%0d arlock=%0d rvalid=%0d rready=%0d rdata=%h rid=%0d rresp=%0d rlast=%0d ap=%0d wp=%0d ar=%0d partial=%0d faults=%0d",
            $time,phase,h_awvalid,h_awready,h_awaddr,h_awid,h_awlen,h_awsize,h_awprot,h_awburst,h_awcache,h_awlock,h_wvalid,h_wready,h_wdata,h_wstrb,h_wlast,h_bvalid,h_bready,h_bid,h_bresp,h_arvalid,h_arready,h_araddr,h_arid,h_arlen,h_arsize,h_arprot,h_arburst,h_arcache,h_arlock,h_rvalid,h_rready,h_rdata,h_rid,h_rresp,h_rlast,host_aw_pending,host_w_pending,host_ar_pending,host_w_partial,host_fault_sources);
    end
    always @(posedge device_clk)if(active && device_rst_n)begin
        $display("AXI4_CDC_BUS side=D t=%0d phase=%0d awvalid=%0d awready=%0d awaddr=%h awid=%0d awlen=%0d awsize=%0d awprot=%0d awburst=%0d awcache=%0d awlock=%0d wvalid=%0d wready=%0d wdata=%h wstrb=%h wlast=%0d bvalid=%0d bready=%0d bid=%0d bresp=%0d arvalid=%0d arready=%0d araddr=%h arid=%0d arlen=%0d arsize=%0d arprot=%0d arburst=%0d arcache=%0d arlock=%0d rvalid=%0d rready=%0d rdata=%h rid=%0d rresp=%0d rlast=%0d ap=%0d wp=%0d ar=%0d partial=%0d faults=%0d",
            $time,phase,d_awvalid,d_awready,d_awaddr,d_awid,d_awlen,d_awsize,d_awprot,d_awburst,d_awcache,d_awlock,d_wvalid,d_wready,d_wdata,d_wstrb,d_wlast,d_bvalid,d_bready,d_bid,d_bresp,d_arvalid,d_arready,d_araddr,d_arid,d_arlen,d_arsize,d_arprot,d_arburst,d_arcache,d_arlock,d_rvalid,d_rready,d_rdata,d_rid,d_rresp,d_rlast,device_aw_pending,device_w_pending,device_ar_pending,device_w_partial,device_fault_sources);
    end
    reg [11:0] last_fault=0;
    always @(posedge mgmt_clk)begin
        #1;
        if(active && phase<3 && external_drain_ack)begin
            must(host_source_fenced && h_bcount==N && h_rcount==1045,"no certificate before all host responses");
            must(fault_mgmt==0,"no certificate after fault report");
            must(host_aw_pending==0 && host_w_pending==0 && host_ar_pending==0 &&
                 device_aw_pending==0 && device_w_pending==0 && device_ar_pending==0,"both ledgers empty at every certificate");
        end
        if(active && (|fault_mgmt))must(!external_drain_ack,"reported fault always revokes certificate");
        if(active && fault_mgmt!=last_fault)begin event_log("FAULT_REPORT");last_fault=fault_mgmt;end
    end
    initial begin
        for(integer id=0;id<16;id=id+1)begin r_index[id]=id;r_beat[id]=0;end
        $display("AXI4_PAIR_PROFILE device_period_ps=%0d host_period_ps=7000 mgmt_period_ps=10000 transactions=24 write_depth=32",DEVICE_PERIOD_PS);
        repeat(32)@(posedge mgmt_clk);#1000;cold_rst_n=1;
        repeat(64)@(posedge host_clk);@(negedge host_clk);host_rst_n=1;
        repeat(64)@(posedge device_clk);@(negedge device_clk);device_rst_n=1;
        repeat(16)@(posedge mgmt_clk);#1000;active=1;event_log("START");
        fork
            begin repeat(40)@(posedge host_clk);for(integer n=0;n<N;n=n+1)host_aw(n);end
            begin for(integer n=0;n<N;n=n+1)for(integer beat=0;beat<lens[n%7];beat=beat+1)host_w(n,beat);end
            begin repeat(25)@(posedge host_clk);for(integer n=0;n<N;n=n+1)host_ar(n);end
        join
        @(negedge host_clk);host_source_fenced=1;
        @(negedge mgmt_clk);req_valid=1;event_log("FENCED_REQUEST");
        wait(h_bvalid && h_rvalid && dut.hreq);#1;
        must(!external_drain_ack,"held responses prevent proof");event_log("HELD_RESPONSES");
        @(posedge device_clk);#1;run_device=0;event_log("STOP_DEVICE");
        release_responses=1;
        repeat(80)@(posedge mgmt_clk);#1;
        must(!external_drain_ack && fault_mgmt==0,"stopped device holds pending proof without fault");event_log("DEVICE_WAIT");
        run_device=1;event_log("RESUME_DEVICE");
        wait(external_drain_ack);#1;
        must(h_awcount==N && h_wcount==N && h_arcount==N && h_bcount==N && h_rcount==r_sent,"traffic counts at proof");
        must(!host_fault_sources && !device_fault_sources,"healthy first proof");event_log("PROOF_ONE");
        wait(!host_mailbox_busy && !device_mailbox_busy);
        @(posedge host_clk);#1;run_host=0;phase=1;tag=tag+64'h100000001;#1;
        must(!external_drain_ack,"new tag immediately masks cached proof");event_log("STOP_HOST_NEW_TAG");
        repeat(50)@(posedge mgmt_clk);#1;
        must(!external_drain_ack && host_mailbox_busy,"fresh proof requires host clock");event_log("HOST_WAIT");
        run_host=1;event_log("RESUME_HOST");wait(external_drain_ack);#1;event_log("PROOF_TWO");
        wait(!host_mailbox_busy && !device_mailbox_busy);
        @(negedge mgmt_clk);phase=2;generation=generation+64'h100000001;#1;
        must(!external_drain_ack,"new generation immediately masks cached proof");event_log("NEW_GENERATION");
        wait(external_drain_ack);#1;event_log("PROOF_THREE");
        wait(!host_mailbox_busy && !device_mailbox_busy);
        @(negedge device_clk);phase=3;inject_orphan=1;event_log("INJECT_ORPHAN_R");
        wait(device_fault_sources==4);#1;run_device=0;event_log("STOP_DEVICE_FAULT");
        repeat(4)@(posedge mgmt_clk);#1;
        must(fault_mgmt[8],"device fault must report with source clock stopped");
        must(!external_drain_ack,"device fault revokes cached proof");event_log("DEVICE_FAULT_REVOKED");
        repeat(12)@(posedge mgmt_clk);#1;
        tag=tag+1;generation=generation+1;event_log("FAULT_FRESH_REQUEST");
        repeat(20)@(posedge mgmt_clk);#1;
        must(!external_drain_ack && !host_mailbox_busy && !device_mailbox_busy,"sticky fault blocks new mailboxes");
        run_device=1;event_log("RESUME_DEVICE_FAULT");
        wait(host_fault_sources==4 && fault_mgmt==12'h104);#1;
        repeat(10)@(posedge mgmt_clk);#1;
        must(orphan_sent && h_rcount==r_sent+1 && !external_drain_ack,"real orphan crossed CDC and remains revoked");
        event_log("TERMINAL");active=0;
        $display("AXI4_PAIR_SUITE_PASS device_period_ps=%0d writes=24 reads=24 proofs=3 native_endpoint=0",DEVICE_PERIOD_PS);$finish;
    end
    initial begin #200000000;$fatal(1,"AXI4_PAIR_TIMEOUT");end
endmodule
