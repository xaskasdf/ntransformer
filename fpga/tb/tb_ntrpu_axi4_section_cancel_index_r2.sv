`timescale 1ps / 1ps
module tb_ntrpu_axi4_section_cancel_index_r2;
    parameter integer DEVICE_PERIOD_PS=5000;
    localparam integer N=24;
    reg mgmt_clk=0,host_clk=0,device_clk=0,run_host=1,run_device=1;
    always #5000 mgmt_clk=~mgmt_clk;
    always #3500 if(run_host)host_clk=~host_clk;
    initial begin #600;forever begin #(DEVICE_PERIOD_PS/2);if(run_device)device_clk=~device_clk;end end
    reg cold_rst_n=0,host_rst_n=0,device_rst_n=0,active=0;
    reg req_valid=0,release_responses=0,inject_orphan=0;
    wire host_source_fenced;
    reg [63:0] generation=64'h100000001,tag=64'h200000001;
    integer phase=0,hcycle=0,dcycle=0;
    wire section_ack,upstream_certificate,pair_certificate,host_certificate,device_certificate;
    wire host_mailbox_busy,pair_host_busy,pair_device_busy;
    wire close_host,host_proof_request,host_drain_ack,local_idle,protocol_fault;
    wire [63:0] host_proof_tag,host_drain_tag;
    wire [6:0] admission_fault_sources;
    wire [5:0] admission_aw_pending,admission_w_pending,aw_left,w_left;
    wire [7:0] admission_ar_pending;
    wire [8:0] admission_w_partial;
    wire ar_left;
    wire [1:0] state_debug;
    wire [5:0] host_fault_sources,device_fault_sources;
    wire [18:0] fault_mgmt;
    wire [5:0] host_aw_pending,host_w_pending,device_aw_pending,device_w_pending;
    wire [7:0] host_ar_pending,device_ar_pending;
    wire [8:0] host_w_partial,device_w_partial;
    reg s_awvalid=0;
    wire s_awready;
    reg [63:0] s_awaddr=0;
    reg [3:0] s_awid=0;
    reg [7:0] s_awlen=0;
    reg [2:0] s_awsize=0;
    reg [2:0] s_awprot=0;
    reg [1:0] s_awburst=0;
    reg [3:0] s_awcache=0;
    reg s_awlock=0;
    reg s_wvalid=0;
    wire s_wready;
    reg [511:0] s_wdata=0;
    reg [63:0] s_wstrb=0;
    reg s_wlast=0;
    wire s_bvalid;
    reg s_bready=0;
    wire [3:0] s_bid;
    wire [1:0] s_bresp;
    reg s_arvalid=0;
    wire s_arready;
    reg [63:0] s_araddr=0;
    reg [3:0] s_arid=0;
    reg [7:0] s_arlen=0;
    reg [2:0] s_arsize=0;
    reg [2:0] s_arprot=0;
    reg [1:0] s_arburst=0;
    reg [3:0] s_arcache=0;
    reg s_arlock=0;
    wire s_rvalid;
    reg s_rready=0;
    wire [511:0] s_rdata;
    wire [3:0] s_rid;
    wire [1:0] s_rresp;
    wire s_rlast;
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
    wire h_awvalid;
    wire h_awready;
    wire [63:0] h_awaddr;
    wire [3:0] h_awid;
    wire [7:0] h_awlen;
    wire [2:0] h_awsize;
    wire [2:0] h_awprot;
    wire [1:0] h_awburst;
    wire [3:0] h_awcache;
    wire h_awlock;
    wire h_wvalid;
    wire h_wready;
    wire [511:0] h_wdata;
    wire [63:0] h_wstrb;
    wire h_wlast;
    wire h_bvalid;
    wire h_bready;
    wire [3:0] h_bid;
    wire [1:0] h_bresp;
    wire h_arvalid;
    wire h_arready;
    wire [63:0] h_araddr;
    wire [3:0] h_arid;
    wire [7:0] h_arlen;
    wire [2:0] h_arsize;
    wire [2:0] h_arprot;
    wire [1:0] h_arburst;
    wire [3:0] h_arcache;
    wire h_arlock;
    wire h_rvalid;
    wire h_rready;
    wire [511:0] h_rdata;
    wire [3:0] h_rid;
    wire [1:0] h_rresp;
    wire h_rlast;
    wire [3:0] h_awregion;
    wire [3:0] h_awqos;
    wire [3:0] h_arregion;
    wire [3:0] h_arqos;
    reg [3:0] s_awregion=0;
    wire [3:0] d_awregion;
    reg [3:0] s_awqos=0;
    wire [3:0] d_awqos;
    reg [3:0] s_arregion=0;
    wire [3:0] d_arregion;
    reg [3:0] s_arqos=0;
    wire [3:0] d_arqos;
    ntrpu_axi4_retained_admission_r2 admission(
        .clk(host_clk),.cold_rst_n(cold_rst_n),.close_request(close_host),
        .proof_request(host_proof_request),.proof_tag(host_proof_tag),
        .s_awvalid(s_awvalid),
        .s_awready(s_awready),
        .s_awaddr(s_awaddr),
        .s_awid(s_awid),
        .s_awlen(s_awlen),
        .s_awsize(s_awsize),
        .s_awprot(s_awprot),
        .s_awburst(s_awburst),
        .s_awcache(s_awcache),
        .s_awlock(s_awlock),
        .s_wvalid(s_wvalid),
        .s_wready(s_wready),
        .s_wdata(s_wdata),
        .s_wstrb(s_wstrb),
        .s_wlast(s_wlast),
        .s_bvalid(s_bvalid),
        .s_bready(s_bready),
        .s_bid(s_bid),
        .s_bresp(s_bresp),
        .s_arvalid(s_arvalid),
        .s_arready(s_arready),
        .s_araddr(s_araddr),
        .s_arid(s_arid),
        .s_arlen(s_arlen),
        .s_arsize(s_arsize),
        .s_arprot(s_arprot),
        .s_arburst(s_arburst),
        .s_arcache(s_arcache),
        .s_arlock(s_arlock),
        .s_rvalid(s_rvalid),
        .s_rready(s_rready),
        .s_rdata(s_rdata),
        .s_rid(s_rid),
        .s_rresp(s_rresp),
        .s_rlast(s_rlast),
        .s_awregion(s_awregion),
        .s_awqos(s_awqos),
        .s_arregion(s_arregion),
        .s_arqos(s_arqos),
        .m_awvalid(h_awvalid),
        .m_awready(h_awready),
        .m_awaddr(h_awaddr),
        .m_awid(h_awid),
        .m_awlen(h_awlen),
        .m_awsize(h_awsize),
        .m_awprot(h_awprot),
        .m_awburst(h_awburst),
        .m_awcache(h_awcache),
        .m_awlock(h_awlock),
        .m_wvalid(h_wvalid),
        .m_wready(h_wready),
        .m_wdata(h_wdata),
        .m_wstrb(h_wstrb),
        .m_wlast(h_wlast),
        .m_bvalid(h_bvalid),
        .m_bready(h_bready),
        .m_bid(h_bid),
        .m_bresp(h_bresp),
        .m_arvalid(h_arvalid),
        .m_arready(h_arready),
        .m_araddr(h_araddr),
        .m_arid(h_arid),
        .m_arlen(h_arlen),
        .m_arsize(h_arsize),
        .m_arprot(h_arprot),
        .m_arburst(h_arburst),
        .m_arcache(h_arcache),
        .m_arlock(h_arlock),
        .m_rvalid(h_rvalid),
        .m_rready(h_rready),
        .m_rdata(h_rdata),
        .m_rid(h_rid),
        .m_rresp(h_rresp),
        .m_rlast(h_rlast),
        .m_awregion(h_awregion),
        .m_awqos(h_awqos),
        .m_arregion(h_arregion),
        .m_arqos(h_arqos),
        .source_fenced(host_source_fenced),.drain_ack(host_drain_ack),.drain_tag(host_drain_tag),
        .local_idle(local_idle),.protocol_fault(protocol_fault),.fault_sources(admission_fault_sources),
        .aw_pending(admission_aw_pending),.w_pending(admission_w_pending),.ar_pending(admission_ar_pending),
        .w_partial(admission_w_partial),.aw_left(aw_left),.w_left(w_left),.ar_left(ar_left),.state_debug(state_debug));
    ntrpu_axi4_section_certificate_r2 dut(
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
        .s_axi_awqos(h_awqos),.m_axi_awqos(d_awqos),.s_axi_arqos(h_arqos),.m_axi_arqos(d_arqos),
        .s_axi_awregion(h_awregion),.m_axi_awregion(d_awregion),.s_axi_arregion(h_arregion),.m_axi_arregion(d_arregion),
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
    integer r_cursor=15;
    reg orphan_offer=0,orphan_sent=0;
    function [1:0] response(input integer n);
        response=(n%3==0)?0:((n%3==1)?2:3);
    endfunction
    function [511:0] pattern(input [31:0] seed);
        for(integer j=0;j<16;j=j+1) pattern[j*32+:32]=seed^(32'h9e3779b9*j);
    endfunction
    function [63:0] mask(input integer beat,input integer size);
        mask=(size==4)?(64'ha5f3<<((beat%4)*16)):
             (64'hf0ff55aa12345678^(64'h0101010101010101*(beat%16)));
    endfunction
    task must(input bit ok,input string text);
        if(!ok)$fatal(1,"AXI4_SECTION_ASSERT phase=%0d %s",phase,text);
    endtask
    task event_log(input string name);
        $display("AXI4_SECTION_EVENT name=%s t=%0d phase=%0d gen=%h tag=%h req=%0d ack=%0d uc=%0d pc=%0d hc=%0d dc=%0d ub=%0d hb=%0d db=%0d fm=%0d af=%0d hf=%0d df=%0d close=%0d fenced=%0d proof_req=%0d proof_ack=%0d proof_tag=%h drain_tag=%h state=%0d idle=%0d awleft=%0d wleft=%0d arleft=%0d ap=%0d wp=%0d arp=%0d partial=%0d hap=%0d hwp=%0d har=%0d dap=%0d dwp=%0d dar=%0d hrst=%0d drst=%0d cold=%0d sbv=%0d srv=%0d sbs=%0d srs=%0d daw=%0d dw=%0d dar_count=%0d dbs=%0d drs=%0d",
            name,$time,phase,generation,tag,req_valid,section_ack,upstream_certificate,pair_certificate,host_certificate,device_certificate,host_mailbox_busy,pair_host_busy,pair_device_busy,fault_mgmt,admission_fault_sources,host_fault_sources,device_fault_sources,close_host,host_source_fenced,host_proof_request,host_drain_ack,host_proof_tag,host_drain_tag,state_debug,local_idle,aw_left,w_left,ar_left,admission_aw_pending,admission_w_pending,admission_ar_pending,admission_w_partial,host_aw_pending,host_w_pending,host_ar_pending,device_aw_pending,device_w_pending,device_ar_pending,host_rst_n,device_rst_n,cold_rst_n,s_bvalid,s_rvalid,h_bcount,h_rcount,aw_count,w_count,ar_count,b_sent,r_sent);
    endtask
    task host_aw(input integer n);
        @(negedge host_clk);
        s_awid=n%16;s_awaddr=64'h100000000+n*4096;s_awlen=lens[n%7]-1;
        s_awregion=n%16;s_awqos=15-n%16;
        s_awsize=(n%7==5)?4:6;s_awprot=n%8;s_awcache=3+4*(n%4);s_awburst=1;s_awlock=0;s_awvalid=1;
        do @(posedge host_clk);while(!s_awready);
        @(negedge host_clk);s_awvalid=0;
    endtask
    task host_w(input integer n,input integer beat);
        @(negedge host_clk);
        s_wdata=pattern(n*4096^(beat<<8));s_wstrb=mask(beat,(n%7==5)?4:6);
        s_wlast=beat==lens[n%7]-1;s_wvalid=1;
        do @(posedge host_clk);while(!s_wready);
        @(negedge host_clk);s_wvalid=0;
    endtask
    task host_ar(input integer n);
        @(negedge host_clk);
        s_arid=n%16;s_araddr=64'h200000000+n*4096;s_arlen=lens[n%7]-1;
        s_arregion=n%16;s_arqos=15-n%16;
        s_arsize=(n%7==5)?4:6;s_arprot=n%8;s_arcache=3+4*(n%4);s_arburst=1;s_arlock=0;s_arvalid=1;
        do @(posedge host_clk);while(!s_arready);
        @(negedge host_clk);s_arvalid=0;
    endtask
    always @(negedge host_clk) if(active)begin
        s_bready=release_responses && (phase>=3 || hcycle%5!=1);
        s_rready=release_responses && (phase>=3 || hcycle%7!=2);
    end
    always @(negedge device_clk) if(active)begin
        d_awready=device_rst_n && aw_count<N && dcycle%5!=2;
        d_wready=device_rst_n && w_count<N && dcycle%7!=3;
        d_arready=device_rst_n && ar_count<N && dcycle%4!=1;
    end
    always @(posedge host_clk)if(active && host_rst_n)begin
        hcycle=hcycle+1;
        if(s_awvalid && s_awready)h_awcount=h_awcount+1;
        if(s_wvalid && s_wready && s_wlast)h_wcount=h_wcount+1;
        if(s_arvalid && s_arready)h_arcount=h_arcount+1;
        if(s_bvalid && s_bready)h_bcount=h_bcount+1;
        if(s_rvalid && s_rready)h_rcount=h_rcount+1;
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
            must(w_beat<256,"bounded W burst");
            must(d_wdata===pattern(w_count*4096^(w_beat<<8)),"device W payload");
            must(d_wstrb===mask(w_beat,(w_count%7==5)?4:6),"device W strobe");
            w_beat=w_beat+1;
            if(d_wlast)begin w_len[w_count]=w_beat;w_count=w_count+1;w_beat=0;end
        end
        if(paired<aw_count && paired<w_count)begin
            must(aw_len[paired]==w_len[paired],"device LEN LAST association");
            must(aw_id[paired]==paired%16 && aw_addr[paired]==64'h100000000+paired*4096,"full write address and ID");
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
                d_bid<=transaction%16;d_bresp<=response(transaction);d_bvalid<=1;
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
                d_rid<=selected;d_rresp<=response(r_beat[selected]);
                d_rdata<=pattern(transaction*4096^(r_beat[selected]<<8)^selected);
                d_rlast<=r_beat[selected]==ar_len[transaction]-1;d_rvalid<=1;orphan_offer<=0;
            end else if(inject_orphan && !orphan_sent)begin
                d_rid<=3;d_rresp<=2;d_rdata<=pattern(32'hbad4c0de);d_rlast<=1;d_rvalid<=1;orphan_offer<=1;
            end else begin d_rvalid<=0;orphan_offer<=0;end
        end
    end
    always @(posedge host_clk)if(active)begin
        $display("AXI4_SECTION_BUS side=S t=%0d phase=%0d awvalid=%0d awready=%0d awaddr=%h awid=%0d awlen=%0d awsize=%0d awprot=%0d awburst=%0d awcache=%0d awlock=%0d wvalid=%0d wready=%0d wdata=%h wstrb=%h wlast=%0d bvalid=%0d bready=%0d bid=%0d bresp=%0d arvalid=%0d arready=%0d araddr=%h arid=%0d arlen=%0d arsize=%0d arprot=%0d arburst=%0d arcache=%0d arlock=%0d rvalid=%0d rready=%0d rdata=%h rid=%0d rresp=%0d rlast=%0d awregion=%0d awqos=%0d arregion=%0d arqos=%0d rst=%0d",
            $time,phase,s_awvalid,s_awready,s_awaddr,s_awid,s_awlen,s_awsize,s_awprot,s_awburst,s_awcache,s_awlock,s_wvalid,s_wready,s_wdata,s_wstrb,s_wlast,s_bvalid,s_bready,s_bid,s_bresp,s_arvalid,s_arready,s_araddr,s_arid,s_arlen,s_arsize,s_arprot,s_arburst,s_arcache,s_arlock,s_rvalid,s_rready,s_rdata,s_rid,s_rresp,s_rlast,s_awregion,s_awqos,s_arregion,s_arqos,host_rst_n);
    end
    always @(posedge host_clk)if(active)begin
        $display("AXI4_SECTION_BUS side=H t=%0d phase=%0d awvalid=%0d awready=%0d awaddr=%h awid=%0d awlen=%0d awsize=%0d awprot=%0d awburst=%0d awcache=%0d awlock=%0d wvalid=%0d wready=%0d wdata=%h wstrb=%h wlast=%0d bvalid=%0d bready=%0d bid=%0d bresp=%0d arvalid=%0d arready=%0d araddr=%h arid=%0d arlen=%0d arsize=%0d arprot=%0d arburst=%0d arcache=%0d arlock=%0d rvalid=%0d rready=%0d rdata=%h rid=%0d rresp=%0d rlast=%0d awregion=%0d awqos=%0d arregion=%0d arqos=%0d rst=%0d",
            $time,phase,h_awvalid,h_awready,h_awaddr,h_awid,h_awlen,h_awsize,h_awprot,h_awburst,h_awcache,h_awlock,h_wvalid,h_wready,h_wdata,h_wstrb,h_wlast,h_bvalid,h_bready,h_bid,h_bresp,h_arvalid,h_arready,h_araddr,h_arid,h_arlen,h_arsize,h_arprot,h_arburst,h_arcache,h_arlock,h_rvalid,h_rready,h_rdata,h_rid,h_rresp,h_rlast,h_awregion,h_awqos,h_arregion,h_arqos,host_rst_n);
    end
    always @(posedge device_clk)if(active)begin
        $display("AXI4_SECTION_BUS side=D t=%0d phase=%0d awvalid=%0d awready=%0d awaddr=%h awid=%0d awlen=%0d awsize=%0d awprot=%0d awburst=%0d awcache=%0d awlock=%0d wvalid=%0d wready=%0d wdata=%h wstrb=%h wlast=%0d bvalid=%0d bready=%0d bid=%0d bresp=%0d arvalid=%0d arready=%0d araddr=%h arid=%0d arlen=%0d arsize=%0d arprot=%0d arburst=%0d arcache=%0d arlock=%0d rvalid=%0d rready=%0d rdata=%h rid=%0d rresp=%0d rlast=%0d awregion=%0d awqos=%0d arregion=%0d arqos=%0d rst=%0d",
            $time,phase,d_awvalid,d_awready,d_awaddr,d_awid,d_awlen,d_awsize,d_awprot,d_awburst,d_awcache,d_awlock,d_wvalid,d_wready,d_wdata,d_wstrb,d_wlast,d_bvalid,d_bready,d_bid,d_bresp,d_arvalid,d_arready,d_araddr,d_arid,d_arlen,d_arsize,d_arprot,d_arburst,d_arcache,d_arlock,d_rvalid,d_rready,d_rdata,d_rid,d_rresp,d_rlast,d_awregion,d_awqos,d_arregion,d_arqos,device_rst_n);
    end
    always @(posedge host_clk)if(active)begin
        $display("AXI4_SECTION_HOST t=%0d phase=%0d gen=%h tag=%h req=%0d ack=%0d uc=%0d pc=%0d hc=%0d dc=%0d ub=%0d hb=%0d db=%0d fm=%0d af=%0d hf=%0d df=%0d close=%0d fenced=%0d proof_req=%0d proof_ack=%0d proof_tag=%h drain_tag=%h state=%0d idle=%0d awleft=%0d wleft=%0d arleft=%0d ap=%0d wp=%0d arp=%0d partial=%0d hap=%0d hwp=%0d har=%0d dap=%0d dwp=%0d dar=%0d hrst=%0d drst=%0d cold=%0d sbv=%0d srv=%0d sbs=%0d srs=%0d daw=%0d dw=%0d dar_count=%0d dbs=%0d drs=%0d",
            $time,phase,generation,tag,req_valid,section_ack,upstream_certificate,pair_certificate,host_certificate,device_certificate,host_mailbox_busy,pair_host_busy,pair_device_busy,fault_mgmt,admission_fault_sources,host_fault_sources,device_fault_sources,close_host,host_source_fenced,host_proof_request,host_drain_ack,host_proof_tag,host_drain_tag,state_debug,local_idle,aw_left,w_left,ar_left,admission_aw_pending,admission_w_pending,admission_ar_pending,admission_w_partial,host_aw_pending,host_w_pending,host_ar_pending,device_aw_pending,device_w_pending,device_ar_pending,host_rst_n,device_rst_n,cold_rst_n,s_bvalid,s_rvalid,h_bcount,h_rcount,aw_count,w_count,ar_count,b_sent,r_sent);
    end
    always @(posedge mgmt_clk)begin
        #1;
        if(active)begin
            $display("AXI4_SECTION_MB box=upstream t=%0d req=%0d gen=%h tag=%h kind=%0d send=%0d rcv=%0d busy=%0d payload_gen=%h payload_tag=%h payload_kind=%0d ack=%0d ack_gen=%h ack_tag=%h ack_kind=%0d obsolete=%0d peer_req=%0d peer_gen=%h peer_tag=%h peer_kind=%0d peer_ack=%0d dest_ack=%0d dest_req=%0d",
                $time,dut.upstream_mailbox.req_valid,dut.upstream_mailbox.req_generation,dut.upstream_mailbox.req_tag,dut.upstream_mailbox.req_kind,dut.upstream_mailbox.send,dut.upstream_mailbox.received_ack,dut.upstream_mailbox.busy,dut.upstream_mailbox.payload[130:67],dut.upstream_mailbox.payload[66:3],dut.upstream_mailbox.payload[2:0],dut.upstream_mailbox.ack_valid,dut.upstream_mailbox.ack_generation,dut.upstream_mailbox.ack_tag,dut.upstream_mailbox.ack_kind,dut.upstream_mailbox.obsolete_completions,dut.upstream_mailbox.peer_req_valid,dut.upstream_mailbox.peer_req_generation,dut.upstream_mailbox.peer_req_tag,dut.upstream_mailbox.peer_req_kind,dut.upstream_mailbox.peer_ack_valid,dut.upstream_mailbox.dest_ack,dut.upstream_mailbox.dest_req);
            $display("AXI4_SECTION_MB box=host t=%0d req=%0d gen=%h tag=%h kind=%0d send=%0d rcv=%0d busy=%0d payload_gen=%h payload_tag=%h payload_kind=%0d ack=%0d ack_gen=%h ack_tag=%h ack_kind=%0d obsolete=%0d peer_req=%0d peer_gen=%h peer_tag=%h peer_kind=%0d peer_ack=%0d dest_ack=%0d dest_req=%0d",
                $time,dut.pair.host_mailbox.req_valid,dut.pair.host_mailbox.req_generation,dut.pair.host_mailbox.req_tag,dut.pair.host_mailbox.req_kind,dut.pair.host_mailbox.send,dut.pair.host_mailbox.received_ack,dut.pair.host_mailbox.busy,dut.pair.host_mailbox.payload[130:67],dut.pair.host_mailbox.payload[66:3],dut.pair.host_mailbox.payload[2:0],dut.pair.host_mailbox.ack_valid,dut.pair.host_mailbox.ack_generation,dut.pair.host_mailbox.ack_tag,dut.pair.host_mailbox.ack_kind,dut.pair.host_mailbox.obsolete_completions,dut.pair.host_mailbox.peer_req_valid,dut.pair.host_mailbox.peer_req_generation,dut.pair.host_mailbox.peer_req_tag,dut.pair.host_mailbox.peer_req_kind,dut.pair.host_mailbox.peer_ack_valid,dut.pair.host_mailbox.dest_ack,dut.pair.host_mailbox.dest_req);
            $display("AXI4_SECTION_MB box=device t=%0d req=%0d gen=%h tag=%h kind=%0d send=%0d rcv=%0d busy=%0d payload_gen=%h payload_tag=%h payload_kind=%0d ack=%0d ack_gen=%h ack_tag=%h ack_kind=%0d obsolete=%0d peer_req=%0d peer_gen=%h peer_tag=%h peer_kind=%0d peer_ack=%0d dest_ack=%0d dest_req=%0d",
                $time,dut.pair.device_mailbox.req_valid,dut.pair.device_mailbox.req_generation,dut.pair.device_mailbox.req_tag,dut.pair.device_mailbox.req_kind,dut.pair.device_mailbox.send,dut.pair.device_mailbox.received_ack,dut.pair.device_mailbox.busy,dut.pair.device_mailbox.payload[130:67],dut.pair.device_mailbox.payload[66:3],dut.pair.device_mailbox.payload[2:0],dut.pair.device_mailbox.ack_valid,dut.pair.device_mailbox.ack_generation,dut.pair.device_mailbox.ack_tag,dut.pair.device_mailbox.ack_kind,dut.pair.device_mailbox.obsolete_completions,dut.pair.device_mailbox.peer_req_valid,dut.pair.device_mailbox.peer_req_generation,dut.pair.device_mailbox.peer_req_tag,dut.pair.device_mailbox.peer_req_kind,dut.pair.device_mailbox.peer_ack_valid,dut.pair.device_mailbox.dest_ack,dut.pair.device_mailbox.dest_req);
            $display("AXI4_SECTION_MGMT t=%0d phase=%0d gen=%h tag=%h req=%0d ack=%0d uc=%0d pc=%0d hc=%0d dc=%0d ub=%0d hb=%0d db=%0d fm=%0d af=%0d hf=%0d df=%0d close=%0d fenced=%0d proof_req=%0d proof_ack=%0d proof_tag=%h drain_tag=%h state=%0d idle=%0d awleft=%0d wleft=%0d arleft=%0d ap=%0d wp=%0d arp=%0d partial=%0d hap=%0d hwp=%0d har=%0d dap=%0d dwp=%0d dar=%0d hrst=%0d drst=%0d cold=%0d sbv=%0d srv=%0d sbs=%0d srs=%0d daw=%0d dw=%0d dar_count=%0d dbs=%0d drs=%0d",
            $time,phase,generation,tag,req_valid,section_ack,upstream_certificate,pair_certificate,host_certificate,device_certificate,host_mailbox_busy,pair_host_busy,pair_device_busy,fault_mgmt,admission_fault_sources,host_fault_sources,device_fault_sources,close_host,host_source_fenced,host_proof_request,host_drain_ack,host_proof_tag,host_drain_tag,state_debug,local_idle,aw_left,w_left,ar_left,admission_aw_pending,admission_w_pending,admission_ar_pending,admission_w_partial,host_aw_pending,host_w_pending,host_ar_pending,device_aw_pending,device_w_pending,device_ar_pending,host_rst_n,device_rst_n,cold_rst_n,s_bvalid,s_rvalid,h_bcount,h_rcount,aw_count,w_count,ar_count,b_sent,r_sent);
            if(|fault_mgmt)must(!section_ack,"reported fault revokes section proof");
        end
    end
    initial begin
        for(integer id=0;id<16;id=id+1)begin r_index[id]=id;r_beat[id]=0;end
        $display("AXI4_SECTION_PROFILE device_period_ps=%0d host_period_ps=7000 mgmt_period_ps=10000 transactions=24 write_depth=32",DEVICE_PERIOD_PS);
        repeat(32)@(posedge mgmt_clk);#1000;cold_rst_n=1;
        repeat(64)@(posedge host_clk);@(negedge host_clk);host_rst_n=1;
        repeat(64)@(posedge device_clk);@(negedge device_clk);device_rst_n=1;
        repeat(16)@(posedge mgmt_clk);#1000;active=1;event_log("START");
        fork
            begin repeat(40)@(posedge host_clk);for(integer n=0;n<N;n=n+1)host_aw(n);end
            begin for(integer n=0;n<N-1;n=n+1)for(integer beat=0;beat<lens[n%7];beat=beat+1)host_w(n,beat);end
            begin repeat(25)@(posedge host_clk);for(integer n=0;n<N;n=n+1)host_ar(n);end
        join
        host_w(23,0);host_w(23,1);
        @(negedge mgmt_clk);req_valid=1;event_log("PARTIAL_GAP_REQUEST");
        wait(state_debug==1);#1;
        must(w_left==1 && aw_left==0 && admission_w_partial==2 && !s_wvalid,"finite partial-W gap prefix");
        event_log("FINITE_PREFIX");
        wait(host_mailbox_busy && host_proof_request);#1;
        @(posedge host_clk);#1;run_host=0;event_log("UPSTREAM_STOP");
        @(negedge mgmt_clk);req_valid=0;#1;event_log("UPSTREAM_CANCEL");
        repeat(8)@(posedge mgmt_clk);#1;
        must(host_mailbox_busy && dut.upstream_mailbox.send && !section_ack,"cancel retains busy upstream transfer");
        event_log("UPSTREAM_CANCEL_WAIT");
        @(negedge mgmt_clk);tag=tag+64'h100000001;req_valid=1;#1;event_log("UPSTREAM_REPLACE");
        repeat(8)@(posedge mgmt_clk);#1;
        must(host_mailbox_busy && !section_ack,"replacement waits for stopped host");event_log("UPSTREAM_REPLACE_WAIT");
        run_host=1;event_log("UPSTREAM_RESUME");
        host_w(23,2);host_w(23,3);
        wait(s_bvalid && s_rvalid && host_proof_request);#1;
        must(!section_ack,"held protected responses prevent proof");event_log("HELD_RESPONSES");
        @(posedge device_clk);#1;run_device=0;event_log("STOP_DEVICE");
        release_responses=1;
        repeat(80)@(posedge mgmt_clk);#1;
        must(!section_ack && fault_mgmt==0,"stopped device leaves proof pending");event_log("DEVICE_WAIT");
        run_device=1;event_log("RESUME_DEVICE");wait(section_ack);#1;
        must(h_bcount==N && h_rcount==1045 && local_idle,"all protected normal replies before certificate");event_log("PROOF_ONE");
        @(negedge host_clk);release_responses=0;phase=1;
        fork
            host_aw(24);
            begin for(integer beat=0;beat<lens[24%7];beat=beat+1)host_w(24,beat);end
            host_ar(24);
        join
        wait(s_bvalid && s_rvalid);#1;must(!local_idle && section_ack,"local replies excluded from section proof");
        event_log("LOCAL_HELD");
        // Both connected raw interfaces stay quiescent. Keep admission and
        // proof lifetimes outside these reset pins; never reinitialize them.
        fork
            begin @(posedge host_clk);#1;host_rst_n=0;event_log("HOST_RESET_ASSERT");end
            begin @(posedge device_clk);#1;device_rst_n=0;event_log("DEVICE_RESET_ASSERT");end
        join
        event_log("CDC_RESET_ASSERT");
        fork
            begin repeat(64)@(posedge host_clk);end
            begin repeat(64)@(posedge device_clk);end
        join
        #1;
        must(section_ack && s_bvalid && s_rvalid && !local_idle && state_debug==2 && fault_mgmt==0,"CDC reset preserves retained section and local replies");
        fork
            begin @(posedge host_clk);#1;host_rst_n=1;event_log("HOST_RESET_RELEASE");end
            begin @(posedge device_clk);#1;device_rst_n=1;event_log("DEVICE_RESET_RELEASE");end
        join
        event_log("CDC_RESET_RELEASE");
        repeat(64)@(posedge mgmt_clk);
        wait(!host_mailbox_busy && !pair_host_busy && !pair_device_busy);
        @(posedge host_clk);#1;run_host=0;tag=tag+64'h100000001;#1;
        must(!section_ack,"tag change immediately masks certificate");event_log("STOP_HOST_NEW_TAG");
        repeat(20)@(posedge mgmt_clk);#1;
        must(host_mailbox_busy && dut.upstream_mailbox.send,"new request launched while host stopped");
        @(negedge mgmt_clk);req_valid=0;#1;event_log("HOST_STOP_CANCEL");
        repeat(8)@(posedge mgmt_clk);#1;event_log("HOST_STOP_CANCEL_WAIT");
        @(negedge mgmt_clk);tag=tag+1;req_valid=1;#1;event_log("HOST_STOP_REPLACE");
        repeat(20)@(posedge mgmt_clk);#1;
        must(!section_ack && host_mailbox_busy,"fresh upstream proof requires host edge");event_log("HOST_WAIT");
        run_host=1;event_log("RESUME_HOST");
        wait(upstream_certificate);#1;
        // Freeze H before the new pair request can return a certificate.
        @(posedge host_clk);#1;run_host=0;event_log("PAIR_HOST_STOP");
        wait(pair_host_busy);#1;
        must(!host_certificate && !section_ack,"host pair transfer pending");event_log("PAIR_HOST_BUSY");
        @(posedge device_clk);#1;run_device=0;event_log("PAIR_DEVICE_STOP");
        @(negedge mgmt_clk);req_valid=0;#1;event_log("PAIR_HOST_CANCEL");
        repeat(8)@(posedge mgmt_clk);#1;
        must(pair_host_busy && dut.pair.host_mailbox.send && !section_ack,"cancel retains busy host pair transfer");
        event_log("PAIR_HOST_CANCEL_WAIT");
        @(negedge mgmt_clk);tag=tag+1;req_valid=1;#1;event_log("PAIR_HOST_REPLACE");
        repeat(8)@(posedge mgmt_clk);#1;event_log("PAIR_HOST_REPLACE_WAIT");
        run_host=1;event_log("PAIR_HOST_RESUME");
        wait(pair_device_busy && host_certificate && upstream_certificate);#1;
        must(!device_certificate && !section_ack,"device pair transfer pending");event_log("PAIR_DEVICE_BUSY");
        @(negedge mgmt_clk);req_valid=0;#1;event_log("PAIR_DEVICE_CANCEL");
        repeat(8)@(posedge mgmt_clk);#1;
        must(pair_device_busy && dut.pair.device_mailbox.send && !section_ack,"cancel retains busy device transfer");
        event_log("PAIR_DEVICE_CANCEL_WAIT");
        @(negedge mgmt_clk);tag=tag+1;req_valid=1;#1;event_log("PAIR_DEVICE_REPLACE");
        repeat(8)@(posedge mgmt_clk);#1;event_log("PAIR_DEVICE_REPLACE_WAIT");
        run_device=1;event_log("PAIR_DEVICE_RESUME");
        wait(section_ack);#1;event_log("PROOF_TWO");
        wait(!host_mailbox_busy && !pair_host_busy && !pair_device_busy);
        @(negedge mgmt_clk);phase=2;generation=generation+64'h100000001;#1;
        must(!section_ack,"new generation immediately masks certificate");event_log("NEW_GENERATION");
        wait(section_ack);#1;event_log("PROOF_THREE");
        wait(!host_mailbox_busy && !pair_host_busy && !pair_device_busy);
        @(negedge device_clk);phase=3;inject_orphan=1;event_log("INJECT_ORPHAN_R");
        wait(device_fault_sources==4);#1;run_device=0;event_log("STOP_DEVICE_FAULT");
        repeat(4)@(posedge mgmt_clk);#1;
        must(fault_mgmt[15] && !section_ack,"registered device fault crosses after device clock stops");event_log("DEVICE_FAULT_REVOKED");
        run_device=1;event_log("RESUME_DEVICE_FAULT");
        wait(admission_fault_sources==2);#1;run_host=0;event_log("STOP_HOST_FAULT");
        repeat(4)@(posedge mgmt_clk);#1;
        must(fault_mgmt[1] && !section_ack,"registered admission alarm crosses without additional host edge");event_log("HOST_FAULT_REVOKED");
        tag=tag+1;generation=generation+1;event_log("FAULT_FRESH_REQUEST");
        repeat(20)@(posedge mgmt_clk);#1;
        must(!section_ack && !host_mailbox_busy && !pair_host_busy && !pair_device_busy,"fault forbids fresh mailboxes");
        run_host=1;event_log("RESUME_HOST_FAULT");
        repeat(8)@(posedge host_clk);#1;
        must(s_bvalid && s_rvalid && h_bcount==24 && h_rcount==1045,"raw orphan never replaces retained local replies");
        @(negedge host_clk);release_responses=1;event_log("LOCAL_RELEASE");
        wait(h_bcount==25 && h_rcount==1061);
        repeat(12)@(posedge mgmt_clk);#1;
        must(orphan_sent && fault_mgmt==19'd33282 && !section_ack && !local_idle,"quarantine remains after local replies complete");
        must(!h_rready && h_rvalid && !s_rvalid && !s_bvalid,"orphan held at raw CDC output");
        must(dut.upstream_mailbox.obsolete_completions==2 && dut.pair.host_mailbox.obsolete_completions==1 && dut.pair.device_mailbox.obsolete_completions==1,"all four obsolete completions retired without certification");
        event_log("TERMINAL");active=0;
        $display("AXI4_SECTION_SUITE_PASS device_period_ps=%0d writes=24 reads=24 local_writes=1 local_reads=1 proofs=3 cdc_resets=1 cancellations=4 native_endpoint=0",DEVICE_PERIOD_PS);$finish;
    end
    initial begin #200000000;$fatal(1,"AXI4_SECTION_TIMEOUT");end
endmodule
