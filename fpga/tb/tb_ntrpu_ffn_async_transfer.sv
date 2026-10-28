`timescale 1ns/1ps
module tb_ntrpu_ffn_async_transfer;
`ifdef NTRPU_FFN_PROFILE
    ntrpu_ffn_profile profiler();
`endif
    logic host_clk=0,host_rst_n=0;
    int host_half_ps=2000,host_phase_ps=700,host_edges=0;
    bit host_clock_ready=0;
    initial begin wait(host_clock_ready);#(host_phase_ps/1000.0);forever #(host_half_ps/1000.0) host_clk=~host_clk;end
    always @(posedge host_clk) host_edges++;
    logic  clk;
    logic  rst_n;
    logic  job_valid;
    logic  job_ready;
    logic [1:0] job_layout;
    logic [13:0] job_dim;
    logic [13:0] job_hidden;
    logic [5:0] job_tile_k [0:2];
    logic [63:0] job_addr [0:7];
    logic [63:0] job_base [0:7];
    logic [63:0] job_bytes [0:7];
    logic [31:0] job_buffer_epoch [0:7];
    logic [31:0] job_sequence;
    logic [31:0] job_epoch;
    logic [31:0] expected_epoch;
    logic  cancel_valid;
    logic  cancel_ready;
    logic  done_valid;
    logic  done_ready;
    logic [3:0] done_status;
    logic [3:0] done_child_status;
    logic [2:0] done_stages;
    logic [1:0] done_phase;
    logic [16:0] done_bytes;
    logic [17:0] done_stage_bytes;
    logic [31:0] done_sequence;
    logic [31:0] done_epoch;
    logic [31:0] done_hls_status;
    logic  activation_selected;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize;
    logic [2:0] m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid;
    logic [3:0] m_axi_arcache;
    logic  m_axi_arlock;
    logic  m_axi_arvalid;
    logic  m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic  m_axi_rlast;
    logic  m_axi_rvalid;
    logic  m_axi_rready;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize;
    logic [2:0] m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid;
    logic [3:0] m_axi_awcache;
    logic  m_axi_awlock;
    logic  m_axi_awvalid;
    logic  m_axi_awready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic  m_axi_wlast;
    logic  m_axi_wvalid;
    logic  m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic  m_axi_bvalid;
    logic  m_axi_bready;

    logic[63:0] s_host_axi_araddr, s_host_axi_awaddr;
    logic[7:0] s_host_axi_arlen, s_host_axi_awlen;
    logic[2:0] s_host_axi_arsize, s_host_axi_arprot, s_host_axi_awsize, s_host_axi_awprot;
    logic[1:0] s_host_axi_arburst, s_host_axi_awburst;
    logic[3:0] s_host_axi_arid, s_host_axi_arcache, s_host_axi_awid, s_host_axi_awcache;
    logic s_host_axi_arlock, s_host_axi_arvalid, s_host_axi_awlock, s_host_axi_awvalid;
    logic s_host_axi_arready, s_host_axi_awready;
    logic[511:0] s_host_axi_rdata;
    logic[1:0] s_host_axi_rresp;
    logic[3:0] s_host_axi_rid;
    logic s_host_axi_rlast, s_host_axi_rvalid;
    logic s_host_axi_rready;
    logic[511:0] s_host_axi_wdata;
    logic[63:0] s_host_axi_wstrb;
    logic s_host_axi_wlast, s_host_axi_wvalid;
    logic s_host_axi_wready;
    logic[3:0] s_host_axi_bid;
    logic[1:0] s_host_axi_bresp;
    logic s_host_axi_bvalid;
    logic s_host_axi_bready;

    ntrpu_ffn_async_host_bfm #(.OUTPUT_FP32(1)) dut(.*);
    always #2.5 clk=~clk;
    localparam int LIMIT=262144;
    logic [7:0] memory[0:7][0:LIMIT-1],host_image[0:3][0:LIMIT-1],fixture_bytes[0:LIMIT-1];
    logic [7:0] golden[0:3][0:65535];
    bit written[0:7][0:LIMIT-1];
    logic [63:0] bases[0:7];
    int spans[0:7];
    string fixture_dir;
    int ident=0,seed=1,rng,host_rng,clock_count=0;
    int core_ar,core_r,core_aw,core_w,core_b,core_bytes,down_bytes;
    int host_aw,host_w,host_b,host_ar,host_r,host_load_bytes,host_result_bytes;
    int phase_cycles[0:3],core_cycles,total_cycles=0,total_bytes=0;
    int ar_stalls=0,aw_stalls=0,w_stalls=0,done_stalls=0,contention=0;
    bit computing=0,server_idle=1;
    wire [1:0] phase=dut.device.fabric.device.core.controller.phase;
    function automatic void need(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL FFN TRANSFER id=%0d clock=%0d: %s",ident,clock_count,why);
    endfunction
    function automatic int buffer_id(input logic [63:0] addr);
        for(int b=0;b<8;b++) if(addr>=bases[b]+4032 && addr<bases[b]+4032+64'(spans[b])) return b;
        need(0,"memory address outside registered spans");return -1;
    endfunction
    function automatic logic [511:0] word_at(input logic [63:0] addr);
        logic [511:0] value;
        int b,off;
        b=buffer_id(addr);off=int'(addr-bases[b]);
        need(off+64<=4032+spans[b],"read bounds");
        for(int lane=0;lane<64;lane++) begin
            need(written[b][off+lane],"read before host/core produced data");
            value[lane*8+:8]=memory[b][off+lane];
        end
        return value;
    endfunction
    task automatic tick;@(posedge clk);#1;endtask
    task automatic host_tick;@(posedge host_clk);#0.1;endtask
    // AXI memory peer. Only accepted bus writes can modify payload storage.
    initial begin : memory_peer
        bit reading,aw_seen,read_host,write_host,arhs,rhs,awhs,whs,bhs;
        logic [63:0] read_addr,write_addr;
        logic [3:0] read_id,write_id;
        logic [511:0] queue[0:3];
        logic [63:0] masks[0:3];bit lasts[0:3];
        int read_left,burst_len,received,stored,b_wait,b,off;
        reading=0;aw_seen=0;read_host=0;write_host=0;received=0;stored=0;b_wait=0;
        m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;
        m_axi_rvalid=0;m_axi_bvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;m_axi_bid=0;m_axi_bresp=0;
        forever begin
            @(negedge clk);
            if(!rst_n) begin
                reading=0;aw_seen=0;received=0;stored=0;b_wait=0;
                m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;m_axi_rvalid=0;m_axi_bvalid=0;server_idle=1;
            end else begin
                m_axi_arready=!reading && ($random(rng)&3)!=0;
                m_axi_awready=!aw_seen && ($random(rng)&3)!=0;
                m_axi_wready=received<4 && (!aw_seen || received<burst_len) && ($random(rng)&3)!=0;
                if(reading && !m_axi_rvalid && ($random(rng)&3)!=0) begin
                    m_axi_rvalid=1;m_axi_rdata=word_at(read_addr);m_axi_rid=read_id;
                    m_axi_rresp=0;m_axi_rlast=read_left==1;
                end
                if(aw_seen && received==burst_len && !m_axi_bvalid) begin
                    b_wait++;
                    if(b_wait>=3 && ($random(rng)&3)!=0) begin m_axi_bvalid=1;m_axi_bid=write_id;m_axi_bresp=0;end
                end
                @(posedge clk);
                arhs=m_axi_arvalid&&m_axi_arready;rhs=m_axi_rvalid&&m_axi_rready;
                awhs=m_axi_awvalid&&m_axi_awready;whs=m_axi_wvalid&&m_axi_wready;bhs=m_axi_bvalid&&m_axi_bready;
                if(arhs) begin
                    need(!reading && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arcache==2 &&
                         m_axi_arprot==0 && !m_axi_arlock && (m_axi_arid==0 || m_axi_arid==9),"shared AR contract");
                    need(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"shared AR page");
                    read_host=m_axi_arid==9;read_id=m_axi_arid;read_addr=m_axi_araddr;
                    read_left=int'(m_axi_arlen)+1;reading=1;
                    b=buffer_id(read_addr);
                    if(!read_host) begin
                        need(computing && (phase==0 ? b==0 || b==1:phase==1 ? b==0 || b==2:phase==2 ? b==4 || b==5:b==6 || b==3),"core read phase");
                        core_ar++;
                    end else host_ar++;
                    $display("NTRPU_TRANSFER_AR id=%0d host=%0d addr=%016h beats=%0d",ident,read_host,read_addr,read_left);
                end
                if(rhs) begin
                    need(reading && read_left>0,"unexpected memory R");
                    if(read_host) host_r++;else core_r++;
                    read_left--;read_addr+=64;if(read_left==0) reading=0;
                end
                if(awhs) begin
                    need(!aw_seen && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awcache==2 &&
                         m_axi_awprot==0 && !m_axi_awlock && (m_axi_awid==0 || m_axi_awid==9),"shared AW contract");
                    write_host=m_axi_awid==9;write_id=m_axi_awid;write_addr=m_axi_awaddr;
                    burst_len=int'(m_axi_awlen)+1;need(burst_len<=4 && int'(write_addr[11:0])+burst_len*64<=4096,"AW burst/page");
                    aw_seen=1;b=buffer_id(write_addr);
                    if(write_host) begin need(b<=3 && !computing,"host writes immutable inputs before start");host_aw++;end
                    else begin need(computing && b==int'(phase)+4,"core write phase");core_aw++;end
                    $display("NTRPU_TRANSFER_AW id=%0d host=%0d addr=%016h beats=%0d",ident,write_host,write_addr,burst_len);
                end
                if(whs) begin
                    need(received<4,"shared W queue overflow");
                    queue[received]=m_axi_wdata;masks[received]=m_axi_wstrb;lasts[received]=m_axi_wlast;received++;
                end
                if(aw_seen) while(stored<received) begin
                    b=buffer_id(write_addr);off=int'(write_addr-bases[b])+stored*64;
                    need(off+64<=4032+spans[b] && lasts[stored]==(stored==burst_len-1) && masks[stored]=='1,"shared W bounds/last/strobe");
                    for(int lane=0;lane<64;lane++) begin
                        need(!written[b][off+lane],"duplicate input/output write");
                        need(queue[stored][lane*8+:8]===(write_host ? host_image[b][off+lane]:golden[b-4][off+lane-4032]),"physical memory byte");
                        memory[b][off+lane]=queue[stored][lane*8+:8];written[b][off+lane]=1;
                    end
                    if(write_host) begin
                        host_w++;host_load_bytes+=64;
                        $display("NTRPU_TRANSFER_LOAD id=%0d buffer=%0d offset=%0d data=%0128h",ident,b,off-4032,queue[stored]);
                    end else begin
                        core_w++;core_bytes+=64;if(b==7) down_bytes+=64;
                        $display("NTRPU_FFN_MEMORY_W id=%0d buffer=%0d offset=%0d strb=%016h data=%0128h",ident,b,off-4032,masks[stored],queue[stored]);
                    end
                    stored++;
                end
                if(bhs) begin
                    need(aw_seen && received==burst_len && stored==received,"B drain");
                    if(write_host) host_b++;else core_b++;
                    aw_seen=0;received=0;stored=0;b_wait=0;
                end
                server_idle=!reading && !aw_seen && received==0 && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid;
                #1;if(rhs) m_axi_rvalid=0;if(bhs) m_axi_bvalid=0;
            end
        end
    end
    always @(posedge clk) begin
        clock_count++;
        need(clock_count<1500000,"integration time budget");
        if(computing) begin
            core_cycles++;phase_cycles[int'(phase)]++;
            ar_stalls+=int'(dut.device.fabric.s_axi_arvalid[0]&&!dut.device.fabric.s_axi_arready[0]);
            aw_stalls+=int'(dut.device.fabric.s_axi_awvalid[0]&&!dut.device.fabric.s_axi_awready[0]);
            w_stalls+=int'(dut.device.fabric.s_axi_wvalid[0]&&!dut.device.fabric.s_axi_wready[0]);
            if(dut.device.fabric.s_axi_arvalid==2'b11) contention++;
        end
        if(done_valid && !done_ready) done_stalls++;
    end
    task automatic host_aw_send(input logic [63:0] addr,input int beats,delay_cycles);
        repeat(delay_cycles) host_tick();
        @(negedge host_clk);s_host_axi_awaddr=addr;s_host_axi_awlen=8'(beats-1);s_host_axi_awvalid=1;
        do begin @(posedge host_clk);end while(!s_host_axi_awready);
        #1;@(negedge host_clk);s_host_axi_awvalid=0;s_host_axi_awaddr='hbadbad04;
    endtask
    task automatic host_w_send(input int b,off,beats,delay_cycles);
        repeat(delay_cycles) host_tick();
        for(int beat=0;beat<beats;beat++) begin
            @(negedge host_clk);s_host_axi_wvalid=1;s_host_axi_wlast=beat==beats-1;s_host_axi_wstrb='1;
            for(int lane=0;lane<64;lane++) s_host_axi_wdata[lane*8+:8]=host_image[b][4032+off+beat*64+lane];
            do begin @(posedge host_clk);end while(!s_host_axi_wready);
            $display("NTRPU_TRANSFER_HOST_W id=%0d buffer=%0d offset=%0d data=%0128h",ident,b,off+beat*64,s_host_axi_wdata);
            #1;@(negedge host_clk);s_host_axi_wvalid=0;s_host_axi_wdata='0;
        end
    endtask
    task automatic load_buffer(input int b);
        int off,beats,order;
        off=0;
        while(off<spans[b]) begin
            beats=(spans[b]-off)/64;if(beats>4) beats=4;
            if(beats>(4096-((4032+off)%4096))/64) beats=(4096-((4032+off)%4096))/64;
            order=(off/64)%3;
            fork
                host_aw_send(bases[b]+4032+64'(off),beats,order==2 ? 4:0);
                host_w_send(b,off,beats,order==1 ? 4:0);
            join
            while(!s_host_axi_bvalid) host_tick();
            repeat(3) begin host_tick();need(s_host_axi_bvalid && s_host_axi_bid==9 && s_host_axi_bresp==0,"host B held");end
            @(negedge host_clk);s_host_axi_bready=1;host_tick();
            @(negedge host_clk);s_host_axi_bready=0;
            $display("NTRPU_TRANSFER_HOST_B id=%0d buffer=%0d offset=%0d beats=%0d",ident,b,off,beats);
            off+=beats*64;
        end
    endtask
    task automatic host_read(input int b,off,beats);
        logic [511:0] value;
        @(negedge host_clk);s_host_axi_araddr=bases[b]+4032+64'(off);s_host_axi_arlen=8'(beats-1);s_host_axi_arvalid=1;
        do begin @(posedge host_clk);end while(!s_host_axi_arready);
        #1;@(negedge host_clk);s_host_axi_arvalid=0;s_host_axi_araddr='hbadbad05;
        for(int beat=0;beat<beats;beat++) begin
            while(!s_host_axi_rvalid) host_tick();
            value=s_host_axi_rdata;
            repeat(2) begin
                host_tick();need(s_host_axi_rvalid && s_host_axi_rdata===value && s_host_axi_rid==9 && s_host_axi_rresp==0 &&
                    s_host_axi_rlast===(beat==beats-1),"host R stability/identity");
            end
            for(int lane=0;lane<64;lane++)
                need(value[lane*8+:8]===(b==7 ? golden[3][off+beat*64+lane]:host_image[b][4032+off+beat*64+lane]),"host readback byte");
            $display("NTRPU_TRANSFER_HOST_R id=%0d buffer=%0d offset=%0d final=%0d data=%0128h",ident,b,off+beat*64,b==7,value);
            if(b==7) host_result_bytes+=64;
            @(negedge host_clk);s_host_axi_rready=1;host_tick();
            @(negedge host_clk);s_host_axi_rready=0;
        end
    endtask
    task automatic prepare;
        need(server_idle && job_ready,"previous run drained");
        core_ar=0;core_r=0;core_aw=0;core_w=0;core_b=0;core_bytes=0;down_bytes=0;core_cycles=0;
        host_ar=0;host_r=0;host_aw=0;host_w=0;host_b=0;host_load_bytes=0;host_result_bytes=0;
        for(int p=0;p<4;p++) phase_cycles[p]=0;
        for(int b=0;b<8;b++) begin
            bases[b]=64'h100000000+64'(b)*64'h20000000;
            for(int i=0;i<LIMIT;i++) begin memory[b][i]=8'ha5;written[b][i]=0;end
        end
        spans[0]=512;spans[1]=195840;spans[2]=195840;spans[3]=200192;
        spans[4]=5760;spans[5]=5760;spans[6]=1472;spans[7]=1024;
        for(int b=0;b<4;b++) begin
            if(b==0) $readmemh($sformatf("%s/job-%03d-input.hex",fixture_dir,ident),fixture_bytes,0,4032+spans[b]-1);
            else $readmemh($sformatf("%s/weight-%0d.hex",fixture_dir,b),fixture_bytes,0,4032+spans[b]-1);
            for(int i=0;i<4032+spans[b];i++) host_image[b][i]=fixture_bytes[i];
        end
        for(int b=4;b<8;b++) begin
            if(b==7) $readmemh($sformatf("%s/job-%03d-final-fp32.hex",fixture_dir,ident),fixture_bytes,0,spans[b]-1);
            else $readmemh($sformatf("%s/job-%03d-buffer-%0d.hex",fixture_dir,ident,b),fixture_bytes,0,spans[b]-1);
            for(int i=0;i<spans[b];i++) golden[b-4][i]=fixture_bytes[i];
        end
    endtask
    initial begin
        int load_start,load_end,case_start,compute_end,read_end,first_contention,off,beats;
        int edge_start,edge_load,edge_compute,edge_read;
        longint ps_start,ps_load,ps_compute,ps_read;
        clk=0;rst_n=0;job_valid=0;cancel_valid=0;done_ready=0;
        s_host_axi_araddr=0;s_host_axi_awaddr=0;s_host_axi_arlen=0;s_host_axi_awlen=0;
        s_host_axi_arsize=6;s_host_axi_awsize=6;s_host_axi_arprot=0;s_host_axi_awprot=0;
        s_host_axi_arburst=1;s_host_axi_awburst=1;s_host_axi_arid=9;s_host_axi_awid=9;
        s_host_axi_arcache=2;s_host_axi_awcache=2;s_host_axi_arlock=0;s_host_axi_awlock=0;
        s_host_axi_arvalid=0;s_host_axi_awvalid=0;s_host_axi_wvalid=0;s_host_axi_rready=0;s_host_axi_bready=0;
        s_host_axi_wdata=0;s_host_axi_wstrb=0;s_host_axi_wlast=0;
        if($value$plusargs("SEED=%d",seed)) begin end
        rng=seed;host_rng=seed+1;
        need($value$plusargs("FIXTURE=%s",fixture_dir),"fixture path");
        $display("NTRPU_FFN_MEMORY_SEED seed=%0d",seed);
        $display("NTRPU_FFN_MEMORY_FORMAT fp32=1");
        if($value$plusargs("HOST_HALF_PS=%d",host_half_ps)) begin end
        need(host_half_ps>=1700 && host_half_ps<=3100,"host period range");
        host_clock_ready=1;
        fork
            begin repeat(64) @(negedge clk);rst_n=1;end
            begin repeat(64) @(negedge host_clk);host_rst_n=1;end
        join
        repeat(32) tick();
        $display("NTRPU_ASYNC_CONFIG host_period_ps=%0d host_phase_ps=%0d device_period_ps=5000 reset_edges=64 time_ps=%0d",2*host_half_ps,host_phase_ps,longint'($realtime*1000));
        for(int sample=1;sample<=3;sample++) begin
            ident=sample;prepare();load_start=clock_count;edge_start=host_edges;ps_start=longint'($realtime*1000);
            for(int b=0;b<4;b++) load_buffer(b);
            load_end=clock_count;edge_load=host_edges;ps_load=longint'($realtime*1000);
            need(host_load_bytes==592384 && host_aw==host_b,"host load byte/drain count");
            first_contention=contention;case_start=clock_count;
            @(negedge clk);job_valid=1;job_dim=256;job_hidden=720;job_layout=2;
            job_sequence='hfffffffe;job_epoch=32'(ident+600);expected_epoch=job_epoch;
            job_tile_k[0]=4;job_tile_k[1]=4;job_tile_k[2]=8;
            for(int b=0;b<8;b++) begin job_addr[b]=bases[b]+4032;job_base[b]=bases[b];job_bytes[b]=64'(4032+spans[b]);job_buffer_epoch[b]=job_epoch;end
            do begin @(posedge clk);end while(!job_ready);
            #1;job_valid=0;computing=1;job_dim=0;job_hidden=0;job_layout=0;job_sequence=0;job_epoch=0;expected_epoch=0;
            for(int b=0;b<8;b++) begin job_addr[b]=0;job_base[b]=0;job_bytes[b]=0;job_buffer_epoch[b]=0;end
            for(int p=0;p<3;p++) job_tile_k[p]=0;
            // Read immutable weights while the FFN competes for the same memory.
            wait(dut.device.fabric.s_axi_arvalid[0]);
            for(int k=0;k<64;k++) begin
                off=64*((k*7)%256);beats=4;
                if(beats>(4096-((4032+off)%4096))/64) beats=(4096-((4032+off)%4096))/64;
                host_read(1,off,beats);
            end
            while(!done_valid) tick();
            compute_end=clock_count;computing=0;edge_compute=host_edges;ps_compute=longint'($realtime*1000);
            need(done_status==0 && done_child_status==0 && done_stages==4 && done_phase==3 && done_bytes==1024 &&
                done_stage_bytes==14016 && done_sequence=='hfffffffe && done_epoch==32'(ident+600) && done_hls_status==0,"MMIO completion");
            need(core_aw==core_b && core_bytes==14016 && down_bytes==1024 && server_idle,"complete core memory drain");
            need(contention>first_contention,"host/core read contention exercised");
            $display("NTRPU_TRANSFER_COMPLETE id=%0d clock=%0d",ident,clock_count);
            for(int off=0;off<1024;) begin
                beats=4;if(beats>(1024-off)/64) beats=(1024-off)/64;
                if(beats>(4096-((4032+off)%4096))/64) beats=(4096-((4032+off)%4096))/64;
                host_read(7,off,beats);off+=beats*64;
            end
            read_end=clock_count;edge_read=host_edges;ps_read=longint'($realtime*1000);need(host_result_bytes==1024,"host result byte count");
            @(negedge clk);done_ready=1;tick();@(negedge clk);done_ready=0;
            while(!job_ready) tick();
            for(int b=0;b<8;b++) for(int i=0;i<LIMIT;i++)
                need(written[b][i]===(i>=4032 && i<4032+spans[b]) &&
                    (written[b][i] || memory[b][i]===8'ha5),"payload coverage and guards");
            $display("NTRPU_FFN_MEMORY_CASE id=%0d dim=256 hidden=720 mode=0 target=0 reset=0 cycles=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d bytes=%0d down_bytes=%0d p0=%0d p1=%0d p2=%0d p3=%0d",
                ident,core_cycles,core_ar,core_r,core_aw,core_w,core_b,core_bytes,down_bytes,phase_cycles[0],phase_cycles[1],phase_cycles[2],phase_cycles[3]);
            $display("NTRPU_TRANSFER_CASE id=%0d load_cycles=%0d control_compute_cycles=%0d result_cycles=%0d host_ar=%0d host_r=%0d host_aw=%0d host_w=%0d host_b=%0d load_bytes=%0d result_bytes=%0d contention=%0d",
                ident,load_end-load_start,compute_end-case_start,read_end-compute_end,host_ar,host_r,host_aw,host_w,host_b,host_load_bytes,host_result_bytes,contention-first_contention);
            $display("NTRPU_ASYNC_METRICS id=%0d host_load_edges=%0d host_compute_edges=%0d host_read_edges=%0d load_ps=%0d compute_ps=%0d read_ps=%0d start_ps=%0d end_ps=%0d",ident,edge_load-edge_start,edge_compute-edge_load,edge_read-edge_compute,ps_load-ps_start,ps_compute-ps_load,ps_read-ps_compute,ps_start,ps_read);
            total_cycles+=core_cycles;total_bytes+=core_bytes;
        end
        $display("NTRPU_FFN_MEMORY_PASS cases=3 cycles=%0d bytes=%0d resets=0 cancels=0 ar_stalls=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d",
            total_cycles,total_bytes,ar_stalls,aw_stalls,w_stalls,done_stalls);
        $display("NTRPU_FFN_TRANSFER_PASS cases=3 load_bytes=1777152 result_bytes=3072");
        $finish;
    end
endmodule

