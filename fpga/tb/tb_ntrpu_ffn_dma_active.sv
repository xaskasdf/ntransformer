`timescale 1ns/1ps
// Real FFN and DMA read overlap. Legal AXI B backpressure holds the selected
// compute phase active; clock loss resets both, then a fresh full FFN must pass.
module tb_ntrpu_ffn_dma_active;
    logic device_osc=0,host_osc=0,mgmt_clk=0,external_reset_async=1,rearm=1;
    bit device_enabled=1,host_enabled=1,host_clock_ready=0;
    wire clk=device_osc && device_enabled,host_clk=host_osc && host_enabled;
    wire rst_n,host_rst_n,device_admit,host_admit,running;
    wire peers_reset_req,rearm_ready,peers_held,peers_ready,host_dma_held,host_dma_ready;
    logic memory_peer_held=0,memory_peer_ready=0;
    integer host_half_ps=2000,quiet_edges=0,up_edges=0,fault=0;
    longint dt=0,ht=0,mt=0;
    always #2.5 device_osc=~device_osc;
    always #5 mgmt_clk=~mgmt_clk;
    initial begin wait(host_clock_ready);#0.7;forever #(host_half_ps/1000.0) host_osc=~host_osc;end
    always @(posedge clk) dt++;
    always @(posedge host_clk) ht++;
    always @(posedge mgmt_clk) mt++;
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

    ntrpu_ffn_supervised_host_bfm #(.OUTPUT_FP32(1)) dut(.*);
    ntrpu_axi_dma_bfm dma(.clk(host_clk),.rst_n(host_rst_n),.held(host_dma_held),.ready(host_dma_ready),.*);
    localparam int LIMIT=16384;
    logic [7:0] memory[0:7][0:LIMIT-1],host_image[0:3][0:LIMIT-1],fixture_bytes[0:LIMIT-1];
    logic [7:0] golden[0:3][0:65535];
    bit written[0:7][0:LIMIT-1];
    logic [63:0] bases[0:7];
    int spans[0:7];
    integer d=32,h=32;
    localparam bit OUTPUT_FP32=1;
    localparam integer DOWN_BYTES=4;
    bit pause_host_reads=0,pause_core_b=0,target_committed=0;
    integer target_phase=0;
    int ident=0,seed=1,rng,host_rng,clock_count=0;
    int core_ar,core_r,core_aw,core_w,core_b,core_bytes,down_bytes;
    int host_aw,host_w,host_b,host_ar,host_r,host_load_bytes,host_result_bytes;
    int phase_cycles[0:3],core_cycles,total_cycles=0,total_bytes=0;
    int ar_stalls=0,aw_stalls=0,w_stalls=0,done_stalls=0,contention=0;
    bit computing=0,server_idle=1;
    wire [1:0] phase=dut.device.fabric.fabric.device.core.controller.phase;
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
            @(negedge clk or negedge rst_n);
            if(!rst_n) begin
                reading=0;aw_seen=0;received=0;stored=0;b_wait=0;
                m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;m_axi_rvalid=0;m_axi_bvalid=0;server_idle=1;
            end else begin
                m_axi_arready=!reading && ($random(rng)&3)!=0;
                m_axi_awready=!aw_seen && ($random(rng)&3)!=0;
                m_axi_wready=received<4 && (!aw_seen || received<burst_len) && ($random(rng)&3)!=0;
                if(reading && !(read_host && pause_host_reads) && !m_axi_rvalid && ($random(rng)&3)!=0) begin
                    m_axi_rvalid=1;m_axi_rdata=word_at(read_addr);m_axi_rid=read_id;
                    m_axi_rresp=0;m_axi_rlast=read_left==1;
                end
                if(aw_seen && received==burst_len && !m_axi_bvalid) begin
                    b_wait++;
                    if(b_wait>=3 && !(pause_core_b && !write_host && int'(phase)==target_phase) && ($random(rng)&3)!=0) begin m_axi_bvalid=1;m_axi_bid=write_id;m_axi_bresp=0;end
                end
                @(posedge clk or negedge rst_n);
                if(!rst_n) begin
                    reading=0;aw_seen=0;received=0;stored=0;b_wait=0;
                    m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;m_axi_rvalid=0;m_axi_bvalid=0;server_idle=1;
                    continue;
                end
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
                        need(write_host || !written[b][off+lane],"duplicate core output write");
                        need(queue[stored][lane*8+:8]===(write_host ? host_image[b][off+lane]:golden[b-4][off+lane-4032]),"physical memory byte");
                        memory[b][off+lane]=queue[stored][lane*8+:8];written[b][off+lane]=1;
                    end
                    if(write_host) begin
                        host_w++;host_load_bytes+=64;
                        $display("NTRPU_TRANSFER_LOAD id=%0d buffer=%0d offset=%0d data=%0128h",ident,b,off-4032,queue[stored]);
                    end else begin
                        core_w++;core_bytes+=64;if(b==7) down_bytes+=64;
                        if(pause_core_b && int'(phase)==target_phase) target_committed=1;
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
            ar_stalls+=int'(dut.device.fabric.fabric.s_axi_arvalid[0]&&!dut.device.fabric.fabric.s_axi_arready[0]);
            aw_stalls+=int'(dut.device.fabric.fabric.s_axi_awvalid[0]&&!dut.device.fabric.fabric.s_axi_awready[0]);
            w_stalls+=int'(dut.device.fabric.fabric.s_axi_wvalid[0]&&!dut.device.fabric.fabric.s_axi_wready[0]);
            if(dut.device.fabric.fabric.s_axi_arvalid==2'b11) contention++;
        end
        if(done_valid && !done_ready) done_stalls++;
    end
    function automatic int q(input int index);
        if(index%32==0) return 127;
        if(index%32==1) return 1;
        return (index%32)*7-113;
    endfunction
    function automatic logic [7:0] expected_byte(input int b,input int off);
        int row,lane,k,source;
        logic [63:0] bits;
        if(b==6) begin
            k=off/64;lane=off%64;
            if(lane<32) return k*32+lane<h ? 8'(q(k*32+lane)):8'd0;
            bits=$realtobits(real'(257+2*k));
            if(lane<40) return bits[(lane-32)*8+:8];
            return 0;
        end
        row=off/((b==7) ? DOWN_BYTES:8);lane=off%((b==7) ? DOWN_BYTES:8);
        if(b==4) bits=$realtobits(real'(257+2*(row/32)));
        else if(b==5) bits=$realtobits(real'(q(row)));
        else begin
            source=(row*7)%h;
            if(OUTPUT_FP32) bits={32'b0,$shortrealtobits(shortreal'((257+2*(source/32))*q(source)))};
            else bits=$realtobits(real'((257+2*(source/32))*q(source)));
        end
        return bits[lane*8+:8];
    endfunction
    always @(posedge clk) begin
        if(!rst_n && server_idle && !m_axi_arready && !m_axi_awready && !m_axi_wready &&
           !m_axi_rvalid && !m_axi_bvalid) begin
            if(quiet_edges==7) memory_peer_held<=1;else quiet_edges<=quiet_edges+1;
        end else begin quiet_edges<=0;memory_peer_held<=0;end
        if(rst_n) begin
            if(up_edges==7) memory_peer_ready<=1;else up_edges<=up_edges+1;
        end else begin up_edges<=0;memory_peer_ready<=0;end
    end
    task automatic trace(input string kind);
        $display("NTRPU_DMA_REC_%s id=%0d fault=%0d ps=%0d dt=%0d ht=%0d mt=%0d den=%0d hen=%0d ext=%0d rst=%0d hrst=%0d da=%0d ha=%0d run=%0d dh=%0d dr=%0d mh=%0d mr=%0d",
            kind,ident,fault,longint'($realtime*1000),dt,ht,mt,device_enabled,host_enabled,external_reset_async,
            rst_n,host_rst_n,device_admit,host_admit,running,host_dma_held,host_dma_ready,memory_peer_held,memory_peer_ready);
    endtask
    always @(rst_n or host_rst_n or host_admit or device_admit or running)
        begin #0.001;trace("OUT");end
    always @(posedge mgmt_clk) begin #0.001;trace("M");end
    task automatic await_run();
        integer watchdog;
        watchdog=0;
        while(!(running && device_admit && host_admit && job_ready)) begin
            @(negedge mgmt_clk);watchdog++;
            need(watchdog<400,"recovery timeout");
        end
        repeat(8) @(negedge host_clk);
        need(!dma.busy && !s_host_axi_bvalid && !s_host_axi_rvalid && !dut.raw_irq && !done_valid &&
             !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && server_idle,"stale traffic after recovery");
    endtask
    task automatic clear_metrics();
        need(server_idle && !dma.busy && job_ready,"clear metrics only when drained");
        core_ar=0;core_r=0;core_aw=0;core_w=0;core_b=0;core_bytes=0;down_bytes=0;core_cycles=0;
        host_ar=0;host_r=0;host_aw=0;host_w=0;host_b=0;host_load_bytes=0;host_result_bytes=0;
        for(integer p=0;p<4;p++) phase_cycles[p]=0;
        // Payload RAM is retained across reset. Only the coverage of writes
        // for the next FFN is cleared, so old intermediates cannot satisfy it.
        for(integer b=4;b<8;b++) for(integer i=0;i<LIMIT;i++) written[b][i]=0;
    endtask
    task automatic images();
        logic[63:0] one;
        integer off,source;
        one=$realtobits(1.0);
        for(integer lane=0;lane<64;lane++) host_image[0][4032+lane]=0;
        for(integer lane=0;lane<32;lane++) host_image[0][4032+lane]=8'(q(lane));
        for(integer lane=0;lane<8;lane++) host_image[0][4064+lane]=one[lane*8+:8];
        for(integer b=1;b<=3;b++) begin
            off=4032;
            for(integer row=0;row<32;row++) begin
                host_image[b][off]=b==1 ? 8'h04:8'h00;
                host_image[b][off+1]=b==1 ? 8'h5c:8'h3c;
                source=b==1 ? 1:b==2 ? row:(row*7)%32;
                for(integer lane=0;lane<32;lane++) host_image[b][off+2+lane]=lane==source ? 1:0;
                off+=34;
            end
            need(off==4032+spans[b],"synthetic packed weight length");
        end
        for(integer b=4;b<8;b++) for(integer off=0;off<spans[b];off++) golden[b-4][off]=expected_byte(b,off);
    endtask
    task automatic load_inputs();
        logic[511:0] words[0:3];
        integer beats,off;
        bit aborted;
        for(integer b=0;b<4;b++) begin
            off=0;
            while(off<spans[b]) begin
                beats=(spans[b]-off)/64;if(beats>4) beats=4;
                if(beats>(4096-((4032+off)%4096))/64) beats=(4096-((4032+off)%4096))/64;
                for(integer beat=0;beat<beats;beat++) for(integer lane=0;lane<64;lane++)
                    words[beat][lane*8+:8]=host_image[b][4032+off+beat*64+lane];
                dma.write(bases[b]+4032+64'(off),beats,words,(off/64)%3==2 ? 4:0,(off/64)%3==1 ? 4:0,3,aborted);
                need(!aborted,"normal load aborted");off+=beats*64;
            end
        end
        need(server_idle && host_aw==host_b,"input loading did not drain");
    endtask
    function automatic logic[63:0] memory_hash();
        logic[63:0] value;
        value=64'hcbf29ce484222325;
        for(integer b=0;b<8;b++) for(integer i=0;i<LIMIT;i++) value=(value^64'(memory[b][i]))*64'h100000001b3;
        return value;
    endfunction
    task automatic run_ffn(input bit start_only=0);
        logic[511:0] words[0:3];
        bit aborted;
        integer watchdog;
        need(host_load_bytes==3328,"all synthetic inputs must arrive through host AXI");
        $display("NTRPU_DMA_REC_FFN_BEGIN id=%0d ps=%0d",ident,longint'($realtime*1000));
        @(negedge clk);job_valid=1;job_layout=2;job_dim=32;job_hidden=32;
        job_sequence='hfffffffe;job_epoch=32'(ident+600);expected_epoch=job_epoch;
        for(integer p=0;p<3;p++) job_tile_k[p]=2;
        for(integer b=0;b<8;b++) begin
            job_addr[b]=bases[b]+4032;job_base[b]=bases[b];job_bytes[b]=64'(4032+spans[b]);job_buffer_epoch[b]=job_epoch;
        end
        do begin @(posedge clk);end while(!job_ready);
        #1;job_valid=0;computing=1;job_dim=0;job_hidden=0;job_layout=0;job_sequence=0;job_epoch=0;expected_epoch=0;
        for(integer b=0;b<8;b++) begin job_addr[b]=0;job_base[b]=0;job_bytes[b]=0;job_buffer_epoch[b]=0;end
        for(integer p=0;p<3;p++) job_tile_k[p]=0;
        if(start_only) return;
        watchdog=0;
        while(!done_valid) begin tick();watchdog++;need(watchdog<30000,"FFN timeout");end
        computing=0;
        need(done_status==0 && done_child_status==0 && done_stages==4 && done_phase==3 &&
             done_bytes==128 && done_stage_bytes==704 && done_sequence=='hfffffffe &&
             done_epoch==32'(ident+600) && done_hls_status==0,"FFN completion identity/accounting");
        need(core_aw==core_b && core_bytes==704 && down_bytes==128 && server_idle,"FFN memory drain");
        for(integer off=0;off<128;off+=64) begin
            dma.read(bases[7]+4032+64'(off),1,2,words,aborted);need(!aborted,"result read aborted");
            for(integer lane=0;lane<64;lane++) need(words[0][lane*8+:8]===golden[3][off+lane],"result byte");
            host_result_bytes+=64;
        end
        @(negedge clk);done_ready=1;tick();@(negedge clk);done_ready=0;
        watchdog=0;while(!job_ready) begin tick();watchdog++;need(watchdog<5000,"FFN software retirement");end
        for(integer b=0;b<8;b++) for(integer i=0;i<LIMIT;i++)
            need(written[b][i]===(i>=4032 && i<4032+spans[b]) &&
                 (written[b][i] || memory[b][i]===8'ha5),"retained RAM coverage/guards");
        $display("NTRPU_FFN_MEMORY_CASE id=%0d dim=32 hidden=32 mode=0 target=0 reset=0 cycles=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d bytes=%0d down_bytes=%0d p0=%0d p1=%0d p2=%0d p3=%0d",
            ident,core_cycles,core_ar,core_r,core_aw,core_w,core_b,core_bytes,down_bytes,phase_cycles[0],phase_cycles[1],phase_cycles[2],phase_cycles[3]);
        $display("NTRPU_DMA_REC_FFN_END id=%0d ps=%0d load_bytes=%0d result_bytes=%0d",ident,longint'($realtime*1000),host_load_bytes,host_result_bytes);
        total_cycles+=core_cycles;total_bytes+=core_bytes;
    endtask
    initial begin : scenarios
        logic[511:0] words[0:3],readback[0:3];
        logic[63:0] retained_hash;
        bit aborted,stop_host;
        integer kind,watchdog,aw_before,w_before,b_before,ar_before,r_before;
        job_valid=0;cancel_valid=0;done_ready=0;job_layout=0;job_dim=0;job_hidden=0;
        job_sequence=0;job_epoch=0;expected_epoch=0;
        for(integer p=0;p<3;p++) job_tile_k[p]=0;
        for(integer b=0;b<8;b++) begin
            bases[b]=64'h100000000+64'(b)*64'h20000000;
            job_addr[b]=0;job_base[b]=0;job_bytes[b]=0;job_buffer_epoch[b]=0;
            for(integer i=0;i<LIMIT;i++) begin memory[b][i]=8'ha5;written[b][i]=0;end
        end
        spans[0]=64;spans[1]=1088;spans[2]=1088;spans[3]=1088;
        spans[4]=256;spans[5]=256;spans[6]=64;spans[7]=128;
        if($value$plusargs("SEED=%d",seed)) begin end
        if($value$plusargs("HOST_HALF_PS=%d",host_half_ps)) begin end
        need(host_half_ps==2000 || host_half_ps==3100,"host period");
        rng=seed;host_rng=seed+1;host_clock_ready=1;
        $display("NTRPU_FFN_MEMORY_SEED seed=%0d",seed);
        $display("NTRPU_FFN_MEMORY_FORMAT fp32=1");
        $display("NTRPU_DMA_ACTIVE_CONFIG host_period_ps=%0d device_period_ps=5000 mgmt_period_ps=10000 dim=32 hidden=32",host_half_ps*2);
        repeat(64) @(negedge mgmt_clk);external_reset_async=0;await_run();
        for(integer iteration=1;iteration<=16;iteration++) begin
            ident=2*iteration-1;fault=iteration;target_phase=(iteration-1)%4;
            kind=(iteration-1)/8;stop_host=((iteration-1)/4)%2==0;
            pause_host_reads=0;pause_core_b=1;target_committed=0;
            images();clear_metrics();load_inputs();run_ffn(1);
            watchdog=0;
            while(!target_committed) begin tick();watchdog++;need(watchdog<30000,"target phase first physical write");end
            need(computing && !done_valid && int'(phase)==target_phase,"concurrent compute pending");
            pause_host_reads=kind==0;
            $display("NTRPU_DMA_ACTIVE_FAULT id=%0d fault=%0d target=%0d kind=%0d stop_host=%0d ps=%0d core_bytes=%0d",ident,fault,target_phase,kind,stop_host,longint'($realtime*1000),core_bytes);
            fork
                begin dma.read(bases[1]+4032,1,kind==1 ? 10000:2,readback,aborted);end
                begin
                    wait(dma.phase==(kind==0 ? 4:5));
                    need(computing && !done_valid && int'(phase)==target_phase && dma.busy,"both transactions active at fault");
                    if(kind==0) need(!s_host_axi_rvalid,"AR-pending read has no response");
                    else need(s_host_axi_rvalid && !s_host_axi_rready,"read response retained");
                    if(stop_host) begin @(negedge host_osc);#0.001;host_enabled=0;end
                    else begin @(negedge device_osc);#0.001;device_enabled=0;end
                    trace("STOP");watchdog=0;
                    while(rst_n || host_rst_n) begin @(negedge mgmt_clk);watchdog++;need(watchdog<300,"automatic concurrent reset deadline");end
                    #0.001;computing=0;
                    need(!external_reset_async && !host_admit && !device_admit && !running && server_idle && !done_valid,"both owners aborted");
                    trace("BLOCKED");retained_hash=memory_hash();
                    $display("NTRPU_DMA_ACTIVE_ABORT id=%0d fault=%0d target=%0d bytes=%0d down_bytes=%0d cycles=%0d ps=%0d",ident,fault,target_phase,core_bytes,down_bytes,core_cycles,longint'($realtime*1000));
                    repeat(100) @(negedge mgmt_clk);
                    if(stop_host) begin @(negedge host_osc);#0.001;host_enabled=1;end
                    else begin @(negedge device_osc);#0.001;device_enabled=1;end
                    trace("RESUME");await_run();
                    need(memory_hash()===retained_hash,"payload RAM changed during concurrent recovery");
                    $display("NTRPU_DMA_REC_RETAIN id=%0d hash=%016h",ident,retained_hash);
                    trace("RECOVERED");
                end
            join
            need(aborted && dma.aborts==iteration,"one DMA abort per concurrent fault");
            need(dut.host_cases==ident && !dut.busy && !done_valid,"aborted control owner released");
            pause_host_reads=0;pause_core_b=0;ident++;
            images();clear_metrics();load_inputs();run_ffn();
            $display("NTRPU_DMA_ACTIVE_RECOVERY id=%0d fault=%0d target=%0d ps=%0d",ident,fault,target_phase,longint'($realtime*1000));
        end
        $display("NTRPU_FFN_MEMORY_PASS cases=32 cycles=%0d bytes=%0d resets=16 cancels=0 ar_stalls=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d",total_cycles,total_bytes,ar_stalls,aw_stalls,w_stalls,done_stalls);
        $display("NTRPU_DMA_ACTIVE_PASS faults=16 host=8 device=8 aborted_ffns=16 ffns=16 fp32=512 aborts=%0d",dma.aborts);
        $finish;
    end
    initial begin #5000000;$fatal(1,"FAIL DMA recovery global watchdog");end
endmodule
