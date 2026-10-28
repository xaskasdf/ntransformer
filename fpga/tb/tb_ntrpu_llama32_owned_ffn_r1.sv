`timescale 1ns/1ps
module tb_ntrpu_llama32_owned_ffn_r1;
`ifdef NTRPU_FFN_ASYNC_TEST
    logic host_clk=0,host_rst_n=0;
    int host_half_ps=2000;
    bit host_clock_ready=0;
    initial begin wait(host_clock_ready);#0.7;forever #(host_half_ps/1000.0) host_clk=~host_clk;end
    task automatic reset_domains();
        @(negedge clk);rst_n=0;host_rst_n=0;
        $display("NTRPU_FFN_IRQ_RESET_ASSERT time_ps=%0d",longint'($realtime*1000));
        fork
            begin repeat(64) @(negedge clk);rst_n=1;end
            begin repeat(64) @(negedge host_clk);host_rst_n=1;end
        join
        repeat(32) @(negedge clk);
        $display("NTRPU_FFN_IRQ_RESET_READY time_ps=%0d",longint'($realtime*1000));
    endtask
`endif
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

    always #2.5 clk=~clk;
`ifdef NTRPU_FFN_FP32
    localparam bit OUTPUT_FP32=1;
`else
    localparam bit OUTPUT_FP32=0;
`endif
    localparam int DOWN_BYTES=OUTPUT_FP32 ? 4:8;
`ifdef NTRPU_FFN_ASYNC_TEST
    ntrpu_ffn_async_host_bfm #(.WATCHDOG_CYCLES(65536),.OUTPUT_FP32(OUTPUT_FP32)) dut(.*);
    wire [1:0] observed_phase=dut.device.fabric.device.core.controller.phase;
    localparam int CANCEL_RESPONSE_DELAY=512;
`elsif NTRPU_FFN_HOST
    ntrpu_ffn_host_bfm #(.WATCHDOG_CYCLES(65536),.OUTPUT_FP32(OUTPUT_FP32)) dut(.*);
    wire [1:0] observed_phase=dut.device.core.controller.phase;
    localparam int CANCEL_RESPONSE_DELAY=100;
`else
    logic [63:0] public_expected_epoch;
    wire [63:0] public_sequence,public_epoch,public_ffn_address,public_residual_address;
    wire [13:0] public_ffn_bytes,public_residual_bytes;
    wire public_result_valid,owner_busy,private_tags_exhausted;
    wire [1:0] observed_phase=dut.core.controller.phase;
    assign done_sequence=public_sequence[31:0];
    assign done_epoch=public_epoch[31:0];
    ntrpu_llama32_owned_ffn_r1 #(.WATCHDOG_CYCLES(65536),.INITIAL_TAG(64'h00000000fffffffc)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(job_valid),
        .command_ready(job_ready),
        .command_sequence({32'h12345678,job_sequence}),
        .command_epoch({32'hfedcba98,job_epoch}),
        .expected_epoch(public_expected_epoch),
        .command_source_valid(job_valid),
        .command_source_sequence({32'h12345678,job_sequence}),
        .command_source_epoch({32'hfedcba98,job_epoch}),
        .command_addr(job_addr),
        .command_base(job_base),
        .command_bytes(job_bytes),
        .command_residual_address(64'h0000700000000fc0),
        .command_residual_base(64'h0000700000000000),
        .command_residual_bytes(64'd12224),
        .cancel_valid(cancel_valid),
        .cancel_ready(cancel_ready),
        .busy(owner_busy),
        .completion_valid(done_valid),
        .completion_ready(done_ready),
        .completion_status(done_status),
        .completion_child_status(done_child_status),
        .completion_stages(done_stages),
        .completion_phase(done_phase),
        .completion_bytes(done_bytes),
        .completion_stage_bytes(done_stage_bytes),
        .completion_hls_status(done_hls_status),
        .completion_sequence(public_sequence),
        .completion_epoch(public_epoch),
        .result_valid(public_result_valid),
        .result_ffn_address(public_ffn_address),
        .result_residual_address(public_residual_address),
        .result_ffn_bytes(public_ffn_bytes),
        .result_residual_bytes(public_residual_bytes),
        .tags_exhausted(private_tags_exhausted),
        .activation_selected(activation_selected),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arid(m_axi_arid),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arlock(m_axi_arlock),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awid(m_axi_awid),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready)
    );
    localparam int CANCEL_RESPONSE_DELAY=25;
`endif
`ifdef NTRPU_FFN_LLAMA_REAL
    localparam int LIMIT=73728;
    localparam bit LARGE=1;
    bit [7:0] llama_gate_weights [0:17829823];
    bit [7:0] llama_up_weights [0:17829823];
    bit [7:0] llama_down_weights [0:17829823];
    function automatic logic [7:0] llama_weight_byte(input int b,input int off);
        case(b)
            1:return llama_gate_weights[off];
            2:return llama_up_weights[off];
            3:return llama_down_weights[off];
            default:return 8'hxx;
        endcase
    endfunction
    task automatic verify_llama_weights(input int b);
        logic [63:0] hash,expected;
        hash=64'hcbf29ce484222325;
        for(int off=0;off<17829824;off++)
            hash=(hash ^ {56'b0,llama_weight_byte(b,off)})*64'h00000100000001b3;
        case(b)
            1:expected=64'hd22214a430288927;
            2:expected=64'h2f357a92b56df048;
            3:expected=64'h0ad507f50ff8e421;
            default:expected=64'hx;
        endcase
        require(hash===expected,$sformatf("full real weight image hash buffer=%0d got=%016h expected=%016h",b,hash,expected));
        $display("NTRPU_LLAMA32_WEIGHT_IMAGE buffer=%0d bytes=17829824 fnv=%016h",b,hash);
    endtask
`elsif NTRPU_FFN_LARGE
    localparam int LIMIT=73728;
    localparam bit LARGE=1;
`elsif NTRPU_FFN_FIXTURE
    localparam int LIMIT=262144;
    localparam bit LARGE=0;
`else
    localparam int LIMIT=32768;
    localparam bit LARGE=0;
`endif
    logic [7:0] memory[0:7][0:LIMIT-1];
    bit written[0:7][0:LIMIT-1];
    bit fixture_mode=0;
    string fixture_dir;
`ifdef NTRPU_FFN_FIXTURE
    logic [7:0] fixture_bytes[0:LIMIT-1],golden[0:3][0:65535];
`endif
    logic [63:0] bases[0:7];
    int spans[0:7];
    int seed=1,rng,ident=0,cycles=0,d,h,mode,target;
    int total_cycles=0,total_writes=0,total_resets=0,total_cancel=0;
    int ar_stalls=0,aw_stalls=0,w_stalls=0,done_stalls=0;
    logic [63:0] read_hash[0:7];
    int read_beats_by_buffer[0:7];
    function automatic void require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL FFN MEMORY case=%0d phase=%0d cycle=%0d: %s",ident,observed_phase,cycles,why);
    endfunction
    function automatic int q(input int index);
        if(index%32==0) return 127;
        if(index%32==1) return 1;
        return (index%32)*7-113;
    endfunction
    function automatic int buffer_id(input logic [63:0] addr);
        for(int b=0;b<8;b++) if(addr>=bases[b] && addr<bases[b]+(LARGE ? 64'h20000000:64'(LIMIT))) return b;
        require(0,"unregistered physical address");return -1;
    endfunction
    // Virtual immutable Q8 images avoid allocating three 17 MiB arrays in XSIM.
    // Full groups of 32 rows, tile_k=32; small smoke has a shortened K tile.
    function automatic logic [7:0] large_weight_byte(input int b,input int off);
        int kb,index,lane,group_index,within_group,tile_width,k,row,source,g;
        logic [15:0] half_scale;
        kb=(b==3 ? h:d)/32;index=off/34;lane=off%34;
        group_index=index/(32*kb);within_group=index%(32*kb);
        tile_width=kb<32 ? kb:32;
        k=(within_group/(32*tile_width))*tile_width+(within_group%(32*tile_width))/32;
        row=group_index*32+within_group%32;
        g=257+2*(row/32);
        // Integers 257..767 are exact in half; spacing changes at 512.
        half_scale=b!=1 ? 16'h3c00:g<512 ? 16'h5c00+16'((g-256)*4):16'h6000+16'((g-512)*2);
        if(lane<2) return half_scale[lane*8+:8];
        source=b==1 ? 1:b==2 ? row%d:(row*7)%h;
        return k*32+lane-2==source ? 8'd1:8'd0;
    endfunction
    function automatic logic [511:0] read_word(input logic [63:0] addr);
        int b,off;
        logic [511:0] value;
        b=buffer_id(addr);off=int'(addr-bases[b]);value=0;
        require(off>=4032 && off+64<=4032+spans[b],"read bounds");
        for(int lane=0;lane<64;lane++) begin
            if(b>=4) require(written[b][off+lane],"read unproduced intermediate byte");
`ifdef NTRPU_FFN_LLAMA_REAL
            value[lane*8+:8]=b>=1 && b<=3 ? llama_weight_byte(b,off+lane):memory[b][off+lane];
`else
            value[lane*8+:8]=LARGE && b>=1 && b<=3 ? large_weight_byte(b,off+lane-4032):memory[b][off+lane];
`endif
        end
        return value;
    endfunction
    function automatic logic [7:0] expected_byte(input int b,input int off);
        int row,lane,k,source;
        logic [63:0] bits;
`ifdef NTRPU_FFN_FIXTURE
        if(fixture_mode) return golden[b-4][off];
`endif
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
    task automatic prepare;
        int kb,rows,off,gn,kcount,source,wi;
        logic [15:0] half_scale;
        logic [63:0] bits;
        for(int b=0;b<8;b++) begin
            bases[b]=64'h100000000+64'(b)*64'h20000000;
            for(int i=0;i<LIMIT;i++) begin memory[b][i]=8'ha5;written[b][i]=0;end
            read_hash[b]=64'hcbf29ce484222325;read_beats_by_buffer[b]=0;
        end
        spans[0]=((d+31)/32)*64;spans[1]=h*((d+31)/32)*34;spans[2]=spans[1];
        spans[3]=d*((h+31)/32)*34;spans[4]=((h+7)/8)*64;spans[5]=spans[4];
        spans[6]=((h+31)/32)*64;spans[7]=((d*DOWN_BYTES+63)/64)*64;
`ifdef NTRPU_FFN_FIXTURE
        if(fixture_mode) begin
            $readmemh($sformatf("%s/job-%03d-input.hex",fixture_dir,ident),fixture_bytes,0,4032+spans[0]-1);
            for(int i=0;i<4032+spans[0];i++) memory[0][i]=fixture_bytes[i];
            for(int b=1;b<=3;b++) begin
`ifdef NTRPU_FFN_LLAMA_REAL
                require(spans[b]==17825792,"complete real weight image size");
                case(b)
                    1:$readmemh($sformatf("%s/weight-1.hex",fixture_dir),llama_gate_weights,0,17829823);
                    2:$readmemh($sformatf("%s/weight-2.hex",fixture_dir),llama_up_weights,0,17829823);
                    3:$readmemh($sformatf("%s/weight-3.hex",fixture_dir),llama_down_weights,0,17829823);
                endcase
                verify_llama_weights(b);
`else
                $readmemh($sformatf("%s/weight-%0d.hex",fixture_dir,b),fixture_bytes,0,4032+spans[b]-1);
                for(int i=0;i<4032+spans[b];i++) memory[b][i]=fixture_bytes[i];
`endif
            end
            for(int b=4;b<8;b++) begin
                if(b==7 && OUTPUT_FP32) $readmemh($sformatf("%s/job-%03d-final-fp32.hex",fixture_dir,ident),fixture_bytes,0,spans[b]-1);
                else $readmemh($sformatf("%s/job-%03d-buffer-%0d.hex",fixture_dir,ident,b),fixture_bytes,0,spans[b]-1);
                for(int i=0;i<spans[b];i++) golden[b-4][i]=fixture_bytes[i];
            end
            return;
        end
`endif
        for(int k=0;k<(d+31)/32;k++) begin
            for(int lane=0;lane<64;lane++) memory[0][4032+k*64+lane]=0;
            for(int lane=0;lane<32;lane++) memory[0][4032+k*64+lane]=8'(q(k*32+lane));
            bits=$realtobits(1.0);
            for(int lane=0;lane<8;lane++) memory[0][4032+k*64+32+lane]=bits[lane*8+:8];
        end
        // Pack layout2 independently: row group, K tile, K block, row, lane.
        for(int b=1;b<=3;b++) begin
            if(LARGE) continue;
            rows=b==3 ? d:h;kb=((b==3 ? h:d)+31)/32;off=4032;
            for(int first=0;first<rows;first+=32) begin
                gn=rows-first>32 ? 32:rows-first;
                for(int tile=0;tile<kb;tile+=2) begin
                    kcount=kb-tile>2 ? 2:kb-tile;
                    for(int k=tile;k<tile+kcount;k++) for(int r=first;r<first+gn;r++) begin
                        half_scale=b==1 ? 16'h5c04+16'((r/32)*8):16'h3c00;
                        memory[b][off]=half_scale[7:0];memory[b][off+1]=half_scale[15:8];
                        source=b==1 ? 1:b==2 ? r%d:(r*7)%h;
                        for(int lane=0;lane<32;lane++) memory[b][off+2+lane]=k*32+lane==source ? 1:0;
                        off+=34;
                    end
                end
            end
            require(off==4032+spans[b],"packed weight length");
        end
    endtask
    task automatic run_case(input int nd,input int nh,input int scenario,input int phase_target=0);
        bit accepted,finished,reading,aw_seen,cancelled,reset_seen,fault_sent;
        bit ar_hold,aw_hold,w_hold,done_hold,old_select;
        bit jhs,chs,arhs,rhs,awhs,whs,bhs;
        logic [88:0] saved_ar,saved_aw;
        logic [576:0] saved_w;
        logic [143:0] saved_done;
        logic [511:0] wbuf[0:3];logic [63:0] masks[0:3];bit lasts[0:3];
        logic [63:0] read_addr,write_addr;
        int read_left,burst_len,received,stored,b_wait,done_wait;
        int ar_count,r_count,aw_count,w_count,b_count,byte_count,down_bytes,phase_bytes[0:3],phase_cycles[0:3];
        int b,off,phase_now,status,expected_stages;
        ident++;d=nd;h=nh;mode=scenario;target=phase_target;cycles=0;
        accepted=0;finished=0;reading=0;aw_seen=0;cancelled=0;reset_seen=0;fault_sent=0;
        ar_hold=0;aw_hold=0;w_hold=0;done_hold=0;old_select=0;
        read_left=0;burst_len=0;received=0;stored=0;b_wait=0;done_wait=0;
        ar_count=0;r_count=0;aw_count=0;w_count=0;b_count=0;byte_count=0;down_bytes=0;
        for(int p=0;p<4;p++) begin phase_bytes[p]=0;phase_cycles[p]=0;end
        @(negedge clk);require(job_ready,"previous FFN still active");prepare();
        job_valid=1;cancel_valid=0;done_ready=0;job_layout=2;job_dim=14'(d);job_hidden=14'(h);
        job_sequence=32'hfffffffe;job_epoch=32'(ident+600);expected_epoch=job_epoch;
        public_expected_epoch={32'hfedcba98,job_epoch};
        $display("OWNFFN_BEGIN id=%0d sequence=%016h epoch=%016h source_sample=1 token=0 residual=0000700000000fc0",ident,{32'h12345678,job_sequence},public_expected_epoch);
        for(int p=0;p<3;p++) job_tile_k[p]=2;
        if(LARGE) for(int p=0;p<3;p++) job_tile_k[p]=32;
`ifdef NTRPU_FFN_LLAMA_REAL
        if(fixture_mode) for(int p=0;p<3;p++) job_tile_k[p]=32;
`else
        if(fixture_mode) begin job_tile_k[0]=4;job_tile_k[1]=4;job_tile_k[2]=8;end
`endif
        for(int i=0;i<8;i++) begin
            job_base[i]=bases[i];job_addr[i]=bases[i]+4032;job_bytes[i]=64'(4032+spans[i]);job_buffer_epoch[i]=job_epoch;
        end
        if(mode==5) job_bytes[7]--;
        m_axi_rvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;
        m_axi_bvalid=0;m_axi_bresp=0;m_axi_bid=0;
        while(!finished) begin
            phase_now=int'(observed_phase);
            m_axi_arready=!reading && ($random(rng)&3)!=0;
            m_axi_awready=!aw_seen && ($random(rng)&3)!=0;
            m_axi_wready=received<4 && (!aw_seen || received<burst_len) && ($random(rng)&3)!=0;
            done_ready=done_wait>=6;
            if(reading && !m_axi_rvalid && ($random(rng)&3)!=0) begin
                m_axi_rvalid=1;m_axi_rdata=read_word(read_addr);m_axi_rlast=read_left==1;m_axi_rresp=0;
                if(mode==1 && phase_now==target && !fault_sent) begin m_axi_rresp=2;fault_sent=1;end
            end
            if(aw_seen && received==burst_len && !m_axi_bvalid) begin
                b_wait++;
                if(mode==3 && phase_now==target && !cancelled) cancel_valid=1;
                if(b_wait>=(mode==3 && phase_now==target ? CANCEL_RESPONSE_DELAY:3) && ($random(rng)&3)!=0) begin
                    m_axi_bvalid=1;m_axi_bresp=0;
                    if(mode==2 && phase_now==target && !fault_sent) begin m_axi_bresp=2;fault_sent=1;end
                end
            end
            @(posedge clk);cycles++;phase_cycles[phase_now]++;require(cycles<(LARGE ? 30000000:fixture_mode ? 1000000:250000),"simulation timeout");
            if(LARGE && cycles%100000==0) $display("NTRPU_FFN_LARGE_PROGRESS cycles=%0d phase=%0d bytes=%0d",cycles,phase_now,byte_count);
            jhs=job_valid&&job_ready;chs=cancel_valid&&cancel_ready;arhs=m_axi_arvalid&&m_axi_arready;
            rhs=m_axi_rvalid&&m_axi_rready;awhs=m_axi_awvalid&&m_axi_awready;whs=m_axi_wvalid&&m_axi_wready;bhs=m_axi_bvalid&&m_axi_bready;
            if(ar_hold) require(m_axi_arvalid && saved_ar==={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock},"AR stability");
            if(aw_hold) require(m_axi_awvalid && saved_aw==={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock},"AW stability");
            if(w_hold) require(m_axi_wvalid && saved_w==={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"W stability");
            if(done_hold) require(done_valid && saved_done==={done_status,done_child_status,done_phase,done_stages,done_bytes,done_stage_bytes,done_sequence,done_epoch,done_hls_status},"completion stability");
            ar_hold=m_axi_arvalid&&!m_axi_arready;aw_hold=m_axi_awvalid&&!m_axi_awready;w_hold=m_axi_wvalid&&!m_axi_wready;done_hold=done_valid&&!done_ready;
            saved_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            saved_done={done_status,done_child_status,done_phase,done_stages,done_bytes,done_stage_bytes,done_sequence,done_epoch,done_hls_status};
            ar_stalls+=int'(ar_hold);aw_stalls+=int'(aw_hold);w_stalls+=int'(w_hold);done_stalls+=int'(done_hold);
            if(activation_selected!=old_select) require(!reading && !aw_seen && received==0 && !m_axi_rvalid && !m_axi_bvalid,"AXI owner changed before drain");
            old_select=activation_selected;
            if(jhs) begin require(!accepted,"duplicate root job");accepted=1;end
            if(dut.child_job_valid && dut.child_job_ready)begin
                require({dut.child_job_epoch,dut.child_job_sequence}===64'h00000000fffffffc,"Independent FFN tag");
                $display("OWNFFN_CHILD id=%0d cycle=%0d tag=%016h",ident,cycles,{dut.child_job_epoch,dut.child_job_sequence});
            end
            if(dut.child_done_ready)require(done_valid && done_ready,"FFN child released before public ACK");
            if(!public_result_valid)require(public_ffn_address===0 && public_residual_address===0 && public_ffn_bytes===0 && public_residual_bytes===0,"Private descriptor leak");

            if(chs) begin require(!cancelled,"duplicate cancel");cancelled=1;total_cancel++;end
            if(arhs) begin
                require(accepted && !reading,"overlapping read bursts");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && !m_axi_arlock && m_axi_arprot==0,"AR attributes");
                require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR page crossing");
                b=buffer_id(m_axi_araddr);
                require(phase_now==0 ? b==0 || b==1:phase_now==1 ? b==0 || b==2:phase_now==2 ? b==4 || b==5:b==6 || b==3,"phase read ownership");
                if(LARGE) $display("NTRPU_FFN_LARGE_AR id=%0d phase=%0d buffer=%0d offset=%0d beats=%0d",ident,phase_now,b,m_axi_araddr-bases[b]-4032,int'(m_axi_arlen)+1);
                read_addr=m_axi_araddr;read_left=int'(m_axi_arlen)+1;reading=1;ar_count++;
            end
            if(rhs) begin
                require(reading && read_left>0,"unexpected R");
                if(LARGE) begin
                    b=buffer_id(read_addr);read_beats_by_buffer[b]++;
                    for(int word_index=0;word_index<8;word_index++)
                        read_hash[b]=(read_hash[b]^m_axi_rdata[word_index*64+:64])*64'h100000001b3;
                end
                read_left--;read_addr+=64;r_count++;if(read_left==0) reading=0;
            end
            if(awhs) begin
                require(!aw_seen,"overlapping AW");aw_seen=1;write_addr=m_axi_awaddr;burst_len=int'(m_axi_awlen)+1;aw_count++;
                require(burst_len<=4 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && !m_axi_awlock && m_axi_awprot==0,"AW attributes");
                require(int'(write_addr[11:0])+burst_len*64<=4096,"AW page crossing");
                b=buffer_id(write_addr);require(b==phase_now+4,"phase write ownership");
                $display("OWNFFN_AW id=%0d cycle=%0d phase=%0d buffer=%0d offset=%0d beats=%0d",ident,cycles,phase_now,b,write_addr-bases[b]-4032,burst_len);
            end
            if(whs) begin
                require(received<4,"W buffer overflow");wbuf[received]=m_axi_wdata;masks[received]=m_axi_wstrb;lasts[received]=m_axi_wlast;received++;w_count++;
                for(int lane=0;lane<64;lane++) if(m_axi_wstrb[lane]) begin byte_count++;phase_bytes[phase_now]++;if(phase_now==3) down_bytes++;end
            end
            if(aw_seen) while(stored<received) begin
                b=buffer_id(write_addr);off=int'(write_addr-bases[b])+stored*64;
                require(off>=4032 && off+64<=4032+spans[b] && lasts[stored]==(stored==burst_len-1),"write bounds/WLAST");
                for(int lane=0;lane<64;lane++) if(masks[stored][lane]) begin
                    require(!written[b][off+lane],"duplicate output byte");
                    require(wbuf[stored][lane*8+:8]===expected_byte(b,off+lane-4032),$sformatf("computed byte mismatch buffer=%0d offset=%0d got=%02h want=%02h",b,off+lane-4032,wbuf[stored][lane*8+:8],expected_byte(b,off+lane-4032)));
                    // Keep payload private in wbuf until accepted successful B.
                end
                $display("NTRPU_FFN_MEMORY_W id=%0d buffer=%0d offset=%0d strb=%016h data=%0128h",ident,b,off-4032,masks[stored],wbuf[stored]);
                stored++;
            end
            if(bhs) begin
                require(aw_seen && received==burst_len && stored==received,"B before AW/W");
                b=buffer_id(write_addr);
                require(m_axi_bresp==0,"Normal fixture received a failed write response");
                for(int beat=0;beat<stored;beat++)begin
                    off=int'(write_addr-bases[b])+beat*64;
                    for(int lane=0;lane<64;lane++)if(masks[beat][lane])begin
                        require(!written[b][off+lane],"Duplicate B-committed byte");
                        written[b][off+lane]=1;memory[b][off+lane]=wbuf[beat][lane*8+:8];
                    end
                end
                $display("OWNFFN_B id=%0d cycle=%0d phase=%0d buffer=%0d offset=%0d beats=%0d resp=%0d",ident,cycles,phase_now,b,write_addr-bases[b]-4032,burst_len,m_axi_bresp);
                b_count++;aw_seen=0;received=0;stored=0;b_wait=0;
            end
            if(done_valid) begin
                done_wait++;status=mode==1 || mode==2 ? 6:mode==3 ? 7:mode==5 ? 1:0;
                expected_stages=mode==0 ? 4:mode==5 ? 0:target;
                require(!reading && !aw_seen && received==0 && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && aw_count==b_count,"root completion before AXI drain");
                require(done_status==status && done_stages==expected_stages,$sformatf("completion status/stages got=%0d/%0d child=%0d",done_status,done_stages,done_child_status));
                require(done_sequence==32'hfffffffe && done_epoch==32'(ident+600) && done_hls_status==0,"root identity/HLS status");
                require(public_sequence===64'h12345678fffffffe && public_epoch===(64'hfedcba9800000000|64'(ident+600)),"Full public completion identity");
                require(public_result_valid && public_ffn_address===bases[7]+4032 && public_residual_address===64'h0000700000000fc0 && public_ffn_bytes===8192 && public_residual_bytes===8192,"Retained output descriptors");
                if(done_wait==1)$display("OWNFFN_COMPLETE id=%0d cycle=%0d sequence=%016h epoch=%016h status=%0d child=%0d stages=%0d phase=%0d bytes=%0d stage_bytes=%0d hls=%08h ffn=%016h residual=%016h valid=%0d",ident,cycles,public_sequence,public_epoch,done_status,done_child_status,done_stages,done_phase,done_bytes,done_stage_bytes,done_hls_status,public_ffn_address,public_residual_address,public_result_valid);
                if(done_ready)$display("OWNFFN_ACK id=%0d cycle=%0d child_ready=%0d",ident,cycles,dut.child_done_ready);

                require(done_stage_bytes==byte_count && done_bytes==down_bytes,"root byte accounting");
                if(mode==0) require(down_bytes==d*DOWN_BYTES && byte_count==2*h*8+spans[6]+d*DOWN_BYTES,"full chain byte count");
                if(mode==5) require(ar_count==0 && byte_count==0,"invalid down descriptor caused traffic");
                if(done_ready) finished=1;
            end
            #1;
            if(jhs) begin
                job_valid=0;job_dim=0;job_hidden=0;job_layout=0;job_sequence=0;job_epoch=0;expected_epoch=0;
                for(int i=0;i<8;i++) begin job_addr[i]=0;job_base[i]=0;job_bytes[i]=0;job_buffer_epoch[i]=0;end
                for(int i=0;i<3;i++) job_tile_k[i]=0;
            end
            if(chs) cancel_valid=0;if(rhs) m_axi_rvalid=0;if(bhs) m_axi_bvalid=0;
            if(mode==4 && phase_now==target && phase_bytes[target]>0) begin
                @(negedge clk);rst_n=0;job_valid=0;cancel_valid=0;m_axi_rvalid=0;m_axi_bvalid=0;
`ifdef NTRPU_FFN_ASYNC_TEST
                reset_domains();
`else
                repeat(4) @(negedge clk);rst_n=1;repeat(8) @(negedge clk);
`endif
                require(job_ready && !done_valid && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid,"reset recovery");
                reset_seen=1;total_resets++;finished=1;
            end
            if(!finished) @(negedge clk);
        end
        for(int b=4;b<8;b++) for(int i=0;i<LIMIT;i++) begin
            if(written[b][i]) require(memory[b][i]===expected_byte(b,i-4032),"stored numerical byte");
            else require(memory[b][i]==8'ha5,"unwritten guard modified");
        end
        require(cancelled==(mode==3) && reset_seen==(mode==4) && fault_sent==(mode==1 || mode==2),"fault coverage");
        if(LARGE) for(int b=0;b<8;b++)
            $display("NTRPU_FFN_LARGE_READ id=%0d buffer=%0d beats=%0d hash=%016h",ident,b,read_beats_by_buffer[b],read_hash[b]);
        $display("NTRPU_FFN_MEMORY_CASE id=%0d dim=%0d hidden=%0d mode=%0d target=%0d reset=%0d cycles=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d bytes=%0d down_bytes=%0d p0=%0d p1=%0d p2=%0d p3=%0d",ident,d,h,mode,target,reset_seen,cycles,ar_count,r_count,aw_count,w_count,b_count,byte_count,down_bytes,phase_cycles[0],phase_cycles[1],phase_cycles[2],phase_cycles[3]);
        total_cycles+=cycles;total_writes+=byte_count;
        cancel_valid=0;repeat(3) @(negedge clk);
`ifdef NTRPU_FFN_HOST
        // The host reads completion first and sends ACK after the consumer accepts it.
        begin
            int ack_wait;
            ack_wait=0;
            while(!job_ready) begin @(negedge clk);ack_wait++;require(ack_wait<
`ifdef NTRPU_FFN_ASYNC_TEST
                    5000
`else
                    100
`endif
                    ,"host ACK timeout");end
        end
`endif
    endtask
    initial begin
        int fd,items,samples,fixture_dim,fixture_hidden;
        string magic;
        clk=0;rst_n=0;job_valid=0;cancel_valid=0;done_ready=0;m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;m_axi_rvalid=0;m_axi_bvalid=0;
        if($value$plusargs("SEED=%d",seed)) begin end
        rng=seed;$display("NTRPU_FFN_MEMORY_SEED seed=%0d",seed);
        $display("NTRPU_FFN_MEMORY_FORMAT fp32=%0d",OUTPUT_FP32);
`ifdef NTRPU_FFN_ASYNC_TEST
        if($value$plusargs("HOST_HALF_PS=%d",host_half_ps)) begin end
        host_clock_ready=1;reset_domains();
        $display("NTRPU_FFN_IRQ_CONFIG host_period_ps=%0d device_period_ps=5000 host_phase_ps=700 reset_edges=64",2*host_half_ps);
`else
        repeat(4) @(negedge clk);rst_n=1;
`endif
`ifdef NTRPU_FFN_LARGE
        require(OUTPUT_FP32 && $value$plusargs("DIM=%d",fixture_dim) && $value$plusargs("HIDDEN=%d",fixture_hidden),"large shape arguments");
        require(fixture_dim==2048 && (fixture_hidden==96 || fixture_hidden==8192),"large shape contract");
        $display("NTRPU_FFN_LARGE_CONFIG dim=%0d hidden=%0d tile_k=32",fixture_dim,fixture_hidden);
        run_case(fixture_dim,fixture_hidden,0);
        require(ident==1 && total_resets==0 && total_cancel==0 && ar_stalls>0 && aw_stalls>0 && w_stalls>0 && done_stalls>0,"large coverage");
`elsif NTRPU_FFN_FIXTURE
        require($value$plusargs("FIXTURE=%s",fixture_dir),"missing fixture path");
        fd=$fopen({fixture_dir,"/jobs.txt"},"r");require(fd!=0,"fixture descriptors open");
        items=$fscanf(fd,"%s %d %d %d",magic,samples,fixture_dim,fixture_hidden);$fclose(fd);
        require(items==4 && magic=="NTRFFNMEM01" && samples==1 && fixture_dim==2048 && fixture_hidden==8192 && OUTPUT_FP32,"full Llama fixture shape/count");
        $display("NTRPU_FFN_LARGE_CONFIG dim=2048 hidden=8192 tile_k=32");
        $display("NTRPU_LLAMA32_REAL_WEIGHTS layer=0 source_sample=1 bytes=53477376");
        fixture_mode=1;
        for(int sample=0;sample<samples;sample++) run_case(fixture_dim,fixture_hidden,0);
        require(ident==1 && total_resets==0 && total_cancel==0 && ar_stalls>0 && aw_stalls>0 && w_stalls>0 && done_stalls>0,"full real fixture coverage");
`else
        run_case(32,32,0);run_case(64,96,0);run_case(64,80,0);
        for(int mode=1;mode<=4;mode++) for(int p=0;p<4;p++) run_case(64,96,mode,p);
        run_case(64,96,0);run_case(64,96,5);
        require(ident==21 && total_resets==4 && total_cancel==4 && ar_stalls>0 && aw_stalls>0 && w_stalls>0 && done_stalls>0,"bank coverage");
`endif
        $display("NTRPU_FFN_MEMORY_PASS cases=%0d cycles=%0d bytes=%0d resets=%0d cancels=%0d ar_stalls=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d",ident,total_cycles,total_writes,total_resets,total_cancel,ar_stalls,aw_stalls,w_stalls,done_stalls);
        $display("OWNFFN_PASS jobs=1 public_bits=64 retained_bytes=8192 input_abi_requantized=0");
        $finish;
    end
endmodule
