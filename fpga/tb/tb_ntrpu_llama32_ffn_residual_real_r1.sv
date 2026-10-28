`timescale 1ns/1ps
`default_nettype none
// Actual kernels only. Expected outputs are never materialized as DUT inputs.
module tb_ntrpu_llama32_ffn_residual_real_r1;
    logic clk,rst_n;
    logic command_valid;
    logic [63:0] command_sequence;
    logic [63:0] command_epoch;
    logic command_source_valid;
    logic [63:0] command_source_sequence;
    logic [63:0] command_source_epoch;
    logic [7:0][63:0] command_addr_drive;
    wire [63:0] command_addr [0:7];
    logic [7:0][63:0] command_base_drive;
    wire [63:0] command_base [0:7];
    logic [7:0][63:0] command_bytes_drive;
    wire [63:0] command_bytes [0:7];
    logic [63:0] command_residual_address;
    logic [63:0] command_residual_base;
    logic [63:0] command_residual_bytes;
    logic [63:0] expected_epoch;
    logic [63:0] command_output_address,command_output_base,command_output_bytes;
    wire command_ready;
    logic cancel_valid;
    wire cancel_ready,busy;
    wire completion_valid;
    logic completion_ready;
    wire [3:0] completion_status,completion_ffn_status,completion_ffn_child_status,completion_residual_status;
    wire [2:0] completion_ffn_stages;
    wire [1:0] completion_ffn_phase;
    wire [16:0] completion_ffn_bytes;
    wire [17:0] completion_ffn_stage_bytes;
    wire [31:0] completion_ffn_hls_status,completion_hls_status;
    wire [9:0] completion_read_beats;
    wire [7:0] completion_write_beats;
    wire [63:0] completion_sequence,completion_epoch;
    wire result_valid;
    wire [63:0] result_address;
    wire [13:0] result_bytes;
    wire tags_exhausted,activation_selected,residual_selected;
    wire [63:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize;
    wire [2:0] m_axi_arprot;
    wire [1:0] m_axi_arburst;
    wire [3:0] m_axi_arid;
    wire [3:0] m_axi_arcache;
    wire m_axi_arlock;
    wire m_axi_arvalid;
    logic m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast;
    logic m_axi_rvalid;
    wire m_axi_rready;
    wire [63:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize;
    wire [2:0] m_axi_awprot;
    wire [1:0] m_axi_awburst;
    wire [3:0] m_axi_awid;
    wire [3:0] m_axi_awcache;
    wire m_axi_awlock;
    wire m_axi_awvalid;
    logic m_axi_awready;
    wire [511:0] m_axi_wdata;
    wire [63:0] m_axi_wstrb;
    wire m_axi_wlast;
    wire m_axi_wvalid;
    logic m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic m_axi_bvalid;
    wire m_axi_bready;
    for(genvar i=0;i<8;i++)begin
        assign command_addr[i]=command_addr_drive[i];
        assign command_base[i]=command_base_drive[i];
        assign command_bytes[i]=command_bytes_drive[i];
    end
    ntrpu_llama32_ffn_residual_r3 #(.INITIAL_TAG(64'h00000000fffffffc)) dut(.*);
    wire [1:0] observed_phase=dut.ffn.core.controller.phase;
    localparam bit OUTPUT_FP32=1;
    localparam int DOWN_BYTES=4;
    always #2.5 clk=~clk;
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
    logic [7:0] memory[0:9][0:LIMIT-1];
    bit written[0:9][0:LIMIT-1];
    bit fixture_mode=0;
    string fixture_dir,extra_dir;
    logic [7:0] fixture_bytes[0:LIMIT-1],golden[0:4][0:65535];
    logic [63:0] bases[0:9];
    int spans[0:9];
    int seed=1,rng,ident=0,cycles=0,d,h,mode,target;
    int total_cycles=0,total_writes=0,total_resets=0,total_cancel=0;
    int ar_stalls=0,aw_stalls=0,w_stalls=0,done_stalls=0;
    logic [63:0] read_hash[0:9];
    int read_beats_by_buffer[0:9];
    function automatic void require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL FFN MEMORY case=%0d phase=%0d cycle=%0d: %s",ident,observed_phase,cycles,why);
    endfunction
    function automatic int q(input int index);
        if(index%32==0) return 127;
        if(index%32==1) return 1;
        return (index%32)*7-113;
    endfunction
    function automatic int buffer_id(input logic [63:0] addr);
        for(int b=0;b<10;b++) if(addr>=bases[b] && addr<bases[b]+(LARGE ? 64'h20000000:64'(LIMIT))) return b;
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
            if(b>=4 && b!=8) require(written[b][off+lane],"read unproduced intermediate byte");
            value[lane*8+:8]=b>=1 && b<=3 ? llama_weight_byte(b,off+lane):memory[b][off+lane];
        end
        return value;
    endfunction
    function automatic logic [7:0] expected_byte(input int b,input int off);
        int row,lane,k,source;
        logic [63:0] bits;
        if(fixture_mode) return golden[b==9 ? 4:b-4][off];
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
        for(int b=0;b<10;b++) begin
            bases[b]=64'h100000000+64'(b)*64'h20000000;
            for(int i=0;i<LIMIT;i++) begin memory[b][i]=8'ha5;written[b][i]=0;end
            read_hash[b]=64'hcbf29ce484222325;read_beats_by_buffer[b]=0;
        end
        spans[0]=((d+31)/32)*64;spans[1]=h*((d+31)/32)*34;spans[2]=spans[1];
        spans[3]=d*((h+31)/32)*34;spans[4]=((h+7)/8)*64;spans[5]=spans[4];
        spans[6]=((h+31)/32)*64;spans[7]=((d*DOWN_BYTES+63)/64)*64;
        bases[8]=64'h0000700000000000;bases[9]=64'h0000710000000000;
        spans[8]=8192;spans[9]=8192;
        if(fixture_mode) begin
            $readmemh($sformatf("%s/job-%03d-input.hex",fixture_dir,ident),fixture_bytes,0,4032+spans[0]-1);
            for(int i=0;i<4032+spans[0];i++) memory[0][i]=fixture_bytes[i];
            for(int b=1;b<=3;b++) begin
                require(spans[b]==17825792,"complete real weight image size");
                case(b)
                    1:$readmemh($sformatf("%s/weight-1.hex",fixture_dir),llama_gate_weights,0,17829823);
                    2:$readmemh($sformatf("%s/weight-2.hex",fixture_dir),llama_up_weights,0,17829823);
                    3:$readmemh($sformatf("%s/weight-3.hex",fixture_dir),llama_down_weights,0,17829823);
                endcase
                verify_llama_weights(b);
            end
            for(int b=4;b<8;b++) begin
                if(b==7 && OUTPUT_FP32) $readmemh($sformatf("%s/job-%03d-final-fp32.hex",fixture_dir,ident),fixture_bytes,0,spans[b]-1);
                else $readmemh($sformatf("%s/job-%03d-buffer-%0d.hex",fixture_dir,ident,b),fixture_bytes,0,spans[b]-1);
                for(int i=0;i<spans[b];i++) golden[b-4][i]=fixture_bytes[i];
            end
            $readmemh({extra_dir,"/residual-input.hex"},fixture_bytes,0,12223);
            for(int i=0;i<12224;i++)memory[8][i]=fixture_bytes[i];
            $readmemh({extra_dir,"/layer-output.hex"},fixture_bytes,0,8191);
            for(int i=0;i<8192;i++)golden[4][i]=fixture_bytes[i];
            return;
        end
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

    task automatic run_complete;
        bit accepted=0,finished=0,reading=0,aw_seen=0;
        bit ar_hold=0,aw_hold=0,w_hold=0,old_res=0,old_activation=0;
        bit jhs,arhs,rhs,awhs,whs,bhs,seen_ffn=0,seen_res=0;
        bit hls_busy=0;int hls_starts=0,hls_dones=0,hls_words=0;
        logic [88:0] saved_ar,saved_aw;
        logic [576:0] saved_w;
        logic [511:0] wbuf[0:3];logic [63:0] masks[0:3];bit lasts[0:3];
        logic [63:0] read_addr,write_addr;
        int read_left=0,burst_len=0,received=0,stored=0,b_wait=0,done_wait=0;
        int ar_count=0,r_count=0,aw_count=0,w_count=0,b_count=0,byte_count=0;
        int phase_cycles[0:4],phase_reads[0:4],phase_writes[0:4];
        int b,off,phase_now,ffn_done_cycle=0,first_res_ar=0,last_res_b=0;
        ident=1;d=2048;h=8192;mode=0;target=0;cycles=0;fixture_mode=1;
        for(int p=0;p<5;p++)begin phase_cycles[p]=0;phase_reads[p]=0;phase_writes[p]=0;end
        @(negedge clk);require(command_ready,"Initial admission unavailable");prepare();
        command_valid=1;command_sequence=64'h12345678fffffffe;command_epoch=64'hfedcba9800000259;expected_epoch=command_epoch;
        command_source_valid=1;command_source_sequence=command_sequence;command_source_epoch=command_epoch;
        for(int b=0;b<8;b++)begin command_addr_drive[b]=bases[b]+4032;command_base_drive[b]=bases[b];command_bytes_drive[b]=64'(4032+spans[b]);end
        command_residual_address=bases[8]+4032;command_residual_base=bases[8];command_residual_bytes=12224;
        command_output_address=bases[9]+4032;command_output_base=bases[9];command_output_bytes=12224;
        $display("FFRREAL_BEGIN cycle=0 sequence=%016h epoch=%016h source_sample=1 token=0 residual=%016h output=%016h",command_sequence,command_epoch,command_residual_address,command_output_address);
        while(!finished)begin
            phase_now=residual_selected ? 4:int'(observed_phase);
            m_axi_arready=!reading && ($random(rng)&3)!=0;
            m_axi_awready=!aw_seen && ($random(rng)&3)!=0;
            m_axi_wready=received<4 && (!aw_seen || received<burst_len) && ($random(rng)&3)!=0;
            completion_ready=done_wait>=6;
            if(reading && !m_axi_rvalid && ($random(rng)&3)!=0)begin
                m_axi_rvalid=1;m_axi_rdata=read_word(read_addr);m_axi_rlast=read_left==1;m_axi_rresp=0;m_axi_rid=0;
            end
            if(aw_seen && received==burst_len && !m_axi_bvalid)begin
                b_wait++;
                if(b_wait>=3 && ($random(rng)&3)!=0)begin m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;end
            end
            @(posedge clk);cycles++;phase_cycles[phase_now]++;
            require(cycles<30000000,"Full composition exceeded cycle limit");
            if(cycles%100000==0)$display("FFRREAL_PROGRESS cycle=%0d phase=%0d bytes=%0d",cycles,phase_now,byte_count);
            jhs=command_valid&&command_ready;arhs=m_axi_arvalid&&m_axi_arready;rhs=m_axi_rvalid&&m_axi_rready;
            awhs=m_axi_awvalid&&m_axi_awready;whs=m_axi_wvalid&&m_axi_wready;bhs=m_axi_bvalid&&m_axi_bready;
            if(ar_hold)require(m_axi_arvalid && saved_ar==={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock},"Held AR");
            if(aw_hold)require(m_axi_awvalid && saved_aw==={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock},"Held AW");
            if(w_hold)require(m_axi_wvalid && saved_w==={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"Held W");
            ar_hold=m_axi_arvalid&&!m_axi_arready;aw_hold=m_axi_awvalid&&!m_axi_awready;w_hold=m_axi_wvalid&&!m_axi_wready;
            saved_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            ar_stalls+=int'(ar_hold);aw_stalls+=int'(aw_hold);w_stalls+=int'(w_hold);done_stalls+=int'(completion_valid&&!completion_ready);
            if(jhs)begin require(!accepted,"Duplicate public command");accepted=1;$display("FFRREAL_ADMIT cycle=%0d",cycles);end
            if(busy)require(!command_ready,"Busy admission");
            if(activation_selected!=old_activation || residual_selected!=old_res)begin
                require(!reading && !aw_seen && received==0 && !m_axi_rvalid && !m_axi_bvalid,"AXI ownership changed before drain");
                if(residual_selected!=old_res)$display("FFRREAL_SWITCH cycle=%0d residual=%0d",cycles,residual_selected);
            end
            old_activation=activation_selected;old_res=residual_selected;
            if(dut.ffn.child_job_valid && dut.ffn.child_job_ready)begin
                require({dut.ffn.child_job_epoch,dut.ffn.child_job_sequence}===64'h00000000fffffffc,"Private FFN tag");
                $display("FFRREAL_FFN_CHILD cycle=%0d tag=%016h",cycles,{dut.ffn.child_job_epoch,dut.ffn.child_job_sequence});
            end
            if(dut.ffn_completion_valid && !seen_ffn)begin
                require(!reading && !aw_seen && received==0 && byte_count==155648,"FFN completion before B commit");
                require(dut.ffn_result_valid && dut.ffn_result_ffn_address==bases[7]+4032 && dut.ffn_result_residual_address==bases[8]+4032 &&
                    dut.ffn_result_ffn_bytes==8192 && dut.ffn_result_residual_bytes==8192,"FFN source descriptors");
                require(dut.ffn_completion_status==0 && dut.ffn_completion_child_status==0 && dut.ffn_completion_stages==4 && dut.ffn_completion_phase==3 &&
                    dut.ffn_completion_bytes==8192 && dut.ffn_completion_stage_bytes==155648 && dut.ffn_completion_hls_status==0,"FFN completion shape");
                seen_ffn=1;ffn_done_cycle=cycles;
                $display("FFRREAL_FFN_COMPLETE cycle=%0d sequence=%016h epoch=%016h status=%0d child=%0d stages=%0d phase=%0d bytes=%0d stage_bytes=%0d hls=%08h",cycles,dut.ffn_completion_sequence,dut.ffn_completion_epoch,dut.ffn_completion_status,dut.ffn_completion_child_status,dut.ffn_completion_stages,dut.ffn_completion_phase,dut.ffn_completion_bytes,dut.ffn_completion_stage_bytes,dut.ffn_completion_hls_status);
            end
            if(dut.res_command_valid && dut.res_command_ready)begin
                require(seen_ffn && !seen_res && residual_selected && !dut.ffn_completion_ready,"Residual started without held FFN");seen_res=1;
                $display("FFRREAL_RES_START cycle=%0d sequence=%016h epoch=%016h input=%016h branch=%016h output=%016h",cycles,dut.res_command_sequence,dut.res_command_epoch,dut.res_command_input_address,dut.res_command_branch_address,dut.res_command_residual_address);
            end
            if(dut.ffn.child_done_ready || dut.ffn_completion_ready || dut.res_completion_ready)begin
                require(completion_valid && completion_ready && dut.ffn.child_done_ready && dut.ffn_completion_ready && dut.res_completion_ready,"Atomic final release");
                $display("FFRREAL_RELEASE cycle=%0d private_ffn=%0d ffn=%0d residual=%0d",cycles,dut.ffn.child_done_ready,dut.ffn_completion_ready,dut.res_completion_ready);
            end
            if(!result_valid)require(result_address==0 && result_bytes==0,"Private final descriptor leak");
            if(dut.res.hls_start)begin
                require(!hls_busy && hls_starts==0 && phase_reads[4]==256 && dut.res.hls_elements==2048,"Residual HLS start");
                hls_busy=1;hls_starts++;$display("FFRREAL_HLS_START cycle=%0d",cycles);
            end
            if(dut.res.hls_residual_ce && dut.res.hls_residual_we)begin
                require(hls_busy && dut.res.hls_residual_address==hls_words,"Residual HLS write order");
                for(int lane=0;lane<4;lane++)require(dut.res.hls_residual_d[lane*8+:8]===golden[4][hls_words*4+lane],"Residual HLS exact output");
                $display("FFRREAL_HLS_WRITE cycle=%0d index=%0d bits=%08h",cycles,hls_words,dut.res.hls_residual_d);hls_words++;
            end
            if(dut.res.hls_done)begin
                require(hls_busy && hls_words==2048 && dut.res.hls_return==0 && dut.res.hls_ready,"Residual HLS completion");
                hls_busy=0;hls_dones++;$display("FFRREAL_HLS_DONE cycle=%0d return=%0d",cycles,dut.res.hls_return);
            end
            if(arhs)begin
                require(accepted && !reading,"Overlapping read bursts");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && !m_axi_arlock && m_axi_arprot==0,"AR attributes");
                require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR page crossing");
                b=buffer_id(m_axi_araddr);
                require(phase_now==0 ? b==0 || b==1:phase_now==1 ? b==0 || b==2:phase_now==2 ? b==4 || b==5:phase_now==3 ? b==6 || b==3:b==8 || b==7,"Read phase ownership");
                if(phase_now==4)begin require(seen_res && seen_ffn,"Residual traffic before admission");if(first_res_ar==0)first_res_ar=cycles;end
                $display("FFRREAL_AR cycle=%0d phase=%0d buffer=%0d offset=%0d beats=%0d",cycles,phase_now,b,m_axi_araddr-bases[b]-4032,int'(m_axi_arlen)+1);
                read_addr=m_axi_araddr;read_left=int'(m_axi_arlen)+1;reading=1;ar_count++;
            end
            if(rhs)begin
                require(reading && read_left>0 && m_axi_rlast==(read_left==1) && m_axi_rresp==0 && m_axi_rid==0,"R shape");
                require(residual_selected ? dut.res_m_axi_rvalid && !dut.ffn_m_axi_rvalid:dut.ffn_m_axi_rvalid && !dut.res_m_axi_rvalid,"R isolation");
                b=buffer_id(read_addr);read_beats_by_buffer[b]=read_beats_by_buffer[b]+1;
                for(int word_index=0;word_index<8;word_index++)read_hash[b]=(read_hash[b]^m_axi_rdata[word_index*64+:64])*64'h100000001b3;
                if(phase_now==4)$display("FFRREAL_RES_R cycle=%0d buffer=%0d offset=%0d data=%0128h",cycles,b,read_addr-bases[b]-4032,m_axi_rdata);
                phase_reads[phase_now]++;read_left--;read_addr+=64;r_count++;if(read_left==0)reading=0;
            end
            if(awhs)begin
                require(!aw_seen,"Overlapping AW");aw_seen=1;write_addr=m_axi_awaddr;burst_len=int'(m_axi_awlen)+1;aw_count++;
                require(burst_len<=4 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && !m_axi_awlock && m_axi_awprot==0,"AW attributes");
                require(int'(write_addr[11:0])+burst_len*64<=4096,"AW page crossing");
                b=buffer_id(write_addr);require(b==(phase_now==4 ? 9:phase_now+4),"Write phase ownership");
                if(phase_now==4)require(burst_len==1 && hls_dones==1,"Residual store contract");
                $display("FFRREAL_AW cycle=%0d phase=%0d buffer=%0d offset=%0d beats=%0d",cycles,phase_now,b,write_addr-bases[b]-4032,burst_len);
            end
            if(whs)begin
                require(received<4 && m_axi_wstrb=='1,"W extent/strobe");wbuf[received]=m_axi_wdata;masks[received]=m_axi_wstrb;lasts[received]=m_axi_wlast;received++;w_count++;
                byte_count+=64;phase_writes[phase_now]++;
            end
            if(aw_seen)while(stored<received)begin
                b=buffer_id(write_addr);off=int'(write_addr-bases[b])+stored*64;
                require(off>=4032 && off+64<=4032+spans[b] && lasts[stored]==(stored==burst_len-1),"Write bounds/WLAST");
                for(int lane=0;lane<64;lane++)begin
                    require(!written[b][off+lane],"Duplicate output byte");
                    require(wbuf[stored][lane*8+:8]===expected_byte(b,off+lane-4032),$sformatf("Computed output mismatch buffer=%0d offset=%0d",b,off+lane-4032));
                end
                $display("FFRREAL_W cycle=%0d buffer=%0d offset=%0d strb=%016h data=%0128h",cycles,b,off-4032,masks[stored],wbuf[stored]);stored++;
            end
            if(bhs)begin
                require(aw_seen && received==burst_len && stored==received && m_axi_bresp==0 && m_axi_bid==0,"B before AW/W or error");
                require(residual_selected ? dut.res_m_axi_bvalid && !dut.ffn_m_axi_bvalid:dut.ffn_m_axi_bvalid && !dut.res_m_axi_bvalid,"B isolation");
                b=buffer_id(write_addr);
                for(int beat=0;beat<stored;beat++)begin
                    off=int'(write_addr-bases[b])+beat*64;
                    for(int lane=0;lane<64;lane++)begin
                        require(!written[b][off+lane],"Duplicate B-committed byte");written[b][off+lane]=1;memory[b][off+lane]=wbuf[beat][lane*8+:8];
                    end
                end
                $display("FFRREAL_B cycle=%0d phase=%0d buffer=%0d offset=%0d beats=%0d resp=%0d",cycles,phase_now,b,write_addr-bases[b]-4032,burst_len,m_axi_bresp);
                if(phase_now==4)last_res_b=cycles;
                b_count++;aw_seen=0;received=0;stored=0;b_wait=0;
            end
            if(completion_valid)begin
                done_wait++;
                require(!reading && !aw_seen && received==0 && !hls_busy && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && aw_count==b_count,"Completion before full drain");
                require(completion_status==0 && completion_ffn_status==0 && completion_ffn_child_status==0 && completion_residual_status==0 && completion_ffn_hls_status==0 && completion_hls_status==0,"Public completion status");
                require(completion_sequence==64'h12345678fffffffe && completion_epoch==64'hfedcba9800000259,"Public 64-bit identity");
                require(completion_ffn_stages==4 && completion_ffn_phase==3 && completion_ffn_bytes==8192 && completion_ffn_stage_bytes==155648 && completion_read_beats==256 && completion_write_beats==128,"Public accounting");
                require(result_valid && result_address==bases[9]+4032 && result_bytes==8192 && byte_count==163840,"Final published bytes");
                require(dut.ffn_completion_valid && dut.res_completion_valid,"Lost final ownership");
                if(done_wait==1)$display("FFRREAL_COMPLETE cycle=%0d sequence=%016h epoch=%016h status=%0d ffn=%0d child=%0d residual=%0d stages=%0d phase=%0d ffn_bytes=%0d stage_bytes=%0d ffn_hls=%08h hls=%08h reads=%0d writes=%0d valid=%0d address=%016h bytes=%0d",cycles,completion_sequence,completion_epoch,completion_status,completion_ffn_status,completion_ffn_child_status,completion_residual_status,completion_ffn_stages,completion_ffn_phase,completion_ffn_bytes,completion_ffn_stage_bytes,completion_ffn_hls_status,completion_hls_status,completion_read_beats,completion_write_beats,result_valid,result_address,result_bytes);
                if(completion_ready)begin $display("FFRREAL_ACK cycle=%0d",cycles);finished=1;end
            end
            #1;
            if(jhs)begin
                command_valid=0;command_sequence=0;command_epoch=0;command_source_valid=0;command_source_sequence=0;command_source_epoch=0;
                command_addr_drive=0;command_base_drive=0;command_bytes_drive=0;
                command_residual_address=0;command_residual_base=0;command_residual_bytes=0;command_output_address=0;command_output_base=0;command_output_bytes=0;
            end
            if(rhs)m_axi_rvalid=0;if(bhs)m_axi_bvalid=0;
            if(!finished)@(negedge clk);
        end
        for(int b=4;b<10;b++)if(b!=8)for(int i=0;i<LIMIT;i++)begin
            if(i>=4032 && i<4032+spans[b])require(written[b][i] && memory[b][i]===expected_byte(b,i-4032),"Stored output coverage/value");
            else require(!written[b][i] && memory[b][i]==8'ha5,"Output guard modified");
        end
        require(hls_starts==1 && hls_dones==1 && hls_words==2048 && ar_stalls>0 && aw_stalls>0 && w_stalls>0 && done_stalls==6,"Coverage");
        for(int b=0;b<10;b++)$display("FFRREAL_READ buffer=%0d beats=%0d hash=%016h",b,read_beats_by_buffer[b],read_hash[b]);
        for(int p=0;p<5;p++)$display("FFRREAL_PHASE phase=%0d cycles=%0d reads=%0d writes=%0d",p,phase_cycles[p],phase_reads[p],phase_writes[p]);
        $display("FFRREAL_PASS jobs=1 seed=%0d cycles=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d bytes=%0d ffn_done=%0d first_res_ar=%0d last_res_b=%0d ar_stalls=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d",seed,cycles,ar_count,r_count,aw_count,w_count,b_count,byte_count,ffn_done_cycle,first_res_ar,last_res_b,ar_stalls,aw_stalls,w_stalls,done_stalls);
        @(negedge clk);completion_ready=0;repeat(3)@(negedge clk);
        require(command_ready && !busy && !completion_valid && !result_valid && !residual_selected,"Final reusable state");
        $display("FFRREAL_REUSABLE jobs=1");
    endtask
    initial begin
        int fd,items,samples,fixture_dim,fixture_hidden;string magic;
        clk=0;rst_n=0;command_valid=0;cancel_valid=0;completion_ready=0;expected_epoch=0;
        m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;m_axi_rvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;m_axi_bvalid=0;m_axi_bresp=0;m_axi_bid=0;
        if($value$plusargs("SEED=%d",seed))begin end
        rng=seed;require($value$plusargs("FIXTURE=%s",fixture_dir) && $value$plusargs("EXTRA=%s",extra_dir),"Missing fixture directories");
        fd=$fopen({fixture_dir,"/jobs.txt"},"r");require(fd!=0,"Fixture descriptor open");items=$fscanf(fd,"%s %d %d %d",magic,samples,fixture_dim,fixture_hidden);$fclose(fd);
        require(items==4 && magic=="NTRFFNMEM01" && samples==1 && fixture_dim==2048 && fixture_hidden==8192,"Complete real-weight fixture geometry");
        $display("FFRREAL_CONFIG dim=2048 hidden=8192 tile_k=32 weight_bytes=53477376 input_abi_requantized=0");
        repeat(4)@(negedge clk);rst_n=1;run_complete();$finish;
    end
endmodule
`default_nettype wire
