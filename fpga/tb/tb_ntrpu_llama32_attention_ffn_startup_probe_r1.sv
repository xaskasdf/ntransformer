`timescale 1ns/1ps
`default_nettype none
// Full arithmetic hierarchy. Q/K/V are external; all downstream intermediates require B commit.
module tb_ntrpu_llama32_attention_ffn_startup_probe_r1;
    logic clk;
    logic rst_n;
    logic bind_valid;
    logic bind_ready;
    logic [63:0] bind_cache_base;
    logic [63:0] bind_region_base;
    logic [63:0] bind_region_bytes;
    logic [31:0] bind_capacity;
    logic [63:0] bind_sequence;
    logic [63:0] bind_epoch;
    logic [63:0] expected_epoch;
    logic cache_valid;
    logic [63:0] cache_base;
    logic [63:0] cache_region_base;
    logic [63:0] cache_region_bytes;
    logic [63:0] cache_sequence;
    logic [63:0] cache_epoch;
    logic [31:0] cache_capacity;
    logic append_valid;
    logic append_ready;
    logic append_prepared;
    logic [31:0] append_layer;
    logic [31:0] append_position;
    logic [63:0] append_sequence;
    logic [63:0] append_epoch;
    logic data_valid;
    logic data_ready;
    logic [511:0] data_bits;
    logic [5:0] data_index;
    logic data_last;
    logic [63:0] data_sequence;
    logic [63:0] data_epoch;
    logic commit_valid;
    logic commit_ready;
    logic [3:0] commit_status;
    logic [63:0] commit_sequence;
    logic [63:0] commit_epoch;
    logic append_cancel_valid;
    logic append_cancel_ready;
    logic append_done_valid;
    logic append_done_ready;
    logic append_done_bind;
    logic [3:0] append_done_status;
    logic [12:0] append_done_bytes;
    logic [31:0] append_done_committed_tokens;
    logic [63:0] append_done_sequence;
    logic [63:0] append_done_epoch;
    logic lease_active;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize;
    logic [2:0] m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid;
    logic [3:0] m_axi_awcache;
    logic m_axi_awlock;
    logic m_axi_awvalid;
    logic m_axi_awready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast;
    logic m_axi_wvalid;
    logic m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic m_axi_bvalid;
    logic m_axi_bready;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize;
    logic [2:0] m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid;
    logic [3:0] m_axi_arcache;
    logic m_axi_arlock;
    logic m_axi_arvalid;
    logic m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast;
    logic m_axi_rvalid;
    logic m_axi_rready;
    logic command_valid;
    logic command_ready;
    logic [31:0] command_position;
    logic [31:0] command_layer;
    logic [63:0] command_sequence;
    logic [63:0] command_epoch;
    logic cancel_valid;
    logic cancel_ready;
    logic completion_valid;
    logic completion_ready;
    logic [3:0] completion_status;
    logic result_valid;
    logic [63:0] command_lease_base;
    logic q_valid;
    logic q_ready;
    logic [511:0] q_data;
    logic [7:0] q_index;
    logic q_last;
    logic [63:0] q_sequence;
    logic [63:0] q_epoch;
    logic [63:0] completion_sequence;
    logic [63:0] completion_epoch;
    logic cache_lock;
    logic maintenance_busy;
    logic metadata_fault;
    logic [31:0] committed_tokens;
    logic busy;
    logic tag_exhausted;
    logic [31:0] completion_hls_status;
    logic [63:0] result_address;
    logic [13:0] result_bytes;
    logic [63:0] command_weight_addr;
    logic [63:0] command_weight_region_base;
    logic [63:0] command_weight_region_bytes;
    logic [63:0] command_activation_addr;
    logic [63:0] command_activation_region_base;
    logic [63:0] command_activation_region_bytes;
    logic [63:0] command_output_addr;
    logic [63:0] command_output_region_base;
    logic [63:0] command_output_region_bytes;
    logic [63:0] command_post_input_address;
    logic [63:0] command_post_input_base;
    logic [63:0] command_post_input_bytes;
    logic [63:0] command_post_weight_address;
    logic [63:0] command_post_weight_base;
    logic [63:0] command_post_weight_bytes;
    logic [63:0] command_post_residual_address;
    logic [63:0] command_post_residual_base;
    logic [63:0] command_post_residual_bytes;
    logic [63:0] command_post_ffn_address;
    logic [63:0] command_post_ffn_base;
    logic [63:0] command_post_ffn_bytes;
    logic [31:0] completion_post_hls_status;
    logic [9:0] completion_post_read_beats;
    logic [7:0] completion_residual_beats;
    logic [6:0] completion_ffn_beats;
    logic [3:0] completion_ffn_status;
    logic [3:0] completion_ffn_child_status;
    logic [3:0] completion_residual_status;
    logic [2:0] completion_ffn_stages;
    logic [1:0] completion_ffn_phase;
    logic [16:0] completion_ffn_bytes;
    logic [17:0] completion_ffn_stage_bytes;
    logic [31:0] completion_ffn_hls_status;
    logic [9:0] completion_read_beats;
    logic [7:0] completion_write_beats;
    logic activation_selected;
    logic residual_selected;
    logic [3:0] completion_attention_status;
    logic [3:0] completion_ffn_residual_status;
    logic [31:0] completion_attention_hls_status;
    logic ffn_selected;
    wire [63:0] command_ffn_addr [1:7];
    wire [63:0] command_ffn_base [1:7];
    wire [63:0] command_ffn_bytes [1:7];
    logic [63:0] command_final_address;
    logic [63:0] command_final_base;
    logic [63:0] command_final_bytes;
    logic [7:1][63:0] f_addr,f_base,f_bytes;
    for(genvar i=1;i<8;i++)begin
        assign command_ffn_addr[i]=f_addr[i];assign command_ffn_base[i]=f_base[i];assign command_ffn_bytes[i]=f_bytes[i];
    end
    localparam logic [63:0] WBASE=64'h100000000,ABASE=64'h8000000000,OBASE=64'ha000000000,KBASE=64'hc000000000;
    localparam logic [63:0] FIRST_P=64'h00000000ffffffc0;
    ntrpu_llama32_attention_ffn_r4 #(.INITIAL_QUANT_TAG(FIRST_P),.INITIAL_PROJECTION_TAG(FIRST_P),.INITIAL_FFN_TAG(64'hfffffffc)) dut(.*);
    logic [511:0] weight_image[0:69694],activation_image[0:255],destination[0:127];
    logic [63:0] reference64[0:8191];logic [31:0] reference32[0:8191],fp32_image[0:8191];
    logic [511:0] fixture_kv[0:47807],kv_memory[0:8191],materialized_activation[0:63],pending_data[0:3];
    logic [31:0] fixture_query[0:9151];
    bit kv_written[0:8191];logic [63:0] activation_written=0;logic [3:0] pending_last;
    logic [2047:0] seen_source;
    string fixture,root_fixture;
    integer seed=1,rng=1,cycles=0,jobs=0,mode_now=0,sample_now=0,publics=0,normal_starts=0;
    // Update the timestamp before triggering posedge monitors, avoiding active-region races.
    always begin
        #2.5;
        if(!clk)cycles=cycles+1;
        clk=~clk;
    end
    integer ars=0,rs=0,aws=0,ws=0,bs=0,groups=0,tiles=0,computed=0,replies=0;
    integer qaws=0,qws=0,qbs=0,kars=0,krs=0,kaws=0,kws=0,kbs=0;
    integer total_ar=0,total_r=0,total_aw=0,total_w=0,total_b=0,total_results=0;
    integer total_qaw=0,total_qw=0,total_qb=0,total_kar=0,total_kr=0,total_kaw=0,total_kw=0,total_kb=0;
    integer quant_words=0,source_reads=0,root_acks=0,qacks=0,packs=0,outacks=0,starts=0,dones=0;
    integer total_quant_words=0,total_source_reads=0,total_root=0,total_qacks=0,total_packs=0,total_outacks=0;
    integer heads=0,head_replies=0,copies=0,head_copies=0,qbeats=0,binds=0,appends=0,total_heads=0,total_copies=0,total_qbeats=0;
    integer append_pos=0,append_aw_cursor=0,append_w_cursor=0;
    integer r_index=0,r_beats=0,aw_beats=0,w_pending=0,write_cursor=0,aw_number=0,aw_kind=0;
    integer quiet=0,max_quiet=0,ar_stalls=0,r_stalls=0,aw_stalls=0,w_stalls=0,pending_input_address=0;
    logic r_active=0,r_is_kv=0,aw_pending=0,input_pending=0,quant_public_seen=0,root_public_seen=0;
    logic hold_ar=0,hold_aw=0,hold_w=0,hold_r=0;
    logic [63:0] r_address,aw_address,sequence_model,epoch_model,tag_model,lease_model;
    logic [88:0] saved_ar,saved_aw;logic [576:0] saved_w;logic [518:0] saved_r;

    localparam logic [63:0] PIB=64'ha100000000,PWB=64'ha200000000,PRB=64'ha300000000,PQB=64'ha400000000;
    logic [31:0] post_source[0:24575],post_residual[0:8191];logic [63:0] post_abi[0:2047];
    logic [511:0] committed_O[0:127],post_destination[0:191];logic [127:0] committed_O_valid;
    integer pars=0,prs=0,paws=0,pws=0,pbs=0,post_starts=0,post_dones=0,post_quiet=0,post_max_quiet=0;
    integer total_par=0,total_pr=0,total_pb=0,post_pairs=0;
    logic r_is_post=0,post_core_busy=0;
    string post_fixture;
    function automatic logic [511:0] post_output_word(input integer index);
        logic [511:0] value;
        if(index<128)for(integer lane=0;lane<16;lane++)value[lane*32+:32]=post_residual[sample_now*2048+index*16+lane];
        else for(integer lane=0;lane<8;lane++)value[lane*64+:64]=post_abi[sample_now*512+(index-128)*8+lane];
        return value;
    endfunction
    function automatic logic [63:0] post_output_address(input integer index);
        return index<128 ? PRB+4032+64'(index*64):PQB+4032+64'((index-128)*64);
    endfunction
    wire source_ce=dut.attention.projected.root.result_ce;
    wire [10:0] source_address=dut.attention.projected.root.result_address;
    function automatic void require(input bit ok,input string why);
        if(!ok)$fatal(1,"PRREAL_FAIL cycle=%0d job=%0d mode=%0d: %s",cycles,jobs,mode_now,why);
    endfunction
    task automatic tick;@(posedge clk);#1;endtask
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

    // The FFN input and residual buffers start poisoned, just like all other outputs.
    localparam int FLIMIT=73728;
    logic [7:0] fmem[0:9][0:FLIMIT-1],golden[0:4][0:65535],fbytes[0:FLIMIT-1];
    bit fwritten[0:9][0:FLIMIT-1];
    logic [63:0] bases[0:9],read_hash[0:9];
    int spans[0:9],read_beats_by_buffer[0:9];
    string ffn_fixture,extra_fixture;
    int far=0,fr=0,faw=0,fw=0,fb=0,fstored=0;
    int fphase_reads[0:4],fphase_writes[0:4],fphase_cycles[0:4];
    int ffn_done_cycle=0,first_res_ar=0,last_res_b=0,attention_done_cycle=0;
    int fhls_starts=0,fhls_dones=0,fhls_words=0,source_reads_abi=0,source_reads_residual=0;
    bit seen_ffn=0,seen_res=0,fhls_busy=0,old_ffn=0,old_residual=0,old_activation=0,seen_attention=0;
    bit post_checked=0,complete_seen=0;
    int final_acks=0,attention_acks=0,ffn_acks=0,ffn_admits=0;
    wire [1:0] observed_phase=dut.ffn.ffn.core.controller.phase;
    wire [2:0] fphase=residual_selected ? 3'd4:{1'b0,observed_phase};
    function automatic int fbuffer(input logic [63:0] address);
        for(int k=0;k<10;k++)if(address>=bases[k]+4032 && address<bases[k]+64'(4032+spans[k]))return k;
        $fatal(1,"AFREAL_FAIL unknown FFN address %h",address);return -1;
    endfunction
    function automatic logic [511:0] f_read_word(input logic [63:0] address);
        logic [511:0] value;int k,off;
        k=fbuffer(address);off=int'(address-bases[k]);
        require(off>=4032 && off+64<=4032+spans[k],"FFN read bounds");
        for(int lane=0;lane<64;lane++)begin
            if(k<1 || k>3)require(fwritten[k][off+lane],"FFN read before producer B commit");
            value[lane*8+:8]=k>=1 && k<=3 ? llama_weight_byte(k,off+lane):fmem[k][off+lane];
        end
        return value;
    endfunction
    function automatic logic [7:0] f_expected(input int k,input int off);
        require(k==4 || k==5 || k==6 || k==7 || k==9,"Expected output buffer");
        return golden[k==9 ? 4:k-4][off];
    endfunction
    task automatic prepare_ffn;
        for(int k=0;k<10;k++)begin
            bases[k]=64'h100000000+64'(k)*64'h20000000;
            for(int off=0;off<FLIMIT;off++)begin fmem[k][off]=8'ha5;fwritten[k][off]=0;end
            read_hash[k]=64'hcbf29ce484222325;read_beats_by_buffer[k]=0;
        end
        bases[0]=PQB;bases[8]=PRB;bases[9]=64'h0000710000000000;
        spans[0]=4096;spans[1]=17825792;spans[2]=17825792;spans[3]=17825792;
        spans[4]=65536;spans[5]=65536;spans[6]=16384;spans[7]=8192;spans[8]=8192;spans[9]=8192;
        $readmemh({ffn_fixture,"/weight-1.hex"},llama_gate_weights,0,17829823);
        $readmemh({ffn_fixture,"/weight-2.hex"},llama_up_weights,0,17829823);
        $readmemh({ffn_fixture,"/weight-3.hex"},llama_down_weights,0,17829823);
        for(int k=1;k<=3;k++)verify_llama_weights(k);
        for(int k=4;k<8;k++)begin
            if(k==7)$readmemh({ffn_fixture,"/job-001-final-fp32.hex"},fbytes,0,spans[k]-1);
            else $readmemh($sformatf("%s/job-001-buffer-%0d.hex",ffn_fixture,k),fbytes,0,spans[k]-1);
            for(int off=0;off<spans[k];off++)golden[k-4][off]=fbytes[off];
        end
        $readmemh({extra_fixture,"/layer-output.hex"},fbytes,0,8191);
        for(int off=0;off<8192;off++)golden[4][off]=fbytes[off];
        for(int p=0;p<5;p++)begin fphase_cycles[p]=0;fphase_reads[p]=0;fphase_writes[p]=0;end
    endtask

    function automatic logic [511:0] read_word(input logic [63:0] address);

        logic [511:0] value;
        if(ffn_selected)return f_read_word(address);
        if(address>=OBASE+4032 && address+64<=OBASE+12224)begin
            if(!committed_O_valid[int'((address-OBASE-4032)>>6)])$fatal(1,"APREAL_FAIL O read before B");
            return committed_O[int'((address-OBASE-4032)>>6)];
        end
        if((address>=PIB+4032 && address+64<=PIB+12224) || (address>=PWB+4032 && address+64<=PWB+12224))begin
            for(integer lane=0;lane<16;lane++)begin
                if(address<PWB)value[lane*32+:32]=post_source[sample_now*6144+int'((address-PIB-4032)/4)+lane];
                else value[lane*32+:32]=post_source[sample_now*6144+4096+int'((address-PWB-4032)/4)+lane];
            end
            return value;
        end
        if(address>=KBASE && address+64<=KBASE+524288)begin
            if(!kv_written[int'((address-KBASE)>>6)])$fatal(1,"PRREAL_FAIL KV read before commit");
            return kv_memory[int'((address-KBASE)>>6)];
        end
        if(address>=WBASE+4032 && address+64<=WBASE+4460480)return weight_image[int'((address-WBASE)>>6)];
        if(!(address>=ABASE+4032 && address+64<=ABASE+8128))$fatal(1,"PRREAL_FAIL invalid read address");
        if(!activation_written[int'((address-ABASE-4032)>>6)])$fatal(1,"PRREAL_FAIL ABI1 read before commit");
        return materialized_activation[int'((address-ABASE-4032)>>6)];
    endfunction
    function automatic logic [511:0] output_word(input integer index);
        logic [511:0] value;
        for(integer i=0;i<16;i++)value[i*32+:32]=reference32[sample_now*2048+index*16+i];
        return value;
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_arready=0;m_axi_rvalid=0;m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;end
        else begin
            m_axi_arready=$urandom_range(0,3)!=0;m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if(r_active && !m_axi_rvalid && $urandom_range(0,3)!=0)begin
                m_axi_rdata=read_word(r_address+64'(r_index*64));m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=r_index==r_beats-1;m_axi_rvalid=1;
            end
            if(aw_pending && w_pending==aw_beats && !m_axi_bvalid && $urandom_range(0,3)!=0)begin
                for(integer i=0;i<w_pending;i++)require(pending_last[i]==(i==aw_beats-1),"W burst last");
                m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;
            end
        end
    end
    always @(posedge clk)begin
        if(!rst_n)begin hold_ar=0;hold_r=0;hold_aw=0;hold_w=0;r_active=0;aw_pending=0;w_pending=0;input_pending=0;end
        else begin
            if(busy)require(cache_lock && !bind_ready && !append_ready && !dut.attention.projected.root.bind_valid && !dut.attention.projected.root.append_valid,"Composed maintenance lock");
            if(command_valid && command_ready)
                $display("PRREAL_COMMAND cycle=%0d job=%0d position=%0d sequence=%016h epoch=%016h lease=%016h",cycles,jobs,command_position,command_sequence,command_epoch,command_lease_base);
            if(q_valid && q_ready)begin
                $display("PRREAL_Q cycle=%0d job=%0d index=%0d data=%0128h",cycles,jobs,q_index,q_data);qbeats++;total_qbeats++;
            end
            if(data_valid && data_ready)
                $display("PRREAL_DATA cycle=%0d job=%0d position=%0d index=%0d data=%0128h",cycles,jobs,append_pos,data_index,data_bits);
            if(dut.attention.projected.root.head_command_valid && dut.attention.projected.root.head_command_ready)begin
                require(mode_now==0 && qbeats==128 && heads==head_replies && heads<32,"Head owner/shape");
                require(dut.attention.projected.root.head_command_head==32'(heads) && dut.attention.projected.root.head_command_position==32'(sample_now) && dut.attention.projected.root.head_command_layer==0 &&
                    dut.attention.projected.root.head_command_sequence==sequence_model && dut.attention.projected.root.head_command_epoch==epoch_model && dut.attention.projected.root.head_command_lease==lease_model+64'(heads),"Head descriptor/identity");
                for(integer lane=0;lane<64;lane++)require(dut.attention.projected.root.head_command_query[lane*32+:32]===fixture_query[sample_now*2048+heads*64+lane],"Head Q content");
                $display("PRREAL_HEAD cycle=%0d job=%0d head=%0d sequence=%016h epoch=%016h lease=%016h",cycles,jobs,heads,sequence_model,epoch_model,dut.attention.projected.root.head_command_lease);
                heads++;total_heads++;head_copies=0;
            end
            if(dut.attention.projected.root.head_completion_valid && dut.attention.projected.root.head_completion_ready)begin
                require(heads==head_replies+1 && head_copies==64 && dut.attention.projected.root.head_completion_status==0 && !lease_active && !r_active,"Head completion drain");
                $display("PRREAL_HEAD_DONE cycle=%0d job=%0d head=%0d status=%0d reads=%0d",cycles,jobs,head_replies,dut.attention.projected.root.head_completion_status,head_copies);head_replies++;
            end
            if(dut.attention.projected.root.completion_valid && !root_public_seen)begin
                require(dut.attention.projected.root.completion_sequence==sequence_model && dut.attention.projected.root.completion_epoch==epoch_model && !r_active && !aw_pending && w_pending==0,"Root publication identity/drain");
                require(mode_now==0 ? (heads==32 && copies==2048 && dut.attention.projected.root.result_valid && dut.attention.projected.root.completion_status==0):
                    (heads==0 && copies==0 && !dut.attention.projected.root.result_valid && dut.attention.projected.root.completion_status==7),"Root publication arithmetic/cancel");
                root_public_seen=1;
                $display("PRREAL_ROOT_PUBLIC cycle=%0d job=%0d status=%0d sequence=%016h epoch=%016h valid=%0d",cycles,jobs,dut.attention.projected.root.completion_status,dut.attention.projected.root.completion_sequence,dut.attention.projected.root.completion_epoch,dut.attention.projected.root.result_valid);
            end
            if(input_pending)begin
                require(dut.attention.projected.output_path.quantize_store.hls_input_q===fp32_image[sample_now*2048+pending_input_address],"Real synchronous root to quantizer data");
                $display("PRREAL_INPUT cycle=%0d job=%0d index=%0d data=%08h",cycles,jobs,pending_input_address,dut.attention.projected.output_path.quantize_store.hls_input_q);
            end
            input_pending=source_ce;pending_input_address=int'(source_address);
            if(source_ce)begin
                require(mode_now==0 && root_acks==0 && root_public_seen && dut.attention.projected.root.result_valid && dut.attention.projected.output_owned,"Source read ownership");
                seen_source[source_address]=1;source_reads++;total_source_reads++;
                $display("PRREAL_READ cycle=%0d job=%0d index=%0d",cycles,jobs,source_address);
            end
            if(dut.attention.projected.output_path.quantize_store.hls_start)begin
                starts++;require(starts==1 && dut.attention.projected.output_path.quantize_store.hls_elements==2048,"Quantizer extent");
                $display("PRREAL_START cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.attention.projected.output_path.quantize_store.hls_output_ce && dut.attention.projected.output_path.quantize_store.hls_output_we)begin
                require(quant_words<512 && dut.attention.projected.output_path.quantize_store.hls_output_address==9'(quant_words),"Quantizer order");
                require(dut.attention.projected.output_path.quantize_store.hls_output_d===activation_image[sample_now*64+quant_words/8][(quant_words%8)*64+:64],"Generated ABI1 bytes");
                $display("PRREAL_WORD cycle=%0d job=%0d index=%0d data=%016h",cycles,jobs,quant_words,dut.attention.projected.output_path.quantize_store.hls_output_d);quant_words++;total_quant_words++;
            end
            if(dut.attention.projected.output_path.quantize_store.hls_done)begin
                dones++;require(dones==1 && dut.attention.projected.output_path.quantize_store.hls_ready && dut.attention.projected.output_path.quantize_store.hls_return==0 && quant_words==512 && seen_source=='1,"Quantizer result coverage");
                $display("PRREAL_DONE cycle=%0d job=%0d status=%0d words=%0d",cycles,jobs,dut.attention.projected.output_path.quantize_store.hls_return,quant_words);
            end
            if(dut.attention.projected.root.completion_valid && dut.attention.projected.root.completion_ready)begin
                require(root_acks==0 && (mode_now==0 ? dones==1:(completion_valid && completion_ready)),"Root release before consumer drain");
                root_acks++;total_root++;
                $display("PRREAL_ROOT_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.attention.projected.root.completion_sequence,dut.attention.projected.root.completion_epoch);
            end
            if(dut.attention.projected.output_path.q_completion_valid && !quant_public_seen)begin
                require(qaws==64 && qws==64 && qbs==64 && activation_written=='1 && !aw_pending && w_pending==0,"Quant publication before commit");
                require(dut.attention.projected.output_path.q_completion_status==0 && dut.attention.projected.output_path.q_completion_hls_status==0 && dut.attention.projected.output_path.q_completion_records==64 && dut.attention.projected.output_path.q_result_valid &&
                    dut.attention.projected.output_path.q_result_address==ABASE+4032 && dut.attention.projected.output_path.q_result_bytes==4096,"Quantized region contract");quant_public_seen=1;
                $display("PRREAL_QUANT_PUBLIC cycle=%0d job=%0d status=%0d hls=%0d sequence=%016h epoch=%016h records=%0d valid=%0d address=%016h bytes=%0d",cycles,jobs,
                    dut.attention.projected.output_path.q_completion_status,dut.attention.projected.output_path.q_completion_hls_status,dut.attention.projected.output_path.q_completion_sequence,dut.attention.projected.output_path.q_completion_epoch,
                    dut.attention.projected.output_path.q_completion_records,dut.attention.projected.output_path.q_result_valid,dut.attention.projected.output_path.q_result_address,dut.attention.projected.output_path.q_result_bytes);
            end
            if(dut.attention.projected.output_path.projection.command_valid && dut.attention.projected.output_path.projection.command_ready)begin
                require(quant_public_seen && qacks==0 && dut.attention.projected.output_path.q_completion_valid,"O without retained ABI1");
                $display("PRREAL_PROJECTION_COMMAND cycle=%0d job=%0d sequence=%016h epoch=%016h address=%016h",cycles,jobs,
                    dut.attention.projected.output_path.projection.command_sequence,dut.attention.projected.output_path.projection.command_epoch,dut.attention.projected.output_path.projection.command_activation_addr);
            end
            if(dut.attention.projected.output_path.q_completion_valid && dut.attention.projected.output_path.quantize_store.completion_ready)begin
                require(completion_valid && completion_ready && !r_active && !aw_pending && w_pending==0,"Early quant owner release");qacks++;total_qacks++;
                $display("PRREAL_QACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.attention.projected.output_path.q_completion_sequence,dut.attention.projected.output_path.q_completion_epoch);
            end
            if(dut.attention.projected.output_path.p_completion_valid && dut.attention.projected.output_path.projection.completion_ready)begin
                require(completion_valid && completion_ready && dut.attention.projected.output_path.quantize_store.completion_ready,"Split output owner release");packs++;total_packs++;
                $display("PRREAL_PACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.attention.projected.output_path.p_completion_sequence,dut.attention.projected.output_path.p_completion_epoch);
            end
            if(dut.attention.projected.output_path.completion_valid && dut.attention.projected.output_path.completion_ready)begin
                outacks++;total_outacks++;$display("PRREAL_OUT_ACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.attention.projected.output_path.projection.control.active_child)begin
                if(dut.attention.projected.output_path.projection.child_activity)quiet=0;else quiet++;
                if(quiet>max_quiet)max_quiet=quiet;
            end else quiet=0;
            if(hold_ar)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock}===saved_ar,"Held AR changed");
            if(hold_aw)require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"Held AW changed");
            if(hold_w)require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"Held W changed");
            if(hold_r)require(m_axi_rvalid && {m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast}===saved_r,"Held R changed");
            hold_ar=m_axi_arvalid && !m_axi_arready;hold_aw=m_axi_awvalid && !m_axi_awready;hold_w=m_axi_wvalid && !m_axi_wready;hold_r=m_axi_rvalid && !m_axi_rready;
            saved_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};saved_r={m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast};
            ar_stalls+=int'(hold_ar);aw_stalls+=int'(hold_aw);w_stalls+=int'(hold_w);r_stalls+=int'(hold_r);
            if(dut.attention.projected.output_path.projection.child_job_valid && dut.attention.projected.output_path.projection.child_job_ready)begin
                require({dut.attention.projected.output_path.projection.child_job_epoch,dut.attention.projected.output_path.projection.child_job_sequence}==tag_model,"Matrix identity");
                require(dut.attention.projected.output_path.projection.child_job_rows==2048 && dut.attention.projected.output_path.projection.child_job_k_blocks==64 && dut.attention.projected.output_path.projection.child_job_tile_k==32 &&
                    dut.attention.projected.output_path.projection.child_job_layout==2 && dut.attention.projected.output_path.projection.child_job_output_fp32,"Full O shape");normal_starts++;
                $display("PRREAL_MATRIX cycle=%0d job=%0d tag=%016h",cycles,jobs,tag_model);
            end
            if(dut.attention.projected.output_path.projection.matrix.e_job_valid && dut.attention.projected.output_path.projection.matrix.e_job_ready)begin
                require(groups==replies && !r_active && !aw_pending && w_pending==0,"O group overlap");
                require(dut.attention.projected.output_path.projection.matrix.e_job_rows==32 && dut.attention.projected.output_path.projection.matrix.e_job_output_fp32 && dut.attention.projected.output_path.projection.matrix.e_job_k_blocks==64 &&
                    {dut.attention.projected.output_path.projection.matrix.e_job_epoch,dut.attention.projected.output_path.projection.matrix.e_job_sequence}==tag_model+64'(groups),"O group identity/shape");
                $display("PRREAL_GROUP cycle=%0d job=%0d group=%0d tag=%016h",cycles,jobs,groups,{dut.attention.projected.output_path.projection.matrix.e_job_epoch,dut.attention.projected.output_path.projection.matrix.e_job_sequence});groups++;
            end
            if(dut.attention.projected.output_path.projection.matrix.e_tile_valid && dut.attention.projected.output_path.projection.matrix.e_tile_ready)begin
                require(dut.attention.projected.output_path.projection.matrix.e_tile_addr==WBASE+4032+64'(tiles*34816) && dut.attention.projected.output_path.projection.matrix.e_tile_k_count==32 &&
                    dut.attention.projected.output_path.projection.matrix.e_tile_k_base==9'((tiles%2)*32),"O tile order");
                $display("PRREAL_TILE cycle=%0d job=%0d tile=%0d address=%016h",cycles,jobs,tiles,dut.attention.projected.output_path.projection.matrix.e_tile_addr);tiles++;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!r_active && m_axi_araddr[5:0]==0 && int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR admission/alignment/page");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR attributes");
                r_address=m_axi_araddr;r_beats=int'(m_axi_arlen)+1;r_index=0;r_active=1;r_is_post=dut.attention.post_owned;r_is_kv=!dut.attention.projected.output_owned;
                if(ffn_selected)begin
                    require(fphase==0 ? fbuffer(r_address)==0 || fbuffer(r_address)==1:fphase==1 ? fbuffer(r_address)==0 || fbuffer(r_address)==2:fphase==2 ? fbuffer(r_address)==4 || fbuffer(r_address)==5:fphase==3 ? fbuffer(r_address)==6 || fbuffer(r_address)==3:fbuffer(r_address)==8 || fbuffer(r_address)==7,"FFN read phase");
                    if(fphase==4)begin require(seen_ffn && seen_res,"Residual admission before AR");if(first_res_ar==0)first_res_ar=cycles;end
                    $display("FFRREAL_AR cycle=%0d phase=%0d buffer=%0d offset=%0d beats=%0d",cycles,fphase,fbuffer(r_address),r_address-bases[fbuffer(r_address)]-4032,r_beats);far++;
                end else
                if(r_is_post)begin
                    require(committed_O_valid=='1 && dut.attention.projected.completion_valid && !dut.attention.projected.completion_ready,"Post input ownership");
                    $display("APREAL_AR cycle=%0d job=%0d address=%016h beats=%0d",cycles,jobs,r_address,r_beats);pars++;total_par++;
                end else if(r_is_kv)begin
                    require(heads==head_replies+1 && r_address>=KBASE && r_address+64'(r_beats*64)<=KBASE+270336,"KV AR owner/range");
                    $display("PRREAL_KAR cycle=%0d job=%0d ar=%0d address=%016h beats=%0d head=%0d",cycles,jobs,kars,r_address,r_beats,heads-1);kars++;total_kar++;
                end else begin
                    require(groups>0 && qbs==64 && qacks==0 && dut.attention.projected.output_path.q_completion_valid && activation_written=='1,"O AR ownership");
                    $display("PRREAL_AR cycle=%0d job=%0d ar=%0d address=%016h beats=%0d group=%0d",cycles,jobs,ars,r_address,r_beats,groups-1);ars++;total_ar++;
                end
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(r_active && m_axi_rlast==(r_index==r_beats-1) && r_is_kv==!dut.attention.projected.output_owned && r_is_post==dut.attention.post_owned,"R owner/burst");
                if(ffn_selected)begin: f_read
                    int k,off;k=fbuffer(r_address+64'(r_index*64));off=int'(r_address+64'(r_index*64)-bases[k]-4032);
                    read_beats_by_buffer[k]++;
                    for(int word_index=0;word_index<8;word_index++)read_hash[k]=(read_hash[k]^m_axi_rdata[word_index*64+:64])*64'h100000001b3;
                    require(residual_selected ? dut.ffn.res_m_axi_rvalid && !dut.ffn.ffn_m_axi_rvalid:dut.ffn.ffn_m_axi_rvalid && !dut.ffn.res_m_axi_rvalid,"FFN R isolation");
                    if(k==0 || k==8)begin
                        $display("AFREAL_SOURCE_R cycle=%0d buffer=%0d offset=%0d data=%0128h",cycles,k,off,m_axi_rdata);
                        if(k==0)source_reads_abi++;else source_reads_residual++;
                    end
                    if(fphase==4)$display("FFRREAL_RES_R cycle=%0d buffer=%0d offset=%0d data=%0128h",cycles,k,off,m_axi_rdata);
                    fr++;fphase_reads[fphase]++;
                end else if(r_is_post)begin
                    $display("APREAL_R cycle=%0d job=%0d index=%0d address=%016h data=%0128h",cycles,jobs,prs,r_address+64'(r_index*64),m_axi_rdata);prs++;total_pr++;
                end else if(r_is_kv)begin $display("PRREAL_KR cycle=%0d job=%0d ar=%0d beat=%0d address=%016h data=%0128h last=%0d",cycles,jobs,kars-1,r_index,r_address+64'(r_index*64),m_axi_rdata,m_axi_rlast);krs++;total_kr++;end
                else begin $display("PRREAL_R cycle=%0d job=%0d ar=%0d beat=%0d address=%016h data=%0128h last=%0d",cycles,jobs,ars-1,r_index,r_address+64'(r_index*64),m_axi_rdata,m_axi_rlast);rs++;total_r++;end
                r_index++;if(m_axi_rlast)r_active=0;m_axi_rvalid<=0;
            end
            if(dut.attention.projected.output_path.projection.matrix.engine.result_valid && dut.attention.projected.output_path.projection.matrix.engine.result_ready)begin
                require(dut.attention.projected.output_path.projection.matrix.engine.result_row==5'(computed%32) && dut.attention.projected.output_path.projection.matrix.engine.result_data===reference64[sample_now*2048+computed],"O FP64 numerical result");
                $display("PRREAL_MAC cycle=%0d job=%0d row=%0d data=%016h",cycles,jobs,computed,dut.attention.projected.output_path.projection.matrix.engine.result_data);computed++;total_results++;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awaddr[5:0]==0 && int'(m_axi_awaddr[11:0])+(int'(m_axi_awlen)+1)*64<=4096,"AW admission/alignment/page");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                aw_kind=ffn_selected ? 4:dut.attention.post_owned ? 3:!dut.attention.projected.output_owned ? 0:dut.attention.projected.output_path.projection_owned ? 2:1;
                if(aw_kind==4)begin: f_aw
                    int k;k=fbuffer(m_axi_awaddr);
                    require(k==(fphase==4 ? 9:int'(fphase)+4) && m_axi_awlen<=3,"FFN AW phase/extent");
                    if(fphase==4)require(m_axi_awlen==0 && fhls_dones==1,"Residual store contract");
                    $display("FFRREAL_AW cycle=%0d phase=%0d buffer=%0d offset=%0d beats=%0d",cycles,fphase,k,m_axi_awaddr-bases[k]-4032,int'(m_axi_awlen)+1);faw++;
                end else if(aw_kind==3)begin
                    require(m_axi_awlen==0 && paws==pbs && m_axi_awaddr==post_output_address(paws),"Post AW ownership/address");
                    aw_number=paws;paws++;
                end else if(aw_kind==0)begin
                    require(m_axi_awlen<=3 && m_axi_awaddr==KBASE+4032+64'(((append_aw_cursor/4)*4+append_pos)*256+(append_aw_cursor%4)*64),"KV AW address/order");
                    aw_number=kaws;append_aw_cursor+=int'(m_axi_awlen)+1;
                    $display("PRREAL_KAW cycle=%0d job=%0d aw=%0d position=%0d address=%016h beats=%0d",cycles,jobs,kaws,append_pos,m_axi_awaddr,int'(m_axi_awlen)+1);kaws++;total_kaw++;
                end else if(aw_kind==1)begin
                    require(root_acks==1 && qaws<64 && qaws==qbs && m_axi_awaddr==ABASE+4032+64'(qaws*64) && m_axi_awlen==0,"ABI1 AW address/order");aw_number=qaws;
                    $display("PRREAL_QAW cycle=%0d job=%0d record=%0d address=%016h",cycles,jobs,qaws,m_axi_awaddr);qaws++;total_qaw++;
                end else begin
                    require(m_axi_awaddr==OBASE+4032+64'(write_cursor*64) && m_axi_awlen<=1,"O AW address/order");aw_number=aws;write_cursor+=int'(m_axi_awlen)+1;
                    $display("PRREAL_AW cycle=%0d job=%0d aw=%0d address=%016h beats=%0d",cycles,jobs,aws,m_axi_awaddr,int'(m_axi_awlen)+1);aws++;total_aw++;
                end
                aw_pending=1;aw_beats=int'(m_axi_awlen)+1;aw_address=m_axi_awaddr;
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(w_pending<4 && m_axi_wstrb=='1,"W capacity/mask");
                if(ffn_selected)begin fw++;fphase_writes[fphase]++;end
                else if(dut.attention.post_owned)begin
                    require(pws==pbs && pws<192 && m_axi_wlast && m_axi_wdata===post_output_word(pws),"Actual composed post W bytes");pws++;
                end else if(!dut.attention.projected.output_owned)begin
                    require(append_w_cursor<64 && m_axi_wdata===fixture_kv[append_pos*64+append_w_cursor],"Actual KV write bytes");
                    $display("PRREAL_KW cycle=%0d job=%0d position=%0d index=%0d data=%0128h last=%0d",cycles,jobs,append_pos,append_w_cursor,m_axi_wdata,m_axi_wlast);append_w_cursor++;kws++;total_kw++;
                end else if(!dut.attention.projected.output_path.projection_owned)begin
                    require(root_acks==1 && qws<64 && qws==qbs && m_axi_wlast && m_axi_wdata===activation_image[sample_now*64+qws],"Actual ABI1 write bytes");
                    $display("PRREAL_QW cycle=%0d job=%0d record=%0d data=%0128h",cycles,jobs,qws,m_axi_wdata);qws++;total_qw++;
                end else begin
                    require(ws<128 && m_axi_wdata===output_word(ws),"Actual FP32 projection bytes");destination[ws]=m_axi_wdata;
                    $display("PRREAL_W cycle=%0d job=%0d word=%0d data=%0128h last=%0d",cycles,jobs,ws,m_axi_wdata,m_axi_wlast);ws++;total_w++;
                end
                pending_data[w_pending]=m_axi_wdata;pending_last[w_pending]=m_axi_wlast;w_pending++;
            end
            if(ffn_selected && aw_pending)while(fstored<w_pending)begin: f_w
                int k,off;k=fbuffer(aw_address);off=int'(aw_address-bases[k])+fstored*64;
                require(off>=4032 && off+64<=4032+spans[k] && pending_last[fstored]==(fstored==aw_beats-1),"FFN W bounds/last");
                for(int lane=0;lane<64;lane++)require(!fwritten[k][off+lane] && pending_data[fstored][lane*8+:8]===f_expected(k,off+lane-4032),"FFN full numerical W bytes");
                $display("FFRREAL_W cycle=%0d buffer=%0d offset=%0d strb=ffffffffffffffff data=%0128h",cycles,k,off-4032,pending_data[fstored]);fstored++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending==aw_beats && aw_kind==(ffn_selected ? 4:dut.attention.post_owned ? 3:!dut.attention.projected.output_owned ? 0:dut.attention.projected.output_path.projection_owned ? 2:1),"B owner/drain");
                if(aw_kind==4)begin: f_b
                    int k,off;k=fbuffer(aw_address);
                    require(fstored==w_pending && m_axi_bresp==0 && m_axi_bid==0,"FFN B contract");
                    require(residual_selected ? dut.ffn.res_m_axi_bvalid && !dut.ffn.ffn_m_axi_bvalid:dut.ffn.ffn_m_axi_bvalid && !dut.ffn.res_m_axi_bvalid,"FFN B isolation");
                    for(int beat=0;beat<aw_beats;beat++)begin
                        off=int'(aw_address-bases[k])+beat*64;
                        for(int lane=0;lane<64;lane++)begin
                            require(!fwritten[k][off+lane],"Duplicate FFN B byte");fwritten[k][off+lane]=1;fmem[k][off+lane]=pending_data[beat][lane*8+:8];
                        end
                    end
                    $display("FFRREAL_B cycle=%0d phase=%0d buffer=%0d offset=%0d beats=%0d resp=%0d",cycles,fphase,k,aw_address-bases[k]-4032,aw_beats,m_axi_bresp);
                    if(fphase==4)last_res_b=cycles;fb++;fstored=0;
                end else if(aw_kind==3)begin
                    require(aw_number==pbs && paws==pbs+1 && pws==pbs+1 && aw_address==post_output_address(pbs),"Post B materialization");
                    post_destination[pbs]=pending_data[0];
                    begin: commit_post_to_ffn
                        int k,off;k=pbs<128 ? 8:0;off=4032+(pbs<128 ? pbs:pbs-128)*64;
                        for(int lane=0;lane<64;lane++)begin
                            require(!fwritten[k][off+lane],"Duplicate post source byte");fwritten[k][off+lane]=1;fmem[k][off+lane]=pending_data[0][lane*8+:8];
                        end
                    end
                    $display("APREAL_B cycle=%0d job=%0d index=%0d address=%016h data=%0128h",cycles,jobs,pbs,aw_address,pending_data[0]);pbs++;total_pb++;
                end else if(aw_kind==0)begin
                    for(integer i=0;i<aw_beats;i++)begin
                        require(aw_address>=KBASE && aw_address+64'((i+1)*64)<=KBASE+270336,"KV commit bounds");
                        kv_memory[int'((aw_address-KBASE)>>6)+i]=pending_data[i];kv_written[int'((aw_address-KBASE)>>6)+i]=1;
                    end
                    $display("PRREAL_KB cycle=%0d job=%0d aw=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);kbs++;total_kb++;
                end else if(aw_kind==1)begin
                    require(aw_number==qbs && qaws==qbs+1 && qws==qbs+1 && !activation_written[aw_number],"ABI1 commit order");
                    materialized_activation[aw_number]=pending_data[0];activation_written[aw_number]=1;
                    $display("PRREAL_QB cycle=%0d job=%0d record=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);qbs++;total_qb++;
                end else begin
                    for(integer i=0;i<aw_beats;i++)begin
                        require(aw_address>=OBASE+4032 && aw_address+64'((i+1)*64)<=OBASE+12224,"O materialization bounds");
                        committed_O[int'((aw_address-OBASE-4032)>>6)+i]=pending_data[i];
                        committed_O_valid[int'((aw_address-OBASE-4032)>>6)+i]=1;
                    end
                    $display("PRREAL_B cycle=%0d job=%0d aw=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);bs++;total_b++;end
                aw_pending=0;w_pending=0;m_axi_bvalid<=0;
            end
            if(dut.attention.projected.output_path.projection.matrix.e_done_valid && dut.attention.projected.output_path.projection.matrix.e_done_ready)begin
                require(dut.attention.projected.output_path.projection.matrix.e_done_status==0 && dut.attention.projected.output_path.projection.matrix.e_done_bytes==128,"O group completion");
                $display("PRREAL_GROUP_DONE cycle=%0d job=%0d group=%0d status=%0d bytes=%0d",cycles,jobs,replies,dut.attention.projected.output_path.projection.matrix.e_done_status,dut.attention.projected.output_path.projection.matrix.e_done_bytes);replies++;
            end
            if(dut.attention.projected.output_path.projection.child_done_valid && dut.attention.projected.output_path.projection.child_done_ready)
                $display("PRREAL_MATRIX_DONE cycle=%0d job=%0d status=%0d bytes=%0d rows=%0d tag=%016h",cycles,jobs,dut.attention.projected.output_path.projection.child_done_status,
                    dut.attention.projected.output_path.projection.child_done_bytes,dut.attention.projected.output_path.projection.child_done_rows,{dut.attention.projected.output_path.projection.child_done_epoch,dut.attention.projected.output_path.projection.child_done_sequence});
            if(!result_valid)require(result_address==0 && result_bytes==0,"Private output address leak");
        end
    end

    // Passive startup observation. Original stimulus and assertions follow unchanged.
    always @(negedge clk)begin
        #0.1;
        if(cycles<16)$display("AFSTART cycle=%0d reset=%b core_reset=%b start=%b fsm=%h pipe_enable=%b ce=%b we=%b address=%h data=%h owner_active=%b owner_count=%0d",
            cycles,rst_n,dut.attention.post.arithmetic.core.ap_rst,
            dut.attention.post.hls_start,dut.attention.post.arithmetic.core.ap_CS_fsm,
            dut.attention.post.arithmetic.core.grp_ntrpu_post_attention_r1_Pipeline_VITIS_LOOP_33_1_fu_147.ap_enable_reg_pp0_iter1,
            dut.attention.post.hls_residual_ce,dut.attention.post.hls_residual_we,
            dut.attention.post.hls_residual_address,dut.attention.post.hls_residual_d,
            dut.attention.post.path.owner.active_hls,dut.attention.post.path.owner.residual_count);
    end
    initial begin #200;$display("AFSTART_LIMIT reached=200ns");$finish;end
    always @(posedge clk)begin
        if(rst_n)begin
            if(dut.attention.post.command_valid && dut.attention.post.command_ready)begin
                require(mode_now==0 && committed_O_valid=='1 && dut.attention.projected.result_valid && dut.attention.projected.completion_valid && !dut.attention.projected.completion_ready && !r_active && !aw_pending && w_pending==0,"Actual O handoff ownership/drain");
                require(dut.attention.post.command_branch_address==OBASE+4032 && dut.attention.post.command_source_sequence==sequence_model && dut.attention.post.command_source_epoch==epoch_model,"Actual O handoff identity");
                $display("APREAL_HANDOFF cycle=%0d job=%0d sequence=%016h epoch=%016h address=%016h bytes=%0d",cycles,jobs,dut.attention.post.command_source_sequence,dut.attention.post.command_source_epoch,dut.attention.post.command_branch_address,dut.attention.projected.result_bytes);
            end
            if(dut.attention.post.hls_start)begin
                require(prs==384 && !post_core_busy && !dut.attention.projected.completion_ready,"Actual post HLS inputs/drain");
                post_starts++;post_core_busy=1;post_quiet=0;
                $display("APREAL_START cycle=%0d job=%0d",cycles,jobs);
            end
            if(post_core_busy)begin
                if(|dut.attention.post.hls_input_ce || (dut.attention.post.hls_residual_ce && dut.attention.post.hls_residual_we) || (dut.attention.post.hls_ffn_ce && dut.attention.post.hls_ffn_we) || dut.attention.post.hls_done)post_quiet=0;
                else post_quiet++;
                if(post_quiet>post_max_quiet)post_max_quiet=post_quiet;
                require(post_max_quiet<262144,"Post HLS quiet watchdog");
            end
            if(dut.attention.post.hls_residual_ce && dut.attention.post.hls_residual_we)
                require(dut.attention.post.hls_residual_d===post_residual[sample_now*2048+int'(dut.attention.post.hls_residual_address)],"Actual post residual arithmetic");
            if(dut.attention.post.hls_ffn_ce && dut.attention.post.hls_ffn_we)
                require(dut.attention.post.hls_ffn_d===post_abi[sample_now*512+int'(dut.attention.post.hls_ffn_address)],"Actual post ABI1 arithmetic");
            if(dut.attention.post.hls_done)begin
                require(post_core_busy && dut.attention.post.hls_ready && dut.attention.post.hls_return==0,"Actual post HLS completion");post_core_busy=0;post_dones++;
                $display("APREAL_DONE cycle=%0d job=%0d return=%0d",cycles,jobs,dut.attention.post.hls_return);
            end
            if(!dut.attention.result_valid)require(dut.attention.result_residual_address==0 && dut.attention.result_ffn_address==0 && dut.attention.result_residual_bytes==0 && dut.attention.result_ffn_bytes==0,"Post private descriptor leak");
            if(dut.attention.result_valid && !post_checked)begin
                post_checked=1;
                require(dut.attention.completion_valid && pbs==192 && !post_core_busy && dut.attention.result_residual_address==PRB+4032 && dut.attention.result_ffn_address==PQB+4032 && dut.attention.result_residual_bytes==8192 && dut.attention.result_ffn_bytes==4096,"Post published descriptor");
                for(integer k=0;k<192;k++)require(post_destination[k]===post_output_word(k),"Post B-committed pair");
            end
        end
    end
    always @(posedge clk)begin: observe_copy
        integer address,head;
        if(rst_n && dut.attention.projected.root.head_result_ce)begin
            address=int'(dut.attention.projected.root.head_result_address);head=heads-1;
            require(heads==head_replies+1 && address==head_copies && dut.attention.projected.root.head_completion_valid && dut.attention.projected.root.head_result_valid,"Head copy ownership/order");
            #1;
            require(dut.attention.projected.root.head_result_q_valid && dut.attention.projected.root.head_result_q===fp32_image[sample_now*2048+head*64+address],"Actual attention FP32 result");
            $display("PRREAL_COPY cycle=%0d job=%0d head=%0d index=%0d data=%08h",cycles,jobs,head,address,dut.attention.projected.root.head_result_q);head_copies++;copies++;total_copies++;
        end
    end
    task automatic prepare_cache;
        integer ticks,index;
        @(negedge clk);bind_cache_base=KBASE+4032;bind_region_base=KBASE;bind_region_bytes=270336;bind_capacity=4;
        bind_sequence=sequence_model;bind_epoch=epoch_model;bind_valid=1;command_valid=1;
        #1;require(bind_ready && !command_ready,"Bind priority");
        $display("PRREAL_BIND cycle=%0d job=%0d base=%016h capacity=4 sequence=%016h epoch=%016h",cycles,jobs,bind_cache_base,sequence_model,epoch_model);
        tick();@(negedge clk);bind_valid=0;command_valid=0;ticks=0;
        while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<100,"Bind timeout");end
        require(append_done_bind && append_done_status==0 && append_done_sequence==sequence_model && append_done_epoch==epoch_model,"Bind result");
        append_done_ready=1;tick();@(negedge clk);append_done_ready=0;binds++;
        require(cache_valid && !maintenance_busy && !metadata_fault && cache_base==KBASE+4032 && cache_capacity==4,"Trusted cache bind");
        $display("PRREAL_BIND_DONE cycle=%0d job=%0d",cycles,jobs);
        for(integer pos=0;pos<4;pos++)begin
            append_pos=pos;append_aw_cursor=0;append_w_cursor=0;
            append_layer=0;append_position=32'(pos);append_sequence=sequence_model;append_epoch=epoch_model;append_valid=1;command_valid=1;
            #1;require(append_ready && !command_ready,"Append priority");
            $display("PRREAL_APPEND cycle=%0d job=%0d position=%0d",cycles,jobs,pos);
            tick();@(negedge clk);append_valid=0;command_valid=0;ticks=0;
            while(!append_prepared)begin @(negedge clk);ticks++;require(ticks<100,"Append admission");end
            index=0;
            while(index<64)begin
                data_valid=$urandom_range(0,3)!=0;data_bits=fixture_kv[pos*64+index];data_index=6'(index);data_last=index==63;
                data_sequence=sequence_model;data_epoch=epoch_model;commit_valid=data_valid && index==63;commit_status=0;commit_sequence=sequence_model;commit_epoch=epoch_model;
                @(posedge clk);if(data_valid && data_ready)index++;
                @(negedge clk);
            end
            data_valid=0;commit_valid=0;ticks=0;
            while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<4000,"Append completion");end
            require(!append_done_bind && append_done_status==0 && append_done_bytes==4096 && append_done_committed_tokens==32'(pos+1) &&
                append_done_sequence==sequence_model && append_done_epoch==epoch_model && append_aw_cursor==64 && append_w_cursor==64 && !aw_pending && w_pending==0,"Append full commit");
            $display("PRREAL_APPEND_DONE cycle=%0d job=%0d position=%0d status=%0d bytes=%0d committed=%0d",cycles,jobs,pos,append_done_status,append_done_bytes,append_done_committed_tokens);
            repeat(3)tick();@(negedge clk);append_done_ready=1;tick();@(negedge clk);append_done_ready=0;appends++;
            require(committed_tokens==32'(pos+1) && !maintenance_busy,"Committed metadata publication");
        end
    endtask
    task automatic run_job(input integer mode,input integer sample_id);
        integer limit;logic [3:0] wanted;
        @(negedge clk);require(!busy && !completion_valid && !r_active && !aw_pending && w_pending==0,"Previous owner active");jobs++;mode_now=mode;sample_now=sample_id;
        ars=0;rs=0;aws=0;ws=0;bs=0;groups=0;tiles=0;computed=0;replies=0;qaws=0;qws=0;qbs=0;kars=0;krs=0;kaws=0;kws=0;kbs=0;
        quant_words=0;source_reads=0;root_acks=0;qacks=0;packs=0;outacks=0;starts=0;dones=0;heads=0;head_replies=0;copies=0;head_copies=0;qbeats=0;
        quiet=0;max_quiet=0;write_cursor=0;activation_written=0;seen_source=0;quant_public_seen=0;root_public_seen=0;input_pending=0;
        for(integer i=0;i<64;i++)materialized_activation[i]='x;
        for(integer i=0;i<8192;i++)kv_written[i]=0;
        for(integer i=0;i<128;i++)destination[i]='1;
        committed_O_valid=0;pars=0;prs=0;paws=0;pws=0;pbs=0;post_starts=0;post_dones=0;post_core_busy=0;post_max_quiet=0;post_quiet=0;
        for(integer k=0;k<192;k++)post_destination[k]='1;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);expected_epoch=epoch_model;
        tag_model=FIRST_P+64'(normal_starts*64);lease_model=64'h00000000ffffffe0+64'((jobs-1)*64);
        $display("PRREAL_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d sequence=%016h epoch=%016h tag=%016h",cycles,jobs,mode,sample_now,sequence_model,epoch_model,tag_model);
        prepare_cache();
        command_sequence=sequence_model;command_epoch=epoch_model;command_position=32'(sample_now);command_layer=0;command_lease_base=lease_model;
        command_weight_addr=WBASE+4032;command_weight_region_base=WBASE;command_weight_region_bytes=4460480;
        command_activation_addr=ABASE+4032;command_activation_region_base=ABASE;command_activation_region_bytes=8128;
        command_output_addr=OBASE+4032;command_output_region_base=OBASE;command_output_region_bytes=12224;command_valid=1;
        command_post_input_address=PIB+4032;command_post_input_base=PIB;command_post_input_bytes=12224;
        command_post_weight_address=PWB+4032;command_post_weight_base=PWB;command_post_weight_bytes=12224;
        command_post_residual_address=PRB+4032;command_post_residual_base=PRB;command_post_residual_bytes=12224;
        command_post_ffn_address=PQB+4032;command_post_ffn_base=PQB;command_post_ffn_bytes=8128;
        for(int k=1;k<8;k++)begin f_addr[k]=bases[k]+4032;f_base[k]=bases[k];f_bytes[k]=64'(4032+spans[k]);end
        command_final_address=bases[9]+4032;command_final_base=bases[9];command_final_bytes=12224;
        #1;require(command_ready,"Root admission");tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;command_lease_base=0;
        command_weight_addr=0;command_weight_region_base=0;command_weight_region_bytes=0;command_activation_addr=0;command_activation_region_base=0;command_activation_region_bytes=0;
        command_output_addr=0;command_output_region_base=0;command_output_region_bytes=0;bind_valid=1;append_valid=1;
        command_post_input_address=0;
        command_post_input_base=0;
        command_post_input_bytes=0;
        command_post_weight_address=0;
        command_post_weight_base=0;
        command_post_weight_bytes=0;
        command_post_residual_address=0;
        command_post_residual_base=0;
        command_post_residual_bytes=0;
        command_post_ffn_address=0;
        command_post_ffn_base=0;
        command_post_ffn_bytes=0;
        f_addr=0;f_base=0;f_bytes=0;command_final_address=0;command_final_base=0;command_final_bytes=0;
        limit=0;while(!q_ready)begin @(negedge clk);limit++;require(limit<1000,"Validated outer admission to Q");end
        for(integer beat=0;beat<(mode==1 ? 8:128);beat++)begin
            repeat($urandom_range(0,2))@(negedge clk);
            q_valid=1;q_index=8'(beat);q_last=beat==127;q_sequence=sequence_model;q_epoch=epoch_model;
            for(integer lane=0;lane<16;lane++)q_data[lane*32+:32]=fixture_query[sample_now*2048+beat*16+lane];
            #1;require(q_ready,"Root Q input");tick();@(negedge clk);q_valid=0;
        end
        if(mode==1)begin
            cancel_valid=1;require(cancel_ready,"Root cancel ready");$display("PRREAL_CANCEL cycle=%0d job=%0d qbeats=%0d",cycles,jobs,qbeats);
            tick();@(negedge clk);cancel_valid=0;
        end
        limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<32000000,"Composed completion timeout");end
        wanted=mode==1 ? 7:0;
        require(completion_status==wanted && completion_sequence==sequence_model && completion_epoch==epoch_model && completion_hls_status==0,"Composed completion identity/status");
        require(result_valid==(mode==0) && !r_active && !aw_pending && w_pending==0 && kaws==68 && kws==256 && kbs==68,"Composed publication/drain");
        if(mode==0)begin
            require(result_address==bases[9]+4032 && result_bytes==8192 && computed==2048 && groups==64 && replies==64 && tiles==128 && ars==1344 && rs==73728 && aws==66 && ws==128 && bs==66,"Full O coverage");
            require(heads==32 && head_replies==32 && copies==2048 && kars==80 && krs==(sample_now+1)*256 && qbeats==128,"Full root coverage");
            require(quant_words==512 && starts==1 && dones==1 && root_acks==1 && qaws==64 && qws==64 && qbs==64,"Full quantization coverage");
        end else require(computed==0 && groups==0 && ars==0 && rs==0 && aws==0 && ws==0 && bs==0 && kars==0 && krs==0 && heads==0 && copies==0 && quant_words==0 && starts==0 && dones==0 && root_acks==0 && qaws==0 && qws==0 && qbs==0 && qbeats==8,"Cancelled root reached output");
        require(qacks==0 && packs==0 && outacks==0 && max_quiet<65536,"Retained output owner");
        require(completion_post_hls_status==0 && completion_post_read_beats==(mode==0 ? 384:0) && completion_residual_beats==(mode==0 ? 128:0) && completion_ffn_beats==(mode==0 ? 64:0),"Integrated post counters/status");
        require(pars==(mode==0 ? 9:0) && prs==(mode==0 ? 384:0) && paws==(mode==0 ? 192:0) && pws==paws && pbs==paws && post_starts==(mode==0) && post_dones==(mode==0),"Integrated post traffic coverage");
        if(mode==0)post_pairs++;
        $display("APREAL_COMPLETE cycle=%0d job=%0d status=%0d valid=%0d bytes=%0d ar=%0d r=%0d b=%0d hls=%0d max_quiet=%0d",cycles,jobs,completion_status,result_valid,dut.attention.result_residual_bytes+dut.attention.result_ffn_bytes,pars,prs,pbs,completion_post_hls_status,post_max_quiet);

        repeat(7)begin tick();require(completion_valid && completion_status==wanted && !command_ready && !bind_ready && !append_ready,"Held composed completion");end
        $display("PRREAL_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h valid=%0d bytes=%0d groups=%0d tiles=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d max_quiet=%0d",cycles,jobs,mode,completion_status,completion_sequence,completion_epoch,result_valid,result_bytes,groups,tiles,computed,ars,rs,aws,ws,bs,max_quiet);
        publics++;@(negedge clk);bind_valid=0;append_valid=0;completion_ready=1;tick();@(negedge clk);completion_ready=0;
        $display("PRREAL_RELEASE cycle=%0d job=%0d",cycles,jobs);
        repeat(6)@(negedge clk);
        require(!busy && !completion_valid && root_acks==1 && qacks==(mode==0) && packs==(mode==0) && outacks==(mode==0),"Final owner release");
    endtask
    initial begin
        require(dut.attention.INITIAL_QUANT_TAG[5:0]==0 && dut.attention.INITIAL_PROJECTION_TAG[5:0]==0,"Aligned 64-tag reservation");
        clk=0;
        rst_n=0;
        bind_valid=0;
        bind_cache_base=0;
        bind_region_base=0;
        bind_region_bytes=0;
        bind_capacity=0;
        bind_sequence=0;
        bind_epoch=0;
        expected_epoch=0;
        append_valid=0;
        append_layer=0;
        append_position=0;
        append_sequence=0;
        append_epoch=0;
        data_valid=0;
        data_bits=0;
        data_index=0;
        data_last=0;
        data_sequence=0;
        data_epoch=0;
        commit_valid=0;
        commit_status=0;
        commit_sequence=0;
        commit_epoch=0;
        append_cancel_valid=0;
        append_done_ready=0;
        m_axi_awready=0;
        m_axi_wready=0;
        m_axi_bid=0;
        m_axi_bresp=0;
        m_axi_bvalid=0;
        m_axi_arready=0;
        m_axi_rdata=0;
        m_axi_rresp=0;
        m_axi_rid=0;
        m_axi_rlast=0;
        m_axi_rvalid=0;
        command_valid=0;
        command_position=0;
        command_layer=0;
        command_sequence=0;
        command_epoch=0;
        cancel_valid=0;
        completion_ready=0;
        command_lease_base=0;
        q_valid=0;
        q_data=0;
        q_index=0;
        q_last=0;
        q_sequence=0;
        q_epoch=0;
        command_weight_addr=0;
        command_weight_region_base=0;
        command_weight_region_bytes=0;
        command_activation_addr=0;
        command_activation_region_base=0;
        command_activation_region_bytes=0;
        command_output_addr=0;
        command_output_region_base=0;
        command_output_region_bytes=0;

        command_post_input_address=0;
        command_post_input_base=0;
        command_post_input_bytes=0;
        command_post_weight_address=0;
        command_post_weight_base=0;
        command_post_weight_bytes=0;
        command_post_residual_address=0;
        command_post_residual_base=0;
        command_post_residual_bytes=0;
        command_post_ffn_address=0;
        command_post_ffn_base=0;
        command_post_ffn_bytes=0;
        if($value$plusargs("SEED=%d",seed))begin end rng=seed;rng=$urandom(rng);
        require($value$plusargs("FIXTURE=%s",fixture) && $value$plusargs("ROOTFIXTURE=%s",root_fixture),"Missing immutable fixtures");
        $readmemh({fixture,"/weight.hex"},weight_image);$readmemh({fixture,"/activation.hex"},activation_image);
        $readmemh({fixture,"/attention-fp32.hex"},fp32_image);$readmemh({fixture,"/reference-fp64.hex"},reference64);$readmemh({fixture,"/reference-fp32.hex"},reference32);
        $readmemh({root_fixture,"/kv_beats.hex"},fixture_kv);$readmemh({root_fixture,"/query.hex"},fixture_query);

        require($value$plusargs("POSTFIXTURE=%s",post_fixture),"Missing immutable post fixture");
        $readmemh({post_fixture,"/source32.hex"},post_source);$readmemh({post_fixture,"/residual32.hex"},post_residual);$readmemh({post_fixture,"/abi64.hex"},post_abi);
        $display("PRREAL_CONFIG seed=%0d rows=2048 k_blocks=64 tile_k=32 watchdog=65536",seed);
        require($value$plusargs("FFNFIXTURE=%s",ffn_fixture) && $value$plusargs("EXTRA=%s",extra_fixture),"Missing FFN/extra fixture");
        f_addr=0;f_base=0;f_bytes=0;command_final_address=0;command_final_base=0;command_final_bytes=0;
        prepare_ffn();
        $display("AFREAL_CONFIG dim=2048 hidden=8192 sample=1 token=0 attention_position=0 full_weight_bytes=57933824 input_abi_requantized=0");
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        run_job(0,0);
        require(jobs==1 && publics==1 && normal_starts==1 && binds==1 && appends==4 && total_heads==32 && total_copies==2048 && total_qbeats==128 && total_results==2048,"Integrated upstream case coverage");
        require(post_pairs==1 && total_par==9 && total_pr==384 && total_pb==192,"Integrated post coverage");
        require(far==16486 && fr==887040 && faw==994 && fw==2560 && fb==994,"Integrated full FFN traffic");
        require(source_reads_abi==32768 && source_reads_residual==128,"Live post source read coverage");
        require(fhls_starts==1 && fhls_dones==1 && fhls_words==2048 && ffn_admits==1 && final_acks==1 && attention_acks==1 && ffn_acks==1,"Integrated final ownership coverage");
        for(int k=0;k<10;k++)if(k<1 || k>3)for(int off=0;off<FLIMIT;off++)begin
            if(off>=4032 && off<4032+spans[k])begin
                require(fwritten[k][off],"Complete B-committed source/output coverage");
                if(k!=0 && k!=8)require(fmem[k][off]===f_expected(k,off-4032),"Final memory numerical coverage");
            end else require(!fwritten[k][off] && fmem[k][off]==8'ha5,"Output guard changed");
        end
        for(int k=0;k<10;k++)$display("FFRREAL_READ buffer=%0d beats=%0d hash=%016h",k,read_beats_by_buffer[k],read_hash[k]);
        for(int p=0;p<5;p++)$display("FFRREAL_PHASE phase=%0d cycles=%0d reads=%0d writes=%0d",p,fphase_cycles[p],fphase_reads[p],fphase_writes[p]);
        require(command_ready && !ffn_selected && !residual_selected && !busy && !completion_valid && !result_valid,"Full hierarchy reusable");
        $display("AFREAL_PASS jobs=1 seed=%0d cycles=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d bytes=163840 attention_done=%0d ffn_done=%0d first_res_ar=%0d last_res_b=%0d abi_reads=%0d residual_reads=%0d",seed,cycles,far,fr,faw,fw,fb,attention_done_cycle,ffn_done_cycle,first_res_ar,last_res_b,source_reads_abi,source_reads_residual);
        $display("AFREAL_REUSABLE jobs=1");
        $finish;
    end

    always @(posedge clk)if(rst_n)begin
        if(ffn_selected)fphase_cycles[fphase]++;
        if(cycles%100000==0)$display("AFREAL_PROGRESS cycle=%0d ffn=%0d phase=%0d bytes=%0d",cycles,ffn_selected,fphase,fw*64);
        if(ffn_selected!=old_ffn || residual_selected!=old_residual || activation_selected!=old_activation)begin
            require(!r_active && !aw_pending && w_pending==0 && !m_axi_rvalid && !m_axi_bvalid,"Hierarchy switched before AXI drain");
            if(ffn_selected!=old_ffn)$display("AFREAL_SWITCH cycle=%0d ffn=%0d",cycles,ffn_selected);
            if(residual_selected!=old_residual)$display("FFRREAL_SWITCH cycle=%0d residual=%0d",cycles,residual_selected);
        end
        old_ffn=ffn_selected;old_residual=residual_selected;old_activation=activation_selected;
        if(dut.attention.completion_valid && !seen_attention)begin
            require(dut.attention.result_valid && pbs==192 && !post_core_busy && !r_active && !aw_pending && w_pending==0,"Producer published before B drain");
            require(dut.attention.completion_status==0 && dut.attention.completion_sequence==sequence_model && dut.attention.completion_epoch==epoch_model,"Producer identity/status");
            seen_attention=1;attention_done_cycle=cycles;
            $display("AFREAL_ATTENTION_COMPLETE cycle=%0d sequence=%016h epoch=%016h residual=%016h abi=%016h",cycles,sequence_model,epoch_model,dut.attention.result_residual_address,dut.attention.result_ffn_address);
        end
        if(dut.ffn.command_valid && dut.ffn.command_ready)begin
            require(seen_attention && ffn_selected && !dut.attention.completion_ready && dut.ffn.command_addr[0]==PQB+4032 && dut.ffn.command_residual_address==PRB+4032,"Actual post to FFN ownership");
            require(dut.ffn.command_sequence==sequence_model && dut.ffn.command_epoch==epoch_model,"Actual FFN admission identity");
            ffn_admits++;
            $display("AFREAL_FFN_ADMIT cycle=%0d sequence=%016h epoch=%016h abi=%016h residual=%016h",cycles,sequence_model,epoch_model,dut.ffn.command_addr[0],dut.ffn.command_residual_address);
        end
        if(dut.ffn.ffn.child_job_valid && dut.ffn.ffn.child_job_ready)begin
            require({dut.ffn.ffn.child_job_epoch,dut.ffn.ffn.child_job_sequence}==64'hfffffffc,"Private full FFN tag");
            $display("FFRREAL_FFN_CHILD cycle=%0d tag=%016h",cycles,{dut.ffn.ffn.child_job_epoch,dut.ffn.ffn.child_job_sequence});
        end
        if(dut.ffn.ffn_completion_valid && !seen_ffn)begin
            require(!r_active && !aw_pending && w_pending==0 && fw*64==155648,"FFN completed before B commit");
            require(dut.ffn.ffn_result_valid && dut.ffn.ffn_result_ffn_address==bases[7]+4032 && dut.ffn.ffn_result_residual_address==PRB+4032,"Actual FFN output descriptors");
            require(dut.ffn.ffn_completion_status==0 && dut.ffn.ffn_completion_child_status==0 && dut.ffn.ffn_completion_stages==4 && dut.ffn.ffn_completion_phase==3 && dut.ffn.ffn_completion_bytes==8192 && dut.ffn.ffn_completion_stage_bytes==155648 && dut.ffn.ffn_completion_hls_status==0,"Full FFN completion accounting");
            seen_ffn=1;ffn_done_cycle=cycles;
            $display("FFRREAL_FFN_COMPLETE cycle=%0d sequence=%016h epoch=%016h status=0 child=0 stages=4 phase=3 bytes=8192 stage_bytes=155648 hls=00000000",cycles,dut.ffn.ffn_completion_sequence,dut.ffn.ffn_completion_epoch);
        end
        if(dut.ffn.res_command_valid && dut.ffn.res_command_ready)begin
            require(seen_ffn && !seen_res && residual_selected && !dut.ffn.ffn_completion_ready,"Residual starts without retained FFN");seen_res=1;
            $display("FFRREAL_RES_START cycle=%0d sequence=%016h epoch=%016h input=%016h branch=%016h output=%016h",cycles,dut.ffn.res_command_sequence,dut.ffn.res_command_epoch,dut.ffn.res_command_input_address,dut.ffn.res_command_branch_address,dut.ffn.res_command_residual_address);
        end
        if(dut.ffn.res.hls_start)begin
            require(!fhls_busy && fhls_starts==0 && fphase_reads[4]==256 && dut.ffn.res.hls_elements==2048,"Residual HLS inputs/extent");
            fhls_busy=1;fhls_starts++;$display("FFRREAL_HLS_START cycle=%0d",cycles);
        end
        if(dut.ffn.res.hls_residual_ce && dut.ffn.res.hls_residual_we)begin
            require(fhls_busy && dut.ffn.res.hls_residual_address==fhls_words,"Residual HLS order");
            for(int lane=0;lane<4;lane++)require(dut.ffn.res.hls_residual_d[lane*8+:8]===golden[4][fhls_words*4+lane],"Residual exact FP32 output");
            $display("FFRREAL_HLS_WRITE cycle=%0d index=%0d bits=%08h",cycles,fhls_words,dut.ffn.res.hls_residual_d);fhls_words++;
        end
        if(dut.ffn.res.hls_done)begin
            require(fhls_busy && fhls_words==2048 && dut.ffn.res.hls_return==0 && dut.ffn.res.hls_ready,"Residual HLS completion");fhls_busy=0;fhls_dones++;
            $display("FFRREAL_HLS_DONE cycle=%0d return=%0d",cycles,dut.ffn.res.hls_return);
        end
        if(completion_valid)begin
            if(!complete_seen)begin
                complete_seen=1;
                $display("AFREAL_COMPLETE cycle=%0d sequence=%016h epoch=%016h status=%0d attention=%0d ffn_residual=%0d ffn=%0d child=%0d residual=%0d stages=%0d phase=%0d ffn_bytes=%0d stage_bytes=%0d attention_hls=%08h post_hls=%08h ffn_hls=%08h hls=%08h post_reads=%0d post_residual=%0d post_abi=%0d reads=%0d writes=%0d valid=%0d address=%016h bytes=%0d",cycles,completion_sequence,completion_epoch,completion_status,completion_attention_status,completion_ffn_residual_status,completion_ffn_status,completion_ffn_child_status,completion_residual_status,completion_ffn_stages,completion_ffn_phase,completion_ffn_bytes,completion_ffn_stage_bytes,completion_attention_hls_status,completion_post_hls_status,completion_ffn_hls_status,completion_hls_status,completion_post_read_beats,completion_residual_beats,completion_ffn_beats,completion_read_beats,completion_write_beats,result_valid,result_address,result_bytes);
            end
            require(result_valid && result_address==bases[9]+4032 && result_bytes==8192 && fw*64==163840 && !fhls_busy,"Full result contract");
            require(completion_status==0 && completion_attention_status==0 && completion_ffn_residual_status==0 && completion_ffn_status==0 && completion_ffn_child_status==0 && completion_residual_status==0 && completion_ffn_hls_status==0 && completion_hls_status==0 && completion_attention_hls_status==0 && completion_post_hls_status==0,"All arithmetic status fields");
            require(completion_ffn_stages==4 && completion_ffn_phase==3 && completion_ffn_bytes==8192 && completion_ffn_stage_bytes==155648 && completion_read_beats==256 && completion_write_beats==128,"Final arithmetic counters");
            require(dut.attention.completion_valid && dut.ffn.completion_valid && dut.ffn.ffn_completion_valid && dut.ffn.res_completion_valid,"Retained complete hierarchy");
        end
        if(dut.attention.completion_ready || dut.ffn.completion_ready || dut.ffn.ffn_completion_ready || dut.ffn.res_completion_ready)begin
            require(completion_valid && completion_ready && dut.attention.completion_ready && dut.ffn.completion_ready && dut.ffn.ffn.child_done_ready && dut.ffn.ffn_completion_ready && dut.ffn.res_completion_ready,"Atomic complete hierarchy ACK");
            attention_acks++;ffn_acks++;
        end
        if(completion_valid && completion_ready)begin
            final_acks++;
            $display("AFREAL_ACK cycle=%0d sequence=%016h epoch=%016h attention=%0d ffn=%0d private_ffn=%0d residual=%0d",cycles,completion_sequence,completion_epoch,dut.attention.completion_ready,dut.ffn.completion_ready,dut.ffn.ffn.child_done_ready,dut.ffn.res_completion_ready);
        end
    end

    initial begin #165000000;$fatal(1,"PRREAL_FAIL global timeout");end
endmodule
`default_nettype wire
