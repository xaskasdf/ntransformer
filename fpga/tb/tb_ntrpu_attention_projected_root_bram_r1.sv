`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_projected_root_bram_r1;
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
    logic [16:0] result_bytes;
    logic [63:0] command_weight_addr;
    logic [63:0] command_weight_region_base;
    logic [63:0] command_weight_region_bytes;
    logic [63:0] command_activation_addr;
    logic [63:0] command_activation_region_base;
    logic [63:0] command_activation_region_bytes;
    logic [63:0] command_output_addr;
    logic [63:0] command_output_region_base;
    logic [63:0] command_output_region_bytes;
    localparam logic [63:0] WBASE=64'h100000000,ABASE=64'h8000000000,OBASE=64'ha000000000,KBASE=64'hc000000000;
    localparam logic [63:0] FIRST_P=64'h00000000ffffffc0;
    ntrpu_attention_projected_root_r2 #(.INITIAL_QUANT_TAG(FIRST_P),.INITIAL_PROJECTION_TAG(FIRST_P)) dut(.*);
    always #2.5 clk=~clk;
    logic [511:0] weight_image[0:69694],activation_image[0:255],destination[0:127];
    logic [63:0] reference64[0:8191];logic [31:0] reference32[0:8191],fp32_image[0:8191];
    logic [511:0] fixture_kv[0:47807],kv_memory[0:8191],materialized_activation[0:63],pending_data[0:3];
    logic [31:0] fixture_query[0:9151];
    bit kv_written[0:8191];logic [63:0] activation_written=0;logic [3:0] pending_last;
    logic [2047:0] seen_source;
    string fixture,root_fixture;
    integer seed=1,rng=1,cycles=0,jobs=0,mode_now=0,sample_now=0,publics=0,normal_starts=0;
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
    wire source_ce=dut.root.result_ce;
    wire [10:0] source_address=dut.root.result_address;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"PRREAL_FAIL cycle=%0d job=%0d mode=%0d: %s",cycles,jobs,mode_now,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [511:0] read_word(input logic [63:0] address);
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
        cycles++;
        if(!rst_n)begin hold_ar=0;hold_r=0;hold_aw=0;hold_w=0;r_active=0;aw_pending=0;w_pending=0;input_pending=0;end
        else begin
            if(busy)require(cache_lock && !bind_ready && !append_ready && !dut.root.bind_valid && !dut.root.append_valid,"Composed maintenance lock");
            if(command_valid && command_ready)
                $display("PRREAL_COMMAND cycle=%0d job=%0d position=%0d sequence=%016h epoch=%016h lease=%016h",cycles,jobs,command_position,command_sequence,command_epoch,command_lease_base);
            if(q_valid && q_ready)begin
                $display("PRREAL_Q cycle=%0d job=%0d index=%0d data=%0128h",cycles,jobs,q_index,q_data);qbeats++;total_qbeats++;
            end
            if(data_valid && data_ready)
                $display("PRREAL_DATA cycle=%0d job=%0d position=%0d index=%0d data=%0128h",cycles,jobs,append_pos,data_index,data_bits);
            if(dut.root.head_command_valid && dut.root.head_command_ready)begin
                require(mode_now==0 && qbeats==128 && heads==head_replies && heads<32,"Head owner/shape");
                require(dut.root.head_command_head==32'(heads) && dut.root.head_command_position==32'(sample_now) && dut.root.head_command_layer==0 &&
                    dut.root.head_command_sequence==sequence_model && dut.root.head_command_epoch==epoch_model && dut.root.head_command_lease==lease_model+64'(heads),"Head descriptor/identity");
                for(integer lane=0;lane<64;lane++)require(dut.root.head_command_query[lane*32+:32]===fixture_query[sample_now*2048+heads*64+lane],"Head Q content");
                $display("PRREAL_HEAD cycle=%0d job=%0d head=%0d sequence=%016h epoch=%016h lease=%016h",cycles,jobs,heads,sequence_model,epoch_model,dut.root.head_command_lease);
                heads++;total_heads++;head_copies=0;
            end
            if(dut.root.head_completion_valid && dut.root.head_completion_ready)begin
                require(heads==head_replies+1 && head_copies==64 && dut.root.head_completion_status==0 && !lease_active && !r_active,"Head completion drain");
                $display("PRREAL_HEAD_DONE cycle=%0d job=%0d head=%0d status=%0d reads=%0d",cycles,jobs,head_replies,dut.root.head_completion_status,head_copies);head_replies++;
            end
            if(dut.root.completion_valid && !root_public_seen)begin
                require(dut.root.completion_sequence==sequence_model && dut.root.completion_epoch==epoch_model && !r_active && !aw_pending && w_pending==0,"Root publication identity/drain");
                require(mode_now==0 ? (heads==32 && copies==2048 && dut.root.result_valid && dut.root.completion_status==0):
                    (heads==0 && copies==0 && !dut.root.result_valid && dut.root.completion_status==7),"Root publication arithmetic/cancel");
                root_public_seen=1;
                $display("PRREAL_ROOT_PUBLIC cycle=%0d job=%0d status=%0d sequence=%016h epoch=%016h valid=%0d",cycles,jobs,dut.root.completion_status,dut.root.completion_sequence,dut.root.completion_epoch,dut.root.result_valid);
            end
            if(input_pending)begin
                require(dut.output_path.quantize_store.hls_input_q===fp32_image[sample_now*2048+pending_input_address],"Real synchronous root to quantizer data");
                $display("PRREAL_INPUT cycle=%0d job=%0d index=%0d data=%08h",cycles,jobs,pending_input_address,dut.output_path.quantize_store.hls_input_q);
            end
            input_pending=source_ce;pending_input_address=int'(source_address);
            if(source_ce)begin
                require(mode_now==0 && root_acks==0 && root_public_seen && dut.root.result_valid && dut.output_owned,"Source read ownership");
                seen_source[source_address]=1;source_reads++;total_source_reads++;
                $display("PRREAL_READ cycle=%0d job=%0d index=%0d",cycles,jobs,source_address);
            end
            if(dut.output_path.quantize_store.hls_start)begin
                starts++;require(starts==1 && dut.output_path.quantize_store.hls_elements==2048,"Quantizer extent");
                $display("PRREAL_START cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.output_path.quantize_store.hls_output_ce && dut.output_path.quantize_store.hls_output_we)begin
                require(quant_words<512 && dut.output_path.quantize_store.hls_output_address==9'(quant_words),"Quantizer order");
                require(dut.output_path.quantize_store.hls_output_d===activation_image[sample_now*64+quant_words/8][(quant_words%8)*64+:64],"Generated ABI1 bytes");
                $display("PRREAL_WORD cycle=%0d job=%0d index=%0d data=%016h",cycles,jobs,quant_words,dut.output_path.quantize_store.hls_output_d);quant_words++;total_quant_words++;
            end
            if(dut.output_path.quantize_store.hls_done)begin
                dones++;require(dones==1 && dut.output_path.quantize_store.hls_ready && dut.output_path.quantize_store.hls_return==0 && quant_words==512 && seen_source=='1,"Quantizer result coverage");
                $display("PRREAL_DONE cycle=%0d job=%0d status=%0d words=%0d",cycles,jobs,dut.output_path.quantize_store.hls_return,quant_words);
            end
            if(dut.root.completion_valid && dut.root.completion_ready)begin
                require(root_acks==0 && (mode_now==0 ? dones==1:(completion_valid && completion_ready)),"Root release before consumer drain");
                root_acks++;total_root++;
                $display("PRREAL_ROOT_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.root.completion_sequence,dut.root.completion_epoch);
            end
            if(dut.output_path.q_completion_valid && !quant_public_seen)begin
                require(qaws==64 && qws==64 && qbs==64 && activation_written=='1 && !aw_pending && w_pending==0,"Quant publication before commit");
                require(dut.output_path.q_completion_status==0 && dut.output_path.q_completion_hls_status==0 && dut.output_path.q_completion_records==64 && dut.output_path.q_result_valid &&
                    dut.output_path.q_result_address==ABASE+4032 && dut.output_path.q_result_bytes==4096,"Quantized region contract");quant_public_seen=1;
                $display("PRREAL_QUANT_PUBLIC cycle=%0d job=%0d status=%0d hls=%0d sequence=%016h epoch=%016h records=%0d valid=%0d address=%016h bytes=%0d",cycles,jobs,
                    dut.output_path.q_completion_status,dut.output_path.q_completion_hls_status,dut.output_path.q_completion_sequence,dut.output_path.q_completion_epoch,
                    dut.output_path.q_completion_records,dut.output_path.q_result_valid,dut.output_path.q_result_address,dut.output_path.q_result_bytes);
            end
            if(dut.output_path.projection.command_valid && dut.output_path.projection.command_ready)begin
                require(quant_public_seen && qacks==0 && dut.output_path.q_completion_valid,"O without retained ABI1");
                $display("PRREAL_PROJECTION_COMMAND cycle=%0d job=%0d sequence=%016h epoch=%016h address=%016h",cycles,jobs,
                    dut.output_path.projection.command_sequence,dut.output_path.projection.command_epoch,dut.output_path.projection.command_activation_addr);
            end
            if(dut.output_path.q_completion_valid && dut.output_path.quantize_store.completion_ready)begin
                require(completion_valid && completion_ready && !r_active && !aw_pending && w_pending==0,"Early quant owner release");qacks++;total_qacks++;
                $display("PRREAL_QACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.output_path.q_completion_sequence,dut.output_path.q_completion_epoch);
            end
            if(dut.output_path.p_completion_valid && dut.output_path.projection.completion_ready)begin
                require(completion_valid && completion_ready && dut.output_path.quantize_store.completion_ready,"Split output owner release");packs++;total_packs++;
                $display("PRREAL_PACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.output_path.p_completion_sequence,dut.output_path.p_completion_epoch);
            end
            if(dut.output_path.completion_valid && dut.output_path.completion_ready)begin
                outacks++;total_outacks++;$display("PRREAL_OUT_ACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.output_path.projection.control.active_child)begin
                if(dut.output_path.projection.child_activity)quiet=0;else quiet++;
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
            if(dut.output_path.projection.child_job_valid && dut.output_path.projection.child_job_ready)begin
                require({dut.output_path.projection.child_job_epoch,dut.output_path.projection.child_job_sequence}==tag_model,"Matrix identity");
                require(dut.output_path.projection.child_job_rows==2048 && dut.output_path.projection.child_job_k_blocks==64 && dut.output_path.projection.child_job_tile_k==32 &&
                    dut.output_path.projection.child_job_layout==2 && dut.output_path.projection.child_job_output_fp32,"Full O shape");normal_starts++;
                $display("PRREAL_MATRIX cycle=%0d job=%0d tag=%016h",cycles,jobs,tag_model);
            end
            if(dut.output_path.projection.matrix.e_job_valid && dut.output_path.projection.matrix.e_job_ready)begin
                require(groups==replies && !r_active && !aw_pending && w_pending==0,"O group overlap");
                require(dut.output_path.projection.matrix.e_job_rows==32 && dut.output_path.projection.matrix.e_job_output_fp32 && dut.output_path.projection.matrix.e_job_k_blocks==64 &&
                    {dut.output_path.projection.matrix.e_job_epoch,dut.output_path.projection.matrix.e_job_sequence}==tag_model+64'(groups),"O group identity/shape");
                $display("PRREAL_GROUP cycle=%0d job=%0d group=%0d tag=%016h",cycles,jobs,groups,{dut.output_path.projection.matrix.e_job_epoch,dut.output_path.projection.matrix.e_job_sequence});groups++;
            end
            if(dut.output_path.projection.matrix.e_tile_valid && dut.output_path.projection.matrix.e_tile_ready)begin
                require(dut.output_path.projection.matrix.e_tile_addr==WBASE+4032+64'(tiles*34816) && dut.output_path.projection.matrix.e_tile_k_count==32 &&
                    dut.output_path.projection.matrix.e_tile_k_base==9'((tiles%2)*32),"O tile order");
                $display("PRREAL_TILE cycle=%0d job=%0d tile=%0d address=%016h",cycles,jobs,tiles,dut.output_path.projection.matrix.e_tile_addr);tiles++;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!r_active && m_axi_araddr[5:0]==0 && int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR admission/alignment/page");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR attributes");
                r_address=m_axi_araddr;r_beats=int'(m_axi_arlen)+1;r_index=0;r_active=1;r_is_kv=!dut.output_owned;
                if(r_is_kv)begin
                    require(heads==head_replies+1 && r_address>=KBASE && r_address+64'(r_beats*64)<=KBASE+270336,"KV AR owner/range");
                    $display("PRREAL_KAR cycle=%0d job=%0d ar=%0d address=%016h beats=%0d head=%0d",cycles,jobs,kars,r_address,r_beats,heads-1);kars++;total_kar++;
                end else begin
                    require(groups>0 && qbs==64 && qacks==0 && dut.output_path.q_completion_valid && activation_written=='1,"O AR ownership");
                    $display("PRREAL_AR cycle=%0d job=%0d ar=%0d address=%016h beats=%0d group=%0d",cycles,jobs,ars,r_address,r_beats,groups-1);ars++;total_ar++;
                end
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(r_active && m_axi_rlast==(r_index==r_beats-1) && r_is_kv==!dut.output_owned,"R owner/burst");
                if(r_is_kv)begin $display("PRREAL_KR cycle=%0d job=%0d ar=%0d beat=%0d address=%016h data=%0128h last=%0d",cycles,jobs,kars-1,r_index,r_address+64'(r_index*64),m_axi_rdata,m_axi_rlast);krs++;total_kr++;end
                else begin $display("PRREAL_R cycle=%0d job=%0d ar=%0d beat=%0d address=%016h data=%0128h last=%0d",cycles,jobs,ars-1,r_index,r_address+64'(r_index*64),m_axi_rdata,m_axi_rlast);rs++;total_r++;end
                r_index++;if(m_axi_rlast)r_active=0;m_axi_rvalid<=0;
            end
            if(dut.output_path.projection.matrix.engine.result_valid && dut.output_path.projection.matrix.engine.result_ready)begin
                require(dut.output_path.projection.matrix.engine.result_row==5'(computed%32) && dut.output_path.projection.matrix.engine.result_data===reference64[sample_now*2048+computed],"O FP64 numerical result");
                $display("PRREAL_MAC cycle=%0d job=%0d row=%0d data=%016h",cycles,jobs,computed,dut.output_path.projection.matrix.engine.result_data);computed++;total_results++;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awaddr[5:0]==0 && int'(m_axi_awaddr[11:0])+(int'(m_axi_awlen)+1)*64<=4096,"AW admission/alignment/page");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                aw_kind=!dut.output_owned ? 0:dut.output_path.projection_owned ? 2:1;
                if(aw_kind==0)begin
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
                if(!dut.output_owned)begin
                    require(append_w_cursor<64 && m_axi_wdata===fixture_kv[append_pos*64+append_w_cursor],"Actual KV write bytes");
                    $display("PRREAL_KW cycle=%0d job=%0d position=%0d index=%0d data=%0128h last=%0d",cycles,jobs,append_pos,append_w_cursor,m_axi_wdata,m_axi_wlast);append_w_cursor++;kws++;total_kw++;
                end else if(!dut.output_path.projection_owned)begin
                    require(root_acks==1 && qws<64 && qws==qbs && m_axi_wlast && m_axi_wdata===activation_image[sample_now*64+qws],"Actual ABI1 write bytes");
                    $display("PRREAL_QW cycle=%0d job=%0d record=%0d data=%0128h",cycles,jobs,qws,m_axi_wdata);qws++;total_qw++;
                end else begin
                    require(ws<128 && m_axi_wdata===output_word(ws),"Actual FP32 projection bytes");destination[ws]=m_axi_wdata;
                    $display("PRREAL_W cycle=%0d job=%0d word=%0d data=%0128h last=%0d",cycles,jobs,ws,m_axi_wdata,m_axi_wlast);ws++;total_w++;
                end
                pending_data[w_pending]=m_axi_wdata;pending_last[w_pending]=m_axi_wlast;w_pending++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending==aw_beats && aw_kind==(!dut.output_owned ? 0:dut.output_path.projection_owned ? 2:1),"B owner/drain");
                if(aw_kind==0)begin
                    for(integer i=0;i<aw_beats;i++)begin
                        require(aw_address>=KBASE && aw_address+64'((i+1)*64)<=KBASE+270336,"KV commit bounds");
                        kv_memory[int'((aw_address-KBASE)>>6)+i]=pending_data[i];kv_written[int'((aw_address-KBASE)>>6)+i]=1;
                    end
                    $display("PRREAL_KB cycle=%0d job=%0d aw=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);kbs++;total_kb++;
                end else if(aw_kind==1)begin
                    require(aw_number==qbs && qaws==qbs+1 && qws==qbs+1 && !activation_written[aw_number],"ABI1 commit order");
                    materialized_activation[aw_number]=pending_data[0];activation_written[aw_number]=1;
                    $display("PRREAL_QB cycle=%0d job=%0d record=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);qbs++;total_qb++;
                end else begin $display("PRREAL_B cycle=%0d job=%0d aw=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);bs++;total_b++;end
                aw_pending=0;w_pending=0;m_axi_bvalid<=0;
            end
            if(dut.output_path.projection.matrix.e_done_valid && dut.output_path.projection.matrix.e_done_ready)begin
                require(dut.output_path.projection.matrix.e_done_status==0 && dut.output_path.projection.matrix.e_done_bytes==128,"O group completion");
                $display("PRREAL_GROUP_DONE cycle=%0d job=%0d group=%0d status=%0d bytes=%0d",cycles,jobs,replies,dut.output_path.projection.matrix.e_done_status,dut.output_path.projection.matrix.e_done_bytes);replies++;
            end
            if(dut.output_path.projection.child_done_valid && dut.output_path.projection.child_done_ready)
                $display("PRREAL_MATRIX_DONE cycle=%0d job=%0d status=%0d bytes=%0d rows=%0d tag=%016h",cycles,jobs,dut.output_path.projection.child_done_status,
                    dut.output_path.projection.child_done_bytes,dut.output_path.projection.child_done_rows,{dut.output_path.projection.child_done_epoch,dut.output_path.projection.child_done_sequence});
            if(!result_valid)require(result_address==0 && result_bytes==0,"Private output address leak");
        end
    end
    always @(posedge clk)begin: observe_copy
        integer address,head;
        if(rst_n && dut.root.head_result_ce)begin
            address=int'(dut.root.head_result_address);head=heads-1;
            require(heads==head_replies+1 && address==head_copies && dut.root.head_completion_valid && dut.root.head_result_valid,"Head copy ownership/order");
            #1;
            require(dut.root.head_result_q_valid && dut.root.head_result_q===fp32_image[sample_now*2048+head*64+address],"Actual attention FP32 result");
            $display("PRREAL_COPY cycle=%0d job=%0d head=%0d index=%0d data=%08h",cycles,jobs,head,address,dut.root.head_result_q);head_copies++;copies++;total_copies++;
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
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);expected_epoch=epoch_model;
        tag_model=FIRST_P+64'(normal_starts*64);lease_model=64'h00000000ffffffe0+64'((jobs-1)*64);
        $display("PRREAL_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d sequence=%016h epoch=%016h tag=%016h",cycles,jobs,mode,sample_now,sequence_model,epoch_model,tag_model);
        prepare_cache();
        command_sequence=sequence_model;command_epoch=epoch_model;command_position=32'(sample_now);command_layer=0;command_lease_base=lease_model;
        command_weight_addr=WBASE+4032;command_weight_region_base=WBASE;command_weight_region_bytes=4460480;
        command_activation_addr=ABASE+4032;command_activation_region_base=ABASE;command_activation_region_bytes=8128;
        command_output_addr=OBASE+4032;command_output_region_base=OBASE;command_output_region_bytes=12224;command_valid=1;
        #1;require(command_ready,"Root admission");tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;command_lease_base=0;
        command_weight_addr=0;command_weight_region_base=0;command_weight_region_bytes=0;command_activation_addr=0;command_activation_region_base=0;command_activation_region_bytes=0;
        command_output_addr=0;command_output_region_base=0;command_output_region_bytes=0;bind_valid=1;append_valid=1;
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
        limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<3000000,"Composed completion timeout");end
        wanted=mode==1 ? 7:0;
        require(completion_status==wanted && completion_sequence==sequence_model && completion_epoch==epoch_model && completion_hls_status==0,"Composed completion identity/status");
        require(result_valid==(mode==0) && !r_active && !aw_pending && w_pending==0 && kaws==68 && kws==256 && kbs==68,"Composed publication/drain");
        if(mode==0)begin
            require(result_address==OBASE+4032 && result_bytes==8192 && computed==2048 && groups==64 && replies==64 && tiles==128 && ars==1344 && rs==73728 && aws==66 && ws==128 && bs==66,"Full O coverage");
            require(heads==32 && head_replies==32 && copies==2048 && kars==80 && krs==(sample_now+1)*256 && qbeats==128,"Full root coverage");
            require(quant_words==512 && starts==1 && dones==1 && root_acks==1 && qaws==64 && qws==64 && qbs==64,"Full quantization coverage");
        end else require(computed==0 && groups==0 && ars==0 && rs==0 && aws==0 && ws==0 && bs==0 && kars==0 && krs==0 && heads==0 && copies==0 && quant_words==0 && starts==0 && dones==0 && root_acks==0 && qaws==0 && qws==0 && qbs==0 && qbeats==8,"Cancelled root reached output");
        require(qacks==0 && packs==0 && outacks==0 && max_quiet<65536,"Retained output owner");
        repeat(7)begin tick();require(completion_valid && completion_status==wanted && !command_ready && !bind_ready && !append_ready,"Held composed completion");end
        $display("PRREAL_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h valid=%0d bytes=%0d groups=%0d tiles=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d max_quiet=%0d",cycles,jobs,mode,completion_status,completion_sequence,completion_epoch,result_valid,result_bytes,groups,tiles,computed,ars,rs,aws,ws,bs,max_quiet);
        publics++;@(negedge clk);bind_valid=0;append_valid=0;completion_ready=1;tick();@(negedge clk);completion_ready=0;
        require(!busy && !completion_valid && root_acks==1 && qacks==(mode==0) && packs==(mode==0) && outacks==(mode==0),"Final owner release");
        $display("PRREAL_RELEASE cycle=%0d job=%0d",cycles,jobs);
    endtask
    initial begin
        require(dut.INITIAL_QUANT_TAG[5:0]==0 && dut.INITIAL_PROJECTION_TAG[5:0]==0,"Aligned 64-tag reservation");
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

        if($value$plusargs("SEED=%d",seed))begin end rng=seed;rng=$urandom(rng);
        require($value$plusargs("FIXTURE=%s",fixture) && $value$plusargs("ROOTFIXTURE=%s",root_fixture),"Missing immutable fixtures");
        $readmemh({fixture,"/weight.hex"},weight_image);$readmemh({fixture,"/activation.hex"},activation_image);
        $readmemh({fixture,"/attention-fp32.hex"},fp32_image);$readmemh({fixture,"/reference-fp64.hex"},reference64);$readmemh({fixture,"/reference-fp32.hex"},reference32);
        $readmemh({root_fixture,"/kv_beats.hex"},fixture_kv);$readmemh({root_fixture,"/query.hex"},fixture_query);
        $display("PRREAL_CONFIG seed=%0d rows=2048 k_blocks=64 tile_k=32 watchdog=65536",seed);
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        run_job(0,0);run_job(1,1);run_job(0,3);
        require(jobs==3 && publics==3 && normal_starts==2 && binds==3 && appends==12 && total_heads==64 && total_copies==4096 && total_qbeats==264 && total_results==4096 &&
            total_ar==2688 && total_r==147456 && total_aw==132 && total_w==256 && total_b==132 && total_qaw==128 && total_qw==128 && total_qb==128 &&
            total_kar==160 && total_kr==1280 && total_kaw==204 && total_kw==768 && total_kb==204 && total_quant_words==1024 && total_root==3 && total_qacks==2 && total_packs==2 && total_outacks==2,"Suite accounting");
        $display("PRREAL_PASS seed=%0d jobs=%0d publics=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d ar_stalls=%0d r_stalls=%0d aw_stalls=%0d w_stalls=%0d cycles=%0d qreads=%0d qwords=%0d root=%0d qacks=%0d packs=%0d qaw=%0d qw=%0d qb=%0d heads=%0d copies=%0d qbeats=%0d binds=%0d appends=%0d kar=%0d kr=%0d kaw=%0d kw=%0d kb=%0d outacks=%0d",seed,jobs,publics,total_results,total_ar,total_r,total_aw,total_w,total_b,ar_stalls,r_stalls,aw_stalls,w_stalls,cycles,total_source_reads,total_quant_words,total_root,total_qacks,total_packs,total_qaw,total_qw,total_qb,total_heads,total_copies,total_qbeats,binds,appends,total_kar,total_kr,total_kaw,total_kw,total_kb,total_outacks);
        $finish;
    end
    initial begin #100000000;$fatal(1,"PRREAL_FAIL global timeout");end
endmodule
`default_nettype wire
