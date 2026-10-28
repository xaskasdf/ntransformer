`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_output_path_real_r1;
    logic clk,rst_n;
    logic command_valid;
    logic command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic [63:0] command_weight_addr,command_weight_region_base,command_weight_region_bytes;
    logic [63:0] command_activation_addr,command_activation_region_base,command_activation_region_bytes;
    logic [63:0] command_output_addr,command_output_region_base,command_output_region_bytes;
    logic cancel_valid;
    logic cancel_ready,busy;
    logic completion_valid;
    logic completion_ready;
    logic [3:0] completion_status;
    logic [63:0] completion_sequence,completion_epoch;
    logic result_valid;
    logic [63:0] result_address;
    logic [16:0] result_bytes;
    logic tag_exhausted;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize, m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid, m_axi_arcache;
    logic m_axi_arlock, m_axi_arvalid;
    logic m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast, m_axi_rvalid;
    logic m_axi_rready;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize, m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid, m_axi_awcache;
    logic m_axi_awlock, m_axi_awvalid;
    logic m_axi_awready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast, m_axi_wvalid;
    logic m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic m_axi_bvalid;
    logic m_axi_bready;
    localparam logic [63:0] WBASE=64'h100000000,ABASE=64'h8000000000,OBASE=64'ha000000000;
    localparam logic [63:0] FIRST_TAG=64'h00000000ffffff80;
    ntrpu_attention_output_path_r1 #(.INITIAL_QUANT_TAG(FIRST_TAG),.INITIAL_PROJECTION_TAG(FIRST_TAG)) dut(.*);
    logic source_completion_valid=0,source_completion_ready,source_result_valid=0;
    logic [3:0] source_completion_status=0;
    logic [63:0] source_sequence=0,source_epoch=0;
    logic source_result_ce,source_result_q_valid=0;
    logic [10:0] source_result_address;
    logic [31:0] source_result_q=0,completion_hls_status;
    logic [31:0] fp32_image[0:8191];
    logic [511:0] materialized_activation[0:63],pending_data[0:1];
    logic [63:0] activation_written=0;
    logic [2047:0] seen_source;
    logic input_pending=0,source_reused=0,quant_public_seen=0,aw_projection=0;
    integer pending_input_address=0,source_reads=0,quant_words=0,root_acks=0,qacks=0,packs=0,starts=0,dones=0;
    integer qaws=0,qws=0,qbs=0,total_qaw=0,total_qw=0,total_qb=0;
    integer total_source_reads=0,total_quant_words=0,total_root=0,total_qacks=0,total_packs=0;
    always #2.5 clk=~clk;
    logic [511:0] weight_image[0:69694],activation_image[0:255],destination[0:127];
    logic [63:0] reference64[0:8191];logic [31:0] reference32[0:8191];
    string fixture;
    integer seed=1,random_state=1,cycles=0,jobs=0,mode_now=0,sample_now=0;
    integer ars=0,rs=0,aws=0,ws=0,bs=0,groups=0,tiles=0,computed=0,replies=0;
    integer total_ar=0,total_r=0,total_aw=0,total_w=0,total_b=0,total_results=0,publics=0;
    integer r_index=0,r_beats=0,aw_beats=0,w_pending=0,write_cursor=0,aw_number=0;
    integer quiet=0,max_quiet=0,job_start=0,ar_stalls=0,r_stalls=0,aw_stalls=0,w_stalls=0;
    logic r_active=0,aw_pending=0,hold_ar=0,hold_aw=0,hold_w=0,hold_r=0;
    logic [63:0] r_address,aw_address,sequence_model,epoch_model,tag_model;
    logic [88:0] saved_ar,saved_aw;logic [576:0] saved_w;logic [518:0] saved_r;
    logic [1:0] pending_last;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"OPATH_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,mode_now,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [511:0] read_word(input logic [63:0] address);
        if(address>=WBASE+4032 && address+64<=WBASE+4460480)return weight_image[int'((address-WBASE)>>6)];
        if(!(address>=ABASE+4032 && address+64<=ABASE+8128))
            $fatal(1,"OPATH_FAIL job=%0d cycle=%0d: Read outside immutable images",jobs,cycles);
        if(!activation_written[int'((address-ABASE-4032)>>6)])
            $fatal(1,"OPATH_FAIL read before committed quantized memory");
        return materialized_activation[int'((address-ABASE-4032)>>6)];
    endfunction
    function automatic logic [511:0] output_word(input int index);
        logic [511:0] value;
        for(integer i=0;i<16;i++)value[i*32+:32]=reference32[sample_now*2048+index*16+i];
        return value;
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_arready=0;m_axi_rvalid=0;m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;end
        else begin
            if(root_acks==1 && !source_reused)begin
                source_sequence=sequence_model^64'h0000010000000000;
                source_epoch=epoch_model^64'h0000100000000000;
                source_completion_valid=1;source_result_valid=1;source_result_q=32'h7fc00001;source_reused=1;
                $display("OPATH_REUSE cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,source_sequence,source_epoch);
            end
            m_axi_arready=$urandom_range(0,3)!=0;m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if(r_active && !m_axi_rvalid && $urandom_range(0,3)!=0)begin
                m_axi_rdata=read_word(r_address+64'(r_index*64));m_axi_rresp=0;m_axi_rid=0;
                m_axi_rlast=r_index==r_beats-1;m_axi_rvalid=1;
            end
            if(aw_pending && w_pending==aw_beats && !m_axi_bvalid && $urandom_range(0,3)!=0)begin
                for(integer i=0;i<w_pending;i++)require(pending_last[i]==(i==aw_beats-1),"Write burst last");
                m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;
            end
        end
    end
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin hold_ar=0;hold_r=0;hold_aw=0;hold_w=0;r_active=0;aw_pending=0;w_pending=0;source_result_q_valid<=0;input_pending=0;end
        else begin
            if(input_pending)begin
                require(dut.quantize_store.hls_input_q===fp32_image[sample_now*2048+pending_input_address],"Generated quantizer synchronous input");
                $display("OPATH_INPUT cycle=%0d job=%0d index=%0d data=%08h",cycles,jobs,pending_input_address,dut.quantize_store.hls_input_q);
            end
            input_pending=source_result_ce;pending_input_address=int'(source_result_address);
            source_result_q_valid<=source_result_ce && source_result_valid;
            if(source_result_ce)begin
                require(root_acks==0 && source_completion_valid && source_result_valid && source_sequence==sequence_model && source_epoch==epoch_model,"Source read after release or wrong identity");
                source_result_q<=fp32_image[sample_now*2048+int'(source_result_address)];
                seen_source[source_result_address]=1;source_reads++;total_source_reads++;
                $display("OPATH_READ cycle=%0d job=%0d index=%0d data=%08h",cycles,jobs,source_result_address,fp32_image[sample_now*2048+int'(source_result_address)]);
            end
            if(dut.quantize_store.hls_start)begin
                starts++;require(starts==1 && dut.quantize_store.hls_elements==2048,"Quantizer launch extent");
                $display("OPATH_START cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.quantize_store.hls_output_ce && dut.quantize_store.hls_output_we)begin
                require(quant_words<512 && dut.quantize_store.hls_output_address==9'(quant_words),"ABI1 word order");
                require(dut.quantize_store.hls_output_d===activation_image[sample_now*64+quant_words/8][(quant_words%8)*64+:64],"Real quantization bytes");
                $display("OPATH_WORD cycle=%0d job=%0d index=%0d data=%016h",cycles,jobs,quant_words,dut.quantize_store.hls_output_d);
                quant_words++;total_quant_words++;
            end
            if(dut.quantize_store.hls_done)begin
                dones++;require(dones==1 && dut.quantize_store.hls_ready && dut.quantize_store.hls_return==0 && quant_words==512 && seen_source=='1,"Quantizer full result");
                $display("OPATH_DONE cycle=%0d job=%0d status=%0d words=%0d",cycles,jobs,dut.quantize_store.hls_return,quant_words);
            end
            if(source_completion_valid && source_completion_ready)begin
                require(root_acks==0 && dones==1 && source_sequence==sequence_model && source_epoch==epoch_model,"Upstream release");
                root_acks++;total_root++;source_completion_valid<=0;source_result_valid<=0;
                $display("OPATH_ROOT_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,source_sequence,source_epoch);
            end
            if(dut.q_completion_valid && !quant_public_seen)begin
                require(qaws==64 && qws==64 && qbs==64 && activation_written=='1 && !aw_pending && w_pending==0,"ABI1 publication before AXI drain");
                require(dut.q_completion_status==0 && dut.q_completion_hls_status==0 && dut.q_completion_records==64 && dut.q_result_valid && dut.q_result_address==ABASE+4032 && dut.q_result_bytes==4096,"ABI1 publication contract");
                quant_public_seen=1;
                $display("OPATH_QUANT_PUBLIC cycle=%0d job=%0d status=%0d hls=%0d sequence=%016h epoch=%016h records=%0d valid=%0d address=%016h bytes=%0d",cycles,jobs,dut.q_completion_status,dut.q_completion_hls_status,dut.q_completion_sequence,dut.q_completion_epoch,dut.q_completion_records,dut.q_result_valid,dut.q_result_address,dut.q_result_bytes);
            end
            if(dut.projection.command_valid && dut.projection.command_ready)begin
                require(quant_public_seen && qacks==0 && dut.q_completion_valid,"Projection without retained quantized region");
                require(dut.projection.command_sequence==sequence_model && dut.projection.command_epoch==epoch_model,"Projection parent identity");
                $display("OPATH_PROJECTION_COMMAND cycle=%0d job=%0d sequence=%016h epoch=%016h address=%016h",cycles,jobs,dut.projection.command_sequence,dut.projection.command_epoch,dut.projection.command_activation_addr);
            end
            if(dut.quantize_store.completion_valid && dut.quantize_store.completion_ready)begin
                require(completion_valid && completion_ready && !r_active && !aw_pending && w_pending==0,"Early region release");qacks++;total_qacks++;
                $display("OPATH_QACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.q_completion_sequence,dut.q_completion_epoch);
            end
            if(dut.projection.completion_valid && dut.projection.completion_ready)begin
                require(completion_valid && completion_ready && dut.quantize_store.completion_ready,"Split child ownership release");packs++;total_packs++;
                $display("OPATH_PACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.p_completion_sequence,dut.p_completion_epoch);
            end
            if(dut.projection.control.active_child)begin
                if(dut.projection.child_activity)quiet=0;else quiet++;
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
            if(dut.projection.child_job_valid && dut.projection.child_job_ready)begin
                require({dut.projection.child_job_epoch,dut.projection.child_job_sequence}==tag_model,"Reserved matrix identity");
                require(dut.projection.child_job_rows==2048 && dut.projection.child_job_k_blocks==64 && dut.projection.child_job_tile_k==32 && dut.projection.child_job_layout==2 && dut.projection.child_job_output_fp32,"Full O shape");
                $display("OPATH_MATRIX cycle=%0d job=%0d tag=%016h",cycles,jobs,{dut.projection.child_job_epoch,dut.projection.child_job_sequence});
            end
            if(dut.projection.matrix.e_job_valid && dut.projection.matrix.e_job_ready)begin
                require(groups==replies && !r_active && !aw_pending && w_pending==0,"Group overlap");
                require(dut.projection.matrix.e_job_rows==32 && dut.projection.matrix.e_job_output_fp32 && dut.projection.matrix.e_job_k_blocks==64,"Group shape");
                require({dut.projection.matrix.e_job_epoch,dut.projection.matrix.e_job_sequence}==tag_model+64'(groups),"Group identity");
                require(dut.projection.matrix.e_job_output_addr==OBASE+4032+64'(groups*128),"Group output address");
                $display("OPATH_GROUP cycle=%0d job=%0d group=%0d tag=%016h",cycles,jobs,groups,{dut.projection.matrix.e_job_epoch,dut.projection.matrix.e_job_sequence});groups++;
            end
            if(dut.projection.matrix.e_tile_valid && dut.projection.matrix.e_tile_ready)begin
                require(dut.projection.matrix.e_tile_addr==WBASE+4032+64'(tiles*34816) && dut.projection.matrix.e_tile_k_count==32 && dut.projection.matrix.e_tile_k_base==9'((tiles%2)*32),"O tile sequence");
                $display("OPATH_TILE cycle=%0d job=%0d tile=%0d address=%016h",cycles,jobs,tiles,dut.projection.matrix.e_tile_addr);tiles++;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!r_active && groups>0 && m_axi_araddr[5:0]==0,"AR admission/alignment");
                require(qbs==64 && qacks==0 && dut.q_completion_valid && activation_written=='1,"Read outside retained ABI1 lifetime");
                require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR crosses 4 KiB");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR attributes");
                r_address=m_axi_araddr;r_beats=int'(m_axi_arlen)+1;r_index=0;r_active=1;
                $display("OPATH_AR cycle=%0d job=%0d ar=%0d address=%016h beats=%0d group=%0d",cycles,jobs,ars,r_address,r_beats,groups-1);ars++;total_ar++;
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(r_active && m_axi_rlast==(r_index==r_beats-1),"Read burst accounting");
                $display("OPATH_R cycle=%0d job=%0d ar=%0d beat=%0d address=%016h data=%0128h last=%0d",cycles,jobs,ars-1,r_index,r_address+64'(r_index*64),m_axi_rdata,m_axi_rlast);
                r_index++;rs++;total_r++;if(m_axi_rlast)r_active=0;m_axi_rvalid<=0;
            end
            if(dut.projection.matrix.engine.result_valid && dut.projection.matrix.engine.result_ready)begin
                require(dut.projection.matrix.engine.result_row==5'(computed%32),"Result row order");
                require(dut.projection.matrix.engine.result_data===reference64[sample_now*2048+computed],"Real O FP64 arithmetic");
                $display("OPATH_MAC cycle=%0d job=%0d row=%0d data=%016h",cycles,jobs,computed,dut.projection.matrix.engine.result_data);computed++;total_results++;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending,"Overlapping write bursts");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                aw_projection=dut.projection_owned;
                if(aw_projection)begin
                    require(m_axi_awaddr==OBASE+4032+64'(write_cursor*64),"O AW address");
                    require(m_axi_awlen<=1 && int'(m_axi_awaddr[11:0])+(int'(m_axi_awlen)+1)*64<=4096,"O AW page");
                    aw_number=aws;write_cursor+=int'(m_axi_awlen)+1;
                    $display("OPATH_AW cycle=%0d job=%0d aw=%0d address=%016h beats=%0d",cycles,jobs,aws,m_axi_awaddr,int'(m_axi_awlen)+1);aws++;total_aw++;
                end else begin
                    require(root_acks==1 && qaws<64 && qaws==qbs && m_axi_awaddr==ABASE+4032+64'(qaws*64) && m_axi_awlen==0,"ABI1 AW ownership/address");
                    aw_number=qaws;
                    $display("OPATH_QAW cycle=%0d job=%0d record=%0d address=%016h",cycles,jobs,qaws,m_axi_awaddr);qaws++;total_qaw++;
                end
                aw_pending=1;aw_beats=int'(m_axi_awlen)+1;aw_address=m_axi_awaddr;
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(w_pending<2 && m_axi_wstrb==64'hffffffffffffffff,"W capacity/mask");
                if(dut.projection_owned)begin
                    require(ws<128 && m_axi_wdata===output_word(ws),"Real O FP32 AXI bytes");
                    destination[ws]=m_axi_wdata;
                    $display("OPATH_W cycle=%0d job=%0d word=%0d data=%0128h last=%0d",cycles,jobs,ws,m_axi_wdata,m_axi_wlast);ws++;total_w++;
                end else begin
                    require(root_acks==1 && qws<64 && qws==qbs && m_axi_wlast && m_axi_wdata===activation_image[sample_now*64+qws],"Materialized ABI1 W bytes");
                    $display("OPATH_QW cycle=%0d job=%0d record=%0d data=%0128h",cycles,jobs,qws,m_axi_wdata);qws++;total_qw++;
                end
                pending_data[w_pending]=m_axi_wdata;pending_last[w_pending]=m_axi_wlast;w_pending++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending==aw_beats && aw_projection==dut.projection_owned,"B drain/owner");
                if(aw_projection)begin
                    $display("OPATH_B cycle=%0d job=%0d aw=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);bs++;total_b++;
                end else begin
                    require(aw_number==qbs && qaws==qbs+1 && qws==qbs+1 && !activation_written[aw_number],"ABI1 commit order");
                    materialized_activation[aw_number]=pending_data[0];activation_written[aw_number]=1;
                    $display("OPATH_QB cycle=%0d job=%0d record=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);qbs++;total_qb++;
                end
                aw_pending=0;w_pending=0;m_axi_bvalid<=0;
            end
            if(dut.projection.matrix.e_done_valid && dut.projection.matrix.e_done_ready)begin
                require(!r_active && !aw_pending && w_pending==0 && groups==replies+1,"Group completion before AXI drain");
                $display("OPATH_GROUP_DONE cycle=%0d job=%0d group=%0d status=%0d bytes=%0d",cycles,jobs,replies,dut.projection.matrix.e_done_status,dut.projection.matrix.e_done_bytes);replies++;
            end
            if(dut.projection.child_done_valid && dut.projection.child_done_ready)begin
                require({dut.projection.child_done_epoch,dut.projection.child_done_sequence}==tag_model,"Matrix completion identity");
                $display("OPATH_MATRIX_DONE cycle=%0d job=%0d status=%0d bytes=%0d rows=%0d tag=%016h",cycles,jobs,dut.projection.child_done_status,dut.projection.child_done_bytes,dut.projection.child_done_rows,{dut.projection.child_done_epoch,dut.projection.child_done_sequence});
            end
            if(result_valid)begin
                require(completion_valid && computed==2048 && ws==128 && aws==bs && !r_active && !aw_pending && w_pending==0,"Publication before completion/drain");
                require(result_address==OBASE+4032 && result_bytes==8192,"Public output extent");
                for(integer i=0;i<128;i++)require(destination[i]===output_word(i),"Published FP32 memory");
            end else require(result_address==0 && result_bytes==0,"Private result leak");
        end
    end
    task automatic run_job(input integer mode,input integer sample_id);
        integer limit;logic [3:0] status_model;
        @(negedge clk);require(!busy && !completion_valid,"Previous owner active");jobs++;mode_now=mode;sample_now=sample_id;
        ars=0;rs=0;aws=0;ws=0;bs=0;groups=0;tiles=0;computed=0;replies=0;quiet=0;max_quiet=0;write_cursor=0;
        source_reads=0;quant_words=0;root_acks=0;qacks=0;packs=0;starts=0;dones=0;
        qaws=0;qws=0;qbs=0;activation_written=0;seen_source=0;source_reused=0;quant_public_seen=0;input_pending=0;
        for(integer i=0;i<64;i++)materialized_activation[i]='x;
        for(integer i=0;i<128;i++)destination[i]='1;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);tag_model=FIRST_TAG+64'((jobs-1)*64);
        command_sequence=sequence_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_weight_addr=WBASE+4032;command_weight_region_base=WBASE;command_weight_region_bytes=4460480;
        command_activation_addr=ABASE+4032;command_activation_region_base=ABASE;command_activation_region_bytes=8128;
        command_output_addr=OBASE+4032;command_output_region_base=OBASE;command_output_region_bytes=12224;
        source_sequence=sequence_model;source_epoch=epoch_model;source_completion_valid=1;source_completion_status=0;source_result_valid=1;
        command_valid=1;job_start=cycles;
        $display("OPATH_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d sequence=%016h epoch=%016h tag=%016h",cycles,jobs,mode,sample_now,sequence_model,epoch_model,tag_model);
        #1;require(command_ready,"Admission after source completion");tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;
        command_weight_addr=0;command_weight_region_base=0;command_weight_region_bytes=0;
        command_activation_addr=0;command_activation_region_base=0;command_activation_region_bytes=0;
        command_output_addr=0;command_output_region_base=0;command_output_region_bytes=0;
        if(mode==1)begin
            wait(dut.projection.command_valid);@(negedge clk);require(cancel_ready,"Handoff cancel admission");cancel_valid=1;
            $display("OPATH_CANCEL cycle=%0d job=%0d reads=%0d",cycles,jobs,rs);
            tick();@(negedge clk);cancel_valid=0;
        end
        limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<3000000,"Full O completion timeout");end
        status_model=mode==1 ? 7:0;
        require(completion_status==status_model && completion_sequence==sequence_model && completion_epoch==epoch_model,"Full parent completion");
        require(result_valid==(mode==0) && !r_active && !aw_pending && w_pending==0 && aws==bs,"Completion publication/drain");
        if(mode==0)require(computed==2048 && groups==64 && replies==64 && tiles==128 && ars==1344 && rs==73728 && aws==66 && ws==128 && bs==66,"Full O coverage");
        else require(computed==0 && aws==0 && ws==0 && bs==0,"Failed O publication");
        if(mode==1)require(ars==0 && rs==0 && groups==0 && tiles==0,"Handoff cancel before reads");
        require(quant_words==512 && starts==1 && dones==1 && root_acks==1 && qaws==64 && qws==64 && qbs==64 && qacks==0 && packs==0 && completion_hls_status==0,"Full quantization and retained region");
        require(max_quiet<65536,"Parent watchdog activity gap");
        repeat(7)begin tick();require(completion_valid && completion_status==status_model && !command_ready,"Held public completion");end
        $display("OPATH_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h valid=%0d bytes=%0d groups=%0d tiles=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d max_quiet=%0d",cycles,jobs,mode,completion_status,completion_sequence,completion_epoch,result_valid,result_bytes,groups,tiles,computed,ars,rs,aws,ws,bs,max_quiet);
        publics++;@(negedge clk);completion_ready=1;tick();@(negedge clk);completion_ready=0;source_completion_valid=0;source_result_valid=0;
        require(!busy && !completion_valid && qacks==1 && packs==1,"Atomic child owner release");
        $display("OPATH_RELEASE cycle=%0d job=%0d",cycles,jobs);
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;expected_epoch=0;
        m_axi_arready=0;m_axi_rvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;
        m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;m_axi_bresp=0;m_axi_bid=0;
        require($value$plusargs("FIXTURE=%s",fixture),"Missing real O fixture");
        $readmemh({fixture,"/weight.hex"},weight_image);$readmemh({fixture,"/activation.hex"},activation_image);
        $readmemh({fixture,"/attention-fp32.hex"},fp32_image);
        $readmemh({fixture,"/reference-fp64.hex"},reference64);$readmemh({fixture,"/reference-fp32.hex"},reference32);
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        $display("OPATH_CONFIG seed=%0d rows=2048 k_blocks=64 tile_k=32 watchdog=65536",seed);
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        run_job(0,0);run_job(0,1);run_job(1,3);run_job(0,2);
        require(jobs==4 && publics==4 && total_results==6144 && total_aw==198 && total_w==384 && total_b==198 && total_qaw==256 && total_qw==256 && total_qb==256 && total_root==4 && total_qacks==4 && total_packs==4,"Suite coverage");
        $display("OPATH_PASS seed=%0d jobs=%0d publics=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d ar_stalls=%0d r_stalls=%0d aw_stalls=%0d w_stalls=%0d cycles=%0d qreads=%0d qwords=%0d root=%0d qacks=%0d packs=%0d qaw=%0d qw=%0d qb=%0d",seed,jobs,publics,total_results,total_ar,total_r,total_aw,total_w,total_b,ar_stalls,r_stalls,aw_stalls,w_stalls,cycles,total_source_reads,total_quant_words,total_root,total_qacks,total_packs,total_qaw,total_qw,total_qb);
        $finish;
    end
    initial begin #100000000;$fatal(1,"OPATH_FAIL global timeout");end
endmodule
`default_nettype wire
