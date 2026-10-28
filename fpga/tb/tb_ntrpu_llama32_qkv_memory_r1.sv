`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_llama32_qkv_memory_r1;
    logic clk,rst_n,command_valid,cancel_valid,completion_ready;
    wire command_ready,cancel_ready,busy,completion_valid;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic [6:0][63:0] drive_addr,drive_base,drive_bytes;
    wire [63:0] command_addr[0:6],command_base[0:6],command_bytes[0:6];
    for(genvar i=0;i<7;i++)begin
        assign command_addr[i]=drive_addr[i];assign command_base[i]=drive_base[i];assign command_bytes[i]=drive_bytes[i];
    end
    logic source_completion_valid,source_result_valid;
    wire source_completion_ready;
    logic [3:0] source_completion_status;
    logic [63:0] source_sequence,source_epoch,source_result_address;
    logic [12:0] source_result_bytes;
    wire [3:0] completion_status,completion_source_status,completion_child_status;
    wire [63:0] completion_sequence,completion_epoch;
    wire [1:0] completion_phase,completion_matrices;
    wire [14:0] completion_total_bytes;
    wire [16:0] completion_child_bytes;
    wire [13:0] completion_child_rows;
    wire completion_revoked,result_valid,tags_exhausted;
    wire [63:0] result_q_address,result_k_address,result_v_address;
    wire [13:0] result_q_bytes,result_k_bytes,result_v_bytes;
    wire [63:0] m_axi_araddr,m_axi_awaddr;
    wire [7:0] m_axi_arlen,m_axi_awlen;
    wire [2:0] m_axi_arsize,m_axi_arprot,m_axi_awsize,m_axi_awprot;
    wire [1:0] m_axi_arburst,m_axi_awburst;
    wire [3:0] m_axi_arid,m_axi_arcache,m_axi_awid,m_axi_awcache;
    wire m_axi_arlock,m_axi_arvalid,m_axi_rready,m_axi_awlock,m_axi_awvalid,m_axi_wlast,m_axi_wvalid,m_axi_bready;
    logic m_axi_arready,m_axi_rvalid,m_axi_rlast,m_axi_awready,m_axi_wready,m_axi_bvalid;
    logic [511:0] m_axi_rdata;
    wire [511:0] m_axi_wdata;
    wire [63:0] m_axi_wstrb;
    logic [1:0] m_axi_rresp,m_axi_bresp;
    logic [3:0] m_axi_rid,m_axi_bid;
    localparam logic [63:0] ABASE=64'h8000000000,FIRST_TAG=64'h00000000ffffffc0;
    ntrpu_llama32_qkv_memory_r1 #(.INITIAL_CHILD_TAG(FIRST_TAG)) dut(.*);
    always #2.5 clk=~clk;
    logic [511:0] q_weights[0:69694],k_weights[0:17470],v_weights[0:17470];
    logic [511:0] activations[0:255],destination[0:191];
    logic [63:0] reference64[0:12287];
    logic [31:0] reference32[0:12287];
    string fixture;
    integer seed=1,random_state=1,cycles=0,jobs=0,mode_now=0,sample_now=0,target_now=0;
    integer phase_now=0,matrices=0,matrix_acks=0,good_matrices=0,source_acks=0,publics=0,cancel_acks=0;
    integer ars=0,rs=0,aws=0,ws=0,bs=0,groups=0,tiles=0,computed=0,replies=0;
    integer total_ar=0,total_r=0,total_aw=0,total_w=0,total_b=0,total_results=0,total_matrices=0,total_groups=0;
    integer r_index=0,r_beats=0,aw_beats=0,w_pending=0,write_cursor=0,aw_number=0;
    integer phase_start=0,job_start=0,ar_stalls=0,r_stalls=0,aw_stalls=0,w_stalls=0;
    integer job_ar=0,job_r=0,job_aw=0,job_w=0,job_b=0,job_results=0;
    logic r_active=0,aw_pending=0,hold_ar=0,hold_aw=0,hold_w=0,hold_r=0,done_seen=0,public_seen=0;
    logic [63:0] r_address,aw_address,sequence_model,epoch_model,tag_model,next_tag;
    logic [88:0] saved_ar,saved_aw;
    logic [576:0] saved_w;
    logic [518:0] saved_r;
    logic [1:0] pending_last;
    function automatic logic [63:0] weight_base(input integer phase);
        return 64'h100000000+64'(phase)*64'h20000000;
    endfunction
    function automatic logic [63:0] output_base(input integer phase);
        return 64'ha000000000+64'(phase)*64'h10000000;
    endfunction
    function automatic integer row_offset(input integer phase);
        return phase==0 ? 0:phase==1 ? 2048:2560;
    endfunction
    function automatic integer rows_for(input integer phase);
        return phase==0 ? 2048:512;
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"QKVM_FAIL job=%0d mode=%0d phase=%0d cycle=%0d: %s",jobs,mode_now,phase_now,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [511:0] read_word(input logic [63:0] address);
        if(address>=ABASE+4032 && address+64<=ABASE+8128)
            return activations[sample_now*64+int'((address-ABASE-4032)>>6)];
        if(!(address>=weight_base(phase_now)+4032 && address+64<=weight_base(phase_now)+(phase_now==0 ? 4460480:1118144)))
            $fatal(1,"QKVM_FAIL job=%0d phase=%0d cycle=%0d: Read outside current immutable matrix",jobs,phase_now,cycles);
        case(phase_now)
            0:return q_weights[int'((address-weight_base(0))>>6)];
            1:return k_weights[int'((address-weight_base(1))>>6)];
            default:return v_weights[int'((address-weight_base(2))>>6)];
        endcase
    endfunction
    function automatic logic [511:0] output_word(input integer index);
        logic [511:0] value;
        for(integer i=0;i<16;i++)value[i*32+:32]=reference32[sample_now*3072+index*16+i];
        return value;
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_arready=0;m_axi_rvalid=0;m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;end
        else begin
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
        if(!rst_n)begin hold_ar=0;hold_aw=0;hold_w=0;hold_r=0;r_active=0;aw_pending=0;w_pending=0;end
        else begin
            if(busy)require(!command_ready && completion_sequence==sequence_model && completion_epoch==epoch_model,"Captured public ownership");
            if(hold_ar)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock}===saved_ar,"Held AR changed");
            if(hold_aw)require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"Held AW changed");
            if(hold_w)require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"Held W changed");
            if(hold_r)require(m_axi_rvalid && {m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast}===saved_r,"Held R changed");
            hold_ar=m_axi_arvalid && !m_axi_arready;hold_aw=m_axi_awvalid && !m_axi_awready;hold_w=m_axi_wvalid && !m_axi_wready;hold_r=m_axi_rvalid && !m_axi_rready;
            saved_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};saved_r={m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast};
            ar_stalls+=int'(hold_ar);aw_stalls+=int'(hold_aw);w_stalls+=int'(hold_w);r_stalls+=int'(hold_r);
            if(dut.child_job_valid && dut.child_job_ready)begin
                require(matrices==matrix_acks && !r_active && !aw_pending && w_pending==0,"Overlapping matrix ownership");
                phase_now=matrices;require(phase_now<3,"Extra matrix");phase_start=cycles;
                ars=0;rs=0;aws=0;ws=0;bs=0;groups=0;tiles=0;computed=0;replies=0;write_cursor=0;done_seen=0;
                require({dut.child_job_epoch,dut.child_job_sequence}==tag_model+64'(phase_now*64),"Matrix identity");
                require(dut.child_job_rows==rows_for(phase_now) && dut.child_job_k_blocks==64 && dut.child_job_tile_k==32 && dut.child_job_layout==2 && dut.child_job_output_fp32,"Full QKV shape");
                require(dut.child_job_weight_addr==weight_base(phase_now)+4032 && dut.child_job_activation_addr==ABASE+4032 && dut.child_job_output_addr==output_base(phase_now)+4032,"Matrix descriptors");
                $display("QKVM_MATRIX cycle=%0d job=%0d phase=%0d tag=%016h",cycles,jobs,phase_now,{dut.child_job_epoch,dut.child_job_sequence});
                if(phase_now==0)next_tag=next_tag+192;
                matrices++;total_matrices++;
            end
            if(dut.matrix.e_job_valid && dut.matrix.e_job_ready)begin
                require(groups==replies && !r_active && !aw_pending && w_pending==0,"Group overlap");
                require(dut.matrix.e_job_rows==32 && dut.matrix.e_job_output_fp32 && dut.matrix.e_job_k_blocks==64,"Group shape");
                require({dut.matrix.e_job_epoch,dut.matrix.e_job_sequence}==tag_model+64'(phase_now*64+groups),"Group identity");
                require(dut.matrix.e_job_output_addr==output_base(phase_now)+4032+64'(groups*128),"Group output address");
                $display("QKVM_GROUP cycle=%0d job=%0d phase=%0d group=%0d tag=%016h",cycles,jobs,phase_now,groups,{dut.matrix.e_job_epoch,dut.matrix.e_job_sequence});groups++;total_groups++;
            end
            if(dut.matrix.e_tile_valid && dut.matrix.e_tile_ready)begin
                require(dut.matrix.e_tile_addr==weight_base(phase_now)+4032+64'(tiles*34816) && dut.matrix.e_tile_k_count==32 && dut.matrix.e_tile_k_base==9'((tiles%2)*32),"Tile sequence");
                $display("QKVM_TILE cycle=%0d job=%0d phase=%0d tile=%0d address=%016h",cycles,jobs,phase_now,tiles,dut.matrix.e_tile_addr);tiles++;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!r_active && groups>0 && m_axi_araddr[5:0]==0,"AR admission/alignment");
                require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR crosses 4 KiB");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR attributes");
                r_address=m_axi_araddr;r_beats=int'(m_axi_arlen)+1;r_index=0;r_active=1;
                $display("QKVM_AR cycle=%0d job=%0d phase=%0d ar=%0d address=%016h beats=%0d group=%0d",cycles,jobs,phase_now,ars,r_address,r_beats,groups-1);ars++;job_ar++;total_ar++;
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(r_active && m_axi_rlast==(r_index==r_beats-1),"Read burst accounting");
                $display("QKVM_R cycle=%0d job=%0d phase=%0d ar=%0d beat=%0d address=%016h data=%0128h last=%0d",cycles,jobs,phase_now,ars-1,r_index,r_address+64'(r_index*64),m_axi_rdata,m_axi_rlast);
                r_index++;rs++;job_r++;total_r++;if(m_axi_rlast)r_active=0;m_axi_rvalid<=0;
            end
            if(dut.matrix.engine.result_valid && dut.matrix.engine.result_ready)begin
                require(dut.matrix.engine.result_row==5'(computed%32),"Result row order");
                require(computed<rows_for(phase_now) && dut.matrix.engine.result_data===reference64[sample_now*3072+row_offset(phase_now)+computed],"Real QKV FP64 arithmetic");
                $display("QKVM_MAC cycle=%0d job=%0d phase=%0d row=%0d data=%016h",cycles,jobs,phase_now,computed,dut.matrix.engine.result_data);computed++;job_results++;total_results++;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awaddr==output_base(phase_now)+4032+64'(write_cursor*64),"AW phase address");
                require(m_axi_awlen<=1 && int'(m_axi_awaddr[11:0])+(int'(m_axi_awlen)+1)*64<=4096,"AW burst/page");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                aw_pending=1;aw_beats=int'(m_axi_awlen)+1;aw_address=m_axi_awaddr;aw_number=aws;write_cursor+=aw_beats;
                $display("QKVM_AW cycle=%0d job=%0d phase=%0d aw=%0d address=%016h beats=%0d",cycles,jobs,phase_now,aws,m_axi_awaddr,aw_beats);aws++;job_aw++;total_aw++;
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(ws<rows_for(phase_now)/16 && w_pending<2 && m_axi_wdata===output_word(row_offset(phase_now)/16+ws),"Real QKV FP32 AXI bytes");
                require(m_axi_wstrb==64'hffffffffffffffff,"FP32 byte mask");
                destination[row_offset(phase_now)/16+ws]=m_axi_wdata;pending_last[w_pending]=m_axi_wlast;w_pending++;
                $display("QKVM_W cycle=%0d job=%0d phase=%0d word=%0d data=%0128h last=%0d",cycles,jobs,phase_now,ws,m_axi_wdata,m_axi_wlast);ws++;job_w++;total_w++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending==aw_beats,"B before AW/W drain");
                $display("QKVM_B cycle=%0d job=%0d phase=%0d aw=%0d response=%0d",cycles,jobs,phase_now,aw_number,m_axi_bresp);
                bs++;job_b++;total_b++;aw_pending=0;w_pending=0;m_axi_bvalid<=0;
            end
            if(dut.matrix.e_done_valid && dut.matrix.e_done_ready)begin
                require(!r_active && !aw_pending && w_pending==0 && groups==replies+1,"Group completion before drain");
                $display("QKVM_GROUP_DONE cycle=%0d job=%0d phase=%0d group=%0d status=%0d bytes=%0d",cycles,jobs,phase_now,replies,dut.matrix.e_done_status,dut.matrix.e_done_bytes);replies++;
            end
            if(dut.child_cancel_valid && dut.child_cancel_ready)begin
                require(mode_now==2 && phase_now==target_now,"Unexpected child cancellation");cancel_acks++;
                $display("QKVM_CHILD_CANCEL cycle=%0d job=%0d phase=%0d",cycles,jobs,phase_now);
            end
            if(dut.child_done_valid && !done_seen)begin
                done_seen=1;
                require(!r_active && !aw_pending && w_pending==0 && aws==bs && groups==replies,"Matrix completion before drain");
                if(mode_now==2 && phase_now==target_now)begin
                    require(dut.child_done_status==7 && dut.child_done_rows==0 && dut.child_done_bytes==0 && computed==0 && ws==0,"Cancelled matrix publication");
                    require(groups==1 && tiles==1 && rs>=70,"Cancellation did not exercise active reads");
                end else begin
                    require(dut.child_done_status==0 && dut.child_done_rows==rows_for(phase_now) && dut.child_done_bytes==rows_for(phase_now)*4,"Successful matrix shape");
                    require(computed==rows_for(phase_now) && groups==rows_for(phase_now)/32 && tiles==rows_for(phase_now)/16 && ars==groups*21 && rs==groups*1152 && ws==rows_for(phase_now)/16 && aws==(phase_now==0 ? 66:17),"Full matrix coverage");good_matrices++;
                end
                $display("QKVM_MATRIX_DONE cycle=%0d job=%0d phase=%0d status=%0d bytes=%0d rows=%0d tag=%016h groups=%0d tiles=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d elapsed=%0d",cycles,jobs,phase_now,dut.child_done_status,dut.child_done_bytes,dut.child_done_rows,{dut.child_done_epoch,dut.child_done_sequence},groups,tiles,computed,ars,rs,aws,ws,bs,cycles-phase_start);
            end
            if(completion_valid && completion_ready)begin
                require(!public_seen,"Duplicate public completion");public_seen=1;publics++;
                $display("QKVM_PUBLIC cycle=%0d job=%0d status=%0d sequence=%016h epoch=%016h valid=%0d revoked=%0d matrices=%0d bytes=%0d",cycles,jobs,completion_status,completion_sequence,completion_epoch,result_valid,completion_revoked,completion_matrices,completion_total_bytes);
            end
            if(dut.child_done_valid && dut.child_done_ready)begin
                require({dut.child_done_epoch,dut.child_done_sequence}==tag_model+64'(phase_now*64),"Matrix completion identity");
                if(phase_now==2 || mode_now==2 && phase_now==target_now)require(public_seen,"Terminal matrix released early");
                $display("QKVM_MATRIX_ACK cycle=%0d job=%0d phase=%0d public=%0d",cycles,jobs,phase_now,public_seen);matrix_acks++;
            end
            if(source_completion_ready)begin
                require(source_completion_valid && public_seen && matrix_acks==matrices && !r_active && !aw_pending,"Source released before publication/drain");
                source_acks++;$display("QKVM_SOURCE_ACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(result_valid)begin
                require(completion_valid && completion_status==0 && good_matrices==3 && !completion_revoked && completion_total_bytes==12288,"Partial or revoked result published");
                require(result_q_address==output_base(0)+4032 && result_k_address==output_base(1)+4032 && result_v_address==output_base(2)+4032 && result_q_bytes==8192 && result_k_bytes==2048 && result_v_bytes==2048,"Public result descriptors");
                for(integer i=0;i<192;i++)require(destination[i]===output_word(i),"Published QKV memory");
            end else require(result_q_address==0 && result_k_address==0 && result_v_address==0 && result_q_bytes==0 && result_k_bytes==0 && result_v_bytes==0,"Private result leak");
        end
    end
    task automatic run_job(input integer mode,input integer sample_id,input integer target);
        integer limit,status_model,sa,pu,cc;
        @(negedge clk);require(!busy,"Previous owner active");jobs++;mode_now=mode;sample_now=sample_id;target_now=target;
        matrices=0;matrix_acks=0;good_matrices=0;done_seen=0;public_seen=0;phase_now=0;
        job_ar=0;job_r=0;job_aw=0;job_w=0;job_b=0;job_results=0;
        sa=source_acks;pu=publics;cc=cancel_acks;
        for(integer i=0;i<192;i++)destination[i]='1;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);tag_model=next_tag;
        command_sequence=sequence_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        source_completion_valid=1;source_completion_status=0;source_sequence=sequence_model;source_epoch=epoch_model;
        source_result_valid=1;source_result_address=ABASE+4032;source_result_bytes=4096;
        drive_addr[0]=ABASE+4032;drive_base[0]=ABASE;drive_bytes[0]=8128;
        for(integer i=0;i<3;i++)begin
            drive_addr[i+1]=weight_base(i)+4032;drive_base[i+1]=weight_base(i);drive_bytes[i+1]=i==0 ? 4460480:1118144;
            drive_addr[i+4]=output_base(i)+4032;drive_base[i+4]=output_base(i);drive_bytes[i+4]=4032+rows_for(i)*4;
        end
        if(mode==1)drive_bytes[6]=drive_bytes[6]-64;
        command_valid=1;job_start=cycles;#1;require(command_ready,"Command admission");
        $display("QKVM_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d target=%0d sequence=%016h epoch=%016h tag=%016h",cycles,jobs,mode,sample_id,target,sequence_model,epoch_model,tag_model);
        tick();@(negedge clk);command_valid=0;command_sequence='1;command_epoch='1;drive_addr='1;drive_base='1;drive_bytes='1;
        if(mode==2)begin
            wait(matrices==target+1 && phase_now==target && rs>=70);@(negedge clk);require(cancel_ready,"Cancel admission");cancel_valid=1;
            $display("QKVM_CANCEL cycle=%0d job=%0d phase=%0d reads=%0d",cycles,jobs,phase_now,rs);
            tick();@(negedge clk);cancel_valid=0;
        end
        limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<12000000,"QKV completion deadline");end
        status_model=mode==1 ? 1:mode==2 ? 7:0;
        require(completion_status==status_model && completion_sequence==sequence_model && completion_epoch==epoch_model,"Public completion identity/status");
        require(result_valid==(mode==0 || mode==3) && !r_active && !aw_pending && job_aw==job_b,"Completion result/drain");
        if(mode==0 || mode==3)require(good_matrices==3 && job_results==3072 && job_r==110592 && job_w==192 && job_aw==100,"Full QKV coverage");
        if(mode==1)require(matrices==0 && job_ar==0 && job_aw==0 && job_results==0,"Bounds before any traffic");
        if(mode==2)require(good_matrices==target && matrices==target+1 && cancel_acks==cc+1,"Cancel phase coverage");
        repeat(7)begin tick();require(completion_valid && completion_status==status_model && !command_ready && source_acks==sa,"Held public lifetime");end
        if(mode==3)begin
            @(negedge clk);expected_epoch=~epoch_model;#1;require(!result_valid && completion_revoked,"Immediate late revocation");tick();
            $display("QKVM_REVOKE cycle=%0d job=%0d",cycles,jobs);
            @(negedge clk);expected_epoch=epoch_model;repeat(3)tick();
            require(completion_valid && completion_status==0 && !result_valid && completion_revoked,"Epoch rollback revived result");
        end
        $display("QKVM_COMPLETE cycle=%0d job=%0d mode=%0d matrices=%0d good=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d elapsed=%0d",cycles,jobs,mode,matrices,good_matrices,job_results,job_ar,job_r,job_aw,job_w,job_b,cycles-job_start);
        @(negedge clk);completion_ready=1;tick();
        require(source_acks==sa+1 && publics==pu+1 && !busy && matrix_acks==matrices,"Exact final release");
        @(negedge clk);completion_ready=0;source_completion_valid=0;tick();require(!completion_valid && !result_valid,"Stale public output");
        $display("QKVM_END cycle=%0d job=%0d",cycles,jobs);
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;expected_epoch=0;
        source_completion_valid=0;source_completion_status=0;source_sequence=0;source_epoch=0;source_result_valid=0;source_result_address=0;source_result_bytes=0;
        command_sequence=0;command_epoch=0;drive_addr=0;drive_base=0;drive_bytes=0;next_tag=FIRST_TAG;
        m_axi_arready=0;m_axi_rvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;
        m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;m_axi_bresp=0;m_axi_bid=0;
        require($value$plusargs("FIXTURE=%s",fixture),"Missing real QKV fixture");
        $readmemh({fixture,"/attn_q-weights.hex"},q_weights);$readmemh({fixture,"/attn_k-weights.hex"},k_weights);$readmemh({fixture,"/attn_v-weights.hex"},v_weights);
        $readmemh({fixture,"/activation.hex"},activations);$readmemh({fixture,"/reference-fp64.hex"},reference64);$readmemh({fixture,"/reference-fp32.hex"},reference32);
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        $display("QKVM_CONFIG seed=%0d q_rows=2048 kv_rows=512 k_blocks=64 tile_k=32 watchdog=65536 phase_limit=16777216",seed);
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        run_job(0,0,0);run_job(0,1,0);run_job(0,2,0);run_job(3,3,0);run_job(1,0,0);
        run_job(2,0,0);run_job(2,1,1);run_job(2,2,2);run_job(0,3,0);
        require(jobs==9 && publics==9 && source_acks==9 && cancel_acks==3 && total_matrices==21 && total_groups==627 && total_results==19968 && total_w==1248 && total_aw==649 && total_b==649,"Suite coverage");
        $display("QKVM_PASS seed=%0d jobs=%0d publics=%0d source_acks=%0d cancels=%0d matrices=%0d groups=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d ar_stalls=%0d r_stalls=%0d aw_stalls=%0d w_stalls=%0d cycles=%0d",seed,jobs,publics,source_acks,cancel_acks,total_matrices,total_groups,total_results,total_ar,total_r,total_aw,total_w,total_b,ar_stalls,r_stalls,aw_stalls,w_stalls,cycles);
        $finish;
    end
    initial begin #500000000;$fatal(1,"QKVM_FAIL global timeout");end
endmodule
`default_nettype wire
