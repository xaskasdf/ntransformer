`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_final_vocab_arithmetic_r1;
    logic clk,rst_n,command_valid,cancel_valid,completion_ready;
    wire command_ready,cancel_ready,busy,completion_valid;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic [4:0][63:0] drive_addr,drive_base,drive_bytes;
    wire [63:0] command_addr[0:4],command_base[0:4],command_bytes[0:4];
    for(genvar i=0;i<5;i++)begin
        assign command_addr[i]=drive_addr[i];assign command_base[i]=drive_base[i];assign command_bytes[i]=drive_bytes[i];
    end
    logic [63:0] command_source_sequence,command_source_epoch;
    logic command_source_valid;
    wire [63:0] completion_sequence,completion_epoch;
    wire [3:0] completion_status,completion_norm_status,completion_vocab_status,completion_matrix_status;
    wire [31:0] completion_norm_hls_status;
    wire [9:0] completion_norm_read_beats;
    wire [6:0] completion_norm_records;
    wire [3:0] completion_phase;
    wire [4:0] completion_matrices;
    wire [18:0] completion_total_bytes;
    wire completion_revoked,result_valid,norm_tag_exhausted,vocab_tags_exhausted;
    wire [63:0] result_address;
    wire [18:0] result_bytes;
    wire source_completion_valid=dut.norm_completion_valid;
    wire source_completion_ready=dut.norm_completion_ready;
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
    localparam logic [63:0] ABASE=64'h8000000000,FIRST_TAG=64'h00000000fffff000;
    ntrpu_final_vocab_real_r1 #(.INITIAL_VOCAB_TAG(FIRST_TAG)) dut(.*);
    always #2.5 clk=~clk;
    logic [511:0] weights[0:278590];
    logic [511:0] activations[0:63],destination[0:8015];
    logic [63:0] reference64[0:128255];
    logic [31:0] reference32[0:128255];
    string fixture,norm_fixture;
    logic [31:0] source_words[0:4095];
    logic [511:0] abi_memory[0:63],pending_data[0:1];
    logic [63:0] abi_committed=0;
    integer n_ar=0,n_r=0,n_aw=0,n_w=0,n_b=0,n_starts=0,n_dones=0,n_words=0,n_reads=0;
    integer n_total_ar=0,n_total_r=0,n_total_w=0,n_total_b=0,n_total_starts=0,n_total_dones=0,n_total_words=0;
    logic norm_core_busy=0;
    wire norm_hls_start=dut.normalization.hls_start;
    wire norm_hls_done=dut.normalization.hls_done;
    wire norm_hls_ready=dut.normalization.hls_ready;
    wire [31:0] norm_hls_return=dut.normalization.hls_return;
    wire [1:0] norm_hls_input_ce=dut.normalization.hls_input_ce;
    wire norm_hls_output_we=dut.normalization.hls_output_ce && dut.normalization.hls_output_we;
    wire [8:0] norm_hls_output_address=dut.normalization.hls_output_address;
    wire [63:0] norm_hls_output_data=dut.normalization.hls_output_d;
    function automatic logic [63:0] norm_base(input integer plane);
        return 64'h6000000000+64'(plane)*64'h10000000;
    endfunction
    function automatic logic [511:0] norm_input(input integer plane,input integer beat);
        logic [511:0] value;
        for(integer i=0;i<16;i++)begin
            value[i*32+:32]=source_words[plane*2048+beat*16+i];
        end
        return value;
    endfunction

    integer result_file=0,phase_file=0;
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
        return 64'h100000000+64'(phase)*64'd17825792;
    endfunction
    function automatic logic [63:0] output_base(input integer phase);
        return 64'ha000000000+64'(phase)*64'd32768;
    endfunction
    function automatic integer row_offset(input integer phase);
        return phase*8192;
    endfunction
    function automatic integer rows_for(input integer phase);
        return phase==15 ? 5376:8192;
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"FVREAL_FAIL job=%0d mode=%0d phase=%0d cycle=%0d: %s",jobs,mode_now,phase_now,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [511:0] read_word(input logic [63:0] address);
        for(integer p=0;p<2;p++)if(address>=norm_base(p)+4032 && address+64<=norm_base(p)+12224)begin
            if(dut.vocab_bus_owned)$fatal(1,"FVREAL_FAIL QKV read norm input");
            return norm_input(p,int'((address-norm_base(p)-4032)>>6));
        end
        if(address>=ABASE+4032 && address+64<=ABASE+8128)begin
            if(!dut.vocab_bus_owned || !abi_committed[int'((address-ABASE-4032)>>6)] || n_b!=64)
                $fatal(1,"FVREAL_FAIL ABI1 read before committed normalization write");
            return abi_memory[int'((address-ABASE-4032)>>6)];
        end
        if(!(address>=weight_base(phase_now)+4032 && address+64<=weight_base(phase_now)+(4032+rows_for(phase_now)*2176)))
            $fatal(1,"FVREAL_FAIL job=%0d phase=%0d cycle=%0d: Read outside current immutable matrix",jobs,phase_now,cycles);
        return weights[int'((address-weight_base(phase_now))>>6)];
    endfunction
    function automatic logic [511:0] output_word(input integer index);
        logic [511:0] value;
        for(integer i=0;i<16;i++)value[i*32+:32]=reference32[index*16+i];
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
            if(dut.vocab_bus_owned)begin
                require(!dut.norm_m_axi_arvalid && !dut.norm_m_axi_awvalid && !dut.norm_m_axi_wvalid,"Norm bus not drained at QKV ownership");
                require(dut.norm_completion_valid,"Lost norm lease during QKV");
            end else require(!dut.vocab_m_axi_arvalid && !dut.vocab_m_axi_awvalid && !dut.vocab_m_axi_wvalid,"Unowned QKV bus traffic");
            if(norm_hls_start)begin
                require(!norm_core_busy && n_r==256 && !r_active && n_aw==0,"Norm start before reads drained");
                norm_core_busy=1;n_starts++;n_total_starts++;n_words=0;n_reads=0;
                $display("FVREAL_NSTART cycle=%0d job=%0d",cycles,jobs);
            end
            if(norm_core_busy)begin
                if(|norm_hls_input_ce)n_reads++;
                if(norm_hls_output_we)begin
                    require(norm_hls_output_address==n_words && n_words<512,"Norm HLS output order");
                    require(norm_hls_output_data===activations[n_words/8][(n_words%8)*64+:64],"Actual norm HLS ABI1 bytes");
                    $display("FVREAL_NWRITE cycle=%0d job=%0d index=%0d data=%016h",cycles,jobs,n_words,norm_hls_output_data);
                    n_words++;n_total_words++;
                end
            end
            if(norm_hls_done)begin
                require(norm_core_busy && norm_hls_ready,"Norm HLS terminal protocol");
                norm_core_busy=0;n_dones++;n_total_dones++;
                require(norm_hls_return==0 && n_words==512,"Norm arithmetic termination");
                $display("FVREAL_NDONE cycle=%0d job=%0d status=%0d words=%0d",cycles,jobs,norm_hls_return,n_words);
            end
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
            if(dut.projection.child_job_valid && dut.projection.child_job_ready)begin
                require(matrices==matrix_acks && !r_active && !aw_pending && w_pending==0,"Overlapping matrix ownership");
                require(n_b==64 && abi_committed==64'hffffffffffffffff && n_dones==1 && dut.norm_completion_valid && dut.norm_result_valid,"Vocabulary before real norm publication");
                phase_now=matrices;require(phase_now<16,"Extra matrix");phase_start=cycles;
                ars=0;rs=0;aws=0;ws=0;bs=0;groups=0;tiles=0;computed=0;replies=0;write_cursor=0;done_seen=0;
                require({dut.projection.child_job_epoch,dut.projection.child_job_sequence}==tag_model+64'(phase_now*256),"Matrix identity");
                require(dut.projection.child_job_rows==rows_for(phase_now) && dut.projection.child_job_k_blocks==64 && dut.projection.child_job_tile_k==32 && dut.projection.child_job_layout==2 && dut.projection.child_job_output_fp32,"Full vocabulary segment shape");
                require(dut.projection.child_job_weight_addr==weight_base(phase_now)+4032 && dut.projection.child_job_activation_addr==ABASE+4032 && dut.projection.child_job_output_addr==output_base(phase_now)+4032,"Matrix descriptors");
                $display("FVREAL_MATRIX cycle=%0d job=%0d phase=%0d tag=%016h",cycles,jobs,phase_now,{dut.projection.child_job_epoch,dut.projection.child_job_sequence});
                if(phase_now==0)next_tag=next_tag+4096;
                $readmemh($sformatf("%s/segment-%02d-weights.hex",fixture,phase_now),weights,0,(4032+rows_for(phase_now)*2176)/64-1);
                matrices++;total_matrices++;
            end
            if(dut.projection.matrix.e_job_valid && dut.projection.matrix.e_job_ready)begin
                require(groups==replies && !r_active && !aw_pending && w_pending==0,"Group overlap");
                require(dut.projection.matrix.e_job_rows==32 && dut.projection.matrix.e_job_output_fp32 && dut.projection.matrix.e_job_k_blocks==64,"Group shape");
                require({dut.projection.matrix.e_job_epoch,dut.projection.matrix.e_job_sequence}==tag_model+64'(phase_now*256+groups),"Group identity");
                require(dut.projection.matrix.e_job_output_addr==output_base(phase_now)+4032+64'(groups*128),"Group output address");
                $display("FVREAL_GROUP cycle=%0d job=%0d phase=%0d group=%0d tag=%016h",cycles,jobs,phase_now,groups,{dut.projection.matrix.e_job_epoch,dut.projection.matrix.e_job_sequence});groups++;total_groups++;
            end
            if(dut.projection.matrix.e_tile_valid && dut.projection.matrix.e_tile_ready)begin
                require(dut.projection.matrix.e_tile_addr==weight_base(phase_now)+4032+64'(tiles*34816) && dut.projection.matrix.e_tile_k_count==32 && dut.projection.matrix.e_tile_k_base==9'((tiles%2)*32),"Tile sequence");
                $display("FVREAL_TILE cycle=%0d job=%0d phase=%0d tile=%0d address=%016h",cycles,jobs,phase_now,tiles,dut.projection.matrix.e_tile_addr);tiles++;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                if(dut.vocab_bus_owned)begin
                require(!r_active && groups>0 && m_axi_araddr[5:0]==0,"AR admission/alignment");
                require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR crosses 4 KiB");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR attributes");
                r_address=m_axi_araddr;r_beats=int'(m_axi_arlen)+1;r_index=0;r_active=1;
                ars++;job_ar++;total_ar++;
                end else begin
                require(!r_active && !norm_core_busy && n_aw==0,"Norm AR ownership");
                require(m_axi_araddr==norm_base(n_r/128)+4032+64'((n_r%128)*64),"Norm AR address");
                require(m_axi_arlen<64 && int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"Norm AR extent");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"Norm AR attributes");
                r_address=m_axi_araddr;r_beats=int'(m_axi_arlen)+1;r_index=0;r_active=1;
                $display("FVREAL_NAR cycle=%0d job=%0d ar=%0d address=%016h beats=%0d",cycles,jobs,n_ar,r_address,r_beats);n_ar++;n_total_ar++;
                end
            end
            if(m_axi_rvalid && m_axi_rready)begin
                if(dut.vocab_bus_owned)begin
                require(r_active && m_axi_rlast==(r_index==r_beats-1),"Read burst accounting");
                if(r_address>=ABASE+4032 && r_address+64<=ABASE+8128)
                    $display("FVREAL_R cycle=%0d job=%0d phase=%0d group=%0d index=%0d address=%016h data=%0128h",cycles,jobs,phase_now,groups-1,int'((r_address+64'(r_index*64)-ABASE-4032)>>6),r_address+64'(r_index*64),m_axi_rdata);
                r_index++;rs++;job_r++;total_r++;if(m_axi_rlast)r_active=0;m_axi_rvalid<=0;
                end else begin
                require(r_active && m_axi_rlast==(r_index==r_beats-1),"Norm R burst");
                $display("FVREAL_NR cycle=%0d job=%0d ar=%0d beat=%0d address=%016h data=%0128h response=%0d last=%0d",cycles,jobs,n_ar-1,r_index,r_address+64'(r_index*64),m_axi_rdata,m_axi_rresp,m_axi_rlast);
                r_index++;n_r++;n_total_r++;if(m_axi_rlast)r_active=0;m_axi_rvalid<=0;
                end
            end
            if(dut.projection.matrix.engine.result_valid && dut.projection.matrix.engine.result_ready)begin
                require(dut.projection.matrix.engine.result_row==5'(computed%32),"Result row order");
                require(computed<rows_for(phase_now) && dut.projection.matrix.engine.result_data===reference64[row_offset(phase_now)+computed],"Real vocabulary FP64 arithmetic");
                $display("FVREAL_MAC cycle=%0d job=%0d phase=%0d row=%0d data=%016h",cycles,jobs,phase_now,computed,dut.projection.matrix.engine.result_data);computed++;job_results++;total_results++;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                if(dut.vocab_bus_owned)begin
                require(!aw_pending && m_axi_awaddr==output_base(phase_now)+4032+64'(write_cursor*64),"AW phase address");
                require(m_axi_awlen<=1 && int'(m_axi_awaddr[11:0])+(int'(m_axi_awlen)+1)*64<=4096,"AW burst/page");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                aw_pending=1;aw_beats=int'(m_axi_awlen)+1;aw_address=m_axi_awaddr;aw_number=aws;write_cursor+=aw_beats;
                $display("FVREAL_AW cycle=%0d job=%0d phase=%0d aw=%0d address=%016h beats=%0d",cycles,jobs,phase_now,aws,m_axi_awaddr,aw_beats);aws++;job_aw++;total_aw++;
                end else begin
                require(!aw_pending && n_dones==1 && n_words==512 && m_axi_awlen==0 && m_axi_awaddr==ABASE+4032+64'(n_aw*64),"Norm AW shape/address");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"Norm AW attributes");
                aw_pending=1;aw_beats=1;aw_address=m_axi_awaddr;aw_number=n_aw;
                $display("FVREAL_NAW cycle=%0d job=%0d aw=%0d address=%016h beats=1",cycles,jobs,n_aw,m_axi_awaddr);n_aw++;
                end
            end
            if(m_axi_wvalid && m_axi_wready)begin
                if(dut.vocab_bus_owned)begin
                require(ws<rows_for(phase_now)/16 && w_pending<2 && m_axi_wdata===output_word(row_offset(phase_now)/16+ws),"Real vocabulary FP32 AXI bytes");
                require(m_axi_wstrb==64'hffffffffffffffff,"FP32 byte mask");
                if(result_file!=0)$fwrite(result_file,"%0128h\n",m_axi_wdata);
                destination[row_offset(phase_now)/16+ws]=m_axi_wdata;pending_last[w_pending]=m_axi_wlast;w_pending++;
                $display("FVREAL_W cycle=%0d job=%0d phase=%0d word=%0d data=%0128h last=%0d",cycles,jobs,phase_now,ws,m_axi_wdata,m_axi_wlast);ws++;job_w++;total_w++;
                end else begin
                require(n_w<64 && w_pending==0 && m_axi_wdata===activations[n_w] && m_axi_wlast && m_axi_wstrb==64'hffffffffffffffff,"Norm AXI bytes");
                pending_data[w_pending]=m_axi_wdata;pending_last[w_pending]=m_axi_wlast;w_pending++;
                $display("FVREAL_NW cycle=%0d job=%0d word=%0d data=%0128h last=%0d",cycles,jobs,n_w,m_axi_wdata,m_axi_wlast);n_w++;n_total_w++;
                end
            end
            if(m_axi_bvalid && m_axi_bready)begin
                if(dut.vocab_bus_owned)begin
                require(aw_pending && w_pending==aw_beats,"B before AW/W drain");
                $display("FVREAL_B cycle=%0d job=%0d phase=%0d aw=%0d response=%0d",cycles,jobs,phase_now,aw_number,m_axi_bresp);
                bs++;job_b++;total_b++;aw_pending=0;w_pending=0;m_axi_bvalid<=0;
                end else begin
                require(aw_pending && aw_beats==1 && w_pending==1 && !abi_committed[aw_number] && aw_address==ABASE+4032+64'(aw_number*64),"Norm B commit");
                require(m_axi_bresp==0 && m_axi_bid==0,"Successful normalization commit");
                abi_memory[aw_number]=pending_data[0];abi_committed[aw_number]=1;
                $display("FVREAL_NB cycle=%0d job=%0d aw=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);
                n_b++;n_total_b++;aw_pending=0;w_pending=0;m_axi_bvalid<=0;
                end
            end
            if(dut.projection.matrix.e_done_valid && dut.projection.matrix.e_done_ready)begin
                require(!r_active && !aw_pending && w_pending==0 && groups==replies+1,"Group completion before drain");
                $display("FVREAL_GROUP_DONE cycle=%0d job=%0d phase=%0d group=%0d status=%0d bytes=%0d",cycles,jobs,phase_now,replies,dut.projection.matrix.e_done_status,dut.projection.matrix.e_done_bytes);replies++;
            end
            if(dut.projection.child_cancel_valid && dut.projection.child_cancel_ready)begin
                require(mode_now==2 && phase_now==target_now,"Unexpected child cancellation");cancel_acks++;
                $display("FVREAL_CHILD_CANCEL cycle=%0d job=%0d phase=%0d",cycles,jobs,phase_now);
            end
            if(dut.projection.child_done_valid && !done_seen)begin
                done_seen=1;
                require(!r_active && !aw_pending && w_pending==0 && aws==bs && groups==replies,"Matrix completion before drain");
                if(mode_now==2 && phase_now==target_now)begin
                    require(dut.projection.child_done_status==7 && dut.projection.child_done_rows==0 && dut.projection.child_done_bytes==0 && computed==0 && ws==0,"Cancelled matrix publication");
                    require(groups==1 && tiles==1 && rs>=70,"Cancellation did not exercise active reads");
                end else begin
                    require(dut.projection.child_done_status==0 && dut.projection.child_done_rows==rows_for(phase_now) && dut.projection.child_done_bytes==rows_for(phase_now)*4,"Successful matrix shape");
                    require(computed==rows_for(phase_now) && groups==rows_for(phase_now)/32 && tiles==rows_for(phase_now)/16 && ars==groups*21 && rs==groups*1152 && ws==rows_for(phase_now)/16 && aws==rows_for(phase_now)/32+(rows_for(phase_now)/32+31)/32,"Full matrix coverage");good_matrices++;
                end
                if(dut.projection.child_done_status==0)begin
                    phase_file=$fopen($sformatf("vocab-phase-%02d.hex",phase_now),"w");require(phase_file!=0,"Open phase result");
                    for(integer i=0;i<rows_for(phase_now)/16;i++)$fwrite(phase_file,"%0128h\n",destination[row_offset(phase_now)/16+i]);
                    $fclose(phase_file);phase_file=0;
                end
                $display("FVREAL_MATRIX_DONE cycle=%0d job=%0d phase=%0d status=%0d bytes=%0d rows=%0d tag=%016h groups=%0d tiles=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d elapsed=%0d",cycles,jobs,phase_now,dut.projection.child_done_status,dut.projection.child_done_bytes,dut.projection.child_done_rows,{dut.projection.child_done_epoch,dut.projection.child_done_sequence},groups,tiles,computed,ars,rs,aws,ws,bs,cycles-phase_start);
            end
            if(completion_valid && completion_ready)begin
                require(!public_seen,"Duplicate public completion");public_seen=1;publics++;
                $display("FVREAL_PUBLIC cycle=%0d job=%0d status=%0d sequence=%016h epoch=%016h valid=%0d revoked=%0d matrices=%0d bytes=%0d",cycles,jobs,completion_status,completion_sequence,completion_epoch,result_valid,completion_revoked,completion_matrices,completion_total_bytes);
            end
            if(dut.projection.child_done_valid && dut.projection.child_done_ready)begin
                require({dut.projection.child_done_epoch,dut.projection.child_done_sequence}==tag_model+64'(phase_now*256),"Matrix completion identity");
                if(phase_now==15 || mode_now==2 && phase_now==target_now)require(public_seen,"Terminal matrix released early");
                $display("FVREAL_MATRIX_ACK cycle=%0d job=%0d phase=%0d public=%0d",cycles,jobs,phase_now,public_seen);matrix_acks++;
            end
            if(source_completion_valid && source_completion_ready)begin
                require(source_completion_valid && public_seen && matrix_acks==matrices && !r_active && !aw_pending,"Source released before publication/drain");
                source_acks++;$display("FVREAL_SOURCE_ACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(result_valid)begin
                require(completion_valid && completion_status==0 && good_matrices==16 && !completion_revoked && completion_total_bytes==513024,"Partial or revoked result published");
                require(result_address==output_base(0)+4032 && result_bytes==513024,"Public result descriptors");
                for(integer i=0;i<8016;i++)require(destination[i]===output_word(i),"Published vocabulary memory");
            end else require(result_address==0 && result_bytes==0,"Private result leak");
        end
    end
    task automatic run_job(input integer mode,input integer sample_id,input integer target);
        integer limit,status_model,sa,pu,cc;
        @(negedge clk);require(!busy,"Previous owner active");jobs++;mode_now=mode;sample_now=sample_id;target_now=target;
        matrices=0;matrix_acks=0;good_matrices=0;done_seen=0;public_seen=0;phase_now=0;
        job_ar=0;job_r=0;job_aw=0;job_w=0;job_b=0;job_results=0;
        n_ar=0;n_r=0;n_aw=0;n_w=0;n_b=0;n_starts=0;n_dones=0;n_words=0;n_reads=0;norm_core_busy=0;abi_committed=0;
        for(integer j=0;j<64;j++)abi_memory[j]='1;
        sa=source_acks;pu=publics;cc=cancel_acks;
        for(integer i=0;i<8016;i++)destination[i]='1;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);tag_model=next_tag;
        command_sequence=sequence_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_source_valid=1;command_source_sequence=sequence_model;command_source_epoch=epoch_model;
        for(integer p=0;p<2;p++)begin drive_addr[p]=norm_base(p)+4032;drive_base[p]=norm_base(p);drive_bytes[p]=12224;end
        drive_addr[2]=ABASE+4032;drive_base[2]=ABASE;drive_bytes[2]=8128;
        drive_addr[3]=weight_base(0)+4032;drive_base[3]=weight_base(0);drive_bytes[3]=4032+279085056;
        drive_addr[4]=output_base(0)+4032;drive_base[4]=output_base(0);drive_bytes[4]=4032+513024;
        if(mode==1)drive_bytes[4]=drive_bytes[4]-64;
        if(mode==0)begin result_file=$fopen("vocab-result.hex","w");require(result_file!=0,"Open result file");end
        command_valid=1;job_start=cycles;#1;require(command_ready,"Command admission");
        $display("FVREAL_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d target=%0d sequence=%016h epoch=%016h tag=%016h",cycles,jobs,mode,sample_id,target,sequence_model,epoch_model,tag_model);
        tick();@(negedge clk);command_valid=0;command_sequence='1;command_epoch='1;drive_addr='1;drive_base='1;drive_bytes='1;command_source_valid=0;command_source_sequence=0;command_source_epoch=0;
        if(mode==2)begin
            wait(matrices==target+1 && phase_now==target && rs>=70);@(negedge clk);require(cancel_ready,"Cancel admission");cancel_valid=1;
            $display("FVREAL_CANCEL cycle=%0d job=%0d phase=%0d reads=%0d",cycles,jobs,phase_now,rs);
            tick();@(negedge clk);cancel_valid=0;
        end
        limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<200000000,"Vocabulary completion deadline");end
        status_model=mode==1 ? 1:mode==2 ? 7:0;
        require(completion_status==status_model && completion_sequence==sequence_model && completion_epoch==epoch_model,"Public completion identity/status");
        require(result_valid==(mode==0) && !r_active && !aw_pending && job_aw==job_b,"Completion result/drain");
        if(mode==0)require(good_matrices==16 && job_results==128256 && job_r==4617216 && job_w==8016 && job_aw==4134,"Full vocabulary coverage");
        if(mode==1)require(matrices==0 && job_ar==0 && job_aw==0 && job_results==0,"Bounds before any traffic");
        if(mode==2)require(good_matrices==target && matrices==target+1 && cancel_acks==cc+1,"Cancel phase coverage");
        require(completion_norm_status==0 && completion_norm_hls_status==0,"Norm completion status");
        require(completion_norm_read_beats==n_r && completion_norm_records==n_b,"Norm completion counters");
        $display("FVREAL_NCOMPLETE cycle=%0d job=%0d status=%0d hls=%0d reads=%0d records=%0d starts=%0d dones=%0d words=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d",cycles,jobs,completion_norm_status,completion_norm_hls_status,completion_norm_read_beats,completion_norm_records,n_starts,n_dones,n_words,n_ar,n_r,n_aw,n_w,n_b);
        repeat(7)begin tick();require(completion_valid && completion_status==status_model && !command_ready && source_acks==sa,"Held public lifetime");end
        $display("FVREAL_COMPLETE cycle=%0d job=%0d mode=%0d matrices=%0d good=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d elapsed=%0d",cycles,jobs,mode,matrices,good_matrices,job_results,job_ar,job_r,job_aw,job_w,job_b,cycles-job_start);
        @(negedge clk);completion_ready=1;tick();
        require(!completion_valid,"Public ACK not consumed");
        @(negedge clk);completion_ready=0;
        limit=0;while(busy)begin tick();limit++;require(limit<32,"Final drain deadline");end
        require(source_acks==sa+(mode==1 ? 0:1) && publics==pu+1 && matrix_acks==matrices && command_ready,"Exact final release");
        @(negedge clk);completion_ready=0;tick();require(!completion_valid && !result_valid,"Stale public output");
        if(result_file!=0)begin $fclose(result_file);result_file=0;end
        $display("FVREAL_END cycle=%0d job=%0d",cycles,jobs);
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;expected_epoch=0;
        command_source_valid=0;command_source_sequence=0;command_source_epoch=0;
        command_sequence=0;command_epoch=0;drive_addr=0;drive_base=0;drive_bytes=0;next_tag=FIRST_TAG;
        m_axi_arready=0;m_axi_rvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;
        m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;m_axi_bresp=0;m_axi_bid=0;
        require($value$plusargs("FIXTURE=%s",fixture),"Missing full real vocabulary fixture");
        require($value$plusargs("NORM_FIXTURE=%s",norm_fixture),"Missing final normalization fixture");
        $readmemh({norm_fixture,"/source32.hex"},source_words);
        $readmemh({fixture,"/output-abi.hex"},activations);
        $readmemh({fixture,"/logits-fp64.hex"},reference64);$readmemh({fixture,"/logits-fp32.hex"},reference32);
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        $display("FVREAL_CONFIG seed=%0d rows=128256 segments=16 tail_rows=5376 k_blocks=64 tile_k=32 watchdog=65536 phase_limit=16777216",seed);
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        run_job(1,0,0);run_job(2,0,0);run_job(0,0,0);
        require(jobs==3 && publics==3 && source_acks==2 && cancel_acks==1 && total_matrices==17 && total_groups==4009 && total_results==128256 && total_w==8016 && total_aw==4134 && total_b==4134,"Suite coverage");
        require(n_total_starts==2 && n_total_dones==2 && n_total_words==1024 && n_total_ar==12 && n_total_r==512 && n_total_w==128 && n_total_b==128,"Normalization suite coverage");
        $display("FVREAL_NSUMMARY starts=%0d dones=%0d words=%0d ar=%0d r=%0d w=%0d b=%0d",n_total_starts,n_total_dones,n_total_words,n_total_ar,n_total_r,n_total_w,n_total_b);
        $display("FVREAL_PASS seed=%0d jobs=%0d publics=%0d source_acks=%0d cancels=%0d matrices=%0d groups=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d ar_stalls=%0d r_stalls=%0d aw_stalls=%0d w_stalls=%0d cycles=%0d",seed,jobs,publics,source_acks,cancel_acks,total_matrices,total_groups,total_results,total_ar,total_r,total_aw,total_w,total_b,ar_stalls,r_stalls,aw_stalls,w_stalls,cycles);
        $finish;
    end
    initial begin #1500000000;$fatal(1,"FVREAL_FAIL global timeout");end
endmodule
`default_nettype wire
