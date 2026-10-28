// External memory and passive producer taps for the real input/attention bank.
// ABI1 and Q/K/V storage starts invalid and changes only on successful B.
    logic [511:0] q_weights[0:69694],k_weights[0:17470],v_weights[0:17470];
    logic [511:0] abi_reference[0:255],abi_memory[0:63],qkv_memory[0:191],p_buffer[0:1];
    logic [31:0] source_words[0:16383],projection32[0:12287];
    logic [63:0] projection64[0:12287];
    bit abi_valid[0:63],qkv_valid[0:191],p_last[0:1];
    bit p_read_active=0,p_aw_pending=0,p_norm_running=0,p_done_seen=0;
    logic [63:0] p_read_base,p_aw_address;
    integer p_read_count=0,p_read_index=0,p_buffered=0,p_aw_beats=0;
    integer p_phase=0,p_matrices=0,p_matrix_acks=0,p_mac=0,p_words=0,p_abi_commits=0,p_qkv_commits=0;
    integer p_reads=0,p_norm_reads=0,p_writes=0,p_bursts=0,p_starts=0,p_dones=0,p_acks=0,c_starts=0,c_acks=0;
    integer p_norm_starts=0,p_norm_dones=0,p_mac_total=0,p_read_total=0,p_write_total=0;
    bit p_qkv_owned;
    assign p_qkv_owned=dut.producer.qkv_bus_owned;
    function automatic logic [63:0] allocation_base(input integer i);
        return 64'(i+1)<<40;
    endfunction
    function automatic logic [63:0] allocation_address(input integer i);
        return allocation_base(i)+4032;
    endfunction
    function automatic logic [63:0] allocation_bytes(input integer i);
        case(i)
            0,1,6:return 12224;
            2:return 8128;
            3:return 4460480;
            4,5:return 1118144;
            9:return 4288;
            default:return 6080;
        endcase
    endfunction
    function automatic integer projection_offset(input integer p);
        return p==0 ? 0:p==1 ? 2048:2560;
    endfunction
    function automatic integer projection_rows(input integer p);
        return p==0 ? 2048:512;
    endfunction
    function automatic logic [511:0] producer_read(input logic [63:0] a);
        logic [511:0] value;integer index;
        for(integer plane=0;plane<2;plane++)if(a>=allocation_address(plane) && a+64<=allocation_address(plane)+8192)begin
            if(p_qkv_owned)$fatal(1,"IAR_FAIL projection reading normalization source");
            index=int'((a-allocation_address(plane))/4);
            for(integer lane=0;lane<16;lane++)value[lane*32+:32]=source_words[sample*4096+plane*2048+index+lane];
            return value;
        end
        if(a>=allocation_address(2) && a+64<=allocation_address(2)+4096)begin
            index=int'((a-allocation_address(2))/64);
            if(!p_qkv_owned || p_abi_commits!=64 || !abi_valid[index])$fatal(1,"IAR_FAIL ABI read before successful B");
            return abi_memory[index];
        end
        if(!p_qkv_owned || a<allocation_address(p_phase+3) || a+64>allocation_base(p_phase+3)+allocation_bytes(p_phase+3))
            $fatal(1,"IAR_FAIL source read outside current weight allocation");
        index=int'((a-allocation_base(p_phase+3))/64);
        case(p_phase)
            0:return q_weights[index];
            1:return k_weights[index];
            default:return v_weights[index];
        endcase
    endfunction
    function automatic logic [31:0] committed_projection_word(input integer index);
        if(index<0 || index>=3072 || !qkv_valid[index/16] || p_qkv_commits!=192 || !dut.producer_completion_valid)
            $fatal(1,"IAR_FAIL consumer read without held, fully committed QKV");
        return qkv_memory[index/16][(index%16)*32+:32];
    endfunction
    function automatic logic [511:0] projected_reference(input integer index);
        logic [511:0] value;
        for(integer lane=0;lane<16;lane++)value[lane*32+:32]=projection32[sample*3072+index*16+lane];
        return value;
    endfunction
    always @(negedge clk)begin
        input_axi_arready=rst_n && !p_read_active && (cycle+seed)%5!=0;
        input_axi_awready=rst_n && !p_aw_pending && (cycle+seed)%4!=0;
        input_axi_wready=rst_n && p_buffered<2 && (cycle+seed)%3!=0;
        if(!rst_n)begin input_axi_rvalid=0;input_axi_bvalid=0;end
        else begin
            if(!input_axi_rvalid && p_read_active && (cycle+seed)%4!=0)begin
                input_axi_rdata=producer_read(p_read_base+64'(p_read_index*64));
                input_axi_rresp=0;input_axi_rid=0;input_axi_rlast=p_read_index==p_read_count-1;input_axi_rvalid=1;
            end
            if(!input_axi_bvalid && p_aw_pending && p_buffered==p_aw_beats && (cycle+seed)%3!=0)begin
                for(integer i=0;i<p_buffered;i++)require(p_last[i]==(i==p_buffered-1),"Producer WLAST framing before B");
                input_axi_bresp=0;input_axi_bid=0;input_axi_bvalid=1;
            end
        end
    end
    // Called from the single posedge observer, so transcript cycle order is deterministic.
    task automatic observe_producer;
        integer index,base_index;
        if(dut.producer_command_valid && dut.producer_command_ready)begin
            require(mode!=1 && p_matrices==0 && p_abi_commits==0 && p_qkv_commits==0,"Producer admission after full preflight");p_starts++;
            $display("IAR_PSTART cycle=%0d job=%0d sequence=%016h epoch=%016h",cycle,job,dut.producer_command_sequence,dut.producer_command_epoch);
        end
        if(dut.producer.normalization.hls_start)begin
            require(!p_norm_running && p_norm_reads==256 && !p_read_active && p_writes==0,"Norm HLS after actual input reads");
            p_norm_running=1;p_norm_starts++;
            $display("IAR_PNSTART cycle=%0d job=%0d",cycle,job);
        end
        if(dut.producer.normalization.hls_output_ce && dut.producer.normalization.hls_output_we)begin
            require(p_norm_running && dut.producer.normalization.hls_output_address==p_words && p_words<512,"Norm sequential output");
            require(dut.producer.normalization.hls_output_d===abi_reference[sample*64+p_words/8][(p_words%8)*64+:64],"Actual norm output versus reference");
            $display("IAR_PNWRITE cycle=%0d job=%0d index=%0d data=%016h",cycle,job,p_words,dut.producer.normalization.hls_output_d);p_words++;
        end
        if(dut.producer.normalization.hls_done)begin
            require(p_norm_running && dut.producer.normalization.hls_ready && dut.producer.normalization.hls_return==0 && p_words==512,"Norm arithmetic completion");
            p_norm_running=0;p_norm_dones++;
            $display("IAR_PNDONE cycle=%0d job=%0d status=%0d words=%0d",cycle,job,dut.producer.normalization.hls_return,p_words);
        end
        if(dut.producer.projection.child_job_valid && dut.producer.projection.child_job_ready)begin
            require(p_matrices==p_matrix_acks && p_matrices<3 && p_abi_commits==64 && !p_read_active && !p_aw_pending && p_buffered==0,"Matrix admission and ABI lease");
            p_phase=p_matrices;p_matrices++;p_mac=0;
            require(dut.producer.projection.child_job_rows==projection_rows(p_phase) && dut.producer.projection.child_job_k_blocks==64 && dut.producer.projection.child_job_tile_k==32 && dut.producer.projection.child_job_layout==2 && dut.producer.projection.child_job_output_fp32,"Full matrix shape");
            require(dut.producer.projection.child_job_weight_addr==allocation_address(p_phase+3) && dut.producer.projection.child_job_activation_addr==allocation_address(2) && dut.producer.projection.child_job_output_addr==allocation_address(p_phase+6),"Retained matrix allocations");
            $display("IAR_PMATRIX cycle=%0d job=%0d phase=%0d tag=%016h rows=%0d",cycle,job,p_phase,{dut.producer.projection.child_job_epoch,dut.producer.projection.child_job_sequence},dut.producer.projection.child_job_rows);
        end
        if(input_axi_arvalid && input_axi_arready)begin
            require(!p_read_active && input_axi_araddr[5:0]==0 && int'(input_axi_araddr[11:0])+(int'(input_axi_arlen)+1)*64<=4096,"Producer read burst/page ownership");
            require(input_axi_arsize==6 && input_axi_arburst==1 && input_axi_arid==0 && input_axi_arcache==2 && input_axi_arprot==0 && !input_axi_arlock,"Producer AR attributes");
            p_read_active=1;p_read_base=input_axi_araddr;p_read_count=int'(input_axi_arlen)+1;p_read_index=0;
            $display("IAR_PAR cycle=%0d job=%0d kind=%0d phase=%0d address=%016h beats=%0d",cycle,job,p_qkv_owned,p_phase,p_read_base,p_read_count);
        end
        if(input_axi_rvalid && input_axi_rready)begin
            require(p_read_active && input_axi_rlast==(p_read_index==p_read_count-1),"Producer R framing");
            $display("IAR_PR cycle=%0d job=%0d kind=%0d phase=%0d address=%016h beat=%0d last=%0d data=%0128h",cycle,job,p_qkv_owned,p_phase,p_read_base+64'(p_read_index*64),p_read_index,input_axi_rlast,input_axi_rdata);
            p_read_index++;p_reads++;p_read_total++;if(!p_qkv_owned)p_norm_reads++;
            if(input_axi_rlast)p_read_active=0;input_axi_rvalid<=0;
        end
        if(dut.producer.projection.matrix.engine.result_valid && dut.producer.projection.matrix.engine.result_ready)begin
            require(p_mac<projection_rows(p_phase) && dut.producer.projection.matrix.engine.result_row==5'(p_mac%32),"Actual matrix MAC position");
            require(dut.producer.projection.matrix.engine.result_data===projection64[sample*3072+projection_offset(p_phase)+p_mac],"Every actual QKV FP64 arithmetic result");
            $display("IAR_PMAC cycle=%0d job=%0d phase=%0d index=%0d data=%016h",cycle,job,p_phase,p_mac,dut.producer.projection.matrix.engine.result_data);p_mac++;p_mac_total++;
        end
        if(input_axi_awvalid && input_axi_awready)begin
            require(!p_aw_pending && input_axi_awaddr[5:0]==0 && input_axi_awlen<=1 && int'(input_axi_awaddr[11:0])+(int'(input_axi_awlen)+1)*64<=4096,"Producer AW burst/page ownership");
            require(input_axi_awsize==6 && input_axi_awburst==1 && input_axi_awid==0 && input_axi_awcache==2 && input_axi_awprot==0 && !input_axi_awlock,"Producer AW attributes");
            p_aw_pending=1;p_aw_address=input_axi_awaddr;p_aw_beats=int'(input_axi_awlen)+1;
            require(p_aw_address==(p_qkv_owned ? allocation_address(p_phase+6)+64'((p_qkv_commits-projection_offset(p_phase)/16)*64):allocation_address(2)+64'(p_abi_commits*64)),"Exact producer write location");
            $display("IAR_PAW cycle=%0d job=%0d kind=%0d phase=%0d address=%016h beats=%0d",cycle,job,p_qkv_owned,p_phase,p_aw_address,p_aw_beats);
        end
        if(input_axi_wvalid && input_axi_wready)begin
            require(p_buffered<2 && input_axi_wstrb==64'hffffffffffffffff,"Producer write mask and buffering");
            require(p_writes<(p_qkv_owned ? 256:64),"Producer write extent");
            require(input_axi_wdata===(p_qkv_owned ? projected_reference(p_writes-64):abi_reference[sample*64+p_writes]),"Producer ABI/QKV arithmetic bytes");
            p_buffer[p_buffered]=input_axi_wdata;p_last[p_buffered]=input_axi_wlast;p_buffered++;
            $display("IAR_PW cycle=%0d job=%0d kind=%0d phase=%0d index=%0d last=%0d data=%0128h",cycle,job,p_qkv_owned,p_phase,p_writes,input_axi_wlast,input_axi_wdata);p_writes++;p_write_total++;
        end
        if(input_axi_bvalid && input_axi_bready)begin
            require(p_aw_pending && p_buffered==p_aw_beats && input_axi_bresp==0,"Producer successful B owns AW/W");
            for(integer i=0;i<p_buffered;i++)begin
                if(p_qkv_owned)begin
                    index=projection_offset(p_phase)/16+int'((p_aw_address-allocation_address(p_phase+6))/64)+i;
                    require(index>=0 && index<192 && !qkv_valid[index],"Unique QKV memory commit");
                    qkv_memory[index]=p_buffer[i];qkv_valid[index]=1;p_qkv_commits++;
                end else begin
                    index=int'((p_aw_address-allocation_address(2))/64)+i;
                    require(index>=0 && index<64 && !abi_valid[index],"Unique ABI1 memory commit");
                    abi_memory[index]=p_buffer[i];abi_valid[index]=1;p_abi_commits++;
                end
            end
            $display("IAR_PB cycle=%0d job=%0d kind=%0d phase=%0d address=%016h beats=%0d response=%0d",cycle,job,p_qkv_owned,p_phase,p_aw_address,p_buffered,input_axi_bresp);
            p_aw_pending=0;p_buffered=0;p_bursts++;input_axi_bvalid<=0;
        end
        if(dut.producer.projection.child_done_valid && dut.producer.projection.child_done_ready)begin
            require(p_mac==projection_rows(p_phase) && !p_read_active && !p_aw_pending && p_buffered==0,"Matrix drains before ACK");
            require(dut.producer.projection.child_done_status==0 && dut.producer.projection.child_done_rows==projection_rows(p_phase) && dut.producer.projection.child_done_bytes==projection_rows(p_phase)*4,"Actual matrix result certificate");
            if(p_phase==2)require(completions==job && c_acks==c_starts,"Final matrix retained through downstream/public lifetime");
            $display("IAR_PMACK cycle=%0d job=%0d phase=%0d tag=%016h status=%0d rows=%0d bytes=%0d",cycle,job,p_phase,{dut.producer.projection.child_done_epoch,dut.producer.projection.child_done_sequence},dut.producer.projection.child_done_status,dut.producer.projection.child_done_rows,dut.producer.projection.child_done_bytes);p_matrix_acks++;
        end
        if(dut.producer_completion_valid && !p_done_seen)begin
            require(p_matrices==3 && p_matrix_acks==2 && p_abi_commits==64 && p_qkv_commits==192 && p_reads==110848 && p_writes==256 && p_bursts==164,"Complete real producer extent");
            require(!p_read_active && !p_aw_pending && p_buffered==0,"Producer complete after memory drain");p_done_seen=1;p_dones++;
            $display("IAR_PDONE cycle=%0d job=%0d status=%0d norm=%0d qkv=%0d matrix=%0d hls=%0d reads=%0d records=%0d phase=%0d matrices=%0d bytes=%0d",cycle,job,dut.producer_completion_status,dut.producer_completion_norm_status,dut.producer_completion_qkv_status,dut.producer_completion_matrix_status,dut.producer_completion_norm_hls_status,dut.producer_completion_norm_read_beats,dut.producer_completion_norm_records,dut.producer_completion_phase,dut.producer_completion_matrices,dut.producer_completion_total_bytes);
        end
        if(dut.consumer_command_valid && dut.consumer_command_ready)begin
            require(p_done_seen && p_qkv_commits==192 && dut.producer_completion_valid,"Consumer begins from held actual producer");c_starts++;
            $display("IAR_CSTART cycle=%0d job=%0d sequence=%016h epoch=%016h layer=%0d position=%0d",cycle,job,dut.consumer_command_sequence,dut.consumer_command_epoch,dut.consumer_command_layer,dut.consumer_command_position);
        end
        if(dut.consumer_completion_valid && dut.consumer_completion_ready)begin
            require(completions==job,"Consumer retained until public ACK");c_acks++;
            $display("IAR_CACK cycle=%0d job=%0d",cycle,job);
        end
        if(dut.producer_completion_valid && dut.producer_completion_ready)begin
            require(completions==job && c_acks==c_starts && !dut.consumer_busy,"Source retained until consumer drain");p_acks++;
            $display("IAR_PACK cycle=%0d job=%0d",cycle,job);
        end
    endtask
