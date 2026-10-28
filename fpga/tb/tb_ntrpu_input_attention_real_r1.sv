`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_input_attention_real_r1;
    logic  clk=0;
    logic  rst_n=0;
    logic  command_valid=0;
    wire  command_ready;
    logic [63:0] command_sequence=0;
    logic [63:0] command_epoch=0;
    logic [63:0] command_lease_base=0;
    logic [63:0] expected_epoch=0;
    logic [31:0] command_layer=0;
    logic [31:0] command_position=0;
    logic [31:0] command_coefficient_position=0;
    wire [63:0] command_addr [0:9];
    logic [0:9][63:0] drive_command_addr=0;
    wire [63:0] command_base [0:9];
    logic [0:9][63:0] drive_command_base=0;
    wire [63:0] command_bytes [0:9];
    logic [0:9][63:0] drive_command_bytes=0;
    logic  cancel_valid=0;
    wire  cancel_ready;
    wire  busy;
    wire  cache_exclusive;
    wire  completion_valid;
    logic  completion_ready=0;
    wire [63:0] completion_sequence;
    wire [63:0] completion_epoch;
    wire [31:0] completion_layer;
    wire [31:0] completion_position;
    wire [3:0] completion_status;
    wire [3:0] completion_source_status;
    wire [3:0] completion_child_status;
    wire  completion_kv_committed;
    wire [12:0] completion_kv_bytes;
    wire [31:0] completion_kv_tokens;
    wire  completion_revoked;
    wire  result_valid;
    logic  result_ce=0;
    logic [10:0] result_address=0;
    wire [31:0] result_q;
    wire  result_q_valid;
    logic [63:0] command_source_sequence=0;
    logic [63:0] command_source_epoch=0;
    logic  command_source_valid=0;
    wire [3:0] completion_norm_status;
    wire [3:0] completion_qkv_status;
    wire [3:0] completion_matrix_status;
    wire [31:0] completion_norm_hls_status;
    wire [9:0] completion_norm_read_beats;
    wire [6:0] completion_norm_records;
    wire [1:0] completion_phase;
    wire [1:0] completion_matrices;
    wire [14:0] completion_total_bytes;
    wire  norm_tag_exhausted;
    wire  qkv_tags_exhausted;
    wire [63:0] input_axi_araddr;
    wire [7:0] input_axi_arlen;
    wire [2:0] input_axi_arsize;
    wire [2:0] input_axi_arprot;
    wire [1:0] input_axi_arburst;
    wire [3:0] input_axi_arid;
    wire [3:0] input_axi_arcache;
    wire  input_axi_arlock;
    wire  input_axi_arvalid;
    logic  input_axi_arready=0;
    logic [511:0] input_axi_rdata=0;
    logic [1:0] input_axi_rresp=0;
    logic [3:0] input_axi_rid=0;
    logic  input_axi_rlast=0;
    logic  input_axi_rvalid=0;
    wire  input_axi_rready;
    wire [63:0] input_axi_awaddr;
    wire [7:0] input_axi_awlen;
    wire [2:0] input_axi_awsize;
    wire [2:0] input_axi_awprot;
    wire [1:0] input_axi_awburst;
    wire [3:0] input_axi_awid;
    wire [3:0] input_axi_awcache;
    wire  input_axi_awlock;
    wire  input_axi_awvalid;
    logic  input_axi_awready=0;
    wire [511:0] input_axi_wdata;
    wire [63:0] input_axi_wstrb;
    wire  input_axi_wlast;
    wire  input_axi_wvalid;
    logic  input_axi_wready=0;
    logic [3:0] input_axi_bid=0;
    logic [1:0] input_axi_bresp=0;
    logic  input_axi_bvalid=0;
    wire  input_axi_bready;
    wire [63:0] v_axi_araddr;
    wire [7:0] v_axi_arlen;
    wire [2:0] v_axi_arsize;
    wire [2:0] v_axi_arprot;
    wire [1:0] v_axi_arburst;
    wire [3:0] v_axi_arid;
    wire [3:0] v_axi_arcache;
    wire  v_axi_arlock;
    wire  v_axi_arvalid;
    wire  v_axi_arready;
    wire [511:0] v_axi_rdata;
    logic [1:0] v_axi_rresp=0;
    logic [3:0] v_axi_rid=0;
    wire  v_axi_rlast;
    wire  v_axi_rvalid;
    wire  v_axi_rready;
    wire [63:0] rope_axi_araddr;
    wire [7:0] rope_axi_arlen;
    wire [2:0] rope_axi_arsize;
    wire [2:0] rope_axi_arprot;
    wire [1:0] rope_axi_arburst;
    wire [3:0] rope_axi_arid;
    wire [3:0] rope_axi_arcache;
    wire  rope_axi_arlock;
    wire  rope_axi_arvalid;
    wire  rope_axi_arready;
    wire [511:0] rope_axi_rdata;
    logic [1:0] rope_axi_rresp=0;
    logic [3:0] rope_axi_rid=0;
    wire  rope_axi_rlast;
    wire  rope_axi_rvalid;
    wire  rope_axi_rready;
    logic  bind_valid=0;
    wire  bind_ready;
    logic [63:0] bind_cache_base=0;
    logic [63:0] bind_region_base=0;
    logic [63:0] bind_region_bytes=0;
    logic [31:0] bind_capacity=0;
    logic [63:0] bind_sequence=0;
    logic [63:0] bind_epoch=0;
    wire  cache_valid;
    wire [63:0] cache_base;
    wire [63:0] cache_region_base;
    wire [63:0] cache_region_bytes;
    wire [63:0] cache_sequence;
    wire [63:0] cache_epoch;
    wire [31:0] cache_capacity;
    wire [63:0] kv_axi_awaddr;
    wire [7:0] kv_axi_awlen;
    wire [2:0] kv_axi_awsize;
    wire [2:0] kv_axi_awprot;
    wire [1:0] kv_axi_awburst;
    wire [3:0] kv_axi_awid;
    wire [3:0] kv_axi_awcache;
    wire  kv_axi_awlock;
    wire  kv_axi_awvalid;
    logic  kv_axi_awready=0;
    wire [511:0] kv_axi_wdata;
    wire [63:0] kv_axi_wstrb;
    wire  kv_axi_wlast;
    wire  kv_axi_wvalid;
    logic  kv_axi_wready=0;
    logic [3:0] kv_axi_bid=0;
    logic [1:0] kv_axi_bresp=0;
    logic  kv_axi_bvalid=0;
    wire  kv_axi_bready;
    wire [63:0] kv_axi_araddr;
    wire [7:0] kv_axi_arlen;
    wire [2:0] kv_axi_arsize;
    wire [2:0] kv_axi_arprot;
    wire [1:0] kv_axi_arburst;
    wire [3:0] kv_axi_arid;
    wire [3:0] kv_axi_arcache;
    wire  kv_axi_arlock;
    wire  kv_axi_arvalid;
    wire  kv_axi_arready;
    wire [511:0] kv_axi_rdata;
    logic [1:0] kv_axi_rresp=0;
    logic [3:0] kv_axi_rid=0;
    wire  kv_axi_rlast;
    wire  kv_axi_rvalid;
    wire  kv_axi_rready;
    wire  root_cache_lock;
    wire  cache_metadata_fault;
    wire [31:0] cache_query_layer;
    wire  cache_query_valid;
    wire [31:0] cache_query_committed_tokens;
    wire [31:0] cache_committed_tokens;
    wire  bind_completion_valid;
    logic  bind_completion_ready=0;
    wire [3:0] bind_completion_status;
    wire [63:0] bind_completion_sequence;
    wire [63:0] bind_completion_epoch;
    for(genvar g=0;g<10;g++)begin: descriptors
        assign command_addr[g]=drive_command_addr[g];
        assign command_base[g]=drive_command_base[g];
        assign command_bytes[g]=drive_command_bytes[g];
    end
    always #2.5 clk=~clk;
    ntrpu_input_attention_r1 dut(.*);
    wire [63:0] ra[0:2];wire [7:0] rn[0:2];wire av[0:2],rr[0:2];
    logic ar[0:2],rv[0:2],rl[0:2];logic [511:0] rd[0:2];
    assign ra[0]=rope_axi_araddr;assign rn[0]=rope_axi_arlen;assign av[0]=rope_axi_arvalid;assign rr[0]=rope_axi_rready;
    assign rope_axi_arready=ar[0];assign rope_axi_rdata=rd[0];assign rope_axi_rvalid=rv[0];assign rope_axi_rlast=rl[0];
    assign ra[1]=v_axi_araddr;assign rn[1]=v_axi_arlen;assign av[1]=v_axi_arvalid;assign rr[1]=v_axi_rready;
    assign v_axi_arready=ar[1];assign v_axi_rdata=rd[1];assign v_axi_rvalid=rv[1];assign v_axi_rlast=rl[1];
    assign ra[2]=kv_axi_araddr;assign rn[2]=kv_axi_arlen;assign av[2]=kv_axi_arvalid;assign rr[2]=kv_axi_rready;
    assign kv_axi_arready=ar[2];assign kv_axi_rdata=rd[2];assign kv_axi_rvalid=rv[2];assign kv_axi_rlast=rl[2];

`define RP dut.consumer.front.rope
`define AT dut.consumer.attention.engine
    logic [31:0] inputs[0:10495],values[0:2047],rotated[0:10239],reference[0:8191];
    logic [511:0] memory[0:4095],write_buffer[0:3];bit memory_valid[0:4095];
    bit write_last[0:3],read_active[0:2];logic [63:0] read_base[0:2];integer read_count[0:2],read_index[0:2];
    bit aw_pending=0;logic [63:0] aw_address;integer aw_beats=0,buffered=0;
    integer cycle=0,seed=1,job=0,sample=0,mode=0,group=0,prefix=0;
    integer binds=0,bindacks=0,admissions=0,completions=0,public_words=0,errors=0;
    integer rbeats[0:2],total_rbeats[0:2],wbeats=0,total_wbeats=0,bursts=0;
    integer rp_writes=0,rp_starts=0,rp_dones=0,heads=0,total_heads=0,head_acks=0;
    integer at_starts=0,at_dones=0,state_writes=0,output_writes=0,qreads=0,kreads=0,vreads=0;
    integer head=0,rp_reads=0,rp_quiet=0;
    bit rp_running=0,at_running=0,head_owned=0;
    logic [103:0] seen_state;logic [63:0] seen_output;
    logic [63:0] seq_model,epoch_model,lease_model;
    localparam logic [63:0] BASE=((64'd14<<40)+64'd4032);
    string fixture;
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

    task automatic require(input bit good,input string reason);
        if(!good)$fatal(1,"IAR_FAIL cycle=%0d job=%0d mode=%0d %s",cycle,job,mode,reason);
    endtask
    function automatic bit close32(input logic [31:0] a,b);
        logic [31:0] x,y;
        x=a[31] ? ~a:a|32'h80000000;y=b[31] ? ~b:b|32'h80000000;
        return a[30:23]!=255 && b[30:23]!=255 && ((a[30:0]==0 && b[30:0]==0) || (x>y ? x-y:y-x)<=1);
    endfunction
    function automatic logic [31:0] input_word(input logic [63:0] address);
        if(address>=allocation_address(6) && address<(allocation_address(6)+8192))return committed_projection_word(int'((address-allocation_address(6))/4));
        if(address>=allocation_address(7) && address<(allocation_address(7)+2048))return committed_projection_word(2048+int'((address-allocation_address(7))/4));
        if(address>=allocation_address(9) && address<(allocation_address(9)+256))return inputs[sample*2624+2560+int'((address-allocation_address(9))/4)];
        return 32'hxxxxxxxx;
    endfunction
    function automatic logic [511:0] response_data(input integer port,input logic [63:0] address);
        logic [511:0] data;
        if(port==2)return memory[int'((address-BASE)/64)];
        for(integer lane=0;lane<16;lane++)begin
            if(port==0)data[lane*32+:32]=input_word(address+64'(lane*4));
            else data[lane*32+:32]=committed_projection_word(2560+int'((address-allocation_address(8))/4)+lane);
        end
        return data;
    endfunction
    function automatic logic [31:0] kv_word(input integer index);
        if(index<512)return rotated[sample*2560+2048+index];
        return values[sample*512+index-512];
    endfunction
    // Only external memory stimuli are driven here. All internal taps are passive.
    always @(negedge clk)begin
        for(integer p=0;p<3;p++)begin
            ar[p]=rst_n && !read_active[p] && (cycle+seed+p)%5!=0;
            if(!rst_n)rv[p]=0;
            else if(!rv[p] && read_active[p] && (cycle+seed+p)%4!=0)begin
                rd[p]=response_data(p,read_base[p]+64'(read_index[p]*64));
                rl[p]=read_index[p]==read_count[p]-1;rv[p]=1;
            end
        end
        kv_axi_awready=rst_n && !aw_pending && (cycle+seed)%4!=0;
        kv_axi_wready=rst_n && buffered<4 && (cycle+seed)%3!=0;
        if(!rst_n)kv_axi_bvalid=0;
        else if(!kv_axi_bvalid && aw_pending && buffered==aw_beats && (cycle+seed)%3!=0)begin
            for(integer i=0;i<buffered;i++)require(write_last[i]==(i==buffered-1),"Complete WLAST framing before B");
            kv_axi_bresp=(mode==2 && wbeats==64) ? 2:0;kv_axi_bvalid=1;
        end
    end
    always @(posedge clk)begin : observe
        bit qc,kc,vc,rc0,rc1;integer qa,ka,va,ra0,ra1;
        logic [63:0] expected_address;integer remaining,beats,offset;
        cycle++;
        if(!rst_n)begin
            for(integer p=0;p<3;p++)begin read_active[p]=0;read_index[p]=0;read_count[p]=0;end
            aw_pending=0;buffered=0;
        end else begin
            require(cycle<120000000,"Global simulation deadline");
            observe_producer();
            if(bind_valid && bind_ready)begin
                binds++;bind_valid<=0;
                $display("IAR_BIND cycle=%0d group=%0d sequence=%016h epoch=%016h capacity=%0d",cycle,group,bind_sequence,bind_epoch,bind_capacity);
            end
            if(bind_completion_valid && bind_completion_ready)begin
                bindacks++;
                $display("IAR_BIND_ACK cycle=%0d group=%0d status=%0d sequence=%016h epoch=%016h",cycle,group,bind_completion_status,bind_completion_sequence,bind_completion_epoch);
            end
            if(command_valid && command_ready)begin
                admissions++;command_valid<=0;
                $display("IAR_ADMIT cycle=%0d job=%0d sequence=%016h epoch=%016h layer=%0d position=%0d",cycle,job,command_sequence,command_epoch,command_layer,command_position);
            end
            for(integer p=0;p<3;p++)begin
                if(av[p] && ar[p])begin
                    require(!read_active[p],"One outstanding read burst per master");
                    if(p==0)begin
                        if(rbeats[p]<128)begin expected_address=allocation_address(6)+64'(rbeats[p]*64);remaining=128-rbeats[p];end
                        else if(rbeats[p]<160)begin expected_address=allocation_address(7)+64'((rbeats[p]-128)*64);remaining=160-rbeats[p];end
                        else begin expected_address=allocation_address(9)+64'((rbeats[p]-160)*64);remaining=164-rbeats[p];end
                        require(rope_axi_arsize==6 && rope_axi_arburst==1 && rope_axi_arid==0,"RoPE AXI attributes");
                    end else if(p==1)begin
                        expected_address=allocation_address(8)+64'(rbeats[p]*64);remaining=32-rbeats[p];
                        require(v_axi_arsize==6 && v_axi_arburst==1 && v_axi_arid==0,"V AXI attributes");
                    end else begin
                        expected_address=ra[p];remaining=64;
                        require(kv_axi_arsize==6 && kv_axi_arburst==1 && kv_axi_arid==0,"KV AXI attributes");
                        require(head_owned && ra[p]>=BASE && ra[p]+64'(int'(rn[p])+1)*64<=BASE+262144,"KV read owns a head and allocation");
                        for(integer i=0;i<=int'(rn[p]);i++)require(memory_valid[int'((ra[p]-BASE)/64)+i],"Read only physically written KV");
                    end
                    beats=remaining<(64-int'(ra[p][11:6])) ? remaining:(64-int'(ra[p][11:6]));
                    require(ra[p][5:0]==0 && ra[p]==expected_address && int'(rn[p])+1>0 && int'(rn[p])+1<=beats,"Read allocation and page bounds");
                    if(p<2)require(int'(rn[p])+1==beats,"Source read extent");
                    read_active[p]=1;read_base[p]=ra[p];read_count[p]=int'(rn[p])+1;read_index[p]=0;
                    $display("IAR_AR cycle=%0d job=%0d port=%0d address=%016h beats=%0d",cycle,job,p,ra[p],read_count[p]);
                end
                if(rv[p] && rr[p])begin
                    require(read_active[p] && rl[p]==(read_index[p]==read_count[p]-1),"Owned read response");
                    $display("IAR_R cycle=%0d job=%0d port=%0d address=%016h last=%0d data=%0128h",cycle,job,p,read_base[p]+64'(read_index[p]*64),rl[p],rd[p]);
                    rbeats[p]++;total_rbeats[p]++;read_index[p]++;rv[p]<=0;
                    if(rl[p])read_active[p]=0;
                end
            end
            if(kv_axi_awvalid && kv_axi_awready)begin
                require(!aw_pending && kv_axi_awsize==6 && kv_axi_awburst==1 && kv_axi_awid==0,"Owned write address");
                require(kv_axi_awaddr>=BASE && kv_axi_awaddr+64'(int'(kv_axi_awlen)+1)*64<=BASE+262144 && kv_axi_awaddr[5:0]==0,"Write allocation");
                aw_pending=1;aw_address=kv_axi_awaddr;aw_beats=int'(kv_axi_awlen)+1;
                require(aw_beats<=4 && aw_beats<=(64-int'(aw_address[11:6])),"Write head/page bounds");
                $display("IAR_AW cycle=%0d job=%0d address=%016h beats=%0d",cycle,job,aw_address,aw_beats);
            end
            if(kv_axi_wvalid && kv_axi_wready)begin
                require(buffered<4 && wbeats<64 && kv_axi_wstrb==64'hffffffffffffffff,"Write frame bounds and byte enables");
                for(integer lane=0;lane<16;lane++)require(kv_axi_wdata[lane*32+:32]===kv_word(wbeats*16+lane),"KV writes contain actual rotated K and source V");
                write_buffer[buffered]=kv_axi_wdata;write_last[buffered]=kv_axi_wlast;buffered++;
                $display("IAR_W cycle=%0d job=%0d index=%0d last=%0d data=%0128h",cycle,job,wbeats,kv_axi_wlast,kv_axi_wdata);
                wbeats++;total_wbeats++;
            end
            if(kv_axi_bvalid && kv_axi_bready)begin
                require(aw_pending && buffered==aw_beats,"B after matching AW and all W");
                if(kv_axi_bresp==0)for(integer i=0;i<buffered;i++)begin
                    offset=int'((aw_address-BASE)/64)+i;memory[offset]=write_buffer[i];memory_valid[offset]=1;
                end
                $display("IAR_B cycle=%0d job=%0d beats=%0d response=%0d",cycle,job,buffered,kv_axi_bresp);
                aw_pending=0;buffered=0;bursts++;kv_axi_bvalid<=0;
            end
            if(`RP.hls_start && !rp_running)begin
                require(rbeats[0]==164,"Real RoPE after all source reads");rp_running=1;rp_starts++;rp_quiet=0;
                $display("IAR_ROPE_START cycle=%0d job=%0d position=%0d",cycle,job,`RP.hls_position);
            end
            rc0=rp_running && `RP.hls_input_ce[0];rc1=rp_running && `RP.hls_input_ce[1];
            ra0=int'(`RP.hls_input_address[0]);ra1=int'(`RP.hls_input_address[1]);
            if(`RP.hls_output_ce && `RP.hls_output_we)begin
                require(rp_running && int'(`RP.hls_output_address)==rp_writes && rp_writes<2560,"Real RoPE sequential publication");
                require(`RP.hls_output_d===rotated[sample*2560+rp_writes],"Every original RoPE arithmetic word");
                $display("IAR_ROPE_WRITE cycle=%0d job=%0d index=%0d data=%08h",cycle,job,rp_writes,`RP.hls_output_d);rp_writes++;
            end
            if(`RP.hls_done && rp_running)begin
                require(`RP.hls_ready && `RP.hls_return==0 && rp_writes==2560,"Real RoPE completion");
                $display("IAR_ROPE_DONE cycle=%0d job=%0d status=%0d writes=%0d",cycle,job,`RP.hls_return,rp_writes);
                rp_running=0;rp_dones++;
            end
            if(dut.consumer.attention.head_command_valid && dut.consumer.attention.head_command_ready)begin
                require(!head_owned && heads<32,"One owned attention head");head_owned=1;head=heads;heads++;total_heads++;
                require(dut.consumer.attention.head_command_head==32'(head) && dut.consumer.attention.head_command_position==32'(sample) && dut.consumer.attention.head_command_layer==0,"Head identity");
                require(dut.consumer.attention.head_command_sequence==seq_model && dut.consumer.attention.head_command_epoch==epoch_model && dut.consumer.attention.head_command_lease==lease_model+64'(head),"Head lease identity");
                for(integer lane=0;lane<64;lane++)require(dut.consumer.attention.head_command_query[lane*32+:32]===rotated[sample*2560+head*64+lane],"Head Q equals real RoPE output");
                $display("IAR_HEAD cycle=%0d job=%0d head=%0d sequence=%016h epoch=%016h lease=%016h query=%0512h",cycle,job,head,seq_model,epoch_model,dut.consumer.attention.head_command_lease,dut.consumer.attention.head_command_query);
            end
            if(`AT.hls_start && !at_running)begin
                require(head_owned && `AT.hls_tile_start==0 && `AT.hls_tile_count==32'(sample+1) && `AT.hls_query_head==32'(head) && `AT.hls_kv_head==32'(head/4),"Real attention tile owns published prefix");
                at_running=1;at_starts++;seen_state=0;seen_output=0;
                $display("IAR_AT_START cycle=%0d job=%0d head=%0d count=%0d sequence=%016h epoch=%016h",cycle,job,head,`AT.hls_tile_count,`AT.hls_sequence,`AT.hls_epoch);
            end
            qc=at_running && `AT.hls_query_ce;kc=at_running && `AT.hls_keys_ce;vc=at_running && `AT.hls_values_ce;
            qa=int'(`AT.hls_query_address);ka=int'(`AT.hls_keys_address);va=int'(`AT.hls_values_address);
            if(`AT.hls_state_out_ce && `AT.hls_state_out_we)begin
                require(at_running && `AT.hls_state_out_address<104 && !seen_state[`AT.hls_state_out_address],"Unique private attention state");
                seen_state[`AT.hls_state_out_address]=1;state_writes++;
                $display("IAR_STATE cycle=%0d job=%0d head=%0d index=%0d data=%016h",cycle,job,head,`AT.hls_state_out_address,`AT.hls_state_out_d);
            end
            if(`AT.hls_output_ce && `AT.hls_output_we)begin
                require(at_running && !seen_output[`AT.hls_output_address],"Unique actual attention output");
                require(close32(`AT.hls_output_d,reference[sample*2048+head*64+int'(`AT.hls_output_address)]),"Actual attention output versus Decimal90 reference");
                seen_output[`AT.hls_output_address]=1;output_writes++;
                $display("IAR_OUTPUT cycle=%0d job=%0d head=%0d index=%0d data=%08h",cycle,job,head,`AT.hls_output_address,`AT.hls_output_d);
            end
            if(`AT.hls_done && at_running)begin
                require(`AT.hls_ready && `AT.hls_return==0 && (&seen_state) && (&seen_output),"Actual attention completion and coverage");
                $display("IAR_AT_DONE cycle=%0d job=%0d head=%0d status=%0d",cycle,job,head,`AT.hls_return);
                at_running=0;at_dones++;
            end
            if(dut.consumer.attention.head_completion_valid && dut.consumer.attention.head_completion_ready)begin
                require(head_owned && !at_running && dut.consumer.attention.head_completion_status==0,"Head drained before release");
                $display("IAR_HEAD_ACK cycle=%0d job=%0d head=%0d",cycle,job,head);head_owned=0;head_acks++;
            end
            if(completion_valid && completion_ready)begin
                $display("IAR_ACK cycle=%0d job=%0d",cycle,job);completions++;
            end
            #1;
            if(rc0)begin
                require(ra0<2560 && `RP.hls_input_q[0]===inputs[sample*2624+ra0],"Original RoPE Q/K input read");rp_reads++;
                $display("IAR_ROPE_READ cycle=%0d job=%0d plane=0 index=%0d data=%08h",cycle,job,ra0,`RP.hls_input_q[0]);
            end
            if(rc1)begin
                require(ra1<64 && `RP.hls_input_q[1]===inputs[sample*2624+2560+ra1],"Original RoPE coefficient read");rp_reads++;
                $display("IAR_ROPE_READ cycle=%0d job=%0d plane=1 index=%0d data=%08h",cycle,job,ra1,`RP.hls_input_q[1]);
            end
            if(qc)begin
                require(qa<64 && `AT.hls_query_q===rotated[sample*2560+head*64+qa],"Actual HLS Q read");qreads++;
                $display("IAR_Q cycle=%0d job=%0d head=%0d index=%0d data=%08h",cycle,job,head,qa,`AT.hls_query_q);
            end
            if(kc)begin
                require(ka<(sample+1)*64 && `AT.hls_keys_q_valid && `AT.hls_keys_q===rotated[(ka/64)*2560+2048+(head/4)*64+ka%64],"Actual HLS K from written cache");kreads++;
                $display("IAR_K cycle=%0d job=%0d head=%0d index=%0d data=%08h",cycle,job,head,ka,`AT.hls_keys_q);
            end
            if(vc)begin
                require(va<(sample+1)*64 && `AT.hls_values_q_valid && `AT.hls_values_q===values[(va/64)*512+(head/4)*64+va%64],"Actual HLS V from written cache");vreads++;
                $display("IAR_V cycle=%0d job=%0d head=%0d index=%0d data=%08h",cycle,job,head,va,`AT.hls_values_q);
            end
        end
    end
    task automatic step;@(posedge clk);#2;endtask
    task automatic bind_cache(input integer which);
        integer limit;
        group=which;seq_model=64'h1234567800000000+64'(which);epoch_model=64'hfedcba9800000000+64'(which);expected_epoch=epoch_model;
        bind_cache_base=BASE;bind_region_base=BASE-4032;bind_region_bytes=266240;bind_capacity=4;
        bind_sequence=seq_model;bind_epoch=epoch_model;bind_valid=1;bind_completion_ready=0;
        limit=0;while(!bind_completion_valid)begin step();limit++;require(limit<50,"Bind progress");end
        repeat(5)begin require(bind_completion_status==0 && !command_ready && busy,"Held successful bind");step();end
        bind_completion_ready=1;step();bind_completion_ready=0;step();
        require(cache_valid && !cache_metadata_fault && !busy,"Cache authority after bind ACK");
    endtask
    task automatic run_case(input integer which_sample,which_mode,which_group,which_prefix);
        integer limit,admit_before,index,want;
        job++;sample=which_sample;mode=which_mode;prefix=which_prefix;heads=0;head_acks=0;rp_writes=0;rp_reads=0;wbeats=0;
        for(integer p=0;p<3;p++)rbeats[p]=0;
        require(!p_read_active && !p_aw_pending && p_buffered==0,"Producer memory idle between jobs");
        p_phase=0;p_matrices=0;p_matrix_acks=0;p_mac=0;p_words=0;p_abi_commits=0;p_qkv_commits=0;
        p_reads=0;p_norm_reads=0;p_writes=0;p_bursts=0;p_done_seen=0;
        for(integer i=0;i<64;i++)begin abi_valid[i]=0;abi_memory[i]='x;end
        for(integer i=0;i<192;i++)begin qkv_valid[i]=0;qkv_memory[i]='x;end
        if(group!=which_group)bind_cache(which_group);
        require(!busy && command_ready && !head_owned && !rp_running && !at_running,"Idle before actual job");
        lease_model=64'h1000+64'(job*64);
        command_sequence=seq_model;command_epoch=epoch_model;command_layer=0;command_position=32'(sample);command_lease_base=lease_model;
        command_source_valid=1;command_source_sequence=seq_model;command_source_epoch=epoch_model;command_coefficient_position=32'(sample);
        for(integer i=0;i<10;i++)begin
            drive_command_addr[i]=allocation_address(i);drive_command_base[i]=allocation_base(i);drive_command_bytes[i]=allocation_bytes(i);
        end
        want=mode==0 ? 0:mode==1 ? 1:3;completion_ready=0;command_valid=1;admit_before=admissions;
        $display("IAR_BEGIN cycle=%0d job=%0d sample=%0d mode=%0d group=%0d prefix=%0d sequence=%016h epoch=%016h",cycle,job,sample,mode,group,prefix,seq_model,epoch_model);
        while(admissions==admit_before)step();repeat(2)step();
        command_sequence=0;command_epoch=0;command_layer=32'hffffffff;command_position=32'hffffffff;command_lease_base=0;
        command_source_valid=0;command_source_sequence=0;command_source_epoch=0;command_coefficient_position=32'hffffffff;
        drive_command_addr=0;drive_command_base=0;drive_command_bytes=0;
        limit=0;while(!completion_valid)begin step();limit++;require(limit<20000000,"Per-job completion deadline");end
        require(completion_status==4'(want) && completion_source_status==0 && completion_child_status==(mode==2 ? 3:0),"Exact complete-owner status");
        require(completion_sequence==seq_model && completion_epoch==epoch_model && completion_layer==0 && completion_position==32'(sample),"Retained public identity");
        require(!p_read_active && !p_aw_pending && p_buffered==0 && !aw_pending && buffered==0 && !head_owned && !at_running,"All four memory masters drained");
        if(mode==1)begin
            require(p_matrices==0 && p_reads==0 && p_writes==0 && !p_done_seen && rbeats[0]==0 && rbeats[1]==0 && rp_writes==0,"Position rejected before producer or consumer traffic");
        end else begin
            require(p_matrices==3 && p_matrix_acks==2 && p_abi_commits==64 && p_qkv_commits==192 && p_done_seen,"Actual producer retained until public retirement");
            require(rbeats[0]==164 && rbeats[1]==32 && rp_writes==2560,"Actual consumer completed full capture");
            require(completion_norm_status==0 && completion_qkv_status==0 && completion_matrix_status==0 && completion_norm_hls_status==0 && completion_norm_read_beats==256 && completion_norm_records==64 && completion_phase==2 && completion_matrices==3 && completion_total_bytes==12288,"Whole real producer certificate");
        end
        require(completion_kv_committed==(mode==0) && completion_kv_bytes==(mode==1 ? 0:4096),"KV commit and byte certificate");
        require(cache_query_valid && cache_query_layer==0 && cache_query_committed_tokens==32'(mode==0 ? sample+1:prefix),"Exact trusted current-layer prefix");
        $display("IAR_COMPLETE cycle=%0d job=%0d status=%0d child=%0d committed=%0d tokens=%0d bytes=%0d heads=%0d",cycle,job,completion_status,completion_child_status,completion_kv_committed,completion_kv_tokens,completion_kv_bytes,heads);
        if(mode==0)begin
            require(heads==32 && head_acks==32 && result_valid,"Whole attention success");
            for(integer i=0;i<2048;i++)begin
                index=(i*17+31)%2048;result_ce=1;result_address=11'(index);step();
                require(result_q_valid && close32(result_q,reference[sample*2048+index]),"Every public word versus independent attention");
                $display("IAR_RESULT cycle=%0d job=%0d index=%0d data=%08h",cycle,job,index,result_q);public_words++;
            end
            result_ce=0;step();
        end else begin errors++;require(heads==0 && !result_valid,"Error does not start attention");end
        bind_valid=1;bind_sequence=seq_model+123;bind_epoch=epoch_model+1;
        repeat(7)begin step();require(!bind_ready && !dut.consumer_bind_valid && busy && cache_exclusive && completion_valid,"Held public result excludes rebind");end
        bind_valid=0;completion_ready=1;step();completion_ready=0;
        limit=0;while(busy)begin step();limit++;require(limit<100,"Final ownership drain");end
        require(p_matrix_acks==(mode==1 ? 0:3),"All actual matrix replies retired");
        $display("IAR_RELEASE cycle=%0d job=%0d prefix=%0d",cycle,job,mode==0 ? sample+1:prefix);
    endtask
    initial begin
        require($value$plusargs("FIXTURE=%s",fixture),"Fixture required");if(!$value$plusargs("SEED=%d",seed))seed=1;
        $readmemh({fixture,"/consumer/input.hex"},inputs);$readmemh({fixture,"/consumer/v.hex"},values);
        $readmemh({fixture,"/consumer/rope-expected.hex"},rotated);$readmemh({fixture,"/consumer/attention-expected.hex"},reference);
        $readmemh({fixture,"/producer/norm/source32.hex"},source_words);
        $readmemh({fixture,"/producer/qkv/activation.hex"},abi_reference);
        $readmemh({fixture,"/producer/qkv/attn_q-weights.hex"},q_weights);
        $readmemh({fixture,"/producer/qkv/attn_k-weights.hex"},k_weights);
        $readmemh({fixture,"/producer/qkv/attn_v-weights.hex"},v_weights);
        $readmemh({fixture,"/producer/qkv/reference-fp32.hex"},projection32);
        $readmemh({fixture,"/producer/qkv/reference-fp64.hex"},projection64);
        for(integer i=0;i<4096;i++)begin memory_valid[i]=0;memory[i]='x;end
        for(integer p=0;p<3;p++)begin ar[p]=0;rv[p]=0;rd[p]=0;rl[p]=0;rbeats[p]=0;total_rbeats[p]=0;end
        repeat(5)step();rst_n=1;repeat(3)step();
        run_case(0,0,1,0);run_case(1,0,1,1);run_case(1,1,1,2);run_case(3,1,1,2);
        run_case(2,2,1,2);run_case(2,0,1,2);run_case(3,0,1,3);run_case(0,0,2,0);
        require(job==8 && admissions==8 && completions==8 && binds==2 && bindacks==2 && errors==3,"Complete job coverage");
        require(rp_starts==6 && rp_dones==6 && at_starts==160 && at_dones==160 && total_heads==160 && public_words==10240,"Both real arithmetic cores exercised");
        require(total_rbeats[0]==984 && total_rbeats[1]==192 && total_wbeats==384,"Actual source/write coverage");
        require(p_starts==6 && p_dones==6 && p_acks==6 && c_starts==6 && c_acks==6 && p_norm_starts==6 && p_norm_dones==6 && p_mac_total==18432,"All four original arithmetic cores used and drained");
        require(p_read_total==665088 && p_write_total==1536,"Actual full-size producer memory traffic");
        $display("IAR_PRODUCER_SUMMARY starts=%0d dones=%0d acks=%0d consumer_starts=%0d consumer_acks=%0d norm_starts=%0d norm_dones=%0d mac=%0d reads=%0d writes=%0d",p_starts,p_dones,p_acks,c_starts,c_acks,p_norm_starts,p_norm_dones,p_mac_total,p_read_total,p_write_total);
        $display("IAR_PASS seed=%0d jobs=%0d binds=%0d errors=%0d rope=%0d heads=%0d words=%0d source=%0d values=%0d kv_reads=%0d writes=%0d state=%0d output=%0d q=%0d k=%0d v=%0d cycles=%0d",seed,job,binds,errors,rp_dones,total_heads,public_words,total_rbeats[0],total_rbeats[1],total_rbeats[2],total_wbeats,state_writes,output_writes,qreads,kreads,vreads,cycle);
        $finish;
    end
`undef RP
`undef AT
endmodule
`default_nettype wire
