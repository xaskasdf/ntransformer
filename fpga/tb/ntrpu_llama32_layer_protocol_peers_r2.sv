// SYNTHETIC CONTROL PEER; never compile with real HLS/Vivado arithmetic.
module ntrpu_input_attention_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter integer MATRIX_TIMEOUT_CYCLES=16777216,
    parameter integer HANDOFF_TIMEOUT_CYCLES=67108864,
    parameter integer OWNER_TIMEOUT_CYCLES=268435456
) (
    input wire  clk ,
    input wire  rst_n ,
    input wire  command_valid ,
    output wire  command_ready ,
    input wire [63:0] command_sequence ,
    input wire [63:0] command_epoch ,
    input wire [63:0] command_lease_base ,
    input wire [63:0] expected_epoch ,
    input wire [31:0] command_layer ,
    input wire [31:0] command_position ,
    input wire [31:0] command_coefficient_position ,
    input wire [63:0] command_addr [0:9],
    input wire [63:0] command_base [0:9],
    input wire [63:0] command_bytes [0:9],
    input wire  cancel_valid ,
    output wire  cancel_ready ,
    output wire  busy ,
    output wire  cache_exclusive ,
    output wire  completion_valid ,
    input wire  completion_ready ,
    output wire [63:0] completion_sequence ,
    output wire [63:0] completion_epoch ,
    output wire [31:0] completion_layer ,
    output wire [31:0] completion_position ,
    output wire [3:0] completion_status ,
    output wire [3:0] completion_source_status ,
    output wire [3:0] completion_child_status ,
    output wire  completion_kv_committed ,
    output wire [12:0] completion_kv_bytes ,
    output wire [31:0] completion_kv_tokens ,
    output wire  completion_revoked ,
    output wire  result_valid ,
    input wire  result_ce ,
    input wire [10:0] result_address ,
    output wire [31:0] result_q ,
    output wire  result_q_valid ,
    input wire [63:0] command_source_sequence ,
    input wire [63:0] command_source_epoch ,
    input wire  command_source_valid ,
    output wire [3:0] completion_norm_status ,
    output wire [3:0] completion_qkv_status ,
    output wire [3:0] completion_matrix_status ,
    output wire [31:0] completion_norm_hls_status ,
    output wire [9:0] completion_norm_read_beats ,
    output wire [6:0] completion_norm_records ,
    output wire [1:0] completion_phase ,
    output wire [1:0] completion_matrices ,
    output wire [14:0] completion_total_bytes ,
    output wire  norm_tag_exhausted ,
    output wire  qkv_tags_exhausted ,
    output wire [63:0] input_axi_araddr ,
    output wire [7:0] input_axi_arlen ,
    output wire [2:0] input_axi_arsize ,
    output wire [2:0] input_axi_arprot ,
    output wire [1:0] input_axi_arburst ,
    output wire [3:0] input_axi_arid ,
    output wire [3:0] input_axi_arcache ,
    output wire  input_axi_arlock ,
    output wire  input_axi_arvalid ,
    input wire  input_axi_arready ,
    input wire [511:0] input_axi_rdata ,
    input wire [1:0] input_axi_rresp ,
    input wire [3:0] input_axi_rid ,
    input wire  input_axi_rlast ,
    input wire  input_axi_rvalid ,
    output wire  input_axi_rready ,
    output wire [63:0] input_axi_awaddr ,
    output wire [7:0] input_axi_awlen ,
    output wire [2:0] input_axi_awsize ,
    output wire [2:0] input_axi_awprot ,
    output wire [1:0] input_axi_awburst ,
    output wire [3:0] input_axi_awid ,
    output wire [3:0] input_axi_awcache ,
    output wire  input_axi_awlock ,
    output wire  input_axi_awvalid ,
    input wire  input_axi_awready ,
    output wire [511:0] input_axi_wdata ,
    output wire [63:0] input_axi_wstrb ,
    output wire  input_axi_wlast ,
    output wire  input_axi_wvalid ,
    input wire  input_axi_wready ,
    input wire [3:0] input_axi_bid ,
    input wire [1:0] input_axi_bresp ,
    input wire  input_axi_bvalid ,
    output wire  input_axi_bready ,
    output wire [63:0] v_axi_araddr ,
    output wire [7:0] v_axi_arlen ,
    output wire [2:0] v_axi_arsize ,
    output wire [2:0] v_axi_arprot ,
    output wire [1:0] v_axi_arburst ,
    output wire [3:0] v_axi_arid ,
    output wire [3:0] v_axi_arcache ,
    output wire  v_axi_arlock ,
    output wire  v_axi_arvalid ,
    input wire  v_axi_arready ,
    input wire [511:0] v_axi_rdata ,
    input wire [1:0] v_axi_rresp ,
    input wire [3:0] v_axi_rid ,
    input wire  v_axi_rlast ,
    input wire  v_axi_rvalid ,
    output wire  v_axi_rready ,
    output wire [63:0] rope_axi_araddr ,
    output wire [7:0] rope_axi_arlen ,
    output wire [2:0] rope_axi_arsize ,
    output wire [2:0] rope_axi_arprot ,
    output wire [1:0] rope_axi_arburst ,
    output wire [3:0] rope_axi_arid ,
    output wire [3:0] rope_axi_arcache ,
    output wire  rope_axi_arlock ,
    output wire  rope_axi_arvalid ,
    input wire  rope_axi_arready ,
    input wire [511:0] rope_axi_rdata ,
    input wire [1:0] rope_axi_rresp ,
    input wire [3:0] rope_axi_rid ,
    input wire  rope_axi_rlast ,
    input wire  rope_axi_rvalid ,
    output wire  rope_axi_rready ,
    input wire  bind_valid ,
    output wire  bind_ready ,
    input wire [63:0] bind_cache_base ,
    input wire [63:0] bind_region_base ,
    input wire [63:0] bind_region_bytes ,
    input wire [31:0] bind_capacity ,
    input wire [63:0] bind_sequence ,
    input wire [63:0] bind_epoch ,
    output wire  cache_valid ,
    output wire [63:0] cache_base ,
    output wire [63:0] cache_region_base ,
    output wire [63:0] cache_region_bytes ,
    output wire [63:0] cache_sequence ,
    output wire [63:0] cache_epoch ,
    output wire [31:0] cache_capacity ,
    output wire [63:0] kv_axi_awaddr ,
    output wire [7:0] kv_axi_awlen ,
    output wire [2:0] kv_axi_awsize ,
    output wire [2:0] kv_axi_awprot ,
    output wire [1:0] kv_axi_awburst ,
    output wire [3:0] kv_axi_awid ,
    output wire [3:0] kv_axi_awcache ,
    output wire  kv_axi_awlock ,
    output wire  kv_axi_awvalid ,
    input wire  kv_axi_awready ,
    output wire [511:0] kv_axi_wdata ,
    output wire [63:0] kv_axi_wstrb ,
    output wire  kv_axi_wlast ,
    output wire  kv_axi_wvalid ,
    input wire  kv_axi_wready ,
    input wire [3:0] kv_axi_bid ,
    input wire [1:0] kv_axi_bresp ,
    input wire  kv_axi_bvalid ,
    output wire  kv_axi_bready ,
    output wire [63:0] kv_axi_araddr ,
    output wire [7:0] kv_axi_arlen ,
    output wire [2:0] kv_axi_arsize ,
    output wire [2:0] kv_axi_arprot ,
    output wire [1:0] kv_axi_arburst ,
    output wire [3:0] kv_axi_arid ,
    output wire [3:0] kv_axi_arcache ,
    output wire  kv_axi_arlock ,
    output wire  kv_axi_arvalid ,
    input wire  kv_axi_arready ,
    input wire [511:0] kv_axi_rdata ,
    input wire [1:0] kv_axi_rresp ,
    input wire [3:0] kv_axi_rid ,
    input wire  kv_axi_rlast ,
    input wire  kv_axi_rvalid ,
    output wire  kv_axi_rready ,
    output wire  root_cache_lock ,
    output wire  cache_metadata_fault ,
    output wire [31:0] cache_query_layer ,
    output wire  cache_query_valid ,
    output wire [31:0] cache_query_committed_tokens ,
    output wire [31:0] cache_committed_tokens ,
    output wire  bind_completion_valid ,
    input wire  bind_completion_ready ,
    output wire [3:0] bind_completion_status ,
    output wire [63:0] bind_completion_sequence ,
    output wire [63:0] bind_completion_epoch 
);

    reg bound=0,bind_response=0;
    reg [63:0] c_base=0,c_region=0,c_bytes=0,c_seq=0,c_epoch=0;
    reg [31:0] c_capacity=0,prefix [0:15];
    reg [31:0] q=0;
    reg qvalid=0;
    reg cfg_cache_fault=0;
    integer i;
    always @(posedge clk)begin
        if(!rst_n)begin bound<=0;bind_response<=0;qvalid<=0;for(i=0;i<16;i=i+1)prefix[i]<=0;end
        else begin
            if(bind_valid && bind_ready)begin
                bound<=1;bind_response<=1;c_base<=bind_cache_base;c_region<=bind_region_base;c_bytes<=bind_region_bytes;
                c_seq<=bind_sequence;c_epoch<=bind_epoch;c_capacity<=bind_capacity;
                for(i=0;i<16;i=i+1)prefix[i]<=0;
            end
            if(bind_completion_valid && bind_completion_ready)bind_response<=0;
            if(state==1 && delay_count==0 && !cancel_valid && expected_epoch==epoch_reg && cfg_status==0)
                prefix[layer_reg]<=position_reg+1;
            qvalid<=result_ce && result_valid;
            if(result_ce && result_valid)q<=32'h51000000 ^ {21'd0,result_address} ^ seq_reg[31:0];
        end
    end

    assign command_ready=state==0 && !cfg_block;
    assign cancel_ready=state==1;
    assign busy=state!=0;
    assign cache_exclusive=state!=0;
    assign completion_valid=state==2;
    assign completion_sequence=seq_reg ^ (cfg_foreign ? 64'h8000000000000000:64'd0);
    assign completion_epoch=epoch_reg;
    assign completion_layer=layer_reg;
    assign completion_position=position_reg;
    assign completion_status=status_reg;
    assign completion_source_status='0;
    assign completion_child_status='0;
    assign completion_kv_committed=status_reg==0;
    assign completion_kv_bytes=4096;
    assign completion_kv_tokens=position_reg+1;
    assign completion_revoked='0;
    assign result_valid=state==2 && status_reg==0 && !cfg_revoke;
    assign result_q=q;
    assign result_q_valid=qvalid;
    assign completion_norm_status='0;
    assign completion_qkv_status='0;
    assign completion_matrix_status='0;
    assign completion_norm_hls_status='0;
    assign completion_norm_read_beats=cfg_badshape ? 255:256;
    assign completion_norm_records=64;
    assign completion_phase=2;
    assign completion_matrices=3;
    assign completion_total_bytes=12288;
    assign norm_tag_exhausted='0;
    assign qkv_tags_exhausted='0;
    assign input_axi_araddr='0;
    assign input_axi_arlen='0;
    assign input_axi_arsize='0;
    assign input_axi_arprot='0;
    assign input_axi_arburst='0;
    assign input_axi_arid='0;
    assign input_axi_arcache='0;
    assign input_axi_arlock='0;
    assign input_axi_arvalid='0;
    assign input_axi_rready='0;
    assign input_axi_awaddr='0;
    assign input_axi_awlen='0;
    assign input_axi_awsize='0;
    assign input_axi_awprot='0;
    assign input_axi_awburst='0;
    assign input_axi_awid='0;
    assign input_axi_awcache='0;
    assign input_axi_awlock='0;
    assign input_axi_awvalid='0;
    assign input_axi_wdata='0;
    assign input_axi_wstrb='0;
    assign input_axi_wlast='0;
    assign input_axi_wvalid='0;
    assign input_axi_bready='0;
    assign v_axi_araddr='0;
    assign v_axi_arlen='0;
    assign v_axi_arsize='0;
    assign v_axi_arprot='0;
    assign v_axi_arburst='0;
    assign v_axi_arid='0;
    assign v_axi_arcache='0;
    assign v_axi_arlock='0;
    assign v_axi_arvalid='0;
    assign v_axi_rready='0;
    assign rope_axi_araddr='0;
    assign rope_axi_arlen='0;
    assign rope_axi_arsize='0;
    assign rope_axi_arprot='0;
    assign rope_axi_arburst='0;
    assign rope_axi_arid='0;
    assign rope_axi_arcache='0;
    assign rope_axi_arlock='0;
    assign rope_axi_arvalid='0;
    assign rope_axi_rready='0;
    assign bind_ready=state==0 && !bind_response;
    assign cache_valid=bound;
    assign cache_base=c_base;
    assign cache_region_base=c_region;
    assign cache_region_bytes=c_bytes;
    assign cache_sequence=c_seq;
    assign cache_epoch=c_epoch;
    assign cache_capacity=c_capacity;
    assign kv_axi_awaddr='0;
    assign kv_axi_awlen='0;
    assign kv_axi_awsize='0;
    assign kv_axi_awprot='0;
    assign kv_axi_awburst='0;
    assign kv_axi_awid='0;
    assign kv_axi_awcache='0;
    assign kv_axi_awlock='0;
    assign kv_axi_awvalid='0;
    assign kv_axi_wdata='0;
    assign kv_axi_wstrb='0;
    assign kv_axi_wlast='0;
    assign kv_axi_wvalid='0;
    assign kv_axi_bready='0;
    assign kv_axi_araddr='0;
    assign kv_axi_arlen='0;
    assign kv_axi_arsize='0;
    assign kv_axi_arprot='0;
    assign kv_axi_arburst='0;
    assign kv_axi_arid='0;
    assign kv_axi_arcache='0;
    assign kv_axi_arlock='0;
    assign kv_axi_arvalid='0;
    assign kv_axi_rready='0;
    assign root_cache_lock='0;
    assign cache_metadata_fault=cfg_cache_fault;
    assign cache_query_layer=command_layer;
    assign cache_query_valid=bound && command_layer<16;
    assign cache_query_committed_tokens=command_layer<16 ? prefix[command_layer]:0;
    assign cache_committed_tokens='0;
    assign bind_completion_valid=bind_response;
    assign bind_completion_status='0;
    assign bind_completion_sequence=c_seq;
    assign bind_completion_epoch=c_epoch;

    integer state=0,delay_count=0,starts=0,acks=0;
    integer cfg_delay=8,cfg_drain=5,cfg_status=0;
    reg cfg_badshape=0,cfg_foreign=0,cfg_revoke=0,cfg_block=0;
    reg [63:0] seq_reg=0,epoch_reg=0,out_reg=0,residual_reg=0,abi_reg=0;
    reg [31:0] layer_reg=0,position_reg=0;
    reg [3:0] status_reg=0;
    integer read_sent=0,read_received=0;
    always @(posedge clk)begin
        if(!rst_n)begin state<=0;starts<=0;acks<=0;delay_count<=0;status_reg<=0;read_sent<=0;read_received<=0;end
        else case(state)
            0:if(command_valid && command_ready)begin
                state<=1;delay_count<=cfg_delay;starts<=starts+1;status_reg<=0;
                seq_reg<=command_sequence;epoch_reg<=command_epoch;read_sent<=0;read_received<=0;
                layer_reg<=command_layer;position_reg<=command_position;
            end
            1:begin
                if(cancel_valid || expected_epoch!=epoch_reg)begin status_reg<=cancel_valid ? 7:2;state<=2;end
                else if(delay_count==0)begin status_reg<=cfg_status;state<=2;end
                else delay_count<=delay_count-1;
            end
            2:if(completion_valid && completion_ready)begin
                if(cfg_foreign)$fatal(1,"LAYER_FAIL foreign peer acknowledged");
                state<=3;delay_count<=cfg_drain;acks<=acks+1;
            end
            3:if(delay_count==0)state<=0;else delay_count<=delay_count-1;
        endcase
    end

endmodule

// SYNTHETIC CONTROL PEER; never compile with real HLS/Vivado arithmetic.
module ntrpu_attention_output_path_r1 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter logic [63:0] INITIAL_QUANT_TAG=64'd0,
    parameter logic [63:0] INITIAL_PROJECTION_TAG=64'd0
) (
    input wire  clk ,
    input wire  rst_n ,
    input wire  command_valid ,
    output wire  command_ready ,
    input wire [63:0] command_sequence ,
    input wire [63:0] command_epoch ,
    input wire [63:0] expected_epoch ,
    input wire [63:0] command_weight_addr ,
    input wire [63:0] command_weight_region_base ,
    input wire [63:0] command_weight_region_bytes ,
    input wire [63:0] command_activation_addr ,
    input wire [63:0] command_activation_region_base ,
    input wire [63:0] command_activation_region_bytes ,
    input wire [63:0] command_output_addr ,
    input wire [63:0] command_output_region_base ,
    input wire [63:0] command_output_region_bytes ,
    input wire  cancel_valid ,
    output wire  cancel_ready ,
    output wire  busy ,
    output wire  completion_valid ,
    input wire  completion_ready ,
    output wire [3:0] completion_status ,
    output wire [63:0] completion_sequence ,
    output wire [63:0] completion_epoch ,
    output wire  result_valid ,
    output wire [63:0] result_address ,
    output wire [16:0] result_bytes ,
    output wire  tag_exhausted ,
    output wire [63:0] m_axi_araddr ,
    output wire [7:0] m_axi_arlen ,
    output wire [2:0] m_axi_arsize ,
    output wire [2:0] m_axi_arprot ,
    output wire [1:0] m_axi_arburst ,
    output wire [3:0] m_axi_arid ,
    output wire [3:0] m_axi_arcache ,
    output wire  m_axi_arlock ,
    output wire  m_axi_arvalid ,
    input wire  m_axi_arready ,
    input wire [511:0] m_axi_rdata ,
    input wire [1:0] m_axi_rresp ,
    input wire [3:0] m_axi_rid ,
    input wire  m_axi_rlast ,
    input wire  m_axi_rvalid ,
    output wire  m_axi_rready ,
    output wire [63:0] m_axi_awaddr ,
    output wire [7:0] m_axi_awlen ,
    output wire [2:0] m_axi_awsize ,
    output wire [2:0] m_axi_awprot ,
    output wire [1:0] m_axi_awburst ,
    output wire [3:0] m_axi_awid ,
    output wire [3:0] m_axi_awcache ,
    output wire  m_axi_awlock ,
    output wire  m_axi_awvalid ,
    input wire  m_axi_awready ,
    output wire [511:0] m_axi_wdata ,
    output wire [63:0] m_axi_wstrb ,
    output wire  m_axi_wlast ,
    output wire  m_axi_wvalid ,
    input wire  m_axi_wready ,
    input wire [3:0] m_axi_bid ,
    input wire [1:0] m_axi_bresp ,
    input wire  m_axi_bvalid ,
    output wire  m_axi_bready ,
    input wire  source_completion_valid ,
    output wire  source_completion_ready ,
    input wire [3:0] source_completion_status ,
    input wire [63:0] source_sequence ,
    input wire [63:0] source_epoch ,
    input wire  source_result_valid ,
    output wire  source_result_ce ,
    output wire [10:0] source_result_address ,
    input wire [31:0] source_result_q ,
    input wire  source_result_q_valid ,
    output wire [31:0] completion_hls_status 
);
    assign command_ready=state==0 && !cfg_block;
    assign cancel_ready=state==1;
    assign busy=state!=0;
    assign completion_valid=state==2;
    assign completion_status=status_reg;
    assign completion_sequence=seq_reg ^ (cfg_foreign ? 64'h8000000000000000:64'd0);
    assign completion_epoch=epoch_reg;
    assign result_valid=state==2 && status_reg==0 && !cfg_revoke;
    assign result_address=out_reg;
    assign result_bytes=cfg_badshape ? 4096:8192;
    assign tag_exhausted='0;
    assign m_axi_araddr=64'h12e;
    assign m_axi_arlen=8'h35;
    assign m_axi_arsize=3'h4;
    assign m_axi_arprot=3'h3;
    assign m_axi_arburst=2'h2;
    assign m_axi_arid=4'h1;
    assign m_axi_arcache=4'h8;
    assign m_axi_arlock=1'h1;
    assign m_axi_arvalid=1'h0;
    assign m_axi_rready=1'h1;
    assign m_axi_awaddr=64'h19e;
    assign m_axi_awlen=8'ha5;
    assign m_axi_awsize=3'h4;
    assign m_axi_awprot=3'h3;
    assign m_axi_awburst=2'h2;
    assign m_axi_awid=4'h1;
    assign m_axi_awcache=4'h8;
    assign m_axi_awlock=1'h1;
    assign m_axi_awvalid=1'h0;
    assign m_axi_wdata=512'h1e4;
    assign m_axi_wstrb=64'h1eb;
    assign m_axi_wlast=1'h0;
    assign m_axi_wvalid=1'h1;
    assign m_axi_bready=1'h0;
    assign source_completion_ready=state==2 && source_completion_valid;
    assign source_result_ce=state==1 && read_sent<2048 && source_result_valid;
    assign source_result_address=read_sent[10:0];
    assign completion_hls_status='0;

    integer state=0,delay_count=0,starts=0,acks=0;
    integer cfg_delay=8,cfg_drain=5,cfg_status=0;
    reg cfg_badshape=0,cfg_foreign=0,cfg_revoke=0,cfg_block=0;
    reg [63:0] seq_reg=0,epoch_reg=0,out_reg=0,residual_reg=0,abi_reg=0;
    reg [31:0] layer_reg=0,position_reg=0;
    reg [3:0] status_reg=0;
    integer read_sent=0,read_received=0;
    always @(posedge clk)begin
        if(!rst_n)begin state<=0;starts<=0;acks<=0;delay_count<=0;status_reg<=0;read_sent<=0;read_received<=0;end
        else case(state)
            0:if(command_valid && command_ready)begin
                state<=1;delay_count<=cfg_delay;starts<=starts+1;status_reg<=0;
                seq_reg<=command_sequence;epoch_reg<=command_epoch;read_sent<=0;read_received<=0;
                out_reg<=command_output_addr;
            end
            1:begin

                if(source_result_ce)read_sent<=read_sent+1;
                if(source_result_q_valid)begin
                    if(source_result_q !== (32'h51000000 ^ read_received ^ seq_reg[31:0]))
                        $fatal(1,"LAYER_FAIL source RAM address/data mismatch word=%0d",read_received);
                    read_received<=read_received+1;
                end
                if(cancel_valid || expected_epoch!=epoch_reg)begin status_reg<=cancel_valid ? 7:2;state<=2;end
                else if(read_received==2048)begin status_reg<=cfg_status;state<=2;end

            end
            2:if(completion_valid && completion_ready)begin
                if(cfg_foreign)$fatal(1,"LAYER_FAIL foreign peer acknowledged");
                state<=3;delay_count<=cfg_drain;acks<=acks+1;
            end
            3:if(delay_count==0)state<=0;else delay_count<=delay_count-1;
        endcase
    end

endmodule

// SYNTHETIC CONTROL PEER; never compile with real HLS/Vivado arithmetic.
module ntrpu_post_attention_real_r2 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144
) (
    input wire  clk ,
    input wire  rst_n ,
    input wire  command_valid ,
    output wire  command_ready ,
    input wire [63:0] command_sequence ,
    input wire [63:0] command_epoch ,
    input wire [63:0] expected_epoch ,
    input wire [63:0] command_source_sequence ,
    input wire [63:0] command_source_epoch ,
    input wire  command_source_valid ,
    input wire [63:0] command_input_address ,
    input wire [63:0] command_input_base ,
    input wire [63:0] command_input_bytes ,
    input wire [63:0] command_branch_address ,
    input wire [63:0] command_branch_base ,
    input wire [63:0] command_branch_bytes ,
    input wire [63:0] command_weight_address ,
    input wire [63:0] command_weight_base ,
    input wire [63:0] command_weight_bytes ,
    input wire [63:0] command_residual_address ,
    input wire [63:0] command_residual_base ,
    input wire [63:0] command_residual_bytes ,
    input wire [63:0] command_ffn_address ,
    input wire [63:0] command_ffn_base ,
    input wire [63:0] command_ffn_bytes ,
    input wire  cancel_valid ,
    output wire  cancel_ready ,
    output wire  busy ,
    output wire  completion_valid ,
    input wire  completion_ready ,
    output wire [3:0] completion_status ,
    output wire [31:0] completion_hls_status ,
    output wire [63:0] completion_sequence ,
    output wire [63:0] completion_epoch ,
    output wire [9:0] completion_read_beats ,
    output wire [7:0] completion_residual_beats ,
    output wire [6:0] completion_ffn_beats ,
    output wire  result_valid ,
    output wire [63:0] result_residual_address ,
    output wire [63:0] result_ffn_address ,
    output wire [13:0] result_residual_bytes ,
    output wire [12:0] result_ffn_bytes ,
    output wire [63:0] m_axi_araddr ,
    output wire [7:0] m_axi_arlen ,
    output wire [2:0] m_axi_arsize ,
    output wire [2:0] m_axi_arprot ,
    output wire [1:0] m_axi_arburst ,
    output wire [3:0] m_axi_arid ,
    output wire [3:0] m_axi_arcache ,
    output wire  m_axi_arlock ,
    output wire  m_axi_arvalid ,
    input wire  m_axi_arready ,
    input wire [511:0] m_axi_rdata ,
    input wire [1:0] m_axi_rresp ,
    input wire [3:0] m_axi_rid ,
    input wire  m_axi_rlast ,
    input wire  m_axi_rvalid ,
    output wire  m_axi_rready ,
    output wire [63:0] m_axi_awaddr ,
    output wire [7:0] m_axi_awlen ,
    output wire [2:0] m_axi_awsize ,
    output wire [2:0] m_axi_awprot ,
    output wire [1:0] m_axi_awburst ,
    output wire [3:0] m_axi_awid ,
    output wire [3:0] m_axi_awcache ,
    output wire  m_axi_awlock ,
    output wire  m_axi_awvalid ,
    input wire  m_axi_awready ,
    output wire [511:0] m_axi_wdata ,
    output wire [63:0] m_axi_wstrb ,
    output wire  m_axi_wlast ,
    output wire  m_axi_wvalid ,
    input wire  m_axi_wready ,
    input wire [3:0] m_axi_bid ,
    input wire [1:0] m_axi_bresp ,
    input wire  m_axi_bvalid ,
    output wire  m_axi_bready 
);
    assign command_ready=state==0 && !cfg_block;
    assign cancel_ready=state==1;
    assign busy=state!=0;
    assign completion_valid=state==2;
    assign completion_status=status_reg;
    assign completion_hls_status='0;
    assign completion_sequence=seq_reg ^ (cfg_foreign ? 64'h8000000000000000:64'd0);
    assign completion_epoch=epoch_reg;
    assign completion_read_beats=cfg_badshape ? 383:384;
    assign completion_residual_beats=128;
    assign completion_ffn_beats=64;
    assign result_valid=state==2 && status_reg==0 && !cfg_revoke;
    assign result_residual_address=residual_reg;
    assign result_ffn_address=abi_reg;
    assign result_residual_bytes=8192;
    assign result_ffn_bytes=4096;
    assign m_axi_araddr=64'h1c5;
    assign m_axi_arlen=8'hcc;
    assign m_axi_arsize=3'h3;
    assign m_axi_arprot=3'h2;
    assign m_axi_arburst=2'h1;
    assign m_axi_arid=4'h8;
    assign m_axi_arcache=4'hf;
    assign m_axi_arlock=1'h0;
    assign m_axi_arvalid=1'h1;
    assign m_axi_rready=1'h0;
    assign m_axi_awaddr=64'h235;
    assign m_axi_awlen=8'h3c;
    assign m_axi_awsize=3'h3;
    assign m_axi_awprot=3'h2;
    assign m_axi_awburst=2'h1;
    assign m_axi_awid=4'h8;
    assign m_axi_awcache=4'hf;
    assign m_axi_awlock=1'h0;
    assign m_axi_awvalid=1'h1;
    assign m_axi_wdata=512'h27b;
    assign m_axi_wstrb=64'h282;
    assign m_axi_wlast=1'h1;
    assign m_axi_wvalid=1'h0;
    assign m_axi_bready=1'h1;

    integer state=0,delay_count=0,starts=0,acks=0;
    integer cfg_delay=8,cfg_drain=5,cfg_status=0;
    reg cfg_badshape=0,cfg_foreign=0,cfg_revoke=0,cfg_block=0;
    reg [63:0] seq_reg=0,epoch_reg=0,out_reg=0,residual_reg=0,abi_reg=0;
    reg [31:0] layer_reg=0,position_reg=0;
    reg [3:0] status_reg=0;
    integer read_sent=0,read_received=0;
    always @(posedge clk)begin
        if(!rst_n)begin state<=0;starts<=0;acks<=0;delay_count<=0;status_reg<=0;read_sent<=0;read_received<=0;end
        else case(state)
            0:if(command_valid && command_ready)begin
                state<=1;delay_count<=cfg_delay;starts<=starts+1;status_reg<=0;
                seq_reg<=command_sequence;epoch_reg<=command_epoch;read_sent<=0;read_received<=0;
                residual_reg<=command_residual_address;abi_reg<=command_ffn_address;
            end
            1:begin
                if(cancel_valid || expected_epoch!=epoch_reg)begin status_reg<=cancel_valid ? 7:2;state<=2;end
                else if(delay_count==0)begin status_reg<=cfg_status;state<=2;end
                else delay_count<=delay_count-1;
            end
            2:if(completion_valid && completion_ready)begin
                if(cfg_foreign)$fatal(1,"LAYER_FAIL foreign peer acknowledged");
                state<=3;delay_count<=cfg_drain;acks<=acks+1;
            end
            3:if(delay_count==0)state<=0;else delay_count<=delay_count-1;
        endcase
    end

endmodule

// SYNTHETIC CONTROL PEER; never compile with real HLS/Vivado arithmetic.
module ntrpu_llama32_ffn_residual_r3 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter logic [63:0] INITIAL_TAG=64'd0
) (
    input wire  clk ,
    input wire  rst_n ,
    input wire  command_valid ,
    input wire [63:0] command_sequence ,
    input wire [63:0] command_epoch ,
    input wire  command_source_valid ,
    input wire [63:0] command_source_sequence ,
    input wire [63:0] command_source_epoch ,
    input wire [63:0] command_addr [0:7],
    input wire [63:0] command_base [0:7],
    input wire [63:0] command_bytes [0:7],
    input wire [63:0] command_residual_address ,
    input wire [63:0] command_residual_base ,
    input wire [63:0] command_residual_bytes ,
    input wire [63:0] expected_epoch ,
    input wire [63:0] command_output_address ,
    input wire [63:0] command_output_base ,
    input wire [63:0] command_output_bytes ,
    output wire  command_ready ,
    input wire  cancel_valid ,
    output wire  cancel_ready ,
    output wire  busy ,
    output wire  completion_valid ,
    input wire  completion_ready ,
    output wire [3:0] completion_status ,
    output wire [3:0] completion_ffn_status ,
    output wire [3:0] completion_ffn_child_status ,
    output wire [3:0] completion_residual_status ,
    output wire [2:0] completion_ffn_stages ,
    output wire [1:0] completion_ffn_phase ,
    output wire [16:0] completion_ffn_bytes ,
    output wire [17:0] completion_ffn_stage_bytes ,
    output wire [31:0] completion_ffn_hls_status ,
    output wire [31:0] completion_hls_status ,
    output wire [9:0] completion_read_beats ,
    output wire [7:0] completion_write_beats ,
    output wire [63:0] completion_sequence ,
    output wire [63:0] completion_epoch ,
    output wire  result_valid ,
    output wire [63:0] result_address ,
    output wire [13:0] result_bytes ,
    output wire  tags_exhausted ,
    output wire  activation_selected ,
    output wire  residual_selected ,
    output wire [63:0] m_axi_araddr ,
    output wire [7:0] m_axi_arlen ,
    output wire [2:0] m_axi_arsize ,
    output wire [2:0] m_axi_arprot ,
    output wire [1:0] m_axi_arburst ,
    output wire [3:0] m_axi_arid ,
    output wire [3:0] m_axi_arcache ,
    output wire  m_axi_arlock ,
    output wire  m_axi_arvalid ,
    input wire  m_axi_arready ,
    input wire [511:0] m_axi_rdata ,
    input wire [1:0] m_axi_rresp ,
    input wire [3:0] m_axi_rid ,
    input wire  m_axi_rlast ,
    input wire  m_axi_rvalid ,
    output wire  m_axi_rready ,
    output wire [63:0] m_axi_awaddr ,
    output wire [7:0] m_axi_awlen ,
    output wire [2:0] m_axi_awsize ,
    output wire [2:0] m_axi_awprot ,
    output wire [1:0] m_axi_awburst ,
    output wire [3:0] m_axi_awid ,
    output wire [3:0] m_axi_awcache ,
    output wire  m_axi_awlock ,
    output wire  m_axi_awvalid ,
    input wire  m_axi_awready ,
    output wire [511:0] m_axi_wdata ,
    output wire [63:0] m_axi_wstrb ,
    output wire  m_axi_wlast ,
    output wire  m_axi_wvalid ,
    input wire  m_axi_wready ,
    input wire [3:0] m_axi_bid ,
    input wire [1:0] m_axi_bresp ,
    input wire  m_axi_bvalid ,
    output wire  m_axi_bready 
);
    assign command_ready=state==0 && !cfg_block;
    assign cancel_ready=state==1;
    assign busy=state!=0;
    assign completion_valid=state==2;
    assign completion_status=status_reg;
    assign completion_ffn_status='0;
    assign completion_ffn_child_status='0;
    assign completion_residual_status='0;
    assign completion_ffn_stages=4;
    assign completion_ffn_phase=3;
    assign completion_ffn_bytes=8192;
    assign completion_ffn_stage_bytes=155648;
    assign completion_ffn_hls_status='0;
    assign completion_hls_status='0;
    assign completion_read_beats=cfg_badshape ? 255:256;
    assign completion_write_beats=128;
    assign completion_sequence=seq_reg ^ (cfg_foreign ? 64'h8000000000000000:64'd0);
    assign completion_epoch=epoch_reg;
    assign result_valid=state==2 && status_reg==0 && !cfg_revoke;
    assign result_address=out_reg;
    assign result_bytes=8192;
    assign tags_exhausted='0;
    assign activation_selected='0;
    assign residual_selected='0;
    assign m_axi_araddr=64'h208;
    assign m_axi_arlen=8'hf;
    assign m_axi_arsize=3'h6;
    assign m_axi_arprot=3'h5;
    assign m_axi_arburst=2'h0;
    assign m_axi_arid=4'hb;
    assign m_axi_arcache=4'h2;
    assign m_axi_arlock=1'h1;
    assign m_axi_arvalid=1'h0;
    assign m_axi_rready=1'h1;
    assign m_axi_awaddr=64'h278;
    assign m_axi_awlen=8'h7f;
    assign m_axi_awsize=3'h6;
    assign m_axi_awprot=3'h5;
    assign m_axi_awburst=2'h0;
    assign m_axi_awid=4'hb;
    assign m_axi_awcache=4'h2;
    assign m_axi_awlock=1'h1;
    assign m_axi_awvalid=1'h0;
    assign m_axi_wdata=512'h2be;
    assign m_axi_wstrb=64'h2c5;
    assign m_axi_wlast=1'h0;
    assign m_axi_wvalid=1'h1;
    assign m_axi_bready=1'h0;

    integer state=0,delay_count=0,starts=0,acks=0;
    integer cfg_delay=8,cfg_drain=5,cfg_status=0;
    reg cfg_badshape=0,cfg_foreign=0,cfg_revoke=0,cfg_block=0;
    reg [63:0] seq_reg=0,epoch_reg=0,out_reg=0,residual_reg=0,abi_reg=0;
    reg [31:0] layer_reg=0,position_reg=0;
    reg [3:0] status_reg=0;
    integer read_sent=0,read_received=0;
    always @(posedge clk)begin
        if(!rst_n)begin state<=0;starts<=0;acks<=0;delay_count<=0;status_reg<=0;read_sent<=0;read_received<=0;end
        else case(state)
            0:if(command_valid && command_ready)begin
                state<=1;delay_count<=cfg_delay;starts<=starts+1;status_reg<=0;
                seq_reg<=command_sequence;epoch_reg<=command_epoch;read_sent<=0;read_received<=0;
                out_reg<=command_output_address;
            end
            1:begin
                if(cancel_valid || expected_epoch!=epoch_reg)begin status_reg<=cancel_valid ? 7:2;state<=2;end
                else if(delay_count==0)begin status_reg<=cfg_status;state<=2;end
                else delay_count<=delay_count-1;
            end
            2:if(completion_valid && completion_ready)begin
                if(cfg_foreign)$fatal(1,"LAYER_FAIL foreign peer acknowledged");
                state<=3;delay_count<=cfg_drain;acks<=acks+1;
            end
            3:if(delay_count==0)state<=0;else delay_count<=delay_count-1;
        endcase
    end

endmodule
