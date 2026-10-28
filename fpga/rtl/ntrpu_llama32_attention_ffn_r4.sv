// Actual attention -> O -> post-O -> FFN -> residual, with one retained AXI owner.
// Q and committed K/V are supplied externally; this is not full token inference.
// Sixteen disjoint allocations are captured before attention can write memory.
// Global reset must quiesce both complete children and the AXI memory target.
`default_nettype none
module ntrpu_llama32_attention_ffn_r4 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter logic [63:0] INITIAL_QUANT_TAG=64'd0,
    parameter logic [63:0] INITIAL_PROJECTION_TAG=64'd0,
    parameter logic [63:0] INITIAL_FFN_TAG=64'd0
) (
    input wire clk,
    input wire rst_n,
    input wire bind_valid,
    output wire bind_ready,
    input wire [63:0] bind_cache_base,
    input wire [63:0] bind_region_base,
    input wire [63:0] bind_region_bytes,
    input wire [31:0] bind_capacity,
    input wire [63:0] bind_sequence,
    input wire [63:0] bind_epoch,
    input wire [63:0] expected_epoch,
    output wire cache_valid,
    output wire [63:0] cache_base,
    output wire [63:0] cache_region_base,
    output wire [63:0] cache_region_bytes,
    output wire [63:0] cache_sequence,
    output wire [63:0] cache_epoch,
    output wire [31:0] cache_capacity,
    input wire append_valid,
    output wire append_ready,
    output wire append_prepared,
    input wire [31:0] append_layer,
    input wire [31:0] append_position,
    input wire [63:0] append_sequence,
    input wire [63:0] append_epoch,
    input wire data_valid,
    output wire data_ready,
    input wire [511:0] data_bits,
    input wire [5:0] data_index,
    input wire data_last,
    input wire [63:0] data_sequence,
    input wire [63:0] data_epoch,
    input wire commit_valid,
    output wire commit_ready,
    input wire [3:0] commit_status,
    input wire [63:0] commit_sequence,
    input wire [63:0] commit_epoch,
    input wire append_cancel_valid,
    output wire append_cancel_ready,
    output wire append_done_valid,
    input wire append_done_ready,
    output wire append_done_bind,
    output wire [3:0] append_done_status,
    output wire [12:0] append_done_bytes,
    output wire [31:0] append_done_committed_tokens,
    output wire [63:0] append_done_sequence,
    output wire [63:0] append_done_epoch,
    output wire lease_active,
    output wire [63:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [2:0] m_axi_awprot,
    output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awid,
    output wire [3:0] m_axi_awcache,
    output wire m_axi_awlock,
    output wire m_axi_awvalid,
    input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast,
    output wire m_axi_wvalid,
    input wire m_axi_wready,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready,
    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [2:0] m_axi_arprot,
    output wire [1:0] m_axi_arburst,
    output wire [3:0] m_axi_arid,
    output wire [3:0] m_axi_arcache,
    output wire m_axi_arlock,
    output wire m_axi_arvalid,
    input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire [3:0] m_axi_rid,
    input wire m_axi_rlast,
    input wire m_axi_rvalid,
    output wire m_axi_rready,
    input wire command_valid,
    output wire command_ready,
    input wire [31:0] command_position,
    input wire [31:0] command_layer,
    input wire [63:0] command_sequence,
    input wire [63:0] command_epoch,
    input wire cancel_valid,
    output wire cancel_ready,
    output wire completion_valid,
    input wire completion_ready,
    output wire [3:0] completion_status,
    output wire result_valid,
    input wire [63:0] command_lease_base,
    input wire q_valid,
    output wire q_ready,
    input wire [511:0] q_data,
    input wire [7:0] q_index,
    input wire q_last,
    input wire [63:0] q_sequence,
    input wire [63:0] q_epoch,
    output wire [63:0] completion_sequence,
    output wire [63:0] completion_epoch,
    output wire cache_lock,
    output wire maintenance_busy,
    output wire metadata_fault,
    output wire [31:0] committed_tokens,
    output wire busy,
    output wire tag_exhausted,
    output wire [31:0] completion_hls_status,
    output wire [63:0] result_address,
    output wire [13:0] result_bytes,
    input wire [63:0] command_weight_addr,
    input wire [63:0] command_weight_region_base,
    input wire [63:0] command_weight_region_bytes,
    input wire [63:0] command_activation_addr,
    input wire [63:0] command_activation_region_base,
    input wire [63:0] command_activation_region_bytes,
    input wire [63:0] command_output_addr,
    input wire [63:0] command_output_region_base,
    input wire [63:0] command_output_region_bytes,
    input wire [63:0] command_post_input_address,
    input wire [63:0] command_post_input_base,
    input wire [63:0] command_post_input_bytes,
    input wire [63:0] command_post_weight_address,
    input wire [63:0] command_post_weight_base,
    input wire [63:0] command_post_weight_bytes,
    input wire [63:0] command_post_residual_address,
    input wire [63:0] command_post_residual_base,
    input wire [63:0] command_post_residual_bytes,
    input wire [63:0] command_post_ffn_address,
    input wire [63:0] command_post_ffn_base,
    input wire [63:0] command_post_ffn_bytes,
    output wire [31:0] completion_post_hls_status,
    output wire [9:0] completion_post_read_beats,
    output wire [7:0] completion_residual_beats,
    output wire [6:0] completion_ffn_beats,
    output wire [3:0] completion_ffn_status,
    output wire [3:0] completion_ffn_child_status,
    output wire [3:0] completion_residual_status,
    output wire [2:0] completion_ffn_stages,
    output wire [1:0] completion_ffn_phase,
    output wire [16:0] completion_ffn_bytes,
    output wire [17:0] completion_ffn_stage_bytes,
    output wire [31:0] completion_ffn_hls_status,
    output wire [9:0] completion_read_beats,
    output wire [7:0] completion_write_beats,
    output wire activation_selected,
    output wire residual_selected,
    output wire [3:0] completion_attention_status,
    output wire [3:0] completion_ffn_residual_status,
    output wire [31:0] completion_attention_hls_status,
    output wire ffn_selected,
    input wire [63:0] command_ffn_addr [1:7],
    input wire [63:0] command_ffn_base [1:7],
    input wire [63:0] command_ffn_bytes [1:7],
    input wire [63:0] command_final_address,
    input wire [63:0] command_final_base,
    input wire [63:0] command_final_bytes
);
    typedef enum logic [3:0] {IDLE,CHECK_SPANS,CHECK_PAIRS,ATTENTION_OFFER,WAIT_ATTENTION,WAIT_HANDOFF,LOCAL_RESPONSE,FINAL_RESPONSE,WAIT_RELEASE} state_t;
    state_t state;
    logic [15:0][63:0] address,allocation_base,allocation_bytes;
    logic [3:0] span_index,pair_left,pair_right,fault,public_status;
    logic cancel_seen,attention_launched,handoff_launched;
    logic [63:0] saved_cache_sequence,saved_cache_epoch;
    logic [31:0] saved_cache_capacity;
    logic [9:0] saved_post_read_beats;
    logic [7:0] saved_residual_beats;
    logic [6:0] saved_ffn_beats;
    logic [31:0] held_command_position;
    logic [31:0] held_command_layer;
    logic [63:0] held_command_sequence;
    logic [63:0] held_command_epoch;
    logic [63:0] held_command_lease_base;
    logic [63:0] held_command_weight_addr;
    logic [63:0] held_command_weight_region_base;
    logic [63:0] held_command_weight_region_bytes;
    logic [63:0] held_command_activation_addr;
    logic [63:0] held_command_activation_region_base;
    logic [63:0] held_command_activation_region_bytes;
    logic [63:0] held_command_output_addr;
    logic [63:0] held_command_output_region_base;
    logic [63:0] held_command_output_region_bytes;
    logic [63:0] held_command_post_input_address;
    logic [63:0] held_command_post_input_base;
    logic [63:0] held_command_post_input_bytes;
    logic [63:0] held_command_post_weight_address;
    logic [63:0] held_command_post_weight_base;
    logic [63:0] held_command_post_weight_bytes;
    logic [63:0] held_command_post_residual_address;
    logic [63:0] held_command_post_residual_base;
    logic [63:0] held_command_post_residual_bytes;
    logic [63:0] held_command_post_ffn_address;
    logic [63:0] held_command_post_ffn_base;
    logic [63:0] held_command_post_ffn_bytes;
    wire a_bind_valid;
    wire a_bind_ready;
    wire [63:0] a_bind_cache_base;
    wire [63:0] a_bind_region_base;
    wire [63:0] a_bind_region_bytes;
    wire [31:0] a_bind_capacity;
    wire [63:0] a_bind_sequence;
    wire [63:0] a_bind_epoch;
    wire a_cache_valid;
    wire [63:0] a_cache_base;
    wire [63:0] a_cache_region_base;
    wire [63:0] a_cache_region_bytes;
    wire [63:0] a_cache_sequence;
    wire [63:0] a_cache_epoch;
    wire [31:0] a_cache_capacity;
    wire a_append_valid;
    wire a_append_ready;
    wire a_append_prepared;
    wire [31:0] a_append_layer;
    wire [31:0] a_append_position;
    wire [63:0] a_append_sequence;
    wire [63:0] a_append_epoch;
    wire a_data_valid;
    wire a_data_ready;
    wire [511:0] a_data_bits;
    wire [5:0] a_data_index;
    wire a_data_last;
    wire [63:0] a_data_sequence;
    wire [63:0] a_data_epoch;
    wire a_commit_valid;
    wire a_commit_ready;
    wire [3:0] a_commit_status;
    wire [63:0] a_commit_sequence;
    wire [63:0] a_commit_epoch;
    wire a_append_cancel_valid;
    wire a_append_cancel_ready;
    wire a_append_done_valid;
    wire a_append_done_ready;
    wire a_append_done_bind;
    wire [3:0] a_append_done_status;
    wire [12:0] a_append_done_bytes;
    wire [31:0] a_append_done_committed_tokens;
    wire [63:0] a_append_done_sequence;
    wire [63:0] a_append_done_epoch;
    wire a_lease_active;
    wire [63:0] a_m_axi_awaddr;
    wire [7:0] a_m_axi_awlen;
    wire [2:0] a_m_axi_awsize;
    wire [2:0] a_m_axi_awprot;
    wire [1:0] a_m_axi_awburst;
    wire [3:0] a_m_axi_awid;
    wire [3:0] a_m_axi_awcache;
    wire a_m_axi_awlock;
    wire a_m_axi_awvalid;
    wire a_m_axi_awready;
    wire [511:0] a_m_axi_wdata;
    wire [63:0] a_m_axi_wstrb;
    wire a_m_axi_wlast;
    wire a_m_axi_wvalid;
    wire a_m_axi_wready;
    wire [3:0] a_m_axi_bid;
    wire [1:0] a_m_axi_bresp;
    wire a_m_axi_bvalid;
    wire a_m_axi_bready;
    wire [63:0] a_m_axi_araddr;
    wire [7:0] a_m_axi_arlen;
    wire [2:0] a_m_axi_arsize;
    wire [2:0] a_m_axi_arprot;
    wire [1:0] a_m_axi_arburst;
    wire [3:0] a_m_axi_arid;
    wire [3:0] a_m_axi_arcache;
    wire a_m_axi_arlock;
    wire a_m_axi_arvalid;
    wire a_m_axi_arready;
    wire [511:0] a_m_axi_rdata;
    wire [1:0] a_m_axi_rresp;
    wire [3:0] a_m_axi_rid;
    wire a_m_axi_rlast;
    wire a_m_axi_rvalid;
    wire a_m_axi_rready;
    wire a_command_valid;
    wire a_command_ready;
    wire [31:0] a_command_position;
    wire [31:0] a_command_layer;
    wire [63:0] a_command_sequence;
    wire [63:0] a_command_epoch;
    wire a_cancel_valid;
    wire a_cancel_ready;
    wire a_completion_valid;
    wire a_completion_ready;
    wire [3:0] a_completion_status;
    wire a_result_valid;
    wire [63:0] a_command_lease_base;
    wire a_q_valid;
    wire a_q_ready;
    wire [511:0] a_q_data;
    wire [7:0] a_q_index;
    wire a_q_last;
    wire [63:0] a_q_sequence;
    wire [63:0] a_q_epoch;
    wire [63:0] a_completion_sequence;
    wire [63:0] a_completion_epoch;
    wire a_cache_lock;
    wire a_maintenance_busy;
    wire a_metadata_fault;
    wire [31:0] a_committed_tokens;
    wire a_busy;
    wire a_tag_exhausted;
    wire [31:0] a_completion_hls_status;
    wire [63:0] a_result_address;
    wire [16:0] a_result_bytes;
    wire [63:0] a_command_weight_addr;
    wire [63:0] a_command_weight_region_base;
    wire [63:0] a_command_weight_region_bytes;
    wire [63:0] a_command_activation_addr;
    wire [63:0] a_command_activation_region_base;
    wire [63:0] a_command_activation_region_bytes;
    wire [63:0] a_command_output_addr;
    wire [63:0] a_command_output_region_base;
    wire [63:0] a_command_output_region_bytes;
    wire [63:0] a_command_post_input_address;
    wire [63:0] a_command_post_input_base;
    wire [63:0] a_command_post_input_bytes;
    wire [63:0] a_command_post_weight_address;
    wire [63:0] a_command_post_weight_base;
    wire [63:0] a_command_post_weight_bytes;
    wire [63:0] a_command_post_residual_address;
    wire [63:0] a_command_post_residual_base;
    wire [63:0] a_command_post_residual_bytes;
    wire [63:0] a_command_post_ffn_address;
    wire [63:0] a_command_post_ffn_base;
    wire [63:0] a_command_post_ffn_bytes;
    wire [31:0] a_completion_post_hls_status;
    wire [9:0] a_completion_post_read_beats;
    wire [7:0] a_completion_residual_beats;
    wire [6:0] a_completion_ffn_beats;
    wire [63:0] a_result_residual_address;
    wire [63:0] a_result_ffn_address;
    wire [13:0] a_result_residual_bytes;
    wire [12:0] a_result_ffn_bytes;
    wire f_command_valid;
    wire [63:0] f_command_sequence;
    wire [63:0] f_command_epoch;
    wire f_command_source_valid;
    wire [63:0] f_command_source_sequence;
    wire [63:0] f_command_source_epoch;
    wire [63:0] f_command_addr [0:7];
    wire [63:0] f_command_base [0:7];
    wire [63:0] f_command_bytes [0:7];
    wire [63:0] f_command_residual_address;
    wire [63:0] f_command_residual_base;
    wire [63:0] f_command_residual_bytes;
    wire [63:0] f_command_output_address;
    wire [63:0] f_command_output_base;
    wire [63:0] f_command_output_bytes;
    wire f_command_ready;
    wire f_cancel_valid;
    wire f_cancel_ready;
    wire f_busy;
    wire f_completion_valid;
    wire f_completion_ready;
    wire [3:0] f_completion_status;
    wire [3:0] f_completion_ffn_status;
    wire [3:0] f_completion_ffn_child_status;
    wire [3:0] f_completion_residual_status;
    wire [2:0] f_completion_ffn_stages;
    wire [1:0] f_completion_ffn_phase;
    wire [16:0] f_completion_ffn_bytes;
    wire [17:0] f_completion_ffn_stage_bytes;
    wire [31:0] f_completion_ffn_hls_status;
    wire [31:0] f_completion_hls_status;
    wire [9:0] f_completion_read_beats;
    wire [7:0] f_completion_write_beats;
    wire [63:0] f_completion_sequence;
    wire [63:0] f_completion_epoch;
    wire f_result_valid;
    wire [63:0] f_result_address;
    wire [13:0] f_result_bytes;
    wire f_tags_exhausted;
    wire f_activation_selected;
    wire f_residual_selected;
    wire [63:0] f_m_axi_araddr;
    wire [7:0] f_m_axi_arlen;
    wire [2:0] f_m_axi_arsize;
    wire [2:0] f_m_axi_arprot;
    wire [1:0] f_m_axi_arburst;
    wire [3:0] f_m_axi_arid;
    wire [3:0] f_m_axi_arcache;
    wire f_m_axi_arlock;
    wire f_m_axi_arvalid;
    wire f_m_axi_arready;
    wire [511:0] f_m_axi_rdata;
    wire [1:0] f_m_axi_rresp;
    wire [3:0] f_m_axi_rid;
    wire f_m_axi_rlast;
    wire f_m_axi_rvalid;
    wire f_m_axi_rready;
    wire [63:0] f_m_axi_awaddr;
    wire [7:0] f_m_axi_awlen;
    wire [2:0] f_m_axi_awsize;
    wire [2:0] f_m_axi_awprot;
    wire [1:0] f_m_axi_awburst;
    wire [3:0] f_m_axi_awid;
    wire [3:0] f_m_axi_awcache;
    wire f_m_axi_awlock;
    wire f_m_axi_awvalid;
    wire f_m_axi_awready;
    wire [511:0] f_m_axi_wdata;
    wire [63:0] f_m_axi_wstrb;
    wire f_m_axi_wlast;
    wire f_m_axi_wvalid;
    wire f_m_axi_wready;
    wire [3:0] f_m_axi_bid;
    wire [1:0] f_m_axi_bresp;
    wire f_m_axi_bvalid;
    wire f_m_axi_bready;
    wire h_command_valid;
    wire h_command_ready;
    wire [63:0] h_command_sequence;
    wire [63:0] h_command_epoch;
    wire [63:0] h_command_output_address;
    wire [63:0] h_command_output_base;
    wire [63:0] h_command_output_bytes;
    wire [63:0] h_command_ffn_addr [1:7];
    wire [63:0] h_command_ffn_base [1:7];
    wire [63:0] h_command_ffn_bytes [1:7];
    wire [63:0] h_command_protected_base [0:7];
    wire [63:0] h_command_protected_bytes [0:7];
    wire h_cancel_valid;
    wire h_cancel_ready;
    wire h_busy;
    wire h_source_completion_valid;
    wire h_source_completion_ready;
    wire [3:0] h_source_completion_status;
    wire [63:0] h_source_sequence;
    wire [63:0] h_source_epoch;
    wire h_source_result_valid;
    wire [63:0] h_source_residual_address;
    wire [63:0] h_source_ffn_address;
    wire [13:0] h_source_residual_bytes;
    wire [12:0] h_source_ffn_bytes;
    wire [31:0] h_source_root_hls_status;
    wire [31:0] h_source_post_hls_status;
    wire [9:0] h_source_post_read_beats;
    wire [7:0] h_source_residual_beats;
    wire [6:0] h_source_ffn_beats;
    wire h_child_command_valid;
    wire h_child_command_ready;
    wire [63:0] h_child_command_sequence;
    wire [63:0] h_child_command_epoch;
    wire [63:0] h_child_source_sequence;
    wire [63:0] h_child_source_epoch;
    wire h_child_source_valid;
    wire [63:0] h_child_command_addr [0:7];
    wire [63:0] h_child_command_base [0:7];
    wire [63:0] h_child_command_bytes [0:7];
    wire [63:0] h_child_residual_address;
    wire [63:0] h_child_residual_base;
    wire [63:0] h_child_residual_bytes;
    wire [63:0] h_child_output_address;
    wire [63:0] h_child_output_base;
    wire [63:0] h_child_output_bytes;
    wire h_child_cancel_valid;
    wire h_child_completion_valid;
    wire h_child_completion_ready;
    wire [3:0] h_child_completion_status;
    wire [31:0] h_child_hls_status;
    wire [31:0] h_child_ffn_hls_status;
    wire [3:0] h_child_ffn_status;
    wire [3:0] h_child_ffn_child_status;
    wire [3:0] h_child_residual_status;
    wire [2:0] h_child_ffn_stages;
    wire [1:0] h_child_ffn_phase;
    wire [16:0] h_child_ffn_bytes;
    wire [17:0] h_child_ffn_stage_bytes;
    wire [63:0] h_child_sequence;
    wire [63:0] h_child_epoch;
    wire [9:0] h_child_read_beats;
    wire [7:0] h_child_write_beats;
    wire h_child_result_valid;
    wire [63:0] h_child_result_address;
    wire [13:0] h_child_result_bytes;
    wire h_ffn_bus_owned;
    wire h_completion_valid;
    wire h_completion_ready;
    wire [3:0] h_completion_status;
    wire [3:0] h_completion_source_status;
    wire [3:0] h_completion_child_status;
    wire [31:0] h_completion_hls_status;
    wire [31:0] h_completion_ffn_hls_status;
    wire [3:0] h_completion_ffn_status;
    wire [3:0] h_completion_ffn_child_status;
    wire [3:0] h_completion_residual_status;
    wire [2:0] h_completion_ffn_stages;
    wire [1:0] h_completion_ffn_phase;
    wire [16:0] h_completion_ffn_bytes;
    wire [17:0] h_completion_ffn_stage_bytes;
    wire [31:0] h_completion_source_root_hls_status;
    wire [31:0] h_completion_source_post_hls_status;
    wire [63:0] h_completion_sequence;
    wire [63:0] h_completion_epoch;
    wire [9:0] h_completion_read_beats;
    wire [7:0] h_completion_write_beats;
    wire h_result_valid;
    wire [63:0] h_result_address;
    wire [13:0] h_result_bytes;
    wire epoch_bad=busy && expected_epoch!=held_command_epoch;
    wire cache_changed=busy && (!a_cache_valid || a_metadata_fault || a_cache_base!=address[0] ||
        a_cache_region_base!=allocation_base[0] || a_cache_region_bytes!=allocation_bytes[0] ||
        a_cache_sequence!=saved_cache_sequence || a_cache_epoch!=saved_cache_epoch || a_cache_capacity!=saved_cache_capacity);
    wire attention_identity=a_completion_sequence==held_command_sequence && a_completion_epoch==held_command_epoch;
    wire ffn_identity=f_completion_sequence==held_command_sequence && f_completion_epoch==held_command_epoch;
    wire source_shape_bad=a_completion_status==0 && (!a_result_valid || a_result_address!=address[3] || a_result_bytes!=8192 ||
        a_result_residual_address!=address[6] || a_result_residual_bytes!=8192 || a_result_ffn_address!=address[7] || a_result_ffn_bytes!=4096);
    wire [3:0] fault_now=fault!=0 ? fault:epoch_bad ? 4'd2:cancel_valid ? 4'd7:cache_changed ? 4'd11:
        attention_launched && a_completion_valid && !attention_identity ? 4'd11:
        state==WAIT_ATTENTION && a_completion_valid && attention_identity && source_shape_bad ? 4'd5:4'd0;
    // Only registered child faults feed cancellation, avoiding result/cancel feedback loops.
    wire lifecycle_fault=fault!=0 || cancel_seen || cancel_valid || epoch_bad;
    wire can_release=a_completion_valid && attention_identity && (!h_ffn_bus_owned || (f_completion_valid && ffn_identity));
    wire [64:0] left_end={1'b0,allocation_base[pair_left]}+{1'b0,allocation_bytes[pair_left]};
    wire [64:0] right_end={1'b0,allocation_base[pair_right]}+{1'b0,allocation_bytes[pair_right]};
    wire pair_overlap={1'b0,allocation_base[pair_left]}<right_end && {1'b0,allocation_base[pair_right]}<left_end;
    function automatic logic [63:0] required_span(input logic [3:0] i);
        case(i)
            0:return 64'd1; // Cache engine validated the full layout at bind.
            1:return 64'd4456448;
            2,7:return 64'd4096;
            8,9,10:return 64'd17825792;
            11,12:return 64'd65536;
            13:return 64'd16384;
            default:return 64'd8192;
        endcase
    endfunction
    function automatic logic span_valid(input logic [63:0] addr,base,bytes,span);
        logic [64:0] ending,limit;
        ending={1'b0,addr}+{1'b0,span};limit={1'b0,base}+{1'b0,bytes};
        return addr[5:0]==0 && bytes!=0 && !ending[64] && !limit[64] && addr>=base && ending<=limit;
    endfunction
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && !bind_valid && !append_valid && a_command_ready && !a_busy && !a_completion_valid && !a_maintenance_busy && !a_lease_active && f_command_ready && !f_busy && !f_completion_valid && !h_busy;
    assign cancel_ready=busy && !cancel_seen;
    assign completion_valid=state==LOCAL_RESPONSE || (state==FINAL_RESPONSE && h_completion_valid && can_release);
    assign completion_status=state==LOCAL_RESPONSE ? fault:public_status;
    assign completion_sequence=held_command_sequence;
    assign completion_epoch=held_command_epoch;
    assign result_valid=state==FINAL_RESPONSE && completion_valid && public_status==0 && h_result_valid && !lifecycle_fault && !cache_changed;
    assign result_address=result_valid ? h_result_address:64'd0;
    assign result_bytes=result_valid ? h_result_bytes:14'd0;
    assign ffn_selected=h_ffn_bus_owned;
    assign activation_selected=ffn_selected && f_activation_selected;
    assign residual_selected=ffn_selected && f_residual_selected;
    assign tag_exhausted=a_tag_exhausted || f_tags_exhausted;
    assign cache_lock=busy || a_cache_lock;
    assign bind_ready=state==IDLE && a_bind_ready;
    assign append_ready=state==IDLE && a_append_ready;
    assign q_ready=attention_launched && !ffn_selected && a_q_ready;
    assign a_bind_valid=state==IDLE && bind_valid;
    assign a_append_valid=state==IDLE && append_valid;
    assign a_command_valid=state==ATTENTION_OFFER;
    assign a_cancel_valid=(attention_launched || state==ATTENTION_OFFER) && lifecycle_fault;
    assign a_completion_ready=h_source_completion_ready;
    assign a_q_valid=attention_launched && !ffn_selected && q_valid;
    assign h_command_valid=state==WAIT_ATTENTION && a_completion_valid && attention_identity;
    assign h_command_sequence=held_command_sequence;
    assign h_command_epoch=held_command_epoch;
    assign h_command_output_address=address[15];
    assign h_command_output_base=allocation_base[15];
    assign h_command_output_bytes=allocation_bytes[15];
    assign h_cancel_valid=busy && lifecycle_fault;
    assign h_source_completion_valid=a_completion_valid;
    assign h_source_completion_status=a_completion_status;
    assign h_source_sequence=a_completion_sequence;
    assign h_source_epoch=a_completion_epoch;
    assign h_source_result_valid=a_result_valid && !source_shape_bad;
    assign h_source_residual_address=a_result_residual_address;
    assign h_source_residual_bytes=a_result_residual_bytes;
    assign h_source_ffn_address=a_result_ffn_address;
    assign h_source_ffn_bytes=a_result_ffn_bytes;
    assign h_source_root_hls_status=a_completion_hls_status;
    assign h_source_post_hls_status=a_completion_post_hls_status;
    assign h_source_post_read_beats=a_completion_post_read_beats;
    assign h_source_residual_beats=a_completion_residual_beats;
    assign h_source_ffn_beats=a_completion_ffn_beats;
    assign h_child_command_ready=f_command_ready;
    assign h_child_completion_valid=f_completion_valid;
    assign h_child_completion_status=f_completion_status;
    assign h_child_hls_status=f_completion_hls_status;
    assign h_child_ffn_hls_status=f_completion_ffn_hls_status;
    assign h_child_sequence=f_completion_sequence;
    assign h_child_epoch=f_completion_epoch;
    assign h_child_read_beats=f_completion_read_beats;
    assign h_child_write_beats=f_completion_write_beats;
    assign h_child_result_valid=f_result_valid;
    assign h_child_result_address=f_result_address;
    assign h_child_result_bytes=f_result_bytes;
    assign h_completion_ready=state==FINAL_RESPONSE && completion_ready && can_release;
    assign f_command_valid=h_child_command_valid;
    assign f_command_sequence=h_child_command_sequence;
    assign f_command_epoch=h_child_command_epoch;
    assign f_command_source_valid=h_child_source_valid;
    assign f_command_source_sequence=h_child_source_sequence;
    assign f_command_source_epoch=h_child_source_epoch;
    assign f_cancel_valid=h_child_cancel_valid;
    assign f_completion_ready=h_child_completion_ready;
    assign completion_attention_status=handoff_launched ? h_completion_source_status:4'd0;
    assign completion_ffn_residual_status=handoff_launched ? h_completion_child_status:4'd0;
    assign completion_attention_hls_status=handoff_launched ? h_completion_source_root_hls_status:32'd0;
    assign completion_post_hls_status=handoff_launched ? h_completion_source_post_hls_status:32'd0;
    assign completion_post_read_beats=saved_post_read_beats;
    assign completion_residual_beats=saved_residual_beats;
    assign completion_ffn_beats=saved_ffn_beats;
    assign h_child_ffn_status=f_completion_ffn_status;
    assign h_child_ffn_child_status=f_completion_ffn_child_status;
    assign h_child_residual_status=f_completion_residual_status;
    assign h_child_ffn_stages=f_completion_ffn_stages;
    assign h_child_ffn_phase=f_completion_ffn_phase;
    assign h_child_ffn_bytes=f_completion_ffn_bytes;
    assign h_child_ffn_stage_bytes=f_completion_ffn_stage_bytes;
    assign completion_ffn_status=handoff_launched ? h_completion_ffn_status:'0;
    assign completion_ffn_child_status=handoff_launched ? h_completion_ffn_child_status:'0;
    assign completion_residual_status=handoff_launched ? h_completion_residual_status:'0;
    assign completion_ffn_stages=handoff_launched ? h_completion_ffn_stages:'0;
    assign completion_ffn_phase=handoff_launched ? h_completion_ffn_phase:'0;
    assign completion_ffn_bytes=handoff_launched ? h_completion_ffn_bytes:'0;
    assign completion_ffn_stage_bytes=handoff_launched ? h_completion_ffn_stage_bytes:'0;
    assign completion_ffn_hls_status=handoff_launched ? h_completion_ffn_hls_status:'0;
    assign completion_hls_status=handoff_launched ? h_completion_hls_status:'0;
    assign completion_read_beats=handoff_launched ? h_completion_read_beats:'0;
    assign completion_write_beats=handoff_launched ? h_completion_write_beats:'0;
    assign f_command_residual_address=h_child_residual_address;
    assign f_command_output_address=h_child_output_address;
    assign f_command_residual_base=h_child_residual_base;
    assign f_command_output_base=h_child_output_base;
    assign f_command_residual_bytes=h_child_residual_bytes;
    assign f_command_output_bytes=h_child_output_bytes;
    assign a_bind_cache_base=bind_cache_base;
    assign a_bind_region_base=bind_region_base;
    assign a_bind_region_bytes=bind_region_bytes;
    assign a_bind_capacity=bind_capacity;
    assign a_bind_sequence=bind_sequence;
    assign a_bind_epoch=bind_epoch;
    assign cache_valid=a_cache_valid;
    assign cache_base=a_cache_base;
    assign cache_region_base=a_cache_region_base;
    assign cache_region_bytes=a_cache_region_bytes;
    assign cache_sequence=a_cache_sequence;
    assign cache_epoch=a_cache_epoch;
    assign cache_capacity=a_cache_capacity;
    assign append_prepared=a_append_prepared;
    assign a_append_layer=append_layer;
    assign a_append_position=append_position;
    assign a_append_sequence=append_sequence;
    assign a_append_epoch=append_epoch;
    assign a_data_valid=state==IDLE && data_valid;
    assign data_ready=state==IDLE && a_data_ready;
    assign a_data_bits=data_bits;
    assign a_data_index=data_index;
    assign a_data_last=data_last;
    assign a_data_sequence=data_sequence;
    assign a_data_epoch=data_epoch;
    assign a_commit_valid=state==IDLE && commit_valid;
    assign commit_ready=state==IDLE && a_commit_ready;
    assign a_commit_status=commit_status;
    assign a_commit_sequence=commit_sequence;
    assign a_commit_epoch=commit_epoch;
    assign a_append_cancel_valid=state==IDLE && append_cancel_valid;
    assign append_cancel_ready=state==IDLE && a_append_cancel_ready;
    assign append_done_valid=a_append_done_valid;
    assign a_append_done_ready=state==IDLE && append_done_ready;
    assign append_done_bind=a_append_done_bind;
    assign append_done_status=a_append_done_status;
    assign append_done_bytes=a_append_done_bytes;
    assign append_done_committed_tokens=a_append_done_committed_tokens;
    assign append_done_sequence=a_append_done_sequence;
    assign append_done_epoch=a_append_done_epoch;
    assign lease_active=a_lease_active;
    assign a_command_position=state==IDLE ? command_position:held_command_position;
    assign a_command_layer=state==IDLE ? command_layer:held_command_layer;
    assign a_command_sequence=state==IDLE ? command_sequence:held_command_sequence;
    assign a_command_epoch=state==IDLE ? command_epoch:held_command_epoch;
    assign a_command_lease_base=state==IDLE ? command_lease_base:held_command_lease_base;
    assign a_q_data=q_data;
    assign a_q_index=q_index;
    assign a_q_last=q_last;
    assign a_q_sequence=q_sequence;
    assign a_q_epoch=q_epoch;
    assign maintenance_busy=a_maintenance_busy;
    assign metadata_fault=a_metadata_fault;
    assign committed_tokens=a_committed_tokens;
    assign a_command_weight_addr=state==IDLE ? command_weight_addr:held_command_weight_addr;
    assign a_command_weight_region_base=state==IDLE ? command_weight_region_base:held_command_weight_region_base;
    assign a_command_weight_region_bytes=state==IDLE ? command_weight_region_bytes:held_command_weight_region_bytes;
    assign a_command_activation_addr=state==IDLE ? command_activation_addr:held_command_activation_addr;
    assign a_command_activation_region_base=state==IDLE ? command_activation_region_base:held_command_activation_region_base;
    assign a_command_activation_region_bytes=state==IDLE ? command_activation_region_bytes:held_command_activation_region_bytes;
    assign a_command_output_addr=state==IDLE ? command_output_addr:held_command_output_addr;
    assign a_command_output_region_base=state==IDLE ? command_output_region_base:held_command_output_region_base;
    assign a_command_output_region_bytes=state==IDLE ? command_output_region_bytes:held_command_output_region_bytes;
    assign a_command_post_input_address=state==IDLE ? command_post_input_address:held_command_post_input_address;
    assign a_command_post_input_base=state==IDLE ? command_post_input_base:held_command_post_input_base;
    assign a_command_post_input_bytes=state==IDLE ? command_post_input_bytes:held_command_post_input_bytes;
    assign a_command_post_weight_address=state==IDLE ? command_post_weight_address:held_command_post_weight_address;
    assign a_command_post_weight_base=state==IDLE ? command_post_weight_base:held_command_post_weight_base;
    assign a_command_post_weight_bytes=state==IDLE ? command_post_weight_bytes:held_command_post_weight_bytes;
    assign a_command_post_residual_address=state==IDLE ? command_post_residual_address:held_command_post_residual_address;
    assign a_command_post_residual_base=state==IDLE ? command_post_residual_base:held_command_post_residual_base;
    assign a_command_post_residual_bytes=state==IDLE ? command_post_residual_bytes:held_command_post_residual_bytes;
    assign a_command_post_ffn_address=state==IDLE ? command_post_ffn_address:held_command_post_ffn_address;
    assign a_command_post_ffn_base=state==IDLE ? command_post_ffn_base:held_command_post_ffn_base;
    assign a_command_post_ffn_bytes=state==IDLE ? command_post_ffn_bytes:held_command_post_ffn_bytes;
    for(genvar i=0;i<8;i++)begin
        assign h_command_protected_base[i]=allocation_base[i];
        assign h_command_protected_bytes[i]=allocation_bytes[i];
        assign f_command_addr[i]=h_child_command_addr[i];
        assign f_command_base[i]=h_child_command_base[i];
        assign f_command_bytes[i]=h_child_command_bytes[i];
    end
    for(genvar i=1;i<8;i++)begin
        assign h_command_ffn_addr[i]=address[i+7];
        assign h_command_ffn_base[i]=allocation_base[i+7];
        assign h_command_ffn_bytes[i]=allocation_bytes[i+7];
    end
    assign m_axi_awaddr=ffn_selected ? f_m_axi_awaddr:a_m_axi_awaddr;
    assign m_axi_awlen=ffn_selected ? f_m_axi_awlen:a_m_axi_awlen;
    assign m_axi_awsize=ffn_selected ? f_m_axi_awsize:a_m_axi_awsize;
    assign m_axi_awprot=ffn_selected ? f_m_axi_awprot:a_m_axi_awprot;
    assign m_axi_awburst=ffn_selected ? f_m_axi_awburst:a_m_axi_awburst;
    assign m_axi_awid=ffn_selected ? f_m_axi_awid:a_m_axi_awid;
    assign m_axi_awcache=ffn_selected ? f_m_axi_awcache:a_m_axi_awcache;
    assign m_axi_awlock=ffn_selected ? f_m_axi_awlock:a_m_axi_awlock;
    assign m_axi_awvalid=ffn_selected ? f_m_axi_awvalid:a_m_axi_awvalid;
    assign a_m_axi_awready=!ffn_selected && m_axi_awready;
    assign f_m_axi_awready=ffn_selected && m_axi_awready;
    assign m_axi_wdata=ffn_selected ? f_m_axi_wdata:a_m_axi_wdata;
    assign m_axi_wstrb=ffn_selected ? f_m_axi_wstrb:a_m_axi_wstrb;
    assign m_axi_wlast=ffn_selected ? f_m_axi_wlast:a_m_axi_wlast;
    assign m_axi_wvalid=ffn_selected ? f_m_axi_wvalid:a_m_axi_wvalid;
    assign a_m_axi_wready=!ffn_selected && m_axi_wready;
    assign f_m_axi_wready=ffn_selected && m_axi_wready;
    assign a_m_axi_bid=m_axi_bid;
    assign f_m_axi_bid=m_axi_bid;
    assign a_m_axi_bresp=m_axi_bresp;
    assign f_m_axi_bresp=m_axi_bresp;
    assign a_m_axi_bvalid=!ffn_selected && m_axi_bvalid;
    assign f_m_axi_bvalid=ffn_selected && m_axi_bvalid;
    assign m_axi_bready=ffn_selected ? f_m_axi_bready:a_m_axi_bready;
    assign m_axi_araddr=ffn_selected ? f_m_axi_araddr:a_m_axi_araddr;
    assign m_axi_arlen=ffn_selected ? f_m_axi_arlen:a_m_axi_arlen;
    assign m_axi_arsize=ffn_selected ? f_m_axi_arsize:a_m_axi_arsize;
    assign m_axi_arprot=ffn_selected ? f_m_axi_arprot:a_m_axi_arprot;
    assign m_axi_arburst=ffn_selected ? f_m_axi_arburst:a_m_axi_arburst;
    assign m_axi_arid=ffn_selected ? f_m_axi_arid:a_m_axi_arid;
    assign m_axi_arcache=ffn_selected ? f_m_axi_arcache:a_m_axi_arcache;
    assign m_axi_arlock=ffn_selected ? f_m_axi_arlock:a_m_axi_arlock;
    assign m_axi_arvalid=ffn_selected ? f_m_axi_arvalid:a_m_axi_arvalid;
    assign a_m_axi_arready=!ffn_selected && m_axi_arready;
    assign f_m_axi_arready=ffn_selected && m_axi_arready;
    assign a_m_axi_rdata=m_axi_rdata;
    assign f_m_axi_rdata=m_axi_rdata;
    assign a_m_axi_rresp=m_axi_rresp;
    assign f_m_axi_rresp=m_axi_rresp;
    assign a_m_axi_rid=m_axi_rid;
    assign f_m_axi_rid=m_axi_rid;
    assign a_m_axi_rlast=m_axi_rlast;
    assign f_m_axi_rlast=m_axi_rlast;
    assign a_m_axi_rvalid=!ffn_selected && m_axi_rvalid;
    assign f_m_axi_rvalid=ffn_selected && m_axi_rvalid;
    assign m_axi_rready=ffn_selected ? f_m_axi_rready:a_m_axi_rready;
    ntrpu_attention_post_path_r2 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES),.INITIAL_QUANT_TAG(INITIAL_QUANT_TAG),.INITIAL_PROJECTION_TAG(INITIAL_PROJECTION_TAG)) attention (
        .clk(clk),
        .rst_n(rst_n),
        .bind_valid(a_bind_valid),
        .bind_ready(a_bind_ready),
        .bind_cache_base(a_bind_cache_base),
        .bind_region_base(a_bind_region_base),
        .bind_region_bytes(a_bind_region_bytes),
        .bind_capacity(a_bind_capacity),
        .bind_sequence(a_bind_sequence),
        .bind_epoch(a_bind_epoch),
        .expected_epoch(expected_epoch),
        .cache_valid(a_cache_valid),
        .cache_base(a_cache_base),
        .cache_region_base(a_cache_region_base),
        .cache_region_bytes(a_cache_region_bytes),
        .cache_sequence(a_cache_sequence),
        .cache_epoch(a_cache_epoch),
        .cache_capacity(a_cache_capacity),
        .append_valid(a_append_valid),
        .append_ready(a_append_ready),
        .append_prepared(a_append_prepared),
        .append_layer(a_append_layer),
        .append_position(a_append_position),
        .append_sequence(a_append_sequence),
        .append_epoch(a_append_epoch),
        .data_valid(a_data_valid),
        .data_ready(a_data_ready),
        .data_bits(a_data_bits),
        .data_index(a_data_index),
        .data_last(a_data_last),
        .data_sequence(a_data_sequence),
        .data_epoch(a_data_epoch),
        .commit_valid(a_commit_valid),
        .commit_ready(a_commit_ready),
        .commit_status(a_commit_status),
        .commit_sequence(a_commit_sequence),
        .commit_epoch(a_commit_epoch),
        .append_cancel_valid(a_append_cancel_valid),
        .append_cancel_ready(a_append_cancel_ready),
        .append_done_valid(a_append_done_valid),
        .append_done_ready(a_append_done_ready),
        .append_done_bind(a_append_done_bind),
        .append_done_status(a_append_done_status),
        .append_done_bytes(a_append_done_bytes),
        .append_done_committed_tokens(a_append_done_committed_tokens),
        .append_done_sequence(a_append_done_sequence),
        .append_done_epoch(a_append_done_epoch),
        .lease_active(a_lease_active),
        .m_axi_awaddr(a_m_axi_awaddr),
        .m_axi_awlen(a_m_axi_awlen),
        .m_axi_awsize(a_m_axi_awsize),
        .m_axi_awprot(a_m_axi_awprot),
        .m_axi_awburst(a_m_axi_awburst),
        .m_axi_awid(a_m_axi_awid),
        .m_axi_awcache(a_m_axi_awcache),
        .m_axi_awlock(a_m_axi_awlock),
        .m_axi_awvalid(a_m_axi_awvalid),
        .m_axi_awready(a_m_axi_awready),
        .m_axi_wdata(a_m_axi_wdata),
        .m_axi_wstrb(a_m_axi_wstrb),
        .m_axi_wlast(a_m_axi_wlast),
        .m_axi_wvalid(a_m_axi_wvalid),
        .m_axi_wready(a_m_axi_wready),
        .m_axi_bid(a_m_axi_bid),
        .m_axi_bresp(a_m_axi_bresp),
        .m_axi_bvalid(a_m_axi_bvalid),
        .m_axi_bready(a_m_axi_bready),
        .m_axi_araddr(a_m_axi_araddr),
        .m_axi_arlen(a_m_axi_arlen),
        .m_axi_arsize(a_m_axi_arsize),
        .m_axi_arprot(a_m_axi_arprot),
        .m_axi_arburst(a_m_axi_arburst),
        .m_axi_arid(a_m_axi_arid),
        .m_axi_arcache(a_m_axi_arcache),
        .m_axi_arlock(a_m_axi_arlock),
        .m_axi_arvalid(a_m_axi_arvalid),
        .m_axi_arready(a_m_axi_arready),
        .m_axi_rdata(a_m_axi_rdata),
        .m_axi_rresp(a_m_axi_rresp),
        .m_axi_rid(a_m_axi_rid),
        .m_axi_rlast(a_m_axi_rlast),
        .m_axi_rvalid(a_m_axi_rvalid),
        .m_axi_rready(a_m_axi_rready),
        .command_valid(a_command_valid),
        .command_ready(a_command_ready),
        .command_position(a_command_position),
        .command_layer(a_command_layer),
        .command_sequence(a_command_sequence),
        .command_epoch(a_command_epoch),
        .cancel_valid(a_cancel_valid),
        .cancel_ready(a_cancel_ready),
        .completion_valid(a_completion_valid),
        .completion_ready(a_completion_ready),
        .completion_status(a_completion_status),
        .result_valid(a_result_valid),
        .command_lease_base(a_command_lease_base),
        .q_valid(a_q_valid),
        .q_ready(a_q_ready),
        .q_data(a_q_data),
        .q_index(a_q_index),
        .q_last(a_q_last),
        .q_sequence(a_q_sequence),
        .q_epoch(a_q_epoch),
        .completion_sequence(a_completion_sequence),
        .completion_epoch(a_completion_epoch),
        .cache_lock(a_cache_lock),
        .maintenance_busy(a_maintenance_busy),
        .metadata_fault(a_metadata_fault),
        .committed_tokens(a_committed_tokens),
        .busy(a_busy),
        .tag_exhausted(a_tag_exhausted),
        .completion_hls_status(a_completion_hls_status),
        .result_address(a_result_address),
        .result_bytes(a_result_bytes),
        .command_weight_addr(a_command_weight_addr),
        .command_weight_region_base(a_command_weight_region_base),
        .command_weight_region_bytes(a_command_weight_region_bytes),
        .command_activation_addr(a_command_activation_addr),
        .command_activation_region_base(a_command_activation_region_base),
        .command_activation_region_bytes(a_command_activation_region_bytes),
        .command_output_addr(a_command_output_addr),
        .command_output_region_base(a_command_output_region_base),
        .command_output_region_bytes(a_command_output_region_bytes),
        .command_post_input_address(a_command_post_input_address),
        .command_post_input_base(a_command_post_input_base),
        .command_post_input_bytes(a_command_post_input_bytes),
        .command_post_weight_address(a_command_post_weight_address),
        .command_post_weight_base(a_command_post_weight_base),
        .command_post_weight_bytes(a_command_post_weight_bytes),
        .command_post_residual_address(a_command_post_residual_address),
        .command_post_residual_base(a_command_post_residual_base),
        .command_post_residual_bytes(a_command_post_residual_bytes),
        .command_post_ffn_address(a_command_post_ffn_address),
        .command_post_ffn_base(a_command_post_ffn_base),
        .command_post_ffn_bytes(a_command_post_ffn_bytes),
        .completion_post_hls_status(a_completion_post_hls_status),
        .completion_post_read_beats(a_completion_post_read_beats),
        .completion_residual_beats(a_completion_residual_beats),
        .completion_ffn_beats(a_completion_ffn_beats),
        .result_residual_address(a_result_residual_address),
        .result_ffn_address(a_result_ffn_address),
        .result_residual_bytes(a_result_residual_bytes),
        .result_ffn_bytes(a_result_ffn_bytes)
    );
    ntrpu_llama32_ffn_residual_r3 #(.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES),.INITIAL_TAG(INITIAL_FFN_TAG)) ffn (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(f_command_valid),
        .command_sequence(f_command_sequence),
        .command_epoch(f_command_epoch),
        .command_source_valid(f_command_source_valid),
        .command_source_sequence(f_command_source_sequence),
        .command_source_epoch(f_command_source_epoch),
        .command_addr(f_command_addr),
        .command_base(f_command_base),
        .command_bytes(f_command_bytes),
        .command_residual_address(f_command_residual_address),
        .command_residual_base(f_command_residual_base),
        .command_residual_bytes(f_command_residual_bytes),
        .expected_epoch(expected_epoch),
        .command_output_address(f_command_output_address),
        .command_output_base(f_command_output_base),
        .command_output_bytes(f_command_output_bytes),
        .command_ready(f_command_ready),
        .cancel_valid(f_cancel_valid),
        .cancel_ready(f_cancel_ready),
        .busy(f_busy),
        .completion_valid(f_completion_valid),
        .completion_ready(f_completion_ready),
        .completion_status(f_completion_status),
        .completion_ffn_status(f_completion_ffn_status),
        .completion_ffn_child_status(f_completion_ffn_child_status),
        .completion_residual_status(f_completion_residual_status),
        .completion_ffn_stages(f_completion_ffn_stages),
        .completion_ffn_phase(f_completion_ffn_phase),
        .completion_ffn_bytes(f_completion_ffn_bytes),
        .completion_ffn_stage_bytes(f_completion_ffn_stage_bytes),
        .completion_ffn_hls_status(f_completion_ffn_hls_status),
        .completion_hls_status(f_completion_hls_status),
        .completion_read_beats(f_completion_read_beats),
        .completion_write_beats(f_completion_write_beats),
        .completion_sequence(f_completion_sequence),
        .completion_epoch(f_completion_epoch),
        .result_valid(f_result_valid),
        .result_address(f_result_address),
        .result_bytes(f_result_bytes),
        .tags_exhausted(f_tags_exhausted),
        .activation_selected(f_activation_selected),
        .residual_selected(f_residual_selected),
        .m_axi_araddr(f_m_axi_araddr),
        .m_axi_arlen(f_m_axi_arlen),
        .m_axi_arsize(f_m_axi_arsize),
        .m_axi_arprot(f_m_axi_arprot),
        .m_axi_arburst(f_m_axi_arburst),
        .m_axi_arid(f_m_axi_arid),
        .m_axi_arcache(f_m_axi_arcache),
        .m_axi_arlock(f_m_axi_arlock),
        .m_axi_arvalid(f_m_axi_arvalid),
        .m_axi_arready(f_m_axi_arready),
        .m_axi_rdata(f_m_axi_rdata),
        .m_axi_rresp(f_m_axi_rresp),
        .m_axi_rid(f_m_axi_rid),
        .m_axi_rlast(f_m_axi_rlast),
        .m_axi_rvalid(f_m_axi_rvalid),
        .m_axi_rready(f_m_axi_rready),
        .m_axi_awaddr(f_m_axi_awaddr),
        .m_axi_awlen(f_m_axi_awlen),
        .m_axi_awsize(f_m_axi_awsize),
        .m_axi_awprot(f_m_axi_awprot),
        .m_axi_awburst(f_m_axi_awburst),
        .m_axi_awid(f_m_axi_awid),
        .m_axi_awcache(f_m_axi_awcache),
        .m_axi_awlock(f_m_axi_awlock),
        .m_axi_awvalid(f_m_axi_awvalid),
        .m_axi_awready(f_m_axi_awready),
        .m_axi_wdata(f_m_axi_wdata),
        .m_axi_wstrb(f_m_axi_wstrb),
        .m_axi_wlast(f_m_axi_wlast),
        .m_axi_wvalid(f_m_axi_wvalid),
        .m_axi_wready(f_m_axi_wready),
        .m_axi_bid(f_m_axi_bid),
        .m_axi_bresp(f_m_axi_bresp),
        .m_axi_bvalid(f_m_axi_bvalid),
        .m_axi_bready(f_m_axi_bready)
    );
    ntrpu_post_ffn_handoff_r2 handoff (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(h_command_valid),
        .command_ready(h_command_ready),
        .command_sequence(h_command_sequence),
        .command_epoch(h_command_epoch),
        .expected_epoch(expected_epoch),
        .command_output_address(h_command_output_address),
        .command_output_base(h_command_output_base),
        .command_output_bytes(h_command_output_bytes),
        .command_ffn_addr(h_command_ffn_addr),
        .command_ffn_base(h_command_ffn_base),
        .command_ffn_bytes(h_command_ffn_bytes),
        .command_protected_base(h_command_protected_base),
        .command_protected_bytes(h_command_protected_bytes),
        .cancel_valid(h_cancel_valid),
        .cancel_ready(h_cancel_ready),
        .busy(h_busy),
        .source_completion_valid(h_source_completion_valid),
        .source_completion_ready(h_source_completion_ready),
        .source_completion_status(h_source_completion_status),
        .source_sequence(h_source_sequence),
        .source_epoch(h_source_epoch),
        .source_result_valid(h_source_result_valid),
        .source_residual_address(h_source_residual_address),
        .source_ffn_address(h_source_ffn_address),
        .source_residual_bytes(h_source_residual_bytes),
        .source_ffn_bytes(h_source_ffn_bytes),
        .source_root_hls_status(h_source_root_hls_status),
        .source_post_hls_status(h_source_post_hls_status),
        .source_post_read_beats(h_source_post_read_beats),
        .source_residual_beats(h_source_residual_beats),
        .source_ffn_beats(h_source_ffn_beats),
        .child_command_valid(h_child_command_valid),
        .child_command_ready(h_child_command_ready),
        .child_command_sequence(h_child_command_sequence),
        .child_command_epoch(h_child_command_epoch),
        .child_source_sequence(h_child_source_sequence),
        .child_source_epoch(h_child_source_epoch),
        .child_source_valid(h_child_source_valid),
        .child_command_addr(h_child_command_addr),
        .child_command_base(h_child_command_base),
        .child_command_bytes(h_child_command_bytes),
        .child_residual_address(h_child_residual_address),
        .child_residual_base(h_child_residual_base),
        .child_residual_bytes(h_child_residual_bytes),
        .child_output_address(h_child_output_address),
        .child_output_base(h_child_output_base),
        .child_output_bytes(h_child_output_bytes),
        .child_cancel_valid(h_child_cancel_valid),
        .child_completion_valid(h_child_completion_valid),
        .child_completion_ready(h_child_completion_ready),
        .child_completion_status(h_child_completion_status),
        .child_hls_status(h_child_hls_status),
        .child_ffn_hls_status(h_child_ffn_hls_status),
        .child_ffn_status(h_child_ffn_status),
        .child_ffn_child_status(h_child_ffn_child_status),
        .child_residual_status(h_child_residual_status),
        .child_ffn_stages(h_child_ffn_stages),
        .child_ffn_phase(h_child_ffn_phase),
        .child_ffn_bytes(h_child_ffn_bytes),
        .child_ffn_stage_bytes(h_child_ffn_stage_bytes),
        .child_sequence(h_child_sequence),
        .child_epoch(h_child_epoch),
        .child_read_beats(h_child_read_beats),
        .child_write_beats(h_child_write_beats),
        .child_result_valid(h_child_result_valid),
        .child_result_address(h_child_result_address),
        .child_result_bytes(h_child_result_bytes),
        .ffn_bus_owned(h_ffn_bus_owned),
        .completion_valid(h_completion_valid),
        .completion_ready(h_completion_ready),
        .completion_status(h_completion_status),
        .completion_source_status(h_completion_source_status),
        .completion_child_status(h_completion_child_status),
        .completion_hls_status(h_completion_hls_status),
        .completion_ffn_hls_status(h_completion_ffn_hls_status),
        .completion_ffn_status(h_completion_ffn_status),
        .completion_ffn_child_status(h_completion_ffn_child_status),
        .completion_residual_status(h_completion_residual_status),
        .completion_ffn_stages(h_completion_ffn_stages),
        .completion_ffn_phase(h_completion_ffn_phase),
        .completion_ffn_bytes(h_completion_ffn_bytes),
        .completion_ffn_stage_bytes(h_completion_ffn_stage_bytes),
        .completion_source_root_hls_status(h_completion_source_root_hls_status),
        .completion_source_post_hls_status(h_completion_source_post_hls_status),
        .completion_sequence(h_completion_sequence),
        .completion_epoch(h_completion_epoch),
        .completion_read_beats(h_completion_read_beats),
        .completion_write_beats(h_completion_write_beats),
        .result_valid(h_result_valid),
        .result_address(h_result_address),
        .result_bytes(h_result_bytes)
    );
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;allocation_base<=0;allocation_bytes<=0;
            span_index<=0;pair_left<=0;pair_right<=1;fault<=0;public_status<=0;
            cancel_seen<=0;attention_launched<=0;handoff_launched<=0;
            saved_cache_sequence<=0;saved_cache_epoch<=0;saved_cache_capacity<=0;
            saved_post_read_beats<=0;saved_residual_beats<=0;saved_ffn_beats<=0;
            held_command_position<=0;
            held_command_layer<=0;
            held_command_sequence<=0;
            held_command_epoch<=0;
            held_command_lease_base<=0;
            held_command_weight_addr<=0;
            held_command_weight_region_base<=0;
            held_command_weight_region_bytes<=0;
            held_command_activation_addr<=0;
            held_command_activation_region_base<=0;
            held_command_activation_region_bytes<=0;
            held_command_output_addr<=0;
            held_command_output_region_base<=0;
            held_command_output_region_bytes<=0;
            held_command_post_input_address<=0;
            held_command_post_input_base<=0;
            held_command_post_input_bytes<=0;
            held_command_post_weight_address<=0;
            held_command_post_weight_base<=0;
            held_command_post_weight_bytes<=0;
            held_command_post_residual_address<=0;
            held_command_post_residual_base<=0;
            held_command_post_residual_bytes<=0;
            held_command_post_ffn_address<=0;
            held_command_post_ffn_base<=0;
            held_command_post_ffn_bytes<=0;
        end else begin
            if(busy && state!=LOCAL_RESPONSE)begin
                fault<=fault_now;
                if(cancel_valid)cancel_seen<=1;
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    span_index<=0;pair_left<=0;pair_right<=1;public_status<=0;
                    cancel_seen<=cancel_valid;attention_launched<=0;handoff_launched<=0;
                    saved_post_read_beats<=0;saved_residual_beats<=0;saved_ffn_beats<=0;
                    saved_cache_sequence<=a_cache_sequence;saved_cache_epoch<=a_cache_epoch;saved_cache_capacity<=a_cache_capacity;
                    if(command_epoch!=expected_epoch || !a_cache_valid || command_sequence!=a_cache_sequence || command_epoch!=a_cache_epoch)fault<=2;
                    else if(cancel_valid)fault<=7;
                    else if(a_metadata_fault)fault<=11;
                    else if(command_layer>=16 || a_cache_capacity==0 || a_cache_capacity>131072 || command_lease_base==0 || command_lease_base>64'hffffffffffffffe0)fault<=1;
                    else fault<=0;
                    held_command_position<=command_position;
                    held_command_layer<=command_layer;
                    held_command_sequence<=command_sequence;
                    held_command_epoch<=command_epoch;
                    held_command_lease_base<=command_lease_base;
                    held_command_weight_addr<=command_weight_addr;
                    held_command_weight_region_base<=command_weight_region_base;
                    held_command_weight_region_bytes<=command_weight_region_bytes;
                    held_command_activation_addr<=command_activation_addr;
                    held_command_activation_region_base<=command_activation_region_base;
                    held_command_activation_region_bytes<=command_activation_region_bytes;
                    held_command_output_addr<=command_output_addr;
                    held_command_output_region_base<=command_output_region_base;
                    held_command_output_region_bytes<=command_output_region_bytes;
                    held_command_post_input_address<=command_post_input_address;
                    held_command_post_input_base<=command_post_input_base;
                    held_command_post_input_bytes<=command_post_input_bytes;
                    held_command_post_weight_address<=command_post_weight_address;
                    held_command_post_weight_base<=command_post_weight_base;
                    held_command_post_weight_bytes<=command_post_weight_bytes;
                    held_command_post_residual_address<=command_post_residual_address;
                    held_command_post_residual_base<=command_post_residual_base;
                    held_command_post_residual_bytes<=command_post_residual_bytes;
                    held_command_post_ffn_address<=command_post_ffn_address;
                    held_command_post_ffn_base<=command_post_ffn_base;
                    held_command_post_ffn_bytes<=command_post_ffn_bytes;
                    address[0]<=a_cache_base;
                    allocation_base[0]<=a_cache_region_base;
                    allocation_bytes[0]<=a_cache_region_bytes;
                    address[1]<=command_weight_addr;
                    allocation_base[1]<=command_weight_region_base;
                    allocation_bytes[1]<=command_weight_region_bytes;
                    address[2]<=command_activation_addr;
                    allocation_base[2]<=command_activation_region_base;
                    allocation_bytes[2]<=command_activation_region_bytes;
                    address[3]<=command_output_addr;
                    allocation_base[3]<=command_output_region_base;
                    allocation_bytes[3]<=command_output_region_bytes;
                    address[4]<=command_post_input_address;
                    allocation_base[4]<=command_post_input_base;
                    allocation_bytes[4]<=command_post_input_bytes;
                    address[5]<=command_post_weight_address;
                    allocation_base[5]<=command_post_weight_base;
                    allocation_bytes[5]<=command_post_weight_bytes;
                    address[6]<=command_post_residual_address;
                    allocation_base[6]<=command_post_residual_base;
                    allocation_bytes[6]<=command_post_residual_bytes;
                    address[7]<=command_post_ffn_address;
                    allocation_base[7]<=command_post_ffn_base;
                    allocation_bytes[7]<=command_post_ffn_bytes;
                    address[8]<=command_ffn_addr[1];
                    allocation_base[8]<=command_ffn_base[1];
                    allocation_bytes[8]<=command_ffn_bytes[1];
                    address[9]<=command_ffn_addr[2];
                    allocation_base[9]<=command_ffn_base[2];
                    allocation_bytes[9]<=command_ffn_bytes[2];
                    address[10]<=command_ffn_addr[3];
                    allocation_base[10]<=command_ffn_base[3];
                    allocation_bytes[10]<=command_ffn_bytes[3];
                    address[11]<=command_ffn_addr[4];
                    allocation_base[11]<=command_ffn_base[4];
                    allocation_bytes[11]<=command_ffn_bytes[4];
                    address[12]<=command_ffn_addr[5];
                    allocation_base[12]<=command_ffn_base[5];
                    allocation_bytes[12]<=command_ffn_bytes[5];
                    address[13]<=command_ffn_addr[6];
                    allocation_base[13]<=command_ffn_base[6];
                    allocation_bytes[13]<=command_ffn_bytes[6];
                    address[14]<=command_ffn_addr[7];
                    allocation_base[14]<=command_ffn_base[7];
                    allocation_bytes[14]<=command_ffn_bytes[7];
                    address[15]<=command_final_address;
                    allocation_base[15]<=command_final_base;
                    allocation_bytes[15]<=command_final_bytes;
                    state<=CHECK_SPANS;
                end
                CHECK_SPANS:begin
                    if(fault_now!=0)state<=LOCAL_RESPONSE;
                    else if(!span_valid(address[span_index],allocation_base[span_index],allocation_bytes[span_index],required_span(span_index)))begin fault<=1;state<=LOCAL_RESPONSE;end
                    else if(span_index==15)state<=CHECK_PAIRS;
                    else span_index<=span_index+1'b1;
                end
                CHECK_PAIRS:begin
                    if(fault_now!=0)state<=LOCAL_RESPONSE;
                    else if(pair_overlap)begin fault<=1;state<=LOCAL_RESPONSE;end
                    else if(pair_left==14)state<=ATTENTION_OFFER;
                    else if(pair_right==15)begin pair_left<=pair_left+1'b1;pair_right<=pair_left+2'd2;end
                    else pair_right<=pair_right+1'b1;
                end
                ATTENTION_OFFER:if(a_command_ready)begin attention_launched<=1;state<=WAIT_ATTENTION;end
                WAIT_ATTENTION:if(h_command_valid && h_command_ready)begin
                    handoff_launched<=1;state<=WAIT_HANDOFF;
                    saved_post_read_beats<=a_completion_post_read_beats;saved_residual_beats<=a_completion_residual_beats;saved_ffn_beats<=a_completion_ffn_beats;
                end
                WAIT_HANDOFF:if(h_completion_valid)begin public_status<=fault_now!=0 ? fault_now:h_completion_status;state<=FINAL_RESPONSE;end
                FINAL_RESPONSE:if(completion_valid && completion_ready)state<=WAIT_RELEASE;
                WAIT_RELEASE:if(!h_busy && !a_busy && !a_completion_valid && !f_busy && !f_completion_valid)state<=IDLE;
                LOCAL_RESPONSE:if(completion_ready)state<=IDLE;
                default:begin state<=IDLE;fault<=11;end
            endcase
        end
    end
endmodule
`default_nettype wire
