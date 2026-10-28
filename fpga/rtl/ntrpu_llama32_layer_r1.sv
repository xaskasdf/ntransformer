// Complete single-layer datapath: input/RoPE/attention -> O -> post-O -> FFN/residual.
// One cache authority; the parent retains every allocation through final ACK and drain.
// Four upstream AXI masters plus one retained downstream AXI owner; shared reset quiesces all peers.
`default_nettype none
module ntrpu_llama32_layer_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter integer MATRIX_TIMEOUT_CYCLES=16777216,
    parameter integer HANDOFF_TIMEOUT_CYCLES=67108864,
    parameter integer OWNER_TIMEOUT_CYCLES=268435456
) (
    input wire clk,
    input wire rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,
    input wire [63:0] command_epoch,
    input wire [63:0] command_lease_base,
    input wire [63:0] expected_epoch,
    input wire [31:0] command_layer,
    input wire [31:0] command_position,
    input wire [31:0] command_coefficient_position,
    input wire [63:0] command_addr [0:23],
    input wire [63:0] command_base [0:23],
    input wire [63:0] command_bytes [0:23],
    output wire cache_exclusive,
    input wire [63:0] command_source_sequence,
    input wire [63:0] command_source_epoch,
    input wire command_source_valid,
    output wire norm_tag_exhausted,
    output wire qkv_tags_exhausted,
    output wire [63:0] input_axi_araddr,
    output wire [7:0] input_axi_arlen,
    output wire [2:0] input_axi_arsize,
    output wire [2:0] input_axi_arprot,
    output wire [1:0] input_axi_arburst,
    output wire [3:0] input_axi_arid,
    output wire [3:0] input_axi_arcache,
    output wire input_axi_arlock,
    output wire input_axi_arvalid,
    input wire input_axi_arready,
    input wire [511:0] input_axi_rdata,
    input wire [1:0] input_axi_rresp,
    input wire [3:0] input_axi_rid,
    input wire input_axi_rlast,
    input wire input_axi_rvalid,
    output wire input_axi_rready,
    output wire [63:0] input_axi_awaddr,
    output wire [7:0] input_axi_awlen,
    output wire [2:0] input_axi_awsize,
    output wire [2:0] input_axi_awprot,
    output wire [1:0] input_axi_awburst,
    output wire [3:0] input_axi_awid,
    output wire [3:0] input_axi_awcache,
    output wire input_axi_awlock,
    output wire input_axi_awvalid,
    input wire input_axi_awready,
    output wire [511:0] input_axi_wdata,
    output wire [63:0] input_axi_wstrb,
    output wire input_axi_wlast,
    output wire input_axi_wvalid,
    input wire input_axi_wready,
    input wire [3:0] input_axi_bid,
    input wire [1:0] input_axi_bresp,
    input wire input_axi_bvalid,
    output wire input_axi_bready,
    output wire [63:0] v_axi_araddr,
    output wire [7:0] v_axi_arlen,
    output wire [2:0] v_axi_arsize,
    output wire [2:0] v_axi_arprot,
    output wire [1:0] v_axi_arburst,
    output wire [3:0] v_axi_arid,
    output wire [3:0] v_axi_arcache,
    output wire v_axi_arlock,
    output wire v_axi_arvalid,
    input wire v_axi_arready,
    input wire [511:0] v_axi_rdata,
    input wire [1:0] v_axi_rresp,
    input wire [3:0] v_axi_rid,
    input wire v_axi_rlast,
    input wire v_axi_rvalid,
    output wire v_axi_rready,
    output wire [63:0] rope_axi_araddr,
    output wire [7:0] rope_axi_arlen,
    output wire [2:0] rope_axi_arsize,
    output wire [2:0] rope_axi_arprot,
    output wire [1:0] rope_axi_arburst,
    output wire [3:0] rope_axi_arid,
    output wire [3:0] rope_axi_arcache,
    output wire rope_axi_arlock,
    output wire rope_axi_arvalid,
    input wire rope_axi_arready,
    input wire [511:0] rope_axi_rdata,
    input wire [1:0] rope_axi_rresp,
    input wire [3:0] rope_axi_rid,
    input wire rope_axi_rlast,
    input wire rope_axi_rvalid,
    output wire rope_axi_rready,
    input wire bind_valid,
    output wire bind_ready,
    input wire [63:0] bind_cache_base,
    input wire [63:0] bind_region_base,
    input wire [63:0] bind_region_bytes,
    input wire [31:0] bind_capacity,
    input wire [63:0] bind_sequence,
    input wire [63:0] bind_epoch,
    output wire cache_valid,
    output wire [63:0] cache_base,
    output wire [63:0] cache_region_base,
    output wire [63:0] cache_region_bytes,
    output wire [63:0] cache_sequence,
    output wire [63:0] cache_epoch,
    output wire [31:0] cache_capacity,
    output wire [63:0] kv_axi_awaddr,
    output wire [7:0] kv_axi_awlen,
    output wire [2:0] kv_axi_awsize,
    output wire [2:0] kv_axi_awprot,
    output wire [1:0] kv_axi_awburst,
    output wire [3:0] kv_axi_awid,
    output wire [3:0] kv_axi_awcache,
    output wire kv_axi_awlock,
    output wire kv_axi_awvalid,
    input wire kv_axi_awready,
    output wire [511:0] kv_axi_wdata,
    output wire [63:0] kv_axi_wstrb,
    output wire kv_axi_wlast,
    output wire kv_axi_wvalid,
    input wire kv_axi_wready,
    input wire [3:0] kv_axi_bid,
    input wire [1:0] kv_axi_bresp,
    input wire kv_axi_bvalid,
    output wire kv_axi_bready,
    output wire [63:0] kv_axi_araddr,
    output wire [7:0] kv_axi_arlen,
    output wire [2:0] kv_axi_arsize,
    output wire [2:0] kv_axi_arprot,
    output wire [1:0] kv_axi_arburst,
    output wire [3:0] kv_axi_arid,
    output wire [3:0] kv_axi_arcache,
    output wire kv_axi_arlock,
    output wire kv_axi_arvalid,
    input wire kv_axi_arready,
    input wire [511:0] kv_axi_rdata,
    input wire [1:0] kv_axi_rresp,
    input wire [3:0] kv_axi_rid,
    input wire kv_axi_rlast,
    input wire kv_axi_rvalid,
    output wire kv_axi_rready,
    output wire root_cache_lock,
    output wire cache_metadata_fault,
    output wire [31:0] cache_query_layer,
    output wire cache_query_valid,
    output wire [31:0] cache_query_committed_tokens,
    output wire [31:0] cache_committed_tokens,
    output wire bind_completion_valid,
    input wire bind_completion_ready,
    output wire [3:0] bind_completion_status,
    output wire [63:0] bind_completion_sequence,
    output wire [63:0] bind_completion_epoch,
    input wire cancel_valid,
    input wire completion_ready,
    output wire cancel_ready,
    output wire busy,
    output wire completion_valid,
    output wire completion_revoked,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,
    output logic [63:0] completion_epoch,
    output logic [31:0] completion_layer,
    output logic [31:0] completion_position,
    output logic [2:0] completion_stage,
    output wire result_valid,
    output wire [63:0] result_address,
    output wire [13:0] result_bytes,
    output wire [1:0] tail_selected,
    output wire projection_tags_exhausted,
    output wire ffn_tags_exhausted,
    output logic [3:0] completion_preflight_status,
    output logic [4:0] completion_preflight_region,
    output logic [4:0] completion_preflight_other_region,
    output wire [63:0] tail_axi_araddr,
    output wire [7:0] tail_axi_arlen,
    output wire [2:0] tail_axi_arsize,
    output wire [2:0] tail_axi_arprot,
    output wire [1:0] tail_axi_arburst,
    output wire [3:0] tail_axi_arid,
    output wire [3:0] tail_axi_arcache,
    output wire tail_axi_arlock,
    output wire tail_axi_arvalid,
    input wire tail_axi_arready,
    input wire [511:0] tail_axi_rdata,
    input wire [1:0] tail_axi_rresp,
    input wire [3:0] tail_axi_rid,
    input wire tail_axi_rlast,
    input wire tail_axi_rvalid,
    output wire tail_axi_rready,
    output wire [63:0] tail_axi_awaddr,
    output wire [7:0] tail_axi_awlen,
    output wire [2:0] tail_axi_awsize,
    output wire [2:0] tail_axi_awprot,
    output wire [1:0] tail_axi_awburst,
    output wire [3:0] tail_axi_awid,
    output wire [3:0] tail_axi_awcache,
    output wire tail_axi_awlock,
    output wire tail_axi_awvalid,
    input wire tail_axi_awready,
    output wire [511:0] tail_axi_wdata,
    output wire [63:0] tail_axi_wstrb,
    output wire tail_axi_wlast,
    output wire tail_axi_wvalid,
    input wire tail_axi_wready,
    input wire [3:0] tail_axi_bid,
    input wire [1:0] tail_axi_bresp,
    input wire tail_axi_bvalid,
    output wire tail_axi_bready,
    output logic [3:0] completion_front_status,
    output logic [3:0] completion_front_source_status,
    output logic [3:0] completion_front_child_status,
    output logic completion_front_kv_committed,
    output logic [12:0] completion_front_kv_bytes,
    output logic [31:0] completion_front_kv_tokens,
    output logic completion_front_revoked,
    output logic [3:0] completion_front_norm_status,
    output logic [3:0] completion_front_qkv_status,
    output logic [3:0] completion_front_matrix_status,
    output logic [31:0] completion_front_norm_hls_status,
    output logic [9:0] completion_front_norm_read_beats,
    output logic [6:0] completion_front_norm_records,
    output logic [1:0] completion_front_phase,
    output logic [1:0] completion_front_matrices,
    output logic [14:0] completion_front_total_bytes,
    output logic [3:0] completion_projection_status,
    output logic [31:0] completion_projection_hls_status,
    output logic [3:0] completion_post_status,
    output logic [31:0] completion_post_hls_status,
    output logic [9:0] completion_post_read_beats,
    output logic [7:0] completion_post_residual_beats,
    output logic [6:0] completion_post_ffn_beats,
    output logic [3:0] completion_lease_status,
    output logic [3:0] completion_lease_source_status,
    output logic [3:0] completion_lease_child_status,
    output logic [31:0] completion_lease_hls_status,
    output logic [31:0] completion_lease_ffn_hls_status,
    output logic [3:0] completion_lease_ffn_status,
    output logic [3:0] completion_lease_ffn_child_status,
    output logic [3:0] completion_lease_residual_status,
    output logic [2:0] completion_lease_ffn_stages,
    output logic [1:0] completion_lease_ffn_phase,
    output logic [16:0] completion_lease_ffn_bytes,
    output logic [17:0] completion_lease_ffn_stage_bytes,
    output logic [31:0] completion_lease_source_root_hls_status,
    output logic [31:0] completion_lease_source_post_hls_status,
    output logic [9:0] completion_lease_read_beats,
    output logic [7:0] completion_lease_write_beats
);
    wire gate_command_valid;
    wire gate_command_ready;
    wire [31:0] gate_command_layer;
    wire [31:0] gate_query_layer;
    wire gate_cancel_valid;
    wire gate_cancel_ready;
    wire gate_busy;
    wire gate_completion_valid;
    wire gate_completion_ready;
    wire [3:0] gate_completion_status;
    wire [63:0] held_sequence;
    wire [63:0] held_epoch;
    wire [31:0] held_layer;
    wire [31:0] held_position;
    wire gate_completion_revoked;
    wire gate_result_valid;
    wire [63:0] address [0:23];
    wire [63:0] allocation_base [0:23];
    wire [63:0] allocation_bytes [0:23];
    wire [63:0] held_lease;
    wire [31:0] held_coefficient_position;
    wire [31:0] initial_prefix;
    wire [4:0] gate_completion_region;
    wire [4:0] gate_completion_other_region;
    wire front_command_valid;
    wire front_command_ready;
    wire [63:0] front_command_addr [0:9];
    wire [63:0] front_command_base [0:9];
    wire [63:0] front_command_bytes [0:9];
    wire front_cancel_valid;
    wire front_cancel_ready;
    wire front_busy;
    wire front_cache_exclusive;
    wire front_completion_valid;
    wire front_completion_ready;
    wire [63:0] front_completion_sequence;
    wire [63:0] front_completion_epoch;
    wire [31:0] front_completion_layer;
    wire [31:0] front_completion_position;
    wire [3:0] front_completion_status;
    wire [3:0] front_completion_source_status;
    wire [3:0] front_completion_child_status;
    wire front_completion_kv_committed;
    wire [12:0] front_completion_kv_bytes;
    wire [31:0] front_completion_kv_tokens;
    wire front_completion_revoked;
    wire front_result_valid;
    wire front_result_ce;
    wire [10:0] front_result_address;
    wire [31:0] front_result_q;
    wire front_result_q_valid;
    wire [3:0] front_completion_norm_status;
    wire [3:0] front_completion_qkv_status;
    wire [3:0] front_completion_matrix_status;
    wire [31:0] front_completion_norm_hls_status;
    wire [9:0] front_completion_norm_read_beats;
    wire [6:0] front_completion_norm_records;
    wire [1:0] front_completion_phase;
    wire [1:0] front_completion_matrices;
    wire [14:0] front_completion_total_bytes;
    wire front_bind_valid;
    wire front_bind_ready;
    wire projection_command_valid;
    wire projection_command_ready;
    wire [63:0] projection_command_weight_addr;
    wire [63:0] projection_command_weight_region_base;
    wire [63:0] projection_command_weight_region_bytes;
    wire [63:0] projection_command_activation_addr;
    wire [63:0] projection_command_activation_region_base;
    wire [63:0] projection_command_activation_region_bytes;
    wire [63:0] projection_command_output_addr;
    wire [63:0] projection_command_output_region_base;
    wire [63:0] projection_command_output_region_bytes;
    wire projection_cancel_valid;
    wire projection_cancel_ready;
    wire projection_busy;
    wire projection_completion_valid;
    wire projection_completion_ready;
    wire [3:0] projection_completion_status;
    wire [63:0] projection_completion_sequence;
    wire [63:0] projection_completion_epoch;
    wire projection_result_valid;
    wire [63:0] projection_result_address;
    wire [16:0] projection_result_bytes;
    wire [63:0] projection_m_axi_araddr;
    wire [7:0] projection_m_axi_arlen;
    wire [2:0] projection_m_axi_arsize;
    wire [2:0] projection_m_axi_arprot;
    wire [1:0] projection_m_axi_arburst;
    wire [3:0] projection_m_axi_arid;
    wire [3:0] projection_m_axi_arcache;
    wire projection_m_axi_arlock;
    wire projection_m_axi_arvalid;
    wire projection_m_axi_arready;
    wire [511:0] projection_m_axi_rdata;
    wire [1:0] projection_m_axi_rresp;
    wire [3:0] projection_m_axi_rid;
    wire projection_m_axi_rlast;
    wire projection_m_axi_rvalid;
    wire projection_m_axi_rready;
    wire [63:0] projection_m_axi_awaddr;
    wire [7:0] projection_m_axi_awlen;
    wire [2:0] projection_m_axi_awsize;
    wire [2:0] projection_m_axi_awprot;
    wire [1:0] projection_m_axi_awburst;
    wire [3:0] projection_m_axi_awid;
    wire [3:0] projection_m_axi_awcache;
    wire projection_m_axi_awlock;
    wire projection_m_axi_awvalid;
    wire projection_m_axi_awready;
    wire [511:0] projection_m_axi_wdata;
    wire [63:0] projection_m_axi_wstrb;
    wire projection_m_axi_wlast;
    wire projection_m_axi_wvalid;
    wire projection_m_axi_wready;
    wire [3:0] projection_m_axi_bid;
    wire [1:0] projection_m_axi_bresp;
    wire projection_m_axi_bvalid;
    wire projection_m_axi_bready;
    wire projection_source_completion_valid;
    wire projection_source_completion_ready;
    wire [3:0] projection_source_completion_status;
    wire [63:0] projection_source_sequence;
    wire [63:0] projection_source_epoch;
    wire projection_source_result_valid;
    wire projection_source_result_ce;
    wire [10:0] projection_source_result_address;
    wire [31:0] projection_source_result_q;
    wire projection_source_result_q_valid;
    wire [31:0] projection_completion_hls_status;
    wire post_command_valid;
    wire post_command_ready;
    wire [63:0] post_command_source_sequence;
    wire [63:0] post_command_source_epoch;
    wire post_command_source_valid;
    wire [63:0] post_command_input_address;
    wire [63:0] post_command_input_base;
    wire [63:0] post_command_input_bytes;
    wire [63:0] post_command_branch_address;
    wire [63:0] post_command_branch_base;
    wire [63:0] post_command_branch_bytes;
    wire [63:0] post_command_weight_address;
    wire [63:0] post_command_weight_base;
    wire [63:0] post_command_weight_bytes;
    wire [63:0] post_command_residual_address;
    wire [63:0] post_command_residual_base;
    wire [63:0] post_command_residual_bytes;
    wire [63:0] post_command_ffn_address;
    wire [63:0] post_command_ffn_base;
    wire [63:0] post_command_ffn_bytes;
    wire post_cancel_valid;
    wire post_cancel_ready;
    wire post_busy;
    wire post_completion_valid;
    wire post_completion_ready;
    wire [3:0] post_completion_status;
    wire [31:0] post_completion_hls_status;
    wire [63:0] post_completion_sequence;
    wire [63:0] post_completion_epoch;
    wire [9:0] post_completion_read_beats;
    wire [7:0] post_completion_residual_beats;
    wire [6:0] post_completion_ffn_beats;
    wire post_result_valid;
    wire [63:0] post_result_residual_address;
    wire [63:0] post_result_ffn_address;
    wire [13:0] post_result_residual_bytes;
    wire [12:0] post_result_ffn_bytes;
    wire [63:0] post_m_axi_araddr;
    wire [7:0] post_m_axi_arlen;
    wire [2:0] post_m_axi_arsize;
    wire [2:0] post_m_axi_arprot;
    wire [1:0] post_m_axi_arburst;
    wire [3:0] post_m_axi_arid;
    wire [3:0] post_m_axi_arcache;
    wire post_m_axi_arlock;
    wire post_m_axi_arvalid;
    wire post_m_axi_arready;
    wire [511:0] post_m_axi_rdata;
    wire [1:0] post_m_axi_rresp;
    wire [3:0] post_m_axi_rid;
    wire post_m_axi_rlast;
    wire post_m_axi_rvalid;
    wire post_m_axi_rready;
    wire [63:0] post_m_axi_awaddr;
    wire [7:0] post_m_axi_awlen;
    wire [2:0] post_m_axi_awsize;
    wire [2:0] post_m_axi_awprot;
    wire [1:0] post_m_axi_awburst;
    wire [3:0] post_m_axi_awid;
    wire [3:0] post_m_axi_awcache;
    wire post_m_axi_awlock;
    wire post_m_axi_awvalid;
    wire post_m_axi_awready;
    wire [511:0] post_m_axi_wdata;
    wire [63:0] post_m_axi_wstrb;
    wire post_m_axi_wlast;
    wire post_m_axi_wvalid;
    wire post_m_axi_wready;
    wire [3:0] post_m_axi_bid;
    wire [1:0] post_m_axi_bresp;
    wire post_m_axi_bvalid;
    wire post_m_axi_bready;
    wire lease_command_valid;
    wire lease_command_ready;
    wire [63:0] lease_command_output_address;
    wire [63:0] lease_command_output_base;
    wire [63:0] lease_command_output_bytes;
    wire [63:0] lease_command_ffn_addr [1:7];
    wire [63:0] lease_command_ffn_base [1:7];
    wire [63:0] lease_command_ffn_bytes [1:7];
    wire [63:0] lease_command_protected_base [0:7];
    wire [63:0] lease_command_protected_bytes [0:7];
    wire lease_cancel_valid;
    wire lease_cancel_ready;
    wire lease_busy;
    wire lease_source_completion_valid;
    wire lease_source_completion_ready;
    wire [3:0] lease_source_completion_status;
    wire [63:0] lease_source_sequence;
    wire [63:0] lease_source_epoch;
    wire lease_source_result_valid;
    wire [63:0] lease_source_residual_address;
    wire [63:0] lease_source_ffn_address;
    wire [13:0] lease_source_residual_bytes;
    wire [12:0] lease_source_ffn_bytes;
    wire [31:0] lease_source_root_hls_status;
    wire [31:0] lease_source_post_hls_status;
    wire [9:0] lease_source_post_read_beats;
    wire [7:0] lease_source_residual_beats;
    wire [6:0] lease_source_ffn_beats;
    wire ffn_command_valid;
    wire ffn_command_ready;
    wire [63:0] ffn_command_sequence;
    wire [63:0] ffn_command_epoch;
    wire [63:0] ffn_command_source_sequence;
    wire [63:0] ffn_command_source_epoch;
    wire ffn_command_source_valid;
    wire [63:0] ffn_command_addr [0:7];
    wire [63:0] ffn_command_base [0:7];
    wire [63:0] ffn_command_bytes [0:7];
    wire [63:0] ffn_command_residual_address;
    wire [63:0] ffn_command_residual_base;
    wire [63:0] ffn_command_residual_bytes;
    wire [63:0] ffn_command_output_address;
    wire [63:0] ffn_command_output_base;
    wire [63:0] ffn_command_output_bytes;
    wire ffn_cancel_valid;
    wire ffn_completion_valid;
    wire ffn_completion_ready;
    wire [3:0] ffn_completion_status;
    wire [31:0] ffn_completion_hls_status;
    wire [31:0] ffn_completion_ffn_hls_status;
    wire [3:0] ffn_completion_ffn_status;
    wire [3:0] ffn_completion_ffn_child_status;
    wire [3:0] ffn_completion_residual_status;
    wire [2:0] ffn_completion_ffn_stages;
    wire [1:0] ffn_completion_ffn_phase;
    wire [16:0] ffn_completion_ffn_bytes;
    wire [17:0] ffn_completion_ffn_stage_bytes;
    wire [63:0] ffn_completion_sequence;
    wire [63:0] ffn_completion_epoch;
    wire [9:0] ffn_completion_read_beats;
    wire [7:0] ffn_completion_write_beats;
    wire ffn_result_valid;
    wire [63:0] ffn_result_address;
    wire [13:0] ffn_result_bytes;
    wire lease_ffn_bus_owned;
    wire lease_completion_valid;
    wire lease_completion_ready;
    wire [3:0] lease_completion_status;
    wire [3:0] lease_completion_source_status;
    wire [3:0] lease_completion_child_status;
    wire [31:0] lease_completion_hls_status;
    wire [31:0] lease_completion_ffn_hls_status;
    wire [3:0] lease_completion_ffn_status;
    wire [3:0] lease_completion_ffn_child_status;
    wire [3:0] lease_completion_residual_status;
    wire [2:0] lease_completion_ffn_stages;
    wire [1:0] lease_completion_ffn_phase;
    wire [16:0] lease_completion_ffn_bytes;
    wire [17:0] lease_completion_ffn_stage_bytes;
    wire [31:0] lease_completion_source_root_hls_status;
    wire [31:0] lease_completion_source_post_hls_status;
    wire [63:0] lease_completion_sequence;
    wire [63:0] lease_completion_epoch;
    wire [9:0] lease_completion_read_beats;
    wire [7:0] lease_completion_write_beats;
    wire lease_result_valid;
    wire [63:0] lease_result_address;
    wire [13:0] lease_result_bytes;
    wire ffn_cancel_ready;
    wire ffn_busy;
    wire ffn_activation_selected;
    wire ffn_residual_selected;
    wire [63:0] ffn_m_axi_araddr;
    wire [7:0] ffn_m_axi_arlen;
    wire [2:0] ffn_m_axi_arsize;
    wire [2:0] ffn_m_axi_arprot;
    wire [1:0] ffn_m_axi_arburst;
    wire [3:0] ffn_m_axi_arid;
    wire [3:0] ffn_m_axi_arcache;
    wire ffn_m_axi_arlock;
    wire ffn_m_axi_arvalid;
    wire ffn_m_axi_arready;
    wire [511:0] ffn_m_axi_rdata;
    wire [1:0] ffn_m_axi_rresp;
    wire [3:0] ffn_m_axi_rid;
    wire ffn_m_axi_rlast;
    wire ffn_m_axi_rvalid;
    wire ffn_m_axi_rready;
    wire [63:0] ffn_m_axi_awaddr;
    wire [7:0] ffn_m_axi_awlen;
    wire [2:0] ffn_m_axi_awsize;
    wire [2:0] ffn_m_axi_awprot;
    wire [1:0] ffn_m_axi_awburst;
    wire [3:0] ffn_m_axi_awid;
    wire [3:0] ffn_m_axi_awcache;
    wire ffn_m_axi_awlock;
    wire ffn_m_axi_awvalid;
    wire ffn_m_axi_awready;
    wire [511:0] ffn_m_axi_wdata;
    wire [63:0] ffn_m_axi_wstrb;
    wire ffn_m_axi_wlast;
    wire ffn_m_axi_wvalid;
    wire ffn_m_axi_wready;
    wire [3:0] ffn_m_axi_bid;
    wire [1:0] ffn_m_axi_bresp;
    wire ffn_m_axi_bvalid;
    wire ffn_m_axi_bready;
    wire [139:0] front_snapshot={front_completion_status,front_completion_source_status,front_completion_child_status,front_completion_kv_committed,front_completion_kv_bytes,front_completion_kv_tokens,front_completion_revoked,front_completion_norm_status,front_completion_qkv_status,front_completion_matrix_status,front_completion_norm_hls_status,front_completion_norm_read_beats,front_completion_norm_records,front_completion_phase,front_completion_matrices,front_completion_total_bytes,front_result_valid};
    logic [139:0] saved_front_snapshot;
    wire [117:0] projection_snapshot={projection_completion_status,projection_completion_hls_status,projection_result_valid,projection_result_address,projection_result_bytes};
    logic [117:0] saved_projection_snapshot;
    wire [216:0] post_snapshot={post_completion_status,post_completion_hls_status,post_completion_read_beats,post_completion_residual_beats,post_completion_ffn_beats,post_result_valid,post_result_residual_address,post_result_ffn_address,post_result_residual_bytes,post_result_ffn_bytes};
    logic [216:0] saved_post_snapshot;
    wire [288:0] lease_snapshot={lease_completion_status,lease_completion_source_status,lease_completion_child_status,lease_completion_hls_status,lease_completion_ffn_hls_status,lease_completion_ffn_status,lease_completion_ffn_child_status,lease_completion_residual_status,lease_completion_ffn_stages,lease_completion_ffn_phase,lease_completion_ffn_bytes,lease_completion_ffn_stage_bytes,lease_completion_source_root_hls_status,lease_completion_source_post_hls_status,lease_completion_read_beats,lease_completion_write_beats,lease_result_valid,lease_result_address,lease_result_bytes};
    logic [288:0] saved_lease_snapshot;
    assign front_command_addr[0]=address[0];
    assign front_command_base[0]=allocation_base[0];
    assign front_command_bytes[0]=allocation_bytes[0];
    assign front_command_addr[1]=address[1];
    assign front_command_base[1]=allocation_base[1];
    assign front_command_bytes[1]=allocation_bytes[1];
    assign front_command_addr[2]=address[2];
    assign front_command_base[2]=allocation_base[2];
    assign front_command_bytes[2]=allocation_bytes[2];
    assign front_command_addr[3]=address[3];
    assign front_command_base[3]=allocation_base[3];
    assign front_command_bytes[3]=allocation_bytes[3];
    assign front_command_addr[4]=address[4];
    assign front_command_base[4]=allocation_base[4];
    assign front_command_bytes[4]=allocation_bytes[4];
    assign front_command_addr[5]=address[5];
    assign front_command_base[5]=allocation_base[5];
    assign front_command_bytes[5]=allocation_bytes[5];
    assign front_command_addr[6]=address[6];
    assign front_command_base[6]=allocation_base[6];
    assign front_command_bytes[6]=allocation_bytes[6];
    assign front_command_addr[7]=address[7];
    assign front_command_base[7]=allocation_base[7];
    assign front_command_bytes[7]=allocation_bytes[7];
    assign front_command_addr[8]=address[8];
    assign front_command_base[8]=allocation_base[8];
    assign front_command_bytes[8]=allocation_bytes[8];
    assign front_command_addr[9]=address[9];
    assign front_command_base[9]=allocation_base[9];
    assign front_command_bytes[9]=allocation_bytes[9];
    assign projection_command_weight_addr=address[10];
    assign projection_command_weight_region_base=allocation_base[10];
    assign projection_command_weight_region_bytes=allocation_bytes[10];
    assign projection_command_activation_addr=address[11];
    assign projection_command_activation_region_base=allocation_base[11];
    assign projection_command_activation_region_bytes=allocation_bytes[11];
    assign projection_command_output_addr=address[12];
    assign projection_command_output_region_base=allocation_base[12];
    assign projection_command_output_region_bytes=allocation_bytes[12];
    assign post_command_input_address=address[0];
    assign post_command_input_base=allocation_base[0];
    assign post_command_input_bytes=allocation_bytes[0];
    assign post_command_branch_address=address[12];
    assign post_command_branch_base=allocation_base[12];
    assign post_command_branch_bytes=allocation_bytes[12];
    assign post_command_weight_address=address[13];
    assign post_command_weight_base=allocation_base[13];
    assign post_command_weight_bytes=allocation_bytes[13];
    assign post_command_residual_address=address[14];
    assign post_command_residual_base=allocation_base[14];
    assign post_command_residual_bytes=allocation_bytes[14];
    assign post_command_ffn_address=address[15];
    assign post_command_ffn_base=allocation_base[15];
    assign post_command_ffn_bytes=allocation_bytes[15];
    assign lease_command_output_address=address[23];
    assign lease_command_output_base=allocation_base[23];
    assign lease_command_output_bytes=allocation_bytes[23];
    assign lease_command_ffn_addr[1]=address[16];
    assign lease_command_ffn_base[1]=allocation_base[16];
    assign lease_command_ffn_bytes[1]=allocation_bytes[16];
    assign lease_command_ffn_addr[2]=address[17];
    assign lease_command_ffn_base[2]=allocation_base[17];
    assign lease_command_ffn_bytes[2]=allocation_bytes[17];
    assign lease_command_ffn_addr[3]=address[18];
    assign lease_command_ffn_base[3]=allocation_base[18];
    assign lease_command_ffn_bytes[3]=allocation_bytes[18];
    assign lease_command_ffn_addr[4]=address[19];
    assign lease_command_ffn_base[4]=allocation_base[19];
    assign lease_command_ffn_bytes[4]=allocation_bytes[19];
    assign lease_command_ffn_addr[5]=address[20];
    assign lease_command_ffn_base[5]=allocation_base[20];
    assign lease_command_ffn_bytes[5]=allocation_bytes[20];
    assign lease_command_ffn_addr[6]=address[21];
    assign lease_command_ffn_base[6]=allocation_base[21];
    assign lease_command_ffn_bytes[6]=allocation_bytes[21];
    assign lease_command_ffn_addr[7]=address[22];
    assign lease_command_ffn_base[7]=allocation_base[22];
    assign lease_command_ffn_bytes[7]=allocation_bytes[22];
    assign lease_command_protected_base[0]=saved_cache_region;
    assign lease_command_protected_bytes[0]=saved_cache_bytes;
    assign lease_command_protected_base[1]=allocation_base[10];
    assign lease_command_protected_bytes[1]=allocation_bytes[10];
    assign lease_command_protected_base[2]=allocation_base[11];
    assign lease_command_protected_bytes[2]=allocation_bytes[11];
    assign lease_command_protected_base[3]=allocation_base[12];
    assign lease_command_protected_bytes[3]=allocation_bytes[12];
    assign lease_command_protected_base[4]=allocation_base[0];
    assign lease_command_protected_bytes[4]=allocation_bytes[0];
    assign lease_command_protected_base[5]=allocation_base[13];
    assign lease_command_protected_bytes[5]=allocation_bytes[13];
    assign lease_command_protected_base[6]=allocation_base[14];
    assign lease_command_protected_bytes[6]=allocation_bytes[14];
    assign lease_command_protected_base[7]=allocation_base[15];
    assign lease_command_protected_bytes[7]=allocation_bytes[15];
    assign tail_axi_araddr=tail_owner==2 ? ffn_m_axi_araddr:tail_owner==1 ? post_m_axi_araddr:projection_m_axi_araddr;
    assign tail_axi_arlen=tail_owner==2 ? ffn_m_axi_arlen:tail_owner==1 ? post_m_axi_arlen:projection_m_axi_arlen;
    assign tail_axi_arsize=tail_owner==2 ? ffn_m_axi_arsize:tail_owner==1 ? post_m_axi_arsize:projection_m_axi_arsize;
    assign tail_axi_arprot=tail_owner==2 ? ffn_m_axi_arprot:tail_owner==1 ? post_m_axi_arprot:projection_m_axi_arprot;
    assign tail_axi_arburst=tail_owner==2 ? ffn_m_axi_arburst:tail_owner==1 ? post_m_axi_arburst:projection_m_axi_arburst;
    assign tail_axi_arid=tail_owner==2 ? ffn_m_axi_arid:tail_owner==1 ? post_m_axi_arid:projection_m_axi_arid;
    assign tail_axi_arcache=tail_owner==2 ? ffn_m_axi_arcache:tail_owner==1 ? post_m_axi_arcache:projection_m_axi_arcache;
    assign tail_axi_arlock=tail_owner==2 ? ffn_m_axi_arlock:tail_owner==1 ? post_m_axi_arlock:projection_m_axi_arlock;
    assign tail_axi_arvalid=tail_owner==2 ? ffn_m_axi_arvalid:tail_owner==1 ? post_m_axi_arvalid:projection_m_axi_arvalid;
    assign projection_m_axi_arready=(tail_owner==0 && tail_axi_arready);
    assign post_m_axi_arready=(tail_owner==1 && tail_axi_arready);
    assign ffn_m_axi_arready=(tail_owner==2 && tail_axi_arready);
    assign projection_m_axi_rdata=tail_axi_rdata;
    assign post_m_axi_rdata=tail_axi_rdata;
    assign ffn_m_axi_rdata=tail_axi_rdata;
    assign projection_m_axi_rresp=tail_axi_rresp;
    assign post_m_axi_rresp=tail_axi_rresp;
    assign ffn_m_axi_rresp=tail_axi_rresp;
    assign projection_m_axi_rid=tail_axi_rid;
    assign post_m_axi_rid=tail_axi_rid;
    assign ffn_m_axi_rid=tail_axi_rid;
    assign projection_m_axi_rlast=tail_axi_rlast;
    assign post_m_axi_rlast=tail_axi_rlast;
    assign ffn_m_axi_rlast=tail_axi_rlast;
    assign projection_m_axi_rvalid=(tail_owner==0 && tail_axi_rvalid);
    assign post_m_axi_rvalid=(tail_owner==1 && tail_axi_rvalid);
    assign ffn_m_axi_rvalid=(tail_owner==2 && tail_axi_rvalid);
    assign tail_axi_rready=tail_owner==2 ? ffn_m_axi_rready:tail_owner==1 ? post_m_axi_rready:projection_m_axi_rready;
    assign tail_axi_awaddr=tail_owner==2 ? ffn_m_axi_awaddr:tail_owner==1 ? post_m_axi_awaddr:projection_m_axi_awaddr;
    assign tail_axi_awlen=tail_owner==2 ? ffn_m_axi_awlen:tail_owner==1 ? post_m_axi_awlen:projection_m_axi_awlen;
    assign tail_axi_awsize=tail_owner==2 ? ffn_m_axi_awsize:tail_owner==1 ? post_m_axi_awsize:projection_m_axi_awsize;
    assign tail_axi_awprot=tail_owner==2 ? ffn_m_axi_awprot:tail_owner==1 ? post_m_axi_awprot:projection_m_axi_awprot;
    assign tail_axi_awburst=tail_owner==2 ? ffn_m_axi_awburst:tail_owner==1 ? post_m_axi_awburst:projection_m_axi_awburst;
    assign tail_axi_awid=tail_owner==2 ? ffn_m_axi_awid:tail_owner==1 ? post_m_axi_awid:projection_m_axi_awid;
    assign tail_axi_awcache=tail_owner==2 ? ffn_m_axi_awcache:tail_owner==1 ? post_m_axi_awcache:projection_m_axi_awcache;
    assign tail_axi_awlock=tail_owner==2 ? ffn_m_axi_awlock:tail_owner==1 ? post_m_axi_awlock:projection_m_axi_awlock;
    assign tail_axi_awvalid=tail_owner==2 ? ffn_m_axi_awvalid:tail_owner==1 ? post_m_axi_awvalid:projection_m_axi_awvalid;
    assign projection_m_axi_awready=(tail_owner==0 && tail_axi_awready);
    assign post_m_axi_awready=(tail_owner==1 && tail_axi_awready);
    assign ffn_m_axi_awready=(tail_owner==2 && tail_axi_awready);
    assign tail_axi_wdata=tail_owner==2 ? ffn_m_axi_wdata:tail_owner==1 ? post_m_axi_wdata:projection_m_axi_wdata;
    assign tail_axi_wstrb=tail_owner==2 ? ffn_m_axi_wstrb:tail_owner==1 ? post_m_axi_wstrb:projection_m_axi_wstrb;
    assign tail_axi_wlast=tail_owner==2 ? ffn_m_axi_wlast:tail_owner==1 ? post_m_axi_wlast:projection_m_axi_wlast;
    assign tail_axi_wvalid=tail_owner==2 ? ffn_m_axi_wvalid:tail_owner==1 ? post_m_axi_wvalid:projection_m_axi_wvalid;
    assign projection_m_axi_wready=(tail_owner==0 && tail_axi_wready);
    assign post_m_axi_wready=(tail_owner==1 && tail_axi_wready);
    assign ffn_m_axi_wready=(tail_owner==2 && tail_axi_wready);
    assign projection_m_axi_bid=tail_axi_bid;
    assign post_m_axi_bid=tail_axi_bid;
    assign ffn_m_axi_bid=tail_axi_bid;
    assign projection_m_axi_bresp=tail_axi_bresp;
    assign post_m_axi_bresp=tail_axi_bresp;
    assign ffn_m_axi_bresp=tail_axi_bresp;
    assign projection_m_axi_bvalid=(tail_owner==0 && tail_axi_bvalid);
    assign post_m_axi_bvalid=(tail_owner==1 && tail_axi_bvalid);
    assign ffn_m_axi_bvalid=(tail_owner==2 && tail_axi_bvalid);
    assign tail_axi_bready=tail_owner==2 ? ffn_m_axi_bready:tail_owner==1 ? post_m_axi_bready:projection_m_axi_bready;

    localparam logic [3:0] OK=0,BAD_COMPLETION=5,CANCELLED=7,TIMED_OUT=8,BAD_PEER=11;
    typedef enum logic [4:0] {IDLE,WAIT_GATE,FRONT_OFFER,WAIT_FRONT,O_OFFER,WAIT_O,
        POST_OFFER,WAIT_POST,LEASE_OFFER,WAIT_LEASE,RESPONSE,ACK_LEASE,DRAIN_LEASE,
        ACK_POST,DRAIN_POST,ACK_O,DRAIN_O,ACK_FRONT,DRAIN_FRONT} state_t;
    state_t state;
    logic gate_passed,front_owned,projection_owned,post_owned,lease_owned,front_released;
    logic front_seen,projection_seen,post_seen,lease_seen,revoked;
    logic [3:0] fault,fault_now;
    logic [1:0] tail_owner;
    logic [31:0] watchdog,saved_capacity,prefix_after_front;
    logic [63:0] saved_cache_base,saved_cache_region,saved_cache_bytes,saved_cache_sequence,saved_cache_epoch;
    wire owner_busy=state!=IDLE;
    wire all_idle=!front_busy && !front_completion_valid && !front_cache_exclusive &&
        !projection_busy && !projection_completion_valid && !post_busy && !post_completion_valid &&
        !lease_busy && !lease_completion_valid && !ffn_busy && !ffn_completion_valid && !gate_busy;
    wire admission_allowed=state==IDLE && !bind_valid && !bind_completion_valid && all_idle;
    wire gate_identity=held_sequence==completion_sequence && held_epoch==completion_epoch;
    wire front_identity=front_completion_sequence==completion_sequence && front_completion_epoch==completion_epoch &&
        front_completion_layer==completion_layer && front_completion_position==completion_position;
    wire projection_identity=projection_completion_sequence==completion_sequence && projection_completion_epoch==completion_epoch;
    wire post_identity=post_completion_sequence==completion_sequence && post_completion_epoch==completion_epoch;
    wire lease_identity=lease_completion_sequence==completion_sequence && lease_completion_epoch==completion_epoch;
    wire ffn_identity=ffn_completion_sequence==completion_sequence && ffn_completion_epoch==completion_epoch;
    wire cache_changed=gate_passed && (!cache_valid || cache_metadata_fault || !cache_query_valid ||
        cache_base!=saved_cache_base || cache_region_base!=saved_cache_region || cache_region_bytes!=saved_cache_bytes ||
        cache_sequence!=saved_cache_sequence || cache_epoch!=saved_cache_epoch || cache_capacity!=saved_capacity ||
        cache_query_layer!=completion_layer ||
        (!front_owned && cache_query_committed_tokens!=initial_prefix) ||
        (front_owned && !front_seen && cache_query_committed_tokens!=initial_prefix &&
            cache_query_committed_tokens!=initial_prefix+1) ||
        (front_seen && cache_query_committed_tokens!=prefix_after_front));
    wire front_success=front_completion_status==OK && !front_completion_revoked && front_result_valid &&
        front_completion_source_status==OK && front_completion_child_status==OK &&
        front_completion_norm_status==OK && front_completion_qkv_status==OK && front_completion_matrix_status==OK &&
        front_completion_norm_hls_status==0 && front_completion_norm_read_beats==256 && front_completion_norm_records==64 &&
        front_completion_phase==2 && front_completion_matrices==3 && front_completion_total_bytes==12288 &&
        front_completion_kv_committed && front_completion_kv_bytes==4096 &&
        front_completion_kv_tokens==completion_position+1 && cache_query_committed_tokens==completion_position+1;
    wire projection_success=projection_completion_status==OK && projection_completion_hls_status==0 &&
        projection_result_valid && projection_result_address==address[12] && projection_result_bytes==8192;
    wire post_success=post_completion_status==OK && post_completion_hls_status==0 && post_result_valid &&
        post_completion_read_beats==384 && post_completion_residual_beats==128 && post_completion_ffn_beats==64 &&
        post_result_residual_address==address[14] && post_result_residual_bytes==8192 &&
        post_result_ffn_address==address[15] && post_result_ffn_bytes==4096;
    wire lease_success=lease_completion_status==OK && lease_result_valid &&
        lease_result_address==address[23] && lease_result_bytes==8192;
    wire foreign_peer=(front_owned && !front_released && front_completion_valid && !front_identity) ||
        (projection_owned && projection_completion_valid && !projection_identity) ||
        (post_owned && post_completion_valid && !post_identity) ||
        (lease_owned && lease_completion_valid && !lease_identity) ||
        (lease_ffn_bus_owned && ffn_completion_valid && !ffn_identity);
    wire changed_peer=(front_seen && !front_released &&
            (!front_completion_valid || front_snapshot!=saved_front_snapshot)) ||
        (projection_seen && projection_owned && state!=DRAIN_O && state!=ACK_FRONT && state!=DRAIN_FRONT &&
            (!projection_completion_valid || projection_snapshot!=saved_projection_snapshot)) ||
        (post_seen && post_owned && !lease_owned && state!=DRAIN_POST &&
            (!post_completion_valid || post_snapshot!=saved_post_snapshot)) ||
        (lease_seen && lease_owned && state==RESPONSE &&
            (!lease_completion_valid || lease_snapshot!=saved_lease_snapshot));
    wire waiting=state==FRONT_OFFER || state==WAIT_FRONT || state==O_OFFER || state==WAIT_O ||
        state==POST_OFFER || state==WAIT_POST || state==LEASE_OFFER || state==WAIT_LEASE;
    wire progress=(front_command_valid && front_command_ready) ||
        (state==WAIT_FRONT && front_completion_valid && front_identity) ||
        (projection_command_valid && projection_command_ready) ||
        (state==WAIT_O && projection_completion_valid && projection_identity) ||
        (post_command_valid && post_command_ready) ||
        (state==WAIT_POST && post_completion_valid && post_identity) ||
        (lease_command_valid && lease_command_ready) ||
        (state==WAIT_LEASE && lease_completion_valid && lease_identity);
    wire timed_out=waiting && watchdog>=32'(OWNER_TIMEOUT_CYCLES-1) && !progress;
    wire retiring=state>=ACK_LEASE;
    // Only stored peer faults feed child cancellation. Keep result/cancel feedback out.
    wire cancel_children=owner_busy && (fault!=OK || revoked || cancel_valid || expected_epoch!=completion_epoch);
    wire lease_can_release=post_completion_valid && post_identity &&
        (!lease_ffn_bus_owned || (ffn_completion_valid && ffn_identity));
    assign busy=owner_busy || !all_idle;
    assign cache_exclusive=owner_busy || !all_idle;
    assign command_ready=admission_allowed && gate_command_ready;
    assign gate_command_valid=command_valid && admission_allowed;
    assign gate_command_layer=owner_busy ? held_layer:command_layer;
    assign gate_cancel_valid=state==WAIT_GATE && cancel_children;
    assign gate_completion_ready=state==WAIT_GATE && gate_completion_valid && gate_identity;
    assign front_bind_valid=bind_valid && state==IDLE && all_idle;
    assign bind_ready=front_bind_ready && state==IDLE && all_idle;
    assign cancel_ready=owner_busy && !retiring && fault==OK && !revoked;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked;
    assign result_valid=completion_valid && completion_status==OK && !revoked && fault_now==OK &&
        lease_owned && lease_completion_valid && lease_identity && lease_success;
    assign result_address=result_valid ? address[23]:64'd0;
    assign result_bytes=result_valid ? 14'd8192:14'd0;
    assign front_command_valid=state==FRONT_OFFER && fault_now==OK;
    assign projection_command_valid=state==O_OFFER && fault_now==OK && front_completion_valid && front_identity && front_success;
    assign post_command_valid=state==POST_OFFER && fault_now==OK && projection_completion_valid && projection_identity && projection_success;
    assign lease_command_valid=state==LEASE_OFFER && fault_now==OK && post_completion_valid && post_identity && post_success;
    assign front_cancel_valid=front_owned && !front_released && cancel_children;
    assign projection_cancel_valid=projection_owned && cancel_children;
    assign post_cancel_valid=post_owned && cancel_children;
    assign lease_cancel_valid=lease_owned && cancel_children;
    assign front_completion_ready=front_owned && !front_released && front_completion_valid && front_identity &&
        (projection_owned ? projection_source_completion_ready:state==ACK_FRONT);
    assign projection_completion_ready=state==ACK_O && projection_owned && projection_completion_valid && projection_identity;
    assign post_completion_ready=lease_owned ? lease_source_completion_ready:
        (state==ACK_POST && post_owned && post_completion_valid && post_identity);
    assign lease_completion_ready=state==ACK_LEASE && lease_owned && lease_completion_valid && lease_identity && lease_can_release;
    assign projection_source_completion_valid=front_owned && !front_released && front_completion_valid;
    assign projection_source_completion_status=front_completion_status;
    assign projection_source_sequence=front_completion_sequence;
    assign projection_source_epoch=front_completion_epoch;
    assign projection_source_result_valid=front_result_valid && front_identity && !front_released;
    assign front_result_ce=projection_owned && !front_released && front_identity && projection_source_result_ce;
    assign front_result_address=projection_source_result_address;
    assign projection_source_result_q=front_result_q;
    assign projection_source_result_q_valid=front_result_q_valid && !front_released;
    assign post_command_source_valid=projection_completion_valid && projection_identity && projection_success;
    assign post_command_source_sequence=projection_completion_sequence;
    assign post_command_source_epoch=projection_completion_epoch;
    assign lease_source_completion_valid=post_completion_valid;
    assign lease_source_completion_status=post_completion_status;
    assign lease_source_sequence=post_completion_sequence;
    assign lease_source_epoch=post_completion_epoch;
    assign lease_source_result_valid=post_result_valid;
    assign lease_source_residual_address=post_result_residual_address;
    assign lease_source_ffn_address=post_result_ffn_address;
    assign lease_source_residual_bytes=post_result_residual_bytes;
    assign lease_source_ffn_bytes=post_result_ffn_bytes;
    assign lease_source_root_hls_status=projection_completion_hls_status;
    assign lease_source_post_hls_status=post_completion_hls_status;
    assign lease_source_post_read_beats=post_completion_read_beats;
    assign lease_source_residual_beats=post_completion_residual_beats;
    assign lease_source_ffn_beats=post_completion_ffn_beats;
    assign tail_selected=tail_owner;
    always_comb begin
        fault_now=fault;
        if(owner_busy && fault==OK && !revoked)begin
            if(expected_epoch!=completion_epoch)fault_now=4'd2;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(cache_changed || foreign_peer || changed_peer)fault_now=BAD_PEER;
            else if(timed_out)fault_now=TIMED_OUT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;gate_passed<=0;front_owned<=0;projection_owned<=0;post_owned<=0;lease_owned<=0;
            front_released<=0;front_seen<=0;projection_seen<=0;post_seen<=0;lease_seen<=0;revoked<=0;
            fault<=OK;tail_owner<=0;watchdog<=0;saved_capacity<=0;prefix_after_front<=0;
            saved_cache_base<=0;saved_cache_region<=0;saved_cache_bytes<=0;saved_cache_sequence<=0;saved_cache_epoch<=0;
            completion_sequence<=0;completion_epoch<=0;completion_layer<=0;completion_position<=0;completion_status<=OK;completion_stage<=0;
            saved_front_snapshot<=0;saved_projection_snapshot<=0;saved_post_snapshot<=0;saved_lease_snapshot<=0;
            completion_front_status<=0;
            completion_front_source_status<=0;
            completion_front_child_status<=0;
            completion_front_kv_committed<=0;
            completion_front_kv_bytes<=0;
            completion_front_kv_tokens<=0;
            completion_front_revoked<=0;
            completion_front_norm_status<=0;
            completion_front_qkv_status<=0;
            completion_front_matrix_status<=0;
            completion_front_norm_hls_status<=0;
            completion_front_norm_read_beats<=0;
            completion_front_norm_records<=0;
            completion_front_phase<=0;
            completion_front_matrices<=0;
            completion_front_total_bytes<=0;
            completion_projection_status<=0;
            completion_projection_hls_status<=0;
            completion_post_status<=0;
            completion_post_hls_status<=0;
            completion_post_read_beats<=0;
            completion_post_residual_beats<=0;
            completion_post_ffn_beats<=0;
            completion_lease_status<=0;
            completion_lease_source_status<=0;
            completion_lease_child_status<=0;
            completion_lease_hls_status<=0;
            completion_lease_ffn_hls_status<=0;
            completion_lease_ffn_status<=0;
            completion_lease_ffn_child_status<=0;
            completion_lease_residual_status<=0;
            completion_lease_ffn_stages<=0;
            completion_lease_ffn_phase<=0;
            completion_lease_ffn_bytes<=0;
            completion_lease_ffn_stage_bytes<=0;
            completion_lease_source_root_hls_status<=0;
            completion_lease_source_post_hls_status<=0;
            completion_lease_read_beats<=0;
            completion_lease_write_beats<=0;
            completion_preflight_status<=0;completion_preflight_region<=31;completion_preflight_other_region<=31;
        end else begin
            if(waiting && !progress && fault==OK)watchdog<=watchdog+1;else watchdog<=0;
            if(owner_busy && fault_now!=OK)begin fault<=fault_now;revoked<=1;end
            if(front_completion_valid && front_completion_ready)front_released<=1;
            if(ffn_command_valid && ffn_command_ready)tail_owner<=2;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_layer<=command_layer;completion_position<=command_position;completion_status<=OK;completion_stage<=0;
                    saved_cache_base<=cache_base;saved_cache_region<=cache_region_base;saved_cache_bytes<=cache_region_bytes;
                    saved_cache_sequence<=cache_sequence;saved_cache_epoch<=cache_epoch;saved_capacity<=cache_capacity;
                    prefix_after_front<=cache_query_committed_tokens;gate_passed<=0;
                    front_owned<=0;projection_owned<=0;post_owned<=0;lease_owned<=0;front_released<=0;
                    front_seen<=0;projection_seen<=0;post_seen<=0;lease_seen<=0;revoked<=0;fault<=OK;tail_owner<=0;
                    saved_front_snapshot<=0;saved_projection_snapshot<=0;saved_post_snapshot<=0;saved_lease_snapshot<=0;
            completion_front_status<=0;
            completion_front_source_status<=0;
            completion_front_child_status<=0;
            completion_front_kv_committed<=0;
            completion_front_kv_bytes<=0;
            completion_front_kv_tokens<=0;
            completion_front_revoked<=0;
            completion_front_norm_status<=0;
            completion_front_qkv_status<=0;
            completion_front_matrix_status<=0;
            completion_front_norm_hls_status<=0;
            completion_front_norm_read_beats<=0;
            completion_front_norm_records<=0;
            completion_front_phase<=0;
            completion_front_matrices<=0;
            completion_front_total_bytes<=0;
            completion_projection_status<=0;
            completion_projection_hls_status<=0;
            completion_post_status<=0;
            completion_post_hls_status<=0;
            completion_post_read_beats<=0;
            completion_post_residual_beats<=0;
            completion_post_ffn_beats<=0;
            completion_lease_status<=0;
            completion_lease_source_status<=0;
            completion_lease_child_status<=0;
            completion_lease_hls_status<=0;
            completion_lease_ffn_hls_status<=0;
            completion_lease_ffn_status<=0;
            completion_lease_ffn_child_status<=0;
            completion_lease_residual_status<=0;
            completion_lease_ffn_stages<=0;
            completion_lease_ffn_phase<=0;
            completion_lease_ffn_bytes<=0;
            completion_lease_ffn_stage_bytes<=0;
            completion_lease_source_root_hls_status<=0;
            completion_lease_source_post_hls_status<=0;
            completion_lease_read_beats<=0;
            completion_lease_write_beats<=0;
            completion_preflight_status<=0;completion_preflight_region<=31;completion_preflight_other_region<=31;
                    state<=WAIT_GATE;
                end
                WAIT_GATE:if(gate_completion_ready)begin
                    completion_preflight_status<=gate_completion_status;
                    completion_preflight_region<=gate_completion_region;completion_preflight_other_region<=gate_completion_other_region;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!gate_result_valid)begin completion_status<=gate_completion_status==OK ? BAD_PEER:gate_completion_status;state<=RESPONSE;end
                    else begin gate_passed<=1;state<=FRONT_OFFER;end
                end
                FRONT_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(front_command_ready)begin front_owned<=1;completion_stage<=1;state<=WAIT_FRONT;end
                WAIT_FRONT:if(front_completion_valid && front_identity)begin
                    front_seen<=1;saved_front_snapshot<=front_snapshot;prefix_after_front<=cache_query_committed_tokens;
                    completion_front_status<=front_completion_status;
                    completion_front_source_status<=front_completion_source_status;
                    completion_front_child_status<=front_completion_child_status;
                    completion_front_kv_committed<=front_completion_kv_committed;
                    completion_front_kv_bytes<=front_completion_kv_bytes;
                    completion_front_kv_tokens<=front_completion_kv_tokens;
                    completion_front_revoked<=front_completion_revoked;
                    completion_front_norm_status<=front_completion_norm_status;
                    completion_front_qkv_status<=front_completion_qkv_status;
                    completion_front_matrix_status<=front_completion_matrix_status;
                    completion_front_norm_hls_status<=front_completion_norm_hls_status;
                    completion_front_norm_read_beats<=front_completion_norm_read_beats;
                    completion_front_norm_records<=front_completion_norm_records;
                    completion_front_phase<=front_completion_phase;
                    completion_front_matrices<=front_completion_matrices;
                    completion_front_total_bytes<=front_completion_total_bytes;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(front_completion_status!=OK)begin completion_status<=front_completion_status;state<=RESPONSE;end
                    else if(!front_success)begin completion_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else state<=O_OFFER;
                end
                O_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(projection_command_ready)begin projection_owned<=1;completion_stage<=2;state<=WAIT_O;end
                WAIT_O:if(projection_completion_valid && projection_identity)begin
                    projection_seen<=1;saved_projection_snapshot<=projection_snapshot;
                    completion_projection_status<=projection_completion_status;
                    completion_projection_hls_status<=projection_completion_hls_status;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(projection_completion_status!=OK)begin completion_status<=projection_completion_status;state<=RESPONSE;end
                    else if(!projection_success)begin completion_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else state<=POST_OFFER;
                end
                POST_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(post_command_ready)begin post_owned<=1;tail_owner<=1;completion_stage<=3;state<=WAIT_POST;end
                WAIT_POST:if(post_completion_valid && post_identity)begin
                    post_seen<=1;saved_post_snapshot<=post_snapshot;
                    completion_post_status<=post_completion_status;
                    completion_post_hls_status<=post_completion_hls_status;
                    completion_post_read_beats<=post_completion_read_beats;
                    completion_post_residual_beats<=post_completion_residual_beats;
                    completion_post_ffn_beats<=post_completion_ffn_beats;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(post_completion_status!=OK)begin completion_status<=post_completion_status;state<=RESPONSE;end
                    else if(!post_success)begin completion_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else state<=LEASE_OFFER;
                end
                LEASE_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(lease_command_ready)begin lease_owned<=1;completion_stage<=4;state<=WAIT_LEASE;end
                WAIT_LEASE:if(lease_completion_valid && lease_identity)begin
                    lease_seen<=1;saved_lease_snapshot<=lease_snapshot;
                    completion_lease_status<=lease_completion_status;
                    completion_lease_source_status<=lease_completion_source_status;
                    completion_lease_child_status<=lease_completion_child_status;
                    completion_lease_hls_status<=lease_completion_hls_status;
                    completion_lease_ffn_hls_status<=lease_completion_ffn_hls_status;
                    completion_lease_ffn_status<=lease_completion_ffn_status;
                    completion_lease_ffn_child_status<=lease_completion_ffn_child_status;
                    completion_lease_residual_status<=lease_completion_residual_status;
                    completion_lease_ffn_stages<=lease_completion_ffn_stages;
                    completion_lease_ffn_phase<=lease_completion_ffn_phase;
                    completion_lease_ffn_bytes<=lease_completion_ffn_bytes;
                    completion_lease_ffn_stage_bytes<=lease_completion_ffn_stage_bytes;
                    completion_lease_source_root_hls_status<=lease_completion_source_root_hls_status;
                    completion_lease_source_post_hls_status<=lease_completion_source_post_hls_status;
                    completion_lease_read_beats<=lease_completion_read_beats;
                    completion_lease_write_beats<=lease_completion_write_beats;
                    completion_status<=fault_now!=OK ? fault_now:lease_completion_status!=OK ? lease_completion_status:
                        lease_success ? OK:BAD_COMPLETION;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=lease_owned ? ACK_LEASE:ACK_POST;
                ACK_LEASE:if(lease_completion_ready)state<=DRAIN_LEASE;
                DRAIN_LEASE:if(!lease_busy && !lease_completion_valid && !ffn_busy && !ffn_completion_valid && !post_busy && !post_completion_valid)begin
                    lease_owned<=0;post_owned<=0;tail_owner<=0;state<=ACK_O;
                end
                ACK_POST:if(!post_owned || post_completion_ready)state<=DRAIN_POST;
                DRAIN_POST:if(!post_busy && !post_completion_valid)begin post_owned<=0;tail_owner<=0;state<=ACK_O;end
                ACK_O:if(!projection_owned || projection_completion_ready)state<=DRAIN_O;
                DRAIN_O:if(!projection_busy && !projection_completion_valid)begin projection_owned<=0;state<=ACK_FRONT;end
                ACK_FRONT:if(!front_owned || front_released || front_completion_ready)state<=DRAIN_FRONT;
                DRAIN_FRONT:if(!front_busy && !front_completion_valid && !front_cache_exclusive && !gate_busy &&
                    !projection_busy && !projection_completion_valid && !post_busy && !post_completion_valid &&
                    !lease_busy && !lease_completion_valid && !ffn_busy && !ffn_completion_valid)begin
                    front_owned<=0;gate_passed<=0;state<=IDLE;
                end
                default:begin fault<=BAD_PEER;revoked<=1;completion_status<=BAD_PEER;state<=RESPONSE;end
            endcase
        end
    end

    ntrpu_llama32_layer_preflight_r1 gate(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(gate_command_valid),
        .command_ready(gate_command_ready),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .expected_epoch(expected_epoch),
        .command_source_valid(command_source_valid),
        .command_source_sequence(command_source_sequence),
        .command_source_epoch(command_source_epoch),
        .command_lease_base(command_lease_base),
        .command_layer(gate_command_layer),
        .command_position(command_position),
        .command_coefficient_position(command_coefficient_position),
        .command_addr(command_addr),
        .command_base(command_base),
        .command_bytes(command_bytes),
        .cache_valid(cache_valid),
        .cache_metadata_fault(cache_metadata_fault),
        .cache_base(cache_base),
        .cache_region_base(cache_region_base),
        .cache_region_bytes(cache_region_bytes),
        .cache_sequence(cache_sequence),
        .cache_epoch(cache_epoch),
        .cache_capacity(cache_capacity),
        .cache_query_layer(gate_query_layer),
        .cache_query_valid(cache_query_valid),
        .cache_query_committed_tokens(cache_query_committed_tokens),
        .cancel_valid(gate_cancel_valid),
        .cancel_ready(gate_cancel_ready),
        .busy(gate_busy),
        .completion_valid(gate_completion_valid),
        .completion_ready(gate_completion_ready),
        .completion_status(gate_completion_status),
        .completion_sequence(held_sequence),
        .completion_epoch(held_epoch),
        .completion_layer(held_layer),
        .completion_position(held_position),
        .completion_revoked(gate_completion_revoked),
        .result_valid(gate_result_valid),
        .validated_addr(address),
        .validated_base(allocation_base),
        .validated_bytes(allocation_bytes),
        .validated_lease_base(held_lease),
        .validated_coefficient_position(held_coefficient_position),
        .validated_prefix(initial_prefix),
        .completion_region(gate_completion_region),
        .completion_other_region(gate_completion_other_region)
    );
    ntrpu_input_attention_r1 #(.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES),.MATRIX_TIMEOUT_CYCLES(MATRIX_TIMEOUT_CYCLES),.HANDOFF_TIMEOUT_CYCLES(HANDOFF_TIMEOUT_CYCLES),.OWNER_TIMEOUT_CYCLES(OWNER_TIMEOUT_CYCLES)) front(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(front_command_valid),
        .command_ready(front_command_ready),
        .command_sequence(completion_sequence),
        .command_epoch(completion_epoch),
        .command_lease_base(held_lease),
        .expected_epoch(expected_epoch),
        .command_layer(gate_query_layer),
        .command_position(held_position),
        .command_coefficient_position(held_coefficient_position),
        .command_addr(front_command_addr),
        .command_base(front_command_base),
        .command_bytes(front_command_bytes),
        .cancel_valid(front_cancel_valid),
        .cancel_ready(front_cancel_ready),
        .busy(front_busy),
        .cache_exclusive(front_cache_exclusive),
        .completion_valid(front_completion_valid),
        .completion_ready(front_completion_ready),
        .completion_sequence(front_completion_sequence),
        .completion_epoch(front_completion_epoch),
        .completion_layer(front_completion_layer),
        .completion_position(front_completion_position),
        .completion_status(front_completion_status),
        .completion_source_status(front_completion_source_status),
        .completion_child_status(front_completion_child_status),
        .completion_kv_committed(front_completion_kv_committed),
        .completion_kv_bytes(front_completion_kv_bytes),
        .completion_kv_tokens(front_completion_kv_tokens),
        .completion_revoked(front_completion_revoked),
        .result_valid(front_result_valid),
        .result_ce(front_result_ce),
        .result_address(front_result_address),
        .result_q(front_result_q),
        .result_q_valid(front_result_q_valid),
        .command_source_sequence(completion_sequence),
        .command_source_epoch(completion_epoch),
        .command_source_valid(1'b1),
        .completion_norm_status(front_completion_norm_status),
        .completion_qkv_status(front_completion_qkv_status),
        .completion_matrix_status(front_completion_matrix_status),
        .completion_norm_hls_status(front_completion_norm_hls_status),
        .completion_norm_read_beats(front_completion_norm_read_beats),
        .completion_norm_records(front_completion_norm_records),
        .completion_phase(front_completion_phase),
        .completion_matrices(front_completion_matrices),
        .completion_total_bytes(front_completion_total_bytes),
        .norm_tag_exhausted(norm_tag_exhausted),
        .qkv_tags_exhausted(qkv_tags_exhausted),
        .input_axi_araddr(input_axi_araddr),
        .input_axi_arlen(input_axi_arlen),
        .input_axi_arsize(input_axi_arsize),
        .input_axi_arprot(input_axi_arprot),
        .input_axi_arburst(input_axi_arburst),
        .input_axi_arid(input_axi_arid),
        .input_axi_arcache(input_axi_arcache),
        .input_axi_arlock(input_axi_arlock),
        .input_axi_arvalid(input_axi_arvalid),
        .input_axi_arready(input_axi_arready),
        .input_axi_rdata(input_axi_rdata),
        .input_axi_rresp(input_axi_rresp),
        .input_axi_rid(input_axi_rid),
        .input_axi_rlast(input_axi_rlast),
        .input_axi_rvalid(input_axi_rvalid),
        .input_axi_rready(input_axi_rready),
        .input_axi_awaddr(input_axi_awaddr),
        .input_axi_awlen(input_axi_awlen),
        .input_axi_awsize(input_axi_awsize),
        .input_axi_awprot(input_axi_awprot),
        .input_axi_awburst(input_axi_awburst),
        .input_axi_awid(input_axi_awid),
        .input_axi_awcache(input_axi_awcache),
        .input_axi_awlock(input_axi_awlock),
        .input_axi_awvalid(input_axi_awvalid),
        .input_axi_awready(input_axi_awready),
        .input_axi_wdata(input_axi_wdata),
        .input_axi_wstrb(input_axi_wstrb),
        .input_axi_wlast(input_axi_wlast),
        .input_axi_wvalid(input_axi_wvalid),
        .input_axi_wready(input_axi_wready),
        .input_axi_bid(input_axi_bid),
        .input_axi_bresp(input_axi_bresp),
        .input_axi_bvalid(input_axi_bvalid),
        .input_axi_bready(input_axi_bready),
        .v_axi_araddr(v_axi_araddr),
        .v_axi_arlen(v_axi_arlen),
        .v_axi_arsize(v_axi_arsize),
        .v_axi_arprot(v_axi_arprot),
        .v_axi_arburst(v_axi_arburst),
        .v_axi_arid(v_axi_arid),
        .v_axi_arcache(v_axi_arcache),
        .v_axi_arlock(v_axi_arlock),
        .v_axi_arvalid(v_axi_arvalid),
        .v_axi_arready(v_axi_arready),
        .v_axi_rdata(v_axi_rdata),
        .v_axi_rresp(v_axi_rresp),
        .v_axi_rid(v_axi_rid),
        .v_axi_rlast(v_axi_rlast),
        .v_axi_rvalid(v_axi_rvalid),
        .v_axi_rready(v_axi_rready),
        .rope_axi_araddr(rope_axi_araddr),
        .rope_axi_arlen(rope_axi_arlen),
        .rope_axi_arsize(rope_axi_arsize),
        .rope_axi_arprot(rope_axi_arprot),
        .rope_axi_arburst(rope_axi_arburst),
        .rope_axi_arid(rope_axi_arid),
        .rope_axi_arcache(rope_axi_arcache),
        .rope_axi_arlock(rope_axi_arlock),
        .rope_axi_arvalid(rope_axi_arvalid),
        .rope_axi_arready(rope_axi_arready),
        .rope_axi_rdata(rope_axi_rdata),
        .rope_axi_rresp(rope_axi_rresp),
        .rope_axi_rid(rope_axi_rid),
        .rope_axi_rlast(rope_axi_rlast),
        .rope_axi_rvalid(rope_axi_rvalid),
        .rope_axi_rready(rope_axi_rready),
        .bind_valid(front_bind_valid),
        .bind_ready(front_bind_ready),
        .bind_cache_base(bind_cache_base),
        .bind_region_base(bind_region_base),
        .bind_region_bytes(bind_region_bytes),
        .bind_capacity(bind_capacity),
        .bind_sequence(bind_sequence),
        .bind_epoch(bind_epoch),
        .cache_valid(cache_valid),
        .cache_base(cache_base),
        .cache_region_base(cache_region_base),
        .cache_region_bytes(cache_region_bytes),
        .cache_sequence(cache_sequence),
        .cache_epoch(cache_epoch),
        .cache_capacity(cache_capacity),
        .kv_axi_awaddr(kv_axi_awaddr),
        .kv_axi_awlen(kv_axi_awlen),
        .kv_axi_awsize(kv_axi_awsize),
        .kv_axi_awprot(kv_axi_awprot),
        .kv_axi_awburst(kv_axi_awburst),
        .kv_axi_awid(kv_axi_awid),
        .kv_axi_awcache(kv_axi_awcache),
        .kv_axi_awlock(kv_axi_awlock),
        .kv_axi_awvalid(kv_axi_awvalid),
        .kv_axi_awready(kv_axi_awready),
        .kv_axi_wdata(kv_axi_wdata),
        .kv_axi_wstrb(kv_axi_wstrb),
        .kv_axi_wlast(kv_axi_wlast),
        .kv_axi_wvalid(kv_axi_wvalid),
        .kv_axi_wready(kv_axi_wready),
        .kv_axi_bid(kv_axi_bid),
        .kv_axi_bresp(kv_axi_bresp),
        .kv_axi_bvalid(kv_axi_bvalid),
        .kv_axi_bready(kv_axi_bready),
        .kv_axi_araddr(kv_axi_araddr),
        .kv_axi_arlen(kv_axi_arlen),
        .kv_axi_arsize(kv_axi_arsize),
        .kv_axi_arprot(kv_axi_arprot),
        .kv_axi_arburst(kv_axi_arburst),
        .kv_axi_arid(kv_axi_arid),
        .kv_axi_arcache(kv_axi_arcache),
        .kv_axi_arlock(kv_axi_arlock),
        .kv_axi_arvalid(kv_axi_arvalid),
        .kv_axi_arready(kv_axi_arready),
        .kv_axi_rdata(kv_axi_rdata),
        .kv_axi_rresp(kv_axi_rresp),
        .kv_axi_rid(kv_axi_rid),
        .kv_axi_rlast(kv_axi_rlast),
        .kv_axi_rvalid(kv_axi_rvalid),
        .kv_axi_rready(kv_axi_rready),
        .root_cache_lock(root_cache_lock),
        .cache_metadata_fault(cache_metadata_fault),
        .cache_query_layer(cache_query_layer),
        .cache_query_valid(cache_query_valid),
        .cache_query_committed_tokens(cache_query_committed_tokens),
        .cache_committed_tokens(cache_committed_tokens),
        .bind_completion_valid(bind_completion_valid),
        .bind_completion_ready(bind_completion_ready),
        .bind_completion_status(bind_completion_status),
        .bind_completion_sequence(bind_completion_sequence),
        .bind_completion_epoch(bind_completion_epoch)
    );
    ntrpu_attention_output_path_r1 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES)) projection(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(projection_command_valid),
        .command_ready(projection_command_ready),
        .command_sequence(completion_sequence),
        .command_epoch(completion_epoch),
        .expected_epoch(expected_epoch),
        .command_weight_addr(projection_command_weight_addr),
        .command_weight_region_base(projection_command_weight_region_base),
        .command_weight_region_bytes(projection_command_weight_region_bytes),
        .command_activation_addr(projection_command_activation_addr),
        .command_activation_region_base(projection_command_activation_region_base),
        .command_activation_region_bytes(projection_command_activation_region_bytes),
        .command_output_addr(projection_command_output_addr),
        .command_output_region_base(projection_command_output_region_base),
        .command_output_region_bytes(projection_command_output_region_bytes),
        .cancel_valid(projection_cancel_valid),
        .cancel_ready(projection_cancel_ready),
        .busy(projection_busy),
        .completion_valid(projection_completion_valid),
        .completion_ready(projection_completion_ready),
        .completion_status(projection_completion_status),
        .completion_sequence(projection_completion_sequence),
        .completion_epoch(projection_completion_epoch),
        .result_valid(projection_result_valid),
        .result_address(projection_result_address),
        .result_bytes(projection_result_bytes),
        .tag_exhausted(projection_tags_exhausted),
        .m_axi_araddr(projection_m_axi_araddr),
        .m_axi_arlen(projection_m_axi_arlen),
        .m_axi_arsize(projection_m_axi_arsize),
        .m_axi_arprot(projection_m_axi_arprot),
        .m_axi_arburst(projection_m_axi_arburst),
        .m_axi_arid(projection_m_axi_arid),
        .m_axi_arcache(projection_m_axi_arcache),
        .m_axi_arlock(projection_m_axi_arlock),
        .m_axi_arvalid(projection_m_axi_arvalid),
        .m_axi_arready(projection_m_axi_arready),
        .m_axi_rdata(projection_m_axi_rdata),
        .m_axi_rresp(projection_m_axi_rresp),
        .m_axi_rid(projection_m_axi_rid),
        .m_axi_rlast(projection_m_axi_rlast),
        .m_axi_rvalid(projection_m_axi_rvalid),
        .m_axi_rready(projection_m_axi_rready),
        .m_axi_awaddr(projection_m_axi_awaddr),
        .m_axi_awlen(projection_m_axi_awlen),
        .m_axi_awsize(projection_m_axi_awsize),
        .m_axi_awprot(projection_m_axi_awprot),
        .m_axi_awburst(projection_m_axi_awburst),
        .m_axi_awid(projection_m_axi_awid),
        .m_axi_awcache(projection_m_axi_awcache),
        .m_axi_awlock(projection_m_axi_awlock),
        .m_axi_awvalid(projection_m_axi_awvalid),
        .m_axi_awready(projection_m_axi_awready),
        .m_axi_wdata(projection_m_axi_wdata),
        .m_axi_wstrb(projection_m_axi_wstrb),
        .m_axi_wlast(projection_m_axi_wlast),
        .m_axi_wvalid(projection_m_axi_wvalid),
        .m_axi_wready(projection_m_axi_wready),
        .m_axi_bid(projection_m_axi_bid),
        .m_axi_bresp(projection_m_axi_bresp),
        .m_axi_bvalid(projection_m_axi_bvalid),
        .m_axi_bready(projection_m_axi_bready),
        .source_completion_valid(projection_source_completion_valid),
        .source_completion_ready(projection_source_completion_ready),
        .source_completion_status(projection_source_completion_status),
        .source_sequence(projection_source_sequence),
        .source_epoch(projection_source_epoch),
        .source_result_valid(projection_source_result_valid),
        .source_result_ce(projection_source_result_ce),
        .source_result_address(projection_source_result_address),
        .source_result_q(projection_source_result_q),
        .source_result_q_valid(projection_source_result_q_valid),
        .completion_hls_status(projection_completion_hls_status)
    );
    ntrpu_post_attention_real_r2 #(.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) post(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(post_command_valid),
        .command_ready(post_command_ready),
        .command_sequence(completion_sequence),
        .command_epoch(completion_epoch),
        .expected_epoch(expected_epoch),
        .command_source_sequence(post_command_source_sequence),
        .command_source_epoch(post_command_source_epoch),
        .command_source_valid(post_command_source_valid),
        .command_input_address(post_command_input_address),
        .command_input_base(post_command_input_base),
        .command_input_bytes(post_command_input_bytes),
        .command_branch_address(post_command_branch_address),
        .command_branch_base(post_command_branch_base),
        .command_branch_bytes(post_command_branch_bytes),
        .command_weight_address(post_command_weight_address),
        .command_weight_base(post_command_weight_base),
        .command_weight_bytes(post_command_weight_bytes),
        .command_residual_address(post_command_residual_address),
        .command_residual_base(post_command_residual_base),
        .command_residual_bytes(post_command_residual_bytes),
        .command_ffn_address(post_command_ffn_address),
        .command_ffn_base(post_command_ffn_base),
        .command_ffn_bytes(post_command_ffn_bytes),
        .cancel_valid(post_cancel_valid),
        .cancel_ready(post_cancel_ready),
        .busy(post_busy),
        .completion_valid(post_completion_valid),
        .completion_ready(post_completion_ready),
        .completion_status(post_completion_status),
        .completion_hls_status(post_completion_hls_status),
        .completion_sequence(post_completion_sequence),
        .completion_epoch(post_completion_epoch),
        .completion_read_beats(post_completion_read_beats),
        .completion_residual_beats(post_completion_residual_beats),
        .completion_ffn_beats(post_completion_ffn_beats),
        .result_valid(post_result_valid),
        .result_residual_address(post_result_residual_address),
        .result_ffn_address(post_result_ffn_address),
        .result_residual_bytes(post_result_residual_bytes),
        .result_ffn_bytes(post_result_ffn_bytes),
        .m_axi_araddr(post_m_axi_araddr),
        .m_axi_arlen(post_m_axi_arlen),
        .m_axi_arsize(post_m_axi_arsize),
        .m_axi_arprot(post_m_axi_arprot),
        .m_axi_arburst(post_m_axi_arburst),
        .m_axi_arid(post_m_axi_arid),
        .m_axi_arcache(post_m_axi_arcache),
        .m_axi_arlock(post_m_axi_arlock),
        .m_axi_arvalid(post_m_axi_arvalid),
        .m_axi_arready(post_m_axi_arready),
        .m_axi_rdata(post_m_axi_rdata),
        .m_axi_rresp(post_m_axi_rresp),
        .m_axi_rid(post_m_axi_rid),
        .m_axi_rlast(post_m_axi_rlast),
        .m_axi_rvalid(post_m_axi_rvalid),
        .m_axi_rready(post_m_axi_rready),
        .m_axi_awaddr(post_m_axi_awaddr),
        .m_axi_awlen(post_m_axi_awlen),
        .m_axi_awsize(post_m_axi_awsize),
        .m_axi_awprot(post_m_axi_awprot),
        .m_axi_awburst(post_m_axi_awburst),
        .m_axi_awid(post_m_axi_awid),
        .m_axi_awcache(post_m_axi_awcache),
        .m_axi_awlock(post_m_axi_awlock),
        .m_axi_awvalid(post_m_axi_awvalid),
        .m_axi_awready(post_m_axi_awready),
        .m_axi_wdata(post_m_axi_wdata),
        .m_axi_wstrb(post_m_axi_wstrb),
        .m_axi_wlast(post_m_axi_wlast),
        .m_axi_wvalid(post_m_axi_wvalid),
        .m_axi_wready(post_m_axi_wready),
        .m_axi_bid(post_m_axi_bid),
        .m_axi_bresp(post_m_axi_bresp),
        .m_axi_bvalid(post_m_axi_bvalid),
        .m_axi_bready(post_m_axi_bready)
    );
    ntrpu_post_ffn_handoff_r2 lease(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(lease_command_valid),
        .command_ready(lease_command_ready),
        .command_sequence(completion_sequence),
        .command_epoch(completion_epoch),
        .expected_epoch(expected_epoch),
        .command_output_address(lease_command_output_address),
        .command_output_base(lease_command_output_base),
        .command_output_bytes(lease_command_output_bytes),
        .command_ffn_addr(lease_command_ffn_addr),
        .command_ffn_base(lease_command_ffn_base),
        .command_ffn_bytes(lease_command_ffn_bytes),
        .command_protected_base(lease_command_protected_base),
        .command_protected_bytes(lease_command_protected_bytes),
        .cancel_valid(lease_cancel_valid),
        .cancel_ready(lease_cancel_ready),
        .busy(lease_busy),
        .source_completion_valid(lease_source_completion_valid),
        .source_completion_ready(lease_source_completion_ready),
        .source_completion_status(lease_source_completion_status),
        .source_sequence(lease_source_sequence),
        .source_epoch(lease_source_epoch),
        .source_result_valid(lease_source_result_valid),
        .source_residual_address(lease_source_residual_address),
        .source_ffn_address(lease_source_ffn_address),
        .source_residual_bytes(lease_source_residual_bytes),
        .source_ffn_bytes(lease_source_ffn_bytes),
        .source_root_hls_status(lease_source_root_hls_status),
        .source_post_hls_status(lease_source_post_hls_status),
        .source_post_read_beats(lease_source_post_read_beats),
        .source_residual_beats(lease_source_residual_beats),
        .source_ffn_beats(lease_source_ffn_beats),
        .child_command_valid(ffn_command_valid),
        .child_command_ready(ffn_command_ready),
        .child_command_sequence(ffn_command_sequence),
        .child_command_epoch(ffn_command_epoch),
        .child_source_sequence(ffn_command_source_sequence),
        .child_source_epoch(ffn_command_source_epoch),
        .child_source_valid(ffn_command_source_valid),
        .child_command_addr(ffn_command_addr),
        .child_command_base(ffn_command_base),
        .child_command_bytes(ffn_command_bytes),
        .child_residual_address(ffn_command_residual_address),
        .child_residual_base(ffn_command_residual_base),
        .child_residual_bytes(ffn_command_residual_bytes),
        .child_output_address(ffn_command_output_address),
        .child_output_base(ffn_command_output_base),
        .child_output_bytes(ffn_command_output_bytes),
        .child_cancel_valid(ffn_cancel_valid),
        .child_completion_valid(ffn_completion_valid),
        .child_completion_ready(ffn_completion_ready),
        .child_completion_status(ffn_completion_status),
        .child_hls_status(ffn_completion_hls_status),
        .child_ffn_hls_status(ffn_completion_ffn_hls_status),
        .child_ffn_status(ffn_completion_ffn_status),
        .child_ffn_child_status(ffn_completion_ffn_child_status),
        .child_residual_status(ffn_completion_residual_status),
        .child_ffn_stages(ffn_completion_ffn_stages),
        .child_ffn_phase(ffn_completion_ffn_phase),
        .child_ffn_bytes(ffn_completion_ffn_bytes),
        .child_ffn_stage_bytes(ffn_completion_ffn_stage_bytes),
        .child_sequence(ffn_completion_sequence),
        .child_epoch(ffn_completion_epoch),
        .child_read_beats(ffn_completion_read_beats),
        .child_write_beats(ffn_completion_write_beats),
        .child_result_valid(ffn_result_valid),
        .child_result_address(ffn_result_address),
        .child_result_bytes(ffn_result_bytes),
        .ffn_bus_owned(lease_ffn_bus_owned),
        .completion_valid(lease_completion_valid),
        .completion_ready(lease_completion_ready),
        .completion_status(lease_completion_status),
        .completion_source_status(lease_completion_source_status),
        .completion_child_status(lease_completion_child_status),
        .completion_hls_status(lease_completion_hls_status),
        .completion_ffn_hls_status(lease_completion_ffn_hls_status),
        .completion_ffn_status(lease_completion_ffn_status),
        .completion_ffn_child_status(lease_completion_ffn_child_status),
        .completion_residual_status(lease_completion_residual_status),
        .completion_ffn_stages(lease_completion_ffn_stages),
        .completion_ffn_phase(lease_completion_ffn_phase),
        .completion_ffn_bytes(lease_completion_ffn_bytes),
        .completion_ffn_stage_bytes(lease_completion_ffn_stage_bytes),
        .completion_source_root_hls_status(lease_completion_source_root_hls_status),
        .completion_source_post_hls_status(lease_completion_source_post_hls_status),
        .completion_sequence(lease_completion_sequence),
        .completion_epoch(lease_completion_epoch),
        .completion_read_beats(lease_completion_read_beats),
        .completion_write_beats(lease_completion_write_beats),
        .result_valid(lease_result_valid),
        .result_address(lease_result_address),
        .result_bytes(lease_result_bytes)
    );
    ntrpu_llama32_ffn_residual_r3 #(.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) ffn(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(ffn_command_valid),
        .command_sequence(ffn_command_sequence),
        .command_epoch(ffn_command_epoch),
        .command_source_valid(ffn_command_source_valid),
        .command_source_sequence(ffn_command_source_sequence),
        .command_source_epoch(ffn_command_source_epoch),
        .command_addr(ffn_command_addr),
        .command_base(ffn_command_base),
        .command_bytes(ffn_command_bytes),
        .command_residual_address(ffn_command_residual_address),
        .command_residual_base(ffn_command_residual_base),
        .command_residual_bytes(ffn_command_residual_bytes),
        .expected_epoch(expected_epoch),
        .command_output_address(ffn_command_output_address),
        .command_output_base(ffn_command_output_base),
        .command_output_bytes(ffn_command_output_bytes),
        .command_ready(ffn_command_ready),
        .cancel_valid(ffn_cancel_valid),
        .cancel_ready(ffn_cancel_ready),
        .busy(ffn_busy),
        .completion_valid(ffn_completion_valid),
        .completion_ready(ffn_completion_ready),
        .completion_status(ffn_completion_status),
        .completion_ffn_status(ffn_completion_ffn_status),
        .completion_ffn_child_status(ffn_completion_ffn_child_status),
        .completion_residual_status(ffn_completion_residual_status),
        .completion_ffn_stages(ffn_completion_ffn_stages),
        .completion_ffn_phase(ffn_completion_ffn_phase),
        .completion_ffn_bytes(ffn_completion_ffn_bytes),
        .completion_ffn_stage_bytes(ffn_completion_ffn_stage_bytes),
        .completion_ffn_hls_status(ffn_completion_ffn_hls_status),
        .completion_hls_status(ffn_completion_hls_status),
        .completion_read_beats(ffn_completion_read_beats),
        .completion_write_beats(ffn_completion_write_beats),
        .completion_sequence(ffn_completion_sequence),
        .completion_epoch(ffn_completion_epoch),
        .result_valid(ffn_result_valid),
        .result_address(ffn_result_address),
        .result_bytes(ffn_result_bytes),
        .tags_exhausted(ffn_tags_exhausted),
        .activation_selected(ffn_activation_selected),
        .residual_selected(ffn_residual_selected),
        .m_axi_araddr(ffn_m_axi_araddr),
        .m_axi_arlen(ffn_m_axi_arlen),
        .m_axi_arsize(ffn_m_axi_arsize),
        .m_axi_arprot(ffn_m_axi_arprot),
        .m_axi_arburst(ffn_m_axi_arburst),
        .m_axi_arid(ffn_m_axi_arid),
        .m_axi_arcache(ffn_m_axi_arcache),
        .m_axi_arlock(ffn_m_axi_arlock),
        .m_axi_arvalid(ffn_m_axi_arvalid),
        .m_axi_arready(ffn_m_axi_arready),
        .m_axi_rdata(ffn_m_axi_rdata),
        .m_axi_rresp(ffn_m_axi_rresp),
        .m_axi_rid(ffn_m_axi_rid),
        .m_axi_rlast(ffn_m_axi_rlast),
        .m_axi_rvalid(ffn_m_axi_rvalid),
        .m_axi_rready(ffn_m_axi_rready),
        .m_axi_awaddr(ffn_m_axi_awaddr),
        .m_axi_awlen(ffn_m_axi_awlen),
        .m_axi_awsize(ffn_m_axi_awsize),
        .m_axi_awprot(ffn_m_axi_awprot),
        .m_axi_awburst(ffn_m_axi_awburst),
        .m_axi_awid(ffn_m_axi_awid),
        .m_axi_awcache(ffn_m_axi_awcache),
        .m_axi_awlock(ffn_m_axi_awlock),
        .m_axi_awvalid(ffn_m_axi_awvalid),
        .m_axi_awready(ffn_m_axi_awready),
        .m_axi_wdata(ffn_m_axi_wdata),
        .m_axi_wstrb(ffn_m_axi_wstrb),
        .m_axi_wlast(ffn_m_axi_wlast),
        .m_axi_wvalid(ffn_m_axi_wvalid),
        .m_axi_wready(ffn_m_axi_wready),
        .m_axi_bid(ffn_m_axi_bid),
        .m_axi_bresp(ffn_m_axi_bresp),
        .m_axi_bvalid(ffn_m_axi_bvalid),
        .m_axi_bready(ffn_m_axi_bready)
    );
endmodule
`default_nettype wire
