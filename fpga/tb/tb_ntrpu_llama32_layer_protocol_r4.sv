`timescale 1ns/1ps
`define CHECK(C,M) if((C)!==1'b1) $fatal(1,"LAYER_FAIL case=%0d cycle=%0d %s",case_id,cycle,M);
module tb_ntrpu_llama32_layer_protocol_r4;
    reg  clk;
    reg  rst_n;
    reg  command_valid;
    wire  command_ready;
    reg [63:0] command_sequence;
    reg [63:0] command_epoch;
    reg [63:0] command_lease_base;
    reg [63:0] expected_epoch;
    reg [31:0] command_layer;
    reg [31:0] command_position;
    reg [31:0] command_coefficient_position;
    reg [23:0][63:0] addr_driver;
    wire [63:0] command_addr [0:23];
    for(genvar i_addr_driver=0;i_addr_driver<24;i_addr_driver=i_addr_driver+1)assign command_addr[i_addr_driver]=addr_driver[i_addr_driver];
    reg [23:0][63:0] base_driver;
    wire [63:0] command_base [0:23];
    for(genvar i_base_driver=0;i_base_driver<24;i_base_driver=i_base_driver+1)assign command_base[i_base_driver]=base_driver[i_base_driver];
    reg [23:0][63:0] bytes_driver;
    wire [63:0] command_bytes [0:23];
    for(genvar i_bytes_driver=0;i_bytes_driver<24;i_bytes_driver=i_bytes_driver+1)assign command_bytes[i_bytes_driver]=bytes_driver[i_bytes_driver];
    wire  cache_exclusive;
    reg [63:0] command_source_sequence;
    reg [63:0] command_source_epoch;
    reg  command_source_valid;
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
    reg  input_axi_arready;
    reg [511:0] input_axi_rdata;
    reg [1:0] input_axi_rresp;
    reg [3:0] input_axi_rid;
    reg  input_axi_rlast;
    reg  input_axi_rvalid;
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
    reg  input_axi_awready;
    wire [511:0] input_axi_wdata;
    wire [63:0] input_axi_wstrb;
    wire  input_axi_wlast;
    wire  input_axi_wvalid;
    reg  input_axi_wready;
    reg [3:0] input_axi_bid;
    reg [1:0] input_axi_bresp;
    reg  input_axi_bvalid;
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
    reg  v_axi_arready;
    reg [511:0] v_axi_rdata;
    reg [1:0] v_axi_rresp;
    reg [3:0] v_axi_rid;
    reg  v_axi_rlast;
    reg  v_axi_rvalid;
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
    reg  rope_axi_arready;
    reg [511:0] rope_axi_rdata;
    reg [1:0] rope_axi_rresp;
    reg [3:0] rope_axi_rid;
    reg  rope_axi_rlast;
    reg  rope_axi_rvalid;
    wire  rope_axi_rready;
    reg  bind_valid;
    wire  bind_ready;
    reg [63:0] bind_cache_base;
    reg [63:0] bind_region_base;
    reg [63:0] bind_region_bytes;
    reg [31:0] bind_capacity;
    reg [63:0] bind_sequence;
    reg [63:0] bind_epoch;
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
    reg  kv_axi_awready;
    wire [511:0] kv_axi_wdata;
    wire [63:0] kv_axi_wstrb;
    wire  kv_axi_wlast;
    wire  kv_axi_wvalid;
    reg  kv_axi_wready;
    reg [3:0] kv_axi_bid;
    reg [1:0] kv_axi_bresp;
    reg  kv_axi_bvalid;
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
    reg  kv_axi_arready;
    reg [511:0] kv_axi_rdata;
    reg [1:0] kv_axi_rresp;
    reg [3:0] kv_axi_rid;
    reg  kv_axi_rlast;
    reg  kv_axi_rvalid;
    wire  kv_axi_rready;
    wire  root_cache_lock;
    wire  cache_metadata_fault;
    wire [31:0] cache_query_layer;
    wire  cache_query_valid;
    wire [31:0] cache_query_committed_tokens;
    wire [31:0] cache_committed_tokens;
    wire  bind_completion_valid;
    reg  bind_completion_ready;
    wire [3:0] bind_completion_status;
    wire [63:0] bind_completion_sequence;
    wire [63:0] bind_completion_epoch;
    reg  cancel_valid;
    reg  completion_ready;
    wire  cancel_ready;
    wire  busy;
    wire  completion_valid;
    wire  completion_revoked;
    wire [3:0] completion_status;
    wire [63:0] completion_sequence;
    wire [63:0] completion_epoch;
    wire [31:0] completion_layer;
    wire [31:0] completion_position;
    wire [2:0] completion_stage;
    wire  result_valid;
    wire [63:0] result_address;
    wire [13:0] result_bytes;
    wire [1:0] tail_selected;
    wire  projection_tags_exhausted;
    wire  ffn_tags_exhausted;
    wire [3:0] completion_preflight_status;
    wire [4:0] completion_preflight_region;
    wire [4:0] completion_preflight_other_region;
    wire [63:0] tail_axi_araddr;
    wire [7:0] tail_axi_arlen;
    wire [2:0] tail_axi_arsize;
    wire [2:0] tail_axi_arprot;
    wire [1:0] tail_axi_arburst;
    wire [3:0] tail_axi_arid;
    wire [3:0] tail_axi_arcache;
    wire  tail_axi_arlock;
    wire  tail_axi_arvalid;
    reg  tail_axi_arready;
    reg [511:0] tail_axi_rdata;
    reg [1:0] tail_axi_rresp;
    reg [3:0] tail_axi_rid;
    reg  tail_axi_rlast;
    reg  tail_axi_rvalid;
    wire  tail_axi_rready;
    wire [63:0] tail_axi_awaddr;
    wire [7:0] tail_axi_awlen;
    wire [2:0] tail_axi_awsize;
    wire [2:0] tail_axi_awprot;
    wire [1:0] tail_axi_awburst;
    wire [3:0] tail_axi_awid;
    wire [3:0] tail_axi_awcache;
    wire  tail_axi_awlock;
    wire  tail_axi_awvalid;
    reg  tail_axi_awready;
    wire [511:0] tail_axi_wdata;
    wire [63:0] tail_axi_wstrb;
    wire  tail_axi_wlast;
    wire  tail_axi_wvalid;
    reg  tail_axi_wready;
    reg [3:0] tail_axi_bid;
    reg [1:0] tail_axi_bresp;
    reg  tail_axi_bvalid;
    wire  tail_axi_bready;
    wire [3:0] completion_front_status;
    wire [3:0] completion_front_source_status;
    wire [3:0] completion_front_child_status;
    wire  completion_front_kv_committed;
    wire [12:0] completion_front_kv_bytes;
    wire [31:0] completion_front_kv_tokens;
    wire  completion_front_revoked;
    wire [3:0] completion_front_norm_status;
    wire [3:0] completion_front_qkv_status;
    wire [3:0] completion_front_matrix_status;
    wire [31:0] completion_front_norm_hls_status;
    wire [9:0] completion_front_norm_read_beats;
    wire [6:0] completion_front_norm_records;
    wire [1:0] completion_front_phase;
    wire [1:0] completion_front_matrices;
    wire [14:0] completion_front_total_bytes;
    wire [3:0] completion_projection_status;
    wire [31:0] completion_projection_hls_status;
    wire [3:0] completion_post_status;
    wire [31:0] completion_post_hls_status;
    wire [9:0] completion_post_read_beats;
    wire [7:0] completion_post_residual_beats;
    wire [6:0] completion_post_ffn_beats;
    wire [3:0] completion_lease_status;
    wire [3:0] completion_lease_source_status;
    wire [3:0] completion_lease_child_status;
    wire [31:0] completion_lease_hls_status;
    wire [31:0] completion_lease_ffn_hls_status;
    wire [3:0] completion_lease_ffn_status;
    wire [3:0] completion_lease_ffn_child_status;
    wire [3:0] completion_lease_residual_status;
    wire [2:0] completion_lease_ffn_stages;
    wire [1:0] completion_lease_ffn_phase;
    wire [16:0] completion_lease_ffn_bytes;
    wire [17:0] completion_lease_ffn_stage_bytes;
    wire [31:0] completion_lease_source_root_hls_status;
    wire [31:0] completion_lease_source_post_hls_status;
    wire [9:0] completion_lease_read_beats;
    wire [7:0] completion_lease_write_beats;
    ntrpu_llama32_layer_r1 #(.OWNER_TIMEOUT_CYCLES(6000)) dut(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(command_valid),
        .command_ready(command_ready),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .command_lease_base(command_lease_base),
        .expected_epoch(expected_epoch),
        .command_layer(command_layer),
        .command_position(command_position),
        .command_coefficient_position(command_coefficient_position),
        .command_addr(command_addr),
        .command_base(command_base),
        .command_bytes(command_bytes),
        .cache_exclusive(cache_exclusive),
        .command_source_sequence(command_source_sequence),
        .command_source_epoch(command_source_epoch),
        .command_source_valid(command_source_valid),
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
        .bind_valid(bind_valid),
        .bind_ready(bind_ready),
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
        .bind_completion_epoch(bind_completion_epoch),
        .cancel_valid(cancel_valid),
        .completion_ready(completion_ready),
        .cancel_ready(cancel_ready),
        .busy(busy),
        .completion_valid(completion_valid),
        .completion_revoked(completion_revoked),
        .completion_status(completion_status),
        .completion_sequence(completion_sequence),
        .completion_epoch(completion_epoch),
        .completion_layer(completion_layer),
        .completion_position(completion_position),
        .completion_stage(completion_stage),
        .result_valid(result_valid),
        .result_address(result_address),
        .result_bytes(result_bytes),
        .tail_selected(tail_selected),
        .projection_tags_exhausted(projection_tags_exhausted),
        .ffn_tags_exhausted(ffn_tags_exhausted),
        .completion_preflight_status(completion_preflight_status),
        .completion_preflight_region(completion_preflight_region),
        .completion_preflight_other_region(completion_preflight_other_region),
        .tail_axi_araddr(tail_axi_araddr),
        .tail_axi_arlen(tail_axi_arlen),
        .tail_axi_arsize(tail_axi_arsize),
        .tail_axi_arprot(tail_axi_arprot),
        .tail_axi_arburst(tail_axi_arburst),
        .tail_axi_arid(tail_axi_arid),
        .tail_axi_arcache(tail_axi_arcache),
        .tail_axi_arlock(tail_axi_arlock),
        .tail_axi_arvalid(tail_axi_arvalid),
        .tail_axi_arready(tail_axi_arready),
        .tail_axi_rdata(tail_axi_rdata),
        .tail_axi_rresp(tail_axi_rresp),
        .tail_axi_rid(tail_axi_rid),
        .tail_axi_rlast(tail_axi_rlast),
        .tail_axi_rvalid(tail_axi_rvalid),
        .tail_axi_rready(tail_axi_rready),
        .tail_axi_awaddr(tail_axi_awaddr),
        .tail_axi_awlen(tail_axi_awlen),
        .tail_axi_awsize(tail_axi_awsize),
        .tail_axi_awprot(tail_axi_awprot),
        .tail_axi_awburst(tail_axi_awburst),
        .tail_axi_awid(tail_axi_awid),
        .tail_axi_awcache(tail_axi_awcache),
        .tail_axi_awlock(tail_axi_awlock),
        .tail_axi_awvalid(tail_axi_awvalid),
        .tail_axi_awready(tail_axi_awready),
        .tail_axi_wdata(tail_axi_wdata),
        .tail_axi_wstrb(tail_axi_wstrb),
        .tail_axi_wlast(tail_axi_wlast),
        .tail_axi_wvalid(tail_axi_wvalid),
        .tail_axi_wready(tail_axi_wready),
        .tail_axi_bid(tail_axi_bid),
        .tail_axi_bresp(tail_axi_bresp),
        .tail_axi_bvalid(tail_axi_bvalid),
        .tail_axi_bready(tail_axi_bready),
        .completion_front_status(completion_front_status),
        .completion_front_source_status(completion_front_source_status),
        .completion_front_child_status(completion_front_child_status),
        .completion_front_kv_committed(completion_front_kv_committed),
        .completion_front_kv_bytes(completion_front_kv_bytes),
        .completion_front_kv_tokens(completion_front_kv_tokens),
        .completion_front_revoked(completion_front_revoked),
        .completion_front_norm_status(completion_front_norm_status),
        .completion_front_qkv_status(completion_front_qkv_status),
        .completion_front_matrix_status(completion_front_matrix_status),
        .completion_front_norm_hls_status(completion_front_norm_hls_status),
        .completion_front_norm_read_beats(completion_front_norm_read_beats),
        .completion_front_norm_records(completion_front_norm_records),
        .completion_front_phase(completion_front_phase),
        .completion_front_matrices(completion_front_matrices),
        .completion_front_total_bytes(completion_front_total_bytes),
        .completion_projection_status(completion_projection_status),
        .completion_projection_hls_status(completion_projection_hls_status),
        .completion_post_status(completion_post_status),
        .completion_post_hls_status(completion_post_hls_status),
        .completion_post_read_beats(completion_post_read_beats),
        .completion_post_residual_beats(completion_post_residual_beats),
        .completion_post_ffn_beats(completion_post_ffn_beats),
        .completion_lease_status(completion_lease_status),
        .completion_lease_source_status(completion_lease_source_status),
        .completion_lease_child_status(completion_lease_child_status),
        .completion_lease_hls_status(completion_lease_hls_status),
        .completion_lease_ffn_hls_status(completion_lease_ffn_hls_status),
        .completion_lease_ffn_status(completion_lease_ffn_status),
        .completion_lease_ffn_child_status(completion_lease_ffn_child_status),
        .completion_lease_residual_status(completion_lease_residual_status),
        .completion_lease_ffn_stages(completion_lease_ffn_stages),
        .completion_lease_ffn_phase(completion_lease_ffn_phase),
        .completion_lease_ffn_bytes(completion_lease_ffn_bytes),
        .completion_lease_ffn_stage_bytes(completion_lease_ffn_stage_bytes),
        .completion_lease_source_root_hls_status(completion_lease_source_root_hls_status),
        .completion_lease_source_post_hls_status(completion_lease_source_post_hls_status),
        .completion_lease_read_beats(completion_lease_read_beats),
        .completion_lease_write_beats(completion_lease_write_beats)
    );

    integer seed=1,cycle=0,jobs=0,case_id=0,waits=0,mode=0,target=0;
    integer start_front,start_o,start_post,start_ffn,drain_start;
    reg [63:0] wanted_sequence,wanted_epoch;
    reg [31:0] wanted_layer,wanted_position;
    reg [3:0] wanted_status;
    reg [23:0][63:0] saved_address,saved_base,saved_bytes;
    reg injected=0;
    always #5 clk=~clk;
    always @(posedge clk)begin
        cycle<=cycle+1;
        if(rst_n && busy)begin
            `CHECK(!command_ready && !bind_ready,"ownership released while busy")
            `CHECK(cache_exclusive,"cache reservation missing")
        end
        if(rst_n)begin
            `CHECK(tail_axi_araddr===(tail_selected==2 ? dut.ffn.m_axi_araddr:tail_selected==1 ? dut.post.m_axi_araddr:dut.projection.m_axi_araddr),"tail output araddr")
            `CHECK(tail_axi_arlen===(tail_selected==2 ? dut.ffn.m_axi_arlen:tail_selected==1 ? dut.post.m_axi_arlen:dut.projection.m_axi_arlen),"tail output arlen")
            `CHECK(tail_axi_arsize===(tail_selected==2 ? dut.ffn.m_axi_arsize:tail_selected==1 ? dut.post.m_axi_arsize:dut.projection.m_axi_arsize),"tail output arsize")
            `CHECK(tail_axi_arprot===(tail_selected==2 ? dut.ffn.m_axi_arprot:tail_selected==1 ? dut.post.m_axi_arprot:dut.projection.m_axi_arprot),"tail output arprot")
            `CHECK(tail_axi_arburst===(tail_selected==2 ? dut.ffn.m_axi_arburst:tail_selected==1 ? dut.post.m_axi_arburst:dut.projection.m_axi_arburst),"tail output arburst")
            `CHECK(tail_axi_arid===(tail_selected==2 ? dut.ffn.m_axi_arid:tail_selected==1 ? dut.post.m_axi_arid:dut.projection.m_axi_arid),"tail output arid")
            `CHECK(tail_axi_arcache===(tail_selected==2 ? dut.ffn.m_axi_arcache:tail_selected==1 ? dut.post.m_axi_arcache:dut.projection.m_axi_arcache),"tail output arcache")
            `CHECK(tail_axi_arlock===(tail_selected==2 ? dut.ffn.m_axi_arlock:tail_selected==1 ? dut.post.m_axi_arlock:dut.projection.m_axi_arlock),"tail output arlock")
            `CHECK(tail_axi_arvalid===(tail_selected==2 ? dut.ffn.m_axi_arvalid:tail_selected==1 ? dut.post.m_axi_arvalid:dut.projection.m_axi_arvalid),"tail output arvalid")
            `CHECK(dut.projection.m_axi_arready===((tail_selected==0 && tail_axi_arready)),"tail input projection arready")
            `CHECK(dut.post.m_axi_arready===((tail_selected==1 && tail_axi_arready)),"tail input post arready")
            `CHECK(dut.ffn.m_axi_arready===((tail_selected==2 && tail_axi_arready)),"tail input ffn arready")
            `CHECK(dut.projection.m_axi_rdata===(tail_axi_rdata),"tail input projection rdata")
            `CHECK(dut.post.m_axi_rdata===(tail_axi_rdata),"tail input post rdata")
            `CHECK(dut.ffn.m_axi_rdata===(tail_axi_rdata),"tail input ffn rdata")
            `CHECK(dut.projection.m_axi_rresp===(tail_axi_rresp),"tail input projection rresp")
            `CHECK(dut.post.m_axi_rresp===(tail_axi_rresp),"tail input post rresp")
            `CHECK(dut.ffn.m_axi_rresp===(tail_axi_rresp),"tail input ffn rresp")
            `CHECK(dut.projection.m_axi_rid===(tail_axi_rid),"tail input projection rid")
            `CHECK(dut.post.m_axi_rid===(tail_axi_rid),"tail input post rid")
            `CHECK(dut.ffn.m_axi_rid===(tail_axi_rid),"tail input ffn rid")
            `CHECK(dut.projection.m_axi_rlast===(tail_axi_rlast),"tail input projection rlast")
            `CHECK(dut.post.m_axi_rlast===(tail_axi_rlast),"tail input post rlast")
            `CHECK(dut.ffn.m_axi_rlast===(tail_axi_rlast),"tail input ffn rlast")
            `CHECK(dut.projection.m_axi_rvalid===((tail_selected==0 && tail_axi_rvalid)),"tail input projection rvalid")
            `CHECK(dut.post.m_axi_rvalid===((tail_selected==1 && tail_axi_rvalid)),"tail input post rvalid")
            `CHECK(dut.ffn.m_axi_rvalid===((tail_selected==2 && tail_axi_rvalid)),"tail input ffn rvalid")
            `CHECK(tail_axi_rready===(tail_selected==2 ? dut.ffn.m_axi_rready:tail_selected==1 ? dut.post.m_axi_rready:dut.projection.m_axi_rready),"tail output rready")
            `CHECK(tail_axi_awaddr===(tail_selected==2 ? dut.ffn.m_axi_awaddr:tail_selected==1 ? dut.post.m_axi_awaddr:dut.projection.m_axi_awaddr),"tail output awaddr")
            `CHECK(tail_axi_awlen===(tail_selected==2 ? dut.ffn.m_axi_awlen:tail_selected==1 ? dut.post.m_axi_awlen:dut.projection.m_axi_awlen),"tail output awlen")
            `CHECK(tail_axi_awsize===(tail_selected==2 ? dut.ffn.m_axi_awsize:tail_selected==1 ? dut.post.m_axi_awsize:dut.projection.m_axi_awsize),"tail output awsize")
            `CHECK(tail_axi_awprot===(tail_selected==2 ? dut.ffn.m_axi_awprot:tail_selected==1 ? dut.post.m_axi_awprot:dut.projection.m_axi_awprot),"tail output awprot")
            `CHECK(tail_axi_awburst===(tail_selected==2 ? dut.ffn.m_axi_awburst:tail_selected==1 ? dut.post.m_axi_awburst:dut.projection.m_axi_awburst),"tail output awburst")
            `CHECK(tail_axi_awid===(tail_selected==2 ? dut.ffn.m_axi_awid:tail_selected==1 ? dut.post.m_axi_awid:dut.projection.m_axi_awid),"tail output awid")
            `CHECK(tail_axi_awcache===(tail_selected==2 ? dut.ffn.m_axi_awcache:tail_selected==1 ? dut.post.m_axi_awcache:dut.projection.m_axi_awcache),"tail output awcache")
            `CHECK(tail_axi_awlock===(tail_selected==2 ? dut.ffn.m_axi_awlock:tail_selected==1 ? dut.post.m_axi_awlock:dut.projection.m_axi_awlock),"tail output awlock")
            `CHECK(tail_axi_awvalid===(tail_selected==2 ? dut.ffn.m_axi_awvalid:tail_selected==1 ? dut.post.m_axi_awvalid:dut.projection.m_axi_awvalid),"tail output awvalid")
            `CHECK(dut.projection.m_axi_awready===((tail_selected==0 && tail_axi_awready)),"tail input projection awready")
            `CHECK(dut.post.m_axi_awready===((tail_selected==1 && tail_axi_awready)),"tail input post awready")
            `CHECK(dut.ffn.m_axi_awready===((tail_selected==2 && tail_axi_awready)),"tail input ffn awready")
            `CHECK(tail_axi_wdata===(tail_selected==2 ? dut.ffn.m_axi_wdata:tail_selected==1 ? dut.post.m_axi_wdata:dut.projection.m_axi_wdata),"tail output wdata")
            `CHECK(tail_axi_wstrb===(tail_selected==2 ? dut.ffn.m_axi_wstrb:tail_selected==1 ? dut.post.m_axi_wstrb:dut.projection.m_axi_wstrb),"tail output wstrb")
            `CHECK(tail_axi_wlast===(tail_selected==2 ? dut.ffn.m_axi_wlast:tail_selected==1 ? dut.post.m_axi_wlast:dut.projection.m_axi_wlast),"tail output wlast")
            `CHECK(tail_axi_wvalid===(tail_selected==2 ? dut.ffn.m_axi_wvalid:tail_selected==1 ? dut.post.m_axi_wvalid:dut.projection.m_axi_wvalid),"tail output wvalid")
            `CHECK(dut.projection.m_axi_wready===((tail_selected==0 && tail_axi_wready)),"tail input projection wready")
            `CHECK(dut.post.m_axi_wready===((tail_selected==1 && tail_axi_wready)),"tail input post wready")
            `CHECK(dut.ffn.m_axi_wready===((tail_selected==2 && tail_axi_wready)),"tail input ffn wready")
            `CHECK(dut.projection.m_axi_bid===(tail_axi_bid),"tail input projection bid")
            `CHECK(dut.post.m_axi_bid===(tail_axi_bid),"tail input post bid")
            `CHECK(dut.ffn.m_axi_bid===(tail_axi_bid),"tail input ffn bid")
            `CHECK(dut.projection.m_axi_bresp===(tail_axi_bresp),"tail input projection bresp")
            `CHECK(dut.post.m_axi_bresp===(tail_axi_bresp),"tail input post bresp")
            `CHECK(dut.ffn.m_axi_bresp===(tail_axi_bresp),"tail input ffn bresp")
            `CHECK(dut.projection.m_axi_bvalid===((tail_selected==0 && tail_axi_bvalid)),"tail input projection bvalid")
            `CHECK(dut.post.m_axi_bvalid===((tail_selected==1 && tail_axi_bvalid)),"tail input post bvalid")
            `CHECK(dut.ffn.m_axi_bvalid===((tail_selected==2 && tail_axi_bvalid)),"tail input ffn bvalid")
            `CHECK(tail_axi_bready===(tail_selected==2 ? dut.ffn.m_axi_bready:tail_selected==1 ? dut.post.m_axi_bready:dut.projection.m_axi_bready),"tail output bready")
        end
        if(rst_n && dut.front.command_valid && dut.front.command_ready)begin
            `CHECK(dut.front.command_sequence==wanted_sequence && dut.front.command_epoch==wanted_epoch,"front admission identity")
            `CHECK(dut.front.command_layer==wanted_layer && dut.front.command_position==wanted_position,"captured layer position")
            `CHECK(dut.front.command_addr[0]==saved_address[0],"front descriptor command_addr[0]")
            `CHECK(dut.front.command_base[0]==saved_base[0],"front descriptor command_base[0]")
            `CHECK(dut.front.command_bytes[0]==saved_bytes[0],"front descriptor command_bytes[0]")
            `CHECK(dut.front.command_addr[1]==saved_address[1],"front descriptor command_addr[1]")
            `CHECK(dut.front.command_base[1]==saved_base[1],"front descriptor command_base[1]")
            `CHECK(dut.front.command_bytes[1]==saved_bytes[1],"front descriptor command_bytes[1]")
            `CHECK(dut.front.command_addr[2]==saved_address[2],"front descriptor command_addr[2]")
            `CHECK(dut.front.command_base[2]==saved_base[2],"front descriptor command_base[2]")
            `CHECK(dut.front.command_bytes[2]==saved_bytes[2],"front descriptor command_bytes[2]")
            `CHECK(dut.front.command_addr[3]==saved_address[3],"front descriptor command_addr[3]")
            `CHECK(dut.front.command_base[3]==saved_base[3],"front descriptor command_base[3]")
            `CHECK(dut.front.command_bytes[3]==saved_bytes[3],"front descriptor command_bytes[3]")
            `CHECK(dut.front.command_addr[4]==saved_address[4],"front descriptor command_addr[4]")
            `CHECK(dut.front.command_base[4]==saved_base[4],"front descriptor command_base[4]")
            `CHECK(dut.front.command_bytes[4]==saved_bytes[4],"front descriptor command_bytes[4]")
            `CHECK(dut.front.command_addr[5]==saved_address[5],"front descriptor command_addr[5]")
            `CHECK(dut.front.command_base[5]==saved_base[5],"front descriptor command_base[5]")
            `CHECK(dut.front.command_bytes[5]==saved_bytes[5],"front descriptor command_bytes[5]")
            `CHECK(dut.front.command_addr[6]==saved_address[6],"front descriptor command_addr[6]")
            `CHECK(dut.front.command_base[6]==saved_base[6],"front descriptor command_base[6]")
            `CHECK(dut.front.command_bytes[6]==saved_bytes[6],"front descriptor command_bytes[6]")
            `CHECK(dut.front.command_addr[7]==saved_address[7],"front descriptor command_addr[7]")
            `CHECK(dut.front.command_base[7]==saved_base[7],"front descriptor command_base[7]")
            `CHECK(dut.front.command_bytes[7]==saved_bytes[7],"front descriptor command_bytes[7]")
            `CHECK(dut.front.command_addr[8]==saved_address[8],"front descriptor command_addr[8]")
            `CHECK(dut.front.command_base[8]==saved_base[8],"front descriptor command_base[8]")
            `CHECK(dut.front.command_bytes[8]==saved_bytes[8],"front descriptor command_bytes[8]")
            `CHECK(dut.front.command_addr[9]==saved_address[9],"front descriptor command_addr[9]")
            `CHECK(dut.front.command_base[9]==saved_base[9],"front descriptor command_base[9]")
            `CHECK(dut.front.command_bytes[9]==saved_bytes[9],"front descriptor command_bytes[9]")
        end
        if(rst_n && dut.projection.command_valid && dut.projection.command_ready)begin
            `CHECK(dut.projection.command_sequence==wanted_sequence && dut.projection.command_epoch==wanted_epoch,"projection admission identity")
            `CHECK(dut.projection.command_weight_addr==saved_address[10],"projection descriptor command_weight_addr")
            `CHECK(dut.projection.command_weight_region_base==saved_base[10],"projection descriptor command_weight_region_base")
            `CHECK(dut.projection.command_weight_region_bytes==saved_bytes[10],"projection descriptor command_weight_region_bytes")
            `CHECK(dut.projection.command_activation_addr==saved_address[11],"projection descriptor command_activation_addr")
            `CHECK(dut.projection.command_activation_region_base==saved_base[11],"projection descriptor command_activation_region_base")
            `CHECK(dut.projection.command_activation_region_bytes==saved_bytes[11],"projection descriptor command_activation_region_bytes")
            `CHECK(dut.projection.command_output_addr==saved_address[12],"projection descriptor command_output_addr")
            `CHECK(dut.projection.command_output_region_base==saved_base[12],"projection descriptor command_output_region_base")
            `CHECK(dut.projection.command_output_region_bytes==saved_bytes[12],"projection descriptor command_output_region_bytes")
        end
        if(rst_n && dut.post.command_valid && dut.post.command_ready)begin
            `CHECK(dut.post.command_sequence==wanted_sequence && dut.post.command_epoch==wanted_epoch,"post admission identity")
            `CHECK(dut.post.command_input_address==saved_address[0],"post descriptor command_input_address")
            `CHECK(dut.post.command_input_base==saved_base[0],"post descriptor command_input_base")
            `CHECK(dut.post.command_input_bytes==saved_bytes[0],"post descriptor command_input_bytes")
            `CHECK(dut.post.command_branch_address==saved_address[12],"post descriptor command_branch_address")
            `CHECK(dut.post.command_branch_base==saved_base[12],"post descriptor command_branch_base")
            `CHECK(dut.post.command_branch_bytes==saved_bytes[12],"post descriptor command_branch_bytes")
            `CHECK(dut.post.command_weight_address==saved_address[13],"post descriptor command_weight_address")
            `CHECK(dut.post.command_weight_base==saved_base[13],"post descriptor command_weight_base")
            `CHECK(dut.post.command_weight_bytes==saved_bytes[13],"post descriptor command_weight_bytes")
            `CHECK(dut.post.command_residual_address==saved_address[14],"post descriptor command_residual_address")
            `CHECK(dut.post.command_residual_base==saved_base[14],"post descriptor command_residual_base")
            `CHECK(dut.post.command_residual_bytes==saved_bytes[14],"post descriptor command_residual_bytes")
            `CHECK(dut.post.command_ffn_address==saved_address[15],"post descriptor command_ffn_address")
            `CHECK(dut.post.command_ffn_base==saved_base[15],"post descriptor command_ffn_base")
            `CHECK(dut.post.command_ffn_bytes==saved_bytes[15],"post descriptor command_ffn_bytes")
        end
        if(rst_n && dut.ffn.command_valid && dut.ffn.command_ready)begin
            `CHECK(dut.ffn.command_sequence==wanted_sequence && dut.ffn.command_epoch==wanted_epoch,"ffn admission identity")
            `CHECK(dut.ffn.command_addr[0]==saved_address[15],"ffn descriptor command_addr[0]")
            `CHECK(dut.ffn.command_base[0]==saved_address[15],"ffn descriptor command_base[0]")
            `CHECK(dut.ffn.command_bytes[0]==64'd4096,"ffn descriptor command_bytes[0]")
            `CHECK(dut.ffn.command_addr[1]==saved_address[16],"ffn descriptor command_addr[1]")
            `CHECK(dut.ffn.command_base[1]==saved_base[16],"ffn descriptor command_base[1]")
            `CHECK(dut.ffn.command_bytes[1]==saved_bytes[16],"ffn descriptor command_bytes[1]")
            `CHECK(dut.ffn.command_addr[2]==saved_address[17],"ffn descriptor command_addr[2]")
            `CHECK(dut.ffn.command_base[2]==saved_base[17],"ffn descriptor command_base[2]")
            `CHECK(dut.ffn.command_bytes[2]==saved_bytes[17],"ffn descriptor command_bytes[2]")
            `CHECK(dut.ffn.command_addr[3]==saved_address[18],"ffn descriptor command_addr[3]")
            `CHECK(dut.ffn.command_base[3]==saved_base[18],"ffn descriptor command_base[3]")
            `CHECK(dut.ffn.command_bytes[3]==saved_bytes[18],"ffn descriptor command_bytes[3]")
            `CHECK(dut.ffn.command_addr[4]==saved_address[19],"ffn descriptor command_addr[4]")
            `CHECK(dut.ffn.command_base[4]==saved_base[19],"ffn descriptor command_base[4]")
            `CHECK(dut.ffn.command_bytes[4]==saved_bytes[19],"ffn descriptor command_bytes[4]")
            `CHECK(dut.ffn.command_addr[5]==saved_address[20],"ffn descriptor command_addr[5]")
            `CHECK(dut.ffn.command_base[5]==saved_base[20],"ffn descriptor command_base[5]")
            `CHECK(dut.ffn.command_bytes[5]==saved_bytes[20],"ffn descriptor command_bytes[5]")
            `CHECK(dut.ffn.command_addr[6]==saved_address[21],"ffn descriptor command_addr[6]")
            `CHECK(dut.ffn.command_base[6]==saved_base[21],"ffn descriptor command_base[6]")
            `CHECK(dut.ffn.command_bytes[6]==saved_bytes[21],"ffn descriptor command_bytes[6]")
            `CHECK(dut.ffn.command_addr[7]==saved_address[22],"ffn descriptor command_addr[7]")
            `CHECK(dut.ffn.command_base[7]==saved_base[22],"ffn descriptor command_base[7]")
            `CHECK(dut.ffn.command_bytes[7]==saved_bytes[22],"ffn descriptor command_bytes[7]")
            `CHECK(dut.ffn.command_residual_address==saved_address[14],"ffn descriptor command_residual_address")
            `CHECK(dut.ffn.command_residual_base==saved_address[14],"ffn descriptor command_residual_base")
            `CHECK(dut.ffn.command_residual_bytes==64'd8192,"ffn descriptor command_residual_bytes")
            `CHECK(dut.ffn.command_output_address==saved_address[23],"ffn descriptor command_output_address")
            `CHECK(dut.ffn.command_output_base==saved_base[23],"ffn descriptor command_output_base")
            `CHECK(dut.ffn.command_output_bytes==saved_bytes[23],"ffn descriptor command_output_bytes")
        end
        if(cycle>1000000)$fatal(1,"LAYER_FAIL global test deadline");
    end
    task tick;begin @(posedge clk);#1;@(negedge clk);end endtask
    task defaults;begin
        dut.front.cfg_badshape=0;dut.front.cfg_foreign=0;dut.front.cfg_revoke=0;dut.front.cfg_block=0;dut.front.cfg_status=0;dut.front.cfg_delay=8+seed%7;dut.front.cfg_drain=5+seed%5;
        dut.projection.cfg_badshape=0;dut.projection.cfg_foreign=0;dut.projection.cfg_revoke=0;dut.projection.cfg_block=0;dut.projection.cfg_status=0;dut.projection.cfg_delay=8+seed%7;dut.projection.cfg_drain=5+seed%5;
        dut.post.cfg_badshape=0;dut.post.cfg_foreign=0;dut.post.cfg_revoke=0;dut.post.cfg_block=0;dut.post.cfg_status=0;dut.post.cfg_delay=8+seed%7;dut.post.cfg_drain=5+seed%5;
        dut.ffn.cfg_badshape=0;dut.ffn.cfg_foreign=0;dut.ffn.cfg_revoke=0;dut.ffn.cfg_block=0;dut.ffn.cfg_status=0;dut.ffn.cfg_delay=8+seed%7;dut.ffn.cfg_drain=5+seed%5;
        dut.front.cfg_cache_fault=0;
        command_valid=0;cancel_valid=0;completion_ready=0;bind_valid=0;bind_completion_ready=0;
        command_sequence=64'h1234567887654321;command_epoch=64'hfedcba9876543210;
        expected_epoch=command_epoch;command_source_sequence=command_sequence;command_source_epoch=command_epoch;command_source_valid=1;
        command_lease_base=64'h8000000012340000;command_layer=jobs%16;command_position=0;command_coefficient_position=0;
        addr_driver[0]=(64'd1<<40)+4032;base_driver[0]=(64'd1<<40);bytes_driver[0]=12224;
        addr_driver[1]=(64'd2<<40)+4032;base_driver[1]=(64'd2<<40);bytes_driver[1]=12224;
        addr_driver[2]=(64'd3<<40)+4032;base_driver[2]=(64'd3<<40);bytes_driver[2]=8128;
        addr_driver[3]=(64'd4<<40)+4032;base_driver[3]=(64'd4<<40);bytes_driver[3]=4460480;
        addr_driver[4]=(64'd5<<40)+4032;base_driver[4]=(64'd5<<40);bytes_driver[4]=1118144;
        addr_driver[5]=(64'd6<<40)+4032;base_driver[5]=(64'd6<<40);bytes_driver[5]=1118144;
        addr_driver[6]=(64'd7<<40)+4032;base_driver[6]=(64'd7<<40);bytes_driver[6]=12224;
        addr_driver[7]=(64'd8<<40)+4032;base_driver[7]=(64'd8<<40);bytes_driver[7]=6080;
        addr_driver[8]=(64'd9<<40)+4032;base_driver[8]=(64'd9<<40);bytes_driver[8]=6080;
        addr_driver[9]=(64'd10<<40)+4032;base_driver[9]=(64'd10<<40);bytes_driver[9]=4288;
        addr_driver[10]=(64'd11<<40)+4032;base_driver[10]=(64'd11<<40);bytes_driver[10]=4460480;
        addr_driver[11]=(64'd12<<40)+4032;base_driver[11]=(64'd12<<40);bytes_driver[11]=8128;
        addr_driver[12]=(64'd13<<40)+4032;base_driver[12]=(64'd13<<40);bytes_driver[12]=12224;
        addr_driver[13]=(64'd14<<40)+4032;base_driver[13]=(64'd14<<40);bytes_driver[13]=12224;
        addr_driver[14]=(64'd15<<40)+4032;base_driver[14]=(64'd15<<40);bytes_driver[14]=12224;
        addr_driver[15]=(64'd16<<40)+4032;base_driver[15]=(64'd16<<40);bytes_driver[15]=8128;
        addr_driver[16]=(64'd17<<40)+4032;base_driver[16]=(64'd17<<40);bytes_driver[16]=17829824;
        addr_driver[17]=(64'd18<<40)+4032;base_driver[17]=(64'd18<<40);bytes_driver[17]=17829824;
        addr_driver[18]=(64'd19<<40)+4032;base_driver[18]=(64'd19<<40);bytes_driver[18]=17829824;
        addr_driver[19]=(64'd20<<40)+4032;base_driver[19]=(64'd20<<40);bytes_driver[19]=69568;
        addr_driver[20]=(64'd21<<40)+4032;base_driver[20]=(64'd21<<40);bytes_driver[20]=69568;
        addr_driver[21]=(64'd22<<40)+4032;base_driver[21]=(64'd22<<40);bytes_driver[21]=20416;
        addr_driver[22]=(64'd23<<40)+4032;base_driver[22]=(64'd23<<40);bytes_driver[22]=12224;
        addr_driver[23]=(64'd24<<40)+4032;base_driver[23]=(64'd24<<40);bytes_driver[23]=12224;
        bind_sequence=command_sequence;bind_epoch=command_epoch;bind_cache_base=(64'd25<<40)+4032;
        bind_region_base=(64'd25<<40);bind_region_bytes=262144+4032;bind_capacity=4;
        bind_valid=1;waits=0;
        while(!bind_ready)begin tick();waits=waits+1;`CHECK(waits<40,"bind readiness deadline") end
        tick();bind_valid=0;waits=0;
        while(!bind_completion_valid)begin tick();waits=waits+1;`CHECK(waits<40,"bind completion deadline") end
        bind_completion_ready=1;tick();bind_completion_ready=0;
    end endtask
    task admission;integer i;begin
        wanted_sequence=command_sequence;wanted_epoch=command_epoch;wanted_layer=command_layer;wanted_position=command_position;
        start_front=dut.front.starts;start_o=dut.projection.starts;start_post=dut.post.starts;start_ffn=dut.ffn.starts;
        for(i=0;i<24;i=i+1)begin saved_address[i]=addr_driver[i];saved_base[i]=base_driver[i];saved_bytes[i]=bytes_driver[i];end
        waits=0;command_valid=1;
        while(!command_ready)begin tick();waits=waits+1;`CHECK(waits<40,"admission deadline") end
        tick();command_valid=0;
        command_sequence=0;command_epoch=0;command_source_sequence=0;command_source_epoch=0;command_source_valid=0;
        command_layer=31;command_position=999;command_coefficient_position=999;command_lease_base=0;
        addr_driver='0;base_driver='0;bytes_driver='0;
        $display("LAYER_ADMIT case=%0d mode=%0d target=%0d cycle=%0d sequence=%016h epoch=%016h layer=%0d position=%0d",case_id,mode,target,cycle,wanted_sequence,wanted_epoch,wanted_layer,wanted_position);
    end endtask
    task wait_response;begin
        waits=0;injected=0;
        while(!completion_valid)begin
            if(!injected && mode==3 && completion_stage==target)begin cancel_valid=1;injected=1;end
            if(!injected && mode==4 && completion_stage==target)begin expected_epoch=wanted_epoch ^ 64'h8000000000000000;injected=1;end
            if(mode==7 && !injected && target==1 && dut.front.completion_valid)begin
                repeat(4)begin tick();`CHECK(!dut.front.completion_ready,"foreign identity acknowledged") end
                dut.front.cfg_foreign=0;injected=1;
            end
            if(mode==7 && !injected && target==2 && dut.projection.completion_valid)begin
                repeat(4)begin tick();`CHECK(!dut.projection.completion_ready,"foreign identity acknowledged") end
                dut.projection.cfg_foreign=0;injected=1;
            end
            if(mode==7 && !injected && target==3 && dut.post.completion_valid)begin
                repeat(4)begin tick();`CHECK(!dut.post.completion_ready,"foreign identity acknowledged") end
                dut.post.cfg_foreign=0;injected=1;
            end
            if(mode==7 && !injected && target==4 && dut.ffn.completion_valid)begin
                repeat(4)begin tick();`CHECK(!dut.ffn.completion_ready,"foreign identity acknowledged") end
                dut.ffn.cfg_foreign=0;injected=1;
            end
            tick();waits=waits+1;
            `CHECK(waits<14000,"public response deadline")
        end
        `CHECK(completion_sequence==wanted_sequence && completion_epoch==wanted_epoch,"public identity")
        `CHECK(completion_layer==wanted_layer && completion_position==wanted_position,"public layer position")
        if(completion_status!=wanted_status)$display("LAYER_STATUS actual=%0d expected=%0d stage=%0d gate=%0d fault=%0d",completion_status,wanted_status,completion_stage,completion_preflight_status,dut.fault);
        `CHECK(completion_status==wanted_status,"public status")
        if(wanted_status==0)begin
            `CHECK(result_valid && result_address==saved_address[23] && result_bytes==8192,"successful result descriptor")
            `CHECK(dut.front.starts==start_front+1 && dut.projection.starts==start_o+1 && dut.post.starts==start_post+1 && dut.ffn.starts==start_ffn+1,"single ordered launch")
            `CHECK(dut.projection.read_received==2048 && dut.front.acks==dut.front.starts,"front RAM consumed before early release")
            `CHECK(tail_selected==2,"FFN must retain tail bus")
        end else begin `CHECK(!result_valid,"fault exposed valid result") end
        $display("LAYER_RESULT case=%0d status=%0d stage=%0d revoked=%0d result=%0d front=%0d projection=%0d post=%0d ffn=%0d words=%0d cycle=%0d",case_id,completion_status,completion_stage,completion_revoked,result_valid,dut.front.starts-start_front,dut.projection.starts-start_o,dut.post.starts-start_post,dut.ffn.starts-start_ffn,dut.projection.read_received,cycle);
        repeat(3+seed%5)begin tick();`CHECK(completion_valid && completion_status==wanted_status,"held public response changed") end
    end endtask
    task release_job;begin
        completion_ready=1;tick();completion_ready=0;cancel_valid=0;expected_epoch=wanted_epoch;drain_start=cycle;waits=0;
        while(busy)begin
            if(dut.ffn.busy && dut.ffn.starts>start_ffn)begin `CHECK(tail_selected==2,"tail released before FFN drain") end
            tick();waits=waits+1;`CHECK(waits<100,"drain deadline")
        end
        `CHECK(!completion_valid && !result_valid && command_ready && bind_ready && !cache_exclusive,"idle after drain")
        `CHECK(dut.front.starts==dut.front.acks && dut.projection.starts==dut.projection.acks && dut.post.starts==dut.post.acks && dut.ffn.starts==dut.ffn.acks,"all and only owned peers acknowledged")
        $display("LAYER_RELEASE case=%0d drain=%0d cycle=%0d",case_id,cycle-drain_start,cycle);
        jobs=jobs+1;case_id=case_id+1;
    end endtask
    initial begin
        clk='0;
        rst_n='0;
        command_valid='0;
        command_sequence='0;
        command_epoch='0;
        command_lease_base='0;
        expected_epoch='0;
        command_layer='0;
        command_position='0;
        command_coefficient_position='0;
        addr_driver='0;
        base_driver='0;
        bytes_driver='0;
        command_source_sequence='0;
        command_source_epoch='0;
        command_source_valid='0;
        input_axi_arready='0;
        input_axi_rdata='0;
        input_axi_rresp='0;
        input_axi_rid='0;
        input_axi_rlast='0;
        input_axi_rvalid='0;
        input_axi_awready='0;
        input_axi_wready='0;
        input_axi_bid='0;
        input_axi_bresp='0;
        input_axi_bvalid='0;
        v_axi_arready='0;
        v_axi_rdata='0;
        v_axi_rresp='0;
        v_axi_rid='0;
        v_axi_rlast='0;
        v_axi_rvalid='0;
        rope_axi_arready='0;
        rope_axi_rdata='0;
        rope_axi_rresp='0;
        rope_axi_rid='0;
        rope_axi_rlast='0;
        rope_axi_rvalid='0;
        bind_valid='0;
        bind_cache_base='0;
        bind_region_base='0;
        bind_region_bytes='0;
        bind_capacity='0;
        bind_sequence='0;
        bind_epoch='0;
        kv_axi_awready='0;
        kv_axi_wready='0;
        kv_axi_bid='0;
        kv_axi_bresp='0;
        kv_axi_bvalid='0;
        kv_axi_arready='0;
        kv_axi_rdata='0;
        kv_axi_rresp='0;
        kv_axi_rid='0;
        kv_axi_rlast='0;
        kv_axi_rvalid='0;
        bind_completion_ready='0;
        cancel_valid='0;
        completion_ready='0;
        tail_axi_arready='0;
        tail_axi_rdata='0;
        tail_axi_rresp='0;
        tail_axi_rid='0;
        tail_axi_rlast='0;
        tail_axi_rvalid='0;
        tail_axi_awready='0;
        tail_axi_wready='0;
        tail_axi_bid='0;
        tail_axi_bresp='0;
        tail_axi_bvalid='0;
        tail_axi_arready='1;
        tail_axi_rdata='1;
        tail_axi_rresp='1;
        tail_axi_rid='1;
        tail_axi_rlast='1;
        tail_axi_rvalid='1;
        tail_axi_awready='1;
        tail_axi_wready='1;
        tail_axi_bid='1;
        tail_axi_bresp='1;
        tail_axi_bvalid='1;
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        repeat(3)tick();rst_n=1;tick();
        mode=0;target=0;wanted_status=0;
        for(integer n=0;n<4;n=n+1)begin
            defaults();admission();wait_response();release_job();
            // Reuse the same binding and committed layer prefix without a reset.
            command_sequence=wanted_sequence;command_epoch=wanted_epoch;command_source_sequence=wanted_sequence;
            command_source_epoch=wanted_epoch;command_source_valid=1;command_layer=wanted_layer;
            command_position=1;command_coefficient_position=1;command_lease_base=64'h8000000012340100;
            addr_driver=saved_address;base_driver=saved_base;bytes_driver=saved_bytes;
            admission();wait_response();release_job();
        end
        // Reject each malformed allocation before any arithmetic peer starts.
        mode=1;wanted_status=1;
        for(integer n=0;n<24;n=n+1)begin
            defaults();target=n;bytes_driver[n]=1;admission();wait_response();
            `CHECK(dut.front.starts==start_front && dut.projection.starts==start_o && dut.post.starts==start_post && dut.ffn.starts==start_ffn,"preflight allowed malformed allocation")
            release_job();
        end
        // Shape certificates are checked at every boundary, including the real lease.
        mode=2;wanted_status=5;
        for(target=1;target<=4;target=target+1)begin
            defaults();case(target)
                1:dut.front.cfg_badshape=1;2:dut.projection.cfg_badshape=1;3:dut.post.cfg_badshape=1;4:dut.ffn.cfg_badshape=1;
            endcase
            admission();wait_response();`CHECK(completion_stage==target,"bad shape reached a later stage") release_job();
        end
        for(mode=3;mode<=4;mode=mode+1)begin
            wanted_status=mode==3 ? 7:2;
            for(target=1;target<=4;target=target+1)begin defaults();admission();wait_response();`CHECK(injected,"lifecycle fault not injected") release_job();end
        end
        mode=5;wanted_status=8;
        for(target=1;target<=3;target=target+1)begin
            defaults();case(target)1:dut.front.cfg_block=1;2:dut.projection.cfg_block=1;3:dut.post.cfg_block=1;endcase
            admission();wait_response();release_job();
        end
        // Held success is revoked immediately if a retained peer/cache changes.
        mode=6;wanted_status=0;
        for(target=0;target<4;target=target+1)begin
            defaults();admission();wait_response();
            case(target)0:dut.front.cfg_cache_fault=1;1:dut.projection.cfg_revoke=1;2:dut.post.cfg_revoke=1;3:dut.ffn.cfg_revoke=1;endcase
            #1;`CHECK(!result_valid,"late fault visible through result_valid")
            tick();`CHECK(completion_revoked && completion_status==0 && !result_valid,"late fault not sticky or changed status")
            dut.front.cfg_cache_fault=0;dut.projection.cfg_revoke=0;dut.post.cfg_revoke=0;dut.ffn.cfg_revoke=0;
            tick();`CHECK(!result_valid && completion_revoked,"transient late fault forgotten") release_job();
        end
        mode=7;wanted_status=11;
        for(target=1;target<=4;target=target+1)begin
            defaults();case(target)1:dut.front.cfg_foreign=1;2:dut.projection.cfg_foreign=1;3:dut.post.cfg_foreign=1;4:dut.ffn.cfg_foreign=1;endcase
            admission();wait_response();`CHECK(injected,"foreign identity not injected") release_job();
        end
        $display("LAYER_PASS seed=%0d jobs=%0d cycles=%0d arithmetic=synthetic",seed,jobs,cycle);$finish;
    end

endmodule
`undef CHECK
