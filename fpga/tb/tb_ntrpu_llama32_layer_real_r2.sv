`timescale 1ns/1ps
`default_nettype none
`define CHECK(C,M) if((C)!==1'b1) $fatal(1,"LREAL_FAIL cycle=%0d %s",cycle,M);
module tb_ntrpu_llama32_layer_real_r2;
    localparam integer FRAMES=1017348;
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
    wire [63:0] command_addr [0:23];
    reg [23:0][63:0] drive_command_addr;
    for(genvar g_command_addr=0;g_command_addr<24;g_command_addr++)assign command_addr[g_command_addr]=drive_command_addr[g_command_addr];
    wire [63:0] command_base [0:23];
    reg [23:0][63:0] drive_command_base;
    for(genvar g_command_base=0;g_command_base<24;g_command_base++)assign command_base[g_command_base]=drive_command_base[g_command_base];
    wire [63:0] command_bytes [0:23];
    reg [23:0][63:0] drive_command_bytes;
    for(genvar g_command_bytes=0;g_command_bytes<24;g_command_bytes++)assign command_bytes[g_command_bytes]=drive_command_bytes[g_command_bytes];
    wire  cache_exclusive;
    reg [63:0] command_source_sequence;
    reg [63:0] command_source_epoch;
    reg  command_source_valid;
    wire  norm_tag_exhausted;
    wire  qkv_tags_exhausted;
    wire [63:0] input_axi_araddr;
    assign bus_araddr[0]=input_axi_araddr;
    wire [7:0] input_axi_arlen;
    assign bus_arlen[0]=input_axi_arlen;
    wire [2:0] input_axi_arsize;
    assign bus_arsize[0]=input_axi_arsize;
    wire [2:0] input_axi_arprot;
    assign bus_arprot[0]=input_axi_arprot;
    wire [1:0] input_axi_arburst;
    assign bus_arburst[0]=input_axi_arburst;
    wire [3:0] input_axi_arid;
    assign bus_arid[0]=input_axi_arid;
    wire [3:0] input_axi_arcache;
    assign bus_arcache[0]=input_axi_arcache;
    wire  input_axi_arlock;
    assign bus_arlock[0]=input_axi_arlock;
    wire  input_axi_arvalid;
    assign bus_arvalid[0]=input_axi_arvalid;
    wire  input_axi_arready;
    assign input_axi_arready=bus_arready[0];
    wire [511:0] input_axi_rdata;
    assign input_axi_rdata=bus_rdata[0];
    wire [1:0] input_axi_rresp;
    assign input_axi_rresp=bus_rresp[0];
    wire [3:0] input_axi_rid;
    assign input_axi_rid=bus_rid[0];
    wire  input_axi_rlast;
    assign input_axi_rlast=bus_rlast[0];
    wire  input_axi_rvalid;
    assign input_axi_rvalid=bus_rvalid[0];
    wire  input_axi_rready;
    assign bus_rready[0]=input_axi_rready;
    wire [63:0] input_axi_awaddr;
    assign bus_awaddr[0]=input_axi_awaddr;
    wire [7:0] input_axi_awlen;
    assign bus_awlen[0]=input_axi_awlen;
    wire [2:0] input_axi_awsize;
    assign bus_awsize[0]=input_axi_awsize;
    wire [2:0] input_axi_awprot;
    assign bus_awprot[0]=input_axi_awprot;
    wire [1:0] input_axi_awburst;
    assign bus_awburst[0]=input_axi_awburst;
    wire [3:0] input_axi_awid;
    assign bus_awid[0]=input_axi_awid;
    wire [3:0] input_axi_awcache;
    assign bus_awcache[0]=input_axi_awcache;
    wire  input_axi_awlock;
    assign bus_awlock[0]=input_axi_awlock;
    wire  input_axi_awvalid;
    assign bus_awvalid[0]=input_axi_awvalid;
    wire  input_axi_awready;
    assign input_axi_awready=bus_awready[0];
    wire [511:0] input_axi_wdata;
    assign bus_wdata[0]=input_axi_wdata;
    wire [63:0] input_axi_wstrb;
    assign bus_wstrb[0]=input_axi_wstrb;
    wire  input_axi_wlast;
    assign bus_wlast[0]=input_axi_wlast;
    wire  input_axi_wvalid;
    assign bus_wvalid[0]=input_axi_wvalid;
    wire  input_axi_wready;
    assign input_axi_wready=bus_wready[0];
    wire [3:0] input_axi_bid;
    assign input_axi_bid=bus_bid[0];
    wire [1:0] input_axi_bresp;
    assign input_axi_bresp=bus_bresp[0];
    wire  input_axi_bvalid;
    assign input_axi_bvalid=bus_bvalid[0];
    wire  input_axi_bready;
    assign bus_bready[0]=input_axi_bready;
    wire [63:0] v_axi_araddr;
    assign bus_araddr[2]=v_axi_araddr;
    wire [7:0] v_axi_arlen;
    assign bus_arlen[2]=v_axi_arlen;
    wire [2:0] v_axi_arsize;
    assign bus_arsize[2]=v_axi_arsize;
    wire [2:0] v_axi_arprot;
    assign bus_arprot[2]=v_axi_arprot;
    wire [1:0] v_axi_arburst;
    assign bus_arburst[2]=v_axi_arburst;
    wire [3:0] v_axi_arid;
    assign bus_arid[2]=v_axi_arid;
    wire [3:0] v_axi_arcache;
    assign bus_arcache[2]=v_axi_arcache;
    wire  v_axi_arlock;
    assign bus_arlock[2]=v_axi_arlock;
    wire  v_axi_arvalid;
    assign bus_arvalid[2]=v_axi_arvalid;
    wire  v_axi_arready;
    assign v_axi_arready=bus_arready[2];
    wire [511:0] v_axi_rdata;
    assign v_axi_rdata=bus_rdata[2];
    wire [1:0] v_axi_rresp;
    assign v_axi_rresp=bus_rresp[2];
    wire [3:0] v_axi_rid;
    assign v_axi_rid=bus_rid[2];
    wire  v_axi_rlast;
    assign v_axi_rlast=bus_rlast[2];
    wire  v_axi_rvalid;
    assign v_axi_rvalid=bus_rvalid[2];
    wire  v_axi_rready;
    assign bus_rready[2]=v_axi_rready;
    wire [63:0] rope_axi_araddr;
    assign bus_araddr[1]=rope_axi_araddr;
    wire [7:0] rope_axi_arlen;
    assign bus_arlen[1]=rope_axi_arlen;
    wire [2:0] rope_axi_arsize;
    assign bus_arsize[1]=rope_axi_arsize;
    wire [2:0] rope_axi_arprot;
    assign bus_arprot[1]=rope_axi_arprot;
    wire [1:0] rope_axi_arburst;
    assign bus_arburst[1]=rope_axi_arburst;
    wire [3:0] rope_axi_arid;
    assign bus_arid[1]=rope_axi_arid;
    wire [3:0] rope_axi_arcache;
    assign bus_arcache[1]=rope_axi_arcache;
    wire  rope_axi_arlock;
    assign bus_arlock[1]=rope_axi_arlock;
    wire  rope_axi_arvalid;
    assign bus_arvalid[1]=rope_axi_arvalid;
    wire  rope_axi_arready;
    assign rope_axi_arready=bus_arready[1];
    wire [511:0] rope_axi_rdata;
    assign rope_axi_rdata=bus_rdata[1];
    wire [1:0] rope_axi_rresp;
    assign rope_axi_rresp=bus_rresp[1];
    wire [3:0] rope_axi_rid;
    assign rope_axi_rid=bus_rid[1];
    wire  rope_axi_rlast;
    assign rope_axi_rlast=bus_rlast[1];
    wire  rope_axi_rvalid;
    assign rope_axi_rvalid=bus_rvalid[1];
    wire  rope_axi_rready;
    assign bus_rready[1]=rope_axi_rready;
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
    assign bus_awaddr[3]=kv_axi_awaddr;
    wire [7:0] kv_axi_awlen;
    assign bus_awlen[3]=kv_axi_awlen;
    wire [2:0] kv_axi_awsize;
    assign bus_awsize[3]=kv_axi_awsize;
    wire [2:0] kv_axi_awprot;
    assign bus_awprot[3]=kv_axi_awprot;
    wire [1:0] kv_axi_awburst;
    assign bus_awburst[3]=kv_axi_awburst;
    wire [3:0] kv_axi_awid;
    assign bus_awid[3]=kv_axi_awid;
    wire [3:0] kv_axi_awcache;
    assign bus_awcache[3]=kv_axi_awcache;
    wire  kv_axi_awlock;
    assign bus_awlock[3]=kv_axi_awlock;
    wire  kv_axi_awvalid;
    assign bus_awvalid[3]=kv_axi_awvalid;
    wire  kv_axi_awready;
    assign kv_axi_awready=bus_awready[3];
    wire [511:0] kv_axi_wdata;
    assign bus_wdata[3]=kv_axi_wdata;
    wire [63:0] kv_axi_wstrb;
    assign bus_wstrb[3]=kv_axi_wstrb;
    wire  kv_axi_wlast;
    assign bus_wlast[3]=kv_axi_wlast;
    wire  kv_axi_wvalid;
    assign bus_wvalid[3]=kv_axi_wvalid;
    wire  kv_axi_wready;
    assign kv_axi_wready=bus_wready[3];
    wire [3:0] kv_axi_bid;
    assign kv_axi_bid=bus_bid[3];
    wire [1:0] kv_axi_bresp;
    assign kv_axi_bresp=bus_bresp[3];
    wire  kv_axi_bvalid;
    assign kv_axi_bvalid=bus_bvalid[3];
    wire  kv_axi_bready;
    assign bus_bready[3]=kv_axi_bready;
    wire [63:0] kv_axi_araddr;
    assign bus_araddr[3]=kv_axi_araddr;
    wire [7:0] kv_axi_arlen;
    assign bus_arlen[3]=kv_axi_arlen;
    wire [2:0] kv_axi_arsize;
    assign bus_arsize[3]=kv_axi_arsize;
    wire [2:0] kv_axi_arprot;
    assign bus_arprot[3]=kv_axi_arprot;
    wire [1:0] kv_axi_arburst;
    assign bus_arburst[3]=kv_axi_arburst;
    wire [3:0] kv_axi_arid;
    assign bus_arid[3]=kv_axi_arid;
    wire [3:0] kv_axi_arcache;
    assign bus_arcache[3]=kv_axi_arcache;
    wire  kv_axi_arlock;
    assign bus_arlock[3]=kv_axi_arlock;
    wire  kv_axi_arvalid;
    assign bus_arvalid[3]=kv_axi_arvalid;
    wire  kv_axi_arready;
    assign kv_axi_arready=bus_arready[3];
    wire [511:0] kv_axi_rdata;
    assign kv_axi_rdata=bus_rdata[3];
    wire [1:0] kv_axi_rresp;
    assign kv_axi_rresp=bus_rresp[3];
    wire [3:0] kv_axi_rid;
    assign kv_axi_rid=bus_rid[3];
    wire  kv_axi_rlast;
    assign kv_axi_rlast=bus_rlast[3];
    wire  kv_axi_rvalid;
    assign kv_axi_rvalid=bus_rvalid[3];
    wire  kv_axi_rready;
    assign bus_rready[3]=kv_axi_rready;
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
    assign bus_araddr[4]=tail_axi_araddr;
    wire [7:0] tail_axi_arlen;
    assign bus_arlen[4]=tail_axi_arlen;
    wire [2:0] tail_axi_arsize;
    assign bus_arsize[4]=tail_axi_arsize;
    wire [2:0] tail_axi_arprot;
    assign bus_arprot[4]=tail_axi_arprot;
    wire [1:0] tail_axi_arburst;
    assign bus_arburst[4]=tail_axi_arburst;
    wire [3:0] tail_axi_arid;
    assign bus_arid[4]=tail_axi_arid;
    wire [3:0] tail_axi_arcache;
    assign bus_arcache[4]=tail_axi_arcache;
    wire  tail_axi_arlock;
    assign bus_arlock[4]=tail_axi_arlock;
    wire  tail_axi_arvalid;
    assign bus_arvalid[4]=tail_axi_arvalid;
    wire  tail_axi_arready;
    assign tail_axi_arready=bus_arready[4];
    wire [511:0] tail_axi_rdata;
    assign tail_axi_rdata=bus_rdata[4];
    wire [1:0] tail_axi_rresp;
    assign tail_axi_rresp=bus_rresp[4];
    wire [3:0] tail_axi_rid;
    assign tail_axi_rid=bus_rid[4];
    wire  tail_axi_rlast;
    assign tail_axi_rlast=bus_rlast[4];
    wire  tail_axi_rvalid;
    assign tail_axi_rvalid=bus_rvalid[4];
    wire  tail_axi_rready;
    assign bus_rready[4]=tail_axi_rready;
    wire [63:0] tail_axi_awaddr;
    assign bus_awaddr[4]=tail_axi_awaddr;
    wire [7:0] tail_axi_awlen;
    assign bus_awlen[4]=tail_axi_awlen;
    wire [2:0] tail_axi_awsize;
    assign bus_awsize[4]=tail_axi_awsize;
    wire [2:0] tail_axi_awprot;
    assign bus_awprot[4]=tail_axi_awprot;
    wire [1:0] tail_axi_awburst;
    assign bus_awburst[4]=tail_axi_awburst;
    wire [3:0] tail_axi_awid;
    assign bus_awid[4]=tail_axi_awid;
    wire [3:0] tail_axi_awcache;
    assign bus_awcache[4]=tail_axi_awcache;
    wire  tail_axi_awlock;
    assign bus_awlock[4]=tail_axi_awlock;
    wire  tail_axi_awvalid;
    assign bus_awvalid[4]=tail_axi_awvalid;
    wire  tail_axi_awready;
    assign tail_axi_awready=bus_awready[4];
    wire [511:0] tail_axi_wdata;
    assign bus_wdata[4]=tail_axi_wdata;
    wire [63:0] tail_axi_wstrb;
    assign bus_wstrb[4]=tail_axi_wstrb;
    wire  tail_axi_wlast;
    assign bus_wlast[4]=tail_axi_wlast;
    wire  tail_axi_wvalid;
    assign bus_wvalid[4]=tail_axi_wvalid;
    wire  tail_axi_wready;
    assign tail_axi_wready=bus_wready[4];
    wire [3:0] tail_axi_bid;
    assign tail_axi_bid=bus_bid[4];
    wire [1:0] tail_axi_bresp;
    assign tail_axi_bresp=bus_bresp[4];
    wire  tail_axi_bvalid;
    assign tail_axi_bvalid=bus_bvalid[4];
    wire  tail_axi_bready;
    assign bus_bready[4]=tail_axi_bready;
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
    wire [63:0] bus_araddr[0:4];
    wire [7:0] bus_arlen[0:4];
    wire [2:0] bus_arsize[0:4];
    wire [2:0] bus_arprot[0:4];
    wire [1:0] bus_arburst[0:4];
    wire [3:0] bus_arid[0:4];
    wire [3:0] bus_arcache[0:4];
    wire  bus_arlock[0:4];
    wire  bus_arvalid[0:4];
    reg  bus_arready[0:4];
    reg [511:0] bus_rdata[0:4];
    reg [1:0] bus_rresp[0:4];
    reg [3:0] bus_rid[0:4];
    reg  bus_rlast[0:4];
    reg  bus_rvalid[0:4];
    wire  bus_rready[0:4];
    wire [63:0] bus_awaddr[0:4];
    assign bus_awaddr[1]='0;
    assign bus_awaddr[2]='0;
    wire [7:0] bus_awlen[0:4];
    assign bus_awlen[1]='0;
    assign bus_awlen[2]='0;
    wire [2:0] bus_awsize[0:4];
    assign bus_awsize[1]='0;
    assign bus_awsize[2]='0;
    wire [2:0] bus_awprot[0:4];
    assign bus_awprot[1]='0;
    assign bus_awprot[2]='0;
    wire [1:0] bus_awburst[0:4];
    assign bus_awburst[1]='0;
    assign bus_awburst[2]='0;
    wire [3:0] bus_awid[0:4];
    assign bus_awid[1]='0;
    assign bus_awid[2]='0;
    wire [3:0] bus_awcache[0:4];
    assign bus_awcache[1]='0;
    assign bus_awcache[2]='0;
    wire  bus_awlock[0:4];
    assign bus_awlock[1]='0;
    assign bus_awlock[2]='0;
    wire  bus_awvalid[0:4];
    assign bus_awvalid[1]='0;
    assign bus_awvalid[2]='0;
    reg  bus_awready[0:4];
    wire [511:0] bus_wdata[0:4];
    assign bus_wdata[1]='0;
    assign bus_wdata[2]='0;
    wire [63:0] bus_wstrb[0:4];
    assign bus_wstrb[1]='0;
    assign bus_wstrb[2]='0;
    wire  bus_wlast[0:4];
    assign bus_wlast[1]='0;
    assign bus_wlast[2]='0;
    wire  bus_wvalid[0:4];
    assign bus_wvalid[1]='0;
    assign bus_wvalid[2]='0;
    reg  bus_wready[0:4];
    reg [3:0] bus_bid[0:4];
    reg [1:0] bus_bresp[0:4];
    reg  bus_bvalid[0:4];
    wire  bus_bready[0:4];
    assign bus_bready[1]='0;
    assign bus_bready[2]='0;
    ntrpu_llama32_layer_r1 dut(
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

    localparam [63:0] SEQ=64'h1234567800000001,EPOCH=64'hfedcba9800000001,LEASE=64'h8000000012340000;
    reg [511:0] memory[0:FRAMES-1],golden[0:FRAMES-1];
    bit valid_frame[0:FRAMES-1],written[0:FRAMES-1];
    integer starts[0:24],sizes[0:24];bit writable[0:24];
    reg [31:0] rope_expected[0:2559],attention_expected[0:2047];
    reg [63:0] read_hash[0:4][0:24];integer region_reads[0:4][0:24];
    integer writes[0:24];integer cycle=0,seed=1,limit,public_cycle,total_writes=0;
    integer r_count[0:4],r_index[0:4],w_count[0:4],w_beats[0:4];
    integer ars[0:4],rs[0:4],aws[0:4],ws[0:4],bs[0:4],r_stalls[0:4],w_stalls[0:4];
    bit read_active[0:4],aw_active[0:4],ar_stalled[0:4],aw_stalled[0:4],w_stalled[0:4];
    reg [63:0] r_base[0:4],w_base[0:4];reg [3:0] read_id[0:4],write_id[0:4];
    reg [511:0] w_buffer[0:4][0:63];bit w_last[0:4][0:63];
    reg [88:0] old_ar[0:4],old_aw[0:4];reg [576:0] old_w[0:4];
    reg [3:0] stage_seen=0,stage_started=0,stage_acked=0;
    integer heads=0,head_acks=0,head=-1,rope_words=0,at_starts=0,at_dones=0,attention_words=0;
    integer source_reads=0,hls_inputs=0;bit source_pending=0;integer source_address=0;
    bit hls_input_pending=0;integer hls_input_address=0;reg [2047:0] source_seen=0;
    bit rp_running=0,at_running=0;reg [103:0] state_seen=0;reg [63:0] output_seen=0;
    string fixture;
    always #2.5 clk=~clk;
    always @(posedge clk)cycle<=cycle+1;
    function automatic [63:0] region_address(input integer r);return (64'(r+1)<<40)+4032;endfunction
    function automatic integer region_index(input [63:0] a);
        integer r;r=int'(a>>40)-1;
        if(r<0 || r>24 || a[5:0]!=0)$fatal(1,"LREAL_FAIL unmapped or unaligned address %016h",a);
        if(a<region_address(r) || a+64>region_address(r)+64'(sizes[r]*64))$fatal(1,"LREAL_FAIL address outside payload %016h",a);
        return r;
    endfunction
    function automatic integer frame_index(input [63:0] a);
        integer r;r=region_index(a);return starts[r]+int'((a-region_address(r))/64);
    endfunction
    function automatic [511:0] read_frame(input [63:0] a);
        integer index;index=frame_index(a);
        if(!valid_frame[index])$fatal(1,"LREAL_FAIL read before actual B commit %016h",a);
        return memory[index];
    endfunction
    function automatic bit read_permission(input integer p,r);
        case(p)
            0:return r<=5;
            1:return r==6 || r==7 || r==9;
            2:return r==8;
            3:return r==24;
            4:case(tail_selected)
                0:return r==10 || r==11;
                1:return r==0 || r==12 || r==13;
                2:return r==14 || (r>=15 && r<=22);
                default:return 0;
            endcase
            default:return 0;
        endcase
    endfunction
    function automatic bit write_permission(input integer p,r);
        case(p)
            0:return r==2 || (r>=6 && r<=8);
            3:return r==24;
            4:case(tail_selected)
                0:return r==11 || r==12;
                1:return r==14 || r==15;
                2:return r>=19 && r<=23;
                default:return 0;
            endcase
            default:return 0;
        endcase
    endfunction
    // Only the actual DUT write payload is installed, and only after B accepts.
    // Expected images are assertions; read_frame never returns them.
    always @(negedge clk)begin
        for(integer p=0;p<5;p=p+1)begin
            bus_arready[p]=rst_n && !read_active[p] && (cycle+p+seed)%5!=0;
            bus_awready[p]=rst_n && !aw_active[p] && (cycle+p+seed)%4!=0;
            bus_wready[p]=rst_n && w_count[p]<64 && (cycle+p+seed)%3!=0;
            if(rst_n)begin
                if(!bus_rvalid[p] && read_active[p] && (cycle+p+seed)%4!=0)begin
                    bus_rdata[p]=read_frame(r_base[p]+64'(r_index[p]*64));bus_rid[p]=read_id[p];
                    bus_rresp[p]=0;bus_rlast[p]=r_index[p]==r_count[p]-1;bus_rvalid[p]=1;
                end
                if(!bus_bvalid[p] && aw_active[p] && w_count[p]==w_beats[p] && (cycle+p+seed)%3!=0)begin
                    bus_bid[p]=write_id[p];bus_bresp[p]=0;bus_bvalid[p]=1;
                end
            end
        end
    end
    always @(posedge clk)begin: memory_service
        integer r,index,off;reg [63:0] a;reg [511:0] data;
        if(rst_n)for(integer p=0;p<5;p=p+1)begin
            if(ar_stalled[p])`CHECK(bus_arvalid[p] && {bus_araddr[p],bus_arlen[p],bus_arsize[p],bus_arburst[p],bus_arid[p],bus_arprot[p],bus_arcache[p],bus_arlock[p]}===old_ar[p],"AR changed under backpressure")
            if(aw_stalled[p])`CHECK(bus_awvalid[p] && {bus_awaddr[p],bus_awlen[p],bus_awsize[p],bus_awburst[p],bus_awid[p],bus_awprot[p],bus_awcache[p],bus_awlock[p]}===old_aw[p],"AW changed under backpressure")
            if(w_stalled[p])`CHECK(bus_wvalid[p] && {bus_wdata[p],bus_wstrb[p],bus_wlast[p]}===old_w[p],"W changed under backpressure")
            ar_stalled[p]=bus_arvalid[p] && !bus_arready[p];aw_stalled[p]=bus_awvalid[p] && !bus_awready[p];w_stalled[p]=bus_wvalid[p] && !bus_wready[p];
            old_ar[p]={bus_araddr[p],bus_arlen[p],bus_arsize[p],bus_arburst[p],bus_arid[p],bus_arprot[p],bus_arcache[p],bus_arlock[p]};
            old_aw[p]={bus_awaddr[p],bus_awlen[p],bus_awsize[p],bus_awburst[p],bus_awid[p],bus_awprot[p],bus_awcache[p],bus_awlock[p]};
            old_w[p]={bus_wdata[p],bus_wstrb[p],bus_wlast[p]};
            if(ar_stalled[p])r_stalls[p]++;if(w_stalled[p])w_stalls[p]++;
            if(bus_arvalid[p] && bus_arready[p])begin
                a=bus_araddr[p];r=region_index(a);off=int'((a-region_address(r))/64);
                `CHECK(!read_active[p] && bus_arlen[p]<64 && bus_arsize[p]==6 && bus_arburst[p]==1,"AR protocol")
                `CHECK(off+int'(bus_arlen[p])+1<=sizes[r] && int'(a[11:0])+(int'(bus_arlen[p])+1)*64<=4096,"AR bounds/page")
                `CHECK(read_permission(p,r),"Wrong master or tail owner read")
                for(integer i=0;i<=int'(bus_arlen[p]);i++)begin `CHECK(valid_frame[starts[r]+off+i],"AR before B commit") end
                r_base[p]=a;r_count[p]=int'(bus_arlen[p])+1;r_index[p]=0;read_id[p]=bus_arid[p];read_active[p]=1;
                $display("LREAL_AR cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,ars[p],r,off*64,r_count[p]);ars[p]++;
            end
            if(bus_rvalid[p] && bus_rready[p])begin
                a=r_base[p]+64'(r_index[p]*64);r=region_index(a);data=read_frame(a);
                `CHECK(read_active[p] && bus_rdata[p]===data && bus_rid[p]==read_id[p] && bus_rresp[p]==0 && bus_rlast[p]==(r_index[p]==r_count[p]-1),"R payload/framing")
                for(integer lane=0;lane<8;lane++)read_hash[p][r]=(read_hash[p][r]^data[lane*64+:64])*64'h100000001b3;
                if(writable[r])$display("LREAL_R cycle=%0d port=%0d region=%0d offset=%0d data=%0128h",cycle,p,r,int'(a-region_address(r)),data);
                region_reads[p][r]++;rs[p]++;r_index[p]++;bus_rvalid[p]<=0;
                if(r_index[p]==r_count[p])read_active[p]=0;
            end
            if(bus_awvalid[p] && bus_awready[p])begin
                a=bus_awaddr[p];r=region_index(a);off=int'((a-region_address(r))/64);
                `CHECK(!aw_active[p] && writable[r] && write_permission(p,r),"AW ownership")
                `CHECK(bus_awlen[p]<64 && bus_awsize[p]==6 && bus_awburst[p]==1,"AW protocol")
                `CHECK(off+int'(bus_awlen[p])+1<=sizes[r] && int'(a[11:0])+(int'(bus_awlen[p])+1)*64<=4096,"AW bounds/page")
                w_base[p]=a;w_beats[p]=int'(bus_awlen[p])+1;write_id[p]=bus_awid[p];aw_active[p]=1;
                $display("LREAL_AW cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,aws[p],r,off*64,w_beats[p]);aws[p]++;
            end
            if(bus_wvalid[p] && bus_wready[p])begin
                `CHECK(w_count[p]<64 && bus_wstrb[p]==64'hffffffffffffffff,"W extent/strobe")
                w_buffer[p][w_count[p]]=bus_wdata[p];w_last[p][w_count[p]]=bus_wlast[p];w_count[p]++;ws[p]++;
                $display("LREAL_W cycle=%0d port=%0d index=%0d last=%0d data=%0128h",cycle,p,ws[p]-1,bus_wlast[p],bus_wdata[p]);
            end
            if(bus_bvalid[p] && bus_bready[p])begin
                `CHECK(aw_active[p] && w_count[p]==w_beats[p] && bus_bresp[p]==0 && bus_bid[p]==write_id[p],"B before full write")
                r=region_index(w_base[p]);off=int'((w_base[p]-region_address(r))/64);
                for(integer i=0;i<w_beats[p];i++)begin
                    index=starts[r]+off+i;data=w_buffer[p][i];
                    `CHECK(!written[index] && w_last[p][i]==(i==w_beats[p]-1),"Duplicate commit or WLAST")
                    `CHECK(data===golden[index],"Actual intermediate differs from immutable reference")
                    memory[index]=data;valid_frame[index]=1;written[index]=1;writes[r]++;total_writes++;
                end
                $display("LREAL_B cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,bs[p],r,off*64,w_beats[p]);
                bs[p]++;aw_active[p]=0;w_count[p]=0;bus_bvalid[p]<=0;
            end
        end
    end
`define RP dut.front.consumer.front.rope
`define AT dut.front.consumer.attention.engine
    always @(posedge clk)begin: observe_real_arithmetic
        if(rst_n)begin
            if(busy)begin `CHECK(!command_ready && !bind_ready && cache_exclusive,"Global ownership lost") end
            if(`RP.hls_start && !rp_running)begin
                `CHECK(rs[1]==164 && rope_words==0,"RoPE must consume all actual Q/K/coefficient reads")
                rp_running=1;$display("LREAL_ROPE_START cycle=%0d",cycle);
            end
            if(`RP.hls_output_ce && `RP.hls_output_we)begin
                `CHECK(rp_running && `RP.hls_output_address==rope_words && rope_words<2560,"RoPE output order")
                `CHECK(`RP.hls_output_d===rope_expected[rope_words],"Real RoPE arithmetic")
                $display("LREAL_ROPE cycle=%0d index=%0d data=%08h",cycle,rope_words,`RP.hls_output_d);rope_words++;
            end
            if(`RP.hls_done && rp_running)begin
                `CHECK(rope_words==2560 && `RP.hls_ready && `RP.hls_return==0,"RoPE completion")
                rp_running=0;$display("LREAL_ROPE_DONE cycle=%0d",cycle);
            end
            if(dut.front.consumer.attention.head_command_valid && dut.front.consumer.attention.head_command_ready)begin
                `CHECK(heads==head_acks && heads<32 && dut.front.consumer.attention.head_command_head==heads,"Attention head ownership/order")
                head=heads;heads++;
                `CHECK(dut.front.consumer.attention.head_command_sequence==SEQ && dut.front.consumer.attention.head_command_epoch==EPOCH,"Attention head identity")
                `CHECK(dut.front.consumer.attention.head_command_lease==LEASE+64'(head) && cache_query_committed_tokens==1,"Attention committed prefix/lease")
                $display("LREAL_HEAD cycle=%0d head=%0d",cycle,head);
            end
            if(`AT.hls_start && !at_running)begin
                `CHECK(!at_running && heads==head_acks+1 && `AT.hls_tile_start==0 && `AT.hls_tile_count==1,"Attention tile ownership")
                at_running=1;at_starts++;state_seen=0;output_seen=0;
            end
            if(`AT.hls_state_out_ce && `AT.hls_state_out_we)begin
                `CHECK(at_running && `AT.hls_state_out_address<104 && !state_seen[`AT.hls_state_out_address],"Unique attention state word")
                state_seen[`AT.hls_state_out_address]=1;
                $display("LREAL_STATE cycle=%0d head=%0d index=%0d data=%016h",cycle,head,`AT.hls_state_out_address,`AT.hls_state_out_d);
            end
            if(`AT.hls_output_ce && `AT.hls_output_we)begin
                `CHECK(at_running && `AT.hls_output_address<64 && !output_seen[`AT.hls_output_address],"Unique attention output")
                `CHECK(`AT.hls_output_d===attention_expected[head*64+int'(`AT.hls_output_address)],"Attention FP32 bytes for downstream quantization")
                output_seen[`AT.hls_output_address]=1;attention_words++;
                $display("LREAL_ATTENTION cycle=%0d head=%0d index=%0d data=%08h",cycle,head,`AT.hls_output_address,`AT.hls_output_d);
            end
            if(`AT.hls_done && at_running)begin
                `CHECK((&state_seen) && (&output_seen) && `AT.hls_return==0 && `AT.hls_ready,"Complete attention state/output")
                at_running=0;at_dones++;
            end
            if(dut.front.consumer.attention.head_completion_valid && dut.front.consumer.attention.head_completion_ready)begin
                `CHECK(!at_running && heads==head_acks+1,"Attention head drain")head_acks++;
            end
            if(hls_input_pending)begin
                `CHECK(dut.projection.quantize_store.hls_input_q===attention_expected[hls_input_address],"Actual synchronous HLS input from retained front RAM")
                $display("LREAL_HLS_INPUT cycle=%0d index=%0d data=%08h",cycle,hls_input_address,dut.projection.quantize_store.hls_input_q);hls_inputs++;
            end
            hls_input_pending=dut.front_result_ce;hls_input_address=int'(dut.front_result_address);
            if(dut.front_result_ce)begin
                `CHECK(!source_pending && dut.front_result_valid && dut.front_completion_valid && source_reads<8192,"Projection source RAM ownership/order")
                source_pending=1;source_address=int'(dut.front_result_address);
            end
            if(dut.front_command_valid && dut.front_command_ready)begin
                `CHECK(!stage_started[0],"Stage admitted twice")stage_started[0]=1;
                $display("LREAL_START cycle=%0d stage=1",cycle);end
            if(dut.front_completion_valid && !stage_seen[0])begin
                `CHECK(stage_started[0] && dut.front_completion_status==0 && dut.front_completion_sequence==SEQ && dut.front_completion_epoch==EPOCH,"Stage completion certificate")stage_seen[0]=1;
                $display("LREAL_DONE cycle=%0d stage=1",cycle);end
            if(dut.front_completion_valid && dut.front_completion_ready)begin
                `CHECK(stage_seen[0] && !stage_acked[0],"Unowned stage ACK")stage_acked[0]=1;
                `CHECK(source_reads>=2048 && (&source_seen) && hls_inputs==source_reads,"Front ACK before source RAM read")
                $display("LREAL_ACK cycle=%0d stage=1",cycle);end
            if(dut.projection_command_valid && dut.projection_command_ready)begin
                `CHECK(!stage_started[1],"Stage admitted twice")stage_started[1]=1;
                $display("LREAL_START cycle=%0d stage=2",cycle);end
            if(dut.projection_completion_valid && !stage_seen[1])begin
                `CHECK(stage_started[1] && dut.projection_completion_status==0 && dut.projection_completion_sequence==SEQ && dut.projection_completion_epoch==EPOCH,"Stage completion certificate")stage_seen[1]=1;
                $display("LREAL_DONE cycle=%0d stage=2",cycle);end
            if(dut.projection_completion_valid && dut.projection_completion_ready)begin
                `CHECK(stage_seen[1] && !stage_acked[1],"Unowned stage ACK")stage_acked[1]=1;
                `CHECK(cycle>public_cycle && !completion_valid,"Tail ACK before public retirement")
                $display("LREAL_ACK cycle=%0d stage=2",cycle);end
            if(dut.post_command_valid && dut.post_command_ready)begin
                `CHECK(!stage_started[2],"Stage admitted twice")stage_started[2]=1;
                $display("LREAL_START cycle=%0d stage=3",cycle);end
            if(dut.post_completion_valid && !stage_seen[2])begin
                `CHECK(stage_started[2] && dut.post_completion_status==0 && dut.post_completion_sequence==SEQ && dut.post_completion_epoch==EPOCH,"Stage completion certificate")stage_seen[2]=1;
                $display("LREAL_DONE cycle=%0d stage=3",cycle);end
            if(dut.post_completion_valid && dut.post_completion_ready)begin
                `CHECK(stage_seen[2] && !stage_acked[2],"Unowned stage ACK")stage_acked[2]=1;
                `CHECK(cycle>public_cycle && !completion_valid,"Tail ACK before public retirement")
                $display("LREAL_ACK cycle=%0d stage=3",cycle);end
            if(dut.ffn_command_valid && dut.ffn_command_ready)begin
                `CHECK(!stage_started[3],"Stage admitted twice")stage_started[3]=1;
                $display("LREAL_START cycle=%0d stage=4",cycle);end
            if(dut.ffn_completion_valid && !stage_seen[3])begin
                `CHECK(stage_started[3] && dut.ffn_completion_status==0 && dut.ffn_completion_sequence==SEQ && dut.ffn_completion_epoch==EPOCH,"Stage completion certificate")stage_seen[3]=1;
                $display("LREAL_DONE cycle=%0d stage=4",cycle);end
            if(dut.ffn_completion_valid && dut.ffn_completion_ready)begin
                `CHECK(stage_seen[3] && !stage_acked[3],"Unowned stage ACK")stage_acked[3]=1;
                `CHECK(cycle>public_cycle && !completion_valid,"Tail ACK before public retirement")
                $display("LREAL_ACK cycle=%0d stage=4",cycle);end
            if(cycle%1000000==0)$display("LREAL_PROGRESS cycle=%0d stage=%0d writes=%0d",cycle,completion_stage,total_writes);
            `CHECK(cycle<40000000,"global cycle deadline")
            #1;
            if(source_pending)begin
                `CHECK(dut.front_result_q_valid && dut.front_result_q===attention_expected[source_address],"Actual attention RAM reaches O")
                $display("LREAL_SOURCE cycle=%0d index=%0d data=%08h",cycle,source_address,dut.front_result_q);source_seen[source_address]=1;source_reads++;source_pending=0;
            end
        end
    end
    task tick;begin @(posedge clk);#1;@(negedge clk);#0.1;end endtask
    initial begin
        clk=0;
        rst_n=0;
        command_valid=0;
        command_sequence=0;
        command_epoch=0;
        command_lease_base=0;
        expected_epoch=0;
        command_layer=0;
        command_position=0;
        command_coefficient_position=0;
        drive_command_addr=0;
        drive_command_base=0;
        drive_command_bytes=0;
        command_source_sequence=0;
        command_source_epoch=0;
        command_source_valid=0;
        bind_valid=0;
        bind_cache_base=0;
        bind_region_base=0;
        bind_region_bytes=0;
        bind_capacity=0;
        bind_sequence=0;
        bind_epoch=0;
        bind_completion_ready=0;
        cancel_valid=0;
        completion_ready=0;
        bus_arready[0]='0;
        bus_arready[1]='0;
        bus_arready[2]='0;
        bus_arready[3]='0;
        bus_arready[4]='0;
        bus_rdata[0]='0;
        bus_rdata[1]='0;
        bus_rdata[2]='0;
        bus_rdata[3]='0;
        bus_rdata[4]='0;
        bus_rresp[0]='0;
        bus_rresp[1]='0;
        bus_rresp[2]='0;
        bus_rresp[3]='0;
        bus_rresp[4]='0;
        bus_rid[0]='0;
        bus_rid[1]='0;
        bus_rid[2]='0;
        bus_rid[3]='0;
        bus_rid[4]='0;
        bus_rlast[0]='0;
        bus_rlast[1]='0;
        bus_rlast[2]='0;
        bus_rlast[3]='0;
        bus_rlast[4]='0;
        bus_rvalid[0]='0;
        bus_rvalid[1]='0;
        bus_rvalid[2]='0;
        bus_rvalid[3]='0;
        bus_rvalid[4]='0;
        bus_awready[0]='0;
        bus_awready[1]='0;
        bus_awready[2]='0;
        bus_awready[3]='0;
        bus_awready[4]='0;
        bus_wready[0]='0;
        bus_wready[1]='0;
        bus_wready[2]='0;
        bus_wready[3]='0;
        bus_wready[4]='0;
        bus_bid[0]='0;
        bus_bid[1]='0;
        bus_bid[2]='0;
        bus_bid[3]='0;
        bus_bid[4]='0;
        bus_bresp[0]='0;
        bus_bresp[1]='0;
        bus_bresp[2]='0;
        bus_bresp[3]='0;
        bus_bresp[4]='0;
        bus_bvalid[0]='0;
        bus_bvalid[1]='0;
        bus_bvalid[2]='0;
        bus_bvalid[3]='0;
        bus_bvalid[4]='0;
        `CHECK($value$plusargs("FIXTURE=%s",fixture),"Missing frozen fixture")
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        for(integer p=0;p<5;p++)begin
            read_active[p]=0;aw_active[p]=0;r_count[p]=0;r_index[p]=0;w_count[p]=0;w_beats[p]=0;
            ars[p]=0;rs[p]=0;aws[p]=0;ws[p]=0;bs[p]=0;r_stalls[p]=0;w_stalls[p]=0;
            ar_stalled[p]=0;aw_stalled[p]=0;w_stalled[p]=0;
            for(integer r=0;r<25;r++)begin read_hash[p][r]=64'hcbf29ce484222325;region_reads[p][r]=0;end
        end
        starts[0]=0;sizes[0]=128;writable[0]=0;writes[0]=0;
        $readmemh({fixture,"/source-00.hex"},memory,0,127);
        for(integer i=0;i<=127;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[1]=128;sizes[1]=128;writable[1]=0;writes[1]=0;
        $readmemh({fixture,"/source-01.hex"},memory,128,255);
        for(integer i=128;i<=255;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[2]=256;sizes[2]=64;writable[2]=1;writes[2]=0;
        $readmemh({fixture,"/expected-02.hex"},golden,256,319);
        for(integer i=256;i<=319;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[3]=320;sizes[3]=69632;writable[3]=0;writes[3]=0;
        $readmemh({fixture,"/source-03.hex"},memory,320,69951);
        for(integer i=320;i<=69951;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[4]=69952;sizes[4]=17408;writable[4]=0;writes[4]=0;
        $readmemh({fixture,"/source-04.hex"},memory,69952,87359);
        for(integer i=69952;i<=87359;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[5]=87360;sizes[5]=17408;writable[5]=0;writes[5]=0;
        $readmemh({fixture,"/source-05.hex"},memory,87360,104767);
        for(integer i=87360;i<=104767;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[6]=104768;sizes[6]=128;writable[6]=1;writes[6]=0;
        $readmemh({fixture,"/expected-06.hex"},golden,104768,104895);
        for(integer i=104768;i<=104895;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[7]=104896;sizes[7]=32;writable[7]=1;writes[7]=0;
        $readmemh({fixture,"/expected-07.hex"},golden,104896,104927);
        for(integer i=104896;i<=104927;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[8]=104928;sizes[8]=32;writable[8]=1;writes[8]=0;
        $readmemh({fixture,"/expected-08.hex"},golden,104928,104959);
        for(integer i=104928;i<=104959;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[9]=104960;sizes[9]=4;writable[9]=0;writes[9]=0;
        $readmemh({fixture,"/source-09.hex"},memory,104960,104963);
        for(integer i=104960;i<=104963;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[10]=104964;sizes[10]=69632;writable[10]=0;writes[10]=0;
        $readmemh({fixture,"/source-10.hex"},memory,104964,174595);
        for(integer i=104964;i<=174595;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[11]=174596;sizes[11]=64;writable[11]=1;writes[11]=0;
        $readmemh({fixture,"/expected-11.hex"},golden,174596,174659);
        for(integer i=174596;i<=174659;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[12]=174660;sizes[12]=128;writable[12]=1;writes[12]=0;
        $readmemh({fixture,"/expected-12.hex"},golden,174660,174787);
        for(integer i=174660;i<=174787;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[13]=174788;sizes[13]=128;writable[13]=0;writes[13]=0;
        $readmemh({fixture,"/source-13.hex"},memory,174788,174915);
        for(integer i=174788;i<=174915;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[14]=174916;sizes[14]=128;writable[14]=1;writes[14]=0;
        $readmemh({fixture,"/expected-14.hex"},golden,174916,175043);
        for(integer i=174916;i<=175043;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[15]=175044;sizes[15]=64;writable[15]=1;writes[15]=0;
        $readmemh({fixture,"/expected-15.hex"},golden,175044,175107);
        for(integer i=175044;i<=175107;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[16]=175108;sizes[16]=278528;writable[16]=0;writes[16]=0;
        $readmemh({fixture,"/source-16.hex"},memory,175108,453635);
        for(integer i=175108;i<=453635;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[17]=453636;sizes[17]=278528;writable[17]=0;writes[17]=0;
        $readmemh({fixture,"/source-17.hex"},memory,453636,732163);
        for(integer i=453636;i<=732163;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[18]=732164;sizes[18]=278528;writable[18]=0;writes[18]=0;
        $readmemh({fixture,"/source-18.hex"},memory,732164,1010691);
        for(integer i=732164;i<=1010691;i++)begin valid_frame[i]=1;written[i]=0;end
        starts[19]=1010692;sizes[19]=1024;writable[19]=1;writes[19]=0;
        $readmemh({fixture,"/expected-19.hex"},golden,1010692,1011715);
        for(integer i=1010692;i<=1011715;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[20]=1011716;sizes[20]=1024;writable[20]=1;writes[20]=0;
        $readmemh({fixture,"/expected-20.hex"},golden,1011716,1012739);
        for(integer i=1011716;i<=1012739;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[21]=1012740;sizes[21]=256;writable[21]=1;writes[21]=0;
        $readmemh({fixture,"/expected-21.hex"},golden,1012740,1012995);
        for(integer i=1012740;i<=1012995;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[22]=1012996;sizes[22]=128;writable[22]=1;writes[22]=0;
        $readmemh({fixture,"/expected-22.hex"},golden,1012996,1013123);
        for(integer i=1012996;i<=1013123;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[23]=1013124;sizes[23]=128;writable[23]=1;writes[23]=0;
        $readmemh({fixture,"/expected-23.hex"},golden,1013124,1013251);
        for(integer i=1013124;i<=1013251;i++)begin valid_frame[i]=0;written[i]=0;end
        starts[24]=1013252;sizes[24]=4096;writable[24]=1;writes[24]=0;
        $readmemh({fixture,"/expected-24.hex"},golden,1013252,1017347);
        for(integer i=1013252;i<=1017347;i++)begin valid_frame[i]=0;written[i]=0;end
        $readmemh({fixture,"/rope.hex"},rope_expected);$readmemh({fixture,"/attention.hex"},attention_expected);
        repeat(4)tick();rst_n=1;tick();
        expected_epoch=EPOCH;bind_sequence=SEQ;bind_epoch=EPOCH;bind_cache_base=region_address(24);
        bind_region_base=64'd25<<40;bind_region_bytes=262144+4032;bind_capacity=4;bind_valid=1;limit=0;
        while(!bind_ready)begin tick();limit++;`CHECK(limit<100,"bind readiness deadline") end
        tick();bind_valid=0;limit=0;
        while(!bind_completion_valid)begin tick();limit++;`CHECK(limit<100,"bind completion deadline") end
        `CHECK(bind_completion_status==0 && cache_valid && cache_capacity==4,"Bind succeeded")
        bind_completion_ready=1;tick();bind_completion_ready=0;
        command_sequence=SEQ;command_epoch=EPOCH;command_source_sequence=SEQ;command_source_epoch=EPOCH;
        command_source_valid=1;command_layer=0;command_position=0;command_coefficient_position=0;command_lease_base=LEASE;
        drive_command_addr[0]=64'h0000010000000fc0;drive_command_base[0]=64'h0000010000000000;drive_command_bytes[0]=12224;
        drive_command_addr[1]=64'h0000020000000fc0;drive_command_base[1]=64'h0000020000000000;drive_command_bytes[1]=12224;
        drive_command_addr[2]=64'h0000030000000fc0;drive_command_base[2]=64'h0000030000000000;drive_command_bytes[2]=8128;
        drive_command_addr[3]=64'h0000040000000fc0;drive_command_base[3]=64'h0000040000000000;drive_command_bytes[3]=4460480;
        drive_command_addr[4]=64'h0000050000000fc0;drive_command_base[4]=64'h0000050000000000;drive_command_bytes[4]=1118144;
        drive_command_addr[5]=64'h0000060000000fc0;drive_command_base[5]=64'h0000060000000000;drive_command_bytes[5]=1118144;
        drive_command_addr[6]=64'h0000070000000fc0;drive_command_base[6]=64'h0000070000000000;drive_command_bytes[6]=12224;
        drive_command_addr[7]=64'h0000080000000fc0;drive_command_base[7]=64'h0000080000000000;drive_command_bytes[7]=6080;
        drive_command_addr[8]=64'h0000090000000fc0;drive_command_base[8]=64'h0000090000000000;drive_command_bytes[8]=6080;
        drive_command_addr[9]=64'h00000a0000000fc0;drive_command_base[9]=64'h00000a0000000000;drive_command_bytes[9]=4288;
        drive_command_addr[10]=64'h00000b0000000fc0;drive_command_base[10]=64'h00000b0000000000;drive_command_bytes[10]=4460480;
        drive_command_addr[11]=64'h00000c0000000fc0;drive_command_base[11]=64'h00000c0000000000;drive_command_bytes[11]=8128;
        drive_command_addr[12]=64'h00000d0000000fc0;drive_command_base[12]=64'h00000d0000000000;drive_command_bytes[12]=12224;
        drive_command_addr[13]=64'h00000e0000000fc0;drive_command_base[13]=64'h00000e0000000000;drive_command_bytes[13]=12224;
        drive_command_addr[14]=64'h00000f0000000fc0;drive_command_base[14]=64'h00000f0000000000;drive_command_bytes[14]=12224;
        drive_command_addr[15]=64'h0000100000000fc0;drive_command_base[15]=64'h0000100000000000;drive_command_bytes[15]=8128;
        drive_command_addr[16]=64'h0000110000000fc0;drive_command_base[16]=64'h0000110000000000;drive_command_bytes[16]=17829824;
        drive_command_addr[17]=64'h0000120000000fc0;drive_command_base[17]=64'h0000120000000000;drive_command_bytes[17]=17829824;
        drive_command_addr[18]=64'h0000130000000fc0;drive_command_base[18]=64'h0000130000000000;drive_command_bytes[18]=17829824;
        drive_command_addr[19]=64'h0000140000000fc0;drive_command_base[19]=64'h0000140000000000;drive_command_bytes[19]=69568;
        drive_command_addr[20]=64'h0000150000000fc0;drive_command_base[20]=64'h0000150000000000;drive_command_bytes[20]=69568;
        drive_command_addr[21]=64'h0000160000000fc0;drive_command_base[21]=64'h0000160000000000;drive_command_bytes[21]=20416;
        drive_command_addr[22]=64'h0000170000000fc0;drive_command_base[22]=64'h0000170000000000;drive_command_bytes[22]=12224;
        drive_command_addr[23]=64'h0000180000000fc0;drive_command_base[23]=64'h0000180000000000;drive_command_bytes[23]=12224;
        command_valid=1;limit=0;
        while(!command_ready)begin tick();limit++;`CHECK(limit<100,"admission deadline") end
        $display("LREAL_ADMIT cycle=%0d sequence=%016h epoch=%016h layer=0 position=0 token=0",cycle+1,SEQ,EPOCH);
        tick();command_valid=0;command_sequence=0;command_epoch=0;command_layer=31;command_position=999;
        command_source_valid=0;command_source_sequence=0;command_source_epoch=0;command_coefficient_position=999;command_lease_base=0;
        drive_command_addr=0;drive_command_base=0;drive_command_bytes=0;
        while(!completion_valid)tick();
        `CHECK(completion_status==0 && !completion_revoked && completion_sequence==SEQ && completion_epoch==EPOCH,"Whole layer status/identity")
        `CHECK(result_valid && result_address==region_address(23) && result_bytes==8192 && completion_stage==4,"Whole layer result")
        `CHECK(stage_started==15 && stage_seen==15 && stage_acked==1,"Retain tail and release consumed front")
        `CHECK(heads==32 && head_acks==32 && at_starts==32 && at_dones==32 && attention_words==2048 && source_reads>=2048 && (&source_seen) && hls_inputs==source_reads && rope_words==2560,"Full source/attention coverage")
        `CHECK(cache_query_layer==0 && cache_query_committed_tokens==1 && total_writes==3264,"Complete memory/cache publication")
        for(integer r=0;r<24;r++)if(writable[r])begin
            `CHECK(writes[r]==sizes[r],"Complete written allocation")
            for(integer i=0;i<sizes[r];i++)begin `CHECK(written[starts[r]+i] && memory[starts[r]+i]===golden[starts[r]+i],"Committed final image") end
        end
        for(integer p=0;p<5;p++)begin
            `CHECK(!read_active[p] && !aw_active[p] && w_count[p]==0 && !bus_rvalid[p] && !bus_bvalid[p],"All memory masters drained")
            $display("LREAL_PORT port=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d ar_stalls=%0d w_stalls=%0d",p,ars[p],rs[p],aws[p],ws[p],bs[p],r_stalls[p],w_stalls[p]);
            for(integer r=0;r<25;r++)$display("LREAL_READ port=%0d region=%0d beats=%0d hash=%016h",p,r,region_reads[p][r],read_hash[p][r]);
        end
        public_cycle=cycle;$display("LREAL_COMPLETE cycle=%0d status=%0d sequence=%016h epoch=%016h address=%016h bytes=%0d prefix=%0d",cycle,completion_status,completion_sequence,completion_epoch,result_address,result_bytes,cache_query_committed_tokens);
        repeat(7)begin tick();`CHECK(completion_valid && result_valid && !completion_revoked && !command_ready,"Held result/ownership") end
        completion_ready=1;$display("LREAL_PUBLIC_ACK cycle=%0d",cycle+1);tick();completion_ready=0;limit=0;
        while(busy)begin tick();limit++;`CHECK(limit<100,"Final drain deadline") end
        `CHECK(stage_acked==15 && command_ready && bind_ready && !cache_exclusive && !completion_valid && !result_valid,"Whole layer reusable after drain")
        $display("LREAL_PASS seed=%0d jobs=1 cycles=%0d writes=%0d heads=%0d source_words=%0d",seed,cycle,total_writes,heads,source_reads);$finish;
    end
`undef RP
`undef AT
endmodule
`undef CHECK
`default_nettype wire
