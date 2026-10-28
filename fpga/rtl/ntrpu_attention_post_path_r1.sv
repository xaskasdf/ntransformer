// Actual root -> O -> residual/RMSNorm/Q8. One AXI owner and retained child responses.
// Global reset must quiesce both children and the shared AXI target.
`default_nettype none
module ntrpu_attention_post_path_r1 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter logic [63:0] INITIAL_QUANT_TAG=64'd0,
    parameter logic [63:0] INITIAL_PROJECTION_TAG=64'd0
) (
    input wire logic clk,
    input wire logic rst_n,
    input wire logic bind_valid,
    output wire logic bind_ready,
    input wire logic [63:0] bind_cache_base,
    input wire logic [63:0] bind_region_base,
    input wire logic [63:0] bind_region_bytes,
    input wire logic [31:0] bind_capacity,
    input wire logic [63:0] bind_sequence,
    input wire logic [63:0] bind_epoch,
    input wire logic [63:0] expected_epoch,
    output wire logic cache_valid,
    output wire logic [63:0] cache_base,
    output wire logic [63:0] cache_region_base,
    output wire logic [63:0] cache_region_bytes,
    output wire logic [63:0] cache_sequence,
    output wire logic [63:0] cache_epoch,
    output wire logic [31:0] cache_capacity,
    input wire logic append_valid,
    output wire logic append_ready,
    output wire logic append_prepared,
    input wire logic [31:0] append_layer,
    input wire logic [31:0] append_position,
    input wire logic [63:0] append_sequence,
    input wire logic [63:0] append_epoch,
    input wire logic data_valid,
    output wire logic data_ready,
    input wire logic [511:0] data_bits,
    input wire logic [5:0] data_index,
    input wire logic data_last,
    input wire logic [63:0] data_sequence,
    input wire logic [63:0] data_epoch,
    input wire logic commit_valid,
    output wire logic commit_ready,
    input wire logic [3:0] commit_status,
    input wire logic [63:0] commit_sequence,
    input wire logic [63:0] commit_epoch,
    input wire logic append_cancel_valid,
    output wire logic append_cancel_ready,
    output wire logic append_done_valid,
    input wire logic append_done_ready,
    output wire logic append_done_bind,
    output wire logic [3:0] append_done_status,
    output wire logic [12:0] append_done_bytes,
    output wire logic [31:0] append_done_committed_tokens,
    output wire logic [63:0] append_done_sequence,
    output wire logic [63:0] append_done_epoch,
    output wire logic lease_active,
    output wire logic [63:0] m_axi_awaddr,
    output wire logic [7:0] m_axi_awlen,
    output wire logic [2:0] m_axi_awsize,
    output wire logic [2:0] m_axi_awprot,
    output wire logic [1:0] m_axi_awburst,
    output wire logic [3:0] m_axi_awid,
    output wire logic [3:0] m_axi_awcache,
    output wire logic m_axi_awlock,
    output wire logic m_axi_awvalid,
    input wire logic m_axi_awready,
    output wire logic [511:0] m_axi_wdata,
    output wire logic [63:0] m_axi_wstrb,
    output wire logic m_axi_wlast,
    output wire logic m_axi_wvalid,
    input wire logic m_axi_wready,
    input wire logic [3:0] m_axi_bid,
    input wire logic [1:0] m_axi_bresp,
    input wire logic m_axi_bvalid,
    output wire logic m_axi_bready,
    output wire logic [63:0] m_axi_araddr,
    output wire logic [7:0] m_axi_arlen,
    output wire logic [2:0] m_axi_arsize,
    output wire logic [2:0] m_axi_arprot,
    output wire logic [1:0] m_axi_arburst,
    output wire logic [3:0] m_axi_arid,
    output wire logic [3:0] m_axi_arcache,
    output wire logic m_axi_arlock,
    output wire logic m_axi_arvalid,
    input wire logic m_axi_arready,
    input wire logic [511:0] m_axi_rdata,
    input wire logic [1:0] m_axi_rresp,
    input wire logic [3:0] m_axi_rid,
    input wire logic m_axi_rlast,
    input wire logic m_axi_rvalid,
    output wire logic m_axi_rready,
    input wire logic command_valid,
    output wire logic command_ready,
    input wire logic [31:0] command_position,
    input wire logic [31:0] command_layer,
    input wire logic [63:0] command_sequence,
    input wire logic [63:0] command_epoch,
    input wire logic cancel_valid,
    output wire logic cancel_ready,
    output wire logic completion_valid,
    input wire logic completion_ready,
    output wire logic [3:0] completion_status,
    output wire logic result_valid,
    input wire logic [63:0] command_lease_base,
    input wire logic q_valid,
    output wire logic q_ready,
    input wire logic [511:0] q_data,
    input wire logic [7:0] q_index,
    input wire logic q_last,
    input wire logic [63:0] q_sequence,
    input wire logic [63:0] q_epoch,
    output wire logic [63:0] completion_sequence,
    output wire logic [63:0] completion_epoch,
    output wire logic cache_lock,
    output wire logic maintenance_busy,
    output wire logic metadata_fault,
    output wire logic [31:0] committed_tokens,
    output wire logic busy,
    output wire logic tag_exhausted,
    output wire logic [31:0] completion_hls_status,
    output wire logic [63:0] result_address,
    output wire logic [16:0] result_bytes,
    input wire logic [63:0] command_weight_addr,
    input wire logic [63:0] command_weight_region_base,
    input wire logic [63:0] command_weight_region_bytes,
    input wire logic [63:0] command_activation_addr,
    input wire logic [63:0] command_activation_region_base,
    input wire logic [63:0] command_activation_region_bytes,
    input wire logic [63:0] command_output_addr,
    input wire logic [63:0] command_output_region_base,
    input wire logic [63:0] command_output_region_bytes,
    input wire logic [63:0] command_post_input_address,
    input wire logic [63:0] command_post_input_base,
    input wire logic [63:0] command_post_input_bytes,
    input wire logic [63:0] command_post_weight_address,
    input wire logic [63:0] command_post_weight_base,
    input wire logic [63:0] command_post_weight_bytes,
    input wire logic [63:0] command_post_residual_address,
    input wire logic [63:0] command_post_residual_base,
    input wire logic [63:0] command_post_residual_bytes,
    input wire logic [63:0] command_post_ffn_address,
    input wire logic [63:0] command_post_ffn_base,
    input wire logic [63:0] command_post_ffn_bytes,
    output wire logic [31:0] completion_post_hls_status,
    output wire logic [9:0] completion_post_read_beats,
    output wire logic [7:0] completion_residual_beats,
    output wire logic [6:0] completion_ffn_beats,
    output wire logic [63:0] result_residual_address,
    output wire logic [63:0] result_ffn_address,
    output wire logic [13:0] result_residual_bytes,
    output wire logic [12:0] result_ffn_bytes
);
    localparam logic [3:0] OK=0,BAD_BOUNDS=1,BAD_IDENTITY=2,BAD_COMPLETION=5,CANCELLED=7,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,WAIT_ROOT,OFFER_POST,WAIT_POST,RESPONSE} state_t;
    state_t state;
    logic root_owned,post_owned,revoked,cancel_seen,epoch_bad,child_cancel,release_public;
    logic root_identity_match,post_identity_match,bounds_ok;
    logic [3:0] fault,fault_now,admission_fault;
    logic [63:0] saved_command_post_input_address;
    logic [63:0] saved_command_post_input_base;
    logic [63:0] saved_command_post_input_bytes;
    logic [63:0] saved_command_post_weight_address;
    logic [63:0] saved_command_post_weight_base;
    logic [63:0] saved_command_post_weight_bytes;
    logic [63:0] saved_command_post_residual_address;
    logic [63:0] saved_command_post_residual_base;
    logic [63:0] saved_command_post_residual_bytes;
    logic [63:0] saved_command_post_ffn_address;
    logic [63:0] saved_command_post_ffn_base;
    logic [63:0] saved_command_post_ffn_bytes;
    logic [63:0] saved_command_output_addr;
    logic [63:0] saved_command_output_region_base;
    logic [63:0] saved_command_output_region_bytes;
    wire r_append_ready;
    wire r_bind_ready;
    wire r_busy;
    wire r_cache_lock;
    wire r_command_ready;
    wire [63:0] r_completion_epoch;
    wire [31:0] r_completion_hls_status;
    wire [63:0] r_completion_sequence;
    wire [3:0] r_completion_status;
    wire r_completion_valid;
    wire [63:0] r_m_axi_araddr;
    wire [1:0] r_m_axi_arburst;
    wire [3:0] r_m_axi_arcache;
    wire [3:0] r_m_axi_arid;
    wire [7:0] r_m_axi_arlen;
    wire r_m_axi_arlock;
    wire [2:0] r_m_axi_arprot;
    wire r_m_axi_arready;
    wire [2:0] r_m_axi_arsize;
    wire r_m_axi_arvalid;
    wire [63:0] r_m_axi_awaddr;
    wire [1:0] r_m_axi_awburst;
    wire [3:0] r_m_axi_awcache;
    wire [3:0] r_m_axi_awid;
    wire [7:0] r_m_axi_awlen;
    wire r_m_axi_awlock;
    wire [2:0] r_m_axi_awprot;
    wire r_m_axi_awready;
    wire [2:0] r_m_axi_awsize;
    wire r_m_axi_awvalid;
    wire [3:0] r_m_axi_bid;
    wire r_m_axi_bready;
    wire [1:0] r_m_axi_bresp;
    wire r_m_axi_bvalid;
    wire [511:0] r_m_axi_rdata;
    wire [3:0] r_m_axi_rid;
    wire r_m_axi_rlast;
    wire r_m_axi_rready;
    wire [1:0] r_m_axi_rresp;
    wire r_m_axi_rvalid;
    wire [511:0] r_m_axi_wdata;
    wire r_m_axi_wlast;
    wire r_m_axi_wready;
    wire [63:0] r_m_axi_wstrb;
    wire r_m_axi_wvalid;
    wire r_q_ready;
    wire [63:0] r_result_address;
    wire [16:0] r_result_bytes;
    wire r_result_valid;
    wire p_busy;
    wire p_cancel_ready;
    wire p_command_ready;
    wire [63:0] p_completion_epoch;
    wire [6:0] p_completion_ffn_beats;
    wire [31:0] p_completion_hls_status;
    wire [9:0] p_completion_read_beats;
    wire [7:0] p_completion_residual_beats;
    wire [63:0] p_completion_sequence;
    wire [3:0] p_completion_status;
    wire p_completion_valid;
    wire [63:0] p_m_axi_araddr;
    wire [1:0] p_m_axi_arburst;
    wire [3:0] p_m_axi_arcache;
    wire [3:0] p_m_axi_arid;
    wire [7:0] p_m_axi_arlen;
    wire p_m_axi_arlock;
    wire [2:0] p_m_axi_arprot;
    wire p_m_axi_arready;
    wire [2:0] p_m_axi_arsize;
    wire p_m_axi_arvalid;
    wire [63:0] p_m_axi_awaddr;
    wire [1:0] p_m_axi_awburst;
    wire [3:0] p_m_axi_awcache;
    wire [3:0] p_m_axi_awid;
    wire [7:0] p_m_axi_awlen;
    wire p_m_axi_awlock;
    wire [2:0] p_m_axi_awprot;
    wire p_m_axi_awready;
    wire [2:0] p_m_axi_awsize;
    wire p_m_axi_awvalid;
    wire [3:0] p_m_axi_bid;
    wire p_m_axi_bready;
    wire [1:0] p_m_axi_bresp;
    wire p_m_axi_bvalid;
    wire [511:0] p_m_axi_rdata;
    wire [3:0] p_m_axi_rid;
    wire p_m_axi_rlast;
    wire p_m_axi_rready;
    wire [1:0] p_m_axi_rresp;
    wire p_m_axi_rvalid;
    wire [511:0] p_m_axi_wdata;
    wire p_m_axi_wlast;
    wire p_m_axi_wready;
    wire [63:0] p_m_axi_wstrb;
    wire p_m_axi_wvalid;
    wire [63:0] p_result_ffn_address;
    wire [12:0] p_result_ffn_bytes;
    wire [63:0] p_result_residual_address;
    wire [13:0] p_result_residual_bytes;
    wire p_result_valid;
    function automatic logic span_ok(input logic [63:0] address,base,bytes,input logic [13:0] extent);
        logic [64:0] limit,ending;
        begin
            limit={1'b0,base}+{1'b0,bytes};ending={1'b0,address}+65'(extent);
            span_ok=address[5:0]==0 && bytes!=0 && !limit[64] && !ending[64] && address>=base && ending<=limit;
        end
    endfunction
    function automatic logic overlap(input logic [63:0] a,b,input logic [13:0] na,nb);
        overlap={1'b0,a}<{1'b0,b}+65'(nb) && {1'b0,b}<{1'b0,a}+65'(na);
    endfunction
    assign bounds_ok=span_ok(command_output_addr,command_output_region_base,command_output_region_bytes,8192) &&
        span_ok(command_post_input_address,command_post_input_base,command_post_input_bytes,8192) &&
        span_ok(command_post_weight_address,command_post_weight_base,command_post_weight_bytes,8192) &&
        span_ok(command_post_residual_address,command_post_residual_base,command_post_residual_bytes,8192) &&
        span_ok(command_post_ffn_address,command_post_ffn_base,command_post_ffn_bytes,4096) &&
        !overlap(command_output_addr,command_post_input_address,8192,8192) &&
        !overlap(command_output_addr,command_post_weight_address,8192,8192) &&
        !overlap(command_post_residual_address,command_post_ffn_address,8192,4096);
    assign admission_fault=command_epoch!=expected_epoch ? BAD_IDENTITY:cancel_valid ? CANCELLED:!bounds_ok ? BAD_BOUNDS:OK;
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && r_command_ready && !p_busy;
    assign bind_ready=state==IDLE && r_bind_ready;
    assign append_ready=state==IDLE && r_append_ready;
    assign cache_lock=busy || r_cache_lock;
    assign q_ready=root_owned && !post_owned && r_q_ready;
    assign cancel_ready=busy && !cancel_seen;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    // Registered fault prevents combinational cross-child cancel feedback.
    assign child_cancel=busy && (cancel_valid || epoch_bad || fault!=OK);
    assign root_identity_match=r_completion_sequence==completion_sequence && r_completion_epoch==completion_epoch;
    assign post_identity_match=p_completion_sequence==completion_sequence && p_completion_epoch==completion_epoch;
    assign completion_valid=state==RESPONSE && (!root_owned || (r_completion_valid && root_identity_match)) &&
        (!post_owned || (p_completion_valid && post_identity_match));
    assign release_public=completion_valid && completion_ready;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad && !cancel_valid &&
        post_owned && r_result_valid && p_result_valid;
    assign result_address=result_valid ? r_result_address:64'd0;
    assign result_bytes=result_valid ? r_result_bytes:17'd0;
    assign result_residual_address=result_valid ? p_result_residual_address:64'd0;
    assign result_ffn_address=result_valid ? p_result_ffn_address:64'd0;
    assign result_residual_bytes=result_valid ? p_result_residual_bytes:14'd0;
    assign result_ffn_bytes=result_valid ? p_result_ffn_bytes:13'd0;
    assign m_axi_awaddr=post_owned ? p_m_axi_awaddr:r_m_axi_awaddr;
    assign m_axi_awlen=post_owned ? p_m_axi_awlen:r_m_axi_awlen;
    assign m_axi_awsize=post_owned ? p_m_axi_awsize:r_m_axi_awsize;
    assign m_axi_awprot=post_owned ? p_m_axi_awprot:r_m_axi_awprot;
    assign m_axi_awburst=post_owned ? p_m_axi_awburst:r_m_axi_awburst;
    assign m_axi_awid=post_owned ? p_m_axi_awid:r_m_axi_awid;
    assign m_axi_awcache=post_owned ? p_m_axi_awcache:r_m_axi_awcache;
    assign m_axi_awlock=post_owned ? p_m_axi_awlock:r_m_axi_awlock;
    assign m_axi_awvalid=post_owned ? p_m_axi_awvalid:r_m_axi_awvalid;
    assign r_m_axi_awready=!post_owned && m_axi_awready;
    assign p_m_axi_awready=post_owned && m_axi_awready;
    assign m_axi_wdata=post_owned ? p_m_axi_wdata:r_m_axi_wdata;
    assign m_axi_wstrb=post_owned ? p_m_axi_wstrb:r_m_axi_wstrb;
    assign m_axi_wlast=post_owned ? p_m_axi_wlast:r_m_axi_wlast;
    assign m_axi_wvalid=post_owned ? p_m_axi_wvalid:r_m_axi_wvalid;
    assign r_m_axi_wready=!post_owned && m_axi_wready;
    assign p_m_axi_wready=post_owned && m_axi_wready;
    assign r_m_axi_bid=m_axi_bid;
    assign p_m_axi_bid=m_axi_bid;
    assign r_m_axi_bresp=m_axi_bresp;
    assign p_m_axi_bresp=m_axi_bresp;
    assign r_m_axi_bvalid=!post_owned && m_axi_bvalid;
    assign p_m_axi_bvalid=post_owned && m_axi_bvalid;
    assign m_axi_bready=post_owned ? p_m_axi_bready:r_m_axi_bready;
    assign m_axi_araddr=post_owned ? p_m_axi_araddr:r_m_axi_araddr;
    assign m_axi_arlen=post_owned ? p_m_axi_arlen:r_m_axi_arlen;
    assign m_axi_arsize=post_owned ? p_m_axi_arsize:r_m_axi_arsize;
    assign m_axi_arprot=post_owned ? p_m_axi_arprot:r_m_axi_arprot;
    assign m_axi_arburst=post_owned ? p_m_axi_arburst:r_m_axi_arburst;
    assign m_axi_arid=post_owned ? p_m_axi_arid:r_m_axi_arid;
    assign m_axi_arcache=post_owned ? p_m_axi_arcache:r_m_axi_arcache;
    assign m_axi_arlock=post_owned ? p_m_axi_arlock:r_m_axi_arlock;
    assign m_axi_arvalid=post_owned ? p_m_axi_arvalid:r_m_axi_arvalid;
    assign r_m_axi_arready=!post_owned && m_axi_arready;
    assign p_m_axi_arready=post_owned && m_axi_arready;
    assign r_m_axi_rdata=m_axi_rdata;
    assign p_m_axi_rdata=m_axi_rdata;
    assign r_m_axi_rresp=m_axi_rresp;
    assign p_m_axi_rresp=m_axi_rresp;
    assign r_m_axi_rid=m_axi_rid;
    assign p_m_axi_rid=m_axi_rid;
    assign r_m_axi_rlast=m_axi_rlast;
    assign p_m_axi_rlast=m_axi_rlast;
    assign r_m_axi_rvalid=!post_owned && m_axi_rvalid;
    assign p_m_axi_rvalid=post_owned && m_axi_rvalid;
    assign m_axi_rready=post_owned ? p_m_axi_rready:r_m_axi_rready;
    ntrpu_attention_projected_root_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),
        .INITIAL_QUANT_TAG(INITIAL_QUANT_TAG),.INITIAL_PROJECTION_TAG(INITIAL_PROJECTION_TAG)) projected (
        .clk(clk),
        .rst_n(rst_n),
        .bind_valid(state==IDLE && bind_valid),
        .bind_ready(r_bind_ready),
        .bind_cache_base(bind_cache_base),
        .bind_region_base(bind_region_base),
        .bind_region_bytes(bind_region_bytes),
        .bind_capacity(bind_capacity),
        .bind_sequence(bind_sequence),
        .bind_epoch(bind_epoch),
        .expected_epoch(expected_epoch),
        .cache_valid(cache_valid),
        .cache_base(cache_base),
        .cache_region_base(cache_region_base),
        .cache_region_bytes(cache_region_bytes),
        .cache_sequence(cache_sequence),
        .cache_epoch(cache_epoch),
        .cache_capacity(cache_capacity),
        .append_valid(state==IDLE && append_valid),
        .append_ready(r_append_ready),
        .append_prepared(append_prepared),
        .append_layer(append_layer),
        .append_position(append_position),
        .append_sequence(append_sequence),
        .append_epoch(append_epoch),
        .data_valid(data_valid),
        .data_ready(data_ready),
        .data_bits(data_bits),
        .data_index(data_index),
        .data_last(data_last),
        .data_sequence(data_sequence),
        .data_epoch(data_epoch),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_status(commit_status),
        .commit_sequence(commit_sequence),
        .commit_epoch(commit_epoch),
        .append_cancel_valid(append_cancel_valid),
        .append_cancel_ready(append_cancel_ready),
        .append_done_valid(append_done_valid),
        .append_done_ready(append_done_ready),
        .append_done_bind(append_done_bind),
        .append_done_status(append_done_status),
        .append_done_bytes(append_done_bytes),
        .append_done_committed_tokens(append_done_committed_tokens),
        .append_done_sequence(append_done_sequence),
        .append_done_epoch(append_done_epoch),
        .lease_active(lease_active),
        .m_axi_awaddr(r_m_axi_awaddr),
        .m_axi_awlen(r_m_axi_awlen),
        .m_axi_awsize(r_m_axi_awsize),
        .m_axi_awprot(r_m_axi_awprot),
        .m_axi_awburst(r_m_axi_awburst),
        .m_axi_awid(r_m_axi_awid),
        .m_axi_awcache(r_m_axi_awcache),
        .m_axi_awlock(r_m_axi_awlock),
        .m_axi_awvalid(r_m_axi_awvalid),
        .m_axi_awready(r_m_axi_awready),
        .m_axi_wdata(r_m_axi_wdata),
        .m_axi_wstrb(r_m_axi_wstrb),
        .m_axi_wlast(r_m_axi_wlast),
        .m_axi_wvalid(r_m_axi_wvalid),
        .m_axi_wready(r_m_axi_wready),
        .m_axi_bid(r_m_axi_bid),
        .m_axi_bresp(r_m_axi_bresp),
        .m_axi_bvalid(r_m_axi_bvalid),
        .m_axi_bready(r_m_axi_bready),
        .m_axi_araddr(r_m_axi_araddr),
        .m_axi_arlen(r_m_axi_arlen),
        .m_axi_arsize(r_m_axi_arsize),
        .m_axi_arprot(r_m_axi_arprot),
        .m_axi_arburst(r_m_axi_arburst),
        .m_axi_arid(r_m_axi_arid),
        .m_axi_arcache(r_m_axi_arcache),
        .m_axi_arlock(r_m_axi_arlock),
        .m_axi_arvalid(r_m_axi_arvalid),
        .m_axi_arready(r_m_axi_arready),
        .m_axi_rdata(r_m_axi_rdata),
        .m_axi_rresp(r_m_axi_rresp),
        .m_axi_rid(r_m_axi_rid),
        .m_axi_rlast(r_m_axi_rlast),
        .m_axi_rvalid(r_m_axi_rvalid),
        .m_axi_rready(r_m_axi_rready),
        .command_valid(state==IDLE && command_valid && !p_busy && admission_fault==OK),
        .command_ready(r_command_ready),
        .command_position(command_position),
        .command_layer(command_layer),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .cancel_valid(root_owned && child_cancel),
        .cancel_ready(cancel_ready),
        .completion_valid(r_completion_valid),
        .completion_ready(root_owned && root_identity_match && release_public),
        .completion_status(r_completion_status),
        .result_valid(r_result_valid),
        .command_lease_base(command_lease_base),
        .q_valid(root_owned && !post_owned && q_valid),
        .q_ready(r_q_ready),
        .q_data(q_data),
        .q_index(q_index),
        .q_last(q_last),
        .q_sequence(q_sequence),
        .q_epoch(q_epoch),
        .completion_sequence(r_completion_sequence),
        .completion_epoch(r_completion_epoch),
        .cache_lock(r_cache_lock),
        .maintenance_busy(maintenance_busy),
        .metadata_fault(metadata_fault),
        .committed_tokens(committed_tokens),
        .busy(r_busy),
        .tag_exhausted(tag_exhausted),
        .completion_hls_status(r_completion_hls_status),
        .result_address(r_result_address),
        .result_bytes(r_result_bytes),
        .command_weight_addr(command_weight_addr),
        .command_weight_region_base(command_weight_region_base),
        .command_weight_region_bytes(command_weight_region_bytes),
        .command_activation_addr(command_activation_addr),
        .command_activation_region_base(command_activation_region_base),
        .command_activation_region_bytes(command_activation_region_bytes),
        .command_output_addr(command_output_addr),
        .command_output_region_base(command_output_region_base),
        .command_output_region_bytes(command_output_region_bytes)
    );
    ntrpu_post_attention_real_r2 #(.AXI_WATCHDOG_CYCLES(WATCHDOG_CYCLES),
        .HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) post (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(state==OFFER_POST),
        .command_ready(p_command_ready),
        .command_sequence(completion_sequence),
        .command_epoch(completion_epoch),
        .expected_epoch(expected_epoch),
        .command_source_sequence(r_completion_sequence),
        .command_source_epoch(r_completion_epoch),
        .command_source_valid(r_result_valid && root_identity_match && r_completion_status==OK),
        .command_input_address(saved_command_post_input_address),
        .command_input_base(saved_command_post_input_base),
        .command_input_bytes(saved_command_post_input_bytes),
        .command_branch_address(saved_command_output_addr),
        .command_branch_base(saved_command_output_region_base),
        .command_branch_bytes(saved_command_output_region_bytes),
        .command_weight_address(saved_command_post_weight_address),
        .command_weight_base(saved_command_post_weight_base),
        .command_weight_bytes(saved_command_post_weight_bytes),
        .command_residual_address(saved_command_post_residual_address),
        .command_residual_base(saved_command_post_residual_base),
        .command_residual_bytes(saved_command_post_residual_bytes),
        .command_ffn_address(saved_command_post_ffn_address),
        .command_ffn_base(saved_command_post_ffn_base),
        .command_ffn_bytes(saved_command_post_ffn_bytes),
        .cancel_valid(child_cancel),
        .cancel_ready(p_cancel_ready),
        .busy(p_busy),
        .completion_valid(p_completion_valid),
        .completion_ready(post_owned && post_identity_match && release_public),
        .completion_status(p_completion_status),
        .completion_hls_status(p_completion_hls_status),
        .completion_sequence(p_completion_sequence),
        .completion_epoch(p_completion_epoch),
        .completion_read_beats(p_completion_read_beats),
        .completion_residual_beats(p_completion_residual_beats),
        .completion_ffn_beats(p_completion_ffn_beats),
        .result_valid(p_result_valid),
        .result_residual_address(p_result_residual_address),
        .result_ffn_address(p_result_ffn_address),
        .result_residual_bytes(p_result_residual_bytes),
        .result_ffn_bytes(p_result_ffn_bytes),
        .m_axi_araddr(p_m_axi_araddr),
        .m_axi_arlen(p_m_axi_arlen),
        .m_axi_arsize(p_m_axi_arsize),
        .m_axi_arprot(p_m_axi_arprot),
        .m_axi_arburst(p_m_axi_arburst),
        .m_axi_arid(p_m_axi_arid),
        .m_axi_arcache(p_m_axi_arcache),
        .m_axi_arlock(p_m_axi_arlock),
        .m_axi_arvalid(p_m_axi_arvalid),
        .m_axi_arready(p_m_axi_arready),
        .m_axi_rdata(p_m_axi_rdata),
        .m_axi_rresp(p_m_axi_rresp),
        .m_axi_rid(p_m_axi_rid),
        .m_axi_rlast(p_m_axi_rlast),
        .m_axi_rvalid(p_m_axi_rvalid),
        .m_axi_rready(p_m_axi_rready),
        .m_axi_awaddr(p_m_axi_awaddr),
        .m_axi_awlen(p_m_axi_awlen),
        .m_axi_awsize(p_m_axi_awsize),
        .m_axi_awprot(p_m_axi_awprot),
        .m_axi_awburst(p_m_axi_awburst),
        .m_axi_awid(p_m_axi_awid),
        .m_axi_awcache(p_m_axi_awcache),
        .m_axi_awlock(p_m_axi_awlock),
        .m_axi_awvalid(p_m_axi_awvalid),
        .m_axi_awready(p_m_axi_awready),
        .m_axi_wdata(p_m_axi_wdata),
        .m_axi_wstrb(p_m_axi_wstrb),
        .m_axi_wlast(p_m_axi_wlast),
        .m_axi_wvalid(p_m_axi_wvalid),
        .m_axi_wready(p_m_axi_wready),
        .m_axi_bid(p_m_axi_bid),
        .m_axi_bresp(p_m_axi_bresp),
        .m_axi_bvalid(p_m_axi_bvalid),
        .m_axi_bready(p_m_axi_bready)
    );
    always_comb begin
        fault_now=fault;
        if(busy && state!=RESPONSE && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(root_owned && r_completion_valid && !root_identity_match)fault_now=BAD_PEER;
            else if(post_owned && p_completion_valid && !post_identity_match)fault_now=BAD_PEER;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;root_owned<=0;post_owned<=0;revoked<=0;cancel_seen<=0;fault<=OK;
            completion_status<=OK;completion_sequence<=0;completion_epoch<=0;
            completion_hls_status<=0;completion_post_hls_status<=0;
            completion_post_read_beats<=0;completion_residual_beats<=0;completion_ffn_beats<=0;
            saved_command_post_input_address<=0;
            saved_command_post_input_base<=0;
            saved_command_post_input_bytes<=0;
            saved_command_post_weight_address<=0;
            saved_command_post_weight_base<=0;
            saved_command_post_weight_bytes<=0;
            saved_command_post_residual_address<=0;
            saved_command_post_residual_base<=0;
            saved_command_post_residual_bytes<=0;
            saved_command_post_ffn_address<=0;
            saved_command_post_ffn_base<=0;
            saved_command_post_ffn_bytes<=0;
            saved_command_output_addr<=0;
            saved_command_output_region_base<=0;
            saved_command_output_region_bytes<=0;
        end else begin
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(busy && state!=RESPONSE)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_status<=admission_fault;fault<=admission_fault;revoked<=admission_fault!=OK;cancel_seen<=cancel_valid;
                    completion_hls_status<=0;completion_post_hls_status<=0;
                    completion_post_read_beats<=0;completion_residual_beats<=0;completion_ffn_beats<=0;
                    root_owned<=admission_fault==OK;post_owned<=0;
                    saved_command_post_input_address<=command_post_input_address;
                    saved_command_post_input_base<=command_post_input_base;
                    saved_command_post_input_bytes<=command_post_input_bytes;
                    saved_command_post_weight_address<=command_post_weight_address;
                    saved_command_post_weight_base<=command_post_weight_base;
                    saved_command_post_weight_bytes<=command_post_weight_bytes;
                    saved_command_post_residual_address<=command_post_residual_address;
                    saved_command_post_residual_base<=command_post_residual_base;
                    saved_command_post_residual_bytes<=command_post_residual_bytes;
                    saved_command_post_ffn_address<=command_post_ffn_address;
                    saved_command_post_ffn_base<=command_post_ffn_base;
                    saved_command_post_ffn_bytes<=command_post_ffn_bytes;
                    saved_command_output_addr<=command_output_addr;
                    saved_command_output_region_base<=command_output_region_base;
                    saved_command_output_region_bytes<=command_output_region_bytes;
                    state<=admission_fault==OK ? WAIT_ROOT:RESPONSE;
                end
                WAIT_ROOT:if(r_completion_valid && root_identity_match)begin
                    completion_hls_status<=r_completion_hls_status;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(r_completion_status!=OK)begin completion_status<=r_completion_status;state<=RESPONSE;end
                    else if(!r_result_valid || r_result_address!=saved_command_output_addr || r_result_bytes!=8192)begin
                        completion_status<=BAD_COMPLETION;state<=RESPONSE;
                    end else begin post_owned<=1;state<=OFFER_POST;end
                end
                // After VALID is offered the handshake is drained, even if cancelled.
                OFFER_POST:if(p_command_ready)state<=WAIT_POST;
                WAIT_POST:if(p_completion_valid && post_identity_match)begin
                    completion_post_hls_status<=p_completion_hls_status;
                    completion_post_read_beats<=p_completion_read_beats;
                    completion_residual_beats<=p_completion_residual_beats;
                    completion_ffn_beats<=p_completion_ffn_beats;
                    if(fault_now!=OK)completion_status<=fault_now;
                    else if(p_completion_status!=OK)completion_status<=p_completion_status;
                    else if(!p_result_valid || p_result_residual_address!=saved_command_post_residual_address ||
                        p_result_ffn_address!=saved_command_post_ffn_address || p_result_residual_bytes!=8192 || p_result_ffn_bytes!=4096)
                        completion_status<=BAD_COMPLETION;
                    else completion_status<=OK;
                    state<=RESPONSE;
                end
                RESPONSE:if(release_public)begin root_owned<=0;post_owned<=0;state<=IDLE;end
                default:begin state<=IDLE;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
