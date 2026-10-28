// Actual attention root -> generated quantize/store -> actual O projection.
// Global reset requires coordinated quiesce of both children and AXI memory.
`default_nettype none
module ntrpu_attention_projected_root_r2 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter logic [63:0] INITIAL_QUANT_TAG=64'd0,
    parameter logic [63:0] INITIAL_PROJECTION_TAG=64'd0
) (
    input wire logic clk,
    input wire logic rst_n,
    input wire logic bind_valid,
    output logic bind_ready,
    input wire logic [63:0] bind_cache_base,
    input wire logic [63:0] bind_region_base,
    input wire logic [63:0] bind_region_bytes,
    input wire logic [31:0] bind_capacity,
    input wire logic [63:0] bind_sequence,
    input wire logic [63:0] bind_epoch,
    input wire logic [63:0] expected_epoch,
    output logic cache_valid,
    output logic [63:0] cache_base,
    output logic [63:0] cache_region_base,
    output logic [63:0] cache_region_bytes,
    output logic [63:0] cache_sequence,
    output logic [63:0] cache_epoch,
    output logic [31:0] cache_capacity,
    input wire logic append_valid,
    output logic append_ready,
    output logic append_prepared,
    input wire logic [31:0] append_layer,
    input wire logic [31:0] append_position,
    input wire logic [63:0] append_sequence,
    input wire logic [63:0] append_epoch,
    input wire logic data_valid,
    output logic data_ready,
    input wire logic [511:0] data_bits,
    input wire logic [5:0] data_index,
    input wire logic data_last,
    input wire logic [63:0] data_sequence,
    input wire logic [63:0] data_epoch,
    input wire logic commit_valid,
    output logic commit_ready,
    input wire logic [3:0] commit_status,
    input wire logic [63:0] commit_sequence,
    input wire logic [63:0] commit_epoch,
    input wire logic append_cancel_valid,
    output logic append_cancel_ready,
    output logic append_done_valid,
    input wire logic append_done_ready,
    output logic append_done_bind,
    output logic [3:0] append_done_status,
    output logic [12:0] append_done_bytes,
    output logic [31:0] append_done_committed_tokens,
    output logic [63:0] append_done_sequence,
    output logic [63:0] append_done_epoch,
    output logic lease_active,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize,
    output logic [2:0] m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid,
    output logic [3:0] m_axi_awcache,
    output logic m_axi_awlock,
    output logic m_axi_awvalid,
    input wire logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,
    output logic m_axi_wvalid,
    input wire logic m_axi_wready,
    input wire logic [3:0] m_axi_bid,
    input wire logic [1:0] m_axi_bresp,
    input wire logic m_axi_bvalid,
    output logic m_axi_bready,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,
    output logic [2:0] m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,
    output logic [3:0] m_axi_arcache,
    output logic m_axi_arlock,
    output logic m_axi_arvalid,
    input wire logic m_axi_arready,
    input wire logic [511:0] m_axi_rdata,
    input wire logic [1:0] m_axi_rresp,
    input wire logic [3:0] m_axi_rid,
    input wire logic m_axi_rlast,
    input wire logic m_axi_rvalid,
    output logic m_axi_rready,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [31:0] command_position,
    input wire logic [31:0] command_layer,
    input wire logic [63:0] command_sequence,
    input wire logic [63:0] command_epoch,
    input wire logic cancel_valid,
    output logic cancel_ready,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic result_valid,
    input wire logic [63:0] command_lease_base,
    input wire logic q_valid,
    output logic q_ready,
    input wire logic [511:0] q_data,
    input wire logic [7:0] q_index,
    input wire logic q_last,
    input wire logic [63:0] q_sequence,
    input wire logic [63:0] q_epoch,
    output logic [63:0] completion_sequence,
    output logic [63:0] completion_epoch,
    output logic cache_lock,
    output logic maintenance_busy,
    output logic metadata_fault,
    output logic [31:0] committed_tokens,
    output logic busy,
    output logic tag_exhausted,
    output logic [31:0] completion_hls_status,
    output logic [63:0] result_address,
    output logic [16:0] result_bytes,
    input wire logic [63:0] command_weight_addr,
    input wire logic [63:0] command_weight_region_base,
    input wire logic [63:0] command_weight_region_bytes,
    input wire logic [63:0] command_activation_addr,
    input wire logic [63:0] command_activation_region_base,
    input wire logic [63:0] command_activation_region_bytes,
    input wire logic [63:0] command_output_addr,
    input wire logic [63:0] command_output_region_base,
    input wire logic [63:0] command_output_region_bytes
);

    localparam logic [3:0] OK=0,BAD_IDENTITY=2,BAD_COMPLETION=5,CANCELLED=7,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,WAIT_ROOT,OFFER_OUTPUT,WAIT_OUTPUT,RESPONSE} state_t;
    state_t state;
    logic [3:0] fault,fault_now;
    logic revoked,cancel_seen,epoch_bad,child_cancel,output_owned,root_released;
    logic root_identity_match,output_identity_match,release_public;
    logic [63:0] saved_command_weight_addr;
    logic [63:0] saved_command_weight_region_base;
    logic [63:0] saved_command_weight_region_bytes;
    logic [63:0] saved_command_activation_addr;
    logic [63:0] saved_command_activation_region_base;
    logic [63:0] saved_command_activation_region_bytes;
    logic [63:0] saved_command_output_addr;
    logic [63:0] saved_command_output_region_base;
    logic [63:0] saved_command_output_region_bytes;
    logic r_bind_ready;
    logic r_append_ready;
    logic [63:0] r_m_axi_awaddr;
    logic [7:0] r_m_axi_awlen;
    logic [2:0] r_m_axi_awsize;
    logic [2:0] r_m_axi_awprot;
    logic [1:0] r_m_axi_awburst;
    logic [3:0] r_m_axi_awid;
    logic [3:0] r_m_axi_awcache;
    logic r_m_axi_awlock;
    logic r_m_axi_awvalid;
    logic r_m_axi_awready;
    logic [511:0] r_m_axi_wdata;
    logic [63:0] r_m_axi_wstrb;
    logic r_m_axi_wlast;
    logic r_m_axi_wvalid;
    logic r_m_axi_wready;
    logic [3:0] r_m_axi_bid;
    logic [1:0] r_m_axi_bresp;
    logic r_m_axi_bvalid;
    logic r_m_axi_bready;
    logic [63:0] r_m_axi_araddr;
    logic [7:0] r_m_axi_arlen;
    logic [2:0] r_m_axi_arsize;
    logic [2:0] r_m_axi_arprot;
    logic [1:0] r_m_axi_arburst;
    logic [3:0] r_m_axi_arid;
    logic [3:0] r_m_axi_arcache;
    logic r_m_axi_arlock;
    logic r_m_axi_arvalid;
    logic r_m_axi_arready;
    logic [511:0] r_m_axi_rdata;
    logic [1:0] r_m_axi_rresp;
    logic [3:0] r_m_axi_rid;
    logic r_m_axi_rlast;
    logic r_m_axi_rvalid;
    logic r_m_axi_rready;
    logic r_command_ready;
    logic r_cancel_ready;
    logic r_completion_valid;
    logic [3:0] r_completion_status;
    logic r_result_valid;
    logic [31:0] r_result_q;
    logic r_result_q_valid;
    logic [63:0] r_completion_sequence;
    logic [63:0] r_completion_epoch;
    logic r_cache_lock;
    logic o_command_ready;
    logic o_cancel_ready;
    logic o_busy;
    logic o_completion_valid;
    logic [3:0] o_completion_status;
    logic [63:0] o_completion_sequence;
    logic [63:0] o_completion_epoch;
    logic o_result_valid;
    logic [63:0] o_result_address;
    logic [16:0] o_result_bytes;
    logic o_tag_exhausted;
    logic [63:0] o_m_axi_araddr;
    logic [7:0] o_m_axi_arlen;
    logic [2:0] o_m_axi_arsize;
    logic [2:0] o_m_axi_arprot;
    logic [1:0] o_m_axi_arburst;
    logic [3:0] o_m_axi_arid;
    logic [3:0] o_m_axi_arcache;
    logic o_m_axi_arlock;
    logic o_m_axi_arvalid;
    logic o_m_axi_arready;
    logic [511:0] o_m_axi_rdata;
    logic [1:0] o_m_axi_rresp;
    logic [3:0] o_m_axi_rid;
    logic o_m_axi_rlast;
    logic o_m_axi_rvalid;
    logic o_m_axi_rready;
    logic [63:0] o_m_axi_awaddr;
    logic [7:0] o_m_axi_awlen;
    logic [2:0] o_m_axi_awsize;
    logic [2:0] o_m_axi_awprot;
    logic [1:0] o_m_axi_awburst;
    logic [3:0] o_m_axi_awid;
    logic [3:0] o_m_axi_awcache;
    logic o_m_axi_awlock;
    logic o_m_axi_awvalid;
    logic o_m_axi_awready;
    logic [511:0] o_m_axi_wdata;
    logic [63:0] o_m_axi_wstrb;
    logic o_m_axi_wlast;
    logic o_m_axi_wvalid;
    logic o_m_axi_wready;
    logic [3:0] o_m_axi_bid;
    logic [1:0] o_m_axi_bresp;
    logic o_m_axi_bvalid;
    logic o_m_axi_bready;
    logic o_source_completion_ready;
    logic o_source_result_ce;
    logic [10:0] o_source_result_address;
    logic [31:0] o_completion_hls_status;

    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && r_command_ready && !o_busy;
    assign bind_ready=state==IDLE && r_bind_ready;
    assign append_ready=state==IDLE && r_append_ready;
    assign cache_lock=busy || r_cache_lock;
    assign cancel_ready=busy && !cancel_seen;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    assign child_cancel=busy && (cancel_valid || fault_now!=OK);
    assign root_identity_match=r_completion_sequence==completion_sequence && r_completion_epoch==completion_epoch;
    assign output_identity_match=o_completion_sequence==completion_sequence && o_completion_epoch==completion_epoch;
    assign completion_valid=state==RESPONSE && (output_owned ?
        (root_released && o_completion_valid && output_identity_match):
        (r_completion_valid && root_identity_match));
    assign release_public=completion_valid && completion_ready;
    assign tag_exhausted=o_tag_exhausted;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad &&
        !cancel_valid && output_owned && o_result_valid;
    assign result_address=result_valid ? o_result_address:64'd0;
    assign result_bytes=result_valid ? o_result_bytes:17'd0;
    // Root completion proves KV traffic drained. Maintenance remains blocked
    // after the quantizer releases root memory and until the final public ACK.
    assign m_axi_araddr=output_owned ? o_m_axi_araddr:r_m_axi_araddr;
    assign m_axi_arlen=output_owned ? o_m_axi_arlen:r_m_axi_arlen;
    assign m_axi_arsize=output_owned ? o_m_axi_arsize:r_m_axi_arsize;
    assign m_axi_arprot=output_owned ? o_m_axi_arprot:r_m_axi_arprot;
    assign m_axi_arburst=output_owned ? o_m_axi_arburst:r_m_axi_arburst;
    assign m_axi_arid=output_owned ? o_m_axi_arid:r_m_axi_arid;
    assign m_axi_arcache=output_owned ? o_m_axi_arcache:r_m_axi_arcache;
    assign m_axi_arlock=output_owned ? o_m_axi_arlock:r_m_axi_arlock;
    assign m_axi_arvalid=output_owned ? o_m_axi_arvalid:r_m_axi_arvalid;
    assign r_m_axi_arready=!output_owned && m_axi_arready;
    assign o_m_axi_arready=output_owned && m_axi_arready;
    assign r_m_axi_rdata=m_axi_rdata;
    assign o_m_axi_rdata=m_axi_rdata;
    assign r_m_axi_rresp=m_axi_rresp;
    assign o_m_axi_rresp=m_axi_rresp;
    assign r_m_axi_rid=m_axi_rid;
    assign o_m_axi_rid=m_axi_rid;
    assign r_m_axi_rlast=m_axi_rlast;
    assign o_m_axi_rlast=m_axi_rlast;
    assign r_m_axi_rvalid=!output_owned && m_axi_rvalid;
    assign o_m_axi_rvalid=output_owned && m_axi_rvalid;
    assign m_axi_rready=output_owned ? o_m_axi_rready:r_m_axi_rready;
    assign m_axi_awaddr=output_owned ? o_m_axi_awaddr:r_m_axi_awaddr;
    assign m_axi_awlen=output_owned ? o_m_axi_awlen:r_m_axi_awlen;
    assign m_axi_awsize=output_owned ? o_m_axi_awsize:r_m_axi_awsize;
    assign m_axi_awprot=output_owned ? o_m_axi_awprot:r_m_axi_awprot;
    assign m_axi_awburst=output_owned ? o_m_axi_awburst:r_m_axi_awburst;
    assign m_axi_awid=output_owned ? o_m_axi_awid:r_m_axi_awid;
    assign m_axi_awcache=output_owned ? o_m_axi_awcache:r_m_axi_awcache;
    assign m_axi_awlock=output_owned ? o_m_axi_awlock:r_m_axi_awlock;
    assign m_axi_awvalid=output_owned ? o_m_axi_awvalid:r_m_axi_awvalid;
    assign r_m_axi_awready=!output_owned && m_axi_awready;
    assign o_m_axi_awready=output_owned && m_axi_awready;
    assign m_axi_wdata=output_owned ? o_m_axi_wdata:r_m_axi_wdata;
    assign m_axi_wstrb=output_owned ? o_m_axi_wstrb:r_m_axi_wstrb;
    assign m_axi_wlast=output_owned ? o_m_axi_wlast:r_m_axi_wlast;
    assign m_axi_wvalid=output_owned ? o_m_axi_wvalid:r_m_axi_wvalid;
    assign r_m_axi_wready=!output_owned && m_axi_wready;
    assign o_m_axi_wready=output_owned && m_axi_wready;
    assign r_m_axi_bid=m_axi_bid;
    assign o_m_axi_bid=m_axi_bid;
    assign r_m_axi_bresp=m_axi_bresp;
    assign o_m_axi_bresp=m_axi_bresp;
    assign r_m_axi_bvalid=!output_owned && m_axi_bvalid;
    assign o_m_axi_bvalid=output_owned && m_axi_bvalid;
    assign m_axi_bready=output_owned ? o_m_axi_bready:r_m_axi_bready;
    ntrpu_attention_root_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) root(
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
        .command_valid(state==IDLE && command_valid && !o_busy),
        .command_ready(r_command_ready),
        .command_position(command_position),
        .command_layer(command_layer),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .cancel_valid(child_cancel),
        .cancel_ready(r_cancel_ready),
        .completion_valid(r_completion_valid),
        .completion_ready(root_identity_match && (output_owned ? o_source_completion_ready:release_public)),
        .completion_status(r_completion_status),
        .result_valid(r_result_valid),
        .result_ce(o_source_result_ce && output_owned && !root_released),
        .result_address(o_source_result_address),
        .result_q(r_result_q),
        .result_q_valid(r_result_q_valid),
        .command_lease_base(command_lease_base),
        .q_valid(q_valid),
        .q_ready(q_ready),
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
        .committed_tokens(committed_tokens)
    );
    ntrpu_attention_output_path_r2 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.INITIAL_QUANT_TAG(INITIAL_QUANT_TAG),.INITIAL_PROJECTION_TAG(INITIAL_PROJECTION_TAG)) output_path(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(state==OFFER_OUTPUT),
        .command_ready(o_command_ready),
        .command_sequence(completion_sequence),
        .command_epoch(completion_epoch),
        .expected_epoch(expected_epoch),
        .command_weight_addr(saved_command_weight_addr),
        .command_weight_region_base(saved_command_weight_region_base),
        .command_weight_region_bytes(saved_command_weight_region_bytes),
        .command_activation_addr(saved_command_activation_addr),
        .command_activation_region_base(saved_command_activation_region_base),
        .command_activation_region_bytes(saved_command_activation_region_bytes),
        .command_output_addr(saved_command_output_addr),
        .command_output_region_base(saved_command_output_region_base),
        .command_output_region_bytes(saved_command_output_region_bytes),
        .cancel_valid(child_cancel),
        .cancel_ready(o_cancel_ready),
        .busy(o_busy),
        .completion_valid(o_completion_valid),
        .completion_ready(release_public && output_owned),
        .completion_status(o_completion_status),
        .completion_sequence(o_completion_sequence),
        .completion_epoch(o_completion_epoch),
        .result_valid(o_result_valid),
        .result_address(o_result_address),
        .result_bytes(o_result_bytes),
        .tag_exhausted(o_tag_exhausted),
        .m_axi_araddr(o_m_axi_araddr),
        .m_axi_arlen(o_m_axi_arlen),
        .m_axi_arsize(o_m_axi_arsize),
        .m_axi_arprot(o_m_axi_arprot),
        .m_axi_arburst(o_m_axi_arburst),
        .m_axi_arid(o_m_axi_arid),
        .m_axi_arcache(o_m_axi_arcache),
        .m_axi_arlock(o_m_axi_arlock),
        .m_axi_arvalid(o_m_axi_arvalid),
        .m_axi_arready(o_m_axi_arready),
        .m_axi_rdata(o_m_axi_rdata),
        .m_axi_rresp(o_m_axi_rresp),
        .m_axi_rid(o_m_axi_rid),
        .m_axi_rlast(o_m_axi_rlast),
        .m_axi_rvalid(o_m_axi_rvalid),
        .m_axi_rready(o_m_axi_rready),
        .m_axi_awaddr(o_m_axi_awaddr),
        .m_axi_awlen(o_m_axi_awlen),
        .m_axi_awsize(o_m_axi_awsize),
        .m_axi_awprot(o_m_axi_awprot),
        .m_axi_awburst(o_m_axi_awburst),
        .m_axi_awid(o_m_axi_awid),
        .m_axi_awcache(o_m_axi_awcache),
        .m_axi_awlock(o_m_axi_awlock),
        .m_axi_awvalid(o_m_axi_awvalid),
        .m_axi_awready(o_m_axi_awready),
        .m_axi_wdata(o_m_axi_wdata),
        .m_axi_wstrb(o_m_axi_wstrb),
        .m_axi_wlast(o_m_axi_wlast),
        .m_axi_wvalid(o_m_axi_wvalid),
        .m_axi_wready(o_m_axi_wready),
        .m_axi_bid(o_m_axi_bid),
        .m_axi_bresp(o_m_axi_bresp),
        .m_axi_bvalid(o_m_axi_bvalid),
        .m_axi_bready(o_m_axi_bready),
        .source_completion_valid(r_completion_valid),
        .source_completion_ready(o_source_completion_ready),
        .source_completion_status(r_completion_status),
        .source_sequence(r_completion_sequence),
        .source_epoch(r_completion_epoch),
        .source_result_valid(r_result_valid),
        .source_result_ce(o_source_result_ce),
        .source_result_address(o_source_result_address),
        .source_result_q(r_result_q),
        .source_result_q_valid(r_result_q_valid),
        .completion_hls_status(o_completion_hls_status)
    );

    always_comb begin
        fault_now=fault;
        if(busy && state!=RESPONSE && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(!root_released && r_completion_valid && !root_identity_match)fault_now=BAD_PEER;
            else if(output_owned && o_completion_valid && !output_identity_match)fault_now=BAD_PEER;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;fault<=OK;revoked<=0;cancel_seen<=0;output_owned<=0;root_released<=0;
            completion_status<=OK;completion_hls_status<=0;completion_sequence<=0;completion_epoch<=0;
            saved_command_weight_addr<=0;
            saved_command_weight_region_base<=0;
            saved_command_weight_region_bytes<=0;
            saved_command_activation_addr<=0;
            saved_command_activation_region_base<=0;
            saved_command_activation_region_bytes<=0;
            saved_command_output_addr<=0;
            saved_command_output_region_base<=0;
            saved_command_output_region_bytes<=0;
        end else begin
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(busy && state!=RESPONSE)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            if(r_completion_valid && root_identity_match && (output_owned ? o_source_completion_ready:release_public))root_released<=1;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_status<=OK;completion_hls_status<=0;fault<=OK;revoked<=0;cancel_seen<=0;
                    output_owned<=0;root_released<=0;
                    saved_command_weight_addr<=command_weight_addr;
                    saved_command_weight_region_base<=command_weight_region_base;
                    saved_command_weight_region_bytes<=command_weight_region_bytes;
                    saved_command_activation_addr<=command_activation_addr;
                    saved_command_activation_region_base<=command_activation_region_base;
                    saved_command_activation_region_bytes<=command_activation_region_bytes;
                    saved_command_output_addr<=command_output_addr;
                    saved_command_output_region_base<=command_output_region_base;
                    saved_command_output_region_bytes<=command_output_region_bytes;
                    state<=WAIT_ROOT;
                end
                WAIT_ROOT:if(r_completion_valid && root_identity_match)begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(r_completion_status!=OK)begin completion_status<=r_completion_status;state<=RESPONSE;end
                    else if(!r_result_valid)begin completion_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else begin output_owned<=1;state<=OFFER_OUTPUT;end
                end
                // Once offered, finish the output-path handshake even after revocation.
                OFFER_OUTPUT:if(o_command_ready)state<=WAIT_OUTPUT;
                WAIT_OUTPUT:if(o_completion_valid && output_identity_match)begin
                    completion_hls_status<=o_completion_hls_status;
                    if(fault_now!=OK)completion_status<=fault_now;
                    else if(o_completion_status!=OK)completion_status<=o_completion_status;
                    else if(!o_result_valid || o_result_address!=saved_command_output_addr || o_result_bytes!=8192)
                        completion_status<=BAD_COMPLETION;
                    else completion_status<=OK;
                    state<=RESPONSE;
                end
                RESPONSE:if(release_public)begin output_owned<=0;state<=IDLE;end
                default:begin state<=IDLE;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
