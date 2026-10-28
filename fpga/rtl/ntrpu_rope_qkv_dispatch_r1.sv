// Actual archived RoPE input hardware followed by capture and dispatch.
// Distinct AXI read ports: Q/K/coefficients and V. No fabric arbitration here.
// Caller excludes other cache users from admission through final source drain.
// Source allocations stay immutable; reset must quiesce all children and peers.
`default_nettype none
module ntrpu_rope_qkv_dispatch_r1 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144
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
    input wire cancel_valid,
    output wire cancel_ready,
    output wire busy,
    output wire cache_exclusive,
    input wire cache_idle,
    input wire cache_valid,
    input wire cache_metadata_fault,
    input wire [63:0] cache_sequence,
    input wire [63:0] cache_epoch,
    input wire [31:0] cache_capacity,
    input wire [31:0] cache_committed_tokens,
    output wire append_valid,
    input wire append_ready,
    input wire append_prepared,
    output wire [31:0] append_layer,
    output wire [31:0] append_position,
    output wire [63:0] append_sequence,
    output wire [63:0] append_epoch,
    output wire data_valid,
    input wire data_ready,
    output wire [511:0] data_bits,
    output wire [5:0] data_index,
    output wire data_last,
    output wire [63:0] data_sequence,
    output wire [63:0] data_epoch,
    output wire commit_valid,
    input wire commit_ready,
    output wire [3:0] commit_status,
    output wire [63:0] commit_sequence,
    output wire [63:0] commit_epoch,
    output wire append_cancel_valid,
    input wire append_cancel_ready,
    input wire append_done_valid,
    output wire append_done_ready,
    input wire append_done_bind,
    input wire [3:0] append_done_status,
    input wire [12:0] append_done_bytes,
    input wire [31:0] append_done_committed_tokens,
    input wire [63:0] append_done_sequence,
    input wire [63:0] append_done_epoch,
    output wire root_command_valid,
    input wire root_command_ready,
    output wire [31:0] root_command_layer,
    output wire [31:0] root_command_position,
    output wire [63:0] root_command_sequence,
    output wire [63:0] root_command_epoch,
    output wire [63:0] root_command_lease_base,
    output wire q_valid,
    input wire q_ready,
    output wire [511:0] q_data,
    output wire [7:0] q_index,
    output wire q_last,
    output wire [63:0] q_sequence,
    output wire [63:0] q_epoch,
    output wire root_cancel_valid,
    input wire root_cancel_ready,
    input wire root_completion_valid,
    output wire root_completion_ready,
    input wire [3:0] root_status,
    input wire [63:0] root_sequence,
    input wire [63:0] root_epoch,
    input wire root_result_valid,
    output wire root_result_ce,
    output wire [10:0] root_result_address,
    input wire [31:0] root_result_q,
    input wire root_result_q_valid,
    output wire completion_valid,
    input wire completion_ready,
    output wire [3:0] completion_status,
    output wire [3:0] completion_append_status,
    output wire [3:0] completion_root_status,
    output wire [63:0] completion_sequence,
    output wire [63:0] completion_epoch,
    output wire [31:0] completion_layer,
    output wire [31:0] completion_position,
    output wire [6:0] completion_kv_frames,
    output wire [7:0] completion_q_frames,
    output wire [12:0] completion_kv_bytes,
    output wire [31:0] completion_kv_tokens,
    output wire completion_kv_committed,
    output wire result_valid,
    output wire completion_revoked,
    input wire result_ce,
    input wire [10:0] result_address,
    output wire [31:0] result_q,
    output wire result_q_valid,
    input wire [63:0] command_v_address,
    input wire [63:0] command_v_base,
    input wire [63:0] command_v_bytes,
    input wire command_v_source_valid,
    input wire [63:0] command_v_source_sequence,
    input wire [63:0] command_v_source_epoch,
    input wire [31:0] command_v_source_position,
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
    output wire [3:0] completion_capture_status,
    output wire [11:0] completion_capture_qk_words,
    output wire [5:0] completion_capture_v_beats,
    input wire [63:0] command_source_sequence,
    input wire [63:0] command_source_epoch,
    input wire command_source_valid,
    input wire [31:0] command_source_position,
    input wire [31:0] command_coefficient_position,
    input wire [63:0] command_q_address,
    input wire [63:0] command_q_base,
    input wire [63:0] command_q_bytes,
    input wire [63:0] command_k_address,
    input wire [63:0] command_k_base,
    input wire [63:0] command_k_bytes,
    input wire [63:0] command_coefficient_address,
    input wire [63:0] command_coefficient_base,
    input wire [63:0] command_coefficient_bytes,
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
    output wire [3:0] completion_rope_status,
    output wire [31:0] completion_rope_hls_status,
    output wire [9:0] completion_rope_beats
);
    typedef enum logic [2:0] {IDLE,ROPE,OFFER,RUN,DRAIN} state_t;
    state_t state;
    logic cancel_seen;
    logic [63:0] held_command_sequence;
    logic [63:0] held_command_epoch;
    logic [63:0] held_command_lease_base;
    logic [31:0] held_command_layer;
    logic [31:0] held_command_position;
    logic [63:0] held_command_v_address;
    logic [63:0] held_command_v_base;
    logic [63:0] held_command_v_bytes;
    logic held_command_v_source_valid;
    logic [63:0] held_command_v_source_sequence;
    logic [63:0] held_command_v_source_epoch;
    logic [31:0] held_command_v_source_position;
    wire rope_cancel_ready;
    wire rope_busy;
    wire rope_completion_valid;
    wire rope_completion_ready;
    wire [3:0] rope_completion_status;
    wire [63:0] rope_completion_sequence;
    wire [63:0] rope_completion_epoch;
    wire [9:0] rope_completion_beats;
    wire [31:0] rope_completion_position;
    wire [31:0] rope_completion_coefficient_position;
    wire rope_result_valid;
    wire [31:0] rope_completion_hls_status;
    wire rope_completion_revoked;
    wire rope_result_ce;
    wire [11:0] rope_result_address;
    wire [31:0] rope_result_q;
    wire rope_result_q_valid;
    wire rope_command_ready;
    wire capture_command_ready;
    wire capture_cancel_ready;
    wire capture_busy;
    wire capture_cache_exclusive;
    wire capture_completion_valid;
    wire capture_result_valid;
    wire capture_completion_revoked;
    wire [31:0] capture_result_q;
    wire capture_result_q_valid;
    wire retained_cancel=cancel_seen || (cancel_valid && state!=DRAIN);
    assign busy=state!=IDLE;
    assign cache_exclusive=busy;
    assign command_ready=state==IDLE && rope_command_ready && capture_command_ready;
    assign cancel_ready=busy && state!=DRAIN && !cancel_seen;
    assign completion_valid=state==RUN && capture_completion_valid;
    assign result_valid=state==RUN && capture_result_valid;
    assign result_q_valid=state==RUN && capture_result_q_valid;
    assign result_q=result_q_valid ? capture_result_q:32'd0;
    assign completion_revoked=capture_completion_revoked;
    assign completion_rope_status=rope_completion_status;
    assign completion_rope_hls_status=rope_completion_hls_status;
    assign completion_rope_beats=rope_completion_beats;
    always_ff @(posedge clk)begin
        if(!rst_n)begin state<=IDLE;cancel_seen<=0;
            held_command_sequence<=0;
            held_command_epoch<=0;
            held_command_lease_base<=0;
            held_command_layer<=0;
            held_command_position<=0;
            held_command_v_address<=0;
            held_command_v_base<=0;
            held_command_v_bytes<=0;
            held_command_v_source_valid<=0;
            held_command_v_source_sequence<=0;
            held_command_v_source_epoch<=0;
            held_command_v_source_position<=0;
        end else begin
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    cancel_seen<=cancel_valid;state<=ROPE;
                    held_command_sequence<=command_sequence;
                    held_command_epoch<=command_epoch;
                    held_command_lease_base<=command_lease_base;
                    held_command_layer<=command_layer;
                    held_command_position<=command_position;
                    held_command_v_address<=command_v_address;
                    held_command_v_base<=command_v_base;
                    held_command_v_bytes<=command_v_bytes;
                    held_command_v_source_valid<=command_v_source_valid;
                    held_command_v_source_sequence<=command_v_source_sequence;
                    held_command_v_source_epoch<=command_v_source_epoch;
                    held_command_v_source_position<=command_v_source_position;
                end
                ROPE:if(rope_completion_valid)state<=OFFER;
                OFFER:if(capture_command_ready)state<=RUN;
                RUN:if(capture_completion_valid && completion_ready)state<=DRAIN;
                DRAIN:if(!rope_busy && !capture_busy)begin cancel_seen<=0;state<=IDLE;end
                default:state<=DRAIN;
            endcase
        end
    end
    ntrpu_rope_input_real_r1 #(.AXI_WATCHDOG_CYCLES(WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) rope(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(state==IDLE && command_valid && command_ready),
        .command_ready(rope_command_ready),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .expected_epoch(expected_epoch),
        .command_source_sequence(command_source_sequence),
        .command_source_epoch(command_source_epoch),
        .command_source_valid(command_source_valid),
        .command_position(command_position),
        .command_source_position(command_source_position),
        .command_coefficient_position(command_coefficient_position),
        .command_q_address(command_q_address),
        .command_q_base(command_q_base),
        .command_q_bytes(command_q_bytes),
        .command_k_address(command_k_address),
        .command_k_base(command_k_base),
        .command_k_bytes(command_k_bytes),
        .command_coefficient_address(command_coefficient_address),
        .command_coefficient_base(command_coefficient_base),
        .command_coefficient_bytes(command_coefficient_bytes),
        .cancel_valid(retained_cancel),
        .cancel_ready(rope_cancel_ready),
        .busy(rope_busy),
        .completion_valid(rope_completion_valid),
        .completion_ready(rope_completion_ready),
        .completion_status(rope_completion_status),
        .completion_sequence(rope_completion_sequence),
        .completion_epoch(rope_completion_epoch),
        .completion_beats(rope_completion_beats),
        .completion_position(rope_completion_position),
        .completion_coefficient_position(rope_completion_coefficient_position),
        .result_valid(rope_result_valid),
        .completion_hls_status(rope_completion_hls_status),
        .completion_revoked(rope_completion_revoked),
        .result_ce(rope_result_ce),
        .result_address(rope_result_address),
        .result_q(rope_result_q),
        .result_q_valid(rope_result_q_valid),
        .m_axi_araddr(rope_axi_araddr),
        .m_axi_arlen(rope_axi_arlen),
        .m_axi_arsize(rope_axi_arsize),
        .m_axi_arprot(rope_axi_arprot),
        .m_axi_arburst(rope_axi_arburst),
        .m_axi_arid(rope_axi_arid),
        .m_axi_arcache(rope_axi_arcache),
        .m_axi_arlock(rope_axi_arlock),
        .m_axi_arvalid(rope_axi_arvalid),
        .m_axi_arready(rope_axi_arready),
        .m_axi_rdata(rope_axi_rdata),
        .m_axi_rresp(rope_axi_rresp),
        .m_axi_rid(rope_axi_rid),
        .m_axi_rlast(rope_axi_rlast),
        .m_axi_rvalid(rope_axi_rvalid),
        .m_axi_rready(rope_axi_rready)
    );
    ntrpu_qkv_capture_dispatch_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) capture(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(state==OFFER),
        .command_ready(capture_command_ready),
        .command_sequence(held_command_sequence),
        .command_epoch(held_command_epoch),
        .command_lease_base(held_command_lease_base),
        .expected_epoch(expected_epoch),
        .command_layer(held_command_layer),
        .command_position(held_command_position),
        .cancel_valid(retained_cancel),
        .cancel_ready(capture_cancel_ready),
        .busy(capture_busy),
        .cache_exclusive(capture_cache_exclusive),
        .cache_idle(cache_idle),
        .cache_valid(cache_valid),
        .cache_metadata_fault(cache_metadata_fault),
        .cache_sequence(cache_sequence),
        .cache_epoch(cache_epoch),
        .cache_capacity(cache_capacity),
        .cache_committed_tokens(cache_committed_tokens),
        .append_valid(append_valid),
        .append_ready(append_ready),
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
        .root_command_valid(root_command_valid),
        .root_command_ready(root_command_ready),
        .root_command_layer(root_command_layer),
        .root_command_position(root_command_position),
        .root_command_sequence(root_command_sequence),
        .root_command_epoch(root_command_epoch),
        .root_command_lease_base(root_command_lease_base),
        .q_valid(q_valid),
        .q_ready(q_ready),
        .q_data(q_data),
        .q_index(q_index),
        .q_last(q_last),
        .q_sequence(q_sequence),
        .q_epoch(q_epoch),
        .root_cancel_valid(root_cancel_valid),
        .root_cancel_ready(root_cancel_ready),
        .root_completion_valid(root_completion_valid),
        .root_completion_ready(root_completion_ready),
        .root_status(root_status),
        .root_sequence(root_sequence),
        .root_epoch(root_epoch),
        .root_result_valid(root_result_valid),
        .root_result_ce(root_result_ce),
        .root_result_address(root_result_address),
        .root_result_q(root_result_q),
        .root_result_q_valid(root_result_q_valid),
        .completion_valid(capture_completion_valid),
        .completion_ready(state==RUN && completion_ready),
        .completion_status(completion_status),
        .completion_append_status(completion_append_status),
        .completion_root_status(completion_root_status),
        .completion_sequence(completion_sequence),
        .completion_epoch(completion_epoch),
        .completion_layer(completion_layer),
        .completion_position(completion_position),
        .completion_kv_frames(completion_kv_frames),
        .completion_q_frames(completion_q_frames),
        .completion_kv_bytes(completion_kv_bytes),
        .completion_kv_tokens(completion_kv_tokens),
        .completion_kv_committed(completion_kv_committed),
        .result_valid(capture_result_valid),
        .completion_revoked(capture_completion_revoked),
        .result_ce(state==RUN && result_ce),
        .result_address(result_address),
        .result_q(capture_result_q),
        .result_q_valid(capture_result_q_valid),
        .command_v_address(held_command_v_address),
        .command_v_base(held_command_v_base),
        .command_v_bytes(held_command_v_bytes),
        .command_v_source_valid(held_command_v_source_valid),
        .command_v_source_sequence(held_command_v_source_sequence),
        .command_v_source_epoch(held_command_v_source_epoch),
        .command_v_source_position(held_command_v_source_position),
        .source_completion_valid(rope_completion_valid),
        .source_completion_ready(rope_completion_ready),
        .source_status(rope_completion_status),
        .source_sequence(rope_completion_sequence),
        .source_epoch(rope_completion_epoch),
        .source_position(rope_completion_position),
        .source_coefficient_position(rope_completion_coefficient_position),
        .source_hls_status(rope_completion_hls_status),
        .source_beats(rope_completion_beats),
        .source_result_valid(rope_result_valid),
        .source_ce(rope_result_ce),
        .source_address(rope_result_address),
        .source_q(rope_result_q),
        .source_q_valid(rope_result_q_valid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arid(m_axi_arid),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arlock(m_axi_arlock),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .completion_capture_status(completion_capture_status),
        .completion_capture_qk_words(completion_capture_qk_words),
        .completion_capture_v_beats(completion_capture_v_beats)
    );
endmodule
`default_nettype wire
