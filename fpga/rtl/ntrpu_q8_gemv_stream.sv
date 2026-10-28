// One GEMV job, private row accumulators, sequential K tiles and a real HLS MAC.
// A job's weights must be prepacked row-major within each submitted K tile.
// All interfaces share clk. Reset must also reset/quiesce the external AXI peer.
module ntrpu_q8_gemv_stream #(
    parameter integer WATCHDOG_CYCLES = 65536
) (
    input logic clk, rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [5:0] job_rows,
    input logic [8:0] job_k_blocks,
    input logic [31:0] job_sequence, job_epoch,
    input logic [63:0] job_region_base, job_region_bytes,
    input logic activation_valid,
    output logic activation_ready,
    input logic [7:0] activation_index,
    input logic [255:0] activation_values,
    input logic [63:0] activation_scale,
    input logic [31:0] activation_epoch,
    input logic tile_valid,
    output logic tile_ready,
    input logic [63:0] tile_addr,
    input logic [8:0] tile_k_base,
    input logic [5:0] tile_k_count,
    input logic [31:0] tile_tag, tile_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic result_valid,
    input logic result_ready,
    output logic [63:0] result_data,
    output logic [4:0] result_row,
    output logic result_last,
    output logic [31:0] result_sequence, result_epoch,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
    output logic [31:0] done_sequence, done_epoch,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize, m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid, m_axi_arcache,
    output logic m_axi_arlock, m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready
);
    localparam logic [3:0] OK=0, BAD_COMMAND=1, BAD_ACTIVATION=2,
        TRANSPORT_ERROR=3, DECODE_ERROR=4, BLOCK_ERROR=5, MAC_ERROR=6,
        CANCELLED=7, TIMEOUT=8;
    typedef enum logic [3:0] {IDLE, PREPARE, TILE_SIZE, TILE_END, TILE_CHECK,
                             ISSUE, RUN, RESULTS, COMPLETE} state_t;
    typedef enum logic [1:0] {MAC_IDLE, MAC_START, MAC_WAIT} mac_state_t;
    state_t state;
    mac_state_t mac_state;
    logic [5:0] rows, row_cursor, column_cursor, k_count, output_row, mac_row;
    logic [8:0] total_k, next_k, k_base;
    logic [31:0] sequence_id, epoch, active_tag, active_bytes;
    logic [63:0] region_base, active_addr;
    logic [64:0] region_end, active_end;
    logic [10:0] expected_blocks, received_blocks, computed_blocks;
    logic [255:0] activation_present;
    (* ram_style = "block" *) logic [319:0] activation_mem [0:255];
    logic [319:0] activation_q;
    logic [63:0] accumulators [0:31];
    logic [255:0] weights_q;
    logic [15:0] weight_scale_q;
    logic [63:0] prior_q, mac_result, saved_mac_result;
    logic mac_done, mac_result_valid, saved_mac_valid;
    logic [31:0] mac_status;
    logic [3:0] error_code, error_now;
    logic [31:0] idle_cycles;
    logic load_seen, decode_seen, load_failed;
    logic load_ready, load_done_valid, decode_done_valid, block_valid, block_ready;
    logic [1:0] load_done_status, decode_done_status;
    logic [31:0] load_done_bytes, load_done_tag, decode_done_tag, block_tag;
    logic [10:0] decode_done_blocks;
    logic [9:0] block_index;
    logic [15:0] block_scale;
    logic [255:0] block_weights;
    logic block_last;
    logic [11:0] proposed_blocks;
    logic [9:0] proposed_k_end;
    logic [64:0] proposed_region_end;
    logic all_activations, block_fire, progress_event, watchdog_expired;
    logic [8:0] activation_address;

    assign proposed_blocks = 12'(rows) * 12'(tile_k_count);
    assign proposed_k_end = {1'b0,tile_k_base} + 10'(tile_k_count);
    assign proposed_region_end = {1'b0,job_region_base} + {1'b0,job_region_bytes};
    assign activation_address = k_base + 9'(column_cursor);
    always_comb begin
        all_activations = 1;
        for (int i=0; i<256; i++)
            if (i < int'(total_k) && !activation_present[i]) all_activations = 0;
    end
    assign job_ready = state == IDLE;
    assign activation_ready = state == PREPARE && next_k == 0 && !cancel_valid;
    assign tile_ready = state == PREPARE && !(activation_valid && next_k == 0) && !cancel_valid;
    assign cancel_ready = state == PREPARE || state == TILE_SIZE || state == TILE_END ||
                          state == TILE_CHECK || state == ISSUE || state == RUN;
    assign result_valid = state == RESULTS;
    assign result_data = accumulators[output_row[4:0]];
    assign result_row = output_row[4:0];
    assign result_last = output_row + 6'd1 == rows;
    assign result_sequence = sequence_id;
    assign result_epoch = epoch;
    assign done_valid = state == COMPLETE;
    assign done_status = error_code;
    assign done_sequence = sequence_id;
    assign done_epoch = epoch;
    assign block_ready = state == RUN && (error_code != OK || mac_state == MAC_IDLE);
    assign block_fire = block_valid && block_ready;
    assign progress_event = (m_axi_arvalid && m_axi_arready) || (m_axi_rvalid && m_axi_rready) ||
                            block_fire || (state == RUN && (load_done_valid || decode_done_valid || mac_done)) ||
                            (state == ISSUE && load_ready);
    assign watchdog_expired = (state == ISSUE || state == RUN) &&
                               idle_cycles >= 32'(WATCHDOG_CYCLES-1) && !progress_event;

    // Sticky first failure, including events coincident with the last MAC/done.
    always_comb begin
        error_now = error_code;
        if (error_code == OK) begin
            if (cancel_valid && cancel_ready) error_now = CANCELLED;
            else if (watchdog_expired) error_now = TIMEOUT;
            else if (state == RUN) begin
                if (load_done_valid && (load_done_status != 0 || load_done_tag != active_tag ||
                                       load_done_bytes != active_bytes)) error_now = TRANSPORT_ERROR;
                else if (decode_done_valid && (decode_done_status != 0 || decode_done_tag != active_tag ||
                                               decode_done_blocks != expected_blocks)) error_now = DECODE_ERROR;
                else if (block_fire && (received_blocks >= expected_blocks || block_tag != active_tag ||
                         block_index != received_blocks[9:0] ||
                         block_last != (received_blocks + 11'd1 == expected_blocks))) error_now = BLOCK_ERROR;
                else if (mac_state == MAC_WAIT && mac_done &&
                         (mac_status != 0 || !(saved_mac_valid || mac_result_valid))) error_now = MAC_ERROR;
            end
        end
    end

    // RAM payload is not reset; the per-job valid mask excludes stale entries.
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (activation_valid && activation_ready && activation_epoch == epoch &&
                {1'b0,activation_index} < total_k)
                activation_mem[activation_index] <= {activation_scale,activation_values};
            if (block_fire && error_now == OK)
                activation_q <= activation_mem[activation_address[7:0]];
        end
    end

    ntrpu_q8_weight_stream weights (
        .clk, .rst_n, .load_valid(state == ISSUE), .load_ready,
        .load_addr(active_addr), .load_bytes(active_bytes), .load_tag(active_tag),
        .load_done_valid, .load_done_ready(state == RUN), .load_done_status, .load_done_bytes, .load_done_tag,
        .block_scale, .block_weights, .block_index, .block_tag, .block_last, .block_valid, .block_ready,
        .decode_done_valid, .decode_done_ready(state == RUN), .decode_done_status, .decode_done_blocks, .decode_done_tag,
        .m_axi_araddr, .m_axi_arlen, .m_axi_arsize, .m_axi_arburst, .m_axi_arid,
        .m_axi_arlock, .m_axi_arcache, .m_axi_arprot, .m_axi_arvalid, .m_axi_arready,
        .m_axi_rdata, .m_axi_rresp, .m_axi_rid, .m_axi_rlast, .m_axi_rvalid, .m_axi_rready
    );
    ntrpu_q8_block_mac mac (
        .ap_clk(clk), .ap_rst(!rst_n), .ap_start(mac_state == MAC_START),
        .ap_done(mac_done), .ap_idle(), .ap_ready(),
        .weights_0(weights_q[63:0]), .weights_1(weights_q[127:64]),
        .weights_2(weights_q[191:128]), .weights_3(weights_q[255:192]),
        .input_0(activation_q[63:0]), .input_1(activation_q[127:64]),
        .input_2(activation_q[191:128]), .input_3(activation_q[255:192]),
        .weight_scale(weight_scale_q), .input_scale(activation_q[319:256]),
        .accumulator(prior_q), .result(mac_result), .result_ap_vld(mac_result_valid), .ap_return(mac_status)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; mac_state <= MAC_IDLE; rows <= 0; total_k <= 0; next_k <= 0;
            row_cursor <= 0; column_cursor <= 0; k_count <= 0; k_base <= 0; output_row <= 0; mac_row <= 0;
            sequence_id <= 0; epoch <= 0; active_tag <= 0; active_bytes <= 0;
            region_base <= 0; region_end <= 0; active_addr <= 0; active_end <= 0;
            expected_blocks <= 0; received_blocks <= 0; computed_blocks <= 0;
            activation_present <= 0; weights_q <= 0; weight_scale_q <= 0; prior_q <= 0;
            saved_mac_result <= 0; saved_mac_valid <= 0; error_code <= OK;
            load_seen <= 0; decode_seen <= 0; load_failed <= 0; idle_cycles <= 0;
            for (int i=0; i<32; i++) accumulators[i] <= 0;
        end else begin
            if (state == ISSUE || state == RUN) begin
                error_code <= error_now;
                if (progress_event) idle_cycles <= 0;
                else if (idle_cycles < 32'(WATCHDOG_CYCLES)) idle_cycles <= idle_cycles + 32'd1;
            end else idle_cycles <= 0;
            case (state)
                IDLE: if (job_valid) begin
                    rows <= job_rows; total_k <= job_k_blocks; next_k <= 0;
                    sequence_id <= job_sequence; epoch <= job_epoch;
                    region_base <= job_region_base; region_end <= proposed_region_end;
                    activation_present <= 0; output_row <= 0; error_code <= OK;
                    for (int i=0; i<32; i++) accumulators[i] <= 0;
                    if (job_rows == 0 || job_rows > 32 || job_k_blocks == 0 || job_k_blocks > 256 ||
                        job_region_bytes == 0 || proposed_region_end[64]) begin
                        error_code <= BAD_COMMAND; state <= COMPLETE;
                    end else state <= PREPARE;
                end
                PREPARE: begin
                    if (cancel_valid) begin error_code <= CANCELLED; state <= COMPLETE; end
                    else if (activation_valid && activation_ready) begin
                        if (activation_epoch != epoch || {1'b0,activation_index} >= total_k) begin
                            error_code <= BAD_ACTIVATION; state <= COMPLETE;
                        end else activation_present[activation_index] <= 1;
                    end else if (tile_valid && tile_ready) begin
                        if (tile_epoch != epoch || tile_k_base != next_k || tile_k_count == 0 || tile_k_count > 32 ||
                            proposed_k_end > {1'b0,total_k}) begin
                            error_code <= BAD_COMMAND; state <= COMPLETE;
                        end else if (!all_activations) begin error_code <= BAD_ACTIVATION; state <= COMPLETE; end
                        else begin
                            active_addr <= tile_addr; active_tag <= tile_tag;
                            k_base <= tile_k_base; k_count <= tile_k_count; expected_blocks <= proposed_blocks[10:0];
                            row_cursor <= 0; column_cursor <= 0; received_blocks <= 0; computed_blocks <= 0;
                            load_seen <= 0; decode_seen <= 0; load_failed <= 0; state <= TILE_SIZE;
                        end
                    end
                end
                // Register the descriptor arithmetic before touching the reader.
                // This removes multiply -> length -> 65-bit bounds from one cycle.
                TILE_SIZE: begin
                    if (cancel_valid) begin error_code <= CANCELLED; state <= COMPLETE; end
                    else if (expected_blocks == 0 || expected_blocks > 1024 || expected_blocks[4:0] != 0) begin
                        error_code <= BAD_COMMAND; state <= COMPLETE;
                    end else begin
                        active_bytes <= (32'(expected_blocks) << 5) + (32'(expected_blocks) << 1);
                        state <= TILE_END;
                    end
                end
                TILE_END: begin
                    if (cancel_valid) begin error_code <= CANCELLED; state <= COMPLETE; end
                    else begin
                        active_end <= {1'b0,active_addr} + {33'b0,active_bytes};
                        state <= TILE_CHECK;
                    end
                end
                TILE_CHECK: begin
                    if (cancel_valid) begin error_code <= CANCELLED; state <= COMPLETE; end
                    else if (active_addr[5:0] != 0 || active_addr < region_base || active_end > region_end) begin
                        error_code <= BAD_COMMAND; state <= COMPLETE;
                    end else state <= ISSUE;
                end
                ISSUE: if (load_ready) state <= RUN;
                RUN: begin
                    if (load_done_valid) begin load_seen <= 1; load_failed <= load_done_status != 0; end
                    if (decode_done_valid) decode_seen <= 1;
                    if (block_fire) begin
                        received_blocks <= received_blocks + 11'd1;
                        if (error_now == OK) begin
                            weights_q <= block_weights; weight_scale_q <= block_scale;
                            prior_q <= accumulators[row_cursor[4:0]]; mac_row <= row_cursor;
                            saved_mac_valid <= 0; mac_state <= MAC_START;
                            if (column_cursor + 6'd1 == k_count) begin
                                column_cursor <= 0; row_cursor <= row_cursor + 6'd1;
                            end else column_cursor <= column_cursor + 6'd1;
                        end
                    end
                    if (mac_state == MAC_START) mac_state <= MAC_WAIT;
                    if (mac_state == MAC_WAIT) begin
                        if (mac_result_valid) begin saved_mac_result <= mac_result; saved_mac_valid <= 1; end
                        if (mac_done) begin
                            mac_state <= MAC_IDLE;
                            if (error_now == OK) begin
                                accumulators[mac_row[4:0]] <= mac_result_valid ? mac_result : saved_mac_result;
                                computed_blocks <= computed_blocks + 11'd1;
                            end
                        end
                    end
                    if (load_seen && (load_failed || decode_seen) && mac_state == MAC_IDLE) begin
                        if (error_now != OK) state <= COMPLETE;
                        else if (received_blocks != expected_blocks || computed_blocks != expected_blocks) begin
                            error_code <= BLOCK_ERROR; state <= COMPLETE;
                        end else if (k_base + 9'(k_count) == total_k) begin
                            output_row <= 0; state <= RESULTS;
                        end else begin next_k <= k_base + 9'(k_count); state <= PREPARE; end
                    end
                end
                RESULTS: if (result_ready) begin
                    if (result_last) state <= COMPLETE;
                    else output_row <= output_row + 6'd1;
                end
                COMPLETE: if (done_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
