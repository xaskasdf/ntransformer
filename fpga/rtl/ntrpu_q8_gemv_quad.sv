// Four HLS MAC lanes; isolated experimental variant. Layout 2 is [K block within tile][row][34 Q8 bytes].
// Rows map modulo four to a lane; each row retains strictly increasing K order.
// Scalar and dual baselines are preserved; qualify before FFN integration.
// All interfaces share clk. Reset must also reset/quiesce the external AXI peer.
module ntrpu_q8_gemv_quad #(
    parameter integer WATCHDOG_CYCLES = 65536
) (
    input logic clk, rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [1:0] job_layout, // Must be 2; row-major layout 1 is rejected.
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
    mac_state_t lane_state [0:3];
    logic [1:0] mac_state; // Aggregate occupancy for validation/diagnostics.
    logic [5:0] rows, row_cursor, column_cursor, k_count, output_row;
    logic [5:0] lane_row [0:3];
    logic [1:0] selected_lane;
    logic [2:0] completed_now;
    logic [8:0] total_k, next_k, k_base;
    logic [31:0] sequence_id, epoch, active_tag, active_bytes;
    logic [63:0] region_base, active_addr;
    logic [64:0] region_end, active_end;
    logic [10:0] expected_blocks, received_blocks, computed_blocks;
    logic [255:0] activation_present;
    (* ram_style = "block" *) logic [319:0] activation_mem [0:255];
    (* ram_style = "block" *) logic [319:0] activation_mem_23 [0:255];
    logic [319:0] activation_q [0:3];
    logic [63:0] accumulators [0:31];
    logic [255:0] weights_q [0:3];
    logic [15:0] weight_scale_q [0:3];
    logic [63:0] prior_q [0:3], lane_result [0:3], saved_lane_result [0:3];
    logic [3:0] lane_done, lane_result_valid, saved_lane_valid;
    logic [31:0] lane_status [0:3];
    logic mac_done;
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
    logic activation_write;
    logic [7:0] activation_port0_address;

`ifdef NTRPU_QUAD_BLOCK_FIFO
    // Two registered entries break the MAC-ready -> Q8 reservoir path.
    // Input readiness depends only on registered occupancy, never on pop.
    // A full FIFO deliberately waits one cycle after a pop before refilling.
    logic decoded_block_valid, decoded_block_ready, decoded_block_last;
    logic [15:0] decoded_block_scale;
    logic [255:0] decoded_block_weights;
    logic [9:0] decoded_block_index;
    logic [31:0] decoded_block_tag;
    logic [314:0] block_fifo [0:1];
    logic block_fifo_read, block_fifo_write;
    logic [1:0] block_fifo_count;
    logic decoded_block_fire;
    assign decoded_block_ready = state == RUN && block_fifo_count < 2;
    assign decoded_block_fire = decoded_block_valid && decoded_block_ready;
    assign block_valid = block_fifo_count != 0;
    assign {block_scale,block_weights,block_index,block_tag,block_last} = block_fifo[block_fifo_read];
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            block_fifo_read <= 0; block_fifo_write <= 0; block_fifo_count <= 0;
        end else begin
            if (decoded_block_fire) begin
                block_fifo[block_fifo_write] <= {decoded_block_scale,decoded_block_weights,
                    decoded_block_index,decoded_block_tag,decoded_block_last};
                block_fifo_write <= !block_fifo_write;
            end
            if (block_fire) block_fifo_read <= !block_fifo_read;
            case ({decoded_block_fire,block_fire})
                2'b10: block_fifo_count <= block_fifo_count + 2'd1;
                2'b01: block_fifo_count <= block_fifo_count - 2'd1;
                default: ;
            endcase
        end
    end
`endif

    assign selected_lane = row_cursor[1:0];
    always_comb begin
        mac_state = MAC_IDLE;
        completed_now = 0;
        for (int lane=0; lane<4; lane++) begin
            if (lane_state[lane] == MAC_WAIT) mac_state = MAC_WAIT;
            else if (lane_state[lane] == MAC_START && mac_state == MAC_IDLE) mac_state = MAC_START;
            if (lane_state[lane] == MAC_WAIT && lane_done[lane]) completed_now = completed_now + 3'd1;
        end
    end
    assign mac_done = |lane_done;
    assign proposed_blocks = 12'(rows) * 12'(tile_k_count);
    assign proposed_k_end = {1'b0,tile_k_base} + 10'(tile_k_count);
    assign proposed_region_end = {1'b0,job_region_base} + {1'b0,job_region_bytes};
    assign activation_address = k_base + 9'(column_cursor);
    assign activation_write = activation_valid && activation_ready && activation_epoch == epoch &&
                              {1'b0,activation_index} < total_k;
    // Port A writes during PREPARE or reads for lane 0 during RUN. Port B
    // reads for lane 1. A mirrored RAM provides the same ports for lanes 2/3.
    // Both copies share writes and the per-job activation-valid mask.
    assign activation_port0_address = activation_write ? activation_index : activation_address[7:0];
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
    assign block_ready = state == RUN && (error_code != OK || lane_state[selected_lane] == MAC_IDLE);
    assign block_fire = block_valid && block_ready;
    assign progress_event = (m_axi_arvalid && m_axi_arready) || (m_axi_rvalid && m_axi_rready) ||
`ifdef NTRPU_QUAD_BLOCK_FIFO
                            decoded_block_fire ||
`endif
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
                else begin
                    for (int lane=0; lane<4; lane++)
                        if (lane_state[lane] == MAC_WAIT && lane_done[lane] &&
                            (lane_status[lane] != 0 || !(saved_lane_valid[lane] || lane_result_valid[lane])))
                            error_now = MAC_ERROR;
                end
            end
        end
    end

    // Both payload copies are written only in PREPARE. Each read has an
    // explicit address and output register, preserving two-port BRAM inference.
    // Contents are not reset; activation_present excludes stale job contents.
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (activation_write) begin
                activation_mem[activation_port0_address] <= {activation_scale,activation_values};
                activation_mem_23[activation_port0_address] <= {activation_scale,activation_values};
            end
            if (block_fire && error_now == OK && selected_lane == 0)
                activation_q[0] <= activation_mem[activation_port0_address];
            if (block_fire && error_now == OK && selected_lane == 1)
                activation_q[1] <= activation_mem[activation_address[7:0]];
            if (block_fire && error_now == OK && selected_lane == 2)
                activation_q[2] <= activation_mem_23[activation_port0_address];
            if (block_fire && error_now == OK && selected_lane == 3)
                activation_q[3] <= activation_mem_23[activation_address[7:0]];
        end
    end

    ntrpu_q8_weight_stream weights (
        .clk, .rst_n, .load_valid(state == ISSUE), .load_ready,
        .load_addr(active_addr), .load_bytes(active_bytes), .load_tag(active_tag),
        .load_done_valid, .load_done_ready(state == RUN), .load_done_status, .load_done_bytes, .load_done_tag,
`ifdef NTRPU_QUAD_BLOCK_FIFO
        .block_scale(decoded_block_scale), .block_weights(decoded_block_weights),
        .block_index(decoded_block_index), .block_tag(decoded_block_tag),
        .block_last(decoded_block_last), .block_valid(decoded_block_valid), .block_ready(decoded_block_ready),
`else
        .block_scale, .block_weights, .block_index, .block_tag, .block_last, .block_valid, .block_ready,
`endif
        .decode_done_valid, .decode_done_ready(state == RUN), .decode_done_status, .decode_done_blocks, .decode_done_tag,
        .m_axi_araddr, .m_axi_arlen, .m_axi_arsize, .m_axi_arburst, .m_axi_arid,
        .m_axi_arlock, .m_axi_arcache, .m_axi_arprot, .m_axi_arvalid, .m_axi_arready,
        .m_axi_rdata, .m_axi_rresp, .m_axi_rid, .m_axi_rlast, .m_axi_rvalid, .m_axi_rready
    );
    for (genvar lane=0; lane<4; lane++) begin : workers
        ntrpu_q8_block_mac mac (
            .ap_clk(clk), .ap_rst(!rst_n), .ap_start(lane_state[lane] == MAC_START),
            .ap_done(lane_done[lane]), .ap_idle(), .ap_ready(),
            .weights_0(weights_q[lane][63:0]), .weights_1(weights_q[lane][127:64]),
            .weights_2(weights_q[lane][191:128]), .weights_3(weights_q[lane][255:192]),
            .input_0(activation_q[lane][63:0]), .input_1(activation_q[lane][127:64]),
            .input_2(activation_q[lane][191:128]), .input_3(activation_q[lane][255:192]),
            .weight_scale(weight_scale_q[lane]), .input_scale(activation_q[lane][319:256]),
            .accumulator(prior_q[lane]), .result(lane_result[lane]),
            .result_ap_vld(lane_result_valid[lane]), .ap_return(lane_status[lane])
        );
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; rows <= 0; total_k <= 0; next_k <= 0;
            row_cursor <= 0; column_cursor <= 0; k_count <= 0; k_base <= 0; output_row <= 0;
            sequence_id <= 0; epoch <= 0; active_tag <= 0; active_bytes <= 0;
            region_base <= 0; region_end <= 0; active_addr <= 0; active_end <= 0;
            expected_blocks <= 0; received_blocks <= 0; computed_blocks <= 0;
            activation_present <= 0; saved_lane_valid <= 0; error_code <= OK;
            for (int lane=0; lane<4; lane++) begin
                lane_state[lane] <= MAC_IDLE; lane_row[lane] <= 0;
                weights_q[lane] <= 0; weight_scale_q[lane] <= 0; prior_q[lane] <= 0;
                saved_lane_result[lane] <= 0;
            end
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
                    if (job_layout != 2 || job_rows == 0 || job_rows > 32 || job_k_blocks == 0 || job_k_blocks > 256 ||
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
                            weights_q[selected_lane] <= block_weights;
                            weight_scale_q[selected_lane] <= block_scale;
                            prior_q[selected_lane] <= accumulators[row_cursor[4:0]];
                            lane_row[selected_lane] <= row_cursor;
                            saved_lane_valid[selected_lane] <= 0; lane_state[selected_lane] <= MAC_START;
                            if (row_cursor + 6'd1 == rows) begin
                                row_cursor <= 0; column_cursor <= column_cursor + 6'd1;
                            end else row_cursor <= row_cursor + 6'd1;
                        end
                    end
                    if (error_now == OK) computed_blocks <= computed_blocks + 11'(completed_now);
                    for (int lane=0; lane<4; lane++) begin
                        if (lane_state[lane] == MAC_START) lane_state[lane] <= MAC_WAIT;
                        if (lane_state[lane] == MAC_WAIT) begin
                            if (lane_result_valid[lane]) begin
                                saved_lane_result[lane] <= lane_result[lane]; saved_lane_valid[lane] <= 1;
                            end
                            if (lane_done[lane]) begin
                                lane_state[lane] <= MAC_IDLE;
                                if (error_now == OK)
                                    accumulators[lane_row[lane][4:0]] <= lane_result_valid[lane] ?
                                        lane_result[lane] : saved_lane_result[lane];
                            end
                        end
                    end
                    if (load_seen && (load_failed || decode_seen) && mac_state == MAC_IDLE
`ifdef NTRPU_QUAD_BLOCK_FIFO
                        && !block_valid && !decoded_block_valid
`endif
                    ) begin
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
