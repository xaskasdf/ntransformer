// 64-byte beats -> Q8_0 blocks (FP16 bits + 32 INT8 lanes), without arithmetic.
// Outputs are provisional until done_status == 0. See docs/fpga/Q8_UNPACKER.md.
module ntrpu_q8_unpacker (
    input  logic         clk,
    input  logic         rst_n,
    input  logic [511:0] in_data,
    input  logic         in_valid,
    output logic         in_ready,
    input  logic         in_last,
    input  logic [31:0]  in_tag,
    output logic [15:0]  block_scale,
    output logic [255:0] block_weights,
    output logic [9:0]   block_index,
    output logic [31:0]  block_tag,
    output logic         block_last,
    output logic         block_valid,
    input  logic         block_ready,
    output logic         done_valid,
    input  logic         done_ready,
    output logic [1:0]   done_status,
    output logic [10:0]  done_blocks,
    output logic [31:0]  done_tag
);
    typedef enum logic [2:0] {IDLE, STREAM, DRAIN, FLUSH, COMPLETE} state_t;
    state_t state;
    logic [1023:0] reservoir, reservoir_next;
    logic [7:0] byte_count, count_next;
    logic [9:0] input_beats;
    logic [4:0] group_beats;
    logic [9:0] next_index;
    logic last_seen, push, pop, format_error, scale_error;

    assign in_ready = state == IDLE || state == DRAIN ||
                      (state == STREAM && !last_seen && byte_count <= 64);
    assign push = in_valid && in_ready;
    // Valid packet lengths are 17, 34, ..., 544 beats: 32..1024 Q8 blocks.
    assign format_error = state == STREAM && push &&
                          (in_tag != done_tag || (in_last && group_beats != 16) ||
                           (input_beats == 543 && !in_last));
    assign scale_error = state == STREAM && byte_count >= 34 && reservoir[14:10] == 5'b11111;
    assign pop = state == STREAM && byte_count >= 34 && (!block_valid || block_ready) &&
                 !format_error && !scale_error;
    assign done_valid = state == COMPLETE;

    always_comb begin
        reservoir_next = reservoir;
        count_next = byte_count;
        if (pop) begin
            reservoir_next = reservoir >> 272;
            count_next = byte_count - 8'd34;
        end
        if (push && state == STREAM) begin
            reservoir_next = reservoir_next | ({512'b0, in_data} << (int'(count_next)*8));
            count_next = count_next + 8'd64;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            reservoir <= 0;
            byte_count <= 0;
            input_beats <= 0;
            group_beats <= 0;
            next_index <= 0;
            last_seen <= 0;
            block_scale <= 0;
            block_weights <= 0;
            block_index <= 0;
            block_tag <= 0;
            block_last <= 0;
            block_valid <= 0;
            done_status <= 0;
            done_blocks <= 0;
            done_tag <= 0;
        end else begin
            if (block_valid && block_ready) begin
                block_valid <= 0;
                done_blocks <= done_blocks + 11'd1;
            end
            case (state)
                IDLE: if (push) begin
                    done_tag <= in_tag;
                    done_status <= 0;
                    done_blocks <= 0;
                    next_index <= 0;
                    input_beats <= 1;
                    group_beats <= 1;
                    last_seen <= in_last;
                    if (in_last) begin
                        done_status <= 1; // 64 bytes cannot contain an integral Q8 packet.
                        reservoir <= 0;
                        byte_count <= 0;
                        state <= FLUSH;
                    end else begin
                        reservoir <= {512'b0, in_data};
                        byte_count <= 64;
                        state <= STREAM;
                    end
                end
                STREAM: begin
                    if (format_error || scale_error) begin
                        done_status <= format_error ? 2'd1 : 2'd2;
                        reservoir <= 0;
                        byte_count <= 0;
                        state <= (last_seen || (push && in_last)) ? FLUSH : DRAIN;
                        // A previously staged block retains VALID/data through backpressure.
                    end else begin
                        reservoir <= reservoir_next;
                        byte_count <= count_next;
                        if (push) begin
                            input_beats <= input_beats + 10'd1;
                            group_beats <= group_beats == 16 ? 5'd0 : group_beats + 5'd1;
                            if (in_last) last_seen <= 1;
                        end
                        if (pop) begin
                            block_scale <= reservoir[15:0];
                            block_weights <= reservoir[271:16];
                            block_index <= next_index;
                            block_tag <= done_tag;
                            block_last <= last_seen && byte_count == 34;
                            block_valid <= 1;
                            next_index <= next_index + 10'd1;
                        end
                        if (last_seen && count_next == 0) state <= FLUSH;
                    end
                end
                DRAIN: if (push && in_last) state <= FLUSH;
                FLUSH: if (!block_valid) state <= COMPLETE;
                COMPLETE: if (done_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
