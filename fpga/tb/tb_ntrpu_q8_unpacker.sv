`timescale 1ns/1ps
module tb_ntrpu_q8_unpacker;
    logic clk = 0;
    always #2.5 clk = !clk;
    logic rst_n = 0;
    logic [511:0] in_data = 0;
    logic in_valid = 0, in_ready, in_last = 0;
    logic [31:0] in_tag = 0;
    logic [15:0] block_scale;
    logic [255:0] block_weights;
    logic [9:0] block_index;
    logic [31:0] block_tag;
    logic block_last, block_valid, block_ready = 0;
    logic done_valid, done_ready = 0;
    logic [1:0] done_status;
    logic [10:0] done_blocks;
    logic [31:0] done_tag;
    ntrpu_q8_unpacker dut (.*);
    int cases = 0, blocks = 0, seed = 1, random_state;
    int input_stalls = 0, output_stalls = 0, done_stalls = 0;
    logic [31:0] alignments = 0;
    int lengths [0:7] = '{17, 34, 51, 68, 136, 272, 527, 544};
    logic [15:0] bad_scales [0:3] = '{16'h7c00, 16'hfc00, 16'h7e01, 16'hfc01};
    int bad_positions [0:3] = '{0, 31, 511, 1023};

    function automatic logic [15:0] scale(input int b);
        case (b % 8)
            0: scale = 16'h0000;
            1: scale = 16'h8000;
            2: scale = 16'h0001;
            3: scale = 16'h83ff;
            4: scale = 16'h3c00;
            5: scale = 16'hbc00;
            6: scale = 16'h7bff;
            7: scale = 16'hfbff;
        endcase
    endfunction
    function automatic int weight(input int b, input int lane, input int tag);
        return (b*29 + lane*17 + (tag & 255)) & 255;
    endfunction
    function automatic logic [511:0] beat_data(input int beat, input int tag,
                                              input int bad_block, input logic [15:0] bad_bits);
        int offset, b, position;
        logic [15:0] s;
        for (int j = 0; j < 64; j++) begin
            offset = beat*64 + j;
            b = offset / 34;
            position = offset % 34;
            s = b == bad_block ? bad_bits : scale(b);
            case (position)
                0: beat_data[j*8 +: 8] = s[7:0];
                1: beat_data[j*8 +: 8] = s[15:8];
                default: beat_data[j*8 +: 8] = 8'(weight(b, position-2, tag));
            endcase
        end
    endfunction
    task automatic require(input bit condition, input string message);
        if (!condition) $fatal(1, "FAIL frame=%0d: %s", cases, message);
    endtask
    task automatic frame(input int beats, input int bad_block = -1,
                         input logic [15:0] bad_bits = 16'h7c00, input int bad_tag_beat = -1);
        int sent, received, cycles, done_wait, expected_status, last_count, signed_value;
        logic [31:0] tag;
        bit finished, input_accepted, was_stalled, was_done_stalled;
        logic [314:0] saved_block;
        logic [44:0] saved_done;
        cases++;
        tag = 32'hfff00000 + 32'(cases);
        sent = 0;
        received = 0;
        cycles = 0;
        done_wait = 0;
        last_count = 0;
        finished = 0;
        input_accepted = 0;
        was_stalled = 0;
        was_done_stalled = 0;
        expected_status = (beats < 17 || beats > 544 || beats % 17 != 0 || bad_tag_beat >= 0) ? 1 :
                          (bad_block >= 0 ? 2 : 0);
        in_valid = 0;
        while (!finished) begin
            @(negedge clk);
            if (input_accepted) in_valid = 0;
            if (!in_valid && sent < beats && $urandom_range(0, 3) != 0) begin
                in_valid = 1;
                in_data = beat_data(sent, int'(tag), bad_block, bad_bits);
                in_tag = sent == bad_tag_beat ? tag ^ 32'h100 : tag;
                in_last = sent == beats-1;
            end
            block_ready = !(cycles >= 12 && cycles < 48) && $urandom_range(0, 3) != 0;
            done_ready = done_wait >= 7 && $urandom_range(0, 2) != 0;
            @(posedge clk);
            cycles++;
            require(cycles < 20000, "timeout");
            if (was_stalled)
                require(block_valid && {block_scale, block_weights, block_index, block_tag, block_last} === saved_block,
                        "block changed while stalled");
            if (was_done_stalled)
                require(done_valid && {done_status, done_blocks, done_tag} === saved_done,
                        "completion changed while stalled");
            was_stalled = block_valid && !block_ready;
            saved_block = {block_scale, block_weights, block_index, block_tag, block_last};
            was_done_stalled = done_valid && !done_ready;
            saved_done = {done_status, done_blocks, done_tag};
            if (in_valid && !in_ready) input_stalls++;
            if (was_stalled) output_stalls++;
            if (was_done_stalled) done_stalls++;
            input_accepted = in_valid && in_ready;
            if (input_accepted) sent++;
            if (block_valid && block_ready) begin
                require(block_tag == tag && block_index == received, "wrong tag/order");
                require(received < 1024 && (bad_block < 0 || received < bad_block), "bad/extra block emitted");
                require(block_scale == scale(received), "FP16 bits/endian mismatch");
                for (int lane = 0; lane < 32; lane++) begin
                    signed_value = weight(received, lane, int'(tag));
                    if (signed_value >= 128) signed_value -= 256;
                    require(int'($signed(block_weights[lane*8 +: 8])) == signed_value, "INT8 lane mismatch");
                end
                if (expected_status == 0)
                    require(block_last == (received+1 == beats*64/34), "wrong block last");
                else require(!block_last, "failed packet published last");
                if (block_last) last_count++;
                alignments[((received*34)%64)/2] = 1;
                received++;
                blocks++;
            end
            if (done_valid) begin
                done_wait++;
                require(sent == beats && !block_valid && !in_ready, "premature completion");
                require(done_tag == tag && done_status == expected_status && done_blocks == received,
                        "wrong completion");
                if (expected_status == 0)
                    require(received == beats*64/34 && last_count == 1, "incomplete valid packet");
                if (done_ready) finished = 1;
            end
            #1;
        end
        in_valid = 0;
        done_ready = 0;
        $display("FRAME %0d PASS beats=%0d status=%0d blocks=%0d", cases, beats, expected_status, received);
    endtask
    initial begin
        if ($value$plusargs("SEED=%d", seed)) begin end
        random_state = $urandom(seed);
        repeat (4) @(negedge clk);
        rst_n = 1;
        for (int i = 0; i < 8; i++) frame(lengths[i]);
        for (int s = 0; s < 4; s++)
            for (int p = 0; p < 4; p++) frame(544, bad_positions[p], bad_scales[s]);
        frame(1);
        frame(16);
        frame(18);
        frame(543);
        frame(545);
        frame(600);
        frame(544, -1, 0, 1);
        frame(544, -1, 0, 16);
        frame(544, -1, 0, 543);
        frame(17);
        @(negedge clk);
        rst_n = 0;
        repeat (4) @(negedge clk);
        require(!block_valid && !done_valid, "reset retained output");
        rst_n = 1;
        frame(544);
        require(alignments == 32'hffffffff && input_stalls > 0 && output_stalls > 0 && done_stalls > 0,
                "missing alignment/backpressure coverage");
        $display("COVERAGE alignments=%h in_stalls=%0d out_stalls=%0d done_stalls=%0d",
                 alignments, input_stalls, output_stalls, done_stalls);
        $display("NTRPU_Q8_UNPACKER_PASS cases=%0d bursts=0 blocks=%0d", cases, blocks);
        $finish;
    end
endmodule
