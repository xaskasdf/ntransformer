`timescale 1ns/1ps
module tb_ntrpu_q8_weight_stream;
    logic clk = 0;
    always #2.5 clk = !clk;
    logic rst_n = 0;
    logic load_valid = 0, load_ready, load_done_valid, load_done_ready = 0;
    logic [63:0] load_addr = 0;
    logic [31:0] load_bytes = 0, load_tag = 0, load_done_bytes, load_done_tag;
    logic [1:0] load_done_status;
    logic [15:0] block_scale;
    logic [255:0] block_weights;
    logic [9:0] block_index;
    logic [31:0] block_tag;
    logic block_last, block_valid, block_ready = 0;
    logic decode_done_valid, decode_done_ready = 0;
    logic [1:0] decode_done_status;
    logic [10:0] decode_done_blocks;
    logic [31:0] decode_done_tag;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize, m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid, m_axi_arcache;
    logic m_axi_arlock, m_axi_arvalid, m_axi_arready = 0;
    logic [511:0] m_axi_rdata = 0;
    logic [1:0] m_axi_rresp = 0;
    logic [3:0] m_axi_rid = 0;
    logic m_axi_rlast = 0, m_axi_rvalid = 0, m_axi_rready;
    ntrpu_q8_weight_stream dut (.*);

    int sizes [0:9] = '{34816,34816,1088,34816,34816,64,1088,34880,2176,34816};
    int bad_block [0:9] = '{-1,-1,-1,1023,-1,-1,0,-1,-1,-1};
    int load_status [0:9] = '{0,0,0,0,2,0,0,1,0,0};
    int decode_status [0:9] = '{0,0,0,2,-1,1,2,-1,0,0};
    bit accepted [0:9], load_seen [0:9], load_accepted [0:9];
    int received [0:9];
    int queue [0:9], queue_read = 0, queue_write = 0;
    int current_tag = 0, issued = 0, returned = 0;
    int burst_first = 0, burst_size = 0, beat = 0;
    bit memory_active = 0, r_accepted = 0, consumer_enabled = 0;
    int cycles = 0, bursts = 0, blocks = 0, decoded = 0, last_count = 0;
    int overlap_cycles = 0, block_stalls = 0, decode_stalls = 0, done_wait = 0;
    int seed = 1, random_state;
    bit was_stalled = 0, was_done_stalled = 0, ar_stalled = 0, load_stalled = 0;
    logic [314:0] saved_block;
    logic [44:0] saved_done;
    logic [65:0] saved_load;
    logic [71:0] saved_ar;

    function automatic logic [63:0] address(input int tag);
        return 64'h100000000 + (64'(tag) << 28) + (tag == 0 ? 64'hfc0 : 64'h40);
    endfunction
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
        return (b*29 + lane*17 + tag) & 255;
    endfunction
    // Construct external memory one byte at a time, independently of the RTL reservoir.
    function automatic logic [511:0] memory_word(input int word_index, input int tag);
        int offset, b, position;
        logic [15:0] s;
        for (int j = 0; j < 64; j++) begin
            offset = word_index*64 + j;
            b = offset/34;
            position = offset%34;
            s = b == bad_block[tag] ? 16'h7e01 : scale(b);
            if (position == 0) memory_word[j*8 +: 8] = s[7:0];
            else if (position == 1) memory_word[j*8 +: 8] = s[15:8];
            else memory_word[j*8 +: 8] = 8'(weight(b, position-2, tag));
        end
    endfunction
    task automatic require(input bit condition, input string message);
        if (!condition) $fatal(1, "FAIL cycle=%0d load=%0d: %s", cycles, current_tag, message);
    endtask

    task automatic step;
        int expected_beats, value, expected_blocks;
        bit load_handshake;
        @(negedge clk);
        m_axi_arready = $urandom_range(0,3) != 0;
        block_ready = consumer_enabled && $urandom_range(0,3) != 0;
        load_done_ready = $urandom_range(0,2) != 0;
        decode_done_ready = done_wait >= 7 && $urandom_range(0,2) != 0;
        if (r_accepted) m_axi_rvalid = 0;
        if (!m_axi_rvalid && memory_active && $urandom_range(0,3) != 0) begin
            m_axi_rvalid = 1;
            m_axi_rdata = memory_word(burst_first + beat, current_tag);
            m_axi_rresp = current_tag == 4 && burst_first + beat == 543 ? 2 : 0;
            m_axi_rlast = beat == burst_size-1;
        end
        @(posedge clk);
        cycles++;
        require(cycles < 50000, "timeout");
        if (was_stalled)
            require(block_valid && {block_scale,block_weights,block_index,block_tag,block_last} === saved_block,
                    "block changed while stalled");
        if (was_done_stalled)
            require(decode_done_valid && {decode_done_status,decode_done_blocks,decode_done_tag} === saved_done,
                    "decode completion changed while stalled");
        if (ar_stalled) require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen} === saved_ar, "AR stability");
        if (load_stalled)
            require(load_done_valid && {load_done_status,load_done_bytes,load_done_tag} === saved_load, "load completion stability");
        was_stalled = block_valid && !block_ready;
        saved_block = {block_scale,block_weights,block_index,block_tag,block_last};
        was_done_stalled = decode_done_valid && !decode_done_ready;
        saved_done = {decode_done_status,decode_done_blocks,decode_done_tag};
        ar_stalled = m_axi_arvalid && !m_axi_arready;
        saved_ar = {m_axi_araddr,m_axi_arlen};
        load_stalled = load_done_valid && !load_done_ready;
        saved_load = {load_done_status,load_done_bytes,load_done_tag};
        if (was_stalled) block_stalls++;
        if (was_done_stalled) decode_stalls++;
        if (memory_active && block_valid && block_ready) overlap_cycles++;
        load_handshake = load_valid && load_ready;
        if (load_handshake) begin
            require(load_tag < 10 && !accepted[load_tag], "duplicate load");
            current_tag = int'(load_tag);
            accepted[current_tag] = 1;
            issued = 0;
            returned = 0;
        end
        if (m_axi_arvalid && m_axi_arready) begin
            require(accepted[current_tag] && !memory_active && load_status[current_tag] != 1, "unexpected AR");
            require(m_axi_araddr == address(current_tag) + 64'(issued*64), "AXI address mismatch");
            expected_beats = sizes[current_tag]/64 - issued;
            if (expected_beats > (4096-int'(m_axi_araddr[11:0]))/64)
                expected_beats = (4096-int'(m_axi_araddr[11:0]))/64;
            require(m_axi_arlen+1 == expected_beats && m_axi_arsize == 6 && m_axi_arburst == 1, "AXI geometry");
            burst_first = issued;
            burst_size = int'(m_axi_arlen)+1;
            issued += burst_size;
            beat = 0;
            memory_active = 1;
            bursts++;
        end
        r_accepted = m_axi_rvalid && m_axi_rready;
        if (r_accepted) begin
            require(memory_active, "R without AR");
            returned++;
            beat++;
            if (beat == burst_size) memory_active = 0;
        end
        if (load_done_valid) begin
            require(load_done_tag == current_tag && !memory_active, "load tag/completion order");
            require(load_done_status == load_status[current_tag], "load status mismatch");
            require(load_done_bytes == (current_tag == 4 ? 543*64 : (current_tag == 7 ? 0 : sizes[current_tag])),
                    "load byte count mismatch");
            if (!load_seen[current_tag]) begin
                load_seen[current_tag] = 1;
                if (load_done_status == 0) begin
                    queue[queue_write] = current_tag;
                    queue_write++;
                end
            end
            if (load_done_ready) load_accepted[current_tag] = 1;
        end
        if (block_valid) begin
            require(queue_read < queue_write && block_tag == queue[queue_read], "block escaped transport commit/order");
        end
        if (block_valid && block_ready) begin
            require(block_index == received[block_tag] && block_scale == scale(received[block_tag]), "block index/scale mismatch");
            require(bad_block[block_tag] < 0 || received[block_tag] < bad_block[block_tag], "invalid scale escaped decoder");
            for (int lane = 0; lane < 32; lane++) begin
                value = weight(received[block_tag],lane,int'(block_tag));
                if (value >= 128) value -= 256;
                require(int'($signed(block_weights[lane*8 +: 8])) == value, "INT8 lane mismatch");
            end
            require(block_last == (decode_status[block_tag] == 0 && received[block_tag]+1 == sizes[block_tag]/34),
                    "wrong block last");
            if (block_last) last_count++;
            received[block_tag]++;
            blocks++;
        end
        if (decode_done_valid) begin
            done_wait++;
            require(queue_read < queue_write && decode_done_tag == queue[queue_read] && !block_valid, "decode completion order");
            expected_blocks = decode_done_tag == 3 ? 1023 :
                              (decode_status[decode_done_tag] == 0 ? sizes[decode_done_tag]/34 : 0);
            require(decode_done_status == decode_status[decode_done_tag] && decode_done_blocks == expected_blocks &&
                    received[decode_done_tag] == expected_blocks, "decode result mismatch");
            if (decode_done_ready) begin
                queue_read++;
                decoded++;
                done_wait = 0;
            end
        end
        #1;
        if (load_handshake) load_valid = 0;
    endtask
    task automatic prepare(input int tag);
        require(!load_valid, "overwritten load");
        load_addr = address(tag);
        load_bytes = sizes[tag];
        load_tag = tag;
        load_valid = 1;
    endtask
    task automatic send(input int tag);
        prepare(tag);
        while (!load_accepted[tag]) step();
    endtask
    initial begin
        if ($value$plusargs("SEED=%d",seed)) begin end
        random_state = $urandom(seed);
        for (int i = 0; i < 10; i++) begin
            accepted[i] = 0;
            load_seen[i] = 0;
            load_accepted[i] = 0;
            received[i] = 0;
        end
        repeat (4) @(negedge clk);
        rst_n = 1;
        @(posedge clk);
        #1;
        send(0);
        send(1);
        prepare(2);
        repeat (32) begin
            step();
            require(!accepted[2], "banks overwritten during decoder backpressure");
        end
        consumer_enabled = 1;
        while (!load_accepted[2]) step();
        for (int tag = 3; tag < 10; tag++) send(tag);
        while (decoded != 8) step();
        repeat (16) step();
        require(blocks == 4191 && last_count == 5 && overlap_cycles > 0 && block_stalls > 0 && decode_stalls > 0,
                "missing integration coverage");
        $display("COVERAGE decoded=%0d last=%0d overlap=%0d block_stalls=%0d done_stalls=%0d",
                 decoded,last_count,overlap_cycles,block_stalls,decode_stalls);
        $display("NTRPU_Q8_STREAM_PASS cases=10 bursts=%0d blocks=%0d",bursts,blocks);
        $finish;
    end
endmodule
