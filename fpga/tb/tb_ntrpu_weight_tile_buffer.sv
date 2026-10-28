`timescale 1ns/1ps
module tb_ntrpu_weight_tile_buffer;
    logic clk = 0;
    always #2.5 clk = !clk;
    logic rst_n = 0;
    logic load_valid = 0, load_ready, load_done_valid, load_done_ready = 0;
    logic [63:0] load_addr = 0;
    logic [31:0] load_bytes = 0, load_tag = 0, load_done_bytes, load_done_tag;
    logic [1:0] load_done_status;
    logic [511:0] tile_data;
    logic tile_valid, tile_ready = 0, tile_last;
    logic [31:0] tile_tag;
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
    ntrpu_weight_tile_buffer dut (.*);

    int consumer_mode = 0, completion_mode = 1; // 0=stop, 1=ready, 2=random.
    int cases = 0, cycles = 0, bursts = 0, overlap_cycles = 0, full_stalls = 0;
    int completed_tiles = 0, rejected_loads = 0, stalled_outputs = 0, stalled_completions = 0;
    int seed = 1, random_state;
    bit accepted [0:255], done_seen [0:255], done_accepted [0:255];
    logic [63:0] addresses [0:255];
    int unsigned sizes [0:255], status_expected [0:255], bytes_expected [0:255];
    int fault_mode [0:255], fault_index [0:255], output_bytes [0:255];
    int queue [0:255];
    int queue_read = 0, queue_write = 0, current_tag = 0;
    bit memory_active = 0, r_accepted = 0;
    logic [63:0] burst_addr;
    int nominal_beats, actual_beats, beat, burst_first, returned = 0, issued = 0;
    bit ar_stalled = 0, out_stalled = 0, done_stalled = 0;
    logic [63:0] saved_araddr;
    logic [7:0] saved_arlen;
    logic [511:0] saved_data;
    logic saved_last;
    logic [31:0] saved_tag, saved_done_tag, saved_done_bytes;
    logic [1:0] saved_status;

    function automatic logic [511:0] pattern(input logic [63:0] addr);
        for (int j = 0; j < 8; j++)
            pattern[j*64 +: 64] = (addr + 64'(j*8)) ^ 64'h6352a1e8f709dc4b;
    endfunction
    task automatic require(input bit condition, input string message);
        if (!condition) $fatal(1, "FAIL cycle=%0d tag=%0d: %s", cycles, current_tag, message);
    endtask

    task automatic step;
        int expected_beats;
        bit command_handshake;
        @(negedge clk);
        m_axi_arready = $urandom_range(0, 3) != 0;
        tile_ready = consumer_mode == 1 || (consumer_mode == 2 && $urandom_range(0, 3) != 0);
        load_done_ready = completion_mode == 1 || (completion_mode == 2 && $urandom_range(0, 2) != 0);
        if (r_accepted) m_axi_rvalid = 0;
        if (!m_axi_rvalid && memory_active && $urandom_range(0, 3) != 0) begin
            m_axi_rvalid = 1;
            m_axi_rdata = pattern(burst_addr + 64'(beat*64));
            m_axi_rresp = 0;
            m_axi_rid = 0;
            m_axi_rlast = beat == actual_beats - 1;
            if (burst_first + beat == fault_index[current_tag]) begin
                if (fault_mode[current_tag] == 1) m_axi_rresp = 2;
                if (fault_mode[current_tag] == 2) m_axi_rresp = 3;
                if (fault_mode[current_tag] == 3) m_axi_rid = 1;
            end
        end
        @(posedge clk);
        cycles++;
        require(cycles < 500000, "timeout");
        if (ar_stalled)
            require(m_axi_arvalid && saved_araddr == m_axi_araddr && saved_arlen == m_axi_arlen,
                    "AR changed under backpressure");
        if (out_stalled)
            require(tile_valid && tile_data === saved_data && tile_last == saved_last && tile_tag == saved_tag,
                    "tile payload/tag changed under backpressure");
        if (done_stalled)
            require(load_done_valid && load_done_tag == saved_done_tag && load_done_bytes == saved_done_bytes &&
                    load_done_status == saved_status, "completion changed under backpressure");
        ar_stalled = m_axi_arvalid && !m_axi_arready;
        saved_araddr = m_axi_araddr;
        saved_arlen = m_axi_arlen;
        out_stalled = tile_valid && !tile_ready;
        saved_data = tile_data;
        saved_last = tile_last;
        saved_tag = tile_tag;
        done_stalled = load_done_valid && !load_done_ready;
        saved_done_tag = load_done_tag;
        saved_done_bytes = load_done_bytes;
        saved_status = load_done_status;
        if (out_stalled) stalled_outputs++;
        if (done_stalled) stalled_completions++;
        if (memory_active && tile_valid && tile_ready) overlap_cycles++;
        command_handshake = load_valid && load_ready;
        if (command_handshake) begin
            require(load_tag < 256 && !accepted[load_tag], "duplicate/invalid load");
            current_tag = int'(load_tag);
            accepted[current_tag] = 1;
            issued = 0;
            returned = 0;
        end
        if (m_axi_arvalid && m_axi_arready) begin
            require(accepted[current_tag] && !memory_active && status_expected[current_tag] != 1,
                    "unexpected AR");
            require(fault_mode[current_tag] == 0 || returned <= fault_index[current_tag], "AR after failed burst");
            require(m_axi_araddr == addresses[current_tag] + 64'(issued*64), "wrong address sequence");
            expected_beats = int'(sizes[current_tag]/64) - issued;
            if (expected_beats > (4096 - int'(m_axi_araddr[11:0]))/64)
                expected_beats = (4096 - int'(m_axi_araddr[11:0]))/64;
            require(m_axi_arlen + 1 == expected_beats && m_axi_arsize == 6 && m_axi_arburst == 1,
                    "wrong burst geometry");
            burst_addr = m_axi_araddr;
            nominal_beats = int'(m_axi_arlen) + 1;
            actual_beats = nominal_beats;
            burst_first = issued;
            if (fault_index[current_tag] >= issued && fault_index[current_tag] < issued + nominal_beats) begin
                if (fault_mode[current_tag] == 4) actual_beats = fault_index[current_tag] - issued + 1;
                if (fault_mode[current_tag] == 5) begin
                    require(fault_index[current_tag] == issued + nominal_beats - 1, "bad late-RLAST stimulus");
                    actual_beats += 2;
                end
            end
            issued += nominal_beats;
            beat = 0;
            memory_active = 1;
            bursts++;
        end
        r_accepted = m_axi_rvalid && m_axi_rready;
        if (r_accepted) begin
            require(memory_active, "unexpected R acceptance");
            beat++;
            returned++;
            if (beat == actual_beats) memory_active = 0;
        end
        if (load_done_valid) begin
            require(load_done_tag == current_tag && !memory_active, "wrong/premature completion");
            require(load_done_status == status_expected[current_tag] && load_done_bytes == bytes_expected[current_tag],
                    "wrong completion status or delivered count");
            if (!done_seen[current_tag]) begin
                done_seen[current_tag] = 1;
                if (load_done_status == 0) begin
                    queue[queue_write] = current_tag;
                    queue_write++;
                end else rejected_loads++;
            end
            if (load_done_ready) done_accepted[current_tag] = 1;
        end
        if (tile_valid) begin
            require(queue_read < queue_write, "uncommitted/failed tile visible");
            require(tile_tag == queue[queue_read] && done_seen[tile_tag] && status_expected[tile_tag] == 0,
                    "tile order/tag/commit mismatch");
        end
        if (tile_valid && tile_ready) begin
            require(tile_data === pattern(addresses[tile_tag] + 64'(output_bytes[tile_tag])), "RAM data mismatch");
            require(tile_last == (output_bytes[tile_tag] + 64 == sizes[tile_tag]), "wrong tile last");
            output_bytes[tile_tag] += 64;
            if (tile_last) begin
                queue_read++;
                completed_tiles++;
            end
        end
        #1;
        if (command_handshake) load_valid = 0;
    endtask

    task automatic prepare(input logic [63:0] addr, input int unsigned size,
                           input int fault = 0, input int at_beat = 0, input bit invalid = 0);
        require(!load_valid && cases < 256, "overwritten load request");
        addresses[cases] = addr;
        sizes[cases] = size;
        fault_mode[cases] = fault;
        fault_index[cases] = at_beat;
        status_expected[cases] = invalid ? 1 : (fault == 0 ? 0 : (fault <= 2 ? 2 : 3));
        bytes_expected[cases] = invalid ? 0 : (fault == 0 ? size : at_beat*64);
        load_tag = cases;
        load_addr = addr;
        load_bytes = size;
        load_valid = 1;
        cases++;
    endtask
    task automatic finish_load;
        int tag;
        tag = int'(load_tag);
        while (!done_accepted[tag]) step();
    endtask
    task automatic send(input logic [63:0] addr, input int unsigned size,
                        input int fault = 0, input int at_beat = 0, input bit invalid = 0);
        prepare(addr, size, fault, at_beat, invalid);
        finish_load();
    endtask
    task automatic drain;
        while (queue_read < queue_write || tile_valid) step();
        repeat (8) step();
    endtask
    task automatic reset_domain;
        require(!memory_active && !load_valid, "test only resets a quiescent memory fabric");
        @(negedge clk);
        rst_n = 0;
        m_axi_rvalid = 0;
        m_axi_arready = 0;
        tile_ready = 0;
        load_done_ready = 0;
        repeat (4) @(negedge clk);
        require(!tile_valid && !load_done_valid && !m_axi_arvalid, "reset did not hide RAM");
        ar_stalled = 0;
        out_stalled = 0;
        done_stalled = 0;
        r_accepted = 0;
        queue_read = 0;
        queue_write = 0;
        rst_n = 1;
        // Return after a rising edge, as step() does, so no unobserved command
        // handshake can occur between the caller's prepare() and the next step.
        @(posedge clk);
        #1;
    endtask

    initial begin
        if ($value$plusargs("SEED=%d", seed)) begin end
        random_state = $urandom(seed);
        for (int i = 0; i < 256; i++) begin
            accepted[i] = 0;
            done_seen[i] = 0;
            done_accepted[i] = 0;
            output_bytes[i] = 0;
        end
        reset_domain();
        // Fill both banks and hold a third command: neither occupied bank may be overwritten.
        send(64'h100000040, 34816);
        send(64'h200000000, 34816);
        prepare(64'h7fffffffc0, 34816);
        repeat (32) begin
            step();
            require(!accepted[2], "third tile accepted while both banks occupied");
            full_stalls++;
        end
        consumer_mode = 2;
        completion_mode = 2;
        finish_load();
        drain();
        // Late errors must not publish even one beat; retry must reuse the failed bank safely.
        for (int f = 1; f <= 5; f++) begin
            send(64'h300000000, 34816, f, f == 4 ? 540 : 543);
            repeat (8) step();
            send(64'h310000000 + 64'(f*4096), 128);
            drain();
        end
        send(64'h400000000, 4096, 1, 0);
        send(64'h400000000, 4096);
        drain();
        send(64'h400000000, 4096, 2, 32);
        send(64'h400000000, 4096);
        drain();
        // Failure/retry in one bank cannot change the other bank's stalled output.
        consumer_mode = 0;
        send(64'h500000000, 4096);
        send(64'h510000000, 34816, 1, 543);
        send(64'h520000000, 34816);
        repeat (24) step();
        consumer_mode = 2;
        drain();
        send(0, 0, 0, 0, 1);
        send(0, 63, 0, 0, 1);
        send(0, 65, 0, 0, 1);
        send(0, 34880, 0, 0, 1);
        send(1, 64, 0, 0, 1);
        send(64'hffffffffffffffc0, 128, 0, 0, 1);
        drain();
        // Metadata reset invalidates two committed banks without clearing the RAM arrays.
        consumer_mode = 0;
        send(64'h600000000, 34816);
        send(64'h610000000, 34816);
        reset_domain();
        consumer_mode = 2;
        send(64'h620000000, 64);
        drain();
        for (int i = 0; i < 20; i++)
            send(64'h700000000 + 64'(i*65536 + $urandom_range(0, 63)*64),
                 $urandom_range(1, 544)*64);
        drain();
        // Successful data is published when the reader completes, even if notification is stalled.
        completion_mode = 0;
        consumer_mode = 1;
        prepare(64'h800000000, 128);
        while (!done_seen[cases-1]) step();
        repeat (24) step();
        require(output_bytes[cases-1] == 128 && !done_accepted[cases-1] && !load_ready,
                "completion backpressure broke publication or command serialization");
        completion_mode = 1;
        finish_load();
        drain();
        require(overlap_cycles > 0 && full_stalls == 32 && stalled_outputs > 0 && stalled_completions > 0,
                "missing overlap/backpressure coverage");
        require(cases == 50 && rejected_loads == 14 && completed_tiles == 34, "missing case coverage");
        $display("COVERAGE overlap=%0d full=%0d output_stalls=%0d completion_stalls=%0d rejected=%0d tiles=%0d",
                 overlap_cycles, full_stalls, stalled_outputs, stalled_completions, rejected_loads, completed_tiles);
        $display("NTRPU_TILE_BUFFER_PASS cases=%0d bursts=%0d tiles=%0d", cases, bursts, completed_tiles);
        $finish;
    end
endmodule
