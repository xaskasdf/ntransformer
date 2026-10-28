`timescale 1ns/1ps
module tb_ntrpu_axi_read_master;
    logic clk = 0;
    always #2.5 clk = !clk;
    logic rst_n = 0;
    logic cmd_valid = 0, cmd_ready, done_valid, done_ready = 0;
    logic [63:0] cmd_addr = 0;
    logic [31:0] cmd_bytes = 0, done_bytes;
    logic [1:0] done_status;
    logic [511:0] out_data;
    logic out_valid, out_ready = 0, out_last;
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
    ntrpu_axi_read_master dut (.*);

    int cases = 0;
    int total_bursts = 0;
    longint total_bytes = 0;
    int ar_stalls = 0, output_stalls = 0, r_stalls = 0, done_stalls = 0;
    int seed = 1;
    int random_state;
    function automatic logic [511:0] pattern(input logic [63:0] addr);
        for (int j = 0; j < 8; j++)
            pattern[j*64 +: 64] = (addr + 64'(j*8)) ^ 64'hf17ea50d6b3294c8;
    endfunction
    task automatic require(input bit condition, input string message);
        if (!condition) $fatal(1, "FAIL case=%0d: %s", cases, message);
    endtask

    // fault: 0=none, 1=SLVERR, 2=DECERR, 3=RID, 4=early RLAST, 5=late RLAST.
    task automatic run_case(input logic [63:0] base, input int unsigned size,
                            input int fault, input int fault_at, input bit invalid,
                            input int fault_burst = 1);
        bit command_sent, memory_active, complete;
        bit ar_stalled, output_stalled, completion_stalled;
        logic [63:0] next_addr, burst_addr, saved_araddr;
        logic [7:0] saved_arlen;
        logic [511:0] saved_data;
        bit saved_last;
        logic [1:0] saved_status;
        logic [31:0] saved_bytes;
        int unsigned remaining, received, returned, ar_count, nominal_beats;
        int unsigned beat, actual_beats, expected_beats, done_wait, cycles;
        int unsigned expected_bytes, expected_status, prior_bytes;
        command_sent = 0;
        memory_active = 0;
        complete = 0;
        ar_stalled = 0;
        output_stalled = 0;
        completion_stalled = 0;
        next_addr = base;
        remaining = size;
        received = 0;
        returned = 0;
        ar_count = 0;
        beat = 0;
        nominal_beats = 0;
        done_wait = 0;
        cycles = 0;
        expected_status = invalid ? 1 : ((fault == 1 || fault == 2) ? 2 : (fault != 0 ? 3 : 0));
        prior_bytes = fault_burst == 1 ? 0 : 4096 - int'(base[11:0]) + (fault_burst - 2)*4096;
        expected_bytes = invalid ? 0 : (fault != 0 ? prior_bytes + fault_at * 64 : size);
        cases++;
        @(negedge clk);
        cmd_addr = base;
        cmd_bytes = size;
        cmd_valid = 1;
        m_axi_rvalid = 0;
        while (!complete) begin
            // Drive only on falling edges; inspect handshakes before rising-edge NBA updates.
            m_axi_arready = cycles >= 6 && ($urandom_range(0, 3) != 0);
            out_ready = !(cycles >= 8 && cycles < 35) && ($urandom_range(0, 3) != 0);
            done_ready = done_wait >= 7 && ($urandom_range(0, 2) != 0);
            if (!m_axi_rvalid && memory_active && $urandom_range(0, 3) != 0) begin
                m_axi_rvalid = 1;
                m_axi_rdata = pattern(burst_addr + 64'(beat*64));
                m_axi_rresp = 0;
                m_axi_rid = 0;
                m_axi_rlast = beat == actual_beats - 1;
                if (ar_count == fault_burst && beat == fault_at) begin
                    if (fault == 1) m_axi_rresp = 2'b10;
                    if (fault == 2) m_axi_rresp = 2'b11;
                    if (fault == 3) m_axi_rid = 4'd1;
                end
            end
            @(posedge clk);
            cycles++;
            require(cycles < 500000, "timeout");
            if (command_sent && !cmd_valid) require(!cmd_ready, "accepted command while busy");
            if (ar_stalled)
                require(m_axi_arvalid && m_axi_araddr == saved_araddr &&
                        m_axi_arlen == saved_arlen, "AR changed while stalled");
            if (output_stalled)
                require(out_valid && out_data == saved_data && out_last == saved_last,
                        "output changed while stalled");
            if (completion_stalled)
                require(done_valid && done_status == saved_status && done_bytes == saved_bytes,
                        "completion changed while stalled");
            ar_stalled = m_axi_arvalid && !m_axi_arready;
            saved_araddr = m_axi_araddr;
            saved_arlen = m_axi_arlen;
            output_stalled = out_valid && !out_ready;
            saved_data = out_data;
            saved_last = out_last;
            completion_stalled = done_valid && !done_ready;
            saved_status = done_status;
            saved_bytes = done_bytes;
            if (ar_stalled) ar_stalls++;
            if (output_stalled) output_stalls++;
            if (m_axi_rvalid && !m_axi_rready) r_stalls++;
            if (completion_stalled) done_stalls++;
            if (cmd_valid && cmd_ready) begin
                require(!command_sent, "duplicate command");
                command_sent = 1;
            end
            if (m_axi_arvalid && m_axi_arready) begin
                require(command_sent && !invalid && !memory_active, "unexpected/outstanding AR");
                require(fault == 0 || ar_count < fault_burst, "new burst issued after error");
                expected_beats = remaining / 64;
                if (expected_beats > (4096 - int'(next_addr[11:0])) / 64)
                    expected_beats = (4096 - int'(next_addr[11:0])) / 64;
                require(m_axi_araddr == next_addr && m_axi_arlen + 1 == expected_beats,
                        "wrong burst address or length");
                require(m_axi_arsize == 6 && m_axi_arburst == 1 && m_axi_arid == 0 &&
                        m_axi_arlock == 0 && m_axi_arcache == 2 && m_axi_arprot == 0,
                        "wrong AXI attributes");
                require(int'(m_axi_araddr[11:0]) + (int'(m_axi_arlen) + 1)*64 <= 4096,
                        "burst crosses 4 KiB");
                nominal_beats = int'(m_axi_arlen) + 1;
                actual_beats = nominal_beats;
                if (fault == 4 && ar_count + 1 == fault_burst) actual_beats = fault_at + 1;
                if (fault == 5 && ar_count + 1 == fault_burst) actual_beats = nominal_beats + 2;
                burst_addr = m_axi_araddr;
                next_addr += 64'(nominal_beats*64);
                remaining -= nominal_beats*64;
                beat = 0;
                memory_active = 1;
                ar_count++;
            end
            if (out_valid && out_ready) begin
                require(received < expected_bytes, "unexpected output after error/end");
                require(out_data === pattern(base + 64'(received)), "payload/address mismatch");
                require(out_last == (fault == 0 && received + 64 == size), "wrong command last");
                received += 64;
            end
            if (m_axi_rvalid && m_axi_rready) begin
                require(memory_active, "R accepted without AR");
                returned++;
                beat++;
                if (beat == actual_beats) memory_active = 0;
            end
            if (done_valid) begin
                done_wait++;
                require(command_sent && !memory_active && !out_valid, "premature completion");
                require(done_status == expected_status, "wrong completion status");
                require(done_bytes == expected_bytes && received == expected_bytes,
                        "wrong delivered byte count");
                if (done_ready) complete = 1;
            end
            // Snapshot R handshake before changing anything at the next falling edge.
            begin
                bit accepted;
                accepted = m_axi_rvalid && m_axi_rready;
                @(negedge clk);
                if (accepted) m_axi_rvalid = 0;
            end
            if (command_sent) cmd_valid = 0;
        end
        if (invalid) require(ar_count == 0, "invalid command reached memory");
        else if (fault != 0) require(ar_count == fault_burst && returned == prior_bytes/64 + actual_beats,
                                    "failed to drain faulted burst");
        total_bursts += ar_count;
        total_bytes += received;
        done_ready = 0;
        m_axi_arready = 0;
        out_ready = 0;
        $display("CASE %0d PASS addr=%h bytes=%0d fault=%0d bursts=%0d delivered=%0d cycles=%0d",
                 cases, base, size, fault, ar_count, received, cycles);
    endtask

    initial begin
        if ($value$plusargs("SEED=%d", seed)) begin end
        random_state = $urandom(seed);
        repeat (4) @(negedge clk);
        rst_n = 1;
        run_case(0, 64, 0, 0, 0);
        run_case(64'hfc0, 128, 0, 0, 0);
        run_case(64'h1000, 4096, 0, 0, 0);
        run_case(64'h100000040, 34816, 0, 0, 0); // 32 x 1024 Q8_0 tile: 34 KiB.
        run_case(64'h200000000, 69632, 0, 0, 0); // 32 x 2048 row tile.
        run_case(64'h7ffffff000, 278528, 0, 0, 0); // 32 x 8192, crosses 512 GiB.
        run_case(64'hffffffffffffffc0, 64, 0, 0, 0);
        run_case(0, 0, 0, 0, 1);
        run_case(1, 64, 0, 0, 1);
        run_case(0, 63, 0, 0, 1);
        run_case(0, 65, 0, 0, 1);
        run_case(64'hffffffffffffffc0, 128, 0, 0, 1);
        for (int f = 1; f <= 3; f++) begin
            run_case(0, 8192, f, 0, 0);
            run_case(0, 8192, f, 7, 0);
            run_case(0, 8192, f, 63, 0);
        end
        run_case(0, 8192, 4, 0, 0);
        run_case(0, 8192, 4, 12, 0);
        run_case(0, 8192, 5, 63, 0);
        run_case(64'hfc0, 12288, 1, 12, 0, 3);
        run_case(64'h100000000, 12288, 4, 11, 0, 2);
        run_case(64'h100000000, 16384, 5, 63, 0, 3);
        for (int i = 0; i < 24; i++)
            run_case(64'h300000000 + 64'($urandom_range(0, 511)*64),
                     $urandom_range(1, 320)*64, 0, 0, 0);
        @(negedge clk);
        rst_n = 0;
        repeat (3) @(negedge clk);
        require(!done_valid && !out_valid && !m_axi_arvalid, "reset left traffic");
        rst_n = 1;
        run_case(64'h400000000, 128, 0, 0, 0);
        require(ar_stalls > 0 && output_stalls > 0 && r_stalls > 0 && done_stalls > 0,
                "missing backpressure coverage");
        $display("STALL_CYCLES ar=%0d output=%0d r=%0d done=%0d", ar_stalls, output_stalls, r_stalls, done_stalls);
        $display("NTRPU_AXI_READER_PASS cases=%0d bursts=%0d bytes=%0d", cases, total_bursts, total_bytes);
        $finish;
    end
endmodule
