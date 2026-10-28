`timescale 1ns / 1ps

module tb_ntrpu_peer_rx_guard;
    parameter integer MAX_BEATS = 4;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, stop = 0, source_stopped = 0, ingress_isolated = 0;
    reg [63:0] quiesce_epoch = 0, rearm_epoch = 0;
    reg rearm_valid = 0;
    wire rearm_ready, quarantined, lease_held, axi_busy;
    wire [63:0] epoch;
    reg cmd_valid = 0, data_valid = 0, data_last = 0;
    wire cmd_ready, data_ready;
    reg [63:0] cmd_epoch = 0, cmd_tag = 0, cmd_addr = 0;
    reg [7:0] cmd_len = 0;
    reg [511:0] data = 0;
    wire cpl_valid;
    reg cpl_ready = 0;
    wire [63:0] cpl_epoch, cpl_tag;
    wire [2:0] cpl_status;
    wire [63:0] m_axi_awaddr, m_axi_wstrb;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize, m_axi_awprot;
    wire [1:0] m_axi_awburst;
    wire [3:0] m_axi_awcache, m_axi_awid;
    wire m_axi_awlock, m_axi_awvalid, m_axi_wlast, m_axi_wvalid, m_axi_bready;
    wire [511:0] m_axi_wdata;
    reg m_axi_awready = 0, m_axi_wready = 0, m_axi_bvalid = 0;
    reg [1:0] m_axi_bresp = 0;
    reg [3:0] m_axi_bid = 0;

    ntrpu_peer_rx_guard #(.MAX_BEATS(MAX_BEATS)) dut (.*);

    integer cycles = 0, cases = 0, aw_count = 0, w_count = 0, b_count = 0;
    integer expected_beats = 0, seen_w = 0, fill_index = 0;
    reg [63:0] expected_addr = 0, expected_tag = 0, expected_epoch = 0;
    reg seen_aw = 0, bus_owed = 0, printed_cpl = 0;
    reg previous_busy = 0, previous_stop = 0, aw_stalled = 0, w_stalled = 0;
    reg cpl_stalled = 0, foreign_seen = 0;
    reg [87:0] held_aw;
    reg [576:0] held_w;
    reg [130:0] held_cpl;
    wire [87:0] aw_bundle = {m_axi_awaddr, m_axi_awlen, m_axi_awsize,
                            m_axi_awburst, m_axi_awcache, m_axi_awid,
                            m_axi_awprot};
    wire [576:0] w_bundle = {m_axi_wdata, m_axi_wstrb, m_axi_wlast};
    wire [130:0] cpl_bundle = {cpl_epoch, cpl_tag, cpl_status};
    reg [31:0] rng = 1;

    task need(input bit condition, input string message);
        if (!condition) $fatal(1, "FAIL cycle=%0d tag=%0d %s", cycles, expected_tag, message);
    endtask

    function automatic [511:0] pattern(input [63:0] tag, input integer beat);
        integer byte_index;
        begin
            for (byte_index = 0; byte_index < 64; byte_index = byte_index + 1)
                pattern[byte_index*8 +: 8] = ((tag + beat*64 + byte_index)*29 + 17) & 255;
        end
    endfunction

    task step;
        @(posedge clk);
        #1;
        @(negedge clk);
    endtask

    task randomize_ready;
        rng = rng ^ (rng << 13);
        rng = rng ^ (rng >> 17);
        rng = rng ^ (rng << 5);
        m_axi_awready = rng[0];
        m_axi_wready = rng[2];
    endtask

    // Monitor only public pins; no access to the DUT's state or buffer.
    always @(posedge clk) begin
        cycles = cycles + 1;
        if (cycles > 30000) $fatal(1, "FAIL watchdog");
        if (rst_n) begin
            if (aw_stalled) need(m_axi_awvalid && aw_bundle === held_aw, "AW unstable under pause");
            if (w_stalled) need(m_axi_wvalid && w_bundle === held_w, "W unstable under pause");
            if (cpl_stalled) need(cpl_valid && cpl_bundle === held_cpl, "completion unstable under pause");
            if (axi_busy || cpl_valid) need(lease_held, "window released while referenced");
            if (stop || quarantined) need(!cmd_ready, "admission open while stopped");
            if (stop) need(!data_ready, "payload admitted while stopped");
            if (stop && !previous_stop) $display("TRACE STOP");
            if (rearm_valid && rearm_ready) begin
                need(!bus_owed && !axi_busy && !cpl_valid, "rearm with traffic/completion pending");
                need(source_stopped && ingress_isolated && quiesce_epoch == epoch &&
                     rearm_epoch > epoch, "rearm lacks current certificates");
                $display("TRACE REARM %0d", rearm_epoch);
            end
            if (cmd_valid && cmd_ready) begin
                need(!bus_owed, "overlapping jobs");
                expected_tag = cmd_tag;
                expected_epoch = cmd_epoch;
                expected_addr = cmd_addr;
                expected_beats = {1'b0, cmd_len} + 1;
                seen_aw = 0;
                seen_w = 0;
                printed_cpl = 0;
                foreign_seen = 0;
                $display("TRACE CMD %0d %0d %016h %0d", cmd_tag, cmd_epoch, cmd_addr, expected_beats);
            end
            if (axi_busy && !previous_busy) begin
                need(!bus_owed, "second launch before B");
                bus_owed = 1;
                $display("TRACE LAUNCH");
            end
            if (m_axi_awvalid && m_axi_awready) begin
                need(bus_owed && !seen_aw, "unexpected/duplicate AW");
                need(m_axi_awaddr == expected_addr && m_axi_awlen + 1 == expected_beats,
                     "AW address/length mismatch");
                need(m_axi_awsize == 6 && m_axi_awburst == 1 && m_axi_awcache == 0 &&
                     m_axi_awid == 0 && m_axi_awprot == 0 && !m_axi_awlock, "AXI profile");
                seen_aw = 1;
                aw_count = aw_count + 1;
                $display("TRACE AW %016h %0d", m_axi_awaddr, m_axi_awlen + 1);
            end
            if (m_axi_wvalid && m_axi_wready) begin
                need(bus_owed && seen_w < expected_beats, "unexpected/extra W");
                need(m_axi_wdata === pattern(expected_tag, seen_w), "payload mismatch");
                need(&m_axi_wstrb, "partial WSTRB");
                need(m_axi_wlast == (seen_w == expected_beats - 1), "WLAST mismatch");
                $display("TRACE W %0d %0128h", m_axi_wlast, m_axi_wdata);
                seen_w = seen_w + 1;
                w_count = w_count + 1;
            end
            if (m_axi_bvalid && m_axi_bready && m_axi_bid != 0 && !foreign_seen) begin
                foreign_seen = 1;
                $display("TRACE FOREIGN_B");
            end
            if (m_axi_bvalid && m_axi_bready && m_axi_bid == 0) begin
                need(bus_owed && seen_aw && seen_w == expected_beats, "premature B");
                bus_owed = 0;
                b_count = b_count + 1;
                $display("TRACE B %0d", m_axi_bresp);
            end
            if (cpl_valid && !printed_cpl) begin
                need(cpl_tag == expected_tag && cpl_epoch == expected_epoch, "completion identity");
                if (cpl_status == 0) need(!bus_owed && seen_aw && seen_w == expected_beats,
                                         "success before final B");
                printed_cpl = 1;
                $display("TRACE CPL %0d", cpl_status);
            end
            if (cpl_valid && cpl_ready) $display("TRACE RETIRE");
            previous_busy = axi_busy;
            previous_stop = stop;
            aw_stalled = m_axi_awvalid && !m_axi_awready;
            w_stalled = m_axi_wvalid && !m_axi_wready;
            cpl_stalled = cpl_valid && !cpl_ready;
            held_aw = aw_bundle;
            held_w = w_bundle;
            held_cpl = cpl_bundle;
        end
    end

    task arm;
        stop = 0;
        rearm_epoch = epoch + 1;
        quiesce_epoch = epoch;
        source_stopped = 1;
        ingress_isolated = 1;
        rearm_valid = 1;
        #1;
        need(rearm_ready, "expected safe rearm");
        step();
        rearm_valid = 0;
        source_stopped = 0;
        ingress_isolated = 0;
        need(cmd_ready && !lease_held && !quarantined, "arm did not open admission");
    endtask

    task issue(input string name, input [63:0] addr, input integer count, input bit stale);
        cases = cases + 1;
        $display("CASE %0d %s", cases, name);
        cmd_tag = cases;
        cmd_epoch = stale ? epoch - 1 : epoch;
        cmd_addr = addr;
        cmd_len = count - 1;
        cmd_valid = 1;
        m_axi_awready = 0;
        m_axi_wready = 0;
        fill_index = 0;
        #1;
        need(cmd_ready, "command not accepted");
        step();
        cmd_valid = 0;
        need(lease_held && !cmd_ready, "missing command reservation");
    endtask

    task feed(input integer count, input bit end_last);
        integer index;
        for (index = 0; index < count; index = index + 1) begin
            data_valid = 1;
            data = pattern(cmd_tag, fill_index);
            data_last = end_last && index == count - 1;
            #1;
            need(data_ready, "payload refused");
            step();
            fill_index = fill_index + 1;
        end
        data_valid = 0;
    endtask

    task drain_channels;
        integer tries;
        tries = 0;
        while (!m_axi_bready && tries < 2000) begin
            randomize_ready();
            step();
            tries = tries + 1;
        end
        need(m_axi_bready, "AW/W failed to drain");
        need(!m_axi_awvalid && !m_axi_wvalid, "extra AW/W after drain");
    endtask

    task respond(input [1:0] response);
        m_axi_bresp = response;
        m_axi_bvalid = 1;
        #1;
        need(m_axi_bready, "B refused after AW/W");
        step();
        m_axi_bvalid = 0;
        need(!axi_busy, "AXI still busy after B");
    endtask

    task retire(input [2:0] status);
        need(cpl_valid && cpl_status == status, "wrong or missing completion");
        repeat (3) step();
        cpl_ready = 1;
        step();
        cpl_ready = 0;
        need(!cpl_valid, "completion not retired");
        if (quarantined) need(lease_held && !cmd_ready, "error retirement released window");
        else need(!lease_held && cmd_ready, "success retirement did not release window");
    endtask

    task halt;
        stop = 1;
        step();
        stop = 0;
        need(quarantined && !cmd_ready, "stop not latched");
    endtask

    integer n, old_aw, old_w, count;
    initial begin
        repeat (3) step();
        rst_n = 1;
        step();
        need(quarantined && !cmd_ready, "cold start admitted work");
        arm();

        issue("all W before AW, high address, page end", 64'h100000f00, 4, 0);
        feed(4, 1);
        step();
        m_axi_wready = 1;
        repeat (4) step();
        need(m_axi_awvalid && !m_axi_wvalid && !m_axi_bready, "AW/W independence");
        m_axi_awready = 1;
        step();
        repeat (5) step();
        need(!cpl_valid && lease_held && axi_busy, "bridge acceptance treated as final completion");
        respond(0);
        retire(0);

        issue("AW before W, last 256 bytes of 512 GiB", 64'h7fffffff00, 4, 0);
        feed(4, 1);
        step();
        m_axi_awready = 1;
        repeat (4) step();
        need(!m_axi_awvalid && m_axi_wvalid, "W stall lost payload");
        drain_channels();
        respond(0);
        retire(0);

        issue("stop with both VALID stalled, retire before late writes/B", 64'h200000000, 4, 0);
        feed(4, 1);
        step();
        repeat (3) step();
        halt();
        retire(1);
        source_stopped = 1;
        ingress_isolated = 1;
        quiesce_epoch = epoch;
        rearm_epoch = epoch + 1;
        rearm_valid = 1;
        repeat (3) begin
            #1; need(!rearm_ready && lease_held, "rearm before AW/W"); step();
        end
        drain_channels();
        repeat (5) begin
            need(!rearm_ready && !cpl_valid, "rearm before B / duplicate completion"); step();
        end
        respond(0);
        rearm_valid = 0;
        need(!cpl_valid && lease_held, "late B changed retired failure");
        arm();

        issue("stop with incomplete local payload; certificate checks", 64'h300000000, 4, 0);
        feed(2, 0);
        halt();
        retire(1);
        rearm_valid = 1;
        rearm_epoch = epoch + 1;
        quiesce_epoch = epoch;
        #1; need(!rearm_ready, "missing both certificates"); step();
        source_stopped = 1;
        #1; need(!rearm_ready, "missing ingress isolation"); step();
        source_stopped = 0;
        ingress_isolated = 1;
        #1; need(!rearm_ready, "missing source stop"); step();
        source_stopped = 1;
        quiesce_epoch = epoch - 1;
        #1; need(!rearm_ready, "stale certificates"); step();
        quiesce_epoch = epoch;
        rearm_epoch = epoch;
        #1; need(!rearm_ready, "same epoch reused"); step();
        rearm_valid = 0;
        arm();

        old_aw = aw_count;
        old_w = w_count;
        issue("stop after buffering but before AXI launch", 64'h400000000, 4, 0);
        feed(4, 1);
        halt();
        retire(1);
        need(aw_count == old_aw && w_count == old_w && !axi_busy, "unlaunched payload escaped");
        arm();

        for (n = 2; n <= 3; n = n + 1) begin
            issue("final AXI error retains quarantined window", 64'h500000000, 2, 0);
            feed(2, 1);
            step();
            drain_channels();
            respond(n[1:0]);
            retire(4);
            arm();
        end

        old_aw = aw_count;
        issue("reject unaligned address", 64'h600000001, 1, 0);
        retire(2); arm();
        issue("reject 4 KiB crossing", 64'h600000fc0, 2, 0);
        retire(2); arm();
        issue("reject excessive buffer length", 64'h600000000, MAX_BEATS + 1, 0);
        retire(2); arm();
        issue("reject stale epoch", 64'h600000000, 1, 1);
        retire(2); arm();
        issue("reject early LAST before any AXI exposure", 64'h600000000, 4, 0);
        feed(1, 1);
        retire(3); arm();
        issue("reject missing LAST before any AXI exposure", 64'h600000000, 1, 0);
        feed(1, 0);
        retire(3); arm();
        need(aw_count == old_aw, "invalid request issued AW");

        issue("stop concurrent with final B has priority", 64'h700000000, 2, 0);
        feed(2, 1); step(); drain_channels();
        stop = 1;
        respond(0);
        stop = 0;
        retire(1); arm();

        issue("published success remains stable across later stop", 64'h700001000, 1, 0);
        feed(1, 1); step(); drain_channels(); respond(0);
        step();
        halt();
        retire(0); arm();

        issue("stop after some W accepted", 64'h700002000, 4, 0);
        feed(4, 1); step();
        m_axi_awready = 1;
        m_axi_wready = 1;
        repeat (2) step();
        m_axi_wready = 0;
        halt(); retire(1);
        drain_channels(); respond(0); arm();

        // Independent deterministic seeds, bounded pauses, variable payload size.
        for (n = 1; n <= 20; n = n + 1) begin
            rng = n;
            count = n == 20 ? MAX_BEATS : (n * 7) % MAX_BEATS + 1;
            issue("seeded independent AW/W pauses", 64'h800000000 + n*4096, count, 0);
            feed(count, 1); step(); drain_channels();
            repeat (n % 7) step();
            respond(0);
            retire(0);
        end

        issue("foreign BID cannot discharge outstanding write", 64'h900000000, 1, 0);
        feed(1, 1); step(); drain_channels();
        m_axi_bid = 7;
        m_axi_bvalid = 1;
        step();
        m_axi_bvalid = 0;
        m_axi_bid = 0;
        retire(4);
        source_stopped = 1;
        ingress_isolated = 1;
        quiesce_epoch = epoch;
        rearm_epoch = epoch + 1;
        rearm_valid = 1;
        repeat (5) begin
            #1; need(!rearm_ready && axi_busy && lease_held && m_axi_bready,
                     "foreign B incorrectly drained the write"); step();
        end
        $display("PASS cases=%0d max_beats=%0d aw=%0d w=%0d b=%0d cycles=%0d",
                 cases, MAX_BEATS, aw_count, w_count, b_count, cycles);
        $finish;
    end
endmodule
