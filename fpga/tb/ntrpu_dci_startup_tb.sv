`timescale 1ps/1ps
module ntrpu_dci_startup_tb;
    logic clk = 0, eos = 0, board_reset = 1, allow_lock = 1;
    wire dci_rst, vendor_locked, mig_reset, ready;
    wire locked = vendor_locked && allow_lock;
    int pulses = 0, releases = 0;
    time pulse_start = 0, pulse_end = 0, last_clock = 0;
    bit complete = 0;
    always #2000 clk = ~clk;
    DCIRESET dci (.RST(dci_rst), .LOCKED(vendor_locked));
    ntrpu_dci_startup dut (
        .clk(clk), .eos_async(eos), .board_reset_async(board_reset),
        .dci_locked_async(locked), .dci_rst(dci_rst), .mig_sys_rst(mig_reset), .ready(ready)
    );
    always @(posedge clk) last_clock = $time;
    always @(posedge dci_rst) begin
        if (!eos || $time < 100000) $fatal(1, "NTRPU_DCI premature pulse");
        pulse_start = $time;
        pulses++;
    end
    always @(negedge dci_rst) if (pulse_start != 0) begin
        pulse_end = $time;
        if (pulse_end-pulse_start != 256000) $fatal(1, "NTRPU_DCI truncated high pulse");
        $display("NTRPU_DCI_PULSE high_ps=%0d", pulse_end-pulse_start);
    end
    always @(negedge mig_reset) begin
        if (!eos || !locked || board_reset || dci_rst || $time != last_clock ||
            pulse_end == 0 || $time-pulse_end < 256000)
            $fatal(1, "NTRPU_DCI unsafe reset release");
        releases++;
    end
    task automatic expect_held(input int clocks);
        repeat (clocks) begin
            @(negedge clk);
            if (mig_reset !== 1 || ready !== 0) $fatal(1, "NTRPU_DCI reset not held");
        end
    endtask
    initial begin
        expect_held(50); // No EOS, no calibration pulse.
        if (pulses != 0) $fatal(1, "NTRPU_DCI pulse before EOS");
        #731 eos = 1;
        @(posedge dci_rst);
        #30001 board_reset = 0;
        #7003 board_reset = 1; // Short asynchronous external reset during pulse.
        #5001 board_reset = 0;
        @(negedge dci_rst);
        allow_lock = 0; // Test input gate, never force/change vendor primitive.
        expect_held(100);
        #719 allow_lock = 1;
        wait (ready === 1);
        if (releases != 1 || pulses != 1) $fatal(1, "NTRPU_DCI startup counts");
        $display("NTRPU_DCI_STARTUP_PASS delayed_lock=1 pulse_interrupted=0");

        #701 board_reset = 1;
        #1 if (!mig_reset) $fatal(1, "NTRPU_DCI external reset not asynchronous");
        #503 board_reset = 0; // Captured even though shorter than one clock.
        expect_held(1);
        wait (ready === 1);
        #719 allow_lock = 0;
        #1 if (!mig_reset) $fatal(1, "NTRPU_DCI lock loss not asynchronous");
        #503 allow_lock = 1;
        expect_held(1);
        wait (ready === 1);
        #713 eos = 0;
        #1 if (!mig_reset) $fatal(1, "NTRPU_DCI EOS loss not asynchronous");
        #509 eos = 1;
        expect_held(1);
        wait (ready === 1);
        repeat (40) @(negedge clk);
        if (pulses != 1 || releases != 4) $fatal(1, "NTRPU_DCI unexpected retrigger/count");
        complete = 1;
        $display("NTRPU_DCI_RECOVERY_PASS board_reset=1 lock_loss=1 eos_loss=1");
        $display("NTRPU_DCI_PASS pulses=%0d releases=%0d time_ps=%0d", pulses, releases, $time);
        $finish;
    end
    initial begin #10000000; $fatal(1, "NTRPU_DCI timeout"); end
    final if (!complete) $error("NTRPU_DCI_INCOMPLETE");
endmodule
