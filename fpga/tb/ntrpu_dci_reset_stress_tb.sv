`timescale 1ps/1ps
// Digital fault handover and clock-stop tests; not an analog glitch/MTBF model.
module ntrpu_dci_reset_stress_tb;
    logic clk = 0, clock_enable = 1;
    logic eos = 0, board_reset = 1, allow_lock = 1;
    wire vendor_locked, dci_rst, mig_reset, ready;
    wire locked = vendor_locked && allow_lock;
    int pulses = 0, releases = 0, cases = 0, clock_stops = 0;
    int healthy_edges = 0;
    bit monitor = 0, complete = 0;
    time last_clock = 0;
    DCIRESET primitive_dci (.RST(dci_rst), .LOCKED(vendor_locked));
    ntrpu_dci_startup dut (
        .clk(clk), .eos_async(eos), .board_reset_async(board_reset),
        .dci_locked_async(locked), .dci_rst(dci_rst),
        .mig_sys_rst(mig_reset), .ready(ready)
    );
    always #2000 if (clock_enable) clk = ~clk;
    always @(posedge dci_rst) pulses++;
    always @(posedge board_reset or negedge eos or negedge locked) begin
        healthy_edges = 0;
        if (monitor) begin
            #1;
            if (mig_reset !== 1 || ready !== 0) $fatal(1, "NTRPU_DCI_STRESS missed asynchronous fault");
        end
    end
    always @(posedge clk) begin
        last_clock = $time;
        if (board_reset || !eos || !locked) healthy_edges = 0;
        else healthy_edges++;
    end
    always @(negedge mig_reset) if (monitor) begin
        if (board_reset || !eos || !locked || $time != last_clock || healthy_edges < 2)
            $fatal(1, "NTRPU_DCI_STRESS early/asynchronous release");
        releases++;
    end
    task automatic faults(input logic [2:0] mask);
        {allow_lock, eos, board_reset} = {~mask[2], ~mask[1], mask[0]};
    endtask
    task automatic held;
        if (mig_reset !== 1 || ready !== 0) $fatal(1, "NTRPU_DCI_STRESS unexpected release");
    endtask
    task automatic handover(input logic [2:0] first, second, input int phase, gap_mode);
        wait (ready === 1);
        @(negedge clk); #101; faults(first);
        #1; held();
        repeat (3) @(negedge clk);
        #(phase);
        case (gap_mode)
            0: begin faults(first | second); #5; faults(second); end
            1: faults(second);
            2: begin faults(0); #5; faults(second); end
        endcase
        #1; held();
        repeat (3) begin @(negedge clk); held(); end
        #101; faults(0);
        wait (ready === 1);
        @(negedge clk);
        cases++;
    endtask
    initial begin
        repeat (50) @(negedge clk);
        #101; faults(0);
        wait (ready === 1);
        @(negedge clk); monitor = 1;
        for (int first = 1; first < 8; first++)
            for (int second = 1; second < 8; second++)
                for (int phase_id = 0; phase_id < 4; phase_id++)
                    for (int gap_mode = 0; gap_mode < 3; gap_mode++) begin
                        case (phase_id)
                            0: handover(3'(first), 3'(second), 17, gap_mode);
                            1: handover(3'(first), 3'(second), 1983, gap_mode);
                            2: handover(3'(first), 3'(second), 2017, gap_mode);
                            3: handover(3'(first), 3'(second), 3983, gap_mode);
                        endcase
                    end
        // Assert and remove each source with no reference edges. Reset must be
        // captured and remain held until the clock returns and two edges pass.
        for (int source_id = 0; source_id < 3; source_id++) begin
            @(negedge clk); #17; clock_enable = 0;
            #501; faults(3'(1 << source_id));
            #1; held();
            #9999; faults(0);
            #12000; held();
            clock_enable = 1;
            wait (ready === 1);
            @(negedge clk); clock_stops++;
        end
        if (pulses != 1 || cases != 588 || clock_stops != 3 || releases != 591)
            $fatal(1, "NTRPU_DCI_STRESS count mismatch cases=%0d stops=%0d releases=%0d", cases, clock_stops, releases);
        complete = 1;
        $display("NTRPU_DCI_STRESS_PASS cases=%0d clock_stops=%0d pulses=%0d releases=%0d time_ps=%0d",
                 cases, clock_stops, pulses, releases, $time);
        $finish;
    end
    initial begin #100000000; $fatal(1, "NTRPU_DCI_STRESS timeout"); end
    final if (!complete) $error("NTRPU_DCI_STRESS_INCOMPLETE");
endmodule
