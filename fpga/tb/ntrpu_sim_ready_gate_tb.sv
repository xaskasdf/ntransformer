`timescale 1ps/1ps
module ntrpu_sim_ready_gate_tb #(parameter integer OFFER_DELAY = 0);
    reg clk = 0, resetn = 0, valid = 0, ready = 0;
    reg [63:0] payload = 0;
    wire in_ready, out_valid;
    wire [63:0] out_payload;
    integer sent = 0, received = 0, cycle = 0, mode = 0;
    bit corrupted = 0, advance = 1;
    always #2000 clk = !clk;
    ntrpu_sim_ready_gate #(.PERIOD(5), .BLOCKED(3), .OFFER_DELAY(OFFER_DELAY), .CHANNEL("UNIT")) dut (
        .clk(clk), .resetn(resetn), .s_valid(valid), .s_ready(in_ready),
        .s_payload(payload), .m_valid(out_valid), .m_ready(ready), .m_payload(out_payload)
    );
    function automatic [63:0] expected(input integer i);
        expected = 64'habcde00123450000 + i;
    endfunction
    initial begin
        if (!$value$plusargs("MODE=%d", mode)) mode = 0;
        repeat (4) @(negedge clk);
        resetn = 1;
        while (received < 64) begin
            // Change stimuli only on falling edges; stalled source holds VALID/data.
            ready = (cycle % 17) >= 11;
            if (advance) begin
                valid = sent < 64 && (cycle % 7 != 0);
                payload = expected(sent);
            end
            if (mode == 1 && dut.held_in && !corrupted) begin
                payload = payload ^ 64'h100;
                corrupted = 1;
            end
            @(posedge clk);
            advance = !valid || in_ready;
            if (valid && in_ready) sent++;
            if (out_valid && ready) begin
                if (out_payload !== expected(received)) $fatal(1, "UNIT_DATA offset=%0d", received);
                received++;
            end
            @(negedge clk);
            cycle++;
        end
        valid = 0;
        repeat (2) @(negedge clk);
        dut.verify_complete(64);
        if (sent != received || dut.sink_stalls == 0) $fatal(1, "UNIT_COVERAGE");
        if (dut.source_stalls < 64*OFFER_DELAY) $fatal(1, "UNIT_OFFER_DELAY_COVERAGE");
        if (mode != 0) $fatal(1, "UNIT_NEGATIVE_NOT_DETECTED");
        $display("NTRPU_READY_GATE_UNIT_PASS sent=%0d received=%0d cycles=%0d offer_delay=%0d", sent, received, cycle, OFFER_DELAY);
        $finish;
    end
    initial begin #10000000; $fatal(1, "UNIT_TIMEOUT"); end
endmodule

module ntrpu_sim_ready_gate_delay4_tb;
    ntrpu_sim_ready_gate_tb #(.OFFER_DELAY(4)) test();
endmodule

module ntrpu_sim_ready_gate_delay9_tb;
    ntrpu_sim_ready_gate_tb #(.OFFER_DELAY(9)) test();
endmodule
