// Simulation-only transparent VALID/READY gate. No data storage or CDC.
// Once offered, a beat stays offered until the consumer accepts it.
`timescale 1ps/1ps
module ntrpu_sim_ready_gate #(
    parameter integer WIDTH = 64,
    parameter integer PERIOD = 11,
    parameter integer BLOCKED = 6,
    parameter integer OFFER_DELAY = 0,
    parameter CHANNEL = "W"
) (
    input wire clk, resetn,
    input wire s_valid,
    output wire s_ready,
    input wire [WIDTH-1:0] s_payload,
    output wire m_valid,
    input wire m_ready,
    output wire [WIDTH-1:0] m_payload
);
    integer phase = 0, beats = 0, source_stalls = 0, sink_stalls = 0;
    integer offer_wait = 0;
    bit held_in = 0, held_out = 0;
    reg [WIDTH-1:0] saved_in, saved_out;
    wire periodic_enabled = held_out || (phase >= BLOCKED);
    // Optional deterministic latency on every offer, including back-to-back
    // transactions. Default zero preserves the original periodic fixture.
    wire enabled = OFFER_DELAY == 0 ? periodic_enabled :
                   (held_out || offer_wait == OFFER_DELAY);
    assign m_valid = resetn && enabled && s_valid;
    assign s_ready = resetn && enabled && m_ready;
    assign m_payload = s_payload;

    initial if (WIDTH < 1 || BLOCKED < 1 || BLOCKED >= PERIOD || OFFER_DELAY < 0)
        $fatal(1, "GATE_BAD_PARAMETERS");
    always @(posedge clk) begin
        if (!resetn) begin
            phase <= 0;
            held_in <= 0;
            held_out <= 0;
            offer_wait <= 0;
        end else begin
            if (s_valid && s_ready) offer_wait <= 0;
            else if (s_valid && offer_wait < OFFER_DELAY) offer_wait <= offer_wait + 1;
            phase <= phase == PERIOD-1 ? 0 : phase+1;
            if ($isunknown({s_valid, m_ready})) $fatal(1, "GATE_UNKNOWN_CONTROL %s", CHANNEL);
            if (held_in && (s_valid !== 1'b1 || s_payload !== saved_in))
                $fatal(1, "GATE_SOURCE_CHANGED %s", CHANNEL);
            if (held_out && (m_valid !== 1'b1 || m_payload !== saved_out))
                $fatal(1, "GATE_OUTPUT_CHANGED %s", CHANNEL);
            if ((s_valid && s_ready) !== (m_valid && m_ready))
                $fatal(1, "GATE_HANDSHAKE_MISMATCH %s", CHANNEL);
            held_in <= s_valid && !s_ready;
            held_out <= m_valid && !m_ready;
            saved_in <= s_payload;
            saved_out <= m_payload;
            if (s_valid && !s_ready) source_stalls++;
            if (m_valid && !m_ready) sink_stalls++;
            if (s_valid && s_ready) beats++;
        end
    end
    task verify_complete(input integer expected_beats);
        begin
            if (beats != expected_beats || source_stalls == 0 || held_in || held_out)
                $fatal(1, "GATE_INCOMPLETE %s beats=%0d stalls=%0d", CHANNEL, beats, source_stalls);
            $display("NTRPU_AXI_GATE_COMPLETE channel=%s beats=%0d source_stalls=%0d sink_stalls=%0d",
                CHANNEL, beats, source_stalls, sink_stalls);
        end
    endtask
endmodule
