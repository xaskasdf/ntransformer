// One global DCI calibration after FPGA configuration, using an always-on clock.
// Warm board resets hold MIG in reset but do not interrupt/restart DCIRESET.
module ntrpu_dci_startup #(
    parameter integer CLOCK_PERIOD_PS = 4000,
    parameter integer GUARD_CYCLES = 64,
    parameter integer PULSE_CYCLES = 64
) (
    input  wire clk,
    input  wire eos_async,
    input  wire board_reset_async,
    input  wire dci_locked_async,
    output logic dci_rst = 1'b0,
    output wire mig_sys_rst,
    output wire ready
);
    localparam integer COUNT_WIDTH = $clog2(
        (GUARD_CYCLES > PULSE_CYCLES ? GUARD_CYCLES : PULSE_CYCLES) + 1);
    typedef enum logic [1:0] {WAIT_EOS, LOW_GUARD, HIGH_PULSE, SETTLE} state_t;
    state_t state = WAIT_EOS;
    logic [COUNT_WIDTH-1:0] count = 0;
    logic sequence_done = 0;
    (* ASYNC_REG = "TRUE" *) logic [1:0] eos_sync = 0;
    (* ASYNC_REG = "TRUE" *) logic [1:0] reset_release = 2'b11;

    // Only reconfiguration restarts this state machine. EOS is a startup event.
    // An external reset cannot shorten a DCI high/low interval.
    always_ff @(posedge clk) begin
        eos_sync <= {eos_sync[0], eos_async};
        case (state)
            WAIT_EOS: if (eos_sync[1]) begin
                count <= 0;
                state <= LOW_GUARD;
            end
            LOW_GUARD: if (count == GUARD_CYCLES-1) begin
                count <= 0;
                dci_rst <= 1;
                state <= HIGH_PULSE;
            end else count <= count + 1'b1;
            HIGH_PULSE: if (count == PULSE_CYCLES-1) begin
                count <= 0;
                dci_rst <= 0;
                state <= SETTLE;
            end else count <= count + 1'b1;
            SETTLE: if (!sequence_done) begin
                if (count == GUARD_CYCLES-1) sequence_done <= 1;
                else count <= count + 1'b1;
            end
            default: begin
                dci_rst <= 0;
                sequence_done <= 0;
                count <= 0;
                state <= WAIT_EOS;
            end
        endcase
    end

    // Asynchronous assertion, two-clock synchronous release. Any lock/EOS loss
    // reasserts MIG reset; returning signals cannot create a short release pulse.
    wire hold_reset = board_reset_async || !eos_async || !dci_locked_async || !sequence_done;
    always_ff @(posedge clk or posedge hold_reset) begin
        if (hold_reset) reset_release <= 2'b11;
        else reset_release <= {reset_release[0], 1'b0};
    end
    assign mig_sys_rst = reset_release[1];
    assign ready = !reset_release[1];

    // synthesis translate_off
    initial begin
        if (CLOCK_PERIOD_PS <= 0 || GUARD_CYCLES < 2 || PULSE_CYCLES < 2 ||
            GUARD_CYCLES * CLOCK_PERIOD_PS < 100000 ||
            PULSE_CYCLES * CLOCK_PERIOD_PS < 100000)
            $fatal(1, "NTRPU_DCI configured pulse/guard shorter than 100 ns");
    end
    // synthesis translate_on
endmodule

// Board integration: exactly one instance for all channels using global DCI.
// clk must exist independently of MIG sys_rst and UI clocks.
module ntrpu_board_startup (
    input wire clk,
    input wire board_reset_async,
    output wire mig_sys_rst,
    output wire ready
);
    wire eos, dci_rst, dci_locked;
    STARTUPE3 startup (
        .EOS(eos), .CFGCLK(), .CFGMCLK(), .PREQ(), .DI(),
        .DO(4'b0), .DTS(4'b1111), .FCSBO(1'b1), .FCSBTS(1'b1),
        .GSR(1'b0), .GTS(1'b0), .KEYCLEARB(1'b1), .PACK(1'b0),
        .USRCCLKO(1'b0), .USRCCLKTS(1'b1), .USRDONEO(1'b1), .USRDONETS(1'b1)
    );
    DCIRESET dci (.RST(dci_rst), .LOCKED(dci_locked));
    ntrpu_dci_startup sequencer (
        .clk(clk), .eos_async(eos), .board_reset_async(board_reset_async),
        .dci_locked_async(dci_locked), .dci_rst(dci_rst),
        .mig_sys_rst(mig_sys_rst), .ready(ready)
    );
endmodule
