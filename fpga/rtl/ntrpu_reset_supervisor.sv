// Application reset coordinator. mgmt_clk MUST remain present independently
// of XDMA, compute and memory clocks. All attached AXI peers must participate.
// This is not a PCIe/PERST or DDR physical reset generator.
module ntrpu_reset_release (
    input wire clk, hold_async,
    output wire released
);
    // Explicit INIT=0 keeps reset/admission closed during configuration too.
    // XPM_CDC_ASYNC_RST initializes to its DEASSERTED level, independent of
    // polarity, so its output alone does not provide this startup invariant.
    (* ASYNC_REG = "TRUE" *) logic [2:0] release_ff = 3'b000;
    always_ff @(posedge clk or posedge hold_async) begin
        if (hold_async) release_ff <= 3'b000;
        else release_ff <= {release_ff[1:0], 1'b1};
    end
    assign released = release_ff[2];
endmodule

module ntrpu_clock_probe #(
    parameter integer TIMEOUT_CYCLES = 64
) (
    input wire mgmt_clk, observed_clk,
    output logic alive = 1'b0
);
    localparam integer CW = $clog2(TIMEOUT_CYCLES + 1);
    logic request = 1'b1;
    logic [CW-1:0] age = 0;
    wire echo, returned;
    // Request remains stable until its round trip finishes. Unlike a free
    // running toggle, it cannot alias against the slower management clock.
    // Probes deliberately keep running during application resets.
    xpm_cdc_single #(.DEST_SYNC_FF(3), .INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1), .SRC_INPUT_REG(0)) outward (
        .src_clk(1'b0), .src_in(request), .dest_clk(observed_clk), .dest_out(echo));
    xpm_cdc_single #(.DEST_SYNC_FF(3), .INIT_SYNC_FF(1),
        .SIM_ASSERT_CHK(1), .SRC_INPUT_REG(0)) inward (
        .src_clk(1'b0), .src_in(echo), .dest_clk(mgmt_clk), .dest_out(returned));
    always_ff @(posedge mgmt_clk) begin
        if (returned == request) begin
            request <= !request;
            age <= 0;
            alive <= 1;
        end else if (age == TIMEOUT_CYCLES-1) begin
            alive <= 0;
        end else age <= age + 1'b1;
    end
endmodule

module ntrpu_reset_hold_counter #(
    parameter integer HOLD_CYCLES = 64
) (
    input wire clk, clear_async,
    output logic done = 0
);
    localparam integer CW = $clog2(HOLD_CYCLES + 1);
    wire clear;
    logic [CW-1:0] count = 0;
    xpm_cdc_async_rst #(.DEST_SYNC_FF(3), .INIT_SYNC_FF(1),
        .RST_ACTIVE_HIGH(1)) clear_cdc (
        .src_arst(clear_async), .dest_clk(clk), .dest_arst(clear));
    always_ff @(posedge clk or posedge clear) begin
        if (clear) begin count <= 0; done <= 0; end
        else if (!done) begin
            if (count == HOLD_CYCLES-1) done <= 1;
            else count <= count + 1'b1;
        end
    end
endmodule

module ntrpu_reset_supervisor #(
    parameter integer CLOCK_TIMEOUT_CYCLES = 64,
    parameter integer HOLD_CYCLES = 64,
    parameter integer CLEAR_CYCLES = 8
) (
    input wire mgmt_clk, host_clk, device_clk,
    input wire external_reset_async,
    // Management-domain contract: peers_held acknowledges that every attached
    // AXI peer is in reset; peers_ready acknowledges recovery after release.
    input wire peers_held, peers_ready, rearm,
    output wire peers_reset_req, host_reset_n, device_reset_n,
    output wire host_admit, device_admit,
    output wire rearm_ready,
    output wire host_alive, device_alive,
    output logic running = 0
);
    typedef enum logic [1:0] {CLEAR, HOLD, RELEASE, RUN} state_t;
    state_t state = CLEAR;
    localparam integer CW = $clog2(CLEAR_CYCLES + 1);
    logic [CW-1:0] clear_count = 0;
    wire mgmt_reset, host_done, device_done, host_done_m, device_done_m;
    wire host_up_m, device_up_m;
    wire fault = external_reset_async || !host_alive || !device_alive;
    // Registered source requests avoid combinational logic at CDC reset pins.
    // External reset presets asynchronously; clock-loss flags are already
    // management-domain registers and are sampled there without reset logic.
    logic clear_counters = 1'b1;
    logic hold_reset = 1'b1;
    logic block_admission = 1'b1;
    ntrpu_clock_probe #(.TIMEOUT_CYCLES(CLOCK_TIMEOUT_CYCLES)) host_probe (
        .mgmt_clk(mgmt_clk), .observed_clk(host_clk), .alive(host_alive));
    ntrpu_clock_probe #(.TIMEOUT_CYCLES(CLOCK_TIMEOUT_CYCLES)) device_probe (
        .mgmt_clk(mgmt_clk), .observed_clk(device_clk), .alive(device_alive));
    xpm_cdc_async_rst #(.DEST_SYNC_FF(3), .INIT_SYNC_FF(1), .RST_ACTIVE_HIGH(1)) management_reset (
        .src_arst(external_reset_async), .dest_clk(mgmt_clk), .dest_arst(mgmt_reset));
    ntrpu_reset_hold_counter #(.HOLD_CYCLES(HOLD_CYCLES)) host_hold (
        .clk(host_clk), .clear_async(clear_counters), .done(host_done));
    ntrpu_reset_hold_counter #(.HOLD_CYCLES(HOLD_CYCLES)) device_hold (
        .clk(device_clk), .clear_async(clear_counters), .done(device_done));
    // A reset just after counter completion may truncate done to one sample.
    // Clear its qualification immediately; only sustained completion is ready.
    ntrpu_reset_release host_count_cdc (
        .hold_async(!host_done),.clk(mgmt_clk),.released(host_done_m));
    ntrpu_reset_release device_count_cdc (
        .hold_async(!device_done),.clk(mgmt_clk),.released(device_done_m));
    ntrpu_reset_release host_reset_cdc (
        .hold_async(hold_reset), .clk(host_clk), .released(host_reset_n));
    ntrpu_reset_release device_reset_cdc (
        .hold_async(hold_reset), .clk(device_clk), .released(device_reset_n));
    // Repeated resets may truncate the high level. Assert low asynchronously
    // and accept high only after three management edges, starting from INIT=0.
    ntrpu_reset_release host_up_cdc (
        .hold_async(!host_reset_n),.clk(mgmt_clk),.released(host_up_m));
    ntrpu_reset_release device_up_cdc (
        .hold_async(!device_reset_n),.clk(mgmt_clk),.released(device_up_m));
    // Admission closes asynchronously even when its local clock has stopped.
    // Release is local and synchronous, after BOTH resets have been observed up.
    ntrpu_reset_release host_admit_cdc (
        .hold_async(block_admission), .clk(host_clk), .released(host_admit));
    ntrpu_reset_release device_admit_cdc (
        .hold_async(block_admission), .clk(device_clk), .released(device_admit));
    assign peers_reset_req = hold_reset;
    assign rearm_ready = state == HOLD && !mgmt_reset && !fault &&
                         host_done_m && device_done_m && peers_held;
    always_ff @(posedge mgmt_clk or posedge mgmt_reset) begin
        if (mgmt_reset) begin
            state <= CLEAR;
            clear_count <= 0;
            running <= 0;
            clear_counters <= 1;
            hold_reset <= 1;
            block_admission <= 1;
        end else if (!host_alive || !device_alive) begin
            // mgmt_clk is independent and continuous. This adds at most one
            // management cycle to detection and still asserts local resets
            // asynchronously even if the affected local clock has stopped.
            state <= CLEAR;
            clear_count <= 0;
            running <= 0;
            clear_counters <= 1;
            hold_reset <= 1;
            block_admission <= 1;
        end else begin
            case (state)
                CLEAR: begin
                    running <= 0;
                    if (clear_count == CLEAR_CYCLES-1) begin
                        if (!host_done_m && !device_done_m) begin
                            state <= HOLD;
                            clear_counters <= 0;
                        end
                    end else clear_count <= clear_count + 1'b1;
                end
                HOLD: if (rearm && rearm_ready) begin
                    state <= RELEASE;
                    hold_reset <= 0;
                end
                RELEASE: if (host_up_m && device_up_m && peers_ready) begin
                    state <= RUN;
                    running <= 1;
                    block_admission <= 0;
                end
                RUN: if (!peers_ready) begin
                    state <= CLEAR;
                    clear_count <= 0;
                    running <= 0;
                    clear_counters <= 1;
                    hold_reset <= 1;
                    block_admission <= 1;
                end
                default: begin
                    state <= CLEAR; clear_count <= 0; running <= 0;
                    clear_counters <= 1; hold_reset <= 1; block_admission <= 1;
                end
            endcase
        end
    end
    // synthesis translate_off
    initial begin
        if (CLOCK_TIMEOUT_CYCLES < 16 || HOLD_CYCLES < 64 || CLEAR_CYCLES < 4)
            $fatal(1, "NTRPU_SUPERVISOR invalid timeout/hold/clear configuration");
    end
    // synthesis translate_on
endmodule
