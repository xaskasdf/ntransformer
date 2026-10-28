// One cold epoch for a shared receiver and two independently clocked memories.
// No degraded mode: both memories must be ready. After RUN, no warm rearm.
// mgmt_clk is independent and continuous; sys_rst of each MIG is external.
module ntrpu_dual_mig_cold_start (
    input wire mgmt_clk, source_clk,
    input wire [1:0] ui_clk, calibrated, ui_rst, memory_bvalid, memory_rvalid,
    input wire external_reset_async, rearm, source_clean,
    output wire source_reset_n, admit, rearm_ready, peers_reset_req,
    output wire [1:0] memory_reset_n, memory_alive,
    output wire source_alive, restart_required,
    output logic running = 0
);
    typedef enum logic [1:0] {CLEAR, HOLD, RELEASE, RUN} state_t;
    state_t state = CLEAR;
    logic [3:0] clear_count = 0, source_count = 0;
    logic opened = 0, clear_counters = 1, hold_reset = 1, block_admission = 1;
    logic source_held = 0;
    wire mgmt_reset, source_done, source_done_m, source_up_m, source_held_m;
    wire [1:0] memory_done_m, memory_up_m, memory_held_m, memory_ready_m;
    wire clocks_alive = source_alive && (&memory_alive);
    ntrpu_clock_probe source_probe (.mgmt_clk(mgmt_clk), .observed_clk(source_clk), .alive(source_alive));
    xpm_cdc_async_rst #(.DEST_SYNC_FF(3), .INIT_SYNC_FF(1), .RST_ACTIVE_HIGH(1)) management_reset (
        .src_arst(external_reset_async), .dest_clk(mgmt_clk), .dest_arst(mgmt_reset));
    ntrpu_reset_hold_counter source_hold (.clk(source_clk), .clear_async(clear_counters), .done(source_done));
    ntrpu_reset_release source_count_cdc (.clk(mgmt_clk), .hold_async(!source_done), .released(source_done_m));
    ntrpu_reset_release source_reset_cdc (.clk(source_clk), .hold_async(hold_reset), .released(source_reset_n));
    ntrpu_reset_release source_up_cdc (.clk(mgmt_clk), .hold_async(!source_reset_n), .released(source_up_m));
    ntrpu_reset_release source_held_cdc (.clk(mgmt_clk), .hold_async(!source_held), .released(source_held_m));
    ntrpu_reset_release admission_cdc (.clk(source_clk), .hold_async(block_admission), .released(admit));
    always_ff @(posedge source_clk) begin
        if (source_reset_n || !source_clean) begin source_count <= 0; source_held <= 0; end
        else if (!source_held) begin
            if (source_count == 7) source_held <= 1;
            else source_count <= source_count + 1'b1;
        end
    end
    for (genvar g=0; g<2; g++) begin: memory
        wire done;
        logic [3:0] count = 0;
        logic held = 0, ready = 0;
        ntrpu_clock_probe probe (.mgmt_clk(mgmt_clk), .observed_clk(ui_clk[g]), .alive(memory_alive[g]));
        ntrpu_reset_hold_counter reset_hold (.clk(ui_clk[g]), .clear_async(clear_counters), .done(done));
        ntrpu_reset_release count_cdc (.clk(mgmt_clk), .hold_async(!done), .released(memory_done_m[g]));
        ntrpu_reset_release reset_cdc (.clk(ui_clk[g]), .hold_async(hold_reset), .released(memory_reset_n[g]));
        ntrpu_reset_release up_cdc (.clk(mgmt_clk), .hold_async(!memory_reset_n[g]), .released(memory_up_m[g]));
        ntrpu_reset_release held_cdc (.clk(mgmt_clk), .hold_async(!held), .released(memory_held_m[g]));
        ntrpu_reset_release ready_cdc (.clk(mgmt_clk), .hold_async(!ready), .released(memory_ready_m[g]));
        always_ff @(posedge ui_clk[g]) begin
            ready <= calibrated[g] && !ui_rst[g];
            if (ui_rst[g] || !calibrated[g] || memory_reset_n[g] || memory_bvalid[g] || memory_rvalid[g]) begin count <= 0; held <= 0; end
            else if (!held) begin
                if (count == 7) held <= 1;
                else count <= count + 1'b1;
            end
        end
    end
    // Latch on the transition itself. Sampling running one cycle later leaves
    // a rearm window if a faster source opens and external reset arrives first.
    always_ff @(posedge mgmt_clk)
        if (!mgmt_reset && clocks_alive && state == RELEASE && source_up_m &&
            (&memory_up_m) && (&memory_ready_m)) opened <= 1;
    assign restart_required = opened && !running;
    assign peers_reset_req = hold_reset;
    assign rearm_ready = state == HOLD && !opened && !mgmt_reset && !external_reset_async && clocks_alive &&
                         source_done_m && (&memory_done_m) && source_held_m && (&memory_held_m);
    always_ff @(posedge mgmt_clk or posedge mgmt_reset) begin
        if (mgmt_reset) begin
            state <= CLEAR; clear_count <= 0; running <= 0;
            clear_counters <= 1; hold_reset <= 1; block_admission <= 1;
        end else if (!clocks_alive) begin
            state <= CLEAR; clear_count <= 0; running <= 0;
            clear_counters <= 1; hold_reset <= 1; block_admission <= 1;
        end else case (state)
            CLEAR: begin
                running <= 0;
                if (clear_count == 7) begin
                    if (!source_done_m && !(|memory_done_m)) begin state <= HOLD; clear_counters <= 0; end
                end else clear_count <= clear_count + 1'b1;
            end
            HOLD: if (rearm && rearm_ready) begin state <= RELEASE; hold_reset <= 0; end
            RELEASE: if (source_up_m && (&memory_up_m) && (&memory_ready_m)) begin
                state <= RUN; running <= 1; block_admission <= 0;
            end
            RUN: if (!( &memory_ready_m)) begin
                state <= CLEAR; clear_count <= 0; running <= 0;
                clear_counters <= 1; hold_reset <= 1; block_admission <= 1;
            end
            default: begin
                state <= CLEAR; clear_count <= 0; running <= 0;
                clear_counters <= 1; hold_reset <= 1; block_admission <= 1;
            end
        endcase
    end
endmodule
