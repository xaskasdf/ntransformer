// Cold-start adapter only. mgmt_clk must remain independent and continuous.
// After first RUN, rearm is permanently disabled until FPGA reconfiguration.
// A runtime loss invalidates the epoch; this does not qualify MIG warm reset.
module ntrpu_mig_cold_start (
    input wire mgmt_clk, source_clk, ui_clk,
    input wire external_reset_async, rearm,
    input wire calibrated, ui_rst,
    input wire source_clean, memory_bvalid, memory_rvalid,
    output wire source_reset_n, memory_reset_n, admit,
    output wire peers_reset_req, rearm_ready, running, restart_required,
    output wire host_alive, device_alive
);
    logic source_held = 0, memory_held = 0, memory_ready = 0;
    logic [3:0] source_count = 0, memory_count = 0;
    logic opened = 0;
    wire source_held_m, memory_held_m, memory_ready_m;
    wire raw_rearm_ready, unused_device_admit;
    always_ff @(posedge source_clk) begin
        if (source_reset_n || !source_clean) begin
            source_count <= 0;
            source_held <= 0;
        end else if (!source_held) begin
            if (source_count == 7) source_held <= 1;
            else source_count <= source_count + 1'b1;
        end
    end
    always_ff @(posedge ui_clk) begin
        memory_ready <= calibrated && !ui_rst;
        if (memory_reset_n || memory_bvalid || memory_rvalid) begin
            memory_count <= 0;
            memory_held <= 0;
        end else if (!memory_held) begin
            if (memory_count == 7) memory_held <= 1;
            else memory_count <= memory_count + 1'b1;
        end
    end
    ntrpu_reset_release source_status (.clk(mgmt_clk), .hold_async(!source_held), .released(source_held_m));
    ntrpu_reset_release memory_status (.clk(mgmt_clk), .hold_async(!memory_held), .released(memory_held_m));
    ntrpu_reset_release memory_available (.clk(mgmt_clk), .hold_async(!memory_ready), .released(memory_ready_m));
    always_ff @(posedge mgmt_clk) if (running) opened <= 1;
    assign rearm_ready = raw_rearm_ready && !opened;
    assign restart_required = opened && (!running || !memory_ready_m);
    ntrpu_reset_supervisor supervisor (
        .mgmt_clk(mgmt_clk), .host_clk(source_clk), .device_clk(ui_clk),
        .external_reset_async(external_reset_async),
        .peers_held(source_held_m && memory_held_m), .peers_ready(memory_ready_m),
        .rearm(rearm && !opened), .peers_reset_req(peers_reset_req),
        .host_reset_n(source_reset_n), .device_reset_n(memory_reset_n),
        .host_admit(admit), .device_admit(unused_device_admit),
        .rearm_ready(raw_rearm_ready), .host_alive(host_alive), .device_alive(device_alive),
        .running(running)
    );
endmodule
