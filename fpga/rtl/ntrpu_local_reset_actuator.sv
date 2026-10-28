`timescale 1ns / 1ps
// Local reset phase actuator. All commands and peer certificates are already
// qualified in the independent, continuously running mgmt_clk domain.
// The parent MUST prove ownership and COMPLETE drain before requesting HELD.
// This module proves local reset edges, not upstream isolation or DMA drain.
// Every command identity must be unique within the configuration lifetime.
module ntrpu_local_reset_actuator #(
    parameter integer HOLD_CYCLES = 64,
    parameter integer CLEAR_CYCLES = 8
) (
    input wire mgmt_clk, host_clk, device_clk, cold_rst_n,
    input wire req_valid, hold_request, peers_held, peers_ready,
    input wire [2:0] req_kind,
    input wire [63:0] req_generation, req_tag,
    output wire ack_valid,
    output wire [2:0] ack_kind,
    output wire [63:0] ack_generation, ack_tag,
    output wire host_reset_n, device_reset_n, peers_reset_req,
    output wire host_held, device_held, host_up, device_up,
    output wire [2:0] state_debug
);
    localparam [2:0] LOCAL_HELD=2, LOCAL_READY=5;
    localparam [2:0] IDLE=0, CLEAR=1, COUNT=2, HELD=3, RELEASE=4, READY=5;
    localparam integer CW=$clog2(CLEAR_CYCLES+1);
    reg [2:0] state=IDLE, saved_kind=0;
    reg [63:0] saved_generation=0, saved_tag=0;
    reg have_command=0, command_started=0, answer=0;
    reg hold_reset=0, clear_counters=1;
    reg [CW-1:0] clear_count=0;
    wire host_done, device_done;
    wire same_identity=have_command && saved_kind==req_kind &&
        saved_generation==req_generation && saved_tag==req_tag;
    wire current_command=req_valid && same_identity && cold_rst_n;
    assign ack_kind=saved_kind;
    assign ack_generation=saved_generation;
    assign ack_tag=saved_tag;
    // Withdrawal, replacement or loss of qualified peer readiness masks an
    // old level immediately. No old certificate can acquire a new identity.
    assign ack_valid=answer && current_command &&
        ((saved_kind==LOCAL_HELD && state==HELD && hold_request &&
          host_held && device_held && peers_held) ||
         (saved_kind==LOCAL_READY && state==READY && !hold_request &&
          host_up && device_up && peers_ready));
    assign peers_reset_req=hold_reset;
    assign state_debug=state;

    ntrpu_reset_hold_counter #(.HOLD_CYCLES(HOLD_CYCLES)) host_counter (
        .clk(host_clk),.clear_async(clear_counters),.done(host_done));
    ntrpu_reset_hold_counter #(.HOLD_CYCLES(HOLD_CYCLES)) device_counter (
        .clk(device_clk),.clear_async(clear_counters),.done(device_done));
    ntrpu_reset_release host_count_cdc (
        .clk(mgmt_clk),.hold_async(!host_done),.released(host_held));
    ntrpu_reset_release device_count_cdc (
        .clk(mgmt_clk),.hold_async(!device_done),.released(device_held));
    ntrpu_reset_release host_release (
        .clk(host_clk),.hold_async(hold_reset),.released(host_reset_n));
    ntrpu_reset_release device_release (
        .clk(device_clk),.hold_async(hold_reset),.released(device_reset_n));
    ntrpu_reset_release host_up_cdc (
        .clk(mgmt_clk),.hold_async(!host_reset_n),.released(host_up));
    ntrpu_reset_release device_up_cdc (
        .clk(mgmt_clk),.hold_async(!device_reset_n),.released(device_up));

    always @(posedge mgmt_clk) begin
        // Configuration only. Runtime reset MUST NOT drive cold_rst_n.
        if(!cold_rst_n) begin
            state<=IDLE; hold_reset<=0; clear_counters<=1; clear_count<=0;
            have_command<=0; command_started<=0; answer<=0;
            saved_kind<=0; saved_generation<=0; saved_tag<=0;
        end else begin
            // Once asserted, reset remains held even if its command vanishes.
            // Clear must be observed low at BOTH counters before reusing them.
            case(state)
                CLEAR: begin
                    if(clear_count<CLEAR_CYCLES-1) clear_count<=clear_count+1'b1;
                    else if(!host_held && !device_held) begin
                        clear_counters<=0; state<=COUNT;
                    end
                end
                COUNT: if(host_held && device_held) state<=HELD;
                RELEASE: if(host_up && device_up && peers_ready) state<=READY;
                default: begin end
            endcase
            if(!req_valid) answer<=0;
            if(req_valid && !same_identity) begin
                saved_kind<=req_kind; saved_generation<=req_generation;
                saved_tag<=req_tag; have_command<=1;
                command_started<=0; answer<=0;
            end else if(current_command) begin
                if(saved_kind==LOCAL_HELD && hold_request) begin
                    if(!command_started) begin
                        // A fresh HELD identity always starts a fresh count,
                        // including retries while outputs remain asserted.
                        hold_reset<=1; clear_counters<=1; clear_count<=0;
                        state<=CLEAR; command_started<=1; answer<=0;
                    end else if(state==HELD && host_held && device_held && peers_held)
                        answer<=1;
                end else if(saved_kind==LOCAL_READY && !hold_request) begin
                    if(!command_started && state==HELD && peers_held) begin
                        hold_reset<=0; state<=RELEASE;
                        command_started<=1; answer<=0;
                    end else if(command_started && state==READY &&
                                host_up && device_up && peers_ready) answer<=1;
                end
            end
        end
    end
    // synthesis translate_off
    initial if(HOLD_CYCLES<64 || CLEAR_CYCLES<4)
        $fatal(1,"LOCAL_RESET_PARAMETERS");
    // synthesis translate_on
endmodule
