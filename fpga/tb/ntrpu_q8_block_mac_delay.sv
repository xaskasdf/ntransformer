`timescale 1ns/1ps
// Simulation-only completion delay around the real, otherwise unchanged HLS RTL.
// The testbench drives hold_completion and optional invalid input-scale injection.
// Clock, arithmetic implementation and result-valid are unchanged. A held
// genuine completion/status is delivered once after release.
module ntrpu_q8_block_mac (
    input wire ap_clk, ap_rst, ap_start,
    output wire ap_done, ap_idle, ap_ready,
    input wire [63:0] weights_0, weights_1, weights_2, weights_3,
    input wire [63:0] input_0, input_1, input_2, input_3,
    input wire [15:0] weight_scale,
    input wire [63:0] input_scale, accumulator,
    output wire [63:0] result,
    output wire result_ap_vld,
    output wire [31:0] ap_return
);
    logic hold_completion=0;
    logic inject_invalid_scale=0;
    logic pending=0;
    logic [31:0] saved_status=0;
    wire raw_done;
    wire [31:0] raw_status;
    ntrpu_q8_block_mac_raw actual (
        .ap_clk, .ap_rst, .ap_start, .ap_done(raw_done), .ap_idle, .ap_ready,
        .weights_0, .weights_1, .weights_2, .weights_3,
        .input_0, .input_1, .input_2, .input_3,
        .weight_scale, .input_scale(inject_invalid_scale ? 64'h7ff8000000000000:input_scale),
        .accumulator, .result, .result_ap_vld,
        .ap_return(raw_status)
    );
    assign ap_done = !hold_completion && (raw_done || pending);
    assign ap_return = pending ? saved_status : raw_status;
    always @(posedge ap_clk) begin
        if (ap_rst) begin pending<=0; saved_status<=0; end
        else begin
            if (ap_start && pending) $fatal(1,"FAIL MAC DELAY: start before held completion retired");
            if (hold_completion && raw_done) begin
                if (pending) $fatal(1,"FAIL MAC DELAY: duplicate genuine completion");
                pending<=1; saved_status<=raw_status;
            end else if (!hold_completion && pending) pending<=0;
        end
    end
endmodule
