// Compile-only exact generated quantizer interface. Do not simulate or synthesize.
module ntrpu_quantize_abi1_r1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        input_r_address0,
        input_r_ce0,
        input_r_q0,
        output_r_address0,
        output_r_ce0,
        output_r_we0,
        output_r_d0,
        elements,
        ap_return
);
input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output  [10:0] input_r_address0;
output   input_r_ce0;
input  [31:0] input_r_q0;
output  [8:0] output_r_address0;
output   output_r_ce0;
output   output_r_we0;
output  [63:0] output_r_d0;
input  [31:0] elements;
output  [31:0] ap_return;
endmodule
