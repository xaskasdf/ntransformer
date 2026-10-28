// Syntax-only MAC interface extracted from the qualified generated core. Never simulate or synthesize this stub.
module ntrpu_q8_block_mac (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        weights_0,
        weights_1,
        weights_2,
        weights_3,
        input_0,
        input_1,
        input_2,
        input_3,
        weight_scale,
        input_scale,
        accumulator,
        result,
        result_ap_vld,
        ap_return
);
input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [63:0] weights_0;
input  [63:0] weights_1;
input  [63:0] weights_2;
input  [63:0] weights_3;
input  [63:0] input_0;
input  [63:0] input_1;
input  [63:0] input_2;
input  [63:0] input_3;
input  [15:0] weight_scale;
input  [63:0] input_scale;
input  [63:0] accumulator;
output  [63:0] result;
output   result_ap_vld;
output  [31:0] ap_return;
endmodule
