// Interface preflight ONLY. Deliberately fails if executed.
module ntrpu_norm_q8_abi1_r1 (
    input wire ap_clk,
    input wire ap_rst,
    input wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,
    output wire [10:0] input_r_address0,
    output wire input_r_ce0,
    input wire [31:0] input_r_q0,
    output wire [10:0] weight_address0,
    output wire weight_ce0,
    input wire [31:0] weight_q0,
    output wire [8:0] output_r_address0,
    output wire output_r_ce0,
    output wire output_r_we0,
    output wire [63:0] output_r_d0,
    input wire [31:0] elements,
    input wire [31:0] epsilon_bits,
    output wire [31:0] ap_return
);
    initial $fatal(1,"COMPILE_SIGNATURE_MUST_NOT_EXECUTE");
endmodule
