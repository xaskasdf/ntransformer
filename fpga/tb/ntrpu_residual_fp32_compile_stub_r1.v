// Port connectivity only. Never execute or include in actual Vivado arithmetic.
module ntrpu_residual_fp32_r1 (
    input wire ap_clk,ap_rst,ap_start,
    output wire ap_done,ap_ready,ap_idle,
    output wire [31:0] ap_return,
    input wire [31:0] elements,
    output wire [10:0] residual_address0,
    output wire residual_ce0,
    input wire [31:0] residual_q0,
    output wire [10:0] branch_address0,
    output wire branch_ce0,
    input wire [31:0] branch_q0,
    output wire [10:0] output_r_address0,
    output wire output_r_ce0,output_r_we0,
    output wire [31:0] output_r_d0
);
endmodule
