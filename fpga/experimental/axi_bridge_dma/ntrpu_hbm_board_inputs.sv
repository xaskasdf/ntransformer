`timescale 1ns/1ps
`default_nettype none
// Pads only. The native composition continues to own its three BUFGCEs.
// Board XDC must select electrical standards/termination from the source design.
module ntrpu_hbm_board_inputs (
    input wire mgmt_clk_p, mgmt_clk_n,
    input wire hbm_ref0_clk_p, hbm_ref0_clk_n,
    input wire hbm_ref1_clk_p, hbm_ref1_clk_n,
    input wire supervisor_rst_n_pad,
    output wire mgmt_clk_in, hbm_ref0_clk_in, hbm_ref1_clk_in,
    output wire supervisor_rst_n
);
    IBUFDS management_input(.I(mgmt_clk_p), .IB(mgmt_clk_n), .O(mgmt_clk_in));
    IBUFDS hbm_reference0_input(.I(hbm_ref0_clk_p), .IB(hbm_ref0_clk_n), .O(hbm_ref0_clk_in));
    IBUFDS hbm_reference1_input(.I(hbm_ref1_clk_p), .IB(hbm_ref1_clk_n), .O(hbm_ref1_clk_in));
    IBUF supervisor_input(.I(supervisor_rst_n_pad), .O(supervisor_rst_n));
endmodule
`default_nettype wire
