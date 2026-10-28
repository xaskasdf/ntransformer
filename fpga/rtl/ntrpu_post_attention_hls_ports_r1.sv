// Inspected generated core SHA256: ac90a7877cd58d2ba630cbd3b1dc42fee190ae94d3bf704fc2cc074d5c40bce2
`default_nettype none
module ntrpu_post_attention_hls_ports_r1 (
    input wire clk,rst_n,hls_start,
    output wire hls_done,hls_ready,
    output wire [31:0] hls_return,
    input wire [31:0] hls_elements,hls_epsilon_bits,
    output wire [2:0] hls_input_ce,
    output wire [10:0] hls_input_address [0:2],
    input wire [31:0] hls_input_q [0:2],
    output wire hls_residual_ce,hls_residual_we,
    output wire [10:0] hls_residual_address,
    output wire [31:0] hls_residual_d,
    output wire hls_ffn_ce,hls_ffn_we,
    output wire [8:0] hls_ffn_address,
    output wire [63:0] hls_ffn_d
);
    wire unused_idle;
    ntrpu_post_attention_r1 core (
        .ap_clk(clk),
        .ap_rst(~rst_n),
        .ap_start(hls_start),
        .ap_done(hls_done),
        .ap_ready(hls_ready),
        .ap_idle(unused_idle),
        .ap_return(hls_return),
        .elements(hls_elements),
        .epsilon_bits(hls_epsilon_bits),
        .residual_input_address0(hls_input_address[0]),
        .residual_input_ce0(hls_input_ce[0]),
        .residual_input_q0(hls_input_q[0]),
        .branch_address0(hls_input_address[1]),
        .branch_ce0(hls_input_ce[1]),
        .branch_q0(hls_input_q[1]),
        .norm_weight_address0(hls_input_address[2]),
        .norm_weight_ce0(hls_input_ce[2]),
        .norm_weight_q0(hls_input_q[2]),
        .residual_output_address0(hls_residual_address),
        .residual_output_ce0(hls_residual_ce),
        .residual_output_we0(hls_residual_we),
        .residual_output_d0(hls_residual_d),
        .ffn_input_address0(hls_ffn_address),
        .ffn_input_ce0(hls_ffn_ce),
        .ffn_input_we0(hls_ffn_we),
        .ffn_input_d0(hls_ffn_d)
    );
endmodule
`default_nettype wire
