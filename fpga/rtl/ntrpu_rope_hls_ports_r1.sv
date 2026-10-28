// Inspected generated core SHA256: 59f30031055ed4f210376d72825d831bbc343301b9f90cf3546c4b805d3280d0
`default_nettype none
module ntrpu_rope_hls_ports_r1 (
    input wire clk,rst_n,hls_start,
    output wire hls_done,hls_ready,
    output wire [31:0] hls_return,
    input wire [31:0] hls_heads,hls_position,hls_coefficient_position,
    output wire [1:0] hls_input_ce,
    output wire [11:0] hls_input_address [0:1],
    input wire [31:0] hls_input_q [0:1],
    output wire hls_output_ce,hls_output_we,
    output wire [11:0] hls_output_address,
    output wire [31:0] hls_output_d
);
    wire unused_idle;
    wire [5:0] coefficient_address;
    assign hls_input_address[1]={6'd0,coefficient_address};
    ntrpu_rope_table_fp32_r1 core (
        .ap_clk(clk),
        .ap_rst(~rst_n),
        .ap_start(hls_start),
        .ap_done(hls_done),
        .ap_ready(hls_ready),
        .ap_idle(unused_idle),
        .ap_return(hls_return),
        .heads(hls_heads),
        .position(hls_position),
        .coefficient_position(hls_coefficient_position),
        .input_r_address0(hls_input_address[0]),
        .input_r_ce0(hls_input_ce[0]),
        .input_r_q0(hls_input_q[0]),
        .coefficients_address0(coefficient_address),
        .coefficients_ce0(hls_input_ce[1]),
        .coefficients_q0(hls_input_q[1]),
        .output_r_address0(hls_output_address),
        .output_r_ce0(hls_output_ce),
        .output_r_we0(hls_output_we),
        .output_r_d0(hls_output_d)
    );
endmodule
`default_nettype wire
