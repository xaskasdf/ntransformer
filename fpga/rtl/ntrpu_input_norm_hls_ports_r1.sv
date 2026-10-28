// Inspected generated core SHA256: 4cd28ac3fb50177206527ec8cc40907b82f2efa583efd80e9d46bd22c597820a
`default_nettype none
module ntrpu_input_norm_hls_ports_r1 (
    input wire clk,rst_n,hls_start,
    output wire hls_done,hls_ready,
    output wire [31:0] hls_return,
    input wire [31:0] hls_elements,hls_epsilon_bits,
    output wire [1:0] hls_input_ce,
    output wire [10:0] hls_input_address [0:1],
    input wire [31:0] hls_input_q [0:1],
    output wire hls_output_ce,hls_output_we,
    output wire [8:0] hls_output_address,
    output wire [63:0] hls_output_d
);
    wire unused_idle;
    ntrpu_norm_q8_abi1_r1 core (
        .ap_clk(clk),
        .ap_rst(~rst_n),
        .ap_start(hls_start),
        .ap_done(hls_done),
        .ap_ready(hls_ready),
        .ap_idle(unused_idle),
        .ap_return(hls_return),
        .elements(hls_elements),
        .epsilon_bits(hls_epsilon_bits),
        .input_r_address0(hls_input_address[0]),
        .input_r_ce0(hls_input_ce[0]),
        .input_r_q0(hls_input_q[0]),
        .weight_address0(hls_input_address[1]),
        .weight_ce0(hls_input_ce[1]),
        .weight_q0(hls_input_q[1]),
        .output_r_address0(hls_output_address),
        .output_r_ce0(hls_output_ce),
        .output_r_we0(hls_output_we),
        .output_r_d0(hls_output_d)
    );
endmodule
`default_nettype wire
