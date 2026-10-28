// Inspected generated core SHA256: 087cb2c4277bd2ac85400a9cf5238e0e9e46b065220b5b294c9b43b891b04223
`default_nettype none
module ntrpu_residual_hls_ports_r1 (
    input wire clk,rst_n,hls_start,
    output wire hls_done,hls_ready,
    output wire [31:0] hls_return,
    input wire [31:0] hls_elements,
    output wire [1:0] hls_input_ce,
    output wire [10:0] hls_input_address [0:1],
    input wire [31:0] hls_input_q [0:1],
    output wire hls_residual_ce,hls_residual_we,
    output wire [10:0] hls_residual_address,
    output wire [31:0] hls_residual_d
);
    wire unused_idle;
    ntrpu_residual_fp32_r1 core (
        .ap_clk(clk),
        .ap_rst(~rst_n),
        .ap_start(hls_start),
        .ap_done(hls_done),
        .ap_ready(hls_ready),
        .ap_idle(unused_idle),
        .ap_return(hls_return),
        .elements(hls_elements),
        .residual_address0(hls_input_address[0]),
        .residual_ce0(hls_input_ce[0]),
        .residual_q0(hls_input_q[0]),
        .branch_address0(hls_input_address[1]),
        .branch_ce0(hls_input_ce[1]),
        .branch_q0(hls_input_q[1]),
        .output_r_address0(hls_residual_address),
        .output_r_ce0(hls_residual_ce),
        .output_r_we0(hls_residual_we),
        .output_r_d0(hls_residual_d)
    );
endmodule
`default_nettype wire
