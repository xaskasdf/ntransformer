// Generated only after exact inspection of the supplied Vitis module.
// Core SHA256: bcfe59a688ad701aa5f18642e9f076affc99d6f40877fd7b54019a9ae38efc48
`default_nettype none
module ntrpu_attention_hls_ports_r1 (
    input wire clk,
    input wire rst_n,
    input wire hls_start,
    output wire hls_done,
    output wire hls_ready,
    output wire [31:0] hls_return,
    input wire [31:0] hls_tile_start,
    input wire [31:0] hls_tile_count,
    input wire [31:0] hls_query_position,
    input wire [31:0] hls_layer,
    input wire [31:0] hls_query_head,
    input wire [31:0] hls_kv_head,
    input wire [63:0] hls_sequence,
    input wire [63:0] hls_epoch,
    output wire [5:0] hls_query_address,
    output wire hls_query_ce,
    input wire [31:0] hls_query_q,
    output wire [11:0] hls_keys_address,
    output wire hls_keys_ce,
    input wire [31:0] hls_keys_q,
    output wire [11:0] hls_values_address,
    output wire hls_values_ce,
    input wire [31:0] hls_values_q,
    output wire [6:0] hls_state_in_address,
    output wire hls_state_in_ce,
    input wire [63:0] hls_state_in_q,
    output wire [6:0] hls_state_out_address,
    output wire hls_state_out_ce,
    output wire hls_state_out_we,
    output wire [63:0] hls_state_out_d,
    output wire [5:0] hls_output_address,
    output wire hls_output_ce,
    output wire hls_output_we,
    output wire [31:0] hls_output_d
);
    wire raw_keys_ce;
    wire [11:0] raw_keys_address;
    wire [31:0] guarded_keys_q;
    wire discarded_keys;
    ntrpu_attention_kv_read_guard_r1 keys_read_guard (
        .clk(clk), .rst_n(rst_n), .tile_count(hls_tile_count),
        .core_ce(raw_keys_ce), .core_address(raw_keys_address), .context_q(hls_keys_q),
        .context_ce(hls_keys_ce), .context_address(hls_keys_address),
        .core_q(guarded_keys_q), .discarded_request(discarded_keys)
    );
    wire raw_values_ce;
    wire [11:0] raw_values_address;
    wire [31:0] guarded_values_q;
    wire discarded_values;
    ntrpu_attention_kv_read_guard_r1 values_read_guard (
        .clk(clk), .rst_n(rst_n), .tile_count(hls_tile_count),
        .core_ce(raw_values_ce), .core_address(raw_values_address), .context_q(hls_values_q),
        .context_ce(hls_values_ce), .context_address(hls_values_address),
        .core_q(guarded_values_q), .discarded_request(discarded_values)
    );
    wire unused_idle;
    wire raw_state_in_ce;
    wire [6:0] raw_state_in_address;
    wire [63:0] guarded_state_in_q;
    ntrpu_attention_state_read_guard_r1 state_read_guard (
        .tile_start(hls_tile_start), .core_ce(raw_state_in_ce),
        .core_address(raw_state_in_address), .context_q(hls_state_in_q),
        .context_ce(hls_state_in_ce), .context_address(hls_state_in_address),
        .core_q(guarded_state_in_q)
    );
    ntrpu_attention_tile_fp32_r1 core (
        .ap_clk(clk),
        .ap_rst(~rst_n),
        .ap_start(hls_start),
        .ap_done(hls_done),
        .ap_ready(hls_ready),
        .ap_idle(unused_idle),
        .ap_return(hls_return),
        .tile_start(hls_tile_start),
        .tile_count(hls_tile_count),
        .query_position(hls_query_position),
        .layer(hls_layer),
        .query_head(hls_query_head),
        .kv_head(hls_kv_head),
        .sequence_r(hls_sequence),
        .epoch(hls_epoch),
        .query_address0(hls_query_address),
        .query_ce0(hls_query_ce),
        .query_q0(hls_query_q),
        .keys_address0(raw_keys_address),
        .keys_ce0(raw_keys_ce),
        .keys_q0(guarded_keys_q),
        .values_address0(raw_values_address),
        .values_ce0(raw_values_ce),
        .values_q0(guarded_values_q),
        .state_input_address0(raw_state_in_address),
        .state_input_ce0(raw_state_in_ce),
        .state_input_q0(guarded_state_in_q),
        .state_output_address0(hls_state_out_address),
        .state_output_ce0(hls_state_out_ce),
        .state_output_we0(hls_state_out_we),
        .state_output_d0(hls_state_out_d),
        .output_r_address0(hls_output_address),
        .output_r_ce0(hls_output_ce),
        .output_r_we0(hls_output_we),
        .output_r_d0(hls_output_d)
    );
endmodule
`default_nettype wire
