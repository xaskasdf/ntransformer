// Vitis schedules header prefetches before deciding whether state is used.
// First-tile speculative reads see zero and cannot expose an older bank.
// Invalid addresses still reach the context's existing BAD_MEMORY check.
`default_nettype none
module ntrpu_attention_state_read_guard_r1 (
    input wire [31:0] tile_start,
    input wire core_ce,
    input wire [6:0] core_address,
    input wire [63:0] context_q,
    output wire context_ce,
    output wire [6:0] context_address,
    output wire [63:0] core_q
);
    assign context_ce=core_ce && (tile_start!=0 || core_address>=104);
    assign context_address=core_address;
    assign core_q=tile_start==0 ? 64'd0:context_q;
endmodule
`default_nettype wire
