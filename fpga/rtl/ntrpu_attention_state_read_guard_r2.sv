// The pinned Vitis core prefetches the initial header and continuation[104].
// Suppress only those documented reads; other invalid addresses still fault.
// Register sentinel suppression to preserve word 103 across address advance.
`default_nettype none
module ntrpu_attention_state_read_guard_r2 (
    input wire clk, rst_n,
    input wire [31:0] tile_start,
    input wire core_ce,
    input wire [6:0] core_address,
    input wire [63:0] context_q,
    output wire context_ce,
    output wire [6:0] context_address,
    output wire [63:0] core_q,
    output wire discarded_request
);
    reg discarded_response;
    assign discarded_request=core_ce && tile_start!=0 && core_address==104;
    assign context_ce=core_ce && !discarded_request && (tile_start!=0 || core_address>=104);
    assign context_address=core_address;
    assign core_q=(tile_start==0 || discarded_response) ? 64'd0:context_q;
    always @(posedge clk)begin
        if(!rst_n)discarded_response<=0;
        else discarded_response<=discarded_request;
    end
endmodule
`default_nettype wire
