// The inspected Vitis copy loop issues a one-past-end speculative read.
// Drop that exact sentinel for short tiles; other bad addresses still fault.
// Track the discarded request across the RAM edge, so the final valid word
// is not zeroed when the core advances its address combinationally.
`default_nettype none
module ntrpu_attention_kv_read_guard_r1 (
    input wire clk, rst_n,
    input wire [31:0] tile_count,
    input wire core_ce,
    input wire [11:0] core_address,
    input wire [31:0] context_q,
    output wire context_ce,
    output wire [11:0] context_address,
    output wire [31:0] core_q,
    output wire discarded_request
);
    reg discarded_response;
    wire short_tile=tile_count>0 && tile_count<64;
    assign discarded_request=core_ce && short_tile && {1'b0,core_address}=={tile_count[6:0],6'd0};
    assign context_ce=core_ce && !discarded_request;
    assign context_address=core_address;
    assign core_q=discarded_response ? 32'd0:context_q;
    always @(posedge clk)begin
        if(!rst_n)discarded_response<=0;
        else discarded_response<=discarded_request;
    end
endmodule
`default_nettype wire
