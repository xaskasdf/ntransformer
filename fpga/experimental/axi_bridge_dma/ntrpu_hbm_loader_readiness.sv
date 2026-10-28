`timescale 1ns/1ps
`default_nettype none
// Inputs are synchronous to management clk. Export one registered CDC source.
// This adds one management cycle; it is not an asynchronous availability kill.
module ntrpu_hbm_loader_readiness (
    input wire clk, cold_rst_n,
    input wire [1:0] ready_mgmt,
    input wire fault_mgmt,
    output reg ready = 0
);
    always @(posedge clk or negedge cold_rst_n) begin
        if (!cold_rst_n) ready <= 0;
        else ready <= (&ready_mgmt) && !fault_mgmt;
    end
endmodule
`default_nettype wire
