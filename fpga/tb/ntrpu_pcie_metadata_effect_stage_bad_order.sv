`timescale 1ns / 1ps
// One registered composition/application step; no table or protocol state.
module ntrpu_pcie_metadata_effect_stage_bad_order (
    input wire clk, cold_rst_n, valid_in,
    input wire [56:0] earlier, later,
    input wire [29:0] seed,
    output reg valid_out=0,
    output reg [56:0] effect_out=0,
    output reg [29:0] metadata_out=0
);
    import ntrpu_pcie_metadata_effect_pkg::*;
    wire [56:0] combined=compose_effect(later,earlier);
    wire [29:0] applied=apply_effect(combined,seed);
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            valid_out<=0; effect_out<=0; metadata_out<=0;
        end else begin
            valid_out<=valid_in;
            if (valid_in) begin
                effect_out<=combined;
                metadata_out<=applied;
            end
        end
    end
endmodule
