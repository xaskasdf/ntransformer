`timescale 1ns / 1ps
// Capture unchanged R12 beat sums; reconstruct its same-edge global candidate.
module ntrpu_pcie_inventory_sum_accumulator_bad_sign (
    input wire clk, cold_rst_n, enable,
    input wire signed [15:0] added_span,
    input wire [14:0] removed_sum,
    output wire signed [22:0] candidate
);
    reg [20:0] previous_low=0;
    reg signed [15:0] saved_added=0;
    reg [14:0] saved_removed=0;
    assign candidate=$signed({2'b0,previous_low})+$signed(saved_added)+$signed({1'b0,saved_removed});

    always @(posedge clk) begin
        if (!cold_rst_n) begin
            previous_low<=0; saved_added<=0; saved_removed<=0;
        end else if (enable) begin
            previous_low<=candidate[20:0];
            saved_added<=added_span; saved_removed<=removed_sum;
        end
    end
endmodule
