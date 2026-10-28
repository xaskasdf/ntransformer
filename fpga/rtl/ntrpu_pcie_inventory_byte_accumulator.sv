`timescale 1ns / 1ps
// Retimed global byte total. Event inputs are captured at edge N; the candidate
// after that edge describes N, for the existing public register at edge N+1.
module ntrpu_pcie_inventory_byte_accumulator (
    input wire clk, cold_rst_n, enable,
    input wire signed [15:0] added_span,
    input wire [51:0] removed_spans,
    output wire signed [22:0] candidate
);
    reg [20:0] previous_low=0;
    reg signed [15:0] saved_added=0;
    reg [51:0] saved_removed=0;
    wire [14:0] removed_sum=
        ({2'b0,saved_removed[12:0]}+{2'b0,saved_removed[25:13]})+
        ({2'b0,saved_removed[38:26]}+{2'b0,saved_removed[51:39]});
    assign candidate=$signed({2'b0,previous_low})+$signed(saved_added)-$signed({1'b0,removed_sum});

    always @(posedge clk) begin
        if (!cold_rst_n) begin
            previous_low<=0; saved_added<=0; saved_removed<=0;
        end else if (enable) begin
            // Deliberately discard high bits between beats, including after a
            // fault. Retaining a wider base would alter the original machine.
            previous_low<=candidate[20:0];
            saved_added<=added_span; saved_removed<=removed_spans;
        end
    end
endmodule
