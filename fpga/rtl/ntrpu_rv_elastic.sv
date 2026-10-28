`timescale 1ns / 1ps

// Two-entry, single-clock ready/valid queue. No combinational path crosses
// the boundary in either direction. Payload is meaningful only with VALID.
module ntrpu_rv_elastic #(parameter integer WIDTH = 8) (
    input wire clk, rst_n,
    input wire s_valid,
    output wire s_ready,
    input wire [WIDTH-1:0] s_data,
    output wire m_valid,
    input wire m_ready,
    output wire [WIDTH-1:0] m_data,
    output wire empty
);
    reg [1:0] count;
    reg [WIDTH-1:0] first, second;
    wire push = s_valid && s_ready;
    wire pop = m_valid && m_ready;
    assign s_ready = rst_n && count < 2;
    assign m_valid = rst_n && count != 0;
    assign m_data = first;
    assign empty = count == 0;

    always @(posedge clk) begin
        if (!rst_n) begin
            count <= 0;
        end else begin
            case ({push, pop})
                2'b10: begin
                    count <= count + 1'b1;
                    if (count == 0) first <= s_data;
                    else second <= s_data;
                end
                2'b01: begin
                    count <= count - 1'b1;
                    if (count == 2) first <= second;
                end
                2'b11: first <= s_data;
                default: begin end
            endcase
        end
    end
endmodule
