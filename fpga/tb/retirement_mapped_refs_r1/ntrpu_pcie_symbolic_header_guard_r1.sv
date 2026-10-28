`timescale 1ns / 1ps
// Preserve R7's fifteen-bit temporal and mixed-sign header predicates.
module ntrpu_pcie_symbolic_header_guard_r1_retirement_mapped_reference_r1 (
    input wire [12:0] coriginal,
    input wire [11:0] caddress,
    input wire [10:0] count,
    input wire normal, co, terminal,
    output reg signed [14:0] delivered,
    output reg length_fault, terminal_fault
);
    reg signed [14:0] available;
    always @* begin
        available=0; delivered=0; length_fault=0; terminal_fault=0;
        if (normal && co) begin
            available=count*4-caddress[1:0];
            delivered=(available<coriginal) ? available : coriginal;
            if (count<1 || count>1024 || delivered<1 ||
                count!=((caddress[1:0]+delivered+3)/4)) length_fault=1;
            if (terminal!==(delivered==coriginal)) terminal_fault=1;
        end
    end
endmodule
