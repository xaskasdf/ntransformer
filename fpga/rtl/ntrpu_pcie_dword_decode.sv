`timescale 1ns / 1ps

// Pure decoder for one ALREADY PAIRED AW/W, single-DWORD bypass write.
// The caller owns AW/W capture, rejected-burst drain, B, lease lifecycle,
// epoch isolation and capacity authorization. This is not an AXI endpoint.
// A reservation is exactly one aligned, preauthorized 64-byte line.
module ntrpu_pcie_dword_decode (
    input  wire         lease_valid,
    input  wire [63:0]  lease_line,
    input  wire [63:0]  awaddr,
    input  wire [7:0]   awlen,
    input  wire [2:0]   awsize,
    input  wire [1:0]   awburst,
    input  wire         awlock,
    input  wire [511:0] wdata,
    input  wire [63:0]  wstrb,
    input  wire         wlast,
    output wire         accept,
    output wire [3:0]   faults, // bit 0 lease, 1 format, 2 line, 3 mask
    output wire [63:0]  line_address,
    output wire [63:0]  byte_valid,
    output wire [511:0] byte_data
);
    wire [63:0] selected_line = {awaddr[63:6], 6'b0};
    wire [63:0] legal_lanes;
    for (genvar lane = 0; lane < 64; lane = lane + 1) begin : lanes
        // An unaligned first beat ends at the ALIGNED DWORD boundary.
        // Do not shift strobes/data by AWADDR again: lanes are bus-relative.
        assign legal_lanes[lane] = (lane / 4 == awaddr[5:2]) &&
                                  (lane >= awaddr[5:0]);
    end
    assign faults[0] = !lease_valid || lease_line[5:0] != 0;
    assign faults[1] = awlen != 0 || awsize != 2 || awburst != 1 || awlock || !wlast;
    assign faults[2] = selected_line != lease_line;
    assign faults[3] = wstrb == 0 || (wstrb & ~legal_lanes) != 0;
    assign accept = faults == 0;
    assign line_address = selected_line;
    assign byte_valid = accept ? wstrb : 64'b0;
    assign byte_data = wdata;
endmodule
