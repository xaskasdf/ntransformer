`timescale 1ns / 1ps

// Proposed NTRPUF01 profile: one already paired, full 64-byte AW/W frame.
// Header: magic, epoch, tag, fragment offset (four little-endian u64 words).
// Payload: 32 bytes, placed at offset 0 or 32 within an authorized line.
// Pure decoder only: caller owns AW/W capture, drain/B, lease authority,
// assembly, replay policy, completions and reset. Not yet an XDMA endpoint.
module ntrpu_pcie_fragment_decode (
    input wire lease_valid,
    input wire [63:0] lease_window, lease_epoch, lease_tag,
    input wire [63:0] awaddr,
    input wire [7:0] awlen,
    input wire [2:0] awsize,
    input wire [1:0] awburst,
    input wire awlock,
    input wire [511:0] wdata,
    input wire [63:0] wstrb,
    input wire wlast,
    output wire accept,
    output wire [6:0] faults, // lease, format, window, magic, epoch, tag, offset
    output wire [63:0] byte_valid,
    output wire [511:0] byte_data
);
    localparam [63:0] MAGIC = 64'h313046555052544e; // ASCII NTRPUF01
    wire high_half = wdata[255:192] == 64'd32;
    assign faults[0] = !lease_valid || lease_window[5:0] != 0;
    assign faults[1] = awlen != 0 || awsize != 6 || awburst != 1 ||
                       awlock || awaddr[5:0] != 0 || !wlast || wstrb != ~64'b0;
    assign faults[2] = awaddr != lease_window;
    assign faults[3] = wdata[63:0] != MAGIC;
    assign faults[4] = wdata[127:64] != lease_epoch;
    assign faults[5] = wdata[191:128] != lease_tag;
    assign faults[6] = wdata[255:192] != 0 && !high_half;
    assign accept = faults == 0;
    wire [63:0] selected_bytes = high_half ? 64'hffffffff00000000 : 64'h00000000ffffffff;
    assign byte_valid = accept ? selected_bytes : 64'b0;
    assign byte_data = high_half ? {wdata[511:256],256'b0} : {256'b0,wdata[511:256]};
endmodule
