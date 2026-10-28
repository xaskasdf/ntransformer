`timescale 1ns / 1ps
// Four independent registered RC arithmetic proposals. Not a protocol monitor.
// Descriptors occupy DWORD positions 0,4,8,12; header_valid is supplied by framing.
// The captured descriptor is received data, never an expected identity/state.
// Non-normal descriptors are retained but do not produce an arithmetic proposal.
module ntrpu_pcie_rc_proposal_reference (
    input wire clk, cold_rst_n,
    input wire [3:0] header_valid,
    input wire [511:0] header_data,
    output reg [3:0] header_seen=0, proposal_valid=0,
    output reg [383:0] captured_headers=0,
    output reg [51:0] delivered_bytes=0, proposed_remaining=0,
    output reg [47:0] proposed_address=0,
    output reg [3:0] length_error=0, terminal_error=0
);
    wire [3:0] normal;
    wire [51:0] delivered_next, remaining_next;
    wire [47:0] address_next;
    wire [3:0] length_next, terminal_next;
    genvar lane;
    generate for (lane=0;lane<4;lane=lane+1) begin: arithmetic
        wire [95:0] h=header_data[lane*128+:96];
        wire [12:0] declared_bytes=h[28:16];
        wire [11:0] declared_address=h[11:0];
        wire [10:0] count=h[42:32];
        // R7's mixed signed/unsigned comparison uses an unsigned 15-bit span.
        // Explicit wrap preserves malformed count=0 and unaligned-address cases.
        wire [14:0] available={2'b0,count,2'b0}-{13'b0,declared_address[1:0]};
        wire [12:0] delivered=(available<{2'b0,declared_bytes}) ? available[12:0] : declared_bytes;
        wire [13:0] covered={1'b0,delivered}+{12'b0,declared_address[1:0]}+14'd3;
        assign normal[lane]=((h[15:12]==0 && h[45:43]==0 && h[46]==0)===1'b1);
        assign delivered_next[lane*13+:13]=delivered;
        assign remaining_next[lane*13+:13]=declared_bytes-delivered;
        assign address_next[lane*12+:12]=declared_address+delivered;
        assign length_next[lane]=(count<1 || count>1024 || delivered<1 || {1'b0,count}!=covered[13:2]);
        assign terminal_next[lane]=(h[30]!=(delivered==declared_bytes));
    end endgenerate

    integer q;
    always @(posedge clk) begin
        header_seen<=0; proposal_valid<=0; captured_headers<=0;
        delivered_bytes<=0; proposed_remaining<=0; proposed_address<=0;
        length_error<=0; terminal_error<=0;
        if (cold_rst_n) begin
            for (q=0;q<4;q=q+1) begin
                if (header_valid[q]) begin
                    header_seen[q]<=1;
                    captured_headers[q*96+:96]<=header_data[q*128+:96];
                    if (normal[q]) begin
                        proposal_valid[q]<=1;
                        delivered_bytes[q*13+:13]<=delivered_next[q*13+:13];
                        proposed_remaining[q*13+:13]<=remaining_next[q*13+:13];
                        proposed_address[q*12+:12]<=address_next[q*12+:12];
                        length_error[q]<=length_next[q];
                        terminal_error[q]<=terminal_next[q];
                    end
                end
            end
        end
    end
endmodule
