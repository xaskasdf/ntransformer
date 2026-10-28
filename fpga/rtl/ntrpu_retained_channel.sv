`timescale 1ns / 1ps
// Single-clock reset boundary. cold_rst_n belongs to the retained lifetime.
// boundary_rst_n resets ONLY the facing AXI interface, never this inventory.
// Requests already accepted remain valid toward the retained consumer.
// Responses may be abandoned only after loss AND the facing interface reset.
// delivered counts facing-interface handshakes, not end-to-end host receipts.
module ntrpu_retained_channel #(
    parameter integer WIDTH=8,
    parameter integer TO_BOUNDARY=0,
    parameter integer COUNT_WIDTH=64
) (
    input wire clk, cold_rst_n, boundary_rst_n, loss,
    input wire s_valid,
    output wire s_ready,
    input wire [WIDTH-1:0] s_data,
    output wire m_valid,
    input wire m_ready,
    output wire [WIDTH-1:0] m_data,
    output wire invalid,
    output reg fenced,
    output reg [1:0] pending,
    output reg [COUNT_WIDTH-1:0] accepted, delivered, abandoned,
    output reg [1:0] fault
);
    reg dead, seen_up;
    reg [WIDTH-1:0] first, second;
    assign invalid=loss || dead || (|fault);
    wire discard=TO_BOUNDARY && (fenced || (invalid && !boundary_rst_n));
    assign s_ready=cold_rst_n && (TO_BOUNDARY ? (discard || pending<2) :
                                  boundary_rst_n && !invalid && pending<2);
    assign m_valid=cold_rst_n && pending!=0 &&
                   (!TO_BOUNDARY || (boundary_rst_n && !discard));
    assign m_data=first;
    wire push=s_valid && s_ready;
    wire pop=m_valid && m_ready;
    wire [2:0] drops=discard ? {1'b0,pending}+{2'b0,push} : 3'd0;
    wire [COUNT_WIDTH:0] add_accepted={1'b0,accepted}+push;
    wire [COUNT_WIDTH:0] add_delivered={1'b0,delivered}+pop;
    wire [COUNT_WIDTH:0] add_abandoned={1'b0,abandoned}+drops;
    always @(posedge clk) begin
        if(!cold_rst_n) begin
            dead<=0;seen_up<=0;fenced<=0;pending<=0;
            accepted<=0;delivered<=0;abandoned<=0;fault<=0;
            first<=0;second<=0;
        end else begin
            if(boundary_rst_n) seen_up<=1;
            if(invalid) dead<=1;
            // A runtime reset without prior invalidation is a contract fault.
            if(seen_up && !boundary_rst_n && !invalid) fault[0]<=1;
            if(discard) fenced<=1;
            if(add_accepted[COUNT_WIDTH] || add_delivered[COUNT_WIDTH] || add_abandoned[COUNT_WIDTH])
                fault[1]<=1;
            accepted<=add_accepted[COUNT_WIDTH] ? {COUNT_WIDTH{1'b1}} : add_accepted[COUNT_WIDTH-1:0];
            delivered<=add_delivered[COUNT_WIDTH] ? {COUNT_WIDTH{1'b1}} : add_delivered[COUNT_WIDTH-1:0];
            abandoned<=add_abandoned[COUNT_WIDTH] ? {COUNT_WIDTH{1'b1}} : add_abandoned[COUNT_WIDTH-1:0];
            if(discard) pending<=0;
            else case({push,pop})
                2'b10:begin
                    pending<=pending+1'b1;
                    if(pending==0) first<=s_data;else second<=s_data;
                end
                2'b01:begin pending<=pending-1'b1;if(pending==2) first<=second;end
                // s_ready is registered-state based: simultaneous push/pop
                // occurs only at occupancy one. No READY combinational path.
                2'b11:first<=s_data;
                default:begin end
            endcase
        end
    end
    // synthesis translate_off
    initial if(WIDTH<1 || COUNT_WIDTH<2 || (TO_BOUNDARY!=0 && TO_BOUNDARY!=1))
        $fatal(1,"RETAINED_CHANNEL_PARAMETERS");
    // synthesis translate_on
endmodule
