`timescale 1ns / 1ps
// Passive AXI4-Lite ledger in ONE interface clock domain. Observe continuously
// from configuration reset, including traffic before the drain request.
// source_fenced is an independent contract: no new transactions can arrive.
// It must cover upstream queues/CDC; this monitor does not impose that fence.
// This does not certify AXI bursts, DMA internals, other interfaces or DDR.
module ntrpu_axil_drain_monitor #(
    parameter integer COUNT_WIDTH=4
) (
    input wire clk, cold_rst_n,
    input wire awvalid, awready, wvalid, wready, bvalid, bready,
    input wire arvalid, arready, rvalid, rready,
    input wire req_valid, source_fenced,
    input wire [63:0] req_tag,
    output wire ack_valid,
    output wire [63:0] ack_tag,
    output reg protocol_fault=0,
    output reg [COUNT_WIDTH-1:0] aw_pending=0, w_pending=0, ar_pending=0
);
    wire aw_fire=awvalid && awready, w_fire=wvalid && wready;
    wire b_fire=bvalid && bready, ar_fire=arvalid && arready;
    wire r_fire=rvalid && rready;
    wire any_valid=awvalid || wvalid || bvalid || arvalid || rvalid;
    wire empty=(aw_pending==0 && w_pending==0 && ar_pending==0);
    reg captured=0;
    reg [63:0] captured_tag=0;
    reg [1:0] quiet_count=0;
    assign ack_tag=captured_tag;
    // Fresh request + two quiet observations + a continuing external fence.
    // A new request cannot relabel an old high ACK, even before the next edge.
    assign ack_valid=cold_rst_n && captured && req_valid && req_tag==captured_tag &&
                     source_fenced && !protocol_fault && empty && !any_valid && quiet_count==2;

    function [COUNT_WIDTH-1:0] next_count(
        input [COUNT_WIDTH-1:0] value, input inc, input dec);
        begin
            case({inc,dec})
                2'b10: next_count=(&value) ? value : value+1'b1;
                2'b01: next_count=(value==0) ? value : value-1'b1;
                default: next_count=value;
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if(!cold_rst_n) begin
            aw_pending<=0;w_pending<=0;ar_pending<=0;protocol_fault<=0;
            captured<=0;captured_tag<=0;quiet_count<=0;
        end else begin
            aw_pending<=next_count(aw_pending,aw_fire,b_fire);
            w_pending<=next_count(w_pending,w_fire,b_fire);
            ar_pending<=next_count(ar_pending,ar_fire,r_fire);
            // Response acceptance needs both prior write halves, or a prior
            // read address. Unknown inventory and overflow fail closed forever.
            if((b_fire && (aw_pending==0 || w_pending==0)) ||
               (r_fire && ar_pending==0) ||
               (aw_fire && !b_fire && (&aw_pending)) ||
               (w_fire && !b_fire && (&w_pending)) ||
               (ar_fire && !r_fire && (&ar_pending))) protocol_fault<=1;
            if(!req_valid) begin captured<=0;quiet_count<=0;end
            else if(!captured || req_tag!=captured_tag) begin
                captured<=1;captured_tag<=req_tag;quiet_count<=0;
            end else if(!source_fenced || protocol_fault || !empty || any_valid) quiet_count<=0;
            else if(quiet_count<2) quiet_count<=quiet_count+1'b1;
        end
    end
    // synthesis translate_off
    initial if(COUNT_WIDTH<1 || COUNT_WIDTH>16) $fatal(1,"AXIL_DRAIN_PARAMETERS");
    // synthesis translate_on
endmodule
