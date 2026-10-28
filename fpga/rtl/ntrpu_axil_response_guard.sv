`timescale 1ns / 1ps
// Host-clock response boundary: manager -> requests -> CDC, CDC -> guard ->
// manager for responses. Observe requests continuously from configuration.
// AXI-Lite has no IDs: this proves prior request credits, NOT response identity
// or correctness of a subordinate that returns a wrong but credited response.
// Only configuration initialization may clear an orphan or unknown inventory.
module ntrpu_axil_response_guard #(
    parameter integer COUNT_WIDTH=8
) (
    input wire clk,cold_rst_n,
    input wire awvalid,awready,wvalid,wready,arvalid,arready,
    input wire c_bvalid,c_rvalid,
    output wire c_bready,c_rready,
    input wire [1:0] c_bresp,c_rresp,
    input wire [31:0] c_rdata,
    output wire h_bvalid,h_rvalid,
    input wire h_bready,h_rready,
    output wire [1:0] h_bresp,h_rresp,
    output wire [31:0] h_rdata,
    output wire [2:0] fault_sources,
    output reg [COUNT_WIDTH-1:0] aw_pending=0,w_pending=0,ar_pending=0
);
    reg orphan_b=0,orphan_r=0,inventory_fault=0;
    wire write_credit=(aw_pending!=0 && w_pending!=0);
    wire read_credit=(ar_pending!=0);
    wire allow_b=write_credit && !orphan_b;
    wire allow_r=read_credit && !orphan_r;
    wire aw_fire=awvalid && awready,w_fire=wvalid && wready;
    wire ar_fire=arvalid && arready;
    wire b_fire=h_bvalid && h_bready,r_fire=h_rvalid && h_rready;
    // A valid response remains offered while stalled: its credits cannot be
    // retired until this handshake. An orphan on the OTHER channel does not
    // gate it. Unknown responses are held below the guard, never acknowledged
    // or reclassified when later AW/W/AR arrive during closure's finite prefix.
    assign h_bvalid=c_bvalid && allow_b;
    assign c_bready=h_bready && allow_b;
    assign h_rvalid=c_rvalid && allow_r;
    assign c_rready=h_rready && allow_r;
    assign h_bresp=c_bresp;
    assign h_rresp=c_rresp;
    assign h_rdata=c_rdata;
    // Export individual registers, not a combinational OR, to CDC consumers.
    assign fault_sources={inventory_fault,orphan_r,orphan_b};

    function [COUNT_WIDTH-1:0] next_count(
        input [COUNT_WIDTH-1:0] value,input inc,input dec);
        case({inc,dec})
            2'b10: next_count=(&value) ? value : value+1'b1;
            2'b01: next_count=(value==0) ? value : value-1'b1;
            default: next_count=value;
        endcase
    endfunction

    always @(posedge clk)begin
        if(!cold_rst_n)begin
            aw_pending<=0;w_pending<=0;ar_pending<=0;
            orphan_b<=0;orphan_r<=0;inventory_fault<=0;
        end else begin
            aw_pending<=next_count(aw_pending,aw_fire,b_fire);
            w_pending<=next_count(w_pending,w_fire,b_fire);
            ar_pending<=next_count(ar_pending,ar_fire,r_fire);
            if(c_bvalid && !write_credit)orphan_b<=1;
            if(c_rvalid && !read_credit)orphan_r<=1;
            // Saturation never silently wraps. With the composed 32-credit
            // admission cap and eight-bit counters overflow is unreachable
            // for conforming traffic. The retained fault forbids certification
            // if that contract is violated; existing offers are not truncated.
            if((aw_fire && !b_fire && (&aw_pending)) ||
               (w_fire && !b_fire && (&w_pending)) ||
               (ar_fire && !r_fire && (&ar_pending)))inventory_fault<=1;
        end
    end
    // synthesis translate_off
    initial if(COUNT_WIDTH<1 || COUNT_WIDTH>16)$fatal(1,"AXIL_RESPONSE_GUARD_PARAMETERS");
    // synthesis translate_on
endmodule
