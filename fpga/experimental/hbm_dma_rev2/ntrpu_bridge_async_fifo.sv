`timescale 1ns/1ps
`default_nettype none
// Revision 2: 4-entry RAM plus one registered receive-domain output slot.
// Physical CDC constraints and implementation qualification remain required.
module ntrpu_bridge_async_fifo #(
    parameter integer WIDTH=8, LGDEPTH=2
) (
    input wire wclk, wrst_n, wvalid,
    output wire wready,
    input wire [WIDTH-1:0] wdata,
    input wire rclk, rrst_n,
    output wire rvalid,
    input wire rready,
    output wire [WIDTH-1:0] rdata
);
    (* ram_style="distributed" *) reg [WIDTH-1:0] memory [0:(1<<LGDEPTH)-1];
    reg [LGDEPTH:0] wp, rp, wg, rg;
    (* ASYNC_REG="TRUE" *) reg [LGDEPTH:0] rs1,rs2,ws1,ws2;
    wire [LGDEPTH:0] full_compare=rs2 ^ ((1<<LGDEPTH)|(1<<(LGDEPTH-1)));
    assign wready=wrst_n && wg!=full_compare;
    // Prefetch transfers ownership to the output slot. The returned Gray
    // pointer frees the RAM location only after its data was captured here.
    (* KEEP="TRUE" *) reg [WIDTH-1:0] output_data;
    reg output_valid;
    wire ram_nonempty=rg!=ws2;
    wire replace_output=!output_valid || rready;
    wire capture=rrst_n && replace_output && ram_nonempty;
    assign rvalid=rrst_n && output_valid;
    assign rdata=output_data;
    // No data reset; output_valid isolates stale storage after cold reset.
    always @(posedge rclk)begin
        if(capture)output_data<=memory[rp[LGDEPTH-1:0]];
    end
    wire [LGDEPTH:0] wn=wp+1'b1, rn=rp+1'b1;
    // RAM contents are intentionally not reset. Validity belongs to the pointers.
    always @(posedge wclk) begin
        if(wvalid && wready)memory[wp[LGDEPTH-1:0]]<=wdata;
    end
    always @(posedge wclk or negedge wrst_n) begin
        if(!wrst_n)begin wp<=0;wg<=0;rs1<=0;rs2<=0;end
        else begin
            rs1<=rg;rs2<=rs1;
            if(wvalid && wready)begin
                wp<=wn;wg<=(wn>>1)^wn;
            end
        end
    end
    always @(posedge rclk or negedge rrst_n) begin
        if(!rrst_n)begin rp<=0;rg<=0;ws1<=0;ws2<=0;output_valid<=0;end
        else begin
            ws1<=wg;ws2<=ws1;
            if(replace_output)output_valid<=ram_nonempty;
            if(capture)begin rp<=rn;rg<=(rn>>1)^rn;end
        end
    end
endmodule
`default_nettype wire
