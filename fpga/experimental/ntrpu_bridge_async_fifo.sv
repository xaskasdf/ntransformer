`timescale 1ns/1ps
`default_nettype none
// Experimental distributed-memory CDC FIFO. Coordinated cold reset only.
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
    reg [WIDTH-1:0] memory [0:(1<<LGDEPTH)-1];
    reg [LGDEPTH:0] wp, rp, wg, rg;
    (* ASYNC_REG="TRUE" *) reg [LGDEPTH:0] rs1,rs2,ws1,ws2;
    wire [LGDEPTH:0] full_compare=rs2 ^ ((1<<LGDEPTH)|(1<<(LGDEPTH-1)));
    assign wready=wrst_n && wg!=full_compare;
    assign rvalid=rrst_n && rg!=ws2;
    assign rdata=memory[rp[LGDEPTH-1:0]];
    wire [LGDEPTH:0] wn=wp+1'b1, rn=rp+1'b1;
    always @(posedge wclk or negedge wrst_n) begin
        if(!wrst_n)begin wp<=0;wg<=0;rs1<=0;rs2<=0;end
        else begin
            rs1<=rg;rs2<=rs1;
            if(wvalid && wready)begin
                memory[wp[LGDEPTH-1:0]]<=wdata;
                wp<=wn;wg<=(wn>>1)^wn;
            end
        end
    end
    always @(posedge rclk or negedge rrst_n) begin
        if(!rrst_n)begin rp<=0;rg<=0;ws1<=0;ws2<=0;end
        else begin
            ws1<=wg;ws2<=ws1;
            if(rvalid && rready)begin rp<=rn;rg<=(rn>>1)^rn;end
        end
    end
endmodule
`default_nettype wire
