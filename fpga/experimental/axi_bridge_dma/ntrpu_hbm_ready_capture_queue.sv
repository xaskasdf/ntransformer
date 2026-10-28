`timescale 1ns/1ps
// Free-slot capture with a combinational next-occupancy indication.
// next_space is consumed by a READY register, never driven directly to HBM.
module ntrpu_hbm_ready_capture_queue #(parameter integer WIDTH=265) (
    input wire clk,rst_n,
    input wire s_valid,
    output wire s_ready,
    input wire [WIDTH-1:0] s_data,
    output wire m_valid,
    input wire m_ready,
    output wire [WIDTH-1:0] m_data,
    output wire empty,
    output wire next_space
);
    reg [1:0] count;
    reg write_slot,read_slot;
    reg [WIDTH-1:0] slot0,slot1;
    wire push=s_valid&&s_ready;
    wire pop=m_valid&&m_ready;
    wire [1:0] next_count=(push&&!pop)?count+2'd1:
                          (!push&&pop)?count-2'd1:count;
    assign next_space=next_count<2;
    assign s_ready=rst_n&&count<2;
    assign m_valid=rst_n&&count!=0;
    assign m_data=read_slot?slot1:slot0;
    assign empty=count==0;
    always @(posedge clk)begin
        if(!rst_n)begin count<=0;write_slot<=0;read_slot<=0;end
        else begin
            count<=next_count;
            if(s_ready)begin
                if(write_slot)slot1<=s_data;
                else slot0<=s_data;
            end
            if(push)write_slot<=!write_slot;
            if(pop)read_slot<=!read_slot;
        end
    end
endmodule
