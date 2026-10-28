`timescale 1ns/1ps
// Isolated candidate: only the unowned write slot may be captured without VALID.
// Pointer/count ownership still changes solely on accepted transfers.
module ntrpu_hbm_response_capture_queue #(parameter integer WIDTH=265) (
    input wire clk,rst_n,
    input wire s_valid,
    output wire s_ready,
    input wire [WIDTH-1:0] s_data,
    output wire m_valid,
    input wire m_ready,
    output wire [WIDTH-1:0] m_data,
    output wire empty
);
    reg [1:0] count;
    reg write_slot,read_slot;
    reg [WIDTH-1:0] slot0,slot1;
    wire push=s_valid&&s_ready;
    wire pop=m_valid&&m_ready;
    assign s_ready=rst_n&&count<2;
    assign m_valid=rst_n&&count!=0;
    assign m_data=read_slot?slot1:slot0;
    assign empty=count==0;
    always @(posedge clk)begin
        if(!rst_n)begin count<=0;write_slot<=0;read_slot<=0;end
        else begin
            case({push,pop})
                2'b10:count<=count+1'b1;
                2'b01:count<=count-1'b1;
                default:begin end
            endcase
            if(s_ready)begin
                if(write_slot)slot1<=s_data;
                else slot0<=s_data;
            end
            if(push)write_slot<=!write_slot;
            if(pop)read_slot<=!read_slot;
        end
    end
endmodule
