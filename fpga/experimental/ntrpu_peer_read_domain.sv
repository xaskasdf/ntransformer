`timescale 1ns/1ps
// Keep at most two reader commands from one physical pseudchannel in flight.
// Different-domain commands wait for local completion, not remote DMA/ACK.
module ntrpu_peer_read_domain (
    input wire clk,rst_n,
    input wire s_valid,
    output wire s_ready,
    input wire [63:0] address,
    output wire m_valid,
    input wire m_ready,
    input wire complete,
    output wire pending,
    output reg fault
);
    reg [1:0] count;
    reg [4:0] domain;
    wire allowed=(count==0 || domain==address[33:29]) && count<2;
    wire accepted=m_valid&&m_ready;
    // Fault propagation stops the owning reader. Do not cancel an offered
    // command here: VALID/address must remain stable under backpressure.
    assign m_valid=s_valid&&allowed;
    assign s_ready=m_ready&&allowed;
    assign pending=count!=0;
    always @(posedge clk)begin
        if(!rst_n)begin count<=0;domain<=0;fault<=0;end
        else begin
            if(complete && count==0)fault<=1;
            else begin
                case({accepted,complete})
                    2'b10:count<=count+1'b1;
                    2'b01:count<=count-1'b1;
                    default:count<=count;
                endcase
            end
            if(accepted && count==0)domain<=address[33:29];
        end
    end
endmodule
