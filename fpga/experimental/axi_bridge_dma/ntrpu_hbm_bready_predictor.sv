`timescale 1ns/1ps
// Predict the existing response state; ownership remains in the loader FSM.
module ntrpu_hbm_bready_predictor(
    input wire offer,response,lane,aw_pending,w_pending,
    input wire [1:0] awready,wready,bvalid,
    output wire [1:0] next_ready
);
    wire enter_response=offer&&(!aw_pending||awready[lane])&&(!w_pending||wready[lane]);
    wire remain_response=response&&!bvalid[lane];
    assign next_ready=(enter_response||remain_response)?(lane?2'b10:2'b01):2'b00;
endmodule
