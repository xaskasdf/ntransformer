`timescale 1ns/1ps
`default_nettype none
// Bundled-data CDC: the source holds its entire sample until the next request.
// All resets must share the canonical cold assertion, with local release sync.
// This RTL still requires physical CDC constraints/review at native integration.
module ntrpu_control_snapshot_tap (
    input wire src_clk,src_cold_rst_n,dst_clk,dst_cold_rst_n,request_token,
    input wire [127:0] src_data,
    output reg [127:0] sample=0,
    output reg done_token=0
);
    (* ASYNC_REG="TRUE" *) reg [2:0] request_sync=0,ack_sync=0;
    reg ack_token=0,settle=0;
    reg [127:0] held_data=0;
    always @(posedge src_clk or negedge src_cold_rst_n) begin
        if(!src_cold_rst_n) begin request_sync<=0;ack_token<=0;held_data<=0;end
        else begin
            request_sync<={request_sync[1:0],request_token};
            if(request_sync[2]!=ack_token) begin
                held_data<=src_data;
                ack_token<=request_sync[2];
            end
        end
    end
    always @(posedge dst_clk or negedge dst_cold_rst_n) begin
        if(!dst_cold_rst_n) begin ack_sync<=0;sample<=0;done_token<=0;settle<=0;end
        else begin
            ack_sync<={ack_sync[1:0],ack_token};
            if(ack_sync[2]!=done_token) begin
                if(settle) begin sample<=held_data;done_token<=ack_sync[2];settle<=0;end
                else settle<=1;
            end else settle<=0;
        end
    end
endmodule

module ntrpu_control_snapshot (
    input wire clk,cold_rst_n,request,release_sample,
    input wire [3:0] src_clk,src_cold_rst_n,
    input wire [511:0] src_data,
    output reg busy=0,valid=0,
    output reg [31:0] epoch=0,
    output reg [511:0] data=0,
    output wire [3:0] progress
);
    reg token=0;
    wire [3:0] done;
    wire [511:0] samples;
    assign progress=busy ? ~(done^{4{token}}) : 4'b0000;
    for(genvar i=0;i<4;i=i+1) begin: taps
        ntrpu_control_snapshot_tap tap(
            .src_clk(src_clk[i]),.src_cold_rst_n(src_cold_rst_n[i]),
            .dst_clk(clk),.dst_cold_rst_n(cold_rst_n),.request_token(token),
            .src_data(src_data[i*128+:128]),.sample(samples[i*128+:128]),.done_token(done[i]));
    end
    always @(posedge clk or negedge cold_rst_n) begin
        if(!cold_rst_n) begin token<=0;busy<=0;valid<=0;epoch<=0;data<=0;end
        else begin
            if(request&&!busy&&!valid) begin token<=~token;busy<=1;end
            if(busy&&(&progress)) begin data<=samples;epoch<=epoch+1;busy<=0;valid<=1;end
            if(release_sample&&valid) valid<=0;
        end
    end
endmodule
`default_nettype wire
