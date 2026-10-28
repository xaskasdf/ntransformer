`timescale 1ns/1ps
`default_nettype none
// APB completion is synchronous to mgmt_clk. LOCKED is asynchronous.
// A sampled loss after arming survives a halted reader and a later relock.
module ntrpu_hbm_native_availability (
    input wire mgmt_clk,cold_rst_n,
    input wire fabric_locked_async,
    input wire [1:0] apb_complete_mgmt,
    output reg [1:0] ready_mgmt,
    output reg armed,fault
);
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] lock_sync;
    always @(posedge mgmt_clk or negedge cold_rst_n) begin
        if(!cold_rst_n)lock_sync<=0;
        else lock_sync<={lock_sync[1:0],fabric_locked_async};
    end
    always @(posedge mgmt_clk or negedge cold_rst_n) begin
        if(!cold_rst_n)begin ready_mgmt<=0;armed<=0;fault<=0;end
        else begin
            if(armed && (!lock_sync[2] || !(&apb_complete_mgmt)))begin
                fault<=1;ready_mgmt<=0;
            end else if(fault)ready_mgmt<=0;
            else begin
                ready_mgmt<=apb_complete_mgmt & {2{lock_sync[2]}};
                if(lock_sync[2] && (&apb_complete_mgmt))armed<=1;
            end
        end
    end
endmodule
`default_nettype wire
