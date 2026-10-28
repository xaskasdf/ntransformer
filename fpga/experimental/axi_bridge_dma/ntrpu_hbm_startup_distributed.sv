`timescale 1ns/1ps
// Isolated candidate: one ordinary distribution stage after the canonical
// three-stage reset synchronizer. No fault or calibration signal resets IP.
module ntrpu_hbm_startup_distributed (
    input wire clk,mem_clk,cold_reset_n,
    input wire [1:0] init_complete_async,adapter_fault_mem,
    output wire src_rst_n,mem_rst_n,
    output reg started,fault
);
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] source_release,memory_release;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] ready_meta,ready_sync,ready_final;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [1:0] fault_meta,fault_sync,fault_final;
    reg [3:0] qualification;
    // Deliberately not ASYNC_REG: implementation may replicate this stage.
    reg memory_distribution;
    assign src_rst_n=source_release[2];
    assign mem_rst_n=memory_distribution;
    always @(posedge clk or negedge cold_reset_n)begin
        if(!cold_reset_n)source_release<=0;
        else source_release<={source_release[1:0],1'b1};
    end
    always @(posedge mem_clk or negedge cold_reset_n)begin
        if(!cold_reset_n)memory_release<=0;
        else memory_release<={memory_release[1:0],1'b1};
    end
    // Asserts even with stopped clocks; releases on the fourth memory edge.
    // Recovery/removal from memory_release[2] must remain timed in integration.
    always @(posedge mem_clk or negedge memory_release[2])begin
        if(!memory_release[2])memory_distribution<=0;
        else memory_distribution<=1'b1;
    end
    always @(posedge clk or negedge src_rst_n)begin
        if(!src_rst_n)begin
            ready_meta<=0;ready_sync<=0;ready_final<=0;
            fault_meta<=0;fault_sync<=0;fault_final<=0;
        end else begin
            ready_meta<={mem_rst_n,init_complete_async};ready_sync<=ready_meta;ready_final<=ready_sync;
            fault_meta<=adapter_fault_mem;fault_sync<=fault_meta;fault_final<=fault_sync;
        end
    end
    always @(posedge clk)begin
        if(!src_rst_n)begin started<=0;fault<=0;qualification<=0;end
        else begin
            if((|fault_final)||(started&&!( &ready_final)))fault<=1;
            if(!started)begin
                if((&ready_final)&&!(|fault_final)&&!fault)begin
                    if(qualification==7)started<=1;
                    else qualification<=qualification+1'b1;
                end else qualification<=0;
            end
        end
    end
endmodule
