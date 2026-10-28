`timescale 1ns/1ps
`default_nettype none
// V3: one release chain per distinct physical source, including native reset.
// SHARED_NATIVE_RESET=1 requires verified identical AXI/control reset sources.
// Management must remain independent and continuous. Not a reset actuator.
module ntrpu_axi_bridge_guard_v3 #(
    parameter integer CLOCK_TIMEOUT=64,
    parameter integer QUALIFY_CYCLES=4,
    parameter integer SHARED_NATIVE_RESET=0
) (
    input wire mgmt_clk, axi_clk, cold_rst_n,
    input wire native_link_up, native_axi_reset_n, native_ctl_reset_n, perst_n,
    output wire client_cold_rst_n, client_link_up, client_stop,
    output reg clock_alive=0,
    output reg armed=0,
    output reg fault=0,
    output reg [1:0] fault_code=0
);
    localparam integer TW=$clog2(CLOCK_TIMEOUT+1);
    localparam integer QW=$clog2(QUALIFY_CYCLES+1);
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] cold_m=0, cold_a=0;
    always @(posedge mgmt_clk or negedge cold_rst_n)
        if(!cold_rst_n)cold_m<=0;else cold_m<={cold_m[1:0],1'b1};
    always @(posedge axi_clk or negedge cold_rst_n)
        if(!cold_rst_n)cold_a<=0;else cold_a<={cold_a[1:0],1'b1};
    assign client_cold_rst_n=cold_a[2];

    reg request=1;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] echo_a=0, echo_m=0;
    reg [TW-1:0] age=0;
    always @(posedge axi_clk)echo_a<={echo_a[1:0],request};
    always @(posedge mgmt_clk)begin
        echo_m<={echo_m[1:0],echo_a[2]};
        if(echo_m[2]==request)begin
            request<=!request;age<=0;clock_alive<=1;
        end else if(age==CLOCK_TIMEOUT-1)clock_alive<=0;
        else age<=age+1'b1;
    end

    localparam integer STATUS_COUNT=SHARED_NATIVE_RESET ? 3 : 4;
    wire [STATUS_COUNT-1:0] native_status;
    generate if(SHARED_NATIVE_RESET)begin:shared_reset
        assign native_status={perst_n,native_axi_reset_n,native_link_up};
    end else begin:distinct_resets
        assign native_status={perst_n,native_ctl_reset_n,native_axi_reset_n,native_link_up};
    end endgenerate
    wire [STATUS_COUNT-1:0] healthy_m,healthy_a;
    for(genvar i=0;i<STATUS_COUNT;i++)begin:availability
        (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] release_m=0,release_a=0;
        always @(posedge mgmt_clk or negedge native_status[i])
            if(!native_status[i])release_m<=0;else release_m<={release_m[1:0],1'b1};
        // One register output drives this asynchronous control, not an AND of
        // status inputs. Management observes a drop before AXI can requalify.
        always @(posedge axi_clk or negedge release_m[2])
            if(!release_m[2])release_a<=0;else release_a<={release_a[1:0],1'b1};
        assign healthy_m[i]=release_m[2];
        assign healthy_a[i]=release_a[2];
    end
    wire all_healthy_m=&healthy_m;
    reg [QW-1:0] qualified=0;
    always @(posedge mgmt_clk)begin
        if(!cold_m[2])begin
            armed<=0;fault<=0;fault_code<=0;qualified<=0;
        end else if(armed)begin
            if(!all_healthy_m || !clock_alive)begin
                fault<=1;
                fault_code<=fault_code | {!clock_alive,!all_healthy_m};
            end
        end else if(all_healthy_m && clock_alive)begin
            if(qualified==QUALIFY_CYCLES-1)armed<=1;
            else qualified<=qualified+1'b1;
        end else qualified<=0;
    end

    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] armed_a=0;
    // Reuse the one AXI-domain cold-reset release; do not independently
    // resynchronize the same external reset for this admission chain.
    always @(posedge axi_clk or negedge cold_a[2])
        if(!cold_a[2])armed_a<=0;else armed_a<={armed_a[1:0],armed};
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] stop_a=0;
    always @(posedge axi_clk or posedge fault)
        if(fault)stop_a<=3'b111;else stop_a<={stop_a[1:0],1'b0};
    assign client_stop=stop_a[2];
    // This conjunction drives the availability port, never a FF CLR/PRE.
    // All permits release on AXI; a native loss or sticky fault closes without AXI.
    assign client_link_up=cold_a[2] && armed_a[2] && (&healthy_a) && !stop_a[2];
    // synthesis translate_off
    // A shared profile is a wiring contract, never an inference from momentary
    // equality. Reject distinct known signals in simulation; native integration
    // separately proves that both IP outputs have the same physical driver.
    always @(native_axi_reset_n or native_ctl_reset_n)begin
        #0; // Let continuous-assignment aliases settle within this time step.
        if(SHARED_NATIVE_RESET &&
           ((native_axi_reset_n===1'b0 && native_ctl_reset_n===1'b1) ||
            (native_axi_reset_n===1'b1 && native_ctl_reset_n===1'b0)))
            $fatal(1,"V3_SHARED_RESET_CONTRACT");
    end
    initial if(CLOCK_TIMEOUT<16 || QUALIFY_CYCLES<3 ||
               (SHARED_NATIVE_RESET!=0 && SHARED_NATIVE_RESET!=1))
        $fatal(1,"BRIDGE_GUARD_PARAMETERS");
    // synthesis translate_on
endmodule
`default_nettype wire
