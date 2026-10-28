`timescale 1ns/1ps
`default_nettype none
// Admission/stop guard, NOT an operational reset actuator. mgmt_clk must be
// independent and continuous. Only configuration may assert cold_rst_n.
module ntrpu_axi_bridge_guard #(
    parameter integer CLOCK_TIMEOUT=64,
    parameter integer QUALIFY_CYCLES=4
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
    wire native_ok=native_link_up && native_axi_reset_n && native_ctl_reset_n && perst_n;
    // Cold reset releases separately in each domain. No operational input
    // can reset a client register or clear a latched fault.
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] cold_m=0, cold_a=0;
    always @(posedge mgmt_clk or negedge cold_rst_n)
        if(!cold_rst_n)cold_m<=0;else cold_m<={cold_m[1:0],1'b1};
    always @(posedge axi_clk or negedge cold_rst_n)
        if(!cold_rst_n)cold_a<=0;else cold_a<={cold_a[1:0],1'b1};
    assign client_cold_rst_n=cold_a[2];

    // Held request/echo handshake, not a free-running toggle: an arbitrary
    // ratio of clocks cannot alias away all progress. Probe survives PERST.
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

    // A low native level closes immediately and remains visible for >=3
    // management edges after recovery. This catches short reset/link pulses
    // in RTL; physical minimum pulse widths and recovery/removal need STA.
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] healthy_m=0;
    always @(posedge mgmt_clk or negedge native_ok)
        if(!native_ok)healthy_m<=0;else healthy_m<={healthy_m[1:0],1'b1};
    reg [QW-1:0] qualified=0;
    always @(posedge mgmt_clk)begin
        if(!cold_m[2])begin
            armed<=0;fault<=0;fault_code<=0;qualified<=0;
        end else if(armed)begin
            if(!healthy_m[2] || !clock_alive)begin
                fault<=1;
                fault_code<=fault_code | {!clock_alive,!healthy_m[2]};
            end
        end else if(healthy_m[2] && clock_alive)begin
            if(qualified==QUALIFY_CYCLES-1)armed<=1;
            else qualified<=qualified+1'b1;
        end else qualified<=0;
    end

    // Close asynchronously even with axi_clk stopped; reopening is local and
    // synchronous. fault is sticky, so operational recovery never reopens it.
    wire allow=armed && !fault && native_ok && cold_rst_n && cold_m[2];
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] admit_a=0;
    always @(posedge axi_clk or negedge allow)
        if(!allow)admit_a<=0;else admit_a<={admit_a[1:0],1'b1};
    assign client_link_up=admit_a[2];
    // Unlike !admit_a, this stays low during startup; asserting stop before
    // initial qualification would permanently fault the existing writer.
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] stop_a=0;
    always @(posedge axi_clk or posedge fault)
        if(fault)stop_a<=3'b111;else stop_a<={stop_a[1:0],1'b0};
    assign client_stop=stop_a[2];
    // synthesis translate_off
    initial if(CLOCK_TIMEOUT<16 || QUALIFY_CYCLES<3)
        $fatal(1,"BRIDGE_GUARD_PARAMETERS");
    // synthesis translate_on
endmodule
`default_nettype wire
