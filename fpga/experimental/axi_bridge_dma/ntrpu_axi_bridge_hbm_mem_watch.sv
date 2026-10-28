`timescale 1ns/1ps
`default_nettype none
// Canonical cold releases are inputs. No operational event resets ownership.
// Independent, continuously running management clock is required.
module ntrpu_axi_bridge_hbm_mem_watch #(
    parameter integer CLOCK_TIMEOUT=64
) (
    input wire mgmt_clk,mem_clk,p_clk,r_clk,
    input wire m_cold_rst_n,mem_cold_rst_n,p_cold_rst_n,r_cold_rst_n,
    output reg clock_alive=0,armed=0,fault=0,
    output wire p_permit,p_fault,r_fault
);
    localparam integer TW=$clog2(CLOCK_TIMEOUT+1);
    reg request=1;
    reg [TW-1:0] age=0;
    reg [2:0] qualified=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] echo_mem=0,echo_m=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] armed_p=0,failed_p=0,failed_r=0;
    always @(posedge mem_clk or negedge mem_cold_rst_n)
        if(!mem_cold_rst_n)echo_mem<=0;else echo_mem<={echo_mem[1:0],request};
    always @(posedge mgmt_clk or negedge m_cold_rst_n)begin
        if(!m_cold_rst_n)begin
            request<=1;age<=0;qualified<=0;echo_m<=0;
            clock_alive<=0;armed<=0;fault<=0;
        end else begin
            echo_m<={echo_m[1:0],echo_mem[2]};
            if(echo_m[2]==request)begin request<=!request;age<=0;clock_alive<=1;end
            else if(age==CLOCK_TIMEOUT-1)clock_alive<=0;
            else age<=age+1'b1;
            if(armed&&!clock_alive)fault<=1;
            if(!armed)begin
                if(clock_alive)begin
                    if(qualified==3)armed<=1;else qualified<=qualified+1'b1;
                end else qualified<=0;
            end
        end
    end
    always @(posedge p_clk or negedge p_cold_rst_n)begin
        if(!p_cold_rst_n)begin armed_p<=0;failed_p<=0;end
        else begin armed_p<={armed_p[1:0],armed};failed_p<={failed_p[1:0],fault};end
    end
    always @(posedge r_clk or negedge r_cold_rst_n)
        if(!r_cold_rst_n)failed_r<=0;else failed_r<={failed_r[1:0],fault};
    assign p_permit=p_cold_rst_n&&armed_p[2]&&!failed_p[2];
    assign p_fault=failed_p[2];
    assign r_fault=failed_r[2];
    // synthesis translate_off
    initial if(CLOCK_TIMEOUT<16)$fatal(1,"HBM_MEM_WATCH_PARAMETER");
    // synthesis translate_on
endmodule
`default_nettype wire
