`timescale 1ns/1ps
`default_nettype none
// Transport only: reset inputs are the canonical, coordinated cold-reset
// releases for each domain. Operational stop/fault never clears FIFO ownership.
// The selected reader must hold each VALID/payload until READY, including CPL.
module ntrpu_axi_bridge_hbm_cdc #(
    parameter integer CLOCK_TIMEOUT=64
) (
    input wire p_clk,r_clk,mgmt_clk,p_cold_rst_n,r_cold_rst_n,m_cold_rst_n,
    input wire p_stop,r_started,r_fault,
    output wire p_started,p_fault,r_stop,
    output reg clock_alive=0,armed=0,management_fault=0,
    input wire [1:0] p_cmd_valid,
    output wire [1:0] p_cmd_ready,
    input wire [127:0] p_cmd_addr,p_cmd_tag,
    input wire [15:0] p_cmd_len,
    output wire [1:0] p_data_valid,p_data_last,
    input wire [1:0] p_data_ready,
    output wire [1023:0] p_data,
    output wire [127:0] p_data_tag,
    output wire [1:0] p_cpl_valid,
    input wire [1:0] p_cpl_ready,
    output wire [127:0] p_cpl_tag,
    output wire [3:0] p_cpl_status,
    output wire [25:0] p_cpl_bytes,
    output wire [1:0] r_cmd_valid,
    input wire [1:0] r_cmd_ready,
    output wire [127:0] r_cmd_addr,r_cmd_tag,
    output wire [15:0] r_cmd_len,
    input wire [1:0] r_data_valid,r_data_last,
    output wire [1:0] r_data_ready,
    input wire [1023:0] r_data,
    input wire [127:0] r_data_tag,
    input wire [1:0] r_cpl_valid,
    output wire [1:0] r_cpl_ready,
    input wire [127:0] r_cpl_tag,
    input wire [3:0] r_cpl_status,
    input wire [25:0] r_cpl_bytes
);
    // A level request/echo cannot alias a stopped clock, unlike free-running
    // toggles sampled at an unfortunate frequency ratio. Management stays on.
    localparam integer TW=$clog2(CLOCK_TIMEOUT+1);
    reg request=1;
    reg [TW-1:0] age=0;
    reg [2:0] qualified=0;
    reg seen_started=0,reader_failed=0,stop_held=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] echo_r=0,echo_m=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] started_m=0,failed_m=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] armed_p=0,failed_p=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] stop_r=0,failed_r=0;
    always @(posedge r_clk or negedge r_cold_rst_n)begin
        if(!r_cold_rst_n)begin
            echo_r<=0;seen_started<=0;reader_failed<=0;stop_r<=0;failed_r<=0;
        end else begin
            echo_r<={echo_r[1:0],request};
            stop_r<={stop_r[1:0],stop_held};
            failed_r<={failed_r[1:0],management_fault};
            if(r_started)seen_started<=1;
            if(r_fault||(seen_started&&!r_started))reader_failed<=1;
        end
    end
    always @(posedge mgmt_clk or negedge m_cold_rst_n)begin
        if(!m_cold_rst_n)begin
            request<=1;age<=0;echo_m<=0;started_m<=0;failed_m<=0;
            clock_alive<=0;qualified<=0;armed<=0;management_fault<=0;
        end else begin
            echo_m<={echo_m[1:0],echo_r[2]};
            started_m<={started_m[1:0],r_started};
            failed_m<={failed_m[1:0],reader_failed};
            if(echo_m[2]==request)begin request<=!request;age<=0;clock_alive<=1;end
            else if(age==CLOCK_TIMEOUT-1)clock_alive<=0;
            else age<=age+1'b1;
            if(failed_m[2]||(armed&&(!clock_alive||!started_m[2])))management_fault<=1;
            if(!armed)begin
                if(clock_alive&&started_m[2]&&!failed_m[2])begin
                    if(qualified==3)armed<=1;else qualified<=qualified+1'b1;
                end else qualified<=0;
            end
        end
    end
    always @(posedge p_clk or negedge p_cold_rst_n)begin
        if(!p_cold_rst_n)begin armed_p<=0;failed_p<=0;stop_held<=0;end
        else begin
            armed_p<={armed_p[1:0],armed};
            failed_p<={failed_p[1:0],management_fault};
            if(p_stop)stop_held<=1;
        end
    end
    assign p_fault=failed_p[2]||stop_held;
    assign p_started=p_cold_rst_n&&armed_p[2]&&!p_fault;
    assign r_stop=stop_r[2]||failed_r[2]||reader_failed;

    for(genvar lane=0;lane<2;lane++)begin:lanes
        wire [135:0] command;
        ntrpu_bridge_async_fifo #(.WIDTH(136)) commands(
            .wclk(p_clk),.wrst_n(p_cold_rst_n),.wvalid(p_cmd_valid[lane]),
            .wready(p_cmd_ready[lane]),
            .wdata({p_cmd_addr[lane*64+:64],p_cmd_tag[lane*64+:64],p_cmd_len[lane*8+:8]}),
            .rclk(r_clk),.rrst_n(r_cold_rst_n),.rvalid(r_cmd_valid[lane]),
            .rready(r_cmd_ready[lane]),.rdata(command));
        assign {r_cmd_addr[lane*64+:64],r_cmd_tag[lane*64+:64],r_cmd_len[lane*8+:8]}=command;
        // One ordered stream for data AND CPL. Even with stalled PCIe, CPL
        // cannot become visible until all earlier data have been consumed.
        wire event_ready,event_valid;
        wire [592:0] event_out;
        wire choose_cpl=!r_data_valid[lane];
        wire [592:0] event_in=choose_cpl ?
            {1'b1,1'b0,r_cpl_tag[lane*64+:64],r_cpl_status[lane*2+:2],r_cpl_bytes[lane*13+:13],512'b0}:
            {1'b0,r_data_last[lane],r_data_tag[lane*64+:64],2'b0,13'b0,r_data[lane*512+:512]};
        ntrpu_bridge_async_fifo #(.WIDTH(593)) returns(
            .wclk(r_clk),.wrst_n(r_cold_rst_n),.wvalid(r_data_valid[lane]||r_cpl_valid[lane]),
            .wready(event_ready),.wdata(event_in),.rclk(p_clk),.rrst_n(p_cold_rst_n),
            .rvalid(event_valid),.rready(event_out[592]?p_cpl_ready[lane]:p_data_ready[lane]),.rdata(event_out));
        assign r_data_ready[lane]=event_ready;
        assign r_cpl_ready[lane]=event_ready&&choose_cpl;
        assign p_data_valid[lane]=event_valid&&!event_out[592];
        assign p_cpl_valid[lane]=event_valid&&event_out[592];
        assign p_data[lane*512+:512]=event_out[511:0];
        assign p_data_last[lane]=event_out[591];
        assign p_data_tag[lane*64+:64]=event_out[590:527];
        assign p_cpl_tag[lane*64+:64]=event_out[590:527];
        assign p_cpl_status[lane*2+:2]=event_out[526:525];
        assign p_cpl_bytes[lane*13+:13]=event_out[524:512];
    end
    // synthesis translate_off
    initial if(CLOCK_TIMEOUT<16)$fatal(1,"HBM_CDC_TIMEOUT_PARAMETER");
    // synthesis translate_on
endmodule
`default_nettype wire
