`timescale 1ns/1ps
`default_nettype none
// BAR register aperture: aligned 1-DW INCR accesses, AXI address 0..4095.
// The wide PCIe bridge bus carries that DW in the addressed byte lane.
// Unsupported bursts drain their declared beats and cannot reach the CSR slave.
module ntrpu_bar32_adapter (
    input wire clk,rst_n,
    input wire [63:0] s_awaddr,s_araddr,
    input wire [3:0] s_awid,s_arid,
    input wire [7:0] s_awlen,s_arlen,
    input wire [2:0] s_awsize,s_arsize,s_awprot,s_arprot,
    input wire [1:0] s_awburst,s_arburst,
    input wire s_awlock,s_arlock,s_awvalid,s_wvalid,s_wlast,s_bready,s_arvalid,s_rready,
    input wire [511:0] s_wdata,
    input wire [63:0] s_wstrb,
    output wire s_awready,s_wready,s_arready,
    output reg [3:0] s_bid=0,s_rid=0,
    output reg [1:0] s_bresp=0,s_rresp=0,
    output reg s_bvalid=0,s_rvalid=0,s_rlast=0,
    output reg [511:0] s_rdata=0,
    output reg [11:0] m_awaddr=0,m_araddr=0,
    output reg [2:0] m_awprot=0,m_arprot=0,
    output reg [31:0] m_wdata=0,
    output reg [3:0] m_wstrb=0,
    output reg m_awvalid=0,m_wvalid=0,m_arvalid=0,
    input wire m_awready,m_wready,m_arready,m_bvalid,m_rvalid,
    input wire [1:0] m_bresp,m_rresp,
    input wire [31:0] m_rdata,
    output wire m_bready,m_rready
);
    localparam [1:0] COLLECT=0,OFFER=1,RESPONSE=2,RETURN=3;
    reg [1:0] w_state=COLLECT,r_state=COLLECT;
    reg aw_full=0,w_full=0,aw_bad=0,w_last=0;
    reg [7:0] w_length=0,w_index=0,r_left=0;
    reg [3:0] w_lane=0,r_lane=0;
    reg [511:0] w_payload=0;
    reg [63:0] w_strobes=0;
    wire [63:0] lane_mask=64'hf<<(w_lane*4);
    assign s_awready=rst_n&&w_state==COLLECT&&!aw_full;
    assign s_wready=rst_n&&w_state==COLLECT&&!w_full;
    assign s_arready=rst_n&&r_state==COLLECT;
    assign m_bready=rst_n&&w_state==RESPONSE;
    assign m_rready=rst_n&&r_state==RESPONSE;

    function automatic invalid(input [63:0] address,input [7:0] length,
        input [2:0] size,input [1:0] burst,input locked);
        invalid=address[63:12]!=0||address[1:0]!=0||length!=0||size!=2||burst!=1||locked;
    endfunction
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            w_state<=COLLECT;r_state<=COLLECT;aw_full<=0;w_full<=0;aw_bad<=0;w_last<=0;
            w_length<=0;w_index<=0;r_left<=0;w_lane<=0;r_lane<=0;w_payload<=0;w_strobes<=0;
            s_bid<=0;s_rid<=0;s_bresp<=0;s_rresp<=0;s_bvalid<=0;s_rvalid<=0;s_rlast<=0;s_rdata<=0;
            m_awaddr<=0;m_araddr<=0;m_awprot<=0;m_arprot<=0;m_wdata<=0;m_wstrb<=0;
            m_awvalid<=0;m_wvalid<=0;m_arvalid<=0;
        end else begin
            if(s_awvalid&&s_awready) begin
                aw_full<=1;aw_bad<=invalid(s_awaddr,s_awlen,s_awsize,s_awburst,s_awlock);
                w_length<=s_awlen;w_index<=0;w_lane<=s_awaddr[5:2];s_bid<=s_awid;
                m_awaddr<=s_awaddr[11:0];m_awprot<=s_awprot;
            end
            if(s_wvalid&&s_wready) begin w_full<=1;w_payload<=s_wdata;w_strobes<=s_wstrb;w_last<=s_wlast;end
            case(w_state)
                COLLECT: if(aw_full&&w_full) begin
                    w_full<=0;
                    if(aw_bad) begin
                        if(w_index==w_length) begin s_bvalid<=1;s_bresp<=3;w_state<=RETURN;end
                        else w_index<=w_index+1;
                    end else if(!w_last||((w_strobes&~lane_mask)!=0)) begin
                        s_bvalid<=1;s_bresp<=2;w_state<=RETURN;
                    end else begin
                        m_wdata<=w_payload[w_lane*32+:32];m_wstrb<=w_strobes[w_lane*4+:4];
                        m_awvalid<=1;m_wvalid<=1;w_state<=OFFER;
                    end
                end
                OFFER: begin
                    if(m_awvalid&&m_awready)m_awvalid<=0;
                    if(m_wvalid&&m_wready)m_wvalid<=0;
                    if((!m_awvalid||m_awready)&&(!m_wvalid||m_wready))w_state<=RESPONSE;
                end
                RESPONSE: if(m_bvalid) begin s_bresp<=m_bresp;s_bvalid<=1;w_state<=RETURN;end
                RETURN: if(s_bvalid&&s_bready) begin s_bvalid<=0;aw_full<=0;w_state<=COLLECT;end
            endcase
            if(s_arvalid&&s_arready) begin
                s_rid<=s_arid;r_left<=s_arlen;r_lane<=s_araddr[5:2];
                m_araddr<=s_araddr[11:0];m_arprot<=s_arprot;
                if(invalid(s_araddr,s_arlen,s_arsize,s_arburst,s_arlock)) begin
                    s_rdata<=0;s_rresp<=3;s_rvalid<=1;s_rlast<=s_arlen==0;r_state<=RETURN;
                end else begin m_arvalid<=1;r_state<=OFFER;end
            end
            case(r_state)
                OFFER: if(m_arvalid&&m_arready) begin m_arvalid<=0;r_state<=RESPONSE;end
                RESPONSE: if(m_rvalid) begin
                    s_rdata<=512'(m_rdata)<<(r_lane*32);s_rresp<=m_rresp;s_rvalid<=1;s_rlast<=1;r_state<=RETURN;
                end
                RETURN: if(s_rvalid&&s_rready) begin
                    if(r_left==0) begin s_rvalid<=0;s_rlast<=0;r_state<=COLLECT;end
                    else begin r_left<=r_left-1;s_rlast<=r_left==1;end
                end
                default: begin end
            endcase
        end
    end
endmodule
`default_nettype wire
