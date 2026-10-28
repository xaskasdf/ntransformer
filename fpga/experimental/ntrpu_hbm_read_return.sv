`timescale 1ns/1ps
// Two independent full-frame read reservations. Cold reset only.
// Success requires completion status zero AND a healthy system epoch.
module ntrpu_hbm_read_return (
    input wire clk,mem_clk,cold_reset_n,
    input wire [1:0] init_complete_async,
    input wire stop,fault_in,
    output wire rst_n,mem_rst_n,started,fault,drained,
    output reg stopped,
    input wire [1:0] cmd_valid,
    output wire [1:0] cmd_ready,
    input wire [127:0] cmd_addr,cmd_tag,
    input wire [15:0] cmd_len,
    output wire [1:0] data_valid,data_last,
    input wire [1:0] data_ready,
    output wire [1023:0] data,
    output wire [127:0] data_tag,
    output wire [1:0] cpl_valid,
    input wire [1:0] cpl_ready,
    output wire [127:0] cpl_tag,
    output wire [3:0] cpl_status,
    output wire [25:0] cpl_bytes,
    output wire [1:0] frame_idle,adapter_pending,adapter_fault,
    output wire [1:0] h_arvalid,
    input wire [1:0] h_arready,
    output wire [67:0] h_araddr,
    output wire [7:0] h_arlen,
    output wire [11:0] h_arid,
    output wire [5:0] h_arsize,
    output wire [3:0] h_arburst,
    input wire [1:0] h_rvalid,h_rlast,
    output wire [1:0] h_rready,
    input wire [511:0] h_rdata,
    input wire [11:0] h_rid,
    input wire [3:0] h_rresp
);
    wire startup_fault;
    wire [1:0] frame_fault;
    reg external_fault;
    wire admit=started&&!fault&&!stop&&!stopped;
    assign fault=startup_fault||external_fault||fault_in||(|frame_fault);
    assign drained=started&&(&frame_idle);
    ntrpu_hbm_startup startup(.clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_reset_n),
        .init_complete_async(init_complete_async),.adapter_fault_mem(adapter_fault),
        .src_rst_n(rst_n),.mem_rst_n(mem_rst_n),.started(started),.fault(startup_fault));
    always @(posedge clk)begin
        if(!rst_n)begin stopped<=0;external_fault<=0;end
        else begin
            if(stop)stopped<=1;
            if(fault_in)external_fault<=1;
        end
    end
    for(genvar lane=0;lane<2;lane++)begin: channels
        wire t_arvalid,t_arready,t_rvalid,t_rready,t_rlast;
        wire [33:0] t_araddr;
        wire [7:0] t_arlen;
        wire [511:0] t_rdata;
        wire [3:0] t_rid;
        wire [1:0] t_rresp;
        wire c_arvalid,c_arready,c_rvalid,c_rready,c_rlast;
        wire [33:0] c_araddr;
        wire [7:0] c_arlen;
        wire [511:0] c_rdata;
        wire [3:0] c_rid;
        wire [1:0] c_rresp;
        ntrpu_hbm_read_frame frame (
            .clk(clk),.rst_n(rst_n),.admit(admit),.cmd_valid(cmd_valid[lane]),.cmd_ready(cmd_ready[lane]),
            .cmd_addr(cmd_addr[lane*64+:64]),.cmd_tag(cmd_tag[lane*64+:64]),.cmd_len(cmd_len[lane*8+:8]),
            .data_valid(data_valid[lane]),.data_ready(data_ready[lane]),.data(data[lane*512+:512]),
            .data_tag(data_tag[lane*64+:64]),.data_last(data_last[lane]),
            .cpl_valid(cpl_valid[lane]),.cpl_ready(cpl_ready[lane]),.cpl_tag(cpl_tag[lane*64+:64]),
            .cpl_status(cpl_status[lane*2+:2]),.cpl_bytes(cpl_bytes[lane*13+:13]),.idle(frame_idle[lane]),.fault(frame_fault[lane]),
            .arvalid(t_arvalid),.arready(t_arready),.araddr(t_araddr),.arlen(t_arlen),
            .rvalid(t_rvalid),.rready(t_rready),.rdata(t_rdata),.rid(t_rid),.rresp(t_rresp),.rlast(t_rlast));
        hbm_read_cdc crossing (
            .s_axi_aclk(clk),.s_axi_aresetn(rst_n),.m_axi_aclk(mem_clk),.m_axi_aresetn(mem_rst_n),
            .s_axi_araddr(t_araddr),.s_axi_arlen(t_arlen),.s_axi_arsize(3'd6),.s_axi_arburst(2'b01),
            .s_axi_arid(4'b0),.s_axi_arlock(1'b0),.s_axi_arcache(4'b0010),.s_axi_arprot(3'b0),.s_axi_arregion(4'b0),.s_axi_arqos(4'b0),
            .s_axi_arvalid(t_arvalid),.s_axi_arready(t_arready),
            .s_axi_rdata(t_rdata),.s_axi_rid(t_rid),.s_axi_rresp(t_rresp),.s_axi_rlast(t_rlast),.s_axi_rvalid(t_rvalid),.s_axi_rready(t_rready),
            .m_axi_araddr(c_araddr),.m_axi_arlen(c_arlen),.m_axi_arsize(),.m_axi_arburst(),
            .m_axi_arid(),.m_axi_arlock(),.m_axi_arcache(),.m_axi_arprot(),.m_axi_arregion(),.m_axi_arqos(),
            .m_axi_arvalid(c_arvalid),.m_axi_arready(c_arready),
            .m_axi_rdata(c_rdata),.m_axi_rid(c_rid),.m_axi_rresp(c_rresp),.m_axi_rlast(c_rlast),.m_axi_rvalid(c_rvalid),.m_axi_rready(c_rready));
        ntrpu_hbm_read_adapter adapter (
            .clk(mem_clk),.rst_n(mem_rst_n),.s_arvalid(c_arvalid),.s_arready(c_arready),.s_araddr(c_araddr),.s_arlen(c_arlen),
            .s_rvalid(c_rvalid),.s_rready(c_rready),.s_rdata(c_rdata),.s_rid(c_rid),.s_rresp(c_rresp),.s_rlast(c_rlast),
            .h_arvalid(h_arvalid[lane]),.h_arready(h_arready[lane]),.h_araddr(h_araddr[lane*34+:34]),.h_arlen(h_arlen[lane*4+:4]),
            .h_arid(h_arid[lane*6+:6]),.h_arsize(h_arsize[lane*3+:3]),.h_arburst(h_arburst[lane*2+:2]),
            .h_rvalid(h_rvalid[lane]),.h_rready(h_rready[lane]),.h_rdata(h_rdata[lane*256+:256]),
            .h_rid(h_rid[lane*6+:6]),.h_rresp(h_rresp[lane*2+:2]),.h_rlast(h_rlast[lane]),
            .pending(adapter_pending[lane]),.fault(adapter_fault[lane]));
    end
endmodule
