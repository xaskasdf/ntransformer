`timescale 1ns/1ps
// Experimental one-source, one-lease admission over two fixed HBM channels.
// The reference pair and ledgers remain unchanged; this is not a PCIe ABI.
module ntrpu_peer_read_shared (
    input wire clk,mem_clk,cold_reset_n,
    input wire [1:0] init_complete_async,
    input wire [1:0] stop,link_up,request_valid,reject_ready,
    input wire [1:0] dma_ready,dma_done_valid,ack_valid,result_ready,
    input wire cfg_valid,revoke,
    input wire [1155:0] cfg,
    input wire [2047:0] request,
    input wire [1159:0] dma_done,ack,
    output wire rst_n,mem_rst_n,started,reader_fault,reader_drained,
    output wire cfg_ready,cfg_error,admission_halted,
    output wire [63:0] admission_watermark,
    output wire [1:0] admission_pending,request_ready,reject_valid,
    output wire [7:0] reject_code,
    output wire [383:0] reject_identity,
    output wire [1:0] dma_valid,dma_last,result_valid,halted,quarantined,
    output wire [1023:0] dma_data,
    output wire [2047:0] dma_context,result_context,
    output wire [127:0] dma_address,result_bytes,
    output wire [7:0] result_status,
    output wire [3:0] contexts_held,
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
    wire [1:0] cmd_valid,cmd_ready,data_valid,data_ready,data_last,cpl_valid,cpl_ready;
    wire [127:0] cmd_addr,cmd_tag,data_tag,cpl_tag;
    wire [15:0] cmd_len;
    wire [1023:0] data;
    wire [3:0] cpl_status;
    wire [25:0] cpl_bytes;
    wire [1:0] lane_cfg_ready,lane_cfg_error,lane_request_valid,lane_request_ready;
    wire [1:0] lane_reject_valid,lane_halted,lane_quarantined;
    wire [3:0] lane_contexts;
    wire [2047:0] lane_request;
    wire guard_cfg_ready,guard_cfg_error;
    wire source_stop=|stop || admission_halted || |lane_halted || |lane_reject_valid;
    wire source_fault=reader_fault || |lane_reject_valid;
    wire cfg_fire=cfg_valid && cfg_ready;
    assign cfg_ready=guard_cfg_ready && (&lane_cfg_ready);
    assign cfg_error=guard_cfg_error || |lane_cfg_error;
    assign halted=lane_halted | {2{admission_halted}};
    assign quarantined=lane_quarantined | {admission_halted && |contexts_held[3:2],admission_halted && |contexts_held[1:0]};
    ntrpu_peer_read_admission guard(
        .clk(clk),.rst_n(rst_n),.enable(started && !cfg_valid),.stop(source_stop),
        .link_up(&link_up),.fault_in(source_fault),.revoke(revoke),
        .cfg_valid(cfg_fire),.cfg_ready(guard_cfg_ready),.cfg(cfg),.cfg_error(guard_cfg_error),
        .request_valid(request_valid),.request_ready(request_ready),.request(request),
        .reject_valid(reject_valid),.reject_ready(reject_ready),.reject_code(reject_code),.reject_identity(reject_identity),
        .admit_valid(lane_request_valid),.admit_ready(lane_request_ready),.admit_request(lane_request),
        .retire_valid(result_valid & result_ready),.retire_context(result_context),
        .retire_status(result_status),.retire_bytes(result_bytes),
        .contexts_held(contexts_held),.pending(admission_pending),.watermark(admission_watermark),.halted(admission_halted));
    for(genvar l=0;l<2;l++)begin: lanes
        ntrpu_peer_read_lane #(.SOURCE_NODE(0)) peer(
            .clk(clk),.rst_n(rst_n),.stop(source_stop),.link_up(&link_up),.reader_fault(source_fault),.revoke(1'b0),
            .cfg_valid(cfg_fire),.cfg_ready(lane_cfg_ready[l]),.cfg(cfg),.cfg_error(lane_cfg_error[l]),
            .request_valid(lane_request_valid[l]),.request_ready(lane_request_ready[l]),.request(lane_request[l*1024+:1024]),
            .reject_valid(lane_reject_valid[l]),.reject_ready(1'b0),.reject_code(),.reject_identity(),
            .cmd_valid(cmd_valid[l]),.cmd_ready(cmd_ready[l]),.cmd_addr(cmd_addr[l*64+:64]),.cmd_tag(cmd_tag[l*64+:64]),.cmd_len(cmd_len[l*8+:8]),
            .data_valid(data_valid[l]),.data_ready(data_ready[l]),.data_last(data_last[l]),.data(data[l*512+:512]),.data_tag(data_tag[l*64+:64]),
            .cpl_valid(cpl_valid[l]),.cpl_ready(cpl_ready[l]),.cpl_tag(cpl_tag[l*64+:64]),.cpl_status(cpl_status[l*2+:2]),.cpl_bytes(cpl_bytes[l*13+:13]),
            .dma_valid(dma_valid[l]),.dma_ready(dma_ready[l]),.dma_last(dma_last[l]),.dma_data(dma_data[l*512+:512]),
            .dma_context(dma_context[l*1024+:1024]),.dma_address(dma_address[l*64+:64]),
            .dma_done_valid(dma_done_valid[l]),.dma_done(dma_done[l*580+:580]),.ack_valid(ack_valid[l]),.ack(ack[l*580+:580]),
            .result_valid(result_valid[l]),.result_ready(result_ready[l]),.result_context(result_context[l*1024+:1024]),
            .result_status(result_status[l*4+:4]),.result_bytes(result_bytes[l*64+:64]),.contexts_held(lane_contexts[l*2+:2]),
            .halted(lane_halted[l]),.quarantined(lane_quarantined[l]));
    end
    ntrpu_hbm_read_return reader(
        .clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_reset_n),.init_complete_async(init_complete_async),
        .stop(source_stop || !(&link_up)),.fault_in(|lane_reject_valid),.rst_n(rst_n),.mem_rst_n(mem_rst_n),.started(started),.fault(reader_fault),.drained(reader_drained),
        .cmd_valid(cmd_valid),.cmd_ready(cmd_ready),.cmd_addr(cmd_addr),.cmd_tag(cmd_tag),.cmd_len(cmd_len),
        .data_valid(data_valid),.data_ready(data_ready),.data_last(data_last),.data(data),.data_tag(data_tag),
        .cpl_valid(cpl_valid),.cpl_ready(cpl_ready),.cpl_tag(cpl_tag),.cpl_status(cpl_status),.cpl_bytes(cpl_bytes),
        .h_arvalid(h_arvalid),.h_arready(h_arready),.h_araddr(h_araddr),.h_arlen(h_arlen),.h_arid(h_arid),.h_arsize(h_arsize),.h_arburst(h_arburst),
        .h_rvalid(h_rvalid),.h_rready(h_rready),.h_rdata(h_rdata),.h_rid(h_rid),.h_rresp(h_rresp),.h_rlast(h_rlast));
endmodule
