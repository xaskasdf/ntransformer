`timescale 1ns/1ps
`default_nettype none
// Revision 3: precomputed exclusive admission bounds; unchanged ready/valid.
// Isolated experimental bridge. One configured, preauthorized lease; two DMA
// channels. No native HBM mapping, PCIe control-plane ABI or XDMA register BFM.
module ntrpu_hbm_dma_bridge (
    input wire clk,dma_clk,cold_reset_n,stop,
    output wire started,rst_n,dma_rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input ntrpu_peer_read_pkg::lease_t cfg,
    input wire [63:0] cfg_writeback,
    output reg cfg_error,
    input wire submit_valid,submit_channel,
    output wire submit_ready,
    input ntrpu_peer_read_pkg::request_t submit_context,
    output reg reject_valid,
    input wire reject_ready,
    output reg [3:0] reject_code,
    input wire [1:0] data_valid,data_last,
    output wire [1:0] data_ready,
    input wire [1023:0] data,
    input wire [2047:0] data_context,
    input wire [1:0] local_valid,ack_valid,result_ready,
    input wire [1159:0] local_event,ack_event,
    output wire [1:0] result_valid,
    output wire [2047:0] result_context,
    output wire [7:0] result_status,
    output wire [127:0] result_bytes,
    output wire [3:0] contexts_held,wire_sent,
    output wire [1:0] halted,dma_fault,
    input wire dma_link_up,
    input wire [1:0] dma_run,dma_error,dma_done,dsc_ready,axis_ready,
    output wire [1:0] dsc_load,axis_valid,axis_last,
    output wire [127:0] dsc_src,dsc_dst,axis_keep,
    output wire [55:0] dsc_len,
    output wire [31:0] dsc_ctl,
    output wire [1023:0] axis_data
);
    import ntrpu_peer_read_pkg::*;
    (* ASYNC_REG="TRUE" *) reg [2:0] sr,dr;
    (* ASYNC_REG="TRUE" *) reg ready_s1,ready_s2,link_s1,link_s2;
    always @(posedge clk or negedge cold_reset_n)begin
        if(!cold_reset_n)begin sr<=0;ready_s1<=0;ready_s2<=0;link_s1<=0;link_s2<=0;end
        else begin sr<={sr[1:0],1'b1};ready_s1<=dma_rst_n;ready_s2<=ready_s1;link_s1<=dma_link_up;link_s2<=link_s1;end
    end
    always @(posedge dma_clk or negedge cold_reset_n)begin
        if(!cold_reset_n)dr<=0;else dr<={dr[1:0],1'b1};
    end
    assign rst_n=sr[2];assign dma_rst_n=dr[2];assign started=rst_n && ready_s2;
    lease_t lease;
    reg configured;
    // Exclusive 65-bit ends preserve an interval ending at 2**64.
    // These registers change atomically with an accepted, valid lease.
    reg [64:0] window_end,published_end,source_capacity,target_capacity;
    reg [63:0] writeback,watermark;
    wire global_stop=stop || |halted;
    wire [1:0] lane_ready,allocate_slot;
    wire [3:0] occupied;
    wire [4095:0] contexts;
    assign cfg_ready=started && contexts_held==0 && !global_stop;
    wire [64:0] wb_end={1'b0,cfg_writeback}+65'd256;
    wire [64:0] cfg_target_end={1'b0,cfg.target_base}+{1'b0,cfg.target_size};
    wire config_bad=cfg.source.session==0 || cfg.source.session!=cfg.target.session ||
        cfg.source.generation==0 || cfg.target.generation==0 || cfg.window_generation==0 || cfg.model_epoch==0 ||
        cfg.source_size==0 || cfg.target_size==0 || |cfg_writeback[5:0] || wb_end>65'h10000000000000000 ||
        cfg_target_end>65'h10000000000000000 ||
        ({1'b0,cfg_writeback}<cfg_target_end && {1'b0,cfg.target_base}<wb_end) ||
        (configured && (cfg.source.session!=lease.source.session || cfg.window_generation<=lease.window_generation));
    wire [64:0] source_start={1'b0,lease.source_base}+{1'b0,submit_context.source_offset};
    wire [64:0] source_limit={1'b0,submit_context.source_offset}+{1'b0,submit_context.bytes};
    wire [64:0] target_start={1'b0,lease.target_base}+{1'b0,submit_context.target_offset};
    wire [64:0] target_limit={1'b0,submit_context.target_offset}+{1'b0,submit_context.bytes};
    // Used only after RANGE and bytes<=4096. Then 13 bits cannot overflow.
    wire [12:0] page_limit={1'b0,source_start[11:0]}+submit_context.bytes[12:0];
    reg [3:0] admission_error;
    wire [3:0] overlap;
    for(genvar e=0;e<4;e=e+1)begin: conflicts
        request_t c;
        assign c=contexts[e*1024+:1024];
        assign overlap[e]=occupied[e] && submit_context.target_offset<c.target_offset+c.bytes &&
                           c.target_offset<submit_context.target_offset+submit_context.bytes;
    end
    always @* begin
        admission_error=OK;
        if(submit_context.source!=lease.source || submit_context.target!=lease.target ||
           submit_context.identity.session!=lease.source.session || submit_context.identity.source!=lease.source.node ||
           submit_context.window_generation!=lease.window_generation || submit_context.model_epoch!=lease.model_epoch)
            admission_error=HANDLE;
        else if(!lease.source_read || !lease.target_write || !lease.source_open || !lease.target_open)admission_error=PERMISSION;
        else if(submit_context.identity.sequence_id<=watermark)admission_error=REPLAY;
        else if(submit_context.bytes==0 || source_limit>{1'b0,lease.source_size} ||
                target_limit>{1'b0,lease.target_size} ||
                submit_context.target_offset<lease.window_offset || target_limit>window_end ||
                source_limit>source_capacity || target_limit>target_capacity)admission_error=RANGE;
        else if(submit_context.source_offset<lease.published_offset ||
                source_limit>published_end)admission_error=UNPUBLISHED;
        else if(submit_context.bytes>4096 || |submit_context.bytes[5:0] || |source_start[5:0] || |target_start[5:0] ||
                |source_start[64:34] || page_limit>13'd4096)admission_error=SPAN;
        else if(|overlap)admission_error=OVERLAP;
    end
    assign submit_ready=started && configured && link_s2 && !global_stop && !cfg_valid && !reject_valid && lane_ready[submit_channel];
    wire accept=submit_valid && submit_ready;
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            configured<=0;lease<='0;writeback<=0;watermark<=0;cfg_error<=0;reject_valid<=0;reject_code<=0;
            window_end<=0;published_end<=0;source_capacity<=0;target_capacity<=0;
        end
        else begin
            cfg_error<=0;
            if(cfg_valid && cfg_ready)begin
                cfg_error<=config_bad;
                if(!config_bad)begin
                    configured<=1;lease<=cfg;writeback<=cfg_writeback;
                    window_end<={1'b0,cfg.window_offset}+{1'b0,cfg.window_bytes};
                    published_end<={1'b0,cfg.published_offset}+{1'b0,cfg.published_bytes};
                    source_capacity<=65'h10000000000000000-{1'b0,cfg.source_base};
                    target_capacity<=65'h10000000000000000-{1'b0,cfg.target_base};
                end
            end
            if(reject_valid && reject_ready)reject_valid<=0;
            if(accept)begin
                if(admission_error!=OK)begin reject_valid<=1;reject_code<=admission_error;end
                else watermark<=submit_context.identity.sequence_id;
            end
        end
    end
    genvar lane_idx;
    generate for(lane_idx=0;lane_idx<2;lane_idx=lane_idx+1)begin: lanes
        wire [63:0] wb=writeback+64'(lane_idx*128)+{57'b0,allocate_slot[lane_idx],6'b0};
        ntrpu_hbm_dma_bridge_lane bridge(
            .clk(clk),.rst_n(rst_n),.dma_clk(dma_clk),.dma_rst_n(dma_rst_n),.stop(global_stop),
            .admit_valid(accept && admission_error==4'd0 && submit_channel==lane_idx),.admit_ready(lane_ready[lane_idx]),
            .admit_context(submit_context),.admit_address(target_start[63:0]),.admit_writeback(wb),
            .data_valid(data_valid[lane_idx]),.data_last(data_last[lane_idx]),.data_ready(data_ready[lane_idx]),
            .data(data[lane_idx*512+:512]),.data_context(data_context[lane_idx*1024+:1024]),
            .local_valid(local_valid[lane_idx]),.ack_valid(ack_valid[lane_idx]),
            .local_event(local_event[lane_idx*580+:580]),.ack_event(ack_event[lane_idx*580+:580]),
            .result_ready(result_ready[lane_idx]),.result_valid(result_valid[lane_idx]),.result_context(result_context[lane_idx*1024+:1024]),
            .result_status(result_status[lane_idx*4+:4]),.result_bytes(result_bytes[lane_idx*64+:64]),
            .held(contexts_held[lane_idx*2+:2]),.occupied(occupied[lane_idx*2+:2]),.held_context(contexts[lane_idx*2048+:2048]),
            .allocate_slot(allocate_slot[lane_idx]),.halted(halted[lane_idx]),.wire_sent(wire_sent[lane_idx*2+:2]),
            .dma_link_up(dma_link_up),.dma_run(dma_run[lane_idx]),.dma_error(dma_error[lane_idx]),.dma_done(dma_done[lane_idx]),
            .dsc_ready(dsc_ready[lane_idx]),.dsc_load(dsc_load[lane_idx]),.dsc_src(dsc_src[lane_idx*64+:64]),.dsc_dst(dsc_dst[lane_idx*64+:64]),
            .dsc_len(dsc_len[lane_idx*28+:28]),.dsc_ctl(dsc_ctl[lane_idx*16+:16]),
            .axis_valid(axis_valid[lane_idx]),.axis_ready(axis_ready[lane_idx]),.axis_last(axis_last[lane_idx]),
            .axis_data(axis_data[lane_idx*512+:512]),.axis_keep(axis_keep[lane_idx*64+:64]),.dma_fault(dma_fault[lane_idx]));
    end endgenerate
endmodule
`default_nettype wire
