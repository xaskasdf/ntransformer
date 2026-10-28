`timescale 1ns/1ps
// Two context reservations, each retained beyond local CPL through remote ACK
// and accepted result. One span per transfer. No timeout, warm reset or rearm.
module ntrpu_peer_region_lane #(
    parameter logic [63:0] SOURCE_NODE=0
) (
    input wire clk, rst_n, stop, link_up, reader_fault, revoke,
    input wire cfg_valid,
    output wire cfg_ready,
    input ntrpu_peer_read_pkg::lease_t cfg,
    output reg cfg_error,
    input wire request_valid,
    output wire request_ready,
    input ntrpu_peer_read_pkg::request_t request,
    input wire [63:0] request_source_address, // Trusted, captured by region admission.
    output reg reject_valid,
    input wire reject_ready,
    output reg [3:0] reject_code,
    output ntrpu_peer_read_pkg::identity_t reject_identity,
    output wire cmd_valid,
    input wire cmd_ready,
    output wire [63:0] cmd_addr, cmd_tag,
    output wire [7:0] cmd_len,
    input wire data_valid, data_last,
    output wire data_ready,
    input wire [511:0] data,
    input wire [63:0] data_tag,
    input wire cpl_valid,
    output wire cpl_ready,
    input wire [63:0] cpl_tag,
    input wire [1:0] cpl_status,
    input wire [12:0] cpl_bytes,
    output wire dma_valid, dma_last,
    input wire dma_ready,
    output wire [511:0] dma_data,
    output ntrpu_peer_read_pkg::request_t dma_context,
    output wire [63:0] dma_address,
    input wire dma_done_valid, ack_valid,
    input ntrpu_peer_read_pkg::event_t dma_done, ack,
    output reg result_valid,
    input wire result_ready,
    output ntrpu_peer_read_pkg::request_t result_context,
    output reg [3:0] result_status,
    output reg [63:0] result_bytes,
    output wire [1:0] contexts_held,
    output reg halted,
    output wire quarantined
);
    import ntrpu_peer_read_pkg::*;
    lease_t lease;
    request_t context0, context1, saved_result;
    // Reserve at the input handshake; commit the wide context one edge later.
    // The reserved slot cannot stall commit, so replacement can occur each edge.
    reg stage_valid, stage_slot;
    request_t stage_request;
    reg [63:0] stage_source, stage_target;
    request_t reserved0, reserved1;
    always @* begin
        reserved0=(stage_valid && !stage_slot)?stage_request:context0;
        reserved1=(stage_valid && stage_slot)?stage_request:context1;
    end
    identity_t saved_reject;
    wire [1023:0] contexts [0:1];
    assign contexts[0]=context0;
    assign contexts[1]=context1;
    request_t selected_data, selected_cpl, selected_issue, selected_result;
    reg configured, revoked;
    reg [63:0] watermark;
    reg [1:0] active, issued, local_done, dma_complete, ack_complete, failed, reported;
    reg alloc_ptr, issue_ptr, retire_ptr;
    reg [63:0] addresses[0:1], destinations[0:1];
    reg [12:0] sent[0:1];
    wire data_index=active[1] && context1.identity.sequence_id==data_tag;
    wire cpl_index=active[1] && context1.identity.sequence_id==cpl_tag;
    always @* begin
        selected_data=data_index?context1:context0;
        selected_cpl=cpl_index?context1:context0;
        selected_issue=issue_ptr?context1:context0;
        selected_result=retire_ptr?context1:context0;
    end
    assign reject_identity=saved_reject;
    assign result_context=saved_result;
    assign contexts_held={1'b0,active[0]}+{1'b0,active[1]};
    assign quarantined=halted && |active;
    assign cfg_ready=rst_n && active==0 && !halted;
    wire cfg_bad=cfg.source.session==0 || cfg.source.session!=cfg.target.session ||
                 cfg.source.node!=SOURCE_NODE || cfg.target.node!=(SOURCE_NODE^64'd1) ||
                 cfg.source.generation==0 || cfg.target.generation==0 || cfg.window_generation==0 ||
                 cfg.model_epoch==0 || (configured && (cfg.source.session!=lease.source.session ||
                 cfg.source.generation<=lease.source.generation ||
                 cfg.target.generation<=lease.target.generation || cfg.window_generation<=lease.window_generation));
    assign request_ready=rst_n && configured && !halted && !stop && link_up && !reader_fault &&
                         !cfg_valid && !reject_valid && !active[alloc_ptr];
    wire [64:0] start_address={1'b0,request_source_address};
    wire [64:0] end_address=start_address+{1'b0,request.bytes}-65'd1;
    wire [64:0] target_address={1'b0,lease.target_base}+{1'b0,request.target_offset};
    wire [64:0] target_end=target_address+{1'b0,request.bytes}-65'd1;
    reg [3:0] admission_error;
    always @* begin
        admission_error=OK;
        if(request.source!=lease.source || request.target!=lease.target ||
           request.identity.session!=lease.source.session || request.identity.source!=SOURCE_NODE ||
           request.window_generation!=lease.window_generation || request.model_epoch!=lease.model_epoch)
            admission_error=HANDLE;
        else if(revoked || revoke || !lease.source_open || !lease.target_open ||
                !lease.source_read || !lease.target_write) admission_error=PERMISSION;
        else if(request.identity.sequence_id<=watermark) admission_error=REPLAY;
        else if(request.bytes==0 || request.source_offset>lease.source_size ||
                request.bytes>lease.source_size-request.source_offset ||
                request.target_offset>lease.target_size || request.bytes>lease.target_size-request.target_offset ||
                request.target_offset<lease.window_offset || request.target_offset-lease.window_offset>lease.window_bytes ||
                request.bytes>lease.window_bytes-(request.target_offset-lease.window_offset) ||
                start_address[64] || end_address[64] || target_address[64] || target_end[64]) admission_error=RANGE;
        else if(request.source_offset<lease.published_offset ||
                request.source_offset-lease.published_offset>lease.published_bytes ||
                request.bytes>lease.published_bytes-(request.source_offset-lease.published_offset)) admission_error=UNPUBLISHED;
        else if(request.bytes>4096 || |request.bytes[5:0] || |start_address[5:0] ||
                |request.source_offset[5:0] || |request.target_offset[5:0] || |target_address[5:0] ||
                |start_address[63:34] || |end_address[63:34] || end_address[63:12]!=start_address[63:12]) admission_error=SPAN;
        else if((active[0] && request.target_offset<reserved0.target_offset+reserved0.bytes &&
                 reserved0.target_offset<request.target_offset+request.bytes) ||
                (active[1] && request.target_offset<reserved1.target_offset+reserved1.bytes &&
                 reserved1.target_offset<request.target_offset+request.bytes)) admission_error=OVERLAP;
    end
    // No combinational cancellation of a command already offered to the reader.
    assign cmd_valid=active[issue_ptr] && !issued[issue_ptr] && !(stage_valid && stage_slot==issue_ptr);
    assign cmd_addr=addresses[issue_ptr];
    assign cmd_tag=selected_issue.identity.sequence_id;
    assign cmd_len=(selected_issue.bytes>>6)-1'b1;
    wire data_match=active[data_index] && issued[data_index] && !local_done[data_index] &&
                    selected_data.identity.sequence_id==data_tag;
    wire cpl_match=active[cpl_index] && issued[cpl_index] && !local_done[cpl_index] &&
                   selected_cpl.identity.sequence_id==cpl_tag;
    assign dma_valid=data_valid && data_match;
    assign data_ready=dma_ready && data_match;
    assign dma_data=data;
    assign dma_last=data_last;
    assign dma_context=selected_data;
    assign dma_address=destinations[data_index];
    assign cpl_ready=cpl_match;
    // No wide context multiplexer precedes event authentication.
    // Admission guarantees distinct sequence IDs for live contexts.
    wire [1:0] dma_accept, ack_accept;
    for(genvar e=0;e<2;e++)begin: event_validation
        request_t event_context;
        assign event_context=contexts[e];
        assign dma_accept[e]=active[e] && issued[e] && !dma_complete[e] &&
            event_context.identity==dma_done.identity && event_context.target==dma_done.target &&
            event_context.window_generation==dma_done.window_generation && dma_done.status==0 &&
            dma_done.bytes==event_context.bytes && sent[e]==event_context.bytes;
        assign ack_accept[e]=active[e] && issued[e] && !ack_complete[e] &&
            event_context.identity==ack.identity && event_context.target==ack.target &&
            event_context.window_generation==ack.window_generation && ack.status==0 &&
            ack.bytes==event_context.bytes && sent[e]==event_context.bytes;
    end
    wire data_error=data_valid && (!data_match || (data_ready &&
                    (sent[data_index]>=selected_data.bytes || data_last!=(sent[data_index]+13'd64==selected_data.bytes))));
    wire cpl_error=cpl_valid && (!cpl_match || cpl_status!=0 || cpl_bytes!=selected_cpl.bytes ||
                               sent[cpl_index]!=selected_cpl.bytes);
    wire dma_error=dma_done_valid && !(|dma_accept);
    wire ack_error=ack_valid && !(|ack_accept);
    wire fail_now=stop || !link_up || reader_fault || data_error || cpl_error || dma_error || ack_error;
    always @(posedge clk) begin
        if(!rst_n) begin
            stage_valid<=0;stage_slot<=0;stage_request<='0;stage_source<=0;stage_target<=0;
            configured<=0;revoked<=0;watermark<=0;lease<='0;
            context0<='0;context1<='0;saved_result<='0;saved_reject<='0;
            active<=0;issued<=0;local_done<=0;dma_complete<=0;ack_complete<=0;failed<=0;reported<=0;
            alloc_ptr<=0;issue_ptr<=0;retire_ptr<=0;halted<=0;cfg_error<=0;
            result_valid<=0;result_status<=0;result_bytes<=0;reject_valid<=0;reject_code<=0;
            for(integer i=0;i<2;i++)begin addresses[i]<=0;destinations[i]<=0;sent[i]<=0;end
        end else begin
            stage_valid<=0;
            if(stage_valid)begin
                if(stage_slot)context1<=stage_request;else context0<=stage_request;
                addresses[stage_slot]<=stage_source;destinations[stage_slot]<=stage_target;
            end
            cfg_error<=0;
            if(revoke)revoked<=1;
            if(cfg_valid && cfg_ready)begin
                cfg_error<=cfg_bad;
                if(!cfg_bad)begin lease<=cfg;configured<=1;revoked<=revoke;end
            end
            if(reject_valid && reject_ready)reject_valid<=0;
            if(request_valid && request_ready)begin
                // These wide enables depend on the handshake, not on admission arithmetic.
                stage_request<=request;stage_slot<=alloc_ptr;
                stage_source<=start_address[63:0];stage_target<=target_address[63:0];
                if(admission_error!=OK)begin
                    reject_valid<=1;reject_code<=admission_error;saved_reject<=request.identity;
                end else begin
                    stage_valid<=1;
                    active[alloc_ptr]<=1;issued[alloc_ptr]<=0;local_done[alloc_ptr]<=0;
                    dma_complete[alloc_ptr]<=0;ack_complete[alloc_ptr]<=0;failed[alloc_ptr]<=0;reported[alloc_ptr]<=0;
                    sent[alloc_ptr]<=0;
                    alloc_ptr<=!alloc_ptr;watermark<=request.identity.sequence_id;
                end
            end
            if(cmd_valid && cmd_ready)begin issued[issue_ptr]<=1;issue_ptr<=!issue_ptr;end
            if(dma_valid && dma_ready)sent[data_index]<=sent[data_index]+13'd64;
            if(cpl_valid && cpl_ready)local_done[cpl_index]<=1;
            for(integer i=0;i<2;i++)begin
                if(dma_done_valid && dma_accept[i])dma_complete[i]<=1;
                if(ack_valid && ack_accept[i])ack_complete[i]<=1;
            end
            if(fail_now)begin
                halted<=1;
                for(integer i=0;i<2;i++)if(active[i] && !(result_valid && retire_ptr==i))failed[i]<=1;
            end
            if(!result_valid && active[retire_ptr] && !reported[retire_ptr] && !(stage_valid && stage_slot==retire_ptr))begin
                if(failed[retire_ptr] || halted || fail_now ||
                   (local_done[retire_ptr] && dma_complete[retire_ptr] && ack_complete[retire_ptr]))begin
                    result_valid<=1;saved_result<=selected_result;
                    result_status<=(failed[retire_ptr] || halted || fail_now)?4'd1:4'd0;
                    result_bytes<=(failed[retire_ptr] || halted || fail_now)?64'd0:selected_result.bytes;
                end
            end
            if(result_valid && result_ready)begin
                result_valid<=0;reported[retire_ptr]<=1;
                // Failed results may be reported, but never release a lease.
                if(result_status==0)active[retire_ptr]<=0;
                retire_ptr<=!retire_ptr;
            end
        end
    end
endmodule
