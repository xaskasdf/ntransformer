`timescale 1ns/1ps
`default_nettype none
// Public-port join for the frozen rev3 bridge and the HBM read-return interface.
// One preauthorized physical lease. The bridge alone authorizes requests and
// retirement; these four mirrors pin HBM addresses and full transfer identities.
// A registered rejection verdict is sampled one cycle after submission.
module ntrpu_hbm_source_join (
    input wire clk,rst_n,started,reader_started,stop,reader_fault,
    input wire [1:0] bridge_halted,
    output reg fault,
    output wire stop_out,
    input wire cfg_valid,
    output wire cfg_ready,cfg_error,
    input ntrpu_peer_read_pkg::lease_t cfg,
    output wire b_cfg_valid,
    input wire b_cfg_ready,b_cfg_error,
    input wire submit_valid,submit_channel,
    output wire submit_ready,
    input ntrpu_peer_read_pkg::request_t submit_context,
    output wire b_submit_valid,
    input wire b_submit_ready,b_reject_valid,
    output wire admission_pending,
    output wire [3:0] source_held,
    output wire [1:0] cmd_valid,
    input wire [1:0] cmd_ready,
    output wire [127:0] cmd_addr,cmd_tag,
    output wire [15:0] cmd_len,
    input wire [1:0] r_data_valid,r_data_last,
    output wire [1:0] r_data_ready,
    input wire [1023:0] r_data,
    input wire [127:0] r_data_tag,
    input wire [1:0] r_cpl_valid,
    output wire [1:0] r_cpl_ready,
    input wire [127:0] r_cpl_tag,
    input wire [3:0] r_cpl_status,
    input wire [25:0] r_cpl_bytes,
    output wire [1:0] data_valid,data_last,local_valid,
    input wire [1:0] data_ready,
    output wire [1023:0] data,
    output wire [2047:0] data_context,
    output wire [1159:0] local_event,
    input wire [1:0] result_valid,result_ready,
    output wire [1:0] b_result_ready,result_permit,
    input wire [2047:0] result_context,
    input wire [7:0] result_status,
    input wire [127:0] result_bytes
);
    import ntrpu_peer_read_pkg::*;
    reg configured,cfg_pending,pending,pending_lane,pending_slot;
    reg [63:0] source_base,candidate_base,pending_address;
    request_t pending_context;
    wire [1:0] alloc_slot,can_reserve,lane_fault;
    wire fail_now=stop || reader_fault || |bridge_halted || |lane_fault;
    assign stop_out=fault || fail_now;
    assign admission_pending=pending;
    wire config_allowed=rst_n && started && reader_started && !stop_out &&
                        !cfg_pending && !pending && source_held==0;
    assign cfg_ready=config_allowed && b_cfg_ready;
    assign b_cfg_valid=cfg_valid && config_allowed;
    assign cfg_error=b_cfg_error;
    wire can_submit=rst_n && started && reader_started && configured && !stop_out &&
                    !cfg_pending && !cfg_valid && !pending && can_reserve[submit_channel];
    assign submit_ready=can_submit && b_submit_ready;
    assign b_submit_valid=can_submit && submit_valid;
    wire reserve=pending && !b_reject_valid;
    always @(posedge clk)begin
        if(!rst_n)begin
            configured<=0;cfg_pending<=0;pending<=0;pending_lane<=0;pending_slot<=0;
            source_base<=0;candidate_base<=0;pending_address<=0;pending_context<='0;fault<=0;
        end else begin
            if(fail_now)fault<=1;
            if(cfg_valid && cfg_ready)begin cfg_pending<=1;candidate_base<=cfg.source_base;end
            if(cfg_pending)begin
                cfg_pending<=0;
                if(!b_cfg_error)begin configured<=1;source_base<=candidate_base;end
            end
            if(submit_valid && submit_ready)begin
                pending<=1;pending_lane<=submit_channel;pending_slot<=alloc_slot[submit_channel];
                pending_context<=submit_context;pending_address<=source_base+submit_context.source_offset;
            end
            if(pending)pending<=0;
        end
    end
    for(genvar lane=0;lane<2;lane=lane+1)begin: lanes
        reg [1023:0] contexts[0:1];
        reg [63:0] addresses[0:1];
        reg [12:0] sent[0:1];
        reg [1:0] active,issued,local_done;
        reg alloc_ptr,issue_ptr;
        request_t c0,c1,issuing,reading,completing,retiring;
        event_t completion;
        wire data_index=active[1] && c1.identity.sequence_id==r_data_tag[lane*64+:64];
        wire cpl_index=active[1] && c1.identity.sequence_id==r_cpl_tag[lane*64+:64];
        wire result_index=active[1] && contexts[1]==result_context[lane*1024+:1024];
        always @* begin
            c0=contexts[0];c1=contexts[1];
            issuing=contexts[issue_ptr];
            reading=contexts[data_index];completing=contexts[cpl_index];retiring=contexts[result_index];
            completion='0;completion.identity=completing.identity;completion.target=completing.target;
            completion.window_generation=completing.window_generation;
            completion.bytes=r_cpl_bytes[lane*13+:13];
        end
        assign source_held[lane*2+:2]={1'b0,active[0]}+{1'b0,active[1]};
        assign alloc_slot[lane]=alloc_ptr;
        assign can_reserve[lane]=!active[alloc_ptr];
        // Accepted work may remain offered after stop. The reader closes new
        // admissions; never retract a command that is already under backpressure.
        assign cmd_valid[lane]=active[issue_ptr] && !issued[issue_ptr];
        assign cmd_addr[lane*64+:64]=addresses[issue_ptr];
        assign cmd_tag[lane*64+:64]=issuing.identity.sequence_id;
        assign cmd_len[lane*8+:8]=(issuing.bytes>>6)-1'b1;
        wire data_match=active[data_index] && issued[data_index] && !local_done[data_index] &&
                        reading.identity.sequence_id==r_data_tag[lane*64+:64];
        wire cpl_match=active[cpl_index] && issued[cpl_index] && !local_done[cpl_index] &&
                       completing.identity.sequence_id==r_cpl_tag[lane*64+:64];
        wire result_match=active[result_index] && contexts[result_index]==result_context[lane*1024+:1024];
        wire data_bad=r_data_valid[lane] && (!data_match || sent[data_index]>=reading.bytes ||
                      r_data_last[lane]!=(sent[data_index]+13'd64==reading.bytes));
        wire cpl_bad=r_cpl_valid[lane] && (!cpl_match || r_cpl_status[lane*2+:2]!=0 ||
                     r_cpl_bytes[lane*13+:13]!=completing.bytes || sent[cpl_index]!=completing.bytes);
        wire result_bad=result_valid[lane] && (!result_match ||
                        (result_status[lane*4+:4]==0 &&
                         (!local_done[result_index] || result_bytes[lane*64+:64]!=retiring.bytes)));
        assign lane_fault[lane]=data_bad || cpl_bad || result_bad;
        assign data_valid[lane]=r_data_valid[lane] && data_match && !data_bad;
        assign r_data_ready[lane]=data_ready[lane] && data_match && !data_bad;
        assign data_last[lane]=r_data_last[lane];
        assign data[lane*512+:512]=r_data[lane*512+:512];
        assign data_context[lane*1024+:1024]=reading;
        assign r_cpl_ready[lane]=cpl_match;
        // No fabricated successful CPL on an HBM error, even if the local
        // frame is drained. The pinned transfer remains owned through failure.
        assign local_valid[lane]=r_cpl_valid[lane] && cpl_match && !cpl_bad;
        assign local_event[lane*580+:580]=completion;
        assign result_permit[lane]=result_match && !result_bad;
        assign b_result_ready[lane]=result_ready[lane] && result_permit[lane];
        always @(posedge clk)begin
            if(!rst_n)begin
                active<=0;issued<=0;local_done<=0;alloc_ptr<=0;issue_ptr<=0;
                for(integer i=0;i<2;i=i+1)begin contexts[i]<=0;addresses[i]<=0;sent[i]<=0;end
            end else begin
                if(reserve && pending_lane==lane)begin
                    active[pending_slot]<=1;issued[pending_slot]<=0;local_done[pending_slot]<=0;
                    contexts[pending_slot]<=pending_context;addresses[pending_slot]<=pending_address;
                    sent[pending_slot]<=0;alloc_ptr<=!alloc_ptr;
                end
                if(cmd_valid[lane] && cmd_ready[lane])begin issued[issue_ptr]<=1;issue_ptr<=!issue_ptr;end
                if(data_valid[lane] && data_ready[lane])sent[data_index]<=sent[data_index]+13'd64;
                if(r_cpl_valid[lane] && r_cpl_ready[lane])local_done[cpl_index]<=1;
                if(result_valid[lane] && b_result_ready[lane] && result_status[lane*4+:4]==0 &&
                   !stop_out)active[result_index]<=0;
            end
        end
    end
endmodule
`default_nettype wire
