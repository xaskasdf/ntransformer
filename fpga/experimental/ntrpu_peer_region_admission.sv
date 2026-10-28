`timescale 1ns/1ps
// One source, one pinned lease, two fixed channels. Four metadata reservations;
// no payload storage, shared-region manager, timeout or warm recovery.
module ntrpu_peer_region_admission #(
    parameter logic [63:0] SOURCE_NODE=0
) (
    input wire clk,rst_n,enable,stop,link_up,fault_in,revoke,
    input wire cfg_valid,
    output wire cfg_ready,
    input ntrpu_peer_read_pkg::lease_t cfg,
    input ntrpu_peer_region_pkg::region_map_t cfg_map,
    output wire cfg_map_bad,
    output wire [127:0] admit_address,
    output reg cfg_error,
    input wire [1:0] request_valid,
    output reg [1:0] request_ready,
    input wire [2047:0] request,
    output reg [1:0] reject_valid,
    input wire [1:0] reject_ready,
    output wire [7:0] reject_code,
    output wire [383:0] reject_identity,
    output wire [1:0] admit_valid,
    input wire [1:0] admit_ready,
    output wire [2047:0] admit_request,
    // Trusted retirements from the peer ledgers, AFTER accepted public results.
    input wire [1:0] retire_valid,
    input wire [2047:0] retire_context,
    input wire [7:0] retire_status,
    input wire [127:0] retire_bytes,
    output wire [3:0] contexts_held,
    output wire [1:0] pending,
    output reg [63:0] watermark,
    output reg halted
);
    import ntrpu_peer_read_pkg::*;
    import ntrpu_peer_region_pkg::*;
    lease_t lease;
    region_map_t pinned_map;
    wire [1:0] map_hit;
    wire [63:0] resolved[0:1];
    reg [63:0] saved_address[0:3];
    ntrpu_peer_region_lookup config_map(.lease_cfg(cfg),.region_map(cfg_map),
        .logical_offset(64'b0),.bytes(64'b0),.cfg_bad(cfg_map_bad),.hit(),.physical_address());

    // Inclusive lower bounds remain explicit; these exclusive ends never wrap.
    reg [64:0] window_end, published_end;
    wire [64:0] request_source_end[0:1],request_target_end[0:1];
    request_t req[0:1],retiring[0:1],saved[0:3];
    for(genvar q=0;q<2;q++)begin: mapping
        ntrpu_peer_region_lookup lookup(.lease_cfg(lease),.region_map(pinned_map),
            .logical_offset(req[q].source_offset),.bytes(req[q].bytes),.cfg_bad(),
            .hit(map_hit[q]),.physical_address(resolved[q]));
    end
    // All other fields are identical to the pinned lease while any slot is active.
    typedef struct packed {
        logic [63:0] sequence_id,source_offset,target_offset,bytes;
    } compact_t;
    compact_t compact[0:3];
    for(genvar i=0;i<4;i++)begin: restore_context
        always @* begin
            saved[i]='0;
            saved[i].identity.session=lease.source.session;
            saved[i].identity.source=SOURCE_NODE;
            saved[i].identity.sequence_id=compact[i].sequence_id;
            saved[i].source=lease.source;saved[i].target=lease.target;
            saved[i].window_generation=lease.window_generation;saved[i].model_epoch=lease.model_epoch;
            saved[i].source_offset=compact[i].source_offset;saved[i].target_offset=compact[i].target_offset;
            saved[i].bytes=compact[i].bytes;
        end
    end
    identity_t rejected[0:1];
    reg [3:0] errors[0:1];
    reg [3:0] active,issued,reported;
    reg [1:0] alloc_ptr,issue_ptr;
    reg configured,revoked,turn;
    wire cfg_bad=cfg_map_bad || cfg.source.session==0 || cfg.source.session!=cfg.target.session ||
        cfg.source.node!=SOURCE_NODE || cfg.target.node!=(SOURCE_NODE^64'd1) ||
        cfg.source.generation==0 || cfg.target.generation==0 || cfg.window_generation==0 ||
        cfg.model_epoch==0 || (configured && (cfg.source.session!=lease.source.session ||
        cfg.source.generation<=lease.source.generation || cfg.target.generation<=lease.target.generation ||
        cfg.window_generation<=lease.window_generation));
    assign cfg_ready=rst_n && active==0 && !halted;
    reg [3:0] admission_error[0:1];
    reg [64:0] first[0:1],last[0:1],target_first[0:1],target_last[0:1];
    reg [1:0] eligible;
    reg chosen;
    reg [1:0] match_found,retire_index,retire_error;
    for(genvar l=0;l<2;l++)begin: channels
        assign request_source_end[l]={1'b0,req[l].source_offset}+{1'b0,req[l].bytes};
        assign request_target_end[l]={1'b0,req[l].target_offset}+{1'b0,req[l].bytes};
        always @* begin req[l]=request[l*1024+:1024];retiring[l]=retire_context[l*1024+:1024];end
        assign admit_valid[l]=rst_n && active[2*l+issue_ptr[l]] && !issued[2*l+issue_ptr[l]];
        assign admit_request[l*1024+:1024]=saved[2*l+issue_ptr[l]];
        assign admit_address[l*64+:64]=saved_address[2*l+issue_ptr[l]];
        assign contexts_held[l*2+:2]={1'b0,active[2*l]}+{1'b0,active[2*l+1]};
        assign pending[l]=(active[2*l]&&!issued[2*l]) || (active[2*l+1]&&!issued[2*l+1]);
        assign reject_identity[l*192+:192]=rejected[l];
        assign reject_code[l*4+:4]=errors[l];
    end
    always @* begin
        for(integer l=0;l<2;l++)begin
            first[l]={1'b0,lease.source_base}+{1'b0,req[l].source_offset};
            last[l]=first[l]+{1'b0,req[l].bytes}-65'd1;
            target_first[l]={1'b0,lease.target_base}+{1'b0,req[l].target_offset};
            target_last[l]=target_first[l]+{1'b0,req[l].bytes}-65'd1;
            admission_error[l]=OK;
            if(req[l].source!=lease.source || req[l].target!=lease.target ||
               req[l].identity.session!=lease.source.session || req[l].identity.source!=SOURCE_NODE ||
               req[l].window_generation!=lease.window_generation || req[l].model_epoch!=lease.model_epoch)
                admission_error[l]=HANDLE;
            else if(revoked || revoke || !lease.source_open || !lease.target_open ||
                    !lease.source_read || !lease.target_write)admission_error[l]=PERMISSION;
            else if(req[l].identity.sequence_id<=watermark)admission_error[l]=REPLAY;
            else if(req[l].bytes==0 || request_source_end[l]>{1'b0,lease.source_size} ||
                request_target_end[l]>{1'b0,lease.target_size} ||
                req[l].target_offset<lease.window_offset || request_target_end[l]>window_end ||
                first[l][64] || last[l][64] || target_first[l][64] || target_last[l][64])admission_error[l]=RANGE;
            else if(req[l].source_offset<lease.published_offset ||
                request_source_end[l]>published_end)admission_error[l]=UNPUBLISHED;
            else if(req[l].bytes>4096 || |req[l].bytes[5:0] || |first[l][5:0] ||
                |req[l].source_offset[5:0] || |req[l].target_offset[5:0] || |target_first[l][5:0] ||
                |first[l][63:34] || |last[l][63:34] || last[l][63:12]!=first[l][63:12])admission_error[l]=SPAN;
            if(admission_error[l]==OK && !map_hit[l])admission_error[l]=MAP_MISS;
            for(integer i=0;i<4;i++)if(admission_error[l]==OK && active[i] &&
                {1'b0,req[l].target_offset} < {1'b0,saved[i].target_offset}+{1'b0,saved[i].bytes} &&
                {1'b0,saved[i].target_offset} < {1'b0,req[l].target_offset}+{1'b0,req[l].bytes})
                admission_error[l]=OVERLAP;
            eligible[l]=request_valid[l] && !reject_valid[l] &&
                (admission_error[l]!=OK || !active[2*l+alloc_ptr[l]]);
            match_found[l]=0;retire_index[l]=0;
            for(integer i=0;i<2;i++)if(active[2*l+i] && saved[2*l+i]==retiring[l])begin
                match_found[l]=1;retire_index[l]=i;
            end
            retire_error[l]=retire_valid[l] && (!match_found[l] || !issued[2*l+retire_index[l]] ||
                reported[2*l+retire_index[l]] ||
                (retire_status[l*4+:4]==0 ? retire_bytes[l*64+:64]!=retiring[l].bytes : retire_bytes[l*64+:64]!=0));
        end
        chosen=eligible[turn]?turn:!turn;
        request_ready=0;
        // One transaction per edge makes watermark/conflict reservation atomic.
        if(rst_n && enable && configured && !cfg_valid && !halted && !stop && link_up &&
           !fault_in && !(|retire_error) && eligible[chosen])request_ready[chosen]=1;
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            pinned_map<='0;for(integer i=0;i<4;i++)saved_address[i]<=0;
            window_end<=0;published_end<=0;
            lease<='0;configured<=0;revoked<=0;watermark<=0;halted<=0;cfg_error<=0;
            active<=0;issued<=0;reported<=0;alloc_ptr<=0;issue_ptr<=0;turn<=0;reject_valid<=0;
            for(integer i=0;i<4;i++)compact[i]<='0;
            for(integer l=0;l<2;l++)begin rejected[l]<='0;errors[l]<=0;end
        end else begin
            cfg_error<=0;
            if(revoke)revoked<=1;
            if(cfg_valid && cfg_ready)begin
                cfg_error<=cfg_bad;
                if(!cfg_bad)begin
                    pinned_map<=cfg_map;
                    lease<=cfg;configured<=1;revoked<=revoke;
                    window_end<={1'b0,cfg.window_offset}+{1'b0,cfg.window_bytes};
                    published_end<={1'b0,cfg.published_offset}+{1'b0,cfg.published_bytes};
                end
            end
            if(stop || !link_up || fault_in || |retire_error)halted<=1;
            for(integer l=0;l<2;l++)begin
                if(reject_valid[l] && reject_ready[l])reject_valid[l]<=0;
                if(request_valid[l] && request_ready[l])begin
                    turn<=!l[0];
                    if(admission_error[l]!=OK)begin
                        reject_valid[l]<=1;errors[l]<=admission_error[l];rejected[l]<=req[l].identity;
                    end else begin
                        saved_address[2*l+alloc_ptr[l]]<=resolved[l];
                        compact[2*l+alloc_ptr[l]]<={req[l].identity.sequence_id,req[l].source_offset,req[l].target_offset,req[l].bytes};active[2*l+alloc_ptr[l]]<=1;
                        issued[2*l+alloc_ptr[l]]<=0;reported[2*l+alloc_ptr[l]]<=0;
                        alloc_ptr[l]<=!alloc_ptr[l];watermark<=req[l].identity.sequence_id;
                    end
                end
                if(admit_valid[l] && admit_ready[l])begin
                    issued[2*l+issue_ptr[l]]<=1;issue_ptr[l]<=!issue_ptr[l];
                end
                if(retire_valid[l] && !retire_error[l])begin
                    reported[2*l+retire_index[l]]<=1;
                    if(retire_status[l*4+:4]==0)active[2*l+retire_index[l]]<=0;
                    else halted<=1; // An accepted failure never releases ownership.
                end
            end
        end
    end
endmodule
