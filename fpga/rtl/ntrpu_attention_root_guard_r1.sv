// Root ownership and trusted per-layer metadata for the 32-head wrapper.
// All engine bind/append/root access must pass through this guard.
`default_nettype none
module ntrpu_attention_root_guard_r1 (
    input wire logic clk, rst_n,
    input wire logic bind_valid,
    output logic bind_ready, engine_bind_valid,
    input wire logic engine_bind_ready,
    input wire logic [63:0] bind_sequence, bind_epoch,
    input wire logic append_valid,
    output logic append_ready, engine_append_valid,
    input wire logic engine_append_ready,
    input wire logic [31:0] append_layer, append_position,
    input wire logic [63:0] append_sequence, append_epoch,
    input wire logic command_valid,
    output logic command_ready, sequencer_command_valid,
    input wire logic sequencer_command_ready, sequencer_cache_lock,
    input wire logic [31:0] command_layer,
    input wire logic engine_lease_active,
    input wire logic engine_done_valid, engine_done_ready, engine_done_bind,
    input wire logic [3:0] engine_done_status,
    input wire logic [12:0] engine_done_bytes,
    input wire logic [31:0] engine_done_committed_tokens,
    input wire logic [63:0] engine_done_sequence, engine_done_epoch,
    input wire logic engine_cache_valid,
    input wire logic [31:0] engine_cache_capacity,
    input wire logic [63:0] engine_cache_sequence, engine_cache_epoch, expected_epoch,
    output logic trusted_cache_valid,
    output logic [31:0] trusted_committed_tokens,
    output logic maintenance_busy, metadata_fault
);
    logic [17:0] prefixes[0:15];
    logic authority_valid, maintenance_bind, maintenance_revoked;
    logic [31:0] held_layer, held_position;
    logic [63:0] held_sequence, held_epoch, trusted_sequence, trusted_epoch, last_epoch;
    logic idle, append_allowed, root_allowed, identity_ok, bind_ok, append_ok;
    assign idle=!maintenance_busy && !sequencer_cache_lock &&
        !engine_lease_active && !engine_done_valid;
    // Maintenance has deterministic priority: bind > append > root.
    assign engine_bind_valid=idle && bind_valid;
    assign bind_ready=idle && engine_bind_ready;
    assign append_allowed=idle && !bind_valid;
    assign engine_append_valid=append_allowed && append_valid;
    assign append_ready=append_allowed && engine_append_ready;
    assign root_allowed=idle && !bind_valid && !append_valid;
    assign sequencer_command_valid=root_allowed && command_valid;
    assign command_ready=root_allowed && sequencer_command_ready;
    assign trusted_cache_valid=authority_valid && !metadata_fault && engine_cache_valid &&
        trusted_sequence==engine_cache_sequence && trusted_epoch==engine_cache_epoch &&
        trusted_epoch==expected_epoch;
    assign trusted_committed_tokens=trusted_cache_valid && command_layer<16 ?
        {14'd0,prefixes[command_layer[3:0]]}:32'd0;
    assign identity_ok=engine_done_bind==maintenance_bind &&
        engine_done_sequence==held_sequence && engine_done_epoch==held_epoch;
    assign bind_ok=identity_ok && engine_cache_valid &&
        engine_cache_sequence==held_sequence && engine_cache_epoch==held_epoch &&
        engine_cache_capacity>0 && engine_cache_capacity<=131072 &&
        held_sequence!=0 && held_epoch!=0 && held_epoch==expected_epoch &&
        !maintenance_revoked && expected_epoch==last_epoch &&
        engine_done_bytes==0 && engine_done_committed_tokens==0;
    assign append_ok=identity_ok && trusted_cache_valid && !maintenance_revoked &&
        held_sequence==trusted_sequence && held_epoch==trusted_epoch &&
        held_layer<16 && held_position<engine_cache_capacity &&
        held_position=={14'd0,prefixes[held_layer[3:0]]} &&
        engine_done_committed_tokens==held_position+32'd1 && engine_done_bytes==4096;

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            authority_valid<=0;maintenance_busy<=0;maintenance_bind<=0;
            maintenance_revoked<=0;metadata_fault<=0;held_layer<=0;held_position<=0;
            held_sequence<=0;held_epoch<=0;trusted_sequence<=0;trusted_epoch<=0;last_epoch<=0;
            for(integer i=0;i<16;i=i+1)prefixes[i]<=0;
        end else begin
            last_epoch<=expected_epoch;
            if(expected_epoch!=last_epoch)begin
                authority_valid<=0;
                if(maintenance_busy)maintenance_revoked<=1;
            end
            if(engine_done_valid && !maintenance_busy)begin
                authority_valid<=0;metadata_fault<=1;
            end
            if(engine_bind_valid && engine_bind_ready)begin
                maintenance_busy<=1;maintenance_bind<=1;authority_valid<=0;
                held_sequence<=bind_sequence;held_epoch<=bind_epoch;
                maintenance_revoked<=bind_epoch!=expected_epoch;
                for(integer i=0;i<16;i=i+1)prefixes[i]<=0;
            end else if(engine_append_valid && engine_append_ready)begin
                maintenance_busy<=1;maintenance_bind<=0;
                held_layer<=append_layer;held_position<=append_position;
                held_sequence<=append_sequence;held_epoch<=append_epoch;
                maintenance_revoked<=append_epoch!=expected_epoch;
            end
            // Metadata becomes visible only when the engine completion is
            // consumed. A held response continues to exclude root admission.
            if(maintenance_busy && engine_done_valid && engine_done_ready)begin
                maintenance_busy<=0;
                if(!identity_ok)begin authority_valid<=0;metadata_fault<=1;end
                else if(maintenance_bind)begin
                    if(engine_done_status==0 && bind_ok)begin
                        authority_valid<=1;metadata_fault<=0;
                        trusted_sequence<=held_sequence;trusted_epoch<=held_epoch;
                    end else begin
                        authority_valid<=0;
                        if(engine_done_status==0)metadata_fault<=1;
                    end
                end else if(engine_done_status==0)begin
                    if(append_ok && expected_epoch==last_epoch)
                        prefixes[held_layer[3:0]]<=engine_done_committed_tokens[17:0];
                    else begin authority_valid<=0;metadata_fault<=1;end
                end
            end
        end
    end
endmodule
`default_nettype wire
