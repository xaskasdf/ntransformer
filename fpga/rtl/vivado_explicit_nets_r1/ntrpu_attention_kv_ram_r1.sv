// Private K/V tile storage: 512-bit fill, two independent 32-bit synchronous
// read ports. Ready means a complete matching KV transaction succeeded.
// Consumer must drain on compute_abort and acknowledge finish before reuse.
// Descriptor metadata must be owned by the query/lease controller. This block
// does not grant KV leases or authenticate arbitrary scheduler descriptors.
`default_nettype none
module ntrpu_attention_kv_ram_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire logic clk,rst_n,
    input wire logic cmd_valid,
    output logic cmd_ready,load_prepared,
    input wire logic [31:0] cmd_start,cmd_tokens,cmd_position,cmd_layer,cmd_query_head,
    input wire logic [63:0] cmd_sequence,cmd_epoch,cmd_lease,expected_epoch,
    input wire logic tile_valid,
    output logic tile_ready,
    input wire logic [511:0] tile_data,
    input wire logic tile_plane,tile_last,
    input wire logic [7:0] tile_index,
    input wire logic [63:0] tile_sequence,tile_epoch,tile_lease,
    input wire logic load_done_valid,
    output logic load_done_ready,
    input wire logic [3:0] load_done_status,
    input wire logic [9:0] load_done_beats,
    input wire logic [15:0] load_done_bytes,
    input wire logic [63:0] load_done_sequence,load_done_epoch,load_done_lease,
    input wire logic cancel_valid,
    output logic cancel_ready,upstream_cancel_valid,
    input wire logic upstream_cancel_ready,
    output logic compute_valid,
    input wire logic compute_ready,
    output logic compute_abort,
    output logic [31:0] compute_start,compute_tokens,compute_position,compute_layer,compute_query_head,compute_kv_head,
    output logic [63:0] compute_sequence,compute_epoch,compute_lease,
    input wire logic keys_ce,values_ce,
    input wire logic [11:0] keys_address,values_address,
    output logic [31:0] keys_q,values_q,
    output logic keys_q_valid,values_q_valid,
    input wire logic finish_valid,
    output logic finish_ready,
    input wire logic [3:0] finish_status,
    output logic done_valid,
    input wire logic done_ready,
    output logic [3:0] done_status,
    output logic [9:0] done_beats,
    output logic done_consumer_started,
    output logic [63:0] done_sequence,done_epoch,done_lease
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_FORMAT=4,CANCELLED=7,TIMEOUT=8,BAD_RAM_ACCESS=9;
    typedef enum logic [2:0] {IDLE,COLLECT,AVAILABLE,EXECUTE,RESPONSE} state_t;
    state_t state;
    logic [3:0] fault_now;
    logic [9:0] received;
    logic [10:0] received_after,expected_beats;
    logic [8:0] plane_beats;
    logic [12:0] valid_words;
    logic [31:0] idle_cycles;
    logic tile_fire,load_fire,bad_tile,expected_plane,watchdog_expired,upstream_cancel_seen;
    logic [8:0] expected_index;
    logic write_enable,k_read_enable,v_read_enable,k_valid,v_valid;
    logic [3:0] k_lane,v_lane;
    logic [31:0] key_bank_q[0:15],value_bank_q[0:15];

    assign cmd_ready=state==IDLE;
    assign load_prepared=state==COLLECT;
    assign tile_ready=state==COLLECT;
    assign load_done_ready=state==COLLECT;
    assign tile_fire=tile_valid && tile_ready;
    assign load_fire=load_done_valid && load_done_ready;
    assign received_after={1'b0,received}+11'(tile_fire);
    assign plane_beats={compute_tokens[6:0],2'b00};
    assign expected_beats={1'b0,plane_beats,1'b0};
    assign valid_words={compute_tokens[6:0],6'b0};
    assign expected_plane={1'b0,received}>={2'b0,plane_beats};
    assign expected_index=expected_plane ? received[8:0]-plane_beats:received[8:0];
    assign cancel_ready=(state==COLLECT || state==AVAILABLE || state==EXECUTE) && done_status==OK;
    assign watchdog_expired=state==COLLECT && idle_cycles>=32'(WATCHDOG_CYCLES-1) && !tile_fire && !load_fire;
    assign upstream_cancel_valid=state==COLLECT && fault_now!=OK && !upstream_cancel_seen;
    assign compute_valid=state==AVAILABLE && fault_now==OK;
    assign compute_abort=state==EXECUTE && fault_now!=OK;
    assign finish_ready=state==EXECUTE;
    assign done_valid=state==RESPONSE;
    assign compute_kv_head=compute_query_head>>2;
    assign done_sequence=compute_sequence;
    assign done_epoch=compute_epoch;
    assign done_lease=compute_lease;
    assign done_beats=received;

    always_comb begin
        bad_tile=received_after>expected_beats || tile_plane!=expected_plane ||
            {1'b0,tile_index}!=expected_index || tile_last!=(received_after==expected_beats) ||
            tile_sequence!=compute_sequence || tile_epoch!=compute_epoch || tile_lease!=compute_lease;
        for(int lane=0;lane<16;lane++) bad_tile |= tile_data[lane*32+23+:8]==8'hff;
        fault_now=done_status;
        if(done_status==OK && (state==COLLECT || state==AVAILABLE || state==EXECUTE))begin
            if(compute_epoch!=expected_epoch) fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready) fault_now=CANCELLED;
            else if(watchdog_expired) fault_now=TIMEOUT;
            else if(tile_fire && bad_tile) fault_now=BAD_FORMAT;
            else if(load_fire)begin
                if(load_done_sequence!=compute_sequence || load_done_epoch!=compute_epoch || load_done_lease!=compute_lease)
                    fault_now=BAD_IDENTITY;
                else if(load_done_status!=OK) fault_now=load_done_status;
                else if(received_after!=expected_beats || {1'b0,load_done_beats}!=received_after ||
                    load_done_bytes!={load_done_beats,6'b0}) fault_now=BAD_FORMAT;
            end else if(state==EXECUTE && ((keys_ce && {1'b0,keys_address}>=valid_words) ||
                (values_ce && {1'b0,values_address}>=valid_words))) fault_now=BAD_RAM_ACCESS;
        end
    end
    assign write_enable=tile_fire && fault_now==OK;
    assign k_read_enable=state==EXECUTE && fault_now==OK && keys_ce;
    assign v_read_enable=state==EXECUTE && fault_now==OK && values_ce;
    // Each lane is 512 x 32: K in the lower half, V in the upper half.
    // Port A fills or reads K; port B reads V. Fill/read phases never overlap.
    generate for(genvar lane=0;lane<16;lane++)begin: banks
        (* ram_style="block" *) logic [31:0] memory[0:511];
        always_ff @(posedge clk)begin
            if(write_enable) memory[{tile_plane,tile_index}]<=tile_data[lane*32+:32];
            else if(k_read_enable && keys_address[3:0]==4'(lane))
                key_bank_q[lane]<=memory[{1'b0,keys_address[11:4]}];
        end
        always_ff @(posedge clk)begin
            if(v_read_enable && values_address[3:0]==4'(lane))
                value_bank_q[lane]<=memory[{1'b1,values_address[11:4]}];
        end
    end endgenerate
    assign keys_q_valid=k_valid && state==EXECUTE && fault_now==OK;
    assign values_q_valid=v_valid && state==EXECUTE && fault_now==OK;
    assign keys_q=keys_q_valid ? key_bank_q[k_lane]:32'b0;
    assign values_q=values_q_valid ? value_bank_q[v_lane]:32'b0;

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;received<=0;done_status<=OK;done_consumer_started<=0;
            compute_start<=0;compute_tokens<=0;compute_position<=0;compute_layer<=0;compute_query_head<=0;
            compute_sequence<=0;compute_epoch<=0;compute_lease<=0;
            idle_cycles<=0;upstream_cancel_seen<=0;k_lane<=0;v_lane<=0;k_valid<=0;v_valid<=0;
        end else begin
            k_valid<=k_read_enable;v_valid<=v_read_enable;
            if(k_read_enable)k_lane<=keys_address[3:0];
            if(v_read_enable)v_lane<=values_address[3:0];
            if(state!=IDLE && state!=RESPONSE)done_status<=fault_now;
            if(state==COLLECT)begin
                if(tile_fire || load_fire)idle_cycles<=0;
                else if(idle_cycles<32'(WATCHDOG_CYCLES))idle_cycles<=idle_cycles+32'd1;
                if(tile_fire && received<1023)received<=received+10'd1;
            end else idle_cycles<=0;
            if(upstream_cancel_valid && upstream_cancel_ready)upstream_cancel_seen<=1;
            case(state)
                IDLE: if(cmd_valid)begin
                    compute_start<=cmd_start;compute_tokens<=cmd_tokens;compute_position<=cmd_position;
                    compute_layer<=cmd_layer;compute_query_head<=cmd_query_head;
                    compute_sequence<=cmd_sequence;compute_epoch<=cmd_epoch;compute_lease<=cmd_lease;
                    received<=0;done_status<=OK;done_consumer_started<=0;upstream_cancel_seen<=0;
                    if(cmd_tokens==0 || cmd_tokens>64 || cmd_position>=131072 || cmd_layer>=16 || cmd_query_head>=32 ||
                        {1'b0,cmd_start}+{1'b0,cmd_tokens}>{1'b0,cmd_position}+33'd1)begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(cmd_sequence==0 || cmd_epoch==0 || cmd_lease==0 || cmd_epoch!=expected_epoch)begin
                        done_status<=BAD_IDENTITY;state<=RESPONSE;
                    end else state<=COLLECT;
                end
                COLLECT: if(load_fire)begin
                    if(fault_now==OK)state<=AVAILABLE;else state<=RESPONSE;
                end
                AVAILABLE:begin
                    if(fault_now!=OK)state<=RESPONSE;
                    else if(compute_ready)begin done_consumer_started<=1;state<=EXECUTE;end
                end
                EXECUTE: if(finish_valid)begin
                    if(fault_now==OK)done_status<=finish_status;
                    state<=RESPONSE;
                end
                RESPONSE: if(done_ready)state<=IDLE;
                default:state<=IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
