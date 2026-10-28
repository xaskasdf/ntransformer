// Retain the actual input-normalization/QKV completion through RoPE/attention.
// The parent captures all ten allocations from its producer command/phase table.
// Allocation metadata is a trusted parent contract, not memory authentication.
// A coordinated reset must quiesce producer, consumer and all AXI peers.
`default_nettype none
module ntrpu_qkv_rope_handoff_r1 #(
    parameter integer CHILD_TIMEOUT_CYCLES=67108864
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,command_lease_base,expected_epoch,
    input wire [31:0] command_layer,command_position,command_coefficient_position,
    input wire [63:0] command_addr [0:9],
    input wire [63:0] command_base [0:9],
    input wire [63:0] command_bytes [0:9],
    input wire cancel_valid,
    output wire cancel_ready,busy,cache_exclusive,
    input wire cache_idle,cache_valid,cache_metadata_fault,
    input wire [63:0] cache_base,cache_region_base,cache_region_bytes,cache_sequence,cache_epoch,
    input wire [31:0] cache_capacity,cache_committed_tokens,
    input wire source_busy,source_completion_valid,
    output wire source_completion_ready,
    input wire [63:0] source_sequence,source_epoch,
    input wire [3:0] source_status,source_norm_status,source_qkv_status,source_matrix_status,
    input wire [31:0] source_hls_status,
    input wire [9:0] source_norm_read_beats,
    input wire [6:0] source_norm_records,
    input wire [1:0] source_phase,source_matrices,
    input wire [14:0] source_total_bytes,
    input wire source_result_valid,source_revoked,
    input wire [63:0] source_result_address [0:2],
    input wire [13:0] source_result_bytes [0:2],
    output wire child_command_valid,
    input wire child_command_ready,child_busy,
    output wire [63:0] child_command_sequence,child_command_epoch,child_command_lease_base,
    output wire [31:0] child_command_layer,child_command_position,child_command_coefficient_position,
    output wire [63:0] child_command_source_sequence,child_command_source_epoch,
    output wire child_command_source_valid,
    output wire [31:0] child_command_source_position,
    output wire [63:0] child_command_v_source_sequence,child_command_v_source_epoch,
    output wire child_command_v_source_valid,
    output wire [31:0] child_command_v_source_position,
    output wire [63:0] child_command_addr [0:3],
    output wire [63:0] child_command_base [0:3],
    output wire [63:0] child_command_bytes [0:3],
    output wire child_cancel_valid,
    input wire child_cancel_ready,
    input wire child_completion_valid,
    output wire child_completion_ready,
    input wire [63:0] child_sequence,child_epoch,
    input wire [31:0] child_layer,child_position,
    input wire [3:0] child_status,child_append_status,child_root_status,child_capture_status,child_rope_status,
    input wire [31:0] child_rope_hls_status,
    input wire [9:0] child_rope_beats,
    input wire [11:0] child_capture_qk_words,
    input wire [5:0] child_capture_v_beats,
    input wire [6:0] child_kv_frames,
    input wire [7:0] child_q_frames,
    input wire [12:0] child_kv_bytes,
    input wire [31:0] child_kv_tokens,
    input wire child_kv_committed,child_result_valid,child_revoked,
    output wire completion_valid,
    input wire completion_ready,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [31:0] completion_layer,completion_position,
    output logic [3:0] completion_status,completion_source_status,completion_child_status,
    output logic completion_kv_committed,
    output logic [12:0] completion_kv_bytes,
    output logic [31:0] completion_kv_tokens,
    output wire completion_revoked,result_valid,
    input wire result_ce,
    input wire [10:0] result_address,
    output wire child_result_ce,
    output wire [10:0] child_result_address,
    input wire [31:0] child_result_q,
    input wire child_result_q_valid,
    output wire [31:0] result_q,
    output wire result_q_valid
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_COMPLETION=5,
        CANCELLED=7,TIMED_OUT=8,BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,CHECK_RANGES,CHECK_ALIASES,OFFER,WAIT_CHILD,
        RESPONSE,DRAIN_CHILD,WAIT_CHILD_IDLE,DRAIN_SOURCE,WAIT_SOURCE_IDLE} state_t;
    state_t state;
    logic [9:0][63:0] address,region_base,region_size;
    logic [3:0] range_index,write_index,other_index;
    logic [63:0] lease,cache_saved_base,cache_saved_region,cache_saved_bytes;
    logic [31:0] capacity,prefix,coefficient_position,watchdog;
    logic source_owned,child_owned,child_seen,cancel_sent,revoked;
    logic [3:0] fault,fault_now;
    logic [319:0] saved_source;
    logic [142:0] saved_child;
    wire [319:0] source_snapshot={source_status,source_norm_status,source_qkv_status,source_matrix_status,
        source_hls_status,source_norm_read_beats,source_norm_records,source_phase,source_matrices,source_total_bytes,
        source_result_valid,source_revoked,source_result_address[0],source_result_address[1],source_result_address[2],
        source_result_bytes[0],source_result_bytes[1],source_result_bytes[2]};
    wire [142:0] child_snapshot={child_status,child_append_status,child_root_status,child_capture_status,child_rope_status,
        child_rope_hls_status,child_rope_beats,child_capture_qk_words,child_capture_v_beats,child_kv_frames,
        child_q_frames,child_kv_bytes,child_kv_tokens,child_kv_committed,child_result_valid,child_revoked};
    wire source_identity=source_sequence==completion_sequence && source_epoch==completion_epoch;
    wire child_identity=child_sequence==completion_sequence && child_epoch==completion_epoch &&
        child_layer==completion_layer && child_position==completion_position;
    wire source_success=source_status==OK && source_norm_status==OK && source_qkv_status==OK &&
        source_matrix_status==OK && source_hls_status==0 && source_norm_read_beats==256 &&
        source_norm_records==64 && source_phase==2 && source_matrices==3 && source_total_bytes==12288 &&
        source_result_valid && !source_revoked && source_result_address[0]==address[6] &&
        source_result_address[1]==address[7] && source_result_address[2]==address[8] &&
        source_result_bytes[0]==8192 && source_result_bytes[1]==2048 && source_result_bytes[2]==2048;
    wire child_success=child_status==OK && child_append_status==OK && child_root_status==OK &&
        child_capture_status==OK && child_rope_status==OK && child_rope_hls_status==0 &&
        child_rope_beats==164 && child_capture_qk_words==2560 && child_capture_v_beats==32 &&
        child_kv_frames==64 && child_q_frames==128 && child_kv_bytes==4096 &&
        child_kv_tokens==completion_position+1 && child_kv_committed && child_result_valid && !child_revoked;
    wire source_bad=source_owned && (!source_completion_valid || !source_identity || source_snapshot!=saved_source);
    wire child_bad=child_owned && ((child_completion_valid && !child_identity) ||
        (child_seen && (!child_completion_valid || child_snapshot!=saved_child)));
    wire cache_bad=busy && (!cache_valid || cache_metadata_fault || cache_sequence!=completion_sequence ||
        cache_epoch!=completion_epoch || cache_base!=cache_saved_base || cache_region_base!=cache_saved_region ||
        cache_region_bytes!=cache_saved_bytes || cache_capacity!=capacity ||
        (!child_owned && !child_seen && cache_committed_tokens!=prefix));
    wire waiting=state==OFFER || state==WAIT_CHILD;
    wire progress=(state==OFFER && child_command_ready) ||
        (state==WAIT_CHILD && child_completion_valid && child_identity);
    wire timed_out=waiting && watchdog>=32'(CHILD_TIMEOUT_CYCLES-1) && !progress;
    wire [64:0] cache_limit={1'b0,cache_saved_region}+{1'b0,cache_saved_bytes};
    wire [64:0] cache_end={1'b0,cache_saved_base}+{17'b0,capacity,16'b0};
    wire [64:0] range_end={1'b0,region_base[range_index]}+{1'b0,region_size[range_index]};
    wire cache_overlap={1'b0,region_base[range_index]}<cache_limit && {1'b0,cache_saved_region}<range_end;
    wire [3:0] write_slot=write_index==0 ? 4'd2:write_index+4'd5;
    wire [64:0] write_end={1'b0,region_base[write_slot]}+{1'b0,region_size[write_slot]};
    wire [64:0] other_end={1'b0,region_base[other_index]}+{1'b0,region_size[other_index]};
    wire alias_write=write_slot!=other_index && {1'b0,region_base[write_slot]}<other_end &&
        {1'b0,region_base[other_index]}<write_end;

    function automatic logic [63:0] span(input logic [3:0] index);
        case(index)
            0,1,6:return 64'd8192;
            2:return 64'd4096;
            3:return 64'd4456448;
            4,5:return 64'd1114112;
            9:return 64'd256;
            default:return 64'd2048;
        endcase
    endfunction
    function automatic logic valid_span(input logic [63:0] addr,base,bytes,needed);
        logic [64:0] last,limit;
        last={1'b0,addr}+{1'b0,needed};limit={1'b0,base}+{1'b0,bytes};
        return addr[5:0]==0 && base[5:0]==0 && bytes[5:0]==0 && bytes!=0 &&
            !last[64] && !limit[64] && addr>=base && last<=limit;
    endfunction
    assign busy=state!=IDLE;
    assign cache_exclusive=busy;
    assign command_ready=state==IDLE && source_completion_valid && cache_idle && !child_busy && !child_completion_valid;
    assign cancel_ready=busy && state!=DRAIN_SOURCE && state!=WAIT_SOURCE_IDLE && fault==OK && !revoked;
    assign child_command_valid=state==OFFER && fault_now==OK;
    assign child_command_sequence=completion_sequence;
    assign child_command_epoch=completion_epoch;
    assign child_command_layer=completion_layer;
    assign child_command_position=completion_position;
    assign child_command_coefficient_position=coefficient_position;
    assign child_command_lease_base=lease;
    assign child_command_source_sequence=completion_sequence;
    assign child_command_source_epoch=completion_epoch;
    assign child_command_source_position=completion_position;
    assign child_command_source_valid=source_owned && !source_bad && source_success && fault_now==OK;
    assign child_command_v_source_sequence=completion_sequence;
    assign child_command_v_source_epoch=completion_epoch;
    assign child_command_v_source_position=completion_position;
    assign child_command_v_source_valid=child_command_source_valid;
    for(genvar i=0;i<4;i++)begin
        assign child_command_addr[i]=address[i==3 ? 9:i+6];
        assign child_command_base[i]=region_base[i==3 ? 9:i+6];
        assign child_command_bytes[i]=region_size[i==3 ? 9:i+6];
    end
    assign child_cancel_valid=child_owned && fault_now!=OK && !cancel_sent && !child_completion_valid;
    assign child_completion_ready=state==DRAIN_CHILD && child_owned && child_identity && child_completion_valid;
    assign source_completion_ready=state==DRAIN_SOURCE && source_owned && source_identity && source_completion_valid && !child_busy;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked;
    assign result_valid=completion_valid && completion_status==OK && !revoked && fault_now==OK &&
        source_owned && child_owned && !source_bad && !child_bad && child_result_valid;
    assign child_result_ce=result_ce && result_valid;
    assign child_result_address=result_address;
    assign result_q=child_result_q;
    assign result_q_valid=child_result_q_valid && result_valid;
    always_comb begin
        fault_now=fault;
        if(busy && fault==OK)begin
            if(expected_epoch!=completion_epoch)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(source_bad || child_bad || cache_bad)fault_now=BAD_PEER;
            else if(timed_out)fault_now=TIMED_OUT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;region_base<=0;region_size<=0;range_index<=0;write_index<=0;other_index<=0;
            lease<=0;cache_saved_base<=0;cache_saved_region<=0;cache_saved_bytes<=0;capacity<=0;prefix<=0;
            coefficient_position<=0;watchdog<=0;source_owned<=0;child_owned<=0;child_seen<=0;cancel_sent<=0;
            revoked<=0;fault<=0;saved_source<=0;saved_child<=0;completion_sequence<=0;completion_epoch<=0;
            completion_layer<=0;completion_position<=0;completion_status<=0;completion_source_status<=0;
            completion_child_status<=0;completion_kv_committed<=0;completion_kv_bytes<=0;completion_kv_tokens<=0;
        end else begin
            if(waiting && !progress && fault==OK)watchdog<=watchdog+1;else watchdog<=0;
            if(busy && fault_now!=OK)begin fault<=fault_now;revoked<=1;end
            if(child_cancel_valid && child_cancel_ready)cancel_sent<=1;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_layer<=command_layer;completion_position<=command_position;
                    coefficient_position<=command_coefficient_position;lease<=command_lease_base;
                    for(integer i=0;i<10;i++)begin address[i]<=command_addr[i];region_base[i]<=command_base[i];region_size[i]<=command_bytes[i];end
                    cache_saved_base<=cache_base;cache_saved_region<=cache_region_base;cache_saved_bytes<=cache_region_bytes;
                    capacity<=cache_capacity;prefix<=cache_committed_tokens;saved_source<=source_snapshot;
                    completion_source_status<=source_status;completion_child_status<=0;
                    completion_kv_committed<=0;completion_kv_bytes<=0;completion_kv_tokens<=0;completion_status<=0;
                    source_owned<=1;child_owned<=0;child_seen<=0;cancel_sent<=0;revoked<=0;fault<=0;
                    range_index<=0;write_index<=0;other_index<=0;
                    if(command_epoch!=expected_epoch || source_sequence!=command_sequence || source_epoch!=command_epoch ||
                        cache_sequence!=command_sequence || cache_epoch!=command_epoch)begin
                        completion_status<=BAD_IDENTITY;fault<=BAD_IDENTITY;state<=RESPONSE;
                    end else state<=CHECK_RANGES;
                end
                CHECK_RANGES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!cache_valid || cache_metadata_fault || capacity==0 || capacity>131072 || completion_layer>=16 ||
                        completion_position>=capacity || completion_position!=prefix || coefficient_position!=completion_position ||
                        lease==0 || lease>64'hffffffffffffffe0 || !valid_span(cache_saved_base,cache_saved_region,cache_saved_bytes,64'd64) ||
                        cache_end[64] || cache_end>cache_limit || !valid_span(address[range_index],region_base[range_index],region_size[range_index],span(range_index)) || cache_overlap)begin
                        completion_status<=BAD_COMMAND;fault<=BAD_COMMAND;state<=RESPONSE;
                    end else if(source_status!=OK || !source_success)begin
                        completion_status<=source_status==OK ? BAD_COMPLETION:source_status;
                        fault<=source_status==OK ? BAD_COMPLETION:source_status;state<=RESPONSE;
                    end else if(range_index==9)state<=CHECK_ALIASES;
                    else range_index<=range_index+1;
                end
                CHECK_ALIASES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(alias_write)begin completion_status<=BAD_COMMAND;fault<=BAD_COMMAND;state<=RESPONSE;end
                    else if(other_index==9)begin
                        other_index<=0;
                        if(write_index==3)state<=OFFER;else write_index<=write_index+1;
                    end else other_index<=other_index+1;
                end
                OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(child_command_ready)begin child_owned<=1;state<=WAIT_CHILD;end
                WAIT_CHILD:if(child_completion_valid && child_identity)begin
                    saved_child<=child_snapshot;child_seen<=1;completion_child_status<=child_status;
                    completion_kv_committed<=child_kv_committed;completion_kv_bytes<=child_kv_bytes;completion_kv_tokens<=child_kv_tokens;
                    if(fault_now!=OK)completion_status<=fault_now;
                    else if(child_status!=OK)completion_status<=child_status;
                    else if(!child_success || cache_committed_tokens!=completion_position+1)completion_status<=BAD_COMPLETION;
                    else completion_status<=OK;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=DRAIN_CHILD;
                DRAIN_CHILD:if(!child_owned || child_completion_ready)begin child_owned<=0;state<=WAIT_CHILD_IDLE;end
                WAIT_CHILD_IDLE:if(!child_busy && !child_completion_valid)state<=DRAIN_SOURCE;
                DRAIN_SOURCE:if(source_completion_ready)begin source_owned<=0;state<=WAIT_SOURCE_IDLE;end
                WAIT_SOURCE_IDLE:if(!source_busy && !source_completion_valid)state<=IDLE;
                default:begin fault<=BAD_PEER;revoked<=1;state<=DRAIN_CHILD;end
            endcase
        end
    end
endmodule
`default_nettype wire
