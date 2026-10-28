// Own a complete Q/K/V snapshot, commit K/V, then stream Q to attention.
// The enclosing scheduler must exclude all other cache bind/append/root users
// while cache_exclusive is asserted, including the final public response/drain.
// Snapshot, append and root peers retain their responses until matching ACK.
// Offered command/data payloads remain stable until accepted or the owning child
// terminates. Reset must quiesce this block, both children, source and AXI together.
// A successful KV append is not rolled back by later cancellation/root failure.
`default_nettype none
module ntrpu_qkv_dispatch_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,command_lease_base,expected_epoch,
    input wire [31:0] command_layer,command_position,
    input wire cancel_valid,
    output wire cancel_ready,busy,cache_exclusive,
    input wire cache_idle,cache_valid,cache_metadata_fault,
    input wire [63:0] cache_sequence,cache_epoch,
    input wire [31:0] cache_capacity,cache_committed_tokens,
    input wire source_completion_valid,
    output wire source_completion_ready,
    input wire [3:0] source_status,
    input wire [63:0] source_sequence,source_epoch,
    input wire [31:0] source_layer,source_position,
    input wire [11:0] source_qk_words,
    input wire [5:0] source_v_beats,
    input wire [7:0] source_frames,
    input wire source_result_valid,
    output wire source_ce,
    output wire [7:0] source_address,
    input wire [511:0] source_q,
    input wire source_q_valid,
    output wire append_valid,
    input wire append_ready,append_prepared,
    output wire [31:0] append_layer,append_position,
    output wire [63:0] append_sequence,append_epoch,
    output wire data_valid,
    input wire data_ready,
    output wire [511:0] data_bits,
    output wire [5:0] data_index,
    output wire data_last,
    output wire [63:0] data_sequence,data_epoch,
    output wire commit_valid,
    input wire commit_ready,
    output wire [3:0] commit_status,
    output wire [63:0] commit_sequence,commit_epoch,
    output wire append_cancel_valid,
    input wire append_cancel_ready,
    input wire append_done_valid,
    output wire append_done_ready,
    input wire append_done_bind,
    input wire [3:0] append_done_status,
    input wire [12:0] append_done_bytes,
    input wire [31:0] append_done_committed_tokens,
    input wire [63:0] append_done_sequence,append_done_epoch,
    output wire root_command_valid,
    input wire root_command_ready,
    output wire [31:0] root_command_layer,root_command_position,
    output wire [63:0] root_command_sequence,root_command_epoch,root_command_lease_base,
    output wire q_valid,
    input wire q_ready,
    output wire [511:0] q_data,
    output wire [7:0] q_index,
    output wire q_last,
    output wire [63:0] q_sequence,q_epoch,
    output wire root_cancel_valid,
    input wire root_cancel_ready,
    input wire root_completion_valid,
    output wire root_completion_ready,
    input wire [3:0] root_status,
    input wire [63:0] root_sequence,root_epoch,
    input wire root_result_valid,
    output wire root_result_ce,
    output wire [10:0] root_result_address,
    input wire [31:0] root_result_q,
    input wire root_result_q_valid,
    output wire completion_valid,
    input wire completion_ready,
    output logic [3:0] completion_status,completion_append_status,completion_root_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [31:0] completion_layer,completion_position,
    output logic [6:0] completion_kv_frames,
    output logic [7:0] completion_q_frames,
    output logic [12:0] completion_kv_bytes,
    output logic [31:0] completion_kv_tokens,
    output logic completion_kv_committed,
    output wire result_valid,completion_revoked,
    input wire result_ce,
    input wire [10:0] result_address,
    output wire [31:0] result_q,
    output wire result_q_valid
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_FORMAT=4,
        BAD_COMPLETION=5,CANCELLED=7,TIMEOUT=8,BAD_PEER=11;
    typedef enum logic [4:0] {IDLE,CHECK,KV_OFFER,KV_START,RAM_REQ,RAM_WAIT,
        STREAM,KV_COMMIT,KV_WAIT,KV_ACK,KV_VERIFY,ROOT_OFFER,ROOT_START,
        ROOT_WAIT,RESPONSE,DRAIN} state_t;
    state_t state;
    logic source_owned,append_owned,root_owned,root_seen,revoked,cancel_seen;
    logic append_cancel_sent,root_cancel_sent,q_phase;
    logic [3:0] fault,fault_now,admission_status,saved_source_status,commit_code;
    logic [11:0] saved_qk_words;
    logic [5:0] saved_v_beats;
    logic [7:0] saved_frames,index;
    logic [63:0] lease_base;
    logic [31:0] saved_capacity,idle_cycles;
    logic [511:0] frame;
    logic finite_frame;
    wire source_identity=source_sequence==completion_sequence && source_epoch==completion_epoch &&
        source_layer==completion_layer && source_position==completion_position;
    wire append_identity=!append_done_bind && append_done_sequence==completion_sequence && append_done_epoch==completion_epoch;
    wire root_identity=root_sequence==completion_sequence && root_epoch==completion_epoch;
    wire source_bad=source_owned && (!source_completion_valid || !source_identity ||
        source_status!=saved_source_status || source_qk_words!=saved_qk_words ||
        source_v_beats!=saved_v_beats || source_frames!=saved_frames ||
        (saved_source_status==OK && !source_result_valid));
    wire epoch_bad=busy && completion_epoch!=expected_epoch;
    wire cache_bad=busy && (!cache_valid || cache_metadata_fault ||
        cache_sequence!=completion_sequence || cache_epoch!=completion_epoch || cache_capacity!=saved_capacity);
    wire root_bad=root_owned && ((root_completion_valid && !root_identity) ||
        (root_seen && (!root_completion_valid || root_status!=completion_root_status ||
        (completion_root_status==OK && !root_result_valid))));
    wire append_held_stable=append_done_valid && append_identity &&
        append_done_status==completion_append_status && append_done_bytes==completion_kv_bytes &&
        append_done_committed_tokens==completion_kv_tokens;
    wire append_bad=append_owned && ((append_done_valid && !append_identity) ||
        (state==KV_ACK && !append_held_stable));
    wire active=busy && state!=RESPONSE && state!=DRAIN;
    wire progress_event=(append_valid && append_ready) || (data_valid && data_ready) ||
        (commit_valid && commit_ready) || (append_done_valid && append_done_ready) ||
        (root_command_valid && root_command_ready) || (q_valid && q_ready) ||
        (state==RAM_WAIT && source_q_valid);
    // Children own their operation lifetime. Never abandon accepted append/root
    // operations merely because physical memory or arithmetic takes a long time.
    wire waiting=state==RAM_WAIT || state==STREAM || state==KV_OFFER || state==ROOT_OFFER || state==KV_START || state==ROOT_START || state==KV_COMMIT;
    wire timeout_now=waiting && !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;
    wire stream_fire=(data_valid && data_ready) || (q_valid && q_ready);
    wire append_success=append_done_status==OK && append_done_bytes==4096 &&
        append_done_committed_tokens==completion_position+1 && completion_kv_frames==64;
    assign busy=state!=IDLE;
    assign cache_exclusive=busy;
    assign command_ready=state==IDLE && cache_idle && !append_done_valid && !root_completion_valid;
    assign cancel_ready=busy && state!=DRAIN && !cancel_seen;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked || epoch_bad || (busy && cancel_valid) || source_bad || cache_bad || root_bad;
    assign result_valid=completion_valid && completion_status==OK && root_owned && root_seen &&
        root_completion_valid && root_identity && root_result_valid && root_status==OK && !completion_revoked;
    assign root_result_ce=result_ce && result_valid;
    assign root_result_address=result_address;
    assign result_q_valid=root_result_q_valid && result_valid;
    assign result_q=result_q_valid ? root_result_q:32'd0;
    assign source_ce=state==RAM_REQ && fault_now==OK;
    assign source_address=q_phase ? index:8'd128+index;
    assign source_completion_ready=state==DRAIN && source_owned && source_completion_valid && source_identity && !append_owned && !root_owned;
    assign append_valid=state==KV_OFFER;
    assign append_layer=completion_layer;
    assign append_position=completion_position;
    assign append_sequence=completion_sequence;
    assign append_epoch=completion_epoch;
    assign data_valid=state==STREAM && !q_phase;
    assign data_bits=frame;
    assign data_index=index[5:0];
    assign data_last=index==63;
    assign data_sequence=completion_sequence;
    assign data_epoch=completion_epoch;
    assign commit_valid=state==KV_COMMIT;
    assign commit_status=commit_code;
    assign commit_sequence=completion_sequence;
    assign commit_epoch=completion_epoch;
    assign append_cancel_valid=append_owned && fault!=OK && !append_cancel_sent && !append_done_valid;
    assign append_done_ready=state==KV_ACK && append_owned && append_held_stable;
    assign root_command_valid=state==ROOT_OFFER;
    // The guard's committed_tokens port selects the command layer even while idle.
    assign root_command_layer=state==IDLE ? command_layer:completion_layer;
    assign root_command_position=completion_position;
    assign root_command_sequence=completion_sequence;
    assign root_command_epoch=completion_epoch;
    assign root_command_lease_base=lease_base;
    assign q_valid=state==STREAM && q_phase;
    assign q_data=frame;
    assign q_index=index;
    assign q_last=index==127;
    assign q_sequence=completion_sequence;
    assign q_epoch=completion_epoch;
    assign root_cancel_valid=root_owned && fault!=OK && !root_cancel_sent && !root_completion_valid;
    assign root_completion_ready=state==DRAIN && root_owned && root_completion_valid && root_identity;

    always_comb begin
        admission_status=OK;
        if(command_epoch!=expected_epoch || !cache_valid || cache_metadata_fault ||
            command_sequence!=cache_sequence || command_epoch!=cache_epoch)admission_status=BAD_IDENTITY;
        else if(command_layer>=16 || cache_capacity==0 || cache_capacity>131072 ||
            command_position>=cache_capacity || command_position!=cache_committed_tokens ||
            command_lease_base==0 || command_lease_base>64'hffffffffffffffe0)admission_status=BAD_COMMAND;
        else if(!source_completion_valid)admission_status=BAD_PEER;
        else if(source_sequence!=command_sequence || source_epoch!=command_epoch ||
            source_layer!=command_layer || source_position!=command_position)admission_status=BAD_IDENTITY;
        else if(source_status!=OK)admission_status=source_status;
        else if(cancel_valid)admission_status=CANCELLED;
        else if(!source_result_valid || source_qk_words!=2560 || source_v_beats!=32 || source_frames!=192)admission_status=BAD_PEER;
        finite_frame=1;
        for(integer lane=0;lane<16;lane++)if(source_q[lane*32+23+:8]==8'hff)finite_frame=0;
        fault_now=fault;
        if(active && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(source_bad || cache_bad || root_bad || append_bad)fault_now=BAD_PEER;
            else if(timeout_now)fault_now=TIMEOUT;
            else if(state==RAM_WAIT && source_q_valid && !finite_frame)fault_now=BAD_FORMAT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;source_owned<=0;append_owned<=0;root_owned<=0;root_seen<=0;
            revoked<=0;cancel_seen<=0;append_cancel_sent<=0;root_cancel_sent<=0;q_phase<=0;
            fault<=OK;saved_source_status<=0;commit_code<=0;saved_qk_words<=0;saved_v_beats<=0;saved_frames<=0;
            index<=0;lease_base<=0;saved_capacity<=0;idle_cycles<=0;frame<=0;
            completion_status<=0;completion_append_status<=0;completion_root_status<=0;
            completion_sequence<=0;completion_epoch<=0;completion_layer<=0;completion_position<=0;
            completion_kv_frames<=0;completion_q_frames<=0;completion_kv_bytes<=0;completion_kv_tokens<=0;completion_kv_committed<=0;
        end else begin
            if(active)fault<=fault_now;
            if(busy && (epoch_bad || cancel_valid || source_bad || cache_bad || root_bad))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(append_cancel_valid && append_cancel_ready)append_cancel_sent<=1;
            if(root_cancel_valid && root_cancel_ready)root_cancel_sent<=1;
            if(waiting)begin
                if(progress_event)idle_cycles<=0;
                else if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            if(source_completion_ready)source_owned<=0;
            if(root_completion_ready)root_owned<=0;
            if(stream_fire)begin
                if(q_phase)completion_q_frames<=completion_q_frames+1;
                else completion_kv_frames<=completion_kv_frames+1;
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_layer<=command_layer;completion_position<=command_position;lease_base<=command_lease_base;
                    saved_capacity<=cache_capacity;saved_source_status<=source_status;
                    saved_qk_words<=source_qk_words;saved_v_beats<=source_v_beats;saved_frames<=source_frames;
                    source_owned<=source_completion_valid && source_sequence==command_sequence && source_epoch==command_epoch &&
                        source_layer==command_layer && source_position==command_position;
                    append_owned<=0;root_owned<=0;root_seen<=0;append_cancel_sent<=0;root_cancel_sent<=0;
                    fault<=admission_status;completion_status<=admission_status;revoked<=admission_status!=OK;cancel_seen<=cancel_valid;
                    completion_append_status<=0;completion_root_status<=0;completion_kv_frames<=0;completion_q_frames<=0;
                    completion_kv_bytes<=0;completion_kv_tokens<=0;completion_kv_committed<=0;q_phase<=0;index<=0;
                    state<=admission_status==OK ? CHECK:RESPONSE;
                end
                CHECK:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else state<=KV_OFFER;
                KV_OFFER:if(append_ready)begin append_owned<=1;state<=KV_START;end
                KV_START:if(append_done_valid)state<=KV_WAIT;
                    else if(fault_now!=OK)begin commit_code<=fault_now;state<=KV_COMMIT;end
                    else if(append_prepared)state<=RAM_REQ;
                RAM_REQ:if(fault_now!=OK)begin
                        if(q_phase)state<=ROOT_WAIT;
                        else begin commit_code<=fault_now;state<=KV_COMMIT;end
                    end else state<=RAM_WAIT;
                RAM_WAIT:if(fault_now!=OK)begin
                        if(q_phase)state<=ROOT_WAIT;
                        else begin commit_code<=fault_now;state<=KV_COMMIT;end
                    end else if(source_q_valid)begin frame<=source_q;state<=STREAM;end
                STREAM:if(q_phase && root_completion_valid && root_identity)begin state<=ROOT_WAIT;end
                    else if(!q_phase && append_done_valid && append_identity)begin state<=KV_WAIT;end
                    else if(stream_fire)begin
                        if(fault_now!=OK)begin
                            if(q_phase)state<=ROOT_WAIT;
                            else begin commit_code<=fault_now;state<=KV_COMMIT;end
                        end else if(index==(q_phase ? 127:63))begin
                            if(q_phase)state<=ROOT_WAIT;
                            else begin commit_code<=OK;state<=KV_COMMIT;end
                        end else begin index<=index+1;state<=RAM_REQ;end
                    end
                KV_COMMIT:if(append_done_valid && append_identity)state<=KV_WAIT;
                    else if(commit_ready)state<=KV_WAIT;
                KV_WAIT:if(append_done_valid && append_identity)begin
                    completion_append_status<=append_done_status;completion_kv_bytes<=append_done_bytes;
                    completion_kv_tokens<=append_done_committed_tokens;
                    if(fault_now==OK)begin
                        if(append_done_bytes>4096 || append_done_bytes[5:0]!=0 || append_done_committed_tokens>saved_capacity)fault<=BAD_COMPLETION;
                        else if(append_done_status!=OK)fault<=append_done_status;
                        else if(!append_success)fault<=BAD_COMPLETION;
                    end
                    state<=KV_ACK;
                end
                KV_ACK:if(append_done_ready)begin append_owned<=0;state<=KV_VERIFY;end
                KV_VERIFY:begin
                    completion_kv_committed<=completion_append_status==OK && completion_kv_bytes==4096 &&
                        completion_kv_tokens==completion_position+1 && !cache_bad && cache_committed_tokens==completion_position+1;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(cache_committed_tokens!=completion_position+1)begin completion_status<=BAD_PEER;fault<=BAD_PEER;state<=RESPONSE;end
                    else begin q_phase<=1;index<=0;state<=ROOT_OFFER;end
                end
                ROOT_OFFER:if(root_command_ready)begin root_owned<=1;state<=ROOT_START;end
                ROOT_START:if(root_completion_valid || fault_now!=OK)state<=ROOT_WAIT;
                    else if(q_ready)state<=RAM_REQ;
                ROOT_WAIT:if(root_completion_valid && root_identity)begin
                    root_seen<=1;completion_root_status<=root_status;
                    if(fault_now!=OK)completion_status<=fault_now;
                    else if(root_status!=OK)completion_status<=root_status;
                    else if(!root_result_valid || completion_q_frames!=128)completion_status<=BAD_COMPLETION;
                    else completion_status<=OK;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=DRAIN;
                DRAIN:if(!source_owned && !append_owned && !root_owned)state<=IDLE;
                default:begin fault<=BAD_PEER;revoked<=1;state<=DRAIN;end
            endcase
        end
    end
endmodule
`default_nettype wire
