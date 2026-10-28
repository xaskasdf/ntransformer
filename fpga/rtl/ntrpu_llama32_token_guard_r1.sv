// Serialize all 16 layers and the final-normalization/vocabulary child.
// This owner retains the caller's complete memory plan, scratch allocations,
// embedding and cache reservation through publication and child drain. The
// enclosing allocator must validate all weight/scratch regions before command.
// Intermediate layer ACK releases that engine, not the enclosing allocation.
// Layer 15 stays held while the head reads its final residual.
// A partially executed token that fails poisons the context. Only a real cache
// bind certificate followed by all sixteen zero-prefix queries clears poison.
// Reset must quiesce/reset both children, AXI peers and the actual KV authority.
`default_nettype none
module ntrpu_llama32_token_guard_r1 #(
    parameter integer CHILD_TIMEOUT_CYCLES=268435456
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,command_lease_base,
    input wire command_source_valid,command_plan_valid,
    input wire [63:0] command_source_sequence,command_source_epoch,
    input wire [31:0] command_position,
    input wire [63:0] command_input_address,command_logits_address,
    input wire [63:0] command_residual_address [0:15],
    input wire cancel_valid,
    output wire cancel_ready,busy,cache_exclusive,bind_allowed,
    input wire bind_request_valid,bind_completion_valid,
    output wire bind_completion_ready,
    input wire [3:0] bind_completion_status,
    input wire [63:0] bind_completion_sequence,bind_completion_epoch,
    input wire cache_idle,cache_valid,cache_metadata_fault,
    input wire [63:0] cache_sequence,cache_epoch,
    input wire [31:0] cache_capacity,
    output wire [31:0] cache_query_layer,
    input wire cache_query_valid,
    input wire [31:0] cache_query_committed_tokens,
    output wire layer_command_valid,
    input wire layer_command_ready,layer_busy,
    output wire [63:0] layer_command_sequence,layer_command_epoch,layer_command_lease_base,
    output wire [31:0] layer_command_layer,layer_command_position,
    output wire [63:0] layer_command_input_address,layer_command_output_address,
    output wire layer_command_source_valid,
    output wire layer_cancel_valid,
    input wire layer_completion_valid,
    output wire layer_completion_ready,
    input wire [3:0] layer_completion_status,
    input wire [63:0] layer_sequence,layer_epoch,
    input wire [31:0] layer_index,layer_position,
    input wire layer_result_valid,layer_revoked,layer_kv_committed,
    input wire [63:0] layer_result_address,
    input wire [13:0] layer_result_bytes,
    input wire [31:0] layer_kv_tokens,
    output wire head_command_valid,
    input wire head_command_ready,head_busy,
    output wire [63:0] head_command_sequence,head_command_epoch,
    output wire [63:0] head_command_input_address,head_command_output_address,
    output wire head_command_source_valid,
    output wire head_cancel_valid,
    input wire head_completion_valid,
    output wire head_completion_ready,
    input wire [3:0] head_completion_status,
    input wire [63:0] head_sequence,head_epoch,
    input wire head_result_valid,head_revoked,
    input wire [63:0] head_result_address,
    input wire [18:0] head_result_bytes,
    output wire completion_valid,
    input wire completion_ready,
    output logic [3:0] completion_status,completion_child_status,binding_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [31:0] completion_position,
    output logic [4:0] completion_layers,
    output wire completion_revoked,result_valid,
    output wire [63:0] result_address,
    output wire [18:0] result_bytes,
    output logic cache_needs_rebind,
    output logic [31:0] committed_tokens
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_COMPLETION=5,
        CANCELLED=7,TIMED_OUT=8,BAD_PEER=11;
    typedef enum logic [3:0] {UNBOUND,VERIFY_BIND,IDLE,CHECK_PLAN,VERIFY_PREFIX,
        OFFER_LAYER,WAIT_LAYER,ACK_LAYER,DRAIN_LAYER,OFFER_HEAD,WAIT_HEAD,
        VERIFY_FINAL,RESPONSE,ACK_HEAD,DRAIN_HEAD,DRAIN_FINAL} state_t;
    state_t state;
    logic [63:0] bound_sequence,bound_epoch,lease_base,input_address,logits_address;
    logic [15:0][63:0] residual_address;
    logic [31:0] capacity,watchdog;
    logic [3:0] step,scan;
    logic owned_layer,owned_head,layer_seen,head_seen,started,revoked,cancel_seen;
    logic [3:0] fault,fault_now;
    wire no_children=!layer_busy && !head_busy && !layer_completion_valid && !head_completion_valid;
    wire context_bad=!cache_valid || cache_metadata_fault || cache_sequence!=bound_sequence ||
        cache_epoch!=bound_epoch || cache_capacity!=capacity || expected_epoch!=bound_epoch;
    wire token_active=state>=CHECK_PLAN;
    wire layer_identity=layer_sequence==completion_sequence && layer_epoch==completion_epoch &&
        layer_index=={28'd0,step} && layer_position==completion_position;
    wire head_identity=head_sequence==completion_sequence && head_epoch==completion_epoch;
    wire layer_success=layer_completion_status==OK && layer_result_valid && !layer_revoked &&
        layer_result_address==residual_address[step] && layer_result_bytes==8192 &&
        layer_kv_committed && layer_kv_tokens==completion_position+1;
    wire head_success=head_completion_status==OK && head_result_valid && !head_revoked &&
        head_result_address==logits_address && head_result_bytes==513024;
    wire layer_changed=owned_layer && layer_seen && state!=DRAIN_FINAL &&
        (!layer_completion_valid || !layer_identity || !layer_success);
    wire head_changed=owned_head && head_seen && state!=DRAIN_HEAD && state!=DRAIN_FINAL &&
        (!head_completion_valid || !head_identity || !head_success);
    wire waiting=state==OFFER_LAYER || state==WAIT_LAYER || state==OFFER_HEAD || state==WAIT_HEAD;
    wire progress=(state==OFFER_LAYER && layer_command_ready) ||
        (state==WAIT_LAYER && layer_completion_valid && layer_identity) ||
        (state==OFFER_HEAD && head_command_ready) ||
        (state==WAIT_HEAD && head_completion_valid && head_identity);
    wire late_bad=context_bad || cancel_valid || layer_changed || head_changed;
    wire [64:0] lease_end={1'b0,command_lease_base}+65'd511;

    function automatic logic valid_span(input logic [63:0] address,bytes);
        logic [64:0] limit;
        limit={1'b0,address}+{1'b0,bytes};
        return address[5:0]==0 && !limit[64];
    endfunction
    function automatic logic overlaps(input logic [63:0] a,an,b,bn);
        return {1'b0,a}<{1'b0,b}+{1'b0,bn} && {1'b0,b}<{1'b0,a}+{1'b0,an};
    endfunction
    assign busy=state!=IDLE && state!=UNBOUND;
    assign cache_exclusive=busy || owned_layer || owned_head;
    assign bind_allowed=(state==IDLE || state==UNBOUND) && no_children;
    assign bind_completion_ready=bind_allowed;
    assign command_ready=state==IDLE && !cache_needs_rebind && !context_bad && no_children &&
        cache_idle && !bind_request_valid && !bind_completion_valid;
    assign cancel_ready=token_active && state<=RESPONSE && !cancel_seen;
    assign cache_query_layer=(state==VERIFY_BIND || state==VERIFY_PREFIX || state==VERIFY_FINAL) ? {28'd0,scan}:{28'd0,step};
    assign layer_command_valid=state==OFFER_LAYER && fault_now==OK;
    assign layer_command_sequence=completion_sequence;
    assign layer_command_epoch=completion_epoch;
    assign layer_command_lease_base=lease_base+({60'd0,step}<<5);
    assign layer_command_layer={28'd0,step};
    assign layer_command_position=completion_position;
    assign layer_command_input_address=step==0 ? input_address:residual_address[step-1'b1];
    assign layer_command_output_address=residual_address[step];
    assign layer_command_source_valid=token_active && fault==OK && !revoked && (step==0 || completion_layers=={1'b0,step});
    assign head_command_valid=state==OFFER_HEAD && fault_now==OK;
    assign head_command_sequence=completion_sequence;
    assign head_command_epoch=completion_epoch;
    assign head_command_input_address=residual_address[15];
    assign head_command_output_address=logits_address;
    assign head_command_source_valid=owned_layer && layer_completion_valid && layer_identity && layer_success && !revoked;
    // Keep live result feedback out of child cancel paths.
    assign layer_cancel_valid=owned_layer && (fault!=OK || revoked || cancel_valid || context_bad);
    assign head_cancel_valid=owned_head && (fault!=OK || revoked || cancel_valid || context_bad);
    assign layer_completion_ready=owned_layer && layer_completion_valid && layer_identity &&
        (state==ACK_LAYER || (state==DRAIN_FINAL && !owned_head && !head_busy && !head_completion_valid));
    assign head_completion_ready=owned_head && head_completion_valid && head_identity && state==ACK_HEAD;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=token_active && (revoked || late_bad);
    assign result_valid=completion_valid && completion_status==OK && completion_layers==16 &&
        owned_layer && owned_head && layer_completion_valid && layer_identity && layer_success &&
        head_completion_valid && head_identity && head_success && !completion_revoked;
    assign result_address=result_valid ? logits_address:64'd0;
    assign result_bytes=result_valid ? 19'd513024:19'd0;

    always_comb begin
        fault_now=fault;
        if(token_active && fault==OK && state<=RESPONSE)begin
            if(expected_epoch!=completion_epoch)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(context_bad || layer_changed || head_changed ||
                (layer_completion_valid && (!owned_layer || !layer_identity)) ||
                (head_completion_valid && (!owned_head || !head_identity)))fault_now=BAD_PEER;
            else if(waiting && watchdog>=CHILD_TIMEOUT_CYCLES-1 && !progress)fault_now=TIMED_OUT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=UNBOUND;bound_sequence<=0;bound_epoch<=0;capacity<=0;committed_tokens<=0;
            cache_needs_rebind<=1;binding_status<=OK;completion_status<=OK;completion_child_status<=OK;
            completion_sequence<=0;completion_epoch<=0;completion_position<=0;completion_layers<=0;
            lease_base<=0;input_address<=0;logits_address<=0;residual_address<=0;step<=0;scan<=0;
            owned_layer<=0;owned_head<=0;layer_seen<=0;head_seen<=0;started<=0;revoked<=0;cancel_seen<=0;
            fault<=OK;watchdog<=0;
        end else begin
            if(token_active && context_bad)cache_needs_rebind<=1;
            if(waiting && !progress)begin
                if(watchdog<CHILD_TIMEOUT_CYCLES)watchdog<=watchdog+1'b1;
            end else watchdog<=0;
            if(token_active && state<=RESPONSE)begin
                if(cancel_valid && cancel_ready)cancel_seen<=1;
                if(fault_now!=OK)begin fault<=fault_now;revoked<=1;if(started || context_bad)cache_needs_rebind<=1;end
            end
            case(state)
                UNBOUND,IDLE:begin
                    if(state==IDLE && context_bad)begin cache_needs_rebind<=1;state<=UNBOUND;end
                    if(bind_completion_valid && bind_completion_ready)begin
                        binding_status<=bind_completion_status;cache_needs_rebind<=1;
                        if(bind_completion_status!=OK || !cache_valid || cache_metadata_fault ||
                            bind_completion_sequence!=cache_sequence || bind_completion_epoch!=cache_epoch ||
                            cache_epoch!=expected_epoch || cache_capacity==0 || cache_capacity>131072)begin
                            binding_status<=BAD_PEER;state<=UNBOUND;
                        end else begin
                            bound_sequence<=cache_sequence;bound_epoch<=cache_epoch;capacity<=cache_capacity;
                            scan<=0;state<=VERIFY_BIND;
                        end
                    end else if(command_valid && command_ready)begin
                        completion_sequence<=command_sequence;completion_epoch<=command_epoch;completion_position<=command_position;
                        lease_base<=command_lease_base;input_address<=command_input_address;logits_address<=command_logits_address;
                        for(integer n=0;n<16;n++)residual_address[n]<=command_residual_address[n];
                        completion_layers<=0;completion_child_status<=OK;step<=0;scan<=0;
                        owned_layer<=0;owned_head<=0;layer_seen<=0;head_seen<=0;started<=0;revoked<=0;cancel_seen<=0;fault<=OK;
                        completion_status<=OK;
                        if(command_sequence!=bound_sequence || command_epoch!=bound_epoch ||
                            command_source_sequence!=command_sequence || command_source_epoch!=command_epoch)begin
                            completion_status<=BAD_IDENTITY;revoked<=1;state<=RESPONSE;
                        end else if(cancel_valid)begin completion_status<=CANCELLED;revoked<=1;state<=RESPONSE;end
                        else if(!command_source_valid || !command_plan_valid || command_position!=committed_tokens ||
                            command_position>=capacity || command_lease_base==0 || lease_end[64] ||
                            !valid_span(command_input_address,8192) || !valid_span(command_logits_address,513024))begin
                            completion_status<=BAD_COMMAND;revoked<=1;state<=RESPONSE;
                        end else state<=CHECK_PLAN;
                    end
                end
                VERIFY_BIND:begin
                    if(context_bad || !cache_idle || !cache_query_valid || cache_query_committed_tokens!=0)begin
                        binding_status<=BAD_PEER;state<=UNBOUND;
                    end else if(scan==15)begin committed_tokens<=0;cache_needs_rebind<=0;binding_status<=OK;state<=IDLE;end
                    else scan<=scan+1'b1;
                end
                CHECK_PLAN:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!valid_span(residual_address[scan],8192) ||
                        overlaps(scan==0 ? input_address:residual_address[scan-1'b1],8192,residual_address[scan],8192) ||
                        (scan==15 && overlaps(residual_address[15],8192,logits_address,513024)))begin
                        completion_status<=BAD_COMMAND;revoked<=1;state<=RESPONSE;
                    end else if(scan==15)begin scan<=0;state<=VERIFY_PREFIX;end
                    else scan<=scan+1'b1;
                end
                VERIFY_PREFIX,VERIFY_FINAL:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!cache_query_valid || cache_query_committed_tokens!=completion_position+(state==VERIFY_FINAL ? 1:0))begin
                        completion_status<=BAD_PEER;fault<=BAD_PEER;revoked<=1;cache_needs_rebind<=1;state<=RESPONSE;
                    end else if(scan==15)begin
                        if(state==VERIFY_FINAL)begin completion_status<=OK;state<=RESPONSE;end
                        else state<=OFFER_LAYER;
                    end else scan<=scan+1'b1;
                end
                OFFER_LAYER:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(layer_command_valid && layer_command_ready)begin
                        owned_layer<=1;layer_seen<=0;started<=1;state<=WAIT_LAYER;
                    end
                end
                WAIT_LAYER:begin
                    if(layer_completion_valid && layer_identity)begin
                        completion_child_status<=layer_completion_status;
                        if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                        else if(!layer_success || !cache_query_valid || cache_query_committed_tokens!=completion_position+1)begin
                            completion_status<=layer_completion_status!=OK ? layer_completion_status:BAD_COMPLETION;
                            revoked<=1;cache_needs_rebind<=1;state<=RESPONSE;
                        end else begin
                            layer_seen<=1;completion_layers<=completion_layers+1'b1;
                            state<=step==15 ? OFFER_HEAD:ACK_LAYER;
                        end
                    end
                end
                ACK_LAYER:begin
                    if(layer_completion_valid && layer_completion_ready)begin
                        owned_layer<=0;layer_seen<=0;state<=DRAIN_LAYER;
                    end
                end
                DRAIN_LAYER:begin
                    if(!layer_busy && !layer_completion_valid)begin
                        if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                        else begin step<=step+1'b1;state<=OFFER_LAYER;end
                    end
                end
                OFFER_HEAD:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(head_command_valid && head_command_ready)begin owned_head<=1;head_seen<=0;state<=WAIT_HEAD;end
                end
                WAIT_HEAD:begin
                    if(head_completion_valid && head_identity)begin
                        completion_child_status<=head_completion_status;
                        if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                        else if(!head_success)begin
                            completion_status<=head_completion_status!=OK ? head_completion_status:BAD_COMPLETION;
                            revoked<=1;cache_needs_rebind<=1;state<=RESPONSE;
                        end else begin head_seen<=1;scan<=0;state<=VERIFY_FINAL;end
                    end
                end
                RESPONSE:begin
                    if(completion_ready)begin
                        if(completion_status!=OK || completion_revoked)begin if(started)cache_needs_rebind<=1;end
                        else committed_tokens<=completion_position+1'b1;
                        state<=ACK_HEAD;
                    end
                end
                ACK_HEAD:begin
                    if(!owned_head)state<=DRAIN_HEAD;
                    else if(head_completion_ready)begin owned_head<=0;head_seen<=0;state<=DRAIN_HEAD;end
                end
                DRAIN_HEAD:begin
                    if(!head_busy && !head_completion_valid)state<=DRAIN_FINAL;
                end
                DRAIN_FINAL:begin
                    if(layer_completion_ready)begin owned_layer<=0;layer_seen<=0;end
                    if(!owned_layer && no_children)begin state<=cache_needs_rebind || context_bad ? UNBOUND:IDLE;end
                end
                default:begin cache_needs_rebind<=1;state<=UNBOUND;end
            endcase
        end
    end
endmodule
`default_nettype wire
