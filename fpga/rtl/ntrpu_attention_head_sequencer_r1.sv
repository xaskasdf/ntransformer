// One root command owns 32 ordered head commands and an atomic output vector.
// The enclosing wrapper must inhibit cache bind/append while cache_lock is high
// and arbitrate root admission against outstanding cache maintenance operations.
`default_nettype none
module ntrpu_attention_head_sequencer_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire logic clk, rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [31:0] command_layer, command_position,
    input wire logic [63:0] command_sequence, command_epoch, command_lease_base,
    input wire logic cache_idle, cache_valid,
    input wire logic [63:0] cache_sequence, cache_epoch, expected_epoch,
    input wire logic [31:0] cache_capacity, cache_committed_tokens,
    output logic cache_lock,
    input wire logic q_valid,
    output logic q_ready,
    input wire logic [511:0] q_data,
    input wire logic [7:0] q_index,
    input wire logic q_last,
    input wire logic [63:0] q_sequence, q_epoch,
    input wire logic cancel_valid,
    output logic cancel_ready,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence, completion_epoch,
    output logic result_valid,
    input wire logic result_ce,
    input wire logic [10:0] result_address,
    output logic [31:0] result_q,
    output logic result_q_valid,
    output logic head_command_valid,
    input wire logic head_command_ready,
    output logic [2047:0] head_command_query,
    output logic [31:0] head_command_layer, head_command_position, head_command_head,
    output logic [63:0] head_command_sequence, head_command_epoch, head_command_lease,
    output logic head_cancel_valid,
    input wire logic head_cancel_ready,
    input wire logic head_completion_valid,
    output logic head_completion_ready,
    input wire logic [3:0] head_completion_status,
    input wire logic head_result_valid,
    output logic head_result_ce,
    output logic [5:0] head_result_address,
    input wire logic [31:0] head_result_q,
    input wire logic head_result_q_valid
);
    localparam logic [3:0] OK=0, BAD_COMMAND=1, BAD_IDENTITY=2,
        BAD_MEMORY=4, CANCELLED=7, TIMEOUT=8, BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,LOAD_Q,PREP_REQUEST,PREP_CAPTURE,ISSUE,
        WAIT_HEAD,COPY_REQUEST,COPY_RETURN,RELEASE_HEAD,DRAIN_HEAD,RESPONSE} state_t;
    state_t state;
    (* ram_style="block" *) logic [511:0] query_mem[0:127];
    (* ram_style="block" *) logic [31:0] output_mem[0:2047];
    logic [511:0] query_slice;
    logic [31:0] result_data;
    logic result_pending, revoked, child_owned, cancel_sent;
    logic [63:0] lease_base;
    logic [6:0] load_index;
    logic [4:0] head_index;
    logic [1:0] slice_index;
    logic [5:0] copy_index;
    logic [31:0] idle_cycles;
    logic [3:0] fault, fault_now, admission_status;
    logic q_finite, q_bad, progress_event, timeout_now, epoch_bad;
    logic output_write;

    assign command_ready=state==IDLE && cache_idle;
    assign cache_lock=state!=IDLE;
    assign cancel_ready=state!=IDLE;
    assign completion_valid=state==RESPONSE;
    assign completion_sequence=head_command_sequence;
    assign completion_epoch=head_command_epoch;
    assign epoch_bad=state!=IDLE && head_command_epoch!=expected_epoch;
    assign q_ready=state==LOAD_Q && fault==OK && !epoch_bad && !cancel_valid;
    assign head_command_valid=state==ISSUE && fault_now==OK;
    assign head_command_head={27'd0,head_index};
    assign head_command_lease=lease_base+{59'd0,head_index};
    // Registered failure avoids cancellation feedback through the child.
    assign head_cancel_valid=state==DRAIN_HEAD && child_owned && !cancel_sent && !head_completion_valid;
    assign head_completion_ready=state==RELEASE_HEAD || state==DRAIN_HEAD;
    assign head_result_ce=state==COPY_REQUEST && fault_now==OK;
    assign head_result_address=copy_index;
    assign result_valid=completion_valid && completion_status==OK && !revoked &&
        !epoch_bad && !cancel_valid;
    assign result_q_valid=result_pending && result_valid;
    assign result_q=result_q_valid ? result_data:32'd0;
    assign output_write=state==COPY_RETURN && head_result_q_valid && fault_now==OK;

    always_comb begin
        admission_status=OK;
        if(!cache_valid || command_sequence!=cache_sequence || command_epoch!=cache_epoch ||
            command_epoch!=expected_epoch) admission_status=BAD_IDENTITY;
        else if(command_layer>=16 || cache_capacity==0 || cache_capacity>131072 ||
            cache_committed_tokens>cache_capacity || command_position>=cache_committed_tokens ||
            command_lease_base==0 || command_lease_base>64'hffffffffffffffe0)
            admission_status=BAD_COMMAND;
        q_finite=1'b1;
        for(integer lane=0;lane<16;lane=lane+1)
            if(q_data[lane*32+23+:8]==8'hff)q_finite=1'b0;
        q_bad=q_index!={1'b0,load_index} || q_last!=(load_index==127) || !q_finite;
        progress_event=(q_valid && q_ready) || (state==ISSUE && head_command_ready &&
            fault==OK && !epoch_bad && !cancel_valid) ||
            (state==COPY_RETURN && head_result_q_valid);
        // The arithmetic child has its own lifetime/drain contract. Never
        // abandon an accepted child merely because its computation is long.
        timeout_now=(state==LOAD_Q || state==ISSUE || state==COPY_RETURN) &&
            !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;
        fault_now=fault;
        if(fault==OK && state!=IDLE && state!=RESPONSE)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(timeout_now)fault_now=TIMEOUT;
            else if(state==LOAD_Q && q_valid && q_ready)begin
                if(q_sequence!=head_command_sequence || q_epoch!=head_command_epoch)fault_now=BAD_IDENTITY;
                else if(q_bad)fault_now=BAD_MEMORY;
            end else if(state==COPY_REQUEST || state==COPY_RETURN || state==RELEASE_HEAD)begin
                if(!head_completion_valid || head_completion_status!=OK || !head_result_valid)
                    fault_now=BAD_PEER;
                else if(state==COPY_RETURN && head_result_q_valid && head_result_q[30:23]==8'hff)
                    fault_now=BAD_MEMORY;
            end
        end
    end

    // Memories have no reset, so each can infer block RAM. Authorization and
    // counters, not physical zeroing, hide stale or incomplete generations.
    always_ff @(posedge clk)begin
        if(rst_n && state==LOAD_Q && q_valid && q_ready && fault_now==OK)
            query_mem[load_index]<=q_data;
        if(rst_n && state==PREP_REQUEST)query_slice<=query_mem[{head_index,slice_index}];
        if(rst_n && output_write)output_mem[{head_index,copy_index}]<=head_result_q;
        if(rst_n && result_ce && result_valid)result_data<=output_mem[result_address];
    end

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;load_index<=0;head_index<=0;slice_index<=0;copy_index<=0;
            idle_cycles<=0;fault<=OK;revoked<=0;child_owned<=0;cancel_sent<=0;
            result_pending<=0;completion_status<=OK;lease_base<=0;
            head_command_query<=0;head_command_layer<=0;head_command_position<=0;
            head_command_sequence<=0;head_command_epoch<=0;
        end else begin
            result_pending<=result_ce && result_valid;
            if(state!=IDLE && (epoch_bad || cancel_valid))revoked<=1'b1;
            if(state==LOAD_Q || state==ISSUE || state==COPY_RETURN)begin
                if(progress_event)idle_cycles<=0;
                else if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            if(head_cancel_valid && head_cancel_ready)cancel_sent<=1'b1;

            if(state!=IDLE && state!=RESPONSE && state!=DRAIN_HEAD && fault_now!=OK)begin
                fault<=fault_now;revoked<=1'b1;
                if(child_owned && !(head_completion_valid && head_completion_ready))state<=DRAIN_HEAD;
                else begin
                    child_owned<=0;completion_status<=fault_now;state<=RESPONSE;
                end
            end else case(state)
                IDLE:if(command_valid && command_ready)begin
                    head_command_layer<=command_layer;head_command_position<=command_position;
                    head_command_sequence<=command_sequence;head_command_epoch<=command_epoch;
                    lease_base<=command_lease_base;load_index<=0;head_index<=0;slice_index<=0;copy_index<=0;
                    fault<=admission_status;revoked<=0;child_owned<=0;cancel_sent<=0;result_pending<=0;
                    completion_status<=admission_status;
                    state<=admission_status==OK ? LOAD_Q:RESPONSE;
                end
                LOAD_Q:if(q_valid && q_ready)begin
                    if(load_index==127)begin slice_index<=0;state<=PREP_REQUEST;end
                    else load_index<=load_index+1;
                end
                PREP_REQUEST:state<=PREP_CAPTURE;
                PREP_CAPTURE:begin
                    head_command_query[slice_index*512+:512]<=query_slice;
                    if(slice_index==3)begin idle_cycles<=0;state<=ISSUE;end
                    else begin slice_index<=slice_index+1;state<=PREP_REQUEST;end
                end
                ISSUE:if(head_command_valid && head_command_ready)begin
                    child_owned<=1'b1;cancel_sent<=0;state<=WAIT_HEAD;
                end
                WAIT_HEAD:if(head_completion_valid)begin
                    if(head_completion_status!=OK || !head_result_valid)begin
                        fault<=head_completion_status!=OK ? head_completion_status:BAD_PEER;
                        revoked<=1'b1;state<=DRAIN_HEAD;
                    end else begin copy_index<=0;state<=COPY_REQUEST;end
                end
                COPY_REQUEST:begin idle_cycles<=0;state<=COPY_RETURN;end
                COPY_RETURN:if(head_result_q_valid)begin
                    if(copy_index==63)state<=RELEASE_HEAD;
                    else begin copy_index<=copy_index+1;state<=COPY_REQUEST;end
                end
                RELEASE_HEAD:if(head_completion_valid && head_completion_ready)begin
                    child_owned<=0;
                    if(head_index==31)begin completion_status<=OK;state<=RESPONSE;end
                    else begin head_index<=head_index+1;slice_index<=0;state<=PREP_REQUEST;end
                end
                DRAIN_HEAD:if(head_completion_valid && head_completion_ready)begin
                    child_owned<=0;completion_status<=fault;state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)begin state<=IDLE;result_pending<=0;end
                default:begin state<=IDLE;revoked<=1'b1;end
            endcase
        end
    end
endmodule
`default_nettype wire
