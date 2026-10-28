// 2048x2048 O projection control; matrix arithmetic/AXI live in the existing core.
// Full parent identity is retained; internal tags reserve 64 row-group sequences.
// Reset must quiesce/reset the dedicated child and AXI peer, including old replies.
`default_nettype none
module ntrpu_attention_projection_control_r1 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter logic [63:0] INITIAL_CHILD_TAG=64'd0
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_weight_addr,command_weight_region_base,command_weight_region_bytes,
    input wire logic [63:0] command_activation_addr,command_activation_region_base,command_activation_region_bytes,
    input wire logic [63:0] command_output_addr,command_output_region_base,command_output_region_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic result_valid,
    output logic [63:0] result_address,
    output logic [16:0] result_bytes,
    output logic tag_exhausted,
    output logic child_job_valid,
    input wire logic child_job_ready,
    output logic [1:0] child_job_layout,
    output logic [13:0] child_job_rows,
    output logic child_job_output_fp32,
    output logic [8:0] child_job_k_blocks,
    output logic [5:0] child_job_tile_k,
    output logic [31:0] child_job_sequence,child_job_epoch,
    output logic [63:0] child_job_weight_addr,child_job_region_base,child_job_region_bytes,
    output logic [63:0] child_job_activation_addr,child_job_activation_region_base,child_job_activation_region_bytes,
    output logic [31:0] child_job_activation_epoch,
    output logic [63:0] child_job_output_addr,child_job_output_region_base,child_job_output_region_bytes,
    output logic [31:0] child_job_output_epoch,
    output logic child_cancel_valid,
    input wire logic child_cancel_ready,child_activity,
    input wire logic child_done_valid,
    output logic child_done_ready,
    input wire logic [3:0] child_done_status,
    input wire logic [16:0] child_done_bytes,
    input wire logic [13:0] child_done_rows,
    input wire logic [31:0] child_done_sequence,child_done_epoch
);
    localparam logic [3:0] OK=0,BAD_IDENTITY=2,BAD_COMPLETION=5,
        CANCELLED=7,TIMEOUT=8,TAGS_EXHAUSTED=9,BAD_PEER=11;
    typedef enum logic [1:0] {IDLE,OFFER,WAIT_CHILD,RESPONSE} state_t;
    state_t state;
    logic [63:0] next_tag;
    logic [31:0] idle_cycles;
    logic [3:0] fault,fault_now;
    logic revoked,cancel_seen,cancel_sent,epoch_bad,identity_match,timeout_now;
    logic progress_event,active_child;

    assign command_ready=state==IDLE && !child_done_valid;
    assign busy=state!=IDLE;
    assign completion_valid=state==RESPONSE;
    assign cancel_ready=busy && !cancel_seen;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    assign active_child=state==OFFER || state==WAIT_CHILD;
    assign child_job_valid=state==OFFER;
    assign child_job_layout=2'd2;
    assign child_job_rows=14'd2048;
    assign child_job_output_fp32=1;
    assign child_job_k_blocks=9'd64;
    assign child_job_tile_k=6'd32;
    assign child_job_activation_epoch=child_job_epoch;
    assign child_job_output_epoch=child_job_epoch;
    assign identity_match=child_done_sequence==child_job_sequence && child_done_epoch==child_job_epoch;
    // A foreign completion cannot prove that our child has drained.
    assign child_done_ready=state==WAIT_CHILD && identity_match;
    assign progress_event=(state==OFFER && child_job_ready) ||
        (state==WAIT_CHILD && (child_activity || (child_done_valid && identity_match)));
    assign timeout_now=active_child && !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;
    assign child_cancel_valid=state==WAIT_CHILD && fault_now!=OK && !cancel_sent;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad && !cancel_valid;
    assign result_address=result_valid ? child_job_output_addr:64'd0;
    assign result_bytes=result_valid ? 17'd8192:17'd0;

    always_comb begin
        fault_now=fault;
        if(fault==OK && active_child)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(child_done_valid && state==WAIT_CHILD && !identity_match)fault_now=BAD_PEER;
            else if(timeout_now)fault_now=TIMEOUT;
        end
    end

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;next_tag<=INITIAL_CHILD_TAG;tag_exhausted<=0;idle_cycles<=0;
            fault<=OK;revoked<=0;cancel_seen<=0;cancel_sent<=0;
            completion_status<=OK;completion_sequence<=0;completion_epoch<=0;
            child_job_sequence<=0;child_job_epoch<=0;
            child_job_weight_addr<=0;child_job_region_base<=0;child_job_region_bytes<=0;
            child_job_activation_addr<=0;child_job_activation_region_base<=0;child_job_activation_region_bytes<=0;
            child_job_output_addr<=0;child_job_output_region_base<=0;child_job_output_region_bytes<=0;
        end else begin
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(active_child)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            if(child_cancel_valid && child_cancel_ready)cancel_sent<=1;
            if(active_child && !progress_event)begin
                if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_status<=OK;revoked<=0;cancel_seen<=0;cancel_sent<=0;fault<=OK;
                    child_job_weight_addr<=command_weight_addr;
                    child_job_region_base<=command_weight_region_base;child_job_region_bytes<=command_weight_region_bytes;
                    child_job_activation_addr<=command_activation_addr;
                    child_job_activation_region_base<=command_activation_region_base;child_job_activation_region_bytes<=command_activation_region_bytes;
                    child_job_output_addr<=command_output_addr;
                    child_job_output_region_base<=command_output_region_base;child_job_output_region_bytes<=command_output_region_bytes;
                    if(command_epoch!=expected_epoch)begin completion_status<=BAD_IDENTITY;state<=RESPONSE;end
                    else if(tag_exhausted || next_tag[5:0]!=0)begin completion_status<=TAGS_EXHAUSTED;state<=RESPONSE;end
                    else begin
                        child_job_sequence<=next_tag[31:0];child_job_epoch<=next_tag[63:32];
                        if(next_tag==64'hffffffffffffffc0)tag_exhausted<=1;
                        else next_tag<=next_tag+64'd64;
                        state<=OFFER;
                    end
                end
                // VALID/descriptor stay stable even if cancellation arrives while blocked.
                OFFER:if(child_job_ready)state<=WAIT_CHILD;
                WAIT_CHILD:if(child_done_valid && child_done_ready)begin
                    if(fault_now!=OK)completion_status<=fault_now;
                    else if(child_done_bytes>8192 || child_done_rows>2048 ||
                        (child_done_status==OK && (child_done_bytes!=8192 || child_done_rows!=2048)))
                        completion_status<=BAD_COMPLETION;
                    else completion_status<=child_done_status;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=IDLE;
                default:begin state<=IDLE;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
