// Own a completed 2048-FP32 root result until the quantizer has drained.
// HLS ports are abstract here: bind only after inspecting generated RTL.
// The source is held, never copied; the 512-word ABI1 destination is private.
// Shared reset requires the source and HLS peer to reset/quiesce together.
`default_nettype none
module ntrpu_attention_quantize_owner_r2 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic cancel_valid,
    output logic cancel_ready,
    output logic busy,
    input wire logic source_completion_valid,
    output logic source_completion_ready,
    input wire logic [3:0] source_completion_status,
    input wire logic [63:0] source_sequence,source_epoch,
    input wire logic source_result_valid,
    output logic source_result_ce,
    output logic [10:0] source_result_address,
    input wire logic [31:0] source_result_q,
    input wire logic source_result_q_valid,
    output logic hls_start,
    input wire logic hls_done,hls_ready,
    input wire logic [31:0] hls_return,
    output logic [31:0] hls_elements,
    input wire logic hls_input_ce,
    input wire logic [10:0] hls_input_address,
    output logic [31:0] hls_input_q,
    input wire logic hls_output_ce,hls_output_we,
    input wire logic [8:0] hls_output_address,
    input wire logic [63:0] hls_output_d,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [31:0] completion_hls_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic result_valid,
    input wire logic result_ce,
    input wire logic [8:0] result_address,
    output logic [63:0] result_q,
    output logic result_q_valid
);
    localparam logic [3:0] OK=0,BAD_IDENTITY=2,BAD_MEMORY=4,
        COMPUTE_ERROR=5,CANCELLED=7,TIMEOUT=8,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,H_START,H_WAIT,RELEASE_SOURCE,RESPONSE} state_t;
    state_t state;
    (* ram_style="block" *) logic [63:0] output_mem[0:511];
    logic [63:0] result_data;
    logic result_pending,input_pending,revoked,source_owned;
    logic [9:0] write_count,write_count_after;
    logic [31:0] idle_cycles;
    logic [3:0] fault,fault_now,admission_status;
    logic epoch_bad,peer_bad,write_event,word_bad,progress_event,timeout_now,active_hls;

    assign command_ready=state==IDLE && source_completion_valid;
    assign busy=state!=IDLE;
    assign cancel_ready=busy;
    assign completion_valid=state==RESPONSE;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    assign peer_bad=source_owned && (!source_completion_valid ||
        source_sequence!=completion_sequence || source_epoch!=completion_epoch ||
        source_completion_status!=OK || !source_result_valid);
    assign active_hls=state==H_START || state==H_WAIT;
    assign hls_start=state==H_START && fault_now==OK;
    assign hls_elements=32'd2048;
    // Ownership captured at admission does not authorize a foreign response.
    // A matching failed response may drain; a permanent identity change needs
    // coordinated reset of source, core and owner.
    assign source_completion_ready=state==RELEASE_SOURCE && source_owned &&
        source_sequence==completion_sequence && source_epoch==completion_epoch;
    assign source_result_ce=active_hls && hls_input_ce && fault_now==OK;
    assign source_result_address=hls_input_address;
    // A synchronous source holds a valid response for the following edge.
    // Do not tie this response to the core's current (next) address or CE.
    assign hls_input_q=active_hls && fault==OK && !epoch_bad && !peer_bad &&
        !cancel_valid && source_result_q_valid ? source_result_q:32'd0;
    assign write_event=active_hls && hls_output_ce && hls_output_we;
    assign write_count_after=write_count+(write_event ? 10'd1:10'd0);
    // Use registered fault state here; fault_now also depends on the watchdog.
    assign progress_event=(active_hls && hls_input_ce && fault==OK && !epoch_bad && !peer_bad && !cancel_valid) ||
        write_event || (state==H_WAIT && hls_done);
    assign timeout_now=state==H_WAIT && !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad && !cancel_valid;
    assign result_q_valid=result_pending && result_valid;
    assign result_q=result_q_valid ? result_data:64'd0;

    always_comb begin
        admission_status=OK;
        if(command_sequence!=source_sequence || command_epoch!=source_epoch || command_epoch!=expected_epoch)
            admission_status=BAD_IDENTITY;
        else if(source_completion_status!=OK)admission_status=source_completion_status;
        else if(!source_result_valid)admission_status=BAD_PEER;
        word_bad=0;
        if(hls_output_address[2:0]<4)begin
            for(integer byte_index=0;byte_index<8;byte_index=byte_index+1)
                if(hls_output_d[byte_index*8+:8]==8'h80)word_bad=1;
        end else if(hls_output_address[2:0]==4)word_bad=hls_output_d>64'h4780204070e1c387;
        else word_bad=hls_output_d!=0;
        fault_now=fault;
        if(fault==OK && state!=IDLE && state!=RESPONSE)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(peer_bad)fault_now=BAD_PEER;
            else if(timeout_now)fault_now=TIMEOUT;
            else if(state==H_WAIT && input_pending && !source_result_q_valid)fault_now=BAD_MEMORY;
            else if(write_event && (write_count>=512 || {1'b0,hls_output_address}!=write_count || word_bad))fault_now=BAD_MEMORY;
            else if(state==H_WAIT && hls_done)begin
                if(!hls_ready)fault_now=BAD_PEER;
                else if(hls_return!=0)fault_now=COMPUTE_ERROR;
                else if(write_count_after!=512)fault_now=BAD_MEMORY;
            end
        end
    end

    always_ff @(posedge clk)begin
        if(rst_n && write_event && fault_now==OK)output_mem[hls_output_address]<=hls_output_d;
        if(rst_n && result_ce && result_valid)result_data<=output_mem[result_address];
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;write_count<=0;idle_cycles<=0;result_pending<=0;input_pending<=0;revoked<=0;source_owned<=0;
            fault<=OK;completion_status<=OK;completion_hls_status<=0;completion_sequence<=0;completion_epoch<=0;
        end else begin
            result_pending<=result_ce && result_valid;
            input_pending<=source_result_ce;
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(state!=IDLE && state!=RESPONSE)begin
                fault<=fault_now;
                if(fault_now!=OK)revoked<=1;
            end
            if(state==H_WAIT && !progress_event)begin
                if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            if(write_event && fault_now==OK)write_count<=write_count_after;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_status<=admission_status;completion_hls_status<=0;
                    write_count<=0;result_pending<=0;revoked<=0;fault<=admission_status;
                    // A mismatched command never consumes someone else's result.
                    source_owned<=command_sequence==source_sequence && command_epoch==source_epoch;
                    if(command_sequence!=source_sequence || command_epoch!=source_epoch)state<=RESPONSE;
                    else state<=admission_status==OK ? H_START:RELEASE_SOURCE;
                end
                H_START:state<=fault_now==OK ? H_WAIT:RELEASE_SOURCE;
                H_WAIT:if(hls_done)begin
                    completion_hls_status<=hls_return;state<=RELEASE_SOURCE;
                end
                // A failed or cancelled core must still complete before this ack.
                RELEASE_SOURCE:if(source_completion_valid && source_completion_ready)begin
                    source_owned<=0;completion_status<=fault_now;state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)begin state<=IDLE;result_pending<=0;end
                default:begin state<=IDLE;revoked<=1;source_owned<=0;end
            endcase
        end
    end
endmodule
`default_nettype wire
