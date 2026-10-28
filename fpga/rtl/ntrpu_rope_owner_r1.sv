// Own 2560 Q/K FP32 words and a 64-word coefficient row through RoPE HLS.
// Position metadata is a lease identifier, not proof of the coefficient phase.
// The producer must supply an immutable, independently verified coefficient row.
// HLS ports are an abstract protocol boundary: bind generated RTL separately.
// A fault drains ap_done before releasing the loader; a hung HLS peer requires
// coordinated reset. Reset must also quiesce the loader and HLS peer.
`default_nettype none
module ntrpu_rope_owner_r1 #(
    parameter integer WATCHDOG_CYCLES=262144
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [31:0] command_position,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    input wire logic source_completion_valid,
    output logic source_completion_ready,
    input wire logic [3:0] source_completion_status,
    input wire logic [63:0] source_sequence,source_epoch,
    input wire logic [31:0] source_position,source_coefficient_position,
    input wire logic source_result_valid,
    output logic [1:0] source_ce,
    output wire [11:0] source_address [0:1],
    input wire logic [31:0] source_q [0:1],
    input wire logic [1:0] source_q_valid,
    output logic hls_start,
    input wire logic hls_done,hls_ready,
    input wire logic [31:0] hls_return,
    output logic [31:0] hls_heads,hls_position,hls_coefficient_position,
    input wire logic [1:0] hls_input_ce,
    input wire logic [11:0] hls_input_address [0:1],
    output wire [31:0] hls_input_q [0:1],
    input wire logic hls_output_ce,hls_output_we,
    input wire logic [11:0] hls_output_address,
    input wire logic [31:0] hls_output_d,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [31:0] completion_hls_status,completion_position,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic result_valid,
    input wire logic result_ce,
    input wire logic [11:0] result_address,
    output logic [31:0] result_q,
    output logic result_q_valid
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_MEMORY=4,
        COMPUTE_ERROR=5,CANCELLED=7,TIMEOUT=8,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,H_START,H_WAIT,RELEASE_SOURCE,RESPONSE} state_t;
    state_t state;
    (* ram_style="block" *) logic [31:0] output_mem[0:2559];
    logic [31:0] result_data,coefficient_position;
    logic result_pending,revoked,source_owned;
    logic [1:0] input_pending;
    logic [11:0] write_count,write_after;
    logic [31:0] idle_cycles;
    logic [3:0] fault,fault_now,admission_status;
    logic epoch_bad,peer_bad,write_event,word_bad,input_bad,input_address_bad;
    logic progress_event,timeout_now,active_hls,source_live;

    assign command_ready=state==IDLE && source_completion_valid;
    assign busy=state!=IDLE;
    assign cancel_ready=busy;
    assign completion_valid=state==RESPONSE;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    assign peer_bad=source_owned && (!source_completion_valid ||
        source_sequence!=completion_sequence || source_epoch!=completion_epoch ||
        source_completion_status!=OK || !source_result_valid ||
        source_position!=completion_position || source_coefficient_position!=coefficient_position);
    assign active_hls=state==H_START || state==H_WAIT;
    assign source_live=active_hls && fault==OK && !epoch_bad && !peer_bad && !cancel_valid;
    assign hls_start=state==H_START && fault_now==OK;
    assign hls_heads=32'd40;
    assign hls_position=completion_position;
    assign hls_coefficient_position=coefficient_position;
    // Only the owned identity may be acknowledged, including its failed result.
    assign source_completion_ready=state==RELEASE_SOURCE && source_owned &&
        source_sequence==completion_sequence && source_epoch==completion_epoch;
    for(genvar plane=0;plane<2;plane=plane+1)begin: inputs
        assign source_ce[plane]=active_hls && hls_input_ce[plane] && fault_now==OK;
        assign source_address[plane]=hls_input_address[plane];
        // Responses belong to the previous CE, independently on each RAM port.
        assign hls_input_q[plane]=source_live && source_q_valid[plane] ? source_q[plane]:32'd0;
    end
    assign write_event=active_hls && hls_output_ce && hls_output_we;
    assign write_after=write_count+(write_event ? 12'd1:12'd0);
    // Registered faults only: do not feed watchdog fault_now back into progress.
    assign progress_event=(source_live && |hls_input_ce) || write_event ||
        (state==H_WAIT && hls_done);
    assign timeout_now=state==H_WAIT && !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad && !cancel_valid;
    assign result_q_valid=result_pending && result_valid;
    assign result_q=result_q_valid ? result_data:32'd0;

    always_comb begin
        admission_status=OK;
        if(command_sequence!=source_sequence || command_epoch!=source_epoch || command_epoch!=expected_epoch)
            admission_status=BAD_IDENTITY;
        else if(command_position>=32'd131072)admission_status=BAD_COMMAND;
        else if(source_position!=command_position || source_coefficient_position!=command_position)admission_status=BAD_IDENTITY;
        else if(source_completion_status!=OK)admission_status=source_completion_status;
        else if(!source_result_valid)admission_status=BAD_PEER;
        input_bad=0;
        for(integer plane=0;plane<2;plane=plane+1)
            if(input_pending[plane] && (!source_q_valid[plane] || source_q[plane][30:23]==8'hff))input_bad=1;
        input_address_bad=(hls_input_ce[0] && hls_input_address[0]>=12'd2560) ||
            (hls_input_ce[1] && hls_input_address[1]>=12'd64);
        word_bad=hls_output_d[30:23]==8'hff;
        fault_now=fault;
        if(fault==OK && state!=IDLE && state!=RESPONSE)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(peer_bad)fault_now=BAD_PEER;
            else if(timeout_now)fault_now=TIMEOUT;
            else if(active_hls && input_address_bad)fault_now=BAD_MEMORY;
            else if(state==H_WAIT && input_bad)fault_now=BAD_MEMORY;
            else if(write_event && (write_count>=2560 || hls_output_address!=write_count || word_bad))fault_now=BAD_MEMORY;
            else if(state==H_WAIT && hls_done)begin
                if(!hls_ready)fault_now=BAD_PEER;
                else if(hls_return!=0)fault_now=COMPUTE_ERROR;
                else if(write_after!=2560)fault_now=BAD_MEMORY;
            end
        end
    end
    always_ff @(posedge clk)begin
        if(rst_n && write_event && fault_now==OK)output_mem[hls_output_address]<=hls_output_d;
        // Registered state keeps cancellation/consumer logic off the BRAM read enable.
        if(rst_n && state==RESPONSE && result_address<12'd2560)result_data<=output_mem[result_address];
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;write_count<=0;idle_cycles<=0;
            result_pending<=0;input_pending<=0;revoked<=0;source_owned<=0;
            fault<=OK;completion_status<=OK;completion_hls_status<=0;completion_sequence<=0;completion_epoch<=0;completion_position<=0;coefficient_position<=0;
        end else begin
            result_pending<=result_ce && result_valid && result_address<12'd2560;
            input_pending<=source_ce;
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(state!=IDLE && state!=RESPONSE)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            if(state==H_WAIT && !progress_event)begin
                if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            if(write_event && fault_now==OK)write_count<=write_after;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_position<=command_position;coefficient_position<=source_coefficient_position;
                    completion_status<=admission_status;completion_hls_status<=0;
                    write_count<=0;result_pending<=0;
                    input_pending<=0;revoked<=0;fault<=admission_status;
                    source_owned<=command_sequence==source_sequence && command_epoch==source_epoch;
                    if(command_sequence!=source_sequence || command_epoch!=source_epoch)state<=RESPONSE;
                    else state<=admission_status==OK ? H_START:RELEASE_SOURCE;
                end
                H_START:state<=fault_now==OK ? H_WAIT:RELEASE_SOURCE;
                H_WAIT:if(hls_done)begin completion_hls_status<=hls_return;state<=RELEASE_SOURCE;end
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
