// Launch normalization then VOCAB while retaining the live ABI1 allocation.
// Regions: final residual, final norm weights, ABI1, tied head weights, logits.
// The caller retains all five allocations through public ACK and final drain.
// Source metadata is a caller snapshot, not an external producer handshake.
// Reset must quiesce both children and their AXI peers. Foreign completions
// are retained until identity recovers or a coordinated reset is applied.
`default_nettype none
module ntrpu_final_vocab_handoff_r1 #(
    parameter integer CHILD_TIMEOUT_CYCLES=268435456
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire [63:0] command_source_sequence,command_source_epoch,
    input wire command_source_valid,
    input wire [63:0] command_addr [0:4],
    input wire [63:0] command_base [0:4],
    input wire [63:0] command_bytes [0:4],
    input wire cancel_valid,
    output wire cancel_ready,busy,
    output wire norm_command_valid,
    input wire norm_command_ready,norm_busy,
    output wire [63:0] norm_command_sequence,norm_command_epoch,
    output wire [63:0] norm_command_addr [0:2],
    output wire [63:0] norm_command_base [0:2],
    output wire [63:0] norm_command_bytes [0:2],
    output wire norm_cancel_valid,
    input wire norm_completion_valid,
    output wire norm_completion_ready,
    input wire [3:0] norm_completion_status,
    input wire [31:0] norm_hls_status,
    input wire [63:0] norm_sequence,norm_epoch,norm_result_address,
    input wire [12:0] norm_result_bytes,
    input wire [9:0] norm_read_beats,
    input wire [6:0] norm_records,
    input wire norm_result_valid,
    output wire vocab_command_valid,
    input wire vocab_command_ready,vocab_busy,
    output wire [63:0] vocab_command_sequence,vocab_command_epoch,
    output wire [63:0] vocab_command_addr [0:2],
    output wire [63:0] vocab_command_base [0:2],
    output wire [63:0] vocab_command_bytes [0:2],
    output wire vocab_cancel_valid,
    input wire vocab_source_ready,
    input wire vocab_completion_valid,
    output wire vocab_completion_ready,
    input wire [3:0] vocab_completion_status,vocab_source_status,vocab_child_status,
    input wire [63:0] vocab_sequence,vocab_epoch,
    input wire [3:0] vocab_phase,
    input wire [4:0] vocab_matrices,
    input wire [18:0] vocab_total_bytes,
    input wire vocab_result_valid,vocab_revoked,
    input wire [63:0] vocab_result_address,
    input wire [18:0] vocab_result_bytes,
    output wire vocab_bus_owned,
    output wire completion_valid,
    input wire completion_ready,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [3:0] completion_status,completion_norm_status,completion_vocab_status,completion_matrix_status,
    output logic [31:0] completion_norm_hls_status,
    output logic [9:0] completion_norm_read_beats,
    output logic [6:0] completion_norm_records,
    output logic [3:0] completion_phase,
    output logic [4:0] completion_matrices,
    output logic [18:0] completion_total_bytes,
    output wire completion_revoked,result_valid,
    output wire [63:0] result_address,
    output wire [18:0] result_bytes
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_COMPLETION=5,
        CANCELLED=7,TIMED_OUT=8,BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,CHECK_RANGES,CHECK_ALIASES,OFFER_NORM,WAIT_NORM,
        OFFER_VOCAB,WAIT_VOCAB,RESPONSE,DRAIN} state_t;
    state_t state;
    logic [4:0][63:0] address,region_base,region_size;
    logic [3:0] range_index,write_index,other_index;
    logic norm_owned,vocab_owned,norm_seen,revoked,cancel_seen;
    logic [3:0] fault,fault_now,admission_status;
    logic [31:0] watchdog;
    wire epoch_bad=busy && expected_epoch!=completion_epoch;
    wire norm_identity=norm_sequence==completion_sequence && norm_epoch==completion_epoch;
    wire vocab_identity=vocab_sequence==completion_sequence && vocab_epoch==completion_epoch;
    wire norm_success=norm_completion_status==OK && norm_hls_status==0 && norm_read_beats==256 &&
        norm_records==64 && norm_result_valid && norm_result_address==address[2] && norm_result_bytes==4096;
    wire norm_changed=norm_seen && (!norm_completion_valid || !norm_identity ||
        norm_completion_status!=completion_norm_status || norm_hls_status!=completion_norm_hls_status ||
        norm_read_beats!=completion_norm_read_beats || norm_records!=completion_norm_records ||
        (completion_norm_status==OK && !norm_success));
    wire vocab_success=vocab_completion_status==OK && vocab_source_status==OK && vocab_child_status==OK &&
        vocab_phase==15 && vocab_matrices==16 && vocab_total_bytes==513024 && vocab_result_valid && !vocab_revoked &&
        vocab_result_address==address[4] && vocab_result_bytes==513024;
    wire vocab_held_bad=vocab_owned && state==RESPONSE && (!vocab_completion_valid || !vocab_identity ||
        vocab_completion_status!=completion_vocab_status || vocab_child_status!=completion_matrix_status ||
        vocab_phase!=completion_phase || vocab_matrices!=completion_matrices || vocab_total_bytes!=completion_total_bytes ||
        (completion_status==OK && !vocab_success));
    wire public_ack=completion_valid && completion_ready;
    wire release_request=public_ack || state==DRAIN;
    wire [64:0] write_end={1'b0,region_base[write_index]}+{1'b0,region_size[write_index]};
    wire [64:0] other_end={1'b0,region_base[other_index]}+{1'b0,region_size[other_index]};
    wire overlap=write_index!=other_index && {1'b0,region_base[write_index]}<other_end &&
        {1'b0,region_base[other_index]}<write_end;
    wire wait_state=state==OFFER_NORM || state==WAIT_NORM || state==OFFER_VOCAB || state==WAIT_VOCAB;
    wire child_progress=(state==OFFER_NORM && norm_command_ready) ||
        (state==WAIT_NORM && norm_completion_valid) ||
        (state==OFFER_VOCAB && vocab_command_ready) || (state==WAIT_VOCAB && vocab_completion_valid);

    function automatic logic [63:0] span(input logic [3:0] index);
        case(index)
            0,1:return 64'd8192;
            2:return 64'd4096;
            3:return 64'd279085056;
            default:return 64'd513024;
        endcase
    endfunction
    function automatic logic span_valid(input logic [63:0] addr,base,bytes,needed);
        logic [64:0] last,limit;
        last={1'b0,addr}+{1'b0,needed};limit={1'b0,base}+{1'b0,bytes};
        return addr[5:0]==0 && base[5:0]==0 && bytes[5:0]==0 && bytes!=0 &&
            !last[64] && !limit[64] && addr>=base && last<=limit;
    endfunction
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && norm_command_ready && !norm_busy && !vocab_busy &&
        !norm_completion_valid && !vocab_completion_valid;
    assign cancel_ready=busy && state!=DRAIN && !cancel_seen;
    assign norm_command_valid=state==OFFER_NORM;
    assign vocab_command_valid=state==OFFER_VOCAB;
    assign norm_command_sequence=completion_sequence;
    assign norm_command_epoch=completion_epoch;
    assign vocab_command_sequence=completion_sequence;
    assign vocab_command_epoch=completion_epoch;
    for(genvar n=0;n<3;n++)begin
        assign norm_command_addr[n]=address[n];
        assign norm_command_base[n]=region_base[n];
        assign norm_command_bytes[n]=region_size[n];

    end
    for(genvar q=0;q<3;q++)begin
        assign vocab_command_addr[q]=address[q+2];
        assign vocab_command_base[q]=region_base[q+2];
        assign vocab_command_bytes[q]=region_size[q+2];
    end
    assign result_address=result_valid ? address[4]:64'd0;
    assign result_bytes=result_valid ? 19'd513024:19'd0;
    // Registered faults keep READY/result/cancel paths free of combinational loops.
    assign norm_cancel_valid=(norm_owned || state==OFFER_NORM) && (fault!=OK || cancel_valid);
    assign vocab_cancel_valid=(vocab_owned || state==OFFER_VOCAB) && (fault!=OK || cancel_valid);
    assign vocab_bus_owned=vocab_owned;
    assign vocab_completion_ready=vocab_owned && vocab_identity && release_request;
    assign norm_completion_ready=norm_owned && norm_identity && release_request &&
        (!vocab_owned || vocab_source_ready);
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=busy && (revoked || epoch_bad || cancel_valid || norm_changed || vocab_held_bad);
    assign result_valid=completion_valid && completion_status==OK && norm_owned && vocab_owned &&
        norm_completion_valid && norm_identity && norm_success && vocab_completion_valid && vocab_identity &&
        vocab_success && !completion_revoked;

    always_comb begin
        admission_status=OK;
        if(command_sequence!=command_source_sequence || command_epoch!=command_source_epoch || command_epoch!=expected_epoch)
            admission_status=BAD_IDENTITY;
        else if(cancel_valid)admission_status=CANCELLED;
        else if(!command_source_valid)admission_status=BAD_PEER;
        fault_now=fault;
        if(busy && state!=RESPONSE && state!=DRAIN && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(norm_changed || (norm_completion_valid && (!norm_owned || !norm_identity)) ||
                (vocab_completion_valid && (!vocab_owned || !vocab_identity)) ||
                (vocab_source_ready && !release_request))fault_now=BAD_PEER;
            else if(wait_state && watchdog>=CHILD_TIMEOUT_CYCLES-1 && !child_progress)fault_now=TIMED_OUT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;region_base<=0;region_size<=0;range_index<=0;write_index<=2;other_index<=0;
            norm_owned<=0;vocab_owned<=0;norm_seen<=0;revoked<=0;cancel_seen<=0;fault<=OK;watchdog<=0;
            completion_sequence<=0;completion_epoch<=0;completion_status<=OK;completion_norm_status<=OK;
            completion_vocab_status<=OK;completion_matrix_status<=OK;completion_norm_hls_status<=0;
            completion_norm_read_beats<=0;completion_norm_records<=0;completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
        end else begin
            if(busy)begin
                if(cancel_valid && cancel_ready)cancel_seen<=1;
                if(epoch_bad || cancel_valid || norm_changed || vocab_held_bad || fault_now!=OK)revoked<=1;
                if(state!=RESPONSE && state!=DRAIN)fault<=fault_now;
            end
            if(wait_state && !child_progress)begin
                if(watchdog<CHILD_TIMEOUT_CYCLES)watchdog<=watchdog+1'b1;
            end else watchdog<=0;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;completion_status<=admission_status;
                    completion_norm_status<=0;completion_vocab_status<=0;completion_matrix_status<=0;completion_norm_hls_status<=0;
                    completion_norm_read_beats<=0;completion_norm_records<=0;completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
                    norm_owned<=0;vocab_owned<=0;norm_seen<=0;revoked<=admission_status!=OK;cancel_seen<=cancel_valid;fault<=admission_status;
                    range_index<=0;write_index<=2;other_index<=0;
                    for(integer i=0;i<5;i++)begin address[i]<=command_addr[i];region_base[i]<=command_base[i];region_size[i]<=command_bytes[i];end
                    state<=admission_status==OK ? CHECK_RANGES:RESPONSE;
                end
                CHECK_RANGES:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!span_valid(address[range_index],region_base[range_index],region_size[range_index],span(range_index)))begin
                        completion_status<=BAD_COMMAND;revoked<=1;state<=RESPONSE;
                    end else if(range_index==4)state<=CHECK_ALIASES;
                    else range_index<=range_index+1'b1;
                CHECK_ALIASES:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(overlap)begin completion_status<=BAD_COMMAND;revoked<=1;state<=RESPONSE;end
                    else if(other_index!=4)other_index<=other_index+1'b1;
                    else if(write_index==4)state<=OFFER_NORM;
                    else begin write_index<=4'd4;other_index<=0;end
                // Once offered, commands remain stable even after cancellation/timeout.
                OFFER_NORM:if(norm_command_ready)begin norm_owned<=1;state<=WAIT_NORM;end
                WAIT_NORM:if(norm_completion_valid)begin
                    completion_norm_status<=norm_completion_status;completion_norm_hls_status<=norm_hls_status;
                    completion_norm_read_beats<=norm_read_beats;completion_norm_records<=norm_records;
                    norm_seen<=norm_identity;
                    if(fault_now!=OK || norm_completion_status!=OK || !norm_success)begin
                        completion_status<=fault_now!=OK ? fault_now:norm_completion_status!=OK ? norm_completion_status:BAD_COMPLETION;
                        revoked<=1;state<=RESPONSE;
                    end else state<=OFFER_VOCAB;
                end
                OFFER_VOCAB:if(vocab_command_ready)begin vocab_owned<=1;state<=WAIT_VOCAB;end
                WAIT_VOCAB:if(vocab_completion_valid)begin
                    completion_vocab_status<=vocab_completion_status;completion_matrix_status<=vocab_child_status;
                    completion_phase<=vocab_phase;completion_matrices<=vocab_matrices;completion_total_bytes<=vocab_total_bytes;
                    completion_status<=fault_now!=OK ? fault_now:vocab_completion_status!=OK ? vocab_completion_status:vocab_success ? OK:BAD_COMPLETION;
                    if(fault_now!=OK || vocab_completion_status!=OK || !vocab_success)revoked<=1;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=DRAIN;
                DRAIN:if(!norm_busy && !vocab_busy && !norm_completion_valid && !vocab_completion_valid)begin
                    norm_owned<=0;vocab_owned<=0;norm_seen<=0;state<=IDLE;
                end
                default:begin revoked<=1;state<=DRAIN;end
            endcase
        end
    end
endmodule
`default_nettype wire
