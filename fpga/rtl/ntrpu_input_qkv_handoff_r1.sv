// Launch normalization then QKV while retaining the live ABI1 allocation.
// Regions: input, norm weights, ABI1, Q/K/V weights, Q/K/V outputs.
// The caller retains all nine allocations through public ACK and final drain.
// Source metadata is a caller snapshot, not an external producer handshake.
// Reset must quiesce both children and their AXI peers. Foreign completions
// are retained until identity recovers or a coordinated reset is applied.
`default_nettype none
module ntrpu_input_qkv_handoff_r1 #(
    parameter integer CHILD_TIMEOUT_CYCLES=16777216
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire [63:0] command_source_sequence,command_source_epoch,
    input wire command_source_valid,
    input wire [63:0] command_addr [0:8],command_base [0:8],command_bytes [0:8],
    input wire cancel_valid,
    output wire cancel_ready,busy,
    output wire norm_command_valid,
    input wire norm_command_ready,norm_busy,
    output wire [63:0] norm_command_sequence,norm_command_epoch,
    output wire [63:0] norm_command_addr [0:2],norm_command_base [0:2],norm_command_bytes [0:2],
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
    output wire qkv_command_valid,
    input wire qkv_command_ready,qkv_busy,
    output wire [63:0] qkv_command_sequence,qkv_command_epoch,
    output wire [63:0] qkv_command_addr [0:6],qkv_command_base [0:6],qkv_command_bytes [0:6],
    output wire qkv_cancel_valid,
    input wire qkv_source_ready,
    input wire qkv_completion_valid,
    output wire qkv_completion_ready,
    input wire [3:0] qkv_completion_status,qkv_source_status,qkv_child_status,
    input wire [63:0] qkv_sequence,qkv_epoch,
    input wire [1:0] qkv_phase,qkv_matrices,
    input wire [14:0] qkv_total_bytes,
    input wire qkv_result_valid,qkv_revoked,
    input wire [63:0] qkv_result_address [0:2],
    input wire [13:0] qkv_result_bytes [0:2],
    output wire qkv_bus_owned,
    output wire completion_valid,
    input wire completion_ready,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [3:0] completion_status,completion_norm_status,completion_qkv_status,completion_matrix_status,
    output logic [31:0] completion_norm_hls_status,
    output logic [9:0] completion_norm_read_beats,
    output logic [6:0] completion_norm_records,
    output logic [1:0] completion_phase,completion_matrices,
    output logic [14:0] completion_total_bytes,
    output wire completion_revoked,result_valid,
    output wire [63:0] result_address [0:2],
    output wire [13:0] result_bytes [0:2]
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_COMPLETION=5,
        CANCELLED=7,TIMED_OUT=8,BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,CHECK_RANGES,CHECK_ALIASES,OFFER_NORM,WAIT_NORM,
        OFFER_QKV,WAIT_QKV,RESPONSE,DRAIN} state_t;
    state_t state;
    logic [8:0][63:0] address,region_base,region_size;
    logic [3:0] range_index,write_index,other_index;
    logic norm_owned,qkv_owned,norm_seen,revoked,cancel_seen;
    logic [3:0] fault,fault_now,admission_status;
    logic [31:0] watchdog;
    wire epoch_bad=busy && expected_epoch!=completion_epoch;
    wire norm_identity=norm_sequence==completion_sequence && norm_epoch==completion_epoch;
    wire qkv_identity=qkv_sequence==completion_sequence && qkv_epoch==completion_epoch;
    wire norm_success=norm_completion_status==OK && norm_hls_status==0 && norm_read_beats==256 &&
        norm_records==64 && norm_result_valid && norm_result_address==address[2] && norm_result_bytes==4096;
    wire norm_changed=norm_seen && (!norm_completion_valid || !norm_identity ||
        norm_completion_status!=completion_norm_status || norm_hls_status!=completion_norm_hls_status ||
        norm_read_beats!=completion_norm_read_beats || norm_records!=completion_norm_records ||
        (completion_norm_status==OK && !norm_success));
    wire qkv_success=qkv_completion_status==OK && qkv_source_status==OK && qkv_child_status==OK &&
        qkv_phase==2 && qkv_matrices==3 && qkv_total_bytes==12288 && qkv_result_valid && !qkv_revoked &&
        qkv_result_address[0]==address[6] && qkv_result_address[1]==address[7] && qkv_result_address[2]==address[8] &&
        qkv_result_bytes[0]==8192 && qkv_result_bytes[1]==2048 && qkv_result_bytes[2]==2048;
    wire qkv_held_bad=qkv_owned && state==RESPONSE && (!qkv_completion_valid || !qkv_identity ||
        qkv_completion_status!=completion_qkv_status || qkv_child_status!=completion_matrix_status ||
        qkv_phase!=completion_phase || qkv_matrices!=completion_matrices || qkv_total_bytes!=completion_total_bytes ||
        (completion_status==OK && !qkv_success));
    wire public_ack=completion_valid && completion_ready;
    wire release_request=public_ack || state==DRAIN;
    wire [64:0] write_end={1'b0,region_base[write_index]}+{1'b0,region_size[write_index]};
    wire [64:0] other_end={1'b0,region_base[other_index]}+{1'b0,region_size[other_index]};
    wire overlap=write_index!=other_index && {1'b0,region_base[write_index]}<other_end &&
        {1'b0,region_base[other_index]}<write_end;
    wire wait_state=state==OFFER_NORM || state==WAIT_NORM || state==OFFER_QKV || state==WAIT_QKV;
    wire child_progress=(state==OFFER_NORM && norm_command_ready) ||
        (state==WAIT_NORM && norm_completion_valid) ||
        (state==OFFER_QKV && qkv_command_ready) || (state==WAIT_QKV && qkv_completion_valid);

    function automatic logic [63:0] span(input logic [3:0] index);
        case(index)
            0,1,6:return 64'd8192;
            2:return 64'd4096;
            3:return 64'd4456448;
            4,5:return 64'd1114112;
            default:return 64'd2048;
        endcase
    endfunction
    function automatic logic span_valid(input logic [63:0] addr,base,bytes,needed);
        logic [64:0] last,limit;
        last={1'b0,addr}+{1'b0,needed};limit={1'b0,base}+{1'b0,bytes};
        return addr[5:0]==0 && base[5:0]==0 && bytes[5:0]==0 && bytes!=0 &&
            !last[64] && !limit[64] && addr>=base && last<=limit;
    endfunction
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && norm_command_ready && !norm_busy && !qkv_busy &&
        !norm_completion_valid && !qkv_completion_valid;
    assign cancel_ready=busy && state!=DRAIN && !cancel_seen;
    assign norm_command_valid=state==OFFER_NORM;
    assign qkv_command_valid=state==OFFER_QKV;
    assign norm_command_sequence=completion_sequence;
    assign norm_command_epoch=completion_epoch;
    assign qkv_command_sequence=completion_sequence;
    assign qkv_command_epoch=completion_epoch;
    for(genvar n=0;n<3;n++)begin
        assign norm_command_addr[n]=address[n];
        assign norm_command_base[n]=region_base[n];
        assign norm_command_bytes[n]=region_size[n];
        assign result_address[n]=result_valid ? address[n+6]:64'd0;
        assign result_bytes[n]=result_valid ? (n==0 ? 14'd8192:14'd2048):14'd0;
    end
    for(genvar q=0;q<7;q++)begin
        assign qkv_command_addr[q]=address[q+2];
        assign qkv_command_base[q]=region_base[q+2];
        assign qkv_command_bytes[q]=region_size[q+2];
    end
    // Registered faults keep READY/result/cancel paths free of combinational loops.
    assign norm_cancel_valid=(norm_owned || state==OFFER_NORM) && (fault!=OK || cancel_valid);
    assign qkv_cancel_valid=(qkv_owned || state==OFFER_QKV) && (fault!=OK || cancel_valid);
    assign qkv_bus_owned=qkv_owned;
    assign qkv_completion_ready=qkv_owned && qkv_identity && release_request;
    assign norm_completion_ready=norm_owned && norm_identity && release_request &&
        (!qkv_owned || qkv_source_ready);
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=busy && (revoked || epoch_bad || cancel_valid || norm_changed || qkv_held_bad);
    assign result_valid=completion_valid && completion_status==OK && norm_owned && qkv_owned &&
        norm_completion_valid && norm_identity && norm_success && qkv_completion_valid && qkv_identity &&
        qkv_success && !completion_revoked;

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
                (qkv_completion_valid && (!qkv_owned || !qkv_identity)) ||
                (qkv_source_ready && !release_request))fault_now=BAD_PEER;
            else if(wait_state && watchdog>=CHILD_TIMEOUT_CYCLES-1 && !child_progress)fault_now=TIMED_OUT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;region_base<=0;region_size<=0;range_index<=0;write_index<=2;other_index<=0;
            norm_owned<=0;qkv_owned<=0;norm_seen<=0;revoked<=0;cancel_seen<=0;fault<=OK;watchdog<=0;
            completion_sequence<=0;completion_epoch<=0;completion_status<=OK;completion_norm_status<=OK;
            completion_qkv_status<=OK;completion_matrix_status<=OK;completion_norm_hls_status<=0;
            completion_norm_read_beats<=0;completion_norm_records<=0;completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
        end else begin
            if(busy)begin
                if(cancel_valid && cancel_ready)cancel_seen<=1;
                if(epoch_bad || cancel_valid || norm_changed || qkv_held_bad || fault_now!=OK)revoked<=1;
                if(state!=RESPONSE && state!=DRAIN)fault<=fault_now;
            end
            if(wait_state && !child_progress)begin
                if(watchdog<CHILD_TIMEOUT_CYCLES)watchdog<=watchdog+1'b1;
            end else watchdog<=0;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;completion_status<=admission_status;
                    completion_norm_status<=0;completion_qkv_status<=0;completion_matrix_status<=0;completion_norm_hls_status<=0;
                    completion_norm_read_beats<=0;completion_norm_records<=0;completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
                    norm_owned<=0;qkv_owned<=0;norm_seen<=0;revoked<=admission_status!=OK;cancel_seen<=cancel_valid;fault<=admission_status;
                    range_index<=0;write_index<=2;other_index<=0;
                    for(integer i=0;i<9;i++)begin address[i]<=command_addr[i];region_base[i]<=command_base[i];region_size[i]<=command_bytes[i];end
                    state<=admission_status==OK ? CHECK_RANGES:RESPONSE;
                end
                CHECK_RANGES:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!span_valid(address[range_index],region_base[range_index],region_size[range_index],span(range_index)))begin
                        completion_status<=BAD_COMMAND;revoked<=1;state<=RESPONSE;
                    end else if(range_index==8)state<=CHECK_ALIASES;
                    else range_index<=range_index+1'b1;
                CHECK_ALIASES:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(overlap)begin completion_status<=BAD_COMMAND;revoked<=1;state<=RESPONSE;end
                    else if(other_index!=8)other_index<=other_index+1'b1;
                    else if(write_index==8)state<=OFFER_NORM;
                    else begin write_index<=write_index==2 ? 4'd6:write_index+1'b1;other_index<=0;end
                // Once offered, commands remain stable even after cancellation/timeout.
                OFFER_NORM:if(norm_command_ready)begin norm_owned<=1;state<=WAIT_NORM;end
                WAIT_NORM:if(norm_completion_valid)begin
                    completion_norm_status<=norm_completion_status;completion_norm_hls_status<=norm_hls_status;
                    completion_norm_read_beats<=norm_read_beats;completion_norm_records<=norm_records;
                    norm_seen<=norm_identity;
                    if(fault_now!=OK || norm_completion_status!=OK || !norm_success)begin
                        completion_status<=fault_now!=OK ? fault_now:norm_completion_status!=OK ? norm_completion_status:BAD_COMPLETION;
                        revoked<=1;state<=RESPONSE;
                    end else state<=OFFER_QKV;
                end
                OFFER_QKV:if(qkv_command_ready)begin qkv_owned<=1;state<=WAIT_QKV;end
                WAIT_QKV:if(qkv_completion_valid)begin
                    completion_qkv_status<=qkv_completion_status;completion_matrix_status<=qkv_child_status;
                    completion_phase<=qkv_phase;completion_matrices<=qkv_matrices;completion_total_bytes<=qkv_total_bytes;
                    completion_status<=fault_now!=OK ? fault_now:qkv_completion_status!=OK ? qkv_completion_status:qkv_success ? OK:BAD_COMPLETION;
                    if(fault_now!=OK || qkv_completion_status!=OK || !qkv_success)revoked<=1;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=DRAIN;
                DRAIN:if(!norm_busy && !qkv_busy && !norm_completion_valid && !qkv_completion_valid)begin
                    norm_owned<=0;qkv_owned<=0;norm_seen<=0;state<=IDLE;
                end
                default:begin revoked<=1;state<=DRAIN;end
            endcase
        end
    end
endmodule
`default_nettype wire
