`default_nettype none
// One retained norm ABI1 -> Q, K and V using one exclusive FP32 matrix engine.
// Descriptor order: ABI1, Q weights, K weights, V weights, Q out, K out, V out.
// Public identities are 64-bit; each matrix gets an independent aligned tag.
// A late lease revocation invalidates result_valid, even with completion held.
// Every public VALID/READY handshake is consumed. Foreign peer responses may
// keep the controller draining afterwards; they are never acknowledged.
// Reset must also reset/quiesce the source, matrix engine and AXI peers.
module ntrpu_llama32_qkv_control_r1 #(
    parameter logic [63:0] INITIAL_TAG=64'd0,
    parameter integer CHILD_TIMEOUT_CYCLES=16777216
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire [63:0] command_addr [0:6],command_base [0:6],command_bytes [0:6],
    input wire cancel_valid,
    output wire cancel_ready,busy,
    input wire source_completion_valid,
    output wire source_completion_ready,
    input wire [3:0] source_completion_status,
    input wire [63:0] source_sequence,source_epoch,source_result_address,
    input wire source_result_valid,
    input wire [12:0] source_result_bytes,
    output wire child_job_valid,
    input wire child_job_ready,
    output wire [1:0] child_job_layout,
    output wire [13:0] child_job_rows,
    output wire child_job_output_fp32,
    output wire [8:0] child_job_k_blocks,
    output wire [5:0] child_job_tile_k,
    output wire [31:0] child_job_sequence,child_job_epoch,
    output wire [63:0] child_job_weight_addr,child_job_region_base,child_job_region_bytes,
    output wire [63:0] child_job_activation_addr,child_job_activation_region_base,child_job_activation_region_bytes,
    output wire [31:0] child_job_activation_epoch,
    output wire [63:0] child_job_output_addr,child_job_output_region_base,child_job_output_region_bytes,
    output wire [31:0] child_job_output_epoch,
    output wire child_cancel_valid,
    input wire child_cancel_ready,
    input wire child_done_valid,
    output wire child_done_ready,
    input wire [3:0] child_done_status,
    input wire [16:0] child_done_bytes,
    input wire [13:0] child_done_rows,
    input wire [31:0] child_done_sequence,child_done_epoch,
    output wire completion_valid,
    input wire completion_ready,
    output logic [3:0] completion_status,completion_source_status,completion_child_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [1:0] completion_phase,completion_matrices,
    output logic [14:0] completion_total_bytes,
    output logic [16:0] completion_child_bytes,
    output logic [13:0] completion_child_rows,
    output wire completion_revoked,result_valid,tags_exhausted,
    output wire [63:0] result_q_address,result_k_address,result_v_address,
    output wire [13:0] result_q_bytes,result_k_bytes,result_v_bytes
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_COMPLETION=5,
        CANCELLED=7,TIMED_OUT=8,TAG_EXHAUSTED=10,BAD_PEER=11;
    localparam logic [63:0] LAST_TAG_BASE=64'hffffffffffffff40;
    typedef enum logic [3:0] {IDLE,CHECK_SOURCE,CHECK_RANGES,CHECK_ALIASES,RESERVE,
        OFFER,WAIT_CHILD,ADVANCE,RESPONSE,DRAIN} state_t;
    state_t state;
    logic [6:0][63:0] address,region_base,region_bytes;
    logic [63:0] next_tag,tag;
    logic [2:0] range_index,write_index,other_index;
    logic [1:0] phase;
    logic source_owned,child_owned,exhausted,revoked,cancel_seen,cancel_sent;
    logic source_snapshot_valid;
    logic [63:0] source_snapshot_address;
    logic [12:0] source_snapshot_bytes;
    logic [3:0] fault,fault_now;
    logic [31:0] watchdog;
    wire epoch_bad=busy && expected_epoch!=completion_epoch;
    wire source_identity=source_sequence==completion_sequence && source_epoch==completion_epoch;
    wire child_identity={child_done_epoch,child_done_sequence}==tag;
    wire source_bad=source_owned && (!source_completion_valid || !source_identity ||
        source_completion_status!=completion_source_status || source_result_valid!=source_snapshot_valid ||
        source_result_address!=source_snapshot_address || source_result_bytes!=source_snapshot_bytes);
    wire child_foreign=child_done_valid && (!child_owned || !child_identity);
    wire [13:0] phase_rows=phase==0 ? 14'd2048:14'd512;
    wire [16:0] phase_bytes=phase==0 ? 17'd8192:17'd2048;
    wire child_success=child_done_status==OK && child_done_rows==phase_rows && child_done_bytes==phase_bytes;
    wire child_shape_bad=child_done_rows>phase_rows || child_done_bytes>phase_bytes ||
        (child_done_status==OK && !child_success);
    wire child_held_bad=child_owned && (state==RESPONSE || state==DRAIN) &&
        (!child_done_valid || !child_identity || child_done_status!=completion_child_status ||
        child_done_rows!=completion_child_rows || child_done_bytes!=completion_child_bytes);
    wire release_safe=(!source_owned || (source_completion_valid && source_identity)) &&
        (!child_owned || (child_done_valid && child_identity));
    wire public_ack=completion_valid && completion_ready;
    wire release_all=(public_ack || state==DRAIN) && release_safe;
    wire intermediate_ack=state==WAIT_CHILD && child_owned && child_done_valid && child_identity &&
        fault_now==OK && child_success && phase<2;
    wire [2:0] weight_index={1'b0,phase}+3'd1;
    wire [2:0] output_index={1'b0,phase}+3'd4;
    wire [64:0] reserved_end={1'b0,next_tag}+65'd192;
    wire [64:0] write_end={1'b0,region_base[write_index]}+{1'b0,region_bytes[write_index]};
    wire [64:0] other_end={1'b0,region_base[other_index]}+{1'b0,region_bytes[other_index]};
    wire overlap=write_index!=other_index && {1'b0,region_base[write_index]}<other_end &&
        {1'b0,region_base[other_index]}<write_end;

    function automatic logic [63:0] span(input logic [2:0] index);
        case(index)
            0:return 64'd4096;
            1:return 64'd4456448;
            2,3:return 64'd1114112;
            4:return 64'd8192;
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
    assign command_ready=state==IDLE && source_completion_valid && child_job_ready && !child_done_valid;
    assign cancel_ready=busy && !cancel_seen;
    assign completion_valid=state==RESPONSE;
    assign source_completion_ready=source_owned && release_all;
    assign child_done_ready=child_owned && (intermediate_ack || release_all);
    assign child_job_valid=state==OFFER;
    assign child_job_layout=2;
    assign child_job_rows=phase_rows;
    assign child_job_output_fp32=1;
    assign child_job_k_blocks=64;
    assign child_job_tile_k=32;
    assign child_job_sequence=tag[31:0];
    assign child_job_epoch=tag[63:32];
    assign child_job_weight_addr=address[weight_index];
    assign child_job_region_base=region_base[weight_index];
    assign child_job_region_bytes=region_bytes[weight_index];
    assign child_job_activation_addr=address[0];
    assign child_job_activation_region_base=region_base[0];
    assign child_job_activation_region_bytes=region_bytes[0];
    assign child_job_activation_epoch=tag[63:32];
    assign child_job_output_addr=address[output_index];
    assign child_job_output_region_base=region_base[output_index];
    assign child_job_output_region_bytes=region_bytes[output_index];
    assign child_job_output_epoch=tag[63:32];
    // Registered fault only: no ready/activity -> cancel -> completion loop.
    assign child_cancel_valid=child_owned && fault!=OK && !cancel_sent;
    assign tags_exhausted=exhausted;
    assign completion_revoked=busy && (revoked || epoch_bad || cancel_valid || source_bad || child_held_bad);
    assign result_valid=completion_valid && completion_status==OK && completion_matrices==3 &&
        completion_total_bytes==12288 && source_owned && child_owned && child_done_valid &&
        child_identity && child_success && !completion_revoked;
    assign result_q_address=result_valid ? address[4]:64'd0;
    assign result_k_address=result_valid ? address[5]:64'd0;
    assign result_v_address=result_valid ? address[6]:64'd0;
    assign result_q_bytes=result_valid ? 14'd8192:14'd0;
    assign result_k_bytes=result_valid ? 14'd2048:14'd0;
    assign result_v_bytes=result_valid ? 14'd2048:14'd0;

    always_comb begin
        fault_now=fault;
        if(busy && state!=RESPONSE && state!=DRAIN && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(source_bad || child_foreign)fault_now=BAD_PEER;
            else if((state==OFFER || state==WAIT_CHILD || state==ADVANCE) && watchdog>=CHILD_TIMEOUT_CYCLES-1 &&
                !(state==OFFER && child_job_ready) &&
                !(state==ADVANCE && child_job_ready && !child_done_valid) &&
                !(state==WAIT_CHILD && child_done_valid && child_identity))fault_now=TIMED_OUT;
        end
    end

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;source_owned<=0;child_owned<=0;revoked<=0;cancel_seen<=0;cancel_sent<=0;
            exhausted<=INITIAL_TAG[5:0]!=0 || INITIAL_TAG>LAST_TAG_BASE;
            next_tag<=INITIAL_TAG;tag<=0;phase<=0;range_index<=0;write_index<=4;other_index<=0;
            watchdog<=0;fault<=OK;source_snapshot_valid<=0;source_snapshot_address<=0;source_snapshot_bytes<=0;
            completion_status<=0;completion_source_status<=0;completion_child_status<=0;
            completion_sequence<=0;completion_epoch<=0;completion_phase<=0;completion_matrices<=0;
            completion_total_bytes<=0;completion_child_bytes<=0;completion_child_rows<=0;
            address<=0;region_base<=0;region_bytes<=0;
        end else begin
            if(busy && (epoch_bad || cancel_valid || source_bad || child_held_bad))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(child_cancel_valid && child_cancel_ready)cancel_sent<=1;
            if(busy && state!=RESPONSE && state!=DRAIN)begin
                fault<=fault_now;
                if(fault_now!=OK)revoked<=1;
            end
            if(state==OFFER || state==WAIT_CHILD || state==ADVANCE)begin
                if(watchdog<CHILD_TIMEOUT_CYCLES)watchdog<=watchdog+1'b1;
            end else watchdog<=0;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_status<=0;completion_source_status<=source_completion_status;completion_child_status<=0;
                    completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
                    completion_child_bytes<=0;completion_child_rows<=0;
                    source_owned<=source_sequence==command_sequence && source_epoch==command_epoch;
                    child_owned<=0;revoked<=0;cancel_seen<=cancel_valid;cancel_sent<=0;
                    source_snapshot_valid<=source_result_valid;source_snapshot_address<=source_result_address;
                    source_snapshot_bytes<=source_result_bytes;
                    phase<=0;range_index<=0;write_index<=4;other_index<=0;
                    for(integer i=0;i<7;i++)begin
                        address[i]<=command_addr[i];region_base[i]<=command_base[i];region_bytes[i]<=command_bytes[i];
                    end
                    if(source_sequence!=command_sequence || source_epoch!=command_epoch || expected_epoch!=command_epoch)
                        fault<=BAD_IDENTITY;
                    else if(cancel_valid)fault<=CANCELLED;
                    else fault<=OK;
                    state<=CHECK_SOURCE;
                end
                CHECK_SOURCE:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(completion_source_status!=OK || !source_snapshot_valid ||
                        source_snapshot_address!=address[0] || source_snapshot_bytes!=4096)begin
                        completion_status<=BAD_COMPLETION;state<=RESPONSE;
                    end else state<=CHECK_RANGES;
                end
                CHECK_RANGES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!span_valid(address[range_index],region_base[range_index],region_bytes[range_index],span(range_index)))begin
                        completion_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(range_index==6)state<=CHECK_ALIASES;
                    else range_index<=range_index+1'b1;
                end
                CHECK_ALIASES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(overlap)begin completion_status<=BAD_COMMAND;state<=RESPONSE;end
                    else if(other_index!=6)other_index<=other_index+1'b1;
                    else if(write_index!=6)begin write_index<=write_index+1'b1;other_index<=0;end
                    else state<=RESERVE;
                end
                RESERVE:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(exhausted)begin completion_status<=TAG_EXHAUSTED;state<=RESPONSE;end
                    else begin
                        tag<=next_tag;next_tag<=reserved_end[63:0];
                        if(reserved_end[64] || reserved_end[63:0]>LAST_TAG_BASE)exhausted<=1;
                        state<=OFFER;
                    end
                end
                OFFER:if(child_job_ready)begin
                    child_owned<=1;cancel_sent<=0;watchdog<=0;state<=WAIT_CHILD;
                end
                WAIT_CHILD:if(child_done_valid && child_identity)begin
                    completion_phase<=phase;completion_child_status<=child_done_status;
                    completion_child_bytes<=child_done_bytes;completion_child_rows<=child_done_rows;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(child_shape_bad)begin completion_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else if(child_done_status!=OK)begin completion_status<=child_done_status;state<=RESPONSE;end
                    else begin
                        completion_matrices<=completion_matrices+1'b1;
                        completion_total_bytes<=completion_total_bytes+15'(phase_bytes);
                        if(phase==2)begin completion_status<=OK;state<=RESPONSE;end
                        else begin child_owned<=0;watchdog<=0;state<=ADVANCE;end
                    end
                end
                ADVANCE:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(child_job_ready && !child_done_valid)begin
                        phase<=phase+1'b1;tag<=tag+64'd64;cancel_sent<=0;watchdog<=0;state<=OFFER;
                    end
                end
                RESPONSE:if(public_ack)begin
                    if(release_safe)begin state<=IDLE;source_owned<=0;child_owned<=0;end
                    else state<=DRAIN;
                end
                DRAIN:if(release_safe)begin state<=IDLE;source_owned<=0;child_owned<=0;end
                default:begin state<=IDLE;source_owned<=0;child_owned<=0;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
