// Hold an owned FFN completion throughout residual calculation and public ACK.
// Source and child must retain their matching completion until acknowledged.
// Reset is coordinated with both peers and memory; foreign identities are never ACKed.
// This controller validates the two operand spans. The enclosing allocator must
// also isolate the output from other still-owned FFN workspace/weight regions.
`default_nettype none
module ntrpu_ffn_residual_handoff_r1 (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire [63:0] command_output_address,command_output_base,command_output_bytes,
    input wire cancel_valid,
    output wire cancel_ready,busy,
    input wire source_completion_valid,
    output wire source_completion_ready,
    input wire [3:0] source_completion_status,
    input wire [63:0] source_sequence,source_epoch,
    input wire source_result_valid,
    input wire [63:0] source_residual_address,source_ffn_address,
    input wire [13:0] source_residual_bytes,source_ffn_bytes,
    output wire child_command_valid,
    input wire child_command_ready,
    output wire [63:0] child_command_sequence,child_command_epoch,
    output wire [63:0] child_source_sequence,child_source_epoch,
    output wire child_source_valid,
    output wire [63:0] child_input_address,child_input_base,child_input_bytes,
    output wire [63:0] child_branch_address,child_branch_base,child_branch_bytes,
    output wire [63:0] child_output_address,child_output_base,child_output_bytes,
    output wire child_cancel_valid,
    input wire child_completion_valid,
    output wire child_completion_ready,
    input wire [3:0] child_completion_status,
    input wire [31:0] child_hls_status,
    input wire [63:0] child_sequence,child_epoch,
    input wire [9:0] child_read_beats,
    input wire [7:0] child_write_beats,
    input wire child_result_valid,
    input wire [63:0] child_result_address,
    input wire [13:0] child_result_bytes,
    output wire residual_bus_owned,
    output wire completion_valid,
    input wire completion_ready,
    output logic [3:0] completion_status,completion_source_status,completion_child_status,
    output logic [31:0] completion_hls_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [9:0] completion_read_beats,
    output logic [7:0] completion_write_beats,
    output wire result_valid,
    output wire [63:0] result_address,
    output wire [13:0] result_bytes
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_COMPLETION=5,CANCELLED=7,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,CHECK,OFFER,WAIT_CHILD,RESPONSE} state_t;
    state_t state;
    logic source_owned,child_owned,revoked,cancel_seen;
    logic [63:0] residual_address,ffn_address,output_address,output_base,output_bytes;
    logic [13:0] residual_bytes,ffn_bytes;
    logic [3:0] fault,fault_now;
    logic source_snapshot_valid;
    wire epoch_bad=busy && expected_epoch!=completion_epoch;
    wire source_identity=source_sequence==completion_sequence && source_epoch==completion_epoch;
    wire child_identity=child_sequence==completion_sequence && child_epoch==completion_epoch;
    wire source_bad=source_owned && (!source_completion_valid || !source_identity ||
        source_completion_status!=completion_source_status ||
        (completion_source_status==OK && (!source_result_valid || source_residual_address!=residual_address ||
        source_ffn_address!=ffn_address || source_residual_bytes!=residual_bytes || source_ffn_bytes!=ffn_bytes)));
    wire child_success=child_completion_status==OK && child_hls_status==0 &&
        child_read_beats==256 && child_write_beats==128 && child_result_valid &&
        child_result_address==output_address && child_result_bytes==8192;
    wire child_shape_bad=child_read_beats>256 || child_write_beats>128 ||
        (child_completion_status==OK && !child_success) || (child_completion_status!=OK && child_result_valid);
    wire child_bad=child_owned && child_completion_valid && !child_identity;
    wire can_release_source=!source_owned || (source_completion_valid && source_identity);
    wire can_release_child=!child_owned || (child_completion_valid && child_identity);
    wire release_public=completion_valid && completion_ready && can_release_source && can_release_child;
    wire [64:0] residual_end={1'b0,residual_address}+65'd8192;
    wire [64:0] ffn_end={1'b0,ffn_address}+65'd8192;
    wire [64:0] output_end={1'b0,output_address}+65'd8192;
    wire [64:0] output_limit={1'b0,output_base}+{1'b0,output_bytes};
    wire bounds_bad=residual_bytes!=8192 || ffn_bytes!=8192 || residual_address[5:0]!=0 ||
        ffn_address[5:0]!=0 || residual_end[64] || ffn_end[64] || output_address[5:0]!=0 ||
        output_bytes==0 || output_end[64] || output_limit[64] || output_address<output_base || output_end>output_limit ||
        ({1'b0,residual_address}<ffn_end && {1'b0,ffn_address}<residual_end) ||
        ({1'b0,output_address}<residual_end && {1'b0,residual_address}<output_end) ||
        ({1'b0,output_address}<ffn_end && {1'b0,ffn_address}<output_end);

    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && source_completion_valid && child_command_ready && !child_completion_valid;
    assign cancel_ready=busy && !cancel_seen;
    assign completion_valid=state==RESPONSE;
    assign source_completion_ready=source_owned && release_public;
    assign child_completion_ready=child_owned && release_public;
    assign residual_bus_owned=child_owned;
    // An offered command is never withdrawn or changed under backpressure.
    assign child_command_valid=state==OFFER;
    assign child_command_sequence=completion_sequence;
    assign child_command_epoch=completion_epoch;
    assign child_source_sequence=completion_sequence;
    assign child_source_epoch=completion_epoch;
    assign child_source_valid=source_snapshot_valid;
    assign child_input_address=residual_address;
    assign child_input_base=residual_address;
    assign child_input_bytes=64'd8192;
    assign child_branch_address=ffn_address;
    assign child_branch_base=ffn_address;
    assign child_branch_bytes=64'd8192;
    assign child_output_address=output_address;
    assign child_output_base=output_base;
    assign child_output_bytes=output_bytes;
    assign child_cancel_valid=child_owned && (cancel_valid || epoch_bad || source_bad || fault!=OK || revoked);
    assign result_valid=completion_valid && completion_status==OK && source_owned && child_owned &&
        source_completion_valid && !source_bad && child_completion_valid && child_identity && child_success &&
        !revoked && !epoch_bad && !cancel_valid;
    assign result_address=result_valid ? output_address:64'd0;
    assign result_bytes=result_valid ? 14'd8192:14'd0;

    always_comb begin
        fault_now=fault;
        if(busy && state!=RESPONSE && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(source_bad || child_bad)fault_now=BAD_PEER;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;source_owned<=0;child_owned<=0;revoked<=0;cancel_seen<=0;source_snapshot_valid<=0;
            residual_address<=0;ffn_address<=0;residual_bytes<=0;ffn_bytes<=0;
            output_address<=0;output_base<=0;output_bytes<=0;fault<=OK;
            completion_status<=OK;completion_source_status<=OK;completion_child_status<=OK;
            completion_hls_status<=0;completion_sequence<=0;completion_epoch<=0;
            completion_read_beats<=0;completion_write_beats<=0;
        end else begin
            if(busy)begin
                if(cancel_valid)cancel_seen<=1;
                if(epoch_bad || cancel_valid || source_bad || child_bad ||
                    (state==RESPONSE && completion_status==OK && child_owned &&
                    (!child_completion_valid || !child_success)))revoked<=1;
                if(state!=RESPONSE)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_status<=OK;completion_source_status<=source_completion_status;
                    completion_child_status<=OK;completion_hls_status<=0;completion_read_beats<=0;completion_write_beats<=0;
                    source_owned<=command_sequence==source_sequence && command_epoch==source_epoch;
                    child_owned<=0;revoked<=0;cancel_seen<=cancel_valid;source_snapshot_valid<=source_result_valid;
                    residual_address<=source_residual_address;ffn_address<=source_ffn_address;
                    residual_bytes<=source_residual_bytes;ffn_bytes<=source_ffn_bytes;
                    output_address<=command_output_address;output_base<=command_output_base;output_bytes<=command_output_bytes;
                    if(command_sequence!=source_sequence || command_epoch!=source_epoch || command_epoch!=expected_epoch)fault<=BAD_IDENTITY;
                    else if(cancel_valid)fault<=CANCELLED;
                    else if(source_completion_status!=OK)fault<=source_completion_status;
                    else if(!source_result_valid)fault<=BAD_PEER;
                    else fault<=OK;
                    state<=CHECK;
                end
                CHECK:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(bounds_bad)begin completion_status<=BAD_COMMAND;state<=RESPONSE;end
                    else begin child_owned<=1;state<=OFFER;end
                end
                OFFER:if(child_command_ready)state<=WAIT_CHILD;
                WAIT_CHILD:if(child_completion_valid && child_identity)begin
                    completion_child_status<=child_completion_status;completion_hls_status<=child_hls_status;
                    completion_read_beats<=child_read_beats;completion_write_beats<=child_write_beats;
                    if(fault_now!=OK)completion_status<=fault_now;
                    else if(child_shape_bad)completion_status<=BAD_COMPLETION;
                    else completion_status<=child_completion_status;
                    state<=RESPONSE;
                end
                RESPONSE:if(release_public)begin state<=IDLE;source_owned<=0;child_owned<=0;end
                default:begin state<=IDLE;source_owned<=0;child_owned<=0;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
