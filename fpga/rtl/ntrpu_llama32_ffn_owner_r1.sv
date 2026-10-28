`default_nettype none
// Llama 3.2 1B: exact post-O ABI1 -> existing D=2048,H=8192 FFN.
// Public identities remain 64-bit. The child gets a separate aligned 64-bit tag.
// Reset must also quiesce/reset the FFN core and its AXI peer.
module ntrpu_llama32_ffn_owner_r1 #(
    parameter logic [63:0] INITIAL_TAG=64'd0
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire command_source_valid,
    input wire [63:0] command_source_sequence,command_source_epoch,
    input wire [63:0] command_addr [0:7],command_base [0:7],command_bytes [0:7],
    input wire [63:0] command_residual_address,command_residual_base,command_residual_bytes,
    input wire cancel_valid,
    output wire cancel_ready,busy,
    output wire completion_valid,
    input wire completion_ready,
    output logic [3:0] completion_status,completion_child_status,
    output logic [2:0] completion_stages,
    output logic [1:0] completion_phase,
    output logic [16:0] completion_bytes,
    output logic [17:0] completion_stage_bytes,
    output logic [31:0] completion_hls_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output wire result_valid,
    output wire [63:0] result_ffn_address,result_residual_address,
    output wire [13:0] result_ffn_bytes,result_residual_bytes,
    output wire tags_exhausted,
    output wire child_job_valid,
    input wire child_job_ready,
    output wire [1:0] child_job_layout,
    output wire [13:0] child_job_dim,child_job_hidden,
    output wire [5:0] child_job_tile_k [0:2],
    output wire [63:0] child_job_addr [0:7],child_job_base [0:7],child_job_bytes [0:7],
    output wire [31:0] child_job_buffer_epoch [0:7],
    output wire [31:0] child_job_sequence,child_job_epoch,child_expected_epoch,
    output wire child_cancel_valid,
    input wire child_cancel_ready,
    input wire child_done_valid,
    output wire child_done_ready,
    input wire [3:0] child_done_status,child_done_child_status,
    input wire [2:0] child_done_stages,
    input wire [1:0] child_done_phase,
    input wire [16:0] child_done_bytes,
    input wire [17:0] child_done_stage_bytes,
    input wire [31:0] child_done_sequence,child_done_epoch,child_done_hls_status
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_COMPLETION=5,
        CANCELLED=7,TAG_EXHAUSTED=10,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,CHECK_RETAINED,CHECK_WRITES,OFFER,WAIT_CHILD,RESPONSE} state_t;
    state_t state;
    logic [63:0] address [0:7],region_base [0:7],region_bytes [0:7];
    logic [63:0] residual_address,residual_base,residual_bytes,next_tag,tag;
    logic owned,exhausted,revoked,cancel_seen;
    logic [3:0] fault,fault_now;
    logic [2:0] write_index;
    wire epoch_bad=busy && expected_epoch!=completion_epoch;
    wire child_identity={child_done_epoch,child_done_sequence}==tag;
    wire child_success=child_done_status==OK && child_done_child_status==OK &&
        child_done_phase==3 && child_done_stages==4 && child_done_bytes==8192 &&
        child_done_stage_bytes==155648 && child_done_hls_status==0;
    wire child_shape_bad=child_done_stages>4 || child_done_bytes>8192 || child_done_stage_bytes>155648 ||
        (child_done_status==OK && !child_success);
    wire release_public=completion_valid && completion_ready &&
        (!owned || (child_done_valid && child_identity));
    wire [63:0] write_span=write_index<6 ? 64'd65536:write_index==6 ? 64'd16384:64'd8192;
    wire [64:0] write_end={1'b0,address[write_index]}+{1'b0,write_span};
    wire [64:0] retained_end={1'b0,residual_base}+{1'b0,residual_bytes};

    function automatic logic span_valid(input logic [63:0] addr,base,bytes,span);
        logic [64:0] last,limit;
        last={1'b0,addr}+{1'b0,span};limit={1'b0,base}+{1'b0,bytes};
        return addr[5:0]==0 && bytes!=0 && !last[64] && !limit[64] && addr>=base && last<=limit;
    endfunction

    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && child_job_ready && !child_done_valid;
    assign cancel_ready=busy && !cancel_seen;
    assign completion_valid=state==RESPONSE;
    assign tags_exhausted=exhausted;
    assign child_job_valid=state==OFFER;
    assign child_job_layout=2;
    assign child_job_dim=2048;
    assign child_job_hidden=8192;
    assign child_job_sequence=tag[31:0];
    assign child_job_epoch=tag[63:32];
    assign child_expected_epoch=tag[63:32];
    for(genvar i=0;i<3;i++)assign child_job_tile_k[i]=32;
    for(genvar i=0;i<8;i++)begin
        assign child_job_addr[i]=address[i];
        assign child_job_base[i]=region_base[i];
        assign child_job_bytes[i]=region_bytes[i];
        assign child_job_buffer_epoch[i]=tag[63:32];
    end
    // Registered faults avoid a combinational cancel/status path through the child.
    assign child_cancel_valid=owned && (cancel_valid || epoch_bad || fault!=OK || revoked);
    assign child_done_ready=owned && release_public;
    assign result_valid=completion_valid && completion_status==OK && owned &&
        child_done_valid && child_identity && child_success && !revoked && fault==OK && !cancel_valid && !epoch_bad;
    assign result_ffn_address=result_valid ? address[7]:64'd0;
    assign result_residual_address=result_valid ? residual_address:64'd0;
    assign result_ffn_bytes=result_valid ? 14'd8192:14'd0;
    assign result_residual_bytes=result_valid ? 14'd8192:14'd0;

    always_comb begin
        fault_now=fault;
        if(busy && state!=RESPONSE && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(owned && child_done_valid && !child_identity)fault_now=BAD_PEER;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;owned<=0;exhausted<=INITIAL_TAG[1:0]!=0;next_tag<=INITIAL_TAG;tag<=0;
            fault<=OK;revoked<=0;cancel_seen<=0;write_index<=4;
            completion_status<=OK;completion_child_status<=0;completion_stages<=0;completion_phase<=0;
            completion_bytes<=0;completion_stage_bytes<=0;completion_hls_status<=0;
            completion_sequence<=0;completion_epoch<=0;
            residual_address<=0;residual_base<=0;residual_bytes<=0;
            for(integer i=0;i<8;i++)begin address[i]<=0;region_base[i]<=0;region_bytes[i]<=0;end
        end else begin
            if(busy && (epoch_bad || cancel_valid || (owned && child_done_valid &&
                (!child_identity || (state==RESPONSE && !child_success)))))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(busy && state!=RESPONSE)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_status<=0;completion_child_status<=0;completion_stages<=0;completion_phase<=0;
                    completion_bytes<=0;completion_stage_bytes<=0;completion_hls_status<=0;
                    owned<=0;revoked<=0;cancel_seen<=cancel_valid;write_index<=4;
                    if(!command_source_valid || command_source_sequence!=command_sequence ||
                        command_source_epoch!=command_epoch || expected_epoch!=command_epoch)fault<=BAD_IDENTITY;
                    else if(cancel_valid)fault<=CANCELLED;
                    else fault<=OK;
                    for(integer i=0;i<8;i++)begin
                        address[i]<=command_addr[i];region_base[i]<=command_base[i];region_bytes[i]<=command_bytes[i];
                    end
                    residual_address<=command_residual_address;residual_base<=command_residual_base;residual_bytes<=command_residual_bytes;
                    state<=CHECK_RETAINED;
                end
                CHECK_RETAINED:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!span_valid(residual_address,residual_base,residual_bytes,64'd8192))begin
                        completion_status<=BAD_COMMAND;state<=RESPONSE;
                    end else state<=CHECK_WRITES;
                end
                CHECK_WRITES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!span_valid(address[write_index],region_base[write_index],region_bytes[write_index],write_span) ||
                        ({1'b0,address[write_index]}<retained_end && {1'b0,residual_base}<write_end))begin
                        completion_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(write_index!=7)write_index<=write_index+1'b1;
                    else if(exhausted)begin completion_status<=TAG_EXHAUSTED;state<=RESPONSE;end
                    else begin
                        tag<=next_tag;owned<=1;state<=OFFER;
                        if(next_tag==64'hfffffffffffffffc)exhausted<=1;
                        else next_tag<=next_tag+64'd4;
                    end
                end
                OFFER:if(child_job_ready)state<=WAIT_CHILD;
                WAIT_CHILD:if(child_done_valid && child_identity)begin
                    completion_child_status<=child_done_child_status;completion_stages<=child_done_stages;
                    completion_phase<=child_done_phase;completion_bytes<=child_done_bytes;
                    completion_stage_bytes<=child_done_stage_bytes;completion_hls_status<=child_done_hls_status;
                    if(fault_now!=OK)completion_status<=fault_now;
                    else if(child_shape_bad)completion_status<=BAD_COMPLETION;
                    else completion_status<=child_done_status;
                    state<=RESPONSE;
                end
                RESPONSE:if(release_public)begin state<=IDLE;owned<=0;end
                default:begin state<=IDLE;owned<=0;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
