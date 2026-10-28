// Validate all allocations for input -> attention -> O -> FFN -> residual.
// The parent reserves the cache and every allocation before admission and keeps
// that reservation after accepting this result. This block performs no AXI I/O.
`default_nettype none
module ntrpu_llama32_layer_preflight_r1 (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire command_source_valid,
    input wire [63:0] command_source_sequence,command_source_epoch,command_lease_base,
    input wire [31:0] command_layer,command_position,command_coefficient_position,
    input wire [63:0] command_addr [0:23],
    input wire [63:0] command_base [0:23],
    input wire [63:0] command_bytes [0:23],
    input wire cache_valid,cache_metadata_fault,
    input wire [63:0] cache_base,cache_region_base,cache_region_bytes,cache_sequence,cache_epoch,
    input wire [31:0] cache_capacity,
    output wire [31:0] cache_query_layer,
    input wire cache_query_valid,
    input wire [31:0] cache_query_committed_tokens,
    input wire cancel_valid,
    output wire cancel_ready,busy,
    output wire completion_valid,
    input wire completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [31:0] completion_layer,completion_position,
    output logic completion_revoked,
    output wire result_valid,
    output wire [63:0] validated_addr [0:23],
    output wire [63:0] validated_base [0:23],
    output wire [63:0] validated_bytes [0:23],
    output logic [63:0] validated_lease_base,
    output logic [31:0] validated_coefficient_position,
    output logic [31:0] validated_prefix,
    output logic [4:0] completion_region,
    output logic [4:0] completion_other_region
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,CANCELLED=7,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,CHECK_CONTEXT,CHECK_RANGES,CHECK_PAIRS,RESPONSE} state_t;
    state_t state;
    logic [23:0][63:0] address,allocation_base,allocation_bytes;
    logic source_valid,saved_cache_valid,saved_metadata_fault,saved_query_valid,context_accepted;
    logic [63:0] source_sequence,source_epoch,saved_cache_base,saved_cache_region,saved_cache_bytes;
    logic [63:0] saved_cache_sequence,saved_cache_epoch;
    logic [31:0] capacity;
    logic [4:0] region_index,pair_left,pair_right;
    logic [3:0] fault_now;
    wire [64:0] cache_limit={1'b0,saved_cache_region}+{1'b0,saved_cache_bytes};
    wire [64:0] cache_end={1'b0,saved_cache_base}+{17'd0,capacity,16'd0};
    wire [64:0] region_end={1'b0,allocation_base[region_index]}+{1'b0,allocation_bytes[region_index]};
    wire [64:0] left_end={1'b0,allocation_base[pair_left]}+{1'b0,allocation_bytes[pair_left]};
    wire [64:0] right_end={1'b0,allocation_base[pair_right]}+{1'b0,allocation_bytes[pair_right]};
    wire cache_overlap={1'b0,allocation_base[region_index]}<cache_limit && {1'b0,saved_cache_region}<region_end;
    wire pair_overlap={1'b0,allocation_base[pair_left]}<right_end && {1'b0,allocation_base[pair_right]}<left_end;
    wire cache_changed=busy && (cache_valid!=saved_cache_valid || cache_metadata_fault!=saved_metadata_fault || cache_query_valid!=saved_query_valid ||
        cache_base!=saved_cache_base || cache_region_base!=saved_cache_region || cache_region_bytes!=saved_cache_bytes ||
        cache_capacity!=capacity || cache_sequence!=saved_cache_sequence || cache_epoch!=saved_cache_epoch ||
        cache_query_committed_tokens!=validated_prefix);

    function automatic logic [63:0] span(input logic [4:0] index);
        case(index)
            2,11,15:return 64'd4096;
            3,10:return 64'd4456448;
            4,5:return 64'd1114112;
            7,8:return 64'd2048;
            9:return 64'd256;
            16,17,18:return 64'd17825792;
            19,20:return 64'd65536;
            21:return 64'd16384;
            default:return 64'd8192;
        endcase
    endfunction
    function automatic logic writable(input logic [4:0] index);
        case(index)
            2,6,7,8,11,12,14,15,19,20,21,22,23:return 1'b1;
            default:return 1'b0;
        endcase
    endfunction
    function automatic logic span_valid(input logic [63:0] addr,base,bytes,needed);
        logic [64:0] last,limit;
        last={1'b0,addr}+{1'b0,needed};limit={1'b0,base}+{1'b0,bytes};
        return addr[5:0]==0 && base[5:0]==0 && bytes[5:0]==0 && bytes!=0 &&
            !last[64] && !limit[64] && addr>=base && last<=limit;
    endfunction
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE;
    assign cache_query_layer=busy ? completion_layer:command_layer;
    assign cancel_ready=busy && completion_status==OK && !completion_revoked;
    assign completion_valid=state==RESPONSE;
    assign result_valid=completion_valid && completion_status==OK && !completion_revoked && fault_now==OK;
    for(genvar i=0;i<24;i++)begin
        assign validated_addr[i]=address[i];
        assign validated_base[i]=allocation_base[i];
        assign validated_bytes[i]=allocation_bytes[i];
    end
    always_comb begin
        fault_now=completion_status;
        if(busy && completion_status==OK && !completion_revoked)begin
            if(expected_epoch!=completion_epoch)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(cache_changed)fault_now=BAD_PEER;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;allocation_base<=0;allocation_bytes<=0;
            source_valid<=0;source_sequence<=0;source_epoch<=0;context_accepted<=0;
            saved_cache_valid<=0;saved_metadata_fault<=0;saved_query_valid<=0;
            saved_cache_base<=0;saved_cache_region<=0;saved_cache_bytes<=0;
            saved_cache_sequence<=0;saved_cache_epoch<=0;capacity<=0;
            region_index<=0;pair_left<=0;pair_right<=1;completion_status<=OK;
            completion_sequence<=0;completion_epoch<=0;completion_layer<=0;completion_position<=0;
            completion_revoked<=0;validated_lease_base<=0;validated_coefficient_position<=0;validated_prefix<=0;
            completion_region<=31;completion_other_region<=31;
        end else begin
            if(busy && fault_now!=OK)completion_revoked<=1;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_layer<=command_layer;completion_position<=command_position;
                    validated_lease_base<=command_lease_base;validated_coefficient_position<=command_coefficient_position;
                    source_valid<=command_source_valid;source_sequence<=command_source_sequence;source_epoch<=command_source_epoch;
                    for(integer i=0;i<24;i++)begin
                        address[i]<=command_addr[i];allocation_base[i]<=command_base[i];allocation_bytes[i]<=command_bytes[i];
                    end
                    saved_cache_valid<=cache_valid;saved_metadata_fault<=cache_metadata_fault;saved_query_valid<=cache_query_valid;
                    saved_cache_base<=cache_base;saved_cache_region<=cache_region_base;saved_cache_bytes<=cache_region_bytes;
                    saved_cache_sequence<=cache_sequence;saved_cache_epoch<=cache_epoch;capacity<=cache_capacity;
                    validated_prefix<=cache_query_committed_tokens;context_accepted<=0;
                    region_index<=0;pair_left<=0;pair_right<=1;completion_status<=OK;completion_revoked<=0;
                    completion_region<=31;completion_other_region<=31;state<=CHECK_CONTEXT;
                end
                CHECK_CONTEXT:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(completion_sequence==0 || completion_epoch==0 || !source_valid ||
                        source_sequence!=completion_sequence || source_epoch!=completion_epoch ||
                        saved_cache_sequence!=completion_sequence || saved_cache_epoch!=completion_epoch)begin
                        completion_status<=BAD_IDENTITY;state<=RESPONSE;
                    end else if(!saved_cache_valid || saved_metadata_fault || !saved_query_valid ||
                        capacity==0 || capacity>131072 || completion_layer>=16 || completion_position>=capacity ||
                        completion_position!=validated_prefix || validated_coefficient_position!=completion_position ||
                        validated_lease_base==0 || validated_lease_base>64'hffffffffffffffe0 ||
                        !span_valid(saved_cache_base,saved_cache_region,saved_cache_bytes,64'd64) ||
                        cache_end[64] || cache_end>cache_limit)begin
                        completion_status<=BAD_COMMAND;state<=RESPONSE;
                    end else begin context_accepted<=1;state<=CHECK_RANGES;end
                end
                CHECK_RANGES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!span_valid(address[region_index],allocation_base[region_index],allocation_bytes[region_index],span(region_index)) || cache_overlap)begin
                        completion_status<=BAD_COMMAND;completion_region<=region_index;state<=RESPONSE;
                    end else if(region_index==23)state<=CHECK_PAIRS;
                    else region_index<=region_index+1;
                end
                CHECK_PAIRS:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if((writable(pair_left) || writable(pair_right)) && pair_overlap)begin
                        completion_status<=BAD_COMMAND;completion_region<=pair_left;completion_other_region<=pair_right;state<=RESPONSE;
                    end else if(pair_right==23)begin
                        if(pair_left==22)state<=RESPONSE;
                        else begin pair_left<=pair_left+1;pair_right<=pair_left+2;end
                    end else pair_right<=pair_right+1;
                end
                // Late revocation leaves an already-presented status/identity stable.
                RESPONSE:if(completion_ready)begin state<=IDLE;context_accepted<=0;end
                default:begin completion_status<=BAD_PEER;completion_revoked<=1;state<=RESPONSE;end
            endcase
        end
    end
endmodule
`default_nettype wire
