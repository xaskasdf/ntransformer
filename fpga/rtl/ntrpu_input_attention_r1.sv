// Actual embedding -> normalization/QKV -> retained RoPE/KV/32-head attention.
// Preflight precedes producer admission; four independent AXI masters share external memory.
// Reset must quiesce the complete hierarchy and all external AXI peers.
`default_nettype none
module ntrpu_input_attention_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter integer MATRIX_TIMEOUT_CYCLES=16777216,
    parameter integer HANDOFF_TIMEOUT_CYCLES=67108864,
    parameter integer OWNER_TIMEOUT_CYCLES=268435456
) (
    input wire clk,
    input wire rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,
    input wire [63:0] command_epoch,
    input wire [63:0] command_lease_base,
    input wire [63:0] expected_epoch,
    input wire [31:0] command_layer,
    input wire [31:0] command_position,
    input wire [31:0] command_coefficient_position,
    input wire [63:0] command_addr [0:9],
    input wire [63:0] command_base [0:9],
    input wire [63:0] command_bytes [0:9],
    input wire cancel_valid,
    output wire cancel_ready,
    output wire busy,
    output wire cache_exclusive,
    output wire completion_valid,
    input wire completion_ready,
    output logic [63:0] completion_sequence,
    output logic [63:0] completion_epoch,
    output logic [31:0] completion_layer,
    output logic [31:0] completion_position,
    output logic [3:0] completion_status,
    output logic [3:0] completion_source_status,
    output logic [3:0] completion_child_status,
    output logic completion_kv_committed,
    output logic [12:0] completion_kv_bytes,
    output logic [31:0] completion_kv_tokens,
    output wire completion_revoked,
    output wire result_valid,
    input wire result_ce,
    input wire [10:0] result_address,
    output wire [31:0] result_q,
    output wire result_q_valid,
    input wire [63:0] command_source_sequence,
    input wire [63:0] command_source_epoch,
    input wire command_source_valid,
    output logic [3:0] completion_norm_status,
    output logic [3:0] completion_qkv_status,
    output logic [3:0] completion_matrix_status,
    output logic [31:0] completion_norm_hls_status,
    output logic [9:0] completion_norm_read_beats,
    output logic [6:0] completion_norm_records,
    output logic [1:0] completion_phase,
    output logic [1:0] completion_matrices,
    output logic [14:0] completion_total_bytes,
    output wire norm_tag_exhausted,
    output wire qkv_tags_exhausted,
    output wire [63:0] input_axi_araddr,
    output wire [7:0] input_axi_arlen,
    output wire [2:0] input_axi_arsize,
    output wire [2:0] input_axi_arprot,
    output wire [1:0] input_axi_arburst,
    output wire [3:0] input_axi_arid,
    output wire [3:0] input_axi_arcache,
    output wire input_axi_arlock,
    output wire input_axi_arvalid,
    input wire input_axi_arready,
    input wire [511:0] input_axi_rdata,
    input wire [1:0] input_axi_rresp,
    input wire [3:0] input_axi_rid,
    input wire input_axi_rlast,
    input wire input_axi_rvalid,
    output wire input_axi_rready,
    output wire [63:0] input_axi_awaddr,
    output wire [7:0] input_axi_awlen,
    output wire [2:0] input_axi_awsize,
    output wire [2:0] input_axi_awprot,
    output wire [1:0] input_axi_awburst,
    output wire [3:0] input_axi_awid,
    output wire [3:0] input_axi_awcache,
    output wire input_axi_awlock,
    output wire input_axi_awvalid,
    input wire input_axi_awready,
    output wire [511:0] input_axi_wdata,
    output wire [63:0] input_axi_wstrb,
    output wire input_axi_wlast,
    output wire input_axi_wvalid,
    input wire input_axi_wready,
    input wire [3:0] input_axi_bid,
    input wire [1:0] input_axi_bresp,
    input wire input_axi_bvalid,
    output wire input_axi_bready,
    output wire [63:0] v_axi_araddr,
    output wire [7:0] v_axi_arlen,
    output wire [2:0] v_axi_arsize,
    output wire [2:0] v_axi_arprot,
    output wire [1:0] v_axi_arburst,
    output wire [3:0] v_axi_arid,
    output wire [3:0] v_axi_arcache,
    output wire v_axi_arlock,
    output wire v_axi_arvalid,
    input wire v_axi_arready,
    input wire [511:0] v_axi_rdata,
    input wire [1:0] v_axi_rresp,
    input wire [3:0] v_axi_rid,
    input wire v_axi_rlast,
    input wire v_axi_rvalid,
    output wire v_axi_rready,
    output wire [63:0] rope_axi_araddr,
    output wire [7:0] rope_axi_arlen,
    output wire [2:0] rope_axi_arsize,
    output wire [2:0] rope_axi_arprot,
    output wire [1:0] rope_axi_arburst,
    output wire [3:0] rope_axi_arid,
    output wire [3:0] rope_axi_arcache,
    output wire rope_axi_arlock,
    output wire rope_axi_arvalid,
    input wire rope_axi_arready,
    input wire [511:0] rope_axi_rdata,
    input wire [1:0] rope_axi_rresp,
    input wire [3:0] rope_axi_rid,
    input wire rope_axi_rlast,
    input wire rope_axi_rvalid,
    output wire rope_axi_rready,
    input wire bind_valid,
    output wire bind_ready,
    input wire [63:0] bind_cache_base,
    input wire [63:0] bind_region_base,
    input wire [63:0] bind_region_bytes,
    input wire [31:0] bind_capacity,
    input wire [63:0] bind_sequence,
    input wire [63:0] bind_epoch,
    output wire cache_valid,
    output wire [63:0] cache_base,
    output wire [63:0] cache_region_base,
    output wire [63:0] cache_region_bytes,
    output wire [63:0] cache_sequence,
    output wire [63:0] cache_epoch,
    output wire [31:0] cache_capacity,
    output wire [63:0] kv_axi_awaddr,
    output wire [7:0] kv_axi_awlen,
    output wire [2:0] kv_axi_awsize,
    output wire [2:0] kv_axi_awprot,
    output wire [1:0] kv_axi_awburst,
    output wire [3:0] kv_axi_awid,
    output wire [3:0] kv_axi_awcache,
    output wire kv_axi_awlock,
    output wire kv_axi_awvalid,
    input wire kv_axi_awready,
    output wire [511:0] kv_axi_wdata,
    output wire [63:0] kv_axi_wstrb,
    output wire kv_axi_wlast,
    output wire kv_axi_wvalid,
    input wire kv_axi_wready,
    input wire [3:0] kv_axi_bid,
    input wire [1:0] kv_axi_bresp,
    input wire kv_axi_bvalid,
    output wire kv_axi_bready,
    output wire [63:0] kv_axi_araddr,
    output wire [7:0] kv_axi_arlen,
    output wire [2:0] kv_axi_arsize,
    output wire [2:0] kv_axi_arprot,
    output wire [1:0] kv_axi_arburst,
    output wire [3:0] kv_axi_arid,
    output wire [3:0] kv_axi_arcache,
    output wire kv_axi_arlock,
    output wire kv_axi_arvalid,
    input wire kv_axi_arready,
    input wire [511:0] kv_axi_rdata,
    input wire [1:0] kv_axi_rresp,
    input wire [3:0] kv_axi_rid,
    input wire kv_axi_rlast,
    input wire kv_axi_rvalid,
    output wire kv_axi_rready,
    output wire root_cache_lock,
    output wire cache_metadata_fault,
    output wire [31:0] cache_query_layer,
    output wire cache_query_valid,
    output wire [31:0] cache_query_committed_tokens,
    output wire [31:0] cache_committed_tokens,
    output wire bind_completion_valid,
    input wire bind_completion_ready,
    output wire [3:0] bind_completion_status,
    output wire [63:0] bind_completion_sequence,
    output wire [63:0] bind_completion_epoch
);
    wire producer_command_valid;
    wire producer_command_ready;
    wire [63:0] producer_command_sequence;
    wire [63:0] producer_command_epoch;
    wire [63:0] producer_command_source_sequence;
    wire [63:0] producer_command_source_epoch;
    wire producer_command_source_valid;
    wire [63:0] producer_command_addr [0:8];
    wire [63:0] producer_command_base [0:8];
    wire [63:0] producer_command_bytes [0:8];
    wire producer_cancel_valid;
    wire producer_cancel_ready;
    wire producer_busy;
    wire producer_completion_valid;
    wire producer_completion_ready;
    wire [63:0] producer_completion_sequence;
    wire [63:0] producer_completion_epoch;
    wire [3:0] producer_completion_status;
    wire [3:0] producer_completion_norm_status;
    wire [3:0] producer_completion_qkv_status;
    wire [3:0] producer_completion_matrix_status;
    wire [31:0] producer_completion_norm_hls_status;
    wire [9:0] producer_completion_norm_read_beats;
    wire [6:0] producer_completion_norm_records;
    wire [1:0] producer_completion_phase;
    wire [1:0] producer_completion_matrices;
    wire [14:0] producer_completion_total_bytes;
    wire producer_completion_revoked;
    wire producer_result_valid;
    wire [63:0] producer_result_address [0:2];
    wire [13:0] producer_result_bytes [0:2];
    wire lease_command_valid;
    wire lease_command_ready;
    wire [63:0] lease_command_sequence;
    wire [63:0] lease_command_epoch;
    wire [63:0] lease_command_lease_base;
    wire [31:0] lease_command_layer;
    wire [31:0] lease_command_position;
    wire [31:0] lease_command_coefficient_position;
    wire [63:0] lease_command_addr [0:9];
    wire [63:0] lease_command_base [0:9];
    wire [63:0] lease_command_bytes [0:9];
    wire lease_cancel_valid;
    wire lease_cancel_ready;
    wire lease_busy;
    wire lease_cache_exclusive;
    wire lease_source_completion_ready;
    wire consumer_command_valid;
    wire consumer_command_ready;
    wire consumer_busy;
    wire [63:0] consumer_command_sequence;
    wire [63:0] consumer_command_epoch;
    wire [63:0] consumer_command_lease_base;
    wire [31:0] consumer_command_layer;
    wire [31:0] consumer_command_position;
    wire [31:0] consumer_command_coefficient_position;
    wire [63:0] consumer_command_source_sequence;
    wire [63:0] consumer_command_source_epoch;
    wire consumer_command_source_valid;
    wire [31:0] consumer_command_source_position;
    wire [63:0] consumer_command_v_source_sequence;
    wire [63:0] consumer_command_v_source_epoch;
    wire consumer_command_v_source_valid;
    wire [31:0] consumer_command_v_source_position;
    wire [63:0] consumer_command_addr [0:3];
    wire [63:0] consumer_command_base [0:3];
    wire [63:0] consumer_command_bytes [0:3];
    wire consumer_cancel_valid;
    wire consumer_cancel_ready;
    wire consumer_completion_valid;
    wire consumer_completion_ready;
    wire [63:0] consumer_completion_sequence;
    wire [63:0] consumer_completion_epoch;
    wire [31:0] consumer_completion_layer;
    wire [31:0] consumer_completion_position;
    wire [3:0] consumer_completion_status;
    wire [3:0] consumer_completion_append_status;
    wire [3:0] consumer_completion_root_status;
    wire [3:0] consumer_completion_capture_status;
    wire [3:0] consumer_completion_rope_status;
    wire [31:0] consumer_completion_rope_hls_status;
    wire [9:0] consumer_completion_rope_beats;
    wire [11:0] consumer_completion_capture_qk_words;
    wire [5:0] consumer_completion_capture_v_beats;
    wire [6:0] consumer_completion_kv_frames;
    wire [7:0] consumer_completion_q_frames;
    wire [12:0] consumer_completion_kv_bytes;
    wire [31:0] consumer_completion_kv_tokens;
    wire consumer_completion_kv_committed;
    wire consumer_result_valid;
    wire consumer_completion_revoked;
    wire lease_completion_valid;
    wire lease_completion_ready;
    wire [63:0] lease_completion_sequence;
    wire [63:0] lease_completion_epoch;
    wire [31:0] lease_completion_layer;
    wire [31:0] lease_completion_position;
    wire [3:0] lease_completion_status;
    wire [3:0] lease_completion_source_status;
    wire [3:0] lease_completion_child_status;
    wire lease_completion_kv_committed;
    wire [12:0] lease_completion_kv_bytes;
    wire [31:0] lease_completion_kv_tokens;
    wire lease_completion_revoked;
    wire lease_result_valid;
    wire lease_result_ce;
    wire [10:0] lease_result_address;
    wire consumer_result_ce;
    wire [10:0] consumer_result_address;
    wire [31:0] consumer_result_q;
    wire consumer_result_q_valid;
    wire [31:0] lease_result_q;
    wire lease_result_q_valid;
    wire consumer_cache_exclusive;
    wire consumer_bind_valid;
    wire consumer_bind_ready;

    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,CANCELLED=7,TIMED_OUT=8,BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,CHECK_RANGES,CHECK_ALIASES,SOURCE_OFFER,SOURCE_WAIT,
        LEASE_OFFER,LEASE_WAIT,RESPONSE,ACK_LEASE,DRAIN_LEASE,ACK_SOURCE,DRAIN_SOURCE} state_t;
    state_t state;
    logic [9:0][63:0] address,region_base,region_size;
    logic [63:0] held_lease,held_source_sequence,held_source_epoch;
    logic held_source_valid,source_owned,lease_owned,source_cancel_sent,lease_cancel_sent,revoked,lease_seen;
    logic [31:0] held_coefficient_position,capacity,prefix,watchdog;
    logic [63:0] saved_cache_base,saved_cache_region,saved_cache_bytes;
    logic [3:0] range_index,write_index,other_index,fault,fault_now;
    logic [59:0] saved_lease;
    wire [59:0] lease_snapshot={lease_completion_status,lease_completion_source_status,lease_completion_child_status,
        lease_completion_kv_committed,lease_completion_kv_bytes,lease_completion_kv_tokens,lease_completion_revoked,lease_result_valid};
    wire owner_busy=state!=IDLE;
    wire source_identity=producer_completion_sequence==completion_sequence && producer_completion_epoch==completion_epoch;
    wire lease_identity=lease_completion_sequence==completion_sequence && lease_completion_epoch==completion_epoch &&
        lease_completion_layer==completion_layer && lease_completion_position==completion_position;
    wire cache_idle=!consumer_busy && !consumer_cache_exclusive && !consumer_completion_valid;
    wire cache_bad=owner_busy && (!cache_valid || cache_metadata_fault || cache_sequence!=completion_sequence ||
        cache_epoch!=completion_epoch || cache_base!=saved_cache_base || cache_region_base!=saved_cache_region ||
        cache_region_bytes!=saved_cache_bytes || cache_capacity!=capacity ||
        (!lease_owned && cache_query_committed_tokens!=prefix));
    wire source_bad=source_owned && !lease_owned &&
        ((producer_completion_valid && !source_identity) || (state==LEASE_OFFER && !producer_completion_valid));
    wire lease_bad=lease_owned && ((lease_completion_valid && !lease_identity) ||
        (lease_seen && state==RESPONSE && (!lease_completion_valid || lease_snapshot!=saved_lease)));
    wire waiting=state==SOURCE_OFFER || state==SOURCE_WAIT || state==LEASE_OFFER || state==LEASE_WAIT;
    wire progress=(producer_command_valid && producer_command_ready) ||
        (state==SOURCE_WAIT && producer_completion_valid && source_identity) ||
        (lease_command_valid && lease_command_ready) || (state==LEASE_WAIT && lease_completion_valid && lease_identity);
    wire timed_out=waiting && watchdog>=32'(OWNER_TIMEOUT_CYCLES-1) && !progress;
    wire [64:0] cache_limit={1'b0,saved_cache_region}+{1'b0,saved_cache_bytes};
    wire [64:0] cache_end={1'b0,saved_cache_base}+{17'd0,capacity,16'd0};
    wire [64:0] range_end={1'b0,region_base[range_index]}+{1'b0,region_size[range_index]};
    wire cache_overlap={1'b0,region_base[range_index]}<cache_limit && {1'b0,saved_cache_region}<range_end;
    wire [3:0] write_slot=write_index==0 ? 4'd2:write_index+4'd5;
    wire [64:0] write_end={1'b0,region_base[write_slot]}+{1'b0,region_size[write_slot]};
    wire [64:0] other_end={1'b0,region_base[other_index]}+{1'b0,region_size[other_index]};
    wire alias_write=write_slot!=other_index && {1'b0,region_base[write_slot]}<other_end &&
        {1'b0,region_base[other_index]}<write_end;
    function automatic logic [63:0] span(input logic [3:0] index);
        case(index)
            0,1,6:return 64'd8192;
            2:return 64'd4096;
            3:return 64'd4456448;
            4,5:return 64'd1114112;
            9:return 64'd256;
            default:return 64'd2048;
        endcase
    endfunction
    function automatic logic valid_span(input logic [63:0] addr,base,bytes,needed);
        logic [64:0] last,limit;
        last={1'b0,addr}+{1'b0,needed};limit={1'b0,base}+{1'b0,bytes};
        return addr[5:0]==0 && base[5:0]==0 && bytes[5:0]==0 && bytes!=0 &&
            !last[64] && !limit[64] && addr>=base && last<=limit;
    endfunction
    assign busy=owner_busy || producer_busy || lease_busy || consumer_busy;
    assign cache_exclusive=owner_busy || producer_busy || lease_busy || consumer_cache_exclusive;
    assign command_ready=!owner_busy && !bind_valid && cache_idle && !producer_busy &&
        !producer_completion_valid && !lease_busy && !lease_completion_valid;
    assign consumer_bind_valid=bind_valid && !owner_busy && !producer_busy && !lease_busy;
    assign bind_ready=consumer_bind_ready && !owner_busy && !producer_busy && !lease_busy;
    assign cache_query_layer=owner_busy ? completion_layer:command_layer;
    assign cancel_ready=owner_busy && state!=ACK_LEASE && state!=DRAIN_LEASE &&
        state!=ACK_SOURCE && state!=DRAIN_SOURCE && fault==OK && !revoked;
    assign producer_command_valid=state==SOURCE_OFFER && fault_now==OK;
    assign producer_command_sequence=completion_sequence;
    assign producer_command_epoch=completion_epoch;
    assign producer_command_source_valid=held_source_valid;
    assign producer_command_source_sequence=held_source_sequence;
    assign producer_command_source_epoch=held_source_epoch;
    assign producer_cancel_valid=source_owned && !lease_owned && fault_now!=OK && !source_cancel_sent && !producer_completion_valid;
    assign producer_completion_ready=lease_owned ? lease_source_completion_ready:
        (state==ACK_SOURCE && source_owned && producer_completion_valid && source_identity);
    assign lease_command_valid=state==LEASE_OFFER && fault_now==OK && producer_completion_valid && source_identity;
    assign lease_command_sequence=completion_sequence;
    assign lease_command_epoch=completion_epoch;
    assign lease_command_layer=completion_layer;
    assign lease_command_position=completion_position;
    assign lease_command_coefficient_position=held_coefficient_position;
    assign lease_command_lease_base=held_lease;
    assign lease_cancel_valid=lease_owned && fault_now!=OK && !lease_cancel_sent;
    assign lease_completion_ready=state==ACK_LEASE && lease_owned && lease_completion_valid && lease_identity;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked || (lease_owned && lease_completion_revoked);
    assign result_valid=completion_valid && completion_status==OK && !completion_revoked && fault_now==OK &&
        lease_owned && lease_completion_valid && lease_identity && lease_result_valid && !lease_bad;
    assign lease_result_ce=result_ce && result_valid;
    assign lease_result_address=result_address;
    assign result_q=lease_result_q;
    assign result_q_valid=result_valid && lease_result_q_valid;
    for(genvar i=0;i<10;i++)begin
        assign lease_command_addr[i]=address[i];
        assign lease_command_base[i]=region_base[i];
        assign lease_command_bytes[i]=region_size[i];
        if(i<9)begin
            assign producer_command_addr[i]=address[i];
            assign producer_command_base[i]=region_base[i];
            assign producer_command_bytes[i]=region_size[i];
        end
    end
    always_comb begin
        fault_now=fault;
        if(owner_busy && fault==OK)begin
            if(expected_epoch!=completion_epoch)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(cache_bad || source_bad || lease_bad)fault_now=BAD_PEER;
            else if(timed_out)fault_now=TIMED_OUT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;region_base<=0;region_size<=0;held_lease<=0;
            held_source_sequence<=0;held_source_epoch<=0;held_source_valid<=0;
            source_owned<=0;lease_owned<=0;source_cancel_sent<=0;lease_cancel_sent<=0;revoked<=0;lease_seen<=0;
            held_coefficient_position<=0;capacity<=0;prefix<=0;watchdog<=0;
            saved_cache_base<=0;saved_cache_region<=0;saved_cache_bytes<=0;
            range_index<=0;write_index<=0;other_index<=0;fault<=0;saved_lease<=0;
            completion_sequence<=0;completion_epoch<=0;completion_layer<=0;completion_position<=0;
            completion_status<=0;completion_source_status<=0;completion_child_status<=0;
            completion_kv_committed<=0;completion_kv_bytes<=0;completion_kv_tokens<=0;
            completion_norm_status<=0;completion_qkv_status<=0;completion_matrix_status<=0;completion_norm_hls_status<=0;
            completion_norm_read_beats<=0;completion_norm_records<=0;completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
        end else begin
            if(waiting && !progress && fault==OK)watchdog<=watchdog+1;else watchdog<=0;
            if(owner_busy && fault_now!=OK)begin fault<=fault_now;revoked<=1;end
            if(producer_cancel_valid && producer_cancel_ready)source_cancel_sent<=1;
            if(lease_cancel_valid && lease_cancel_ready)lease_cancel_sent<=1;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_layer<=command_layer;completion_position<=command_position;
                    held_coefficient_position<=command_coefficient_position;held_lease<=command_lease_base;
                    held_source_sequence<=command_source_sequence;held_source_epoch<=command_source_epoch;held_source_valid<=command_source_valid;
                    for(integer i=0;i<10;i++)begin address[i]<=command_addr[i];region_base[i]<=command_base[i];region_size[i]<=command_bytes[i];end
                    saved_cache_base<=cache_base;saved_cache_region<=cache_region_base;saved_cache_bytes<=cache_region_bytes;
                    capacity<=cache_capacity;prefix<=cache_query_committed_tokens;
                    source_owned<=0;lease_owned<=0;source_cancel_sent<=0;lease_cancel_sent<=0;revoked<=0;lease_seen<=0;fault<=0;
                    range_index<=0;write_index<=0;other_index<=0;saved_lease<=0;
                    completion_status<=0;completion_source_status<=0;completion_child_status<=0;
                    completion_kv_committed<=0;completion_kv_bytes<=0;completion_kv_tokens<=0;
                    completion_norm_status<=0;completion_qkv_status<=0;completion_matrix_status<=0;completion_norm_hls_status<=0;
                    completion_norm_read_beats<=0;completion_norm_records<=0;completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
                    if(command_sequence==0 || command_epoch==0 || command_epoch!=expected_epoch || !command_source_valid ||
                        command_source_sequence!=command_sequence || command_source_epoch!=command_epoch ||
                        cache_sequence!=command_sequence || cache_epoch!=command_epoch)begin
                        completion_status<=BAD_IDENTITY;fault<=BAD_IDENTITY;state<=RESPONSE;
                    end else state<=CHECK_RANGES;
                end
                CHECK_RANGES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!cache_valid || cache_metadata_fault || !cache_query_valid || capacity==0 || capacity>131072 ||
                        completion_layer>=16 || completion_position>=capacity || completion_position!=prefix ||
                        held_coefficient_position!=completion_position || held_lease==0 || held_lease>64'hffffffffffffffe0 ||
                        !valid_span(saved_cache_base,saved_cache_region,saved_cache_bytes,64'd64) ||
                        cache_end[64] || cache_end>cache_limit ||
                        !valid_span(address[range_index],region_base[range_index],region_size[range_index],span(range_index)) || cache_overlap)begin
                        completion_status<=BAD_COMMAND;fault<=BAD_COMMAND;state<=RESPONSE;
                    end else if(range_index==9)state<=CHECK_ALIASES;
                    else range_index<=range_index+1;
                end
                CHECK_ALIASES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(alias_write)begin completion_status<=BAD_COMMAND;fault<=BAD_COMMAND;state<=RESPONSE;end
                    else if(other_index==9)begin
                        other_index<=0;
                        if(write_index==3)state<=SOURCE_OFFER;else write_index<=write_index+1;
                    end else other_index<=other_index+1;
                end
                SOURCE_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(producer_command_ready)begin source_owned<=1;state<=SOURCE_WAIT;end
                SOURCE_WAIT:if(producer_completion_valid && source_identity)begin
                    completion_source_status<=producer_completion_status;
                    completion_norm_status<=producer_completion_norm_status;completion_qkv_status<=producer_completion_qkv_status;
                    completion_matrix_status<=producer_completion_matrix_status;completion_norm_hls_status<=producer_completion_norm_hls_status;
                    completion_norm_read_beats<=producer_completion_norm_read_beats;completion_norm_records<=producer_completion_norm_records;
                    completion_phase<=producer_completion_phase;completion_matrices<=producer_completion_matrices;completion_total_bytes<=producer_completion_total_bytes;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else state<=LEASE_OFFER;
                end
                LEASE_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(lease_command_ready)begin lease_owned<=1;state<=LEASE_WAIT;end
                LEASE_WAIT:if(lease_completion_valid && lease_identity)begin
                    saved_lease<=lease_snapshot;lease_seen<=1;
                    completion_status<=fault_now==OK ? lease_completion_status:fault_now;
                    completion_source_status<=lease_completion_source_status;completion_child_status<=lease_completion_child_status;
                    completion_kv_committed<=lease_completion_kv_committed;completion_kv_bytes<=lease_completion_kv_bytes;completion_kv_tokens<=lease_completion_kv_tokens;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=lease_owned ? ACK_LEASE:ACK_SOURCE;
                ACK_LEASE:if(lease_completion_ready)state<=DRAIN_LEASE;
                DRAIN_LEASE:if(!lease_busy && !lease_completion_valid && !producer_busy && !producer_completion_valid &&
                    !consumer_busy && !consumer_completion_valid)begin source_owned<=0;lease_owned<=0;state<=IDLE;end
                ACK_SOURCE:if(!source_owned || producer_completion_ready)begin source_owned<=0;state<=DRAIN_SOURCE;end
                DRAIN_SOURCE:if(!producer_busy && !producer_completion_valid)state<=IDLE;
                default:begin fault<=BAD_PEER;revoked<=1;completion_status<=BAD_PEER;state<=RESPONSE;end
            endcase
        end
    end

    ntrpu_input_qkv_real_r1 #(.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.NORM_HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES),.MATRIX_TIMEOUT_CYCLES(MATRIX_TIMEOUT_CYCLES),.HANDOFF_TIMEOUT_CYCLES(HANDOFF_TIMEOUT_CYCLES)) producer(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(producer_command_valid),
        .command_ready(producer_command_ready),
        .command_sequence(producer_command_sequence),
        .command_epoch(producer_command_epoch),
        .expected_epoch(expected_epoch),
        .command_source_sequence(producer_command_source_sequence),
        .command_source_epoch(producer_command_source_epoch),
        .command_source_valid(producer_command_source_valid),
        .command_addr(producer_command_addr),
        .command_base(producer_command_base),
        .command_bytes(producer_command_bytes),
        .cancel_valid(producer_cancel_valid),
        .cancel_ready(producer_cancel_ready),
        .busy(producer_busy),
        .completion_valid(producer_completion_valid),
        .completion_ready(producer_completion_ready),
        .completion_sequence(producer_completion_sequence),
        .completion_epoch(producer_completion_epoch),
        .completion_status(producer_completion_status),
        .completion_norm_status(producer_completion_norm_status),
        .completion_qkv_status(producer_completion_qkv_status),
        .completion_matrix_status(producer_completion_matrix_status),
        .completion_norm_hls_status(producer_completion_norm_hls_status),
        .completion_norm_read_beats(producer_completion_norm_read_beats),
        .completion_norm_records(producer_completion_norm_records),
        .completion_phase(producer_completion_phase),
        .completion_matrices(producer_completion_matrices),
        .completion_total_bytes(producer_completion_total_bytes),
        .completion_revoked(producer_completion_revoked),
        .result_valid(producer_result_valid),
        .result_address(producer_result_address),
        .result_bytes(producer_result_bytes),
        .norm_tag_exhausted(norm_tag_exhausted),
        .qkv_tags_exhausted(qkv_tags_exhausted),
        .m_axi_araddr(input_axi_araddr),
        .m_axi_arlen(input_axi_arlen),
        .m_axi_arsize(input_axi_arsize),
        .m_axi_arprot(input_axi_arprot),
        .m_axi_arburst(input_axi_arburst),
        .m_axi_arid(input_axi_arid),
        .m_axi_arcache(input_axi_arcache),
        .m_axi_arlock(input_axi_arlock),
        .m_axi_arvalid(input_axi_arvalid),
        .m_axi_arready(input_axi_arready),
        .m_axi_rdata(input_axi_rdata),
        .m_axi_rresp(input_axi_rresp),
        .m_axi_rid(input_axi_rid),
        .m_axi_rlast(input_axi_rlast),
        .m_axi_rvalid(input_axi_rvalid),
        .m_axi_rready(input_axi_rready),
        .m_axi_awaddr(input_axi_awaddr),
        .m_axi_awlen(input_axi_awlen),
        .m_axi_awsize(input_axi_awsize),
        .m_axi_awprot(input_axi_awprot),
        .m_axi_awburst(input_axi_awburst),
        .m_axi_awid(input_axi_awid),
        .m_axi_awcache(input_axi_awcache),
        .m_axi_awlock(input_axi_awlock),
        .m_axi_awvalid(input_axi_awvalid),
        .m_axi_awready(input_axi_awready),
        .m_axi_wdata(input_axi_wdata),
        .m_axi_wstrb(input_axi_wstrb),
        .m_axi_wlast(input_axi_wlast),
        .m_axi_wvalid(input_axi_wvalid),
        .m_axi_wready(input_axi_wready),
        .m_axi_bid(input_axi_bid),
        .m_axi_bresp(input_axi_bresp),
        .m_axi_bvalid(input_axi_bvalid),
        .m_axi_bready(input_axi_bready)
    );
    ntrpu_qkv_rope_handoff_r1 #(.CHILD_TIMEOUT_CYCLES(HANDOFF_TIMEOUT_CYCLES)) lease(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(lease_command_valid),
        .command_ready(lease_command_ready),
        .command_sequence(lease_command_sequence),
        .command_epoch(lease_command_epoch),
        .command_lease_base(lease_command_lease_base),
        .expected_epoch(expected_epoch),
        .command_layer(lease_command_layer),
        .command_position(lease_command_position),
        .command_coefficient_position(lease_command_coefficient_position),
        .command_addr(lease_command_addr),
        .command_base(lease_command_base),
        .command_bytes(lease_command_bytes),
        .cancel_valid(lease_cancel_valid),
        .cancel_ready(lease_cancel_ready),
        .busy(lease_busy),
        .cache_exclusive(lease_cache_exclusive),
        .cache_idle(cache_idle),
        .cache_valid(cache_valid),
        .cache_metadata_fault(cache_metadata_fault),
        .cache_base(cache_base),
        .cache_region_base(cache_region_base),
        .cache_region_bytes(cache_region_bytes),
        .cache_sequence(cache_sequence),
        .cache_epoch(cache_epoch),
        .cache_capacity(cache_capacity),
        .cache_committed_tokens(cache_query_committed_tokens),
        .source_busy(producer_busy),
        .source_completion_valid(producer_completion_valid),
        .source_completion_ready(lease_source_completion_ready),
        .source_sequence(producer_completion_sequence),
        .source_epoch(producer_completion_epoch),
        .source_status(producer_completion_status),
        .source_norm_status(producer_completion_norm_status),
        .source_qkv_status(producer_completion_qkv_status),
        .source_matrix_status(producer_completion_matrix_status),
        .source_hls_status(producer_completion_norm_hls_status),
        .source_norm_read_beats(producer_completion_norm_read_beats),
        .source_norm_records(producer_completion_norm_records),
        .source_phase(producer_completion_phase),
        .source_matrices(producer_completion_matrices),
        .source_total_bytes(producer_completion_total_bytes),
        .source_result_valid(producer_result_valid),
        .source_revoked(producer_completion_revoked),
        .source_result_address(producer_result_address),
        .source_result_bytes(producer_result_bytes),
        .child_command_valid(consumer_command_valid),
        .child_command_ready(consumer_command_ready),
        .child_busy(consumer_busy),
        .child_command_sequence(consumer_command_sequence),
        .child_command_epoch(consumer_command_epoch),
        .child_command_lease_base(consumer_command_lease_base),
        .child_command_layer(consumer_command_layer),
        .child_command_position(consumer_command_position),
        .child_command_coefficient_position(consumer_command_coefficient_position),
        .child_command_source_sequence(consumer_command_source_sequence),
        .child_command_source_epoch(consumer_command_source_epoch),
        .child_command_source_valid(consumer_command_source_valid),
        .child_command_source_position(consumer_command_source_position),
        .child_command_v_source_sequence(consumer_command_v_source_sequence),
        .child_command_v_source_epoch(consumer_command_v_source_epoch),
        .child_command_v_source_valid(consumer_command_v_source_valid),
        .child_command_v_source_position(consumer_command_v_source_position),
        .child_command_addr(consumer_command_addr),
        .child_command_base(consumer_command_base),
        .child_command_bytes(consumer_command_bytes),
        .child_cancel_valid(consumer_cancel_valid),
        .child_cancel_ready(consumer_cancel_ready),
        .child_completion_valid(consumer_completion_valid),
        .child_completion_ready(consumer_completion_ready),
        .child_sequence(consumer_completion_sequence),
        .child_epoch(consumer_completion_epoch),
        .child_layer(consumer_completion_layer),
        .child_position(consumer_completion_position),
        .child_status(consumer_completion_status),
        .child_append_status(consumer_completion_append_status),
        .child_root_status(consumer_completion_root_status),
        .child_capture_status(consumer_completion_capture_status),
        .child_rope_status(consumer_completion_rope_status),
        .child_rope_hls_status(consumer_completion_rope_hls_status),
        .child_rope_beats(consumer_completion_rope_beats),
        .child_capture_qk_words(consumer_completion_capture_qk_words),
        .child_capture_v_beats(consumer_completion_capture_v_beats),
        .child_kv_frames(consumer_completion_kv_frames),
        .child_q_frames(consumer_completion_q_frames),
        .child_kv_bytes(consumer_completion_kv_bytes),
        .child_kv_tokens(consumer_completion_kv_tokens),
        .child_kv_committed(consumer_completion_kv_committed),
        .child_result_valid(consumer_result_valid),
        .child_revoked(consumer_completion_revoked),
        .completion_valid(lease_completion_valid),
        .completion_ready(lease_completion_ready),
        .completion_sequence(lease_completion_sequence),
        .completion_epoch(lease_completion_epoch),
        .completion_layer(lease_completion_layer),
        .completion_position(lease_completion_position),
        .completion_status(lease_completion_status),
        .completion_source_status(lease_completion_source_status),
        .completion_child_status(lease_completion_child_status),
        .completion_kv_committed(lease_completion_kv_committed),
        .completion_kv_bytes(lease_completion_kv_bytes),
        .completion_kv_tokens(lease_completion_kv_tokens),
        .completion_revoked(lease_completion_revoked),
        .result_valid(lease_result_valid),
        .result_ce(lease_result_ce),
        .result_address(lease_result_address),
        .child_result_ce(consumer_result_ce),
        .child_result_address(consumer_result_address),
        .child_result_q(consumer_result_q),
        .child_result_q_valid(consumer_result_q_valid),
        .result_q(lease_result_q),
        .result_q_valid(lease_result_q_valid)
    );
    ntrpu_rope_attention_query_r1 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) consumer(
        .clk(clk),
        .rst_n(rst_n),
        .command_sequence(consumer_command_sequence),
        .command_epoch(consumer_command_epoch),
        .command_lease_base(consumer_command_lease_base),
        .expected_epoch(expected_epoch),
        .command_layer(consumer_command_layer),
        .command_position(consumer_command_position),
        .cancel_valid(consumer_cancel_valid),
        .cancel_ready(consumer_cancel_ready),
        .completion_valid(consumer_completion_valid),
        .completion_ready(consumer_completion_ready),
        .completion_status(consumer_completion_status),
        .completion_append_status(consumer_completion_append_status),
        .completion_root_status(consumer_completion_root_status),
        .completion_sequence(consumer_completion_sequence),
        .completion_epoch(consumer_completion_epoch),
        .completion_layer(consumer_completion_layer),
        .completion_position(consumer_completion_position),
        .completion_kv_frames(consumer_completion_kv_frames),
        .completion_q_frames(consumer_completion_q_frames),
        .completion_kv_bytes(consumer_completion_kv_bytes),
        .completion_kv_tokens(consumer_completion_kv_tokens),
        .completion_kv_committed(consumer_completion_kv_committed),
        .result_valid(consumer_result_valid),
        .completion_revoked(consumer_completion_revoked),
        .result_ce(consumer_result_ce),
        .result_address(consumer_result_address),
        .result_q(consumer_result_q),
        .result_q_valid(consumer_result_q_valid),
        .command_v_address(consumer_command_addr[2]),
        .command_v_base(consumer_command_base[2]),
        .command_v_bytes(consumer_command_bytes[2]),
        .command_v_source_valid(consumer_command_v_source_valid),
        .command_v_source_sequence(consumer_command_v_source_sequence),
        .command_v_source_epoch(consumer_command_v_source_epoch),
        .command_v_source_position(consumer_command_v_source_position),
        .v_axi_araddr(v_axi_araddr),
        .v_axi_arlen(v_axi_arlen),
        .v_axi_arsize(v_axi_arsize),
        .v_axi_arprot(v_axi_arprot),
        .v_axi_arburst(v_axi_arburst),
        .v_axi_arid(v_axi_arid),
        .v_axi_arcache(v_axi_arcache),
        .v_axi_arlock(v_axi_arlock),
        .v_axi_arvalid(v_axi_arvalid),
        .v_axi_arready(v_axi_arready),
        .v_axi_rdata(v_axi_rdata),
        .v_axi_rresp(v_axi_rresp),
        .v_axi_rid(v_axi_rid),
        .v_axi_rlast(v_axi_rlast),
        .v_axi_rvalid(v_axi_rvalid),
        .v_axi_rready(v_axi_rready),
        .completion_capture_status(consumer_completion_capture_status),
        .completion_capture_qk_words(consumer_completion_capture_qk_words),
        .completion_capture_v_beats(consumer_completion_capture_v_beats),
        .command_source_sequence(consumer_command_source_sequence),
        .command_source_epoch(consumer_command_source_epoch),
        .command_source_valid(consumer_command_source_valid),
        .command_source_position(consumer_command_source_position),
        .command_coefficient_position(consumer_command_coefficient_position),
        .command_q_address(consumer_command_addr[0]),
        .command_q_base(consumer_command_base[0]),
        .command_q_bytes(consumer_command_bytes[0]),
        .command_k_address(consumer_command_addr[1]),
        .command_k_base(consumer_command_base[1]),
        .command_k_bytes(consumer_command_bytes[1]),
        .command_coefficient_address(consumer_command_addr[3]),
        .command_coefficient_base(consumer_command_base[3]),
        .command_coefficient_bytes(consumer_command_bytes[3]),
        .rope_axi_araddr(rope_axi_araddr),
        .rope_axi_arlen(rope_axi_arlen),
        .rope_axi_arsize(rope_axi_arsize),
        .rope_axi_arprot(rope_axi_arprot),
        .rope_axi_arburst(rope_axi_arburst),
        .rope_axi_arid(rope_axi_arid),
        .rope_axi_arcache(rope_axi_arcache),
        .rope_axi_arlock(rope_axi_arlock),
        .rope_axi_arvalid(rope_axi_arvalid),
        .rope_axi_arready(rope_axi_arready),
        .rope_axi_rdata(rope_axi_rdata),
        .rope_axi_rresp(rope_axi_rresp),
        .rope_axi_rid(rope_axi_rid),
        .rope_axi_rlast(rope_axi_rlast),
        .rope_axi_rvalid(rope_axi_rvalid),
        .rope_axi_rready(rope_axi_rready),
        .completion_rope_status(consumer_completion_rope_status),
        .completion_rope_hls_status(consumer_completion_rope_hls_status),
        .completion_rope_beats(consumer_completion_rope_beats),
        .command_valid(consumer_command_valid),
        .command_ready(consumer_command_ready),
        .busy(consumer_busy),
        .cache_exclusive(consumer_cache_exclusive),
        .bind_valid(consumer_bind_valid),
        .bind_ready(consumer_bind_ready),
        .bind_cache_base(bind_cache_base),
        .bind_region_base(bind_region_base),
        .bind_region_bytes(bind_region_bytes),
        .bind_capacity(bind_capacity),
        .bind_sequence(bind_sequence),
        .bind_epoch(bind_epoch),
        .cache_valid(cache_valid),
        .cache_base(cache_base),
        .cache_region_base(cache_region_base),
        .cache_region_bytes(cache_region_bytes),
        .cache_sequence(cache_sequence),
        .cache_epoch(cache_epoch),
        .cache_capacity(cache_capacity),
        .kv_axi_awaddr(kv_axi_awaddr),
        .kv_axi_awlen(kv_axi_awlen),
        .kv_axi_awsize(kv_axi_awsize),
        .kv_axi_awprot(kv_axi_awprot),
        .kv_axi_awburst(kv_axi_awburst),
        .kv_axi_awid(kv_axi_awid),
        .kv_axi_awcache(kv_axi_awcache),
        .kv_axi_awlock(kv_axi_awlock),
        .kv_axi_awvalid(kv_axi_awvalid),
        .kv_axi_awready(kv_axi_awready),
        .kv_axi_wdata(kv_axi_wdata),
        .kv_axi_wstrb(kv_axi_wstrb),
        .kv_axi_wlast(kv_axi_wlast),
        .kv_axi_wvalid(kv_axi_wvalid),
        .kv_axi_wready(kv_axi_wready),
        .kv_axi_bid(kv_axi_bid),
        .kv_axi_bresp(kv_axi_bresp),
        .kv_axi_bvalid(kv_axi_bvalid),
        .kv_axi_bready(kv_axi_bready),
        .kv_axi_araddr(kv_axi_araddr),
        .kv_axi_arlen(kv_axi_arlen),
        .kv_axi_arsize(kv_axi_arsize),
        .kv_axi_arprot(kv_axi_arprot),
        .kv_axi_arburst(kv_axi_arburst),
        .kv_axi_arid(kv_axi_arid),
        .kv_axi_arcache(kv_axi_arcache),
        .kv_axi_arlock(kv_axi_arlock),
        .kv_axi_arvalid(kv_axi_arvalid),
        .kv_axi_arready(kv_axi_arready),
        .kv_axi_rdata(kv_axi_rdata),
        .kv_axi_rresp(kv_axi_rresp),
        .kv_axi_rid(kv_axi_rid),
        .kv_axi_rlast(kv_axi_rlast),
        .kv_axi_rvalid(kv_axi_rvalid),
        .kv_axi_rready(kv_axi_rready),
        .root_cache_lock(root_cache_lock),
        .cache_metadata_fault(cache_metadata_fault),
        .cache_query_layer(cache_query_layer),
        .cache_query_valid(cache_query_valid),
        .cache_query_committed_tokens(cache_query_committed_tokens),
        .cache_committed_tokens(cache_committed_tokens),
        .bind_completion_valid(bind_completion_valid),
        .bind_completion_ready(bind_completion_ready),
        .bind_completion_status(bind_completion_status),
        .bind_completion_sequence(bind_completion_sequence),
        .bind_completion_epoch(bind_completion_epoch)
    );
endmodule
`default_nettype wire
