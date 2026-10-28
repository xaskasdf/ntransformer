// Private Q/state/output ownership for the attention tile ap_memory contract.
// The generated HLS core is an external peer; this module contains no arithmetic.
`default_nettype none
module ntrpu_attention_context_r2 (
    input logic clk, rst_n,
    input logic command_valid,
    output logic command_ready,
    input logic [2047:0] command_query,
    input logic [31:0] command_position, command_layer, command_head,
    input logic [63:0] command_sequence, command_epoch, command_lease, expected_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic query_valid,
    input logic query_ready,
    output logic [31:0] query_position, query_layer, query_head,
    output logic [63:0] query_sequence, query_epoch, query_lease,
    output logic query_cancel_valid,
    input logic compute_valid,
    output logic compute_ready,
    input logic compute_abort,
    input logic [31:0] compute_start, compute_tokens, compute_position, compute_layer,
        compute_query_head, compute_kv_head,
    input logic [63:0] compute_sequence, compute_epoch, compute_lease,
    output logic finish_valid,
    input logic finish_ready,
    output logic [3:0] finish_status,
    input logic query_done_valid,
    output logic query_done_ready,
    input logic [3:0] query_done_status,
    input logic [31:0] query_done_tokens, query_done_tiles,
    input logic [63:0] query_done_sequence, query_done_epoch, query_done_lease,
    output logic completion_valid,
    input logic completion_ready,
    output logic [3:0] completion_status,
    // Live authorization, unlike the immutable completion record. Sample this
    // with every result read/commit; any observed revocation remains sticky.
    output logic result_valid,
    input logic result_ce,
    input logic [5:0] result_address,
    output logic [31:0] result_q,
    output logic result_q_valid,
    output logic hls_start,
    input logic hls_ready, hls_done,
    input logic [31:0] hls_return,
    output logic [31:0] hls_tile_start, hls_tile_count, hls_query_position,
        hls_layer, hls_query_head, hls_kv_head,
    output logic [63:0] hls_sequence, hls_epoch,
    input logic hls_query_ce,
    input logic [5:0] hls_query_address,
    output logic [31:0] hls_query_q,
    input logic hls_state_in_ce,
    input logic [6:0] hls_state_in_address,
    output logic [63:0] hls_state_in_q,
    input logic hls_state_out_ce, hls_state_out_we,
    input logic [6:0] hls_state_out_address,
    input logic [63:0] hls_state_out_d,
    input logic hls_output_ce, hls_output_we,
    input logic [5:0] hls_output_address,
    input logic [31:0] hls_output_d
);
    localparam logic [3:0] OK=0, BAD_COMMAND=1, BAD_IDENTITY=2,
        BAD_MEMORY=4, CANCELLED=7, HLS_ERROR=10, BAD_CERTIFICATE=11;
    typedef enum logic [1:0] {IDLE, ISSUE, ACTIVE, RESPONSE} owner_t;
    typedef enum logic [1:0] {T_IDLE, T_RUN, T_FINISH} tile_t;
    owner_t owner;
    tile_t tile;
    logic [31:0] query_mem[0:63], output_mem[0:63];
    logic [63:0] state_a[0:103], state_b[0:103];
    logic state_bank, ready_seen, done_seen, final_staged, epoch_revoked;
    logic [103:0] state_seen, state_seen_after;
    logic [63:0] output_seen, output_seen_after;
    logic [3:0] fault, fault_now, tile_result;
    logic [31:0] next_start, completed_tiles;
    logic [31:0] result_data;
    logic result_pending, final_tile, state_write, output_write, memory_bad;
    logic command_finite, compute_identity_ok, compute_shape_ok;
    integer lane;

    assign command_ready=owner==IDLE;
    assign cancel_ready=owner==ISSUE || owner==ACTIVE;
    assign query_valid=owner==ISSUE && fault_now==OK;
    // Registered fault breaks the path cancel -> compute_abort feedback loop.
    // Port checks can change combinationally after a RAM write clock edge.
    assign query_cancel_valid=owner==ACTIVE && fault!=OK;
    assign compute_ready=owner==ACTIVE && tile==T_IDLE && fault_now==OK && !query_done_valid;
    assign finish_valid=owner==ACTIVE && tile==T_FINISH;
    assign finish_status=fault_now!=OK ? fault_now:tile_result;
    assign query_done_ready=owner==ACTIVE && tile==T_IDLE;
    assign completion_valid=owner==RESPONSE;
    assign result_valid=completion_valid && completion_status==OK &&
        !epoch_revoked && query_epoch==expected_epoch;
    assign result_q_valid=result_pending && result_valid;
    assign result_q=result_q_valid ? result_data:32'd0;
    assign hls_start=owner==ACTIVE && tile==T_RUN && !ready_seen;
    assign hls_query_position=query_position;
    assign hls_layer=query_layer;
    assign hls_query_head=query_head;
    assign hls_kv_head=query_head>>2;
    assign hls_sequence=query_sequence;
    assign hls_epoch=query_epoch;
    assign final_tile=hls_tile_start+hls_tile_count==query_position+32'd1;
    assign state_write=hls_state_out_ce && hls_state_out_we;
    assign output_write=hls_output_ce && hls_output_we;
    assign compute_identity_ok=compute_sequence==query_sequence && compute_epoch==query_epoch &&
        compute_lease==query_lease && compute_position==query_position && compute_layer==query_layer &&
        compute_query_head==query_head && compute_kv_head==(query_head>>2);
    assign compute_shape_ok=compute_start==next_start && compute_tokens!=0 && compute_tokens<=64 &&
        compute_start<=query_position && compute_tokens<=query_position-compute_start+32'd1 &&
        (compute_tokens==64 || compute_start+compute_tokens==query_position+32'd1);

    always_comb begin
        command_finite=1'b1;
        for(integer i=0;i<64;i=i+1)
            if(command_query[i*32+23+:8]==8'hff)command_finite=1'b0;
        state_seen_after=state_seen;
        output_seen_after=output_seen;
        memory_bad=1'b0;
        if(owner==ACTIVE && tile==T_RUN && !done_seen)begin
            if(hls_state_in_ce && (hls_state_in_address>=104 || hls_tile_start==0))memory_bad=1'b1;
            if(state_write)begin
                if(hls_state_out_address>=104)memory_bad=1'b1;
                else begin
                    if(state_seen[hls_state_out_address])memory_bad=1'b1;
                    state_seen_after[hls_state_out_address]=1'b1;
                end
            end
            if(output_write)begin
                if(!final_tile || output_seen[hls_output_address] || hls_output_d[30:23]==8'hff)
                    memory_bad=1'b1;
                output_seen_after[hls_output_address]=1'b1;
            end
        end else if(owner==ACTIVE && (state_write || output_write || hls_query_ce || hls_state_in_ce))
            memory_bad=1'b1;
        fault_now=fault;
        if(fault==OK && (owner==ISSUE || owner==ACTIVE))begin
            if(query_epoch!=expected_epoch || epoch_revoked)fault_now=BAD_IDENTITY;
            else if(cancel_valid || (tile!=T_IDLE && compute_abort))fault_now=CANCELLED;
            else if(memory_bad)fault_now=BAD_MEMORY;
        end
    end

    // Synchronous, one-cycle ap_memory reads. Raw RAM contents survive reset;
    // ownership/validity prevent their reuse. State input and output never alias.
    always_ff @(posedge clk)begin
        if(rst_n && owner==ACTIVE && tile==T_RUN && !done_seen)begin
            if(hls_query_ce)hls_query_q<=query_mem[hls_query_address];
            if(hls_state_in_ce && hls_state_in_address<104 && hls_tile_start!=0)
                hls_state_in_q<=state_bank ? state_b[hls_state_in_address]:state_a[hls_state_in_address];
            if(state_write && hls_state_out_address<104 && fault_now==OK)begin
                if(state_bank)state_a[hls_state_out_address]<=hls_state_out_d;
                else state_b[hls_state_out_address]<=hls_state_out_d;
            end
            if(output_write && final_tile && fault_now==OK)output_mem[hls_output_address]<=hls_output_d;
        end
        if(rst_n && result_ce && result_valid)result_data<=output_mem[result_address];
    end

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            owner<=IDLE;tile<=T_IDLE;fault<=OK;tile_result<=OK;completion_status<=OK;
            query_position<=0;query_layer<=0;query_head<=0;query_sequence<=0;query_epoch<=0;query_lease<=0;
            hls_tile_start<=0;hls_tile_count<=0;next_start<=0;completed_tiles<=0;
            state_bank<=0;ready_seen<=0;done_seen<=0;final_staged<=0;epoch_revoked<=0;
            state_seen<=0;output_seen<=0;result_pending<=0;
        end else begin
            result_pending<=result_ce && result_valid;
            if(owner!=IDLE && query_epoch!=expected_epoch)epoch_revoked<=1'b1;
            if(owner==ISSUE || owner==ACTIVE)fault<=fault_now;
            case(owner)
                IDLE:if(command_valid)begin
                    query_position<=command_position;query_layer<=command_layer;query_head<=command_head;
                    query_sequence<=command_sequence;query_epoch<=command_epoch;query_lease<=command_lease;
                    for(lane=0;lane<64;lane=lane+1)query_mem[lane]<=command_query[lane*32+:32];
                    next_start<=0;completed_tiles<=0;state_bank<=0;tile<=T_IDLE;fault<=OK;
                    ready_seen<=0;done_seen<=0;state_seen<=0;output_seen<=0;
                    final_staged<=0;epoch_revoked<=0;completion_status<=OK;owner<=ISSUE;
                    if(command_position>=131072 || command_layer>=16 || command_head>=32)begin
                        completion_status<=BAD_COMMAND;owner<=RESPONSE;
                    end else if(command_sequence==0 || command_epoch==0 || command_lease==0 || command_epoch!=expected_epoch)begin
                        completion_status<=BAD_IDENTITY;owner<=RESPONSE;
                    end else if(!command_finite)begin
                        completion_status<=BAD_MEMORY;owner<=RESPONSE;
                    end
                end
                ISSUE:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;owner<=RESPONSE;end
                    else if(query_ready)owner<=ACTIVE;
                end
                ACTIVE:begin
                    case(tile)
                        T_IDLE:if(compute_valid && compute_ready)begin
                            hls_tile_start<=compute_start;hls_tile_count<=compute_tokens;
                            state_seen<=0;output_seen<=0;ready_seen<=0;done_seen<=0;tile_result<=OK;
                            if(!compute_identity_ok)begin
                                fault<=BAD_IDENTITY;tile_result<=BAD_IDENTITY;tile<=T_FINISH;
                            end else if(!compute_shape_ok)begin
                                fault<=BAD_COMMAND;tile_result<=BAD_COMMAND;tile<=T_FINISH;
                            end else tile<=T_RUN;
                        end
                        T_RUN:begin
                            state_seen<=state_seen_after;output_seen<=output_seen_after;
                            if(hls_ready)ready_seen<=1'b1;
                            if(hls_done && !done_seen)begin
                                done_seen<=1'b1;
                                if(fault_now!=OK)tile_result<=fault_now;
                                else if(hls_return!=0)begin tile_result<=HLS_ERROR;fault<=HLS_ERROR;end
                                else if(!(&state_seen_after) || (final_tile ? !(&output_seen_after):(|output_seen_after)))begin
                                    tile_result<=BAD_MEMORY;fault<=BAD_MEMORY;
                                end
                            end
                            // A cancellation cannot truncate the peer's transaction.
                            // Hold start to ready, and wait for done even if ready was early.
                            if((ready_seen || hls_ready) && (done_seen || hls_done))tile<=T_FINISH;
                        end
                        T_FINISH:if(finish_ready)begin
                            if(finish_status==OK)begin
                                next_start<=next_start+hls_tile_count;completed_tiles<=completed_tiles+32'd1;
                                state_bank<=!state_bank;
                                if(final_tile)final_staged<=1'b1;
                            end
                            tile<=T_IDLE;
                        end
                        default:tile<=T_IDLE;
                    endcase
                    if(query_done_valid && query_done_ready)begin
                        completion_status<=fault_now!=OK ? fault_now:query_done_status;
                        if(fault_now==OK && query_done_status==OK &&
                            (query_done_sequence!=query_sequence || query_done_epoch!=query_epoch || query_done_lease!=query_lease ||
                             query_done_tokens!=query_position+32'd1 || query_done_tiles!=completed_tiles ||
                             next_start!=query_position+32'd1 || !final_staged))completion_status<=BAD_CERTIFICATE;
                        owner<=RESPONSE;
                    end
                end
                RESPONSE:if(completion_ready)owner<=IDLE;
                default:owner<=IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
