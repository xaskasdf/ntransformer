// FP32 KV layout: [16 layers][2 planes K,V][8 KV heads][capacity][64].
// Reads one head's K and V tiles into a PRIVATE consumer cache. Only an OK
// completion permits attention. Committed-prefix/lease inputs must come from
// a trusted publisher; this block does not create or authenticate KV leases.
// Cancellation/watchdog drain issued AXI and held output; coordinated reset
// must quiesce the AXI peer and invalidate the consumer cache.
`default_nettype none
module ntrpu_kv_tile_loader_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire logic clk,rst_n,
    input wire logic cmd_valid,
    output logic cmd_ready,
    input wire logic [63:0] cmd_cache_base,cmd_region_base,cmd_region_bytes,
    input wire logic [31:0] cmd_capacity,cmd_committed_tokens,cmd_position,cmd_tile_start,
    input wire logic [31:0] cmd_tile_tokens,cmd_layer,cmd_query_head,cmd_kv_head,
    input wire logic [63:0] cmd_sequence,cmd_epoch,cmd_cache_sequence,cmd_cache_epoch,expected_epoch,
    input wire logic cancel_valid,
    output logic cancel_ready,
    output logic tile_valid,
    input wire logic tile_ready,
    output logic [511:0] tile_data,
    output logic tile_plane,tile_last,
    output logic [7:0] tile_index,
    output logic [63:0] tile_sequence,tile_epoch,
    output logic done_valid,
    input wire logic done_ready,
    output logic [3:0] done_status,
    output logic [9:0] done_beats,
    output logic [15:0] done_bytes,
    output logic [63:0] done_sequence,done_epoch,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,m_axi_arcache,
    output logic m_axi_arlock,m_axi_arvalid,
    input wire logic m_axi_arready,
    input wire logic [511:0] m_axi_rdata,
    input wire logic [1:0] m_axi_rresp,
    input wire logic [3:0] m_axi_rid,
    input wire logic m_axi_rlast,m_axi_rvalid,
    output logic m_axi_rready
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,READ_ERROR=3,
        BAD_FORMAT=4,CANCELLED=7,TIMEOUT=8;
    typedef enum logic [3:0] {IDLE,OFFSETS,BOUNDS,CHECK,ISSUE,READ,DRAIN,RESPONSE} state_t;
    state_t state;
    logic [63:0] cache_base,region_base,region_bytes,address;
    logic [64:0] region_end,cache_end,k_address,v_address,k_end,v_end;
    logic [17:0] capacity,start;
    logic [6:0] tokens;
    logic [3:0] layer;
    logic [2:0] kv_head;
    logic [25:0] k_slot,v_slot;
    logic [8:0] total_beats,received;
    logic [6:0] chunk_beats,chunk_received;
    logic plane;
    logic [31:0] chunk_bytes,idle_cycles;
    logic reader_cmd_valid,reader_done_ready,reader_cmd_ready,reader_done,reader_valid,reader_ready,reader_last;
    logic [1:0] reader_status;
    logic [31:0] reader_bytes;
    logic [511:0] reader_data;
    logic bad_beat,progress_event,watchdog_expired;
    logic [3:0] error_now;

    function automatic logic [6:0] next_chunk(input logic [63:0] addr,input logic [8:0] beats);
        logic [6:0] page;
        page=7'd64-{1'b0,addr[11:6]};
        return beats<{2'b0,page} ? beats[6:0]:page;
    endfunction
    assign cmd_ready=state==IDLE;
    assign cancel_ready=state!=IDLE && state!=RESPONSE && done_status==OK;
    assign done_valid=state==RESPONSE;
    assign done_bytes={done_beats,6'b0};
    assign chunk_bytes={19'b0,chunk_beats,6'b0};
    assign reader_cmd_valid=state==ISSUE && error_now==OK;
    assign reader_done_ready=state==READ;
    assign reader_ready=state==READ && (done_status!=OK || !tile_valid || tile_ready);
    assign progress_event=(m_axi_arvalid && m_axi_arready) || (m_axi_rvalid && m_axi_rready) ||
        (reader_valid && reader_ready) || (tile_valid && tile_ready) ||
        (state==ISSUE && reader_cmd_ready) || (state==READ && reader_done);
    assign watchdog_expired=(state==ISSUE || state==READ || state==DRAIN) &&
        idle_cycles>=32'(WATCHDOG_CYCLES-1) && !progress_event;
    always_comb begin
        bad_beat=received>=total_beats || reader_last!=(chunk_received+7'd1==chunk_beats);
        for(int lane=0;lane<16;lane++) bad_beat |= reader_data[lane*32+23+:8]==8'hff;
        error_now=done_status;
        if(done_status==OK) begin
            if(cancel_valid && cancel_ready) error_now=CANCELLED;
            else if(watchdog_expired) error_now=TIMEOUT;
            else if(state==READ && reader_done &&
                (reader_status!=0 || reader_bytes!=chunk_bytes || chunk_received!=chunk_beats)) error_now=READ_ERROR;
            else if(state==READ && reader_valid && reader_ready && bad_beat) error_now=BAD_FORMAT;
        end
    end
    ntrpu_axi_read_master reader (
        .clk,.rst_n,.cmd_valid(reader_cmd_valid),.cmd_ready(reader_cmd_ready),
        .cmd_addr(address),.cmd_bytes(chunk_bytes),.done_valid(reader_done),.done_ready(reader_done_ready),
        .done_status(reader_status),.done_bytes(reader_bytes),.out_data(reader_data),
        .out_valid(reader_valid),.out_ready(reader_ready),.out_last(reader_last),
        .m_axi_araddr,.m_axi_arlen,.m_axi_arsize,.m_axi_arprot,.m_axi_arburst,.m_axi_arid,
        .m_axi_arcache,.m_axi_arlock,.m_axi_arvalid,.m_axi_arready,.m_axi_rdata,.m_axi_rresp,
        .m_axi_rid,.m_axi_rlast,.m_axi_rvalid,.m_axi_rready
    );
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;cache_base<=0;region_base<=0;region_bytes<=0;address<=0;
            region_end<=0;cache_end<=0;k_address<=0;v_address<=0;k_end<=0;v_end<=0;
            capacity<=0;start<=0;tokens<=0;layer<=0;kv_head<=0;k_slot<=0;v_slot<=0;
            total_beats<=0;received<=0;chunk_beats<=0;chunk_received<=0;plane<=0;idle_cycles<=0;
            tile_valid<=0;tile_data<=0;tile_plane<=0;tile_last<=0;tile_index<=0;tile_sequence<=0;tile_epoch<=0;
            done_status<=OK;done_beats<=0;done_sequence<=0;done_epoch<=0;
        end else begin
            if(tile_valid && tile_ready) begin tile_valid<=0;done_beats<=done_beats+10'd1;end
            if(state!=IDLE && state!=RESPONSE) done_status<=error_now;
            if(state==ISSUE || state==READ || state==DRAIN) begin
                if(progress_event) idle_cycles<=0;
                else if(idle_cycles<32'(WATCHDOG_CYCLES)) idle_cycles<=idle_cycles+32'd1;
            end else idle_cycles<=0;
            case(state)
                IDLE: if(cmd_valid) begin
                    cache_base<=cmd_cache_base;region_base<=cmd_region_base;region_bytes<=cmd_region_bytes;
                    capacity<=cmd_capacity[17:0];start<=cmd_tile_start[17:0];tokens<=cmd_tile_tokens[6:0];
                    layer<=cmd_layer[3:0];kv_head<=cmd_kv_head[2:0];total_beats<={cmd_tile_tokens[6:0],2'b0};
                    received<=0;plane<=0;done_beats<=0;done_status<=OK;
                    done_sequence<=cmd_sequence;done_epoch<=cmd_epoch;tile_sequence<=cmd_sequence;tile_epoch<=cmd_epoch;
                    if(cmd_cache_base[5:0]!=0 || cmd_region_bytes==0 || cmd_capacity==0 || cmd_capacity>131072 ||
                        cmd_tile_tokens==0 || cmd_tile_tokens>64 || cmd_position>=cmd_capacity ||
                        cmd_committed_tokens>cmd_capacity || cmd_committed_tokens<=cmd_position ||
                        {1'b0,cmd_tile_start}+{1'b0,cmd_tile_tokens}>{1'b0,cmd_position}+33'd1 ||
                        cmd_layer>=16 || cmd_query_head>=32 || cmd_kv_head!=cmd_query_head/4) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(cmd_sequence==0 || cmd_epoch==0 || cmd_sequence!=cmd_cache_sequence ||
                        cmd_epoch!=cmd_cache_epoch || cmd_epoch!=expected_epoch) begin done_status<=BAD_IDENTITY;state<=RESPONSE;end
                    else state<=OFFSETS;
                end
                OFFSETS: begin
                    k_slot<=26'({layer,1'b0,kv_head})*26'(capacity)+26'(start);
                    v_slot<=26'({layer,1'b1,kv_head})*26'(capacity)+26'(start);
                    region_end<={1'b0,region_base}+{1'b0,region_bytes};
                    cache_end<={1'b0,cache_base}+{31'b0,capacity,16'b0};
                    if(error_now!=OK) state<=RESPONSE;else state<=BOUNDS;
                end
                BOUNDS: begin
                    k_address<={1'b0,cache_base}+{31'b0,k_slot,8'b0};
                    v_address<={1'b0,cache_base}+{31'b0,v_slot,8'b0};
                    k_end<={1'b0,cache_base}+({39'b0,k_slot}+{58'b0,tokens})*65'd256;
                    v_end<={1'b0,cache_base}+({39'b0,v_slot}+{58'b0,tokens})*65'd256;
                    if(error_now!=OK) state<=RESPONSE;else state<=CHECK;
                end
                CHECK: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(region_end[64] || cache_end[64] || k_address[64] || v_address[64] || k_end[64] || v_end[64] ||
                        cache_base<region_base || cache_end>region_end || k_end>cache_end || v_end>cache_end ||
                        k_address< {1'b0,cache_base} || v_address< {1'b0,cache_base}) begin done_status<=BAD_COMMAND;state<=RESPONSE;end
                    else begin address<=k_address[63:0];chunk_beats<=next_chunk(k_address[63:0],total_beats);state<=ISSUE;end
                end
                ISSUE: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(reader_cmd_ready) begin chunk_received<=0;state<=READ;end
                end
                READ: begin
                    if(reader_valid && reader_ready) begin
                        received<=received+9'd1;chunk_received<=chunk_received+7'd1;
                        if(error_now==OK) begin
                            tile_valid<=1;tile_data<=reader_data;tile_plane<=plane;tile_index<=received[7:0];
                            tile_last<=plane && received+9'd1==total_beats;
                        end
                    end
                    if(reader_done) state<=DRAIN;
                end
                DRAIN: if(!tile_valid) begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(received==total_beats) begin
                        if(plane) begin
                            if(done_beats!={total_beats,1'b0}) done_status<=BAD_FORMAT;
                            state<=RESPONSE;
                        end else begin
                            plane<=1;received<=0;address<=v_address[63:0];
                            chunk_beats<=next_chunk(v_address[63:0],total_beats);state<=ISSUE;
                        end
                    end else begin
                        address<=address+{32'b0,chunk_bytes};
                        chunk_beats<=next_chunk(address+{32'b0,chunk_bytes},total_beats-received);state<=ISSUE;
                    end
                end
                RESPONSE: if(done_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
