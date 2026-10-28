// Own one query lease across all K/V tiles and consumer acknowledgements.
// Consumer state/results remain private until query_done_status==0 is retired.
// This path does not contain the attention arithmetic or its HLS port binding.
`default_nettype none
module ntrpu_attention_kv_path_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input logic clk,rst_n,
    input logic bind_valid,
    output logic bind_ready,
    input logic [63:0] bind_cache_base,bind_region_base,bind_region_bytes,
    input logic [31:0] bind_capacity,
    input logic [63:0] bind_sequence,bind_epoch,expected_epoch,
    output logic cache_valid,
    output logic [63:0] cache_base,cache_region_base,cache_region_bytes,cache_sequence,cache_epoch,
    output logic [31:0] cache_capacity,
    input logic append_valid,
    output logic append_ready,append_prepared,
    input logic [31:0] append_layer,append_position,
    input logic [63:0] append_sequence,append_epoch,
    input logic data_valid,
    output logic data_ready,
    input logic [511:0] data_bits,
    input logic [5:0] data_index,
    input logic data_last,
    input logic [63:0] data_sequence,data_epoch,
    input logic commit_valid,
    output logic commit_ready,
    input logic [3:0] commit_status,
    input logic [63:0] commit_sequence,commit_epoch,
    input logic append_cancel_valid,
    output logic append_cancel_ready,
    output logic append_done_valid,
    input logic append_done_ready,
    output logic append_done_bind,
    output logic [3:0] append_done_status,
    output logic [12:0] append_done_bytes,
    output logic [31:0] append_done_committed_tokens,
    output logic [63:0] append_done_sequence,append_done_epoch,
    input logic query_valid,
    output logic query_ready,lease_active,
    input logic [31:0] query_position,query_layer,query_head,
    input logic [63:0] query_sequence,query_epoch,query_lease,
    input logic query_cancel_valid,
    output logic query_cancel_ready,
    output logic query_done_valid,
    input logic query_done_ready,
    output logic [3:0] query_done_status,
    output logic [31:0] query_done_tokens,query_done_tiles,
    output logic [63:0] query_done_sequence,query_done_epoch,query_done_lease,
    output logic compute_valid,
    input logic compute_ready,
    output logic compute_abort,
    output logic [31:0] compute_start,compute_tokens,compute_position,compute_layer,compute_query_head,compute_kv_head,
    output logic [63:0] compute_sequence,compute_epoch,compute_lease,
    input logic keys_ce,values_ce,
    input logic [11:0] keys_address,values_address,
    output logic [31:0] keys_q,values_q,
    output logic keys_q_valid,values_q_valid,
    input logic finish_valid,
    output logic finish_ready,
    input logic [3:0] finish_status,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize,m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid,m_axi_awcache,
    output logic m_axi_awlock,m_axi_awvalid,
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,m_axi_arcache,
    output logic m_axi_arlock,m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast,m_axi_rvalid,
    output logic m_axi_rready
);
    logic lease_cmd_valid;
    logic lease_cmd_ready;
    logic lease_cmd_release;
    logic [31:0] lease_cmd_layer,lease_cmd_query_head,lease_cmd_position;
    logic [63:0] lease_cmd_sequence,lease_cmd_epoch,lease_cmd_id;
    logic lease_done_valid;
    logic lease_done_ready;
    logic lease_done_release;
    logic [3:0] lease_done_status;
    logic [31:0] lease_done_prefix;
    logic [63:0] lease_done_sequence,lease_done_epoch,lease_done_id;
    logic read_valid;
    logic read_ready;
    logic [31:0] read_start,read_tokens;
    logic [63:0] read_sequence,read_epoch,read_lease_id;
    logic read_cancel_valid;
    logic read_cancel_ready;
    logic tile_valid;
    logic tile_ready;
    logic [511:0] tile_data;
    logic tile_plane,tile_last;
    logic [7:0] tile_index;
    logic [63:0] tile_sequence,tile_epoch,tile_lease_id;
    logic read_done_valid;
    logic read_done_ready;
    logic [3:0] read_done_status;
    logic [9:0] read_done_beats;
    logic [15:0] read_done_bytes;
    logic [63:0] read_done_sequence,read_done_epoch,read_done_lease_id;
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,CANCELLED=7;
    typedef enum logic [3:0] {IDLE,ACQUIRE,ACQUIRE_DONE,RAM_COMMAND,READ_COMMAND,WAIT_TILE,
        RELEASE,RELEASE_DONE,RESPONSE} state_t;
    state_t state;
    logic [31:0] held_position,held_layer,held_head,next_start,tile_tokens;
    logic [3:0] fault_now;
    logic ram_cmd_valid,ram_cmd_ready,ram_prepared,ram_cancel_ready,ram_done_valid,ram_done_ready,ram_started;
    logic [3:0] ram_status;
    logic [9:0] ram_beats;
    logic [63:0] ram_sequence,ram_epoch,ram_lease;
    assign query_ready=state==IDLE && !lease_active && !lease_done_valid;
    assign query_done_valid=state==RESPONSE;
    assign query_cancel_ready=state!=IDLE && state!=RESPONSE && query_done_status==OK;
    always_comb begin
        fault_now=query_done_status;
        if(query_done_status==OK && state!=IDLE && state!=RESPONSE)begin
            if(query_done_epoch!=expected_epoch)fault_now=BAD_IDENTITY;
            else if(query_cancel_valid && query_cancel_ready)fault_now=CANCELLED;
        end
    end
    assign tile_tokens=held_position-next_start>=63 ? 32'd64:held_position-next_start+32'd1;
    assign lease_cmd_valid=(state==ACQUIRE && fault_now==OK) || state==RELEASE;
    assign lease_cmd_release=state==RELEASE;
    assign lease_cmd_layer=held_layer;
    assign lease_cmd_query_head=held_head;
    assign lease_cmd_position=held_position;
    assign lease_cmd_sequence=query_done_sequence;
    assign lease_cmd_epoch=query_done_epoch;
    assign lease_cmd_id=query_done_lease;
    assign lease_done_ready=state==ACQUIRE_DONE || state==RELEASE_DONE;
    assign ram_cmd_valid=state==RAM_COMMAND && fault_now==OK;
    assign read_valid=state==READ_COMMAND && ram_prepared;
    assign read_start=next_start;
    assign read_tokens=tile_tokens;
    assign read_sequence=query_done_sequence;
    assign read_epoch=query_done_epoch;
    assign read_lease_id=query_done_lease;
    assign ram_done_ready=state==WAIT_TILE;

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;held_position<=0;held_layer<=0;held_head<=0;next_start<=0;
            query_done_status<=OK;query_done_tokens<=0;query_done_tiles<=0;
            query_done_sequence<=0;query_done_epoch<=0;query_done_lease<=0;
        end else begin
            if(state!=IDLE && state!=RESPONSE)query_done_status<=fault_now;
            case(state)
                IDLE:if(query_valid && query_ready)begin
                    held_position<=query_position;held_layer<=query_layer;held_head<=query_head;next_start<=0;
                    query_done_sequence<=query_sequence;query_done_epoch<=query_epoch;query_done_lease<=query_lease;
                    query_done_tokens<=0;query_done_tiles<=0;query_done_status<=OK;
                    if(query_position>=131072 || query_layer>=16 || query_head>=32)begin
                        query_done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(query_sequence==0 || query_epoch==0 || query_lease==0 || query_epoch!=expected_epoch)begin
                        query_done_status<=BAD_IDENTITY;state<=RESPONSE;
                    end else state<=ACQUIRE;
                end
                ACQUIRE:begin
                    if(fault_now!=OK)state<=RESPONSE;
                    else if(lease_cmd_ready)state<=ACQUIRE_DONE;
                end
                ACQUIRE_DONE:if(lease_done_valid)begin
                    if(lease_done_status!=OK)begin
                        if(fault_now==OK)query_done_status<=lease_done_status;
                        state<=RESPONSE;
                    end else if(fault_now!=OK)state<=RELEASE;
                    else state<=RAM_COMMAND;
                end
                RAM_COMMAND:begin
                    if(fault_now!=OK)state<=RELEASE;
                    else if(ram_cmd_ready)state<=READ_COMMAND;
                end
                // Even after cancellation the admitted RAM load needs an upstream
                // completion certificate. Issue/drain this read before releasing.
                READ_COMMAND:if(read_valid && read_ready)state<=WAIT_TILE;
                WAIT_TILE:if(ram_done_valid)begin
                    if(fault_now!=OK || ram_status!=OK)begin
                        if(fault_now==OK)query_done_status<=ram_status;
                        state<=RELEASE;
                    end else begin
                        query_done_tokens<=next_start+tile_tokens;
                        query_done_tiles<=query_done_tiles+32'd1;
                        next_start<=next_start+tile_tokens;
                        if(next_start+tile_tokens==held_position+32'd1)state<=RELEASE;
                        else state<=RAM_COMMAND;
                    end
                end
                RELEASE:if(lease_cmd_ready)state<=RELEASE_DONE;
                RELEASE_DONE:if(lease_done_valid)begin
                    if(fault_now==OK)query_done_status<=lease_done_status;
                    state<=RESPONSE;
                end
                RESPONSE:if(query_done_ready)state<=IDLE;
                default:state<=IDLE;
            endcase
        end
    end
    ntrpu_kv_cache_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) cache (
        .clk,
        .rst_n,
        .bind_valid,
        .bind_ready,
        .bind_cache_base,
        .bind_region_base,
        .bind_region_bytes,
        .bind_capacity,
        .bind_sequence,
        .bind_epoch,
        .expected_epoch,
        .cache_valid,
        .cache_base,
        .cache_region_base,
        .cache_region_bytes,
        .cache_sequence,
        .cache_epoch,
        .cache_capacity,
        .append_valid,
        .append_ready,
        .append_prepared,
        .append_layer,
        .append_position,
        .append_sequence,
        .append_epoch,
        .data_valid,
        .data_ready,
        .data_bits,
        .data_index,
        .data_last,
        .data_sequence,
        .data_epoch,
        .commit_valid,
        .commit_ready,
        .commit_status,
        .commit_sequence,
        .commit_epoch,
        .append_cancel_valid,
        .append_cancel_ready,
        .append_done_valid,
        .append_done_ready,
        .append_done_bind,
        .append_done_status,
        .append_done_bytes,
        .append_done_committed_tokens,
        .append_done_sequence,
        .append_done_epoch,
        .lease_cmd_valid,
        .lease_cmd_ready,
        .lease_cmd_release,
        .lease_cmd_layer,
        .lease_cmd_query_head,
        .lease_cmd_position,
        .lease_cmd_sequence,
        .lease_cmd_epoch,
        .lease_cmd_id,
        .lease_active,
        .lease_done_valid,
        .lease_done_ready,
        .lease_done_release,
        .lease_done_status,
        .lease_done_prefix,
        .lease_done_sequence,
        .lease_done_epoch,
        .lease_done_id,
        .read_valid,
        .read_ready,
        .read_start,
        .read_tokens,
        .read_sequence,
        .read_epoch,
        .read_lease_id,
        .read_cancel_valid,
        .read_cancel_ready,
        .tile_valid,
        .tile_ready,
        .tile_data,
        .tile_plane,
        .tile_last,
        .tile_index,
        .tile_sequence,
        .tile_epoch,
        .tile_lease_id,
        .read_done_valid,
        .read_done_ready,
        .read_done_status,
        .read_done_beats,
        .read_done_bytes,
        .read_done_sequence,
        .read_done_epoch,
        .read_done_lease_id,
        .m_axi_awaddr,
        .m_axi_awlen,
        .m_axi_awsize,
        .m_axi_awprot,
        .m_axi_awburst,
        .m_axi_awid,
        .m_axi_awcache,
        .m_axi_awlock,
        .m_axi_awvalid,
        .m_axi_awready,
        .m_axi_wdata,
        .m_axi_wstrb,
        .m_axi_wlast,
        .m_axi_wvalid,
        .m_axi_wready,
        .m_axi_bid,
        .m_axi_bresp,
        .m_axi_bvalid,
        .m_axi_bready,
        .m_axi_araddr,
        .m_axi_arlen,
        .m_axi_arsize,
        .m_axi_arprot,
        .m_axi_arburst,
        .m_axi_arid,
        .m_axi_arcache,
        .m_axi_arlock,
        .m_axi_arvalid,
        .m_axi_arready,
        .m_axi_rdata,
        .m_axi_rresp,
        .m_axi_rid,
        .m_axi_rlast,
        .m_axi_rvalid,
        .m_axi_rready
    );
    ntrpu_attention_kv_ram_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) ram (
        .clk,.rst_n,.cmd_valid(ram_cmd_valid),.cmd_ready(ram_cmd_ready),.load_prepared(ram_prepared),
        .cmd_start(next_start),.cmd_tokens(tile_tokens),.cmd_position(held_position),.cmd_layer(held_layer),.cmd_query_head(held_head),
        .cmd_sequence(query_done_sequence),.cmd_epoch(query_done_epoch),.cmd_lease(query_done_lease),.expected_epoch,
        .tile_valid,.tile_ready,.tile_data,.tile_plane,.tile_last,.tile_index,.tile_sequence,.tile_epoch,.tile_lease(tile_lease_id),
        .load_done_valid(read_done_valid),.load_done_ready(read_done_ready),.load_done_status(read_done_status),
        .load_done_beats(read_done_beats),.load_done_bytes(read_done_bytes),
        .load_done_sequence(read_done_sequence),.load_done_epoch(read_done_epoch),.load_done_lease(read_done_lease_id),
        .cancel_valid(fault_now!=OK),.cancel_ready(ram_cancel_ready),
        .upstream_cancel_valid(read_cancel_valid),.upstream_cancel_ready(read_cancel_ready),
        .compute_valid,.compute_ready,.compute_abort,.compute_start,.compute_tokens,.compute_position,.compute_layer,
        .compute_query_head,.compute_kv_head,.compute_sequence,.compute_epoch,.compute_lease,
        .keys_ce,.values_ce,.keys_address,.values_address,.keys_q,.values_q,.keys_q_valid,.values_q_valid,
        .finish_valid,.finish_ready,.finish_status,
        .done_valid(ram_done_valid),.done_ready(ram_done_ready),.done_status(ram_status),.done_beats(ram_beats),
        .done_consumer_started(ram_started),.done_sequence(ram_sequence),.done_epoch(ram_epoch),.done_lease(ram_lease)
    );
endmodule
`default_nettype wire
