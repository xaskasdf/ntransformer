// One owned KV allocation and one attention-query lease, all on one clock.
// Append may continue while a reader holds a snapshot of a published prefix.
// Binding is blocked until the lease response and all read jobs are drained.
// Release is the consumer's certificate that attention no longer uses this
// lease. It may abort a query between tiles. Reset must quiesce every peer.
`default_nettype none
module ntrpu_kv_cache_r1 #(
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
    input logic lease_cmd_valid,
    output logic lease_cmd_ready,
    input logic lease_cmd_release,
    input logic [31:0] lease_cmd_layer,lease_cmd_query_head,lease_cmd_position,
    input logic [63:0] lease_cmd_sequence,lease_cmd_epoch,lease_cmd_id,
    output logic lease_active,lease_done_valid,
    input logic lease_done_ready,
    output logic lease_done_release,
    output logic [3:0] lease_done_status,
    output logic [31:0] lease_done_prefix,
    output logic [63:0] lease_done_sequence,lease_done_epoch,lease_done_id,
    input logic read_valid,
    output logic read_ready,
    input logic [31:0] read_start,read_tokens,
    input logic [63:0] read_sequence,read_epoch,read_lease_id,
    input logic read_cancel_valid,
    output logic read_cancel_ready,
    output logic tile_valid,
    input logic tile_ready,
    output logic [511:0] tile_data,
    output logic tile_plane,tile_last,
    output logic [7:0] tile_index,
    output logic [63:0] tile_sequence,tile_epoch,tile_lease_id,
    output logic read_done_valid,
    input logic read_done_ready,
    output logic [3:0] read_done_status,
    output logic [9:0] read_done_beats,
    output logic [15:0] read_done_bytes,
    output logic [63:0] read_done_sequence,read_done_epoch,read_done_lease_id,
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
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2;
    logic writer_bind_valid,writer_bind_ready,writer_cmd_valid,writer_cmd_ready,maintenance_allowed;
    logic [31:0] writer_prefix;
    logic [63:0] held_sequence,held_epoch,held_id,last_lease_id;
    logic [31:0] held_layer,held_query_head,held_position,held_prefix,next_start,read_count;
    logic read_busy,read_rejected,loader_cmd_valid,loader_cmd_ready,loader_done_valid,loader_done_ready;
    logic [3:0] rejected_status,admission_status,loader_done_status;
    logic [9:0] loader_done_beats;
    logic [15:0] loader_done_bytes;
    logic [63:0] loader_done_sequence,loader_done_epoch;
    logic loader_cancel_valid,loader_cancel_ready;

    assign maintenance_allowed=!lease_active && !lease_done_valid && !read_busy;
    assign writer_bind_valid=bind_valid && maintenance_allowed;
    assign bind_ready=writer_bind_ready && maintenance_allowed;
    // Bind has priority, then acquire, then append; held leases permit append.
    assign writer_cmd_valid=append_valid && (lease_active || !lease_cmd_valid);
    assign append_ready=writer_cmd_ready && (lease_active || !lease_cmd_valid);
    assign lease_cmd_ready=!lease_done_valid && !read_busy &&
        (lease_active ? lease_cmd_release:(writer_cmd_ready && !bind_valid));
    assign read_ready=lease_active && !lease_done_valid && !lease_cmd_valid && !read_busy && loader_cmd_ready;
    always_comb begin
        admission_status=OK;
        if(read_sequence!=held_sequence || read_epoch!=held_epoch || read_lease_id!=held_id || read_epoch!=expected_epoch)
            admission_status=BAD_IDENTITY;
        else if(read_start!=next_start || read_tokens==0 || read_tokens>64 ||
            {1'b0,read_start}+{1'b0,read_tokens}>{1'b0,held_position}+33'd1) admission_status=BAD_COMMAND;
    end
    assign loader_cmd_valid=read_valid && read_ready && admission_status==OK;
    assign loader_cancel_valid=read_cancel_valid && read_busy && !read_rejected;
    assign read_cancel_ready=read_busy && !read_rejected && loader_cancel_ready;
    assign read_done_valid=read_busy && (read_rejected || loader_done_valid);
    assign loader_done_ready=read_busy && !read_rejected && read_done_ready;
    assign read_done_status=read_rejected ? rejected_status:loader_done_status;
    assign read_done_beats=read_rejected ? 10'b0:loader_done_beats;
    assign read_done_bytes=read_rejected ? 16'b0:loader_done_bytes;
    assign tile_lease_id=held_id;

    ntrpu_kv_append_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) writer (
        .clk,.rst_n,.maintenance_allowed,.bind_valid(writer_bind_valid),.bind_ready(writer_bind_ready),
        .bind_cache_base,.bind_region_base,.bind_region_bytes,.bind_capacity,.bind_sequence,.bind_epoch,.expected_epoch,
        .cache_valid,.cache_base,.cache_region_base,.cache_region_bytes,.cache_sequence,.cache_epoch,.cache_capacity,
        .inspect_layer(lease_cmd_layer[3:0]),.committed_tokens(writer_prefix),
        .cmd_valid(writer_cmd_valid),.cmd_ready(writer_cmd_ready),.prepared(append_prepared),
        .cmd_layer(append_layer),.cmd_position(append_position),.cmd_sequence(append_sequence),.cmd_epoch(append_epoch),
        .data_valid,.data_ready,.data_bits,.data_index,.data_last,.data_sequence,.data_epoch,
        .commit_valid,.commit_ready,.commit_status,.commit_sequence,.commit_epoch,
        .cancel_valid(append_cancel_valid),.cancel_ready(append_cancel_ready),
        .done_valid(append_done_valid),.done_ready(append_done_ready),.done_bind(append_done_bind),
        .done_status(append_done_status),.done_bytes(append_done_bytes),.done_committed_tokens(append_done_committed_tokens),
        .done_sequence(append_done_sequence),.done_epoch(append_done_epoch),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,.m_axi_awcache,.m_axi_awlock,
        .m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,.m_axi_wlast,.m_axi_wvalid,.m_axi_wready,
        .m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready
    );
    ntrpu_kv_tile_loader_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) reader (
        .clk,.rst_n,.cmd_valid(loader_cmd_valid),.cmd_ready(loader_cmd_ready),
        .cmd_cache_base(cache_base),.cmd_region_base(cache_region_base),.cmd_region_bytes(cache_region_bytes),
        .cmd_capacity(cache_capacity),.cmd_committed_tokens(held_prefix),.cmd_position(held_position),
        .cmd_tile_start(read_start),.cmd_tile_tokens(read_tokens),.cmd_layer(held_layer),
        .cmd_query_head(held_query_head),.cmd_kv_head(held_query_head>>2),
        .cmd_sequence(held_sequence),.cmd_epoch(held_epoch),.cmd_cache_sequence(cache_sequence),.cmd_cache_epoch(cache_epoch),
        .expected_epoch,.cancel_valid(loader_cancel_valid),.cancel_ready(loader_cancel_ready),
        .tile_valid,.tile_ready,.tile_data,.tile_plane,.tile_last,.tile_index,.tile_sequence,.tile_epoch,
        .done_valid(loader_done_valid),.done_ready(loader_done_ready),.done_status(loader_done_status),
        .done_beats(loader_done_beats),.done_bytes(loader_done_bytes),.done_sequence(loader_done_sequence),.done_epoch(loader_done_epoch),
        .m_axi_araddr,.m_axi_arlen,.m_axi_arsize,.m_axi_arprot,.m_axi_arburst,.m_axi_arid,.m_axi_arcache,.m_axi_arlock,
        .m_axi_arvalid,.m_axi_arready,.m_axi_rdata,.m_axi_rresp,.m_axi_rid,.m_axi_rlast,.m_axi_rvalid,.m_axi_rready
    );
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            lease_active<=0;lease_done_valid<=0;lease_done_release<=0;lease_done_status<=0;lease_done_prefix<=0;
            lease_done_sequence<=0;lease_done_epoch<=0;lease_done_id<=0;
            held_sequence<=0;held_epoch<=0;held_id<=0;last_lease_id<=0;
            held_layer<=0;held_query_head<=0;held_position<=0;held_prefix<=0;next_start<=0;read_count<=0;
            read_busy<=0;read_rejected<=0;rejected_status<=0;
            read_done_sequence<=0;read_done_epoch<=0;read_done_lease_id<=0;
        end else begin
            if(lease_done_valid && lease_done_ready) lease_done_valid<=0;
            if(lease_cmd_valid && lease_cmd_ready) begin
                lease_done_valid<=1;lease_done_release<=lease_cmd_release;lease_done_status<=OK;
                lease_done_sequence<=lease_cmd_sequence;lease_done_epoch<=lease_cmd_epoch;lease_done_id<=lease_cmd_id;
                lease_done_prefix<=lease_active ? held_prefix:0;
                if(lease_cmd_release) begin
                    if(!lease_active || lease_cmd_sequence!=held_sequence || lease_cmd_epoch!=held_epoch || lease_cmd_id!=held_id)
                        lease_done_status<=BAD_IDENTITY;
                    else lease_active<=0;
                end else if(!cache_valid || lease_cmd_sequence!=cache_sequence || lease_cmd_epoch!=cache_epoch ||
                    lease_cmd_epoch!=expected_epoch || lease_cmd_id==0 || lease_cmd_id<=last_lease_id)
                    lease_done_status<=BAD_IDENTITY;
                else if(lease_cmd_layer>=16 || lease_cmd_query_head>=32 || lease_cmd_position>=cache_capacity || lease_cmd_position>=writer_prefix)
                    lease_done_status<=BAD_COMMAND;
                else begin
                    lease_active<=1;last_lease_id<=lease_cmd_id;held_id<=lease_cmd_id;
                    held_sequence<=cache_sequence;held_epoch<=cache_epoch;held_layer<=lease_cmd_layer;
                    held_query_head<=lease_cmd_query_head;held_position<=lease_cmd_position;held_prefix<=writer_prefix;
                    lease_done_prefix<=writer_prefix;next_start<=0;
                end
            end
            if(read_valid && read_ready) begin
                read_busy<=1;read_rejected<=admission_status!=OK;rejected_status<=admission_status;read_count<=read_tokens;
                read_done_sequence<=read_sequence;read_done_epoch<=read_epoch;read_done_lease_id<=read_lease_id;
            end
            if(read_done_valid && read_done_ready) begin
                read_busy<=0;
                if(read_done_status==OK) next_start<=next_start+read_count;
            end
        end
    end
endmodule
`default_nettype wire
