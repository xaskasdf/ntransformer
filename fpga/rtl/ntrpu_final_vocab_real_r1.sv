// Actual RMSNorm/Q8 -> shared AXI ABI1 -> original vocabulary matrix engine.
// The caller retains all five allocations through public ACK and drain.
// Shared reset must quiesce both children and their external AXI peer.
`default_nettype none
module ntrpu_final_vocab_real_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer NORM_HLS_WATCHDOG_CYCLES=262144,
    parameter integer MATRIX_TIMEOUT_CYCLES=16777216,
    parameter integer HANDOFF_TIMEOUT_CYCLES=268435456,
    parameter logic [63:0] INITIAL_NORM_TAG=64'd0,
    parameter logic [63:0] INITIAL_VOCAB_TAG=64'd0
) (
input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire [63:0] command_source_sequence,command_source_epoch,
    input wire command_source_valid,
    input wire [63:0] command_addr [0:4],
    input wire [63:0] command_base [0:4],
    input wire [63:0] command_bytes [0:4],
    input wire cancel_valid,
    output wire cancel_ready,busy,
    output wire completion_valid,
    input wire completion_ready,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [3:0] completion_status,completion_norm_status,completion_vocab_status,completion_matrix_status,
    output logic [31:0] completion_norm_hls_status,
    output logic [9:0] completion_norm_read_beats,
    output logic [6:0] completion_norm_records,
    output logic [3:0] completion_phase,
    output logic [4:0] completion_matrices,
    output logic [18:0] completion_total_bytes,
    output wire completion_revoked,result_valid,
    output wire [63:0] result_address,
    output wire [18:0] result_bytes,
    output wire norm_tag_exhausted,vocab_tags_exhausted,
    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [2:0] m_axi_arprot,
    output wire [1:0] m_axi_arburst,
    output wire [3:0] m_axi_arid,
    output wire [3:0] m_axi_arcache,
    output wire m_axi_arlock,
    output wire m_axi_arvalid,
    input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire [3:0] m_axi_rid,
    input wire m_axi_rlast,
    input wire m_axi_rvalid,
    output wire m_axi_rready,
    output wire [63:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [2:0] m_axi_awprot,
    output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awid,
    output wire [3:0] m_axi_awcache,
    output wire m_axi_awlock,
    output wire m_axi_awvalid,
    input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast,
    output wire m_axi_wvalid,
    input wire m_axi_wready,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready

);
    wire norm_command_valid;
    wire norm_command_ready;
    wire norm_busy;
    wire [63:0] norm_command_sequence;
    wire [63:0] norm_command_epoch;
    wire [63:0] norm_command_addr [0:2];
    wire [63:0] norm_command_base [0:2];
    wire [63:0] norm_command_bytes [0:2];
    wire norm_cancel_valid;
    wire norm_completion_valid;
    wire norm_completion_ready;
    wire [3:0] norm_completion_status;
    wire [31:0] norm_hls_status;
    wire [63:0] norm_sequence;
    wire [63:0] norm_epoch;
    wire [63:0] norm_result_address;
    wire [12:0] norm_result_bytes;
    wire [9:0] norm_read_beats;
    wire [6:0] norm_records;
    wire norm_result_valid;
    wire vocab_command_valid;
    wire vocab_command_ready;
    wire vocab_busy;
    wire [63:0] vocab_command_sequence;
    wire [63:0] vocab_command_epoch;
    wire [63:0] vocab_command_addr [0:2];
    wire [63:0] vocab_command_base [0:2];
    wire [63:0] vocab_command_bytes [0:2];
    wire vocab_cancel_valid;
    wire vocab_source_ready;
    wire vocab_completion_valid;
    wire vocab_completion_ready;
    wire [3:0] vocab_completion_status;
    wire [3:0] vocab_source_status;
    wire [3:0] vocab_child_status;
    wire [63:0] vocab_sequence;
    wire [63:0] vocab_epoch;
    wire [3:0] vocab_phase;
    wire [4:0] vocab_matrices;
    wire [18:0] vocab_total_bytes;
    wire vocab_result_valid;
    wire vocab_revoked;
    wire [63:0] vocab_result_address;
    wire [18:0] vocab_result_bytes;
    wire vocab_bus_owned;
    wire [63:0] norm_m_axi_araddr;
    wire [7:0] norm_m_axi_arlen;
    wire [2:0] norm_m_axi_arsize;
    wire [2:0] norm_m_axi_arprot;
    wire [1:0] norm_m_axi_arburst;
    wire [3:0] norm_m_axi_arid;
    wire [3:0] norm_m_axi_arcache;
    wire norm_m_axi_arlock;
    wire norm_m_axi_arvalid;
    wire norm_m_axi_rready;
    wire [63:0] norm_m_axi_awaddr;
    wire [7:0] norm_m_axi_awlen;
    wire [2:0] norm_m_axi_awsize;
    wire [2:0] norm_m_axi_awprot;
    wire [1:0] norm_m_axi_awburst;
    wire [3:0] norm_m_axi_awid;
    wire [3:0] norm_m_axi_awcache;
    wire norm_m_axi_awlock;
    wire norm_m_axi_awvalid;
    wire [511:0] norm_m_axi_wdata;
    wire [63:0] norm_m_axi_wstrb;
    wire norm_m_axi_wlast;
    wire norm_m_axi_wvalid;
    wire norm_m_axi_bready;
    wire [63:0] vocab_m_axi_araddr;
    wire [7:0] vocab_m_axi_arlen;
    wire [2:0] vocab_m_axi_arsize;
    wire [2:0] vocab_m_axi_arprot;
    wire [1:0] vocab_m_axi_arburst;
    wire [3:0] vocab_m_axi_arid;
    wire [3:0] vocab_m_axi_arcache;
    wire vocab_m_axi_arlock;
    wire vocab_m_axi_arvalid;
    wire vocab_m_axi_rready;
    wire [63:0] vocab_m_axi_awaddr;
    wire [7:0] vocab_m_axi_awlen;
    wire [2:0] vocab_m_axi_awsize;
    wire [2:0] vocab_m_axi_awprot;
    wire [1:0] vocab_m_axi_awburst;
    wire [3:0] vocab_m_axi_awid;
    wire [3:0] vocab_m_axi_awcache;
    wire vocab_m_axi_awlock;
    wire vocab_m_axi_awvalid;
    wire [511:0] vocab_m_axi_wdata;
    wire [63:0] vocab_m_axi_wstrb;
    wire vocab_m_axi_wlast;
    wire vocab_m_axi_wvalid;
    wire vocab_m_axi_bready;
    ntrpu_final_vocab_handoff_r1 #(.CHILD_TIMEOUT_CYCLES(HANDOFF_TIMEOUT_CYCLES)) handoff(.*);
    ntrpu_input_norm_real_r1 #(
        .AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(NORM_HLS_WATCHDOG_CYCLES),
        .INITIAL_CHILD_TAG(INITIAL_NORM_TAG)
    ) normalization(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(norm_command_valid),
        .command_ready(norm_command_ready),
        .command_sequence(norm_command_sequence),
        .command_epoch(norm_command_epoch),
        .expected_epoch(expected_epoch),
        .command_source_sequence(norm_command_sequence),
        .command_source_epoch(norm_command_epoch),
        .command_source_valid(1'b1),
        .cancel_valid(norm_cancel_valid),
        .cancel_ready(),
        .busy(norm_busy),
        .tag_exhausted(norm_tag_exhausted),
        .completion_valid(norm_completion_valid),
        .completion_ready(norm_completion_ready),
        .completion_status(norm_completion_status),
        .completion_hls_status(norm_hls_status),
        .completion_sequence(norm_sequence),
        .completion_epoch(norm_epoch),
        .completion_read_beats(norm_read_beats),
        .completion_records(norm_records),
        .result_valid(norm_result_valid),
        .completion_revoked(),
        .result_address(norm_result_address),
        .result_bytes(norm_result_bytes),
        .command_input_address(norm_command_addr[0]),
        .command_input_base(norm_command_base[0]),
        .command_input_bytes(norm_command_bytes[0]),
        .command_weight_address(norm_command_addr[1]),
        .command_weight_base(norm_command_base[1]),
        .command_weight_bytes(norm_command_bytes[1]),
        .command_output_address(norm_command_addr[2]),
        .command_output_base(norm_command_base[2]),
        .command_output_bytes(norm_command_bytes[2]),
        .m_axi_araddr(norm_m_axi_araddr),
        .m_axi_arlen(norm_m_axi_arlen),
        .m_axi_arsize(norm_m_axi_arsize),
        .m_axi_arprot(norm_m_axi_arprot),
        .m_axi_arburst(norm_m_axi_arburst),
        .m_axi_arid(norm_m_axi_arid),
        .m_axi_arcache(norm_m_axi_arcache),
        .m_axi_arlock(norm_m_axi_arlock),
        .m_axi_arvalid(norm_m_axi_arvalid),
        .m_axi_arready((!vocab_bus_owned && m_axi_arready)),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid((!vocab_bus_owned && m_axi_rvalid)),
        .m_axi_rready(norm_m_axi_rready),
        .m_axi_awaddr(norm_m_axi_awaddr),
        .m_axi_awlen(norm_m_axi_awlen),
        .m_axi_awsize(norm_m_axi_awsize),
        .m_axi_awprot(norm_m_axi_awprot),
        .m_axi_awburst(norm_m_axi_awburst),
        .m_axi_awid(norm_m_axi_awid),
        .m_axi_awcache(norm_m_axi_awcache),
        .m_axi_awlock(norm_m_axi_awlock),
        .m_axi_awvalid(norm_m_axi_awvalid),
        .m_axi_awready((!vocab_bus_owned && m_axi_awready)),
        .m_axi_wdata(norm_m_axi_wdata),
        .m_axi_wstrb(norm_m_axi_wstrb),
        .m_axi_wlast(norm_m_axi_wlast),
        .m_axi_wvalid(norm_m_axi_wvalid),
        .m_axi_wready((!vocab_bus_owned && m_axi_wready)),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid((!vocab_bus_owned && m_axi_bvalid)),
        .m_axi_bready(norm_m_axi_bready)
    );
    ntrpu_llama32_vocab_memory_r1 #(
        .WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.CHILD_TIMEOUT_CYCLES(MATRIX_TIMEOUT_CYCLES),
        .INITIAL_CHILD_TAG(INITIAL_VOCAB_TAG)
    ) projection(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(vocab_command_valid),
        .command_ready(vocab_command_ready),
        .command_sequence(vocab_command_sequence),
        .command_epoch(vocab_command_epoch),
        .expected_epoch(expected_epoch),
        .command_addr(vocab_command_addr),
        .command_base(vocab_command_base),
        .command_bytes(vocab_command_bytes),
        .cancel_valid(vocab_cancel_valid),
        .cancel_ready(),
        .busy(vocab_busy),
        .source_completion_valid(norm_completion_valid),
        .source_completion_ready(vocab_source_ready),
        .source_completion_status(norm_completion_status),
        .source_sequence(norm_sequence),
        .source_epoch(norm_epoch),
        .source_result_address(norm_result_address),
        .source_result_valid(norm_result_valid),
        .source_result_bytes(norm_result_bytes),
        .completion_valid(vocab_completion_valid),
        .completion_ready(vocab_completion_ready),
        .completion_status(vocab_completion_status),
        .completion_source_status(vocab_source_status),
        .completion_child_status(vocab_child_status),
        .completion_sequence(vocab_sequence),
        .completion_epoch(vocab_epoch),
        .completion_phase(vocab_phase),
        .completion_matrices(vocab_matrices),
        .completion_total_bytes(vocab_total_bytes),
        .completion_child_bytes(),
        .completion_child_rows(),
        .completion_revoked(vocab_revoked),
        .result_valid(vocab_result_valid),
        .tags_exhausted(vocab_tags_exhausted),
        .result_address(vocab_result_address),
        .result_bytes(vocab_result_bytes),
        .m_axi_araddr(vocab_m_axi_araddr),
        .m_axi_arlen(vocab_m_axi_arlen),
        .m_axi_arsize(vocab_m_axi_arsize),
        .m_axi_arprot(vocab_m_axi_arprot),
        .m_axi_arburst(vocab_m_axi_arburst),
        .m_axi_arid(vocab_m_axi_arid),
        .m_axi_arcache(vocab_m_axi_arcache),
        .m_axi_arlock(vocab_m_axi_arlock),
        .m_axi_arvalid(vocab_m_axi_arvalid),
        .m_axi_arready((vocab_bus_owned && m_axi_arready)),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid((vocab_bus_owned && m_axi_rvalid)),
        .m_axi_rready(vocab_m_axi_rready),
        .m_axi_awaddr(vocab_m_axi_awaddr),
        .m_axi_awlen(vocab_m_axi_awlen),
        .m_axi_awsize(vocab_m_axi_awsize),
        .m_axi_awprot(vocab_m_axi_awprot),
        .m_axi_awburst(vocab_m_axi_awburst),
        .m_axi_awid(vocab_m_axi_awid),
        .m_axi_awcache(vocab_m_axi_awcache),
        .m_axi_awlock(vocab_m_axi_awlock),
        .m_axi_awvalid(vocab_m_axi_awvalid),
        .m_axi_awready((vocab_bus_owned && m_axi_awready)),
        .m_axi_wdata(vocab_m_axi_wdata),
        .m_axi_wstrb(vocab_m_axi_wstrb),
        .m_axi_wlast(vocab_m_axi_wlast),
        .m_axi_wvalid(vocab_m_axi_wvalid),
        .m_axi_wready((vocab_bus_owned && m_axi_wready)),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid((vocab_bus_owned && m_axi_bvalid)),
        .m_axi_bready(vocab_m_axi_bready)
    );
    // Selection changes only after norm has drained and VOCAB accepts its command.
    // The handoff retains it until VOCAB and its source have both retired.
    assign m_axi_araddr=vocab_bus_owned ? vocab_m_axi_araddr:norm_m_axi_araddr;
    assign m_axi_arlen=vocab_bus_owned ? vocab_m_axi_arlen:norm_m_axi_arlen;
    assign m_axi_arsize=vocab_bus_owned ? vocab_m_axi_arsize:norm_m_axi_arsize;
    assign m_axi_arprot=vocab_bus_owned ? vocab_m_axi_arprot:norm_m_axi_arprot;
    assign m_axi_arburst=vocab_bus_owned ? vocab_m_axi_arburst:norm_m_axi_arburst;
    assign m_axi_arid=vocab_bus_owned ? vocab_m_axi_arid:norm_m_axi_arid;
    assign m_axi_arcache=vocab_bus_owned ? vocab_m_axi_arcache:norm_m_axi_arcache;
    assign m_axi_arlock=vocab_bus_owned ? vocab_m_axi_arlock:norm_m_axi_arlock;
    assign m_axi_arvalid=vocab_bus_owned ? vocab_m_axi_arvalid:norm_m_axi_arvalid;
    assign m_axi_rready=vocab_bus_owned ? vocab_m_axi_rready:norm_m_axi_rready;
    assign m_axi_awaddr=vocab_bus_owned ? vocab_m_axi_awaddr:norm_m_axi_awaddr;
    assign m_axi_awlen=vocab_bus_owned ? vocab_m_axi_awlen:norm_m_axi_awlen;
    assign m_axi_awsize=vocab_bus_owned ? vocab_m_axi_awsize:norm_m_axi_awsize;
    assign m_axi_awprot=vocab_bus_owned ? vocab_m_axi_awprot:norm_m_axi_awprot;
    assign m_axi_awburst=vocab_bus_owned ? vocab_m_axi_awburst:norm_m_axi_awburst;
    assign m_axi_awid=vocab_bus_owned ? vocab_m_axi_awid:norm_m_axi_awid;
    assign m_axi_awcache=vocab_bus_owned ? vocab_m_axi_awcache:norm_m_axi_awcache;
    assign m_axi_awlock=vocab_bus_owned ? vocab_m_axi_awlock:norm_m_axi_awlock;
    assign m_axi_awvalid=vocab_bus_owned ? vocab_m_axi_awvalid:norm_m_axi_awvalid;
    assign m_axi_wdata=vocab_bus_owned ? vocab_m_axi_wdata:norm_m_axi_wdata;
    assign m_axi_wstrb=vocab_bus_owned ? vocab_m_axi_wstrb:norm_m_axi_wstrb;
    assign m_axi_wlast=vocab_bus_owned ? vocab_m_axi_wlast:norm_m_axi_wlast;
    assign m_axi_wvalid=vocab_bus_owned ? vocab_m_axi_wvalid:norm_m_axi_wvalid;
    assign m_axi_bready=vocab_bus_owned ? vocab_m_axi_bready:norm_m_axi_bready;
endmodule
`default_nettype wire
