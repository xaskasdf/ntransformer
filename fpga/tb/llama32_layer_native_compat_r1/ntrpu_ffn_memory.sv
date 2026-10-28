// Shared-clock FFN: gate -> up -> SwiGLU/quantize -> down.
// One matrix engine, one activation engine; AXI changes owner only after drain.
// Reset must also reset/quiesce the shared AXI peer. Publish only root status OK.
module ntrpu_ffn_memory #(parameter integer WATCHDOG_CYCLES=65536,parameter bit OUTPUT_FP32=1) (

    input logic clk,rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [1:0] job_layout,
    input logic [13:0] job_dim,job_hidden,
    input logic [5:0] job_tile_k [0:2],
    input logic [63:0] job_addr [0:7],
    input logic [63:0] job_base [0:7],
    input logic [63:0] job_bytes [0:7],
    input logic [31:0] job_buffer_epoch [0:7],
    input logic [31:0] job_sequence,job_epoch,expected_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,done_child_status,
    output logic [2:0] done_stages,
    output logic [1:0] done_phase,
    output logic [16:0] done_bytes,
    output logic [17:0] done_stage_bytes,
    output logic [31:0] done_sequence,done_epoch,done_hls_status,
    output logic activation_selected,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize, m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid, m_axi_arcache,
    output logic m_axi_arlock, m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize, m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid, m_axi_awcache,
    output logic m_axi_awlock, m_axi_awvalid,
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast, m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready
 );
    logic  m_job_valid,m_job_output_fp32;
    logic  m_job_ready;
    logic [1:0] m_job_layout;
    logic [13:0] m_job_rows;
    logic [8:0] m_job_k_blocks;
    logic [5:0] m_job_tile_k;
    logic [31:0] m_job_sequence;
    logic [31:0] m_job_epoch;
    logic [63:0] m_job_weight_addr;
    logic [63:0] m_job_region_base;
    logic [63:0] m_job_region_bytes;
    logic [63:0] m_job_activation_addr;
    logic [63:0] m_job_activation_region_base;
    logic [63:0] m_job_activation_region_bytes;
    logic [31:0] m_job_activation_epoch;
    logic [63:0] m_job_output_addr;
    logic [63:0] m_job_output_region_base;
    logic [63:0] m_job_output_region_bytes;
    logic [31:0] m_job_output_epoch;
    logic  m_cancel_valid;
    logic  m_cancel_ready;
    logic  m_done_valid;
    logic  m_done_ready;
    logic [3:0] m_done_status;
    logic [16:0] m_done_bytes;
    logic [13:0] m_done_rows;
    logic [31:0] m_done_sequence;
    logic [31:0] m_done_epoch;
    logic  a_job_valid;
    logic  a_job_ready;
    logic [13:0] a_job_elements;
    logic [31:0] a_job_sequence;
    logic [31:0] a_job_epoch;
    logic [31:0] a_expected_epoch;
    logic [63:0] a_job_gate_addr;
    logic [63:0] a_job_gate_base;
    logic [63:0] a_job_gate_bytes;
    logic [63:0] a_job_up_addr;
    logic [63:0] a_job_up_base;
    logic [63:0] a_job_up_bytes;
    logic [63:0] a_job_output_addr;
    logic [63:0] a_job_output_base;
    logic [63:0] a_job_output_bytes;
    logic [31:0] a_job_gate_epoch;
    logic [31:0] a_job_up_epoch;
    logic [31:0] a_job_output_epoch;
    logic  a_cancel_valid;
    logic  a_cancel_ready;
    logic  a_done_valid;
    logic  a_done_ready;
    logic [3:0] a_done_status;
    logic [8:0] a_done_blocks;
    logic [14:0] a_done_bytes;
    logic [31:0] a_done_sequence;
    logic [31:0] a_done_epoch;
    logic [31:0] a_done_hls_status;
    logic [63:0] matrix_m_axi_araddr;
    logic [7:0] matrix_m_axi_arlen;
    logic [2:0] matrix_m_axi_arsize;
    logic [2:0] matrix_m_axi_arprot;
    logic [1:0] matrix_m_axi_arburst;
    logic [3:0] matrix_m_axi_arid;
    logic [3:0] matrix_m_axi_arcache;
    logic  matrix_m_axi_arlock;
    logic  matrix_m_axi_arvalid;
    logic  matrix_m_axi_arready;
    logic [511:0] matrix_m_axi_rdata;
    logic [1:0] matrix_m_axi_rresp;
    logic [3:0] matrix_m_axi_rid;
    logic  matrix_m_axi_rlast;
    logic  matrix_m_axi_rvalid;
    logic  matrix_m_axi_rready;
    logic [63:0] matrix_m_axi_awaddr;
    logic [7:0] matrix_m_axi_awlen;
    logic [2:0] matrix_m_axi_awsize;
    logic [2:0] matrix_m_axi_awprot;
    logic [1:0] matrix_m_axi_awburst;
    logic [3:0] matrix_m_axi_awid;
    logic [3:0] matrix_m_axi_awcache;
    logic  matrix_m_axi_awlock;
    logic  matrix_m_axi_awvalid;
    logic  matrix_m_axi_awready;
    logic [511:0] matrix_m_axi_wdata;
    logic [63:0] matrix_m_axi_wstrb;
    logic  matrix_m_axi_wlast;
    logic  matrix_m_axi_wvalid;
    logic  matrix_m_axi_wready;
    logic [3:0] matrix_m_axi_bid;
    logic [1:0] matrix_m_axi_bresp;
    logic  matrix_m_axi_bvalid;
    logic  matrix_m_axi_bready;
    logic [63:0] activation_m_axi_araddr;
    logic [7:0] activation_m_axi_arlen;
    logic [2:0] activation_m_axi_arsize;
    logic [2:0] activation_m_axi_arprot;
    logic [1:0] activation_m_axi_arburst;
    logic [3:0] activation_m_axi_arid;
    logic [3:0] activation_m_axi_arcache;
    logic  activation_m_axi_arlock;
    logic  activation_m_axi_arvalid;
    logic  activation_m_axi_arready;
    logic [511:0] activation_m_axi_rdata;
    logic [1:0] activation_m_axi_rresp;
    logic [3:0] activation_m_axi_rid;
    logic  activation_m_axi_rlast;
    logic  activation_m_axi_rvalid;
    logic  activation_m_axi_rready;
    logic [63:0] activation_m_axi_awaddr;
    logic [7:0] activation_m_axi_awlen;
    logic [2:0] activation_m_axi_awsize;
    logic [2:0] activation_m_axi_awprot;
    logic [1:0] activation_m_axi_awburst;
    logic [3:0] activation_m_axi_awid;
    logic [3:0] activation_m_axi_awcache;
    logic  activation_m_axi_awlock;
    logic  activation_m_axi_awvalid;
    logic  activation_m_axi_awready;
    logic [511:0] activation_m_axi_wdata;
    logic [63:0] activation_m_axi_wstrb;
    logic  activation_m_axi_wlast;
    logic  activation_m_axi_wvalid;
    logic  activation_m_axi_wready;
    logic [3:0] activation_m_axi_bid;
    logic [1:0] activation_m_axi_bresp;
    logic  activation_m_axi_bvalid;
    logic  activation_m_axi_bready;
    ntrpu_ffn_sequencer #(.OUTPUT_FP32(OUTPUT_FP32)) controller(.*);
    ntrpu_q8_matrix_memory #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.FP32_CAPABLE(OUTPUT_FP32)) matrix (
        .clk(clk),
        .rst_n(rst_n),
        .job_valid(m_job_valid),
        .job_ready(m_job_ready),
        .job_layout(m_job_layout),
        .job_rows(m_job_rows),
        .job_output_fp32(m_job_output_fp32),
        .job_k_blocks(m_job_k_blocks),
        .job_tile_k(m_job_tile_k),
        .job_sequence(m_job_sequence),
        .job_epoch(m_job_epoch),
        .job_weight_addr(m_job_weight_addr),
        .job_region_base(m_job_region_base),
        .job_region_bytes(m_job_region_bytes),
        .job_activation_addr(m_job_activation_addr),
        .job_activation_region_base(m_job_activation_region_base),
        .job_activation_region_bytes(m_job_activation_region_bytes),
        .job_activation_epoch(m_job_activation_epoch),
        .job_output_addr(m_job_output_addr),
        .job_output_region_base(m_job_output_region_base),
        .job_output_region_bytes(m_job_output_region_bytes),
        .job_output_epoch(m_job_output_epoch),
        .cancel_valid(m_cancel_valid),
        .cancel_ready(m_cancel_ready),
        .done_valid(m_done_valid),
        .done_ready(m_done_ready),
        .done_status(m_done_status),
        .done_bytes(m_done_bytes),
        .done_rows(m_done_rows),
        .done_sequence(m_done_sequence),
        .done_epoch(m_done_epoch),
        .m_axi_araddr(matrix_m_axi_araddr),
        .m_axi_arlen(matrix_m_axi_arlen),
        .m_axi_arsize(matrix_m_axi_arsize),
        .m_axi_arprot(matrix_m_axi_arprot),
        .m_axi_arburst(matrix_m_axi_arburst),
        .m_axi_arid(matrix_m_axi_arid),
        .m_axi_arcache(matrix_m_axi_arcache),
        .m_axi_arlock(matrix_m_axi_arlock),
        .m_axi_arvalid(matrix_m_axi_arvalid),
        .m_axi_arready(matrix_m_axi_arready),
        .m_axi_rdata(matrix_m_axi_rdata),
        .m_axi_rresp(matrix_m_axi_rresp),
        .m_axi_rid(matrix_m_axi_rid),
        .m_axi_rlast(matrix_m_axi_rlast),
        .m_axi_rvalid(matrix_m_axi_rvalid),
        .m_axi_rready(matrix_m_axi_rready),
        .m_axi_awaddr(matrix_m_axi_awaddr),
        .m_axi_awlen(matrix_m_axi_awlen),
        .m_axi_awsize(matrix_m_axi_awsize),
        .m_axi_awprot(matrix_m_axi_awprot),
        .m_axi_awburst(matrix_m_axi_awburst),
        .m_axi_awid(matrix_m_axi_awid),
        .m_axi_awcache(matrix_m_axi_awcache),
        .m_axi_awlock(matrix_m_axi_awlock),
        .m_axi_awvalid(matrix_m_axi_awvalid),
        .m_axi_awready(matrix_m_axi_awready),
        .m_axi_wdata(matrix_m_axi_wdata),
        .m_axi_wstrb(matrix_m_axi_wstrb),
        .m_axi_wlast(matrix_m_axi_wlast),
        .m_axi_wvalid(matrix_m_axi_wvalid),
        .m_axi_wready(matrix_m_axi_wready),
        .m_axi_bid(matrix_m_axi_bid),
        .m_axi_bresp(matrix_m_axi_bresp),
        .m_axi_bvalid(matrix_m_axi_bvalid),
        .m_axi_bready(matrix_m_axi_bready)
    );
    ntrpu_swiglu_memory #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) activation (
        .clk(clk),
        .rst_n(rst_n),
        .job_valid(a_job_valid),
        .job_ready(a_job_ready),
        .job_elements(a_job_elements),
        .job_sequence(a_job_sequence),
        .job_epoch(a_job_epoch),
        .expected_epoch(a_expected_epoch),
        .job_gate_addr(a_job_gate_addr),
        .job_gate_base(a_job_gate_base),
        .job_gate_bytes(a_job_gate_bytes),
        .job_up_addr(a_job_up_addr),
        .job_up_base(a_job_up_base),
        .job_up_bytes(a_job_up_bytes),
        .job_output_addr(a_job_output_addr),
        .job_output_base(a_job_output_base),
        .job_output_bytes(a_job_output_bytes),
        .job_gate_epoch(a_job_gate_epoch),
        .job_up_epoch(a_job_up_epoch),
        .job_output_epoch(a_job_output_epoch),
        .cancel_valid(a_cancel_valid),
        .cancel_ready(a_cancel_ready),
        .done_valid(a_done_valid),
        .done_ready(a_done_ready),
        .done_status(a_done_status),
        .done_blocks(a_done_blocks),
        .done_bytes(a_done_bytes),
        .done_sequence(a_done_sequence),
        .done_epoch(a_done_epoch),
        .done_hls_status(a_done_hls_status),
        .m_axi_araddr(activation_m_axi_araddr),
        .m_axi_arlen(activation_m_axi_arlen),
        .m_axi_arsize(activation_m_axi_arsize),
        .m_axi_arprot(activation_m_axi_arprot),
        .m_axi_arburst(activation_m_axi_arburst),
        .m_axi_arid(activation_m_axi_arid),
        .m_axi_arcache(activation_m_axi_arcache),
        .m_axi_arlock(activation_m_axi_arlock),
        .m_axi_arvalid(activation_m_axi_arvalid),
        .m_axi_arready(activation_m_axi_arready),
        .m_axi_rdata(activation_m_axi_rdata),
        .m_axi_rresp(activation_m_axi_rresp),
        .m_axi_rid(activation_m_axi_rid),
        .m_axi_rlast(activation_m_axi_rlast),
        .m_axi_rvalid(activation_m_axi_rvalid),
        .m_axi_rready(activation_m_axi_rready),
        .m_axi_awaddr(activation_m_axi_awaddr),
        .m_axi_awlen(activation_m_axi_awlen),
        .m_axi_awsize(activation_m_axi_awsize),
        .m_axi_awprot(activation_m_axi_awprot),
        .m_axi_awburst(activation_m_axi_awburst),
        .m_axi_awid(activation_m_axi_awid),
        .m_axi_awcache(activation_m_axi_awcache),
        .m_axi_awlock(activation_m_axi_awlock),
        .m_axi_awvalid(activation_m_axi_awvalid),
        .m_axi_awready(activation_m_axi_awready),
        .m_axi_wdata(activation_m_axi_wdata),
        .m_axi_wstrb(activation_m_axi_wstrb),
        .m_axi_wlast(activation_m_axi_wlast),
        .m_axi_wvalid(activation_m_axi_wvalid),
        .m_axi_wready(activation_m_axi_wready),
        .m_axi_bid(activation_m_axi_bid),
        .m_axi_bresp(activation_m_axi_bresp),
        .m_axi_bvalid(activation_m_axi_bvalid),
        .m_axi_bready(activation_m_axi_bready)
    );
    assign m_axi_araddr=activation_selected ? activation_m_axi_araddr:matrix_m_axi_araddr;
    assign m_axi_arlen=activation_selected ? activation_m_axi_arlen:matrix_m_axi_arlen;
    assign m_axi_arsize=activation_selected ? activation_m_axi_arsize:matrix_m_axi_arsize;
    assign m_axi_arprot=activation_selected ? activation_m_axi_arprot:matrix_m_axi_arprot;
    assign m_axi_arburst=activation_selected ? activation_m_axi_arburst:matrix_m_axi_arburst;
    assign m_axi_arid=activation_selected ? activation_m_axi_arid:matrix_m_axi_arid;
    assign m_axi_arcache=activation_selected ? activation_m_axi_arcache:matrix_m_axi_arcache;
    assign m_axi_arlock=activation_selected ? activation_m_axi_arlock:matrix_m_axi_arlock;
    assign m_axi_arvalid=activation_selected ? activation_m_axi_arvalid:matrix_m_axi_arvalid;
    assign matrix_m_axi_arready=m_axi_arready && !activation_selected;
    assign activation_m_axi_arready=m_axi_arready && activation_selected;
    assign matrix_m_axi_rdata=m_axi_rdata;
    assign activation_m_axi_rdata=m_axi_rdata;
    assign matrix_m_axi_rresp=m_axi_rresp;
    assign activation_m_axi_rresp=m_axi_rresp;
    assign matrix_m_axi_rid=m_axi_rid;
    assign activation_m_axi_rid=m_axi_rid;
    assign matrix_m_axi_rlast=m_axi_rlast;
    assign activation_m_axi_rlast=m_axi_rlast;
    assign matrix_m_axi_rvalid=m_axi_rvalid && !activation_selected;
    assign activation_m_axi_rvalid=m_axi_rvalid && activation_selected;
    assign m_axi_rready=activation_selected ? activation_m_axi_rready:matrix_m_axi_rready;
    assign m_axi_awaddr=activation_selected ? activation_m_axi_awaddr:matrix_m_axi_awaddr;
    assign m_axi_awlen=activation_selected ? activation_m_axi_awlen:matrix_m_axi_awlen;
    assign m_axi_awsize=activation_selected ? activation_m_axi_awsize:matrix_m_axi_awsize;
    assign m_axi_awprot=activation_selected ? activation_m_axi_awprot:matrix_m_axi_awprot;
    assign m_axi_awburst=activation_selected ? activation_m_axi_awburst:matrix_m_axi_awburst;
    assign m_axi_awid=activation_selected ? activation_m_axi_awid:matrix_m_axi_awid;
    assign m_axi_awcache=activation_selected ? activation_m_axi_awcache:matrix_m_axi_awcache;
    assign m_axi_awlock=activation_selected ? activation_m_axi_awlock:matrix_m_axi_awlock;
    assign m_axi_awvalid=activation_selected ? activation_m_axi_awvalid:matrix_m_axi_awvalid;
    assign matrix_m_axi_awready=m_axi_awready && !activation_selected;
    assign activation_m_axi_awready=m_axi_awready && activation_selected;
    assign m_axi_wdata=activation_selected ? activation_m_axi_wdata:matrix_m_axi_wdata;
    assign m_axi_wstrb=activation_selected ? activation_m_axi_wstrb:matrix_m_axi_wstrb;
    assign m_axi_wlast=activation_selected ? activation_m_axi_wlast:matrix_m_axi_wlast;
    assign m_axi_wvalid=activation_selected ? activation_m_axi_wvalid:matrix_m_axi_wvalid;
    assign matrix_m_axi_wready=m_axi_wready && !activation_selected;
    assign activation_m_axi_wready=m_axi_wready && activation_selected;
    assign matrix_m_axi_bid=m_axi_bid;
    assign activation_m_axi_bid=m_axi_bid;
    assign matrix_m_axi_bresp=m_axi_bresp;
    assign activation_m_axi_bresp=m_axi_bresp;
    assign matrix_m_axi_bvalid=m_axi_bvalid && !activation_selected;
    assign activation_m_axi_bvalid=m_axi_bvalid && activation_selected;
    assign m_axi_bready=activation_selected ? activation_m_axi_bready:matrix_m_axi_bready;
endmodule
