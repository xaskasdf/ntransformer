// AXI reader -> private tile banks -> Q8_0 block stream. No GEMV arithmetic yet.
module ntrpu_q8_weight_stream (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         load_valid,
    output logic         load_ready,
    input  logic [63:0]  load_addr,
    input  logic [31:0]  load_bytes,
    input  logic [31:0]  load_tag,
    output logic         load_done_valid,
    input  logic         load_done_ready,
    output logic [1:0]   load_done_status,
    output logic [31:0]  load_done_bytes,
    output logic [31:0]  load_done_tag,
    output logic [15:0]  block_scale,
    output logic [255:0] block_weights,
    output logic [9:0]   block_index,
    output logic [31:0]  block_tag,
    output logic         block_last,
    output logic         block_valid,
    input  logic         block_ready,
    output logic         decode_done_valid,
    input  logic         decode_done_ready,
    output logic [1:0]   decode_done_status,
    output logic [10:0]  decode_done_blocks,
    output logic [31:0]  decode_done_tag,
    output logic [63:0]  m_axi_araddr,
    output logic [7:0]   m_axi_arlen,
    output logic [2:0]   m_axi_arsize,
    output logic [1:0]   m_axi_arburst,
    output logic [3:0]   m_axi_arid,
    output logic         m_axi_arlock,
    output logic [3:0]   m_axi_arcache,
    output logic [2:0]   m_axi_arprot,
    output logic         m_axi_arvalid,
    input  logic         m_axi_arready,
    input  logic [511:0] m_axi_rdata,
    input  logic [1:0]   m_axi_rresp,
    input  logic [3:0]   m_axi_rid,
    input  logic         m_axi_rlast,
    input  logic         m_axi_rvalid,
    output logic         m_axi_rready
);
    logic [511:0] tile_data;
    logic tile_valid, tile_ready, tile_last;
    logic [31:0] tile_tag;
    ntrpu_weight_tile_buffer tiles (.*);
    ntrpu_q8_unpacker unpacker (
        .clk, .rst_n,
        .in_data(tile_data), .in_valid(tile_valid), .in_ready(tile_ready),
        .in_last(tile_last), .in_tag(tile_tag),
        .block_scale, .block_weights, .block_index, .block_tag, .block_last, .block_valid, .block_ready,
        .done_valid(decode_done_valid), .done_ready(decode_done_ready),
        .done_status(decode_done_status), .done_blocks(decode_done_blocks), .done_tag(decode_done_tag)
    );
endmodule
