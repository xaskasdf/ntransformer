// Two private tile banks. Only a successful AXI completion publishes a bank.
// Single clock; see docs/fpga/WEIGHT_TILE_BUFFER.md for ownership and reset rules.
module ntrpu_weight_tile_buffer #(
    parameter integer TILE_BEATS = 544
) (
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
    output logic [511:0] tile_data,
    output logic         tile_valid,
    input  logic         tile_ready,
    output logic         tile_last,
    output logic [31:0]  tile_tag,
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
    localparam integer ADDR_WIDTH = TILE_BEATS > 1 ? $clog2(TILE_BEATS) : 1;
    localparam integer COUNT_WIDTH = $clog2(TILE_BEATS + 1);
    localparam logic [31:0] MAX_BYTES = TILE_BEATS * 64;
    typedef enum logic [1:0] {IDLE, ISSUE, FILL, RESPONSE} state_t;
    state_t state;
    logic producer_bank, consumer_bank, output_bank;
    logic [1:0] bank_valid;
    logic [COUNT_WIDTH-1:0] bank_words [0:1];
    logic [31:0] bank_tags [0:1];
    logic [COUNT_WIDTH-1:0] fill_count, fill_words, read_index;
    logic [63:0] command_addr;
    logic [31:0] command_bytes;
    logic fill_bad;
    (* ram_style = "block" *) logic [511:0] bank0 [0:TILE_BEATS-1];
    (* ram_style = "block" *) logic [511:0] bank1 [0:TILE_BEATS-1];
    logic [511:0] bank0_q, bank1_q;
    logic read_enable, write_enable;
    logic reader_cmd_ready, reader_done_valid;
    logic [1:0] reader_done_status;
    logic [31:0] reader_done_bytes;
    logic [511:0] reader_data;
    logic reader_valid, reader_ready, reader_last;

    assign load_ready = state == IDLE && !bank_valid[producer_bank];
    assign load_done_valid = state == RESPONSE;
    assign reader_ready = state == FILL;
    assign write_enable = reader_valid && reader_ready && fill_count < fill_words;
    assign read_enable = bank_valid[consumer_bank] && read_index < bank_words[consumer_bank] &&
                         (!tile_valid || tile_ready);
    assign tile_data = output_bank ? bank1_q : bank0_q;

    // RAM contents are deliberately not reset. Ownership metadata hides stale data.
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (write_enable && !producer_bank) bank0[fill_count[ADDR_WIDTH-1:0]] <= reader_data;
            if (write_enable && producer_bank) bank1[fill_count[ADDR_WIDTH-1:0]] <= reader_data;
            if (read_enable && !consumer_bank) bank0_q <= bank0[read_index[ADDR_WIDTH-1:0]];
            if (read_enable && consumer_bank) bank1_q <= bank1[read_index[ADDR_WIDTH-1:0]];
        end
    end

    ntrpu_axi_read_master reader (
        .clk, .rst_n,
        .cmd_valid(state == ISSUE), .cmd_ready(reader_cmd_ready),
        .cmd_addr(command_addr), .cmd_bytes(command_bytes),
        .done_valid(reader_done_valid), .done_ready(state == FILL),
        .done_status(reader_done_status), .done_bytes(reader_done_bytes),
        .out_data(reader_data), .out_valid(reader_valid), .out_ready(reader_ready), .out_last(reader_last),
        .m_axi_araddr, .m_axi_arlen, .m_axi_arsize, .m_axi_arburst, .m_axi_arid,
        .m_axi_arlock, .m_axi_arcache, .m_axi_arprot, .m_axi_arvalid, .m_axi_arready,
        .m_axi_rdata, .m_axi_rresp, .m_axi_rid, .m_axi_rlast, .m_axi_rvalid, .m_axi_rready
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            producer_bank <= 0;
            consumer_bank <= 0;
            output_bank <= 0;
            bank_valid <= 0;
            bank_words[0] <= 0;
            bank_words[1] <= 0;
            bank_tags[0] <= 0;
            bank_tags[1] <= 0;
            fill_count <= 0;
            fill_words <= 0;
            read_index <= 0;
            command_addr <= 0;
            command_bytes <= 0;
            fill_bad <= 0;
            load_done_status <= 0;
            load_done_bytes <= 0;
            load_done_tag <= 0;
            tile_valid <= 0;
            tile_tag <= 0;
            tile_last <= 0;
        end else begin
            if (tile_valid && tile_ready) begin
                tile_valid <= 0;
                if (tile_last) begin
                    bank_valid[consumer_bank] <= 0;
                    consumer_bank <= !consumer_bank;
                    read_index <= 0;
                end
            end
            if (read_enable) begin
                output_bank <= consumer_bank;
                tile_valid <= 1;
                tile_tag <= bank_tags[consumer_bank];
                tile_last <= read_index + 1'b1 == bank_words[consumer_bank];
                read_index <= read_index + 1'b1;
            end
            case (state)
                IDLE: if (load_valid && load_ready) begin
                    command_addr <= load_addr;
                    command_bytes <= load_bytes;
                    load_done_tag <= load_tag;
                    load_done_bytes <= 0;
                    load_done_status <= 0;
                    fill_count <= 0;
                    fill_words <= COUNT_WIDTH'(load_bytes >> 6);
                    fill_bad <= 0;
                    if (load_bytes == 0 || load_bytes > MAX_BYTES || load_bytes[5:0] != 0) begin
                        load_done_status <= 1;
                        state <= RESPONSE;
                    end else state <= ISSUE;
                end
                ISSUE: if (reader_cmd_ready) state <= FILL;
                FILL: begin
                    if (reader_valid && reader_ready) begin
                        if (fill_count >= fill_words || reader_last != (fill_count + 1'b1 == fill_words))
                            fill_bad <= 1;
                        if (fill_count < fill_words) fill_count <= fill_count + 1'b1;
                    end
                    if (reader_done_valid) begin
                        load_done_bytes <= reader_done_bytes;
                        if (reader_done_status != 0) load_done_status <= reader_done_status;
                        else if (fill_bad || fill_count != fill_words || reader_done_bytes != command_bytes)
                            load_done_status <= 3;
                        else begin
                            bank_valid[producer_bank] <= 1;
                            bank_words[producer_bank] <= fill_words;
                            bank_tags[producer_bank] <= load_done_tag;
                            producer_bank <= !producer_bank;
                        end
                        state <= RESPONSE;
                    end
                end
                RESPONSE: if (load_done_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
