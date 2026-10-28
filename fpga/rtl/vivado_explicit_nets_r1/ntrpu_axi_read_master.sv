// Single-clock, single-outstanding AXI4 weight reader. See docs/fpga/AXI_READER.md.
module ntrpu_axi_read_master (
    input wire logic         clk,
    input wire logic         rst_n,
    input wire logic         cmd_valid,
    output logic         cmd_ready,
    input wire logic [63:0]  cmd_addr,
    input wire logic [31:0]  cmd_bytes,
    output logic         done_valid,
    input wire logic         done_ready,
    output logic [1:0]   done_status,
    output logic [31:0]  done_bytes,
    output logic [511:0] out_data,
    output logic         out_valid,
    input wire logic         out_ready,
    output logic         out_last,
    output logic [63:0]  m_axi_araddr,
    output logic [7:0]   m_axi_arlen,
    output logic [2:0]   m_axi_arsize,
    output logic [1:0]   m_axi_arburst,
    output logic [3:0]   m_axi_arid,
    output logic         m_axi_arlock,
    output logic [3:0]   m_axi_arcache,
    output logic [2:0]   m_axi_arprot,
    output logic         m_axi_arvalid,
    input wire logic         m_axi_arready,
    input wire logic [511:0] m_axi_rdata,
    input wire logic [1:0]   m_axi_rresp,
    input wire logic [3:0]   m_axi_rid,
    input wire logic         m_axi_rlast,
    input wire logic         m_axi_rvalid,
    output logic         m_axi_rready
);
    localparam logic [1:0] OK = 0, BAD_COMMAND = 1, READ_ERROR = 2, PROTOCOL_ERROR = 3;
    typedef enum logic [2:0] {IDLE, ADDRESS, RECEIVE, FLUSH, COMPLETE} state_t;
    state_t state;
    logic [63:0] address;
    logic [31:0] remaining;
    logic [6:0] burst_beats, beats_left;
    logic [1:0] status;
    logic [31:0] delivered;
    logic [64:0] command_end;
    logic [1:0] beat_error;

    function automatic logic [6:0] next_burst(input logic [63:0] addr,
                                              input logic [31:0] byte_count);
        logic [31:0] count;
        logic [6:0] page_beats;
        begin
            count = byte_count >> 6;
            page_beats = 7'd64 - {1'b0, addr[11:6]};
            next_burst = (count < {25'b0, page_beats}) ? count[6:0] : page_beats;
        end
    endfunction

    assign command_end = {1'b0, cmd_addr} + {33'b0, cmd_bytes} - 65'd1;
    assign cmd_ready = state == IDLE;
    assign done_valid = state == COMPLETE;
    assign done_status = status;
    assign done_bytes = delivered;
    assign m_axi_araddr = address;
    assign m_axi_arlen = {1'b0, burst_beats} - 8'd1;
    assign m_axi_arsize = 3'd6;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arid = 4'd0;
    assign m_axi_arlock = 1'b0;
    assign m_axi_arcache = 4'b0010; // Normal, non-cacheable, non-bufferable.
    assign m_axi_arprot = 3'b000;   // Unprivileged, secure, data: integrate per platform.
    assign m_axi_arvalid = state == ADDRESS;
    // One elastic register decouples AXI R payload from the consumer.
    // After an error, drain the issued burst even if the consumer stalls.
    assign m_axi_rready = state == RECEIVE &&
                          (status != OK || !out_valid || out_ready);
    always_comb begin
        beat_error = OK;
        if (m_axi_rresp != 2'b00)
            beat_error = READ_ERROR;
        else if (m_axi_rid != 0 || beats_left == 0 ||
                 m_axi_rlast != (beats_left == 1))
            beat_error = PROTOCOL_ERROR;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            address <= 0;
            remaining <= 0;
            burst_beats <= 0;
            beats_left <= 0;
            status <= OK;
            delivered <= 0;
            out_valid <= 0;
            out_data <= 0;
            out_last <= 0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 0;
                delivered <= delivered + 32'd64;
            end
            case (state)
                IDLE: if (cmd_valid) begin
                    delivered <= 0;
                    status <= OK;
                    if (cmd_bytes == 0 || cmd_bytes[5:0] != 0 ||
                        cmd_addr[5:0] != 0 || command_end[64]) begin
                        status <= BAD_COMMAND;
                        state <= COMPLETE;
                    end else begin
                        address <= cmd_addr;
                        remaining <= cmd_bytes;
                        burst_beats <= next_burst(cmd_addr, cmd_bytes);
                        state <= ADDRESS;
                    end
                end
                ADDRESS: if (m_axi_arready) begin
                    beats_left <= burst_beats;
                    state <= RECEIVE;
                end
                RECEIVE: if (m_axi_rvalid && m_axi_rready) begin
                    if (status == OK && beat_error != OK)
                        status <= beat_error;
                    if (status == OK && beat_error == OK) begin
                        out_data <= m_axi_rdata;
                        out_last <= remaining == 64;
                        out_valid <= 1;
                    end
                    if (beats_left != 0)
                        beats_left <= beats_left - 7'd1;
                    if (remaining >= 64)
                        remaining <= remaining - 32'd64;
                    if (m_axi_rlast) begin
                        if (status != OK || beat_error != OK || remaining == 64)
                            state <= FLUSH;
                        else begin
                            address <= address + ({57'b0, burst_beats} << 6);
                            burst_beats <= next_burst(
                                address + ({57'b0, burst_beats} << 6), remaining - 32'd64);
                            state <= ADDRESS;
                        end
                    end
                end
                FLUSH: if (!out_valid) state <= COMPLETE;
                COMPLETE: if (done_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
