`timescale 1ns / 1ps

// Isolated, single-clock receiver for one prevalidated peer write span.
// Buffer the whole span before asserting AXI VALID; never cancel exposed AXI.
// rst_n is a SYNCHRONOUS cold/coordinated reset, not an abort mechanism.
module ntrpu_peer_rx_guard #(
    parameter integer MAX_BEATS = 4
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         stop,
    input  wire         source_stopped,
    input  wire         ingress_isolated,
    input  wire [63:0]  quiesce_epoch,
    input  wire         rearm_valid,
    output wire         rearm_ready,
    input  wire [63:0]  rearm_epoch,
    output reg  [63:0]  epoch,
    output reg          quarantined,
    output reg          lease_held,
    output wire         axi_busy,

    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [63:0]  cmd_epoch,
    input  wire [63:0]  cmd_tag,
    input  wire [63:0]  cmd_addr,
    input  wire [7:0]   cmd_len,       // AXI length: beats minus one
    input  wire         data_valid,
    output wire         data_ready,
    input  wire [511:0] data,
    input  wire         data_last,

    output reg          cpl_valid,
    input  wire         cpl_ready,
    output reg  [63:0]  cpl_epoch,
    output reg  [63:0]  cpl_tag,
    output reg  [2:0]   cpl_status,    // 0 OK, 1 STOP, 2 DESC, 3 FRAME, 4 AXI

    output wire [63:0]  m_axi_awaddr,
    output wire [7:0]   m_axi_awlen,
    output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst,
    output wire [3:0]   m_axi_awcache,
    output wire [3:0]   m_axi_awid,
    output wire [2:0]   m_axi_awprot,
    output wire         m_axi_awlock,
    output wire         m_axi_awvalid,
    input  wire         m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0]  m_axi_wstrb,
    output wire         m_axi_wlast,
    output wire         m_axi_wvalid,
    input  wire         m_axi_wready,
    input  wire [1:0]   m_axi_bresp,
    input  wire [3:0]   m_axi_bid,
    input  wire         m_axi_bvalid,
    output wire         m_axi_bready
);
    localparam [2:0] IDLE = 0, FILL = 1, BUFFERED = 2, ACTIVE = 3, FINISHED = 4;
    localparam [2:0] OK = 0, STOPPED = 1, BAD_DESC = 2, BAD_FRAME = 3, AXI_ERROR = 4;
    localparam integer INDEX_BITS = MAX_BEATS <= 1 ? 1 : $clog2(MAX_BEATS);

    reg [2:0] state;
    reg [511:0] payload [0:MAX_BEATS-1];
    reg [63:0] address;
    reg [8:0] beats, fill_count, write_count;
    reg aw_done, reported;
    wire [8:0] requested_beats = {1'b0, cmd_len} + 9'd1;
    wire [14:0] page_end = {3'b0, cmd_addr[11:0]} + {requested_beats, 6'b0};
    wire bad_descriptor = cmd_epoch != epoch || cmd_addr[5:0] != 0 ||
                          requested_beats > MAX_BEATS || page_end > 15'd4096;

    assign cmd_ready = state == IDLE && !lease_held && !quarantined && !stop;
    assign data_ready = state == FILL && !stop;
    assign axi_busy = state == ACTIVE;
    // ACKs certify the CURRENT epoch, not merely an old asserted level.
    assign rearm_ready = quarantined && !stop && !cpl_valid &&
                         (state == IDLE || state == FINISHED) &&
                         source_stopped && ingress_isolated && quiesce_epoch == epoch &&
                         rearm_epoch > epoch;

    assign m_axi_awaddr = address;
    assign m_axi_awlen = beats[7:0] - 8'd1;
    assign m_axi_awsize = 3'd6;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awcache = 4'b0000;
    assign m_axi_awid = 4'd0;
    assign m_axi_awprot = 3'b000;
    assign m_axi_awlock = 1'b0;
    assign m_axi_awvalid = state == ACTIVE && !aw_done;
    assign m_axi_wvalid = state == ACTIVE && write_count < beats;
    assign m_axi_wdata = m_axi_wvalid ? payload[write_count[INDEX_BITS-1:0]] : 512'd0;
    assign m_axi_wstrb = {64{1'b1}};
    assign m_axi_wlast = write_count == beats - 9'd1;
    assign m_axi_bready = state == ACTIVE && aw_done && write_count == beats;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            epoch <= 0;
            quarantined <= 1;
            lease_held <= 0;
            address <= 0;
            beats <= 0;
            fill_count <= 0;
            write_count <= 0;
            aw_done <= 0;
            reported <= 0;
            cpl_valid <= 0;
            cpl_epoch <= 0;
            cpl_tag <= 0;
            cpl_status <= OK;
        end else begin
            if (cpl_valid && cpl_ready) begin
                cpl_valid <= 0;
                if (state == FINISHED && !quarantined && !stop) begin
                    lease_held <= 0;
                    state <= IDLE;
                end
            end

            case (state)
                IDLE: if (cmd_valid && cmd_ready) begin
                    lease_held <= 1;
                    address <= cmd_addr;
                    beats <= requested_beats;
                    fill_count <= 0;
                    write_count <= 0;
                    aw_done <= 0;
                    reported <= bad_descriptor;
                    cpl_epoch <= cmd_epoch;
                    cpl_tag <= cmd_tag;
                    if (bad_descriptor) begin
                        state <= FINISHED;
                        quarantined <= 1;
                        cpl_valid <= 1;
                        cpl_status <= BAD_DESC;
                    end else begin
                        state <= FILL;
                    end
                end
                FILL: if (data_valid && data_ready) begin
                    if (data_last != (fill_count == beats - 9'd1)) begin
                        state <= FINISHED;
                        quarantined <= 1;
                        cpl_valid <= 1;
                        cpl_status <= BAD_FRAME;
                        reported <= 1;
                    end else begin
                        payload[fill_count[INDEX_BITS-1:0]] <= data;
                        fill_count <= fill_count + 9'd1;
                        if (data_last)
                            state <= BUFFERED;
                    end
                end
                BUFFERED: if (!stop) state <= ACTIVE;
                ACTIVE: begin
                    if (m_axi_awvalid && m_axi_awready)
                        aw_done <= 1;
                    if (m_axi_wvalid && m_axi_wready)
                        write_count <= write_count + 9'd1;
                    if (m_axi_bvalid && m_axi_bready && m_axi_bid != 0) begin
                        // Do not confuse a foreign B with the outstanding ID 0.
                        quarantined <= 1;
                        if (!reported) begin
                            cpl_valid <= 1;
                            cpl_status <= AXI_ERROR;
                            reported <= 1;
                        end
                    end
                    if (m_axi_bvalid && m_axi_bready && m_axi_bid == 0) begin
                        state <= FINISHED;
                        if (m_axi_bresp != 0)
                            quarantined <= 1;
                        if (!reported) begin
                            cpl_valid <= 1;
                            cpl_status <= m_axi_bresp == 0 ? OK : AXI_ERROR;
                            reported <= 1;
                        end
                    end
                end
                default: begin end
            endcase

            // Stop has priority over a newly formed completion, including B in
            // the same cycle. A completion already exposed is never rewritten.
            if (stop) begin
                quarantined <= 1;
                if (state == FILL || state == BUFFERED)
                    state <= FINISHED;
                if ((state == FILL || state == BUFFERED || state == ACTIVE) && !reported) begin
                    cpl_valid <= 1;
                    cpl_status <= STOPPED;
                    reported <= 1;
                end
            end

            if (rearm_valid && rearm_ready) begin
                state <= IDLE;
                epoch <= rearm_epoch;
                quarantined <= 0;
                lease_held <= 0;
                reported <= 0;
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (MAX_BEATS < 1 || MAX_BEATS > 64)
            $fatal(1, "MAX_BEATS must be in 1..64 (full 64-byte beats, one 4 KiB page)");
    end
`endif
endmodule
