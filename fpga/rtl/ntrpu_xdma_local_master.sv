`timescale 1ns / 1ps
// One public XDMA AXI-Lite transaction at a time, in axi_aclk's domain.
// Admission may close without retracting an offered AXI request. A timeout
// raises a sticky fault but never invents B/R, cancels VALID, or permits reset.
// cold_rst_n is for FPGA configuration only, not operational recovery.
module ntrpu_xdma_local_master #(
    parameter integer WAIT_LIMIT = 65536
) (
    input wire clk, cold_rst_n, admit,
    input wire req_valid,
    output wire req_ready,
    input wire req_write,
    input wire [31:0] req_address, req_data,
    input wire [3:0] req_strobe,
    input wire [127:0] req_identity,
    output reg rsp_valid = 0,
    input wire rsp_ready,
    output reg [31:0] rsp_data = 0,
    output reg [1:0] rsp_code = 0,
    output reg [127:0] rsp_identity = 0,
    output reg rsp_timeout = 0,
    output reg timeout_latched = 0,
    output wire busy,
    output reg [31:0] m_axil_awaddr = 0,
    output wire [2:0] m_axil_awprot,
    output reg m_axil_awvalid = 0,
    input wire m_axil_awready,
    output reg [31:0] m_axil_wdata = 0,
    output reg [3:0] m_axil_wstrb = 0,
    output reg m_axil_wvalid = 0,
    input wire m_axil_wready,
    input wire [1:0] m_axil_bresp,
    input wire m_axil_bvalid,
    output wire m_axil_bready,
    output reg [31:0] m_axil_araddr = 0,
    output wire [2:0] m_axil_arprot,
    output reg m_axil_arvalid = 0,
    input wire m_axil_arready,
    input wire [31:0] m_axil_rdata,
    input wire [1:0] m_axil_rresp,
    input wire m_axil_rvalid,
    output wire m_axil_rready
);
    localparam [2:0] IDLE=0, WRITE_REQUEST=1, WRITE_RESPONSE=2,
                     READ_REQUEST=3, READ_RESPONSE=4, REPORT=5;
    localparam integer TIMER_WIDTH = WAIT_LIMIT > 1 ? $clog2(WAIT_LIMIT) : 1;
    reg [2:0] state = IDLE;
    reg [TIMER_WIDTH-1:0] elapsed = 0;
    wire active = state == WRITE_REQUEST || state == WRITE_RESPONSE ||
                  state == READ_REQUEST || state == READ_RESPONSE;
    wire expires = active && elapsed == WAIT_LIMIT-1;
    assign req_ready = cold_rst_n && admit && state == IDLE && !timeout_latched;
    assign busy = state != IDLE;
    assign m_axil_awprot = 3'b000;
    assign m_axil_arprot = 3'b000;
    assign m_axil_bready = state == WRITE_RESPONSE;
    assign m_axil_rready = state == READ_RESPONSE;

    always @(posedge clk) begin
        if (!cold_rst_n) begin
            state <= IDLE;
            elapsed <= 0;
            timeout_latched <= 0;
            rsp_valid <= 0;
            rsp_data <= 0;
            rsp_code <= 0;
            rsp_identity <= 0;
            rsp_timeout <= 0;
            m_axil_awaddr <= 0;
            m_axil_awvalid <= 0;
            m_axil_wdata <= 0;
            m_axil_wstrb <= 0;
            m_axil_wvalid <= 0;
            m_axil_araddr <= 0;
            m_axil_arvalid <= 0;
        end else begin
            if (active) begin
                if (expires) timeout_latched <= 1;
                else elapsed <= elapsed + 1'b1;
            end else elapsed <= 0;
            case (state)
                IDLE: if (req_valid && req_ready) begin
                    rsp_identity <= req_identity;
                    rsp_data <= 0;
                    rsp_code <= 0;
                    rsp_timeout <= 0;
                    if (req_write) begin
                        m_axil_awaddr <= req_address;
                        m_axil_awvalid <= 1;
                        m_axil_wdata <= req_data;
                        m_axil_wstrb <= req_strobe;
                        m_axil_wvalid <= 1;
                        state <= WRITE_REQUEST;
                    end else begin
                        m_axil_araddr <= req_address;
                        m_axil_arvalid <= 1;
                        state <= READ_REQUEST;
                    end
                end
                WRITE_REQUEST: begin
                    if (m_axil_awvalid && m_axil_awready) m_axil_awvalid <= 0;
                    if (m_axil_wvalid && m_axil_wready) m_axil_wvalid <= 0;
                    if ((!m_axil_awvalid || m_axil_awready) &&
                        (!m_axil_wvalid || m_axil_wready)) state <= WRITE_RESPONSE;
                end
                WRITE_RESPONSE: if (m_axil_bvalid && m_axil_bready) begin
                    rsp_code <= m_axil_bresp;
                    rsp_timeout <= timeout_latched || expires;
                    rsp_valid <= 1;
                    state <= REPORT;
                end
                READ_REQUEST: if (m_axil_arvalid && m_axil_arready) begin
                    m_axil_arvalid <= 0;
                    state <= READ_RESPONSE;
                end
                READ_RESPONSE: if (m_axil_rvalid && m_axil_rready) begin
                    rsp_data <= m_axil_rdata;
                    rsp_code <= m_axil_rresp;
                    rsp_timeout <= timeout_latched || expires;
                    rsp_valid <= 1;
                    state <= REPORT;
                end
                REPORT: if (rsp_valid && rsp_ready) begin
                    rsp_valid <= 0;
                    state <= IDLE;
                end
                default: timeout_latched <= 1;
            endcase
        end
    end
endmodule
