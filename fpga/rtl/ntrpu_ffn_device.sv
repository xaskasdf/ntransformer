// One-clock host-controlled FFN. Privileged control; no PCIe, DMA or CDC implied.
// rst_n must reset/quiesce the compute engine and its shared AXI memory peer.
module ntrpu_ffn_device #(parameter integer WATCHDOG_CYCLES=65536,parameter bit OUTPUT_FP32=1,EXTERNAL_RETIRE=0) (
`ifdef NTRPU_SUPERVISED_FFN
    output wire irq_source_reset_drained,irq_host_reset_drained,
    output wire irq_source_operational,irq_host_operational,
`endif
    input logic clk,rst_n,
    input logic host_clk,host_rst_n,usr_irq_ack,
    input logic [31:0] s_axil_awaddr,s_axil_wdata,s_axil_araddr,
    input logic [2:0] s_axil_awprot,s_axil_arprot,
    input logic [3:0] s_axil_wstrb,
    input logic s_axil_awvalid,s_axil_wvalid,s_axil_bready,s_axil_arvalid,s_axil_rready,
    output logic s_axil_awready,s_axil_wready,s_axil_bvalid,s_axil_arready,s_axil_rvalid,
    output logic [1:0] s_axil_bresp,s_axil_rresp,
    output logic [31:0] s_axil_rdata,
    output logic irq,
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
    logic job_valid;
    logic job_ready;
    logic [1:0] job_layout;
    logic [13:0] job_dim,job_hidden;
    logic [5:0] job_tile_k [0:2];
    logic [63:0] job_addr [0:7],job_base [0:7],job_bytes [0:7];
    logic [31:0] job_buffer_epoch [0:7];
    logic [31:0] job_sequence,job_epoch,expected_epoch;
    logic cancel_valid;
    logic cancel_ready;
    logic done_valid;
    logic done_ready;
    logic [3:0] done_status,done_child_status;
    logic [2:0] done_stages;
    logic [1:0] done_phase;
    logic [16:0] done_bytes;
    logic [17:0] done_stage_bytes;
    logic [31:0] done_sequence,done_epoch,done_hls_status;

    generate if(EXTERNAL_RETIRE) begin : with_irq
        ntrpu_ffn_irq_control #(.OUTPUT_FP32(OUTPUT_FP32)) control(.*);
    end else begin : legacy
    ntrpu_ffn_control #(.OUTPUT_FP32(OUTPUT_FP32)) control(.notify_event_ready(1'b0),.notify_service_ready(1'b0),.notify_service_rejected(1'b0),
        .notify_retire_valid(1'b0),.notify_retire_sequence(32'b0),.notify_retire_epoch(32'b0),
        .notify_event_valid(),.notify_event_irq(),.notify_service_valid(),.notify_retire_ready(),
        .notify_sequence(),.notify_epoch(),.*);
    end endgenerate
    ntrpu_ffn_memory #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.OUTPUT_FP32(OUTPUT_FP32)) core(.*);
endmodule

