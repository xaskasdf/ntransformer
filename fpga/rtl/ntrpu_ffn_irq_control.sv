// ABI 2.0 CSR plus dual-confirmation completion notification.
// Both domains and compute/host peers require coordinated reset.
module ntrpu_ffn_irq_control #(parameter bit OUTPUT_FP32=1) (
`ifdef NTRPU_SUPERVISED_FFN
    output wire irq_source_reset_drained,irq_host_reset_drained,
    output wire irq_source_operational,irq_host_operational,
`endif
    input logic clk,rst_n,host_clk,host_rst_n,usr_irq_ack,
    input logic [31:0] s_axil_awaddr,s_axil_wdata,s_axil_araddr,
    input logic [2:0] s_axil_awprot,s_axil_arprot,
    input logic [3:0] s_axil_wstrb,
    input logic s_axil_awvalid,s_axil_wvalid,s_axil_bready,s_axil_arvalid,s_axil_rready,
    output logic s_axil_awready,s_axil_wready,s_axil_bvalid,s_axil_arready,s_axil_rvalid,
    output logic [1:0] s_axil_bresp,s_axil_rresp,
    output logic [31:0] s_axil_rdata,
    output logic irq,
    output logic job_valid,
    input logic job_ready,
    output logic [1:0] job_layout,
    output logic [13:0] job_dim,job_hidden,
    output logic [5:0] job_tile_k [0:2],
    output logic [63:0] job_addr [0:7],job_base [0:7],job_bytes [0:7],
    output logic [31:0] job_buffer_epoch [0:7],
    output logic [31:0] job_sequence,job_epoch,expected_epoch,
    output logic cancel_valid,
    input logic cancel_ready,
    input logic done_valid,
    output logic done_ready,
    input logic [3:0] done_status,done_child_status,
    input logic [2:0] done_stages,
    input logic [1:0] done_phase,
    input logic [16:0] done_bytes,
    input logic [17:0] done_stage_bytes,
    input logic [31:0] done_sequence,done_epoch,done_hls_status
);
    logic notify_event_valid,notify_event_irq,notify_service_valid,notify_retire_ready;
    logic notify_event_ready,notify_service_ready,notify_service_rejected,notify_retire_valid;
    logic [31:0] notify_sequence,notify_epoch,notify_retire_sequence,notify_retire_epoch;
    logic [31:0] irq_sequence,irq_epoch;
    ntrpu_ffn_control #(.OUTPUT_FP32(OUTPUT_FP32),.EXTERNAL_RETIRE(1)) control(.irq(),.*);
    ntrpu_irq_transport transport(
`ifdef NTRPU_SUPERVISED_FFN
        .source_reset_drained(irq_source_reset_drained),.host_reset_drained(irq_host_reset_drained),
        .source_operational(irq_source_operational),.host_operational(irq_host_operational),
`else
        .source_reset_drained(),.host_reset_drained(),
        .source_operational(),.host_operational(),
`endif
        .event_valid(notify_event_valid),.event_irq(notify_event_irq),.event_ready(notify_event_ready),
        .event_sequence(notify_sequence),.event_epoch(notify_epoch),
        .service_valid(notify_service_valid),.service_ready(notify_service_ready),
        .service_rejected(notify_service_rejected),.service_sequence(notify_sequence),.service_epoch(notify_epoch),
        .retire_valid(notify_retire_valid),.retire_ready(notify_retire_ready),
        .retire_sequence(notify_retire_sequence),.retire_epoch(notify_retire_epoch),.usr_irq_req(irq),.*);
endmodule
