// Reset/clock-loss supervisor connected to the actual notification transport.
// Only these two protocol peers belong to this island. Integrating AXI, XDMA
// or memory requires their own reset/drain acknowledgements as well.
module ntrpu_supervised_irq_transport (
    input wire mgmt_clk,clk,host_clk,external_reset_async,rearm,
    output wire rst_n,host_rst_n,device_admit,host_admit,
    output wire peers_reset_req,rearm_ready,running,
    output wire peers_held,peers_ready,host_alive,device_alive,
    input wire event_valid,event_irq,
    output wire event_ready,
    input wire [31:0] event_sequence,event_epoch,
    input wire service_valid,
    output wire service_ready,service_rejected,
    input wire [31:0] service_sequence,service_epoch,
    output wire retire_valid,
    input wire retire_ready,
    output wire [31:0] retire_sequence,retire_epoch,
    output wire usr_irq_req,
    input wire usr_irq_ack,
    output wire [31:0] irq_sequence,irq_epoch
);
    wire source_reset_drained,host_reset_drained;
    wire source_operational,host_operational;
    wire [3:0] local_status,mgmt_status;
    assign local_status={host_operational,source_operational,
                         host_reset_drained,source_reset_drained};
    for (genvar i=0;i<4;i++) begin : status_cdc
        xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),
            .SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) sync (
            .src_clk(1'b0),.src_in(local_status[i]),
            .dest_clk(mgmt_clk),.dest_out(mgmt_status[i]));
    end
    assign peers_held=mgmt_status[0] && mgmt_status[1];
    // Operational stays high during ACTIVE/DRAIN; event_ready alone would
    // spuriously reset every time a completion notification became busy.
    assign peers_ready=mgmt_status[2] && mgmt_status[3];
    ntrpu_reset_supervisor supervisor (
        .device_clk(clk),.host_clk(host_clk),.device_reset_n(rst_n),
        .mgmt_clk(mgmt_clk),.external_reset_async(external_reset_async),
        .rearm(rearm),.host_reset_n(host_rst_n),.*);
    wire raw_event_ready,raw_service_ready,raw_retire_valid,raw_irq;
    assign event_ready=device_admit && raw_event_ready;
    assign service_ready=device_admit && raw_service_ready;
    assign retire_valid=device_admit && raw_retire_valid;
    assign usr_irq_req=host_admit && raw_irq;
    ntrpu_irq_transport transport (
        .event_valid(event_valid && device_admit),.event_ready(raw_event_ready),
        .service_valid(service_valid && device_admit),.service_ready(raw_service_ready),
        .retire_ready(retire_ready && device_admit),.retire_valid(raw_retire_valid),
        .usr_irq_req(raw_irq),.*);
endmodule
