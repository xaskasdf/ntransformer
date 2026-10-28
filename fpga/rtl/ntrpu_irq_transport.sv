// One completion notification, with separate MMIO service and MSI/MSI-X ACK.
// Both domains and their peers require coordinated reset. During reset the
// destination drains XPM handshakes without issuing IRQs; clocks must run long
// enough to drain before admission. This is not a reset/clock-loss supervisor.
module ntrpu_irq_transport (
    input logic clk,rst_n,host_clk,host_rst_n,
    input logic event_valid,event_irq,
    output logic event_ready,
    input logic [31:0] event_sequence,event_epoch,
    input logic service_valid,
    output logic service_ready,service_rejected,
    input logic [31:0] service_sequence,service_epoch,
    output logic retire_valid,
    input logic retire_ready,
    output logic [31:0] retire_sequence,retire_epoch,
    output logic usr_irq_req,
    input logic usr_irq_ack,
    output logic [31:0] irq_sequence,irq_epoch,
    // Registered observations for a supervisor; no state-machine override.
    output logic source_reset_drained=0,host_reset_drained=0,
    output logic source_operational=0,host_operational=0
);
    typedef enum logic [1:0] {S_BOOT,S_IDLE,S_ACTIVE,S_DRAIN} source_state_t;
    typedef enum logic [1:0] {H_BOOT,H_IDLE,H_ACTIVE,H_DRAIN} host_state_t;
    source_state_t source_state;
    host_state_t host_state;
    // FPGA configuration initialization only. Application reset MUST drain
    // these transactions, never withdraw src_send ahead of src_rcv (XPM S-2).
    logic [64:0] event_payload=0;
    logic [63:0] service_payload=0;
    logic event_send=0,service_send=0;
    logic [64:0] event_received,host_payload;
    logic [63:0] service_received;
    logic event_rcv,event_req,event_ack;
    logic service_rcv,service_req,service_ack,service_sent;
    logic host_armed,source_armed,irq_reg,transport_seen,service_seen;
    logic matching_service,retiring;

    assign host_armed=host_state==H_IDLE && !usr_irq_ack && !event_req && !service_req;
    assign event_ready=rst_n && source_state==S_IDLE && source_armed;
    assign service_ready=rst_n && source_state==S_ACTIVE && !service_sent;
    assign retire_valid=rst_n && retiring;
    assign retire_sequence=event_payload[31:0];
    assign retire_epoch=event_payload[63:32];
    assign irq_sequence=host_payload[31:0];
    assign irq_epoch=host_payload[63:32];
    assign usr_irq_req=host_rst_n && irq_reg;
    assign matching_service=service_req && service_received==host_payload[63:0];

    // These indicate actual protocol quiescence, not just reset-pin levels.
    // An unacknowledged send is retained until its XPM handshake drains.
    // Wait eight consecutive quiet local edges: a request already in flight
    // can arrive after reset assertion through the three-stage destination CDC.
    // Supervised use requires local periods >= 3.4 ns and mgmt period 10 ns;
    // the supervisor also enforces 64 real reset edges in BOTH domains.
    logic [3:0] source_quiet_count=0,host_quiet_count=0;
    always_ff @(posedge clk) begin
        if (!rst_n && !event_send && !service_send && !event_rcv && !service_rcv) begin
            if (source_quiet_count == 7) source_reset_drained <= 1;
            else source_quiet_count <= source_quiet_count + 1'b1;
        end else begin source_quiet_count <= 0; source_reset_drained <= 0; end
        source_operational <= rst_n && source_state != S_BOOT;
    end
    always_ff @(posedge host_clk) begin
        if (!host_rst_n && !event_req && !service_req && !event_ack && !service_ack) begin
            if (host_quiet_count == 7) host_reset_drained <= 1;
            else host_quiet_count <= host_quiet_count + 1'b1;
        end else begin host_quiet_count <= 0; host_reset_drained <= 0; end
        host_operational <= host_rst_n && host_state != H_BOOT;
    end

    xpm_cdc_handshake #(.DEST_EXT_HSK(1),.DEST_SYNC_FF(3),.SRC_SYNC_FF(3),
        .INIT_SYNC_FF(0),.SIM_ASSERT_CHK(1),.WIDTH(65)) event_cdc (
        .src_clk(clk),.src_in(event_payload),.src_send(event_send),.src_rcv(event_rcv),
        .dest_clk(host_clk),.dest_out(event_received),.dest_req(event_req),.dest_ack(event_ack));
    xpm_cdc_handshake #(.DEST_EXT_HSK(1),.DEST_SYNC_FF(3),.SRC_SYNC_FF(3),
        .INIT_SYNC_FF(0),.SIM_ASSERT_CHK(1),.WIDTH(64)) service_cdc (
        .src_clk(clk),.src_in(service_payload),.src_send(service_send),.src_rcv(service_rcv),
        .dest_clk(host_clk),.dest_out(service_received),.dest_req(service_req),.dest_ack(service_ack));
    // Repeated reset can truncate readiness. Register the source, then accept
    // a sustained high; low clears qualification without a destination clock.
    logic host_armed_registered=0;
    always_ff @(posedge host_clk) host_armed_registered<=host_armed;
    ntrpu_cdc_level_qualifier armed_cdc (
        .src_level(host_armed_registered),.dest_clk(clk),.dest_level(source_armed));

    always_ff @(posedge clk) begin
        if(event_rcv) event_send<=0;
        else if(event_valid && event_ready) begin
            event_payload<={event_irq,event_epoch,event_sequence};event_send<=1;
        end
        if(service_rcv) service_send<=0;
        else if(service_valid && service_ready &&
                {service_epoch,service_sequence}==event_payload[63:0]) begin
            service_payload<=event_payload[63:0];service_send<=1;
        end
    end

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            source_state<=S_BOOT;service_sent<=0;
            service_rejected<=0;retiring<=0;
        end else begin
            service_rejected<=0;
            case(source_state)
                S_BOOT:if(!event_rcv && !service_rcv && !event_send && !service_send && source_armed) source_state<=S_IDLE;
                S_IDLE:if(event_valid && event_ready) begin
                    service_sent<=0;source_state<=S_ACTIVE;
                end
                S_ACTIVE:begin
                    if(service_valid && service_ready) begin
                        if({service_epoch,service_sequence}==event_payload[63:0]) begin
                            service_sent<=1;
                        end else service_rejected<=1;
                    end
                    if(event_rcv) source_state<=S_DRAIN;
                end
                S_DRAIN:begin
                    if(!event_rcv && !service_rcv && !event_send && !service_send && source_armed)
                        retiring<=1;
                    if(retire_valid && retire_ready) begin retiring<=0;source_state<=S_IDLE;end
                end
                default:source_state<=S_BOOT;
            endcase
        end
    end

    always_ff @(posedge host_clk) begin
        if(!host_rst_n) begin
            host_state<=H_BOOT;host_payload<=0;irq_reg<=0;
            transport_seen<=0;service_seen<=0;
            // XPM has no reset port: drain orphaned transfers. Source keeps
            // each send/payload stable until this acknowledgement returns.
            event_ack<=event_req;service_ack<=service_req;
        end else begin
            case(host_state)
                H_BOOT:begin
                    irq_reg<=0;event_ack<=event_req;service_ack<=service_req;
                    if(!event_req && !service_req && !usr_irq_ack) host_state<=H_IDLE;
                end
                H_IDLE:begin
                    event_ack<=0;service_ack<=0;transport_seen<=0;service_seen<=0;
                    if(event_req && !usr_irq_ack) begin
                        host_payload<=event_received;irq_reg<=event_received[64];host_state<=H_ACTIVE;
                    end
                end
                H_ACTIVE:begin
                    if(irq_reg && usr_irq_ack) transport_seen<=1;
                    if(matching_service) begin service_seen<=1;service_ack<=1;end
                    if((!host_payload[64] || transport_seen || (irq_reg && usr_irq_ack)) &&
                       (service_seen || matching_service)) begin
                        irq_reg<=0;event_ack<=1;host_state<=H_DRAIN;
                    end
                end
                H_DRAIN:begin
                    if(!event_req) event_ack<=0;
                    if(!service_req) service_ack<=0;
                    if(!event_req && !service_req && !event_ack && !service_ack && !usr_irq_ack)
                        host_state<=H_IDLE;
                end
                default:host_state<=H_BOOT;
            endcase
        end
    end
endmodule

// Fail-closed level qualification, not an event/pulse synchronizer. A short
// high may be discarded; low asynchronously clears all stages. Sources must
// be registered locally. Integrators must constrain/audit these CLR pins.
module ntrpu_cdc_level_qualifier (
    input wire src_level,dest_clk,
    output wire dest_level
);
    (* ASYNC_REG = "TRUE" *) logic [2:0] level_ff=3'b000;
    always_ff @(posedge dest_clk or negedge src_level) begin
        if(!src_level) level_ff<=3'b000;
        else level_ff<={level_ff[1:0],1'b1};
    end
    assign dest_level=level_ff[2];
endmodule
