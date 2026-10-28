`timescale 1ns / 1ps
// Transport reset guard. Management clock must remain present independently.
// cold_rst_n is CONFIGURATION lifetime only, never PERST/application reset.
// peers_held/ready and arm are management-domain contractual acknowledgements.
// source_invalid_ack is source-domain: ownership has latched invalidation.
// Admission outputs authorize NEW transactions; never AND them into held VALID.
module ntrpu_host_loss_guard #(
    parameter integer CLOCK_TIMEOUT_CYCLES=64,
    parameter integer HOLD_CYCLES=64,
    parameter integer CLEAR_CYCLES=8
) (
    input wire mgmt_clk,host_clk,source_clk,cold_rst_n,
    input wire link_up_async,endpoint_reset_async,
    input wire arm,peers_held,peers_ready,
    input wire source_invalid_ack,source_fault,
    output wire host_reset_n,source_reset_n,peers_reset_req,
    output wire host_admit,source_admit,host_alive,source_alive,
    output reg source_invalidate=0,
    output reg loss_mgmt=0,
    output reg [4:0] causes=0,
    output reg [63:0] first_loss_tick=0,
    output wire waiting_ack,
    output reg running=0,
    output wire [3:0] state_debug
);
    localparam [3:0] BOOT_CLEAR=0,BOOT_HOLD=1,BOOT_RELEASE=2,RUN=3,
        WAIT_ACK=4,FAULT_CLEAR=5,FAULT_HOLD=6,FAULT_RELEASE=7,FAULTED=8;
    localparam integer CW=$clog2(CLEAR_CYCLES+1);
    reg [3:0] state=BOOT_CLEAR;
    reg [CW-1:0] clear_count=0;
    reg [63:0] ticks=0;
    reg hold_reset=1,clear_counts=1,block_admission=1,watching=0;
    reg invalidate_req=0,ack_retained=0,fault_retained=0;
    wire request_source,ack_mgmt,fault_mgmt,link_good,reset_clear;
    wire host_done,source_done,host_done_m,source_done_m,host_up_m,source_up_m;
    wire [4:0] observed_causes={fault_mgmt,!source_alive,!host_alive,!reset_clear,!link_good};
    wire transport_healthy=link_good && reset_clear && host_alive && source_alive;
    assign peers_reset_req=hold_reset;
    assign waiting_ack=state==WAIT_ACK;
    assign state_debug=state;
    // Asynchronous assertion captures even a sub-management-cycle pulse;
    // release requires three local edges. Physical minimum pulse width and
    // board reset wiring are separate implementation/board requirements.
    ntrpu_reset_release link_filter(.clk(mgmt_clk),.hold_async(!link_up_async),.released(link_good));
    ntrpu_reset_release endpoint_filter(.clk(mgmt_clk),.hold_async(endpoint_reset_async),.released(reset_clear));
    ntrpu_clock_probe #(.TIMEOUT_CYCLES(CLOCK_TIMEOUT_CYCLES)) host_probe(
        .mgmt_clk(mgmt_clk),.observed_clk(host_clk),.alive(host_alive));
    ntrpu_clock_probe #(.TIMEOUT_CYCLES(CLOCK_TIMEOUT_CYCLES)) source_probe(
        .mgmt_clk(mgmt_clk),.observed_clk(source_clk),.alive(source_alive));
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),.SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) request_cdc(
        .src_clk(1'b0),.src_in(invalidate_req),.dest_clk(source_clk),.dest_out(request_source));
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),.SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) ack_cdc(
        .src_clk(1'b0),.src_in(ack_retained),.dest_clk(mgmt_clk),.dest_out(ack_mgmt));
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(1),.SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) fault_cdc(
        .src_clk(1'b0),.src_in(fault_retained),.dest_clk(mgmt_clk),.dest_out(fault_mgmt));
    always @(posedge source_clk) begin
        if(!cold_rst_n) begin source_invalidate<=0;ack_retained<=0;fault_retained<=0;end
        else begin
            if(source_fault) fault_retained<=1;
            if(request_source || source_fault) source_invalidate<=1;
            // Only an ACK sampled AFTER an invalidation edge can authorize reset.
            if(source_invalidate && source_invalid_ack) ack_retained<=1;
        end
    end
    ntrpu_reset_hold_counter #(.HOLD_CYCLES(HOLD_CYCLES)) host_span(
        .clk(host_clk),.clear_async(clear_counts),.done(host_done));
    ntrpu_reset_hold_counter #(.HOLD_CYCLES(HOLD_CYCLES)) source_span(
        .clk(source_clk),.clear_async(clear_counts),.done(source_done));
    ntrpu_reset_release host_count_sync(.clk(mgmt_clk),.hold_async(!host_done),.released(host_done_m));
    ntrpu_reset_release source_count_sync(.clk(mgmt_clk),.hold_async(!source_done),.released(source_done_m));
    ntrpu_reset_release host_release(.clk(host_clk),.hold_async(hold_reset),.released(host_reset_n));
    ntrpu_reset_release source_release(.clk(source_clk),.hold_async(hold_reset),.released(source_reset_n));
    ntrpu_reset_release host_up_sync(.clk(mgmt_clk),.hold_async(!host_reset_n),.released(host_up_m));
    ntrpu_reset_release source_up_sync(.clk(mgmt_clk),.hold_async(!source_reset_n),.released(source_up_m));
    ntrpu_reset_release host_admission(.clk(host_clk),.hold_async(block_admission),.released(host_admit));
    ntrpu_reset_release source_admission(.clk(source_clk),.hold_async(block_admission),.released(source_admit));
    always @(posedge mgmt_clk) begin
        if(!cold_rst_n) begin
            state<=BOOT_CLEAR;clear_count<=0;ticks<=0;first_loss_tick<=0;
            hold_reset<=1;clear_counts<=1;block_admission<=1;watching<=0;
            invalidate_req<=0;loss_mgmt<=0;causes<=0;running<=0;
        end else begin
            if(!(&ticks)) ticks<=ticks+1'b1;
            if(watching && |observed_causes) begin
                causes<=causes|observed_causes;
                if(!loss_mgmt) first_loss_tick<=ticks;
                loss_mgmt<=1;invalidate_req<=1;block_admission<=1;running<=0;
            end
            case(state)
                BOOT_CLEAR,FAULT_CLEAR: begin
                    if(clear_count==CLEAR_CYCLES-1) begin
                        if(!host_done_m && !source_done_m) begin
                            clear_counts<=0;
                            state<=state==BOOT_CLEAR ? BOOT_HOLD:FAULT_HOLD;
                        end
                    end else clear_count<=clear_count+1'b1;
                end
                BOOT_HOLD: if(transport_healthy && !fault_mgmt && host_done_m && source_done_m && peers_held && arm) begin
                    hold_reset<=0;state<=BOOT_RELEASE;
                end
                BOOT_RELEASE: begin
                    if(!transport_healthy || fault_mgmt) begin
                        hold_reset<=1;clear_counts<=1;clear_count<=0;state<=BOOT_CLEAR;
                    end else if(host_up_m && source_up_m && peers_ready) begin
                        watching<=1;running<=1;block_admission<=0;state<=RUN;
                    end
                end
                RUN: if(|observed_causes) state<=WAIT_ACK;
                WAIT_ACK: if(ack_mgmt) begin
                    hold_reset<=1;clear_counts<=1;clear_count<=0;state<=FAULT_CLEAR;
                end
                FAULT_HOLD: if(transport_healthy && host_done_m && source_done_m && peers_held) begin
                    hold_reset<=0;state<=FAULT_RELEASE;
                end
                FAULT_RELEASE: begin
                    if(!transport_healthy) begin
                        hold_reset<=1;clear_counts<=1;clear_count<=0;state<=FAULT_CLEAR;
                    end else if(host_up_m && source_up_m && peers_ready) state<=FAULTED;
                end
                FAULTED: if(!transport_healthy) begin
                    hold_reset<=1;clear_counts<=1;clear_count<=0;state<=FAULT_CLEAR;
                end
                default: begin
                    // Fail closed, but never reset retained transport without ACK.
                    watching<=1;loss_mgmt<=1;invalidate_req<=1;
                    block_admission<=1;running<=0;state<=WAIT_ACK;
                end
            endcase
        end
    end
    // synthesis translate_off
    initial if(CLOCK_TIMEOUT_CYCLES<16 || HOLD_CYCLES<64 || CLEAR_CYCLES<4)
        $fatal(1,"HOST_GUARD_PARAMETERS");
    // synthesis translate_on
endmodule
