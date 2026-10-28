`timescale 1ns/1ps
// Management-domain contract engine; not yet connected to the native endpoint.
// All inputs MUST already be qualified in the independent mgmt_clk domain.
// ACK producers must observe req_valid/kind/tag and prove the requested action.
// Idle AXI pins alone are not quiescence or reset acknowledgements.
// cold_rst_n belongs to FPGA configuration lifetime, never PERST/runtime reset.
module ntrpu_endpoint_reset_sequencer #(
    parameter integer HOLD_CYCLES = 5000001,
    parameter [63:0] FIRST_TAG = 64'd1
) (
    input wire mgmt_clk, cold_rst_n, start, retry, external_hold,
    input wire endpoint_held, endpoint_available,
    input wire ack_valid,
    input wire [2:0] ack_kind,
    input wire [63:0] ack_tag,
    output reg req_valid = 0,
    output reg [2:0] req_kind = 0,
    output reg [63:0] req_tag = 0,
    output reg invalidate_req = 0, block_admission = 0,
    output reg endpoint_reset_req = 0, local_reset_req = 0,
    output reg owner_retained = 0, complete = 0, exhausted = 0,
    output reg [31:0] rejected_ack_cycles = 0,
    output wire [3:0] state_debug,
    output wire [63:0] hold_elapsed
);
    localparam [2:0] INVALIDATE=0, QUIESCE=1, LOCAL_HELD=2,
        ENDPOINT_HELD=3, ENDPOINT_READY=4, LOCAL_READY=5;
    localparam [3:0] IDLE=0, WAIT_OWNER=1, WAIT_QUIET=2,
        WAIT_LOCAL_HELD=3, WAIT_ENDPOINT_HELD=4, HOLD=5,
        WAIT_ENDPOINT_READY=6, WAIT_LOCAL_READY=7, COMPLETE=8, LOCKED=9;
    localparam integer CW=$clog2(HOLD_CYCLES+1);
    reg [3:0] state=IDLE;
    reg [CW-1:0] count=0;
    reg [63:0] next_tag=FIRST_TAG;
    reg tags_empty=0;
    wire matching_ack=req_valid && ack_valid && ack_kind==req_kind && ack_tag==req_tag;
    assign state_debug=state;
    assign hold_elapsed=count;

    // Tags never wrap within a configuration. Exhaustion preserves current
    // reset outputs; it cannot reset a non-quiesced interface to escape a wait.
    task issue(input [2:0] kind, input [3:0] destination);
        begin
            if(tags_empty) begin
                req_valid<=0; complete<=0; exhausted<=1; state<=LOCKED;
            end else begin
                req_valid<=1; req_kind<=kind; req_tag<=next_tag; state<=destination;
                if(&next_tag) tags_empty<=1;
                else next_tag<=next_tag+1'b1;
            end
        end
    endtask

    always @(posedge mgmt_clk) begin
        if(!cold_rst_n) begin
            state<=IDLE; count<=0; next_tag<=FIRST_TAG; tags_empty<=0;
            req_valid<=0; req_kind<=0; req_tag<=0; invalidate_req<=0;
            block_admission<=0; endpoint_reset_req<=0; local_reset_req<=0;
            owner_retained<=0; complete<=0; exhausted<=0; rejected_ack_cycles<=0;
        end else begin
            if(ack_valid && !matching_ack && !(&rejected_ack_cycles))
                rejected_ack_cycles<=rejected_ack_cycles+1'b1;
            // After ownership was invalidated, a retry needs a NEW drain
            // certificate before asserting any newly released reset. Outputs
            // already asserted remain held. Admission never reopens.
            if(owner_retained && state!=LOCKED &&
               (retry || ((state==WAIT_ENDPOINT_READY || state==WAIT_LOCAL_READY || state==COMPLETE) &&
                           (external_hold || ((state==WAIT_LOCAL_READY || state==COMPLETE) && !endpoint_available))))) begin
                complete<=0; count<=0; issue(QUIESCE,WAIT_QUIET);
            end else case(state)
                IDLE: if(start) begin
                    invalidate_req<=1; block_admission<=1;
                    issue(INVALIDATE,WAIT_OWNER);
                end
                WAIT_OWNER: if(matching_ack) begin
                    owner_retained<=1; issue(QUIESCE,WAIT_QUIET);
                end
                WAIT_QUIET: if(matching_ack) begin
                    local_reset_req<=1; issue(LOCAL_HELD,WAIT_LOCAL_HELD);
                end
                WAIT_LOCAL_HELD: if(matching_ack) begin
                    endpoint_reset_req<=1; issue(ENDPOINT_HELD,WAIT_ENDPOINT_HELD);
                end
                WAIT_ENDPOINT_HELD: if(matching_ack && endpoint_held) begin
                    req_valid<=0; count<=0; state<=HOLD;
                end
                HOLD: begin
                    // The first counted interval starts AFTER the observed
                    // held ACK edge. No partial initial period is counted.
                    case(endpoint_held)
                        1'b1: begin
                            if(count<HOLD_CYCLES) count<=count+1'b1;
                            if(count>=HOLD_CYCLES-1 && !external_hold) begin
                                endpoint_reset_req<=0; issue(ENDPOINT_READY,WAIT_ENDPOINT_READY);
                            end
                        end
                        default: begin
                            count<=0; issue(ENDPOINT_HELD,WAIT_ENDPOINT_HELD);
                        end
                    endcase
                end
                WAIT_ENDPOINT_READY: if(matching_ack && endpoint_available && !endpoint_held) begin
                    local_reset_req<=0; issue(LOCAL_READY,WAIT_LOCAL_READY);
                end
                WAIT_LOCAL_READY: if(matching_ack && endpoint_available && !endpoint_held) begin
                    req_valid<=0; complete<=1; state<=COMPLETE;
                end
                COMPLETE: begin end
                LOCKED: begin end
                default: begin
                    invalidate_req<=1; block_admission<=1; req_valid<=0;
                    complete<=0; exhausted<=1; state<=LOCKED;
                end
            endcase
        end
    end
    // synthesis translate_off
    initial if(HOLD_CYCLES<2 || FIRST_TAG==0) $fatal(1,"ENDPOINT_SEQUENCE_PARAMETERS");
    // synthesis translate_on
endmodule
