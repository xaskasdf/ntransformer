`timescale 1ns / 1ps
// One outstanding phase certificate across clocks using the actual XPM bus
// handshake. No runtime reset port: state belongs to FPGA configuration life.
// req_generation identifies that life; req_tag must not be reused within it.
// This transports proof requests, NOT retractable commands to assert resets.
// Replacing/canceling a request suppresses its ACK but cannot retract a request
// already delivered to the peer. Missing peer ACK/clock deliberately stalls.
module ntrpu_tagged_reset_mailbox (
    input wire mgmt_clk, peer_clk,
    input wire req_valid,
    input wire [2:0] req_kind,
    input wire [63:0] req_generation, req_tag,
    output wire ack_valid,
    output wire [2:0] ack_kind,
    output wire [63:0] ack_generation, ack_tag,
    output wire busy,
    output reg [31:0] obsolete_completions=0,
    output reg peer_req_valid=0,
    output wire [2:0] peer_req_kind,
    output wire [63:0] peer_req_generation, peer_req_tag,
    input wire peer_ack_valid,
    input wire [2:0] peer_ack_kind,
    input wire [63:0] peer_ack_generation, peer_ack_tag,
    output reg [31:0] rejected_peer_ack_cycles=0
);
    localparam [1:0] IDLE=0, WAIT_ACK=1, DRAIN=2;
    reg [1:0] source_state=IDLE, peer_state=IDLE;
    reg [130:0] payload=0, peer_payload=0;
    reg issued=0, certificate=0, send=0, dest_ack=0, peer_armed=0;
    wire [130:0] request={req_generation,req_tag,req_kind};
    wire [130:0] peer_answer={peer_ack_generation,peer_ack_tag,peer_ack_kind};
    wire [130:0] received;
    wire received_ack, dest_req;
    wire current_request=req_valid && request==payload;
    assign busy=source_state!=IDLE;
    assign ack_valid=certificate && current_request;
    assign {ack_generation,ack_tag,ack_kind}=payload;
    assign {peer_req_generation,peer_req_tag,peer_req_kind}=peer_payload;

    xpm_cdc_handshake #(.WIDTH(131),.DEST_EXT_HSK(1),.DEST_SYNC_FF(3),
        .SRC_SYNC_FF(3),.INIT_SYNC_FF(1),.SIM_ASSERT_CHK(1)) transfer (
        .src_clk(mgmt_clk),.src_in(payload),.src_send(send),.src_rcv(received_ack),
        .dest_clk(peer_clk),.dest_out(received),.dest_req(dest_req),.dest_ack(dest_ack));

    always @(posedge mgmt_clk) begin
        if(!current_request) certificate<=0;
        case(source_state)
            IDLE: if(req_valid && (!issued || request!=payload) && !received_ack) begin
                payload<=request;issued<=1;certificate<=0;send<=1;source_state<=WAIT_ACK;
            end
            WAIT_ACK: if(received_ack) begin
                send<=0;source_state<=DRAIN;certificate<=current_request;
                if(!current_request && !(&obsolete_completions))
                    obsolete_completions<=obsolete_completions+1'b1;
            end
            DRAIN: if(!received_ack) source_state<=IDLE;
            default: begin end // preserve protocol state; never force src_send low
        endcase
    end
    always @(posedge peer_clk) begin
        if(peer_ack_valid && (!peer_req_valid || !peer_armed || peer_answer!=peer_payload) &&
           !(&rejected_peer_ack_cycles)) rejected_peer_ack_cycles<=rejected_peer_ack_cycles+1'b1;
        case(peer_state)
            IDLE: if(dest_req) begin
                peer_payload<=received;peer_req_valid<=1;peer_armed<=0;peer_state<=WAIT_ACK;
            end
            WAIT_ACK: begin
                // Observe ACK low AFTER exposing this request. A high ACK at
                // arrival cannot certify it even if its fields happen to match.
                if(!peer_ack_valid) peer_armed<=1;
                if(peer_armed && peer_ack_valid && peer_answer==peer_payload) begin
                    dest_ack<=1;peer_req_valid<=0;peer_state<=DRAIN;
                end
            end
            DRAIN: if(!dest_req) begin dest_ack<=0;peer_state<=IDLE;end
            default: begin end
        endcase
    end
endmodule
