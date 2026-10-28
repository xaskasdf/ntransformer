`timescale 1ns/1ps
// Isolated single-clock descriptor front end; NOT wired into the R4 receiver.
// DESC = {epoch[31:0], tag[63:0], destination[63:0], source[63:0], len[7:0]}.
// Inputs are prevalidated, independent across destinations, LEN <= 63.
// reserve_can is an ATOMIC same-clock offer of bank+descriptor+result credit.
// Exactly one reserve_take commits that offer; receiver must honor it even
// if stop/fault arrives later. A readiness hint or banks_used is insufficient.
// Payload stays upstream; fetch follows command acceptance. frame_done is
// issued only after all LEN+1 input beats were accepted by the reserved sink.
// stop drains queued work. Fault quarantines unreserved queues, but does not
// withdraw a committed command/fetch. Reset requires a coordinated dead epoch.
module ntrpu_destination_dispatch #(
    parameter integer DEPTH = 8,
    parameter integer COUNT_BITS = $clog2(DEPTH+1)
) (
    input wire clk, rst_n, stop, fault_in,
    input wire [1:0] submit_valid,
    output wire [1:0] submit_ready,
    input wire [463:0] submit_desc,
    output wire [1:0] head_valid,
    output wire [463:0] head_desc,
    output wire [2*COUNT_BITS-1:0] queued,
    input wire [1:0] reserve_can,
    output reg [1:0] reserve_take,
    output wire cmd_valid,
    input wire cmd_ready,
    output wire cmd_channel,
    output wire [231:0] cmd_desc,
    output wire fetch_valid,
    input wire fetch_ready,
    output wire fetch_channel,
    output wire [231:0] fetch_desc,
    input wire frame_done_valid,
    output wire frame_done_ready,
    input wire [63:0] frame_done_tag,
    input wire [31:0] frame_done_epoch,
    input wire [8:0] frame_done_beats,
    input wire fence_valid,
    output wire fence_ready,
    input wire [63:0] fence_id,
    output wire fence_done_valid,
    input wire fence_done_ready,
    output wire [63:0] fence_done_id,
    input wire sink_drained,
    output wire fault, local_empty, drained
);
    localparam integer PTR_BITS = DEPTH<=1 ? 1 : $clog2(DEPTH);
    localparam [1:0] IDLE=0, COMMAND=1, FETCH=2, FRAME=3;
    reg [PTR_BITS-1:0] rd [0:1], wr [0:1];
    reg [COUNT_BITS-1:0] count [0:1];
    reg [1:0] state;
    reg next_channel, selected_channel, sticky_fault;
    reg [231:0] selected_desc;
    reg fence_active, fence_complete;
    reg [63:0] saved_fence;
    wire [1:0] push = submit_valid & submit_ready;
    wire [1:0] eligible = head_valid & reserve_can;
    wire done_match = frame_done_tag == selected_desc[199:136] &&
        frame_done_epoch == selected_desc[231:200] &&
        frame_done_beats == {1'b0, selected_desc[7:0]}+9'd1;

    function [PTR_BITS-1:0] advance;
        input [PTR_BITS-1:0] value;
        begin advance = value == DEPTH-1 ? 0 : value+1'b1; end
    endfunction

    assign fault = sticky_fault || fault_in;
    assign local_empty = count[0]==0 && count[1]==0 && state==IDLE;
    assign drained = rst_n && local_empty && sink_drained && !fence_active;
    assign cmd_valid = rst_n && state==COMMAND;
    assign fetch_valid = rst_n && state==FETCH;
    assign frame_done_ready = rst_n && state==FRAME;
    assign cmd_channel = selected_channel;
    assign fetch_channel = selected_channel;
    assign cmd_desc = selected_desc;
    assign fetch_desc = selected_desc;
    assign fence_ready = rst_n && !fault && !fence_active;
    assign fence_done_valid = rst_n && fence_complete;
    assign fence_done_id = saved_fence;

    genvar g;
    generate for(g=0;g<2;g=g+1) begin: queues
        // Separate 1-write/1-read arrays avoid a multi-writer RAM template.
        reg [231:0] fifo [0:DEPTH-1];
        assign submit_ready[g] = rst_n && !stop && !fault && !fence_active &&
            !fence_valid && count[g]<DEPTH;
        assign head_valid[g] = rst_n && count[g]!=0;
        assign head_desc[g*232+:232] = count[g]!=0 ? fifo[rd[g]] : 232'b0;
        assign queued[g*COUNT_BITS+:COUNT_BITS] = count[g];
        always @(posedge clk) begin
            if (!rst_n) begin
                rd[g]<=0;wr[g]<=0;count[g]<=0;
            end else begin
                if (push[g]) begin
                    fifo[wr[g]]<=submit_desc[g*232+:232];
                    wr[g]<=advance(wr[g]);
                end
                if (reserve_take[g]) rd[g]<=advance(rd[g]);
                case ({push[g],reserve_take[g]})
                    2'b10: count[g]<=count[g]+1'b1;
                    2'b01: count[g]<=count[g]-1'b1;
                    default: count[g]<=count[g];
                endcase
            end
        end
    end endgenerate

    always @* begin
        reserve_take=0;
        if (rst_n && !fault && state==IDLE) begin
            if (eligible[next_channel]) reserve_take[next_channel]=1;
            else if (eligible[!next_channel]) reserve_take[!next_channel]=1;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;next_channel<=0;selected_channel<=0;selected_desc<=0;
            sticky_fault<=0;fence_active<=0;fence_complete<=0;saved_fence<=0;
        end else begin
            if (fault_in) sticky_fault<=1;
            if (fence_valid && fence_ready) begin
                fence_active<=1;saved_fence<=fence_id;
            end
            if (fence_active && !fence_complete && local_empty && sink_drained && !fault)
                fence_complete<=1;
            if (fence_done_valid && fence_done_ready) begin
                fence_active<=0;fence_complete<=0;
            end
            case (state)
                IDLE: if (|reserve_take) begin
                    selected_channel<=reserve_take[1];
                    selected_desc<=head_desc[reserve_take[1]*232+:232];
                    next_channel<=!reserve_take[1];state<=COMMAND;
                end
                COMMAND: if (cmd_ready) state<=FETCH;
                FETCH: if (fetch_ready) state<=FRAME;
                FRAME: if (frame_done_valid) begin
                    if (done_match) state<=IDLE;
                    else sticky_fault<=1;
                end
                default: sticky_fault<=1;
            endcase
        end
    end

    initial begin
        if (DEPTH<1 || COUNT_BITS!=$clog2(DEPTH+1)) $fatal(1,"invalid dispatch depth");
    end
endmodule
