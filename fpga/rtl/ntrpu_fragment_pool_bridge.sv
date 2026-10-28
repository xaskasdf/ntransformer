`timescale 1ns / 1ps
// One reservation, exclusive use of a two-channel pool. Source-clock only.
// ARM/retirement must be joint with the ingress lease and its trusted barrier.
// STOP forbids new ARM but never retracts an exposed line/command/data.
// rst_n is a coordinated cold reset of the complete transport session.
module ntrpu_fragment_pool_bridge #(
    parameter [63:0] FIRST_COOKIE = 64'd1
) (
    input wire clk, rst_n, stop,
    input wire pool_admit, pool_restart_required, pool_fault, pool_drained,
    input wire arm_valid,
    output wire arm_ready,
    input wire [63:0] arm_epoch, arm_tag, arm_destination,
    input wire arm_channel,
    output reg [63:0] epoch,
    output reg lease_held,
    input wire retire_valid,
    output wire retire_ready,
    input wire [63:0] retire_epoch, retire_tag,
    input wire line_valid,
    output wire line_ready,
    input wire [63:0] line_epoch, line_tag, line_address,
    input wire [511:0] line_data,
    output wire done_valid,
    input wire done_ready,
    output wire [63:0] done_epoch, done_tag,
    output reg [3:0] done_status,
    output reg [1:0] done_pool_status,
    output wire cmd_valid,
    input wire cmd_ready,
    output wire cmd_channel,
    output wire [63:0] cmd_addr, cmd_tag,
    output wire [7:0] cmd_len,
    output wire data_valid,
    input wire data_ready,
    output wire [511:0] data,
    input wire [1:0] cpl_valid,
    output wire [1:0] cpl_ready,
    input wire [127:0] cpl_tag,
    input wire [3:0] cpl_status,
    output reg protocol_fault, session_lost, exhausted,
    output wire session_invalid,
    output wire idle,
    output reg [2:0] phase,
    output reg [63:0] next_cookie,
    output reg [31:0] rejected_completions
);
    localparam [2:0] IDLE=0, LEASE=1, COMMAND=2, DATA=3,
                     COMPLETION=4, DRAIN=5, DONE=6, RETIRE=7;
    localparam [3:0] OK=0, STOPPED=1, BAD_LINE=2, POOL_ERROR=4, PROTOCOL_ERROR=5;
    reg [63:0] saved_tag, saved_address, active_cookie;
    reg saved_channel, seen_admit, stopped, pool_error;
    reg [511:0] saved_data;
    reg completion_seen;
    // Runtime availability loss cannot be repaired by a later high level.
    // Consumers must invalidate their session on this signal even if an
    // already exposed DONE is held: its payload must remain immutable.
    wire loss_now = pool_restart_required || (seen_admit && !pool_admit);
    assign session_invalid = session_lost || loss_now;
    wire descriptor_ok = arm_epoch > epoch && arm_destination[5:0]==0 &&
                         arm_destination[63:37]==0;
    wire admission_ok = pool_admit && !session_invalid && !pool_fault && !pool_error &&
                        !protocol_fault && !exhausted && !(|cpl_valid);
    assign arm_ready = rst_n && phase==IDLE && !lease_held && !stop &&
                       admission_ok && pool_drained && descriptor_ok;
    // An ingress line that is already VALID is irrevocable, even on STOP.
    assign line_ready = rst_n && lease_held && phase==LEASE && !session_invalid;
    assign retire_ready = rst_n && lease_held && !session_invalid && pool_drained &&
                          !(|cpl_valid) && retire_epoch==epoch && retire_tag==saved_tag &&
                          (phase==RETIRE || (phase==LEASE && (stop || stopped) && !line_valid));
    assign idle = rst_n && phase==IDLE && !lease_held;
    assign cmd_valid = rst_n && phase==COMMAND;
    assign cmd_channel = saved_channel;
    assign cmd_addr = saved_address;
    assign cmd_tag = active_cookie;
    assign cmd_len = 8'd0;
    assign data_valid = rst_n && phase==DATA;
    assign data = saved_data;
    assign done_valid = rst_n && phase==DONE;
    assign done_epoch = epoch;
    assign done_tag = saved_tag;
    // Reject/drain orphan or foreign completions, including simultaneous ones.
    // A lost session freezes acceptance until coordinated cold recovery.
    assign cpl_ready = {2{rst_n && !session_invalid}};
    wire [1:0] takes = cpl_valid & cpl_ready;
    wire matching = phase==COMPLETION && takes[saved_channel] &&
                    cpl_tag[saved_channel*64 +:64]==active_cookie;
    wire [1:0] foreign_takes = takes & ~(matching ? (2'b01 << saved_channel) : 2'b00);
    wire foreign_now = |foreign_takes;
    wire [1:0] foreign_count = {1'b0,foreign_takes[0]} + {1'b0,foreign_takes[1]};

    always @(posedge clk) begin
        if(!rst_n) begin
            phase<=IDLE;epoch<=0;lease_held<=0;saved_tag<=0;saved_address<=0;
            saved_channel<=0;seen_admit<=0;stopped<=0;pool_error<=0;
            saved_data<=0;active_cookie<=0;next_cookie<=FIRST_COOKIE;
            completion_seen<=0;done_status<=0;done_pool_status<=0;
            protocol_fault<=0;session_lost<=0;exhausted<=0;rejected_completions<=0;
        end else begin
            if(pool_admit) seen_admit<=1;
            if(loss_now) session_lost<=1;
            if(pool_fault) pool_error<=1;
            if(lease_held && stop) stopped<=1;
            if(foreign_now) begin
                protocol_fault<=1;
                if(rejected_completions > 32'hffffffff-foreign_count)
                    rejected_completions<=32'hffffffff;
                else rejected_completions<=rejected_completions+foreign_count;
            end
            if(arm_valid && arm_ready) begin
                epoch<=arm_epoch;saved_tag<=arm_tag;saved_address<=arm_destination;
                saved_channel<=arm_channel;lease_held<=1;phase<=LEASE;
                stopped<=0;completion_seen<=0;done_status<=OK;done_pool_status<=0;
            end
            if(retire_valid && retire_ready) begin lease_held<=0;phase<=IDLE;end
            // Never create a successful completion after availability loss.
            // Exposed handshakes still advance exactly once; freezing COMMAND
            // with VALID/READY high would duplicate a request after a fault.
            case(phase)
                LEASE: if(line_valid && line_ready) begin
                    if(line_epoch!=epoch || line_tag!=saved_tag || line_address!=saved_address) begin
                        protocol_fault<=1;done_status<=BAD_LINE;phase<=DONE;
                    end else if(protocol_fault || foreign_now || pool_fault || pool_error) begin
                        done_status<=PROTOCOL_ERROR;phase<=DONE;
                    end else begin
                        saved_data<=line_data;active_cookie<=next_cookie;phase<=COMMAND;
                        if(next_cookie==64'hffffffffffffffff) exhausted<=1;
                        else next_cookie<=next_cookie+1'b1;
                    end
                end
                COMMAND: if(cmd_valid && cmd_ready) phase<=DATA;
                DATA: if(data_valid && data_ready) phase<=COMPLETION;
                COMPLETION: if(matching) begin
                    done_pool_status<=cpl_status[saved_channel*2 +:2];
                    completion_seen<=1;phase<=DRAIN;
                end
                DRAIN: if(!session_invalid && pool_drained && !(|cpl_valid) && completion_seen) begin
                    done_status<=stopped || stop ? STOPPED :
                                 done_pool_status!=0 ? POOL_ERROR :
                                 protocol_fault || pool_fault || pool_error ? PROTOCOL_ERROR : OK;
                    phase<=DONE;
                end
                DONE: if(done_valid && done_ready) phase<=RETIRE;
                default:begin end
            endcase
        end
    end
endmodule
