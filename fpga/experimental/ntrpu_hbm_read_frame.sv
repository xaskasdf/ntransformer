`timescale 1ns/1ps
// One full 4-KiB response reservation. cmd_len is beats minus one (0..63).
// READ_ERROR returns no payload. Malformed framing quarantines until cold reset.
module ntrpu_hbm_read_frame (
    input wire clk,rst_n,admit,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [63:0] cmd_addr,cmd_tag,
    input wire [7:0] cmd_len,
    output reg data_valid,
    input wire data_ready,
    output reg [511:0] data,
    output wire [63:0] data_tag,
    output reg data_last,
    output wire cpl_valid,
    input wire cpl_ready,
    output wire [63:0] cpl_tag,
    output wire [1:0] cpl_status,
    output wire [12:0] cpl_bytes,
    output wire idle,
    output reg fault,
    output wire arvalid,
    input wire arready,
    output wire [33:0] araddr,
    output wire [7:0] arlen,
    input wire rvalid,
    output wire rready,
    input wire [511:0] rdata,
    input wire [3:0] rid,
    input wire [1:0] rresp,
    input wire rlast
);
    localparam [2:0] IDLE=0,ADDRESS=1,RECEIVE=2,PUBLISH=3,COMPLETE=4,QUARANTINE=5;
    reg [2:0] state;
    reg [63:0] tag;
    reg [33:0] address;
    reg [6:0] length,received,loaded,delivered;
    reg [1:0] status;
    // Shallow, wide storage; no reset on the payload array.
    (* ram_style="distributed" *) reg [511:0] memory [0:63];
    wire [8:0] command_beats={1'b0,cmd_len}+9'd1;
    wire [64:0] command_end={1'b0,cmd_addr}+({56'b0,command_beats}<<6)-65'd1;
    wire command_bad=command_beats>64 || |cmd_addr[5:0] || |cmd_addr[63:34] ||
                     |command_end[64:34] || command_end[63:12]!=cmd_addr[63:12];
    wire bad_response=rvalid && (state!=RECEIVE || rid!=0 || rresp==2'b01 ||
                                received>=length || rlast!=(received+1'b1==length));
    wire store_beat=rvalid&&rready&&!bad_response;
    wire load_beat=state==PUBLISH && (!data_valid||data_ready) && loaded<length;
    assign idle=state==IDLE;
    assign cmd_ready=idle && admit && !fault && !bad_response;
    assign arvalid=state==ADDRESS;
    assign araddr=address;
    assign arlen={1'b0,length}-8'd1;
    // READY depends on registered state, not the incoming ID/VALID/response.
    // A malformed accepted beat is quarantined and never stored/published.
    assign rready=state==RECEIVE;
    assign data_tag=tag;
    assign cpl_tag=tag;
    assign cpl_valid=state==COMPLETE;
    assign cpl_status=status;
    assign cpl_bytes={delivered,6'b0};
    always @(posedge clk) begin
        if(store_beat)memory[received[5:0]]<=rdata;
        if(load_beat)data<=memory[loaded[5:0]];
    end
    always @(posedge clk) begin
        if(!rst_n)begin
            state<=IDLE;tag<=0;address<=0;length<=0;received<=0;loaded<=0;delivered<=0;
            status<=0;fault<=0;data_valid<=0;data_last<=0;
        end else begin
            if(bad_response)fault<=1;
            case(state)
                IDLE:if(cmd_valid&&cmd_ready)begin
                    tag<=cmd_tag;address<=cmd_addr[33:0];length<=command_beats[6:0];
                    received<=0;loaded<=0;delivered<=0;status<=0;
                    if(command_bad)begin status<=1;state<=COMPLETE;end
                    else state<=ADDRESS;
                end
                ADDRESS:if(arready)state<=RECEIVE;
                RECEIVE:begin
                    if(bad_response)state<=QUARANTINE;
                    else if(store_beat)begin
                        received<=received+1'b1;
                        if(rresp!=0)begin status<=2;fault<=1;end
                        if(rlast)begin
                            if(status!=0||rresp!=0)state<=COMPLETE;
                            else state<=PUBLISH;
                        end
                    end
                end
                PUBLISH:begin
                    if(data_valid&&data_ready)begin
                        delivered<=delivered+1'b1;
                        if(data_last)begin data_valid<=0;state<=COMPLETE;end
                    end
                    if(load_beat)begin
                        loaded<=loaded+1'b1;data_valid<=1;data_last<=loaded+1'b1==length;
                    end
                end
                COMPLETE:if(cpl_ready)state<=IDLE;
                default:begin end
            endcase
        end
    end
endmodule
