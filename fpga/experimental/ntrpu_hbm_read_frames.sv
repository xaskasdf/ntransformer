`timescale 1ns/1ps
// Experimental ordered frame queue. Payload is reserved from command through CPL.
// One fetch engine overlaps the ordered publisher; cold reset alone drops debt.
module ntrpu_hbm_read_frames #(parameter integer FRAMES=2) (
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
    localparam [1:0] FREE=0,QUEUED=1,SEALED=2;
    localparam [1:0] FETCH_WAIT=0,FETCH_ADDRESS=1,FETCH_RECEIVE=2,FETCH_QUARANTINE=3;
    localparam [1:0] PUB_WAIT=0,PUB_DATA=1,PUB_CPL=2;
    reg [1:0] bank_state[0:FRAMES-1],status[0:FRAMES-1];
    reg [63:0] tags[0:FRAMES-1];
    reg [33:0] addresses[0:FRAMES-1];
    reg [6:0] lengths[0:FRAMES-1];
    reg alloc_ptr,fetch_ptr,publish_ptr;
    reg [1:0] used,fetch_state,publish_state;
    reg [6:0] received,loaded,delivered;
    reg protocol_fault;
    (* ram_style="distributed" *) reg [511:0] memory[0:64*FRAMES-1];
    wire [8:0] command_beats={1'b0,cmd_len}+9'd1;
    wire [64:0] command_end={1'b0,cmd_addr}+({56'b0,command_beats}<<6)-65'd1;
    wire command_bad=command_beats>64 || |cmd_addr[5:0] || |cmd_addr[63:34] ||
                     |command_end[64:34] || command_end[63:12]!=cmd_addr[63:12];
    wire ar_fire=arvalid&&arready;
    wire malformed=rvalid&&fetch_state==FETCH_RECEIVE&&
                   (rid!=0||rresp==2'b01||received>=lengths[fetch_ptr]||
                    rlast!=(received+1'b1==lengths[fetch_ptr]));
    wire unsolicited=rvalid&&fetch_state!=FETCH_RECEIVE&&!ar_fire;
    wire bad_response=malformed||unsolicited;
    wire reserve_fire=cmd_valid&&cmd_ready;
    wire release_frame=cpl_valid&&cpl_ready;
    wire store_beat=rvalid&&rready&&!bad_response;
    wire seal_beat=store_beat&&rlast;
    wire seal_invalid=fetch_state==FETCH_WAIT&&bank_state[fetch_ptr]==QUEUED&&status[fetch_ptr]==1;
    wire load_beat=publish_state==PUB_DATA&&(!data_valid||data_ready)&&loaded<lengths[publish_ptr];
    wire [6:0] write_index={fetch_ptr,received[5:0]};
    wire [6:0] read_index={publish_ptr,loaded[5:0]};

    assign idle=used==0;
    assign cmd_ready=rst_n&&admit&&!fault&&!bad_response&&used<FRAMES&&bank_state[alloc_ptr]==FREE;
    assign arvalid=fetch_state==FETCH_ADDRESS;
    assign araddr=addresses[fetch_ptr];
    assign arlen={1'b0,lengths[fetch_ptr]}-8'd1;
    assign rready=rst_n&&fetch_state==FETCH_RECEIVE&&!protocol_fault;
    assign data_tag=tags[publish_ptr];
    assign cpl_tag=tags[publish_ptr];
    assign cpl_status=status[publish_ptr];
    assign cpl_bytes={delivered,6'b0};
    assign cpl_valid=publish_state==PUB_CPL;

    always @(posedge clk)begin
        if(store_beat)memory[write_index]<=rdata;
        if(load_beat)data<=memory[read_index];
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            alloc_ptr<=0;fetch_ptr<=0;publish_ptr<=0;used<=0;
            fetch_state<=FETCH_WAIT;publish_state<=PUB_WAIT;
            received<=0;loaded<=0;delivered<=0;fault<=0;protocol_fault<=0;
            data_valid<=0;data_last<=0;
            for(integer i=0;i<FRAMES;i++)begin
                bank_state[i]<=FREE;status[i]<=0;tags[i]<=0;addresses[i]<=0;lengths[i]<=0;
            end
        end else begin
            if(bad_response)begin fault<=1;protocol_fault<=1;end
            case({reserve_fire,release_frame})
                2'b10:used<=used+1'b1;
                2'b01:used<=used-1'b1;
                default:begin end
            endcase
            if(reserve_fire)begin
                tags[alloc_ptr]<=cmd_tag;addresses[alloc_ptr]<=cmd_addr[33:0];
                lengths[alloc_ptr]<=command_beats[6:0];status[alloc_ptr]<=command_bad?1:0;
                bank_state[alloc_ptr]<=QUEUED;alloc_ptr<=FRAMES==1?1'b0:!alloc_ptr;
            end
            case(fetch_state)
                FETCH_WAIT:if(bank_state[fetch_ptr]==QUEUED)begin
                    if(status[fetch_ptr]==1)begin
                        bank_state[fetch_ptr]<=SEALED;fetch_ptr<=FRAMES==1?1'b0:!fetch_ptr;
                    end else if(!protocol_fault&&!bad_response)begin
                        received<=0;fetch_state<=FETCH_ADDRESS;
                    end
                end
                FETCH_ADDRESS:if(ar_fire)begin
                    fetch_state<=protocol_fault||bad_response?FETCH_QUARANTINE:FETCH_RECEIVE;
                end
                FETCH_RECEIVE:begin
                    if(bad_response)fetch_state<=FETCH_QUARANTINE;
                    else if(store_beat)begin
                        received<=received+1'b1;
                        if(rresp!=0)begin status[fetch_ptr]<=2;fault<=1;end
                        if(rlast)begin
                            bank_state[fetch_ptr]<=SEALED;fetch_ptr<=FRAMES==1?1'b0:!fetch_ptr;
                            fetch_state<=FETCH_WAIT;
                        end
                    end
                end
                default:begin end
            endcase
            case(publish_state)
                PUB_WAIT:if(bank_state[publish_ptr]==SEALED)begin
                    loaded<=0;delivered<=0;
                    publish_state<=status[publish_ptr]==0?PUB_DATA:PUB_CPL;
                end
                PUB_DATA:begin
                    if(data_valid&&data_ready)begin
                        delivered<=delivered+1'b1;
                        if(data_last)begin data_valid<=0;publish_state<=PUB_CPL;end
                    end
                    if(load_beat)begin
                        loaded<=loaded+1'b1;data_valid<=1;data_last<=loaded+1'b1==lengths[publish_ptr];
                    end
                end
                PUB_CPL:if(release_frame)begin
                    bank_state[publish_ptr]<=FREE;publish_ptr<=FRAMES==1?1'b0:!publish_ptr;
                    publish_state<=PUB_WAIT;
                end
                default:begin end
            endcase
        end
    end
    initial if(FRAMES!=1&&FRAMES!=2)$error("FRAMES must be 1 or 2");
endmodule
