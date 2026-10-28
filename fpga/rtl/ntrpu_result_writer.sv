// Private vector staging followed by AXI4 writes, at most one burst pending.
// No write before a successful commit and complete, validated result stream.
// AW and W advance independently. Failed writes may modify the private target;
// consume it only after done_status==0. done_bytes counts strobed W handshakes,
// not persistence. Cancellation before commit drains input through commit.
// Reset must also reset/quiesce the AXI peer and discard the destination.
module ntrpu_result_writer #(
    parameter integer WATCHDOG_CYCLES = 65536,
    parameter bit FP32_CAPABLE = 0
) (
    input logic clk, rst_n,
    input logic cmd_valid,
    output logic cmd_ready, prepared,
    input logic [63:0] cmd_addr, cmd_region_base, cmd_region_bytes,
    input logic [5:0] cmd_rows,
    input logic cmd_output_fp32,
    input logic [31:0] cmd_sequence, cmd_epoch, expected_epoch,
    input logic result_valid,
    output logic result_ready,
    input logic [63:0] result_data,
    input logic [4:0] result_row,
    input logic result_last,
    input logic [31:0] result_sequence, result_epoch,
    input logic commit_valid,
    output logic commit_ready,
    input logic [3:0] commit_status,
    input logic [31:0] commit_sequence, commit_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
    output logic [8:0] done_bytes,
    output logic [31:0] done_sequence, done_epoch,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize, m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid, m_axi_awcache,
    output logic m_axi_awlock, m_axi_awvalid,
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast, m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready
);
    localparam logic [3:0] OK=0, BAD_COMMAND=1, BAD_EPOCH=2, WRITE_ERROR=3,
        BAD_RESULT=5, CANCELLED=7, TIMEOUT=8;
    typedef enum logic [2:0] {IDLE, BOUNDS, CHECK, COLLECT, PLAN, SEND, ADVANCE, RESPONSE} state_t;
    state_t state;
    logic [63:0] address,region_base,region_bytes;
    logic [64:0] region_end,vector_end;
    logic [5:0] rows,received;
    logic [6:0] received_after;
    logic [2:0] total_beats,beat_index,burst_beats,sent;
    logic aw_seen, result_fire, write_fire, response_fire, progress_event, watchdog_expired;
    logic [31:0] idle_cycles;
    logic [3:0] error_now;
    logic [8:0] beat_payload_bytes;
    logic [63:0] values [0:31];
    logic output_fp32,cast_valid,cast_ready,cast_fire,cast_last;
    logic [31:0] cast_data,cast_sequence,cast_epoch;
    logic [13:0] cast_index;
    logic [1:0] cast_status;
    logic [5:0] converted;
    logic [2:0] pending,pending_after;
    logic store_enable;
    logic [4:0] store_index;
    logic [63:0] store_data;

    generate if(FP32_CAPABLE) begin : fp32_path
        ntrpu_fp64_to_fp32 converter (
            .clk,.rst_n,.s_valid(result_valid && state==COLLECT && output_fp32),.s_ready(cast_ready),
            .s_data(result_data),.s_index({9'b0,result_row}),.s_last(result_last),
            .s_sequence(result_sequence),.s_epoch(result_epoch),
            .m_valid(cast_valid),.m_ready(state==COLLECT && output_fp32),
            .m_data(cast_data),.m_status(cast_status),.m_index(cast_index),.m_last(cast_last),
            .m_sequence(cast_sequence),.m_epoch(cast_epoch)
        );
    end else begin : binary64_only
        assign cast_ready=0;assign cast_valid=0;assign cast_data=0;assign cast_status=0;
        assign cast_index=0;assign cast_last=0;assign cast_sequence=0;assign cast_epoch=0;
    end endgenerate

    function automatic logic [2:0] next_burst(input logic [63:0] addr,input logic [2:0] count);
        logic [6:0] page;
        page=7'd64-{1'b0,addr[11:6]};
        return {4'b0,count}<page ? count:page[2:0];
    endfunction
    assign cmd_ready=state==IDLE;
    assign prepared=state==COLLECT;
    assign result_ready=state==COLLECT && (!output_fp32 || cast_ready);
    // The producer may offer commit together with the final FP64 item. Do
    // not consume it until every accepted item has emerged from the pipe.
    assign cast_fire=cast_valid && state==COLLECT && output_fp32;
    assign pending_after=pending+3'(result_fire && output_fp32)-3'(cast_fire);
    assign commit_ready=state==COLLECT && (!output_fp32 || pending_after==0);
    // One physical write port: formats are mutually exclusive per command.
    // Separate indexed assignments prevent distributed RAM inference.
    assign store_enable=error_now==OK && (output_fp32 ? cast_fire && converted<32:result_fire && received<32);
    assign store_index=output_fp32 ? converted[4:0]:received[4:0];
    assign store_data=output_fp32 ? {32'b0,cast_data}:result_data;
    assign cancel_ready=(state==BOUNDS || state==CHECK || state==COLLECT) && done_status==OK;
    assign done_valid=state==RESPONSE;
    assign result_fire=result_valid && result_ready;
    assign received_after={1'b0,received}+7'(result_fire);
    assign m_axi_awaddr=address;
    assign m_axi_awlen={5'b0,burst_beats}-8'd1;
    assign m_axi_awsize=3'd6;
    assign m_axi_awburst=2'b01;
    assign m_axi_awid=0;
    assign m_axi_awcache=4'b0010;
    assign m_axi_awprot=0;
    assign m_axi_awlock=0;
    assign m_axi_awvalid=state==SEND && !aw_seen;
    assign m_axi_wvalid=state==SEND && sent<burst_beats;
    assign m_axi_wlast=sent+3'd1==burst_beats;
    assign m_axi_bready=state==SEND && aw_seen && sent==burst_beats;
    assign write_fire=m_axi_wvalid && m_axi_wready;
    assign response_fire=m_axi_bvalid && m_axi_bready;
    assign progress_event=(m_axi_awvalid && m_axi_awready) || write_fire || response_fire;
    assign watchdog_expired=state==SEND && idle_cycles>=32'(WATCHDOG_CYCLES-1) && !progress_event;
    always_comb begin
        m_axi_wdata=0; m_axi_wstrb=0; beat_payload_bytes=0;
        if(output_fp32) begin
            for(int lane=0;lane<16;lane++) begin
                if(int'(beat_index)*16+lane<int'(rows)) begin
                    m_axi_wdata[lane*32+:32]=values[int'(beat_index)*16+lane][31:0];
                    m_axi_wstrb[lane*4+:4]=4'hf;
                    beat_payload_bytes+=9'd4;
                end
            end
        end else begin
            for(int lane=0;lane<8;lane++) begin
                if(int'(beat_index)*8+lane<int'(rows)) begin
                    m_axi_wdata[lane*64+:64]=values[int'(beat_index)*8+lane];
                    m_axi_wstrb[lane*8+:8]=8'hff;
                    beat_payload_bytes+=9'd8;
                end
            end
        end
    end
    always_comb begin
        error_now=done_status;
        if(done_status==OK) begin
            if(cancel_valid && cancel_ready) error_now=CANCELLED;
            else if(watchdog_expired) error_now=TIMEOUT;
            else if(response_fire && (m_axi_bresp!=0 || m_axi_bid!=0)) error_now=WRITE_ERROR;
            else if(result_fire && (received>=rows || {1'b0,result_row}!=received ||
                    result_sequence!=done_sequence || result_epoch!=done_epoch ||
                    result_last!=(received_after=={1'b0,rows}) || result_data[62:52]==11'h7ff)) error_now=BAD_RESULT;
            else if(cast_fire && (cast_status!=0 || converted>=rows || cast_index!={8'b0,converted} ||
                    cast_sequence!=done_sequence || cast_epoch!=done_epoch ||
                    cast_last!=(converted+6'd1==rows))) error_now=BAD_RESULT;
        end
    end
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;address<=0;region_base<=0;region_bytes<=0;region_end<=0;vector_end<=0;
            rows<=0;received<=0;total_beats<=0;beat_index<=0;burst_beats<=0;sent<=0;aw_seen<=0;
            idle_cycles<=0;done_status<=OK;done_bytes<=0;done_sequence<=0;done_epoch<=0;
            output_fp32<=0;converted<=0;pending<=0;
        end else begin
            if(store_enable) values[store_index]<=store_data;
            if(state==COLLECT) pending<=pending_after;
            if(state!=IDLE && state!=RESPONSE) done_status<=error_now;
            if(state==SEND && !progress_event) begin
                if(idle_cycles<32'(WATCHDOG_CYCLES)) idle_cycles<=idle_cycles+32'd1;
            end else idle_cycles<=0;
            if(result_fire) begin
                if(received<63) received<=received+6'd1;
            end
            if(cast_fire) begin
                if(converted<63) converted<=converted+6'd1;
            end
            if(write_fire) begin
                sent<=sent+3'd1;beat_index<=beat_index+3'd1;
                done_bytes<=done_bytes+beat_payload_bytes;
            end
            case(state)
                IDLE: if(cmd_valid) begin
                    address<=cmd_addr;region_base<=cmd_region_base;region_bytes<=cmd_region_bytes;
                    rows<=cmd_rows;received<=0;converted<=0;pending<=0;
                    output_fp32<=FP32_CAPABLE && cmd_output_fp32;
                    total_beats<=FP32_CAPABLE && cmd_output_fp32 ?
                        3'(({1'b0,cmd_rows}+7'd15)>>4):3'(({1'b0,cmd_rows}+7'd7)>>3);
                    beat_index<=0;done_status<=OK;done_bytes<=0;done_sequence<=cmd_sequence;done_epoch<=cmd_epoch;
                    if(cmd_rows==0 || cmd_rows>32 || cmd_addr[5:0]!=0 || cmd_region_bytes==0) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(cmd_epoch!=expected_epoch) begin done_status<=BAD_EPOCH;state<=RESPONSE;end
                    else state<=BOUNDS;
                end
                BOUNDS: begin
                    region_end<={1'b0,region_base}+{1'b0,region_bytes};
                    vector_end<={1'b0,address}+{56'b0,total_beats,6'b0};
                    if(error_now!=OK) state<=RESPONSE;else state<=CHECK;
                end
                CHECK: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(region_end[64] || vector_end[64] || address<region_base || vector_end>region_end) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else state<=COLLECT;
                end
                COLLECT: if(commit_valid && commit_ready) begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(commit_sequence!=done_sequence || commit_epoch!=done_epoch) begin
                        done_status<=BAD_RESULT;state<=RESPONSE;
                    end else if(commit_status!=OK) begin done_status<=commit_status;state<=RESPONSE;end
                    else if(received_after!={1'b0,rows}) begin done_status<=BAD_RESULT;state<=RESPONSE;end
                    else state<=PLAN;
                end
                PLAN: begin
                    burst_beats<=next_burst(address,total_beats-beat_index);
                    sent<=0;aw_seen<=0;state<=SEND;
                end
                SEND: begin
                    if(m_axi_awvalid && m_axi_awready) aw_seen<=1;
                    if(response_fire) begin
                        if(error_now!=OK || beat_index==total_beats) state<=RESPONSE;
                        else state<=ADVANCE;
                    end
                end
                ADVANCE: begin address<=address+{55'b0,burst_beats,6'b0};state<=PLAN;end
                RESPONSE: if(done_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
