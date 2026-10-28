// One private ABI1 activation record per command. The producer must assemble
// all eight words and check HLS completion before offering cmd_valid.
// Capture every command field; no AW/W before bounds and format validation.
// Cancellation/timeouts during SEND retain AW/W until all handshakes and B.
// Only done_status==0 permits publication; reset must also quiesce the AXI peer.
module ntrpu_activation_writer #(
    parameter integer WATCHDOG_CYCLES = 65536
) (
    input logic clk, rst_n,
    input logic cmd_valid,
    output logic cmd_ready,
    input logic [63:0] cmd_addr, cmd_region_base, cmd_region_bytes,
    input logic [511:0] cmd_record,
    input logic [5:0] cmd_elements,
    input logic [31:0] cmd_sequence, cmd_epoch, expected_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
    output logic [6:0] done_bytes,
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
    localparam logic [3:0] OK=0, BAD_COMMAND=1, BAD_EPOCH=2,
        WRITE_ERROR=3, BAD_FORMAT=4, CANCELLED=7, TIMEOUT=8;
    localparam logic [63:0] MAX_SCALE=64'h4780204070e1c387;
    typedef enum logic [2:0] {IDLE, BOUNDS, CHECK, VALIDATE, LAUNCH, SEND, RESPONSE} state_t;
    state_t state;
    logic [63:0] address, region_base, region_bytes;
    logic [64:0] region_end, record_end;
    logic [511:0] record_data;
    logic [5:0] elements;
    logic aw_seen, w_seen, bad_record, bad_padding;
    logic aw_fire, w_fire, b_fire, progress_event, watchdog_expired;
    logic [31:0] idle_cycles;
    logic [3:0] error_now;

    assign cmd_ready=state==IDLE;
    assign cancel_ready=state!=IDLE && state!=RESPONSE && done_status==OK;
    assign done_valid=state==RESPONSE;
    assign m_axi_awaddr=address;
    assign m_axi_awlen=0;
    assign m_axi_awsize=3'd6;
    assign m_axi_awburst=2'b01;
    assign m_axi_awid=0;
    assign m_axi_awcache=4'b0010;
    assign m_axi_awprot=0;
    assign m_axi_awlock=0;
    assign m_axi_awvalid=state==SEND && !aw_seen;
    assign m_axi_wdata=record_data;
    assign m_axi_wstrb=64'hffffffffffffffff;
    assign m_axi_wlast=1;
    assign m_axi_wvalid=state==SEND && !w_seen;
    assign m_axi_bready=state==SEND && aw_seen && w_seen;
    assign aw_fire=m_axi_awvalid && m_axi_awready;
    assign w_fire=m_axi_wvalid && m_axi_wready;
    assign b_fire=m_axi_bvalid && m_axi_bready;
    assign progress_event=aw_fire || w_fire || b_fire;
    assign watchdog_expired=state==SEND && idle_cycles>=32'(WATCHDOG_CYCLES-1) && !progress_event;
    always_comb begin
        bad_padding=0;
        for(int lane=0;lane<32;lane++)
            if(lane>=int'(elements) && record_data[lane*8+:8]!=0) bad_padding=1;
    end
    always_comb begin
        error_now=done_status;
        if(done_status==OK) begin
            if(cancel_valid && cancel_ready) error_now=CANCELLED;
            else if(watchdog_expired) error_now=TIMEOUT;
            else if(b_fire && (m_axi_bresp!=0 || m_axi_bid!=0)) error_now=WRITE_ERROR;
        end
    end
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;address<=0;region_base<=0;region_bytes<=0;region_end<=0;record_end<=0;
            record_data<=0;elements<=0;aw_seen<=0;w_seen<=0;bad_record<=0;idle_cycles<=0;
            done_status<=OK;done_bytes<=0;done_sequence<=0;done_epoch<=0;
        end else begin
            if(state!=IDLE && state!=RESPONSE) done_status<=error_now;
            if(state==SEND && !progress_event) begin
                if(idle_cycles<32'(WATCHDOG_CYCLES)) idle_cycles<=idle_cycles+32'd1;
            end else idle_cycles<=0;
            if(aw_fire) aw_seen<=1;
            if(w_fire) begin w_seen<=1;done_bytes<=7'd64;end
            case(state)
                IDLE: if(cmd_valid) begin
                    address<=cmd_addr;region_base<=cmd_region_base;region_bytes<=cmd_region_bytes;
                    record_data<=cmd_record;elements<=cmd_elements;aw_seen<=0;w_seen<=0;
                    done_status<=OK;done_bytes<=0;done_sequence<=cmd_sequence;done_epoch<=cmd_epoch;
                    if(cmd_elements==0 || cmd_elements>32 || cmd_addr[5:0]!=0 || cmd_region_bytes==0) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(cmd_epoch!=expected_epoch) begin done_status<=BAD_EPOCH;state<=RESPONSE;end
                    else state<=BOUNDS;
                end
                BOUNDS: begin
                    region_end<={1'b0,region_base}+{1'b0,region_bytes};
                    record_end<={1'b0,address}+65'd64;
                    bad_record<=bad_padding || record_data[511:320]!=0 ||
                        (record_data[319] && record_data[318:256]!=0) ||
                        {1'b0,record_data[318:256]}>MAX_SCALE;
                    if(error_now!=OK) state<=RESPONSE;else state<=CHECK;
                end
                CHECK: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(region_end[64] || record_end[64] || address<region_base || record_end>region_end) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else state<=VALIDATE;
                end
                VALIDATE: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(bad_record) begin done_status<=BAD_FORMAT;state<=RESPONSE;end
                    else state<=LAUNCH;
                end
                LAUNCH: if(error_now!=OK) state<=RESPONSE;else state<=SEND;
                SEND: if(b_fire) state<=RESPONSE;
                RESPONSE: if(done_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
