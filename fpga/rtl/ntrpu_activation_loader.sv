// Activation ABI1: INT8[32] at bytes0..31, FP64 scale32..39, zero40..63.
// Writes a PRIVATE consumer cache. Only successful done permits computation.
// A pending output remains valid under cancellation; drain it before completion.
// Reset must reset/quiesce the AXI peer and invalidate the consumer cache.
module ntrpu_activation_loader #(
    parameter integer WATCHDOG_CYCLES = 65536
) (
    input logic clk, rst_n,
    input logic cmd_valid,
    output logic cmd_ready,
    input logic [63:0] cmd_addr, cmd_region_base, cmd_region_bytes,
    input logic [8:0] cmd_blocks,
    input logic [31:0] cmd_sequence, cmd_epoch, expected_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic activation_valid,
    input logic activation_ready,
    output logic [7:0] activation_index,
    output logic [255:0] activation_values,
    output logic [63:0] activation_scale,
    output logic [31:0] activation_epoch,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
    output logic [8:0] done_blocks,
    output logic [31:0] done_sequence, done_epoch,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize, m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid, m_axi_arcache,
    output logic m_axi_arlock, m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready
);
    localparam logic [3:0] OK=0, BAD_COMMAND=1, BAD_EPOCH=2,
        TRANSPORT_ERROR=3, BAD_FORMAT=4, CANCELLED=7, TIMEOUT=8;
    // Exact binary64 representation of float_max/127, matching the HLS MAC.
    localparam logic [63:0] MAX_SCALE=64'h4780204070e1c387;
    typedef enum logic [2:0] {IDLE, BOUNDS, CHECK, ISSUE, READ, DRAIN, RESPONSE} state_t;
    state_t state;
    logic [63:0] address, region_base, region_bytes;
    logic [64:0] region_end, vector_end;
    logic [8:0] total_blocks, received;
    logic [6:0] chunk_blocks, chunk_received;
    logic [31:0] idle_cycles;
    logic [3:0] error_now;
    logic reader_cmd_ready, reader_done_valid, reader_valid, reader_ready, reader_last;
    logic [1:0] reader_status;
    logic [31:0] reader_bytes, chunk_bytes;
    logic [511:0] reader_data;
    logic record_bad, progress_event, watchdog_expired;

    function automatic logic [6:0] next_chunk(input logic [63:0] addr, input logic [8:0] count);
        logic [6:0] page;
        page=7'd64-{1'b0,addr[11:6]};
        return count < {2'b0,page} ? count[6:0] : page;
    endfunction
    assign cmd_ready=state==IDLE;
    assign cancel_ready=state!=IDLE && state!=RESPONSE;
    assign done_valid=state==RESPONSE;
    assign chunk_bytes={19'b0,chunk_blocks,6'b0};
    assign reader_ready=state==READ && (done_status!=OK || !activation_valid || activation_ready);
    assign progress_event=(m_axi_arvalid && m_axi_arready) || (m_axi_rvalid && m_axi_rready) ||
        (reader_valid && reader_ready) || (activation_valid && activation_ready) ||
        (state==READ && reader_done_valid);
    assign watchdog_expired=(state==ISSUE || state==READ || state==DRAIN) &&
        idle_cycles>=32'(WATCHDOG_CYCLES-1) && !progress_event;
    assign record_bad=reader_data[511:320]!=0 ||
        (reader_data[319] && reader_data[318:256]!=0) ||
        {1'b0,reader_data[318:256]}>MAX_SCALE ||
        received>=total_blocks || reader_last!=(chunk_received+7'd1==chunk_blocks);
    always_comb begin
        error_now=done_status;
        if (done_status==OK) begin
            if (cancel_valid && cancel_ready) error_now=CANCELLED;
            else if (watchdog_expired) error_now=TIMEOUT;
            else if (state==READ && reader_done_valid &&
                     (reader_status!=0 || reader_bytes!=chunk_bytes || chunk_received!=chunk_blocks)) error_now=TRANSPORT_ERROR;
            else if (state==READ && reader_valid && reader_ready && record_bad) error_now=BAD_FORMAT;
        end
    end
    ntrpu_axi_read_master reader (
        .clk,.rst_n,.cmd_valid(state==ISSUE && error_now==OK),
        .cmd_ready(reader_cmd_ready),.cmd_addr(address),.cmd_bytes(chunk_bytes),
        .done_valid(reader_done_valid),.done_ready(state==READ),.done_status(reader_status),.done_bytes(reader_bytes),
        .out_data(reader_data),.out_valid(reader_valid),.out_ready(reader_ready),.out_last(reader_last),
        .m_axi_araddr,.m_axi_arlen,.m_axi_arsize,.m_axi_arprot,.m_axi_arburst,.m_axi_arid,
        .m_axi_arcache,.m_axi_arlock,.m_axi_arvalid,.m_axi_arready,.m_axi_rdata,.m_axi_rresp,
        .m_axi_rid,.m_axi_rlast,.m_axi_rvalid,.m_axi_rready
    );
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE; address<=0; region_base<=0; region_bytes<=0; region_end<=0; vector_end<=0;
            total_blocks<=0; received<=0; chunk_blocks<=0; chunk_received<=0; idle_cycles<=0;
            activation_valid<=0; activation_index<=0; activation_values<=0; activation_scale<=0; activation_epoch<=0;
            done_status<=OK; done_blocks<=0; done_sequence<=0; done_epoch<=0;
        end else begin
            if (activation_valid && activation_ready) begin
                activation_valid<=0; done_blocks<=done_blocks+9'd1;
            end
            if (cancel_ready) done_status<=error_now;
            if (state==ISSUE || state==READ || state==DRAIN) begin
                if (progress_event) idle_cycles<=0;
                else if (idle_cycles<32'(WATCHDOG_CYCLES)) idle_cycles<=idle_cycles+32'd1;
            end else idle_cycles<=0;
            case (state)
                IDLE: if (cmd_valid) begin
                    address<=cmd_addr; region_base<=cmd_region_base; region_bytes<=cmd_region_bytes;
                    total_blocks<=cmd_blocks; received<=0; done_blocks<=0; done_status<=OK;
                    done_sequence<=cmd_sequence; done_epoch<=cmd_epoch; activation_epoch<=cmd_epoch;
                    if (cmd_blocks==0 || cmd_blocks>256 || cmd_addr[5:0]!=0 || cmd_region_bytes==0) begin
                        done_status<=BAD_COMMAND; state<=RESPONSE;
                    end else if (cmd_epoch!=expected_epoch) begin done_status<=BAD_EPOCH; state<=RESPONSE; end
                    else state<=BOUNDS;
                end
                BOUNDS: begin
                    region_end<={1'b0,region_base}+{1'b0,region_bytes};
                    vector_end<={1'b0,address}+{50'b0,total_blocks,6'b0};
                    if (error_now!=OK) state<=RESPONSE;
                    else state<=CHECK;
                end
                CHECK: begin
                    if (error_now!=OK) state<=RESPONSE;
                    else if (region_end[64] || vector_end[64] || address<region_base || vector_end>region_end) begin
                        done_status<=BAD_COMMAND; state<=RESPONSE;
                    end else begin chunk_blocks<=next_chunk(address,total_blocks); state<=ISSUE; end
                end
                ISSUE: begin
                    if (error_now!=OK) state<=RESPONSE;
                    else if (reader_cmd_ready) begin chunk_received<=0; state<=READ; end
                end
                READ: begin
                    if (reader_valid && reader_ready) begin
                        received<=received+9'd1; chunk_received<=chunk_received+7'd1;
                        if (error_now==OK) begin
                            activation_valid<=1; activation_index<=received[7:0];
                            activation_values<=reader_data[255:0]; activation_scale<=reader_data[319:256];
                        end
                    end
                    if (reader_done_valid) state<=DRAIN;
                end
                DRAIN: if (!activation_valid) begin
                    if (error_now!=OK) state<=RESPONSE;
                    else if (received==total_blocks) begin
                        if (done_blocks!=total_blocks) done_status<=BAD_FORMAT;
                        state<=RESPONSE;
                    end else begin
                        address<=address+{32'b0,chunk_bytes};
                        chunk_blocks<=next_chunk(address+{32'b0,chunk_bytes},total_blocks-received);
                        state<=ISSUE;
                    end
                end
                RESPONSE: if (done_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
