// Memory-fed GEMV with private AXI result destination and B-confirmed completion.
// Output span must not overlap either registered immutable input region.
// Failure may leave destination bytes modified; only successful completion publishes.
// All interfaces share clk; reset must quiesce/reset the AXI read/write peers.
module ntrpu_q8_gemv_store #(
    parameter integer WATCHDOG_CYCLES = 65536,
    parameter bit FP32_CAPABLE = 0
) (
    input logic clk, rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [1:0] job_layout,
    input logic [5:0] job_rows,
    input logic job_output_fp32,
    input logic [8:0] job_k_blocks,
    input logic [31:0] job_sequence, job_epoch,
    input logic [63:0] job_region_base, job_region_bytes,
    input logic [63:0] job_activation_addr, job_activation_region_base, job_activation_region_bytes,
    input logic [31:0] job_activation_epoch,
    input logic [63:0] job_output_addr, job_output_region_base, job_output_region_bytes,
    input logic [31:0] job_output_epoch,
    input logic tile_valid,
    output logic tile_ready,
    input logic [63:0] tile_addr,
    input logic [8:0] tile_k_base,
    input logic [5:0] tile_k_count,
    input logic [31:0] tile_tag, tile_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
    output logic [8:0] done_bytes,
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
    output logic m_axi_rready,
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
    typedef enum logic [3:0] {IDLE, BOUNDS, CHECK, START_WRITER, PREPARE, ABORT_WRITER,
                             START_CORE, RUN, FINISH, RESPONSE} state_t;
    state_t state;
    logic [1:0] layout;
    logic [5:0] rows;
    logic output_fp32;
    logic [8:0] blocks;
    logic [31:0] activation_epoch,output_epoch;
    logic [63:0] weight_base,weight_bytes,activation_addr,activation_base,activation_bytes;
    logic [63:0] output_addr,output_base,output_bytes;
    logic [64:0] output_end,weight_end,activation_end;
    logic engine_job_ready,engine_tile_ready,engine_cancel_ready,engine_done_valid;
    logic [3:0] engine_status;
    logic [31:0] engine_sequence,engine_epoch;
    logic result_valid,result_ready,result_last;
    logic [63:0] result_data;
    logic [4:0] result_row;
    logic [31:0] result_sequence,result_epoch;
    logic writer_cmd_ready,writer_prepared,writer_cancel_ready,writer_commit_ready,writer_done_valid;
    logic [3:0] writer_status;
    logic [8:0] writer_bytes;
    logic [31:0] writer_sequence,writer_epoch;

    assign job_ready=state==IDLE;
    assign done_valid=state==RESPONSE;
    assign tile_ready=state==RUN && engine_tile_ready;
    assign cancel_ready=state==BOUNDS || state==CHECK ||
        (state==PREPARE && writer_cancel_ready) || (state==RUN && engine_cancel_ready);
    ntrpu_result_writer #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.FP32_CAPABLE(FP32_CAPABLE)) writer (
        .clk,.rst_n,.cmd_valid(state==START_WRITER),.cmd_ready(writer_cmd_ready),.prepared(writer_prepared),
        .cmd_addr(output_addr),.cmd_region_base(output_base),.cmd_region_bytes(output_bytes),.cmd_rows(rows),.cmd_output_fp32(output_fp32),
        .cmd_sequence(done_sequence),.cmd_epoch(output_epoch),.expected_epoch(done_epoch),
        .result_valid(state==RUN && result_valid),.result_ready,.result_data,.result_row,.result_last,.result_sequence,.result_epoch,
        .commit_valid(state==ABORT_WRITER || (state==RUN && engine_done_valid)),.commit_ready(writer_commit_ready),
        .commit_status(state==ABORT_WRITER ? 4'd7:engine_status),
        .commit_sequence(state==ABORT_WRITER ? done_sequence:engine_sequence),
        .commit_epoch(state==ABORT_WRITER ? done_epoch:engine_epoch),
        .cancel_valid(state==PREPARE && cancel_valid),.cancel_ready(writer_cancel_ready),
        .done_valid(writer_done_valid),.done_ready(state==PREPARE || state==ABORT_WRITER || state==FINISH),
        .done_status(writer_status),.done_bytes(writer_bytes),.done_sequence(writer_sequence),.done_epoch(writer_epoch),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,.m_axi_awcache,
        .m_axi_awlock,.m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,.m_axi_wlast,.m_axi_wvalid,
        .m_axi_wready,.m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready
    );
    // Explicit experimental build variant; the default keeps the qualified dual core.
`ifdef NTRPU_GEMV_QUAD
    ntrpu_q8_gemv_quad_memory #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) engine (
`else
    ntrpu_q8_gemv_memory #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) engine (
`endif
        .clk,.rst_n,.job_valid(state==START_CORE),.job_ready(engine_job_ready),.job_layout(layout),
        .job_rows(rows),.job_k_blocks(blocks),.job_sequence(done_sequence),.job_epoch(done_epoch),
        .job_region_base(weight_base),.job_region_bytes(weight_bytes),
        .job_activation_addr(activation_addr),.job_activation_region_base(activation_base),
        .job_activation_region_bytes(activation_bytes),.job_activation_epoch(activation_epoch),
        .tile_valid(state==RUN && tile_valid),.tile_ready(engine_tile_ready),
        .tile_addr,.tile_k_base,.tile_k_count,.tile_tag,.tile_epoch,
        .cancel_valid(state==RUN && cancel_valid),.cancel_ready(engine_cancel_ready),
        .result_valid,.result_ready(state==RUN && result_ready),.result_data,.result_row,.result_last,.result_sequence,.result_epoch,
        .done_valid(engine_done_valid),.done_ready(state==RUN && writer_commit_ready),.done_status(engine_status),
        .done_sequence(engine_sequence),.done_epoch(engine_epoch),
        .m_axi_araddr,.m_axi_arlen,.m_axi_arsize,.m_axi_arprot,.m_axi_arburst,.m_axi_arid,.m_axi_arcache,.m_axi_arlock,
        .m_axi_arvalid,.m_axi_arready,.m_axi_rdata,.m_axi_rresp,.m_axi_rid,.m_axi_rlast,.m_axi_rvalid,.m_axi_rready
    );
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;layout<=0;rows<=0;output_fp32<=0;blocks<=0;activation_epoch<=0;output_epoch<=0;
            weight_base<=0;weight_bytes<=0;activation_addr<=0;activation_base<=0;activation_bytes<=0;
            output_addr<=0;output_base<=0;output_bytes<=0;output_end<=0;weight_end<=0;activation_end<=0;
            done_sequence<=0;done_epoch<=0;done_status<=0;done_bytes<=0;
        end else case(state)
            IDLE: if(job_valid) begin
                layout<=job_layout;rows<=job_rows;output_fp32<=FP32_CAPABLE && job_output_fp32;blocks<=job_k_blocks;
                done_sequence<=job_sequence;done_epoch<=job_epoch;done_status<=0;done_bytes<=0;
                weight_base<=job_region_base;weight_bytes<=job_region_bytes;
                activation_addr<=job_activation_addr;activation_base<=job_activation_region_base;
                activation_bytes<=job_activation_region_bytes;activation_epoch<=job_activation_epoch;
                output_addr<=job_output_addr;output_base<=job_output_region_base;
                output_bytes<=job_output_region_bytes;output_epoch<=job_output_epoch;state<=BOUNDS;
            end
            BOUNDS: begin
                weight_end<={1'b0,weight_base}+{1'b0,weight_bytes};
                activation_end<={1'b0,activation_base}+{1'b0,activation_bytes};
                output_end<={1'b0,output_addr}+(output_fp32 ? ((({59'b0,rows}+65'd15)>>4)<<6):((({59'b0,rows}+65'd7)>>3)<<6));
                if(cancel_valid) begin done_status<=7;state<=RESPONSE;end else state<=CHECK;
            end
            CHECK: begin
                if(cancel_valid) begin done_status<=7;state<=RESPONSE;end
                else if(output_end[64] || weight_end[64] || activation_end[64] ||
                    ({1'b0,output_addr}<weight_end && output_end>{1'b0,weight_base}) ||
                    ({1'b0,output_addr}<activation_end && output_end>{1'b0,activation_base})) begin
                    done_status<=1;state<=RESPONSE;
                end else state<=START_WRITER;
            end
            START_WRITER: if(writer_cmd_ready) state<=PREPARE;
            PREPARE: begin
                if(writer_done_valid) begin done_status<=writer_status;state<=RESPONSE;end
                else if(cancel_valid && writer_cancel_ready) state<=ABORT_WRITER;
                else if(writer_prepared) state<=START_CORE;
            end
            ABORT_WRITER: if(writer_done_valid) begin done_status<=writer_status;state<=RESPONSE;end
            START_CORE: if(engine_job_ready) state<=RUN;
            RUN: if(engine_done_valid && writer_commit_ready) state<=FINISH;
            FINISH: if(writer_done_valid) begin
                done_status<=writer_status==0 && (writer_sequence!=done_sequence || writer_epoch!=done_epoch) ? 4'd5:writer_status;
                done_bytes<=writer_bytes;state<=RESPONSE;
            end
            RESPONSE: if(done_ready) state<=IDLE;
            default: state<=IDLE;
        endcase
    end
endmodule
