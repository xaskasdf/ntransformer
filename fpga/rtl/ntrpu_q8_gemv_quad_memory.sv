// Isolated four-MAC candidate; original dual wrapper remains unchanged.
// One AXI read port loads ABI1 activations, then layout2 weight tiles.
// Job descriptors are captured at acceptance. Buffers must remain immutable
// until completion. No tile is admitted until the entire vector is validated.
// Errors drain the activation loader before cancelling the private GEMV cache.
// Reset must also reset/quiesce the AXI peer. All ports share clk.
module ntrpu_q8_gemv_quad_memory #(
    parameter integer WATCHDOG_CYCLES = 65536
) (
    input logic clk, rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [1:0] job_layout,
    input logic [5:0] job_rows,
    input logic [8:0] job_k_blocks,
    input logic [31:0] job_sequence, job_epoch,
    input logic [63:0] job_region_base, job_region_bytes,
    input logic [63:0] job_activation_addr, job_activation_region_base, job_activation_region_bytes,
    input logic [31:0] job_activation_epoch,
    input logic tile_valid,
    output logic tile_ready,
    input logic [63:0] tile_addr,
    input logic [8:0] tile_k_base,
    input logic [5:0] tile_k_count,
    input logic [31:0] tile_tag, tile_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic result_valid,
    input logic result_ready,
    output logic [63:0] result_data,
    output logic [4:0] result_row,
    output logic result_last,
    output logic [31:0] result_sequence, result_epoch,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
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
    typedef enum logic [2:0] {IDLE, START_CORE, ACT_START, LOADING, COMPUTE, STOP_CORE, RESPONSE} state_t;
    state_t state;
    logic [1:0] layout;
    logic [5:0] rows;
    logic [8:0] blocks;
    logic [63:0] region_base, region_bytes, activation_addr, activation_base, activation_bytes;
    logic [31:0] activation_job_epoch;
    logic core_job_ready, core_tile_ready, core_cancel_ready, core_done_valid, core_done_ready, core_result_valid;
    logic [3:0] core_status;
    logic [31:0] core_sequence, core_epoch;
    logic loader_cmd_ready, loader_cancel_ready, loader_done_valid;
    logic [3:0] loader_status;
    logic [8:0] loader_blocks;
    logic [31:0] loader_sequence, loader_epoch;
    logic activation_valid, activation_ready;
    logic [7:0] activation_index;
    logic [255:0] activation_values;
    logic [63:0] activation_scale;
    logic [31:0] activation_epoch;
    logic select_activation;
    logic [63:0] araddr [0:1];
    logic [7:0] arlen [0:1];
    logic [2:0] arsize [0:1], arprot [0:1];
    logic [1:0] arburst [0:1];
    logic [3:0] arid [0:1], arcache [0:1];
    logic [1:0] arlock, arvalid, rready;

    assign job_ready=state==IDLE;
    assign tile_ready=state==COMPUTE && core_tile_ready;
    assign result_valid=state==COMPUTE && core_result_valid;
    assign done_valid=state==RESPONSE;
    assign cancel_ready=(state==ACT_START && !core_done_valid) ||
        (state==LOADING && loader_cancel_ready) || (state==COMPUTE && core_cancel_ready);
    assign core_done_ready=state==ACT_START || state==STOP_CORE || state==COMPUTE;
    assign select_activation=state==ACT_START || state==LOADING;
    assign m_axi_araddr=araddr[select_activation];
    assign m_axi_arlen=arlen[select_activation];
    assign m_axi_arsize=arsize[select_activation];
    assign m_axi_arprot=arprot[select_activation];
    assign m_axi_arburst=arburst[select_activation];
    assign m_axi_arid=arid[select_activation];
    assign m_axi_arcache=arcache[select_activation];
    assign m_axi_arlock=arlock[select_activation];
    assign m_axi_arvalid=arvalid[select_activation];
    assign m_axi_rready=rready[select_activation];

    ntrpu_activation_loader #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) loader (
        .clk,.rst_n,.cmd_valid(state==ACT_START && !core_done_valid && !cancel_valid),.cmd_ready(loader_cmd_ready),
        .cmd_addr(activation_addr),.cmd_region_base(activation_base),.cmd_region_bytes(activation_bytes),
        .cmd_blocks(blocks),.cmd_sequence(done_sequence),.cmd_epoch(activation_job_epoch),.expected_epoch(done_epoch),
        .cancel_valid(state==LOADING && cancel_valid),.cancel_ready(loader_cancel_ready),
        .activation_valid,.activation_ready,.activation_index,.activation_values,.activation_scale,.activation_epoch,
        .done_valid(loader_done_valid),.done_ready(state==LOADING),.done_status(loader_status),
        .done_blocks(loader_blocks),.done_sequence(loader_sequence),.done_epoch(loader_epoch),
        .m_axi_araddr(araddr[1]),.m_axi_arlen(arlen[1]),.m_axi_arsize(arsize[1]),.m_axi_arprot(arprot[1]),
        .m_axi_arburst(arburst[1]),.m_axi_arid(arid[1]),.m_axi_arcache(arcache[1]),.m_axi_arlock(arlock[1]),
        .m_axi_arvalid(arvalid[1]),.m_axi_arready(m_axi_arready && select_activation),
        .m_axi_rdata,.m_axi_rresp,.m_axi_rid,.m_axi_rlast,
        .m_axi_rvalid(m_axi_rvalid && select_activation),.m_axi_rready(rready[1])
    );
    ntrpu_q8_gemv_quad #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) core (
        .clk,.rst_n,.job_valid(state==START_CORE),.job_ready(core_job_ready),.job_layout(layout),
        .job_rows(rows),.job_k_blocks(blocks),.job_sequence(done_sequence),.job_epoch(done_epoch),
        .job_region_base(region_base),.job_region_bytes(region_bytes),
        .activation_valid,.activation_ready,.activation_index,.activation_values,.activation_scale,.activation_epoch,
        .tile_valid(state==COMPUTE && tile_valid),.tile_ready(core_tile_ready),
        .tile_addr,.tile_k_base,.tile_k_count,.tile_tag,.tile_epoch,
        .cancel_valid(state==STOP_CORE || (state==COMPUTE && cancel_valid)),.cancel_ready(core_cancel_ready),
        .result_valid(core_result_valid),.result_ready(state==COMPUTE && result_ready),
        .result_data,.result_row,.result_last,.result_sequence,.result_epoch,
        .done_valid(core_done_valid),.done_ready(core_done_ready),.done_status(core_status),
        .done_sequence(core_sequence),.done_epoch(core_epoch),
        .m_axi_araddr(araddr[0]),.m_axi_arlen(arlen[0]),.m_axi_arsize(arsize[0]),.m_axi_arprot(arprot[0]),
        .m_axi_arburst(arburst[0]),.m_axi_arid(arid[0]),.m_axi_arcache(arcache[0]),.m_axi_arlock(arlock[0]),
        .m_axi_arvalid(arvalid[0]),.m_axi_arready(m_axi_arready && !select_activation),
        .m_axi_rdata,.m_axi_rresp,.m_axi_rid,.m_axi_rlast,
        .m_axi_rvalid(m_axi_rvalid && !select_activation),.m_axi_rready(rready[0])
    );
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE; layout<=0; rows<=0; blocks<=0;
            region_base<=0; region_bytes<=0; activation_addr<=0; activation_base<=0; activation_bytes<=0;
            activation_job_epoch<=0; done_sequence<=0; done_epoch<=0; done_status<=0;
        end else case (state)
            IDLE: if (job_valid) begin
                layout<=job_layout; rows<=job_rows; blocks<=job_k_blocks;
                region_base<=job_region_base; region_bytes<=job_region_bytes;
                activation_addr<=job_activation_addr; activation_base<=job_activation_region_base;
                activation_bytes<=job_activation_region_bytes; activation_job_epoch<=job_activation_epoch;
                done_sequence<=job_sequence; done_epoch<=job_epoch; done_status<=0; state<=START_CORE;
            end
            START_CORE: if (core_job_ready) state<=ACT_START;
            ACT_START: begin
                if (core_done_valid) begin done_status<=core_status; state<=RESPONSE; end
                else if (cancel_valid) begin done_status<=7; state<=STOP_CORE; end
                else if (loader_cmd_ready) state<=LOADING;
            end
            LOADING: if (loader_done_valid) begin
                if (loader_status!=0) begin
                    done_status<=loader_status==1 || loader_status==2 || loader_status==4 ? 4'd2:loader_status;
                    state<=STOP_CORE;
                end else if (loader_blocks!=blocks || loader_sequence!=done_sequence || loader_epoch!=done_epoch) begin
                    done_status<=2; state<=STOP_CORE;
                end else state<=COMPUTE;
            end
            COMPUTE: if (core_done_valid) begin
                done_status<=core_sequence!=done_sequence || core_epoch!=done_epoch ? 4'd5:core_status;
                state<=RESPONSE;
            end
            STOP_CORE: if (core_done_valid) state<=RESPONSE;
            RESPONSE: if (done_ready) state<=IDLE;
            default: state<=IDLE;
        endcase
    end
endmodule
