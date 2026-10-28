// Sequential loader -> HLS RAM owner -> single AXI store.
// HLS remains an explicit external interface until its generated RTL is bound.
// The caller retains all source regions immutably until this completion ACK;
// command_source_* is a descriptor snapshot, not a producer-release handshake.
// All children and both AXI peers share a coordinated reset. Partial output
// writes remain private; only the successful output descriptor is publishable.
`default_nettype none
module ntrpu_residual_path_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_source_sequence,command_source_epoch,
    input wire logic command_source_valid,
    input wire logic [63:0] command_input_address,command_input_base,command_input_bytes,
    input wire logic [63:0] command_branch_address,command_branch_base,command_branch_bytes,
    input wire logic [63:0] command_residual_address,command_residual_base,command_residual_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [31:0] completion_hls_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [9:0] completion_read_beats,
    output logic [7:0] completion_residual_beats,
    output logic result_valid,
    output logic [63:0] result_residual_address,
    output logic [13:0] result_residual_bytes,
    output logic hls_start,
    input wire logic hls_done,hls_ready,
    input wire logic [31:0] hls_return,
    output logic [31:0] hls_elements,
    input wire logic [1:0] hls_input_ce,
    input wire logic [10:0] hls_input_address [0:1],
    output wire [31:0] hls_input_q [0:1],
    input wire logic hls_residual_ce,hls_residual_we,
    input wire logic [10:0] hls_residual_address,
    input wire logic [31:0] hls_residual_d,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,m_axi_arcache,
    output logic m_axi_arlock,m_axi_arvalid,
    input wire logic m_axi_arready,
    input wire logic [511:0] m_axi_rdata,
    input wire logic [1:0] m_axi_rresp,
    input wire logic [3:0] m_axi_rid,
    input wire logic m_axi_rlast,m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize,m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid,m_axi_awcache,
    output logic m_axi_awlock,m_axi_awvalid,
    input wire logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,m_axi_wvalid,
    input wire logic m_axi_wready,
    input wire logic [3:0] m_axi_bid,
    input wire logic [1:0] m_axi_bresp,
    input wire logic m_axi_bvalid,
    output logic m_axi_bready
);
    typedef enum logic [2:0] {IDLE,OWNER_COMMAND,STORE_COMMAND,WAIT_STORE,RESPONSE} state_t;
    state_t state;
    logic [63:0] sequence_reg,epoch_reg,residual_address,residual_base,residual_bytes;
    logic cancel_seen,revoked,epoch_bad,child_cancel;
    logic [3:0] fault,fault_now;
    logic loader_ready,loader_busy,loader_valid,loader_ack,loader_result;
    logic [3:0] loader_status;
    logic [63:0] loader_sequence,loader_epoch;
    logic [9:0] loader_beats;
    wire [1:0] loader_ce,loader_q_valid;
    wire [10:0] loader_address [0:1];
    wire [31:0] loader_q [0:1];
    logic owner_ready,owner_busy,owner_valid,owner_ack,owner_result;
    logic [3:0] owner_status;
    logic [31:0] owner_hls_status;
    logic [63:0] owner_sequence,owner_epoch;
    logic residual_ce,residual_q_valid;
    logic [10:0] residual_ram_address;
    logic [31:0] residual_q;
    logic store_ready,store_busy,store_valid,store_result;
    logic [3:0] store_status;
    logic [63:0] store_sequence,store_epoch,store_residual_address;
    logic [7:0] store_residual_beats;
    logic [13:0] store_residual_bytes;

    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && loader_ready && !owner_busy && !store_busy;
    assign cancel_ready=busy && !cancel_seen;
    assign child_cancel=busy && (cancel_seen || cancel_valid);
    assign epoch_bad=busy && epoch_reg!=expected_epoch;
    assign completion_valid=state==RESPONSE;
    assign result_valid=completion_valid && completion_status==0 && store_result &&
        !revoked && !epoch_bad && !cancel_valid;
    assign result_residual_address=result_valid ? store_residual_address:64'd0;
    assign result_residual_bytes=result_valid ? store_residual_bytes:14'd0;

    // Capture lifecycle and child failures before any child response is freed.
    // Same-cycle priority is epoch, cancellation, loader, owner, then store.
    always_comb begin
        fault_now=fault;
        if(busy && state!=RESPONSE && fault==0)begin
            if(epoch_bad)fault_now=2;
            else if(cancel_valid)fault_now=7;
            else if(loader_valid && loader_status!=0)fault_now=loader_status;
            else if(owner_valid && owner_status!=0)fault_now=owner_status;
            else if(store_valid && store_status!=0)fault_now=store_status;
        end
    end

    ntrpu_residual_loader_r1 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES)) loader(
        .clk,.rst_n,.command_valid(command_valid && command_ready),.command_ready(loader_ready),
        .command_sequence,.command_epoch,.expected_epoch,.command_source_sequence,.command_source_epoch,.command_source_valid,
        .command_residual_address(command_input_address),.command_residual_base(command_input_base),.command_residual_bytes(command_input_bytes),
        .command_branch_address,.command_branch_base,.command_branch_bytes,
        .cancel_valid(child_cancel),.cancel_ready(),.busy(loader_busy),
        .completion_valid(loader_valid),.completion_ready(loader_ack),.completion_status(loader_status),
        .completion_sequence(loader_sequence),.completion_epoch(loader_epoch),.completion_beats(loader_beats),.result_valid(loader_result),
        .residual_ce(loader_ce[0]),.branch_ce(loader_ce[1]),
        .residual_address(loader_address[0]),.branch_address(loader_address[1]),
        .residual_q(loader_q[0]),.branch_q(loader_q[1]),
        .residual_q_valid(loader_q_valid[0]),.branch_q_valid(loader_q_valid[1]),
        .m_axi_araddr,.m_axi_arlen,.m_axi_arsize,.m_axi_arprot,.m_axi_arburst,.m_axi_arid,.m_axi_arcache,
        .m_axi_arlock,.m_axi_arvalid,.m_axi_arready,.m_axi_rdata,.m_axi_rresp,.m_axi_rid,.m_axi_rlast,.m_axi_rvalid,.m_axi_rready
    );
    ntrpu_residual_owner_r1 #(.WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) owner(
        .clk,.rst_n,.command_valid(state==OWNER_COMMAND),.command_ready(owner_ready),
        .command_sequence(sequence_reg),.command_epoch(epoch_reg),.expected_epoch,
        .cancel_valid(child_cancel),.cancel_ready(),.busy(owner_busy),
        .source_completion_valid(loader_valid),.source_completion_ready(loader_ack),.source_completion_status(loader_status),
        .source_sequence(loader_sequence),.source_epoch(loader_epoch),.source_result_valid(loader_result),
        .source_ce(loader_ce),.source_address(loader_address),.source_q(loader_q),.source_q_valid(loader_q_valid),
        .hls_start,.hls_done,.hls_ready,.hls_return,.hls_elements,.hls_input_ce,.hls_input_address,.hls_input_q,
        .hls_residual_ce,.hls_residual_we,.hls_residual_address,.hls_residual_d,
        .completion_valid(owner_valid),.completion_ready(owner_ack),.completion_status(owner_status),.completion_hls_status(owner_hls_status),
        .completion_sequence(owner_sequence),.completion_epoch(owner_epoch),.result_valid(owner_result),
        .residual_ce,.residual_address(residual_ram_address),
        .residual_q,.residual_q_valid
    );
    ntrpu_residual_store_r1 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES)) store(
        .clk,.rst_n,.command_valid(state==STORE_COMMAND),.command_ready(store_ready),
        .command_sequence(sequence_reg),.command_epoch(epoch_reg),.expected_epoch,
        .command_residual_address(residual_address),.command_residual_base(residual_base),.command_residual_bytes(residual_bytes),
        .cancel_valid(child_cancel),.cancel_ready(),.busy(store_busy),
        .source_completion_valid(owner_valid),.source_completion_ready(owner_ack),.source_completion_status(owner_status),
        .source_sequence(owner_sequence),.source_epoch(owner_epoch),.source_result_valid(owner_result),
        .source_residual_ce(residual_ce),
        .source_residual_address(residual_ram_address),
        .source_residual_q(residual_q),.source_residual_q_valid(residual_q_valid),
        .completion_valid(store_valid),.completion_ready(state==RESPONSE && completion_ready),.completion_status(store_status),
        .completion_sequence(store_sequence),.completion_epoch(store_epoch),
        .completion_residual_beats(store_residual_beats),.result_valid(store_result),
        .result_residual_address(store_residual_address),
        .result_residual_bytes(store_residual_bytes),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,.m_axi_awcache,.m_axi_awlock,
        .m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,.m_axi_wlast,.m_axi_wvalid,.m_axi_wready,
        .m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready
    );
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;sequence_reg<=0;epoch_reg<=0;cancel_seen<=0;revoked<=0;fault<=0;
            residual_address<=0;residual_base<=0;residual_bytes<=0;
            completion_status<=0;completion_hls_status<=0;completion_sequence<=0;completion_epoch<=0;
            completion_read_beats<=0;completion_residual_beats<=0;
        end else begin
            if(busy)begin
                fault<=fault_now;
                if(cancel_valid)cancel_seen<=1;
                if(cancel_valid || epoch_bad || fault_now!=0)revoked<=1;
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    sequence_reg<=command_sequence;epoch_reg<=command_epoch;
                    residual_address<=command_residual_address;residual_base<=command_residual_base;residual_bytes<=command_residual_bytes;
                    cancel_seen<=cancel_valid;revoked<=cancel_valid || command_epoch!=expected_epoch;
                    fault<=command_epoch!=expected_epoch ? 4'd2:cancel_valid ? 4'd7:4'd0;
                    state<=OWNER_COMMAND;
                end
                OWNER_COMMAND:if(owner_ready)state<=STORE_COMMAND;
                STORE_COMMAND:if(store_ready)state<=WAIT_STORE;
                WAIT_STORE:if(store_valid)begin
                    completion_status<=fault_now!=0 ? fault_now:store_status;
                    completion_sequence<=store_sequence;completion_epoch<=store_epoch;
                    completion_hls_status<=owner_hls_status;completion_read_beats<=loader_beats;
                    completion_residual_beats<=store_residual_beats;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=IDLE;
                default:begin state<=IDLE;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
