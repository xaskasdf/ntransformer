// AXI Q/K/phase snapshots -> retained RoPE output RAM. HLS is an external boundary.
// Caller keeps all input regions and the original QKV/V lease until public ACK.
// Reset must reach this wrapper, both children, HLS and the AXI peer together.
// A foreign child response is never acknowledged; it requires recovery/reset.
`default_nettype none
module ntrpu_rope_input_path_r2 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_source_sequence,command_source_epoch,
    input wire logic command_source_valid,
    input wire logic [31:0] command_position,command_source_position,command_coefficient_position,
    input wire logic [63:0] command_q_address,command_q_base,command_q_bytes,
    input wire logic [63:0] command_k_address,command_k_base,command_k_bytes,
    input wire logic [63:0] command_coefficient_address,command_coefficient_base,command_coefficient_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [9:0] completion_beats,
    output logic [31:0] completion_position,completion_coefficient_position,
    output logic result_valid,
    output logic [31:0] completion_hls_status,
    output logic completion_revoked,
    input wire logic result_ce,
    input wire logic [11:0] result_address,
    output logic [31:0] result_q,
    output logic result_q_valid,
    output logic hls_start,
    input wire logic hls_done,hls_ready,
    input wire logic [31:0] hls_return,
    output logic [31:0] hls_heads,hls_position,hls_coefficient_position,
    input wire logic [1:0] hls_input_ce,
    input wire logic [11:0] hls_input_address [0:1],
    output wire [31:0] hls_input_q [0:1],
    input wire logic hls_output_ce,hls_output_we,
    input wire logic [11:0] hls_output_address,
    input wire logic [31:0] hls_output_d,
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
    output logic m_axi_rready
);
    localparam logic [3:0] OK=0;
    typedef enum logic [2:0] {IDLE,LOAD,COMPUTE,RESPONSE,DRAIN} state_t;
    state_t state;
    logic cancel_seen,revoked;
    wire loader_ready,loader_busy,loader_valid,loader_ack,loader_result;
    wire [3:0] loader_status;
    wire [63:0] loader_sequence,loader_epoch;
    wire [31:0] loader_position,loader_coefficient_position;
    wire [9:0] loader_beats;
    wire [1:0] source_ce,source_q_valid;
    wire [11:0] source_address [0:1];
    wire [31:0] source_q [0:1];
    wire owner_ready,owner_busy,owner_valid,owner_result,owner_ack;
    wire [3:0] owner_status;
    wire [31:0] owner_hls_status,owner_position;
    wire [63:0] owner_sequence,owner_epoch;
    wire [31:0] owner_q;
    wire owner_q_valid;
    wire epoch_bad=busy && completion_epoch!=expected_epoch;
    wire child_cancel=busy && (cancel_seen || cancel_valid);
    wire loader_command=state==IDLE && command_valid && !owner_busy;
    wire owner_command=state==LOAD && loader_valid;
    wire owner_match=owner_sequence==completion_sequence && owner_epoch==completion_epoch && owner_position==completion_position;
    wire owner_read=result_ce && result_valid;
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && loader_ready && !owner_busy;
    assign cancel_ready=busy && state!=DRAIN && !cancel_seen;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked || epoch_bad || (busy && cancel_valid) ||
        (completion_valid && (!owner_valid || !owner_match || !owner_result));
    assign result_valid=completion_valid && completion_status==OK && owner_valid && owner_match && owner_result && !completion_revoked;
    assign result_q_valid=owner_q_valid && result_valid;
    assign result_q=result_q_valid ? owner_q:32'd0;
    assign owner_ack=owner_match && ((state==RESPONSE && completion_ready) || state==DRAIN);
    ntrpu_rope_loader_r1 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES)) loader (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(loader_command),
        .command_ready(loader_ready),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .expected_epoch(expected_epoch),
        .command_source_sequence(command_source_sequence),
        .command_source_epoch(command_source_epoch),
        .command_source_valid(command_source_valid),
        .command_position(command_position),
        .command_source_position(command_source_position),
        .command_coefficient_position(command_coefficient_position),
        .command_q_address(command_q_address),
        .command_q_base(command_q_base),
        .command_q_bytes(command_q_bytes),
        .command_k_address(command_k_address),
        .command_k_base(command_k_base),
        .command_k_bytes(command_k_bytes),
        .command_coefficient_address(command_coefficient_address),
        .command_coefficient_base(command_coefficient_base),
        .command_coefficient_bytes(command_coefficient_bytes),
        .cancel_valid(child_cancel),
        .cancel_ready(),
        .busy(loader_busy),
        .completion_valid(loader_valid),
        .completion_ready(loader_ack),
        .completion_status(loader_status),
        .completion_sequence(loader_sequence),
        .completion_epoch(loader_epoch),
        .completion_beats(loader_beats),
        .completion_position(loader_position),
        .completion_coefficient_position(loader_coefficient_position),
        .result_valid(loader_result),
        .qk_ce(source_ce[0]),
        .coefficient_ce(source_ce[1]),
        .qk_address(source_address[0]),
        .coefficient_address(source_address[1]),
        .qk_q(source_q[0]),
        .coefficient_q(source_q[1]),
        .qk_q_valid(source_q_valid[0]),
        .coefficient_q_valid(source_q_valid[1]),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arid(m_axi_arid),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arlock(m_axi_arlock),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );
    ntrpu_rope_owner_r2 #(.WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) owner (
        .clk,.rst_n,.command_valid(owner_command),.command_ready(owner_ready),
        .command_sequence(completion_sequence),.command_epoch(completion_epoch),.expected_epoch,
        .command_position(completion_position),.cancel_valid(child_cancel),.cancel_ready(),.busy(owner_busy),
        .source_completion_valid(loader_valid),.source_completion_ready(loader_ack),.source_completion_status(loader_status),
        .source_sequence(loader_sequence),.source_epoch(loader_epoch),.source_position(loader_position),
        .source_coefficient_position(loader_coefficient_position),.source_result_valid(loader_result),
        .source_ce,.source_address,.source_q,.source_q_valid,
        .hls_start,.hls_done,.hls_ready,.hls_return,.hls_heads,.hls_position,.hls_coefficient_position,
        .hls_input_ce,.hls_input_address,.hls_input_q,.hls_output_ce,.hls_output_we,.hls_output_address,.hls_output_d,
        .completion_valid(owner_valid),.completion_ready(owner_ack),.completion_status(owner_status),
        .completion_hls_status(owner_hls_status),.completion_sequence(owner_sequence),.completion_epoch(owner_epoch),
        .completion_position(owner_position),.result_valid(owner_result),
        .result_ce(owner_read),.result_address,.result_q(owner_q),.result_q_valid(owner_q_valid)
    );
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;cancel_seen<=0;revoked<=0;
            completion_status<=0;completion_sequence<=0;completion_epoch<=0;
            completion_position<=0;completion_coefficient_position<=0;completion_beats<=0;completion_hls_status<=0;
        end else begin
            if(busy)begin
                if(cancel_valid && cancel_ready)cancel_seen<=1;
                if(cancel_valid || epoch_bad ||
                    (state==RESPONSE && (!owner_valid || !owner_match || !owner_result)))revoked<=1;
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_position<=command_position;completion_coefficient_position<=command_coefficient_position;
                    completion_status<=0;completion_hls_status<=0;completion_beats<=0;
                    cancel_seen<=cancel_valid;revoked<=cancel_valid;state<=LOAD;
                end
                LOAD:if(owner_command && owner_ready)begin completion_beats<=loader_beats;state<=COMPUTE;end
                COMPUTE:if(owner_valid && owner_match)begin
                    completion_status<=owner_status;completion_hls_status<=owner_hls_status;state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=DRAIN;
                DRAIN:if(!owner_busy && !loader_busy)begin state<=IDLE;end
                default:begin state<=DRAIN;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
