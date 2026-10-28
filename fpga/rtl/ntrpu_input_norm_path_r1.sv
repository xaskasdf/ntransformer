// Input/weight AXI loader -> RMSNorm-Q8 owner -> retained 4096-byte ABI1 store.
// HLS remains an explicit external interface until its generated RTL is bound.
// The caller retains all source regions immutably until this completion ACK;
// command_source_* is a descriptor snapshot, not a producer-release handshake.
// All children and both AXI peers share a coordinated reset. Partial output
// writes remain private; only the successful output descriptor is publishable.
`default_nettype none
module ntrpu_input_norm_path_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter logic [63:0] INITIAL_CHILD_TAG=64'd0
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_source_sequence,command_source_epoch,
    input wire logic command_source_valid,
    input wire logic [63:0] command_input_address,command_input_base,command_input_bytes,
    input wire logic [63:0] command_weight_address,command_weight_base,command_weight_bytes,
    input wire logic [63:0] command_output_address,command_output_base,command_output_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,tag_exhausted,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [31:0] completion_hls_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [9:0] completion_read_beats,
    output logic [6:0] completion_records,
    output logic result_valid,completion_revoked,
    output logic [63:0] result_address,
    output logic [12:0] result_bytes,
    output logic hls_start,
    input wire logic hls_done,hls_ready,
    input wire logic [31:0] hls_return,
    output logic [31:0] hls_elements,hls_epsilon_bits,
    input wire logic [1:0] hls_input_ce,
    input wire logic [10:0] hls_input_address [0:1],
    output wire [31:0] hls_input_q [0:1],
    input wire logic hls_output_ce,hls_output_we,
    input wire logic [8:0] hls_output_address,
    input wire logic [63:0] hls_output_d,
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
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,CANCELLED=7,TAGS_EXHAUSTED=9,BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,CHECK_REGION,CHECK_ALIAS,OFFER_LOADER,OFFER_OWNER,OFFER_STORE,WAIT_STORE,RESPONSE,DRAIN} state_t;
    state_t state;
    logic [63:0] addresses[0:2],bases[0:2],sizes[0:2];
    logic [64:0] ends[0:2];
    logic [64:0] selected_end,selected_extent;
    logic [1:0] check_index;
    logic [3:0] fault,fault_now,admission_status;
    logic cancel_seen,revoked,store_owned,bounds_bad,alias_bad;
    wire epoch_bad=busy && completion_epoch!=expected_epoch;
    wire child_cancel=busy && (cancel_seen || cancel_valid || fault!=OK);
    wire loader_command=state==OFFER_LOADER;
    wire owner_command=state==OFFER_OWNER;
    wire store_command=state==OFFER_STORE;
    wire loader_ready,loader_busy,loader_valid,loader_ack,loader_result;
    wire [3:0] loader_status;
    wire [63:0] loader_sequence,loader_epoch;
    wire [9:0] loader_beats;
    wire [1:0] loader_ce,loader_q_valid;
    wire [10:0] loader_address [0:1];
    wire [31:0] loader_q [0:1];
    wire owner_ready,owner_busy,owner_valid,owner_ack,owner_result;
    wire [3:0] owner_status;
    wire [31:0] owner_hls_status;
    wire [63:0] owner_sequence,owner_epoch;
    wire abi_ce,abi_q_valid;
    wire [8:0] abi_address;
    wire [63:0] abi_q;
    wire store_ready,store_busy,store_valid,store_result;
    wire [3:0] store_status;
    wire [63:0] store_sequence,store_epoch,store_address;
    wire [6:0] store_records;
    wire [12:0] store_bytes;
    wire loader_match=loader_sequence==completion_sequence && loader_epoch==completion_epoch;
    wire owner_match=owner_sequence==completion_sequence && owner_epoch==completion_epoch;
    wire store_match=store_sequence==completion_sequence && store_epoch==completion_epoch;
    wire store_ack=store_owned && store_match && ((state==RESPONSE && completion_ready) || state==DRAIN);

    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && loader_ready && !loader_busy && !owner_busy && !store_busy;
    assign cancel_ready=busy && state!=DRAIN && !cancel_seen;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked || epoch_bad || (busy && cancel_valid) ||
        (completion_valid && store_owned && (!store_match || !store_result));
    assign result_valid=completion_valid && completion_status==OK && store_owned &&
        store_valid && store_match && store_result && !completion_revoked;
    assign result_address=result_valid ? store_address:64'd0;
    assign result_bytes=result_valid ? store_bytes:13'd0;

    always_comb begin
        admission_status=OK;
        if(command_sequence!=command_source_sequence || command_epoch!=command_source_epoch || command_epoch!=expected_epoch)
            admission_status=BAD_IDENTITY;
        else if(cancel_valid)admission_status=CANCELLED;
        else if(!command_source_valid)admission_status=BAD_PEER;
        else if(tag_exhausted || INITIAL_CHILD_TAG[5:0]!=0)admission_status=TAGS_EXHAUSTED;
        selected_end={1'b0,bases[check_index]}+{1'b0,sizes[check_index]};
        selected_extent={1'b0,addresses[check_index]}+(check_index==2 ? 65'd4096:65'd8192);
        bounds_bad=addresses[check_index][5:0]!=0 || sizes[check_index]==0 || selected_end[64] ||
            selected_extent[64] || addresses[check_index]<bases[check_index] || selected_extent>selected_end;
        alias_bad={1'b0,bases[2]}<ends[check_index] && {1'b0,bases[check_index]}<ends[2];
        fault_now=fault;
        if(busy && state!=RESPONSE && state!=DRAIN && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(state==CHECK_REGION && bounds_bad)fault_now=BAD_COMMAND;
            else if(state==CHECK_ALIAS && alias_bad)fault_now=BAD_COMMAND;
            else if(loader_valid && !loader_match)fault_now=BAD_PEER;
            else if(owner_valid && !owner_match)fault_now=BAD_PEER;
            else if(store_valid && !store_match)fault_now=BAD_PEER;
            else if(loader_valid && loader_status!=OK)fault_now=loader_status;
            else if(owner_valid && owner_status!=OK)fault_now=owner_status;
            else if(store_valid && store_status!=OK)fault_now=store_status;
            else if(loader_valid && !loader_result || owner_valid && !owner_result || store_valid && !store_result)fault_now=BAD_PEER;
        end
    end

    ntrpu_residual_loader_r1 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES)) loader(
        .clk,.rst_n,.command_valid(loader_command),.command_ready(loader_ready),
        .command_sequence(completion_sequence),.command_epoch(completion_epoch),.expected_epoch,
        .command_source_sequence(completion_sequence),.command_source_epoch(completion_epoch),.command_source_valid(1'b1),
        .command_residual_address(addresses[0]),.command_residual_base(bases[0]),.command_residual_bytes(sizes[0]),
        .command_branch_address(addresses[1]),.command_branch_base(bases[1]),.command_branch_bytes(sizes[1]),
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
    ntrpu_input_norm_owner_r1 #(.WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) owner(
        .clk,.rst_n,.command_valid(owner_command),.command_ready(owner_ready),
        .command_sequence(completion_sequence),.command_epoch(completion_epoch),.expected_epoch,
        .cancel_valid(child_cancel),.cancel_ready(),.busy(owner_busy),
        .source_completion_valid(loader_valid),.source_completion_ready(loader_ack),.source_completion_status(loader_status),
        .source_sequence(loader_sequence),.source_epoch(loader_epoch),.source_result_valid(loader_result),
        .source_ce(loader_ce),.source_address(loader_address),.source_q(loader_q),.source_q_valid(loader_q_valid),
        .hls_start,.hls_done,.hls_ready,.hls_return,.hls_elements,.hls_epsilon_bits,.hls_input_ce,.hls_input_address,.hls_input_q,
        .hls_output_ce,.hls_output_we,.hls_output_address,.hls_output_d,
        .completion_valid(owner_valid),.completion_ready(owner_ack),.completion_status(owner_status),.completion_hls_status(owner_hls_status),
        .completion_sequence(owner_sequence),.completion_epoch(owner_epoch),.result_valid(owner_result),
        .result_ce(abi_ce),.result_address(abi_address),.result_q(abi_q),.result_q_valid(abi_q_valid)
    );
    ntrpu_attention_activation_store_r1 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.INITIAL_CHILD_TAG(INITIAL_CHILD_TAG)) store(
        .clk,.rst_n,.command_valid(store_command),.command_ready(store_ready),
        .command_sequence(completion_sequence),.command_epoch(completion_epoch),.expected_epoch,
        .command_address(addresses[2]),.command_region_base(bases[2]),.command_region_bytes(sizes[2]),
        .cancel_valid(child_cancel),.cancel_ready(),.busy(store_busy),.tag_exhausted,
        .source_completion_valid(owner_valid),.source_completion_ready(owner_ack),.source_completion_status(owner_status),
        .source_sequence(owner_sequence),.source_epoch(owner_epoch),.source_result_valid(owner_result),
        .source_result_ce(abi_ce),.source_result_address(abi_address),.source_result_q(abi_q),.source_result_q_valid(abi_q_valid),
        .completion_valid(store_valid),.completion_ready(store_ack),.completion_status(store_status),
        .completion_sequence(store_sequence),.completion_epoch(store_epoch),.completion_records(store_records),
        .result_valid(store_result),.result_address(store_address),.result_bytes(store_bytes),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,.m_axi_awcache,.m_axi_awlock,
        .m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,.m_axi_wlast,.m_axi_wvalid,.m_axi_wready,
        .m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready
    );
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;check_index<=0;fault<=OK;cancel_seen<=0;revoked<=0;store_owned<=0;
            completion_status<=OK;completion_sequence<=0;completion_epoch<=0;
            completion_hls_status<=0;completion_read_beats<=0;completion_records<=0;
            for(integer i=0;i<3;i++)begin addresses[i]<=0;bases[i]<=0;sizes[i]<=0;ends[i]<=0;end
        end else begin
            if(busy)begin
                if(state!=RESPONSE && state!=DRAIN)fault<=fault_now;
                if(cancel_valid && cancel_ready)cancel_seen<=1;
                if(cancel_valid || epoch_bad || fault_now!=OK ||
                    (state==RESPONSE && store_owned && (!store_match || !store_result)))revoked<=1;
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_status<=admission_status;completion_hls_status<=0;completion_read_beats<=0;completion_records<=0;
                    addresses[0]<=command_input_address;bases[0]<=command_input_base;sizes[0]<=command_input_bytes;
                    addresses[1]<=command_weight_address;bases[1]<=command_weight_base;sizes[1]<=command_weight_bytes;
                    addresses[2]<=command_output_address;bases[2]<=command_output_base;sizes[2]<=command_output_bytes;
                    check_index<=0;cancel_seen<=cancel_valid;revoked<=admission_status!=OK;fault<=admission_status;store_owned<=0;
                    state<=admission_status==OK ? CHECK_REGION:RESPONSE;
                end
                CHECK_REGION:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else begin
                        ends[check_index]<=selected_end;
                        if(check_index==2)begin check_index<=0;state<=CHECK_ALIAS;end
                        else check_index<=check_index+1;
                    end
                CHECK_ALIAS:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(check_index==1)state<=OFFER_LOADER;
                    else check_index<=check_index+1;
                OFFER_LOADER:if(loader_ready)state<=OFFER_OWNER;
                OFFER_OWNER:if(owner_ready)state<=OFFER_STORE;
                OFFER_STORE:if(store_ready)begin store_owned<=1;state<=WAIT_STORE;end
                WAIT_STORE:if(store_valid)begin
                    completion_status<=fault_now!=OK ? fault_now:store_status;
                    completion_hls_status<=owner_hls_status;completion_read_beats<=loader_beats;completion_records<=store_records;
                    state<=RESPONSE;
                end
                // Every public VALID/READY is consumed exactly once. A foreign
                // child stays owned in DRAIN until peers retire or shared reset.
                RESPONSE:if(completion_ready)state<=DRAIN;
                DRAIN:if(!loader_busy && !owner_busy && !store_busy)begin state<=IDLE;store_owned<=0;end
                default:begin state<=DRAIN;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
