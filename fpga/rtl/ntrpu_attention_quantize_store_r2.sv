// Root FP32 completion -> generated quantization -> owned ABI1 -> actual AXI writer.
// Global reset must also quiesce the upstream source and external AXI memory.
`default_nettype none
module ntrpu_attention_quantize_store_r2 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter logic [63:0] INITIAL_CHILD_TAG=64'd0
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_address,command_region_base,command_region_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,tag_exhausted,
    input wire logic source_completion_valid,
    output logic source_completion_ready,
    input wire logic [3:0] source_completion_status,
    input wire logic [63:0] source_sequence,source_epoch,
    input wire logic source_result_valid,
    output logic source_result_ce,
    output logic [10:0] source_result_address,
    input wire logic [31:0] source_result_q,
    input wire logic source_result_q_valid,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [31:0] completion_hls_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [6:0] completion_records,
    output logic result_valid,
    output logic [63:0] result_address,
    output logic [12:0] result_bytes,
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
    localparam logic [3:0] OK=0,BAD_IDENTITY=2,CANCELLED=7,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,WAIT_QUANT,OFFER_STORE,WAIT_STORE,RESPONSE} state_t;
    state_t state;
    logic [63:0] address,region_base,region_bytes;
    logic [3:0] fault,fault_now;
    logic revoked,cancel_seen,epoch_bad,store_identity_match;
    logic quant_command_valid,quant_command_ready,quant_busy,quant_cancel_ready;
    logic quant_completion_valid,quant_completion_ready,quant_result_valid;
    logic [3:0] quant_completion_status;
    logic [31:0] quant_hls_status;
    logic [63:0] quant_sequence,quant_epoch,quant_result_q;
    logic quant_result_ce,quant_result_q_valid;
    logic [8:0] quant_result_address;
    logic hls_start,hls_done,hls_ready,hls_idle;
    logic [31:0] hls_return,hls_elements,hls_input_q;
    logic hls_input_ce,hls_output_ce,hls_output_we;
    logic [10:0] hls_input_address;
    logic [8:0] hls_output_address;
    logic [63:0] hls_output_d;
    logic store_command_ready,store_busy,store_cancel_ready,store_completion_valid,store_completion_ready;
    logic [3:0] store_completion_status;
    logic [63:0] store_sequence,store_epoch,store_result_address;
    logic [6:0] store_records;
    logic [12:0] store_result_bytes;
    logic store_result_valid,child_cancel;

    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && quant_command_ready && !store_busy;
    assign quant_command_valid=state==IDLE && command_valid && !store_busy;
    assign completion_valid=state==RESPONSE;
    assign cancel_ready=busy && !cancel_seen;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    assign child_cancel=busy && (cancel_valid || fault_now!=OK);
    assign store_identity_match=store_sequence==completion_sequence && store_epoch==completion_epoch;
    assign store_completion_ready=state==RESPONSE && completion_ready && store_identity_match;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad &&
        !cancel_valid && store_result_valid && store_identity_match;
    assign result_address=result_valid ? store_result_address:64'd0;
    assign result_bytes=result_valid ? store_result_bytes:13'd0;

    ntrpu_attention_quantize_owner_r3 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) quant_owner(
        .clk,.rst_n,.command_valid(quant_command_valid),.command_ready(quant_command_ready),
        .command_sequence,.command_epoch,.expected_epoch,.cancel_valid(child_cancel),.cancel_ready(quant_cancel_ready),.busy(quant_busy),
        .source_completion_valid,.source_completion_ready,.source_completion_status,.source_sequence,.source_epoch,
        .source_result_valid,.source_result_ce,.source_result_address,.source_result_q,.source_result_q_valid,
        .hls_start,.hls_done,.hls_ready,.hls_return,.hls_elements,.hls_input_ce,.hls_input_address,.hls_input_q,
        .hls_output_ce,.hls_output_we,.hls_output_address,.hls_output_d,
        .completion_valid(quant_completion_valid),.completion_ready(quant_completion_ready),
        .completion_status(quant_completion_status),.completion_hls_status(quant_hls_status),
        .completion_sequence(quant_sequence),.completion_epoch(quant_epoch),.result_valid(quant_result_valid),
        .result_ce(quant_result_ce),.result_address(quant_result_address),.result_q(quant_result_q),.result_q_valid(quant_result_q_valid)
    );
    // Exact generated 2026.1 ap_ctrl_hs and ram_1p port names/widths.
    ntrpu_quantize_abi1_r1 quantizer(
        .ap_clk(clk),.ap_rst(!rst_n),.ap_start(hls_start),.ap_done(hls_done),.ap_ready(hls_ready),.ap_idle(hls_idle),
        .input_r_address0(hls_input_address),.input_r_ce0(hls_input_ce),.input_r_q0(hls_input_q),
        .output_r_address0(hls_output_address),.output_r_ce0(hls_output_ce),.output_r_we0(hls_output_we),
        .output_r_d0(hls_output_d),.elements(hls_elements),.ap_return(hls_return)
    );
    ntrpu_attention_activation_store_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.INITIAL_CHILD_TAG(INITIAL_CHILD_TAG)) store(
        .clk,.rst_n,.command_valid(state==OFFER_STORE),.command_ready(store_command_ready),
        .command_sequence(completion_sequence),.command_epoch(completion_epoch),.expected_epoch,
        .command_address(address),.command_region_base(region_base),.command_region_bytes(region_bytes),
        .cancel_valid(child_cancel),.cancel_ready(store_cancel_ready),.busy(store_busy),.tag_exhausted,
        .source_completion_valid(quant_completion_valid),.source_completion_ready(quant_completion_ready),
        .source_completion_status(quant_completion_status),.source_sequence(quant_sequence),.source_epoch(quant_epoch),
        .source_result_valid(quant_result_valid),.source_result_ce(quant_result_ce),.source_result_address(quant_result_address),
        .source_result_q(quant_result_q),.source_result_q_valid(quant_result_q_valid),
        .completion_valid(store_completion_valid),.completion_ready(store_completion_ready),.completion_status(store_completion_status),
        .completion_sequence(store_sequence),.completion_epoch(store_epoch),.completion_records(store_records),
        .result_valid(store_result_valid),.result_address(store_result_address),.result_bytes(store_result_bytes),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,.m_axi_awcache,
        .m_axi_awlock,.m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,.m_axi_wlast,.m_axi_wvalid,
        .m_axi_wready,.m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready
    );
    always_comb begin
        fault_now=fault;
        if(busy && state!=RESPONSE && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(store_completion_valid && state==WAIT_STORE && !store_identity_match)fault_now=BAD_PEER;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;region_base<=0;region_bytes<=0;fault<=OK;revoked<=0;cancel_seen<=0;
            completion_status<=OK;completion_hls_status<=0;completion_records<=0;completion_sequence<=0;completion_epoch<=0;
        end else begin
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(busy && state!=RESPONSE)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    address<=command_address;region_base<=command_region_base;region_bytes<=command_region_bytes;
                    completion_status<=OK;completion_hls_status<=0;completion_records<=0;
                    fault<=OK;revoked<=0;cancel_seen<=0;state<=WAIT_QUANT;
                end
                WAIT_QUANT:if(quant_completion_valid)begin
                    completion_hls_status<=quant_hls_status;state<=OFFER_STORE;
                end
                // The offered store command remains stable even if revoked.
                OFFER_STORE:if(store_command_ready)state<=WAIT_STORE;
                WAIT_STORE:if(store_completion_valid && store_identity_match)begin
                    completion_status<=fault_now!=OK ? fault_now:store_completion_status;
                    completion_records<=store_records;state<=RESPONSE;
                end
                RESPONSE:if(completion_ready && store_identity_match)state<=IDLE;
                default:begin state<=IDLE;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
