// Sequential FP32 -> ABI1 -> full O projection, using the actual child modules.
// Caller owns immutable weights and disjoint work regions until public completion.
// Shared reset must also reset/quiesce the upstream source and external AXI memory.
`default_nettype none
module ntrpu_attention_output_path_r2 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter logic [63:0] INITIAL_QUANT_TAG=64'd0,
    parameter logic [63:0] INITIAL_PROJECTION_TAG=64'd0
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_weight_addr,command_weight_region_base,command_weight_region_bytes,
    input wire logic [63:0] command_activation_addr,command_activation_region_base,command_activation_region_bytes,
    input wire logic [63:0] command_output_addr,command_output_region_base,command_output_region_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic result_valid,
    output logic [63:0] result_address,
    output logic [16:0] result_bytes,
    output logic tag_exhausted,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize, m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid, m_axi_arcache,
    output logic m_axi_arlock, m_axi_arvalid,
    input wire logic m_axi_arready,
    input wire logic [511:0] m_axi_rdata,
    input wire logic [1:0] m_axi_rresp,
    input wire logic [3:0] m_axi_rid,
    input wire logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize, m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid, m_axi_awcache,
    output logic m_axi_awlock, m_axi_awvalid,
    input wire logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast, m_axi_wvalid,
    input wire logic m_axi_wready,
    input wire logic [3:0] m_axi_bid,
    input wire logic [1:0] m_axi_bresp,
    input wire logic m_axi_bvalid,
    output logic m_axi_bready,
    input wire logic source_completion_valid,
    output logic source_completion_ready,
    input wire logic [3:0] source_completion_status,
    input wire logic [63:0] source_sequence,source_epoch,
    input wire logic source_result_valid,
    output logic source_result_ce,
    output logic [10:0] source_result_address,
    input wire logic [31:0] source_result_q,
    input wire logic source_result_q_valid,
    output logic [31:0] completion_hls_status
);

    localparam logic [3:0] OK=0,BAD_IDENTITY=2,BAD_COMPLETION=5,CANCELLED=7,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,WAIT_QUANT,OFFER_PROJECTION,WAIT_PROJECTION,RESPONSE} state_t;
    state_t state;
    logic [63:0] weight_address,weight_base,weight_bytes,activation_address,activation_base,activation_bytes;
    logic [63:0] output_address,output_base,output_bytes;
    logic [3:0] fault,fault_now;
    logic revoked,cancel_seen,epoch_bad,child_cancel,projection_owned;
    logic quant_identity_match,projection_identity_match,children_held,release_children;
    logic q_command_ready;
    logic q_cancel_ready;
    logic q_busy;
    logic q_tag_exhausted;
    logic q_completion_valid;
    logic [3:0] q_completion_status;
    logic [31:0] q_completion_hls_status;
    logic [63:0] q_completion_sequence;
    logic [63:0] q_completion_epoch;
    logic [6:0] q_completion_records;
    logic q_result_valid;
    logic [63:0] q_result_address;
    logic [12:0] q_result_bytes;
    logic [63:0] q_m_axi_awaddr;
    logic [7:0] q_m_axi_awlen;
    logic [2:0] q_m_axi_awsize;
    logic [2:0] q_m_axi_awprot;
    logic [1:0] q_m_axi_awburst;
    logic [3:0] q_m_axi_awid;
    logic [3:0] q_m_axi_awcache;
    logic q_m_axi_awlock;
    logic q_m_axi_awvalid;
    logic q_m_axi_awready;
    logic [511:0] q_m_axi_wdata;
    logic [63:0] q_m_axi_wstrb;
    logic q_m_axi_wlast;
    logic q_m_axi_wvalid;
    logic q_m_axi_wready;
    logic [3:0] q_m_axi_bid;
    logic [1:0] q_m_axi_bresp;
    logic q_m_axi_bvalid;
    logic q_m_axi_bready;
    logic p_command_ready;
    logic p_cancel_ready;
    logic p_busy;
    logic p_completion_valid;
    logic [3:0] p_completion_status;
    logic [63:0] p_completion_sequence;
    logic [63:0] p_completion_epoch;
    logic p_result_valid;
    logic [63:0] p_result_address;
    logic [16:0] p_result_bytes;
    logic p_tag_exhausted;
    logic [63:0] p_m_axi_araddr;
    logic [7:0] p_m_axi_arlen;
    logic [2:0] p_m_axi_arsize;
    logic [2:0] p_m_axi_arprot;
    logic [1:0] p_m_axi_arburst;
    logic [3:0] p_m_axi_arid;
    logic [3:0] p_m_axi_arcache;
    logic p_m_axi_arlock;
    logic p_m_axi_arvalid;
    logic p_m_axi_arready;
    logic [511:0] p_m_axi_rdata;
    logic [1:0] p_m_axi_rresp;
    logic [3:0] p_m_axi_rid;
    logic p_m_axi_rlast;
    logic p_m_axi_rvalid;
    logic p_m_axi_rready;
    logic [63:0] p_m_axi_awaddr;
    logic [7:0] p_m_axi_awlen;
    logic [2:0] p_m_axi_awsize;
    logic [2:0] p_m_axi_awprot;
    logic [1:0] p_m_axi_awburst;
    logic [3:0] p_m_axi_awid;
    logic [3:0] p_m_axi_awcache;
    logic p_m_axi_awlock;
    logic p_m_axi_awvalid;
    logic p_m_axi_awready;
    logic [511:0] p_m_axi_wdata;
    logic [63:0] p_m_axi_wstrb;
    logic p_m_axi_wlast;
    logic p_m_axi_wvalid;
    logic p_m_axi_wready;
    logic [3:0] p_m_axi_bid;
    logic [1:0] p_m_axi_bresp;
    logic p_m_axi_bvalid;
    logic p_m_axi_bready;

    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && q_command_ready && !p_busy;
    assign cancel_ready=busy && !cancel_seen;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    assign child_cancel=busy && (cancel_valid || fault_now!=OK);
    assign quant_identity_match=q_completion_sequence==completion_sequence && q_completion_epoch==completion_epoch;
    assign projection_identity_match=p_completion_sequence==completion_sequence && p_completion_epoch==completion_epoch;
    assign children_held=q_completion_valid && quant_identity_match &&
        (!projection_owned || (p_completion_valid && projection_identity_match));
    assign completion_valid=state==RESPONSE && children_held;
    assign release_children=completion_valid && completion_ready;
    assign tag_exhausted=q_tag_exhausted || p_tag_exhausted;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad &&
        !cancel_valid && projection_owned && q_result_valid && p_result_valid;
    assign result_address=result_valid ? p_result_address:64'd0;
    assign result_bytes=result_valid ? p_result_bytes:17'd0;

    // The quantized region stays owned through all O reads and the public ACK.
    // Switch only after quantization has published and drained its AXI writes.
    assign m_axi_araddr=p_m_axi_araddr;
    assign m_axi_arlen=p_m_axi_arlen;
    assign m_axi_arsize=p_m_axi_arsize;
    assign m_axi_arprot=p_m_axi_arprot;
    assign m_axi_arburst=p_m_axi_arburst;
    assign m_axi_arid=p_m_axi_arid;
    assign m_axi_arcache=p_m_axi_arcache;
    assign m_axi_arlock=p_m_axi_arlock;
    assign m_axi_arvalid=projection_owned && p_m_axi_arvalid;
    assign p_m_axi_arready=projection_owned && m_axi_arready;
    assign p_m_axi_rdata=m_axi_rdata;
    assign p_m_axi_rresp=m_axi_rresp;
    assign p_m_axi_rid=m_axi_rid;
    assign p_m_axi_rlast=m_axi_rlast;
    assign p_m_axi_rvalid=projection_owned && m_axi_rvalid;
    assign m_axi_rready=projection_owned && p_m_axi_rready;
    assign m_axi_awaddr=projection_owned ? p_m_axi_awaddr:q_m_axi_awaddr;
    assign m_axi_awlen=projection_owned ? p_m_axi_awlen:q_m_axi_awlen;
    assign m_axi_awsize=projection_owned ? p_m_axi_awsize:q_m_axi_awsize;
    assign m_axi_awprot=projection_owned ? p_m_axi_awprot:q_m_axi_awprot;
    assign m_axi_awburst=projection_owned ? p_m_axi_awburst:q_m_axi_awburst;
    assign m_axi_awid=projection_owned ? p_m_axi_awid:q_m_axi_awid;
    assign m_axi_awcache=projection_owned ? p_m_axi_awcache:q_m_axi_awcache;
    assign m_axi_awlock=projection_owned ? p_m_axi_awlock:q_m_axi_awlock;
    assign m_axi_awvalid=projection_owned ? p_m_axi_awvalid:q_m_axi_awvalid;
    assign q_m_axi_awready=!projection_owned && m_axi_awready;
    assign p_m_axi_awready=projection_owned && m_axi_awready;
    assign m_axi_wdata=projection_owned ? p_m_axi_wdata:q_m_axi_wdata;
    assign m_axi_wstrb=projection_owned ? p_m_axi_wstrb:q_m_axi_wstrb;
    assign m_axi_wlast=projection_owned ? p_m_axi_wlast:q_m_axi_wlast;
    assign m_axi_wvalid=projection_owned ? p_m_axi_wvalid:q_m_axi_wvalid;
    assign q_m_axi_wready=!projection_owned && m_axi_wready;
    assign p_m_axi_wready=projection_owned && m_axi_wready;
    assign q_m_axi_bid=m_axi_bid;
    assign p_m_axi_bid=m_axi_bid;
    assign q_m_axi_bresp=m_axi_bresp;
    assign p_m_axi_bresp=m_axi_bresp;
    assign q_m_axi_bvalid=!projection_owned && m_axi_bvalid;
    assign p_m_axi_bvalid=projection_owned && m_axi_bvalid;
    assign m_axi_bready=projection_owned ? p_m_axi_bready:q_m_axi_bready;
    ntrpu_attention_quantize_store_r2 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.INITIAL_CHILD_TAG(INITIAL_QUANT_TAG)) quantize_store(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(state==IDLE && command_valid && !p_busy),
        .command_ready(q_command_ready),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .expected_epoch(expected_epoch),
        .command_address(command_activation_addr),
        .command_region_base(command_activation_region_base),
        .command_region_bytes(command_activation_region_bytes),
        .cancel_valid(child_cancel),
        .cancel_ready(q_cancel_ready),
        .busy(q_busy),
        .tag_exhausted(q_tag_exhausted),
        .source_completion_valid(source_completion_valid),
        .source_completion_ready(source_completion_ready),
        .source_completion_status(source_completion_status),
        .source_sequence(source_sequence),
        .source_epoch(source_epoch),
        .source_result_valid(source_result_valid),
        .source_result_ce(source_result_ce),
        .source_result_address(source_result_address),
        .source_result_q(source_result_q),
        .source_result_q_valid(source_result_q_valid),
        .completion_valid(q_completion_valid),
        .completion_ready(release_children),
        .completion_status(q_completion_status),
        .completion_hls_status(q_completion_hls_status),
        .completion_sequence(q_completion_sequence),
        .completion_epoch(q_completion_epoch),
        .completion_records(q_completion_records),
        .result_valid(q_result_valid),
        .result_address(q_result_address),
        .result_bytes(q_result_bytes),
        .m_axi_awaddr(q_m_axi_awaddr),
        .m_axi_awlen(q_m_axi_awlen),
        .m_axi_awsize(q_m_axi_awsize),
        .m_axi_awprot(q_m_axi_awprot),
        .m_axi_awburst(q_m_axi_awburst),
        .m_axi_awid(q_m_axi_awid),
        .m_axi_awcache(q_m_axi_awcache),
        .m_axi_awlock(q_m_axi_awlock),
        .m_axi_awvalid(q_m_axi_awvalid),
        .m_axi_awready(q_m_axi_awready),
        .m_axi_wdata(q_m_axi_wdata),
        .m_axi_wstrb(q_m_axi_wstrb),
        .m_axi_wlast(q_m_axi_wlast),
        .m_axi_wvalid(q_m_axi_wvalid),
        .m_axi_wready(q_m_axi_wready),
        .m_axi_bid(q_m_axi_bid),
        .m_axi_bresp(q_m_axi_bresp),
        .m_axi_bvalid(q_m_axi_bvalid),
        .m_axi_bready(q_m_axi_bready)
    );
    ntrpu_attention_projection_memory_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.INITIAL_CHILD_TAG(INITIAL_PROJECTION_TAG)) projection(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(state==OFFER_PROJECTION),
        .command_ready(p_command_ready),
        .command_sequence(completion_sequence),
        .command_epoch(completion_epoch),
        .expected_epoch(expected_epoch),
        .command_weight_addr(weight_address),
        .command_weight_region_base(weight_base),
        .command_weight_region_bytes(weight_bytes),
        .command_activation_addr(activation_address),
        .command_activation_region_base(activation_base),
        .command_activation_region_bytes(activation_bytes),
        .command_output_addr(output_address),
        .command_output_region_base(output_base),
        .command_output_region_bytes(output_bytes),
        .cancel_valid(child_cancel),
        .cancel_ready(p_cancel_ready),
        .busy(p_busy),
        .completion_valid(p_completion_valid),
        .completion_ready(release_children && projection_owned),
        .completion_status(p_completion_status),
        .completion_sequence(p_completion_sequence),
        .completion_epoch(p_completion_epoch),
        .result_valid(p_result_valid),
        .result_address(p_result_address),
        .result_bytes(p_result_bytes),
        .tag_exhausted(p_tag_exhausted),
        .m_axi_araddr(p_m_axi_araddr),
        .m_axi_arlen(p_m_axi_arlen),
        .m_axi_arsize(p_m_axi_arsize),
        .m_axi_arprot(p_m_axi_arprot),
        .m_axi_arburst(p_m_axi_arburst),
        .m_axi_arid(p_m_axi_arid),
        .m_axi_arcache(p_m_axi_arcache),
        .m_axi_arlock(p_m_axi_arlock),
        .m_axi_arvalid(p_m_axi_arvalid),
        .m_axi_arready(p_m_axi_arready),
        .m_axi_rdata(p_m_axi_rdata),
        .m_axi_rresp(p_m_axi_rresp),
        .m_axi_rid(p_m_axi_rid),
        .m_axi_rlast(p_m_axi_rlast),
        .m_axi_rvalid(p_m_axi_rvalid),
        .m_axi_rready(p_m_axi_rready),
        .m_axi_awaddr(p_m_axi_awaddr),
        .m_axi_awlen(p_m_axi_awlen),
        .m_axi_awsize(p_m_axi_awsize),
        .m_axi_awprot(p_m_axi_awprot),
        .m_axi_awburst(p_m_axi_awburst),
        .m_axi_awid(p_m_axi_awid),
        .m_axi_awcache(p_m_axi_awcache),
        .m_axi_awlock(p_m_axi_awlock),
        .m_axi_awvalid(p_m_axi_awvalid),
        .m_axi_awready(p_m_axi_awready),
        .m_axi_wdata(p_m_axi_wdata),
        .m_axi_wstrb(p_m_axi_wstrb),
        .m_axi_wlast(p_m_axi_wlast),
        .m_axi_wvalid(p_m_axi_wvalid),
        .m_axi_wready(p_m_axi_wready),
        .m_axi_bid(p_m_axi_bid),
        .m_axi_bresp(p_m_axi_bresp),
        .m_axi_bvalid(p_m_axi_bvalid),
        .m_axi_bready(p_m_axi_bready)
    );

    always_comb begin
        fault_now=fault;
        if(busy && state!=RESPONSE && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(q_completion_valid && !quant_identity_match)fault_now=BAD_PEER;
            else if(projection_owned && p_completion_valid && !projection_identity_match)fault_now=BAD_PEER;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;fault<=OK;revoked<=0;cancel_seen<=0;projection_owned<=0;
            completion_status<=OK;completion_hls_status<=0;completion_sequence<=0;completion_epoch<=0;
            weight_address<=0;weight_base<=0;weight_bytes<=0;activation_address<=0;activation_base<=0;activation_bytes<=0;
            output_address<=0;output_base<=0;output_bytes<=0;
        end else begin
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(busy && state!=RESPONSE)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    weight_address<=command_weight_addr;weight_base<=command_weight_region_base;weight_bytes<=command_weight_region_bytes;
                    activation_address<=command_activation_addr;activation_base<=command_activation_region_base;activation_bytes<=command_activation_region_bytes;
                    output_address<=command_output_addr;output_base<=command_output_region_base;output_bytes<=command_output_region_bytes;
                    completion_status<=OK;completion_hls_status<=0;fault<=OK;revoked<=0;cancel_seen<=0;projection_owned<=0;
                    state<=WAIT_QUANT;
                end
                WAIT_QUANT:if(q_completion_valid && quant_identity_match)begin
                    completion_hls_status<=q_completion_hls_status;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(q_completion_status!=OK)begin completion_status<=q_completion_status;state<=RESPONSE;end
                    else if(!q_result_valid || q_result_address!=activation_address || q_result_bytes!=4096 || q_completion_records!=64)begin
                        completion_status<=BAD_COMPLETION;state<=RESPONSE;
                    end else begin projection_owned<=1;state<=OFFER_PROJECTION;end
                end
                // Once offered, keep the O command stable and drain it even if revoked.
                OFFER_PROJECTION:if(p_command_ready)state<=WAIT_PROJECTION;
                WAIT_PROJECTION:if(p_completion_valid && projection_identity_match)begin
                    if(fault_now!=OK)completion_status<=fault_now;
                    else if(p_completion_status!=OK)completion_status<=p_completion_status;
                    else if(!p_result_valid || p_result_address!=output_address || p_result_bytes!=8192)
                        completion_status<=BAD_COMPLETION;
                    else completion_status<=OK;
                    state<=RESPONSE;
                end
                RESPONSE:if(release_children)begin projection_owned<=0;state<=IDLE;end
                default:begin state<=IDLE;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
