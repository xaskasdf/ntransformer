// Complete owned FFN followed by actual residual arithmetic over one AXI port.
// Input/weight/workspace allocations remain reserved through the final ACK.
// The final output must be disjoint from every captured allocation.
// All children and the AXI memory peer share coordinated reset.
`default_nettype none
module ntrpu_llama32_ffn_residual_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter logic [63:0] INITIAL_TAG=64'd0
) (
    input wire clk,rst_n,
    input wire command_valid,
    input wire [63:0] command_sequence,
    input wire [63:0] command_epoch,
    input wire command_source_valid,
    input wire [63:0] command_source_sequence,
    input wire [63:0] command_source_epoch,
    input wire [63:0] command_addr [0:7],
    input wire [63:0] command_base [0:7],
    input wire [63:0] command_bytes [0:7],
    input wire [63:0] command_residual_address,
    input wire [63:0] command_residual_base,
    input wire [63:0] command_residual_bytes,
    input wire [63:0] expected_epoch,
    input wire [63:0] command_output_address,command_output_base,command_output_bytes,
    output wire command_ready,
    input wire cancel_valid,
    output wire cancel_ready,busy,
    output wire completion_valid,
    input wire completion_ready,
    output wire [3:0] completion_status,completion_ffn_status,completion_ffn_child_status,completion_residual_status,
    output wire [2:0] completion_ffn_stages,
    output wire [1:0] completion_ffn_phase,
    output wire [16:0] completion_ffn_bytes,
    output wire [17:0] completion_ffn_stage_bytes,
    output wire [31:0] completion_ffn_hls_status,completion_hls_status,
    output wire [9:0] completion_read_beats,
    output wire [7:0] completion_write_beats,
    output wire [63:0] completion_sequence,completion_epoch,
    output wire result_valid,
    output wire [63:0] result_address,
    output wire [13:0] result_bytes,
    output wire tags_exhausted,activation_selected,residual_selected,
    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [2:0] m_axi_arprot,
    output wire [1:0] m_axi_arburst,
    output wire [3:0] m_axi_arid,
    output wire [3:0] m_axi_arcache,
    output wire m_axi_arlock,
    output wire m_axi_arvalid,
    input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire [3:0] m_axi_rid,
    input wire m_axi_rlast,
    input wire m_axi_rvalid,
    output wire m_axi_rready,
    output wire [63:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [2:0] m_axi_awprot,
    output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awid,
    output wire [3:0] m_axi_awcache,
    output wire m_axi_awlock,
    output wire m_axi_awvalid,
    input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast,
    output wire m_axi_wvalid,
    input wire m_axi_wready,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    typedef enum logic [2:0] {IDLE,CHECK_OUTPUT,CHECK_REGIONS,FFN_OFFER,WAIT_FFN,WAIT_HANDOFF,LOCAL_RESPONSE} state_t;
    state_t state;
    logic [63:0] sequence_reg,epoch_reg,source_sequence,source_epoch;
    logic source_valid,cancel_seen,ffn_launched;
    logic [63:0] address [0:7],region_base [0:7],region_bytes [0:7];
    logic [63:0] residual_address,residual_base,residual_bytes,output_address,output_base,output_bytes;
    logic [3:0] fault,region_index;
    wire epoch_bad=busy && expected_epoch!=epoch_reg;
    wire lifecycle_fault=cancel_seen || cancel_valid || epoch_bad || fault!=0;
    wire [3:0] fault_now=fault!=0 ? fault:epoch_bad ? 4'd2:cancel_valid ? 4'd7:4'd0;
    wire [64:0] output_end={1'b0,output_address}+65'd8192;
    wire [64:0] output_limit={1'b0,output_base}+{1'b0,output_bytes};
    wire output_bad=output_address[5:0]!=0 || output_bytes==0 || output_end[64] || output_limit[64] ||
        output_address<output_base || output_end>output_limit;
    wire [63:0] allocation_base=region_index==8 ? residual_base:region_base[region_index[2:0]];
    wire [63:0] allocation_bytes=region_index==8 ? residual_bytes:region_bytes[region_index[2:0]];
    wire [64:0] allocation_end={1'b0,allocation_base}+{1'b0,allocation_bytes};
    wire allocation_bad=allocation_bytes==0 || allocation_end[64] ||
        ({1'b0,output_address}<allocation_end && {1'b0,allocation_base}<output_end);
    wire ffn_command_valid;
    wire ffn_command_ready;
    wire [63:0] ffn_command_sequence;
    wire [63:0] ffn_command_epoch;
    wire ffn_command_source_valid;
    wire [63:0] ffn_command_source_sequence;
    wire [63:0] ffn_command_source_epoch;
    wire [63:0] ffn_command_addr [0:7];
    wire [63:0] ffn_command_base [0:7];
    wire [63:0] ffn_command_bytes [0:7];
    wire [63:0] ffn_command_residual_address;
    wire [63:0] ffn_command_residual_base;
    wire [63:0] ffn_command_residual_bytes;
    wire ffn_cancel_valid;
    wire ffn_cancel_ready;
    wire ffn_busy;
    wire ffn_completion_valid;
    wire ffn_completion_ready;
    wire [3:0] ffn_completion_status;
    wire [3:0] ffn_completion_child_status;
    wire [2:0] ffn_completion_stages;
    wire [1:0] ffn_completion_phase;
    wire [16:0] ffn_completion_bytes;
    wire [17:0] ffn_completion_stage_bytes;
    wire [31:0] ffn_completion_hls_status;
    wire [63:0] ffn_completion_sequence;
    wire [63:0] ffn_completion_epoch;
    wire ffn_result_valid;
    wire [63:0] ffn_result_ffn_address;
    wire [63:0] ffn_result_residual_address;
    wire [13:0] ffn_result_ffn_bytes;
    wire [13:0] ffn_result_residual_bytes;
    wire ffn_tags_exhausted;
    wire ffn_activation_selected;
    wire [63:0] ffn_m_axi_araddr;
    wire [7:0] ffn_m_axi_arlen;
    wire [2:0] ffn_m_axi_arsize;
    wire [2:0] ffn_m_axi_arprot;
    wire [1:0] ffn_m_axi_arburst;
    wire [3:0] ffn_m_axi_arid;
    wire [3:0] ffn_m_axi_arcache;
    wire ffn_m_axi_arlock;
    wire ffn_m_axi_arvalid;
    wire ffn_m_axi_arready;
    wire [511:0] ffn_m_axi_rdata;
    wire [1:0] ffn_m_axi_rresp;
    wire [3:0] ffn_m_axi_rid;
    wire ffn_m_axi_rlast;
    wire ffn_m_axi_rvalid;
    wire ffn_m_axi_rready;
    wire [63:0] ffn_m_axi_awaddr;
    wire [7:0] ffn_m_axi_awlen;
    wire [2:0] ffn_m_axi_awsize;
    wire [2:0] ffn_m_axi_awprot;
    wire [1:0] ffn_m_axi_awburst;
    wire [3:0] ffn_m_axi_awid;
    wire [3:0] ffn_m_axi_awcache;
    wire ffn_m_axi_awlock;
    wire ffn_m_axi_awvalid;
    wire ffn_m_axi_awready;
    wire [511:0] ffn_m_axi_wdata;
    wire [63:0] ffn_m_axi_wstrb;
    wire ffn_m_axi_wlast;
    wire ffn_m_axi_wvalid;
    wire ffn_m_axi_wready;
    wire [3:0] ffn_m_axi_bid;
    wire [1:0] ffn_m_axi_bresp;
    wire ffn_m_axi_bvalid;
    wire ffn_m_axi_bready;
    wire res_command_valid;
    wire res_command_ready;
    wire [63:0] res_command_sequence;
    wire [63:0] res_command_epoch;
    wire [63:0] res_command_source_sequence;
    wire [63:0] res_command_source_epoch;
    wire res_command_source_valid;
    wire [63:0] res_command_input_address;
    wire [63:0] res_command_input_base;
    wire [63:0] res_command_input_bytes;
    wire [63:0] res_command_branch_address;
    wire [63:0] res_command_branch_base;
    wire [63:0] res_command_branch_bytes;
    wire [63:0] res_command_residual_address;
    wire [63:0] res_command_residual_base;
    wire [63:0] res_command_residual_bytes;
    wire res_cancel_valid;
    wire res_cancel_ready;
    wire res_busy;
    wire res_completion_valid;
    wire res_completion_ready;
    wire [3:0] res_completion_status;
    wire [31:0] res_completion_hls_status;
    wire [63:0] res_completion_sequence;
    wire [63:0] res_completion_epoch;
    wire [9:0] res_completion_read_beats;
    wire [7:0] res_completion_residual_beats;
    wire res_result_valid;
    wire [63:0] res_result_residual_address;
    wire [13:0] res_result_residual_bytes;
    wire [63:0] res_m_axi_araddr;
    wire [7:0] res_m_axi_arlen;
    wire [2:0] res_m_axi_arsize;
    wire [2:0] res_m_axi_arprot;
    wire [1:0] res_m_axi_arburst;
    wire [3:0] res_m_axi_arid;
    wire [3:0] res_m_axi_arcache;
    wire res_m_axi_arlock;
    wire res_m_axi_arvalid;
    wire res_m_axi_arready;
    wire [511:0] res_m_axi_rdata;
    wire [1:0] res_m_axi_rresp;
    wire [3:0] res_m_axi_rid;
    wire res_m_axi_rlast;
    wire res_m_axi_rvalid;
    wire res_m_axi_rready;
    wire [63:0] res_m_axi_awaddr;
    wire [7:0] res_m_axi_awlen;
    wire [2:0] res_m_axi_awsize;
    wire [2:0] res_m_axi_awprot;
    wire [1:0] res_m_axi_awburst;
    wire [3:0] res_m_axi_awid;
    wire [3:0] res_m_axi_awcache;
    wire res_m_axi_awlock;
    wire res_m_axi_awvalid;
    wire res_m_axi_awready;
    wire [511:0] res_m_axi_wdata;
    wire [63:0] res_m_axi_wstrb;
    wire res_m_axi_wlast;
    wire res_m_axi_wvalid;
    wire res_m_axi_wready;
    wire [3:0] res_m_axi_bid;
    wire [1:0] res_m_axi_bresp;
    wire res_m_axi_bvalid;
    wire res_m_axi_bready;
    wire handoff_command_valid;
    wire handoff_command_ready;
    wire [63:0] handoff_command_sequence;
    wire [63:0] handoff_command_epoch;
    wire [63:0] handoff_command_output_address;
    wire [63:0] handoff_command_output_base;
    wire [63:0] handoff_command_output_bytes;
    wire handoff_cancel_valid;
    wire handoff_cancel_ready;
    wire handoff_busy;
    wire handoff_source_completion_valid;
    wire handoff_source_completion_ready;
    wire [3:0] handoff_source_completion_status;
    wire [63:0] handoff_source_sequence;
    wire [63:0] handoff_source_epoch;
    wire handoff_source_result_valid;
    wire [63:0] handoff_source_residual_address;
    wire [63:0] handoff_source_ffn_address;
    wire [13:0] handoff_source_residual_bytes;
    wire [13:0] handoff_source_ffn_bytes;
    wire handoff_child_command_valid;
    wire handoff_child_command_ready;
    wire [63:0] handoff_child_command_sequence;
    wire [63:0] handoff_child_command_epoch;
    wire [63:0] handoff_child_source_sequence;
    wire [63:0] handoff_child_source_epoch;
    wire handoff_child_source_valid;
    wire [63:0] handoff_child_input_address;
    wire [63:0] handoff_child_input_base;
    wire [63:0] handoff_child_input_bytes;
    wire [63:0] handoff_child_branch_address;
    wire [63:0] handoff_child_branch_base;
    wire [63:0] handoff_child_branch_bytes;
    wire [63:0] handoff_child_output_address;
    wire [63:0] handoff_child_output_base;
    wire [63:0] handoff_child_output_bytes;
    wire handoff_child_cancel_valid;
    wire handoff_child_completion_valid;
    wire handoff_child_completion_ready;
    wire [3:0] handoff_child_completion_status;
    wire [31:0] handoff_child_hls_status;
    wire [63:0] handoff_child_sequence;
    wire [63:0] handoff_child_epoch;
    wire [9:0] handoff_child_read_beats;
    wire [7:0] handoff_child_write_beats;
    wire handoff_child_result_valid;
    wire [63:0] handoff_child_result_address;
    wire [13:0] handoff_child_result_bytes;
    wire handoff_residual_bus_owned;
    wire handoff_completion_valid;
    wire handoff_completion_ready;
    wire [3:0] handoff_completion_status;
    wire [3:0] handoff_completion_source_status;
    wire [3:0] handoff_completion_child_status;
    wire [31:0] handoff_completion_hls_status;
    wire [63:0] handoff_completion_sequence;
    wire [63:0] handoff_completion_epoch;
    wire [9:0] handoff_completion_read_beats;
    wire [7:0] handoff_completion_write_beats;
    wire handoff_result_valid;
    wire [63:0] handoff_result_address;
    wire [13:0] handoff_result_bytes;
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && ffn_command_ready && !ffn_completion_valid && !ffn_busy && !handoff_busy && res_command_ready && !res_completion_valid;
    assign cancel_ready=busy && !cancel_seen;
    assign completion_valid=state==LOCAL_RESPONSE || (state==WAIT_HANDOFF && handoff_completion_valid);
    assign completion_status=state==LOCAL_RESPONSE ? fault:handoff_completion_status;
    assign completion_sequence=sequence_reg;
    assign completion_epoch=epoch_reg;
    assign result_valid=state==WAIT_HANDOFF && handoff_result_valid && !lifecycle_fault;
    assign result_address=result_valid ? handoff_result_address:64'd0;
    assign result_bytes=result_valid ? handoff_result_bytes:14'd0;
    assign completion_ffn_status=ffn_launched ? ffn_completion_status:4'd0;
    assign completion_ffn_child_status=ffn_launched ? ffn_completion_child_status:4'd0;
    assign completion_ffn_stages=ffn_launched ? ffn_completion_stages:3'd0;
    assign completion_ffn_phase=ffn_launched ? ffn_completion_phase:2'd0;
    assign completion_ffn_bytes=ffn_launched ? ffn_completion_bytes:17'd0;
    assign completion_ffn_stage_bytes=ffn_launched ? ffn_completion_stage_bytes:18'd0;
    assign completion_ffn_hls_status=ffn_launched ? ffn_completion_hls_status:32'd0;
    assign completion_residual_status=state==LOCAL_RESPONSE ? 4'd0:handoff_completion_child_status;
    assign completion_hls_status=state==LOCAL_RESPONSE ? 32'd0:handoff_completion_hls_status;
    assign completion_read_beats=state==LOCAL_RESPONSE ? 10'd0:handoff_completion_read_beats;
    assign completion_write_beats=state==LOCAL_RESPONSE ? 8'd0:handoff_completion_write_beats;
    assign tags_exhausted=ffn_tags_exhausted;
    assign activation_selected=ffn_activation_selected;
    assign residual_selected=handoff_residual_bus_owned;
    assign ffn_command_valid=state==FFN_OFFER;
    assign ffn_command_sequence=sequence_reg;
    assign ffn_command_epoch=epoch_reg;
    assign ffn_command_source_valid=source_valid;
    assign ffn_command_source_sequence=source_sequence;
    assign ffn_command_source_epoch=source_epoch;
    assign ffn_command_residual_address=residual_address;
    assign ffn_command_residual_base=residual_base;
    assign ffn_command_residual_bytes=residual_bytes;
    assign ffn_cancel_valid=busy && lifecycle_fault;
    assign ffn_completion_ready=handoff_source_completion_ready;
    assign handoff_command_valid=state==WAIT_FFN;
    assign handoff_command_sequence=sequence_reg;
    assign handoff_command_epoch=epoch_reg;
    assign handoff_command_output_address=output_address;
    assign handoff_command_output_base=output_base;
    assign handoff_command_output_bytes=output_bytes;
    assign handoff_cancel_valid=busy && lifecycle_fault;
    assign handoff_completion_ready=state==WAIT_HANDOFF && completion_ready;
    assign handoff_source_completion_valid=ffn_completion_valid;
    assign handoff_source_completion_status=ffn_completion_status;
    assign handoff_source_sequence=ffn_completion_sequence;
    assign handoff_source_epoch=ffn_completion_epoch;
    assign handoff_source_result_valid=ffn_result_valid;
    assign handoff_source_residual_address=ffn_result_residual_address;
    assign handoff_source_residual_bytes=ffn_result_residual_bytes;
    assign handoff_source_ffn_address=ffn_result_ffn_address;
    assign handoff_source_ffn_bytes=ffn_result_ffn_bytes;
    assign handoff_child_command_ready=res_command_ready;
    assign handoff_child_completion_valid=res_completion_valid;
    assign handoff_child_completion_status=res_completion_status;
    assign handoff_child_hls_status=res_completion_hls_status;
    assign handoff_child_sequence=res_completion_sequence;
    assign handoff_child_epoch=res_completion_epoch;
    assign handoff_child_read_beats=res_completion_read_beats;
    assign handoff_child_write_beats=res_completion_residual_beats;
    assign handoff_child_result_valid=res_result_valid;
    assign handoff_child_result_address=res_result_residual_address;
    assign handoff_child_result_bytes=res_result_residual_bytes;
    assign res_command_valid=handoff_child_command_valid;
    assign res_command_sequence=handoff_child_command_sequence;
    assign res_command_epoch=handoff_child_command_epoch;
    assign res_command_source_sequence=handoff_child_source_sequence;
    assign res_command_source_epoch=handoff_child_source_epoch;
    assign res_command_source_valid=handoff_child_source_valid;
    assign res_cancel_valid=handoff_child_cancel_valid;
    assign res_completion_ready=handoff_child_completion_ready;
    assign res_command_input_address=handoff_child_input_address;
    assign res_command_input_base=handoff_child_input_base;
    assign res_command_input_bytes=handoff_child_input_bytes;
    assign res_command_branch_address=handoff_child_branch_address;
    assign res_command_branch_base=handoff_child_branch_base;
    assign res_command_branch_bytes=handoff_child_branch_bytes;
    assign res_command_residual_address=handoff_child_output_address;
    assign res_command_residual_base=handoff_child_output_base;
    assign res_command_residual_bytes=handoff_child_output_bytes;
    for(genvar i=0;i<8;i++)begin
        assign ffn_command_addr[i]=address[i];
        assign ffn_command_base[i]=region_base[i];
        assign ffn_command_bytes[i]=region_bytes[i];
    end
    // FFN completion guarantees all five AXI channels have drained. Handoff
    // takes the bus before offering residual work and retains it through ACK.
    assign m_axi_araddr=residual_selected ? res_m_axi_araddr:ffn_m_axi_araddr;
    assign m_axi_arlen=residual_selected ? res_m_axi_arlen:ffn_m_axi_arlen;
    assign m_axi_arsize=residual_selected ? res_m_axi_arsize:ffn_m_axi_arsize;
    assign m_axi_arprot=residual_selected ? res_m_axi_arprot:ffn_m_axi_arprot;
    assign m_axi_arburst=residual_selected ? res_m_axi_arburst:ffn_m_axi_arburst;
    assign m_axi_arid=residual_selected ? res_m_axi_arid:ffn_m_axi_arid;
    assign m_axi_arcache=residual_selected ? res_m_axi_arcache:ffn_m_axi_arcache;
    assign m_axi_arlock=residual_selected ? res_m_axi_arlock:ffn_m_axi_arlock;
    assign m_axi_arvalid=residual_selected ? res_m_axi_arvalid:ffn_m_axi_arvalid;
    assign ffn_m_axi_arready=!residual_selected && m_axi_arready;
    assign res_m_axi_arready=residual_selected && m_axi_arready;
    assign ffn_m_axi_rdata=m_axi_rdata;
    assign res_m_axi_rdata=m_axi_rdata;
    assign ffn_m_axi_rresp=m_axi_rresp;
    assign res_m_axi_rresp=m_axi_rresp;
    assign ffn_m_axi_rid=m_axi_rid;
    assign res_m_axi_rid=m_axi_rid;
    assign ffn_m_axi_rlast=m_axi_rlast;
    assign res_m_axi_rlast=m_axi_rlast;
    assign ffn_m_axi_rvalid=!residual_selected && m_axi_rvalid;
    assign res_m_axi_rvalid=residual_selected && m_axi_rvalid;
    assign m_axi_rready=residual_selected ? res_m_axi_rready:ffn_m_axi_rready;
    assign m_axi_awaddr=residual_selected ? res_m_axi_awaddr:ffn_m_axi_awaddr;
    assign m_axi_awlen=residual_selected ? res_m_axi_awlen:ffn_m_axi_awlen;
    assign m_axi_awsize=residual_selected ? res_m_axi_awsize:ffn_m_axi_awsize;
    assign m_axi_awprot=residual_selected ? res_m_axi_awprot:ffn_m_axi_awprot;
    assign m_axi_awburst=residual_selected ? res_m_axi_awburst:ffn_m_axi_awburst;
    assign m_axi_awid=residual_selected ? res_m_axi_awid:ffn_m_axi_awid;
    assign m_axi_awcache=residual_selected ? res_m_axi_awcache:ffn_m_axi_awcache;
    assign m_axi_awlock=residual_selected ? res_m_axi_awlock:ffn_m_axi_awlock;
    assign m_axi_awvalid=residual_selected ? res_m_axi_awvalid:ffn_m_axi_awvalid;
    assign ffn_m_axi_awready=!residual_selected && m_axi_awready;
    assign res_m_axi_awready=residual_selected && m_axi_awready;
    assign m_axi_wdata=residual_selected ? res_m_axi_wdata:ffn_m_axi_wdata;
    assign m_axi_wstrb=residual_selected ? res_m_axi_wstrb:ffn_m_axi_wstrb;
    assign m_axi_wlast=residual_selected ? res_m_axi_wlast:ffn_m_axi_wlast;
    assign m_axi_wvalid=residual_selected ? res_m_axi_wvalid:ffn_m_axi_wvalid;
    assign ffn_m_axi_wready=!residual_selected && m_axi_wready;
    assign res_m_axi_wready=residual_selected && m_axi_wready;
    assign ffn_m_axi_bid=m_axi_bid;
    assign res_m_axi_bid=m_axi_bid;
    assign ffn_m_axi_bresp=m_axi_bresp;
    assign res_m_axi_bresp=m_axi_bresp;
    assign ffn_m_axi_bvalid=!residual_selected && m_axi_bvalid;
    assign res_m_axi_bvalid=residual_selected && m_axi_bvalid;
    assign m_axi_bready=residual_selected ? res_m_axi_bready:ffn_m_axi_bready;
    ntrpu_llama32_owned_ffn_r1 #(.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.INITIAL_TAG(INITIAL_TAG)) ffn (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(ffn_command_valid),
        .command_ready(ffn_command_ready),
        .command_sequence(ffn_command_sequence),
        .command_epoch(ffn_command_epoch),
        .expected_epoch(expected_epoch),
        .command_source_valid(ffn_command_source_valid),
        .command_source_sequence(ffn_command_source_sequence),
        .command_source_epoch(ffn_command_source_epoch),
        .command_addr(ffn_command_addr),
        .command_base(ffn_command_base),
        .command_bytes(ffn_command_bytes),
        .command_residual_address(ffn_command_residual_address),
        .command_residual_base(ffn_command_residual_base),
        .command_residual_bytes(ffn_command_residual_bytes),
        .cancel_valid(ffn_cancel_valid),
        .cancel_ready(ffn_cancel_ready),
        .busy(ffn_busy),
        .completion_valid(ffn_completion_valid),
        .completion_ready(ffn_completion_ready),
        .completion_status(ffn_completion_status),
        .completion_child_status(ffn_completion_child_status),
        .completion_stages(ffn_completion_stages),
        .completion_phase(ffn_completion_phase),
        .completion_bytes(ffn_completion_bytes),
        .completion_stage_bytes(ffn_completion_stage_bytes),
        .completion_hls_status(ffn_completion_hls_status),
        .completion_sequence(ffn_completion_sequence),
        .completion_epoch(ffn_completion_epoch),
        .result_valid(ffn_result_valid),
        .result_ffn_address(ffn_result_ffn_address),
        .result_residual_address(ffn_result_residual_address),
        .result_ffn_bytes(ffn_result_ffn_bytes),
        .result_residual_bytes(ffn_result_residual_bytes),
        .tags_exhausted(ffn_tags_exhausted),
        .activation_selected(ffn_activation_selected),
        .m_axi_araddr(ffn_m_axi_araddr),
        .m_axi_arlen(ffn_m_axi_arlen),
        .m_axi_arsize(ffn_m_axi_arsize),
        .m_axi_arprot(ffn_m_axi_arprot),
        .m_axi_arburst(ffn_m_axi_arburst),
        .m_axi_arid(ffn_m_axi_arid),
        .m_axi_arcache(ffn_m_axi_arcache),
        .m_axi_arlock(ffn_m_axi_arlock),
        .m_axi_arvalid(ffn_m_axi_arvalid),
        .m_axi_arready(ffn_m_axi_arready),
        .m_axi_rdata(ffn_m_axi_rdata),
        .m_axi_rresp(ffn_m_axi_rresp),
        .m_axi_rid(ffn_m_axi_rid),
        .m_axi_rlast(ffn_m_axi_rlast),
        .m_axi_rvalid(ffn_m_axi_rvalid),
        .m_axi_rready(ffn_m_axi_rready),
        .m_axi_awaddr(ffn_m_axi_awaddr),
        .m_axi_awlen(ffn_m_axi_awlen),
        .m_axi_awsize(ffn_m_axi_awsize),
        .m_axi_awprot(ffn_m_axi_awprot),
        .m_axi_awburst(ffn_m_axi_awburst),
        .m_axi_awid(ffn_m_axi_awid),
        .m_axi_awcache(ffn_m_axi_awcache),
        .m_axi_awlock(ffn_m_axi_awlock),
        .m_axi_awvalid(ffn_m_axi_awvalid),
        .m_axi_awready(ffn_m_axi_awready),
        .m_axi_wdata(ffn_m_axi_wdata),
        .m_axi_wstrb(ffn_m_axi_wstrb),
        .m_axi_wlast(ffn_m_axi_wlast),
        .m_axi_wvalid(ffn_m_axi_wvalid),
        .m_axi_wready(ffn_m_axi_wready),
        .m_axi_bid(ffn_m_axi_bid),
        .m_axi_bresp(ffn_m_axi_bresp),
        .m_axi_bvalid(ffn_m_axi_bvalid),
        .m_axi_bready(ffn_m_axi_bready)
    );
    ntrpu_residual_real_r1 #(.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)) res (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(res_command_valid),
        .command_ready(res_command_ready),
        .command_sequence(res_command_sequence),
        .command_epoch(res_command_epoch),
        .expected_epoch(expected_epoch),
        .command_source_sequence(res_command_source_sequence),
        .command_source_epoch(res_command_source_epoch),
        .command_source_valid(res_command_source_valid),
        .command_input_address(res_command_input_address),
        .command_input_base(res_command_input_base),
        .command_input_bytes(res_command_input_bytes),
        .command_branch_address(res_command_branch_address),
        .command_branch_base(res_command_branch_base),
        .command_branch_bytes(res_command_branch_bytes),
        .command_residual_address(res_command_residual_address),
        .command_residual_base(res_command_residual_base),
        .command_residual_bytes(res_command_residual_bytes),
        .cancel_valid(res_cancel_valid),
        .cancel_ready(res_cancel_ready),
        .busy(res_busy),
        .completion_valid(res_completion_valid),
        .completion_ready(res_completion_ready),
        .completion_status(res_completion_status),
        .completion_hls_status(res_completion_hls_status),
        .completion_sequence(res_completion_sequence),
        .completion_epoch(res_completion_epoch),
        .completion_read_beats(res_completion_read_beats),
        .completion_residual_beats(res_completion_residual_beats),
        .result_valid(res_result_valid),
        .result_residual_address(res_result_residual_address),
        .result_residual_bytes(res_result_residual_bytes),
        .m_axi_araddr(res_m_axi_araddr),
        .m_axi_arlen(res_m_axi_arlen),
        .m_axi_arsize(res_m_axi_arsize),
        .m_axi_arprot(res_m_axi_arprot),
        .m_axi_arburst(res_m_axi_arburst),
        .m_axi_arid(res_m_axi_arid),
        .m_axi_arcache(res_m_axi_arcache),
        .m_axi_arlock(res_m_axi_arlock),
        .m_axi_arvalid(res_m_axi_arvalid),
        .m_axi_arready(res_m_axi_arready),
        .m_axi_rdata(res_m_axi_rdata),
        .m_axi_rresp(res_m_axi_rresp),
        .m_axi_rid(res_m_axi_rid),
        .m_axi_rlast(res_m_axi_rlast),
        .m_axi_rvalid(res_m_axi_rvalid),
        .m_axi_rready(res_m_axi_rready),
        .m_axi_awaddr(res_m_axi_awaddr),
        .m_axi_awlen(res_m_axi_awlen),
        .m_axi_awsize(res_m_axi_awsize),
        .m_axi_awprot(res_m_axi_awprot),
        .m_axi_awburst(res_m_axi_awburst),
        .m_axi_awid(res_m_axi_awid),
        .m_axi_awcache(res_m_axi_awcache),
        .m_axi_awlock(res_m_axi_awlock),
        .m_axi_awvalid(res_m_axi_awvalid),
        .m_axi_awready(res_m_axi_awready),
        .m_axi_wdata(res_m_axi_wdata),
        .m_axi_wstrb(res_m_axi_wstrb),
        .m_axi_wlast(res_m_axi_wlast),
        .m_axi_wvalid(res_m_axi_wvalid),
        .m_axi_wready(res_m_axi_wready),
        .m_axi_bid(res_m_axi_bid),
        .m_axi_bresp(res_m_axi_bresp),
        .m_axi_bvalid(res_m_axi_bvalid),
        .m_axi_bready(res_m_axi_bready)
    );
    ntrpu_ffn_residual_handoff_r1  handoff (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(handoff_command_valid),
        .command_ready(handoff_command_ready),
        .command_sequence(handoff_command_sequence),
        .command_epoch(handoff_command_epoch),
        .expected_epoch(expected_epoch),
        .command_output_address(handoff_command_output_address),
        .command_output_base(handoff_command_output_base),
        .command_output_bytes(handoff_command_output_bytes),
        .cancel_valid(handoff_cancel_valid),
        .cancel_ready(handoff_cancel_ready),
        .busy(handoff_busy),
        .source_completion_valid(handoff_source_completion_valid),
        .source_completion_ready(handoff_source_completion_ready),
        .source_completion_status(handoff_source_completion_status),
        .source_sequence(handoff_source_sequence),
        .source_epoch(handoff_source_epoch),
        .source_result_valid(handoff_source_result_valid),
        .source_residual_address(handoff_source_residual_address),
        .source_ffn_address(handoff_source_ffn_address),
        .source_residual_bytes(handoff_source_residual_bytes),
        .source_ffn_bytes(handoff_source_ffn_bytes),
        .child_command_valid(handoff_child_command_valid),
        .child_command_ready(handoff_child_command_ready),
        .child_command_sequence(handoff_child_command_sequence),
        .child_command_epoch(handoff_child_command_epoch),
        .child_source_sequence(handoff_child_source_sequence),
        .child_source_epoch(handoff_child_source_epoch),
        .child_source_valid(handoff_child_source_valid),
        .child_input_address(handoff_child_input_address),
        .child_input_base(handoff_child_input_base),
        .child_input_bytes(handoff_child_input_bytes),
        .child_branch_address(handoff_child_branch_address),
        .child_branch_base(handoff_child_branch_base),
        .child_branch_bytes(handoff_child_branch_bytes),
        .child_output_address(handoff_child_output_address),
        .child_output_base(handoff_child_output_base),
        .child_output_bytes(handoff_child_output_bytes),
        .child_cancel_valid(handoff_child_cancel_valid),
        .child_completion_valid(handoff_child_completion_valid),
        .child_completion_ready(handoff_child_completion_ready),
        .child_completion_status(handoff_child_completion_status),
        .child_hls_status(handoff_child_hls_status),
        .child_sequence(handoff_child_sequence),
        .child_epoch(handoff_child_epoch),
        .child_read_beats(handoff_child_read_beats),
        .child_write_beats(handoff_child_write_beats),
        .child_result_valid(handoff_child_result_valid),
        .child_result_address(handoff_child_result_address),
        .child_result_bytes(handoff_child_result_bytes),
        .residual_bus_owned(handoff_residual_bus_owned),
        .completion_valid(handoff_completion_valid),
        .completion_ready(handoff_completion_ready),
        .completion_status(handoff_completion_status),
        .completion_source_status(handoff_completion_source_status),
        .completion_child_status(handoff_completion_child_status),
        .completion_hls_status(handoff_completion_hls_status),
        .completion_sequence(handoff_completion_sequence),
        .completion_epoch(handoff_completion_epoch),
        .completion_read_beats(handoff_completion_read_beats),
        .completion_write_beats(handoff_completion_write_beats),
        .result_valid(handoff_result_valid),
        .result_address(handoff_result_address),
        .result_bytes(handoff_result_bytes)
    );
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;sequence_reg<=0;epoch_reg<=0;source_sequence<=0;source_epoch<=0;source_valid<=0;
            cancel_seen<=0;ffn_launched<=0;fault<=0;region_index<=0;
            residual_address<=0;residual_base<=0;residual_bytes<=0;output_address<=0;output_base<=0;output_bytes<=0;
            for(integer i=0;i<8;i++)begin address[i]<=0;region_base[i]<=0;region_bytes[i]<=0;end
        end else begin
            if(busy && state!=LOCAL_RESPONSE)begin
                fault<=fault_now;
                if(cancel_valid)cancel_seen<=1;
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    sequence_reg<=command_sequence;epoch_reg<=command_epoch;source_sequence<=command_source_sequence;
                    source_epoch<=command_source_epoch;source_valid<=command_source_valid;
                    cancel_seen<=cancel_valid;ffn_launched<=0;region_index<=0;
                    if(!command_source_valid || command_sequence!=command_source_sequence || command_epoch!=command_source_epoch || command_epoch!=expected_epoch)fault<=2;
                    else if(cancel_valid)fault<=7;
                    else fault<=0;
                    for(integer i=0;i<8;i++)begin address[i]<=command_addr[i];region_base[i]<=command_base[i];region_bytes[i]<=command_bytes[i];end
                    residual_address<=command_residual_address;residual_base<=command_residual_base;residual_bytes<=command_residual_bytes;
                    output_address<=command_output_address;output_base<=command_output_base;output_bytes<=command_output_bytes;
                    state<=CHECK_OUTPUT;
                end
                CHECK_OUTPUT:begin
                    if(fault_now!=0)state<=LOCAL_RESPONSE;
                    else if(output_bad)begin fault<=1;state<=LOCAL_RESPONSE;end
                    else state<=CHECK_REGIONS;
                end
                CHECK_REGIONS:begin
                    if(fault_now!=0)state<=LOCAL_RESPONSE;
                    else if(allocation_bad)begin fault<=1;state<=LOCAL_RESPONSE;end
                    else if(region_index==8)state<=FFN_OFFER;
                    else region_index<=region_index+1'b1;
                end
                FFN_OFFER:if(ffn_command_ready)begin ffn_launched<=1;state<=WAIT_FFN;end
                WAIT_FFN:if(handoff_command_ready)state<=WAIT_HANDOFF;
                WAIT_HANDOFF:if(!handoff_busy && !ffn_busy && !ffn_completion_valid && !res_busy && !res_completion_valid)state<=IDLE;
                LOCAL_RESPONSE:if(completion_ready)state<=IDLE;
                default:begin state<=IDLE;fault<=11;end
            endcase
        end
    end
endmodule
`default_nettype wire
