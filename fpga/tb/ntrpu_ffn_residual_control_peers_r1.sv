// NATIVE PROTOCOL PEERS ONLY. Never include this file in actual-arithmetic packages.
`timescale 1ns/1ps
`default_nettype none
module ntrpu_ffn_residual_probe_peer_r1 #(parameter integer PEER_ID=0) (
    input wire clk,rst_n,command_valid,admission_enable,cancel_valid,
    output wire command_ready,busy,completion_valid,
    input wire completion_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,read0,read1,write0,response_sequence_mask,
    input wire [3:0] injected_error,
    output wire [63:0] completion_sequence,completion_epoch,
    output logic [3:0] completion_status,
    output wire result_valid,
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
    typedef enum logic [3:0] {IDLE,AR0,R0,AR1,R1,SEND,BWAIT,DELAY,RESPONSE} state_t;
    state_t state;
    logic [63:0] seq,epoch,read_address0,read_address1,write_address;
    logic [511:0] input0,payload;
    logic aw_seen,w_seen,cancel_seen,revoked;
    logic [1:0] delay_count;
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && admission_enable;
    assign completion_valid=state==RESPONSE;
    assign completion_sequence=seq^response_sequence_mask;
    assign completion_epoch=epoch;
    assign result_valid=completion_valid && completion_status==0 && !revoked && !cancel_valid && expected_epoch==epoch;
    assign m_axi_araddr=state==AR1 ? read_address1:read_address0;
    assign m_axi_arlen=0;assign m_axi_arsize=6;assign m_axi_arburst=1;
    assign m_axi_arid=0;assign m_axi_arcache=2;assign m_axi_arprot=0;assign m_axi_arlock=0;
    assign m_axi_arvalid=state==AR0 || state==AR1;
    assign m_axi_rready=state==R0 || state==R1;
    assign m_axi_awaddr=write_address;assign m_axi_awlen=0;assign m_axi_awsize=6;assign m_axi_awburst=1;
    assign m_axi_awid=0;assign m_axi_awcache=2;assign m_axi_awprot=0;assign m_axi_awlock=0;
    assign m_axi_awvalid=state==SEND && !aw_seen;
    assign m_axi_wdata=payload;assign m_axi_wstrb='1;assign m_axi_wlast=1;assign m_axi_wvalid=state==SEND && !w_seen;
    assign m_axi_bready=state==BWAIT;
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;seq<=0;epoch<=0;read_address0<=0;read_address1<=0;write_address<=0;
            input0<=0;payload<=0;aw_seen<=0;w_seen<=0;cancel_seen<=0;revoked<=0;delay_count<=0;completion_status<=0;
        end else begin
            if(busy && (cancel_valid || expected_epoch!=epoch))begin cancel_seen<=1;revoked<=1;end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    seq<=command_sequence;epoch<=command_epoch;read_address0<=read0;read_address1<=read1;write_address<=write0;
                    aw_seen<=0;w_seen<=0;cancel_seen<=cancel_valid;revoked<=cancel_valid;delay_count<=0;completion_status<=0;state<=AR0;
                end
                AR0:if(m_axi_arready)state<=R0;
                R0:if(m_axi_rvalid)begin input0<=m_axi_rdata;state<=AR1;end
                AR1:if(m_axi_arready)state<=R1;
                R1:if(m_axi_rvalid)begin payload<=input0^m_axi_rdata^{16{PEER_ID==0 ? 32'h46:32'h52}};state<=SEND;end
                SEND:begin
                    if(m_axi_awready)aw_seen<=1;
                    if(m_axi_wready)w_seen<=1;
                    if((aw_seen || m_axi_awready) && (w_seen || m_axi_wready))state<=BWAIT;
                end
                BWAIT:if(m_axi_bvalid)state<=DELAY;
                DELAY:if(delay_count==2)begin
                    completion_status<=cancel_seen || cancel_valid || expected_epoch!=epoch ? 4'd7:injected_error;
                    state<=RESPONSE;
                end else delay_count<=delay_count+1'b1;
                RESPONSE:if(completion_ready)state<=IDLE;
                default:state<=IDLE;
            endcase
        end
    end
endmodule

module ntrpu_llama32_owned_ffn_r1 #(
    parameter integer WATCHDOG_CYCLES=65536, parameter logic [63:0] INITIAL_TAG=64'd0
) (
    input wire clk,
    input wire rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,
    input wire [63:0] command_epoch,
    input wire [63:0] expected_epoch,
    input wire command_source_valid,
    input wire [63:0] command_source_sequence,
    input wire [63:0] command_source_epoch,
    input wire [63:0] command_addr [0:7],
    input wire [63:0] command_base [0:7],
    input wire [63:0] command_bytes [0:7],
    input wire [63:0] command_residual_address,
    input wire [63:0] command_residual_base,
    input wire [63:0] command_residual_bytes,
    input wire cancel_valid,
    output wire cancel_ready,
    output wire busy,
    output wire completion_valid,
    input wire completion_ready,
    output wire [3:0] completion_status,
    output wire [3:0] completion_child_status,
    output wire [2:0] completion_stages,
    output wire [1:0] completion_phase,
    output wire [16:0] completion_bytes,
    output wire [17:0] completion_stage_bytes,
    output wire [31:0] completion_hls_status,
    output wire [63:0] completion_sequence,
    output wire [63:0] completion_epoch,
    output wire result_valid,
    output wire [63:0] result_ffn_address,
    output wire [63:0] result_residual_address,
    output wire [13:0] result_ffn_bytes,
    output wire [13:0] result_residual_bytes,
    output wire tags_exhausted,
    output wire activation_selected,
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
    logic admission_enable=1;
    logic [3:0] injected_error=0;
    logic [63:0] response_sequence_mask=0;
    logic [63:0] output_saved,residual_saved;
    wire peer_result;
    always_ff @(posedge clk)begin
        if(!rst_n)begin output_saved<=0;residual_saved<=0;end
        else if(command_valid && command_ready)begin
            output_saved<=command_addr[7];residual_saved<=command_residual_address;
        end
    end
    assign cancel_ready=busy;
    assign result_valid=peer_result;
    assign completion_hls_status=0;
    assign result_ffn_address=peer_result ? output_saved:64'd0;
    assign result_residual_address=peer_result ? residual_saved:64'd0;
    assign result_ffn_bytes=peer_result ? 14'd8192:14'd0;
    assign result_residual_bytes=peer_result ? 14'd8192:14'd0;
    assign completion_child_status=completion_status;
    assign completion_stages=4;assign completion_phase=3;assign completion_bytes=8192;assign completion_stage_bytes=155648;
    assign tags_exhausted=0;assign activation_selected=0;
    ntrpu_ffn_residual_probe_peer_r1 #(.PEER_ID(0)) peer(
        .clk,.rst_n,.command_valid,.command_ready,.busy,.admission_enable,.cancel_valid,.completion_valid,.completion_ready,
        .command_sequence,.command_epoch,.expected_epoch,.response_sequence_mask,.injected_error,
        .completion_sequence,.completion_epoch,.completion_status,.result_valid(peer_result),
        .read0(command_addr[0]),.read1(command_addr[1]),.write0(command_addr[7]),
        .m_axi_araddr,
        .m_axi_arlen,
        .m_axi_arsize,
        .m_axi_arprot,
        .m_axi_arburst,
        .m_axi_arid,
        .m_axi_arcache,
        .m_axi_arlock,
        .m_axi_arvalid,
        .m_axi_arready,
        .m_axi_rdata,
        .m_axi_rresp,
        .m_axi_rid,
        .m_axi_rlast,
        .m_axi_rvalid,
        .m_axi_rready,
        .m_axi_awaddr,
        .m_axi_awlen,
        .m_axi_awsize,
        .m_axi_awprot,
        .m_axi_awburst,
        .m_axi_awid,
        .m_axi_awcache,
        .m_axi_awlock,
        .m_axi_awvalid,
        .m_axi_awready,
        .m_axi_wdata,
        .m_axi_wstrb,
        .m_axi_wlast,
        .m_axi_wvalid,
        .m_axi_wready,
        .m_axi_bid,
        .m_axi_bresp,
        .m_axi_bvalid,
        .m_axi_bready
    );
endmodule

module ntrpu_residual_real_r1 #(
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
    logic admission_enable=1;
    logic [3:0] injected_error=0;
    logic [63:0] response_sequence_mask=0;
    logic [63:0] output_saved,residual_saved;
    wire peer_result;
    always_ff @(posedge clk)begin
        if(!rst_n)begin output_saved<=0;residual_saved<=0;end
        else if(command_valid && command_ready)begin
            output_saved<=command_residual_address;residual_saved<=command_input_address;
        end
    end
    assign cancel_ready=busy;
    assign result_valid=peer_result;
    assign completion_hls_status=0;
    assign result_residual_address=peer_result ? output_saved:64'd0;
    assign result_residual_bytes=peer_result ? 14'd8192:14'd0;
    assign completion_read_beats=256;assign completion_residual_beats=128;
    ntrpu_ffn_residual_probe_peer_r1 #(.PEER_ID(1)) peer(
        .clk,.rst_n,.command_valid,.command_ready,.busy,.admission_enable,.cancel_valid,.completion_valid,.completion_ready,
        .command_sequence,.command_epoch,.expected_epoch,.response_sequence_mask,.injected_error,
        .completion_sequence,.completion_epoch,.completion_status,.result_valid(peer_result),
        .read0(command_input_address),.read1(command_branch_address),.write0(command_residual_address),
        .m_axi_araddr,
        .m_axi_arlen,
        .m_axi_arsize,
        .m_axi_arprot,
        .m_axi_arburst,
        .m_axi_arid,
        .m_axi_arcache,
        .m_axi_arlock,
        .m_axi_arvalid,
        .m_axi_arready,
        .m_axi_rdata,
        .m_axi_rresp,
        .m_axi_rid,
        .m_axi_rlast,
        .m_axi_rvalid,
        .m_axi_rready,
        .m_axi_awaddr,
        .m_axi_awlen,
        .m_axi_awsize,
        .m_axi_awprot,
        .m_axi_awburst,
        .m_axi_awid,
        .m_axi_awcache,
        .m_axi_awlock,
        .m_axi_awvalid,
        .m_axi_awready,
        .m_axi_wdata,
        .m_axi_wstrb,
        .m_axi_wlast,
        .m_axi_wvalid,
        .m_axi_wready,
        .m_axi_bid,
        .m_axi_bresp,
        .m_axi_bvalid,
        .m_axi_bready
    );
endmodule
`default_nettype wire
