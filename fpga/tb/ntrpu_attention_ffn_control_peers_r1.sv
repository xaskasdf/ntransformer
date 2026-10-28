// NATIVE PROTOCOL PEERS ONLY. Never include this file in actual-arithmetic packages.
`timescale 1ns/1ps
`default_nettype none
module ntrpu_attention_ffn_probe_peer_r1 #(parameter integer PEER_ID=0) (
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

module ntrpu_attention_post_path_r2 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144,
    parameter logic [63:0] INITIAL_QUANT_TAG=64'd0,
    parameter logic [63:0] INITIAL_PROJECTION_TAG=64'd0
) (
    input wire logic clk,
    input wire logic rst_n,
    input wire logic bind_valid,
    output logic bind_ready,
    input wire logic [63:0] bind_cache_base,
    input wire logic [63:0] bind_region_base,
    input wire logic [63:0] bind_region_bytes,
    input wire logic [31:0] bind_capacity,
    input wire logic [63:0] bind_sequence,
    input wire logic [63:0] bind_epoch,
    input wire logic [63:0] expected_epoch,
    output logic cache_valid,
    output logic [63:0] cache_base,
    output logic [63:0] cache_region_base,
    output logic [63:0] cache_region_bytes,
    output logic [63:0] cache_sequence,
    output logic [63:0] cache_epoch,
    output logic [31:0] cache_capacity,
    input wire logic append_valid,
    output logic append_ready,
    output logic append_prepared,
    input wire logic [31:0] append_layer,
    input wire logic [31:0] append_position,
    input wire logic [63:0] append_sequence,
    input wire logic [63:0] append_epoch,
    input wire logic data_valid,
    output logic data_ready,
    input wire logic [511:0] data_bits,
    input wire logic [5:0] data_index,
    input wire logic data_last,
    input wire logic [63:0] data_sequence,
    input wire logic [63:0] data_epoch,
    input wire logic commit_valid,
    output logic commit_ready,
    input wire logic [3:0] commit_status,
    input wire logic [63:0] commit_sequence,
    input wire logic [63:0] commit_epoch,
    input wire logic append_cancel_valid,
    output logic append_cancel_ready,
    output logic append_done_valid,
    input wire logic append_done_ready,
    output logic append_done_bind,
    output logic [3:0] append_done_status,
    output logic [12:0] append_done_bytes,
    output logic [31:0] append_done_committed_tokens,
    output logic [63:0] append_done_sequence,
    output logic [63:0] append_done_epoch,
    output logic lease_active,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize,
    output logic [2:0] m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid,
    output logic [3:0] m_axi_awcache,
    output logic m_axi_awlock,
    output logic m_axi_awvalid,
    input wire logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,
    output logic m_axi_wvalid,
    input wire logic m_axi_wready,
    input wire logic [3:0] m_axi_bid,
    input wire logic [1:0] m_axi_bresp,
    input wire logic m_axi_bvalid,
    output logic m_axi_bready,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,
    output logic [2:0] m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,
    output logic [3:0] m_axi_arcache,
    output logic m_axi_arlock,
    output logic m_axi_arvalid,
    input wire logic m_axi_arready,
    input wire logic [511:0] m_axi_rdata,
    input wire logic [1:0] m_axi_rresp,
    input wire logic [3:0] m_axi_rid,
    input wire logic m_axi_rlast,
    input wire logic m_axi_rvalid,
    output logic m_axi_rready,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [31:0] command_position,
    input wire logic [31:0] command_layer,
    input wire logic [63:0] command_sequence,
    input wire logic [63:0] command_epoch,
    input wire logic cancel_valid,
    output logic cancel_ready,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic result_valid,
    input wire logic [63:0] command_lease_base,
    input wire logic q_valid,
    output logic q_ready,
    input wire logic [511:0] q_data,
    input wire logic [7:0] q_index,
    input wire logic q_last,
    input wire logic [63:0] q_sequence,
    input wire logic [63:0] q_epoch,
    output logic [63:0] completion_sequence,
    output logic [63:0] completion_epoch,
    output logic cache_lock,
    output logic maintenance_busy,
    output logic metadata_fault,
    output logic [31:0] committed_tokens,
    output logic busy,
    output logic tag_exhausted,
    output logic [31:0] completion_hls_status,
    output logic [63:0] result_address,
    output logic [16:0] result_bytes,
    input wire logic [63:0] command_weight_addr,
    input wire logic [63:0] command_weight_region_base,
    input wire logic [63:0] command_weight_region_bytes,
    input wire logic [63:0] command_activation_addr,
    input wire logic [63:0] command_activation_region_base,
    input wire logic [63:0] command_activation_region_bytes,
    input wire logic [63:0] command_output_addr,
    input wire logic [63:0] command_output_region_base,
    input wire logic [63:0] command_output_region_bytes,
    input wire logic [63:0] command_post_input_address,
    input wire logic [63:0] command_post_input_base,
    input wire logic [63:0] command_post_input_bytes,
    input wire logic [63:0] command_post_weight_address,
    input wire logic [63:0] command_post_weight_base,
    input wire logic [63:0] command_post_weight_bytes,
    input wire logic [63:0] command_post_residual_address,
    input wire logic [63:0] command_post_residual_base,
    input wire logic [63:0] command_post_residual_bytes,
    input wire logic [63:0] command_post_ffn_address,
    input wire logic [63:0] command_post_ffn_base,
    input wire logic [63:0] command_post_ffn_bytes,
    output logic [31:0] completion_post_hls_status,
    output logic [9:0] completion_post_read_beats,
    output logic [7:0] completion_residual_beats,
    output logic [6:0] completion_ffn_beats,
    output logic [63:0] result_residual_address,
    output logic [63:0] result_ffn_address,
    output logic [13:0] result_residual_bytes,
    output logic [12:0] result_ffn_bytes
);
    // Deliberately controlled protocol stimulus; no attention/FFN arithmetic.
    logic admission_enable=1,mute_completion=0,mute_result=0;
    logic [3:0] injected_error=0,shape_fault=0;
    logic [63:0] response_sequence_mask=0;
    wire peer_result,peer_completion_valid;
    logic [63:0] output_saved,residual_saved,ffn_saved;
    assign completion_valid=peer_completion_valid && !mute_completion;
    assign result_valid=peer_result && !mute_result && !mute_completion;
    assign cancel_ready=busy;
    logic cache_invalid=0,metadata_injected=0;
    logic [63:0] cache_base_mask=0;
    logic [63:0] saved_cache_base,saved_region_base,saved_region_bytes,saved_cache_sequence,saved_cache_epoch;
    logic [31:0] saved_capacity;
    logic cache_bound;
    assign cache_valid=cache_bound && !cache_invalid;
    assign cache_base=saved_cache_base^cache_base_mask;
    assign cache_region_base=saved_region_base;assign cache_region_bytes=saved_region_bytes;
    assign cache_sequence=saved_cache_sequence;assign cache_epoch=saved_cache_epoch;assign cache_capacity=saved_capacity;
    assign bind_ready=!busy && !append_done_valid;
    assign append_ready=bind_ready && !bind_valid;
    assign data_ready=!busy;assign commit_ready=!busy;assign append_cancel_ready=!busy;
    assign append_prepared=0;assign lease_active=0;
    assign cache_lock=busy;assign maintenance_busy=append_done_valid;
    assign metadata_fault=metadata_injected;assign committed_tokens=cache_bound ? 1:0;
    assign q_ready=busy && !peer_completion_valid;
    assign tag_exhausted=0;
    assign completion_hls_status=shape_fault==3 ? 1:0;
    assign completion_post_hls_status=0;
    assign completion_post_read_beats=shape_fault==4 ? 383:384;
    assign completion_residual_beats=128;assign completion_ffn_beats=64;
    assign result_address=result_valid ? output_saved:64'd0;
    assign result_bytes=result_valid ? 17'd8192:17'd0;
    assign result_residual_address=result_valid ? residual_saved:64'd0;
    assign result_ffn_address=result_valid ? ffn_saved+(shape_fault==1 ? 64'd64:64'd0):64'd0;
    assign result_residual_bytes=result_valid ? (shape_fault==2 ? 14'd8191:14'd8192):14'd0;
    assign result_ffn_bytes=result_valid ? 13'd4096:13'd0;
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            cache_bound<=0;saved_cache_base<=0;saved_region_base<=0;saved_region_bytes<=0;saved_cache_sequence<=0;saved_cache_epoch<=0;saved_capacity<=0;
            append_done_valid<=0;append_done_bind<=0;append_done_status<=0;append_done_bytes<=0;append_done_committed_tokens<=0;append_done_sequence<=0;append_done_epoch<=0;
            output_saved<=0;residual_saved<=0;ffn_saved<=0;
        end else begin
            if(append_done_valid && append_done_ready)append_done_valid<=0;
            if(bind_valid && bind_ready)begin
                cache_bound<=1;saved_cache_base<=bind_cache_base;saved_region_base<=bind_region_base;saved_region_bytes<=bind_region_bytes;
                saved_cache_sequence<=bind_sequence;saved_cache_epoch<=bind_epoch;saved_capacity<=bind_capacity;
                append_done_valid<=1;append_done_bind<=1;append_done_status<=0;append_done_bytes<=0;append_done_committed_tokens<=0;append_done_sequence<=bind_sequence;append_done_epoch<=bind_epoch;
            end else if(append_valid && append_ready)begin
                append_done_valid<=1;append_done_bind<=0;append_done_status<=0;append_done_bytes<=4096;append_done_committed_tokens<=append_position+1;append_done_sequence<=append_sequence;append_done_epoch<=append_epoch;
            end
            if(command_valid && command_ready)begin output_saved<=command_output_addr;residual_saved<=command_post_residual_address;ffn_saved<=command_post_ffn_address;end
        end
    end
    ntrpu_attention_ffn_probe_peer_r1 #(.PEER_ID(0)) peer(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(command_valid),
        .command_ready(command_ready),
        .busy(busy),
        .admission_enable(admission_enable),
        .cancel_valid(cancel_valid),
        .completion_valid(peer_completion_valid),
        .completion_ready(completion_ready),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .expected_epoch(expected_epoch),
        .response_sequence_mask(response_sequence_mask),
        .injected_error(injected_error),
        .completion_sequence(completion_sequence),
        .completion_epoch(completion_epoch),
        .completion_status(completion_status),
        .result_valid(peer_result),
        .read0(command_post_input_address),
        .read1(command_weight_addr),
        .write0(command_post_ffn_address),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awid(m_axi_awid),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
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
endmodule

module ntrpu_llama32_ffn_residual_r3 #(
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
    // Deliberately controlled protocol stimulus; no attention/FFN arithmetic.
    logic admission_enable=1,mute_completion=0,mute_result=0;
    logic [3:0] injected_error=0,shape_fault=0;
    logic [63:0] response_sequence_mask=0;
    wire peer_result,peer_completion_valid;
    logic [63:0] output_saved,residual_saved,ffn_saved;
    assign completion_valid=peer_completion_valid && !mute_completion;
    assign result_valid=peer_result && !mute_result && !mute_completion;
    assign cancel_ready=busy;
    assign completion_ffn_status=0;assign completion_ffn_child_status=0;assign completion_residual_status=0;
    assign completion_ffn_stages=4;assign completion_ffn_phase=3;assign completion_ffn_bytes=8192;assign completion_ffn_stage_bytes=155648;
    assign completion_ffn_hls_status=0;assign completion_hls_status=shape_fault==1 ? 1:0;
    assign completion_read_beats=shape_fault==2 ? 255:256;assign completion_write_beats=128;
    assign result_address=result_valid ? output_saved:64'd0;assign result_bytes=result_valid ? 14'd8192:14'd0;
    assign tags_exhausted=0;assign activation_selected=0;assign residual_selected=busy;
    always_ff @(posedge clk)begin
        if(!rst_n)begin output_saved<=0;residual_saved<=0;ffn_saved<=0;end
        else if(command_valid && command_ready)begin output_saved<=command_output_address;residual_saved<=command_residual_address;ffn_saved<=command_addr[0];end
    end
    ntrpu_attention_ffn_probe_peer_r1 #(.PEER_ID(1)) peer(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(command_valid),
        .command_ready(command_ready),
        .busy(busy),
        .admission_enable(admission_enable),
        .cancel_valid(cancel_valid),
        .completion_valid(peer_completion_valid),
        .completion_ready(completion_ready),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .expected_epoch(expected_epoch),
        .response_sequence_mask(response_sequence_mask),
        .injected_error(injected_error),
        .completion_sequence(completion_sequence),
        .completion_epoch(completion_epoch),
        .completion_status(completion_status),
        .result_valid(peer_result),
        .read0(command_addr[0]),
        .read1(command_residual_address),
        .write0(command_output_address),
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
        .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awid(m_axi_awid),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready)
    );
endmodule
`default_nettype wire
