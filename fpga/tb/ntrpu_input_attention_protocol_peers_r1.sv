`timescale 1ns/1ps
`default_nettype none
module ntrpu_input_qkv_real_r1 #(
    parameter integer AXI_WATCHDOG_CYCLES=65536,
    parameter integer NORM_HLS_WATCHDOG_CYCLES=262144,
    parameter integer MATRIX_TIMEOUT_CYCLES=16777216,
    parameter integer HANDOFF_TIMEOUT_CYCLES=67108864,
    parameter logic [63:0] INITIAL_NORM_TAG=64'd0,
    parameter logic [63:0] INITIAL_QKV_TAG=64'd0
) (
    input wire clk,
    input wire rst_n,
    input wire command_valid,
    output logic command_ready,
    input wire [63:0] command_sequence,
    input wire [63:0] command_epoch,
    input wire [63:0] expected_epoch,
    input wire [63:0] command_source_sequence,
    input wire [63:0] command_source_epoch,
    input wire command_source_valid,
    input wire [63:0] command_addr [0:8],
    input wire [63:0] command_base [0:8],
    input wire [63:0] command_bytes [0:8],
    input wire cancel_valid,
    output logic cancel_ready,
    output logic busy,
    output logic completion_valid,
    input wire completion_ready,
    output logic [63:0] completion_sequence,
    output logic [63:0] completion_epoch,
    output logic [3:0] completion_status,
    output logic [3:0] completion_norm_status,
    output logic [3:0] completion_qkv_status,
    output logic [3:0] completion_matrix_status,
    output logic [31:0] completion_norm_hls_status,
    output logic [9:0] completion_norm_read_beats,
    output logic [6:0] completion_norm_records,
    output logic [1:0] completion_phase,
    output logic [1:0] completion_matrices,
    output logic [14:0] completion_total_bytes,
    output logic completion_revoked,
    output logic result_valid,
    output logic [63:0] result_address [0:2],
    output logic [13:0] result_bytes [0:2],
    output logic norm_tag_exhausted,
    output logic qkv_tags_exhausted,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,
    output logic [2:0] m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,
    output logic [3:0] m_axi_arcache,
    output logic m_axi_arlock,
    output logic m_axi_arvalid,
    input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire [3:0] m_axi_rid,
    input wire m_axi_rlast,
    input wire m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize,
    output logic [2:0] m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid,
    output logic [3:0] m_axi_awcache,
    output logic m_axi_awlock,
    output logic m_axi_awvalid,
    input wire m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,
    output logic m_axi_wvalid,
    input wire m_axi_wready,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output logic m_axi_bready
);
    assign norm_tag_exhausted='0;
    assign qkv_tags_exhausted='0;
    assign m_axi_araddr='0;
    assign m_axi_arlen='0;
    assign m_axi_arsize='0;
    assign m_axi_arprot='0;
    assign m_axi_arburst='0;
    assign m_axi_arid='0;
    assign m_axi_arcache='0;
    assign m_axi_arlock='0;
    assign m_axi_arvalid='0;
    assign m_axi_rready='0;
    assign m_axi_awaddr='0;
    assign m_axi_awlen='0;
    assign m_axi_awsize='0;
    assign m_axi_awprot='0;
    assign m_axi_awburst='0;
    assign m_axi_awid='0;
    assign m_axi_awcache='0;
    assign m_axi_awlock='0;
    assign m_axi_awvalid='0;
    assign m_axi_wdata='0;
    assign m_axi_wstrb='0;
    assign m_axi_wlast='0;
    assign m_axi_wvalid='0;
    assign m_axi_bready='0;

    integer test_delay=5,test_drain=7,test_status=0,test_beats=256;
    logic [63:0] test_sequence_mask=0;
    logic test_revoked=0;
    integer delay_left=0,drain_left=0;
    logic [63:0] saved_sequence,saved_epoch;
    logic cancelled;
    assign command_ready=!busy && !completion_valid;
    assign cancel_ready=busy && !completion_valid && !cancelled;
    assign completion_sequence=saved_sequence^test_sequence_mask;
    assign completion_epoch=saved_epoch;
    assign completion_revoked=test_revoked;
    assign result_valid=completion_valid && completion_status==0 && !test_revoked;
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            busy<=0;completion_valid<=0;saved_sequence<=0;saved_epoch<=0;cancelled<=0;delay_left<=0;drain_left<=0;
            completion_status<=0;completion_norm_status<=0;completion_qkv_status<=0;completion_matrix_status<=0;
            completion_norm_hls_status<=0;completion_norm_read_beats<=0;completion_norm_records<=0;
            completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
            for(integer i=0;i<3;i++)begin result_address[i]<=0;result_bytes[i]<=0;end
        end else begin
            if(command_valid && command_ready)begin
                busy<=1;saved_sequence<=command_sequence;saved_epoch<=command_epoch;cancelled<=0;delay_left<=test_delay;
                for(integer i=0;i<3;i++)begin result_address[i]<=command_addr[6+i];result_bytes[i]<=i==0 ? 8192:2048;end
            end
            if(cancel_valid && cancel_ready)begin cancelled<=1;delay_left<=2;end
            else if(delay_left>0)begin
                delay_left<=delay_left-1;
                if(delay_left==1)begin
                    completion_valid<=1;completion_status<=cancelled ? 7:4'(test_status);
                    completion_norm_status<=0;completion_qkv_status<=0;completion_matrix_status<=0;
                    completion_norm_hls_status<=0;completion_norm_read_beats<=10'(test_beats);completion_norm_records<=64;
                    completion_phase<=2;completion_matrices<=3;completion_total_bytes<=12288;
                end
            end
            if(completion_valid && completion_ready)begin completion_valid<=0;drain_left<=test_drain;end
            if(drain_left>0)begin drain_left<=drain_left-1;if(drain_left==1)busy<=0;end
        end
    end
endmodule

module ntrpu_rope_attention_query_r1 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter integer HLS_WATCHDOG_CYCLES=262144
) (
    input wire clk,
    input wire rst_n,
    input wire [63:0] command_sequence,
    input wire [63:0] command_epoch,
    input wire [63:0] command_lease_base,
    input wire [63:0] expected_epoch,
    input wire [31:0] command_layer,
    input wire [31:0] command_position,
    input wire cancel_valid,
    output logic cancel_ready,
    output logic completion_valid,
    input wire completion_ready,
    output logic [3:0] completion_status,
    output logic [3:0] completion_append_status,
    output logic [3:0] completion_root_status,
    output logic [63:0] completion_sequence,
    output logic [63:0] completion_epoch,
    output logic [31:0] completion_layer,
    output logic [31:0] completion_position,
    output logic [6:0] completion_kv_frames,
    output logic [7:0] completion_q_frames,
    output logic [12:0] completion_kv_bytes,
    output logic [31:0] completion_kv_tokens,
    output logic completion_kv_committed,
    output logic result_valid,
    output logic completion_revoked,
    input wire result_ce,
    input wire [10:0] result_address,
    output logic [31:0] result_q,
    output logic result_q_valid,
    input wire [63:0] command_v_address,
    input wire [63:0] command_v_base,
    input wire [63:0] command_v_bytes,
    input wire command_v_source_valid,
    input wire [63:0] command_v_source_sequence,
    input wire [63:0] command_v_source_epoch,
    input wire [31:0] command_v_source_position,
    output logic [63:0] v_axi_araddr,
    output logic [7:0] v_axi_arlen,
    output logic [2:0] v_axi_arsize,
    output logic [2:0] v_axi_arprot,
    output logic [1:0] v_axi_arburst,
    output logic [3:0] v_axi_arid,
    output logic [3:0] v_axi_arcache,
    output logic v_axi_arlock,
    output logic v_axi_arvalid,
    input wire v_axi_arready,
    input wire [511:0] v_axi_rdata,
    input wire [1:0] v_axi_rresp,
    input wire [3:0] v_axi_rid,
    input wire v_axi_rlast,
    input wire v_axi_rvalid,
    output logic v_axi_rready,
    output logic [3:0] completion_capture_status,
    output logic [11:0] completion_capture_qk_words,
    output logic [5:0] completion_capture_v_beats,
    input wire [63:0] command_source_sequence,
    input wire [63:0] command_source_epoch,
    input wire command_source_valid,
    input wire [31:0] command_source_position,
    input wire [31:0] command_coefficient_position,
    input wire [63:0] command_q_address,
    input wire [63:0] command_q_base,
    input wire [63:0] command_q_bytes,
    input wire [63:0] command_k_address,
    input wire [63:0] command_k_base,
    input wire [63:0] command_k_bytes,
    input wire [63:0] command_coefficient_address,
    input wire [63:0] command_coefficient_base,
    input wire [63:0] command_coefficient_bytes,
    output logic [63:0] rope_axi_araddr,
    output logic [7:0] rope_axi_arlen,
    output logic [2:0] rope_axi_arsize,
    output logic [2:0] rope_axi_arprot,
    output logic [1:0] rope_axi_arburst,
    output logic [3:0] rope_axi_arid,
    output logic [3:0] rope_axi_arcache,
    output logic rope_axi_arlock,
    output logic rope_axi_arvalid,
    input wire rope_axi_arready,
    input wire [511:0] rope_axi_rdata,
    input wire [1:0] rope_axi_rresp,
    input wire [3:0] rope_axi_rid,
    input wire rope_axi_rlast,
    input wire rope_axi_rvalid,
    output logic rope_axi_rready,
    output logic [3:0] completion_rope_status,
    output logic [31:0] completion_rope_hls_status,
    output logic [9:0] completion_rope_beats,
    input wire command_valid,
    output logic command_ready,
    output logic busy,
    output logic cache_exclusive,
    input wire bind_valid,
    output logic bind_ready,
    input wire [63:0] bind_cache_base,
    input wire [63:0] bind_region_base,
    input wire [63:0] bind_region_bytes,
    input wire [31:0] bind_capacity,
    input wire [63:0] bind_sequence,
    input wire [63:0] bind_epoch,
    output logic cache_valid,
    output logic [63:0] cache_base,
    output logic [63:0] cache_region_base,
    output logic [63:0] cache_region_bytes,
    output logic [63:0] cache_sequence,
    output logic [63:0] cache_epoch,
    output logic [31:0] cache_capacity,
    output logic [63:0] kv_axi_awaddr,
    output logic [7:0] kv_axi_awlen,
    output logic [2:0] kv_axi_awsize,
    output logic [2:0] kv_axi_awprot,
    output logic [1:0] kv_axi_awburst,
    output logic [3:0] kv_axi_awid,
    output logic [3:0] kv_axi_awcache,
    output logic kv_axi_awlock,
    output logic kv_axi_awvalid,
    input wire kv_axi_awready,
    output logic [511:0] kv_axi_wdata,
    output logic [63:0] kv_axi_wstrb,
    output logic kv_axi_wlast,
    output logic kv_axi_wvalid,
    input wire kv_axi_wready,
    input wire [3:0] kv_axi_bid,
    input wire [1:0] kv_axi_bresp,
    input wire kv_axi_bvalid,
    output logic kv_axi_bready,
    output logic [63:0] kv_axi_araddr,
    output logic [7:0] kv_axi_arlen,
    output logic [2:0] kv_axi_arsize,
    output logic [2:0] kv_axi_arprot,
    output logic [1:0] kv_axi_arburst,
    output logic [3:0] kv_axi_arid,
    output logic [3:0] kv_axi_arcache,
    output logic kv_axi_arlock,
    output logic kv_axi_arvalid,
    input wire kv_axi_arready,
    input wire [511:0] kv_axi_rdata,
    input wire [1:0] kv_axi_rresp,
    input wire [3:0] kv_axi_rid,
    input wire kv_axi_rlast,
    input wire kv_axi_rvalid,
    output logic kv_axi_rready,
    output logic root_cache_lock,
    output logic cache_metadata_fault,
    input wire [31:0] cache_query_layer,
    output logic cache_query_valid,
    output logic [31:0] cache_query_committed_tokens,
    output logic [31:0] cache_committed_tokens,
    output logic bind_completion_valid,
    input wire bind_completion_ready,
    output logic [3:0] bind_completion_status,
    output logic [63:0] bind_completion_sequence,
    output logic [63:0] bind_completion_epoch
);
    assign v_axi_araddr='0;
    assign v_axi_arlen='0;
    assign v_axi_arsize='0;
    assign v_axi_arprot='0;
    assign v_axi_arburst='0;
    assign v_axi_arid='0;
    assign v_axi_arcache='0;
    assign v_axi_arlock='0;
    assign v_axi_arvalid='0;
    assign v_axi_rready='0;
    assign rope_axi_araddr='0;
    assign rope_axi_arlen='0;
    assign rope_axi_arsize='0;
    assign rope_axi_arprot='0;
    assign rope_axi_arburst='0;
    assign rope_axi_arid='0;
    assign rope_axi_arcache='0;
    assign rope_axi_arlock='0;
    assign rope_axi_arvalid='0;
    assign rope_axi_rready='0;
    assign kv_axi_awaddr='0;
    assign kv_axi_awlen='0;
    assign kv_axi_awsize='0;
    assign kv_axi_awprot='0;
    assign kv_axi_awburst='0;
    assign kv_axi_awid='0;
    assign kv_axi_awcache='0;
    assign kv_axi_awlock='0;
    assign kv_axi_awvalid='0;
    assign kv_axi_wdata='0;
    assign kv_axi_wstrb='0;
    assign kv_axi_wlast='0;
    assign kv_axi_wvalid='0;
    assign kv_axi_bready='0;
    assign kv_axi_araddr='0;
    assign kv_axi_arlen='0;
    assign kv_axi_arsize='0;
    assign kv_axi_arprot='0;
    assign kv_axi_arburst='0;
    assign kv_axi_arid='0;
    assign kv_axi_arcache='0;
    assign kv_axi_arlock='0;
    assign kv_axi_arvalid='0;
    assign kv_axi_rready='0;
    assign root_cache_lock='0;

    integer test_delay=5,test_drain=7,test_status=0,test_bytes=4096;
    logic test_mutate_cache=0;
    logic [63:0] test_sequence_mask=0;
    integer delay_left=0,drain_left=0;
    logic cancelled,authority;
    logic [63:0] saved_sequence,saved_epoch,saved_base;
    logic [31:0] prefixes[0:15];
    assign cache_base=saved_base^(test_mutate_cache ? 64'h100000000:64'd0);
    assign cache_valid=authority && cache_epoch==expected_epoch;
    assign cache_metadata_fault=1'b0;
    assign cache_query_valid=cache_valid && cache_query_layer<16;
    assign cache_query_committed_tokens=cache_query_valid ? prefixes[cache_query_layer[3:0]]:32'd0;
    // Deliberately unrelated to the queried layer; the owner must use the new port.
    assign cache_committed_tokens=32'd77;
    assign cache_exclusive=busy;
    assign bind_ready=!busy && !completion_valid && !bind_completion_valid;
    assign command_ready=!busy && !completion_valid && !bind_valid;
    assign cancel_ready=busy && !completion_valid && !cancelled;
    assign completion_sequence=saved_sequence^test_sequence_mask;
    assign completion_epoch=saved_epoch;
    assign result_valid=completion_valid && completion_status==0 && !completion_revoked;
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            busy<=0;completion_valid<=0;bind_completion_valid<=0;cancelled<=0;authority<=0;
            saved_sequence<=0;saved_epoch<=0;saved_base<=0;delay_left<=0;drain_left<=0;
            cache_region_base<=0;cache_region_bytes<=0;cache_sequence<=0;cache_epoch<=0;cache_capacity<=0;
            bind_completion_status<=0;bind_completion_sequence<=0;bind_completion_epoch<=0;
            completion_layer<=0;completion_position<=0;completion_status<=0;completion_append_status<=0;completion_root_status<=0;
            completion_capture_status<=0;completion_rope_status<=0;completion_rope_hls_status<=0;completion_rope_beats<=0;
            completion_capture_qk_words<=0;completion_capture_v_beats<=0;completion_kv_frames<=0;completion_q_frames<=0;
            completion_kv_bytes<=0;completion_kv_tokens<=0;completion_kv_committed<=0;completion_revoked<=0;
            result_q<=0;result_q_valid<=0;
            for(integer i=0;i<16;i++)prefixes[i]<=0;
        end else begin
            result_q_valid<=result_ce && result_valid;
            if(result_ce && result_valid)result_q<=32'hbadd0000^32'(result_address)^(completion_layer<<16)^completion_position;
            if(bind_valid && bind_ready)begin
                busy<=1;bind_completion_valid<=1;bind_completion_status<=0;
                bind_completion_sequence<=bind_sequence;bind_completion_epoch<=bind_epoch;
                cache_sequence<=bind_sequence;cache_epoch<=bind_epoch;saved_base<=bind_cache_base;
                cache_region_base<=bind_region_base;cache_region_bytes<=bind_region_bytes;cache_capacity<=bind_capacity;authority<=1;
                for(integer i=0;i<16;i++)prefixes[i]<=i%3;
            end
            if(bind_completion_valid && bind_completion_ready)begin bind_completion_valid<=0;busy<=0;end
            if(command_valid && command_ready)begin
                busy<=1;saved_sequence<=command_sequence;saved_epoch<=command_epoch;cancelled<=0;delay_left<=test_delay;
                completion_layer<=command_layer;completion_position<=command_position;completion_revoked<=0;
            end
            if(cancel_valid && cancel_ready)begin cancelled<=1;delay_left<=2;end
            else if(delay_left>0)begin
                delay_left<=delay_left-1;
                if(delay_left==1)begin
                    completion_valid<=1;completion_status<=cancelled ? 7:4'(test_status);
                    completion_append_status<=0;completion_root_status<=0;completion_capture_status<=0;completion_rope_status<=0;
                    completion_rope_hls_status<=0;completion_rope_beats<=164;
                    completion_capture_qk_words<=2560;completion_capture_v_beats<=32;completion_kv_frames<=64;completion_q_frames<=128;
                    completion_kv_bytes<=cancelled || test_status!=0 ? 0:13'(test_bytes);
                    completion_kv_tokens<=completion_position+(cancelled || test_status!=0 ? 0:1);
                    completion_kv_committed<=!cancelled && test_status==0;
                    if(!cancelled && test_status==0)prefixes[completion_layer[3:0]]<=completion_position+1;
                end
            end
            if(completion_valid && completion_ready)begin completion_valid<=0;drain_left<=test_drain;end
            if(drain_left>0)begin drain_left<=drain_left-1;if(drain_left==1)busy<=0;end
        end
    end
endmodule

`default_nettype wire
