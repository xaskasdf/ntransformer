// Protocol-only output path peer; one synthetic word, no quantization/GEMV.
`timescale 1ns/1ps
module ntrpu_attention_output_path_r1 #(
 parameter integer WATCHDOG_CYCLES=65536,
 parameter logic [63:0] INITIAL_QUANT_TAG=0,INITIAL_PROJECTION_TAG=0
) (
    input logic clk,
    input logic rst_n,
    input logic command_valid,
    output logic command_ready,
    input logic [63:0] command_sequence,
    input logic [63:0] command_epoch,
    input logic [63:0] expected_epoch,
    input logic [63:0] command_weight_addr,
    input logic [63:0] command_weight_region_base,
    input logic [63:0] command_weight_region_bytes,
    input logic [63:0] command_activation_addr,
    input logic [63:0] command_activation_region_base,
    input logic [63:0] command_activation_region_bytes,
    input logic [63:0] command_output_addr,
    input logic [63:0] command_output_region_base,
    input logic [63:0] command_output_region_bytes,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic busy,
    output logic completion_valid,
    input logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,
    output logic [63:0] completion_epoch,
    output logic result_valid,
    output logic [63:0] result_address,
    output logic [16:0] result_bytes,
    output logic tag_exhausted,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,
    output logic [2:0] m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,
    output logic [3:0] m_axi_arcache,
    output logic m_axi_arlock,
    output logic m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast,
    input logic m_axi_rvalid,
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
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,
    output logic m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready,
    input logic source_completion_valid,
    output logic source_completion_ready,
    input logic [3:0] source_completion_status,
    input logic [63:0] source_sequence,
    input logic [63:0] source_epoch,
    input logic source_result_valid,
    output logic source_result_ce,
    output logic [10:0] source_result_address,
    input logic [31:0] source_result_q,
    input logic source_result_q_valid,
    output logic [31:0] completion_hls_status
);

    assign m_axi_awlen=0;assign m_axi_awsize=6;assign m_axi_awburst=1;
    assign m_axi_awid=0;assign m_axi_awcache=2;assign m_axi_awprot=0;assign m_axi_awlock=0;
    assign m_axi_wstrb='1;assign m_axi_wlast=1;
    assign m_axi_arlen=0;assign m_axi_arsize=6;assign m_axi_arburst=1;
    assign m_axi_arid=0;assign m_axi_arcache=2;assign m_axi_arprot=0;assign m_axi_arlock=0;

    integer state=0,test_mode=0;
    logic foreign=0,hold_admission=0,cancelled=0,aw_seen=0,w_seen=0;
    logic [3:0] source_error=0;
    logic [63:0] saved_sequence=0,saved_epoch=0,activation=0,destination=0;
    logic [511:0] generated_word=0,projected_word=0;
    assign busy=state!=0;assign tag_exhausted=0;
    assign command_ready=state==0 && source_completion_valid && !hold_admission;
    assign cancel_ready=busy && !cancelled;
    assign source_result_ce=state==1 && !cancel_valid;
    assign source_result_address=saved_sequence[10:0];
    assign source_completion_ready=state==3;
    assign completion_valid=state==8;
    assign completion_sequence=saved_sequence^(foreign ? 64'h0000010000000000:64'd0);
    assign completion_epoch=saved_epoch;
    assign completion_status=cancelled ? 7:source_error!=0 ? source_error:test_mode==1 ? 3:0;
    assign completion_hls_status=test_mode==1 ? 42:0;
    assign result_valid=completion_valid && completion_status==0 && test_mode!=3;
    assign result_address=result_valid ? destination:0;
    assign result_bytes=result_valid ? (test_mode==2 ? 8191:8192):0;
    assign m_axi_awaddr=state==4 ? activation:destination;
    assign m_axi_awvalid=(state==4 || state==7) && !aw_seen;
    assign m_axi_wvalid=(state==4 || state==7) && !w_seen;
    assign m_axi_wdata=state==4 ? generated_word:projected_word;
    assign m_axi_bready=(state==4 || state==7) && aw_seen && w_seen;
    assign m_axi_araddr=activation;
    assign m_axi_arvalid=state==5;assign m_axi_rready=state==6;
    always @(posedge clk)begin
        if(!rst_n)begin state<=0;cancelled<=0;source_error<=0;aw_seen<=0;w_seen<=0;end
        else begin
            if(cancel_valid && busy && state!=8)cancelled<=1;
            case(state)
                0:if(command_valid && command_ready)begin
                    if(command_sequence!==source_sequence || command_epoch!==source_epoch)$fatal(1,"PROOT_FAIL retained source identity");
                    if(command_weight_addr!=64'h100000fc0 || command_weight_region_base!=64'h100000000 || command_weight_region_bytes!=4460480 ||
                       command_activation_addr!=64'h8000000fc0 || command_activation_region_base!=64'h8000000000 || command_activation_region_bytes!=8128 ||
                       command_output_addr!=64'ha000000fc0 || command_output_region_base!=64'ha000000000 || command_output_region_bytes!=12224)$fatal(1,"PROOT_FAIL retained output descriptor");
                    saved_sequence<=command_sequence;saved_epoch<=command_epoch;activation<=command_activation_addr;destination<=command_output_addr;
                    cancelled<=cancel_valid;source_error<=source_completion_status;aw_seen<=0;w_seen<=0;
                    state<=(cancel_valid || source_completion_status!=0 || !source_result_valid) ? 3:1;
                end
                1:state<=cancel_valid ? 3:2;
                2:begin
                    if(!cancel_valid && (!source_result_q_valid || source_result_q!==(32'h3f000000|32'(saved_sequence[10:0]))))$fatal(1,"PROOT_FAIL synchronous root data");
                    generated_word<={16{source_result_q}};state<=3;
                end
                3:if(source_completion_valid && source_completion_ready)begin
                    if(source_sequence!=saved_sequence || source_epoch!=saved_epoch)$fatal(1,"PROOT_FAIL foreign source release");
                    state<=(cancelled || cancel_valid || source_error!=0) ? 8:4;
                end
                4,7:begin
                    if(m_axi_awvalid && m_axi_awready)aw_seen<=1;
                    if(m_axi_wvalid && m_axi_wready)w_seen<=1;
                    if(m_axi_bvalid && m_axi_bready)begin
                        if(m_axi_bresp!=0)$fatal(1,"PROOT_FAIL output B routing");
                        aw_seen<=0;w_seen<=0;state<=state==4 ? 5:8;
                    end
                end
                5:if(m_axi_arready)state<=6;
                6:if(m_axi_rvalid && m_axi_rready)begin
                    if(m_axi_rdata!==generated_word || m_axi_rresp!=0 || !m_axi_rlast)$fatal(1,"PROOT_FAIL output read routing");
                    projected_word<=~m_axi_rdata;state<=7;
                end
                8:if(completion_ready)state<=0;
            endcase
        end
    end
endmodule
