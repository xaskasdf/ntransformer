// Native protocol peer: reads the materialized synthetic quant beat and writes its complement.
`timescale 1ns/1ps
module ntrpu_attention_projection_memory_r1 #(parameter integer WATCHDOG_CYCLES=65536,parameter logic [63:0] INITIAL_CHILD_TAG=0)(

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
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize, m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid, m_axi_awcache,
    output logic m_axi_awlock, m_axi_awvalid,
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast, m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready
);

    integer state=0,test_mode=0;
    logic foreign=0,hold_admission=0,cancelled=0,aw_seen=0,w_seen=0;
    logic [63:0] saved_sequence,saved_epoch,saved_address;
    assign busy=state!=0;
    assign cancel_ready=busy && !cancelled;
    assign tag_exhausted=0;
    assign completion_sequence=saved_sequence ^ (foreign ? 64'h10000000000:64'd0);
    assign completion_epoch=saved_epoch;
    assign m_axi_awaddr=saved_address;
    assign m_axi_awlen=0;
    assign m_axi_awsize=6;
    assign m_axi_awburst=1;
    assign m_axi_awprot=0;
    assign m_axi_awid=0;
    assign m_axi_awcache=2;
    assign m_axi_awlock=0;
    assign m_axi_wstrb='1;
    assign m_axi_wlast=1;

    logic [63:0] saved_activation;
    logic [511:0] read_bits;
    assign command_ready=state==0 && !hold_admission;
    assign completion_valid=state==5;
    assign completion_status=cancelled ? 7:test_mode==7 ? 3:0;
    assign result_valid=completion_valid && completion_status==0 && test_mode!=15;
    assign result_address=result_valid ? saved_address:0;
    assign result_bytes=result_valid ? (test_mode==8 ? 8191:8192):0;
    assign m_axi_araddr=saved_activation;
    assign m_axi_arlen=0;
    assign m_axi_arsize=6;
    assign m_axi_arburst=1;
    assign m_axi_arid=0;
    assign m_axi_arcache=2;
    assign m_axi_arprot=0;
    assign m_axi_arlock=0;
    assign m_axi_arvalid=state==1;
    assign m_axi_rready=state==2;
    assign m_axi_awvalid=state==3 && !aw_seen;
    assign m_axi_wvalid=state==3 && !w_seen;
    assign m_axi_wdata=~read_bits;
    assign m_axi_bready=state==3 && aw_seen && w_seen;
    always @(posedge clk)begin
        if(!rst_n)begin state<=0;cancelled<=0;aw_seen<=0;w_seen<=0;saved_sequence<=0;saved_epoch<=0;saved_address<=0;saved_activation<=0;read_bits<=0;end
        else begin
            if(cancel_valid && cancel_ready)cancelled<=1;
            case(state)
                0:if(command_valid && command_ready)begin
                    saved_sequence<=command_sequence;saved_epoch<=command_epoch;saved_address<=command_output_addr;saved_activation<=command_activation_addr;
                    if(command_weight_addr!==64'h100000fc0 || command_weight_region_base!==64'h100000000 || command_weight_region_bytes!==4460480 ||
                       command_activation_addr!==64'h8000000fc0 || command_activation_region_base!==64'h8000000000 || command_activation_region_bytes!==8128 ||
                       command_output_addr!==64'ha000000fc0 || command_output_region_base!==64'ha000000000 || command_output_region_bytes!==12224)
                        $fatal(1,"PATH_FAIL retained projection descriptor");
                    cancelled<=0;aw_seen<=0;w_seen<=0;state<=1;
                end
                1:if(m_axi_arready)state<=2;
                2:if(m_axi_rvalid)begin
                    if(m_axi_rdata!=={8{64'h0123456789abcdef ^ saved_sequence}} || !m_axi_rlast || m_axi_rresp!=0 || m_axi_rid!=0)$fatal(1,"PATH_FAIL materialized ABI1/routing");
                    read_bits<=m_axi_rdata;state<=3;
                end
                3:begin
                    if(m_axi_awvalid && m_axi_awready)aw_seen<=1;
                    if(m_axi_wvalid && m_axi_wready)w_seen<=1;
                    if(m_axi_bvalid && m_axi_bready)begin if(m_axi_bresp!=0)$fatal(1,"PATH_FAIL B routing");state<=5;end
                end
                5:if(completion_ready)state<=0;
            endcase
        end
    end
endmodule
