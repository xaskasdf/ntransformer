// Native protocol peer, single synthetic beat; never use for arithmetic qualification.
`timescale 1ns/1ps
module ntrpu_attention_quantize_store_r1 #(parameter integer WATCHDOG_CYCLES=65536,parameter logic [63:0] INITIAL_CHILD_TAG=0)(

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

    assign command_ready=state==0 && source_completion_valid && !hold_admission;
    assign completion_valid=state==5;
    assign completion_status=cancelled ? 7:test_mode==1 ? 3:0;
    assign completion_hls_status=test_mode==1 ? 2:0;
    assign completion_records=completion_status==0 ? 64:0;
    assign result_valid=completion_valid && completion_status==0 && test_mode!=14;
    assign result_address=result_valid ? saved_address:0;
    assign result_bytes=result_valid ? (test_mode==2 ? 4095:4096):0;
    assign source_result_ce=state==1;
    assign source_result_address=saved_sequence[10:0];
    assign source_completion_ready=state==4;
    assign m_axi_awvalid=state==3 && !aw_seen;
    assign m_axi_wvalid=state==3 && !w_seen;
    assign m_axi_wdata={8{64'h0123456789abcdef ^ saved_sequence}};
    assign m_axi_bready=state==3 && aw_seen && w_seen;
    always @(posedge clk)begin
        if(!rst_n)begin state<=0;cancelled<=0;aw_seen<=0;w_seen<=0;saved_sequence<=0;saved_epoch<=0;saved_address<=0;end
        else begin
            if(cancel_valid && cancel_ready)cancelled<=1;
            case(state)
                0:if(command_valid && command_ready)begin
                    saved_sequence<=command_sequence;saved_epoch<=command_epoch;saved_address<=command_address;
                    if(source_sequence!==command_sequence || source_epoch!==command_epoch)$fatal(1,"PATH_FAIL source wiring");
                    if(command_region_base+4032!==command_address || command_region_bytes!==8128)$fatal(1,"PATH_FAIL quant descriptor");
                    cancelled<=0;aw_seen<=0;w_seen<=0;state<=1;
                end
                1:state<=2;
                2:begin if(!source_result_q_valid || source_result_q!==(32'h3f000000|32'(saved_sequence[10:0])))$fatal(1,"PATH_FAIL synchronous source wiring");state<=3;end
                3:begin
                    if(m_axi_awvalid && m_axi_awready)aw_seen<=1;
                    if(m_axi_wvalid && m_axi_wready)w_seen<=1;
                    if(m_axi_bvalid && m_axi_bready)begin if(m_axi_bresp!=0)$fatal(1,"PATH_FAIL B routing");state<=4;end
                end
                4:if(source_completion_valid)state<=5;
                5:if(completion_ready)state<=0;
            endcase
        end
    end
endmodule
