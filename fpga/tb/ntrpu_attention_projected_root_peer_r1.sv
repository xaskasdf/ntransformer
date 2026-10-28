// Protocol-only single-beat root peer; no attention arithmetic.
`timescale 1ns/1ps
module ntrpu_attention_root_r1 #(parameter integer WATCHDOG_CYCLES=65536) (
    input logic clk,
    input logic rst_n,
    input logic bind_valid,
    output logic bind_ready,
    input logic [63:0] bind_cache_base,
    input logic [63:0] bind_region_base,
    input logic [63:0] bind_region_bytes,
    input logic [31:0] bind_capacity,
    input logic [63:0] bind_sequence,
    input logic [63:0] bind_epoch,
    input logic [63:0] expected_epoch,
    output logic cache_valid,
    output logic [63:0] cache_base,
    output logic [63:0] cache_region_base,
    output logic [63:0] cache_region_bytes,
    output logic [63:0] cache_sequence,
    output logic [63:0] cache_epoch,
    output logic [31:0] cache_capacity,
    input logic append_valid,
    output logic append_ready,
    output logic append_prepared,
    input logic [31:0] append_layer,
    input logic [31:0] append_position,
    input logic [63:0] append_sequence,
    input logic [63:0] append_epoch,
    input logic data_valid,
    output logic data_ready,
    input logic [511:0] data_bits,
    input logic [5:0] data_index,
    input logic data_last,
    input logic [63:0] data_sequence,
    input logic [63:0] data_epoch,
    input logic commit_valid,
    output logic commit_ready,
    input logic [3:0] commit_status,
    input logic [63:0] commit_sequence,
    input logic [63:0] commit_epoch,
    input logic append_cancel_valid,
    output logic append_cancel_ready,
    output logic append_done_valid,
    input logic append_done_ready,
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
    input logic command_valid,
    output logic command_ready,
    input logic [31:0] command_position,
    input logic [31:0] command_layer,
    input logic [63:0] command_sequence,
    input logic [63:0] command_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic completion_valid,
    input logic completion_ready,
    output logic [3:0] completion_status,
    output logic result_valid,
    input logic result_ce,
    input logic [10:0] result_address,
    output logic [31:0] result_q,
    output logic result_q_valid,
    input logic [63:0] command_lease_base,
    input logic q_valid,
    output logic q_ready,
    input logic [511:0] q_data,
    input logic [7:0] q_index,
    input logic q_last,
    input logic [63:0] q_sequence,
    input logic [63:0] q_epoch,
    output logic [63:0] completion_sequence,
    output logic [63:0] completion_epoch,
    output logic cache_lock,
    output logic maintenance_busy,
    output logic metadata_fault,
    output logic [31:0] committed_tokens
);

    assign m_axi_awlen=0;assign m_axi_awsize=6;assign m_axi_awburst=1;
    assign m_axi_awid=0;assign m_axi_awcache=2;assign m_axi_awprot=0;assign m_axi_awlock=0;
    assign m_axi_wstrb='1;assign m_axi_wlast=1;
    assign m_axi_arlen=0;assign m_axi_arsize=6;assign m_axi_arburst=1;
    assign m_axi_arid=0;assign m_axi_arcache=2;assign m_axi_arprot=0;assign m_axi_arlock=0;

    integer state=0,maintenance=0,test_mode=0;
    logic foreign=0,hold_maintenance=0,cancelled=0,aw_seen=0,w_seen=0;
    logic [63:0] saved_sequence=0,saved_epoch=0,maint_sequence=0,maint_epoch=0;
    logic [31:0] result_data=0;logic result_pending=0;
    assign bind_ready=state==0 && maintenance==0 && !hold_maintenance;
    assign append_ready=bind_ready && !bind_valid;
    assign append_prepared=maintenance==2;
    assign data_ready=0;assign commit_ready=0;assign append_cancel_ready=maintenance!=0;
    assign append_done_valid=maintenance==1 || maintenance==3;
    assign append_done_bind=maintenance==1;assign append_done_status=0;
    assign append_done_bytes=maintenance==3 ? 4096:0;assign append_done_committed_tokens=maintenance==3 ? 1:0;
    assign append_done_sequence=maint_sequence;assign append_done_epoch=maint_epoch;
    assign lease_active=state!=0;assign cache_lock=state!=0;
    assign maintenance_busy=maintenance!=0;assign metadata_fault=0;assign committed_tokens=cache_valid ? 1:0;
    assign command_ready=state==0 && maintenance==0 && !bind_valid && !append_valid && cache_valid;
    assign cancel_ready=state!=0 && !cancelled;
    assign q_ready=state==1;
    assign completion_valid=state==4;
    assign completion_status=cancelled ? 7:test_mode==1 ? 3:0;
    assign completion_sequence=saved_sequence^(foreign ? 64'h0000010000000000:64'd0);
    assign completion_epoch=saved_epoch;
    assign result_valid=completion_valid && completion_status==0 && test_mode!=2;
    assign result_q_valid=result_pending && result_valid;
    assign result_q=result_q_valid ? result_data:0;
    assign m_axi_awaddr=64'h4000000fc0;
    assign m_axi_awvalid=maintenance==2 && !aw_seen;
    assign m_axi_wvalid=maintenance==2 && !w_seen;
    assign m_axi_wdata={8{64'h1122334455667788^maint_sequence}};
    assign m_axi_bready=maintenance==2 && aw_seen && w_seen;
    assign m_axi_araddr=64'h4000000fc0;
    assign m_axi_arvalid=state==2;assign m_axi_rready=state==3;
    always @(posedge clk)begin
        if(!rst_n)begin
            state<=0;maintenance<=0;cancelled<=0;aw_seen<=0;w_seen<=0;cache_valid<=0;result_pending<=0;
            cache_base<=0;cache_region_base<=0;cache_region_bytes<=0;cache_capacity<=0;cache_sequence<=0;cache_epoch<=0;
        end else begin
            result_pending<=result_ce && result_valid;
            if(result_ce && result_valid)result_data<=32'h3f000000|32'(result_address);
            if(cancel_valid && state!=0 && state!=4)cancelled<=1;
            if(bind_valid && bind_ready)begin
                maintenance<=1;maint_sequence<=bind_sequence;maint_epoch<=bind_epoch;
                cache_valid<=1;cache_base<=bind_cache_base;cache_region_base<=bind_region_base;cache_region_bytes<=bind_region_bytes;
                cache_sequence<=bind_sequence;cache_epoch<=bind_epoch;cache_capacity<=bind_capacity;
            end else if(append_valid && append_ready)begin
                maintenance<=2;maint_sequence<=append_sequence;maint_epoch<=append_epoch;aw_seen<=0;w_seen<=0;
            end
            if(append_done_valid && append_done_ready)maintenance<=0;
            if(m_axi_awvalid && m_axi_awready)aw_seen<=1;
            if(m_axi_wvalid && m_axi_wready)w_seen<=1;
            if(m_axi_bvalid && m_axi_bready)begin if(m_axi_bresp!=0)$fatal(1,"PROOT_FAIL maintenance B");maintenance<=3;end
            case(state)
                0:if(command_valid && command_ready)begin
                    saved_sequence<=command_sequence;saved_epoch<=command_epoch;cancelled<=0;
                    if(command_position!=0 || command_layer!=0 || command_lease_base!=64'h1234000000000000)$fatal(1,"PROOT_FAIL root descriptor");
                    state<=1;
                end
                1:if(cancel_valid)state<=4;else if(q_valid && q_ready)begin
                    if(q_index!=0 || !q_last || q_sequence!=saved_sequence || q_epoch!=saved_epoch || q_data!={16{32'hca000000|32'(saved_sequence[15:0])}})$fatal(1,"PROOT_FAIL Q wiring");
                    state<=2;
                end
                2:if(m_axi_arready)state<=3;
                3:if(m_axi_rvalid && m_axi_rready)begin
                    if(m_axi_rdata!={8{64'h1122334455667788^saved_sequence}} || m_axi_rresp!=0 || !m_axi_rlast)$fatal(1,"PROOT_FAIL KV read routing");
                    state<=4;
                end
                4:if(completion_ready)state<=0;
            endcase
        end
    end
endmodule
