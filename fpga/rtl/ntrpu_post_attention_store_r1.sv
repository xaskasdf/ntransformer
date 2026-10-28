// Copy the held post-O pair to two disjoint AXI regions; publish only after B.
// One 64-byte write is in flight. Offered AW/W remain stable through faults.
// The caller owns both destinations privately until success; partial writes
// on failure are not rolled back. Reset must quiesce source and AXI peers.
`default_nettype none
module ntrpu_post_attention_store_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_residual_address,command_residual_base,command_residual_bytes,
    input wire logic [63:0] command_ffn_address,command_ffn_base,command_ffn_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    input wire logic source_completion_valid,
    output logic source_completion_ready,
    input wire logic [3:0] source_completion_status,
    input wire logic [63:0] source_sequence,source_epoch,
    input wire logic source_result_valid,
    output logic source_residual_ce,source_ffn_ce,
    output logic [10:0] source_residual_address,
    output logic [8:0] source_ffn_address,
    input wire logic [31:0] source_residual_q,
    input wire logic [63:0] source_ffn_q,
    input wire logic source_residual_q_valid,source_ffn_q_valid,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [7:0] completion_residual_beats,
    output logic [6:0] completion_ffn_beats,
    output logic result_valid,
    output logic [63:0] result_residual_address,result_ffn_address,
    output logic [13:0] result_residual_bytes,
    output logic [12:0] result_ffn_bytes,
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
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,WRITE_ERROR=3,
        BAD_MEMORY=4,CANCELLED=7,TIMEOUT=8,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,CHECK_BOUNDS,FETCH_REQUEST,FETCH_RESPONSE,
        SEND,RELEASE_SOURCE,RESPONSE} state_t;
    state_t state;
    logic [63:0] residual_address,residual_base,residual_bytes,ffn_address,ffn_base,ffn_bytes;
    logic [64:0] residual_region_end,ffn_region_end,residual_end,ffn_end;
    logic plane;
    logic [6:0] record_index;
    logic [3:0] word_index,fault,fault_now,admission_status;
    logic [511:0] record_data;
    logic source_owned,revoked,cancel_seen,epoch_bad,peer_bad,word_bad,bounds_bad;
    logic aw_seen,w_seen,aw_fire,w_fire,b_fire,active,progress_event,timeout_now;
    logic [31:0] idle_cycles;

    assign busy=state!=IDLE;
    assign active=state!=IDLE && state!=RESPONSE;
    assign command_ready=state==IDLE && source_completion_valid;
    assign completion_valid=state==RESPONSE;
    assign cancel_ready=busy && !cancel_seen;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    assign peer_bad=source_owned && (!source_completion_valid || !source_result_valid ||
        source_completion_status!=OK || source_sequence!=completion_sequence || source_epoch!=completion_epoch);
    assign source_residual_ce=state==FETCH_REQUEST && !plane && fault_now==OK;
    assign source_ffn_ce=state==FETCH_REQUEST && plane && fault_now==OK;
    assign source_residual_address={record_index,word_index};
    assign source_ffn_address={record_index[5:0],word_index[2:0]};
    assign source_completion_ready=state==RELEASE_SOURCE && source_owned &&
        source_sequence==completion_sequence && source_epoch==completion_epoch;
    assign m_axi_awaddr=(plane ? ffn_address:residual_address)+{51'd0,record_index,6'd0};
    assign m_axi_awlen=0;
    assign m_axi_awsize=3'd6;
    assign m_axi_awburst=2'b01;
    assign m_axi_awid=0;
    assign m_axi_awcache=4'b0010;
    assign m_axi_awprot=0;
    assign m_axi_awlock=0;
    assign m_axi_awvalid=state==SEND && !aw_seen;
    assign m_axi_wdata=record_data;
    assign m_axi_wstrb=64'hffffffffffffffff;
    assign m_axi_wlast=1;
    assign m_axi_wvalid=state==SEND && !w_seen;
    assign m_axi_bready=state==SEND && aw_seen && w_seen;
    assign aw_fire=m_axi_awvalid && m_axi_awready;
    assign w_fire=m_axi_wvalid && m_axi_wready;
    assign b_fire=m_axi_bvalid && m_axi_bready;
    assign progress_event=aw_fire || w_fire || b_fire;
    assign timeout_now=state==SEND && !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad && !cancel_valid;
    assign result_residual_address=result_valid ? residual_address:64'd0;
    assign result_ffn_address=result_valid ? ffn_address:64'd0;
    assign result_residual_bytes=result_valid ? 14'd8192:14'd0;
    assign result_ffn_bytes=result_valid ? 13'd4096:13'd0;

    always_comb begin
        admission_status=OK;
        if(command_sequence!=source_sequence || command_epoch!=source_epoch || command_epoch!=expected_epoch)
            admission_status=BAD_IDENTITY;
        else if(source_completion_status!=OK)admission_status=source_completion_status;
        else if(!source_result_valid)admission_status=BAD_PEER;
        residual_region_end={1'b0,residual_base}+{1'b0,residual_bytes};
        ffn_region_end={1'b0,ffn_base}+{1'b0,ffn_bytes};
        residual_end={1'b0,residual_address}+65'd8192;
        ffn_end={1'b0,ffn_address}+65'd4096;
        bounds_bad=residual_address[5:0]!=0 || ffn_address[5:0]!=0 || residual_bytes==0 || ffn_bytes==0 ||
            residual_region_end[64] || ffn_region_end[64] || residual_end[64] || ffn_end[64] ||
            residual_address<residual_base || ffn_address<ffn_base ||
            residual_end>residual_region_end || ffn_end>ffn_region_end ||
            ({1'b0,residual_address}<ffn_end && {1'b0,ffn_address}<residual_end);
        word_bad=0;
        if(!plane)word_bad=source_residual_q[30:23]==8'hff;
        else if(word_index<4)begin
            for(integer lane=0;lane<8;lane=lane+1)
                if(source_ffn_q[lane*8+:8]==8'h80)word_bad=1;
        end else if(word_index==4)word_bad=source_ffn_q>64'h4780204070e1c387;
        else word_bad=source_ffn_q!=0;
        fault_now=fault;
        if(active && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(peer_bad)fault_now=BAD_PEER;
            else if(timeout_now)fault_now=TIMEOUT;
            else if(state==CHECK_BOUNDS && bounds_bad)fault_now=BAD_COMMAND;
            else if(state==FETCH_RESPONSE &&
                (word_bad || (plane ? !source_ffn_q_valid:!source_residual_q_valid)))fault_now=BAD_MEMORY;
            else if(b_fire && (m_axi_bresp!=0 || m_axi_bid!=0))fault_now=WRITE_ERROR;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;residual_address<=0;residual_base<=0;residual_bytes<=0;
            ffn_address<=0;ffn_base<=0;ffn_bytes<=0;plane<=0;record_index<=0;word_index<=0;record_data<=0;
            fault<=OK;source_owned<=0;revoked<=0;cancel_seen<=0;aw_seen<=0;w_seen<=0;idle_cycles<=0;
            completion_status<=OK;completion_sequence<=0;completion_epoch<=0;
            completion_residual_beats<=0;completion_ffn_beats<=0;
        end else begin
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(active)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            if(state==SEND && !progress_event)begin
                if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            if(aw_fire)aw_seen<=1;
            if(w_fire)w_seen<=1;
            if(b_fire && m_axi_bresp==0 && m_axi_bid==0)begin
                if(plane)completion_ffn_beats<=completion_ffn_beats+7'd1;
                else completion_residual_beats<=completion_residual_beats+8'd1;
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    residual_address<=command_residual_address;residual_base<=command_residual_base;residual_bytes<=command_residual_bytes;
                    ffn_address<=command_ffn_address;ffn_base<=command_ffn_base;ffn_bytes<=command_ffn_bytes;
                    completion_status<=admission_status;completion_residual_beats<=0;completion_ffn_beats<=0;fault<=admission_status;
                    source_owned<=command_sequence==source_sequence && command_epoch==source_epoch;
                    revoked<=0;cancel_seen<=0;plane<=0;record_index<=0;word_index<=0;record_data<=0;aw_seen<=0;w_seen<=0;
                    if(command_sequence!=source_sequence || command_epoch!=source_epoch)state<=RESPONSE;
                    else state<=admission_status==OK ? CHECK_BOUNDS:RELEASE_SOURCE;
                end
                CHECK_BOUNDS:state<=fault_now==OK ? FETCH_REQUEST:RELEASE_SOURCE;
                FETCH_REQUEST:state<=fault_now==OK ? FETCH_RESPONSE:RELEASE_SOURCE;
                FETCH_RESPONSE:if(fault_now!=OK)state<=RELEASE_SOURCE;
                    else begin
                        if(plane)record_data[word_index*64+:64]<=source_ffn_q;
                        else record_data[word_index*32+:32]<=source_residual_q;
                        if(word_index==(plane ? 4'd7:4'd15))begin aw_seen<=0;w_seen<=0;state<=SEND;end
                        else begin word_index<=word_index+4'd1;state<=FETCH_REQUEST;end
                    end
                // Once SEND offers either channel, retain both through its B.
                SEND:if(b_fire)begin
                    if(fault_now!=OK)state<=RELEASE_SOURCE;
                    else if(plane && record_index==63)state<=RELEASE_SOURCE;
                    else begin
                        if(!plane && record_index==127)begin plane<=1;record_index<=0;end
                        else record_index<=record_index+7'd1;
                        word_index<=0;state<=FETCH_REQUEST;
                    end
                end
                RELEASE_SOURCE:if(source_completion_valid && source_completion_ready)begin
                    source_owned<=0;completion_status<=fault_now;state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=IDLE;
                default:begin state<=IDLE;source_owned<=0;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
