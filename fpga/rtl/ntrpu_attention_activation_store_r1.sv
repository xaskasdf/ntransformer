// Publish one 4096-byte ABI1 activation after 64 actual AXI write responses.
// Source is synchronous and retained until the dedicated writer has drained.
// Full parent identities are independent of the writer's reserved 32-bit tags.
// Reset requires source and AXI peers to reset/quiesce together.
`default_nettype none
module ntrpu_attention_activation_store_r1 #(
    parameter integer WATCHDOG_CYCLES=65536,
    parameter logic [63:0] INITIAL_CHILD_TAG=64'd0
) (
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
    output logic [8:0] source_result_address,
    input wire logic [63:0] source_result_q,
    input wire logic source_result_q_valid,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
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
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,BAD_MEMORY=4,
        BAD_COMPLETION=5,CANCELLED=7,TIMEOUT=8,TAGS_EXHAUSTED=9,BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,CHECK_BOUNDS,FETCH_REQUEST,FETCH_RESPONSE,
        OFFER,WAIT_WRITE,RELEASE_SOURCE,RESPONSE} state_t;
    state_t state;
    logic [63:0] address,region_base,region_bytes,next_tag,base_tag;
    logic [64:0] region_end,activation_end;
    logic [5:0] record_index;
    logic [2:0] word_index;
    logic [511:0] record_data;
    logic [3:0] fault,fault_now,admission_status;
    logic source_owned,revoked,cancel_seen,epoch_bad,peer_bad,word_bad;
    logic active,progress_event,timeout_now;
    logic [31:0] idle_cycles;
    logic writer_cmd_valid,writer_cmd_ready,writer_cancel_valid,writer_cancel_ready;
    logic writer_done_valid,writer_done_ready,writer_identity_match;
    logic [3:0] writer_done_status;
    logic [6:0] writer_done_bytes;
    logic [31:0] writer_sequence,writer_epoch,writer_done_sequence,writer_done_epoch;
    logic [63:0] writer_address;

    assign busy=state!=IDLE;
    assign active=state!=IDLE && state!=RESPONSE;
    assign command_ready=state==IDLE && source_completion_valid && writer_cmd_ready && !writer_done_valid;
    assign completion_valid=state==RESPONSE;
    assign cancel_ready=busy && !cancel_seen;
    assign epoch_bad=busy && expected_epoch!=completion_epoch;
    assign peer_bad=source_owned && (!source_completion_valid || !source_result_valid ||
        source_completion_status!=OK || source_sequence!=completion_sequence || source_epoch!=completion_epoch);
    assign source_result_ce=state==FETCH_REQUEST && fault_now==OK;
    assign source_result_address={record_index,word_index};
    assign source_completion_ready=state==RELEASE_SOURCE && source_owned && writer_cmd_ready &&
        source_sequence==completion_sequence && source_epoch==completion_epoch;
    assign writer_cmd_valid=state==OFFER;
    assign writer_address=address+{52'd0,record_index,6'd0};
    assign writer_sequence=base_tag[31:0]+{26'd0,record_index};
    assign writer_epoch=base_tag[63:32];
    assign writer_identity_match=writer_done_sequence==writer_sequence && writer_done_epoch==writer_epoch;
    assign writer_done_ready=state==WAIT_WRITE && writer_identity_match;
    assign progress_event=(m_axi_awvalid && m_axi_awready) || (m_axi_wvalid && m_axi_wready) ||
        (m_axi_bvalid && m_axi_bready) || (writer_done_valid && writer_done_ready);
    assign timeout_now=state==WAIT_WRITE && !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;
    assign writer_cancel_valid=state==WAIT_WRITE && fault_now!=OK;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad && !cancel_valid;
    assign result_address=result_valid ? address:64'd0;
    assign result_bytes=result_valid ? 13'd4096:13'd0;

    ntrpu_activation_writer #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) writer(
        .clk,.rst_n,.cmd_valid(writer_cmd_valid),.cmd_ready(writer_cmd_ready),
        .cmd_addr(writer_address),.cmd_region_base(region_base),.cmd_region_bytes(region_bytes),
        .cmd_record(record_data),.cmd_elements(6'd32),.cmd_sequence(writer_sequence),
        .cmd_epoch(writer_epoch),.expected_epoch(writer_epoch),
        .cancel_valid(writer_cancel_valid),.cancel_ready(writer_cancel_ready),
        .done_valid(writer_done_valid),.done_ready(writer_done_ready),.done_status(writer_done_status),
        .done_bytes(writer_done_bytes),.done_sequence(writer_done_sequence),.done_epoch(writer_done_epoch),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,.m_axi_awcache,
        .m_axi_awlock,.m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,.m_axi_wlast,.m_axi_wvalid,
        .m_axi_wready,.m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready
    );

    always_comb begin
        admission_status=OK;
        if(command_sequence!=source_sequence || command_epoch!=source_epoch || command_epoch!=expected_epoch)
            admission_status=BAD_IDENTITY;
        else if(source_completion_status!=OK)admission_status=source_completion_status;
        else if(!source_result_valid)admission_status=BAD_PEER;
        else if(tag_exhausted || next_tag[5:0]!=0)admission_status=TAGS_EXHAUSTED;
        region_end={1'b0,region_base}+{1'b0,region_bytes};
        activation_end={1'b0,address}+65'd4096;
        word_bad=0;
        if(word_index<4)begin
            for(integer lane=0;lane<8;lane=lane+1)
                if(source_result_q[lane*8+:8]==8'h80)word_bad=1;
        end else if(word_index==4)word_bad=source_result_q>64'h4780204070e1c387;
        else word_bad=source_result_q!=0;
        fault_now=fault;
        if(active && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(peer_bad)fault_now=BAD_PEER;
            else if(state==WAIT_WRITE && writer_done_valid && !writer_identity_match)fault_now=BAD_PEER;
            else if(timeout_now)fault_now=TIMEOUT;
            else if(state==CHECK_BOUNDS && (address[5:0]!=0 || region_bytes==0 || region_end[64] ||
                activation_end[64] || address<region_base || activation_end>region_end))fault_now=BAD_COMMAND;
            else if(state==FETCH_RESPONSE && (!source_result_q_valid || word_bad))fault_now=BAD_MEMORY;
            else if(state==WAIT_WRITE && writer_done_valid && writer_done_ready)begin
                if(writer_done_bytes>64 || (writer_done_status==OK && writer_done_bytes!=64))fault_now=BAD_COMPLETION;
                else if(writer_done_status!=OK)fault_now=writer_done_status;
            end
        end
    end

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;region_base<=0;region_bytes<=0;next_tag<=INITIAL_CHILD_TAG;base_tag<=0;
            record_index<=0;word_index<=0;record_data<=0;fault<=OK;source_owned<=0;revoked<=0;cancel_seen<=0;
            idle_cycles<=0;tag_exhausted<=0;completion_status<=OK;completion_records<=0;completion_sequence<=0;completion_epoch<=0;
        end else begin
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(active)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            if(state==WAIT_WRITE && !progress_event)begin
                if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    address<=command_address;region_base<=command_region_base;region_bytes<=command_region_bytes;
                    completion_status<=admission_status;completion_records<=0;fault<=admission_status;
                    source_owned<=command_sequence==source_sequence && command_epoch==source_epoch;
                    revoked<=0;cancel_seen<=0;record_index<=0;word_index<=0;record_data<=0;
                    if(command_sequence!=source_sequence || command_epoch!=source_epoch)state<=RESPONSE;
                    else if(admission_status!=OK)state<=RELEASE_SOURCE;
                    else begin
                        base_tag<=next_tag;
                        if(next_tag==64'hffffffffffffffc0)tag_exhausted<=1;
                        else next_tag<=next_tag+64'd64;
                        state<=CHECK_BOUNDS;
                    end
                end
                CHECK_BOUNDS:state<=fault_now==OK ? FETCH_REQUEST:RELEASE_SOURCE;
                FETCH_REQUEST:state<=fault_now==OK ? FETCH_RESPONSE:RELEASE_SOURCE;
                FETCH_RESPONSE:begin
                    if(fault_now!=OK)state<=RELEASE_SOURCE;
                    else begin
                        record_data[word_index*64+:64]<=source_result_q;
                        if(word_index==7)state<=OFFER;
                        else begin word_index<=word_index+1;state<=FETCH_REQUEST;end
                    end
                end
                // Even after a fault, an offered command remains stable until accepted.
                OFFER:if(writer_cmd_ready)state<=WAIT_WRITE;
                WAIT_WRITE:if(writer_done_valid && writer_done_ready)begin
                    if(writer_done_status==OK && writer_done_bytes==64)completion_records<=completion_records+1;
                    if(fault_now!=OK || record_index==63)state<=RELEASE_SOURCE;
                    else begin record_index<=record_index+1;word_index<=0;state<=FETCH_REQUEST;end
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
