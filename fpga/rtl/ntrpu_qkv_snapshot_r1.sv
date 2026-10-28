// Capture rotated Q/K and DDR-backed V before either downstream consumer starts.
// Frames 0..127 are Q, 128..159 K, 160..191 V; sixteen little-endian FP32 per frame.
// Caller retains immutable V allocation through public ACK and final drain.
// RoPE source is a live held completion; release only after public ACK.
// Layer is supplied by the scheduler: the RoPE source has no independent layer tag.
// Reset must quiesce this block, the source and the AXI peer together.
`default_nettype none
module ntrpu_qkv_snapshot_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire clk,rst_n,
    input wire command_valid,
    output wire command_ready,
    input wire [63:0] command_sequence,command_epoch,expected_epoch,
    input wire [31:0] command_layer,command_position,
    input wire [63:0] command_v_address,command_v_base,command_v_bytes,
    input wire command_v_source_valid,
    input wire [63:0] command_v_source_sequence,command_v_source_epoch,
    input wire [31:0] command_v_source_position,
    input wire source_completion_valid,
    output wire source_completion_ready,
    input wire [3:0] source_status,
    input wire [63:0] source_sequence,source_epoch,
    input wire [31:0] source_position,source_coefficient_position,source_hls_status,
    input wire [9:0] source_beats,
    input wire source_result_valid,
    output wire source_ce,
    output wire [11:0] source_address,
    input wire [31:0] source_q,
    input wire source_q_valid,
    input wire cancel_valid,
    output wire cancel_ready,busy,
    output wire completion_valid,
    input wire completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [31:0] completion_layer,completion_position,
    output logic [11:0] completion_qk_words,
    output logic [5:0] completion_v_beats,
    output wire [7:0] completion_frames,
    output wire result_valid,completion_revoked,
    input wire result_ce,
    input wire [7:0] result_address,
    output wire [511:0] result_q,
    output wire result_q_valid,
    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,m_axi_arprot,
    output wire [1:0] m_axi_arburst,
    output wire [3:0] m_axi_arid,m_axi_arcache,
    output wire m_axi_arlock,m_axi_arvalid,
    input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire [3:0] m_axi_rid,
    input wire m_axi_rlast,m_axi_rvalid,
    output wire m_axi_rready
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,READ_ERROR=3,
        BAD_FORMAT=4,CANCELLED=7,TIMEOUT=8,BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,CHECK,ISSUE,READ,ADVANCE,COPY_REQ,COPY_RESP,RESPONSE,DRAIN} state_t;
    state_t state;
    logic [63:0] v_address,v_base,v_bytes,address;
    logic [5:0] chunk_beats,chunk_received;
    logic [31:0] idle_cycles;
    logic [3:0] fault,fault_now,admission_status,held_source_status;
    logic [31:0] held_source_hls,held_source_coefficient_position;
    logic [9:0] held_source_beats;
    logic source_owned,revoked,cancel_seen,result_pending;
    logic [511:0] packed_words,result_data;
    (* ram_style="block" *) logic [511:0] frames[0:191];
    wire reader_cmd_ready,reader_done,reader_valid,reader_last;
    wire [1:0] reader_status;
    wire [31:0] reader_bytes;
    wire [511:0] reader_data;
    wire active=state!=IDLE && state!=RESPONSE && state!=DRAIN;
    wire epoch_bad=busy && completion_epoch!=expected_epoch;
    wire source_identity=source_sequence==completion_sequence && source_epoch==completion_epoch && source_position==completion_position;
    wire source_changed=source_owned && (!source_completion_valid || !source_identity ||
        source_status!=held_source_status || source_hls_status!=held_source_hls ||
        source_coefficient_position!=held_source_coefficient_position || source_beats!=held_source_beats ||
        (held_source_status==OK && !source_result_valid));
    wire [64:0] region_end={1'b0,v_base}+{1'b0,v_bytes};
    wire [64:0] vector_end={1'b0,v_address}+65'd2048;
    wire bounds_bad=v_address[5:0]!=0 || v_bytes==0 || region_end[64] || vector_end[64] ||
        v_address<v_base || vector_end>region_end;
    wire [31:0] chunk_bytes={20'd0,chunk_beats,6'd0};
    wire reader_cmd_valid=state==ISSUE && fault_now==OK;
    wire reader_ready=state==READ;
    wire reader_done_ready=state==READ;
    wire progress_event=(m_axi_arvalid && m_axi_arready) || (m_axi_rvalid && m_axi_rready) ||
        (reader_valid && reader_ready) || (reader_done && reader_done_ready) ||
        (state==COPY_RESP && source_q_valid);
    wire waiting=state==ISSUE || state==READ || state==COPY_RESP;
    wire timeout_now=waiting && !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;
    logic beat_bad;

    function automatic logic [5:0] next_chunk(input logic [63:0] addr,input logic [5:0] remaining);
        logic [6:0] page;
        page=7'd64-{1'b0,addr[11:6]};
        return {1'b0,remaining}<page ? remaining:page[5:0];
    endfunction
    assign busy=state!=IDLE;
    assign command_ready=state==IDLE && reader_cmd_ready && !reader_done;
    assign cancel_ready=busy && state!=DRAIN && !cancel_seen;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked || epoch_bad || (busy && cancel_valid) || source_changed;
    assign result_valid=completion_valid && completion_status==OK && source_owned && !completion_revoked;
    assign completion_frames=result_valid ? 8'd192:8'd0;
    assign result_q_valid=result_pending && result_valid;
    assign result_q=result_q_valid ? result_data:512'd0;
    assign source_ce=state==COPY_REQ && fault_now==OK;
    assign source_address=completion_qk_words;
    assign source_completion_ready=state==DRAIN && source_owned && source_completion_valid && source_identity;

    always_comb begin
        admission_status=OK;
        if(command_epoch!=expected_epoch || command_v_source_sequence!=command_sequence || command_v_source_epoch!=command_epoch ||
            command_v_source_position!=command_position)admission_status=BAD_IDENTITY;
        else if(command_layer>=16 || command_position>=131072)admission_status=BAD_COMMAND;
        else if(!source_completion_valid)admission_status=BAD_PEER;
        else if(source_sequence!=command_sequence || source_epoch!=command_epoch || source_position!=command_position ||
            source_coefficient_position!=command_position)admission_status=BAD_IDENTITY;
        else if(source_status!=OK)admission_status=source_status;
        else if(cancel_valid)admission_status=CANCELLED;
        else if(!command_v_source_valid || !source_result_valid || source_hls_status!=0 || source_beats!=164)admission_status=BAD_PEER;
        beat_bad=completion_v_beats>=32 || reader_last!=(chunk_received+1==chunk_beats);
        for(integer lane=0;lane<16;lane++)if(reader_data[lane*32+23+:8]==8'hff)beat_bad=1;
        fault_now=fault;
        if(active && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid)fault_now=CANCELLED;
            else if(source_changed)fault_now=BAD_PEER;
            else if(timeout_now)fault_now=TIMEOUT;
            else if(state==CHECK && bounds_bad)fault_now=BAD_COMMAND;
            else if(state==READ && reader_done && (reader_status!=0 || reader_bytes!=chunk_bytes || chunk_received!=chunk_beats))fault_now=READ_ERROR;
            else if(state==READ && reader_valid && beat_bad)fault_now=BAD_FORMAT;
            else if(state==COPY_RESP && source_q_valid && source_q[30:23]==8'hff)fault_now=BAD_FORMAT;
        end
    end
    ntrpu_axi_read_master reader(
        .clk,.rst_n,.cmd_valid(reader_cmd_valid),.cmd_ready(reader_cmd_ready),.cmd_addr(address),.cmd_bytes(chunk_bytes),
        .done_valid(reader_done),.done_ready(reader_done_ready),.done_status(reader_status),.done_bytes(reader_bytes),
        .out_data(reader_data),.out_valid(reader_valid),.out_ready(reader_ready),.out_last(reader_last),
        .m_axi_araddr,.m_axi_arlen,.m_axi_arsize,.m_axi_arprot,.m_axi_arburst,.m_axi_arid,
        .m_axi_arcache,.m_axi_arlock,.m_axi_arvalid,.m_axi_arready,.m_axi_rdata,.m_axi_rresp,
        .m_axi_rid,.m_axi_rlast,.m_axi_rvalid,.m_axi_rready
    );
    always_ff @(posedge clk)begin
        if(rst_n && state==READ && reader_valid && fault_now==OK)frames[8'd160+{2'd0,completion_v_beats}]<=reader_data;
        if(rst_n && state==COPY_RESP && source_q_valid && fault_now==OK && completion_qk_words[3:0]==15)
            frames[completion_qk_words[11:4]]<={source_q,packed_words[479:0]};
        if(rst_n && result_ce && result_valid && result_address<192)result_data<=frames[result_address];
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;v_address<=0;v_base<=0;v_bytes<=0;address<=0;chunk_beats<=0;chunk_received<=0;
            idle_cycles<=0;fault<=OK;held_source_status<=0;held_source_hls<=0;held_source_coefficient_position<=0;held_source_beats<=0;
            source_owned<=0;revoked<=0;cancel_seen<=0;result_pending<=0;packed_words<=0;
            completion_status<=OK;completion_sequence<=0;completion_epoch<=0;completion_layer<=0;completion_position<=0;
            completion_qk_words<=0;completion_v_beats<=0;
        end else begin
            result_pending<=result_ce && result_valid && result_address<192;
            if(busy && (epoch_bad || cancel_valid || source_changed))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(active)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            if(waiting)begin
                if(progress_event)idle_cycles<=0;
                else if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            if(source_completion_ready)source_owned<=0;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_layer<=command_layer;completion_position<=command_position;
                    completion_status<=admission_status;fault<=admission_status;revoked<=admission_status!=OK;cancel_seen<=cancel_valid;
                    completion_qk_words<=0;completion_v_beats<=0;result_pending<=0;packed_words<=0;
                    v_address<=command_v_address;v_base<=command_v_base;v_bytes<=command_v_bytes;
                    held_source_status<=source_status;held_source_hls<=source_hls_status;
                    held_source_coefficient_position<=source_coefficient_position;held_source_beats<=source_beats;
                    source_owned<=source_completion_valid && source_sequence==command_sequence && source_epoch==command_epoch && source_position==command_position;
                    state<=admission_status==OK ? CHECK:RESPONSE;
                end
                CHECK:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else begin address<=v_address;chunk_beats<=next_chunk(v_address,6'd32);state<=ISSUE;end
                ISSUE:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(reader_cmd_ready)begin chunk_received<=0;state<=READ;end
                READ:begin
                    if(reader_valid)begin completion_v_beats<=completion_v_beats+1;chunk_received<=chunk_received+1;end
                    if(reader_done)state<=ADVANCE;
                end
                ADVANCE:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(completion_v_beats==32)state<=COPY_REQ;
                    else begin address<=address+{32'd0,chunk_bytes};chunk_beats<=next_chunk(address+{32'd0,chunk_bytes},6'd32-completion_v_beats);state<=ISSUE;end
                COPY_REQ:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else state<=COPY_RESP;
                COPY_RESP:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(source_q_valid)begin
                        packed_words[completion_qk_words[3:0]*32+:32]<=source_q;
                        completion_qk_words<=completion_qk_words+1;
                        if(completion_qk_words==2559)begin completion_status<=OK;state<=RESPONSE;end
                        else state<=COPY_REQ;
                    end
                RESPONSE:if(completion_ready)begin result_pending<=0;state<=DRAIN;end
                DRAIN:if(!source_owned && reader_cmd_ready && !reader_done)state<=IDLE;
                default:begin state<=DRAIN;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
