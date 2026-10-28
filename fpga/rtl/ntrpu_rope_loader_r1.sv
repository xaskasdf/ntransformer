// Load Q (2048 FP32), K (512 FP32), and a phase row (64 FP32).
// Output RAM 0 concatenates Q then K; RAM 1 contains the phase row.
// Position fields associate leases; immutable coefficient-table identity and
// phase correctness must be verified by the trusted caller before admission.
// Trusted caller supplies region bounds and the source lease; it must keep all
// three regions immutable through completion. These checks do not authenticate
// DMA descriptors. Only an OK, unrevoked completion exposes the private RAMs.
// Cancellation drains the issued burst. A nonresponding AXI peer needs a shared
// reset; never reuse the port merely because the inactivity watchdog expired.
`default_nettype none
module ntrpu_rope_loader_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire logic clk,rst_n,
    input wire logic command_valid,
    output logic command_ready,
    input wire logic [63:0] command_sequence,command_epoch,expected_epoch,
    input wire logic [63:0] command_source_sequence,command_source_epoch,
    input wire logic command_source_valid,
    input wire logic [31:0] command_position,command_source_position,command_coefficient_position,
    input wire logic [63:0] command_q_address,command_q_base,command_q_bytes,
    input wire logic [63:0] command_k_address,command_k_base,command_k_bytes,
    input wire logic [63:0] command_coefficient_address,command_coefficient_base,command_coefficient_bytes,
    input wire logic cancel_valid,
    output logic cancel_ready,busy,
    output logic completion_valid,
    input wire logic completion_ready,
    output logic [3:0] completion_status,
    output logic [63:0] completion_sequence,completion_epoch,
    output logic [9:0] completion_beats,
    output logic [31:0] completion_position,completion_coefficient_position,
    output logic result_valid,
    input wire logic qk_ce,coefficient_ce,
    input wire logic [11:0] qk_address,coefficient_address,
    output logic [31:0] qk_q,coefficient_q,
    output logic qk_q_valid,coefficient_q_valid,
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
    output logic m_axi_rready
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,READ_ERROR=3,
        BAD_FORMAT=4,CANCELLED=7,TIMEOUT=8,BAD_PEER=11;
    typedef enum logic [2:0] {IDLE,CHECK,ISSUE,READ,ADVANCE,RESPONSE} state_t;
    state_t state;
    logic [63:0] addresses[0:2],bases[0:2],sizes[0:2];
    logic [1:0] plane;
    logic [63:0] address;
    logic [7:0] received;
    logic [11:0] stored;
    logic [6:0] chunk_beats,chunk_received;
    logic [31:0] chunk_bytes,idle_cycles;
    logic [511:0] pending_data;
    logic [11:0] pending_address;
    logic [3:0] pending_lane,fault,fault_now,admission_status;
    logic beat_pending,revoked,cancel_seen,epoch_bad,bounds_bad,beat_bad;
    logic active,progress_event,timeout_now;
    logic reader_cmd_valid,reader_cmd_ready,reader_done,reader_done_ready;
    logic reader_valid,reader_ready,reader_last;
    logic [1:0] reader_status;
    logic [31:0] reader_bytes;
    logic [511:0] reader_data;
    logic [64:0] region_end[0:2],vector_end[0:2];
    (* ram_style="block" *) logic [31:0] qk_mem[0:2559];
    (* ram_style="block" *) logic [31:0] coefficient_mem[0:63];
    logic [31:0] qk_data,coefficient_data;
    logic qk_pending,coefficient_pending;
    function automatic logic [7:0] plane_beats(input logic [1:0] which);
        case(which)
            0:return 8'd128;
            1:return 8'd32;
            2:return 8'd4;
            default:return 8'd0;
        endcase
    endfunction
    function automatic logic [6:0] next_chunk(input logic [63:0] addr,input logic [7:0] beats);
        logic [6:0] page;
        page=7'd64-{1'b0,addr[11:6]};
        return beats<{1'b0,page} ? beats[6:0]:page;
    endfunction
    assign busy=state!=IDLE;
    assign active=state!=IDLE && state!=RESPONSE;
    assign command_ready=state==IDLE && reader_cmd_ready && !reader_done;
    assign cancel_ready=busy && !cancel_seen;
    assign completion_valid=state==RESPONSE;
    assign epoch_bad=busy && completion_epoch!=expected_epoch;
    assign result_valid=completion_valid && completion_status==OK && !revoked && !epoch_bad && !cancel_valid;
    assign qk_q_valid=qk_pending && result_valid;
    assign coefficient_q_valid=coefficient_pending && result_valid;
    assign qk_q=qk_q_valid ? qk_data:32'd0;
    assign coefficient_q=coefficient_q_valid ? coefficient_data:32'd0;
    assign chunk_bytes={19'd0,chunk_beats,6'd0};
    assign reader_cmd_valid=state==ISSUE && fault_now==OK;
    // Readiness is derived from registered fault state, not fault_now. This
    // avoids a progress -> timeout -> readiness -> progress combinational loop.
    assign reader_ready=state==READ && (!beat_pending || fault!=OK);
    assign reader_done_ready=state==READ && !beat_pending;
    assign progress_event=(m_axi_arvalid && m_axi_arready) || (m_axi_rvalid && m_axi_rready) ||
        (reader_valid && reader_ready) || (reader_done && reader_done_ready) || (state==READ && beat_pending);
    assign timeout_now=(state==ISSUE || state==READ) && !progress_event && idle_cycles>=WATCHDOG_CYCLES-1;

    always_comb begin
        admission_status=OK;
        if(command_sequence!=command_source_sequence || command_epoch!=command_source_epoch || command_epoch!=expected_epoch)
            admission_status=BAD_IDENTITY;
        else if(command_position>=32'd131072)admission_status=BAD_COMMAND;
        else if(command_source_position!=command_position || command_coefficient_position!=command_position)admission_status=BAD_IDENTITY;
        else if(!command_source_valid)admission_status=BAD_PEER;
        bounds_bad=0;
        for(integer i=0;i<3;i=i+1)begin
            region_end[i]={1'b0,bases[i]}+{1'b0,sizes[i]};
            vector_end[i]={1'b0,addresses[i]}+{51'd0,plane_beats(2'(i)),6'd0};
            if(addresses[i][5:0]!=0 || sizes[i]==0 || region_end[i][64] || vector_end[i][64] ||
                addresses[i]<bases[i] || vector_end[i]>region_end[i])bounds_bad=1;
        end
        beat_bad=received>=plane_beats(plane) || reader_last!=(chunk_received+7'd1==chunk_beats);
        for(integer lane=0;lane<16;lane=lane+1)
            if(reader_data[lane*32+23+:8]==8'hff)beat_bad=1;
        fault_now=fault;
        if(active && fault==OK)begin
            if(epoch_bad)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(timeout_now)fault_now=TIMEOUT;
            else if(state==CHECK && bounds_bad)fault_now=BAD_COMMAND;
            else if(state==READ && reader_done && reader_done_ready &&
                (reader_status!=0 || reader_bytes!=chunk_bytes || chunk_received!=chunk_beats))fault_now=READ_ERROR;
            else if(state==READ && reader_valid && reader_ready && beat_bad)fault_now=BAD_FORMAT;
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
        if(rst_n && state==READ && beat_pending && fault_now==OK)begin
            if(plane<2)qk_mem[pending_address+{8'd0,pending_lane}+(plane==1 ? 12'd2048:12'd0)]<=pending_data[pending_lane*32+:32];
            else if(plane==2)coefficient_mem[pending_address+{8'd0,pending_lane}]<=pending_data[pending_lane*32+:32];
        end
        if(rst_n && qk_ce && result_valid && qk_address<12'd2560)qk_data<=qk_mem[qk_address];
        if(rst_n && coefficient_ce && result_valid && coefficient_address<12'd64)coefficient_data<=coefficient_mem[coefficient_address];
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;plane<=0;address<=0;received<=0;stored<=0;chunk_beats<=0;chunk_received<=0;
            idle_cycles<=0;pending_data<=0;pending_address<=0;pending_lane<=0;beat_pending<=0;
            fault<=OK;revoked<=0;cancel_seen<=0;completion_status<=OK;completion_sequence<=0;completion_epoch<=0;completion_beats<=0;completion_position<=0;completion_coefficient_position<=0;
            qk_pending<=0;coefficient_pending<=0;
            for(integer i=0;i<3;i=i+1)begin addresses[i]<=0;bases[i]<=0;sizes[i]<=0;end
        end else begin
            qk_pending<=qk_ce && result_valid && qk_address<12'd2560;
            coefficient_pending<=coefficient_ce && result_valid && coefficient_address<12'd64;
            if(busy && (epoch_bad || cancel_valid))revoked<=1;
            if(cancel_valid && cancel_ready)cancel_seen<=1;
            if(active)begin fault<=fault_now;if(fault_now!=OK)revoked<=1;end
            if(state==ISSUE || state==READ)begin
                if(progress_event)idle_cycles<=0;
                else if(idle_cycles<WATCHDOG_CYCLES)idle_cycles<=idle_cycles+1;
            end else idle_cycles<=0;
            if(state==READ && beat_pending)begin
                if(fault_now!=OK)beat_pending<=0;
                else begin
                    stored<=stored+12'd1;
                    if(pending_lane==15)beat_pending<=0;
                    else pending_lane<=pending_lane+4'd1;
                end
            end
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    addresses[0]<=command_q_address;addresses[1]<=command_k_address;addresses[2]<=command_coefficient_address;
                    bases[0]<=command_q_base;bases[1]<=command_k_base;bases[2]<=command_coefficient_base;
                    sizes[0]<=command_q_bytes;sizes[1]<=command_k_bytes;sizes[2]<=command_coefficient_bytes;
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;completion_beats<=0;
                    completion_position<=command_position;completion_coefficient_position<=command_coefficient_position;
                    completion_status<=admission_status;fault<=admission_status;revoked<=0;cancel_seen<=0;
                    plane<=0;received<=0;stored<=0;beat_pending<=0;
                    state<=admission_status==OK ? CHECK:RESPONSE;
                end
                CHECK:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else begin address<=addresses[0];chunk_beats<=next_chunk(addresses[0],8'd128);state<=ISSUE;end
                ISSUE:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(reader_cmd_ready)begin chunk_received<=0;state<=READ;end
                READ:begin
                    if(reader_valid && reader_ready)begin
                        received<=received+8'd1;chunk_received<=chunk_received+7'd1;
                        completion_beats<=completion_beats+10'd1;
                        if(fault_now==OK)begin
                            pending_data<=reader_data;pending_address<={1'b0,received[6:0],4'd0};pending_lane<=0;beat_pending<=1;
                        end
                    end
                    if(reader_done && reader_done_ready)state<=ADVANCE;
                end
                ADVANCE:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(received==plane_beats(plane))begin
                        if(stored!={plane_beats(plane),4'd0})begin completion_status<=BAD_FORMAT;state<=RESPONSE;end
                        else if(plane==2)begin completion_status<=OK;state<=RESPONSE;end
                        else begin
                            plane<=plane+2'd1;address<=addresses[plane+2'd1];received<=0;stored<=0;
                            chunk_beats<=next_chunk(addresses[plane+2'd1],plane_beats(plane+2'd1));state<=ISSUE;
                        end
                    end else begin
                        address<=address+{32'd0,chunk_bytes};
                        chunk_beats<=next_chunk(address+{32'd0,chunk_bytes},plane_beats(plane)-received);state<=ISSUE;
                    end
                RESPONSE:if(completion_ready)begin state<=IDLE;qk_pending<=0;coefficient_pending<=0;end
                default:begin state<=IDLE;revoked<=1;end
            endcase
        end
    end
endmodule
`default_nettype wire
