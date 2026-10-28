// Vector gate/up binary64 -> HLS SwiGLU/Q8 -> private ABI1 records in AXI memory.
// All buffers registered, one clock, one outstanding AXI transaction per direction.
// Failed/cancelled vectors may have written private prefixes; publish only on OK.
// Watchdog/cancel drain readers, HLS and writers; coordinated reset recovers hangs.
module ntrpu_swiglu_memory #(
    parameter integer WATCHDOG_CYCLES = 65536
) (
    input logic clk,rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [13:0] job_elements,
    input logic [31:0] job_sequence,job_epoch,expected_epoch,
    input logic [63:0] job_gate_addr,job_gate_base,job_gate_bytes,
    input logic [63:0] job_up_addr,job_up_base,job_up_bytes,
    input logic [63:0] job_output_addr,job_output_base,job_output_bytes,
    input logic [31:0] job_gate_epoch,job_up_epoch,job_output_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
    output logic [8:0] done_blocks,
    output logic [14:0] done_bytes,
    output logic [31:0] done_sequence,done_epoch,done_hls_status,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize,m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid,m_axi_arcache,
    output logic m_axi_arlock,m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast,m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize,m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid,m_axi_awcache,
    output logic m_axi_awlock,m_axi_awvalid,
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_EPOCH=2,READ_ERROR=3,
        WRITE_ERROR=4,COMPUTE_ERROR=5,CANCELLED=7,TIMEOUT=8;
    typedef enum logic [3:0] {IDLE,BOUNDS,CHECK,PLAN,G_ISSUE,G_READ,U_ISSUE,U_READ,
        H_START,H_WAIT,W_ISSUE,W_WAIT,ADVANCE,RESPONSE} state_t;
    state_t state;
    logic [63:0] gate_addr,up_addr,out_addr,gate_base,up_base,out_base,gate_size,up_size,out_size;
    logic [64:0] gate_end,up_end,out_end,gate_region_end,up_region_end,out_region_end;
    logic [16:0] vector_read_bytes;
    logic [14:0] vector_write_bytes;
    logic [13:0] remaining;
    logic [5:0] block_elements;
    logic [8:0] block_bytes;
    logic [2:0] read_beats;
    logic [31:0] idle_cycles;
    logic [3:0] error_now;
    logic reading,reader_cmd,reader_cmd_ready,reader_done,reader_out_valid,reader_out_ready,reader_last;
    logic [1:0] reader_status;
    logic [31:0] reader_bytes;
    logic [511:0] reader_data,record_data;
    logic writer_cmd,writer_ready,writer_done,writer_cancel_ready;
    logic [3:0] writer_status;
    logic [6:0] writer_bytes;
    logic [31:0] writer_sequence,writer_epoch;
    logic hls_done,hls_idle,hls_ready,gate_ce,up_ce,record_ce,record_we;
    logic [4:0] gate_index,up_index;
    logic [2:0] record_index;
    logic [63:0] gate_q,up_q,record_word;
    logic [31:0] hls_status;
    logic [63:0] gate_memory [0:31],up_memory [0:31];
    logic [7:0] word_seen,word_seen_after;
    logic bad_words,record_write,progress_event,watchdog_expired;

    assign job_ready=state==IDLE;
    assign done_valid=state==RESPONSE;
    assign cancel_ready=state!=IDLE && state!=RESPONSE && done_status==OK;
    assign reading=state==G_READ || state==U_READ;
    assign reader_cmd=(state==G_ISSUE || state==U_ISSUE) && error_now==OK;
    assign reader_out_ready=reading;
    assign writer_cmd=state==W_ISSUE && error_now==OK;
    assign record_write=(state==H_START || state==H_WAIT) && record_ce && record_we;
    assign word_seen_after=word_seen | (record_write ? (8'b1 << record_index):8'b0);
    assign progress_event=(m_axi_arvalid && m_axi_arready) || (m_axi_rvalid && m_axi_rready) ||
        (reader_out_valid && reader_out_ready) || ((state==G_ISSUE || state==U_ISSUE) && reader_cmd_ready) ||
        (reading && reader_done) || record_write || (state==H_WAIT && hls_done) ||
        (state==W_ISSUE && writer_ready) || (m_axi_awvalid && m_axi_awready) ||
        (m_axi_wvalid && m_axi_wready) || (state==W_WAIT && writer_done);
    assign watchdog_expired=state>=G_ISSUE && state<=W_WAIT &&
        idle_cycles>=32'(WATCHDOG_CYCLES-1) && !progress_event;
    always_comb begin
        error_now=done_status;
        if(done_status==OK) begin
            if(cancel_valid && cancel_ready) error_now=CANCELLED;
            else if(watchdog_expired) error_now=TIMEOUT;
            else if(reading && reader_done && (reader_status!=0 || reader_bytes!={23'b0,block_bytes} ||
                    read_beats!={1'b0,block_bytes[8:6]})) error_now=READ_ERROR;
            else if(reading && reader_out_valid && reader_out_ready && (read_beats>=4 ||
                    reader_last!=(read_beats+3'd1==block_bytes[8:6]))) error_now=READ_ERROR;
            else if(state==H_WAIT && hls_done && (hls_status!=0 || bad_words ||
                    word_seen_after!=8'hff || (record_write && word_seen[record_index]))) error_now=COMPUTE_ERROR;
            else if(state==W_WAIT && writer_done && (writer_status!=0 || writer_bytes!=64 ||
                    writer_sequence!=done_sequence || writer_epoch!=done_epoch)) error_now=WRITE_ERROR;
        end
    end
    ntrpu_axi_read_master reader (
        .clk,.rst_n,.cmd_valid(reader_cmd),.cmd_ready(reader_cmd_ready),
        .cmd_addr(state==G_ISSUE ? gate_addr:up_addr),.cmd_bytes({23'b0,block_bytes}),
        .done_valid(reader_done),.done_ready(reading),.done_status(reader_status),.done_bytes(reader_bytes),
        .out_data(reader_data),.out_valid(reader_out_valid),.out_ready(reader_out_ready),.out_last(reader_last),
        .m_axi_araddr,.m_axi_arlen,.m_axi_arsize,.m_axi_arprot,.m_axi_arburst,.m_axi_arid,
        .m_axi_arcache,.m_axi_arlock,.m_axi_arvalid,.m_axi_arready,.m_axi_rdata,.m_axi_rresp,
        .m_axi_rid,.m_axi_rlast,.m_axi_rvalid,.m_axi_rready
    );
    ntrpu_swiglu_q8_block activation (
        .ap_clk(clk),.ap_rst(!rst_n),.ap_start(state==H_START),.ap_done(hls_done),
        .ap_idle(hls_idle),.ap_ready(hls_ready),.ap_return(hls_status),.elements({26'b0,block_elements}),
        .gate_address0(gate_index),.gate_ce0(gate_ce),.gate_q0(gate_q),
        .up_address0(up_index),.up_ce0(up_ce),.up_q0(up_q),
        .record_r_address0(record_index),.record_r_ce0(record_ce),.record_r_we0(record_we),.record_r_d0(record_word)
    );
    ntrpu_activation_writer #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) writer (
        .clk,.rst_n,.cmd_valid(writer_cmd),.cmd_ready(writer_ready),.cmd_addr(out_addr),
        .cmd_region_base(out_base),.cmd_region_bytes(out_size),.cmd_record(record_data),
        .cmd_elements(block_elements),.cmd_sequence(done_sequence),.cmd_epoch(done_epoch),.expected_epoch(done_epoch),
        .cancel_valid(state==W_WAIT && error_now!=OK),.cancel_ready(writer_cancel_ready),
        .done_valid(writer_done),.done_ready(state==W_WAIT),.done_status(writer_status),.done_bytes(writer_bytes),
        .done_sequence(writer_sequence),.done_epoch(writer_epoch),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,
        .m_axi_awcache,.m_axi_awlock,.m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,
        .m_axi_wlast,.m_axi_wvalid,.m_axi_wready,.m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready
    );
    // HLS ap_memory read latency is one cycle; output words are assembled privately.
    always_ff @(posedge clk) begin
        if(gate_ce) gate_q<=gate_memory[gate_index];
        if(up_ce) up_q<=up_memory[up_index];
        if(reading && reader_out_valid && reader_out_ready && error_now==OK)
            for(int lane=0;lane<8;lane++) begin
                if(state==G_READ) gate_memory[{read_beats[1:0],3'b000}+5'(lane)]<=reader_data[lane*64+:64];
                else up_memory[{read_beats[1:0],3'b000}+5'(lane)]<=reader_data[lane*64+:64];
            end
    end
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;gate_addr<=0;up_addr<=0;out_addr<=0;gate_base<=0;up_base<=0;out_base<=0;
            gate_size<=0;up_size<=0;out_size<=0;gate_end<=0;up_end<=0;out_end<=0;
            gate_region_end<=0;up_region_end<=0;out_region_end<=0;vector_read_bytes<=0;vector_write_bytes<=0;
            remaining<=0;block_elements<=0;block_bytes<=0;read_beats<=0;idle_cycles<=0;
            word_seen<=0;record_data<=0;bad_words<=0;
            done_status<=OK;done_blocks<=0;done_bytes<=0;done_sequence<=0;done_epoch<=0;done_hls_status<=0;
        end else begin
            if(state!=IDLE && state!=RESPONSE) done_status<=error_now;
            if(state>=G_ISSUE && state<=W_WAIT && !progress_event) begin
                if(idle_cycles<32'(WATCHDOG_CYCLES)) idle_cycles<=idle_cycles+32'd1;
            end else idle_cycles<=0;
            if(reader_out_valid && reader_out_ready) read_beats<=read_beats+3'd1;
            if(record_write) begin
                word_seen<=word_seen_after;record_data[record_index*64+:64]<=record_word;
                if(word_seen[record_index]) bad_words<=1;
            end
            case(state)
                IDLE: if(job_valid) begin
                    gate_addr<=job_gate_addr;up_addr<=job_up_addr;out_addr<=job_output_addr;
                    gate_base<=job_gate_base;up_base<=job_up_base;out_base<=job_output_base;
                    gate_size<=job_gate_bytes;up_size<=job_up_bytes;out_size<=job_output_bytes;
                    remaining<=job_elements;
                    vector_read_bytes<=17'((({1'b0,job_elements}+15'd7)>>3)<<6);
                    vector_write_bytes<=15'((({1'b0,job_elements}+15'd31)>>5)<<6);
                    done_status<=OK;done_blocks<=0;done_bytes<=0;done_hls_status<=0;
                    done_sequence<=job_sequence;done_epoch<=job_epoch;
                    if(job_elements==0 || job_elements>8192 || job_gate_addr[5:0]!=0 || job_up_addr[5:0]!=0 ||
                       job_output_addr[5:0]!=0 || job_gate_bytes==0 || job_up_bytes==0 || job_output_bytes==0) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(job_epoch!=expected_epoch || job_gate_epoch!=job_epoch ||
                                job_up_epoch!=job_epoch || job_output_epoch!=job_epoch) begin
                        done_status<=BAD_EPOCH;state<=RESPONSE;
                    end else state<=BOUNDS;
                end
                BOUNDS: begin
                    gate_end<={1'b0,gate_addr}+{48'b0,vector_read_bytes};
                    up_end<={1'b0,up_addr}+{48'b0,vector_read_bytes};
                    out_end<={1'b0,out_addr}+{50'b0,vector_write_bytes};
                    gate_region_end<={1'b0,gate_base}+{1'b0,gate_size};
                    up_region_end<={1'b0,up_base}+{1'b0,up_size};
                    out_region_end<={1'b0,out_base}+{1'b0,out_size};
                    if(error_now!=OK) state<=RESPONSE;else state<=CHECK;
                end
                CHECK: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(gate_end[64] || up_end[64] || out_end[64] || gate_region_end[64] ||
                        up_region_end[64] || out_region_end[64] || gate_addr<gate_base || up_addr<up_base || out_addr<out_base ||
                        gate_end>gate_region_end || up_end>up_region_end || out_end>out_region_end ||
                        ({1'b0,out_addr}<gate_end && {1'b0,gate_addr}<out_end) ||
                        ({1'b0,out_addr}<up_end && {1'b0,up_addr}<out_end)) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else state<=PLAN;
                end
                PLAN: begin
                    block_elements<=remaining>32 ? 6'd32:remaining[5:0];
                    block_bytes<=remaining>32 ? 9'd256:9'((({1'b0,remaining}+15'd7)>>3)<<6);
                    if(error_now!=OK) state<=RESPONSE;else state<=G_ISSUE;
                end
                G_ISSUE,U_ISSUE: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(reader_cmd_ready) begin read_beats<=0;state<=state==G_ISSUE ? G_READ:U_READ;end
                end
                G_READ,U_READ: if(reader_done) begin
                    if(error_now!=OK) state<=RESPONSE;
                    else state<=state==G_READ ? U_ISSUE:H_START;
                end
                H_START: begin
                    word_seen<=0;record_data<=0;bad_words<=0;state<=H_WAIT;
                end
                H_WAIT: if(hls_done) begin
                    done_hls_status<=hls_status;
                    if(error_now!=OK) state<=RESPONSE;else state<=W_ISSUE;
                end
                W_ISSUE: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(writer_ready) state<=W_WAIT;
                end
                W_WAIT: if(writer_done) begin
                    done_bytes<=done_bytes+{8'b0,writer_bytes};
                    if(error_now!=OK) state<=RESPONSE;
                    else begin done_blocks<=done_blocks+9'd1;state<=ADVANCE;end
                end
                ADVANCE: begin
                    if(error_now!=OK || remaining<=32) state<=RESPONSE;
                    else begin
                        remaining<=remaining-14'd32;gate_addr<=gate_addr+64'd256;
                        up_addr<=up_addr+64'd256;out_addr<=out_addr+64'd64;state<=PLAN;
                    end
                end
                RESPONSE: if(done_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
