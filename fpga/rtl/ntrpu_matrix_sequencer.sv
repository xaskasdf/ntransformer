// Compact matrix layout: [group of <=32 rows][K tile][K block][row][34 Q8 bytes].
// Every tile is a whole number of 64-byte beats; no implicit row/K padding.
// The entire destination stays private until a successful matrix completion.
// All ports share clk. Reset must also quiesce/reset the engine and AXI peers.
module ntrpu_matrix_sequencer #(parameter bit FP32_CAPABLE=0) (
    input logic clk, rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [1:0] job_layout,
    input logic [13:0] job_rows,
    input logic job_output_fp32,
    input logic [8:0] job_k_blocks,
    input logic [5:0] job_tile_k,
    input logic [31:0] job_sequence, job_epoch,
    input logic [63:0] job_weight_addr, job_region_base, job_region_bytes,
    input logic [63:0] job_activation_addr, job_activation_region_base, job_activation_region_bytes,
    input logic [31:0] job_activation_epoch,
    input logic [63:0] job_output_addr, job_output_region_base, job_output_region_bytes,
    input logic [31:0] job_output_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,
    output logic [16:0] done_bytes,
    output logic [13:0] done_rows,
    output logic [31:0] done_sequence, done_epoch,
    output logic e_job_valid,
    input logic e_job_ready,
    output logic [1:0] e_job_layout,
    output logic [5:0] e_job_rows,
    output logic e_job_output_fp32,
    output logic [8:0] e_job_k_blocks,
    output logic [31:0] e_job_sequence, e_job_epoch,
    output logic [63:0] e_job_region_base, e_job_region_bytes,
    output logic [63:0] e_job_activation_addr, e_job_activation_region_base, e_job_activation_region_bytes,
    output logic [31:0] e_job_activation_epoch,
    output logic [63:0] e_job_output_addr, e_job_output_region_base, e_job_output_region_bytes,
    output logic [31:0] e_job_output_epoch,
    output logic e_tile_valid,
    input logic e_tile_ready,
    output logic [63:0] e_tile_addr,
    output logic [8:0] e_tile_k_base,
    output logic [5:0] e_tile_k_count,
    output logic [31:0] e_tile_tag, e_tile_epoch,
    output logic e_cancel_valid,
    input logic e_cancel_ready,
    input logic e_done_valid,
    output logic e_done_ready,
    input logic [3:0] e_done_status,
    input logic [8:0] e_done_bytes,
    input logic [31:0] e_done_sequence, e_done_epoch
);
    typedef enum logic [3:0] {IDLE, SHAPE, SPANS, BOUNDS, RANGES, FORMAT,
                             GROUP, START, RUN, ADVANCE, RESPONSE} state_t;
    state_t state;
    logic [1:0] layout;
    logic [13:0] total_rows, remaining_rows;
    logic [5:0] tile_k, tail_rows, scan_count;
    logic [8:0] scan_k, k_after;
    logic [10:0] scan_product;
    logic [15:0] row_bytes;
    logic [29:0] weight_span;
    logic [16:0] output_span;
    logic [8:0] expected_group_bytes;
    logic [18:0] group_bytes;
    logic [16:0] tile_bytes;
    logic [10:0] tile_blocks;
    typedef enum logic [2:0] {TCOUNT, TBLOCKS, TBYTES, TOFFER, TFINISHED} tile_state_t;
    tile_state_t tile_state;
    logic [63:0] weight_start, group_address;
    logic [64:0] weight_end, weight_region_end, activation_end, activation_region_end;
    logic [64:0] output_end, output_region_end;
    logic cancel_pending, cancel_sent, cancel_fire, tile_fire, completion_bad;

    assign job_ready=state==IDLE;
    assign done_valid=state==RESPONSE;
    assign cancel_ready=state!=IDLE && state!=RESPONSE && !cancel_pending;
    assign cancel_fire=cancel_valid && cancel_ready;
    assign e_job_valid=state==START;
    assign e_job_layout=2;
    assign e_job_epoch=done_epoch;
    assign e_tile_epoch=done_epoch;
    assign e_tile_valid=state==RUN && tile_state==TOFFER;
    assign tile_fire=e_tile_valid && e_tile_ready;
    assign k_after=e_tile_k_base+(tile_fire ? {3'b0,e_tile_k_count}:9'd0);
    assign e_cancel_valid=state==RUN && (cancel_pending || cancel_fire) && !cancel_sent;
    assign e_done_ready=state==RUN;
    assign scan_count=e_job_k_blocks-scan_k>{3'b0,tile_k} ? tile_k:6'(e_job_k_blocks-scan_k);
    assign scan_product=11'(tail_rows)*11'(scan_count);
    assign expected_group_bytes=e_job_output_fp32 ? {1'b0,e_job_rows,2'b0}:{e_job_rows,3'b0};
    assign completion_bad=e_done_sequence!=e_job_sequence || e_done_epoch!=done_epoch ||
        e_done_bytes>expected_group_bytes ||
        (e_done_status==0 && (e_done_bytes!=expected_group_bytes || k_after!=e_job_k_blocks));

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;layout<=0;total_rows<=0;remaining_rows<=0;tile_k<=0;tail_rows<=0;
            scan_k<=0;row_bytes<=0;weight_span<=0;output_span<=0;group_bytes<=0;
            tile_state<=TCOUNT;tile_blocks<=0;tile_bytes<=0;e_tile_k_count<=0;
            weight_start<=0;group_address<=0;weight_end<=0;weight_region_end<=0;
            activation_end<=0;activation_region_end<=0;output_end<=0;output_region_end<=0;
            cancel_pending<=0;cancel_sent<=0;done_status<=0;done_bytes<=0;done_rows<=0;done_sequence<=0;done_epoch<=0;
            e_job_rows<=0;e_job_output_fp32<=0;e_job_k_blocks<=0;e_job_sequence<=0;e_job_region_base<=0;e_job_region_bytes<=0;
            e_job_activation_addr<=0;e_job_activation_region_base<=0;e_job_activation_region_bytes<=0;e_job_activation_epoch<=0;
            e_job_output_addr<=0;e_job_output_region_base<=0;e_job_output_region_bytes<=0;e_job_output_epoch<=0;
            e_tile_addr<=0;e_tile_k_base<=0;e_tile_tag<=0;
        end else if(cancel_fire && state!=START && state!=RUN) begin
            done_status<=7;cancel_pending<=1;state<=RESPONSE;
        end else begin
            if(cancel_fire) cancel_pending<=1;
            if(e_cancel_valid && e_cancel_ready) cancel_sent<=1;
            case(state)
                IDLE: if(job_valid) begin
                    layout<=job_layout;e_job_output_fp32<=FP32_CAPABLE && job_output_fp32;total_rows<=job_rows;remaining_rows<=job_rows;tile_k<=job_tile_k;
                    e_job_k_blocks<=job_k_blocks;done_sequence<=job_sequence;done_epoch<=job_epoch;
                    e_job_sequence<=job_sequence;done_status<=0;done_bytes<=0;done_rows<=0;
                    weight_start<=job_weight_addr;group_address<=job_weight_addr;
                    e_job_region_base<=job_region_base;e_job_region_bytes<=job_region_bytes;
                    e_job_activation_addr<=job_activation_addr;e_job_activation_region_base<=job_activation_region_base;
                    e_job_activation_region_bytes<=job_activation_region_bytes;e_job_activation_epoch<=job_activation_epoch;
                    e_job_output_addr<=job_output_addr;e_job_output_region_base<=job_output_region_base;
                    e_job_output_region_bytes<=job_output_region_bytes;e_job_output_epoch<=job_output_epoch;
                    cancel_pending<=0;cancel_sent<=0;scan_k<=0;e_tile_tag<=0;state<=SHAPE;
                end
                SHAPE: begin
                    if(layout!=2 || total_rows==0 || total_rows>8192 || e_job_k_blocks==0 || e_job_k_blocks>256 ||
                       tile_k==0 || tile_k>32 || weight_start[5:0]!=0 || e_job_activation_addr[5:0]!=0 ||
                       e_job_output_addr[5:0]!=0 || e_job_region_bytes==0 ||
                       e_job_activation_region_bytes==0 || e_job_output_region_bytes==0) begin
                        done_status<=1;state<=RESPONSE;
                    end else if(e_job_activation_epoch!=done_epoch || e_job_output_epoch!=done_epoch) begin
                        done_status<=2;state<=RESPONSE;
                    end else begin
                        row_bytes<=(16'(e_job_k_blocks)<<5)+(16'(e_job_k_blocks)<<1);
                        tail_rows<=total_rows[4:0]==0 ? 6'd32:{1'b0,total_rows[4:0]};state<=SPANS;
                    end
                end
                SPANS: begin
                    weight_span<=30'(total_rows)*30'(row_bytes);
                    output_span<=e_job_output_fp32 ? (((17'(total_rows)+17'd15)>>4)<<6):(((17'(total_rows)+17'd7)>>3)<<6);state<=BOUNDS;
                end
                BOUNDS: begin
                    weight_end<={1'b0,weight_start}+65'(weight_span);
                    weight_region_end<={1'b0,e_job_region_base}+{1'b0,e_job_region_bytes};
                    activation_end<={1'b0,e_job_activation_addr}+(65'(e_job_k_blocks)<<6);
                    activation_region_end<={1'b0,e_job_activation_region_base}+{1'b0,e_job_activation_region_bytes};
                    output_end<={1'b0,e_job_output_addr}+65'(output_span);
                    output_region_end<={1'b0,e_job_output_region_base}+{1'b0,e_job_output_region_bytes};state<=RANGES;
                end
                RANGES: begin
                    if(weight_end[64] || weight_region_end[64] || activation_end[64] || activation_region_end[64] ||
                       output_end[64] || output_region_end[64] || weight_start<e_job_region_base || weight_end>weight_region_end ||
                       e_job_activation_addr<e_job_activation_region_base || activation_end>activation_region_end ||
                       e_job_output_addr<e_job_output_region_base || output_end>output_region_end ||
                       ({1'b0,e_job_output_addr}<weight_region_end && output_end>{1'b0,e_job_region_base}) ||
                       ({1'b0,e_job_output_addr}<activation_region_end && output_end>{1'b0,e_job_activation_region_base})) begin
                        done_status<=1;state<=RESPONSE;
                    end else state<=FORMAT;
                end
                FORMAT: begin
                    if(scan_product[4:0]!=0) begin done_status<=1;state<=RESPONSE;end
                    else if(scan_k+{3'b0,scan_count}==e_job_k_blocks) state<=GROUP;
                    else scan_k<=scan_k+{3'b0,scan_count};
                end
                GROUP: begin
                    e_job_rows<=remaining_rows>32 ? 6'd32:6'(remaining_rows);
                    group_bytes<=19'(remaining_rows>32 ? 6'd32:6'(remaining_rows))*19'(row_bytes);
                    e_tile_addr<=group_address;e_tile_k_base<=0;tile_state<=TCOUNT;cancel_sent<=0;state<=START;
                end
                START: if(e_job_ready) state<=RUN;
                RUN: begin
                    case(tile_state)
                        TCOUNT: begin
                            e_tile_k_count<=e_job_k_blocks-e_tile_k_base>{3'b0,tile_k} ? tile_k:6'(e_job_k_blocks-e_tile_k_base);
                            tile_state<=TBLOCKS;
                        end
                        TBLOCKS: begin tile_blocks<=11'(e_job_rows)*11'(e_tile_k_count);tile_state<=TBYTES;end
                        TBYTES: begin tile_bytes<=(17'(tile_blocks)<<5)+(17'(tile_blocks)<<1);tile_state<=TOFFER;end
                        default: begin end
                    endcase
                    if(tile_fire) begin
                        e_tile_addr<=e_tile_addr+64'(tile_bytes);e_tile_k_base<=k_after;e_tile_tag<=e_tile_tag+1;
                        tile_state<=k_after==e_job_k_blocks ? TFINISHED:TCOUNT;
                    end
                    if(e_done_valid) begin
                        if(e_done_bytes<=expected_group_bytes) done_bytes<=done_bytes+17'(e_done_bytes);
                        if(completion_bad) begin done_status<=5;state<=RESPONSE;end
                        else begin
                            if(e_done_status==0) done_rows<=done_rows+14'(e_job_rows);
                            if(e_done_status!=0) begin done_status<=e_done_status;state<=RESPONSE;end
                            else if(cancel_pending || cancel_fire) begin done_status<=7;state<=RESPONSE;end
                            else if(remaining_rows==14'(e_job_rows)) state<=RESPONSE;
                            else state<=ADVANCE;
                        end
                    end
                end
                ADVANCE: begin
                    remaining_rows<=remaining_rows-14'(e_job_rows);group_address<=group_address+64'(group_bytes);
                    e_job_output_addr<=e_job_output_addr+(64'(e_job_rows)<<(e_job_output_fp32 ? 2:3));
                    e_job_sequence<=e_job_sequence+1;state<=GROUP;
                end
                RESPONSE: if(done_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
