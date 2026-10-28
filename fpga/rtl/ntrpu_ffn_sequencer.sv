// Root FFN descriptors: 0=x ABI1, 1=Wgate, 2=Wup, 3=Wdown,
// 4=gate binary64, 5=up binary64, 6=SwiGLU ABI1, 7=down FP32 when OUTPUT_FP32, otherwise binary64.
// Matrix layout2; K dimensions rounded to blocks of32 by the host packer.
// Every destination is private until the successful fourth completion.
module ntrpu_ffn_sequencer #(parameter bit OUTPUT_FP32=1) (
    input logic clk,rst_n,
    input logic job_valid,
    output logic job_ready,
    input logic [1:0] job_layout,
    input logic [13:0] job_dim,job_hidden,
    input logic [5:0] job_tile_k [0:2],
    input logic [63:0] job_addr [0:7],job_base [0:7],job_bytes [0:7],
    input logic [31:0] job_buffer_epoch [0:7],
    input logic [31:0] job_sequence,job_epoch,expected_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,done_child_status,
    output logic [2:0] done_stages,
    output logic [1:0] done_phase,
    output logic [16:0] done_bytes,
    output logic [17:0] done_stage_bytes,
    output logic [31:0] done_sequence,done_epoch,done_hls_status,
    output logic activation_selected,
    output logic m_job_valid,
    input logic m_job_ready,
    output logic [1:0] m_job_layout,
    output logic [13:0] m_job_rows,
    output logic m_job_output_fp32,
    output logic [8:0] m_job_k_blocks,
    output logic [5:0] m_job_tile_k,
    output logic [31:0] m_job_sequence,m_job_epoch,
    output logic [63:0] m_job_weight_addr,m_job_region_base,m_job_region_bytes,
    output logic [63:0] m_job_activation_addr,m_job_activation_region_base,m_job_activation_region_bytes,
    output logic [31:0] m_job_activation_epoch,
    output logic [63:0] m_job_output_addr,m_job_output_region_base,m_job_output_region_bytes,
    output logic [31:0] m_job_output_epoch,
    output logic m_cancel_valid,
    input logic m_cancel_ready,m_done_valid,
    output logic m_done_ready,
    input logic [3:0] m_done_status,
    input logic [16:0] m_done_bytes,
    input logic [13:0] m_done_rows,
    input logic [31:0] m_done_sequence,m_done_epoch,
    output logic a_job_valid,
    input logic a_job_ready,
    output logic [13:0] a_job_elements,
    output logic [31:0] a_job_sequence,a_job_epoch,a_expected_epoch,
    output logic [63:0] a_job_gate_addr,a_job_gate_base,a_job_gate_bytes,
    output logic [63:0] a_job_up_addr,a_job_up_base,a_job_up_bytes,
    output logic [63:0] a_job_output_addr,a_job_output_base,a_job_output_bytes,
    output logic [31:0] a_job_gate_epoch,a_job_up_epoch,a_job_output_epoch,
    output logic a_cancel_valid,
    input logic a_cancel_ready,a_done_valid,
    output logic a_done_ready,
    input logic [3:0] a_done_status,
    input logic [8:0] a_done_blocks,
    input logic [14:0] a_done_bytes,
    input logic [31:0] a_done_sequence,a_done_epoch,a_done_hls_status
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_EPOCH=2,
        BAD_COMPLETION=5,CHILD_ERROR=6,CANCELLED=7,TIMEOUT=8;
    typedef enum logic [3:0] {IDLE,SHAPE,K_BLOCKS,ROW_BYTES,SPANS,BOUND_SUM,BOUND_CHECK,
        OVERLAP,FORMAT,START,OFFER,RUN,ADVANCE,RESPONSE} state_t;
    state_t state;
    logic [1:0] layout,phase,format_matrix;
    logic [13:0] dim,hidden;
    logic [8:0] dim_blocks,hidden_blocks,scan_k,scan_limit;
    logic [5:0] tile_k [0:2];
    logic [5:0] scan_count,tail_rows;
    logic [10:0] scan_product;
    logic [15:0] gate_row_bytes,down_row_bytes;
    logic [63:0] address [0:7],region_base [0:7],region_bytes [0:7];
    logic [31:0] epochs [0:7];
    logic [29:0] spans [0:7];
    logic [64:0] range_end [0:7],region_end [0:7];
    logic [31:0] expected;
    logic [2:0] range_index,write_index,other_index,weight_index,input_index,output_index;
    logic cancel_pending,cancel_sent,cancel_fire,child_done,completion_bad;
    logic [3:0] child_status;
    logic [16:0] child_bytes,expected_matrix_bytes;
    logic [31:0] child_sequence;
    assign job_ready=state==IDLE;
    assign done_valid=state==RESPONSE;
    assign cancel_ready=state!=IDLE && state!=RESPONSE && !cancel_pending;
    assign cancel_fire=cancel_valid && cancel_ready;
    assign activation_selected=phase==2;
    assign child_sequence=done_sequence+{30'b0,phase};
    assign weight_index=phase==0 ? 3'd1:phase==1 ? 3'd2:3'd3;
    assign input_index=phase==3 ? 3'd6:3'd0;
    assign output_index=phase==0 ? 3'd4:phase==1 ? 3'd5:3'd7;
    assign m_job_valid=state==OFFER && phase!=2;
    assign m_job_layout=2;
    assign m_job_output_fp32=OUTPUT_FP32 && phase==3;
    assign m_job_rows=phase==3 ? dim:hidden;
    assign m_job_k_blocks=phase==3 ? hidden_blocks:dim_blocks;
    assign m_job_tile_k=tile_k[phase==3 ? 2:phase];
    assign m_job_sequence=child_sequence;
    assign m_job_epoch=done_epoch;
    assign m_job_weight_addr=address[weight_index];
    assign m_job_region_base=region_base[weight_index];
    assign m_job_region_bytes=region_bytes[weight_index];
    assign m_job_activation_addr=address[input_index];
    assign m_job_activation_region_base=region_base[input_index];
    assign m_job_activation_region_bytes=region_bytes[input_index];
    assign m_job_activation_epoch=done_epoch;
    assign m_job_output_addr=address[output_index];
    assign m_job_output_region_base=region_base[output_index];
    assign m_job_output_region_bytes=region_bytes[output_index];
    assign m_job_output_epoch=done_epoch;
    assign a_job_valid=state==OFFER && phase==2;
    assign a_job_elements=hidden;
    assign a_job_sequence=child_sequence;
    assign a_job_epoch=done_epoch;
    assign a_expected_epoch=done_epoch;
    assign a_job_gate_addr=address[4];assign a_job_gate_base=region_base[4];assign a_job_gate_bytes=region_bytes[4];
    assign a_job_up_addr=address[5];assign a_job_up_base=region_base[5];assign a_job_up_bytes=region_bytes[5];
    assign a_job_output_addr=address[6];assign a_job_output_base=region_base[6];assign a_job_output_bytes=region_bytes[6];
    assign a_job_gate_epoch=done_epoch;assign a_job_up_epoch=done_epoch;assign a_job_output_epoch=done_epoch;
    assign m_cancel_valid=state==RUN && phase!=2 && (cancel_pending || cancel_fire) && !cancel_sent;
    assign a_cancel_valid=state==RUN && phase==2 && (cancel_pending || cancel_fire) && !cancel_sent;
    assign m_done_ready=state==RUN && phase!=2;
    assign a_done_ready=state==RUN && phase==2;
    assign child_done=phase==2 ? a_done_valid:m_done_valid;
    assign child_status=phase==2 ? a_done_status:m_done_status;
    assign child_bytes=phase==2 ? {2'b0,a_done_bytes}:m_done_bytes;
    assign expected_matrix_bytes=m_job_output_fp32 ? {1'b0,m_job_rows,2'b0}:{m_job_rows,3'b0};
    assign completion_bad=phase==2 ?
        (a_done_sequence!=child_sequence || a_done_epoch!=done_epoch || a_done_blocks>hidden_blocks ||
         {15'b0,a_done_bytes}>spans[6] || a_done_bytes[5:0]!=0 ||
         (a_done_status==OK && (a_done_hls_status!=0 || a_done_blocks!=hidden_blocks || {15'b0,a_done_bytes}!=spans[6]))) :
        (m_done_sequence!=child_sequence || m_done_epoch!=done_epoch || m_done_rows>m_job_rows ||
         m_done_bytes>expected_matrix_bytes || (m_job_output_fp32 ? m_done_bytes[1:0]!=0:m_done_bytes[2:0]!=0) ||
         (m_done_status==OK && (m_done_rows!=m_job_rows || m_done_bytes!=expected_matrix_bytes)));
    assign scan_limit=format_matrix==2 ? hidden_blocks:dim_blocks;
    assign scan_count=scan_limit-scan_k>{3'b0,tile_k[format_matrix]} ? tile_k[format_matrix]:6'(scan_limit-scan_k);
    assign tail_rows=format_matrix==2 ? (dim[4:0]==0 ? 6'd32:{1'b0,dim[4:0]}) :
                                                (hidden[4:0]==0 ? 6'd32:{1'b0,hidden[4:0]});
    assign scan_product=11'(tail_rows)*11'(scan_count);
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;layout<=0;phase<=0;format_matrix<=0;dim<=0;hidden<=0;dim_blocks<=0;hidden_blocks<=0;
            scan_k<=0;gate_row_bytes<=0;down_row_bytes<=0;range_index<=0;write_index<=4;other_index<=0;
            expected<=0;cancel_pending<=0;cancel_sent<=0;
            done_status<=OK;done_child_status<=0;done_stages<=0;done_phase<=0;done_bytes<=0;
            done_stage_bytes<=0;done_sequence<=0;done_epoch<=0;done_hls_status<=0;
            for(int i=0;i<3;i++) tile_k[i]<=0;
            for(int i=0;i<8;i++) begin
                address[i]<=0;region_base[i]<=0;region_bytes[i]<=0;epochs[i]<=0;
                spans[i]<=0;range_end[i]<=0;region_end[i]<=0;
            end
        end else if(cancel_fire && state!=OFFER && state!=RUN) begin
            cancel_pending<=1;done_status<=CANCELLED;state<=RESPONSE;
        end else begin
            if(cancel_fire) begin cancel_pending<=1;done_status<=CANCELLED;end
            if((m_cancel_valid && m_cancel_ready) || (a_cancel_valid && a_cancel_ready)) cancel_sent<=1;
            case(state)
                IDLE: if(job_valid) begin
                    layout<=job_layout;dim<=job_dim;hidden<=job_hidden;expected<=expected_epoch;
                    done_sequence<=job_sequence;done_epoch<=job_epoch;done_status<=OK;done_child_status<=0;
                    done_stages<=0;done_phase<=0;done_bytes<=0;done_stage_bytes<=0;done_hls_status<=0;
                    phase<=0;format_matrix<=0;scan_k<=0;range_index<=0;write_index<=4;other_index<=0;
                    cancel_pending<=0;cancel_sent<=0;
                    for(int i=0;i<3;i++) tile_k[i]<=job_tile_k[i];
                    for(int i=0;i<8;i++) begin
                        address[i]<=job_addr[i];region_base[i]<=job_base[i];region_bytes[i]<=job_bytes[i];epochs[i]<=job_buffer_epoch[i];
                    end
                    state<=SHAPE;
                end
                SHAPE: begin
                    if(layout!=2 || dim==0 || dim>8192 || hidden==0 || hidden>8192 ||
                       tile_k[0]==0 || tile_k[0]>32 || tile_k[1]==0 || tile_k[1]>32 || tile_k[2]==0 || tile_k[2]>32) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(done_epoch!=expected) begin done_status<=BAD_EPOCH;state<=RESPONSE;end
                    else state<=K_BLOCKS;
                end
                K_BLOCKS: begin
                    dim_blocks<=9'(({1'b0,dim}+15'd31)>>5);hidden_blocks<=9'(({1'b0,hidden}+15'd31)>>5);state<=ROW_BYTES;
                end
                ROW_BYTES: begin
                    gate_row_bytes<=(16'(dim_blocks)<<5)+(16'(dim_blocks)<<1);
                    down_row_bytes<=(16'(hidden_blocks)<<5)+(16'(hidden_blocks)<<1);state<=SPANS;
                end
                SPANS: begin
                    spans[0]<=30'(dim_blocks)<<6;
                    spans[1]<=30'(hidden)*30'(gate_row_bytes);spans[2]<=30'(hidden)*30'(gate_row_bytes);
                    spans[3]<=30'(dim)*30'(down_row_bytes);
                    spans[4]<=((30'(hidden)+30'd7)>>3)<<6;spans[5]<=((30'(hidden)+30'd7)>>3)<<6;
                    spans[6]<=30'(hidden_blocks)<<6;spans[7]<=OUTPUT_FP32 ? (((30'(dim)+30'd15)>>4)<<6):(((30'(dim)+30'd7)>>3)<<6);state<=BOUND_SUM;
                end
                BOUND_SUM: begin
                    range_end[range_index]<={1'b0,address[range_index]}+{35'b0,spans[range_index]};
                    region_end[range_index]<={1'b0,region_base[range_index]}+{1'b0,region_bytes[range_index]};state<=BOUND_CHECK;
                end
                BOUND_CHECK: begin
                    if(address[range_index][5:0]!=0 || region_bytes[range_index]==0 ||
                       range_end[range_index][64] || region_end[range_index][64] ||
                       address[range_index]<region_base[range_index] || range_end[range_index]>region_end[range_index]) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(epochs[range_index]!=done_epoch) begin done_status<=BAD_EPOCH;state<=RESPONSE;end
                    else if(range_index==7) state<=OVERLAP;
                    else begin range_index<=range_index+3'd1;state<=BOUND_SUM;end
                end
                OVERLAP: begin
                    if(write_index!=other_index && {1'b0,address[write_index]}<region_end[other_index] &&
                       {1'b0,region_base[other_index]}<range_end[write_index]) begin
                        done_status<=BAD_COMMAND;state<=RESPONSE;
                    end else if(other_index==7) begin
                        other_index<=0;
                        if(write_index==7) state<=FORMAT;else write_index<=write_index+3'd1;
                    end else other_index<=other_index+3'd1;
                end
                FORMAT: begin
                    if(scan_product[4:0]!=0) begin done_status<=BAD_COMMAND;state<=RESPONSE;end
                    else if(scan_k+{3'b0,scan_count}==scan_limit) begin
                        scan_k<=0;
                        if(format_matrix==2) state<=START;else format_matrix<=format_matrix+2'd1;
                    end else scan_k<=scan_k+{3'b0,scan_count};
                end
                START: state<=OFFER;
                OFFER: if(phase==2 ? a_job_ready:m_job_ready) state<=RUN;
                RUN: if(child_done) begin
                    done_phase<=phase;done_child_status<=child_status;
                    if(phase==2) done_hls_status<=a_done_hls_status;
                    if(!completion_bad) begin
                        done_stage_bytes<=done_stage_bytes+{1'b0,child_bytes};
                        if(phase==3) done_bytes<=child_bytes;
                    end
                    if(cancel_pending || cancel_fire) begin done_status<=CANCELLED;state<=RESPONSE;end
                    else if(completion_bad) begin done_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else if(child_status!=OK) begin done_status<=child_status==TIMEOUT ? TIMEOUT:CHILD_ERROR;state<=RESPONSE;end
                    else begin
                        done_stages<=done_stages+3'd1;
                        if(phase==3) state<=RESPONSE;else state<=ADVANCE;
                    end
                end
                ADVANCE: begin phase<=phase+2'd1;cancel_sent<=0;state<=START;end
                RESPONSE: if(done_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
