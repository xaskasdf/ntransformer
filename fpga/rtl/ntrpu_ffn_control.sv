// Privileged single-clock MMIO control for one FFN root. Experimental ABI 1/2.
// The system must coordinate rst_n with the compute engine and its AXI peer.
module ntrpu_ffn_control #(parameter bit OUTPUT_FP32=1, EXTERNAL_RETIRE=0) (
    input logic clk,rst_n,
    input logic [31:0] s_axil_awaddr,s_axil_wdata,s_axil_araddr,
    input logic [2:0] s_axil_awprot,s_axil_arprot,
    input logic [3:0] s_axil_wstrb,
    input logic s_axil_awvalid,s_axil_wvalid,s_axil_bready,s_axil_arvalid,s_axil_rready,
    output logic s_axil_awready,s_axil_wready,s_axil_bvalid,s_axil_arready,s_axil_rvalid,
    output logic [1:0] s_axil_bresp,s_axil_rresp,
    output logic [31:0] s_axil_rdata,
    output logic irq,
    output logic job_valid,
    input logic job_ready,
    output logic [1:0] job_layout,
    output logic [13:0] job_dim,job_hidden,
    output logic [5:0] job_tile_k [0:2],
    output logic [63:0] job_addr [0:7],job_base [0:7],job_bytes [0:7],
    output logic [31:0] job_buffer_epoch [0:7],
    output logic [31:0] job_sequence,job_epoch,expected_epoch,
    output logic cancel_valid,
    input logic cancel_ready,
    input logic done_valid,
    output logic done_ready,
    input logic [3:0] done_status,done_child_status,
    input logic [2:0] done_stages,
    input logic [1:0] done_phase,
    input logic [16:0] done_bytes,
    input logic [17:0] done_stage_bytes,
    input logic [31:0] done_sequence,done_epoch,done_hls_status,
    output logic notify_event_valid,notify_event_irq,notify_service_valid,notify_retire_ready,
    output logic [31:0] notify_sequence,notify_epoch,
    input logic notify_event_ready,notify_service_ready,notify_service_rejected,notify_retire_valid,
    input logic [31:0] notify_retire_sequence,notify_retire_epoch
);
    localparam logic [1:0] OKAY=0,SLVERR=2,DECERR=3;
    logic aw_pending,w_pending,running,completion_pending,irq_enable;
    logic cancel_requested,cancel_sent;
    logic completion_irq,notification_sent,service_requested,service_sent,notification_fault;
    logic [31:0] ack_sequence,ack_epoch;
    logic [31:0] write_address,write_data,byte_mask;
    logic [3:0] write_strobes;
    logic [31:0] config_words[0:5],buffer_words[0:7][0:6],completion_words[0:8];
    logic [31:0] accepted_count,completed_count;
    logic [63:0] elapsed_cycles;
    logic write_fire,job_fire,done_fire;
    logic [2:0] write_kind;
    logic [1:0] write_response;
    logic [31:0] merged_word,allowed_bits,command;
    logic [33:0] read_value;
    logic [16:0] expected_output_bytes;
    integer config_index,buffer_index,buffer_field;

    assign s_axil_awready=rst_n && !aw_pending && !s_axil_bvalid;
    assign s_axil_wready=rst_n && !w_pending && !s_axil_bvalid;
    assign s_axil_arready=rst_n && !s_axil_rvalid;
    assign write_fire=aw_pending && w_pending && !s_axil_bvalid;
    assign job_fire=job_valid && job_ready;
    assign done_ready=running && !completion_pending;
    assign done_fire=done_valid && done_ready;
    assign cancel_valid=running && cancel_requested && !cancel_sent;
    assign irq=completion_pending && (EXTERNAL_RETIRE ? completion_irq:irq_enable);
    assign notify_event_valid=EXTERNAL_RETIRE && completion_pending && !notification_sent && !notification_fault;
    assign notify_event_irq=completion_irq;
    assign notify_sequence=job_sequence;
    assign notify_epoch=job_epoch;
    assign notify_service_valid=EXTERNAL_RETIRE && notification_sent && service_requested && !service_sent && !notification_fault;
    assign notify_retire_ready=EXTERNAL_RETIRE && completion_pending && notification_sent && service_sent &&
        !notification_fault && notify_retire_sequence==job_sequence && notify_retire_epoch==job_epoch;
    assign expected_output_bytes=OUTPUT_FP32 ? {1'b0,job_dim,2'b0}:{job_dim,3'b0};

    function automatic logic address_valid(input logic [31:0] a);
        return a[1:0]==0 && (a<=32'h34 || (a>=32'h40 && a<=32'h60) ||
               (EXTERNAL_RETIRE && (a==32'h38 || a==32'h3c || a==32'h64 || a==32'h68)) ||
               (a>=32'h100 && a<32'h200 && a[4:2]!=7));
    endfunction
    function automatic logic [31:0] read_register(input logic [31:0] a);
        if(a>=32'h20 && a<=32'h34) return config_words[(a-32'h20)>>2];
        if(a>=32'h40 && a<=32'h60) return completion_words[(a-32'h40)>>2];
        if(a>=32'h100 && a<32'h200 && a[4:2]!=7) return buffer_words[a[7:5]][a[4:2]];
        case(a)
            32'h00:return 32'h4652544e; // Little-endian "NTRF".
            32'h04:return EXTERNAL_RETIRE ? 32'h00020000:32'h00010000;
            32'h08:return {28'b0,EXTERNAL_RETIRE,1'b1,1'b1,OUTPUT_FP32};
            32'h0c:return {23'b0,(EXTERNAL_RETIRE && notify_event_ready),notification_fault,
                notification_sent,service_requested,cancel_requested,cancel_valid,completion_pending,running,job_valid};
            32'h10:return 0;
            32'h14:return {31'b0,irq_enable};
            32'h18:return accepted_count;
            32'h1c:return completed_count;
            32'h38:return ack_sequence;
            32'h3c:return ack_epoch;
            32'h64:return job_sequence;
            32'h68:return job_epoch;
            default:return 0;
        endcase
    endfunction
    always_comb begin
        read_value=address_valid(s_axil_araddr) ? {OKAY,read_register(s_axil_araddr)}:{DECERR,32'b0};
        for(int lane=0;lane<4;lane++) byte_mask[lane*8+:8]={8{write_strobes[lane]}};
        command=write_data & byte_mask;
        config_index=int'((write_address-32'h20)>>2);
        buffer_index=int'(write_address[7:5]);buffer_field=int'(write_address[4:2]);
        write_kind=0;write_response=OKAY;merged_word=0;allowed_bits=32'hffffffff;
        if(!address_valid(write_address)) write_response=DECERR;
        else if(write_address==32'h10) begin
            write_kind=1;
            case(command)
                0:begin end
                1:if(job_valid || running || completion_pending ||
                    (EXTERNAL_RETIRE && (!notify_event_ready || notification_fault))) write_response=SLVERR;
                2:if(!(job_valid || running) || cancel_requested || done_fire) write_response=SLVERR;
                4:if(!completion_pending || (EXTERNAL_RETIRE && (service_requested || notification_fault ||
                    ack_sequence!=job_sequence || ack_epoch!=job_epoch))) write_response=SLVERR;
                default:write_response=SLVERR;
            endcase
        end else if(write_address==32'h14) begin
            write_kind=2;allowed_bits=1;
            merged_word=({31'b0,irq_enable}&~byte_mask)|command;
        end else if(write_address>=32'h20 && write_address<=32'h34) begin
            write_kind=3;merged_word=(config_words[config_index]&~byte_mask)|command;
            case(config_index)
                3:allowed_bits=32'h3fff3fff;
                4:allowed_bits=32'h00000003;
                5:allowed_bits=32'h003f3f3f;
                default:begin end
            endcase
        end else if(write_address>=32'h100 && write_address<32'h200) begin
            write_kind=4;merged_word=(buffer_words[buffer_index][buffer_field]&~byte_mask)|command;
        end else if(EXTERNAL_RETIRE && (write_address==32'h38 || write_address==32'h3c)) begin
            write_kind=5;merged_word=((write_address==32'h38 ? ack_sequence:ack_epoch)&~byte_mask)|command;
        end else write_response=SLVERR; // Known read-only register.
        if((merged_word & ~allowed_bits)!=0) write_response=SLVERR;
    end

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            aw_pending<=0;w_pending<=0;write_address<=0;write_data<=0;write_strobes<=0;
            s_axil_bvalid<=0;s_axil_bresp<=OKAY;s_axil_rvalid<=0;s_axil_rresp<=OKAY;s_axil_rdata<=0;
            running<=0;completion_pending<=0;irq_enable<=0;job_valid<=0;
            completion_irq<=0;notification_sent<=0;service_requested<=0;service_sent<=0;notification_fault<=0;
            ack_sequence<=0;ack_epoch<=0;
            cancel_requested<=0;cancel_sent<=0;elapsed_cycles<=0;accepted_count<=0;completed_count<=0;
            job_layout<=0;job_dim<=0;job_hidden<=0;job_sequence<=0;job_epoch<=0;expected_epoch<=0;
            for(int i=0;i<3;i++) job_tile_k[i]<=0;
            for(int i=0;i<6;i++) config_words[i]<=0;
            for(int i=0;i<9;i++) completion_words[i]<=0;
            for(int i=0;i<8;i++) begin
                job_addr[i]<=0;job_base[i]<=0;job_bytes[i]<=0;job_buffer_epoch[i]<=0;
                for(int f=0;f<7;f++) buffer_words[i][f]<=0;
            end
        end else begin
            if(s_axil_awvalid && s_axil_awready) begin aw_pending<=1;write_address<=s_axil_awaddr;end
            if(s_axil_wvalid && s_axil_wready) begin w_pending<=1;write_data<=s_axil_wdata;write_strobes<=s_axil_wstrb;end
            if(s_axil_bvalid && s_axil_bready) s_axil_bvalid<=0;
            if(s_axil_rvalid && s_axil_rready) s_axil_rvalid<=0;
            if(s_axil_arvalid && s_axil_arready) begin
                s_axil_rvalid<=1;s_axil_rresp<=read_value[33:32];s_axil_rdata<=read_value[31:0];
            end
            if(running) elapsed_cycles<=elapsed_cycles+64'd1;
            if(job_fire) begin job_valid<=0;running<=1;elapsed_cycles<=0;accepted_count<=accepted_count+32'd1;end
            if(cancel_valid && cancel_ready) cancel_sent<=1;
            if(EXTERNAL_RETIRE) begin
                if(notify_event_valid && notify_event_ready) notification_sent<=1;
                if(notify_service_valid && notify_service_ready) service_sent<=1;
                // Reject unexpected identities or order; only a coordinated reset recovers.
                if(notify_service_rejected || (notify_retire_valid && !notify_retire_ready)) notification_fault<=1;
                if(notify_retire_valid && notify_retire_ready && !notify_service_rejected) begin
                    completion_pending<=0;notification_sent<=0;service_requested<=0;service_sent<=0;
                end
            end
            if(done_fire) begin
                running<=0;completion_pending<=1;completed_count<=completed_count+32'd1;
                completion_irq<=irq_enable;
                completion_words[0]<={19'b0,done_phase,done_stages,done_child_status,done_status};
                completion_words[1]<={15'b0,done_bytes};completion_words[2]<={14'b0,done_stage_bytes};
                completion_words[3]<=done_sequence;completion_words[4]<=done_epoch;completion_words[5]<=done_hls_status;
                completion_words[6]<=32'(elapsed_cycles+64'd1);
                completion_words[7]<=32'((elapsed_cycles+64'd1)>>32);
                completion_words[8]<={29'b0,
                    (done_status==3 || done_status==4 || done_status>8 ||
                     (done_status==0 && (done_stages!=4 || done_phase!=3 ||
                     done_bytes!=expected_output_bytes || done_child_status!=0 || done_hls_status!=0))),
                    done_epoch!=job_epoch,done_sequence!=job_sequence};
            end
            if(write_fire) begin
                aw_pending<=0;w_pending<=0;s_axil_bvalid<=1;s_axil_bresp<=write_response;
                if(write_response==OKAY) case(write_kind)
                    1:case(command)
                        1:begin
                            job_valid<=1;cancel_requested<=0;cancel_sent<=0;
                            job_sequence<=config_words[0];job_epoch<=config_words[1];expected_epoch<=config_words[2];
                            job_dim<=config_words[3][13:0];job_hidden<=config_words[3][29:16];job_layout<=config_words[4][1:0];
                            for(int i=0;i<3;i++) job_tile_k[i]<=config_words[5][i*8+:6];
                            for(int i=0;i<8;i++) begin
                                job_addr[i]<={buffer_words[i][1],buffer_words[i][0]};
                                job_base[i]<={buffer_words[i][3],buffer_words[i][2]};
                                job_bytes[i]<={buffer_words[i][5],buffer_words[i][4]};
                                job_buffer_epoch[i]<=buffer_words[i][6];
                            end
                        end
                        2:cancel_requested<=1;
                        4:if(EXTERNAL_RETIRE) service_requested<=1;else completion_pending<=0;
                        default:begin end
                    endcase
                    2:irq_enable<=merged_word[0];
                    3:config_words[config_index]<=merged_word;
                    4:buffer_words[buffer_index][buffer_field]<=merged_word;
                    5:if(write_address==32'h38) ack_sequence<=merged_word;else ack_epoch<=merged_word;
                    default:begin end
                endcase
            end
        end
    end
endmodule
