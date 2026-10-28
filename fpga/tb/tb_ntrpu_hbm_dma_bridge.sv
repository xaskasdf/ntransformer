`timescale 1ns/1ps
module tb_ntrpu_hbm_dma_bridge;
    import ntrpu_peer_read_pkg::*;
    reg clk=0,dma_clk=0,clock_enable=1;
    integer seed=0,case_id=0,cycles=0,passes=0;
    reg cold_reset_n=0,stop=0;
    wire started,rst_n,dma_rst_n;
    reg cfg_valid=0;
    wire cfg_ready,cfg_error;
    lease_t cfg;
    reg [63:0] cfg_writeback=64'h100000000000;
    reg submit_valid=0,submit_channel=0,reject_ready=0;
    request_t submit_context;
    wire submit_ready,reject_valid;
    wire [3:0] reject_code;
    reg [1:0] data_valid=0,data_last=0,local_valid=0,ack_valid=0,result_ready=0;
    wire [1:0] data_ready,result_valid,halted,dma_fault;
    reg [1023:0] data=0;
    reg [2047:0] data_context=0;
    reg [1159:0] local_event=0,ack_event=0;
    wire [2047:0] result_context;
    wire [7:0] result_status;
    wire [127:0] result_bytes;
    wire [3:0] contexts_held,wire_sent;
    reg dma_link_up=1;
    reg [1:0] dma_run=0,dma_error=0,dma_done=0;
    reg [1:0] dsc_allow=0,axis_allow=0;
    reg [1:0] dsc_ready=0,axis_ready=0;
    wire [1:0] dsc_load,axis_valid,axis_last;
    wire [127:0] dsc_src,dsc_dst,axis_keep;
    wire [55:0] dsc_len;
    wire [31:0] dsc_ctl;
    wire [1023:0] axis_data;
    ntrpu_hbm_dma_bridge dut(.*);
    always #2.5 if(clock_enable)clk=~clk;
    initial begin
        if($value$plusargs("seed=%d",seed))begin end
        #(0.1*(seed%10));forever #2 dma_clk=~dma_clk;
    end
    initial begin #1000000;$fatal(1,"BR WATCHDOG case=%0d",case_id);end
    integer admitted[0:1],descriptors[0:1],packets[0:1],done_count[0:1],beat_index[0:1];
    reg [63:0] expected_seq[0:1][0:15];
    integer expected_len[0:1][0:15];
    reg [63:0] expected_dst[0:1][0:15];
    integer retired[0:1];
    reg [1:0] held_axis,held_result;
    reg [1023:0] previous_axis;
    reg [1:0] previous_last;
    reg [2047:0] previous_context;
    reg [7:0] previous_status;
    reg [127:0] previous_bytes;
    request_t observed;
    integer j,k,q,offset;

    function automatic request_t request_for(input integer channel,seq,length,index);
        request_t r;
        begin
            r='0;r.identity.session=1;r.identity.source=0;r.identity.sequence_id=seq;
            r.source=cfg.source;r.target=cfg.target;r.window_generation=1;r.model_epoch=7;
            r.source_offset=(64'(channel)<<33)+64'(index*4096);
            r.target_offset=(64'(channel)<<32)+64'(index*4096);r.bytes=length;request_for=r;
        end
    endfunction
    function automatic event_t event_for(input request_t r);
        event_t e;
        begin e='0;e.identity=r.identity;e.target=r.target;e.window_generation=r.window_generation;e.bytes=r.bytes;event_for=e;end
    endfunction
    function automatic [511:0] pattern(input integer seq,byte_offset);
        reg [511:0] v;
        integer b;
        begin for(b=0;b<64;b=b+1)v[b*8+:8]=8'((((byte_offset+b)*37)^((byte_offset+b)>>7)^(seq*53)));pattern=v;end
    endfunction

    always @(negedge dma_clk)begin
        cycles=cycles+1;
        dsc_ready[0]=dsc_allow[0] && ((cycles+seed)%4!=0);
        dsc_ready[1]=dsc_allow[1] && ((cycles+seed)%5!=0);
        axis_ready[0]=axis_allow[0] && ((cycles+seed)%3!=0);
        axis_ready[1]=axis_allow[1] && ((cycles+seed)%4!=0);
    end
    always @(posedge dma_clk)begin
        if(!dma_rst_n)begin
            held_axis=0;previous_axis=0;previous_last=0;
            for(integer c=0;c<2;c=c+1)begin descriptors[c]=0;packets[c]=0;done_count[c]=0;beat_index[c]=0;end
        end else begin
            for(integer c=0;c<2;c=c+1)begin
                if(held_axis[c] && (!axis_valid[c] || axis_data[c*512+:512]!==previous_axis[c*512+:512] ||
                    axis_last[c]!==previous_last[c]))$fatal(1,"BR AXIS_UNSTABLE");
                if(dsc_load[c])begin
                    q=descriptors[c];
                    if(!dsc_ready[c] || !dma_run[c] || q>=admitted[c] ||
                       dsc_dst[c*64+:64]!==expected_dst[c][q] || dsc_len[c*28+:28]!==28'(expected_len[c][q]) ||
                       dsc_src[c*64+:64]!==64'h100000000000+64'(c*128+(q%2)*64) ||
                       dsc_ctl[c*16+:16]!==16'h0010)$fatal(1,"BR DESCRIPTOR_MISMATCH c=%0d q=%0d",c,q);
                    $display("BR DESC %0d %0d %0d %016h %016h %0d",case_id,c,expected_seq[c][q],dsc_src[c*64+:64],dsc_dst[c*64+:64],dsc_len[c*28+:28]);
                    descriptors[c]=q+1;
                end
                if(axis_valid[c] && axis_ready[c])begin
                    q=packets[c];offset=beat_index[c]*64;
                    if(q>=descriptors[c] || axis_data[c*512+:512]!==pattern(expected_seq[c][q],offset) ||
                        axis_keep[c*64+:64]!==64'hffffffffffffffff ||
                        axis_last[c]!=(offset+64==expected_len[c][q]))$fatal(1,"BR PAYLOAD_MISMATCH c=%0d q=%0d off=%0d",c,q,offset);
                    $display("BR DATA %0d %0d %0d %0d %0d %0128h",case_id,c,expected_seq[c][q],offset,axis_last[c],axis_data[c*512+:512]);
                    if(axis_last[c])begin packets[c]=q+1;beat_index[c]=0;end
                    else beat_index[c]=beat_index[c]+1;
                end
                if(dma_done[c])begin
                    $display("BR DONE %0d %0d %0d",case_id,c,done_count[c]);done_count[c]=done_count[c]+1;
                end
                held_axis[c]=axis_valid[c] && !axis_ready[c];
            end
            previous_axis=axis_data;previous_last=axis_last;
        end
    end
    always @(posedge clk)begin
        if(!rst_n)begin held_result=0;retired[0]=0;retired[1]=0;end
        else begin
            for(integer c=0;c<2;c=c+1)begin
                if(held_result[c] && (!result_valid[c] || result_context[c*1024+:1024]!==previous_context[c*1024+:1024] ||
                    result_status[c*4+:4]!==previous_status[c*4+:4] || result_bytes[c*64+:64]!==previous_bytes[c*64+:64]))
                    $fatal(1,"BR RESULT_UNSTABLE");
                if(data_valid[c] && data_ready[c])begin
                    observed=data_context[c*1024+:1024];
                    $display("BR PUSH %0d %0d %0d %0d",case_id,c,observed.identity.sequence_id,data_last[c]);
                end
                if(result_valid[c] && result_ready[c])begin
                    observed=result_context[c*1024+:1024];
                    $display("BR RESULT %0d %0d %0d %0d %0d",case_id,c,observed.identity.sequence_id,result_status[c*4+:4],result_bytes[c*64+:64]);
                    retired[c]=retired[c]+1;
                end
                held_result[c]=result_valid[c] && !result_ready[c];
            end
            previous_context=result_context;previous_status=result_status;previous_bytes=result_bytes;
        end
    end
    for(genvar g=0;g<2;g=g+1)begin: completion_observation
        always @(posedge clk)if(rst_n && dut.lanes[g].bridge.event_valid)
            $display("BR EVENT %0d %0d %0d %0d",case_id,g,dut.lanes[g].bridge.completion[1],dut.lanes[g].bridge.completion[0]);
    end

    task automatic reset_case(input integer number);
        begin
            @(negedge clk);cold_reset_n=0;case_id=number;clock_enable=1;
            stop=0;cfg_valid=0;submit_valid=0;data_valid=0;data_last=0;local_valid=0;ack_valid=0;
            result_ready=0;reject_ready=0;dma_link_up=1;dma_run=0;dma_error=0;dma_done=0;dsc_allow=0;axis_allow=0;
            admitted[0]=0;admitted[1]=0;
            cfg='0;cfg.source.session=1;cfg.source.node=0;cfg.source.generation=1;
            cfg.target.session=1;cfg.target.node=1;cfg.target.generation=1;
            cfg.source_size=64'h400000000;cfg.published_bytes=64'h400000000;
            cfg.target_base=64'h1000000000;cfg.target_size=64'h200000000;
            cfg.window_bytes=64'h200000000;cfg.window_generation=1;cfg.model_epoch=7;
            cfg.source_read=1;cfg.target_write=1;cfg.source_open=1;cfg.target_open=1;
            repeat(6)@(negedge clk);cold_reset_n=1;
            wait(started);@(negedge clk);cfg_valid=1;
            @(posedge clk);while(!cfg_ready)@(posedge clk);
            @(negedge clk);cfg_valid=0;
            if(cfg_error)$fatal(1,"BR CFG_ERROR");
            $display("BR CASE %0d %0d",case_id,seed);
        end
    endtask
    task automatic reserve(input integer c,input request_t r,input integer reject_expected);
        begin
            @(negedge clk);submit_channel=1'(c);submit_context=r;submit_valid=1;
            if(reject_expected==0)begin
                expected_seq[c][admitted[c]]=r.identity.sequence_id;
                expected_len[c][admitted[c]]=32'(r.bytes);
                expected_dst[c][admitted[c]]=cfg.target_base+r.target_offset;
                admitted[c]=admitted[c]+1;
            end
            @(posedge clk);while(!submit_ready)@(posedge clk);
            @(negedge clk);submit_valid=0;
            if(reject_expected!=0)begin
                if(!reject_valid || reject_code!=reject_expected)$fatal(1,"BR REJECT_MISMATCH got=%0d wanted=%0d",reject_code,reject_expected);
                reject_ready=1;@(negedge clk);reject_ready=0;
                $display("BR REJECT %0d %0d",case_id,reject_expected);
            end else begin
                if(reject_valid)$fatal(1,"BR UNEXPECTED_REJECT %0d",reject_code);
                $display("BR ADMIT %0d %0d %0d %0d %016h",case_id,c,r.identity.sequence_id,r.bytes,cfg.target_base+r.target_offset);
            end
        end
    endtask
    task automatic beat(input integer c,input request_t r,input integer off,input bit last_bit);
        begin
            @(negedge clk);data_context[c*1024+:1024]=r;data[c*512+:512]=pattern(32'(r.identity.sequence_id),off);
            data_last[c]=last_bit;data_valid[c]=1;
            @(posedge clk);while(!data_ready[c])@(posedge clk);
            @(negedge clk);data_valid[c]=0;
        end
    endtask
    task automatic packet(input integer c,input request_t r);
        begin for(integer b=0;b<r.bytes;b=b+64)beat(c,r,b,b+64==r.bytes);end
    endtask
    task automatic local_finish(input integer c,input request_t r);
        begin
            @(negedge clk);local_event[c*580+:580]=event_for(r);local_valid[c]=1;
            @(negedge clk);local_valid[c]=0;$display("BR LOCAL %0d %0d %0d",case_id,c,r.identity.sequence_id);
        end
    endtask
    task automatic acknowledge(input integer c,input request_t r);
        begin
            @(negedge clk);ack_event[c*580+:580]=event_for(r);ack_valid[c]=1;
            @(negedge clk);ack_valid[c]=0;$display("BR ACK %0d %0d %0d",case_id,c,r.identity.sequence_id);
        end
    endtask
    task automatic done_pulses(input [1:0] mask,input integer count);
        begin
            @(negedge dma_clk);dma_done=mask;
            repeat(count)@(negedge dma_clk);dma_done=0;
        end
    endtask
    task automatic passed(input [255:0] label);
        begin passes=passes+1;$display("BR PASS %0d %0s",case_id,label);end
    endtask
    task automatic check_quarantine(input integer count);
        begin
            repeat(12)@(negedge clk);
            if(!(halted==3) || contexts_held[1:0]+contexts_held[3:2]!=count)$fatal(1,"BR NOT_QUARANTINED held=%h halt=%h",contexts_held,halted);
            result_ready=3;repeat(12)@(negedge clk);result_ready=0;
            if(contexts_held[1:0]+contexts_held[3:2]!=count || cfg_ready || submit_ready)$fatal(1,"BR LEASE_RELEASED");
        end
    endtask

    request_t a,b,c,d,bad;
    event_t bad_ack;
    reg [2047:0] held_offer;
    integer order_index,step_index,operation,slot_index;
    initial begin
        reset_case(0);
        a=request_for(0,1,64,0);b=request_for(1,2,128,0);c=request_for(0,3,4096,1);d=request_for(1,4,192,1);
        reserve(0,a,0);reserve(1,b,0);reserve(0,c,0);reserve(1,d,0);
        if(contexts_held!==4'b1010 || cfg_ready || submit_ready)$fatal(1,"BR RESERVATION_LIMIT");
        // Run/Ready are independent gates; neither may leak a descriptor.
        dsc_allow=3;repeat(8)@(negedge dma_clk);
        if(descriptors[0] || descriptors[1])$fatal(1,"BR RUN_GATE");
        dsc_allow=0;repeat(2)@(negedge dma_clk);dma_run=3;repeat(6)@(negedge dma_clk);
        if(descriptors[0] || descriptors[1])$fatal(1,"BR READY_GATE");
        dsc_allow=3;axis_allow=3;
        fork
            begin packet(0,a);local_finish(0,a);packet(0,c);local_finish(0,c);end
            begin packet(1,b);local_finish(1,b);packet(1,d);local_finish(1,d);end
        join
        wait(packets[0]==2 && packets[1]==2);wait(wire_sent==4'b1111);
        // ACK is allowed before local DMA notification, not before sent data.
        acknowledge(1,b);acknowledge(0,a);acknowledge(1,d);acknowledge(0,c);
        if(result_valid)$fatal(1,"BR EARLY_RESULT");
        @(negedge clk);clock_enable=0;
        done_pulses(3,2);repeat(20)@(negedge dma_clk);
        if(result_valid)$fatal(1,"BR EVENT_CLOCK_BYPASS");
        clock_enable=1;wait(result_valid==3);
        held_offer=result_context;repeat(15)@(negedge clk);
        if(result_context!==held_offer || contexts_held!==4'b1010)$fatal(1,"BR RESULT_STALL");
        result_ready=3;wait(retired[0]==2 && retired[1]==2);repeat(5)@(negedge clk);
        if(contexts_held || halted || dma_fault)$fatal(1,"BR SUCCESS_LEAK");
        passed("dual_cdc_consecutive_completions");
        if($test$plusargs("happy_only"))begin $display("BR ALL_PASS %0d",passes);$finish;end

        reset_case(1);a=request_for(0,1,128,0);reserve(0,a,0);dma_run=3;dsc_allow=3;axis_allow=0;
        beat(0,a,0,0);wait(axis_valid[0]);held_offer={1536'b0,axis_data[511:0]};
        dma_link_up=0;repeat(12)@(negedge clk);
        if(!axis_valid[0] || axis_data[511:0]!==held_offer[511:0])$fatal(1,"BR LOST_OFFER");
        dma_link_up=1;check_quarantine(1);passed("link_loss_preserves_valid");

        reset_case(2);a=request_for(0,1,128,0);reserve(0,a,0);dma_run=3;dsc_allow=3;axis_allow=3;
        beat(0,a,0,0);wait(beat_index[0]==1);dma_error[0]=1;
        repeat(3)@(negedge dma_clk);dma_error=0;check_quarantine(1);passed("partial_dma_error");

        reset_case(3);a=request_for(0,1,128,0);reserve(0,a,0);dma_run=3;dsc_allow=3;
        wait(descriptors[0]==1);done_pulses(1,1);check_quarantine(1);passed("premature_done");

        reset_case(4);a=request_for(0,1,64,0);reserve(0,a,0);
        acknowledge(0,a);check_quarantine(1);passed("premature_ack");

        reset_case(5);a=request_for(0,1,64,0);reserve(0,a,0);dma_run=3;dsc_allow=3;axis_allow=3;
        packet(0,a);wait(wire_sent[0]);bad=a;bad.window_generation=2;
        acknowledge(0,bad);check_quarantine(1);passed("stale_ack_generation");

        reset_case(6);a=request_for(0,1,64,0);reserve(0,a,0);dma_run=3;dsc_allow=3;axis_allow=3;
        packet(0,a);wait(wire_sent[0]);acknowledge(0,a);acknowledge(0,a);
        check_quarantine(1);passed("duplicate_ack");

        reset_case(7);a=request_for(0,1,128,0);reserve(0,a,0);
        @(negedge clk);data_context[1023:0]=a;data[511:0]=pattern(1,0);data_valid[0]=1;data_last[0]=1;
        @(negedge clk);data_valid=0;check_quarantine(1);passed("bad_last");

        reset_case(8);a=request_for(0,1,64,0);reserve(0,a,0);local_finish(0,a);
        check_quarantine(1);passed("premature_local_cpl");

        reset_case(9);a=request_for(0,1,64,0);reserve(0,a,0);
        bad=a;bad.identity.sequence_id=2;reserve(1,bad,OVERLAP);
        bad=a;bad.identity.sequence_id=2;bad.source_offset=64'h3ffffffc0;bad.bytes=128;reserve(1,bad,RANGE);
        bad=a;bad.identity.sequence_id=2;bad.source_offset=4032;bad.bytes=128;bad.target_offset=4096;reserve(1,bad,SPAN);
        bad=a;bad.identity.sequence_id=2;bad.window_generation=2;reserve(1,bad,HANDLE);
        reserve(1,a,REPLAY);
        @(negedge clk);cfg_valid=1;cfg.window_generation=2;repeat(8)@(negedge clk);
        if(cfg_ready || cfg_error || contexts_held!=1 || descriptors[0] || descriptors[1])$fatal(1,"BR BUSY_MAP");
        cfg_valid=0;passed("admission_and_busy_remap");

        reset_case(10);a=request_for(0,1,64,0);reserve(0,a,0);dma_run=3;dsc_allow=3;axis_allow=3;
        packet(0,a);local_finish(0,a);wait(wire_sent[0]);done_pulses(1,1);acknowledge(0,a);
        wait(result_valid[0]);repeat(3)@(negedge clk);result_ready=1;
        wait(retired[0]==1);@(negedge clk);result_ready=0;
        b=request_for(0,2,64,1);reserve(0,b,0);packet(0,b);wait(wire_sent[1]);
        acknowledge(0,a);check_quarantine(1);passed("stale_ack_after_retirement");

        reset_case(11);a=request_for(0,1,128,0);b=request_for(0,2,64,1);reserve(0,a,0);reserve(0,b,0);
        beat(0,a,0,0);@(negedge clk);data_context[1023:0]=b;data[511:0]=pattern(2,0);data_valid[0]=1;data_last[0]=1;
        @(negedge clk);data_valid=0;check_quarantine(2);passed("packet_interleave");

        for(order_index=0;order_index<6;order_index=order_index+1)begin
            reset_case(12+order_index);a=request_for(0,1,64,0);reserve(0,a,0);
            dma_run=3;dsc_allow=3;axis_allow=3;packet(0,a);wait(wire_sent[0]);
            for(step_index=0;step_index<3;step_index=step_index+1)begin
                case(order_index)
                    0:operation=step_index; // local, DMA, ACK
                    1:operation=(step_index==0)?0:((step_index==1)?2:1);
                    2:operation=(step_index==0)?1:((step_index==1)?0:2);
                    3:operation=(step_index==0)?1:((step_index==1)?2:0);
                    4:operation=(step_index==0)?2:((step_index==1)?0:1);
                    default:operation=2-step_index;
                endcase
                case(operation)
                    0:local_finish(0,a);
                    1:done_pulses(1,1);
                    2:acknowledge(0,a);
                endcase
                repeat(8)@(negedge clk);
                if(step_index<2 && result_valid)$fatal(1,"BR MISSING_OBLIGATION_IGNORED");
            end
            wait(result_valid[0]);result_ready=1;wait(retired[0]==1);repeat(3)@(negedge clk);
            if(contexts_held || halted)$fatal(1,"BR COMPLETION_ORDER");
            passed("completion_order");
        end

        reset_case(18);dma_run=3;dsc_allow=3;axis_allow=3;
        for(slot_index=0;slot_index<8;slot_index=slot_index+1)begin
            a=request_for(0,slot_index+1,64,slot_index);reserve(0,a,0);
            if(wire_sent[slot_index%2])$fatal(1,"BR STALE_SENT_GENERATION");
            packet(0,a);local_finish(0,a);wait(wire_sent[slot_index%2]);
            done_pulses(1,1);acknowledge(0,a);wait(result_valid[0]);result_ready=1;
            wait(retired[0]==slot_index+1);@(negedge clk);result_ready=0;
        end
        if(contexts_held || halted)$fatal(1,"BR SLOT_REUSE");passed("ring_and_generation_wrap");

        for(order_index=0;order_index<4;order_index=order_index+1)begin
            reset_case(19+order_index);a=request_for(0,1,64,0);reserve(0,a,0);
            if(order_index>=1)begin dma_run=3;dsc_allow=3;wait(descriptors[0]==1);end
            if(order_index>=2)begin axis_allow=3;packet(0,a);local_finish(0,a);wait(wire_sent[0]);end
            if(order_index>=3)begin done_pulses(1,1);repeat(8)@(negedge clk);end
            dma_link_up=0;check_quarantine(1);dma_link_up=1;
            repeat(10)@(negedge clk);if(!halted || contexts_held!=1)$fatal(1,"BR LINK_REARM");
            passed("link_loss_boundary");
        end

        reset_case(23);a=request_for(0,1,64,0);reserve(0,a,0);dma_run=3;dsc_allow=3;axis_allow=3;
        packet(0,a);local_finish(0,a);wait(wire_sent[0]);done_pulses(1,1);acknowledge(0,a);
        wait(result_valid[0]);held_offer=result_context;
        dma_link_up=0;repeat(12)@(negedge clk);
        if(!result_valid[0] || result_context!==held_offer || result_status[3:0]!=0)$fatal(1,"BR RESULT_REWRITTEN");
        check_quarantine(1);passed("fault_after_result_offer");

        reset_case(24);a=request_for(0,1,64,0);a.identity.sequence_id=64'h100000001;
        reserve(0,a,0);dma_run=3;dsc_allow=3;axis_allow=3;packet(0,a);wait(wire_sent[0]);
        bad=a;bad.identity.sequence_id=1;acknowledge(0,bad);check_quarantine(1);
        passed("identity_high_bits");
        $display("BR ALL_PASS %0d",passes);$finish;
    end
endmodule
