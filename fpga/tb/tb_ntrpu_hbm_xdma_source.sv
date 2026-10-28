`timescale 1ns/1ps
module tb_ntrpu_hbm_xdma_source;
    import ntrpu_peer_read_pkg::*;
    reg clk=0,dma_clk=0,cold_reset_n=0,stop=0,dma_link_up=1;
    always #2.5 clk=~clk;
    always #2 dma_clk=~dma_clk;
    wire started,rst_n,dma_rst_n,cfg_ready,cfg_error,submit_ready,reject_valid;
    reg cfg_valid=0,submit_valid=0,submit_channel=0,reject_ready=1;
    lease_t cfg;
    request_t submit_context;
    reg [63:0] cfg_writeback=64'h0000000300001000;
    wire [3:0] reject_code;
    reg [1:0] ack_valid=0,result_ready=0;
    reg [1159:0] ack_event=0;
    wire reader_stop,source_fault,admission_pending;
    wire [3:0] source_held;
    wire [1:0] cmd_valid,cmd_ready,r_data_valid,r_data_last,r_data_ready,r_cpl_valid,r_cpl_ready;
    wire [127:0] cmd_addr,cmd_tag,r_data_tag,r_cpl_tag;
    wire [15:0] cmd_len;
    wire [1023:0] r_data;
    wire [3:0] r_cpl_status;
    wire [25:0] r_cpl_bytes;
    wire [1:0] frame_fault,frame_idle;
    wire reader_started=started;
    wire reader_fault=|frame_fault;
    wire [1:0] result_valid,halted,dma_fault;
    wire [2047:0] result_context;
    wire [7:0] result_status;
    wire [127:0] result_bytes;
    wire [3:0] contexts_held,wire_sent;
    
    reg [1:0] dma_run=3;
    reg [1:0] dma_error=0,dma_done=0;
    reg [1:0] dsc_ready=3,axis_ready=3;
    wire [1:0] dsc_load,axis_valid,axis_last;
    wire [127:0] dsc_src,dsc_dst,axis_keep;
    wire [55:0] dsc_len;
    wire [31:0] dsc_ctl;
    wire [1023:0] axis_data;

    reg management_core_ready=1,exclusive_owner=1,management_fault=0;
    reg management_wait_expired=0,management_reset_request=0,guard_start=0;
    reg [63:0] guard_epoch=0;
    wire guard_start_ready,guard_active;
    wire [15:0] guard_faults;
    wire [1:0] admit_descriptor,validated_done;
    wire [63:0] published_count,observed_count,delivered_count,control_snapshot,status_snapshot,count_snapshot;
    wire [31:0] m_axil_awaddr,m_axil_wdata,m_axil_araddr;
    wire [2:0] m_axil_awprot,m_axil_arprot;
    wire [3:0] m_axil_wstrb;
    wire m_axil_awvalid,m_axil_wvalid,m_axil_bready,m_axil_arvalid,m_axil_rready;
    reg m_axil_awready=0,m_axil_wready=0,m_axil_bvalid=0,m_axil_arready=0,m_axil_rvalid=0;
    reg [1:0] m_axil_bresp=0,m_axil_rresp=0;
    reg [31:0] m_axil_rdata=0;
    ntrpu_hbm_xdma_source #(.GUARD_WAIT_LIMIT(4000),.MASTER_WAIT_LIMIT(48)) dut(.*);
    integer phase=0,seed=0,ticks=0,passes=0,round_number=0;
    integer desc[0:1],packet[0:1],offsets[0:1],done_sent[0:1],retired[0:1],credits[0:1];
    integer ar_count[0:1],cpl_count[0:1],command_count[0:1],rejections=0;
    reg allow_done=0,pending=0,ar_open=0,cmd_gate=1,axis_open=1;
    reg [31:0] address_q=0;
    integer delay_q=0;
    wire [1:0] arvalid;
    reg [1:0] arready=0,rvalid=0,rlast=0;
    wire [1:0] rready;
    wire [67:0] araddr;
    wire [15:0] arlen;
    wire [127:0] frame_data_tag,frame_cpl_tag;
    wire [25:0] frame_cpl_bytes;
    assign r_data_tag=frame_data_tag ^ ((phase==12)?128'h0000000100000000:128'b0);
    assign r_cpl_tag=frame_cpl_tag ^ ((phase==14)?128'h0000000100000000:128'b0);
    assign r_cpl_bytes=frame_cpl_bytes ^ ((phase==13)?26'd64:26'b0);
    reg [1023:0] rdata=0;
    reg [7:0] rid=0;
    reg [3:0] rresp=0;
    reg [1:0] mem_busy=0;
    integer mem_beats[0:1],mem_offset[0:1],mem_request[0:1],source_ticks=0;
    reg [1:0] held_axis=0,held_result=0,held_command=0,held_ar=0,held_read=0;
    reg [1023:0] prev_data,prev_read;
    reg [1:0] prev_last,prev_read_last;
    reg [2047:0] prev_context;
    reg [7:0] prev_status;
    reg [127:0] prev_addr,prev_tag,prev_read_tag;
    reg [15:0] prev_len,prev_arlen;
    reg [67:0] prev_araddr;
    for(genvar lane=0;lane<2;lane=lane+1)begin: frames
        wire frame_ready;
        assign cmd_ready[lane]=frame_ready && cmd_gate;
        ntrpu_hbm_read_frames #(.FRAMES(2)) frame(
            .clk(clk),.rst_n(rst_n),.admit(started && !reader_stop),
            .cmd_valid(cmd_valid[lane] && cmd_gate),.cmd_ready(frame_ready),
            .cmd_addr(cmd_addr[lane*64+:64]),.cmd_tag(cmd_tag[lane*64+:64]),.cmd_len(cmd_len[lane*8+:8]),
            .data_valid(r_data_valid[lane]),.data_ready(r_data_ready[lane]),.data(r_data[lane*512+:512]),
            .data_last(r_data_last[lane]),.data_tag(frame_data_tag[lane*64+:64]),
            .cpl_valid(r_cpl_valid[lane]),.cpl_ready(r_cpl_ready[lane]),
            .cpl_tag(frame_cpl_tag[lane*64+:64]),.cpl_status(r_cpl_status[lane*2+:2]),.cpl_bytes(frame_cpl_bytes[lane*13+:13]),
            .fault(frame_fault[lane]),.idle(frame_idle[lane]),
            .arvalid(arvalid[lane]),.arready(arready[lane]),.araddr(araddr[lane*34+:34]),.arlen(arlen[lane*8+:8]),
            .rvalid(rvalid[lane]),.rready(rready[lane]),.rdata(rdata[lane*512+:512]),
            .rid(rid[lane*4+:4]),.rresp(rresp[lane*2+:2]),.rlast(rlast[lane]));
    end
    task require(input bit ok,input string why);
        if(!ok)$fatal(1,"SJ ASSERT phase=%0d %s",phase,why);
    endtask
    function automatic bit success_case;
        return phase<=3 || phase==10 || phase==11 || phase==15;
    endfunction
    function automatic integer length_for(input integer i);
        case(i%4) 0:length_for=64;1:length_for=128;2:length_for=256;3:length_for=4096;endcase
    endfunction
    function automatic [63:0] destination(input integer i);
        return 64'h100000800+(64'(i%2)<<32)+64'((i/2)*4096);
    endfunction
    function automatic [63:0] wb_address(input integer i);
        return 64'h300001000+64'((i%2)*128+((i/2)%2)*64);
    endfunction
    function automatic [7:0] pattern(input integer i,offset);
        return 8'((offset*37)^(offset>>7)^((i+1)*53));
    endfunction
    function automatic request_t context_for(input integer i);
        request_t c;
        c='0;c.identity.session=1;c.identity.source=0;c.identity.sequence_id=64'(i+1);
        c.source=cfg.source;c.target=cfg.target;c.window_generation=cfg.window_generation;c.model_epoch=7;
        c.source_offset=64'h200000000*64'(i%2)+64'((i/2)*4096);
        if(phase==15)c.source_offset=64'h200000000*64'((i%2)^((i/2)%2))+64'((i/2)*4096);
        c.target_offset=destination(i)-cfg.target_base;c.bytes=64'(length_for(i));return c;
    endfunction
    function automatic event_t event_for(input integer i);
        request_t c;event_t e;c=context_for(i);e='0;e.identity=c.identity;e.target=c.target;
        e.window_generation=c.window_generation;e.bytes=c.bytes;return e;
    endfunction
    always @(negedge dma_clk)begin
        if(!dma_rst_n || !cold_reset_n)begin m_axil_arready=0;m_axil_rvalid=0;pending=0;dma_done=0;ticks=0;end
        else begin
            ticks++;
            dsc_ready[0]=(ticks+seed)%4!=0;dsc_ready[1]=(ticks+seed)%5!=0;
            axis_ready[0]=axis_open && (ticks+seed)%3!=0;axis_ready[1]=axis_open && (ticks+seed)%4!=0;
            dma_done=0;
            for(integer c=0;c<2;c++)if(allow_done && done_sent[c]<packet[c])dma_done[c]=1;
            m_axil_arready=!pending && !m_axil_rvalid && (ticks+seed)%3!=0;
            if(pending && !m_axil_rvalid)begin
                if(delay_q>0)delay_q--;
                else begin
                    case(address_q[7:0])
                        'h04:m_axil_rdata='h00f83e7f;
                        'h40:m_axil_rdata=1;
                        'h48:m_axil_rdata=done_sent[address_q[8]];
                        default:$fatal(1,"SJ management address");
                    endcase
                    m_axil_rresp=0;m_axil_rvalid=1;
                end
            end
        end
    end
    always @(posedge dma_clk)begin
        if(!dma_rst_n || !cold_reset_n)begin
            held_axis=0;
            for(integer c=0;c<2;c++)begin desc[c]=0;packet[c]=0;offsets[c]=0;done_sent[c]=0;credits[c]=0;end
        end else begin
            require(!m_axil_awvalid && !m_axil_wvalid,"read-only management");
            if(m_axil_arvalid && m_axil_arready)begin
                require(!pending,"one AR");pending<=1;address_q<=m_axil_araddr;delay_q<=1+(ticks+seed)%4;
                $display("SJ MGMT %0d AR %h",phase,m_axil_araddr);
            end
            if(m_axil_rvalid && m_axil_rready)begin
                pending<=0;m_axil_rvalid<=0;$display("SJ MGMT %0d R %h %h",phase,m_axil_rdata,m_axil_rresp);
            end
            for(integer c=0;c<2;c++)begin
                if(dsc_load[c])begin
                    require(admit_descriptor[c] && dsc_ready[c] && desc[c]<4,"guarded descriptor");
                    require(dsc_dst[c*64+:64]==destination(c+2*desc[c]) && dsc_src[c*64+:64]==wb_address(c+2*desc[c]) &&
                            dsc_len[c*28+:28]==length_for(c+2*desc[c]) && dsc_ctl[c*16+:16]==16'h10,"descriptor contents");
                    $display("SJ DESC %0d %0d %h %h %h %h",phase,c,dsc_src[c*64+:64],dsc_dst[c*64+:64],dsc_len[c*28+:28],dsc_ctl[c*16+:16]);
                    desc[c]++;
                end
                if(held_axis[c])require(axis_valid[c] && axis_data[c*512+:512]===prev_data[c*512+:512] &&
                                        axis_last[c]===prev_last[c],"stalled DMA stable");
                if(axis_valid[c] && axis_ready[c])begin
                    require(packet[c]<desc[c] && axis_keep[c*64+:64]==64'hffffffffffffffff,"packet descriptor");
                    for(integer b=0;b<64;b++)require(axis_data[c*512+b*8+:8]===pattern(c+2*packet[c],offsets[c]+b),"payload bytes");
                    $display("SJ AXIS %0d %0d %h %h %h",phase,c,axis_data[c*512+:512],axis_keep[c*64+:64],axis_last[c]);
                    offsets[c]+=64;require(axis_last[c]==(offsets[c]==length_for(c+2*packet[c])),"last boundary");
                    if(axis_last[c])begin packet[c]++;offsets[c]=0;end
                end
                if(dma_done[c])begin done_sent[c]++;$display("SJ DONE %0d %0d",phase,c);end
                if(validated_done[c])begin require(credits[c]<done_sent[c],"backed credit");credits[c]++;$display("SJ CREDIT %0d %0d",phase,c);end
                held_axis[c]=axis_valid[c]&&!axis_ready[c];
            end
            prev_data=axis_data;prev_last=axis_last;
        end
    end
    always @(negedge clk)begin
        if(!rst_n)begin arready=0;rvalid=0;mem_busy=0;source_ticks=0;end
        else begin
            source_ticks++;
            for(integer c=0;c<2;c++)begin
                arready[c]=ar_open && !mem_busy[c] && (source_ticks+seed+c)%3!=0;
                if(mem_busy[c] && !rvalid[c] && (source_ticks+seed+c)%4!=0)begin
                    for(integer b=0;b<64;b++)rdata[c*512+b*8+:8]=pattern(mem_request[c],mem_offset[c]+b);
                    rlast[c]=(mem_beats[c]==1);rid[c*4+:4]=0;rresp[c*2+:2]=0;
                    if(c==0 && mem_request[c]==0)begin
                        if(phase==4)rresp[c*2+:2]=2;
                        if(phase==5)rid[c*4+:4]=1;
                        if(phase==6)rlast[c]=0;
                    end
                    rvalid[c]=1;
                end
            end
        end
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            held_result=0;held_command=0;held_ar=0;held_read=0;rejections=0;
            for(integer c=0;c<2;c++)begin retired[c]=0;ar_count[c]=0;command_count[c]=0;cpl_count[c]=0;end
        end else begin
            if(cfg_valid && cfg_ready)$display("SJ CFG %0d %h",phase,cfg);
            if(phase==12 && r_data_valid[0])require(!r_data_ready[0],"invalid 64-bit read tag held");
            if(phase==14 && r_cpl_valid[0])require(!r_cpl_ready[0],"invalid 64-bit CPL tag held");
            if(phase==13 && r_cpl_valid[0])require(!dut.local_valid[0],"bad CPL cannot become successful local event");
            if(submit_valid && submit_ready)$display("SJ SUBMIT %0d %0d %h",phase,submit_channel,submit_context);
            if(reject_valid && reject_ready)begin rejections++;$display("SJ REJECT %0d %h",phase,reject_code);end
            for(integer c=0;c<2;c++)begin
                if(held_command[c])require(cmd_valid[c] && cmd_addr[c*64+:64]===prev_addr[c*64+:64] &&
                    cmd_tag[c*64+:64]===prev_tag[c*64+:64] && cmd_len[c*8+:8]===prev_len[c*8+:8],"stalled command stable");
                if(held_ar[c])require(arvalid[c] && araddr[c*34+:34]===prev_araddr[c*34+:34] && arlen[c*8+:8]===prev_arlen[c*8+:8],"stalled AR stable");
                if(held_read[c])require(r_data_valid[c] && r_data[c*512+:512]===prev_read[c*512+:512] &&
                    r_data_last[c]===prev_read_last[c] && r_data_tag[c*64+:64]===prev_read_tag[c*64+:64],"stalled read stable");
                if(cmd_valid[c] && cmd_ready[c])begin
                    require(source_held[c*2+:2]!=0 && contexts_held[c*2+:2]!=0,"both contexts reserved before command");
                    $display("SJ CMD %0d %0d %h %h %h",phase,c,cmd_addr[c*64+:64],cmd_tag[c*64+:64],cmd_len[c*8+:8]);command_count[c]++;
                end
                if(arvalid[c] && arready[c])begin
                    require(ar_count[c]<command_count[c],"AR after reservation");
                    mem_busy[c]<=1;mem_beats[c]=int'(arlen[c*8+:8])+1;mem_offset[c]=0;
                    mem_request[c]=c+2*ar_count[c];ar_count[c]++;
                    $display("SJ AR %0d %0d %h %h",phase,c,araddr[c*34+:34],arlen[c*8+:8]);
                end
                if(rvalid[c] && rready[c])begin
                    $display("SJ R %0d %0d %h %h %h %h",phase,c,rdata[c*512+:512],rid[c*4+:4],rresp[c*2+:2],rlast[c]);
                    rvalid[c]<=0;mem_offset[c]+=64;mem_beats[c]--;
                    if(mem_beats[c]==0)mem_busy[c]<=0;
                end
                if(r_cpl_valid[c] && r_cpl_ready[c])begin
                    cpl_count[c]++;$display("SJ CPL %0d %0d %h %h %h",phase,c,r_cpl_tag[c*64+:64],r_cpl_status[c*2+:2],r_cpl_bytes[c*13+:13]);
                end
                if(ack_valid[c])$display("SJ ACK %0d %0d %h",phase,c,ack_event[c*580+:580]);
                if(held_result[c])require(result_valid[c] && result_context[c*1024+:1024]===prev_context[c*1024+:1024] &&
                    result_status[c*4+:4]===prev_status[c*4+:4],"stalled result stable");
                if(result_valid[c] && result_ready[c])begin
                    require(dut.b_result_ready[c],"retirement accepted jointly");
                    require(result_context[c*1024+:1024]===context_for(c+2*retired[c]),"result full identity");
                    if(success_case())require(result_status[c*4+:4]==0 && result_bytes[c*64+:64]==length_for(c+2*retired[c]),"success result");
                    else if(phase!=9)require(result_status[c*4+:4]==1 && result_bytes[c*64+:64]==0,"error result");
                    else require(result_status[c*4+:4]==(retired[c]==0?0:1),"held success preserved but quarantined");
                    $display("SJ RESULT %0d %0d %h %h %h",phase,c,result_context[c*1024+:1024],result_status[c*4+:4],result_bytes[c*64+:64]);retired[c]++;
                end
                held_command[c]=cmd_valid[c]&&!cmd_ready[c];held_ar[c]=arvalid[c]&&!arready[c];
                held_read[c]=r_data_valid[c]&&!r_data_ready[c];held_result[c]=result_valid[c]&&!result_ready[c];
            end
            prev_addr=cmd_addr;prev_tag=cmd_tag;prev_len=cmd_len;prev_araddr=araddr;prev_arlen=arlen;
            prev_read=r_data;prev_read_last=r_data_last;prev_read_tag=r_data_tag;
            prev_context=result_context;prev_status=result_status;
        end
    end
    task configure;
        @(negedge clk);cfg_valid=1;do @(posedge clk);while(!cfg_ready);
        @(negedge clk);cfg_valid=0;require(!cfg_error,"valid config");repeat(2)@(negedge clk);
    endtask
    task submit(input request_t request,input integer channel,input integer rejection);
        @(negedge clk);submit_context=request;submit_channel=1'(channel);submit_valid=1;
        do @(posedge clk);while(!submit_ready);
        @(negedge clk);submit_valid=0;
        require(reject_valid==(rejection!=0),"submission verdict");
        if(rejection)require(reject_code==rejection,"rejection reason");
        repeat(2)@(negedge clk);
    endtask
    task reserve_four(input integer base);
        request_t bad;
        if(phase==10)begin
            bad=context_for(0);bad.source.generation=2;submit(bad,0,1);
            bad=context_for(0);bad.bytes=0;submit(bad,0,3);
            bad=context_for(0);bad.bytes=65;submit(bad,0,5);
        end
        for(integer i=base;i<base+4;i++)begin
            submit(context_for(i),i%2,0);
            if(phase==10 && i==1)begin
                submit(context_for(0),0,6);
                bad=context_for(2);bad.target_offset=destination(1)-cfg.target_base;submit(bad,0,7);
            end
        end
        require(contexts_held==10 && source_held==10 && !admission_pending,"four pinned contexts");
    endtask
    task acknowledgements(input integer base);
        for(integer q=0;q<2;q++)begin
            @(negedge clk);ack_event[0+:580]=event_for(base+2*q);ack_event[580+:580]=event_for(base+2*q+1);ack_valid=3;
            @(negedge clk);ack_valid=0;
        end
    endtask
    task collect(input integer n);
        @(negedge clk);result_ready=3;wait(retired[0]==n && retired[1]==n);
        repeat(12)@(negedge clk);result_ready=0;
    endtask
    task success(input integer base);
        wait(packet[0]==base/2+2 && packet[1]==base/2+2);repeat(12)@(negedge clk);
        require(frame_idle==3 && source_held==10 && contexts_held==10 && result_valid==0,"frame free is not transfer retirement");
        if(phase==2)begin acknowledgements(base);repeat(12)@(negedge clk);require(result_valid==0,"ACK alone insufficient");end
        @(negedge dma_clk);allow_done=1;wait(credits[0]==base/2+2 && credits[1]==base/2+2);
        repeat(12)@(negedge clk);
        if(phase!=2)begin require(result_valid==0,"DMA alone insufficient");acknowledgements(base);end
        wait(result_valid==3);repeat(8)@(negedge clk);
        require(contexts_held==10 && source_held==10,"backpressure pins both mirrors");
        collect(base/2+2);require(contexts_held==0 && source_held==0 && !source_fault && halted==0,"coordinated retirement");
    endtask
    initial begin
        if(!$value$plusargs("seed=%d",seed))seed=0;
        $dumpfile("source.vcd");$dumpvars(1,tb_ntrpu_hbm_xdma_source);
        for(integer scenario=1;scenario<=15;scenario++)begin
            @(negedge clk);cold_reset_n=0;phase=scenario;stop=0;allow_done=0;ar_open=0;cmd_gate=scenario!=7;axis_open=scenario!=8;
            cfg_valid=0;submit_valid=0;ack_valid=0;result_ready=0;guard_start=0;guard_epoch=64'(scenario);
            round_number=0;repeat(6)@(negedge clk);cold_reset_n=1;wait(started);
            @(negedge dma_clk);require(guard_start_ready,"guard ready");guard_start=1;
            @(negedge dma_clk);guard_start=0;wait(admit_descriptor==3);
            cfg='0;cfg.source.session=1;cfg.source.slot=10;cfg.source.generation=1;
            cfg.target.session=1;cfg.target.node=1;cfg.target.slot=20;cfg.target.generation=1;
            cfg.source_base=64'h4000;cfg.source_size=64'h200100000;
            cfg.target_base=64'h100000000;cfg.target_size=64'h200000000;
            cfg.window_bytes=cfg.target_size;cfg.window_generation=1;cfg.model_epoch=7;
            cfg.published_bytes=cfg.source_size;cfg.source_read=1;cfg.target_write=1;cfg.source_open=1;cfg.target_open=1;
            configure;
            if(scenario==11)begin
                @(negedge clk);cfg.source_base='h10000;cfg_valid=1;do @(posedge clk);while(!cfg_ready);
                @(negedge clk);cfg_valid=0;require(cfg_error,"bad generation config rejected");
                cfg.source_base='h4000;repeat(2)@(negedge clk);
            end
            reserve_four(0);
            @(negedge clk);ar_open=1;
            if(success_case())begin
                if(scenario==11)begin
                    cfg_valid=1;repeat(12)begin @(negedge clk);require(!cfg_ready,"active config blocked");end cfg_valid=0;
                end
                success(0);
                if(scenario==3)begin
                    @(negedge clk);ar_open=0;allow_done=0;cfg.source_base='h8000;
                    cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;configure;
                    reserve_four(4);@(negedge clk);ar_open=1;success(4);
                end
            end else begin
                if(scenario==7)begin
                    repeat(8)@(negedge clk);require(cmd_valid==3 && command_count[0]==0 && command_count[1]==0,"held unissued commands");stop=1;
                end
                if(scenario==8)begin wait(|(r_data_valid & ~r_data_ready));@(negedge clk);stop=1;end
                if(scenario==9)begin
                    wait(packet[0]==2 && packet[1]==2);repeat(12)@(negedge clk);
                    @(negedge dma_clk);allow_done=1;wait(credits[0]==2 && credits[1]==2);
                    repeat(12)@(negedge clk);acknowledgements(0);wait(result_valid==3);
                    @(negedge clk);stop=1;
                end
                wait(halted==3 && source_fault);collect(2);
                require(source_held==10 && contexts_held==10,"error reports retain all reservations");
                @(negedge clk);stop=0;cmd_gate=1;axis_open=1;ar_open=1;allow_done=1;
                repeat(100)@(negedge clk);acknowledgements(0);repeat(40)@(negedge clk);
                require(source_held==10 && contexts_held==10 && source_fault && !submit_ready && !cfg_ready && result_valid==0,"late events cannot rearm");
            end
            if(scenario==10)require(rejections==5,"all invalid requests rejected");
            passes++;$display("SJ CASE %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",phase,source_held,contexts_held,retired[0],retired[1],command_count[0],command_count[1],packet[0],packet[1],rejections);
        end
        $display("SJ PASS %0d %0d",passes,seed);$finish;
    end
    initial begin #1000000;$fatal(1,"SJ watchdog phase=%0d",phase);end
endmodule
