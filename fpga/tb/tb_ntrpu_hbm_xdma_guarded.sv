`timescale 1ns/1ps
module tb_ntrpu_hbm_xdma_guarded;
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
    reg [1:0] data_valid=0,data_last=0,local_valid=0,ack_valid=0,result_ready=0;
    reg [1023:0] data=0;
    reg [2047:0] data_context=0;
    reg [1159:0] local_event=0,ack_event=0;
    wire [1:0] data_ready,result_valid,halted,dma_fault;
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
    ntrpu_hbm_xdma_guarded #(.GUARD_WAIT_LIMIT(4000),.MASTER_WAIT_LIMIT(48)) dut(.*);
    integer phase=0,seed=0,ticks=0,passes=0;
    integer desc[0:1],packet[0:1],offsets[0:1],done_sent[0:1],retired[0:1],credits[0:1];
    reg allow_done=0,stall_ar=0,stall_r=0,pending=0;
    reg [31:0] address_q=0,status_value=1;
    reg [1:0] response_code=0;
    integer delay_q=0;
    reg [1:0] held_axis=0,held_result=0;
    reg [1023:0] prev_data;
    reg [1:0] prev_last;
    reg [2047:0] prev_context;
    reg [7:0] prev_status;
    task require(input bit ok,input string why);
        if(!ok)$fatal(1,"HG ASSERT phase=%0d %s",phase,why);
    endtask
    function automatic integer length_for(input integer i);
        case(i) 0:length_for=64;1:length_for=128;2:length_for=256;3:length_for=4096;endcase
    endfunction
    function automatic [63:0] destination(input integer i);
        destination=64'h100000800+(64'(i%2)<<32)+64'((i/2)*4096);
    endfunction
    function automatic [63:0] wb_address(input integer i);
        wb_address=64'h300001000+64'((i%2)*128+(i/2)*64);
    endfunction
    function automatic [7:0] pattern(input integer i,offset);
        pattern=8'((offset*37)^(offset>>7)^((i+1)*53));
    endfunction
    function automatic request_t context_for(input integer i);
        request_t r;
        r='0;r.identity.session=1;r.identity.source=0;r.identity.sequence_id=i+1;
        r.source=cfg.source;r.target=cfg.target;r.window_generation=1;r.model_epoch=7;
        r.source_offset=64'(i*4096);r.target_offset=destination(i)-cfg.target_base;r.bytes=length_for(i);
        return r;
    endfunction
    function automatic event_t event_for(input integer i);
        request_t r;event_t e;r=context_for(i);e='0;e.identity=r.identity;e.target=r.target;
        e.window_generation=r.window_generation;e.bytes=r.bytes;return e;
    endfunction
    task automatic payload(input integer channel);
        integer i,k,b;
        for(integer slot=0;slot<2;slot=slot+1)begin
            i=channel+2*slot;
            for(k=0;k<length_for(i);k=k+64)begin
                @(negedge clk);
                // Deliberate source gaps exercise packet assembly without changing READY.
                repeat((k/64+channel)%3)@(negedge clk);
                for(b=0;b<64;b=b+1)data[channel*512+b*8+:8]=pattern(i,k+b);
                data_context[channel*1024+:1024]=context_for(i);
                data_last[channel]=(k+64==length_for(i));data_valid[channel]=1;
                do @(posedge clk);while(!data_ready[channel]);
                @(negedge clk);data_valid[channel]=0;
            end
            local_event[channel*580+:580]=event_for(i);local_valid[channel]=1;
            @(negedge clk);local_valid[channel]=0;
        end
    endtask

    always @(negedge dma_clk) begin
        if(!dma_rst_n) begin m_axil_arready=0;m_axil_rvalid=0;pending=0;dma_done=0;ticks=0;end
        else begin
            ticks=ticks+1;
            dsc_ready[0]=(ticks+seed)%4!=0;dsc_ready[1]=(ticks+seed)%5!=0;
            axis_ready[0]=(ticks+seed)%3!=0;axis_ready[1]=(ticks+seed)%4!=0;
            dma_done=0;
            for(integer c=0;c<2;c=c+1) if(allow_done && done_sent[c]<packet[c])dma_done[c]=1;
            m_axil_arready=!stall_ar && !pending && !m_axil_rvalid && (ticks+seed)%3!=0;
            if(pending && !m_axil_rvalid)begin
                if(delay_q>0)delay_q=delay_q-1;
                else if(!stall_r)begin
                    case(address_q[7:0])
                        'h04:m_axil_rdata='h00f83e7f;
                        'h40:m_axil_rdata=status_value;
                        'h48:m_axil_rdata=done_sent[address_q[8]];
                        default:$fatal(1,"HG unexpected address");
                    endcase
                    m_axil_rresp=response_code;m_axil_rvalid=1;
                end
            end
        end
    end
    always @(posedge dma_clk) begin
        if(!dma_rst_n)begin
            held_axis=0;
            for(integer c=0;c<2;c=c+1)begin desc[c]=0;packet[c]=0;offsets[c]=0;done_sent[c]=0;credits[c]=0;end
        end else begin
            require(!m_axil_awvalid && !m_axil_wvalid,"no management writes");
            if(m_axil_arvalid && m_axil_arready)begin
                require(!pending,"one AR");pending<=1;address_q<=m_axil_araddr;delay_q<=1+(ticks+seed)%4;
            end
            if(m_axil_rvalid && m_axil_rready)begin pending<=0;m_axil_rvalid<=0;end
            for(integer c=0;c<2;c=c+1)begin
                if(dsc_load[c])begin
                    require(admit_descriptor[c] && dsc_ready[c] && desc[c]<2,"guarded descriptor admission");
                    require(dsc_dst[c*64+:64]==destination(c+2*desc[c]) && dsc_src[c*64+:64]==wb_address(c+2*desc[c]) &&
                            dsc_len[c*28+:28]==length_for(c+2*desc[c]) && dsc_ctl[c*16+:16]==16'h10,"descriptor contents");desc[c]++;
                end
                if(held_axis[c])require(axis_valid[c] && axis_data[c*512+:512]===prev_data[c*512+:512] &&
                                        axis_last[c]===prev_last[c],"stalled stream retained");
                if(axis_valid[c] && axis_ready[c])begin
                    require(packet[c]<desc[c] && axis_keep[c*64+:64]==64'hffffffffffffffff,"packet belongs to descriptor");
                    for(integer b=0;b<64;b=b+1)
                        require(axis_data[c*512+b*8+:8]===pattern(c+2*packet[c],offsets[c]+b),"payload contents");
                    offsets[c]+=64;require(axis_last[c]==(offsets[c]==length_for(c+2*packet[c])),"last boundary");
                    if(axis_last[c])begin packet[c]++;offsets[c]=0;end
                end
                if(dma_done[c])done_sent[c]++;
                if(validated_done[c])begin require(credits[c]<done_sent[c],"backed completion");credits[c]++;end
                held_axis[c]=axis_valid[c]&&!axis_ready[c];
            end
            prev_data=axis_data;prev_last=axis_last;
        end
    end
    always @(posedge clk)begin
        if(!rst_n)begin retired[0]=0;retired[1]=0;held_result=0;end
        else for(integer c=0;c<2;c=c+1)begin
            if(held_result[c])require(result_valid[c] && result_context[c*1024+:1024]===prev_context[c*1024+:1024] &&
                                     result_status[c*4+:4]===prev_status[c*4+:4],"result stays stable");
            if(result_valid[c] && result_ready[c])begin
                require(retired[c]<2 && result_context[c*1024+:1024]===context_for(c+2*retired[c]),"result identity");
                require(result_status[c*4+:4]==(phase<=2?0:1) && result_bytes[c*64+:64]==(phase<=2?length_for(c+2*retired[c]):0),"result status");
                retired[c]++;
            end
            held_result[c]=result_valid[c]&&!result_ready[c];
            prev_context[c*1024+:1024]=result_context[c*1024+:1024];prev_status[c*4+:4]=result_status[c*4+:4];
        end
    end
    task setup(input integer number);
        begin
            @(negedge clk);cold_reset_n=0;phase=number;stop=0;allow_done=0;stall_ar=0;stall_r=0;
            cfg_valid=0;submit_valid=0;data_valid=0;local_valid=0;ack_valid=0;result_ready=0;
            dma_link_up=1;dma_run=3;dma_error=0;management_core_ready=1;exclusive_owner=1;management_fault=0;
            management_wait_expired=0;management_reset_request=0;guard_start=0;guard_epoch=number;
            status_value=1;response_code=0;
            repeat(6)@(negedge clk);cold_reset_n=1;wait(started);
            @(negedge dma_clk);require(guard_start_ready,"guard start ready");guard_start=1;
            @(negedge dma_clk);guard_start=0;wait(admit_descriptor==3);
            @(negedge clk);
            cfg='0;cfg.source.session=1;cfg.source.node=0;cfg.source.slot=10;cfg.source.generation=1;
            cfg.target.session=1;cfg.target.node=1;cfg.target.slot=20;cfg.target.generation=1;
            cfg.source_size='h100000;cfg.target_base=64'h100000000;cfg.target_size=64'h200000000;
            cfg.window_bytes=cfg.target_size;cfg.window_generation=1;cfg.model_epoch=7;
            cfg.published_bytes=cfg.source_size;cfg.source_read=1;cfg.target_write=1;cfg.source_open=1;cfg.target_open=1;
            cfg_valid=1;do @(posedge clk);while(!cfg_ready);@(negedge clk);cfg_valid=0;require(!cfg_error,"lease");
            for(integer i=0;i<4;i=i+1)begin
                submit_context=context_for(i);submit_channel=i%2;submit_valid=1;
                do @(posedge clk);while(!submit_ready);@(negedge clk);submit_valid=0;require(!reject_valid,"admitted");
                @(negedge clk);
            end
            fork payload(0);payload(1);join
            wait(packet[0]==2 && packet[1]==2);repeat(12)@(negedge clk);
            require(contexts_held==4'b1010 && result_valid==0 && delivered_count==0,"four contexts await completion");
        end
    endtask
    task acknowledgements;
        for(integer q=0;q<2;q=q+1)begin
            @(negedge clk);ack_event[0+:580]=event_for(2*q);ack_event[580+:580]=event_for(2*q+1);ack_valid=3;
            @(negedge clk);ack_valid=0;
        end
    endtask
    task collect;
        begin
            @(negedge clk);result_ready=3;
            wait(retired[0]==2 && retired[1]==2);repeat(12)@(negedge clk);
        end
    endtask
    initial begin
        if(!$value$plusargs("seed=%d",seed))seed=0;
        $dumpfile("guarded.vcd");$dumpvars(0,tb_ntrpu_hbm_xdma_guarded);
        for(integer c=1;c<=11;c=c+1)begin
            setup(c);
            if(c<=2)begin
                if(c==2)begin acknowledgements;repeat(12)@(negedge clk);require(contexts_held==10 && result_valid==0,"ACK cannot replace DMA validation");end
                @(negedge dma_clk);allow_done=1;wait(delivered_count==64'h200000002);
                repeat(12)@(negedge clk);
                if(c==1)begin require(contexts_held==10 && result_valid==0,"DMA validation cannot replace ACK");acknowledgements;end
                collect;require(contexts_held==0 && halted==0 && guard_faults==0,"successful retirement");
            end else begin
                @(negedge dma_clk);
                case(c)
                    3:stall_ar=1;
                    4:stall_r=1;
                    5:status_value='h80001;
                    6:response_code=2;
                    7:exclusive_owner=0;
                    8:dma_link_up=0;
                    9:management_reset_request=1;
                    10:management_core_ready=0;
                    11:management_wait_expired=1;
                endcase
                wait(guard_faults!=0 && halted==3);collect;
                require(contexts_held==10 && delivered_count==0 && observed_count==0 && published_count==64'h200000002,
                        "error reports retain all four owned contexts");
                stall_ar=0;stall_r=0;exclusive_owner=1;dma_link_up=1;management_core_ready=1;
                management_reset_request=0;management_wait_expired=0;response_code=0;status_value=1;
                @(negedge dma_clk);allow_done=1;wait(observed_count==64'h200000002);
                acknowledgements;repeat(80)@(negedge clk);
                require(contexts_held==10 && delivered_count==0 && guard_faults!=0 && halted==3 && !cfg_ready && !submit_ready,
                        "late Done/ACK and restored link cannot release quarantine");
            end
            passes++;$display("HG CASE phase=%0d held=%0d faults=%h retired0=%0d retired1=%0d",phase,contexts_held,guard_faults,retired[0],retired[1]);
        end
        $display("HG PASS cases=%0d seed=%0d",passes,seed);$finish;
    end
    initial begin #1000000;$fatal(1,"HG watchdog phase=%0d",phase);end
endmodule
