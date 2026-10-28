`timescale 1ps/1ps
// Native PIPE gate: rev3 emits real descriptors/data. Reader data and remote ACK
// are fixture inputs; successful DMA is released only after native status reads.
module ntrpu_hbm_xdma_native;
    import ntrpu_peer_read_pkg::*;
    reg clk=0,cold_reset_n=0,stop=0;
    always #2500 clk=~clk;
    wire dma_clk=board.EP.user_clk;
    wire dma_link_up=board.EP.user_lnk_up;
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
    wire [15:0] native_status;
    wire [1:0] dma_run={native_status[14],native_status[6]};
    reg [1:0] dma_error=0,dma_done=0;
    wire [1:0] dsc_ready,axis_ready,dsc_load,axis_valid,axis_last;
    wire [127:0] dsc_src,dsc_dst,axis_keep;
    wire [55:0] dsc_len;
    wire [31:0] dsc_ctl;
    wire [1023:0] axis_data;
    ntrpu_hbm_dma_bridge dut(.*);
    integer phase=0,descriptors[0:1]='{0,0},raw_done[0:1]='{0,0},retired[0:1]='{0,0};
    integer axis_bytes[0:1]='{0,0},packets[0:1]='{0,0};
    reg [1:0] previous_stall=0;
    reg [1023:0] previous_data;
    reg [1:0] previous_last;
    reg [31:0] value;
    reg [63:0] host_wb[0:3];
    reg [7:0] host_memory[0:3][0:4095];
    bit written[0:3][0:4095],wb_written[0:3][0:7];
    integer host_bytes[0:3]='{0,0,0,0},wb_bytes[0:3]='{0,0,0,0};
    reg host_active=0;
    reg [63:0] host_address;
    integer host_length,host_offset,host_packets=0;
    time previous_edge=0;
    integer measured_edges=0;

    task require(input bit condition,input string reason);
        if(!condition)$fatal(1,"HX ASSERT phase=%0d reason=%s",phase,reason);
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
    task read_reg(input [31:0] address);
        board.RP.tx_usrapp.TSK_XDMA_REG_READ(address);value=board.RP.tx_usrapp.P_READ_DATA;
        require(!$isunknown(value),"known register value");
        $display("HX REG t=%0t address=%08h value=%08h",$time,address,value);
    endtask
    task write_reg(input [31:0] address,input [31:0] v);
        $display("HX WRITE t=%0t address=%08h value=%08h",$time,address,v);
        board.RP.tx_usrapp.TSK_XDMA_REG_WRITE(address,v,4'hf);
    endtask
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
    task run_probe;
        integer i,c,q,b;
        require(board.EP.user_resetn===1'b1 && dma_link_up===1'b1,"enumerated endpoint");
        require(board.RP.tx_usrapp.xdma_bar==0 || board.RP.tx_usrapp.xdma_bar==1,"DMA register discovery");
        require(board.RP.tx_usrapp.BAR_INIT_P_BAR_ENABLED[1]==2'b10,"native DMA BAR1 is memory32");
        $display("HX BAR discovered=%0d selected=1 base=%08h",board.RP.tx_usrapp.xdma_bar,board.RP.tx_usrapp.BAR_INIT_P_BAR[1][31:0]);
        // AMD's example loops BAR0 AXI-Lite into local management. Select the
        // real DMA BAR explicitly in the public host driver and verify its ID.
        board.RP.tx_usrapp.xdma_bar=1;
        phase=1;
        for(i=0;i<4;i=i+1)begin
            host_wb[i]=0;
            for(b=0;b<4096;b=b+1)begin host_memory[i][b]=0;written[i][b]=0;end
            for(b=0;b<8;b=b+1)wb_written[i][b]=0;
        end
        repeat(32)@(negedge dma_clk);
        require(measured_edges>=16,"native clock observed");
        read_reg(0);require(value[31:16]==16'h1fc0,"native BAR1 XDMA identifier");
        for(c=0;c<2;c=c+1)begin
            read_reg('h1040+c*'h100);require(value==0,"channel initially idle");
            write_reg('h1004+c*'h100,'h00f83e7f);
            read_reg('h1004+c*'h100);require(value=='h00f83e7f,"error logging enabled; metadata WB enabled; Run=1");
        end
        @(negedge clk);cold_reset_n=1;
        wait(started);@(negedge clk);
        cfg='0;cfg.source.session=1;cfg.source.node=0;cfg.source.slot=10;cfg.source.generation=1;
        cfg.target.session=1;cfg.target.node=1;cfg.target.slot=20;cfg.target.generation=1;
        cfg.source_size=64'h100000;cfg.target_base=64'h100000000;cfg.target_size=64'h200000000;
        cfg.window_bytes=cfg.target_size;cfg.window_generation=1;cfg.model_epoch=7;
        cfg.published_bytes=cfg.source_size;cfg.source_read=1;cfg.target_write=1;cfg.source_open=1;cfg.target_open=1;
        cfg_valid=1;do @(posedge clk);while(!cfg_ready);@(negedge clk);cfg_valid=0;
        require(!cfg_error,"valid lease");
        phase=2;
        for(i=0;i<4;i=i+1)begin
            @(negedge clk);submit_context=context_for(i);submit_channel=i%2;submit_valid=1;
            do @(posedge clk);while(!submit_ready);
            @(negedge clk);submit_valid=0;require(!reject_valid,"request accepted");
        end
        fork payload(0);payload(1);join
        wait(raw_done[0]==2 && raw_done[1]==2);
        wait(host_bytes[0]==64 && host_bytes[1]==128 && host_bytes[2]==256 && host_bytes[3]==4096);
        wait(wb_bytes[0]==8 && wb_bytes[1]==8 && wb_bytes[2]==8 && wb_bytes[3]==8);
        phase=3;
        for(i=0;i<4;i=i+1)begin
            require(host_wb[i]=={32'(length_for(i)),32'h52b40001},"native writeback magic/EOP/length");
            for(b=0;b<length_for(i);b=b+1)require(written[i][b] && host_memory[i][b]==pattern(i,b),"received bytes");
        end
        for(c=0;c<2;c=c+1)begin
            read_reg('h1040+c*'h100);require((value & ~32'h47)==0,"no logged native errors");
            read_reg('h1048+c*'h100);require(value==2,"native descriptor count");
            read_reg('h1004+c*'h100);require(value=='h00f83e7f,"logging configuration retained");
        end
        require(contexts_held==4'b1010 && result_valid==0,"raw Done does not retire unvalidated transfers");
        // Test-only management policy: two stored native Done credits per channel.
        // A hardware register reader/credit provider is a separate integration gate.
        for(q=0;q<2;q=q+1)begin
            @(negedge dma_clk);dma_done=3;@(negedge dma_clk);dma_done=0;
        end
        repeat(32)@(negedge clk);require(result_valid==0,"remote ACK still required");
        for(q=0;q<2;q=q+1)begin
            @(negedge clk);ack_event[0+:580]=event_for(q*2);ack_event[580+:580]=event_for(q*2+1);ack_valid=3;
            @(negedge clk);ack_valid=0;
        end
        repeat(16)@(negedge clk);result_ready=3;
        wait(retired[0]==2 && retired[1]==2);repeat(16)@(negedge clk);
        require(contexts_held==0 && halted==0 && dma_fault==0,"all four contexts retired");
        phase=4;
        $display("HX PASS bytes=4544 writeback_bytes=32 channels=2 descriptors=4 native_hbm=0 hardware_error_reader=0");
    endtask
    always @(posedge dma_clk)begin
        if(phase!=0)begin
            if(previous_edge!=0)begin
                require($time-previous_edge==4000,"native AXI clock period 4000 ps");
                if(measured_edges<32)$display("HX CLOCK t=%0t period=%0t",$time,$time-previous_edge);
                measured_edges=measured_edges+1;
            end
            previous_edge=$time;
            require(board.EP.user_resetn===1'b1 && dma_link_up===1'b1,"native link/reset retained");
            for(integer c=0;c<2;c=c+1)begin
                if(native_status[c*8+3])begin raw_done[c]=raw_done[c]+1;$display("HX DONE t=%0t channel=%0d count=%0d status=%02h",$time,c,raw_done[c],native_status[c*8+:8]);end
                if(dsc_load[c])begin
                    require(dsc_ready[c] && dma_run[c] && descriptors[c]<2,"legal descriptor handshake");
                    require(dsc_dst[c*64+:64]==destination(c+descriptors[c]*2) && dsc_src[c*64+:64]==wb_address(c+descriptors[c]*2),"descriptor addresses");
                    require(dsc_len[c*28+:28]==length_for(c+descriptors[c]*2) && dsc_ctl[c*16+:16]==16'h10,"descriptor length/control");
                    $display("HX DESC t=%0t channel=%0d src=%016h dst=%016h len=%0d ctl=%04h",$time,c,dsc_src[c*64+:64],dsc_dst[c*64+:64],dsc_len[c*28+:28],dsc_ctl[c*16+:16]);
                    descriptors[c]=descriptors[c]+1;
                end
                if(previous_stall[c])require(axis_valid[c] && axis_data[c*512+:512]===previous_data[c*512+:512] && axis_last[c]===previous_last[c],"AXIS stable under native backpressure");
                previous_stall[c]=axis_valid[c] && !axis_ready[c];
                if(axis_valid[c] && axis_ready[c])begin
                    require(axis_keep[c*64+:64]==64'hffffffffffffffff,"full stream beat");
                    axis_bytes[c]=axis_bytes[c]+64;if(axis_last[c])packets[c]=packets[c]+1;
                    $display("HX AXIS t=%0t channel=%0d data=%0128h last=%b",$time,c,axis_data[c*512+:512],axis_last[c]);
                end
            end
            previous_data=axis_data;previous_last=axis_last;
        end
    end
    always @(posedge clk)if(rst_n)begin
        require(halted==0 && dma_fault==0 && !reject_valid,"bridge healthy");
        for(integer c=0;c<2;c=c+1)if(result_valid[c] && result_ready[c])begin
            require(result_context[c*1024+:1024]==context_for(c+retired[c]*2) && result_status[c*4+:4]==0 && result_bytes[c*64+:64]==length_for(c+retired[c]*2),"attributed result");
            $display("HX RESULT t=%0t channel=%0d index=%0d bytes=%0d",$time,c,retired[c],result_bytes[c*64+:64]);retired[c]=retired[c]+1;
        end
    end
    // Passive destination memory: only accepted RP CQ bytes change state.
    always @(posedge board.RP.user_clk)begin: receive
        integer first,b,i,offset,address_hits;reg [63:0] address;reg [7:0] byte_value;
        if(phase!=0 && board.RP.m_axis_cq_tvalid && board.RP.m_axis_cq_tready)begin
            $display("HX CQ t=%0t data=%0128h keep=%04h last=%b user=%046h",$time,board.RP.m_axis_cq_tdata,board.RP.m_axis_cq_tkeep,board.RP.m_axis_cq_tlast,board.RP.m_axis_cq_tuser);
            first=0;
            if(!host_active)begin
                require(board.RP.m_axis_cq_tuser[81:80]===2'b01 && board.RP.m_axis_cq_tdata[78:75]==1,"single nonstraddled memory write");
                require(board.RP.m_axis_cq_tuser[15:0]==16'h0f0f,"full byte enables");
                host_address=board.RP.m_axis_cq_tdata[63:0];host_length=board.RP.m_axis_cq_tdata[74:64]*4;
                host_offset=0;host_active=1;first=16;
            end else require(board.RP.m_axis_cq_tuser[81:80]===0,"no nested CQ packet");
            require(!board.RP.m_axis_cq_tuser[96],"no discontinue");
            for(b=0;b<64;b=b+1)if(board.RP.m_axis_cq_tuser[16+b]===1'b1)begin
                require(b>=first && board.RP.m_axis_cq_tkeep[b/4] && host_offset<host_length,"CQ enabled payload");
                address=host_address+host_offset;byte_value=board.RP.m_axis_cq_tdata[b*8+:8];address_hits=0;
                for(i=0;i<4;i=i+1)begin
                    if(address>=destination(i) && address<destination(i)+length_for(i))begin
                        offset=address-destination(i);require(!written[i][offset],"no duplicate payload byte");
                        host_memory[i][offset]=byte_value;written[i][offset]=1;host_bytes[i]++;address_hits++;
                    end
                    if(address>=wb_address(i) && address<wb_address(i)+8)begin
                        offset=address-wb_address(i);require(!wb_written[i][offset],"no duplicate writeback byte");
                        host_wb[i][offset*8+:8]=byte_value;wb_written[i][offset]=1;wb_bytes[i]++;address_hits++;
                    end
                end
                require(address_hits==1,"authorized full-width destination");host_offset++;
            end
            if(board.RP.m_axis_cq_tlast)begin require(host_offset==host_length,"CQ packet length");host_active=0;host_packets++;end
        end
    end
    initial begin #250000000;$fatal(1,"HX TIMEOUT phase=%0d done=%0d,%0d",phase,raw_done[0],raw_done[1]);end
endmodule
