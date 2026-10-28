    integer wdone[0:1],wbeats[0:1],wtag[0:1];
    reg [63:0] wbase[0:1];
    integer direct_phase=0,mem_cycle=0,ar_count[0:1],read_count[0:1];
    reg [33:0] read_address[0:1][0:1023];
    integer read_length[0:1][0:1023],read_beat[0:1];
    reg [1:0] held_native_r=0;
    reg [264:0] saved_native_r[0:1];
    always @(posedge mem_clk)if(mem_rst_n)begin
        mem_cycle++;
        for(integer l=0;l<2;l++)begin
            if(held_native_r[l])check(raw_rvalid[l] && saved_native_r[l]=={raw_rdata[l*256+:256],raw_rid[l*6+:6],raw_rresp[l*2+:2],raw_rlast[l]},"HBM_RESPONSE_STABILITY");
            held_native_r[l]=raw_rvalid[l] && !(bus_rready[l]&&r_open[l]);
            saved_native_r[l]={raw_rdata[l*256+:256],raw_rid[l*6+:6],raw_rresp[l*2+:2],raw_rlast[l]};
            if(bus_arvalid[l] && ar_open[l] && raw_arready[l])begin
                check(bus_arid[l*6+:6]==0 && bus_arsize[l*3+:3]==5 && bus_arburst[l*2+:2]==1,"NATIVE_ATTRIBUTES");
                read_address[l][ar_count[l]]=bus_araddr[l*34+:34];
                read_length[l][ar_count[l]]=int'(bus_arlen[l*4+:4])+1;ar_count[l]++;
                $display("SN AR %0d %0d %0d %0d %0d %0d %0d",$rtoi($realtime*1000),direct_phase,direct_mode,l,mem_cycle,bus_araddr[l*34+:34],int'(bus_arlen[l*4+:4])+1);
                if(!direct_mode)$display("SF AR %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,l,bus_araddr[l*34+:34],int'(bus_arlen[l*4+:4])+1);
            end
            if(raw_rvalid[l] && bus_rready[l] && r_open[l])begin
                check(read_count[l]<ar_count[l] && raw_rid[l*6+:6]==0 && raw_rresp[l*2+:2]==0,"HBM_RESPONSE");
                for(integer b=0;b<32;b++)check(raw_rdata[l*256+b*8+:8]==value_at(read_address[l][read_count[l]]+read_beat[l]*32+b),"HBM_BYTES");
                check(raw_rlast[l]==(read_beat[l]+1==read_length[l][read_count[l]]),"HBM_LAST");
                $display("SN NR %0d %0d %0d %0d %0d %0d %0d %0d %h",$rtoi($realtime*1000),direct_phase,direct_mode,l,mem_cycle,raw_rid[l*6+:6],raw_rresp[l*2+:2],raw_rlast[l],raw_rdata[l*256+:256]);
                if(raw_rlast[l])begin read_count[l]++;read_beat[l]=0;end else read_beat[l]++;
            end
            if(direct_mode && direct_phase>0)
                $display("SN CYCLE %0d %0d %0d %0d %b %b %b %b",$rtoi($realtime*1000),direct_phase,l,mem_cycle,bus_arvalid[l],raw_arready[l],raw_rvalid[l],bus_rready[l]);
            if(h_bvalid[l]&&h_bready[l])begin
                check(h_bresp[l*2+:2]==0,"HBM_BRESP");
                $display("SN B %0d %0d %0d %0d",$rtoi($realtime*1000),l,h_bid[l*6+:6],h_bresp[l*2+:2]);
            end
        end
    end
    always @(posedge clk)if(wrst_n)begin
        check(!wfault,"NATIVE_WRITER_FAULT");
        for(integer l=0;l<2;l++)begin
            if(wcmd_valid[l]&&wcmd_ready[l])begin
                wtag[l]=wcmd_tag[l*64+:64];wbase[l]=wcmd_addr[l*64+:64];wbeats[l]=0;
                $display("SN WCMD %0d %0d %0d %0d %0d",$rtoi($realtime*1000),l,wtag[l],wbase[l],int'(wcmd_len[l*8+:8])+1);
            end
            if(wdata_valid[l]&&wdata_ready[l])begin
                $display("SN WI %0d %0d %0d %0d %h",$rtoi($realtime*1000),l,wtag[l],wbeats[l],wdata[l*512+:512]);wbeats[l]++;
            end
            if(wcpl_valid[l])begin
                check(wcpl_status[l*2+:2]==0 && wcpl_tag[l*64+:64]==wtag[l],"PRELOAD_CPL");
                wdone[l]++;$display("SN WCPL %0d %0d %0d",$rtoi($realtime*1000),l,wtag[l]);
            end
        end
    end
    task automatic preload(input integer l,seq,input logic[63:0] address);
        integer before_done;
        before_done=wdone[l];tick();wcmd_addr[l*64+:64]=address;wcmd_tag[l*64+:64]=seq;
        wcmd_len[l*8+:8]=63;wcmd_valid[l]=1;do @(posedge clk);while(!wcmd_ready[l]);tick();wcmd_valid[l]=0;
        for(integer beat=0;beat<64;beat++)begin
            for(integer b=0;b<64;b++)wdata[l*512+b*8+:8]=value_at(address+beat*64+b);
            wdata_valid[l]=1;do @(posedge clk);while(!wdata_ready[l]);tick();wdata_valid[l]=0;
        end
        while(wdone[l]==before_done)tick();tick();
    endtask
    task automatic direct_read(input integer l,input logic[33:0] base);
        integer before_reads;
        before_reads=read_count[l];
        for(integer burst=0;burst<8;burst++)begin
            @(negedge mem_clk);#0.001;direct_address[l*34+:34]=base+burst*512;direct_valid[l]=1;
            do @(posedge mem_clk);while(!raw_arready[l]);
            @(negedge mem_clk);#0.001;direct_valid[l]=0;
        end
        while(read_count[l]<before_reads+8)begin @(negedge mem_clk);#0.001;end
    endtask
    task automatic locality(input integer number,input logic[33:0] a0,a1);
        @(negedge mem_clk);#0.001;direct_phase=number;direct_ready=3;
        $display("SN LOCALITY_BEGIN %0d %0d %0d %0d",$rtoi($realtime*1000),number,a0,a1);
        fork direct_read(0,a0);direct_read(1,a1);join
        @(negedge mem_clk);#0.001;direct_ready=0;
        check(ar_count[0]==read_count[0] && ar_count[1]==read_count[1],"DIRECT_DRAIN");
        $display("SN LOCALITY_END %0d %0d",$rtoi($realtime*1000),number);direct_phase=0;
    endtask
    request_t a,b,c,d,v;
    initial begin
        for(integer l=0;l<2;l++)begin
            wdone[l]=0;wbeats[l]=0;wtag[l]=0;ar_count[l]=0;read_count[l]=0;read_beat[l]=0;
        end
        epoch=1;scenario=0;apb_rst_n=0;tick(NATIVE?940:10);apb_rst_n=1;cold_reset_n=1;
        while(!started || !wstarted)tick();tick(5);
        $display("SN BOOT %0d %b %b",$rtoi($realtime*1000),apb_complete_0,apb_complete_1);
        // Each physical byte is written once, through the local ingress port.
        for(integer i=0;i<4;i++)preload(0,i+1,64'(i)*4096);
        for(integer i=0;i<2;i++)preload(1,i+1,64'h200000000+64'(i)*4096);
        check(wdone[0]==4 && wdone[1]==2,"PRELOAD_COUNT");
        $display("SN PRELOAD_DONE %0d",$rtoi($realtime*1000));
        locality(1,0,34'h200000000); // AXI_00->PC0, AXI_16->PC16.
        locality(2,34'h200000000,0); // Crossed stacks.
        locality(3,0,0); // Both ingress ports contend for the same source.
        tick();direct_mode=0;cfg=initial_lease();cfg.source_size=16384;cfg.published_bytes=16384;configure();
        $display("SN PUBLISHED %0d 0",$rtoi($realtime*1000));

        scenario=1;a=make_request(1);b=make_request(2,4096);b.source_offset=0;
        c=make_request(3,8192);c.source_offset=4096;d=make_request(4,12288);d.source_offset=4096;
        expected[1]=a;expected[2]=b;expected[3]=c;expected[4]=d;
        submit(0,a);submit(1,b);submit(0,c);submit(1,d);while(dma_valid!=3)tick();tick(20);
        check(contexts_held==10,"SHARED_FOUR_CONTEXTS");cfg_valid=1;tick(8);check(!cfg_ready,"CFG_WHILE_HELD");cfg_valid=0;
        dma_ready=3;wait_data(a);wait_data(b);wait_data(c);wait_data(d);
        notify_ack(0,c);notify_dma(0,c);notify_ack(1,d);notify_dma(1,d);
        notify_dma(0,a);notify_dma(1,b);tick(6);check(result_valid==0 && contexts_held==10,"EARLY_RELEASE");
        notify_ack(0,a);notify_ack(1,b);while(result_valid!=3)tick();tick(10);
        check(contexts_held==10,"OFFER_RELEASED");result_ready=1;while(results[0]<2)tick();tick(4);
        check(contexts_held==8,"ASYMMETRIC_RETIRE");rejected_request(0,make_request(5,4096),OVERLAP);
        result_ready=3;while(results[1]<2)tick();tick(3);check(contexts_held==0,"CASE1_HELD");passed();

        scenario=2;
        for(integer seq=5;seq<=12;seq++)begin
            integer size_bytes;
            case(seq%4)0:size_bytes=4096;1:size_bytes=64;2:size_bytes=512;3:size_bytes=576;endcase
            v=make_request(seq,(seq%4)*4096,size_bytes);expected[seq]=v;submit((seq-1)%2,v);complete((seq-1)%2,v);
            check(contexts_held==0 && admission_watermark==seq,"REUSE_WATERMARK");
        end
        rejected_request(1,make_request(12,0,64),REPLAY);passed();

        scenario=3;a=make_request(13,0,64);expected[13]=a;submit(0,a);revoke=1;tick();revoke=0;
        rejected_request(1,make_request(14,64,64),PERMISSION);complete(0,a);
        check(contexts_held==0,"CHANGE_REGION_WHILE_HELD");
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;
        cfg.source_base=64'h200000000;cfg.source_size=8192;cfg.published_bytes=8192;configure();
        $display("SN PUBLISHED %0d 8589934592",$rtoi($realtime*1000));
        b=make_request(14,0,64);expected[14]=b;submit(1,b);complete(1,b);
        c=make_request(15,4096);c.source_offset=0;d=make_request(16,8192);d.source_offset=0;
        expected[15]=c;expected[16]=d;submit(0,c);submit(1,d);complete(0,c);complete(1,d);
        check(contexts_held==0,"HIGH_REGION_RETIRE");passed();

        scenario=4;ar_open=0;dma_ready=0;result_ready=0;
        a=make_request(17);b=make_request(18,4096);expected[17]=a;expected[18]=b;submit(0,a);submit(1,b);
        while(h_arvalid!=3)tick();stop[0]=1;tick(10);stop=0;result_ready=3;tick(40);
        check(admission_halted && halted==3 && quarantined==3 && contexts_held==5 && request_ready==0 && !cfg_ready,"GLOBAL_STOP");
        check(received[17]==0 && received[18]==0,"STOP_DATA");passed();
        $display("SG ALL_PASS mode=full cases=4");
        $display("SN ALL_PASS native=%0d",NATIVE);$finish;
    end
    initial begin #150000;$fatal(1,"SN_WATCHDOG case=%0d",scenario);end
