    integer wdone[0:1],wbeats[0:1],wtag[0:1];
    reg [63:0] wbase[0:1];
    always @(posedge clk)if(wrst_n)begin
        check(!wfault,"NATIVE_WRITER_FAULT");
        for(integer l=0;l<2;l++)begin
            if(wcmd_valid[l] && wcmd_ready[l])begin
                wtag[l]=wcmd_tag[l*64+:64];wbase[l]=wcmd_addr[l*64+:64];wbeats[l]=0;
                $display("PN WCMD %0d %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,l,wtag[l],wbase[l],int'(wcmd_len[l*8+:8])+1);
            end
            if(wdata_valid[l] && wdata_ready[l])begin
                $display("PN WI %0d %0d %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,wtag[l],wbeats[l],wdata[l*512+:512]);wbeats[l]++;
            end
            if(wcpl_valid[l])begin
                check(wcpl_status[l*2+:2]==0 && wcpl_tag[l*64+:64]==wtag[l],"PRELOAD_CPL");
                wdone[l]++;$display("PN WCPL %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,l,wtag[l]);
            end
        end
    end
    task automatic preload(input integer l,seq,offset,length);
        integer before_done;
        before_done=wdone[l];tick();
        wcmd_addr[l*64+:64]=l*65536+offset;wcmd_tag[l*64+:64]=seq;
        wcmd_len[l*8+:8]=length/64-1;wcmd_valid[l]=1;
        do @(posedge clk);while(!wcmd_ready[l]);tick();wcmd_valid[l]=0;
        for(integer beat=0;beat<length/64;beat++)begin
            for(integer b=0;b<64;b++)wdata[l*512+b*8+:8]=value_at(l*65536+offset+beat*64+b,l);
            wdata_valid[l]=1;do @(posedge clk);while(!wdata_ready[l]);tick();wdata_valid[l]=0;
        end
        while(wdone[l]==before_done)tick();tick();
    endtask
    initial begin
        check(FULL==1 && PAIR==1,"NATIVE_PROFILE");
        for(integer l=0;l<2;l++)begin wdone[l]=0;wbeats[l]=0;wtag[l]=0;end
        scenario=0;boot();
        $display("PN BOOT %0d %b %b %b %b",$rtoi($realtime*1000),apb_complete_0,apb_complete_1,DRAM_0_STAT_CATTRIP,DRAM_1_STAT_CATTRIP);
        for(integer seq=1;seq<=10;seq++)begin
            integer size_bytes,offset;
            offset=seq<=2?(seq-1)*4096:seq*4096;
            size_bytes=seq<=2?4096:(seq%4==0)?4096:(seq%4==1)?576:(seq%4==2)?512:64;
            fork preload(0,seq,offset,size_bytes);preload(1,seq,offset,size_bytes);join
        end
        check(wdone[0]==10 && wdone[1]==10,"PRELOAD_COUNT");
        configure();$display("PN PUBLISHED %0d",$rtoi($realtime*1000));
