    request_t a,b,c,d,v;
    lease_t saved_cfg;
    region_map_t saved_map;
    initial begin
        for(integer l=0;l<2;l++)begin
            wdone[l]=0;wbeats[l]=0;wtag[l]=0;ar_count[l]=0;read_count[l]=0;read_beat[l]=0;
        end
        epoch=1;scenario=0;apb_rst_n=0;tick(NATIVE?940:10);apb_rst_n=1;cold_reset_n=1;
        while(!started || !wstarted)tick();tick(5);
        $display("SN BOOT %0d %b %b",$rtoi($realtime*1000),apb_complete_0,apb_complete_1);
        for(integer i=0;i<4;i++)preload(0,i+1,64'(i)*4096);
        for(integer i=0;i<2;i++)preload(1,i+1,64'h200000000+64'(i)*4096);
        check(wdone[0]==4 && wdone[1]==2,"PRELOAD_COUNT");
        $display("SN PRELOAD_DONE %0d",$rtoi($realtime*1000));
        locality(1,0,34'h200000000);locality(2,34'h200000000,0);locality(3,0,0);
        tick();direct_mode=0;cfg=initial_lease();cfg_map='0;cfg_map.generation=1;cfg_map.valid=3;
        cfg_map.first.logical_base=0;cfg_map.first.physical_base=0;cfg_map.first.bytes=16384;
        cfg_map.second.logical_base=32768;cfg_map.second.physical_base=64'h200000000;cfg_map.second.bytes=8192;
        configure();$display("SN PUBLISHED %0d 1",$rtoi($realtime*1000));

        scenario=1;a=make_request(1);b=make_request(2,4096);b.source_offset=32768;
        c=make_request(3,8192);c.source_offset=4096;d=make_request(4,12288);d.source_offset=36864;
        expected[1]=a;expected[2]=b;expected[3]=c;expected[4]=d;
        submit(0,a);submit(1,b);submit(0,c);submit(1,d);while(dma_valid!=3)tick();tick(20);
        check(contexts_held==10,"SHARED_FOUR_CONTEXTS");
        saved_cfg=cfg;saved_map=cfg_map;
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;cfg_map.generation=2;
        cfg_map.first.physical_base=64'h100000;
        cfg_valid=1;tick(8);check(!cfg_ready && !cfg_error,"CFG_WHILE_HELD");cfg_valid=0;
        cfg=saved_cfg;cfg_map=saved_map;
        dma_ready=3;wait_data(a);wait_data(b);wait_data(c);wait_data(d);
        notify_ack(0,c);notify_dma(0,c);notify_ack(1,d);notify_dma(1,d);
        notify_dma(0,a);notify_dma(1,b);tick(6);check(result_valid==0 && contexts_held==10,"EARLY_RELEASE");
        notify_ack(0,a);notify_ack(1,b);while(result_valid!=3)tick();tick(10);
        check(contexts_held==10,"OFFER_RELEASED");result_ready=1;while(results[0]<2)tick();tick(4);
        check(contexts_held==8,"ASYMMETRIC_RETIRE");rejected_request(0,make_request(5,4096),OVERLAP);
        result_ready=3;while(results[1]<2)tick();tick(3);check(contexts_held==0,"CASE1_HELD");
        v=make_request(5,0,64);v.source_offset=16384;rejected_request(0,v,MAP_MISS);passed();

        scenario=2;
        for(integer batch=0;batch<2;batch++)begin
            dma_ready=0;r_open=0;
            for(integer index=0;index<4;index++)begin
                integer seq,size_bytes;
                seq=5+batch*4+index;
                case(index)0:size_bytes=64;1:size_bytes=512;2:size_bytes=576;3:size_bytes=4096;endcase
                v=make_request(seq,(seq%4)*4096,size_bytes);
                v.source_offset=((index==1 || index==2)^batch)?32768:0;
                expected[seq]=v;submit(index%2,v);
            end
            tick(20);check(contexts_held==10 && dut.domain_pending==3 && dut.cmd_valid==3 &&
                dut.reader_cmd_valid==0,"DOMAIN_BOUNDARY_PENDING");
            r_open=3;dma_ready=1;tick(20);check(contexts_held==10,"DOMAIN_SWITCH_IS_NOT_RETIREMENT");
            dma_ready=3;
            for(integer index=0;index<4;index++)complete(index%2,expected[5+batch*4+index]);
            check(contexts_held==0 && admission_watermark==8+batch*4,"REUSE_WATERMARK");
        end
        rejected_request(1,make_request(12,0,64),REPLAY);passed();

        scenario=3;saved_cfg=cfg;saved_map=cfg_map;
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;cfg_map.generation=2;
        cfg_map.first.physical_base=cfg_map.second.physical_base;configure(1);
        cfg=saved_cfg;cfg_map=saved_map;
        a=make_request(13,0,64);expected[13]=a;submit(0,a);revoke=1;tick();revoke=0;
        rejected_request(1,make_request(14,64,64),PERMISSION);complete(0,a);
        check(contexts_held==0,"CHANGE_REGION_WHILE_HELD");
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;cfg_map.generation=2;
        cfg_map.first.physical_base=64'h200000000;cfg_map.first.bytes=8192;
        cfg_map.second.physical_base=0;cfg_map.second.bytes=16384;configure();
        $display("SN PUBLISHED %0d 2",$rtoi($realtime*1000));
        b=make_request(14,0,64);b.source_offset=32768;expected[14]=b;submit(1,b);complete(1,b);
        c=make_request(15,4096);c.source_offset=0;d=make_request(16,8192);d.source_offset=32768;
        expected[15]=c;expected[16]=d;submit(0,c);submit(1,d);complete(0,c);complete(1,d);
        check(contexts_held==0,"REMAP_RETIRE");passed();

        scenario=4;ar_open=0;dma_ready=0;result_ready=0;
        a=make_request(17);b=make_request(18,4096);b.source_offset=32768;
        expected[17]=a;expected[18]=b;submit(0,a);submit(1,b);
        while(h_arvalid!=3)tick();stop[0]=1;tick(10);stop=0;result_ready=3;tick(40);
        check(admission_halted && halted==3 && quarantined==3 && contexts_held==5 &&
            request_ready==0 && !cfg_ready,"GLOBAL_STOP");
        check(received[17]==0 && received[18]==0,"STOP_DATA");passed();
        $display("SG ALL_PASS mode=full cases=4");
        $display("SN ALL_PASS native=%0d",NATIVE);$finish;
    end
    initial begin #150000;$fatal(1,"SN_WATCHDOG case=%0d",scenario);end
