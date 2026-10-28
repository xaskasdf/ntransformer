    integer scenario=0,epoch=0,cycles=0,commands=0,writes=0,read_beats=0,output_beats=0,completions=0,dual=0;
    integer expected[0:1];
    integer wdone[0:4095],done[0:4095],length_by_tag[0:4095],expected_by_tag[0:4095];
    integer delivered_by_tag[0:4095],received_by_tag[0:4095];
    integer tags_in_order[0:1][0:511],push[0:1],pop[0:1];
    integer fragment_tag[0:1][0:1023],fragment_left[0:1][0:1023],fp[0:1],fr[0:1];
    reg allow_fault=0;
    reg [1:0] held_data=0,held_cpl=0,held_ar=0;
    reg [1153:0] saved_data;
    reg [157:0] saved_cpl;
    reg [101:0] saved_ar;
    function automatic longint stamp();return longint'($realtime*1000);endfunction
    function automatic [63:0] destination(input integer tag);
        return (tag%2?64'h200000000:0)+64'h10000+64'(tag/2)*4096;
    endfunction
    function automatic [511:0] pattern(input integer tag,input integer beat);
        for(integer j=0;j<16;j++)pattern[j*32+:32]=(tag*32'h9e3779b9)^(beat*32'h10203)^(j*32'h1010101);
    endfunction
    task automatic need(input bit ok,input string why);
        if(!ok)$fatal(1,"RF_FAIL case=%0d epoch=%0d cycle=%0d %s",scenario,epoch,cycles,why);
    endtask
    task automatic step;@(negedge clk);#0.2;endtask
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin
            held_data=0;held_cpl=0;
            for(integer c=0;c<2;c++)begin push[c]=0;pop[c]=0;end
        end else begin
            if(!allow_fault)need(!fault&&!wfault,"UNEXPECTED_FAULT");
            if(!started)need(cmd_ready==0&&!drained,"STARTUP_GATING");
            if(stopped)need(cmd_ready==0,"STOP_RETENTION");
            if((data_valid&data_ready)==3)dual++;
            for(integer c=0;c<2;c++)begin
                need(!held_data[c]||(data_valid[c]&&{data_tag[c*64+:64],data_last[c],data[c*512+:512]}===saved_data[c*577+:577]),"DATA_STABILITY");
                need(!held_cpl[c]||(cpl_valid[c]&&{cpl_tag[c*64+:64],cpl_status[c*2+:2],cpl_bytes[c*13+:13]}===saved_cpl[c*79+:79]),"CPL_STABILITY");
                held_data[c]=data_valid[c]&&!data_ready[c];saved_data[c*577+:577]={data_tag[c*64+:64],data_last[c],data[c*512+:512]};
                held_cpl[c]=cpl_valid[c]&&!cpl_ready[c];saved_cpl[c*79+:79]={cpl_tag[c*64+:64],cpl_status[c*2+:2],cpl_bytes[c*13+:13]};
                if(wcmd_valid[c]&&wcmd_ready[c])begin
                    writes++;$display("RR WCMD %0d %0d %0d %0d %0d %h",stamp(),epoch,c,wcmd_tag[c*64+:64],int'(wcmd_len[c*8+:8])+1,wcmd_addr[c*64+:64]);
                end
                if(wdata_valid[c]&&wdata_ready[c])$display("RR WI %0d %0d %0d %h",stamp(),epoch,c,wdata[c*512+:512]);
                if(wcpl_valid[c])begin
                    need(wcpl_status[c*2+:2]==0,"WRITE_COMPLETION");wdone[wcpl_tag[c*64+:64]]=1;
                    $display("RR WCPL %0d %0d %0d %0d",stamp(),epoch,c,wcpl_tag[c*64+:64]);
                end
                if(cmd_valid[c]&&cmd_ready[c])begin
                    automatic integer t=int'(cmd_tag[c*64+:64]);
                    need(push[c]-pop[c]<FRAMES&&push[c]<512,"FRAME_CAPACITY");
                    tags_in_order[c][push[c]]=t;push[c]++;
                    length_by_tag[t]=int'(cmd_len[c*8+:8])+1;expected_by_tag[t]=expected[c];delivered_by_tag[t]=0;received_by_tag[t]=0;
                    commands++;$display("RR CMD %0d %0d %0d %0d %0d %h",stamp(),epoch,c,t,length_by_tag[t],cmd_addr[c*64+:64]);
                end
                if(data_valid[c])begin
                    automatic integer t=int'(data_tag[c*64+:64]);
                    need(pop[c]<push[c]&&tags_in_order[c][pop[c]]==t,"PUBLICATION_ORDER");
                    need(received_by_tag[t]==2*length_by_tag[t]&&expected_by_tag[t]==0,"PUBLICATION_BEFORE_FULL_FRAME");
                end
                if(data_valid[c]&&data_ready[c])begin
                    automatic integer t=int'(data_tag[c*64+:64]);
                    need(data[c*512+:512]===pattern(t,delivered_by_tag[t])&&data_last[c]==(delivered_by_tag[t]+1==length_by_tag[t]),"RETURN_DATA");
                    $display("RR DATA %0d %0d %0d %0d %0d %0d %h",stamp(),epoch,c,t,delivered_by_tag[t],data_last[c],data[c*512+:512]);
                    delivered_by_tag[t]++;output_beats++;
                end
                if(cpl_valid[c]&&cpl_ready[c])begin
                    automatic integer t=int'(cpl_tag[c*64+:64]);
                    need(pop[c]<push[c]&&tags_in_order[c][pop[c]]==t&&cpl_status[c*2+:2]==expected_by_tag[t],"COMPLETION_ORDER_STATUS");
                    need(delivered_by_tag[t]==(expected_by_tag[t]==0?length_by_tag[t]:0)&&cpl_bytes[c*13+:13]==delivered_by_tag[t]*64,"COMPLETION_BYTES");
                    $display("RR CPL %0d %0d %0d %0d %0d %0d",stamp(),epoch,c,t,cpl_status[c*2+:2],cpl_bytes[c*13+:13]);
                    done[t]=1;pop[c]++;completions++;
                end
            end
        end
    end
    always @(posedge mem_clk)begin
        if(!mem_rst_n)begin held_ar=0;for(integer c=0;c<2;c++)begin native_reads[c]=0;native_addresses[c]=0;fp[c]=0;fr[c]=0;end end
        else for(integer c=0;c<2;c++)begin
            need(!held_ar[c]||(h_arvalid[c]&&{h_araddr[c*34+:34],h_arlen[c*4+:4],h_arid[c*6+:6],h_arsize[c*3+:3],h_arburst[c*2+:2]}===saved_ar[c*51+:51]),"AR_STABILITY");
            held_ar[c]=h_arvalid[c]&&!h_arready[c];saved_ar[c*51+:51]={h_araddr[c*34+:34],h_arlen[c*4+:4],h_arid[c*6+:6],h_arsize[c*3+:3],h_arburst[c*2+:2]};
            if(h_arvalid[c]&&h_arready[c])begin
                automatic longint a=longint'(h_araddr[c*34+:34]);
                automatic integer t=int'(((a&64'h1ffffffff)-64'h10000)/4096)*2+c;
                need(fp[c]<1024&&h_arid[c*6+:6]==0&&h_arsize[c*3+:3]==5&&h_arburst[c*2+:2]==1,"NATIVE_ADDRESS");
                fragment_tag[c][fp[c]]=t;fragment_left[c][fp[c]]=int'(h_arlen[c*4+:4])+1;fp[c]++;
                native_addresses[c]++;$display("RR AR %0d %0d %0d %h %0d",stamp(),epoch,c,h_araddr[c*34+:34],int'(h_arlen[c*4+:4])+1);
            end
            if(h_rvalid[c]&&h_rready[c])begin
                need(fr[c]<fp[c],"RESPONSE_WITHOUT_FRAGMENT");received_by_tag[fragment_tag[c][fr[c]]]++;
                fragment_left[c][fr[c]]--;if(fragment_left[c][fr[c]]==0)fr[c]++;
                native_reads[c]++;read_beats++;
                $display("RR NR %0d %0d %0d %0d %0d %0d %h",stamp(),epoch,c,h_rid[c*6+:6],h_rresp[c*2+:2],h_rlast[c],h_rdata[c*256+:256]);
            end
        end
    end
    for(genvar f=0;f<2;f++)begin: frame_monitor
        always @(posedge clk)if(rst_n)begin
            $display("RF STATE %0d %0d %0d %0d",stamp(),epoch,f,dut.channels[f].frame.used);
            if(dut.channels[f].frame.store_beat)
                $display("RF FILL %0d %0d %0d %0d %0d %0d %0d %h",stamp(),epoch,f,
                    dut.channels[f].frame.tags[dut.channels[f].frame.fetch_ptr],dut.channels[f].frame.received,
                    dut.channels[f].frame.rresp,dut.channels[f].frame.rlast,dut.channels[f].frame.rdata);
            if(dut.channels[f].frame.seal_beat||dut.channels[f].frame.seal_invalid)
                $display("RF SEAL %0d %0d %0d %0d %0d",stamp(),epoch,f,
                    dut.channels[f].frame.tags[dut.channels[f].frame.fetch_ptr],
                    dut.channels[f].frame.seal_invalid?1:((dut.channels[f].frame.status[dut.channels[f].frame.fetch_ptr]!=0||dut.channels[f].frame.rresp!=0)?2:0));
            if(dut.channels[f].frame.store_beat&&data_valid[f]&&data_ready[f])
                $display("RF OVERLAP %0d %0d %0d %0d %0d",stamp(),epoch,f,
                    dut.channels[f].frame.tags[dut.channels[f].frame.fetch_ptr],data_tag[f*64+:64]);
        end
    end
    task automatic boot;
        step;cold_reset_n=0;apb_rst_n=0;mem_pause=0;stop=0;fault_in=0;init_mask=3;allow_fault=0;
        ar_open=3;r_open=3;stray=0;injection=0;inject_index=0;cmd_valid=0;wcmd_valid=0;wdata_valid=0;data_ready=3;cpl_ready=3;
        expected[0]=0;expected[1]=0;
        repeat(NATIVE?940:12)step;epoch++;apb_rst_n=1;cold_reset_n=1;
        wait(started&&wstarted);step;need(mem_rst_n&&wmem_rst_n&&apb_complete_0&&apb_complete_1,"BOOT_STATUS");
        $display("RR BOOT %0d %0d %b %b %b %b",stamp(),epoch,apb_complete_0,apb_complete_1,DRAM_0_STAT_CATTRIP,DRAM_1_STAT_CATTRIP);
    endtask
    task automatic write_frame(input integer c,t,n);
        step;wcmd_addr[c*64+:64]=destination(t);wcmd_tag[c*64+:64]=t;wcmd_len[c*8+:8]=n-1;wcmd_valid[c]=1;
        do @(posedge clk);while(!wcmd_ready[c]);step;wcmd_valid[c]=0;
        for(integer i=0;i<n;i++)begin
            wdata_valid[c]=1;wdata[c*512+:512]=pattern(t,i);
            do @(posedge clk);while(!wdata_ready[c]);step;wdata_valid[c]=0;
        end
        wait(wdone[t]);step;
    endtask
    task automatic command(input integer c,t,n);
        step;cmd_addr[c*64+:64]=destination(t);cmd_tag[c*64+:64]=t;cmd_len[c*8+:8]=n-1;cmd_valid[c]=1;
        do @(posedge clk);while(!cmd_ready[c]);step;cmd_valid[c]=0;
    endtask
    task automatic pair_write(input integer t,n);fork write_frame(0,t,n);write_frame(1,t+1,n);join endtask
    task automatic pair_command(input integer t,n);fork command(0,t,n);command(1,t+1,n);join endtask
    task automatic pair_done(input integer t);wait(done[t]&&done[t+1]);step;endtask
    task automatic begin_case(input integer n);scenario=n;$display("RR CASE %0d %0d",n,stamp());endtask
    task automatic passed;
        repeat(3)step;need(drained,"NOT_DRAINED");$display("RR PASS %0d %0d",scenario,stamp());
    endtask
    initial begin
        $display("RF CONFIG frames=%0d native=%0d common=%0d",FRAMES,NATIVE,COMMON);
        for(integer t=0;t<4096;t++)begin wdone[t]=0;done[t]=0;received_by_tag[t]=0;end
        begin_case(1);boot;
        for(integer j=0;j<4;j++)begin
            automatic integer n;
            case(j)0:n=1;1:n=8;2:n=9;default:n=64;endcase
            pair_write(j*2,n);pair_command(j*2,n);pair_done(j*2);
        end
        passed;
        begin_case(2);
        for(integer j=0;j<FRAMES;j++)pair_write(100+j*2,64);
        data_ready=0;cpl_ready=0;
        for(integer j=0;j<FRAMES;j++)pair_command(100+j*2,64);
        wait(dut.channels[0].frame.bank_state[FRAMES-1]==2&&dut.channels[1].frame.bank_state[FRAMES-1]==2);
        repeat(20)step;need(cmd_ready==0&&!drained,"BOTH_RESERVATIONS");
        $display("RF FULL %0d %0d",stamp(),epoch);
        data_ready=1;wait(cpl_valid[0]);repeat(12)step;need(data_valid[1]&&!drained&&cmd_ready==0,"ASYMMETRIC_HOLD");
        data_ready=3;wait(cpl_valid==3);repeat(12)step;
        $display("RF CPL_HELD %0d %0d",stamp(),epoch);cpl_ready=3;
        pair_done(100+2*(FRAMES-1));passed;
        begin_case(3);if(!NATIVE)boot;
        for(integer j=0;j<12;j++)pair_write(300+2*j,64);
        $display("RF STREAM_BEGIN %0d %0d",stamp(),epoch);
        fork
            begin for(integer j=0;j<12;j++)command(0,300+2*j,64);end
            begin for(integer j=0;j<12;j++)command(1,301+2*j,64);end
        join
        pair_done(322);passed;
        begin_case(4);
        for(integer j=0;j<FRAMES;j++)pair_write(200+j*2,64);
        r_open=0;
        for(integer j=0;j<FRAMES;j++)pair_command(200+j*2,64);
        wait(adapter_pending==3);step;stop=1;repeat(8)step;stop=0;
        need(stopped&&!drained&&cmd_ready==0,"STOP_PENDING");r_open=3;
        pair_done(200+2*(FRAMES-1));passed;
        if(!NATIVE&&FRAMES==2)begin
            for(integer mode=1;mode<=6;mode++)begin
                begin_case(4+mode);boot;
                write_frame(0,400+mode*4,9);write_frame(0,402+mode*4,9);
                data_ready=2;cpl_ready=2;
                command(0,400+mode*4,9);wait(data_valid[0]);
                allow_fault=1;injection=mode;inject_index=native_reads[0]+((mode<=2)?17:0);
                if(mode<=2)expected[0]=2;
                command(0,402+mode*4,9);
                wait(fault);repeat(25)step;
                need(data_valid[0]&&data_tag[63:0]==400+mode*4&&!drained,"FIRST_FRAME_SURVIVES_SECOND_FAULT");
                data_ready=3;cpl_ready=3;wait(done[400+mode*4]);
                if(mode<=2)begin wait(done[402+mode*4]);passed;end
                else begin
                    repeat(40)step;need(!drained&&!done[402+mode*4]&&!data_valid[0]&&!cpl_valid[0],"SECOND_QUARANTINED");
                    $display("RF QUARANTINE %0d %0d %0d",stamp(),epoch,402+mode*4);
                    $display("RR PASS %0d %0d",scenario,stamp());
                end
            end
            begin_case(11);boot;pair_write(500,9);pair_write(502,9);mem_pause=1;
            fork
                begin pair_command(500,9);pair_command(502,9);end
                begin repeat(150)step;need(!drained&&cmd_ready==0,"PAUSED_TWO_FRAMES");mem_half=3.317;mem_pause=0;end
            join
            pair_done(502);passed;
            begin_case(12);boot;pair_write(600,64);pair_write(602,64);r_open=0;
            pair_command(600,64);pair_command(602,64);wait(adapter_pending==3);
            step;allow_fault=1;fault_in=1;step;fault_in=0;need(fault,"EXTERNAL_FAULT");r_open=3;
            pair_done(602);passed;
        end
        $display("RF ALL_PASS cases=%0d commands=%0d writes=%0d native_reads=%0d output_beats=%0d completions=%0d dual=%0d",(!NATIVE&&FRAMES==2)?12:4,commands,writes,read_beats,output_beats,completions,dual);
        $finish;
    end
