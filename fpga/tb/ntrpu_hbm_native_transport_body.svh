    integer memory_cycle=0,native_aw=0,native_w=0,native_b=0,native_ar=0,native_r=0;
    integer native_r_stalls=0,thermal_unknown=0,prestart_samples=0,cpl_stalls=0;
    integer peak_banks=0,read_tag[0:1],read_offset[0:1];
    reg [7:0] used_union=0;
    reg [1:0] held_r=0;
    reg [529:0] saved_r=0;
    bit booted=0;
    always @(negedge clk)begin
        #0.3;
        if(rst_n)begin
            if(!started)begin
                prestart_samples++;
                need(submit_ready==0&&!fetch_valid&&h_awvalid==0&&h_wvalid==0&&!drained,"PRESTART_CLOSED");
            end else begin
                need(!fault&&adapter_fault==0,"CLEAN_EPOCH");
                need(inflight[5:0]==aws[0]-bs[0]&&inflight[11:6]==aws[1]-bs[1],"CDC_LEDGER");
                if(drained)need(inflight==0,"DRAIN_NO_CDC_WORK");
                if($countones(banks_used)>peak_banks)peak_banks=$countones(banks_used);
                used_union=used_union|banks_used;
            end
        end
    end
    always @(posedge clk)if(started)begin
        for(integer lane=0;lane<2;lane++)if(cpl_valid[lane]&&!cpl_ready[lane])cpl_stalls++;
    end
    always @(posedge mem_clk)begin
        memory_cycle++;
        if(booted)begin
            need(apb_complete_0===1'b1&&apb_complete_1===1'b1,"NATIVE_READY");
            need(DRAM_0_STAT_CATTRIP!==1'b1&&DRAM_1_STAT_CATTRIP!==1'b1,"CATTRIP_ASSERTED");
            if(DRAM_0_STAT_CATTRIP!==1'b0||DRAM_1_STAT_CATTRIP!==1'b0)thermal_unknown++;
        end
        if(!mem_rst_n)held_r=0;
        else for(integer lane=0;lane<2;lane++)begin
            if(h_awvalid[lane]&&h_awready[lane])begin
                need(started&&h_awsize[lane*3+:3]==5&&h_awburst[lane*2+:2]==1,"NATIVE_AW");
                native_aw++;$display("NC AW %0d %0d %h %0d %0d",stamp(),lane,h_awaddr[lane*34+:34],int'(h_awlen[lane*4+:4])+1,h_awid[lane*6+:6]);
            end
            if(h_wvalid[lane]&&h_wready[lane])begin
                native_w++;$display("NC W %0d %0d %h %h %b",stamp(),lane,h_wdata[lane*256+:256],h_wstrb[lane*32+:32],h_wlast[lane]);
            end
            if(h_bvalid[lane]&&h_bready[lane])begin
                native_b++;$display("NC B %0d %0d %0d %0d",stamp(),lane,h_bid[lane*6+:6],h_bresp[lane*2+:2]);
            end
            need(!held_r[lane]||(s_rvalid[lane]&&{s_rdata[lane*256+:256],s_rid[lane*6+:6],s_rresp[lane*2+:2],s_rlast[lane]}===saved_r[lane*265+:265]),"NATIVE_R_STABLE");
            held_r[lane]=s_rvalid[lane]&&!s_rready[lane];
            saved_r[lane*265+:265]={s_rdata[lane*256+:256],s_rid[lane*6+:6],s_rresp[lane*2+:2],s_rlast[lane]};
            if(held_r[lane])begin native_r_stalls++;$display("NC RS %0d %0d %h %0d %0d %b",stamp(),lane,s_rdata[lane*256+:256],s_rid[lane*6+:6],s_rresp[lane*2+:2],s_rlast[lane]);end
            if(s_arvalid[lane]&&s_arready[lane])begin
                native_ar++;$display("NC AR %0d %0d %h %0d %0d",stamp(),lane,s_araddr[lane*34+:34],int'(s_arlen[lane*4+:4])+1,s_arid[lane*6+:6]);
            end
            if(s_rvalid[lane]&&s_rready[lane])begin
                native_r++;$display("NC R %0d %0d %h %0d %0d %b",stamp(),lane,s_rdata[lane*256+:256],s_rid[lane*6+:6],s_rresp[lane*2+:2],s_rlast[lane]);
            end
        end
    end
    task automatic push_one(input integer ch,input integer tag,input integer beats);
        step;submit_valid[ch]=1;submit_desc[ch*232+:232]=descriptor(tag,beats);
        for(integer tries=0;tries<16000;tries++)begin
            @(posedge clk);if(submit_ready[ch])begin step;submit_valid[ch]=0;return;end
            step;
        end
        need(0,"SUBMIT_TIMEOUT");
    endtask
    task automatic drain_all;
        for(integer tries=0;tries<4000;tries++)begin
            step;
            if(drained)begin
                need(!fault&&cpls[0]==submitted[0]&&cpls[1]==submitted[1],"CLEAN_DRAIN");
                $display("TE DRAIN %0d",stamp());return;
            end
        end
        need(0,"DRAIN_TIMEOUT");
    endtask
    task automatic send_fence(input integer id);
        step;fence_id=id;fence_valid=1;
        for(integer tries=0;tries<2000;tries++)begin
            @(posedge clk);if(fence_ready)begin step;fence_valid=0;return;end
            step;
        end
        need(0,"FENCE_TIMEOUT");
    endtask
    task automatic read_descriptor(input integer lane,input integer tag);
        integer total,offset,beats,count;reg [511:0] whole;reg [255:0] value;
        total=lengths[tag]*2;offset=0;read_tag[lane]=tag;
        while(offset<total)begin
            beats=(total-offset)>16?16:total-offset;
            @(negedge mem_clk);s_araddr[lane*34+:34]=34'(destination(tag)+64'(offset*32));
            s_arlen[lane*4+:4]=4'(beats-1);s_arid[lane*6+:6]=6'(tag);
            s_arsize[lane*3+:3]=5;s_arburst[lane*2+:2]=1;s_arvalid[lane]=1;
            do @(posedge mem_clk);while(!s_arready[lane]);
            @(negedge mem_clk);s_arvalid[lane]=0;count=0;
            while(count<beats)begin
                @(negedge mem_clk);s_rready[lane]=(memory_cycle%9>=3);
                @(posedge mem_clk);
                if(s_rvalid[lane]&&s_rready[lane])begin
                    whole=pattern(tag,(offset+count)/2);value=whole[((offset+count)%2)*256+:256];
                    need(s_rdata[lane*256+:256]===value,"READBACK_PATTERN");
                    need(s_rid[lane*6+:6]===6'(tag)&&s_rresp[lane*2+:2]===2'b0,"READBACK_ID_RESP");
                    need(s_rlast[lane]==(count+1==beats),"READBACK_LAST");count++;
                end
            end
            @(negedge mem_clk);s_rready[lane]=0;offset+=beats;read_offset[lane]=offset;
        end
        $display("NC VERIFIED %0d %0d %0d %0d",stamp(),lane,tag,total*32);
    endtask
    task automatic verify_pair(input integer first,input integer pairs);
        fork
            for(integer j=0;j<pairs;j++)read_descriptor(0,first+2*j);
            for(integer j=0;j<pairs;j++)read_descriptor(1,first+2*j+1);
        join
    endtask
    task automatic pass_case;
        $display("NC CASE_PASS %0d %0d %0d %0d %0d",scenario,submitted[0],submitted[1],cpls[0],cpls[1]);
    endtask
    initial begin
        #4700;@(negedge apb_clk);apb_rst_n=1;
        step;cold_reset_n=1;$display("NC RELEASE %0d",stamp());
        // Offer actual descriptors before calibration; the supervisor must hold them.
        scenario=1;hold_fetch=1;
        fork push_one(0,10,1);push_one(1,11,1);join
        need(started&&prestart_samples>0,"QUALIFIED_START");
        booted=1;$display("NC BOOT %0d %b %b",stamp(),apb_complete_0,apb_complete_1);
        $display("NC THERMAL %0d %b %b",stamp(),DRAM_0_STAT_CATTRIP,DRAM_1_STAT_CATTRIP);
        hold_fetch=0;
        for(integer j=2;j<8;j++)push_one(j%2,10+j,j<4?8:j<6?9:64);
        drain_all;verify_pair(10,4);pass_case;

        scenario=2;step;hold_cpl=3;
        fork push_one(0,20,8);push_one(1,21,9);join
        send_fence(77);wait(cpl_valid==3);repeat(20)step;
        need(!fence_done_valid&&!drained,"FENCE_WAITS_EXTERNAL_CPL");
        $display("NC FENCE_BLOCKED %0d",stamp());
        hold_cpl=0;wait(fence_done_valid);repeat(7)step;
        need(fence_done_id==77&&submit_ready==0&&!drained,"FENCE_RESULT_HELD");
        fence_done_ready=1;step;fence_done_ready=0;drain_all;
        verify_pair(20,1);pass_case;

        scenario=3;step;hold_cpl=3;
        fork
            for(integer j=0;j<6;j++)push_one(0,30+2*j,8);
            for(integer j=0;j<6;j++)push_one(1,31+2*j,8);
            begin repeat(300)step;need(!drained,"PRESSURE_RETAINS_WORK");hold_cpl=0;end
        join
        drain_all;verify_pair(30,6);pass_case;

        scenario=4;step;hold_fetch=1;
        fork push_one(0,50,8);push_one(1,51,8);join
        stop=1;step;stop=0;repeat(12)step;
        need(submit_ready==0&&!drained,"STOP_RETAINS_ACCEPTED_WORK");hold_fetch=0;
        drain_all;need(submit_ready==0,"STOP_STAYS_CLOSED");verify_pair(50,1);pass_case;
        repeat(20)step;
        $display("NC COUNTS sub=%0d cpl=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d r_stalls=%0d cpl_stalls=%0d peak_banks=%0d used_union=%0d prestart=%0d thermal_unknown=%0d",submitted[0]+submitted[1],cpls[0]+cpls[1],native_aw,native_w,native_b,native_ar,native_r,native_r_stalls,cpl_stalls,peak_banks,used_union,prestart_samples,thermal_unknown);
        need(submitted[0]+submitted[1]==24&&cpls[0]+cpls[1]==24&&native_w==586&&native_r==586&&native_aw==native_b&&native_ar==native_aw&&native_r_stalls>0&&cpl_stalls>0,"FINAL_COUNTS");
        $display("NC ALL_PASS cases=4 time_ps=%0d",stamp());$finish;
    end
    initial begin #80000;$fatal(1,"NATIVE_TRANSPORT_WATCHDOG");end
