    integer scenario=0,cycle=0,aw_count=0,w_count=0,b_count=0,ar_count=0,r_count=0;
    integer b_stalls=0,r_stalls=0,thermal_unknown_cycles=0;
    bit booted=0;
    reg [255:0] expected[bit[33:0]];
    function automatic longint stamp();return longint'($realtime*1000.0);endfunction
    function automatic [255:0] pattern(input integer number,input integer lane,input integer beat);
        for(integer k=0;k<8;k++)pattern[k*32+:32]=32'(number*32'h9e3779b9)^32'(lane*32'h6a09e667)^32'(beat*32'h10203)^32'(k*32'h01010101);
    endfunction
    task automatic need(input bit ok,input string why);
        if(!ok)$fatal(1,"NATIVE_%s case=%0d time_ps=%0d",why,scenario,stamp());
    endtask
    task automatic send_aw(input integer lane,input bit[33:0] address,input integer beats,input bit[5:0] id,input integer delay_cycles);
        repeat(delay_cycles)@(negedge mem_clk);
        @(negedge mem_clk);s_awaddr[lane*34+:34]=address;s_awlen[lane*4+:4]=4'(beats-1);
        s_awid[lane*6+:6]=id;s_awsize[lane*3+:3]=5;s_awburst[lane*2+:2]=1;s_awvalid[lane]=1;
        do @(posedge mem_clk);while(!s_awready[lane]);
        @(negedge mem_clk);s_awvalid[lane]=0;
    endtask
    task automatic send_w(input integer lane,input bit[33:0] address,input integer beats,input integer number,input integer delay_cycles);
        reg[255:0] value,merged;reg[31:0] mask;
        repeat(delay_cycles)@(negedge mem_clk);
        for(integer j=0;j<beats;j++)begin
            @(negedge mem_clk);value=pattern(number,lane,j);
            mask=number==4?(j%2?32'h55555555:32'haaaaaaaa):32'hffffffff;
            s_wdata[lane*256+:256]=value;s_wstrb[lane*32+:32]=mask;s_wlast[lane]=(j+1==beats);s_wvalid[lane]=1;
            do @(posedge mem_clk);while(!s_wready[lane]);
            merged=expected.exists(address+34'(j*32))?expected[address+34'(j*32)]:0;
            for(integer k=0;k<32;k++)if(mask[k])merged[k*8+:8]=value[k*8+:8];
            expected[address+34'(j*32)]=merged;
            @(negedge mem_clk);s_wvalid[lane]=0;
            if(j%3==1)repeat(3)@(negedge mem_clk);
        end
    endtask
    task automatic write_burst(input integer lane,input bit[33:0] address,input integer beats,input integer number);
        bit[5:0] id;id=6'(number*7+lane*17);
        fork
            send_aw(lane,address,beats,id,number==3?9:0);
            send_w(lane,address,beats,number,number==1?7:0);
        join
        wait(s_bvalid[lane]===1'b1);
        // Count complete sampling edges with BREADY low. Seven falling edges
        // alone can cover only six BVALID sampling edges, depending on phase.
        repeat(7)@(posedge mem_clk);
        @(negedge mem_clk);
        s_bready[lane]=1;
        do @(posedge mem_clk);while(!s_bvalid[lane]);
        need(s_bid[lane*6+:6]===id&&s_bresp[lane*2+:2]===2'b0,"B_ID_RESP");
        @(negedge mem_clk);s_bready[lane]=0;
    endtask
    task automatic read_burst(input integer lane,input bit[33:0] address,input integer beats,input integer number);
        integer count;bit[5:0] id;reg[255:0] value;
        id=6'(number*11+lane*19);
        @(negedge mem_clk);s_araddr[lane*34+:34]=address;s_arlen[lane*4+:4]=4'(beats-1);
        s_arid[lane*6+:6]=id;s_arsize[lane*3+:3]=5;s_arburst[lane*2+:2]=1;s_arvalid[lane]=1;
        do @(posedge mem_clk);while(!s_arready[lane]);
        @(negedge mem_clk);s_arvalid[lane]=0;
        count=0;
        while(count<beats)begin
            @(negedge mem_clk);s_rready[lane]=(cycle%9>=4);
            @(posedge mem_clk);
            if(s_rvalid[lane]&&s_rready[lane])begin
                need(expected.exists(address+34'(count*32)),"READ_INITIALIZED");
                value=expected[address+34'(count*32)];
                need(s_rid[lane*6+:6]===id&&s_rresp[lane*2+:2]===2'b0,"R_ID_RESP");
                need(s_rlast[lane]==(count+1==beats),"RLAST");
                if(s_rdata[lane*256+:256]!==value)
                    $fatal(1,"NATIVE_READ_DATA case=%0d lane=%0d beat=%0d addr=%h expected=%h actual=%h",number,lane,count,address+34'(count*32),value,s_rdata[lane*256+:256]);
                count++;
            end
        end
        @(negedge mem_clk);s_rready[lane]=0;
    endtask
    reg[1:0] held_b=0,held_r=0;
    reg[15:0] saved_b;
    reg[529:0] saved_r;
    always @(posedge mem_clk)begin
        cycle++;
        if(booted)begin
            need(apb_complete_0===1'b1&&apb_complete_1===1'b1,"READY");
            // Native model thermal status is observed, never forced. An asserted
            // alarm still fails. X/Z makes thermal qualification unavailable;
            // this data-path-only bench must report that limitation explicitly.
            need(DRAM_0_STAT_CATTRIP!==1'b1&&DRAM_1_STAT_CATTRIP!==1'b1,"CATTRIP_ASSERTED");
            if(DRAM_0_STAT_CATTRIP!==1'b0||DRAM_1_STAT_CATTRIP!==1'b0)thermal_unknown_cycles++;
            for(integer lane=0;lane<2;lane++)begin
                need(!held_b[lane]||(s_bvalid[lane]&&{s_bid[lane*6+:6],s_bresp[lane*2+:2]}===saved_b[lane*8+:8]),"B_STABLE");
                need(!held_r[lane]||(s_rvalid[lane]&&{s_rdata[lane*256+:256],s_rid[lane*6+:6],s_rresp[lane*2+:2],s_rlast[lane]}===saved_r[lane*265+:265]),"R_STABLE");
                held_b[lane]=s_bvalid[lane]&&!s_bready[lane];
                held_r[lane]=s_rvalid[lane]&&!s_rready[lane];
                saved_b[lane*8+:8]={s_bid[lane*6+:6],s_bresp[lane*2+:2]};
                saved_r[lane*265+:265]={s_rdata[lane*256+:256],s_rid[lane*6+:6],s_rresp[lane*2+:2],s_rlast[lane]};
                if(held_b[lane])begin
                    b_stalls++;$display("NT BS %0d %0d %0d %0d %0d",stamp(),scenario,lane,s_bid[lane*6+:6],s_bresp[lane*2+:2]);
                end
                if(held_r[lane])begin
                    r_stalls++;$display("NT RS %0d %0d %0d %h %0d %0d %b",stamp(),scenario,lane,s_rdata[lane*256+:256],s_rid[lane*6+:6],s_rresp[lane*2+:2],s_rlast[lane]);
                end
                if(s_awvalid[lane]&&s_awready[lane])begin
                    aw_count++;$display("NT AW %0d %0d %0d %h %0d %0d",stamp(),scenario,lane,s_awaddr[lane*34+:34],int'(s_awlen[lane*4+:4])+1,s_awid[lane*6+:6]);
                end
                if(s_wvalid[lane]&&s_wready[lane])begin
                    w_count++;$display("NT W %0d %0d %0d %h %h %b",stamp(),scenario,lane,s_wdata[lane*256+:256],s_wstrb[lane*32+:32],s_wlast[lane]);
                end
                if(s_bvalid[lane]&&s_bready[lane])begin
                    b_count++;$display("NT B %0d %0d %0d %0d %0d",stamp(),scenario,lane,s_bid[lane*6+:6],s_bresp[lane*2+:2]);
                end
                if(s_arvalid[lane]&&s_arready[lane])begin
                    ar_count++;$display("NT AR %0d %0d %0d %h %0d %0d",stamp(),scenario,lane,s_araddr[lane*34+:34],int'(s_arlen[lane*4+:4])+1,s_arid[lane*6+:6]);
                end
                if(s_rvalid[lane]&&s_rready[lane])begin
                    r_count++;$display("NT R %0d %0d %0d %h %0d %0d %b",stamp(),scenario,lane,s_rdata[lane*256+:256],s_rid[lane*6+:6],s_rresp[lane*2+:2],s_rlast[lane]);
                end
            end
        end else need(s_awvalid==0&&s_wvalid==0&&s_arvalid==0,"NO_TRAFFIC_BEFORE_BOOT");
    end
    initial begin
        #4700;
        @(negedge apb_clk);apb_rst_n=1;
        @(negedge mem_clk);mem_rst_n=1;
        $display("NT RELEASE %0d",stamp());
        wait(apb_complete_0===1'b1&&apb_complete_1===1'b1);
        repeat(32)@(posedge apb_clk);
        @(negedge mem_clk);booted=1;$display("NT BOOT %0d",stamp());
        $display("NT THERMAL %0d %b %b %h %h",stamp(),DRAM_0_STAT_CATTRIP,DRAM_1_STAT_CATTRIP,DRAM_0_STAT_TEMP,DRAM_1_STAT_TEMP);
        for(integer number=1;number<=5;number++)begin
            bit[33:0] offset;integer beats;
            scenario=number;
            offset=number==1?34'h1000:number==2||number==4?34'h2000:number==3?34'h3000:34'h1fffffe0;
            beats=number==1||number==5?1:number==3?8:16;
            $display("NT CASE %0d %0d",stamp(),number);
            fork
                write_burst(0,offset,beats,number);
                write_burst(1,34'h200000000+offset,beats,number);
            join
            fork
                read_burst(0,offset,beats,number);
                read_burst(1,34'h200000000+offset,beats,number);
            join
            $display("NT CASE_PASS %0d",number);
        end
        repeat(20)@(negedge mem_clk);
        $display("NT THERMAL_END time_ps=%0d unknown_cycles=%0d cattrip0=%b cattrip1=%b",stamp(),thermal_unknown_cycles,DRAM_0_STAT_CATTRIP,DRAM_1_STAT_CATTRIP);
        $display("NT COUNTS aw=%0d w=%0d b=%0d ar=%0d r=%0d b_stalls=%0d r_stalls=%0d",aw_count,w_count,b_count,ar_count,r_count,b_stalls,r_stalls);
        need(aw_count==10&&b_count==10&&ar_count==10&&w_count==84&&r_count==84&&b_stalls>=70&&r_stalls>0,"COUNTS_AND_STALLS");
        $display("NT ALL_PASS cases=5 aw=%0d w=%0d b=%0d ar=%0d r=%0d b_stalls=%0d r_stalls=%0d time_ps=%0d",aw_count,w_count,b_count,ar_count,r_count,b_stalls,r_stalls,stamp());$finish;
    end
    initial begin #100000;$fatal(1,"NATIVE_WATCHDOG");end
