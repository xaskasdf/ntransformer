    // Two independent AXI3 sinks. AW and W may arrive in either order.
    integer hc=0,ha[0:1],hw[0:1],hb[0:1],htail[0:1],hbeat[0:1];
    integer hdue[0:4095],hlen[0:4095],hcount[0:4095];
    reg [33:0] haddress[0:4095];
    reg [1:0] hsent=0,hahold=0,hwhold=0;
    reg [97:0] hasaved;
    reg [577:0] hwsaved;
    always @(negedge clk)begin
        for(integer ch=0;ch<2;ch++)begin
            h_awready[ch]=rst_n&&!hold_aw[ch]&&cycle%5!=0&&ha[ch]-hb[ch]<32;
            h_wready[ch]=rst_n&&!hold_w[ch]&&cycle%7!=0&&hw[ch]-hb[ch]<32;
            if(!h_bvalid[ch]||hsent[ch]||!rst_n)begin
                h_bvalid[ch]=rst_n&&hb[ch]<htail[ch]&&cycle>=hdue[ch*2048+hb[ch]]&&!hold_b[ch];
                h_bid[ch*6+:6]=scenario==16?16:0;
                h_bresp[ch*2+:2]=scenario==17?1:
                    (int'((haddress[ch*2048+hb[ch]]-34'h100000000)/4096)==error_tag?2:0);
            end
        end
    end
    // Log target events at the same edge as transport events; independent
    // replay sorts each edge by dependency, not Verilog process print order.
    always @(posedge clk)begin
        hc++;
        hsent=h_bvalid&h_bready;
        if(!rst_n)begin
            hahold=0;hwhold=0;
            for(integer ch=0;ch<2;ch++)begin ha[ch]=0;hw[ch]=0;hb[ch]=0;htail[ch]=0;hbeat[ch]=0;end
        end else begin
            for(integer ch=0;ch<2;ch++)begin
                need(!hahold[ch]||(h_awvalid[ch]&&{h_awaddr[ch*34+:34],h_awlen[ch*4+:4],h_awsize[ch*3+:3],h_awburst[ch*2+:2],h_awid[ch*6+:6]}===hasaved[ch*49+:49]),"HBM_AW_STABLE");
                need(!hwhold[ch]||(h_wvalid[ch]&&{h_wdata[ch*256+:256],h_wstrb[ch*32+:32],h_wlast[ch]}===hwsaved[ch*289+:289]),"HBM_W_STABLE");
                hahold[ch]=h_awvalid[ch]&&!h_awready[ch];
                hasaved[ch*49+:49]={h_awaddr[ch*34+:34],h_awlen[ch*4+:4],h_awsize[ch*3+:3],h_awburst[ch*2+:2],h_awid[ch*6+:6]};
                hwhold[ch]=h_wvalid[ch]&&!h_wready[ch];hwsaved[ch*289+:289]={h_wdata[ch*256+:256],h_wstrb[ch*32+:32],h_wlast[ch]};
                if(h_awvalid[ch]&&h_awready[ch])begin
                    need(ha[ch]<2048&&h_awid[ch*6+:6]==0&&h_awsize[ch*3+:3]==5&&h_awburst[ch*2+:2]==1,"HBM_AW_ATTRIBUTES");
                    haddress[ch*2048+ha[ch]]=h_awaddr[ch*34+:34];hlen[ch*2048+ha[ch]]=int'(h_awlen[ch*4+:4])+1;
                    $display("HT AW %0d %0d %0d %h %0d",hc,ch,ha[ch],h_awaddr[ch*34+:34],int'(h_awlen[ch*4+:4])+1);ha[ch]++;
                end
                if(h_wvalid[ch]&&h_wready[ch])begin
                    need(hw[ch]<2048&&hbeat[ch]<16,"HBM_BEAT_BOUNDS");
                    $display("HT W %0d %0d %0d %0d %h %h %b",hc,ch,hw[ch],hbeat[ch],h_wdata[ch*256+:256],h_wstrb[ch*32+:32],h_wlast[ch]);
                    hbeat[ch]++;
                    if(h_wlast[ch])begin hcount[ch*2048+hw[ch]]=hbeat[ch];hw[ch]++;hbeat[ch]=0;end
                end
                while(htail[ch]<ha[ch]&&htail[ch]<hw[ch])begin
                    need(hcount[ch*2048+htail[ch]]==hlen[ch*2048+htail[ch]],"HBM_WLAST_LENGTH");
                    hdue[ch*2048+htail[ch]]=cycle+20;htail[ch]++;
                end
                if(h_bvalid[ch]&&h_bready[ch])begin
                    need(hb[ch]<htail[ch],"HBM_B_CAUSAL");
                    $display("HT B %0d %0d %0d %0d %0d",hc,ch,hb[ch],h_bid[ch*6+:6],h_bresp[ch*2+:2]);hb[ch]++;
                end
                need(ha[ch]-hb[ch]<=32&&hw[ch]-hb[ch]<=32,"HBM_CREDITS");
            end
        end
    end
