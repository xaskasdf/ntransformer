    // Testbench-only preflight target. This is not AMD HBM qualification.
    reg [255:0] memory[bit[33:0]];
    reg [1:0] active=0,reading=0;
    reg [33:0] address[0:1],raddress[0:1];
    integer remaining[0:1],rremaining[0:1];
    reg [5:0] id[0:1],read_id[0:1];
    initial begin
        apb_complete_0=0;apb_complete_1=0;
        DRAM_0_STAT_CATTRIP=1'bx;DRAM_1_STAT_CATTRIP=1'bx;
        DRAM_0_STAT_TEMP=0;DRAM_1_STAT_TEMP=0;
        wait(APB_0_PRESET_N&&APB_1_PRESET_N);#200;
        apb_complete_0=1;apb_complete_1=1;
    end
    always @(posedge axi_clk)begin
        if(!axi_cold_reset_n)begin
            active=0;reading=0;s_bvalid=0;s_rvalid=0;s_rlast=0;s_bid=0;s_bresp=0;s_rid=0;s_rresp=0;s_rdata=0;s_rdata_parity=0;
        end else for(integer lane=0;lane<2;lane++)begin
            if(s_bvalid[lane]&&s_bready[lane])s_bvalid[lane]<=0;
            if(s_awvalid[lane]&&s_awready[lane])begin
                active[lane]=1;address[lane]=s_awaddr[lane*34+:34];remaining[lane]=int'(s_awlen[lane*4+:4])+1;id[lane]=s_awid[lane*6+:6];
            end
            if(s_wvalid[lane]&&s_wready[lane])begin
                if(!active[lane]||s_wlast[lane]!=(remaining[lane]==1))$fatal(1,"STUB_WRITE_PROTOCOL");
                memory[address[lane]]=s_wdata[lane*256+:256];
                if(s_wstrb[lane*32+:32]!==32'hffffffff)$fatal(1,"STUB_ONLY_FULL_STROBES");
                address[lane]+=32;remaining[lane]--;
                if(remaining[lane]==0)begin active[lane]=0;s_bid[lane*6+:6]<=id[lane];s_bresp[lane*2+:2]<=0;s_bvalid[lane]<=1;end
            end
            if(s_arvalid[lane]&&s_arready[lane])begin
                reading[lane]=1;raddress[lane]=s_araddr[lane*34+:34];rremaining[lane]=int'(s_arlen[lane*4+:4])+1;read_id[lane]=s_arid[lane*6+:6];
            end
            if(s_rvalid[lane]&&s_rready[lane])begin rremaining[lane]--;raddress[lane]+=32;if(rremaining[lane]==0)reading[lane]=0;end
            if(!s_rvalid[lane]||s_rready[lane])begin
                s_rvalid[lane]<=reading[lane];
                if(reading[lane])begin
                    if(!memory.exists(raddress[lane]))$fatal(1,"STUB_READ_UNWRITTEN");
                    s_rdata[lane*256+:256]<=memory[raddress[lane]];s_rid[lane*6+:6]<=read_id[lane];s_rresp[lane*2+:2]<=0;s_rlast[lane]<=rremaining[lane]==1;
                end
            end
        end
    end
    always @(negedge axi_clk)for(integer lane=0;lane<2;lane++)begin
        s_awready[lane]=axi_cold_reset_n&&!active[lane]&&!s_bvalid[lane];
        s_wready[lane]=axi_cold_reset_n&&active[lane];
        s_arready[lane]=axi_cold_reset_n&&!reading[lane]&&!s_rvalid[lane];
    end
