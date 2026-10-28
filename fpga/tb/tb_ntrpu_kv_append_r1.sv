`timescale 1ns/1ps
module tb_ntrpu_kv_append_r1 #(parameter integer WATCHDOG_CYCLES=32);
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic maintenance_allowed=1,bind_valid=0,bind_ready;
    logic [63:0] bind_cache_base=0,bind_region_base=0,bind_region_bytes=0;
    logic [31:0] bind_capacity=0;
    logic [63:0] bind_sequence=0,bind_epoch=0,expected_epoch=0;
    logic cache_valid;
    logic [63:0] cache_base,cache_region_base,cache_region_bytes,cache_sequence,cache_epoch;
    logic [31:0] cache_capacity,committed_tokens;
    logic [3:0] inspect_layer=0;
    logic cmd_valid=0,cmd_ready,prepared;
    logic [31:0] cmd_layer=0,cmd_position=0;
    logic [63:0] cmd_sequence=0,cmd_epoch=0;
    logic data_valid=0,data_ready,data_last=0;
    logic [511:0] data_bits=0;
    logic [5:0] data_index=0;
    logic [63:0] data_sequence=0,data_epoch=0;
    logic commit_valid=0,commit_ready;
    logic [3:0] commit_status=0;
    logic [63:0] commit_sequence=0,commit_epoch=0;
    logic cancel_valid=0,cancel_ready,done_valid,done_ready=0,done_bind;
    logic [3:0] done_status;
    logic [12:0] done_bytes;
    logic [31:0] done_committed_tokens;
    logic [63:0] done_sequence,done_epoch;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize,m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid,m_axi_awcache;
    logic m_axi_awlock,m_axi_awvalid,m_axi_awready=0;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast,m_axi_wvalid,m_axi_wready=0;
    logic [3:0] m_axi_bid=0;
    logic [1:0] m_axi_bresp=0;
    logic m_axi_bvalid=0,m_axi_bready;
    ntrpu_kv_append_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut(.*);
    localparam logic [63:0] BASE=64'ha000000000;
    int seed=1,random_state,op=0,binds=0,jobs=0,resets=0,successes=0;
    int aw_stalls=0,w_stalls=0,done_stalls=0,before_aw=0;
    int prefix[0:15];
    bit valid_model=0;
    logic [63:0] base_model,seq_model,epoch_model,last_epoch_model=0,next_epoch=64'hfedcba9800000000;
    int cap_model;
    function automatic logic [511:0] pattern(input int id,index,scenario);
        logic [31:0] word;
        for(int lane=0;lane<16;lane++) begin
            word=32'h3f000000 | (32'(id*17+index*19+lane*23)&32'h007fffff);
            if(scenario==32) case(lane%6)
                0:word=0;1:word=32'h80000000;2:word=1;3:word=32'h80000001;
                4:word=32'h7f7fffff;5:word=32'hff7fffff;
            endcase
            pattern[lane*32+:32]=word;
        end
    endfunction
    function automatic logic [63:0] location(input int lay,pos,index);
        return base_model+((64'(lay)*16+64'(index/4))*64'(cap_model)+64'(pos))*256+64'(index%4)*64;
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"FAIL KV_APPEND op=%0d: %s",op,why);
    endtask
    task automatic check_prefixes;
        for(int lay=0;lay<16;lay++) begin
            @(negedge clk);inspect_layer=4'(lay);#1;
            require(committed_tokens==(valid_model ? prefix[lay]:0),"Owned prefix table");
            $display("NTRPU_KVA_PREFIX id=%0d layer=%0d committed=%0d valid=%0d",op,lay,committed_tokens,cache_valid);
        end
        require(cache_valid==valid_model,"Lease validity");
    endtask
    task automatic bind_case(input int cap,offset,scenario,want_status);
        int cycles,wait_done;
        bit accepted,finished;
        logic [63:0] seq,epoch;
        op++;binds++;cycles=0;wait_done=0;accepted=0;finished=0;
        @(negedge clk);require(bind_ready && cmd_ready,"Idle before bind");
        bind_cache_base=BASE+64'(offset);bind_region_base=BASE;bind_region_bytes=64'(cap)*65536+8192;
        bind_capacity=32'(cap);bind_sequence=64'h1234567800000000+64'(op);next_epoch++;bind_epoch=next_epoch;
        expected_epoch=bind_epoch;
        case(scenario)
            1:bind_capacity=0;
            2:bind_capacity=131073;
            3:bind_cache_base++;
            4:bind_region_bytes=64'(offset)+64'(cap)*65536-1;
            5:bind_region_base=bind_cache_base+64;
            6:begin bind_cache_base=64'hfffffffffffff000;bind_region_base=bind_cache_base;bind_region_bytes=8192;end
            7:begin bind_cache_base=64'hfffffffffffff000;bind_region_base=0;bind_region_bytes=64'hffffffffffffffff;end
            8:bind_sequence=0;
            9:begin bind_epoch=0;expected_epoch=0;end
            10:begin bind_epoch=last_epoch_model;expected_epoch=bind_epoch;end
            11:expected_epoch++;
        endcase
        seq=bind_sequence;epoch=bind_epoch;bind_valid=1;
        $display("NTRPU_KVA_BIND id=%0d scenario=%0d capacity=%0d cache=%016h region=%016h bytes=%016h sequence=%016h epoch=%016h expected_epoch=%016h previous_epoch=%016h",op,scenario,bind_capacity,bind_cache_base,bind_region_base,bind_region_bytes,seq,epoch,expected_epoch,last_epoch_model);
        while(!finished) begin
            maintenance_allowed=scenario!=12 || cycles>=10;done_ready=wait_done>=5;
            @(posedge clk);cycles++;
            require(cycles<100,"Bind timeout");require(!m_axi_awvalid && !m_axi_wvalid,"Bind emitted writes");
            if(!maintenance_allowed) require(!bind_ready,"Rebind without quiescence");
            if(bind_valid && bind_ready) begin
                $display("NTRPU_KVA_BIND_ACCEPT id=%0d cycle=%0d maintenance=%0d",op,cycles,maintenance_allowed);
                accepted=1;valid_model=0;for(int i=0;i<16;i++)prefix[i]=0;
                base_model=bind_cache_base;cap_model=int'(bind_capacity);seq_model=seq;epoch_model=epoch;
            end
            if(done_valid) begin
                if(wait_done==0)$display("NTRPU_KVA_BIND_RESULT id=%0d status=%0d valid=%0d bytes=%0d committed=%0d sequence=%016h epoch=%016h",op,done_status,cache_valid,done_bytes,done_committed_tokens,done_sequence,done_epoch);
                require(done_bind && done_status==4'(want_status) && done_bytes==0 && done_committed_tokens==0,"Bind completion");
                require(done_sequence==seq && done_epoch==epoch,"Bind completion identity");
                wait_done++;
                if(want_status==0)begin valid_model=1;last_epoch_model=epoch;end
                if(done_ready)finished=1;
            end
            @(negedge clk);
            if(accepted) begin bind_valid=0;bind_cache_base=0;bind_region_base=0;bind_region_bytes=0;bind_capacity=0;bind_sequence=0;bind_epoch=0;end
        end
        done_ready=0;maintenance_allowed=1;expected_epoch=epoch_model;
        $display("NTRPU_KVA_BIND_DONE id=%0d status=%0d valid=%0d cycles=%0d",op,want_status,valid_model,cycles);
        check_prefixes();
    endtask
    task automatic append_case(input int lay,pos,scenario,want_status);
        int cycles,inputs,writes,bursts,responses,planned,aw_beats,w_burst,done_wait,target_inputs,old_prefix,commit_cycle;
        bit accepted,finished,cancelled,committed,aw_pending,reset_case,reset_now;
        bit aw_hold,w_hold,done_hold,aw_hs,w_hs,b_hs,data_hs,cmd_hs,commit_hs,cancel_hs;
        logic [88:0] saved_aw;
        logic [576:0] saved_w;
        logic [177:0] saved_done;
        logic [63:0] seq,epoch,want_address;
        int want_burst;
        op++;jobs++;cycles=0;inputs=0;writes=0;bursts=0;responses=0;planned=0;aw_beats=0;w_burst=0;done_wait=0;commit_cycle=-1;
        accepted=0;finished=0;cancelled=0;committed=0;aw_pending=0;aw_hold=0;w_hold=0;done_hold=0;
        reset_case=scenario>=33 && scenario<=37;reset_now=0;old_prefix=lay<16 ? prefix[lay]:0;
        target_inputs=scenario==13 ? 63:(scenario==14 ? 65:64);
        @(negedge clk);require(cmd_ready,"Idle before append");inspect_layer=4'(lay);
        cmd_layer=32'(lay);cmd_position=32'(pos);cmd_sequence=seq_model;cmd_epoch=epoch_model;expected_epoch=epoch_model;
        case(scenario)
            1:cmd_layer=16;2:cmd_position++;3:cmd_sequence^=64'h0000010000000000;
            4:cmd_epoch^=64'h0000010000000000;5:expected_epoch++;
        endcase
        old_prefix=cmd_layer<16 ? prefix[cmd_layer]:0;
        seq=cmd_sequence;epoch=cmd_epoch;cmd_valid=1;data_valid=0;commit_valid=0;cancel_valid=0;m_axi_bvalid=0;
        $display("NTRPU_KVA_BEGIN id=%0d scenario=%0d layer=%0d position=%0d prefix=%0d valid=%0d capacity=%0d cache=%016h sequence=%016h epoch=%016h expected_epoch=%016h",op,scenario,cmd_layer,cmd_position,old_prefix,valid_model,cap_model,base_model,seq,epoch,expected_epoch);
        while(!finished) begin
            m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if((scenario==23 || scenario==34) && cycles<200)m_axi_awready=0;
            if(scenario==27 && cycles<WATCHDOG_CYCLES+250)m_axi_awready=0;
            if(scenario==28 && cycles<WATCHDOG_CYCLES+250)m_axi_wready=0;
            done_ready=done_wait>=6 && $urandom_range(0,2)!=0;
            if(scenario==37)done_ready=0;
            data_valid=prepared && inputs<target_inputs && $urandom_range(0,3)!=0;
            if(scenario==26 && cycles<WATCHDOG_CYCLES+32)data_valid=0;
            data_bits=pattern(op,inputs,scenario);data_index=6'(inputs);data_last=inputs==63;data_sequence=seq;data_epoch=epoch;
            if(scenario==6 && inputs==5)data_index=4;
            if(scenario==7 && inputs==6)data_sequence^=64'h0000010000000000;
            if(scenario==8 && inputs==7)data_epoch^=64'h0000010000000000;
            if(scenario==9 && inputs==8)data_last=1;
            if(scenario==10 && inputs==63)data_last=0;
            if((scenario==11 && inputs==0) || (scenario==30 && inputs==5))data_bits[31:0]=32'h7fc12345;
            if(scenario==12 && inputs==63)data_bits[511:480]=32'h7f800000;
            commit_valid=prepared && !committed && (inputs==target_inputs || (data_valid && inputs+1==target_inputs));
            if(scenario==39 && inputs==target_inputs && cycles<commit_cycle+5)commit_valid=0;
            if(scenario==39 && inputs<target_inputs)commit_valid=0;
            commit_status=scenario==17 || scenario==30 ? 4'd6:4'd0;commit_sequence=seq;commit_epoch=epoch;
            if(scenario==15)commit_sequence^=64'h0000010000000000;
            if(scenario==16)commit_epoch^=64'h0000010000000000;
            if(!m_axi_bvalid && aw_pending && w_burst==aw_beats && $urandom_range(0,3)!=0 &&
                !((scenario==29 || scenario==36) && cycles<WATCHDOG_CYCLES+300)) begin
                m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;
                if((scenario==18 && responses==0) || (scenario==19 && planned==64) || scenario==38)m_axi_bresp=2;
                if(scenario==20 && responses==7)m_axi_bid=1;
            end
            if(!cancelled && cancel_ready && ((scenario==21 && inputs>=8) || (scenario==22 && committed && writes==0) ||
                (scenario==23 && m_axi_awvalid) || (scenario==24 && writes>=8) ||
                (scenario==25 && m_axi_bvalid && planned==64) || (scenario==31 && commit_valid) ||
                (scenario==38 && m_axi_bvalid)))cancel_valid=1;
            @(posedge clk);cycles++;
            require(cycles<2*WATCHDOG_CYCLES+5000,"Append timeout");
            aw_hs=m_axi_awvalid && m_axi_awready;w_hs=m_axi_wvalid && m_axi_wready;b_hs=m_axi_bvalid && m_axi_bready;
            data_hs=data_valid && data_ready;cmd_hs=cmd_valid && cmd_ready;commit_hs=commit_valid && commit_ready;cancel_hs=cancel_valid && cancel_ready;
            if(aw_hold)require(m_axi_awvalid && saved_aw=={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock},"AW changed while stalled");
            if(w_hold)require(m_axi_wvalid && saved_w=={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"W changed while stalled");
            if(done_hold)require(done_valid && saved_done=={done_bind,done_status,done_bytes,done_committed_tokens,done_sequence,done_epoch},"Completion changed while stalled");
            aw_hold=m_axi_awvalid && !m_axi_awready;w_hold=m_axi_wvalid && !m_axi_wready;done_hold=done_valid && !done_ready;
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};saved_done={done_bind,done_status,done_bytes,done_committed_tokens,done_sequence,done_epoch};
            aw_stalls+=int'(aw_hold);w_stalls+=int'(w_hold);done_stalls+=int'(done_hold);
            if(cmd_hs)accepted=1;
            if(data_hs) begin
                $display("NTRPU_KVA_DATA id=%0d index=%0d last=%0d sequence=%016h epoch=%016h data=%0128h",op,data_index,data_last,data_sequence,data_epoch,data_bits);
                inputs++;if(inputs==target_inputs)commit_cycle=cycles;
            end
            if(commit_hs)begin committed=1;$display("NTRPU_KVA_COMMIT id=%0d status=%0d sequence=%016h epoch=%016h inputs=%0d",op,commit_status,commit_sequence,commit_epoch,inputs);end
            if(cancel_hs)begin cancelled=1;$display("NTRPU_KVA_CANCEL id=%0d writes=%0d responses=%0d",op,writes,responses);end
            if(aw_hs)begin
                require(committed && !aw_pending,"AW before commit or duplicate outstanding address");
                want_address=location(lay,pos,planned);want_burst=4-planned%4;
                if(want_burst>64-int'((want_address>>6)&63))want_burst=64-int'((want_address>>6)&63);
                require(m_axi_awaddr==want_address && int'(m_axi_awlen)+1==want_burst,"Full address and 4 KiB burst plan");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                aw_pending=1;aw_beats=want_burst;planned+=want_burst;bursts++;
                $display("NTRPU_KVA_AW id=%0d address=%016h beats=%0d",op,m_axi_awaddr,want_burst);
            end
            if(w_hs)begin
                require(committed && writes<64,"W before commit or beyond token");
                require(m_axi_wdata==pattern(op,writes,scenario) && m_axi_wstrb==64'hffffffffffffffff,"Staged physical bytes");
                want_address=location(lay,pos,writes);want_burst=4-writes%4;
                require(m_axi_wlast==(want_burst==1 || ((want_address>>6)&63)==63),"WLAST/page boundary");
                if(!aw_pending)before_aw++;
                $display("NTRPU_KVA_W id=%0d index=%0d last=%0d data=%0128h",op,writes,m_axi_wlast,m_axi_wdata);
                writes++;w_burst++;
            end
            if(b_hs)begin
                require(aw_pending && w_burst==aw_beats,"B before complete AW/W");
                $display("NTRPU_KVA_B id=%0d ordinal=%0d resp=%0d bid=%0d",op,responses,m_axi_bresp,m_axi_bid);
                responses++;aw_pending=0;w_burst=0;
                if(writes==64 && want_status==0)prefix[lay]=pos+1;
            end
            if(done_valid)begin
                if(done_wait==0)$display("NTRPU_KVA_DONE id=%0d status=%0d bytes=%0d committed=%0d sequence=%016h epoch=%016h",op,done_status,done_bytes,done_committed_tokens,done_sequence,done_epoch);
                require(!done_bind && done_status==4'(want_status),$sformatf("Completion status %0d expected %0d",done_status,want_status));
                require(done_bytes==writes*64 && done_sequence==seq && done_epoch==epoch,"Completion counts/identity");
                require(done_committed_tokens==32'(want_status==0 ? pos+1:old_prefix),"Completion prefix");
                require(!aw_pending && responses==bursts,"Completion before B drained");done_wait++;
                if(want_status==0)require(writes==64 && inputs==64,"Incomplete successful token");
                else if(want_status!=3 && want_status!=7 && want_status!=8)require(writes==0 && bursts==0,"Write despite invalid input");
                if(done_ready)finished=1;
            end
            #1;
            if(b_hs && committed_tokens!=32'(old_prefix) && lay<16)
                $display("NTRPU_KVA_PUBLICATION id=%0d committed=%0d responses=%0d writes=%0d",op,committed_tokens,responses,writes);
            if(lay<16)require(committed_tokens==32'(prefix[lay]),"Premature/absent publication");
            require(cache_valid==valid_model,"Append changed lease validity");
            reset_now=(scenario==33 && inputs>=5) || (scenario==34 && m_axi_awvalid) ||
                (scenario==35 && writes>=2) || (scenario==36 && writes>=4 && responses==0) || (scenario==37 && done_valid);
            @(negedge clk);
            if(accepted)begin cmd_valid=0;cmd_layer=0;cmd_position=0;cmd_sequence=0;cmd_epoch=0;end
            if(b_hs)m_axi_bvalid=0;
            if(cancel_hs)cancel_valid=0;
            if(reset_now)begin
                $display("NTRPU_KVA_RESET id=%0d inputs=%0d writes=%0d bursts=%0d responses=%0d prefix=%0d",op,inputs,writes,bursts,responses,prefix[lay]);
                rst_n=0;cmd_valid=0;data_valid=0;commit_valid=0;cancel_valid=0;m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;
                repeat(3)@(negedge clk);
                require(!done_valid && !m_axi_awvalid && !m_axi_wvalid && !cache_valid,"Reset left visible transaction");
                rst_n=1;valid_model=0;last_epoch_model=0;for(int i=0;i<16;i++)prefix[i]=0;
                resets++;finished=1;
            end
        end
        cmd_valid=0;data_valid=0;commit_valid=0;cancel_valid=0;m_axi_bvalid=0;done_ready=0;m_axi_awready=0;m_axi_wready=0;
        if(!reset_case && want_status==0)successes++;
        $display("NTRPU_KVA_CASE id=%0d scenario=%0d status=%0d reset=%0d inputs=%0d writes=%0d bursts=%0d responses=%0d published=%0d cycles=%0d",op,scenario,want_status,reset_case,inputs,writes,bursts,responses,reset_case ? 0:prefix[lay<16 ? lay:0],cycles);
        check_prefixes();
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        $display("NTRPU_KVA_SEED seed=%0d watchdog=%0d",seed,WATCHDOG_CYCLES);
        for(int i=0;i<16;i++)prefix[i]=0;
        repeat(3)@(negedge clk);rst_n=1;
        bind_case(131072,4032,12,0);
        for(int lay=0;lay<16;lay++)append_case(lay,0,0,0);
        append_case(0,1,39,0);
        for(int scenario=1;scenario<=32;scenario++)begin
            if(scenario<=2)append_case(1,1,scenario,1);
            else if(scenario<=5)append_case(1,1,scenario,2);
            else if(scenario<=14)append_case(1,1,scenario,4);
            else if(scenario<=16)append_case(1,1,scenario,2);
            else if(scenario==17)append_case(1,1,scenario,6);
            else if(scenario<=20)append_case(1,1,scenario,3);
            else if(scenario<=25)append_case(1,1,scenario,7);
            else if(scenario<=29)append_case(1,1,scenario,8);
            else if(scenario==30)append_case(1,1,scenario,4);
            else if(scenario==31)append_case(1,1,scenario,7);
            else append_case(1,1,scenario,0);
        end
        append_case(1,2,38,7);append_case(1,2,0,0);
        for(int scenario=33;scenario<=37;scenario++)begin
            append_case(2,scenario==33 ? 1:0,scenario,scenario==37 ? 0:7);
            append_case(2,0,0,2);
            bind_case(17,64,0,0);
        end
        bind_case(3,0,0,0);
        for(int pos=0;pos<3;pos++)append_case(15,pos,0,0);
        append_case(15,3,0,1);append_case(15,1,0,1);
        for(int scenario=1;scenario<=11;scenario++)begin
            bind_case(17,64,scenario,scenario>=8 ? 2:1);
            append_case(0,0,0,2);
            bind_case(17,64,0,0);
        end
        append_case(15,0,0,0);
        require(aw_stalls>0 && w_stalls>0 && done_stalls>0 && before_aw>0,"Backpressure and independent AW/W not exercised");
        $display("NTRPU_KVA_PASS binds=%0d jobs=%0d resets=%0d successes=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d w_before_aw=%0d",binds,jobs,resets,successes,aw_stalls,w_stalls,done_stalls,before_aw);
        $finish;
    end
endmodule
