`timescale 1ns/1ps
module tb_ntrpu_rope_loader_r1;
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic command_valid=0,command_ready,cancel_valid=0,cancel_ready,busy;
    logic [63:0] command_sequence=0,command_epoch=0,expected_epoch=0;
    logic [63:0] command_source_sequence=0,command_source_epoch=0;
    logic command_source_valid=0;
    logic [31:0] command_position=0,command_source_position=0,command_coefficient_position=0;
    logic [31:0] completion_position,completion_coefficient_position,saved_position,saved_coefficient_position;
    logic [63:0] command_q_address=0,command_q_base=0,command_q_bytes=0;
    logic [63:0] command_k_address=0,command_k_base=0,command_k_bytes=0;
    logic [63:0] command_coefficient_address=0,command_coefficient_base=0,command_coefficient_bytes=0;
    logic completion_valid,completion_ready=0;
    logic [3:0] completion_status;
    logic [63:0] completion_sequence,completion_epoch;
    logic [9:0] completion_beats;
    logic result_valid,qk_ce=0,coefficient_ce=0;
    logic [11:0] qk_address=0,coefficient_address=0;
    logic [31:0] qk_q,coefficient_q;
    logic qk_q_valid,coefficient_q_valid;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize,m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid,m_axi_arcache;
    logic m_axi_arlock,m_axi_arvalid,m_axi_arready=0;
    logic [511:0] m_axi_rdata=0;
    logic [1:0] m_axi_rresp=0;
    logic [3:0] m_axi_rid=0;
    logic m_axi_rlast=0,m_axi_rvalid=0,m_axi_rready;
    ntrpu_rope_loader_r1 #(.WATCHDOG_CYCLES(32)) dut(.*);
    int seed=1,random_state,job=0,scenario=0,cycle=0,cycles=0;
    int ar_count=0,r_count=0,ar_total=0,r_total=0,ram_total=0,completions=0,resets=0;
    int ar_stalls=0,r_stalls=0,revocations=0,beat=0,beats=0,burst_plane=0,plan_plane=0,plan_index=0;
    bit bus_active=0,ar_held=0,r_held=0,cmd_hs,cancel_hs,ar_hs,r_hs,done_hs;
    logic [63:0] burst_address,source_addresses[0:2],saved_sequence,saved_epoch;
    logic [71:0] saved_ar;
    logic [518:0] saved_r;

    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"FAIL RPELOAD job=%0d scenario=%0d cycle=%0d %s",job,scenario,cycle,why);
    endtask
    function automatic logic [31:0] memory_word(input logic [63:0] address);
        return 32'h3f000000 | (32'((address>>2)^(address>>33)^(address>>40)^64'(job*97))&32'h007fffff);
    endfunction
    function automatic integer beats_for(input int p);
        case(p) 0:return 128;1:return 32;2:return 4;default:return 0;endcase
    endfunction
    task automatic step;
        logic [63:0] a;
        logic last;
        int lane;
        @(negedge clk);
        m_axi_arready=!bus_active && $urandom_range(0,3)!=0;
        if((scenario==42 || scenario==46 || scenario==50) && cycles<120)m_axi_arready=0;
        if(!m_axi_rvalid && bus_active && $urandom_range(0,3)!=0 && !((scenario==47 || scenario==51) && cycles<120))begin
            a=burst_address+64'(beat*64);
            for(lane=0;lane<16;lane++)m_axi_rdata[lane*32+:32]=memory_word(a+64'(lane*4));
            m_axi_rresp=0;m_axi_rid=0;last=beat==beats-1;
            if(scenario>=30 && scenario<=32 && burst_plane==scenario-30 && a==source_addresses[burst_plane]+64'(beats_for(burst_plane)*64-64))m_axi_rresp=2;
            if(scenario>=33 && scenario<=35 && burst_plane==scenario-33 && a==source_addresses[burst_plane]+64'(beats_for(burst_plane)*64-64))m_axi_rid=1;
            if(scenario>=38 && scenario<=40 && burst_plane==scenario-38 && a==source_addresses[burst_plane]+64'(beats_for(burst_plane)*64-64))m_axi_rdata[511:480]=32'h7fc12345;
            if(scenario==36 && ar_count==1)last=beat==3;
            if(scenario==37 && ar_count==1)last=beat==beats;
            m_axi_rlast=last;m_axi_rvalid=1;
        end
        @(posedge clk);cycle++;cycles++;
        cmd_hs=rst_n&&command_valid&&command_ready;cancel_hs=rst_n&&cancel_valid&&cancel_ready;
        ar_hs=rst_n&&m_axi_arvalid&&m_axi_arready;r_hs=rst_n&&m_axi_rvalid&&m_axi_rready;done_hs=rst_n&&completion_valid&&completion_ready;
        if(rst_n && ar_held)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"Held AR changed");
        if(rst_n && r_held)require(m_axi_rvalid && {m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast}===saved_r,"BFM held R changed");
        ar_held=rst_n&&m_axi_arvalid&&!m_axi_arready;r_held=rst_n&&m_axi_rvalid&&!m_axi_rready;
        saved_ar={m_axi_araddr,m_axi_arlen};saved_r={m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast};
        ar_stalls+=int'(ar_held);r_stalls+=int'(r_held);
        if(!completion_valid)require(!result_valid && !qk_q_valid && !coefficient_q_valid &&
            qk_q==0 && coefficient_q==0,"Private data exposed before completion");
        if(ar_hs)begin
            require(!bus_active && plan_plane<3,"Single outstanding AXI");
            require(m_axi_araddr==source_addresses[plan_plane]+64'(plan_index*64),"AXI plane/address order");
            require(m_axi_arlen+1==((beats_for(plan_plane)-plan_index)<(64-int'(m_axi_araddr[11:6])) ? beats_for(plan_plane)-plan_index:64-int'(m_axi_araddr[11:6])),"Burst/page extent");
            require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AXI attributes");
            $display("RPELOAD_AR cycle=%0d job=%0d plane=%0d address=%016h beats=%0d",cycle,job,plan_plane,m_axi_araddr,m_axi_arlen+1);
        end
        if(r_hs)$display("RPELOAD_R cycle=%0d job=%0d address=%016h response=%0d rid=%0d last=%0d data=%0128h",cycle,job,burst_address+64'(beat*64),m_axi_rresp,m_axi_rid,m_axi_rlast,m_axi_rdata);
        #1;
        if(ar_hs)begin
            bus_active=1;burst_address=saved_ar[71:8];beats=int'(saved_ar[7:0])+1;beat=0;burst_plane=plan_plane;
            ar_count++;ar_total++;plan_index+=beats;
            if(plan_index==beats_for(plan_plane))begin plan_plane++;plan_index=0;end
        end
        if(r_hs)begin
            r_count++;r_total++;beat++;
            if(m_axi_rlast)bus_active=0;
            m_axi_rvalid=0;
        end
        if(cmd_hs)command_valid=0;
        if(cancel_hs)cancel_valid=0;
        require(cycles<18000,"Bounded test timeout");
    endtask

    task automatic run_case(input int mode,want);
        int offset,quiet,index,revoked_at;
        bit sent_cancel,changed_epoch,reset_case;
        logic [205:0] held_completion;
        begin : case_body
        scenario=mode;job++;cycles=0;ar_count=0;r_count=0;plan_plane=0;plan_index=0;
        sent_cancel=0;changed_epoch=0;reset_case=mode>=50 && mode<=52;revoked_at=0;
        require(command_ready && !bus_active && !m_axi_rvalid,"Previous request drained");
        offset=(mode==1 || mode==36 || mode==37) ? 0:(mode==2 ? 128:4032);
        command_q_base=64'h100000000;command_k_base=64'h8000000000;command_coefficient_base=64'hc000000000;
        command_q_bytes=16384;command_k_bytes=16384;command_coefficient_bytes=16384;
        command_q_address=command_q_base+64'(offset);
        command_k_address=command_k_base+64'(offset);
        command_coefficient_address=command_coefficient_base+64'(offset);
        command_sequence=64'h1234567800000000+64'(job);command_epoch=64'hfedcba9800000000+64'(job);
        command_source_sequence=command_sequence;command_source_epoch=command_epoch;expected_epoch=command_epoch;command_source_valid=1;
        saved_position=mode==56 ? 131072:mode==59 ? 131071:mode==0 ? 0:32'(job*2017);
        command_position=saved_position;command_source_position=saved_position;command_coefficient_position=saved_position;
        if(mode==57)command_source_position=saved_position+1;
        if(mode==58)command_coefficient_position=saved_position+1;
        saved_coefficient_position=command_coefficient_position;
        case(mode)
            10:command_q_address++;11:command_k_address++;12:command_coefficient_address++;
            13:command_q_bytes=64'(offset)+8191;14:command_k_bytes=64'(offset)+2047;15:command_coefficient_bytes=64'(offset)+255;
            16:command_q_base=command_q_address+64;17:command_k_base=command_k_address+64;18:command_coefficient_base=command_coefficient_address+64;
            19:begin command_q_address=64'hfffffffffffff000;command_q_base=command_q_address;end
            20:begin command_k_address=64'hfffffffffffff000;command_k_base=command_k_address;end
            21:begin command_coefficient_address=64'hfffffffffffff000;command_coefficient_base=command_coefficient_address;end
            22:command_source_sequence^=64'h0000000100000000;
            23:command_source_epoch^=64'h0000000100000000;
            24:expected_epoch^=64'h0000000100000000;
            25:command_source_valid=0;
            53:command_coefficient_bytes=0;
            54:begin command_k_address=64'hffffffffffffffc0;command_k_base=0;command_k_bytes=64'hffffffffffffffff;end
            61:command_q_bytes=0;
            default:;
        endcase
        source_addresses[0]=command_q_address;source_addresses[1]=command_k_address;source_addresses[2]=command_coefficient_address;
        saved_sequence=command_sequence;saved_epoch=command_epoch;
        $display("RPELOAD_BEGIN job=%0d scenario=%0d sequence=%016h epoch=%016h expected=%016h source_sequence=%016h source_epoch=%016h source_valid=%0d ra=%016h rb=%016h rz=%016h ba=%016h bb=%016h bz=%016h wa=%016h wb=%016h wz=%016h position=%0d source_position=%0d coefficient_position=%0d",
            job,mode,command_sequence,command_epoch,expected_epoch,command_source_sequence,command_source_epoch,command_source_valid,
            command_q_address,command_q_base,command_q_bytes,command_k_address,command_k_base,command_k_bytes,
            command_coefficient_address,command_coefficient_base,command_coefficient_bytes,command_position,command_source_position,command_coefficient_position);
        completion_ready=0;cancel_valid=0;command_valid=1;
        qk_ce=1;coefficient_ce=1;
        while(!completion_valid)begin
            if(!sent_cancel && cancel_ready && ((mode==41 && cycles==1) ||
                (mode==42 && m_axi_arvalid && cycles>=10) || (mode==43 && r_count>=10) || (mode==44 && r_count>=1 && cycles>=10)))begin
                cancel_valid=1;sent_cancel=1;$display("RPELOAD_CANCEL cycle=%0d job=%0d",cycle+1,job);
            end
            if(mode==45 && !changed_epoch && r_count>=130)begin
                expected_epoch^=64'h0000000100000000;changed_epoch=1;revoked_at=cycles;
                $display("RPELOAD_EPOCH cycle=%0d job=%0d expected=%016h",cycle+1,job,expected_epoch);
            end
            if(mode==45 && changed_epoch && cycles==revoked_at+3)expected_epoch=saved_epoch;
            if((mode==50 && m_axi_arvalid && cycles>=10) || (mode==51 && bus_active && cycles>=10))begin
                rst_n=0;bus_active=0;m_axi_rvalid=0;ar_held=0;r_held=0;command_valid=0;
                step();step();rst_n=1;step();resets++;
                require(command_ready && !completion_valid && !result_valid,"Coordinated reset");
                $display("RPELOAD_RESET cycle=%0d job=%0d ar=%0d r=%0d",cycle,job,ar_count,r_count);
                disable case_body;
            end
            step();
            if(cycles==2)begin
                // Command payload can change after acceptance; captured values must win.
                command_sequence=0;command_epoch=0;command_q_address=0;command_k_address=0;command_coefficient_address=0;command_position=32'hffffffff;command_source_position=0;command_coefficient_position=0;
            end
        end
        require(completion_status==want && completion_sequence==saved_sequence && completion_epoch==saved_epoch && completion_position==saved_position && completion_coefficient_position==saved_coefficient_position,"Completion identity/status");
        require(!bus_active && !m_axi_rvalid,"Completion before AXI drain");
        require(result_valid==(want==0),"Result publication");
        if(want==0)require(completion_beats==164 && r_count==164,"Complete Q/K/phase vector");
        if((mode>=10 && mode<=25) || mode==41 || mode==53 || mode==54 || (mode>=56 && mode<=58) || mode==61)require(ar_count==0 && r_count==0,"Rejected command issued AXI");
        held_completion={completion_status,completion_sequence,completion_epoch,completion_beats,completion_position,completion_coefficient_position};
        $display("RPELOAD_COMPLETE cycle=%0d job=%0d status=%0d sequence=%016h epoch=%016h valid=%0d beats=%0d ar=%0d r=%0d position=%0d coefficient_position=%0d",cycle,job,completion_status,completion_sequence,completion_epoch,result_valid,completion_beats,ar_count,r_count,completion_position,completion_coefficient_position);
        completions++;
        if(want==0)begin
            for(index=0;index<2560;index++)begin
                qk_ce=1;coefficient_ce=index<64;
                qk_address=12'((index*17+31)%2560);coefficient_address=12'((index*7+63)%64);
                step();
                require(result_valid && qk_q_valid && coefficient_q_valid==(index<64),"RAM read valid");
                require(qk_q===memory_word(source_addresses[qk_address<2048 ? 0:1]+64'(qk_address<2048 ? qk_address:qk_address-2048)*4),"Q/K RAM concatenation");
                if(index<64)require(coefficient_q===memory_word(source_addresses[2]+64'(coefficient_address)*4),"Coefficient RAM");
                else require(coefficient_q===0,"Inactive coefficient port");
                $display("RPELOAD_RAM cycle=%0d job=%0d qa=%0d q=%08h ca=%0d c=%08h cv=%0d",cycle,job,qk_address,qk_q,coefficient_address,coefficient_q,coefficient_q_valid);
                ram_total+=1+int'(index<64);
            end
            qk_ce=0;coefficient_ce=0;step();
            require(!qk_q_valid && !coefficient_q_valid && qk_q===0 && coefficient_q===0,"Both RAM CE bubbles");
            if(mode==60)begin
                for(int trial=0;trial<2;trial++)begin
                    qk_ce=1;coefficient_ce=1;qk_address=trial==0 ? 2560:4095;coefficient_address=trial==0 ? 64:4095;
                    step();require(result_valid && !qk_q_valid && !coefficient_q_valid && qk_q===0 && coefficient_q===0,"Invalid RAM address");
                    $display("RPELOAD_INVALID_READ cycle=%0d job=%0d qa=%0d ca=%0d",cycle,job,qk_address,coefficient_address);
                end
                qk_address=2559;coefficient_address=63;step();
                require(qk_q_valid && coefficient_q_valid && qk_q===memory_word(source_addresses[1]+2044) && coefficient_q===memory_word(source_addresses[2]+252),"Read recovery at RAM ends");
                qk_ce=0;coefficient_ce=0;step();
            end
        end
        if(mode==48 || mode==49)begin
            if(mode==48)cancel_valid=1;else expected_epoch^=64'h0000000100000000;
            step();expected_epoch=saved_epoch;step();revocations++;
            require(!result_valid && !qk_q_valid && !coefficient_q_valid,"Sticky held-result revocation");
            $display("RPELOAD_REVOKE cycle=%0d job=%0d mode=%0d",cycle,job,mode);
        end
        for(quiet=0;quiet<7;quiet++)begin
            step();require(completion_valid && {completion_status,completion_sequence,completion_epoch,completion_beats,completion_position,completion_coefficient_position}===held_completion,"Held completion changed");
            if(want!=0 || mode==48 || mode==49)require(!result_valid && qk_q==0 && coefficient_q==0,"Rejected data visible");
            require(!command_ready && !m_axi_arvalid,"Owned RAM reused before ACK");
        end
        if(mode==52)begin
            rst_n=0;step();step();rst_n=1;step();resets++;
            require(command_ready && !completion_valid && !result_valid,"Reset held result");
            $display("RPELOAD_RESET cycle=%0d job=%0d ar=%0d r=%0d",cycle,job,ar_count,r_count);
            disable case_body;
        end
        completion_ready=1;step();completion_ready=0;step();
        require(command_ready && !completion_valid && !result_valid && !qk_q_valid && !coefficient_q_valid,"Release to IDLE");
        $display("RPELOAD_RELEASE cycle=%0d job=%0d",cycle,job);
        end
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        repeat(3)@(negedge clk);rst_n=1;step();
        for(int mode=0;mode<3;mode++)run_case(mode,0);
        for(int mode=10;mode<=21;mode++)run_case(mode,1);
        for(int mode=22;mode<=24;mode++)run_case(mode,2);
        run_case(25,11);
        for(int mode=30;mode<=37;mode++)run_case(mode,3);
        for(int mode=38;mode<=40;mode++)run_case(mode,4);
        for(int mode=41;mode<=44;mode++)run_case(mode,7);
        run_case(45,2);run_case(46,8);run_case(47,8);
        run_case(48,0);run_case(49,0);run_case(50,0);run_case(51,0);run_case(52,0);
        run_case(53,1);run_case(54,1);run_case(55,0);
        run_case(56,1);run_case(57,2);run_case(58,2);run_case(59,0);run_case(60,0);run_case(61,1);
        require(ar_stalls>0 && r_stalls>0,"Backpressure coverage");
        $display("RPELOAD_PASS seed=%0d jobs=%0d completions=%0d resets=%0d revocations=%0d ar=%0d r=%0d ram=%0d ar_stalls=%0d r_stalls=%0d cycles=%0d",
            seed,job,completions,resets,revocations,ar_total,r_total,ram_total,ar_stalls,r_stalls,cycle);
        $finish;
    end
endmodule
