`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_residual_store_r2;
    localparam integer WD=64;
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic command_valid=0,command_ready,cancel_valid=0,cancel_ready,busy;
    logic [63:0] command_sequence=0,command_epoch=0,expected_epoch=0;
    logic [63:0] command_residual_address=0,command_residual_base=0,command_residual_bytes=0;
    logic [63:0] command_ffn_address=0,command_ffn_base=0,command_ffn_bytes=0;
    logic source_completion_valid=0,source_completion_ready,source_result_valid=0;
    logic [3:0] source_completion_status=0;
    logic [63:0] source_sequence=0,source_epoch=0;
    logic source_residual_ce,source_ffn_ce=0,source_residual_q_valid=0,source_ffn_q_valid=0;
    logic [10:0] source_residual_address;
    logic [8:0] source_ffn_address=0;
    logic [31:0] source_residual_q=0;
    logic [63:0] source_ffn_q=0;
    logic completion_valid,completion_ready=0,result_valid;
    logic [3:0] completion_status;
    logic [63:0] completion_sequence,completion_epoch,result_residual_address,result_ffn_address=0;
    logic [7:0] completion_residual_beats;
    logic [6:0] completion_ffn_beats=0;
    logic [13:0] result_residual_bytes;
    logic [12:0] result_ffn_bytes=0;
    logic [63:0] m_axi_awaddr,m_axi_wstrb;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize,m_axi_awprot;
    logic [1:0] m_axi_awburst,m_axi_bresp=0;
    logic [3:0] m_axi_awid,m_axi_awcache,m_axi_bid=0;
    logic m_axi_awlock,m_axi_awvalid,m_axi_awready=0,m_axi_wlast,m_axi_wvalid,m_axi_wready=0;
    logic m_axi_bvalid=0,m_axi_bready;
    logic [511:0] m_axi_wdata;
    ntrpu_residual_store_r1 #(.WATCHDOG_CYCLES(WD)) dut(.*);
    logic [31:0] residual_words[0:8191];
    // Legacy q counters/descriptor fields below are fixed at zero.
    logic [511:0] memory[0:127],pending_write;
    logic [63:0] pending_address;
    string fixture;
    integer seed=1,random_state=1,cycles=0,jobs=0,mode_now=0,sample_now=0,resets=0,publics=0,source_acks=0;
    integer rreads=0,qreads=0,aws=0,ws=0,bs=0,good_r=0,good_q=0,launch_cycle=0,last_b_cycle=0;
    integer total_rr=0,total_qr=0,total_aw=0,total_w=0,total_b=0;
    integer aw_stalls=0,w_stalls=0,held=0,aw_first=0,w_first=0;
    logic [63:0] sequence_model,epoch_model,residual_model,ffn_model;
    logic [88:0] saved_aw;
    logic [576:0] saved_w;
    logic hold_aw=0,hold_w=0;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"RSTORE_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,mode_now,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [31:0] residual_word(input int index);
        logic [31:0] value;
        value=residual_words[sample_now*2048+index];
        if((mode_now==23 && index==24) || (mode_now==48 && index==2047))value=32'h7f800000;
        return value;
    endfunction
    function automatic logic [511:0] record_word(input int index);
        logic [511:0] value;
        for(integer i=0;i<16;i++)value[i*32+:32]=residual_words[sample_now*2048+index*16+i];
        return value;
    endfunction
    function automatic logic [63:0] record_address(input int index);
        return residual_model+64'(index*64);
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;end
        else begin
            m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if((m_axi_awvalid || m_axi_wvalid) && launch_cycle==0)launch_cycle=cycles;
            if(bs==0 && mode_now==32 && (launch_cycle==0 || cycles<launch_cycle+WD+12))m_axi_awready=0;
            if(bs==0 && mode_now==33 && (launch_cycle==0 || cycles<launch_cycle+WD+12))m_axi_wready=0;
            if(bs==0 && (mode_now==35 || mode_now==42))m_axi_wready=cycles>=launch_cycle+12 && mode_now!=42;
            if(bs==0 && (mode_now==36 || mode_now==43))m_axi_awready=cycles>=launch_cycle+12 && mode_now!=43;
            if(bs==0 && mode_now==50 && cycles<launch_cycle+12)begin m_axi_awready=0;m_axi_wready=0;end
            if(aws>bs && ws>bs && !m_axi_bvalid && $urandom_range(0,3)!=0 &&
                !(bs==127 && mode_now==34 && cycles<last_b_cycle+WD+40) &&
                !(bs==0 && mode_now==37 && cycles<launch_cycle+12) && mode_now!=44)begin
                m_axi_bvalid=1;
                m_axi_bresp=((mode_now==28 && bs==0) || (mode_now==55 && bs==127)) ? 2:0;
                m_axi_bid=((mode_now==29 && bs==127) || (mode_now==31 && bs==127)) ? 15:0;
            end
        end
    end
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin
            source_residual_q_valid<=0;source_ffn_q_valid<=0;source_residual_q<=0;source_ffn_q<=0;
            hold_aw=0;hold_w=0;
        end else begin
            source_residual_q_valid<=source_residual_ce && source_result_valid && !((mode_now==22 && source_residual_address==17) || (mode_now==54 && source_residual_address==2047));
            source_ffn_q_valid<=source_ffn_ce && source_result_valid && !(mode_now==24 && source_ffn_address==17);
            require(!(source_residual_ce && source_ffn_ce),"Input RAM selection");
            if(source_residual_ce)begin
                require(int'(source_residual_address)==rreads && qreads==0,"Residual source order");
                source_residual_q<=residual_word(int'(source_residual_address));rreads++;total_rr++;
                $display("RSTORE_RREAD cycle=%0d job=%0d index=%0d data=%08h",cycles,jobs,source_residual_address,residual_word(int'(source_residual_address)));
            end
            if(hold_aw)require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"Held AW changed");
            if(hold_w)require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"Held W changed");
            hold_aw=m_axi_awvalid && !m_axi_awready;hold_w=m_axi_wvalid && !m_axi_wready;
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};aw_stalls+=int'(hold_aw);w_stalls+=int'(hold_w);
            if(m_axi_awvalid || m_axi_wvalid || source_residual_ce)
                require(!(mode_now>=1 && mode_now<=18),"Traffic for rejected command");
            if(busy)require(completion_residual_beats==good_r,"Only accepted successful B commits the counter");
            if(m_axi_awvalid && m_axi_awready)begin
                require(aws<128 && aws==bs && m_axi_awaddr==record_address(aws),"AXI address/count");
                require(m_axi_awlen==0 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AXI attributes");
                if(ws==aws)aw_first++;
                pending_address=m_axi_awaddr;
                $display("RSTORE_AW cycle=%0d job=%0d record=%0d address=%016h",cycles,jobs,aws,m_axi_awaddr);
                aws++;total_aw++;
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(ws<128 && ws==bs && m_axi_wdata===record_word(ws),"Actual residual AXI bytes/order");
                require(m_axi_wstrb==64'hffffffffffffffff && m_axi_wlast,"AXI W mask/last");
                if(aws==ws)w_first++;
                pending_write=m_axi_wdata;
                $display("RSTORE_W cycle=%0d job=%0d record=%0d data=%0128h",cycles,jobs,ws,m_axi_wdata);
                ws++;total_w++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aws==bs+1 && ws==bs+1 && pending_address==record_address(bs),"B before matching AW/W");
                if(m_axi_bresp==0 && m_axi_bid==0)begin
                    memory[bs]=pending_write;
                    if(bs<128)good_r++;else good_q++;
                end
                $display("RSTORE_B cycle=%0d job=%0d record=%0d response=%0d id=%0d",cycles,jobs,bs,m_axi_bresp,m_axi_bid);
                bs++;total_b++;last_b_cycle=cycles;m_axi_bvalid<=0;
            end
            if(source_completion_valid && source_completion_ready)begin
                require(source_sequence==sequence_model && source_epoch==epoch_model,"Foreign source acknowledgement");
                require(aws==bs && ws==bs && !m_axi_awvalid && !m_axi_wvalid && !m_axi_bready,"Source released before AXI drain");
                source_acks++;
                $display("RSTORE_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,source_sequence,source_epoch);
                source_completion_valid<=0;source_result_valid<=0;
            end
            if(result_valid)begin
                require(completion_valid && completion_residual_beats==128 && completion_ffn_beats==0 && bs==128,"Premature residual publication");
                require(result_residual_address==residual_model && result_residual_bytes==8192 && result_ffn_address==0 && result_ffn_bytes==0,"Residual descriptor");
                for(integer i=0;i<128;i++)require(memory[i]===record_word(i),"Published committed residual image");
            end else require(result_residual_address==0 && result_ffn_address==0 && result_residual_bytes==0 && result_ffn_bytes==0,"Private descriptor leak");
            if(completion_valid && !completion_ready)held++;
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;resets++;
        $display("RSTORE_RESET cycle=%0d job=%0d count=%0d",cycles,jobs,resets);
        command_valid=0;cancel_valid=0;completion_ready=0;source_completion_valid=0;source_result_valid=0;
        m_axi_bvalid=0;repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(!busy && !completion_valid && !source_completion_ready && !result_valid && !m_axi_awvalid && !m_axi_wvalid,"Reset ownership");
    endtask
    function automatic integer wanted_status(input int mode);
        case(mode)
            1,2,3,20,36:return 2;
            4,28,29,55:return 3;
            5,21,46:return 11;
            6,7,8,9,10,11,12,13,14,15,16,17,18:return 1;
            19,35,37,38,50:return 7;
            22,23,48,54:return 4;
            32,33,34:return 8;
            default:return 0;
        endcase
    endfunction
    task automatic run_job(input int mode);
        integer limit,acks_before;logic [3:0] saved;
        jobs++;mode_now=mode;sample_now=(jobs-1)%4;acks_before=source_acks;
        @(negedge clk);require(!busy,"Previous owner active");
        rreads=0;qreads=0;aws=0;ws=0;bs=0;good_r=0;good_q=0;launch_cycle=0;last_b_cycle=0;
        for(integer i=0;i<128;i++)memory[i]='1;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        residual_model=64'h2200000fc0+64'(jobs*16384);ffn_model=0;
        if(mode==53)residual_model=64'hffffffffffffdfc0;
        source_sequence=sequence_model;source_epoch=epoch_model;expected_epoch=epoch_model;
        command_sequence=sequence_model;command_epoch=epoch_model;
        command_residual_address=residual_model;command_residual_base=residual_model;command_residual_bytes=8192;
        command_ffn_address=ffn_model;command_ffn_base=ffn_model;command_ffn_bytes=0;
        source_completion_valid=1;source_completion_status=mode==4 ? 3:0;source_result_valid=mode!=4 && mode!=5;
        case(mode)
            1:source_sequence=sequence_model^(64'd1<<40);
            2:source_epoch=epoch_model^(64'd1<<44);
            3:expected_epoch=epoch_model^(64'd1<<48);
            6:command_residual_address++;
            7:command_residual_bytes=8191;
            8:begin command_residual_base=64'hfffffffffffff000;command_residual_bytes=8192;end
            9:begin command_residual_address=64'hfffffffffffff040;command_residual_base=0;command_residual_bytes='1;end
            10:command_residual_base=residual_model+64;
            17:command_residual_bytes=0;
            default:;
        endcase
        #1;require(command_ready,"Command admission");command_valid=1;
        $display("RSTORE_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d sequence=%016h epoch=%016h raddr=%016h rbase=%016h rbytes=%016h qaddr=%016h qbase=%016h qbytes=%016h",
            cycles,jobs,mode,sample_now,sequence_model,epoch_model,command_residual_address,command_residual_base,command_residual_bytes,command_ffn_address,command_ffn_base,command_ffn_bytes);
        tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;
        command_residual_address=0;command_residual_base=0;command_residual_bytes=0;command_ffn_address=0;command_ffn_base=0;command_ffn_bytes=0;
        if(mode==3)expected_epoch=epoch_model;
        if(mode==19)begin cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if(mode==20 || mode==21)begin
            wait(rreads>=20);@(negedge clk);
            if(mode==20)expected_epoch=epoch_model^(64'd1<<48);else source_sequence=sequence_model^(64'd1<<40);
            repeat(2)tick();@(negedge clk);expected_epoch=epoch_model;source_sequence=sequence_model;
        end
        if(mode==35 || mode==36 || mode==37 || mode==38 || mode==50)begin
            if(mode==35)wait(aws==1);else if(mode==36)wait(ws==1);
            else if(mode==37)wait(aws==1 && ws==1);else if(mode==38)wait(bs==128);else wait(m_axi_awvalid || m_axi_wvalid);
            @(negedge clk);if(mode==36)expected_epoch=epoch_model^(64'd1<<48);else cancel_valid=1;
            tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;
        end
        if(mode==46 || mode==47)begin
            wait(m_axi_awvalid || m_axi_wvalid);@(negedge clk);source_sequence=sequence_model^(64'd1<<40);
            $display("RSTORE_FOREIGN cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            wait(dut.state==dut.RELEASE_SOURCE);
            repeat(32)begin tick();require(!source_completion_ready && !completion_valid && !result_valid,"Foreign response released");end
            $display("RSTORE_HELD cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            if(mode==46)begin
                @(negedge clk);source_sequence=sequence_model;
                $display("RSTORE_RESTORE cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            end
        end
        if((mode>=41 && mode<=45) || mode==47)begin
            if(mode==41)wait(rreads>=20);
            else if(mode==42)wait(aws==1 && ws==0);
            else if(mode==43)wait(ws==1 && aws==0);
            else if(mode==44)wait(aws==1 && ws==1);
            else if(mode==45)wait(completion_valid);
            $display("RSTORE_ABORT cycle=%0d job=%0d mode=%0d aw=%0d w=%0d b=%0d",cycles,jobs,mode,aws,ws,bs);
            reset_all();
        end else begin
            limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<30000,"Completion timeout");end
            saved=completion_status;
            require(saved==4'(wanted_status(mode)),"Completion status");
            require(completion_sequence==sequence_model && completion_epoch==epoch_model,"Full parent identity");
            require(source_acks-acks_before==((mode==1 || mode==2) ? 0:1),"Source ACK count");
            require(aws==ws && ws==bs && completion_residual_beats==good_r && completion_ffn_beats==good_q,"Completion counts/outstanding AXI");
            repeat(7)begin tick();require(completion_valid && completion_status==saved && !command_ready,"Held completion");end
            if(mode==39 || mode==40)begin
                @(negedge clk);if(mode==39)expected_epoch=epoch_model^(64'd1<<48);else cancel_valid=1;
                #1;require(!result_valid && result_residual_address==0 && result_ffn_address==0 && result_residual_bytes==0 && result_ffn_bytes==0,"Late pair publication leak");
                tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;tick();
                require(!result_valid && completion_status==saved,"Rollback revived pair");
                $display("RSTORE_REVOKED cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            end
            $display("RSTORE_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h rbeats=%0d qbeats=%0d valid=%0d raddr=%016h rbytes=%0d qaddr=%016h qbytes=%0d aw=%0d w=%0d b=%0d",
                cycles,jobs,mode,saved,completion_sequence,completion_epoch,completion_residual_beats,completion_ffn_beats,result_valid,result_residual_address,result_residual_bytes,result_ffn_address,result_ffn_bytes,aws,ws,bs);
            publics++;@(negedge clk);completion_ready=1;tick();@(negedge clk);completion_ready=0;source_completion_valid=0;source_result_valid=0;
            require(!busy && !completion_valid && !result_valid,"Owner release");
        end
    endtask
    initial begin
        require($value$plusargs("FIXTURE=%s",fixture),"Missing fixture");
        $readmemh({fixture,"/residual32.hex"},residual_words);
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        $display("RSTORE_CONFIG seed=%0d watchdog=%0d",seed,WD);reset_all();
        repeat(4)run_job(0);
        for(integer mode=1;mode<=55;mode++)if(!(mode==11 || mode==12 || mode==13 || mode==14 || mode==15 || mode==16 || mode==18 || mode==24 || mode==25 || mode==26 || mode==27 || mode==30 || mode==31 || mode==49 || mode==51 || mode==52))begin run_job(mode);run_job(0);end
        require(jobs==82 && publics==76 && resets==7 && aw_stalls>0 && w_stalls>0 && aw_first>0 && w_first>0,"Suite coverage");
        $display("RSTORE_PASS seed=%0d jobs=%0d publics=%0d acks=%0d resets=%0d rreads=%0d qreads=%0d aw=%0d w=%0d b=%0d aw_stalls=%0d w_stalls=%0d held=%0d aw_first=%0d w_first=%0d cycles=%0d",
            seed,jobs,publics,source_acks,resets,total_rr,total_qr,total_aw,total_w,total_b,aw_stalls,w_stalls,held,aw_first,w_first,cycles);
        $finish;
    end
    initial begin #20000000;$fatal(1,"RSTORE_FAIL global timeout");end
endmodule
`default_nettype wire
