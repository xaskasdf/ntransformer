`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_quantize_store_r1;
    localparam integer WD=65536;
    localparam logic [63:0] FIRST_TAG=64'h00000000ffffff80;
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic command_valid=0,command_ready,cancel_valid=0,cancel_ready,busy,tag_exhausted;
    logic [63:0] command_sequence=0,command_epoch=0,expected_epoch=0;
    logic [63:0] command_address=0,command_region_base=0,command_region_bytes=0;
    logic source_completion_valid=0,source_completion_ready,source_result_valid=0;
    logic [3:0] source_completion_status=0;
    logic [63:0] source_sequence=0,source_epoch=0;
    logic source_result_ce,source_result_q_valid=0;
    logic [10:0] source_result_address;
    logic [31:0] source_result_q=0,completion_hls_status;
    logic completion_valid,completion_ready=0,result_valid;
    logic [3:0] completion_status;
    logic [63:0] completion_sequence,completion_epoch,result_address;
    logic [6:0] completion_records;
    logic [12:0] result_bytes;
    logic [63:0] m_axi_awaddr,m_axi_wstrb;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize,m_axi_awprot;
    logic [1:0] m_axi_awburst,m_axi_bresp=0;
    logic [3:0] m_axi_awid,m_axi_awcache,m_axi_bid=0;
    logic m_axi_awlock,m_axi_awvalid,m_axi_awready=0,m_axi_wlast,m_axi_wvalid,m_axi_wready=0;
    logic m_axi_bvalid=0,m_axi_bready;
    logic [511:0] m_axi_wdata;
    ntrpu_attention_quantize_store_r1 #(.WATCHDOG_CYCLES(WD),.INITIAL_CHILD_TAG(FIRST_TAG)) dut(.*);

    logic [31:0] fp32[0:8191];logic [63:0] words[0:2047];
    logic [511:0] memory[0:63];logic [2047:0] seen;
    string fixture,fpfixture;
    integer seed=1,random_state=1,cycles=0,jobs=0,mode_now=0,sample_now=0,resets=0,publics=0;
    integer reads=0,writes=0,aws=0,ws=0,bs=0,launch_cycle=0,starts=0,dones=0,root_acks=0,quant_acks=0;
    integer total_reads=0,total_writes=0,total_aw=0,total_w=0,total_b=0,total_starts=0,total_dones=0,total_root=0,total_quant=0;
    integer aw_stalls=0,w_stalls=0,held=0,aw_first=0,w_first=0,pending_address=0;
    logic pending_input=0;
    logic [63:0] sequence_model,epoch_model,address_model;
    logic [88:0] saved_aw;logic [576:0] saved_w;logic hold_aw=0,hold_w=0;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"QSTORE_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,mode_now,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [31:0] source_word(input int index);
        if(mode_now==3 && index==2030)return 32'h7fc00001;
        return fp32[sample_now*2048+index];
    endfunction
    function automatic logic [511:0] record_word(input int index);
        logic [511:0] value;
        for(integer i=0;i<8;i++)value[i*64+:64]=words[sample_now*512+index*8+i];
        return value;
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;end
        else begin
            m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if((m_axi_awvalid || m_axi_wvalid) && launch_cycle==0)launch_cycle=cycles;
            if(mode_now==5 && bs==0)m_axi_wready=cycles>=launch_cycle+12;
            if(mode_now==8)m_axi_awready=0;
            if(aws>bs && ws>bs && !m_axi_bvalid && $urandom_range(0,3)!=0)begin
                m_axi_bvalid=1;m_axi_bresp=(mode_now==4 && bs==7) ? 2:0;m_axi_bid=0;
            end
        end
    end
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin source_result_q_valid<=0;source_result_q<=0;hold_aw=0;hold_w=0;pending_input=0;end
        else begin
            if(pending_input && dut.quant_owner.fault==0 && !dut.quant_owner.epoch_bad && !dut.quant_owner.peer_bad && !dut.child_cancel)begin
                require(dut.hls_input_q===source_word(pending_address),"Synchronous HLS input binding");
                $display("QSTORE_INPUT cycle=%0d job=%0d index=%0d data=%08h",cycles,jobs,pending_address,dut.hls_input_q);
            end
            pending_input=source_result_ce;pending_address=int'(source_result_address);
            source_result_q_valid<=source_result_ce && source_result_valid;
            if(source_result_ce)begin
                require(source_completion_valid && source_result_valid && source_sequence==sequence_model && source_epoch==epoch_model,"Unowned source read");
                source_result_q<=source_word(int'(source_result_address));seen[source_result_address]=1;reads++;total_reads++;
                $display("QSTORE_READ cycle=%0d job=%0d index=%0d data=%08h",cycles,jobs,source_result_address,source_word(int'(source_result_address)));
            end
            if(dut.hls_start)begin
                starts++;total_starts++;require(starts==1 && dut.hls_elements==2048,"HLS start extent/count");
                $display("QSTORE_START cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.hls_output_ce && dut.hls_output_we)begin
                require(dut.hls_output_address==9'(writes),"Generated output order");
                if(mode_now!=1 && mode_now!=2 && mode_now!=9)require(dut.hls_output_d===words[sample_now*512+writes],"Generated ABI1 bytes");
                $display("QSTORE_WORD cycle=%0d job=%0d index=%0d data=%016h",cycles,jobs,writes,dut.hls_output_d);
                writes++;total_writes++;
            end
            if(dut.hls_done)begin
                dones++;total_dones++;require(dones==1 && dut.hls_ready,"Generated done/ready");
                require(dut.hls_return==(mode_now==3 ? 2:0),"HLS return status");
                require(writes==(mode_now==3 ? 0:512),"Generated output extent");
                $display("QSTORE_DONE cycle=%0d job=%0d status=%0d words=%0d",cycles,jobs,dut.hls_return,writes);
            end
            if(hold_aw)require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"Held AW changed");
            if(hold_w)require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"Held W changed");
            hold_aw=m_axi_awvalid && !m_axi_awready;hold_w=m_axi_wvalid && !m_axi_wready;
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};aw_stalls+=int'(hold_aw);w_stalls+=int'(hold_w);
            if(m_axi_awvalid && m_axi_awready)begin
                require(root_acks==1 && aws<64 && aws==bs && m_axi_awaddr==address_model+64'(aws*64),"AXI address/count/root release");
                require(m_axi_awlen==0 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AXI attributes");
                if(ws==aws)aw_first++;
                $display("QSTORE_AW cycle=%0d job=%0d record=%0d address=%016h",cycles,jobs,aws,m_axi_awaddr);
                aws++;total_aw++;
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(root_acks==1 && ws<64 && ws==bs && m_axi_wdata===record_word(ws),"ABI1 AXI bytes/order");
                require(m_axi_wstrb==64'hffffffffffffffff && m_axi_wlast,"AXI W mask/last");
                if(aws==ws)w_first++;
                memory[ws]=m_axi_wdata;
                $display("QSTORE_W cycle=%0d job=%0d record=%0d data=%0128h",cycles,jobs,ws,m_axi_wdata);
                ws++;total_w++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aws==bs+1 && ws==bs+1,"B before AW/W");
                $display("QSTORE_B cycle=%0d job=%0d record=%0d response=%0d",cycles,jobs,bs,m_axi_bresp);
                bs++;total_b++;m_axi_bvalid<=0;
            end
            if(source_completion_valid && source_completion_ready)begin
                require(root_acks==0 && dones==1 && source_sequence==sequence_model && source_epoch==epoch_model,"Foreign or premature root acknowledgement");
                root_acks++;total_root++;
                $display("QSTORE_ROOT_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,source_sequence,source_epoch);
                source_completion_valid<=0;source_result_valid<=0;
            end
            if(dut.quant_completion_valid && dut.quant_completion_ready)begin
                require(root_acks==1 && quant_acks==0 && aws==bs && ws==bs,"Quantization released before AXI drain");
                quant_acks++;total_quant++;
                $display("QSTORE_QUANT_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,dut.quant_sequence,dut.quant_epoch);
            end
            if(result_valid)begin
                require(completion_valid && quant_acks==1 && completion_records==64 && bs==64 && result_address==address_model && result_bytes==4096,"Premature region publication");
                for(integer i=0;i<64;i++)require(memory[i]===record_word(i),"Published memory image");
            end else require(result_address==0 && result_bytes==0,"Private region leak");
            if(completion_valid && !completion_ready)held++;
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;resets++;
        $display("QSTORE_RESET cycle=%0d job=%0d count=%0d",cycles,jobs,resets);
        command_valid=0;cancel_valid=0;completion_ready=0;source_completion_valid=0;source_result_valid=0;
        m_axi_bvalid=0;repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(!busy && !completion_valid && !source_completion_ready && !result_valid && !m_axi_awvalid && !m_axi_wvalid,"Reset ownership");
    endtask
    function automatic integer wanted_status(input int mode);
        case(mode)1,5:return 7;2:return 2;3:return 5;4:return 3;default:return 0;endcase
    endfunction
    task automatic run_job(input int mode);
        integer limit;logic [3:0] saved;
        @(negedge clk);jobs++;mode_now=mode;sample_now=(jobs-1)%4;require(!busy,"Previous owner active");
        reads=0;writes=0;aws=0;ws=0;bs=0;launch_cycle=0;starts=0;dones=0;root_acks=0;quant_acks=0;seen=0;
        for(integer i=0;i<64;i++)memory[i]='1;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        address_model=64'h2200000fc0+64'(jobs*8192);
        source_sequence=sequence_model;source_epoch=epoch_model;expected_epoch=epoch_model;
        command_sequence=sequence_model;command_epoch=epoch_model;
        command_address=address_model;command_region_base=address_model;command_region_bytes=4096;
        source_completion_valid=1;source_completion_status=0;source_result_valid=1;
        #1;require(command_ready,"Command admission");command_valid=1;
        $display("QSTORE_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d sequence=%016h epoch=%016h address=%016h",cycles,jobs,mode,sample_now,sequence_model,epoch_model,address_model);
        tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;command_address=0;command_region_base=0;command_region_bytes=0;
        if(mode==1 || mode==2 || mode==7 || mode==9)begin
            wait(reads>=64);@(negedge clk);
            $display("QSTORE_INJECT cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            if(mode==1)cancel_valid=1;
            if(mode==2)expected_epoch=epoch_model^(64'd1<<48);
            if(mode==9)source_sequence=sequence_model^(64'd1<<40);
            if(mode==1 || mode==2)begin repeat(2)tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;end
            if(mode==9)begin
                wait(dones==1);repeat(32)begin tick();require(!source_completion_ready && !completion_valid && !result_valid && !m_axi_awvalid && !m_axi_wvalid,"Foreign source released");end
                $display("QSTORE_FOREIGN_HELD cycle=%0d job=%0d",cycles,jobs);
            end
        end
        if(mode==5)begin
            wait(aws==1);@(negedge clk);cancel_valid=1;
            $display("QSTORE_INJECT cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            tick();@(negedge clk);cancel_valid=0;
        end
        if(mode==10)begin
            wait(root_acks==1);@(negedge clk);
            source_sequence=sequence_model^(64'd1<<40);source_epoch=epoch_model^(64'd1<<44);
            source_completion_valid=1;source_result_valid=1;
            $display("QSTORE_REUSE cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,source_sequence,source_epoch);
        end
        if(mode==7 || mode==8 || mode==9)begin
            if(mode==8)wait(ws==1 && aws==0);
            $display("QSTORE_ABORT cycle=%0d job=%0d mode=%0d aw=%0d w=%0d b=%0d",cycles,jobs,mode,aws,ws,bs);
            reset_all();
        end else begin
            limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<250000,"Completion timeout");end
            saved=completion_status;require(saved==4'(wanted_status(mode)),"Completion status");
            require(completion_hls_status==(mode==3 ? 2:0),"Public HLS status");
            require(completion_sequence==sequence_model && completion_epoch==epoch_model,"Full parent identity");
            require(root_acks==1 && quant_acks==1 && aws==ws && ws==bs,"Completion drain");
            if(mode==0 || mode==4 || mode==5 || mode==6 || mode==10)require(&seen,"Missing FP32 source address");
            require(result_valid==(saved==0),"Initial publication");
            repeat(7)begin tick();require(completion_valid && completion_status==saved && !command_ready,"Held completion");end
            if(mode==6)begin
                @(negedge clk);expected_epoch=epoch_model^(64'd1<<48);
                #1;require(!result_valid && result_address==0 && result_bytes==0,"Late publication leak");
                tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;tick();
                require(!result_valid && completion_status==saved,"Rollback revived region");
                $display("QSTORE_REVOKED cycle=%0d job=%0d",cycles,jobs);
            end
            $display("QSTORE_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d hls=%0d sequence=%016h epoch=%016h records=%0d valid=%0d address=%016h bytes=%0d aw=%0d w=%0d b=%0d",cycles,jobs,mode,saved,completion_hls_status,completion_sequence,completion_epoch,completion_records,result_valid,result_address,result_bytes,aws,ws,bs);
            publics++;@(negedge clk);completion_ready=1;tick();@(negedge clk);completion_ready=0;source_completion_valid=0;source_result_valid=0;
            require(!busy && !completion_valid && !result_valid,"Owner release");
        end
    endtask
    initial begin
        require($value$plusargs("FIXTURE=%s",fixture),"Missing ABI1 fixture");$readmemh(fixture,words);
        require($value$plusargs("FP32=%s",fpfixture),"Missing FP32 fixture");$readmemh(fpfixture,fp32);
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        $display("QSTORE_CONFIG seed=%0d watchdog=%0d",seed,WD);
        reset_all();repeat(4)run_job(0);
        for(integer mode=1;mode<=10;mode++)begin run_job(mode);run_job(0);end
        require(jobs==24 && publics==21 && resets==4 && total_starts==24 && total_dones==23 && total_root==22 && total_quant==21,"Suite ownership coverage");
        require(total_aw==1033 && total_w==1034 && total_b==1033 && aw_stalls>0 && w_stalls>0 && aw_first>0 && w_first>0,"Suite bus coverage");
        $display("QSTORE_PASS seed=%0d jobs=%0d publics=%0d root=%0d quant=%0d resets=%0d reads=%0d writes=%0d starts=%0d dones=%0d aw=%0d w=%0d b=%0d aw_stalls=%0d w_stalls=%0d held=%0d aw_first=%0d w_first=%0d cycles=%0d",seed,jobs,publics,total_root,total_quant,resets,total_reads,total_writes,total_starts,total_dones,total_aw,total_w,total_b,aw_stalls,w_stalls,held,aw_first,w_first,cycles);
        $finish;
    end
    initial begin #40000000;$fatal(1,"QSTORE_FAIL global timeout");end
endmodule
`default_nettype wire
