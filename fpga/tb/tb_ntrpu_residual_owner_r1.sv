`timescale 1ns/1ps
`default_nettype none
// Protocol peers only: distinct RAM patterns, no residual arithmetic.
module tb_ntrpu_residual_owner_r1;
    localparam integer WD=64;
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic command_valid=0,command_ready,cancel_valid=0,cancel_ready,busy;
    logic [63:0] command_sequence=0,command_epoch=0,expected_epoch=0;
    logic source_completion_valid=0,source_completion_ready,source_result_valid=0;
    logic [3:0] source_completion_status=0;
    logic [63:0] source_sequence=0,source_epoch=0;
    wire [1:0] source_ce;
    wire [10:0] source_address [0:1];
    logic [31:0] source_q [0:1];
    logic [1:0] source_q_valid=0;
    logic hls_start,hls_done=0,hls_ready=0;
    logic [31:0] hls_return=0,hls_elements;
    logic [1:0] hls_input_ce=0;
    logic [10:0] hls_input_address [0:1];
    wire [31:0] hls_input_q [0:1];
    logic hls_residual_ce=0,hls_residual_we=0;
    logic [10:0] hls_residual_address=0;
    logic [31:0] hls_residual_d=0;
    logic completion_valid,completion_ready=0,result_valid;
    logic residual_ce=0,residual_q_valid;
    logic [10:0] residual_address=0;
    logic [31:0] residual_q;
    logic [3:0] completion_status;
    logic [31:0] completion_hls_status;
    logic [63:0] completion_sequence,completion_epoch;
    ntrpu_residual_owner_r1 #(.WATCHDOG_CYCLES(WD)) dut(.*);
    integer seed=1,random_state=1,cycles=0,jobs=0,current_mode=0,completions=0,results=0,residual_results=0;
    integer starts=0,dones=0,acks=0,resets=0,reads=0,writes=0,core_reads=0,core_writes=0,core_residual_writes=0,residual_writes=0,held=0;
    logic core_busy=0,block_response=0;
    logic [63:0] sequence_model,epoch_model;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"ROWNER_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,current_mode,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [31:0] source_word(input int plane,input int index);
        return 32'h3f000000 | 32'(plane*1048576+jobs*4096+index);
    endfunction
    function automatic logic [31:0] residual_word(input int index);
        return 32'hbf000000 | 32'(jobs*4096+index);
    endfunction
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin
            source_q_valid<=0;for(integer p=0;p<2;p++)source_q[p]<=0;
        end else begin
            for(integer p=0;p<2;p++)begin
                source_q_valid[p]<=source_ce[p] && source_result_valid &&
                    !(block_response && p==(current_mode==35 ? 1:0));
                if(source_ce[p])begin
                    source_q[p]<=(current_mode==44 && core_reads==64 && p==0) ? 32'h7fc12345:source_word(p,int'(source_address[p]));reads++;
                end
            end
            if(!completion_valid)require(!result_valid,"Premature residual publication");
            if(!result_valid)require(!residual_q_valid && residual_q===0,"Private residual leak");
            if(source_completion_ready && source_completion_valid)begin
                require(!core_busy && !hls_start,"Source released before HLS drain");
                acks++;$display("ROWNER_ACK cycle=%0d job=%0d",cycles,jobs);
                source_completion_valid<=0;source_result_valid<=0;
            end
            if(completion_valid && !completion_ready)held++;
            if(hls_residual_ce && hls_residual_we)residual_writes++;
        end
    end
    task automatic peer_core;
        begin: work
            core_busy=1;core_reads=0;core_writes=0;core_residual_writes=0;starts++;
            require(hls_elements==2048 && source_completion_valid && source_result_valid,"HLS start admission");
            $display("ROWNER_START cycle=%0d job=%0d",cycles,jobs);
            for(integer index=0;index<2048;index++)begin
                @(negedge clk);if(!rst_n)disable work;
                hls_input_ce={index%2==0,1'b1};
                hls_input_address[0]=11'(index*3+17);
                hls_input_address[1]=11'(index*5+31);
                tick();if(!rst_n)disable work;
                for(integer p=0;p<2;p++)begin
                    if(dut.fault_now==0 && source_q_valid[p])
                        require(hls_input_q[p]===source_word(p,int'(hls_input_address[p])),"Independent source RAM response");
                    else if(dut.fault!=0 || !source_q_valid[p])
                        require(hls_input_q[p]===0,"Revoked source response leak");
                end
                core_reads++;
                if(index%37==0)begin
                    @(negedge clk);hls_input_ce=0;
                    repeat($urandom_range(1,3))begin tick();if(!rst_n)disable work;end
                end
                if(current_mode==21 && index==256)begin
                    @(negedge clk);hls_input_ce=0;
                    repeat(WD+3)begin tick();if(!rst_n)disable work;require(!source_completion_ready && !completion_valid,"Timeout abandoned HLS");end
                end
            end
            @(negedge clk);hls_input_ce=0;
            for(integer index=0;index<(current_mode==18 ? 0:current_mode==31 ? 2047:current_mode==34 ? 2049:2048);index++)begin
                @(negedge clk);if(!rst_n)disable work;
                hls_residual_ce=1;hls_residual_we=1;
                hls_residual_address=11'(current_mode==32 && index==64 ? 63:index);
                hls_residual_d=residual_word(index);
                if(current_mode==33 && index==1024)hls_residual_d=32'h7f800000;
                if(current_mode==43 && index==2047)begin hls_done=1;hls_ready=1;hls_return=0;end
                tick();if(!rst_n)disable work;core_residual_writes++;
                if(current_mode==43 && index==2047)begin
                    core_busy=0;dones++;
                    $display("ROWNER_DONE cycle=%0d job=%0d status=0",cycles,jobs);
                    @(negedge clk);hls_done=0;hls_ready=0;hls_residual_ce=0;hls_residual_we=0;
                    disable work;
                end
                if(index%23==0)begin
                    @(negedge clk);hls_residual_ce=0;hls_residual_we=0;
                    repeat($urandom_range(1,3))begin tick();if(!rst_n)disable work;end
                end
            end
            @(negedge clk);hls_residual_ce=0;hls_residual_we=0;
            repeat(4)begin tick();if(!rst_n)disable work;require(!result_valid,"Partial residual publication");end
            @(negedge clk);if(!rst_n)disable work;
            hls_done=1;hls_ready=current_mode!=19;hls_return=current_mode==18 ? 2:current_mode==41 ? 6:0;
            @(posedge clk);if(!rst_n)disable work;
            core_busy=0;dones++;$display("ROWNER_DONE cycle=%0d job=%0d status=%0d",cycles,jobs,hls_return);
            #1;@(negedge clk);hls_done=0;hls_ready=0;
        end
        core_busy=0;hls_residual_ce=0;hls_residual_we=0;hls_input_ce=0;hls_done=0;hls_ready=0;
    endtask
    always begin @(posedge clk);if(rst_n && hls_start)peer_core();end
    task automatic reset_all;
        @(negedge clk);rst_n=0;resets++;
        $display("ROWNER_RESET cycle=%0d job=%0d count=%0d",cycles,jobs,resets);
        command_valid=0;cancel_valid=0;completion_ready=0;residual_ce=0;block_response=0;
        source_completion_valid=0;source_result_valid=0;
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(!busy && !completion_valid && !result_valid && !hls_start && !source_completion_ready && !core_busy,"Reset ownership");
    endtask
    function automatic integer wanted_status(input int mode);
        case(mode)
            1:return 3;
            2,3,4,8,23,40:return 2;
            5,9,10,11,19,37,38:return 11;
            6,7,22,39:return 7;
            12,31,32,33,34,35,44:return 4;
            18,41:return 5;
            21:return 8;
            default:return 0;
        endcase
    endfunction
    task automatic run_job(input int mode);
        integer limit,ack_before,start_before;logic [3:0] saved;
        jobs++;current_mode=mode;ack_before=acks;start_before=starts;core_reads=0;core_writes=0;core_residual_writes=0;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        @(negedge clk);require(!busy && !core_busy,"Previous owner active");
        source_sequence=sequence_model;source_epoch=epoch_model;expected_epoch=epoch_model;
        command_sequence=sequence_model;command_epoch=epoch_model;
        source_completion_valid=1;source_completion_status=mode==1 ? 3:0;source_result_valid=mode!=5 && mode!=1;
        if(mode==2)source_sequence=sequence_model^(64'd1<<40);
        if(mode==3)source_epoch=epoch_model^(64'd1<<44);
        if(mode==4)expected_epoch=epoch_model^(64'd1<<48);
        #1;require(command_ready,"Command readiness");command_valid=1;
        $display("ROWNER_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,sequence_model,epoch_model);
        tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;
        if(mode==4)expected_epoch=epoch_model;
        if(mode==6)begin cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if((mode>=7 && mode<=12) || (mode>=35 && mode<=38))begin
            wait(core_reads>=64);@(negedge clk);
            case(mode)
                7:cancel_valid=1;
                8:expected_epoch=epoch_model^(64'd1<<48);
                9:source_sequence=sequence_model^(64'd1<<40);
                10:source_completion_valid=0;
                11:source_result_valid=0;
                12,35,36:block_response=1;
                37:source_epoch=epoch_model^(64'd1<<44);
                38:source_completion_status=3;
            endcase
            repeat(2)tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;
            source_sequence=sequence_model;source_epoch=epoch_model;source_completion_status=0;source_completion_valid=1;source_result_valid=1;block_response=0;
            require(!completion_valid && !source_completion_ready,"Failed core abandoned before drain");
        end
        if(mode==22 || mode==23)begin
            wait(core_residual_writes>=64);@(negedge clk);
            if(mode==22)cancel_valid=1;else expected_epoch=epoch_model^(64'd1<<48);
            tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;
        end
        if(mode==39 || mode==40)begin
            wait(core_residual_writes>=2048);@(negedge clk);
            if(mode==39)cancel_valid=1;else expected_epoch=epoch_model^(64'd1<<48);
            tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;
        end
        if(mode>=26 && mode<=30)begin
            if(mode==26)wait(core_reads>=64);
            else if(mode==27)wait(core_residual_writes>=64);
            else if(mode==28)wait(hls_done);
            else if(mode==29)wait(completion_valid);
            // Mode 30 resets before the H_START acceptance edge.
            if(mode==30)rst_n=0;
            $display("ROWNER_ABORT_RESET cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            reset_all();
        end else begin
            limit=0;
            while(!completion_valid)begin @(negedge clk);limit++;require(limit<20000,"Completion timeout");end
            saved=completion_status;
            require(saved==4'(wanted_status(mode)) && !core_busy && !source_completion_ready,"Completion status/drain");
            require(completion_sequence==sequence_model && completion_epoch==epoch_model,"Captured 64-bit identity");
            require(completion_hls_status==(mode==18 ? 2:mode==41 ? 6:0),"HLS status retained");
            require(acks-ack_before==((mode==2 || mode==3) ? 0:1),"Source ownership acknowledgement");
            require(starts-start_before==(mode<=6 && mode!=0 ? 0:1),"HLS admission count");
            repeat(7)begin tick();require(completion_valid && completion_status==saved && !command_ready,"Held completion");end
            if(saved==0)for(integer index=0;index<2048;index++)begin
                @(negedge clk);residual_ce=1;residual_address=11'(index*5+seed);
                tick();
                require(residual_q_valid && residual_q===residual_word(int'(residual_address)),"Residual result content/order");residual_results++;
                $display("ROWNER_RESIDUAL cycle=%0d job=%0d index=%0d address=%0d data=%08h",cycles,jobs,index,residual_address,residual_q);
                if(index%17==0)begin
                    @(negedge clk);residual_ce=0;tick();
                    require(!residual_q_valid && residual_q===0,"RAM CE bubble");
                end
            end
            if(mode==24 || mode==25)begin
                @(negedge clk);
                if(mode==24)expected_epoch=epoch_model^(64'd1<<48);else cancel_valid=1;
                #1;require(!result_valid && !residual_q_valid && residual_q===0,"Late revocation leak");
                tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;tick();
                require(!result_valid && !residual_q_valid && completion_status==saved,"Rollback revived result");
                $display("ROWNER_REVOKED cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            end
            $display("ROWNER_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,saved,completion_sequence,completion_epoch);
            completions++;@(negedge clk);residual_ce=0;completion_ready=1;
            tick();@(negedge clk);completion_ready=0;source_completion_valid=0;source_result_valid=0;
            require(!busy && !completion_valid && !result_valid,"Owner release");
        end
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        reset_all();
        for(integer mode=0;mode<=44;mode++)if(!(mode>=13 && mode<=17) && mode!=20 && mode!=36 && mode!=42)begin run_job(mode);if(mode!=0)run_job(0);end
        require(jobs==73 && completions==68 && results==0 && residual_results==81920 && resets==6 && held>0,"Suite coverage");
        $display("ROWNER_PASS seed=%0d watchdog=%0d jobs=%0d completions=%0d results=%0d starts=%0d dones=%0d acks=%0d resets=%0d reads=%0d writes=%0d residual_results=%0d residual_writes=%0d held=%0d cycles=%0d",
            seed,WD,jobs,completions,results,starts,dones,acks,resets,reads,writes,residual_results,residual_writes,held,cycles);
        $finish;
    end
    initial begin #20000000;$fatal(1,"ROWNER_FAIL global timeout");end
endmodule
`default_nettype wire
