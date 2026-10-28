`timescale 1ns/1ps
`default_nettype none
// Protocol peers only: distinct RAM patterns, no RMSNorm/quantization arithmetic.
module tb_ntrpu_input_norm_owner_r1;
    localparam integer WD=64;
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic command_valid=0,command_ready,cancel_valid=0,cancel_ready,busy;
    logic [63:0] command_sequence=0,command_epoch=0,expected_epoch=0;
    logic source_completion_valid=0,source_completion_ready,source_result_valid=0;
    logic [3:0] source_completion_status=0;
    logic [63:0] source_sequence=0,source_epoch=0;
    wire [1:0] source_ce;
    wire [10:0] source_address [0:1];
    logic [1:0][31:0] drive_source_q;
    wire [31:0] source_q [0:1];
    logic [1:0] source_q_valid=0;
    logic hls_start,hls_done=0,hls_ready=0;
    logic [31:0] hls_return=0,hls_elements,hls_epsilon_bits;
    logic [1:0] hls_input_ce=0;
    logic [1:0][10:0] drive_input_address;
    wire [10:0] hls_input_address [0:1];
    for(genvar p=0;p<2;p++)begin
        assign source_q[p]=drive_source_q[p];
        assign hls_input_address[p]=drive_input_address[p];
    end
    wire [31:0] hls_input_q [0:1];
    logic hls_output_ce=0,hls_output_we=0;
    logic [8:0] hls_output_address=0;
    logic [63:0] hls_output_d=0;
    logic completion_valid,completion_ready=0,result_valid;
    logic result_ce=0,result_q_valid;
    logic [8:0] result_address=0;
    logic [63:0] result_q;
    logic [3:0] completion_status;
    logic [31:0] completion_hls_status;
    logic [63:0] completion_sequence,completion_epoch;
    ntrpu_input_norm_owner_r1 #(.WATCHDOG_CYCLES(WD)) dut(.*);
    integer seed=1,random_state=1,cycles=0,jobs=0,current_mode=0,completions=0,results=0;
    integer starts=0,dones=0,acks=0,resets=0,reads=0,writes=0,core_reads=0,core_writes=0,held=0;
    logic core_busy=0,block_response=0;
    logic [63:0] sequence_model,epoch_model;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"INOWNER_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,current_mode,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [31:0] source_word(input int plane,input int index);
        return 32'h3f000000 | 32'(plane*1048576+jobs*4096+index);
    endfunction
    function automatic logic [63:0] record_word(input int index);
        logic [63:0] value;integer q;
        value=0;
        if(index%8<4)for(integer b=0;b<8;b++)begin
            q=(jobs+index+b+seed)%255-127;value[b*8+:8]=8'(q);
        end
        else if(index%8==4)value=64'h3ff0000000000000|64'(jobs*512+index);
        return value;
    endfunction
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin
            source_q_valid<=0;for(integer p=0;p<2;p++)drive_source_q[p]<=0;
        end else begin
            for(integer p=0;p<2;p++)begin
                source_q_valid[p]<=source_ce[p] && source_result_valid &&
                    !(block_response && (current_mode==12 || current_mode==35) && p==(current_mode==35 ? 1:0));
                if(source_ce[p])begin
                    drive_source_q[p]<=block_response && (current_mode==43 && p==0 || current_mode==44 && p==1) ? 32'h7fc12345:source_word(p,int'(source_address[p]));reads++;
                end
            end
            if(!completion_valid)require(!result_valid,"Premature ABI1 publication");
            if(!result_valid)require(!result_q_valid && result_q===0,"Private ABI1 leak");
            if(source_completion_ready && source_completion_valid)begin
                require(!core_busy && !hls_start,"Source released before HLS drain");
                acks++;$display("INOWNER_ACK cycle=%0d job=%0d",cycles,jobs);
                source_completion_valid<=0;source_result_valid<=0;
            end
            if(completion_valid && !completion_ready)held++;
            if(hls_output_ce && hls_output_we)writes++;
        end
    end
    task automatic peer_core;
        begin: work
            core_busy=1;core_reads=0;core_writes=0;starts++;
            require(hls_elements==2048 && hls_epsilon_bits==32'h3727c5ac && source_completion_valid && source_result_valid,"HLS start admission");
            $display("INOWNER_START cycle=%0d job=%0d",cycles,jobs);
            for(integer index=0;index<2048;index++)begin
                @(negedge clk);if(!rst_n)disable work;
                hls_input_ce={index%2==0,1'b1};
                drive_input_address[0]=11'(index*3+17);
                drive_input_address[1]=11'(index*5+31);
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
            for(integer index=0;index<(current_mode==18 || current_mode==42 ? 0:current_mode==13 ? 511:current_mode==20 ? 513:512);index++)begin
                @(negedge clk);if(!rst_n)disable work;
                hls_output_ce=1;hls_output_we=1;
                hls_output_address=9'(current_mode==14 && index==64 ? 63:index);
                hls_output_d=record_word(index);
                if(current_mode==15 && index==5)hls_output_d=1;
                if(current_mode==16 && index==4)hls_output_d=64'h8000000000000000;
                if(current_mode==17 && index==0)hls_output_d[7:0]=8'h80;
                tick();if(!rst_n)disable work;core_writes++;
                if(index%19==0)begin
                    @(negedge clk);hls_output_ce=0;hls_output_we=0;
                    repeat($urandom_range(1,3))begin tick();if(!rst_n)disable work;end
                end
            end
            @(negedge clk);if(!rst_n)disable work;
            hls_output_ce=0;hls_output_we=0;hls_done=1;hls_ready=current_mode!=19;hls_return=current_mode==18 ? 2:current_mode==41 ? 6:0;
            @(posedge clk);if(!rst_n)disable work;
            core_busy=0;dones++;$display("INOWNER_DONE cycle=%0d job=%0d status=%0d",cycles,jobs,hls_return);
            #1;@(negedge clk);hls_done=0;hls_ready=0;
        end
        core_busy=0;hls_input_ce=0;hls_output_ce=0;hls_output_we=0;hls_done=0;hls_ready=0;
    endtask
    always begin @(posedge clk);if(rst_n && hls_start)peer_core();end
    task automatic reset_all;
        @(negedge clk);rst_n=0;resets++;
        $display("INOWNER_RESET cycle=%0d job=%0d count=%0d",cycles,jobs,resets);
        command_valid=0;cancel_valid=0;completion_ready=0;result_ce=0;block_response=0;
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
            12,13,14,15,16,17,20,35,42,43,44:return 4;
            18,41:return 5;
            21:return 8;
            default:return 0;
        endcase
    endfunction
    task automatic run_job(input int mode);
        integer limit,ack_before,start_before;logic [3:0] saved;
        jobs++;current_mode=mode;ack_before=acks;start_before=starts;core_reads=0;core_writes=0;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        @(negedge clk);require(!busy && !core_busy,"Previous owner active");
        source_sequence=sequence_model;source_epoch=epoch_model;expected_epoch=epoch_model;
        command_sequence=sequence_model;command_epoch=epoch_model;
        source_completion_valid=1;source_completion_status=mode==1 ? 3:0;source_result_valid=mode!=5 && mode!=1;
        if(mode==2)source_sequence=sequence_model^(64'd1<<40);
        if(mode==3)source_epoch=epoch_model^(64'd1<<44);
        if(mode==4)expected_epoch=epoch_model^(64'd1<<48);
        #1;require(command_ready,"Command readiness");command_valid=1;
        $display("INOWNER_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,sequence_model,epoch_model);
        tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;
        if(mode==6)begin cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if((mode>=7 && mode<=12) || (mode>=35 && mode<=38) || mode==43 || mode==44)begin
            wait(core_reads>=64);@(negedge clk);
            case(mode)
                7:cancel_valid=1;
                8:expected_epoch=epoch_model^(64'd1<<48);
                9:source_sequence=sequence_model^(64'd1<<40);
                10:source_completion_valid=0;
                11:source_result_valid=0;
                12,35,43,44:block_response=1;
                37:source_epoch=epoch_model^(64'd1<<44);
                38:source_completion_status=3;
            endcase
            repeat(2)tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;
            source_sequence=sequence_model;source_epoch=epoch_model;source_completion_status=0;source_completion_valid=1;source_result_valid=1;block_response=0;
            require(!completion_valid && !source_completion_ready,"Failed core abandoned before drain");
        end
        if(mode==22 || mode==23)begin
            wait(core_writes>=64);@(negedge clk);
            if(mode==22)cancel_valid=1;else expected_epoch=epoch_model^(64'd1<<48);
            tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;
        end
        if(mode>=26 && mode<=30)begin
            if(mode==26)wait(core_reads>=64);
            else if(mode==27)wait(core_writes>=64);
            else if(mode==28)wait(hls_done);
            else if(mode==29)wait(completion_valid);
            // Mode 30 resets before the H_START acceptance edge.
            if(mode==30)rst_n=0;
            $display("INOWNER_ABORT_RESET cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
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
            if(saved==0)for(integer index=0;index<512;index++)begin
                @(negedge clk);result_ce=1;result_address=9'(index*3+seed);tick();
                require(result_q_valid && result_q===record_word(int'(result_address)),"ABI1 result content/order");results++;
                $display("INOWNER_RESULT cycle=%0d job=%0d index=%0d address=%0d data=%016h",cycles,jobs,index,result_address,result_q);
                if(index%17==0)begin
                    @(negedge clk);result_ce=0;tick();require(!result_q_valid && result_q===0,"RAM CE bubble");
                end
            end
            if(mode==24 || mode==25)begin
                @(negedge clk);
                if(mode==24)expected_epoch=epoch_model^(64'd1<<48);else cancel_valid=1;
                #1;require(!result_valid && !result_q_valid && result_q===0,"Late revocation leak");
                tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;tick();
                require(!result_valid && !result_q_valid && completion_status==saved,"Rollback revived result");
                $display("INOWNER_REVOKED cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            end
            $display("INOWNER_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,saved,completion_sequence,completion_epoch);
            completions++;@(negedge clk);result_ce=0;completion_ready=1;
            tick();@(negedge clk);completion_ready=0;source_completion_valid=0;source_result_valid=0;
            require(!busy && !completion_valid && !result_valid,"Owner release");
        end
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        reset_all();
        for(integer mode=0;mode<=44;mode++)begin if(!(mode>=31 && mode<=34) && mode!=36 && mode!=39 && mode!=40)begin run_job(mode);if(mode!=0)run_job(0);end end
        require(jobs==75 && completions==70 && results==20480 && resets==6 && held>0,"Suite coverage");
        $display("INOWNER_PASS seed=%0d watchdog=%0d jobs=%0d completions=%0d results=%0d starts=%0d dones=%0d acks=%0d resets=%0d reads=%0d writes=%0d held=%0d cycles=%0d",
            seed,WD,jobs,completions,results,starts,dones,acks,resets,reads,writes,held,cycles);
        $finish;
    end
    initial begin #20000000;$fatal(1,"INOWNER_FAIL global timeout");end
endmodule
`default_nettype wire
