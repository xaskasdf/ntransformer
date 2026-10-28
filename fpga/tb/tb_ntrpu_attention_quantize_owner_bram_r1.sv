`timescale 1ns/1ps
`default_nettype none
// Protocol peer only: values are tagged records, not quantization arithmetic.
`ifndef AQOWNER_DUT
`define AQOWNER_DUT ntrpu_attention_quantize_owner_r3
`endif
module tb_ntrpu_attention_quantize_owner_bram_r1;
    localparam integer WD=64;
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic command_valid=0,command_ready,cancel_valid=0,cancel_ready,busy;
    logic [63:0] command_sequence=0,command_epoch=0,expected_epoch=0;
    logic source_completion_valid=0,source_completion_ready,source_result_valid=0;
    logic [3:0] source_completion_status=0;
    logic [63:0] source_sequence=0,source_epoch=0;
    logic source_result_ce,source_result_q_valid=0;
    logic [10:0] source_result_address;
    logic [31:0] source_result_q=0;
    logic hls_start,hls_done=0,hls_ready=0;
    logic [31:0] hls_return=0,hls_elements;
    logic hls_input_ce=0;
    logic [10:0] hls_input_address=0;
    logic [31:0] hls_input_q;
    logic hls_output_ce=0,hls_output_we=0;
    logic [8:0] hls_output_address=0;
    logic [63:0] hls_output_d=0;
    logic completion_valid,completion_ready=0,result_valid,result_ce=0,result_q_valid;
    logic [3:0] completion_status;
    logic [31:0] completion_hls_status;
    logic [63:0] completion_sequence,completion_epoch,result_q;
    logic [8:0] result_address=0;
    `AQOWNER_DUT #(.WATCHDOG_CYCLES(WD)) dut(.*);
    integer seed=1,random_state=1,cycles=0,jobs=0,current_mode=0,completions=0,results=0;
    integer starts=0,dones=0,acks=0,resets=0,reads=0,writes=0,core_reads=0,core_writes=0,held=0;
    logic core_busy=0,block_response=0;
    logic [63:0] sequence_model,epoch_model;
    // Identical inputs, independent old owner, compare every public output.
    wire  shadow_command_ready;
    wire  shadow_cancel_ready;
    wire  shadow_busy;
    wire  shadow_source_completion_ready;
    wire  shadow_source_result_ce;
    wire [10:0] shadow_source_result_address;
    wire  shadow_hls_start;
    wire [31:0] shadow_hls_elements;
    wire [31:0] shadow_hls_input_q;
    wire  shadow_completion_valid;
    wire [3:0] shadow_completion_status;
    wire [31:0] shadow_completion_hls_status;
    wire [63:0] shadow_completion_sequence;
    wire [63:0] shadow_completion_epoch;
    wire  shadow_result_valid;
    wire [63:0] shadow_result_q;
    wire  shadow_result_q_valid;
    ntrpu_attention_quantize_owner_r2 #(.WATCHDOG_CYCLES(WD)) shadow (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(command_valid),
        .command_ready(shadow_command_ready),
        .command_sequence(command_sequence),
        .command_epoch(command_epoch),
        .expected_epoch(expected_epoch),
        .cancel_valid(cancel_valid),
        .cancel_ready(shadow_cancel_ready),
        .busy(shadow_busy),
        .source_completion_valid(source_completion_valid),
        .source_completion_ready(shadow_source_completion_ready),
        .source_completion_status(source_completion_status),
        .source_sequence(source_sequence),
        .source_epoch(source_epoch),
        .source_result_valid(source_result_valid),
        .source_result_ce(shadow_source_result_ce),
        .source_result_address(shadow_source_result_address),
        .source_result_q(source_result_q),
        .source_result_q_valid(source_result_q_valid),
        .hls_start(shadow_hls_start),
        .hls_done(hls_done),
        .hls_ready(hls_ready),
        .hls_return(hls_return),
        .hls_elements(shadow_hls_elements),
        .hls_input_ce(hls_input_ce),
        .hls_input_address(hls_input_address),
        .hls_input_q(shadow_hls_input_q),
        .hls_output_ce(hls_output_ce),
        .hls_output_we(hls_output_we),
        .hls_output_address(hls_output_address),
        .hls_output_d(hls_output_d),
        .completion_valid(shadow_completion_valid),
        .completion_ready(completion_ready),
        .completion_status(shadow_completion_status),
        .completion_hls_status(shadow_completion_hls_status),
        .completion_sequence(shadow_completion_sequence),
        .completion_epoch(shadow_completion_epoch),
        .result_valid(shadow_result_valid),
        .result_ce(result_ce),
        .result_address(result_address),
        .result_q(shadow_result_q),
        .result_q_valid(shadow_result_q_valid)
    );
    integer equivalence_samples=0,private_reads=0,private_unrequested=0;
    always @(posedge clk)begin
        if(rst_n && dut.state==dut.RESPONSE)begin
            private_reads++;
            if(!result_ce || !result_valid)private_unrequested++;
        end
    end
    always @(posedge clk or negedge clk)begin
        #1.5;
        if(cycles>0)begin
            require(command_ready===shadow_command_ready,"Public equivalence: command_ready");
            require(cancel_ready===shadow_cancel_ready,"Public equivalence: cancel_ready");
            require(busy===shadow_busy,"Public equivalence: busy");
            require(source_completion_ready===shadow_source_completion_ready,"Public equivalence: source_completion_ready");
            require(source_result_ce===shadow_source_result_ce,"Public equivalence: source_result_ce");
            require(source_result_address===shadow_source_result_address,"Public equivalence: source_result_address");
            require(hls_start===shadow_hls_start,"Public equivalence: hls_start");
            require(hls_elements===shadow_hls_elements,"Public equivalence: hls_elements");
            require(hls_input_q===shadow_hls_input_q,"Public equivalence: hls_input_q");
            require(completion_valid===shadow_completion_valid,"Public equivalence: completion_valid");
            require(completion_status===shadow_completion_status,"Public equivalence: completion_status");
            require(completion_hls_status===shadow_completion_hls_status,"Public equivalence: completion_hls_status");
            require(completion_sequence===shadow_completion_sequence,"Public equivalence: completion_sequence");
            require(completion_epoch===shadow_completion_epoch,"Public equivalence: completion_epoch");
            require(result_valid===shadow_result_valid,"Public equivalence: result_valid");
            require(result_q===shadow_result_q,"Public equivalence: result_q");
            require(result_q_valid===shadow_result_q_valid,"Public equivalence: result_q_valid");
            equivalence_samples++;
        end
    end

    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"AQOWNER_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,current_mode,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [31:0] source_word(input int index);
        return 32'h3f000000 | 32'(jobs*4096+index);
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
        if(!rst_n)begin source_result_q_valid<=0;source_result_q<=0;end
        else begin
            source_result_q_valid<=source_result_ce && source_result_valid && !block_response;
            if(source_result_ce)begin source_result_q<=source_word(int'(source_result_address));reads++;end
            if(!result_valid)require(!result_q_valid && result_q===0,"Private result leak");
            if(source_completion_ready && source_completion_valid)begin
                require(!core_busy && !hls_start,"Source released before HLS drain");
                require(source_sequence==sequence_model && source_epoch==epoch_model,"Foreign source acknowledged");
                acks++;$display("AQOWNER_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,source_sequence,source_epoch);
                source_completion_valid<=0;source_result_valid<=0;
            end
            if(completion_valid && !completion_ready)held++;
            if(hls_output_ce && hls_output_we)writes++;
        end
    end
    task automatic peer_core;
        begin: work
            core_busy=1;core_reads=0;core_writes=0;starts++;
            require(hls_elements==2048 && source_completion_valid && source_result_valid,"HLS start admission");
            $display("AQOWNER_START cycle=%0d job=%0d",cycles,jobs);
            for(integer index=0;index<2048;index++)begin
                @(negedge clk);if(!rst_n)disable work;
                hls_input_ce=1;hls_input_address=11'(index);
                tick();if(!rst_n)disable work;
                if(dut.fault_now==0)require(hls_input_q===source_word(index),"Direct source response");
                else require(hls_input_q===0,"Revoked source response leak");
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
            for(integer index=0;index<(current_mode==18 ? 0:current_mode==13 ? 511:current_mode==20 ? 513:512);index++)begin
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
            hls_output_ce=0;hls_output_we=0;hls_done=1;hls_ready=current_mode!=19;hls_return=current_mode==18 ? 2:0;
            @(posedge clk);if(!rst_n)disable work;
            core_busy=0;dones++;$display("AQOWNER_DONE cycle=%0d job=%0d status=%0d",cycles,jobs,hls_return);
            #1;@(negedge clk);hls_done=0;hls_ready=0;
        end
        core_busy=0;hls_input_ce=0;hls_output_ce=0;hls_output_we=0;hls_done=0;hls_ready=0;
    endtask
    always begin @(posedge clk);if(rst_n && hls_start)peer_core();end
    task automatic reset_all;
        @(negedge clk);rst_n=0;resets++;
        $display("AQOWNER_RESET cycle=%0d job=%0d count=%0d",cycles,jobs,resets);
        command_valid=0;cancel_valid=0;completion_ready=0;result_ce=0;block_response=0;
        source_completion_valid=0;source_result_valid=0;
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(!busy && !completion_valid && !result_valid && !hls_start && !source_completion_ready && !core_busy,"Reset ownership");
    endtask
    function automatic integer wanted_status(input int mode);
        case(mode)
            1:return 3;
            2,3,4,8,23:return 2;
            5,9,10,11,19,33,34:return 11;
            6,7,22:return 7;
            12,13,14,15,16,17,20:return 4;
            18:return 5;
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
        $display("AQOWNER_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,sequence_model,epoch_model);
        tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;
        if(mode==6)begin cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if(mode>=7 && mode<=12)begin
            wait(core_reads>=64);@(negedge clk);
            case(mode)
                7:cancel_valid=1;
                8:expected_epoch=epoch_model^(64'd1<<48);
                9:source_sequence=sequence_model^(64'd1<<40);
                10:source_completion_valid=0;
                11:source_result_valid=0;
                12:block_response=1;
            endcase
            repeat(2)tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;
            source_sequence=sequence_model;source_completion_valid=1;source_result_valid=1;block_response=0;
            require(!completion_valid && !source_completion_ready,"Failed core abandoned before drain");
        end
        if(mode==22 || mode==23)begin
            wait(core_writes>=64);@(negedge clk);
            if(mode==22)cancel_valid=1;else expected_epoch=epoch_model^(64'd1<<48);
            tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;
        end
        if(mode>=31)begin
            if(mode<=32)wait(core_reads>=64);
            else begin wait(hls_done);@(posedge clk);#1;end
            @(negedge clk);
            if(mode==31 || mode==33 || mode==35)source_sequence=sequence_model^(64'd1<<40);
            else source_epoch=epoch_model^(64'd1<<44);
            $display("AQOWNER_FOREIGN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,source_sequence,source_epoch);
            if(mode<=32)begin wait(hls_done);@(posedge clk);#1;end
            repeat(32)begin
                tick();require(!source_completion_ready && !completion_valid && !result_valid && busy && !command_ready,"Foreign source released or published");
                require(acks==ack_before,"Foreign acknowledgement count");
            end
            $display("AQOWNER_HELD cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            if(mode==33 || mode==34)begin
                @(negedge clk);source_sequence=sequence_model;source_epoch=epoch_model;
                $display("AQOWNER_RESTORE cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,source_sequence,source_epoch);
            end
        end
        if((mode>=26 && mode<=30) || mode==31 || mode==32 || mode==35 || mode==36)begin
            if(mode==26)wait(core_reads>=64);
            else if(mode==27)wait(core_writes>=64);
            else if(mode==28)wait(hls_done);
            else if(mode==29)wait(completion_valid);
            // Mode 30 resets before the H_START acceptance edge.
            if(mode==30)rst_n=0;
            $display("AQOWNER_ABORT_RESET cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            reset_all();
        end else begin
            limit=0;
            while(!completion_valid)begin @(negedge clk);limit++;require(limit<20000,"Completion timeout");end
            saved=completion_status;
            require(saved==4'(wanted_status(mode)) && !core_busy && !source_completion_ready,"Completion status/drain");
            require(completion_sequence==sequence_model && completion_epoch==epoch_model,"Captured 64-bit identity");
            require(completion_hls_status==(mode==18 ? 2:0),"HLS status retained");
            require(acks-ack_before==((mode==2 || mode==3) ? 0:1),"Source ownership acknowledgement");
            require(starts-start_before==(mode<=6 && mode!=0 ? 0:1),"HLS admission count");
            repeat(7)begin tick();require(completion_valid && completion_status==saved && !command_ready,"Held completion");end
            if(saved==0)for(integer index=0;index<512;index++)begin
                @(negedge clk);result_ce=1;result_address=9'(index);tick();
                require(result_q_valid && result_q===record_word(index),"ABI1 result content/order");results++;
                $display("AQOWNER_RESULT cycle=%0d job=%0d index=%0d data=%016h",cycles,jobs,index,result_q);
                if(index%17==0)begin @(negedge clk);result_ce=0;result_address=9'(511-index);tick();require(!result_q_valid && result_q===0,"Unrequested speculative data leak");end
            end
            if(mode==24 || mode==25)begin
                @(negedge clk);
                if(mode==24)expected_epoch=epoch_model^(64'd1<<48);else cancel_valid=1;
                #1;require(!result_valid && !result_q_valid && result_q===0,"Late revocation leak");
                tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;tick();
                require(!result_valid && !result_q_valid && completion_status==saved,"Rollback revived result");
                $display("AQOWNER_REVOKED cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            end
            $display("AQOWNER_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,saved,completion_sequence,completion_epoch);
            completions++;@(negedge clk);result_ce=0;completion_ready=1;
            tick();@(negedge clk);completion_ready=0;source_completion_valid=0;source_result_valid=0;
            require(!busy && !completion_valid && !result_valid,"Owner release");
        end
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        reset_all();
        for(integer mode=0;mode<=36;mode++)begin run_job(mode);if(mode!=0)run_job(0);end
        require(jobs==73 && completions==64 && results==19968 && resets==10 && held>0,"Suite coverage");
        $display("AQOWNER_PASS seed=%0d watchdog=%0d jobs=%0d completions=%0d results=%0d starts=%0d dones=%0d acks=%0d resets=%0d reads=%0d writes=%0d held=%0d cycles=%0d",
            seed,WD,jobs,completions,results,starts,dones,acks,resets,reads,writes,held,cycles);
        require(equivalence_samples>360000 && private_reads>results && private_unrequested>1000,"Equivalence/speculation coverage");
        $display("AQBRAM_PASS seed=%0d outputs=%0d samples=%0d private_reads=%0d unrequested=%0d",seed,17,equivalence_samples,private_reads,private_unrequested);
        $finish;
    end
    initial begin #20000000;$fatal(1,"AQOWNER_FAIL global timeout");end
endmodule
`default_nettype wire
