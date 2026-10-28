`timescale 1ns/1ps
`default_nettype none
// Protocol peer only. The actual matrix engine and arithmetic are separate gates.
module tb_ntrpu_attention_projection_control_r1 #(parameter bit EXHAUSTION=0);
    localparam integer WD=24;
    localparam logic [63:0] INITIAL_TAG=EXHAUSTION ? 64'hffffffffffffffc0:64'h00000000ffffff80;
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic command_valid=0,command_ready,cancel_valid=0,cancel_ready,busy;
    logic [63:0] command_sequence=0,command_epoch=0,expected_epoch=0;
    logic [63:0] command_weight_addr=0,command_weight_region_base=0,command_weight_region_bytes=0;
    logic [63:0] command_activation_addr=0,command_activation_region_base=0,command_activation_region_bytes=0;
    logic [63:0] command_output_addr=0,command_output_region_base=0,command_output_region_bytes=0;
    logic completion_valid,completion_ready=0,result_valid,tag_exhausted;
    logic [3:0] completion_status;
    logic [63:0] completion_sequence,completion_epoch,result_address;
    logic [16:0] result_bytes;
    logic child_job_valid,child_job_ready=0,child_job_output_fp32;
    logic [1:0] child_job_layout;
    logic [13:0] child_job_rows;
    logic [8:0] child_job_k_blocks;
    logic [5:0] child_job_tile_k;
    logic [31:0] child_job_sequence,child_job_epoch,child_job_activation_epoch,child_job_output_epoch;
    logic [63:0] child_job_weight_addr,child_job_region_base,child_job_region_bytes;
    logic [63:0] child_job_activation_addr,child_job_activation_region_base,child_job_activation_region_bytes;
    logic [63:0] child_job_output_addr,child_job_output_region_base,child_job_output_region_bytes;
    logic child_cancel_valid,child_cancel_ready=1,child_activity=0,child_done_valid=0,child_done_ready;
    logic [3:0] child_done_status=0;
    logic [16:0] child_done_bytes=0;
    logic [13:0] child_done_rows=0;
    logic [31:0] child_done_sequence=0,child_done_epoch=0;
    ntrpu_attention_projection_control_r1 #(.WATCHDOG_CYCLES(WD),.INITIAL_CHILD_TAG(INITIAL_TAG)) dut(.*);
    integer seed=1,random_state,cycles=0,jobs=0,current_mode=0,children=0,replies=0,cancels=0,publics=0,resets=0,spans=0;
    logic [63:0] next_model=INITIAL_TAG,tag_model,parent_sequence,parent_epoch;
    logic [63:0] weight_model,activation_model,output_model;
    logic [575:0] descriptor_model;
    logic exhausted_model=0;

    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"APROJ_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,current_mode,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    always @(posedge clk)begin
        cycles++;
        if(rst_n)begin
            if(child_job_valid && child_job_ready)begin
                children++;$display("APROJ_CHILD cycle=%0d job=%0d sequence=%08h epoch=%08h",cycles,jobs,child_job_sequence,child_job_epoch);
            end
            if(child_done_valid && child_done_ready)begin
                replies++;$display("APROJ_ACK cycle=%0d job=%0d status=%0d bytes=%0d rows=%0d sequence=%08h epoch=%08h",
                    cycles,jobs,child_done_status,child_done_bytes,child_done_rows,child_done_sequence,child_done_epoch);
            end
            if(child_cancel_valid && child_cancel_ready)begin
                cancels++;$display("APROJ_CANCEL cycle=%0d job=%0d",cycles,jobs);
            end
            if(!result_valid)require(result_bytes==0 && result_address==0,"Private span leak");
            if(result_valid)require(completion_valid && completion_status==0 && completion_epoch==expected_epoch,"Publication authority");
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;resets++;
        command_valid=0;cancel_valid=0;completion_ready=0;child_job_ready=0;child_done_valid=0;child_activity=0;child_cancel_ready=1;
        $display("APROJ_RESET cycle=%0d job=%0d count=%0d",cycles,jobs,resets);
        repeat(4)tick();@(negedge clk);rst_n=1;repeat(2)tick();
        next_model=INITIAL_TAG;exhausted_model=0;
        require(command_ready && !busy && !completion_valid && !result_valid && !child_job_valid && !child_cancel_valid,"Coordinated reset");
    endtask
    task automatic descriptor_check;
        require(child_job_layout==2 && child_job_rows==2048 && child_job_output_fp32 && child_job_k_blocks==64 && child_job_tile_k==32,"Matrix shape");
        require({child_job_epoch,child_job_sequence}==tag_model && child_job_sequence[5:0]==0,"Internal reserved tag");
        require(child_job_activation_epoch==child_job_epoch && child_job_output_epoch==child_job_epoch,"Internal memory epochs");
        require({child_job_weight_addr,child_job_region_base,child_job_region_bytes,
            child_job_activation_addr,child_job_activation_region_base,child_job_activation_region_bytes,
            child_job_output_addr,child_job_output_region_base,child_job_output_region_bytes}===descriptor_model,"Captured descriptor");
    endtask
    function automatic integer wanted_status(input int mode);
        case(mode)
            1,3,5,21:return 2;
            2,4,20,22:return 7;
            6,7:return 11;
            8:return 3;
            9,10,11,12:return 5;
            13,14:return 8;
            24:return 9;
            default:return 0;
        endcase
    endfunction
    task automatic finish_response(input int mode);
        integer limit;logic [3:0] saved;
        limit=0;
        while(!completion_valid)begin tick();limit++;require(limit<12,"Missing completion");end
        saved=completion_status;
        require(saved==4'(wanted_status(mode)),"Completion status");
        require(completion_sequence==parent_sequence && completion_epoch==parent_epoch,"Captured full identity");
        repeat(7)begin tick();require(completion_valid && !command_ready && completion_status==saved,"Held completion");end
        if(saved==0)begin
            require(result_valid && result_bytes==8192 && result_address==output_model,"Completed output span");spans++;
        end else require(!result_valid,"Failed output published");
        if(mode==15 || mode==16)begin
            @(negedge clk);if(mode==15)expected_epoch=parent_epoch^(64'd1<<48);else cancel_valid=1;
            #1;require(!result_valid && result_bytes==0,"Immediate late revocation");
            tick();@(negedge clk);expected_epoch=parent_epoch;cancel_valid=0;tick();
            require(!result_valid && completion_status==saved,"Rollback revived output");
            $display("APROJ_REVOKED cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
        end
        $display("APROJ_PUBLIC cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h valid=%0d address=%016h bytes=%0d",
            cycles,jobs,mode,saved,completion_sequence,completion_epoch,result_valid,result_address,result_bytes);publics++;
        @(negedge clk);completion_ready=1;tick();@(negedge clk);completion_ready=0;cancel_valid=0;expected_epoch=parent_epoch;
        require(command_ready && !busy && !result_valid,"Owner release");
    endtask
    task automatic run_job(input int mode);
        begin: work
            jobs++;current_mode=mode;
            parent_sequence=64'h13579bdf00000000+64'(jobs);parent_epoch=64'h2468ace000000000+64'(jobs);
            weight_model=64'h0000001100000000+64'(jobs)*64'h1000000;
            activation_model=64'h0000002200000000+64'(jobs)*64'h10000;
            output_model=64'h0000003300000000+64'(jobs)*64'h10000;
            @(negedge clk);command_sequence=parent_sequence;command_epoch=parent_epoch;expected_epoch=parent_epoch;
            command_weight_addr=weight_model;command_weight_region_base=weight_model-4032;command_weight_region_bytes=4456448+4032;
            command_activation_addr=activation_model;command_activation_region_base=activation_model;command_activation_region_bytes=4096;
            command_output_addr=output_model;command_output_region_base=output_model;command_output_region_bytes=8192;
            descriptor_model={command_weight_addr,command_weight_region_base,command_weight_region_bytes,
                command_activation_addr,command_activation_region_base,command_activation_region_bytes,
                command_output_addr,command_output_region_base,command_output_region_bytes};
            if(mode==1)expected_epoch=parent_epoch^(64'd1<<48);
            require(command_ready && !child_job_valid,"Admission readiness");command_valid=1;
            $display("APROJ_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,parent_sequence,parent_epoch);
            tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;
            command_weight_addr=0;command_weight_region_base=0;command_weight_region_bytes=0;
            command_activation_addr=0;command_activation_region_base=0;command_activation_region_bytes=0;
            command_output_addr=0;command_output_region_base=0;command_output_region_bytes=0;
            if(mode==1 || mode==24)begin
                require(!child_job_valid,"Rejected command reached child");finish_response(mode);disable work;
            end
            require(!exhausted_model,"Unplanned exhausted command");tag_model=next_model;
            if(next_model==64'hffffffffffffffc0)exhausted_model=1;else next_model+=64'd64;
            require(tag_exhausted==exhausted_model,"Exhaustion state");
            descriptor_check();
            $display("APROJ_OFFER cycle=%0d job=%0d sequence=%08h epoch=%08h weight=%016h activation=%016h output=%016h",
                cycles,jobs,child_job_sequence,child_job_epoch,child_job_weight_addr,child_job_activation_addr,child_job_output_addr);
            if(mode==2)cancel_valid=1;
            if(mode==3)expected_epoch=parent_epoch^(64'd1<<48);
            repeat(mode==13 ? WD+3:3)begin tick();descriptor_check();require(child_job_valid && !completion_valid,"Offer withdrawn before acceptance");end
            @(negedge clk);cancel_valid=0;expected_epoch=parent_epoch;
            if(mode==17)begin
                $display("APROJ_ABORT_RESET cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);reset_all();disable work;
            end
            child_job_ready=1;tick();@(negedge clk);child_job_ready=0;child_activity=1;
            require(!child_job_valid && !completion_valid,"Accepted child ownership");
            repeat($urandom_range(2,6))begin tick();descriptor_check();end
            @(negedge clk);child_activity=0;
            if(mode==18)begin
                $display("APROJ_ABORT_RESET cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);reset_all();disable work;
            end
            if(mode==4 || mode==20)cancel_valid=1;
            if(mode==5)expected_epoch=parent_epoch^(64'd1<<48);
            if(mode==20)child_cancel_ready=0;
            repeat(mode==14 ? WD+3:3)begin
                tick();require(!completion_valid,"Owner released before matching drain");
                if(mode==20)require(child_cancel_valid,"Cancellation not held");
            end
            @(negedge clk);child_cancel_ready=1;cancel_valid=0;expected_epoch=parent_epoch;
            child_done_sequence=tag_model[31:0];child_done_epoch=tag_model[63:32];
            child_done_status=mode==8 || mode==11 ? 3:0;
            child_done_bytes=mode==8 ? 64:mode==9 ? 8191:mode==11 ? 8193:8192;
            child_done_rows=mode==8 ? 16:mode==10 ? 2047:mode==12 ? 2049:2048;
            if(mode==6 || mode==7 || mode==23)begin
                child_done_valid=1;
                if(mode==7)child_done_epoch^=32'h80000000;else child_done_sequence^=32'h80000000;
                repeat(mode==23 ? WD+4:2)begin tick();require(!child_done_ready && !completion_valid && !result_valid,"Foreign reply released owner");end
                if(mode==23)begin
                    $display("APROJ_ABORT_RESET cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);reset_all();disable work;
                end
                @(negedge clk);child_done_sequence=tag_model[31:0];child_done_epoch=tag_model[63:32];
            end
            if(mode==21)expected_epoch=parent_epoch^(64'd1<<48);
            if(mode==22)cancel_valid=1;
            child_done_valid=1;tick();@(negedge clk);child_done_valid=0;cancel_valid=0;expected_epoch=parent_epoch;
            if(mode==19)begin
                require(completion_valid && result_valid,"Reset response phase");
                $display("APROJ_ABORT_RESET cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);reset_all();disable work;
            end
            finish_response(mode);
        end
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        $display("APROJ_CONFIG seed=%0d watchdog=%0d exhaustion=%0d initial=%016h arithmetic_peer=0",seed,WD,EXHAUSTION,INITIAL_TAG);
        reset_all();
        if(EXHAUSTION)begin
            run_job(0);run_job(24);run_job(24);reset_all();run_job(0);
            require(jobs==4 && children==2 && replies==2 && publics==4 && spans==2 && resets==2,"Exhaustion coverage");
        end else begin
            @(negedge clk);child_done_valid=1;child_done_sequence=32'hf00df00d;
            repeat(3)begin tick();require(!command_ready && !child_done_ready && !child_job_valid,"Orphan reply admitted command");end
            $display("APROJ_ORPHAN cycle=%0d job=0",cycles);reset_all();
            for(integer mode=0;mode<=23;mode++)begin run_job(mode);if(mode!=0)run_job(0);end
            require(jobs==47 && children==45 && replies==43 && publics==43 && spans==26 && resets==6,"Main coverage");
        end
        $display("APROJ_PASS seed=%0d exhaustion=%0d jobs=%0d children=%0d replies=%0d cancels=%0d publics=%0d spans=%0d resets=%0d cycles=%0d",
            seed,EXHAUSTION,jobs,children,replies,cancels,publics,spans,resets,cycles);
        $finish;
    end
    initial begin #1000000;$fatal(1,"APROJ_FAIL global timeout");end
endmodule
`default_nettype wire
