// Controller-only RTL bench. The child is a deterministic protocol peer,
// not attention arithmetic, DDR, or the generated HLS core.
`timescale 1ns/1ps
module tb_ntrpu_attention_head_sequencer_r1;
    parameter integer WATCHDOG_CYCLES=32;
    logic clk, rst_n;
    logic command_valid;
    logic command_ready;
    logic [31:0] command_layer, command_position;
    logic [63:0] command_sequence, command_epoch, command_lease_base;
    logic cache_idle, cache_valid;
    logic [63:0] cache_sequence, cache_epoch, expected_epoch;
    logic [31:0] cache_capacity, cache_committed_tokens;
    logic cache_lock;
    logic q_valid;
    logic q_ready;
    logic [511:0] q_data;
    logic [7:0] q_index;
    logic q_last;
    logic [63:0] q_sequence, q_epoch;
    logic cancel_valid;
    logic cancel_ready;
    logic completion_valid;
    logic completion_ready;
    logic [3:0] completion_status;
    logic [63:0] completion_sequence, completion_epoch;
    logic result_valid;
    logic result_ce;
    logic [10:0] result_address;
    logic [31:0] result_q;
    logic result_q_valid;
    logic head_command_valid;
    logic head_command_ready;
    logic [2047:0] head_command_query;
    logic [31:0] head_command_layer, head_command_position, head_command_head;
    logic [63:0] head_command_sequence, head_command_epoch, head_command_lease;
    logic head_cancel_valid;
    logic head_cancel_ready;
    logic head_completion_valid;
    logic head_completion_ready;
    logic [3:0] head_completion_status;
    logic head_result_valid;
    logic head_result_ce;
    logic [5:0] head_result_address;
    logic [31:0] head_result_q;
    logic head_result_q_valid;
    ntrpu_attention_head_sequencer_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut (
        .clk,
        .rst_n,
        .command_valid,
        .command_ready,
        .command_layer,
        .command_position,
        .command_sequence,
        .command_epoch,
        .command_lease_base,
        .cache_idle,
        .cache_valid,
        .cache_sequence,
        .cache_epoch,
        .expected_epoch,
        .cache_capacity,
        .cache_committed_tokens,
        .cache_lock,
        .q_valid,
        .q_ready,
        .q_data,
        .q_index,
        .q_last,
        .q_sequence,
        .q_epoch,
        .cancel_valid,
        .cancel_ready,
        .completion_valid,
        .completion_ready,
        .completion_status,
        .completion_sequence,
        .completion_epoch,
        .result_valid,
        .result_ce,
        .result_address,
        .result_q,
        .result_q_valid,
        .head_command_valid,
        .head_command_ready,
        .head_command_query,
        .head_command_layer,
        .head_command_position,
        .head_command_head,
        .head_command_sequence,
        .head_command_epoch,
        .head_command_lease,
        .head_cancel_valid,
        .head_cancel_ready,
        .head_completion_valid,
        .head_completion_ready,
        .head_completion_status,
        .head_result_valid,
        .head_result_ce,
        .head_result_address,
        .head_result_q,
        .head_result_q_valid
    );
    integer seed, cycle_count=0, job=0, mode=0, child_count=0, released=0;
    integer completions=0, resets=0, compared=0, total_children=0;
    integer peer_head=0, compute_delay=0, read_delay=0, pending_address=0;
    logic peer_busy=0, read_pending=0, block_commands=0, case_active=0;
    logic [63:0] root_sequence, root_epoch, root_lease;
    logic [31:0] root_layer, root_position;
    always #2.5 clk=~clk;
    function automatic [31:0] query_word(input integer j,input integer index);
        query_word=32'h3e000000 | ((j&255)<<12) | index;
    endfunction
    function automatic [31:0] answer_word(input integer j,input integer head,input integer lane);
        answer_word=32'h3f000000 | ((j&255)<<12) | (head<<6) | lane;
    endfunction
    task automatic fail(input string message);
        $display("HEADSEQ_FAIL job=%0d mode=%0d cycle=%0d %s",job,mode,cycle_count,message);
        $fatal(1,"Head sequencer control failure");
    endtask

    assign head_command_ready=!peer_busy && !block_commands && ((cycle_count+seed)%5!=0);
    assign head_cancel_ready=peer_busy && !head_completion_valid;
    assign head_result_valid=head_completion_valid && head_completion_status==0;
    always @(posedge clk)begin
        cycle_count<=cycle_count+1;
        if(cycle_count>3000000)fail("Global cycle budget");
        if(!rst_n)begin
            peer_busy<=0;head_completion_valid<=0;head_completion_status<=0;
            head_result_q<=0;head_result_q_valid<=0;read_pending<=0;compute_delay<=0;read_delay<=0;
        end else begin
            head_result_q_valid<=0;
            if(case_active)begin
                if(result_valid && child_count!=32)fail("Partial publication");
                if(completion_valid && completion_status!=0 && result_valid)fail("Failed result visible");
                if(!result_q_valid && result_q!==0)fail("Invalid read not masked");
            end
            if(head_command_valid && head_command_ready)begin
                if(!cache_lock || peer_busy || child_count>=32)fail("Child ownership");
                if(head_command_head!==child_count || head_command_layer!==root_layer ||
                    head_command_position!==root_position || head_command_sequence!==root_sequence ||
                    head_command_epoch!==root_epoch || head_command_lease!==root_lease+child_count)
                    fail("Child command identity/order");
                for(integer lane=0;lane<64;lane=lane+1)
                    if(head_command_query[lane*32+:32]!==query_word(job,child_count*64+lane))fail("Query concat/order");
                $display("HEADSEQ_CHILD job=%0d head=%0d lease=%016h",job,head_command_head,head_command_lease);
                peer_head<=child_count;child_count<=child_count+1;total_children<=total_children+1;
                peer_busy<=1;compute_delay<=5+((child_count+seed)%7);
                head_completion_status<=0;head_completion_valid<=0;
            end
            if(peer_busy && !head_completion_valid)begin
                if(head_cancel_valid && head_cancel_ready)begin
                    head_completion_valid<=1;head_completion_status<=7;
                end else if(compute_delay==0)begin
                    head_completion_valid<=1;
                    head_completion_status<=(mode==11 && peer_head==7) ? 4'd3:4'd0;
                end else compute_delay<=compute_delay-1;
            end
            if(head_result_ce)begin
                if(!head_result_valid || read_pending)fail("Read ownership/outstanding");
                read_pending<=1;pending_address<=head_result_address;
                read_delay<=1+((head_result_address+seed)%3);
            end
            if(read_pending && !(mode==13 && peer_head==3))begin
                if(read_delay==0)begin
                    head_result_q<=(mode==12 && peer_head==5 && pending_address==7) ?
                        32'h7fc00001:answer_word(job,peer_head,pending_address);
                    head_result_q_valid<=1;read_pending<=0;
                end else read_delay<=read_delay-1;
            end
            if(head_completion_valid && head_completion_ready)begin
                $display("HEADSEQ_RELEASE job=%0d head=%0d status=%0d",job,peer_head,head_completion_status);
                released<=released+1;peer_busy<=0;head_completion_valid<=0;
                read_pending<=0;head_result_q_valid<=0;
            end
        end
    end

    task automatic reset_all;
        @(negedge clk);rst_n=0;case_active=0;command_valid=0;q_valid=0;
        cancel_valid=0;completion_ready=0;result_ce=0;block_commands=0;
        repeat(3)@(posedge clk);
        @(negedge clk);rst_n=1;resets=resets+1;
        @(posedge clk);#1;
        if(cache_lock || completion_valid || result_valid || result_q_valid || peer_busy)fail("Reset retained ownership");
        $display("HEADSEQ_RESET count=%0d job=%0d",resets,job);
    endtask
    task automatic begin_job(input integer scenario);
        @(negedge clk);job=job+1;mode=scenario;case_active=1;child_count=0;released=0;
        root_sequence=64'h1234000000000000+job;root_epoch=64'hfedc000000000000+job;
        root_lease=64'h80000000+job*64;root_layer=2;root_position=4;
        command_layer=root_layer;command_position=root_position;command_sequence=root_sequence;
        command_epoch=root_epoch;command_lease_base=root_lease;
        cache_idle=1;cache_valid=1;cache_sequence=root_sequence;cache_epoch=root_epoch;expected_epoch=root_epoch;
        cache_capacity=256;cache_committed_tokens=5;cancel_valid=0;block_commands=0;
        q_valid=0;completion_ready=0;result_ce=0;
        if(mode==8)command_lease_base=64'hffffffffffffffe1;
        if(mode==9)cache_sequence=root_sequence^1;
        if(mode==10)cache_committed_tokens=4;
        if(mode==33)command_layer=16;
        if(mode==34)command_lease_base=0;
        command_valid=1;
        if(mode==30)begin
            cache_idle=0;repeat(6)begin @(posedge clk);#1;if(command_ready || cache_lock)fail("Cache maintenance admission");end
            @(negedge clk);cache_idle=1;
        end
        do @(posedge clk);while(!command_ready);
        @(negedge clk);command_valid=0;
        // External command inputs may change immediately after acceptance.
        command_layer=15;command_position=123;command_sequence=0;command_epoch=0;command_lease_base=0;
        if(!cache_lock)fail("Missing root ownership");
        $display("HEADSEQ_BEGIN job=%0d mode=%0d",job,mode);
    endtask
    task automatic load_queries(input integer count);
        for(integer index=0;index<count && !completion_valid;index=index+1)begin
            if((index+seed)%9==0)repeat(2)@(negedge clk);
            @(negedge clk);q_valid=1;q_index=index;q_last=index==127;
            q_sequence=root_sequence;q_epoch=root_epoch;
            for(integer lane=0;lane<16;lane=lane+1)q_data[lane*32+:32]=query_word(job,index*16+lane);
            if(mode==2 && index==17)q_index=16;
            if(mode==3 && index==37)q_sequence=root_sequence^1;
            if(mode==4 && index==127)q_data[15*32+:32]=32'h7fc00001;
            if(mode==5 && index==63)q_last=1;
            if(mode==6 && index==127)q_last=0;
            if(mode==31 && index==45)q_epoch=root_epoch^1;
            if(mode==32 && index==18)q_index=200;
            do @(posedge clk);while(!q_ready && !completion_valid);
            @(negedge clk);q_valid=0;
        end
    endtask
    task automatic wait_completion(input integer expected_status,input integer expected_children);
        integer deadline;
        deadline=cycle_count+WATCHDOG_CYCLES+100000;
        while(!completion_valid)begin
            @(posedge clk);#1;
            if(cycle_count>deadline)fail("Root completion timeout");
        end
        if(completion_status!==expected_status || child_count!=expected_children || released!=expected_children ||
            peer_busy || completion_sequence!==root_sequence || completion_epoch!==root_epoch)
            fail($sformatf("Completion expected=%0d got=%0d children=%0d expected=%0d released=%0d",expected_status,completion_status,child_count,expected_children,released));
        repeat(5)begin
            @(posedge clk);#1;
            if(!completion_valid || completion_status!==expected_status || !cache_lock || command_ready)
                fail("Unstable held completion");
        end
    endtask
    task automatic read_results;
        for(integer index=0;index<2048;index=index+1)begin
            if((index+seed)%11==0)@(negedge clk);
            @(negedge clk);result_ce=1;result_address=index;
            @(posedge clk);#1;
            if(!result_valid || !result_q_valid || result_q!==answer_word(job,index/64,index%64))fail("Result concat/order");
            $display("HEADSEQ_WORD job=%0d index=%0d data=%08h",job,index,result_q);compared=compared+1;
            @(negedge clk);result_ce=0;
            @(posedge clk);#1;if(result_q_valid || result_q!==0)fail("Read response valid duration");
        end
    endtask
    task automatic finish_job;
        $display("HEADSEQ_DONE job=%0d mode=%0d status=%0d children=%0d",job,mode,completion_status,child_count);
        @(negedge clk);completion_ready=1;
        @(posedge clk);#1;
        if(cache_lock || result_valid || result_q_valid || completion_valid)fail("Consumed result retained");
        @(negedge clk);completion_ready=0;cancel_valid=0;case_active=0;completions=completions+1;
    endtask
    task automatic success(input integer scenario);
        begin_job(scenario);load_queries(128);wait_completion(0,32);read_results();finish_job();
    endtask
    task automatic revoke_held(input integer use_cancel);
        @(negedge clk);result_ce=1;result_address=0;
        if(use_cancel)cancel_valid=1;else expected_epoch=root_epoch^1;
        #1;if(result_valid || result_q_valid || result_q!==0)fail("Late revoke not immediate");
        @(posedge clk);#1;
        @(negedge clk);cancel_valid=0;expected_epoch=root_epoch;
        repeat(4)begin @(posedge clk);#1;
            if(result_valid || result_q_valid || result_q!==0 || !completion_valid || completion_status!=0)
                fail("Late revoke revived result or changed completion");
        end
        @(negedge clk);result_ce=0;
    endtask

    initial begin
        clk=0;rst_n=0;seed=1;
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        command_valid=0;q_valid=0;cancel_valid=0;completion_ready=0;result_ce=0;
        q_data=0;q_index=0;q_last=0;q_sequence=0;q_epoch=0;result_address=0;
        cache_idle=1;cache_valid=0;cache_sequence=0;cache_epoch=0;expected_epoch=0;
        cache_capacity=0;cache_committed_tokens=0;
        $display("HEADSEQ_CONFIG seed=%0d watchdog=%0d arithmetic_peer=0",seed,WATCHDOG_CYCLES);
        reset_all();success(0);success(1);
        for(integer scenario=2;scenario<=6;scenario=scenario+1)begin
            begin_job(scenario);load_queries(128);wait_completion(scenario==3 ? 2:4,0);finish_job();
        end
        begin_job(7);load_queries(3);wait_completion(8,0);finish_job();
        for(integer scenario=8;scenario<=10;scenario=scenario+1)begin
            begin_job(scenario);wait_completion(scenario==9 ? 2:1,0);finish_job();
        end
        begin_job(11);load_queries(128);wait_completion(3,8);finish_job();
        begin_job(12);load_queries(128);wait_completion(4,6);finish_job();
        begin_job(13);load_queries(128);wait_completion(8,4);finish_job();
        begin_job(14);load_queries(3);@(negedge clk);cancel_valid=1;wait_completion(7,0);finish_job();
        begin_job(15);load_queries(128);wait(peer_busy && peer_head==4 && !head_completion_valid);
        @(negedge clk);cancel_valid=1;wait_completion(7,5);finish_job();
        begin_job(16);load_queries(128);wait(peer_busy && peer_head==2 && !head_completion_valid);
        @(negedge clk);expected_epoch=root_epoch^1;wait_completion(2,3);finish_job();
        begin_job(17);load_queries(128);wait(head_result_ce && peer_head==6 && head_result_address==10);
        @(negedge clk);cancel_valid=1;wait_completion(7,7);finish_job();
        begin_job(18);load_queries(128);wait(head_result_ce && peer_head==8 && head_result_address==10);
        @(negedge clk);expected_epoch=root_epoch^1;wait_completion(2,9);finish_job();
        begin_job(19);load_queries(128);wait_completion(0,32);read_results();revoke_held(0);finish_job();
        begin_job(20);load_queries(128);wait_completion(0,32);read_results();revoke_held(1);finish_job();
        begin_job(21);load_queries(20);reset_all();success(22);
        begin_job(23);load_queries(128);wait(peer_busy && peer_head==3 && !head_completion_valid);reset_all();success(24);
        begin_job(25);load_queries(128);wait(head_result_ce && peer_head==5 && head_result_address==10);reset_all();success(26);
        begin_job(27);load_queries(128);wait_completion(0,32);reset_all();success(28);
        begin_job(29);block_commands=1;load_queries(128);wait_completion(8,0);finish_job();
        success(30);
        for(integer scenario=31;scenario<=32;scenario=scenario+1)begin
            begin_job(scenario);load_queries(128);wait_completion(scenario==31 ? 2:4,0);finish_job();
        end
        for(integer scenario=33;scenario<=34;scenario=scenario+1)begin
            begin_job(scenario);wait_completion(1,0);finish_job();
        end
        if(job!=35 || completions!=31 || resets!=5 || compared!=18432)fail("Summary coverage");
        $display("HEADSEQ_PASS jobs=%0d completions=%0d resets=%0d output_words=%0d children=%0d cycles=%0d",
            job,completions,resets,compared,total_children,cycle_count);
        $finish;
    end
endmodule
