`timescale 1ns/1ps
// Independent RAM consumer, not the generated attention HLS core.
module tb_ntrpu_attention_kv_ram_r1 #(parameter integer WATCHDOG_CYCLES=32);
    logic clk=0,rst_n=0,cmd_valid=0,cmd_ready,load_prepared;
    logic [31:0] cmd_start=0,cmd_tokens=0,cmd_position=0,cmd_layer=0,cmd_query_head=0;
    logic [63:0] cmd_sequence=0,cmd_epoch=0,cmd_lease=0,expected_epoch=100;
    logic tile_valid=0,tile_ready,tile_plane=0,tile_last=0;
    logic [511:0] tile_data=0;
    logic [7:0] tile_index=0;
    logic [63:0] tile_sequence=0,tile_epoch=0,tile_lease=0;
    logic load_done_valid=0,load_done_ready;
    logic [3:0] load_done_status=0;
    logic [9:0] load_done_beats=0;
    logic [15:0] load_done_bytes=0;
    logic [63:0] load_done_sequence=0,load_done_epoch=0,load_done_lease=0;
    logic cancel_valid=0,cancel_ready,upstream_cancel_valid,upstream_cancel_ready=0;
    logic compute_valid,compute_ready=0,compute_abort;
    logic [31:0] compute_start,compute_tokens,compute_position,compute_layer,compute_query_head,compute_kv_head;
    logic [63:0] compute_sequence,compute_epoch,compute_lease;
    logic keys_ce=0,values_ce=0;
    logic [11:0] keys_address=0,values_address=0;
    logic [31:0] keys_q,values_q;
    logic keys_q_valid,values_q_valid;
    logic finish_valid=0,finish_ready;
    logic [3:0] finish_status=0;
    logic done_valid,done_ready=0,done_consumer_started;
    logic [3:0] done_status;
    logic [9:0] done_beats;
    logic [63:0] done_sequence,done_epoch,done_lease;
    ntrpu_attention_kv_ram_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut(.*);
    always #2.5 clk=~clk;
    integer seed=1,job=0,kind=0,resets=0,commands=0,tiles=0,commits=0,computes=0,reads=0,dones=0;
    integer sent=0,ntokens=0,cycles=0;
    logic [31:0] random_state;
    function automatic integer randmod(input integer n);
        random_state=random_state ^ (random_state<<13);
        random_state=random_state ^ (random_state>>17);
        random_state=random_state ^ (random_state<<5);
        randmod=integer'(random_state % 32'(n));
    endfunction
    function automatic logic [31:0] word(input integer j,p,a);
        // Distinguishes job, plane, every row, and every lane; always finite FP32.
        word=32'h3f000000 ^ (32'(j)<<16) ^ (32'(p)<<15) ^ 32'(a);
    endfunction
    task automatic tick; @(posedge clk); #1; endtask
    task automatic require(input logic condition,input string message);
        if(condition!==1'b1)$fatal(1,"AKR_FAIL job=%0d kind=%0d cycle=%0d %s",job,kind,cycles,message);
    endtask
    task automatic hidden;
        require(!keys_q_valid && !values_q_valid && keys_q===0 && values_q===0,"private RAM exposed");
        if(load_prepared)require(!compute_valid,"incomplete tile published");
    endtask
    always @(posedge clk)begin
        cycles=cycles+1;
        if(rst_n)begin
            if(cmd_valid && cmd_ready)begin
                commands=commands+1;
                $display("AKR_CMD job=%0d kind=%0d start=%0d tokens=%0d position=%0d layer=%0d head=%0d seq=%0d epoch=%0d lease=%0d expected=%0d",job,kind,cmd_start,cmd_tokens,cmd_position,cmd_layer,cmd_query_head,cmd_sequence,cmd_epoch,cmd_lease,expected_epoch);
            end
            if(tile_valid && tile_ready)begin
                tiles=tiles+1;
                $display("AKR_TILE job=%0d plane=%0d index=%0d last=%0d seq=%0d epoch=%0d lease=%0d data=%0128h",job,tile_plane,tile_index,tile_last,tile_sequence,tile_epoch,tile_lease,tile_data);
            end
            if(load_done_valid && load_done_ready)begin
                commits=commits+1;
                $display("AKR_COMMIT job=%0d status=%0d beats=%0d bytes=%0d seq=%0d epoch=%0d lease=%0d",job,load_done_status,load_done_beats,load_done_bytes,load_done_sequence,load_done_epoch,load_done_lease);
            end
            if(compute_valid && compute_ready)begin
                computes=computes+1;
                $display("AKR_COMPUTE job=%0d start=%0d tokens=%0d position=%0d layer=%0d head=%0d kv=%0d seq=%0d epoch=%0d lease=%0d",job,compute_start,compute_tokens,compute_position,compute_layer,compute_query_head,compute_kv_head,compute_sequence,compute_epoch,compute_lease);
            end
            if(done_valid && done_ready)begin
                dones=dones+1;
                $display("AKR_DONE job=%0d status=%0d beats=%0d started=%0d seq=%0d epoch=%0d lease=%0d",job,done_status,done_beats,done_consumer_started,done_sequence,done_epoch,done_lease);
            end
        end
        if(cycles>2000000)$fatal(1,"AKR_FAIL global watchdog");
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;cmd_valid=0;tile_valid=0;load_done_valid=0;
        cancel_valid=0;upstream_cancel_ready=0;compute_ready=0;
        keys_ce=0;values_ce=0;finish_valid=0;done_ready=0;expected_epoch=100;
        repeat(3)tick();hidden();require(cmd_ready,"reset did not release command");
        require(!compute_valid && !compute_abort && !done_valid,"reset leaked ownership");
        resets=resets+1;$display("AKR_RESET job=%0d kind=%0d",job,kind);
        @(negedge clk);rst_n=1;tick();
    endtask
    task automatic begin_job(input integer case_kind,tokens,head);
        job=job+1;kind=case_kind;ntokens=tokens;sent=0;
        @(negedge clk);require(cmd_ready,"previous job retained");
        cmd_start=32'(job%5)*64;cmd_tokens=32'(tokens);cmd_position=cmd_start+32'(tokens)-1;
        cmd_layer=32'(job%16);cmd_query_head=32'(head);cmd_sequence=64'(1000+job);
        cmd_epoch=100;cmd_lease=64'(10000+job);expected_epoch=100;
        case(kind)
            1:cmd_tokens=0;
            2:cmd_tokens=65;
            3:cmd_position=cmd_start;
            4:cmd_position=131072;
            5:cmd_layer=16;
            6:cmd_query_head=32;
            7:cmd_sequence=0;
            8:cmd_epoch=0;
            9:cmd_lease=0;
            10:expected_epoch=101;
            11:begin cmd_start=32'hffffffff;cmd_position=0;end
        endcase
        cmd_valid=1;keys_ce=1;values_ce=1;keys_address=0;values_address=0;
        tick();hidden();require(!cmd_ready,"command accepted twice");
        @(negedge clk);cmd_valid=0;keys_ce=0;values_ce=0;
        if(kind==0 || kind>=12)require(load_prepared && tile_ready,"valid command did not prepare load");
        else require(done_valid && !load_prepared && !tile_ready,"bad command started load");
    endtask
    task automatic set_commit(input integer beats,status);
        load_done_valid=1;load_done_status=4'(status);load_done_beats=10'(beats);load_done_bytes=16'(beats*64);
        load_done_sequence=cmd_sequence;load_done_epoch=cmd_epoch;load_done_lease=cmd_lease;
        case(kind)
            19:load_done_sequence=cmd_sequence+1;
            20:load_done_epoch=cmd_epoch+1;
            21:load_done_lease=cmd_lease+1;
            22:load_done_beats=10'(beats+1);
            23:load_done_bytes=16'(beats*64-1);
        endcase
    endtask
    task automatic beat(input integer ordinal,input logic same_commit);
        @(negedge clk);tile_valid=1;tile_plane=ordinal>=ntokens*4;
        tile_index=8'(ordinal%(ntokens*4));tile_last=ordinal==ntokens*8-1;
        tile_sequence=cmd_sequence;tile_epoch=cmd_epoch;tile_lease=cmd_lease;
        for(int lane=0;lane<16;lane++)tile_data[lane*32+:32]=word(job,integer'(tile_plane),integer'(tile_index)*16+lane);
        if(ordinal==1)case(kind)
            12:tile_plane=!tile_plane;
            13:tile_index=tile_index+1;
            14:tile_last=1;
            15:tile_sequence=cmd_sequence+1;
            16:tile_epoch=cmd_epoch+1;
            17:tile_lease=cmd_lease+1;
            18:tile_data[32*7+:32]=32'h7fc00001;
        endcase
        if(kind==46 && ordinal==ntokens*8-1)tile_last=0;
        if(same_commit)set_commit(sent+1,kind==24 ? 3:0);
        require(tile_ready,"load stopped draining");tick();sent=sent+1;hidden();
        @(negedge clk);tile_valid=0;load_done_valid=0;
        repeat(randmod(3))begin tick();hidden();end
    endtask
    task automatic commit_load(input integer status);
        @(negedge clk);set_commit(sent,status);require(load_done_ready,"completion not accepted");
        tick();hidden();@(negedge clk);load_done_valid=0;
    endtask
    task automatic load_complete(input logic same_commit);
        for(int n=0;n<ntokens*8;n++)beat(n,same_commit && n==ntokens*8-1);
        if(!same_commit)commit_load(kind==24 ? 3:0);
    endtask
    task automatic start_compute;
        require(compute_valid && !cmd_ready && !load_prepared && !done_valid,"tile not available atomically");
        // Keep descriptor/ownership stable under consumer backpressure and hostile inputs.
        @(negedge clk);cmd_valid=1;tile_valid=1;load_done_valid=1;keys_ce=1;values_ce=1;
        repeat(5)begin tick();hidden();require(compute_valid && !tile_ready && !load_done_ready && !cmd_ready,"available tile changed");end
        require(compute_start==cmd_start && compute_tokens==cmd_tokens && compute_position==cmd_position &&
            compute_layer==cmd_layer && compute_query_head==cmd_query_head && compute_kv_head==cmd_query_head/4 &&
            compute_sequence==cmd_sequence && compute_epoch==cmd_epoch && compute_lease==cmd_lease,"descriptor mismatch");
        @(negedge clk);cmd_valid=0;tile_valid=0;load_done_valid=0;keys_ce=0;values_ce=0;compute_ready=1;
        tick();require(finish_ready && done_consumer_started,"consumer not started");hidden();
        @(negedge clk);compute_ready=0;
    endtask
    task automatic read_pair(input integer ka,va,input logic kc,vc);
        @(negedge clk);keys_ce=kc;values_ce=vc;keys_address=12'(ka);values_address=12'(va);
        tick();reads=reads+1;
        require(!cmd_ready && !tile_ready && !load_done_ready && !compute_valid && !done_valid,"executing RAM reused");
        require(keys_q_valid===kc && values_q_valid===vc,"RAM latency/enable mismatch");
        require(keys_q===(kc ? word(job,0,ka):32'b0) && values_q===(vc ? word(job,1,va):32'b0),"RAM data mismatch");
        $display("AKR_READ job=%0d ka=%0d va=%0d kc=%0d vc=%0d kvalid=%0d vvalid=%0d k=%08h v=%08h",job,ka,va,kc,vc,keys_q_valid,values_q_valid,keys_q,values_q);
    endtask
    task automatic finish_compute(input integer status);
        @(negedge clk);keys_ce=0;values_ce=0;finish_valid=1;finish_status=4'(status);
        require(finish_ready,"finish blocked");tick();hidden();
        @(negedge clk);finish_valid=0;
    endtask
    task automatic check_done(input integer status,input logic started);
        require(done_valid && done_status==status && done_beats==sent && done_consumer_started==started,"completion mismatch");
        require(done_sequence==cmd_sequence && done_epoch==cmd_epoch && done_lease==cmd_lease,"completion identity mismatch");
        @(negedge clk);cmd_valid=1;tile_valid=1;load_done_valid=1;keys_ce=1;values_ce=1;compute_ready=1;
        repeat(5)begin
            tick();hidden();require(done_valid && done_status==status && done_beats==sent && done_consumer_started==started,"held completion changed");
            require(!cmd_ready && !tile_ready && !load_done_ready && !compute_valid && !finish_ready,"premature reuse");
        end
        @(negedge clk);cmd_valid=0;tile_valid=0;load_done_valid=0;keys_ce=0;values_ce=0;compute_ready=0;done_ready=1;
        tick();require(cmd_ready && !done_valid,"completion did not release RAM");hidden();
        @(negedge clk);done_ready=0;expected_epoch=100;
    endtask
    task automatic cancel_or_revoke(input logic revoke);
        @(negedge clk);
        if(revoke)expected_epoch=101;else begin require(cancel_ready,"cancel refused");cancel_valid=1;end
        #1;hidden();tick();hidden();@(negedge clk);cancel_valid=0;
    endtask
    task automatic drain_cancel;
        require(upstream_cancel_valid && !done_valid && !compute_valid,"upstream cancellation missing");
        repeat(4)begin tick();require(upstream_cancel_valid && !done_valid && !cmd_ready,"cancel was not retained");end
        @(negedge clk);upstream_cancel_ready=1;tick();
        @(negedge clk);upstream_cancel_ready=0;tick();require(!upstream_cancel_valid,"duplicate cancellation");
        commit_load(7);
    endtask
    initial begin : run
        integer token_sizes[0:5];integer expected_status;
        token_sizes[0]=1;token_sizes[1]=2;token_sizes[2]=3;token_sizes[3]=16;token_sizes[4]=63;token_sizes[5]=64;
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        random_state=32'(seed);$display("AKR_START seed=%0d watchdog=%0d",seed,WATCHDOG_CYCLES);
        reset_all();
        for(int h=0;h<32;h++)begin
            begin_job(0,h<6 ? token_sizes[h]:1,h);load_complete(h%2==0);start_compute();
            for(int a=0;a<ntokens*64;a++)read_pair(a,ntokens*64-1-a,1,1);
            for(int a=0;a<32;a++)read_pair(randmod(ntokens*64),randmod(ntokens*64),a%3!=0,a%4!=0);
            finish_compute(0);check_done(0,1);
        end
        for(int c=1;c<=11;c++)begin begin_job(c,2,3);check_done(c>=7 && c<=10 ? 2:1,0);end
        for(int c=12;c<=24;c++)begin
            begin_job(c,2,7);load_complete(c%2==0);
            expected_status=c>=19 && c<=21 ? 2 : (c==24 ? 3:4);
            check_done(expected_status,0);
        end
        // Truncated and overlong streams cannot be published.
        begin_job(25,2,0);beat(0,0);commit_load(0);check_done(4,0);
        begin_job(26,1,0);for(int a=0;a<9;a++)beat(a,0);commit_load(0);check_done(4,0);
        // Cancel/revoke during collection, while available, and during compute.
        for(int c=27;c<=32;c++)begin
            begin_job(c,2,31);
            if(c<=28)begin beat(0,0);cancel_or_revoke(c==28);beat(1,0);drain_cancel();check_done(c==28 ? 2:7,0);end
            else begin
                load_complete(0);
                if(c>=31)begin start_compute();read_pair(0,127,1,1);end
                cancel_or_revoke(c==30 || c==32);
                if(c>=31)begin
                    require(compute_abort && !cmd_ready && !done_valid,"consumer not retained after abort");
                    repeat(5)begin tick();hidden();require(compute_abort && finish_ready,"abort dropped");end
                    finish_compute(0);
                end
                check_done(c==30 || c==32 ? 2:7,c>=31);
            end
        end
        begin_job(33,1,0);repeat(WATCHDOG_CYCLES+2)tick();drain_cancel();check_done(8,0);
        begin_job(34,1,0);beat(0,0);repeat(WATCHDOG_CYCLES+2)tick();drain_cancel();check_done(8,0);
        for(int c=35;c<=36;c++)begin
            begin_job(c,1,5);load_complete(0);start_compute();read_pair(0,63,1,1);
            @(negedge clk);keys_ce=c==35;values_ce=c==36;keys_address=64;values_address=64;
            #1;hidden();require(compute_abort,"out-of-range RAM access did not abort");tick();hidden();
            @(negedge clk);keys_ce=0;values_ce=0;finish_compute(0);check_done(9,1);
        end
        begin_job(37,1,0);load_complete(0);start_compute();finish_compute(5);check_done(5,1);
        // Consumer can perform long arithmetic without memory traffic; no false load watchdog.
        begin_job(38,1,0);load_complete(0);start_compute();repeat(WATCHDOG_CYCLES+2)begin tick();require(!compute_abort,"false compute timeout");end
        read_pair(15,16,1,1);finish_compute(0);check_done(0,1);
        // Reset each ownership phase; a later short job must not expose stale longer tiles.
        for(int c=39;c<=42;c++)begin
            begin_job(c,2,2);
            if(c==39)beat(0,0);else load_complete(0);
            if(c>=41)begin start_compute();read_pair(1,65,1,1);end
            if(c==42)finish_compute(0);
            reset_all();begin_job(43,1,3);load_complete(0);start_compute();
            for(int a=0;a<64;a++)read_pair(a,63-a,1,1);
            finish_compute(0);check_done(0,1);
        end
        // Simultaneous terminal events: cancellation and revocation override success/finish.
        begin_job(44,1,0);for(int a=0;a<7;a++)beat(a,0);
        @(negedge clk);cancel_valid=1;beat(7,1);@(negedge clk);cancel_valid=0;check_done(7,0);
        begin_job(45,1,0);load_complete(1);start_compute();
        @(negedge clk);expected_epoch=101;finish_compute(0);check_done(2,1);
        begin_job(46,1,0);load_complete(1);check_done(4,0);
        $display("AKR_PASS seed=%0d watchdog=%0d jobs=%0d resets=%0d commands=%0d tiles=%0d commits=%0d computes=%0d reads=%0d dones=%0d",seed,WATCHDOG_CYCLES,job,resets,commands,tiles,commits,computes,reads,dones);
        $finish;
    end
endmodule
