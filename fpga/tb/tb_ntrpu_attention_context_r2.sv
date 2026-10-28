`timescale 1ns/1ps
`default_nettype none
// Protocol peer, deliberately NOT an attention arithmetic or generated-HLS model.
module tb_ntrpu_attention_context_r2;
    logic clk=0;always #5 clk=~clk;
    logic rst_n=0;
    logic command_valid=0,command_ready;
    logic [2047:0] command_query=0;
    logic [31:0] command_position=0,command_layer=0,command_head=0;
    logic [63:0] command_sequence=0,command_epoch=0,command_lease=0,expected_epoch=1;
    logic cancel_valid=0,cancel_ready,query_valid,query_ready=0,query_cancel_valid;
    logic [31:0] query_position,query_layer,query_head;
    logic [63:0] query_sequence,query_epoch,query_lease;
    logic compute_valid=0,compute_ready,compute_abort=0;
    logic [31:0] compute_start=0,compute_tokens=0,compute_position=0,compute_layer=0,compute_query_head=0,compute_kv_head=0;
    logic [63:0] compute_sequence=0,compute_epoch=0,compute_lease=0;
    logic finish_valid,finish_ready=0;
    logic [3:0] finish_status;
    logic query_done_valid=0,query_done_ready;
    logic [3:0] query_done_status=0;
    logic [31:0] query_done_tokens=0,query_done_tiles=0;
    logic [63:0] query_done_sequence=0,query_done_epoch=0,query_done_lease=0;
    logic completion_valid,completion_ready=0,result_valid,result_ce=0,result_q_valid;
    logic [3:0] completion_status;
    logic [5:0] result_address=0;
    logic [31:0] result_q;
    logic hls_start,hls_ready=0,hls_done=0;
    logic [31:0] hls_return=0,hls_tile_start,hls_tile_count,hls_query_position,hls_layer,hls_query_head,hls_kv_head;
    logic [63:0] hls_sequence,hls_epoch;
    logic hls_query_ce=0,hls_state_in_ce=0,hls_state_out_ce=0,hls_state_out_we=0,hls_output_ce=0,hls_output_we=0;
    logic [5:0] hls_query_address=0,hls_output_address=0;
    logic [6:0] hls_state_in_address=0,hls_state_out_address=0;
    logic [31:0] hls_query_q,hls_output_d=0;
    logic [63:0] hls_state_in_q,hls_state_out_d=0;
    ntrpu_attention_context_r2 dut(.*);
    integer cases=0,tiles=0,cycles=0,starts=0,reads=0,state_reads=0,result_reads=0,resets=0;
    integer seed=1,random_state,delay_count;
    logic [31:0] held_position,held_layer,held_head;
    logic [63:0] held_sequence,held_epoch,held_lease;
    integer held_tiles;
    function automatic [31:0] qword(input integer index);
        qword=32'h3f000000 | (held_sequence[15:0]<<6) | index;
    endfunction
    function automatic [63:0] sword(input integer tile_end,input integer index);
        sword=(held_sequence<<40) | (64'(tile_end)<<8) | 64'(index);
    endfunction
    function automatic [31:0] oword(input integer index);
        oword=32'h40000000 | (held_sequence[15:0]<<6) | index;
    endfunction
    always @(posedge clk)begin
        cycles=cycles+1;
        if(cycles>150000)$fatal(1,"TIMEOUT");
        if(rst_n)begin
            if(hls_state_out_ce && hls_state_out_we)$display("STATE_WRITE case=%0d tile=%0d index=%0d data=%016h",cases,tiles,hls_state_out_address,hls_state_out_d);
            if(hls_output_ce && hls_output_we)$display("OUTPUT_WRITE case=%0d tile=%0d index=%0d data=%08h",cases,tiles,hls_output_address,hls_output_d);
            if(hls_start && hls_ready)begin starts=starts+1;$display("START case=%0d tile=%0d cycle=%0d",cases,tiles,cycles);end
            if(result_valid && !completion_valid)$fatal(1,"early result authorization");
            if(!result_valid && (result_q_valid || result_q!==0))$fatal(1,"private result exposed");
            if(finish_valid && hls_start)$fatal(1,"finish before start handshake");
            if(completion_valid && (compute_ready || query_valid || hls_start))$fatal(1,"completion does not own context");
        end
    end
    task automatic tick;
        @(posedge clk);#1;
    endtask
    task automatic quiet_peer;
        hls_ready=0;hls_done=0;hls_return=0;hls_query_ce=0;hls_state_in_ce=0;
        hls_state_out_ce=0;hls_state_out_we=0;hls_output_ce=0;hls_output_we=0;
    endtask
    task automatic reset_all;
        @(negedge clk);rst_n=0;quiet_peer();command_valid=0;query_ready=0;compute_valid=0;compute_abort=0;
        finish_ready=0;query_done_valid=0;completion_ready=0;cancel_valid=0;result_ce=0;
        repeat(3)tick();@(negedge clk);rst_n=1;tick();
        if(!command_ready || completion_valid || result_valid || hls_start)$fatal(1,"reset ownership");
        resets=resets+1;$display("RESET index=%0d case=%0d cycle=%0d",resets,cases,cycles);
    endtask
    task automatic command(input integer position,input integer layer,input integer head,input integer bad);
        cases=cases+1;held_sequence=cases;held_epoch=expected_epoch;held_lease=cases;
        held_position=position;held_layer=layer;held_head=head;held_tiles=0;
        @(negedge clk);command_valid=1;command_position=position;command_layer=layer;command_head=head;
        command_sequence=held_sequence;command_epoch=held_epoch;command_lease=held_lease;
        for(integer i=0;i<64;i=i+1)command_query[i*32+:32]=qword(i);
        if(bad==1)command_query[7*32+:32]=32'h7f800001;
        if(bad==2)command_epoch=expected_epoch+1;
        if(bad==3)command_sequence=0;
        if(!command_ready)$fatal(1,"command not ready");tick();
        @(negedge clk);command_valid=0;command_query='1;
        $display("COMMAND case=%0d position=%0d layer=%0d head=%0d bad=%0d cycle=%0d",cases,position,layer,head,bad,cycles);
    endtask
    task automatic issue;
        delay_count=1+($urandom(random_state)%6);
        repeat(delay_count)begin tick();if(!query_valid || result_valid)$fatal(1,"query request not held");end
        if(query_position!==held_position || query_layer!==held_layer || query_head!==held_head ||
            query_sequence!==held_sequence || query_epoch!==held_epoch || query_lease!==held_lease)$fatal(1,"query metadata");
        @(negedge clk);query_ready=1;tick();@(negedge clk);query_ready=0;
    endtask
    task automatic tile_command(input integer first,input integer count,input integer bad);
        @(negedge clk);compute_valid=1;compute_start=first;compute_tokens=count;
        compute_position=held_position;compute_layer=held_layer;compute_query_head=held_head;compute_kv_head=held_head/4;
        compute_sequence=held_sequence;compute_epoch=held_epoch;compute_lease=held_lease;
        if(bad==1)compute_lease=held_lease+1;
        if(bad==2)compute_start=first+1;
        if(bad==3)compute_kv_head=(held_head/4)^1;
        if(!compute_ready)$fatal(1,"compute not ready");tick();@(negedge clk);compute_valid=0;
        tiles=tiles+1;
        $display("TILE case=%0d tile=%0d first=%0d count=%0d bad=%0d cycle=%0d",cases,tiles,first,count,bad,cycles);
    endtask
    // Modes exercise the memory/control protocol. Values are tagged bit patterns,
    // not softmax outputs. Cancellation keeps this peer alive through its done.
    task automatic peer(input integer mode,input integer early_ready);
        integer before_starts;
        integer finish_code;
        logic last_tile;
        before_starts=starts;last_tile=hls_tile_start+hls_tile_count==held_position+1;
        if(hls_query_position!==held_position || hls_layer!==held_layer || hls_query_head!==held_head ||
            hls_kv_head!==(held_head/4) || hls_sequence!==held_sequence || hls_epoch!==held_epoch)$fatal(1,"HLS scalar metadata");
        repeat(5)begin tick();if(!hls_start || finish_valid)$fatal(1,"start not held to ready");end
        if(mode==9)begin @(negedge clk);cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if(early_ready)begin @(negedge clk);hls_ready=1;tick();@(negedge clk);hls_ready=0;end
        for(integer i=0;i<64;i=i+1)begin
            @(negedge clk);hls_query_ce=1;hls_query_address=i;tick();
            if(hls_query_q!==qword(i))$fatal(1,"Q memory case=%0d index=%0d got=%h",cases,i,hls_query_q);
            reads=reads+1;$display("Q case=%0d tile=%0d index=%0d data=%08h",cases,tiles,i,hls_query_q);
        end
        @(negedge clk);hls_query_ce=0;
        if(hls_tile_start!=0)begin
            for(integer i=0;i<104;i=i+1)begin
                @(negedge clk);hls_state_in_ce=1;hls_state_in_address=i;tick();
                if(hls_state_in_q!==sword(hls_tile_start,i))$fatal(1,"state ownership case=%0d index=%0d got=%h",cases,i,hls_state_in_q);
                state_reads=state_reads+1;$display("STATE_READ case=%0d tile=%0d index=%0d data=%016h",cases,tiles,i,hls_state_in_q);
            end
        end
        @(negedge clk);hls_state_in_ce=0;
        if(mode==5)begin hls_state_in_ce=1;hls_state_in_address=127;tick();@(negedge clk);hls_state_in_ce=0;end
        if(mode==10)begin expected_epoch=expected_epoch+1;tick();end
        if(mode==11)begin @(negedge clk);compute_abort=1;tick();end
        for(integer i=0;i<104;i=i+1)begin
            @(negedge clk);hls_state_out_ce=1;hls_state_out_we=1;hls_state_out_address=i;
            hls_state_out_d=sword(hls_tile_start+hls_tile_count,i);
            if(mode==1 && i==103)hls_state_out_we=0;
            if(mode==2 && i==103)hls_state_out_address=0;
            if(mode==6 && i==103)hls_state_out_address=127;
            // On intermediate success, last state write coincides with done.
            if(!last_tile && i==103 && mode!=3)begin hls_done=1;hls_ready=1;hls_return=mode==7 ? 32'd33:0;end
            tick();
        end
        @(negedge clk);hls_state_out_ce=0;hls_state_out_we=0;
        if(last_tile || mode==3)begin
            for(integer i=0;i<64;i=i+1)begin
                @(negedge clk);hls_output_ce=1;hls_output_we=1;hls_output_address=i;hls_output_d=oword(i);
                if(mode==4 && i==7)hls_output_d=32'h7fc00001;
                if(mode==8 && i>=32)hls_output_we=0;
                if(mode==15 && i==63)hls_output_we=0;
                if(i==63)begin hls_done=1;hls_ready=1;hls_return=mode==7 ? 32'd33:0;end
                tick();
            end
        end
        @(negedge clk);quiet_peer();compute_abort=0;#1;
        if(starts!=before_starts+1)$fatal(1,"start handshake count got=%0d expected=%0d",starts,before_starts+1);
        if(!finish_valid)$fatal(1,"missing tile completion");
        if(mode==14)begin hls_output_ce=1;hls_output_we=1;tick();@(negedge clk);quiet_peer();end
        case(mode)
            0:finish_code=0;
            9,11:finish_code=7;
            10:finish_code=2;
            7:finish_code=10;
            default:finish_code=4;
        endcase
        if(finish_status!==4'(finish_code))$fatal(1,"finish case=%0d mode=%0d got=%0d expected=%0d",cases,mode,finish_status,finish_code);
        repeat(5)begin tick();if(!finish_valid || result_valid || finish_status!==4'(finish_code))$fatal(1,"finish hold");end
        @(negedge clk);finish_ready=1;tick();@(negedge clk);finish_ready=0;
        if(finish_code==0)held_tiles=held_tiles+1;
        $display("FINISH case=%0d tile=%0d mode=%0d status=%0d cycle=%0d",cases,tiles,mode,finish_code,cycles);
    endtask
    task automatic bad_tile_finish(input integer expected_status);
        if(hls_start || !finish_valid || finish_status!==4'(expected_status))$fatal(1,"bad descriptor started HLS");
        repeat(3)tick();@(negedge clk);finish_ready=1;tick();@(negedge clk);finish_ready=0;
        $display("REJECT_TILE case=%0d tile=%0d status=%0d cycle=%0d",cases,tiles,expected_status,cycles);
    endtask
    task automatic path_done(input integer status,input integer bad);
        repeat(4)begin tick();if(result_valid)$fatal(1,"result before release certificate");end
        @(negedge clk);query_done_valid=1;query_done_status=status;query_done_tokens=held_position+1;
        query_done_tiles=held_tiles;query_done_sequence=held_sequence;query_done_epoch=held_epoch;query_done_lease=held_lease;
        if(bad==1)query_done_tokens=held_position;
        if(bad==2)query_done_lease=held_lease+1;
        if(bad==3)query_done_tiles=held_tiles+1;
        if(bad==4)expected_epoch=expected_epoch+1;
        if(!query_done_ready)$fatal(1,"query certificate not ready");tick();@(negedge clk);query_done_valid=0;
    endtask
    task automatic result_check(input integer status,input integer revoke);
        if(!completion_valid || completion_status!==4'(status) || result_valid!==(status==0))
            $fatal(1,"completion case=%0d status=%0d expected=%0d result=%b",cases,completion_status,status,result_valid);
        repeat(4)tick();
        if(status==0)begin
            for(integer i=0;i<64;i=i+1)begin
                @(negedge clk);result_ce=1;result_address=i;tick();
                if(!result_q_valid || result_q!==oword(i))$fatal(1,"result data case=%0d index=%0d",cases,i);
                result_reads=result_reads+1;$display("RESULT case=%0d index=%0d data=%08h",cases,i,result_q);
            end
        end
        if(revoke)begin
            @(negedge clk);result_ce=1;expected_epoch=expected_epoch+1;#1;
            if(result_valid || result_q_valid || result_q!==0)$fatal(1,"revoked data immediately exposed");
            tick();@(negedge clk);expected_epoch=held_epoch;tick();
            if(result_valid || result_q_valid || completion_status!==4'(status))$fatal(1,"revocation not sticky / completion changed");
        end
        @(negedge clk);result_ce=0;completion_ready=1;tick();@(negedge clk);completion_ready=0;
        if(!command_ready || result_valid || result_q_valid)$fatal(1,"retirement");
        $display("COMPLETE case=%0d status=%0d revoke=%0d cycle=%0d",cases,status,revoke,cycles);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;
        reset_all();
        // Single, full, boundary-crossing and three-tile queries, including GQA.
        for(integer head=0;head<32;head=head+1)begin
            command(head%2==0 ? 0:64,head%16,head,0);issue();
            if(held_position==64)begin tile_command(0,64,0);peer(0,head%2);end
            tile_command(held_position==64 ? 64:0,1,0);peer(0,head%2);
            path_done(0,0);result_check(0,head==31);
        end
        command(128,15,31,0);issue();tile_command(0,64,0);peer(0,1);
        tile_command(64,64,0);peer(0,0);tile_command(128,1,0);peer(0,1);path_done(0,0);result_check(0,0);
        command(63,0,0,0);issue();tile_command(0,64,0);peer(0,0);path_done(0,0);result_check(0,0);
        for(integer mode=1;mode<=15;mode=mode+1)begin
            if(mode!=12 && mode!=13)begin
                command(mode==3 ? 64:0,0,0,0);issue();tile_command(0,mode==3 ? 64:1,0);peer(mode,1);
                path_done(0,0);result_check(mode==9 || mode==11 ? 7:mode==10 ? 2:mode==7 ? 10:4,0);
            end
        end
        // Errors after a committed private state must not publish earlier work.
        command(64,7,13,0);issue();tile_command(0,64,0);peer(0,1);tile_command(64,1,0);peer(7,1);
        path_done(0,0);result_check(10,0);
        for(integer bad=1;bad<=3;bad=bad+1)begin
            command(0,0,0,0);issue();tile_command(0,1,bad);bad_tile_finish(bad==2 ? 1:2);path_done(0,0);result_check(bad==2 ? 1:2,0);
        end
        for(integer bad=1;bad<=4;bad=bad+1)begin
            command(0,0,0,0);issue();tile_command(0,1,0);peer(0,1);path_done(0,bad);result_check(bad==4 ? 2:11,0);
        end
        command(0,0,0,0);issue();path_done(0,0);result_check(11,0);
        command(0,0,0,0);issue();path_done(3,0);result_check(3,0);
        command(0,0,0,0);@(negedge clk);cancel_valid=1;tick();@(negedge clk);cancel_valid=0;result_check(7,0);
        command(0,0,0,0);issue();@(negedge clk);cancel_valid=1;tick();@(negedge clk);cancel_valid=0;
        if(!query_cancel_valid || compute_ready)$fatal(1,"query cancel not propagated");path_done(7,0);result_check(7,0);
        command(0,0,0,1);result_check(4,0);command(0,0,0,2);result_check(2,0);command(0,0,0,3);result_check(2,0);
        command(131072,0,0,0);result_check(1,0);command(0,16,0,0);result_check(1,0);command(0,0,32,0);result_check(1,0);
        // Coordinated peer resets at admission, live HLS, held finish and result.
        command(0,0,0,0);reset_all();
        command(0,0,0,0);issue();tile_command(0,1,0);repeat(5)tick();reset_all();
        command(0,0,0,0);issue();tile_command(0,1,0);peer(0,1);path_done(0,0);reset_all();
        command(0,0,0,0);issue();tile_command(0,1,0);peer(0,1);path_done(0,0);result_check(0,0);
        $display("PASS_ATTENTION_CONTEXT_PROTOCOL_R2 seed=%0d cases=%0d tiles=%0d starts=%0d q_reads=%0d state_reads=%0d result_reads=%0d resets=%0d cycles=%0d",
            seed,cases,tiles,starts,reads,state_reads,result_reads,resets,cycles);
        $finish;
    end
endmodule
`default_nettype wire
