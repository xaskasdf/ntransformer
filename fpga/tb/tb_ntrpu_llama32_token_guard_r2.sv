`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_llama32_token_guard_r2;
    logic  clk,rst_n;
    logic  command_valid;
    wire  command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch,command_lease_base;
    logic  command_source_valid,command_plan_valid;
    logic [63:0] command_source_sequence,command_source_epoch;
    logic [31:0] command_position;
    logic [63:0] command_input_address,command_logits_address;
    logic [15:0][63:0] drive_residual;
    wire [63:0] command_residual_address [0:15];
    for(genvar n=0;n<16;n++)assign command_residual_address[n]=drive_residual[n];
    logic  cancel_valid;
    wire  cancel_ready,busy,cache_exclusive,bind_allowed;
    logic  bind_request_valid,bind_completion_valid;
    wire  bind_completion_ready;
    logic [3:0] bind_completion_status;
    logic [63:0] bind_completion_sequence,bind_completion_epoch;
    logic  cache_idle,cache_valid,cache_metadata_fault;
    logic [63:0] cache_sequence,cache_epoch;
    logic [31:0] cache_capacity;
    wire [31:0] cache_query_layer;
    logic  cache_query_valid;
    logic [31:0] cache_query_committed_tokens;
    wire  layer_command_valid;
    logic  layer_command_ready,layer_busy;
    wire [63:0] layer_command_sequence,layer_command_epoch,layer_command_lease_base;
    wire [31:0] layer_command_layer,layer_command_position;
    wire [63:0] layer_command_input_address,layer_command_output_address;
    wire  layer_command_source_valid;
    wire  layer_cancel_valid;
    logic  layer_completion_valid;
    wire  layer_completion_ready;
    logic [3:0] layer_completion_status;
    logic [63:0] layer_sequence,layer_epoch;
    logic [31:0] layer_index,layer_position;
    logic  layer_result_valid,layer_revoked,layer_kv_committed;
    logic [63:0] layer_result_address;
    logic [13:0] layer_result_bytes;
    logic [31:0] layer_kv_tokens;
    wire  head_command_valid;
    logic  head_command_ready,head_busy;
    wire [63:0] head_command_sequence,head_command_epoch;
    wire [63:0] head_command_input_address,head_command_output_address;
    wire  head_command_source_valid;
    wire  head_cancel_valid;
    logic  head_completion_valid;
    wire  head_completion_ready;
    logic [3:0] head_completion_status;
    logic [63:0] head_sequence,head_epoch;
    logic  head_result_valid,head_revoked;
    logic [63:0] head_result_address;
    logic [18:0] head_result_bytes;
    wire  completion_valid;
    logic  completion_ready;
    wire [3:0] completion_status,completion_child_status,binding_status;
    wire [63:0] completion_sequence,completion_epoch;
    wire [31:0] completion_position;
    wire [4:0] completion_layers;
    wire  completion_revoked,result_valid;
    wire [63:0] result_address;
    wire [18:0] result_bytes;
    wire  cache_needs_rebind;
    wire [31:0] committed_tokens;

    ntrpu_llama32_token_guard_r1 #(.CHILD_TIMEOUT_CYCLES(64)) dut(.*);
    always #2.5 clk=~clk;
    localparam logic [63:0] EMBED=64'h3000000000,RESID=64'h3000010000,LOGITS=64'h4000000000;
    logic [31:0] prefix[0:15];
    assign cache_query_valid=cache_valid && cache_query_layer<16;
    assign cache_query_committed_tokens=cache_query_valid ? prefix[cache_query_layer]:0;
    integer cycle=0,job=0,mode_now=0,target_now=0,seed=1,random_state=1;
    integer binds=0,publics=0,layer_starts=0,layer_acks=0,head_starts=0,head_acks=0;
    integer l_delay=0,h_delay=0,l_drain=0,h_drain=0,l_foreign=0,h_foreign=0;
    integer saved_layer=0,saved_position=0,job_layers=0,job_heads=0,job_lacks=0,job_hacks=0;
    integer total_success=0,total_poison=0;
    logic l_aborted=0,h_aborted=0,public_seen=0;
    logic [63:0] seq_model,epoch_model,input_model,logits_model,lease_model;
    logic [63:0] last_output;
    function automatic logic [63:0] residual(input integer n);
        return RESID+64'(n%2)*64'h10000;
    endfunction
    task automatic check(input bit ok,input string why);
        if(!ok)$fatal(1,"TOK_FAIL job=%0d mode=%0d target=%0d cycle=%0d %s",job,mode_now,target_now,cycle,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    always @(negedge clk)begin
        layer_command_ready=!layer_busy && !layer_completion_valid && !(mode_now==28 && layer_command_layer==target_now);
        head_command_ready=!head_busy && !head_completion_valid && mode_now!=29;
    end
    always @(posedge clk)begin
        cycle++;
        if(rst_n)begin
            check(!busy || cache_exclusive,"Global reservation dropped");
            check(!busy || !bind_allowed,"Rebind during token ownership");
            if(layer_command_valid && layer_command_ready)begin
                check(job_layers<16 && layer_command_layer==job_layers,"Skipped/repeated layer");
                check(job_lacks==job_layers && !layer_busy && !layer_completion_valid,"Layer reuse before drain");
                check(layer_command_sequence==seq_model && layer_command_epoch==epoch_model,"Layer identity");
                check(layer_command_position==saved_position && layer_command_lease_base==lease_model+64'(job_layers*32),"Position/lease reservation");
                check(layer_command_input_address==(job_layers==0 ? input_model:last_output) && layer_command_output_address==residual(job_layers),"Residual continuity");
                check(layer_command_source_valid,"Missing retained source certificate");
                saved_layer=job_layers;last_output=residual(job_layers);
                layer_sequence<=seq_model;layer_epoch<=epoch_model;layer_index<=job_layers;layer_position<=saved_position;
                layer_result_address<=residual(job_layers);layer_result_bytes<=8192;
                layer_kv_committed<=1;layer_kv_tokens<=saved_position+1;layer_result_valid<=0;layer_revoked<=0;
                layer_completion_status<=0;layer_busy<=1;l_delay=$urandom_range(3,7);l_aborted=0;
                if(mode_now==30 && saved_layer==target_now)l_delay=100;
                prefix[job_layers]<=saved_position+1;
                $display("TOK_LAYER cycle=%0d job=%0d layer=%0d position=%0d input=%016h output=%016h lease=%016h",cycle,job,job_layers,saved_position,layer_command_input_address,layer_command_output_address,layer_command_lease_base);
                job_layers++;layer_starts++;
            end else if(layer_busy && !layer_completion_valid && l_delay>0)begin
                if(layer_cancel_valid)begin l_aborted=1;if(l_delay>2)l_delay=2;end
                l_delay--;
                if(l_delay==0)begin
                    layer_completion_valid<=1;layer_result_valid<=!l_aborted;
                    layer_completion_status<=l_aborted ? 7:0;
                    if(mode_now==8 && saved_layer==target_now)begin layer_completion_status<=3;layer_result_valid<=0;end
                    if(mode_now==9 && saved_layer==target_now)layer_result_bytes<=4096;
                    if(mode_now==10 && saved_layer==target_now)prefix[saved_layer]<=saved_position;
                    if(mode_now==25 && saved_layer==target_now)begin layer_sequence<=seq_model^64'h100000000;l_foreign=8;end
                    $display("TOK_LDONE cycle=%0d job=%0d layer=%0d status=%0d",cycle,job,saved_layer,l_aborted ? 7:(mode_now==8 && saved_layer==target_now) ? 3:0);
                end
            end
            if(l_foreign>0)begin
                check(!layer_completion_ready,"Foreign layer ACK");l_foreign--;
                if(l_foreign==0)layer_sequence<=seq_model;
            end
            if(layer_completion_valid && layer_completion_ready)begin
                if(saved_layer==15)check(public_seen && job_hacks==job_heads && !head_busy,"Last residual released before head drain");
                $display("TOK_LACK cycle=%0d job=%0d layer=%0d public=%0d",cycle,job,saved_layer,public_seen);
                layer_completion_valid<=0;layer_result_valid<=0;l_drain=3;job_lacks++;layer_acks++;
            end else if(l_drain>0)begin l_drain--;if(l_drain==0)layer_busy<=0;end
            if(head_command_valid && head_command_ready)begin
                check(job_layers==16 && job_lacks==15 && layer_completion_valid && layer_index==15,"Head before retained final layer");
                check(head_command_source_valid && head_command_sequence==seq_model && head_command_epoch==epoch_model,"Head source identity");
                check(head_command_input_address==residual(15) && head_command_output_address==logits_model,"Head descriptors");
                check(job_heads==0,"Duplicate head");job_heads++;head_starts++;
                head_busy<=1;h_aborted=0;h_delay=$urandom_range(3,7);head_revoked<=0;
                head_sequence<=seq_model;head_epoch<=epoch_model;head_completion_status<=0;
                head_result_valid<=0;head_result_address<=logits_model;head_result_bytes<=513024;
                if(mode_now==31)h_delay=100;
                $display("TOK_HEAD cycle=%0d job=%0d input=%016h output=%016h",cycle,job,head_command_input_address,head_command_output_address);
            end else if(head_busy && !head_completion_valid && h_delay>0)begin
                if(head_cancel_valid)begin h_aborted=1;if(h_delay>2)h_delay=2;end
                h_delay--;
                if(h_delay==0)begin
                    head_completion_valid<=1;head_result_valid<=!h_aborted;head_completion_status<=h_aborted ? 7:0;
                    if(mode_now==11)begin head_completion_status<=3;head_result_valid<=0;end
                    if(mode_now==12)head_result_bytes<=512960;
                    if(mode_now==13)head_result_valid<=0;
                    if(mode_now==14)prefix[target_now]<=saved_position;
                    if(mode_now==26)begin head_epoch<=epoch_model^64'h100000000;h_foreign=8;end
                    $display("TOK_HDONE cycle=%0d job=%0d status=%0d",cycle,job,h_aborted ? 7:mode_now==11 ? 3:0);
                end
            end
            if(h_foreign>0)begin
                check(!head_completion_ready,"Foreign head ACK");h_foreign--;
                if(h_foreign==0)head_epoch<=epoch_model;
            end
            if(head_completion_valid && head_completion_ready)begin
                check(public_seen,"Head released before public ACK");
                $display("TOK_HACK cycle=%0d job=%0d",cycle,job);
                head_completion_valid<=0;head_result_valid<=0;h_drain=3;job_hacks++;head_acks++;
            end else if(h_drain>0)begin h_drain--;if(h_drain==0)head_busy<=0;end
            if(completion_valid && completion_ready)begin
                check(!public_seen,"Repeated public response");public_seen=1;publics++;
                $display("TOK_PUBLIC cycle=%0d job=%0d status=%0d valid=%0d revoked=%0d layers=%0d position=%0d",cycle,job,completion_status,result_valid,completion_revoked,completion_layers,completion_position);
            end
            if(result_valid)check(completion_valid && completion_status==0 && result_address==logits_model && result_bytes==513024 && job_layers==16 && job_heads==1,"Partial result");
            else check(result_address==0 && result_bytes==0,"Invalid result descriptors leaked");
        end
    end
    task automatic bind_context(input integer bad_prefix);
        integer limit;
        @(negedge clk);check(!busy && !layer_busy && !head_busy && bind_allowed,"Bind ownership");binds++;
        cache_sequence=64'h12345678abcdef00+64'(binds);cache_epoch=64'hfedcba9800000000+64'(binds);
        expected_epoch=cache_epoch;cache_valid=1;cache_metadata_fault=0;cache_idle=1;cache_capacity=4;
        for(integer n=0;n<16;n++)prefix[n]=0;
        if(bad_prefix>=0)prefix[bad_prefix]=1;
        bind_completion_status=0;bind_completion_sequence=cache_sequence;bind_completion_epoch=cache_epoch;
        case(bad_prefix)
            -2:bind_completion_status=3;
            -3:bind_completion_sequence=cache_sequence^64'h100000000;
            -4:bind_completion_epoch=cache_epoch^64'h100000000;
            -5:cache_capacity=0;
            -6:cache_capacity=131073;
            -7:cache_metadata_fault=1;
            -8:cache_valid=0;
            -9:expected_epoch=cache_epoch^64'h100000000;
        endcase
        bind_completion_valid=1;check(bind_completion_ready,"Bind ACK");tick();@(negedge clk);bind_completion_valid=0;
        limit=0;while(busy)begin tick();limit++;check(limit<30,"Bind verification deadline");end
        check(cache_needs_rebind==(bad_prefix!=-1) && command_ready==(bad_prefix==-1),"All-layer zero prefix admission");
        if(bad_prefix==-1)check(committed_tokens==0 && binding_status==0,"Bind zero commit");
        else check(binding_status==11,"Bad bind prefix status");
        $display("TOK_BIND cycle=%0d bind=%0d bad_prefix=%0d poison=%0d",cycle,binds,bad_prefix,cache_needs_rebind);
    endtask
    task automatic run_job(input integer mode,input integer target,input integer position);
        integer limit,expected_status,expected_poison,expected_layers;
        bit late;
        @(negedge clk);check(command_ready,"Command admission");job++;mode_now=mode;target_now=target;
        job_layers=0;job_heads=0;job_lacks=0;job_hacks=0;public_seen=0;saved_position=position;
        seq_model=cache_sequence;epoch_model=cache_epoch;lease_model=64'hfffffffffffffc00;
        input_model=EMBED;logits_model=LOGITS;
        command_sequence=seq_model;command_epoch=epoch_model;command_position=position;command_lease_base=lease_model;
        command_source_valid=1;command_source_sequence=seq_model;command_source_epoch=epoch_model;command_plan_valid=1;
        command_input_address=input_model;command_logits_address=logits_model;
        for(integer n=0;n<16;n++)drive_residual[n]=residual(n);
        if(mode==1)command_plan_valid=0;
        if(mode==2)command_source_sequence=seq_model^64'h100000000;
        if(mode==3)command_position=position+1;
        if(mode==4)command_lease_base=64'hfffffffffffffe01;
        if(mode==5)command_input_address=EMBED+1;
        if(mode>=100 && mode<116)drive_residual[mode-100]=residual(mode-100)+1;
        if(mode>=200 && mode<216)drive_residual[mode-200]=mode==200 ? EMBED:residual(mode-201);
        if(mode==7)prefix[target]=position+1;
        command_valid=1;
        $display("TOK_BEGIN cycle=%0d job=%0d mode=%0d target=%0d position=%0d committed=%0d",cycle,job,mode,target,position,committed_tokens);
        tick();@(negedge clk);command_valid=0;command_sequence='1;command_epoch='1;
        command_input_address='1;command_logits_address='1;drive_residual='1;command_plan_valid=0;command_source_valid=0;
        if(mode==20)begin
            wait(job_layers==target+1 && layer_busy && !layer_completion_valid);@(negedge clk);cancel_valid=1;tick();@(negedge clk);cancel_valid=0;
        end
        if(mode==21 || mode==27)begin
            wait(job_heads==1 && head_busy && !head_completion_valid);@(negedge clk);
            if(mode==21)cancel_valid=1;else layer_result_valid=0;
            tick();@(negedge clk);cancel_valid=0;
        end
        if(mode==36)begin
            wait(job_lacks==target+1 && layer_busy);@(negedge clk);cancel_valid=1;
            tick();@(negedge clk);cancel_valid=0;
        end
        if(mode==37 || mode==38)begin
            @(negedge clk);if(mode==37)cancel_valid=1;else cache_metadata_fault=1;
            tick();@(negedge clk);cancel_valid=0;cache_metadata_fault=0;
        end
        limit=0;while(!completion_valid)begin tick();limit++;check(limit<2000,"Token completion deadline");end
        late=mode==22 || mode==23 || mode==24 || mode==34 || mode==35;
        expected_status=(mode==0 || late) ? 0:mode==2 ? 2:
            (mode==1 || mode==3 || mode==4 || mode==5 || mode>=100) ? 1:
            (mode==8 || mode==11) ? 3:(mode==9 || mode==10 || mode==12 || mode==13) ? 5:
            (mode==20 || mode==21 || mode==36 || mode==37) ? 7:(mode>=28 && mode<=31) ? 8:11;
        expected_poison=mode!=0 && !(mode==1 || mode==2 || mode==3 || mode==4 || mode==5 || mode==37 || mode>=100 || (mode==28 && target==0));
        check(completion_status==expected_status,"Public diagnostic status");
        check(completion_sequence==seq_model && completion_epoch==epoch_model,"Captured public identity");
        check(result_valid==(mode==0 || late),"Result validity");
        repeat(5)begin tick();check(completion_valid && !command_ready && !bind_allowed,"Held token completion");end
        if(late)begin
            @(negedge clk);
            if(mode==22)cancel_valid=1;
            if(mode==23)expected_epoch=epoch_model^64'h100000000;
            if(mode==24)cache_metadata_fault=1;
            if(mode==34)head_result_bytes=512960;
            if(mode==35)layer_result_bytes=4096;
            #1;check(!result_valid && completion_revoked,"Immediate late revocation");tick();
            @(negedge clk);cancel_valid=0;expected_epoch=epoch_model;cache_metadata_fault=0;
            head_result_bytes=513024;layer_result_bytes=8192;repeat(3)tick();
            check(!result_valid && completion_status==0 && completion_revoked,"Late rollback revived result");
        end
        check(job_lacks==(job_layers==16 ? 15:(mode==28 || mode==36 ? target+(mode==36 ? 1:0):job_layers==0 ? 0:job_layers-1)),"Held layer release count");
        @(negedge clk);completion_ready=1;tick();check(!completion_valid,"Public ACK not consumed");
        @(negedge clk);completion_ready=0;limit=0;
        while(busy)begin tick();limit++;check(limit<40,"Drain deadline");end
        check(job_lacks==job_layers && job_hacks==job_heads && !layer_busy && !head_busy,"All child drains");
        check(cache_needs_rebind==expected_poison,"Context poison decision");
        check(committed_tokens==(mode==0 ? position+1:position),"Atomic whole-token commit");
        if(expected_poison)begin
            total_poison++;@(negedge clk);command_valid=1;repeat(3)begin tick();check(!command_ready && !layer_command_valid && !head_command_valid,"Poisoned context accepted work");end
            @(negedge clk);command_valid=0;
        end else check(command_ready,"Reusable clean context");
        if(mode==0)total_success++;
        $display("TOK_END cycle=%0d job=%0d layers=%0d heads=%0d lacks=%0d hacks=%0d poison=%0d committed=%0d",cycle,job,job_layers,job_heads,job_lacks,job_hacks,cache_needs_rebind,committed_tokens);
        mode_now=0;
    endtask
    task automatic fault_and_recover(input integer mode,input integer target);
        bind_context(-1);run_job(mode,target,0);
        if(cache_needs_rebind)bind_context(-1);
        run_job(0,0,0);
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;cancel_valid=0;completion_ready=0;bind_request_valid=0;bind_completion_valid=0;
        command_sequence=0;command_epoch=0;expected_epoch=0;command_lease_base=0;command_source_valid=0;command_plan_valid=0;
        command_source_sequence=0;command_source_epoch=0;command_position=0;command_input_address=0;command_logits_address=0;drive_residual=0;
        bind_completion_status=0;bind_completion_sequence=0;bind_completion_epoch=0;
        cache_idle=1;cache_valid=0;cache_metadata_fault=0;cache_sequence=0;cache_epoch=0;cache_capacity=0;
        layer_command_ready=1;layer_busy=0;layer_completion_valid=0;layer_completion_status=0;layer_sequence=0;layer_epoch=0;
        layer_index=0;layer_position=0;layer_result_valid=0;layer_revoked=0;layer_kv_committed=0;layer_result_address=0;layer_result_bytes=0;layer_kv_tokens=0;
        head_command_ready=1;head_busy=0;head_completion_valid=0;head_completion_status=0;head_sequence=0;head_epoch=0;
        head_result_valid=0;head_revoked=0;head_result_address=0;head_result_bytes=0;
        for(integer n=0;n<16;n++)prefix[n]=0;
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        repeat(4)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        check(cache_needs_rebind && !command_ready,"Reset requires cache bind");
        for(integer n=0;n<16;n++)bind_context(n);
        for(integer n=2;n<=9;n++)bind_context(-n);
        bind_context(-1);run_job(0,0,0);run_job(0,0,1);
        for(integer mode=1;mode<=5;mode++)fault_and_recover(mode,0);
        for(integer n=0;n<16;n++)begin
            fault_and_recover(100+n,n);fault_and_recover(200+n,n);
            fault_and_recover(7,n);fault_and_recover(8,n);fault_and_recover(9,n);fault_and_recover(10,n);
            fault_and_recover(14,n);fault_and_recover(20,n);fault_and_recover(25,n);fault_and_recover(28,n);
        end
        for(integer mode=11;mode<=13;mode++)fault_and_recover(mode,0);
        for(integer mode=21;mode<=24;mode++)fault_and_recover(mode,0);
        fault_and_recover(26,0);fault_and_recover(27,0);fault_and_recover(29,0);
        fault_and_recover(30,7);fault_and_recover(31,0);fault_and_recover(34,0);fault_and_recover(35,0);
        for(integer n=0;n<15;n++)fault_and_recover(36,n);
        fault_and_recover(37,0);fault_and_recover(38,0);
        check(publics==job && layer_starts==layer_acks && head_starts==head_acks,"Suite drain coverage");
        $display("TOK_PASS seed=%0d jobs=%0d binds=%0d publics=%0d layers=%0d lacks=%0d heads=%0d hacks=%0d successes=%0d poisoned=%0d cycles=%0d",seed,job,binds,publics,layer_starts,layer_acks,head_starts,head_acks,total_success,total_poison,cycle);
        $finish;
    end
    initial begin #10000000;$fatal(1,"TOK_FAIL global timeout");end
endmodule
`default_nettype wire
