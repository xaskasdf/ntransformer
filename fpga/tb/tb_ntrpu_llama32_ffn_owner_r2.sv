`timescale 1ns/1ps
`default_nettype none
// Directed protocol peer; no FFN arithmetic or AXI simulation.
module tb_ntrpu_llama32_ffn_owner_r2 #(parameter integer CONFIG=0);
    logic clk,rst_n;
    logic command_valid;
    logic command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic command_source_valid;
    logic [63:0] command_source_sequence,command_source_epoch;
    logic [63:0] command_addr [0:7],command_base [0:7],command_bytes [0:7];
    logic [63:0] command_residual_address,command_residual_base,command_residual_bytes;
    logic cancel_valid;
    logic cancel_ready,busy;
    logic completion_valid;
    logic completion_ready;
    logic [3:0] completion_status,completion_child_status;
    logic [2:0] completion_stages;
    logic [1:0] completion_phase;
    logic [16:0] completion_bytes;
    logic [17:0] completion_stage_bytes;
    logic [31:0] completion_hls_status;
    logic [63:0] completion_sequence,completion_epoch;
    logic result_valid;
    logic [63:0] result_ffn_address,result_residual_address;
    logic [13:0] result_ffn_bytes,result_residual_bytes;
    logic tags_exhausted;
    logic child_job_valid;
    logic child_job_ready;
    logic [1:0] child_job_layout;
    logic [13:0] child_job_dim,child_job_hidden;
    wire [5:0] child_job_tile_k [0:2];
    wire [63:0] child_job_addr [0:7],child_job_base [0:7],child_job_bytes [0:7];
    wire [31:0] child_job_buffer_epoch [0:7];
    logic [31:0] child_job_sequence,child_job_epoch,child_expected_epoch;
    logic child_cancel_valid;
    logic child_cancel_ready;
    logic child_done_valid;
    logic child_done_ready;
    logic [3:0] child_done_status,child_done_child_status;
    logic [2:0] child_done_stages;
    logic [1:0] child_done_phase;
    logic [16:0] child_done_bytes;
    logic [17:0] child_done_stage_bytes;
    logic [31:0] child_done_sequence,child_done_epoch,child_done_hls_status;
    localparam logic [63:0] FIRST=CONFIG==1 ? 64'hfffffffffffffffc : CONFIG==2 ? 64'd3 : 64'h00000000fffffffc;
    ntrpu_llama32_ffn_owner_r2 #(.INITIAL_TAG(FIRST)) dut(.*);
    initial clk=0;always #2.5 clk=~clk;
    integer seed=1,rng=1,cycles=0,jobs=0,mode=0,starts=0,acks=0,completed=0,resets=0,held=0;
    logic [63:0] next_model,sequence_model,epoch_model,residual_model,tag_model;
    logic [63:0] addr_model[0:7],base_model[0:7],bytes_model[0:7];
    logic [63:0] saved_completion_sequence,saved_completion_epoch;
    logic [3:0] saved_status,saved_child;
    logic [2:0] saved_stages;
    logic [1:0] saved_phase;
    logic [16:0] saved_bytes;
    logic [17:0] saved_stage_bytes;
    logic [31:0] saved_hls;
    integer local_starts,local_acks,expected_status;
    bit launched,normal_result,abort_reset;

    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"FOWNER_FAIL cycle=%0d job=%0d mode=%0d: %s",cycles,jobs,mode,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    task automatic reset_pair;
        @(negedge clk);rst_n=0;command_valid=0;cancel_valid=0;completion_ready=0;
        child_job_ready=1;child_cancel_ready=1;child_done_valid=0;
        repeat(3)tick();
        require(!busy && !completion_valid && !result_valid && !child_job_valid && !child_done_ready,"Reset must clear ownership and publication");
        next_model=FIRST;resets++;
        $display("FOWNER_RESET cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
        @(negedge clk);rst_n=1;tick();
    endtask
    task automatic check_descriptor;
        require(child_job_layout===2 && child_job_dim===2048 && child_job_hidden===8192,"Fixed Llama dimensions");
        require({child_job_epoch,child_job_sequence}===tag_model && child_expected_epoch===tag_model[63:32],"Independent aligned private tag");
        for(integer k=0;k<3;k++)require(child_job_tile_k[k]===32,"Q8 tile length");
        for(integer k=0;k<8;k++)begin
            require(child_job_addr[k]===addr_model[k] && child_job_base[k]===base_model[k] && child_job_bytes[k]===bytes_model[k],"Captured descriptor changed while held");
            require(child_job_buffer_epoch[k]===tag_model[63:32],"Private buffer epoch");
        end
    endtask
    always @(posedge clk)begin
        cycles++;
        if(cycles>40000)$fatal(1,"FOWNER_FAIL global watchdog");
        if(rst_n)begin
            if(!result_valid)require(result_ffn_address===0 && result_residual_address===0 && result_ffn_bytes===0 && result_residual_bytes===0,"Private address leak");
            if(result_valid)require(completion_valid && completion_status===0,"Premature success");
            if(busy)require(!command_ready,"Overlapping command accepted");
            if(child_job_valid)check_descriptor();
            if(child_job_valid && child_job_ready)begin
                starts++;
                $display("FOWNER_START cycle=%0d job=%0d mode=%0d tag=%016h",cycles,jobs,mode,{child_job_epoch,child_job_sequence});
            end
            if(child_done_ready)begin
                require(child_done_valid && completion_valid && completion_ready && {child_done_epoch,child_done_sequence}===tag_model,"Child released without matching public ACK");
                acks++;$display("FOWNER_ACK cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            end
            if(completion_valid && !completion_ready)held++;
        end
    end
    task automatic successful_child;
        child_done_sequence=tag_model[31:0];child_done_epoch=tag_model[63:32];
        child_done_status=0;child_done_child_status=0;child_done_phase=3;child_done_stages=4;
        child_done_bytes=8192;child_done_stage_bytes=155648;child_done_hls_status=0;
    endtask
    task automatic one_job(input integer selected_mode);
        mode=selected_mode;jobs++;local_starts=starts;local_acks=acks;launched=0;abort_reset=0;
        expected_status=0;normal_result=1;
        @(negedge clk);cancel_valid=0;completion_ready=0;child_done_valid=0;child_job_ready=1;
        command_valid=0;command_sequence=64'h1234567800000000+64'(jobs);
        command_epoch=64'hfedcba9800000000+64'(jobs*3);expected_epoch=command_epoch;
        command_source_valid=1;command_source_sequence=command_sequence;command_source_epoch=command_epoch;
        for(integer k=0;k<8;k++)begin
            command_base[k]=64'h0000001000000000+64'(k)*64'h04000000+64'(jobs)*64'h200000000;
            command_addr[k]=command_base[k]+64'($urandom_range(0,8))*64;
            command_bytes[k]=k==0 ? 8192:k<4 ? 17829888:k<6 ? 69632:k==6 ? 20480:12288;
        end
        command_residual_base=64'h0000700000000000+64'(jobs)*64'h200000;
        command_residual_address=command_residual_base+64'($urandom_range(0,8))*64;
        command_residual_bytes=16384;
        if(mode==1)expected_status=6;
        if(mode>=2 && mode<=5 || mode==10 || mode==11 || mode==44 || mode==45)expected_status=2;
        if(mode>=6 && mode<=9)expected_status=7;
        if(mode==12 || mode==13)expected_status=11;
        if(mode>=14 && mode<=19 || mode==47)expected_status=5;
        if(mode>=20 && mode<=32 || mode>=41 && mode<=43)expected_status=1;
        if(mode==51)expected_status=10;
        if(expected_status!=0 || mode>=33 && mode<=36 || mode==46)normal_result=0;
        case(mode)
            2:command_source_valid=0;
            3:command_source_sequence=command_sequence^64'h100000000;
            4:command_source_epoch=command_epoch^64'h100000000;
            5:expected_epoch=command_epoch^64'h100000000;
            6:cancel_valid=1;
            20:command_residual_bytes=8191;
            21:command_residual_address=command_residual_base+1;
            22:begin command_residual_base=64'hfffffffffffff000;command_residual_address=command_residual_base;command_residual_bytes=4095;end
            23:begin command_residual_base=64'hffffffffffffc000;command_residual_address=command_residual_base;command_residual_bytes=32768;end
            24,25,26,27:command_bytes[mode-20]=1;
            28,29,30,31:begin command_base[mode-24]=command_residual_base;command_addr[mode-24]=command_residual_address;end
            32:begin command_base[7]=command_residual_base+8192;command_addr[7]=command_base[7];command_residual_address=command_residual_base;end
            41:command_residual_bytes=0;
            42:command_addr[6]=command_base[6]+1;
            43:begin command_base[4]=64'hfffffffffffff000;command_addr[4]=command_base[4];command_bytes[4]=4095;end
        endcase
        sequence_model=command_sequence;epoch_model=command_epoch;residual_model=command_residual_address;tag_model=next_model;
        for(integer k=0;k<8;k++)begin addr_model[k]=command_addr[k];base_model[k]=command_base[k];bytes_model[k]=command_bytes[k];end
        successful_child();
        // Stale DONE and a non-ready child both prevent admission.
        child_done_valid=1;child_done_sequence=32'hdeadbeef;#1;require(!command_ready,"Stale completion admitted");tick();
        @(negedge clk);child_done_valid=0;child_job_ready=0;#1;require(!command_ready,"Busy child admitted");tick();
        @(negedge clk);child_job_ready=1;command_valid=1;#1;require(command_ready,"Idle admission unavailable");tick();
        $display("FOWNER_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h expected=%0d",cycles,jobs,mode,sequence_model,epoch_model,expected_status);
        @(negedge clk);command_valid=0;child_job_ready=0;cancel_valid=mode==7;
        // Destroy live descriptor pins after their acceptance.
        command_sequence=0;command_epoch=0;command_source_valid=0;command_source_sequence=0;command_source_epoch=0;
        command_residual_address=0;command_residual_base=0;command_residual_bytes=0;
        for(integer k=0;k<8;k++)begin command_addr[k]=0;command_base[k]=0;command_bytes[k]=0;end
        if(mode==44)expected_epoch=epoch_model^64'h100000000;
        if(mode==37)begin reset_pair();abort_reset=1;end
        if(!abort_reset)begin
            begin: admission_wait
                for(integer n=0;n<20;n++)begin
                    if(child_job_valid || completion_valid)disable admission_wait;
                    tick();
                end
                require(0,"No bounded preflight response");
            end
            if(child_job_valid)begin
                if(mode==38)begin reset_pair();abort_reset=1;end
                if(!abort_reset)begin
                    @(negedge clk);
                    if(mode==8)cancel_valid=1;
                    if(mode==10)expected_epoch=epoch_model^64'h100000000;
                    repeat(2+$urandom_range(0,5))begin tick();require(child_job_valid && !child_done_ready && !completion_valid,"Offer abandoned under backpressure");end
                    if(mode==8 || mode==10)require(child_cancel_valid,"Offer cancellation absent");
                    @(negedge clk);child_job_ready=1;tick();launched=1;
                    if(next_model!=64'hfffffffffffffffc)next_model+=4;
                    @(negedge clk);child_job_ready=0;
                    if(mode==39)begin reset_pair();abort_reset=1;end
                    if(!abort_reset)begin
                        if(mode==9 || mode==45)cancel_valid=1;
                        if(mode==11 || mode==45)expected_epoch=epoch_model^64'h100000000;
                        repeat(2+$urandom_range(0,5))begin tick();require(!completion_valid && !child_done_ready,"Child abandoned before drain");end
                        if(mode==8 || mode==9 || mode==10 || mode==11 || mode==45)require(child_cancel_valid,"Running cancellation absent");
                        @(negedge clk);successful_child();child_done_valid=1;
                        if(mode==12 || mode==13)begin
                            if(mode==12)child_done_sequence^=32'h10000000;else child_done_epoch^=32'h10000000;
                            repeat(3)begin tick();require(!completion_valid && !child_done_ready && child_cancel_valid,"Foreign completion acquired ownership");end
                            @(negedge clk);successful_child();
                        end
                        case(mode)
                            1:begin child_done_status=6;child_done_child_status=8;child_done_phase=2;child_done_stages=2;child_done_bytes=128;child_done_stage_bytes=123;child_done_hls_status=32'h55;end
                            14:child_done_stages=3;
                            15:child_done_bytes=8191;
                            16:child_done_stage_bytes=155647;
                            17:child_done_phase=2;
                            18:child_done_hls_status=1;
                            19:child_done_child_status=1;
                            47:begin child_done_status=6;child_done_stage_bytes=155649;end
                        endcase
                        tick();require(completion_valid,"Matching child did not complete");
                    end
                end
            end
            if(!abort_reset)begin
                require(completion_valid && completion_status===4'(expected_status),"Public status mismatch");
                require(completion_sequence===sequence_model && completion_epoch===epoch_model,"Truncated public identity");
                require(!child_done_ready,"Early child ACK");
                if(mode==40)begin reset_pair();abort_reset=1;end
                if(!abort_reset)begin
                    if(launched)require(completion_child_status===child_done_child_status && completion_stages===child_done_stages && completion_phase===child_done_phase && completion_bytes===child_done_bytes && completion_stage_bytes===child_done_stage_bytes && completion_hls_status===child_done_hls_status,"Lost child diagnostics");
                    else require(completion_child_status===0 && completion_stages===0 && completion_phase===0 && completion_bytes===0 && completion_stage_bytes===0 && completion_hls_status===0,"Stale child diagnostics on rejection");
                    saved_status=completion_status;saved_child=completion_child_status;saved_stages=completion_stages;saved_phase=completion_phase;saved_bytes=completion_bytes;saved_stage_bytes=completion_stage_bytes;saved_hls=completion_hls_status;
                    if(mode>=33 && mode<=36 || mode==46)begin
                        require(result_valid,"Missing result before late revocation");
                        @(negedge clk);
                        if(mode==33)cancel_valid=1;
                        if(mode==34)expected_epoch=epoch_model^64'h100000000;
                        if(mode==35)child_done_sequence^=32'h100;
                        if(mode==36)child_done_bytes=4096;
                        if(mode==46)child_done_epoch^=32'h100;
                        #1;require(!result_valid && !child_done_ready,"Late fault did not mask result immediately");tick();
                        @(negedge clk);cancel_valid=0;expected_epoch=epoch_model;successful_child();tick();
                        require(!result_valid,"Restored peer resurrected revoked result");
                    end
                    repeat(3+$urandom_range(0,5))begin
                        tick();require(completion_valid && !child_done_ready,"Response ownership not held");
                        require(completion_status===saved_status && completion_child_status===saved_child && completion_stages===saved_stages && completion_phase===saved_phase && completion_bytes===saved_bytes && completion_stage_bytes===saved_stage_bytes && completion_hls_status===saved_hls && completion_sequence===sequence_model && completion_epoch===epoch_model,"Response changed while held");
                        require(result_valid===normal_result,"Result validity mismatch");
                        if(normal_result)require(result_ffn_address===addr_model[7] && result_residual_address===residual_model && result_ffn_bytes===8192 && result_residual_bytes===8192,"Result pair descriptor mismatch");
                    end
                    $display("FOWNER_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d child=%0d stages=%0d phase=%0d bytes=%0d stage_bytes=%0d hls=%08h sequence=%016h epoch=%016h valid=%0d ffn=%016h residual=%016h ffn_bytes=%0d residual_bytes=%0d launched=%0d tag=%016h",cycles,jobs,mode,completion_status,completion_child_status,completion_stages,completion_phase,completion_bytes,completion_stage_bytes,completion_hls_status,completion_sequence,completion_epoch,result_valid,result_ffn_address,result_residual_address,result_ffn_bytes,result_residual_bytes,launched,tag_model);
                    @(negedge clk);completion_ready=1;#1;require(child_done_ready===launched,"Final child ACK mismatch");tick();
                    completed++;@(negedge clk);completion_ready=0;child_done_valid=0;child_job_ready=1;cancel_valid=0;
                    tick();require(!busy && !completion_valid && !result_valid && !child_done_ready,"Ownership leaked after release");
                    require(starts-local_starts==int'(launched) && acks-local_acks==int'(launched),"Start/ACK conservation");
                end
            end
        end
    endtask
    initial begin
        rst_n=0;command_valid=0;command_sequence=0;command_epoch=0;expected_epoch=0;
        command_source_valid=0;command_source_sequence=0;command_source_epoch=0;
        command_residual_address=0;command_residual_base=0;command_residual_bytes=0;
        cancel_valid=0;completion_ready=0;child_job_ready=1;child_cancel_ready=1;child_done_valid=0;
        tag_model=FIRST;successful_child();
        for(integer k=0;k<8;k++)begin command_addr[k]=0;command_base[k]=0;command_bytes[k]=0;end
        if($value$plusargs("SEED=%d",seed))begin end
        rng=seed;rng=$urandom(rng);reset_pair();
        if(CONFIG==0)begin
            for(integer m=0;m<48;m++)begin
                one_job(m);
                if(m>=37 && m<=40)one_job(0);
            end
        end else if(CONFIG==1)begin
            one_job(0);require(tags_exhausted,"Last aligned tag did not exhaust allocator");one_job(51);one_job(51);
        end else begin require(tags_exhausted,"Misaligned initial tag allowed");one_job(51);one_job(51);end
        $display("FOWNER_PASS config=%0d seed=%0d jobs=%0d completed=%0d starts=%0d acks=%0d resets=%0d held=%0d cycles=%0d",CONFIG,seed,jobs,completed,starts,acks,resets,held,cycles);
        $finish;
    end
endmodule
`default_nettype wire
