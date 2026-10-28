`timescale 1ns/1ps
module tb_ntrpu_matrix_sequencer;
    logic job_output_fp32=0;
    logic e_job_output_fp32;
    logic clk=0;
    logic rst_n=0;
    logic job_valid=0;
    logic job_ready;
    logic [1:0] job_layout=0;
    logic [13:0] job_rows=0;
    logic [8:0] job_k_blocks=0;
    logic [5:0] job_tile_k=0;
    logic [31:0] job_sequence=0;
    logic [31:0] job_epoch=0;
    logic [63:0] job_weight_addr=0;
    logic [63:0] job_region_base=0;
    logic [63:0] job_region_bytes=0;
    logic [63:0] job_activation_addr=0;
    logic [63:0] job_activation_region_base=0;
    logic [63:0] job_activation_region_bytes=0;
    logic [31:0] job_activation_epoch=0;
    logic [63:0] job_output_addr=0;
    logic [63:0] job_output_region_base=0;
    logic [63:0] job_output_region_bytes=0;
    logic [31:0] job_output_epoch=0;
    logic cancel_valid=0;
    logic cancel_ready;
    logic done_valid;
    logic done_ready=0;
    logic [3:0] done_status;
    logic [16:0] done_bytes;
    logic [13:0] done_rows;
    logic [31:0] done_sequence;
    logic [31:0] done_epoch;
    logic e_job_valid;
    logic e_job_ready=0;
    logic [1:0] e_job_layout;
    logic [5:0] e_job_rows;
    logic [8:0] e_job_k_blocks;
    logic [31:0] e_job_sequence;
    logic [31:0] e_job_epoch;
    logic [63:0] e_job_region_base;
    logic [63:0] e_job_region_bytes;
    logic [63:0] e_job_activation_addr;
    logic [63:0] e_job_activation_region_base;
    logic [63:0] e_job_activation_region_bytes;
    logic [31:0] e_job_activation_epoch;
    logic [63:0] e_job_output_addr;
    logic [63:0] e_job_output_region_base;
    logic [63:0] e_job_output_region_bytes;
    logic [31:0] e_job_output_epoch;
    logic e_tile_valid;
    logic e_tile_ready=0;
    logic [63:0] e_tile_addr;
    logic [8:0] e_tile_k_base;
    logic [5:0] e_tile_k_count;
    logic [31:0] e_tile_tag;
    logic [31:0] e_tile_epoch;
    logic e_cancel_valid;
    logic e_cancel_ready=0;
    logic e_done_valid=0;
    logic e_done_ready;
    logic [3:0] e_done_status=0;
    logic [8:0] e_done_bytes=0;
    logic [31:0] e_done_sequence=0;
    logic [31:0] e_done_epoch=0;

    always #2.5 clk=~clk;
    ntrpu_matrix_sequencer dut(.*);
    localparam logic [63:0] W=64'h100000000,A=64'h8000000000,O=64'ha000000000;
    int seed=1,random_state,ident=0,cycles=0,total_cycles=0,total_groups=0,total_tiles=0,total_resets=0;
    int n,k,chunk,mode,groups,tiles,k_seen,group_n,row_base,wait_done,delay_count,group_index;
    int sum_bytes,sum_rows,case_resets,job_stalls=0,tile_stalls=0,done_stalls=0,cancel_count=0;
    bit active,pending,aborted,accepted,finished,cancelled,job_hold,tile_hold,done_hold;
    logic [31:0] sequence_root,epoch_root;
    logic [511:0] saved_job;
    logic [143:0] saved_tile;
    logic [98:0] saved_done;
    function automatic void require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL MATRIX SEQUENCER case=%0d cycle=%0d: %s",ident,cycles,why);
    endfunction
    task automatic run_case(input int nr,input int kb,input int tk,input int scenario,input int status);
        bit job_hs,engine_hs,tile_hs,completion_hs,cancel_hs,engine_cancel_hs;
        int wanted_count,wanted_status,wanted_bytes;
        logic [63:0] wanted_addr;
        ident++;n=nr;k=kb;chunk=tk;mode=scenario;cycles=0;groups=0;tiles=0;row_base=0;
        sum_bytes=0;sum_rows=0;case_resets=0;k_seen=0;wait_done=0;delay_count=0;group_index=0;
        active=0;pending=0;aborted=0;accepted=0;finished=0;cancelled=0;job_hold=0;tile_hold=0;done_hold=0;
        sequence_root=32'hfffffff0+32'(ident);epoch_root=32'(ident+300);
        @(negedge clk);require(job_ready,"previous job remains active");
        job_layout=2;job_rows=14'(n);job_k_blocks=9'(k);job_tile_k=6'(chunk);
        job_sequence=sequence_root;job_epoch=epoch_root;job_weight_addr=W+4032;job_region_base=W;job_region_bytes=64'h10000000;
        job_activation_addr=A+4032;job_activation_region_base=A;job_activation_region_bytes=64'h100000;
        job_activation_epoch=epoch_root;job_output_addr=O+4032;job_output_region_base=O;job_output_region_bytes=64'h100000;
        job_output_epoch=epoch_root;job_valid=1;cancel_valid=0;e_done_valid=0;
        case(mode)
            20:job_layout=1;
            21:job_weight_addr++;
            22:job_activation_addr++;
            23:job_output_addr++;
            24:job_region_bytes=64'(4032+n*k*34-1);
            25:job_activation_region_bytes=64'(4032+k*64-1);
            26:job_output_region_bytes=64'(4032+((n+7)/8)*64-1);
            27:job_activation_epoch++;
            28:job_output_epoch++;
            29:job_region_base=64'hfffffffffffff000;
            30:begin job_output_addr=W+4032;job_output_region_base=W;end
            31:begin job_output_addr=A+4032;job_output_region_base=A;end
            32:begin job_output_addr=64'hffffffffffffffc0;job_output_region_base=job_output_addr;job_output_region_bytes=128;end
            33:job_weight_addr=W-64;
        endcase
        while(!finished) begin
            e_job_ready=!active && !e_done_valid && $urandom_range(0,3)!=0;
            if(mode==9 && cycles<40) e_job_ready=0;
            e_tile_ready=active && !pending && !aborted && $urandom_range(0,3)!=0;
            e_cancel_ready=active && !pending && !aborted;
            done_ready=wait_done>=5 && $urandom_range(0,3)!=0;
            if(!cancelled && cancel_ready && ((mode==8 && dut.state==1) ||
                (mode==9 && e_job_valid && cycles>=4) || (mode==10 && active && k_seen>0) ||
                (mode==11 && pending) || (mode==12 && dut.state==9))) cancel_valid=1;
            if(pending && !e_done_valid) begin
                if(delay_count>0) delay_count--;
                else begin
                    e_done_valid=1;e_done_sequence=sequence_root+32'(group_index);e_done_epoch=epoch_root;
                    e_done_status=aborted ? 7:0;e_done_bytes=aborted ? 0:9'(group_n*8);
                    if((mode==1 && group_index==0) || (mode==2 && group_index==1)) begin e_done_status=3;e_done_bytes=64;end
                    if(mode==4 && group_index==1) e_done_sequence++;
                    if(mode==5 && group_index==1) e_done_epoch++;
                    if(mode==6 && group_index==1) e_done_bytes-=8;
                end
            end
            @(posedge clk);cycles++;require(cycles<300000,"timeout");
            job_hs=job_valid&&job_ready;engine_hs=e_job_valid&&e_job_ready;tile_hs=e_tile_valid&&e_tile_ready;
            completion_hs=e_done_valid&&e_done_ready;cancel_hs=cancel_valid&&cancel_ready;
            engine_cancel_hs=e_cancel_valid&&e_cancel_ready;
            if(job_hold) require(e_job_valid && {e_job_rows,e_job_k_blocks,e_job_sequence,e_job_epoch,e_job_region_base,e_job_region_bytes,e_job_activation_addr,e_job_output_addr}===saved_job,"engine job stable");
            if(tile_hold) require(e_tile_valid && {e_tile_addr,e_tile_k_base,e_tile_k_count,e_tile_tag,e_tile_epoch}===saved_tile || completion_hs,"tile stable until engine ends");
            if(done_hold) require(done_valid && {done_status,done_bytes,done_rows,done_sequence,done_epoch}===saved_done,"matrix completion stable");
            job_hold=e_job_valid&&!e_job_ready;tile_hold=e_tile_valid&&!e_tile_ready;done_hold=done_valid&&!done_ready;
            saved_job={e_job_rows,e_job_k_blocks,e_job_sequence,e_job_epoch,e_job_region_base,e_job_region_bytes,e_job_activation_addr,e_job_output_addr};
            saved_tile={e_tile_addr,e_tile_k_base,e_tile_k_count,e_tile_tag,e_tile_epoch};saved_done={done_status,done_bytes,done_rows,done_sequence,done_epoch};
            job_stalls+=int'(job_hold);tile_stalls+=int'(tile_hold);done_stalls+=int'(done_hold);
            if(job_hs) begin require(!accepted,"duplicate matrix job");accepted=1;end
            if(cancel_hs) begin cancelled=1;cancel_count++;end
            if(engine_hs) begin
                require(accepted && !active && !pending,"overlapping engine job");
                group_n=n-row_base>32 ? 32:n-row_base;group_index=groups;
                require(e_job_layout==2 && int'(e_job_rows)==group_n && int'(e_job_k_blocks)==k,"engine shape");
                require(e_job_sequence==sequence_root+32'(groups) && e_job_epoch==epoch_root,"engine identity including wrap");
                require(e_job_region_base==W && e_job_region_bytes==64'h10000000,"registered weights");
                require(e_job_activation_addr==A+4032 && e_job_activation_region_base==A && e_job_activation_region_bytes==64'h100000 && e_job_activation_epoch==epoch_root,"registered activation");
                require(e_job_output_addr==O+4032+64'(row_base*8) && e_job_output_region_base==O && e_job_output_region_bytes==64'h100000 && e_job_output_epoch==epoch_root,"registered output");
                $display("NTRPU_MATRIX_GROUP case=%0d index=%0d first_row=%0d rows=%0d sequence=%0d",ident,groups,row_base,group_n,e_job_sequence);
                groups++;active=1;k_seen=0;aborted=0;
            end
            if(tile_hs) begin
                require(active && !pending && !aborted,"unexpected tile");
                wanted_count=k-k_seen>chunk ? chunk:k-k_seen;
                wanted_addr=W+4032+64'(row_base*k*34+k_seen*group_n*34);
                require(e_tile_addr==wanted_addr && e_tile_addr[5:0]==0,"compact tile address/high bits");
                require(int'(e_tile_k_base)==k_seen && int'(e_tile_k_count)==wanted_count && e_tile_tag==32'(tiles) && e_tile_epoch==epoch_root,"tile count/tag/epoch");
                require(group_n*wanted_count%32==0,"whole AXI beats");
                tiles++;k_seen+=wanted_count;
                if(k_seen==k || mode==7) begin pending=1;delay_count=3;end
            end
            if(engine_cancel_hs) begin aborted=1;pending=1;delay_count=2;end
            if(completion_hs) begin
                require(active && pending,"unexpected engine completion");
                sum_bytes+=int'(e_done_bytes);
                if(e_done_status==0 && e_done_sequence==sequence_root+32'(group_index) && e_done_epoch==epoch_root && int'(e_done_bytes)==group_n*8 && k_seen==k) sum_rows+=group_n;
                active=0;pending=0;row_base+=group_n;tile_hold=0;
            end
            if(done_valid) begin
                wait_done++;require(!active && !pending && !e_job_valid && !e_tile_valid,"matrix done before drain");
                require(done_status==4'(status),$sformatf("status %0d wanted %0d",done_status,status));
                require(done_sequence==sequence_root && done_epoch==epoch_root,"original completion identity");
                require(int'(done_bytes)==sum_bytes && int'(done_rows)==sum_rows,"completion private counters");
                if(status==0) require(groups==(n+31)/32 && tiles==groups*((k+chunk-1)/chunk) && sum_bytes==n*8 && sum_rows==n,"complete matrix");
                if(mode>=20 || (status==1 && groups==0) || mode==8) require(groups==0 && tiles==0,"rejected command issued engine work");
                if(mode!=14 && done_ready) finished=1;
            end
            #1;
            if(job_hs) begin
                job_valid=0;job_layout=0;job_rows=0;job_k_blocks=0;job_tile_k=0;job_sequence=0;job_epoch=0;
                job_weight_addr=0;job_region_base=0;job_region_bytes=0;job_activation_addr=0;job_activation_region_base=0;
                job_activation_region_bytes=0;job_activation_epoch=0;job_output_addr=0;job_output_region_base=0;job_output_region_bytes=0;job_output_epoch=0;
            end
            if(cancel_hs) cancel_valid=0;
            if(completion_hs) e_done_valid=0;
            if((mode==13 && active && k_seen>0) || (mode==14 && done_valid && !done_ready)) begin
                @(negedge clk);rst_n=0;e_done_valid=0;active=0;pending=0;job_valid=0;cancel_valid=0;
                repeat(4) @(negedge clk);rst_n=1;repeat(4) @(negedge clk);
                require(job_ready && !done_valid && !e_job_valid && !e_tile_valid && !e_cancel_valid,"coordinated reset recovery");
                case_resets=1;total_resets++;finished=1;
            end
            if(!finished) @(negedge clk);
        end
        $display("NTRPU_MATRIX_CASE id=%0d rows=%0d blocks=%0d chunk=%0d mode=%0d status=%0d cycles=%0d groups=%0d tiles=%0d bytes=%0d completed_rows=%0d reset=%0d",ident,n,k,chunk,mode,status,cycles,groups,tiles,sum_bytes,sum_rows,case_resets);
        total_cycles+=cycles;total_groups+=groups;total_tiles+=tiles;
        cancel_valid=0;repeat(3) @(negedge clk);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_MATRIX_SEED seed=%0d",seed);random_state=$urandom(seed);
        repeat(4) @(negedge clk);rst_n=1;@(posedge clk);#1;
        run_case(1,32,32,0,0);run_case(31,64,32,0,0);run_case(32,9,4,0,0);run_case(33,64,32,0,0);
        run_case(720,8,4,0,0);run_case(8192,64,32,0,0);run_case(2048,256,32,0,0);run_case(8191,256,32,0,0);
        run_case(8192,256,1,0,0);
        run_case(64,64,32,1,3);run_case(64,64,32,2,3);
        run_case(64,64,32,4,5);run_case(64,64,32,5,5);run_case(64,64,32,6,5);run_case(64,64,32,7,5);
        run_case(64,64,32,8,7);run_case(64,64,32,9,7);run_case(64,64,32,10,7);run_case(64,64,32,11,7);run_case(64,64,32,12,7);
        run_case(64,64,32,13,0);run_case(32,32,32,14,0);run_case(33,64,32,0,0);
        run_case(0,32,32,0,1);run_case(8193,32,32,0,1);run_case(32,0,32,0,1);run_case(32,257,32,0,1);
        run_case(32,32,0,0,1);run_case(32,32,33,0,1);run_case(33,33,32,0,1);
        for(int m=20;m<=33;m++) run_case(64,64,32,m,m==27 || m==28 ? 2:1);
        run_case(64,64,32,0,0);
        require(ident==45 && total_resets==2 && job_stalls>0 && tile_stalls>0 && done_stalls>0 && cancel_count==5,"suite coverage");
        $display("NTRPU_MATRIX_PASS cases=%0d cycles=%0d groups=%0d tiles=%0d resets=%0d job_stalls=%0d tile_stalls=%0d done_stalls=%0d cancels=%0d",ident,total_cycles,total_groups,total_tiles,total_resets,job_stalls,tile_stalls,done_stalls,cancel_count);
        $finish;
    end
endmodule
