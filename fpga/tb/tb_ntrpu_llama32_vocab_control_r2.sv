`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_llama32_vocab_control_r2 #(
    parameter logic [63:0] INITIAL_TAG=64'h00000000ffffff00,
    parameter integer TAG_TEST=0
);
    logic clk,rst_n;
    logic command_valid;
    wire command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic cancel_valid;
    wire cancel_ready,busy;
    logic source_completion_valid;
    wire source_completion_ready;
    logic [3:0] source_completion_status;
    logic [63:0] source_sequence,source_epoch,source_result_address;
    logic source_result_valid;
    logic [12:0] source_result_bytes;
    wire child_job_valid;
    logic child_job_ready;
    wire [1:0] child_job_layout;
    wire [13:0] child_job_rows;
    wire child_job_output_fp32;
    wire [8:0] child_job_k_blocks;
    wire [5:0] child_job_tile_k;
    wire [31:0] child_job_sequence,child_job_epoch;
    wire [63:0] child_job_weight_addr,child_job_region_base,child_job_region_bytes;
    wire [63:0] child_job_activation_addr,child_job_activation_region_base,child_job_activation_region_bytes;
    wire [31:0] child_job_activation_epoch;
    wire [63:0] child_job_output_addr,child_job_output_region_base,child_job_output_region_bytes;
    wire [31:0] child_job_output_epoch;
    wire child_cancel_valid;
    logic child_cancel_ready;
    logic child_done_valid;
    wire child_done_ready;
    logic [3:0] child_done_status;
    logic [16:0] child_done_bytes;
    logic [13:0] child_done_rows;
    logic [31:0] child_done_sequence,child_done_epoch;
    wire completion_valid;
    logic completion_ready;
    wire [3:0] completion_status,completion_source_status,completion_child_status;
    wire [63:0] completion_sequence,completion_epoch;
    wire [3:0] completion_phase;
    wire [4:0] completion_matrices;
    wire [18:0] completion_total_bytes;
    wire [16:0] completion_child_bytes;
    wire [13:0] completion_child_rows;
    wire completion_revoked,result_valid,tags_exhausted;
    wire [63:0] result_address;
    wire [18:0] result_bytes;
    logic [2:0][63:0] drive_addr,drive_base,drive_bytes,model_addr,model_base,model_bytes;
    wire [63:0] command_addr [0:2],command_base [0:2],command_bytes [0:2];
    for(genvar i=0;i<3;i++)begin
        assign command_addr[i]=drive_addr[i];assign command_base[i]=drive_base[i];assign command_bytes[i]=drive_bytes[i];
    end
    ntrpu_llama32_vocab_control_r1 #(.INITIAL_TAG(INITIAL_TAG),.CHILD_TIMEOUT_CYCLES(160)) dut(.*);
    always #2.5 clk=~clk;
    integer seed=1,random_state=1,cycles=0,jobs=0,mode_now=0,phase_model=0;
    integer offers=0,child_acks=0,source_acks=0,publics=0,resets=0,cancels=0,drains=0;
    integer reserved_jobs=0,held_samples=0,revocations=0;
    logic [63:0] model_sequence,model_epoch,model_tag,next_model_tag;
    logic public_seen,expect_terminal,active_child,reject_model;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"VOCAB_FAIL cycle=%0d job=%0d mode=%0d phase=%0d: %s",cycles,jobs,mode_now,phase_model,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic integer wanted(input integer mode);
        if(mode==1 || mode==2 || mode==3 || mode==31 || mode==41 || mode==56)return 2;
        if(mode>=4 && mode<=7 || mode>=51 && mode<=53)return 5;
        if(mode==10 || mode==20 || mode==21)return 1;
        if(mode==30 || mode==40 || mode==55)return 7;
        if(mode>=32 && mode<=38 || mode==43 || mode==54 || mode==57)return 11;
        if(mode==42 || mode==58 || mode==59)return 8;
        if(mode==50)return 3;
        if(mode==90)return 10;
        return 0;
    endfunction
    function automatic logic [63:0] bytes_for(input integer index);
        case(index)
            0:return 4096;
            1:return 279085056;
            default:return 513024;
        endcase
    endfunction
    always @(posedge clk)begin
        cycles++;
        if(rst_n)begin
            if(busy)begin
                require(!command_ready,"Busy admission");
                require(completion_sequence==model_sequence && completion_epoch==model_epoch,
                    "Captured identity changed during ownership");
            end
            if(child_job_valid)begin
                require(!reject_model,"Rejected command offered to child");
                require(child_job_layout==2 && child_job_output_fp32 && child_job_k_blocks==64 && child_job_tile_k==32,"Matrix numeric format");
                require(child_job_rows==(phase_model==15 ? 5376:8192),"Matrix row count");
                require({child_job_epoch,child_job_sequence}==model_tag+64'(phase_model*256),"Reserved full-width tag");
                require(child_job_activation_epoch==child_job_epoch && child_job_output_epoch==child_job_epoch,"Buffer epoch mapping");
                require(child_job_activation_addr==model_addr[0] && child_job_activation_region_base==model_base[0] &&
                    child_job_activation_region_bytes==model_bytes[0],"Captured activation allocation");
                require(child_job_weight_addr==model_addr[1]+64'(phase_model)*64'h1100000 && child_job_region_base==model_base[1] &&
                    child_job_region_bytes==model_bytes[1],"Captured weight allocation");
                require(child_job_output_addr==model_addr[2]+64'(phase_model)*64'h8000 && child_job_output_region_base==model_base[2] &&
                    child_job_output_region_bytes==model_bytes[2],"Captured output allocation");
                if(child_job_ready)begin
                    require(!active_child,"Overlapping matrix children");active_child=1;offers++;
                    $display("VOCAB_OFFER cycle=%0d job=%0d phase=%0d tag=%016h rows=%0d weight=%016h activation=%016h output=%016h",cycles,jobs,phase_model,{child_job_epoch,child_job_sequence},child_job_rows,child_job_weight_addr,child_job_activation_addr,child_job_output_addr);
                end else begin
                    held_samples++;
                    $display("VOCAB_HOLD cycle=%0d job=%0d phase=%0d tag=%016h rows=%0d weight=%016h activation=%016h output=%016h",cycles,jobs,phase_model,{child_job_epoch,child_job_sequence},child_job_rows,child_job_weight_addr,child_job_activation_addr,child_job_output_addr);
                end
            end
            if(child_cancel_valid)begin
                require(active_child,"Unowned child cancellation");
                if(child_cancel_ready)begin cancels++;$display("VOCAB_CANCEL cycle=%0d job=%0d phase=%0d",cycles,jobs,phase_model);end
            end
            if(completion_valid && completion_ready)begin
                require(!public_seen,"Duplicate public handshake");public_seen=1;publics++;
                $display("VOCAB_PUBLIC cycle=%0d job=%0d status=%0d revoked=%0d valid=%0d matrices=%0d bytes=%0d sequence=%016h epoch=%016h output=%016h output_bytes=%0d",cycles,jobs,completion_status,completion_revoked,result_valid,completion_matrices,completion_total_bytes,completion_sequence,completion_epoch,result_address,result_bytes);
            end
            if(child_done_ready)begin
                require(active_child && child_done_valid && {child_done_epoch,child_done_sequence}==model_tag+64'(phase_model*256),"Foreign or unsolicited child ACK");
                if(expect_terminal)require(public_seen,"Terminal child released before public ACK");
                active_child=0;child_acks++;
                $display("VOCAB_CHILD_ACK cycle=%0d job=%0d phase=%0d public=%0d",cycles,jobs,phase_model,public_seen);
            end
            if(source_completion_ready)begin
                require(public_seen && source_completion_valid && source_sequence==model_sequence && source_epoch==model_epoch,"Foreign or premature source ACK");
                require(!active_child || child_done_ready,"Source released before child drain");
                source_acks++;$display("VOCAB_SOURCE_ACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(result_valid)begin
                require(completion_valid && completion_status==0 && completion_matrices==16 && completion_total_bytes==513024 &&
                    !completion_revoked && expected_epoch==model_epoch && source_result_valid && source_completion_valid &&
                    source_sequence==model_sequence && source_epoch==model_epoch && child_done_valid && child_done_status==0,"Premature or revoked publication");
                require(result_address==model_addr[2] && result_bytes==513024,"Public logits descriptor");
            end else require(result_address==0 && result_bytes==0,"Private result descriptor leak");
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;
        source_completion_valid=0;child_done_valid=0;child_job_ready=1;child_cancel_ready=1;
        public_seen=0;active_child=0;expect_terminal=0;reject_model=0;resets++;next_model_tag=INITIAL_TAG;
        repeat(3)tick();@(negedge clk);rst_n=1;tick();
        require(!busy && !completion_valid && !result_valid && !child_done_ready && !source_completion_ready,"Coordinated reset flush");
        $display("VOCAB_RESET cycle=%0d job=%0d",cycles,jobs);
    endtask
    task automatic await_public;
        integer limit;
        limit=0;
        while(!completion_valid && limit<400)begin tick();limit++;end
        require(completion_valid,"Public response deadline");
    endtask
    task automatic finish_public(input integer wanted_status,input bit good_result,input integer expected_source_ack);
        integer sa,ca,pu;logic [134:0] held;
        await_public();
        require(completion_status==wanted_status && result_valid==good_result,"Completion status/result decision");
        require(completion_sequence==model_sequence && completion_epoch==model_epoch,"Full public identity");
        held={completion_status,completion_source_status,completion_child_status,completion_sequence,
            completion_phase,completion_matrices,completion_total_bytes,completion_child_bytes,completion_child_rows};
        repeat(4)begin tick();require(completion_valid &&
            {completion_status,completion_source_status,completion_child_status,completion_sequence,
            completion_phase,completion_matrices,completion_total_bytes,completion_child_bytes,completion_child_rows}==held,"Held completion stability");end
        sa=source_acks;ca=child_acks;pu=publics;
        @(negedge clk);completion_ready=1;tick();
        require(publics==pu+1 && source_acks==sa+expected_source_ack && !busy,"Exact public retirement");
        @(negedge clk);completion_ready=0;child_done_valid=0;source_completion_valid=0;tick();
        require(!completion_valid && !result_valid && !active_child,"Clean retirement");
        $display("VOCAB_END cycle=%0d job=%0d mode=%0d status=%0d source_acks=%0d",cycles,jobs,mode_now,wanted_status,expected_source_ack);
    endtask
    task automatic accept_cancel;
        integer limit,before_count;
        before_count=cancels;limit=0;
        while(!child_cancel_valid && limit<12)begin tick();limit++;end
        require(child_cancel_valid,"Registered child cancellation not forwarded");
        repeat(4)begin
            tick();require(child_cancel_valid && cancels==before_count && active_child,
                "Held child cancellation withdrawn or prematurely consumed");
        end
        @(negedge clk);child_cancel_ready=1;tick();
        require(cancels==before_count+1,"Exact child cancellation handshake");
        @(negedge clk);
    endtask
    task automatic run_job(input integer mode,input integer target,input integer variant);
        integer oi,sa,ca,pu,cn,limit,stop_phase,expected_status,random_delay;
        logic [63:0] good_tag;
        logic early,phase_fault,needs_drain;
        begin : job_body
        jobs++;mode_now=mode;phase_model=0;oi=offers;sa=source_acks;ca=child_acks;pu=publics;cn=cancels;
        @(negedge clk);require(!busy,"Previous transaction still owned");
        model_sequence=64'h13579bdf00000000+64'(jobs);model_epoch=64'hfedcba9800000000+64'(jobs);
        model_tag=next_model_tag;public_seen=0;active_child=0;expect_terminal=0;
        command_sequence=model_sequence;command_epoch=model_epoch;expected_epoch=model_epoch;
        source_completion_valid=1;source_completion_status=0;source_sequence=model_sequence;source_epoch=model_epoch;
        source_result_valid=1;source_result_bytes=4096;
        for(integer i=0;i<3;i++)begin
            drive_addr[i]=64'h100000000+64'(i)*64'h40000000+64'(jobs)*64'h10000;
            drive_base[i]=drive_addr[i]-4032;drive_bytes[i]=bytes_for(i)+4032;
        end
        source_result_address=drive_addr[0];
        child_job_ready=1;child_done_valid=0;child_cancel_ready=!(mode>=40 && mode<=43 || mode>=54 && mode<=58);
        child_done_status=0;child_done_rows=0;child_done_bytes=0;child_done_sequence=0;child_done_epoch=0;
        cancel_valid=0;completion_ready=0;
        case(mode)
            1:source_sequence=source_sequence^64'h0000000100000000;
            2:source_epoch=source_epoch^64'h0000000100000000;
            3:expected_epoch=expected_epoch^64'h0000000100000000;
            4:source_completion_status=3;
            5:source_result_valid=0;
            6:source_result_address=source_result_address+64;
            7:source_result_bytes=4032;
            10:case(variant)
                0:drive_addr[target]=drive_addr[target]+1;
                1:drive_base[target]=drive_base[target]+1;
                2:drive_bytes[target]=drive_bytes[target]-1;
                3:drive_bytes[target]=0;
                4:drive_addr[target]=drive_base[target]-64;
                5:drive_bytes[target]=bytes_for(target)+4032-64;
                6:begin drive_base[target]=64'hffffffffffffff00;drive_addr[target]=drive_base[target];drive_bytes[target]=4096;end
                7:begin drive_addr[target]=64'hffffffffffffffc0;drive_base[target]=drive_addr[target];drive_bytes[target]=64;end
            endcase
            20:begin drive_base[target]=drive_base[variant];drive_addr[target]=drive_addr[variant];drive_bytes[target]=bytes_for(variant)+bytes_for(target)+4032;end
            21:begin
                drive_base[target]=drive_base[variant];drive_addr[target]=drive_addr[variant]+bytes_for(variant)+8192;
                drive_bytes[target]=drive_addr[target]-drive_base[target]+bytes_for(target);
            end
        endcase
        // Source address continues to match the supplied ABI1 for range checks;
        // only source-shape cases deliberately disagree with it.
        if(mode==10 && target==0)source_result_address=drive_addr[0];
        model_addr=drive_addr;model_base=drive_base;model_bytes=drive_bytes;
        expected_status=wanted(mode);
        reject_model=mode>=1 && mode<=7 || mode==10 || mode==20 || mode==21 || mode>=30 && mode<=38 || mode==90;
        $display("VOCAB_BEGIN cycle=%0d job=%0d mode=%0d target=%0d variant=%0d expect=%0d sequence=%016h epoch=%016h tag=%016h",cycles,jobs,mode,target,variant,expected_status,model_sequence,model_epoch,model_tag);
        command_valid=1;#1;require(command_ready,"Ready admission");tick();
        @(negedge clk);command_valid=0;child_job_ready=0;
        command_sequence=~model_sequence;command_epoch=~model_epoch;drive_addr='1;drive_base='1;drive_bytes='1;
        if(mode==30)cancel_valid=1;
        if(mode==31)expected_epoch=~model_epoch;
        case(mode)
            32:source_sequence=source_sequence^64'h0000000100000000;
            33:source_epoch=source_epoch^64'h0000000100000000;
            34:source_completion_status=3;
            35:source_result_valid=0;
            36:source_result_address=source_result_address+64;
            37:source_result_bytes=4032;
            38:source_completion_valid=0;
        endcase
        if(mode>=30 && mode<=38)begin
            tick();@(negedge clk);cancel_valid=0;expected_epoch=model_epoch;
            source_sequence=model_sequence;source_epoch=model_epoch;source_completion_status=0;
            source_result_valid=1;source_result_address=model_addr[0];source_result_bytes=4096;source_completion_valid=1;
        end
        early=mode>=1 && mode<=7 || mode==10 || mode==20 || mode==21 || mode>=30 && mode<=38 || mode==90;
        if(early)begin
            expect_terminal=1;finish_public(expected_status,0,mode==1 || mode==2 ? 0:1);
            require(offers==oi && child_acks==ca,"Rejected descriptor reached child");
            if(mode==90)require(tags_exhausted,"Tag exhaustion missing");
            disable job_body;
        end
        stop_phase=mode>=50 && mode<=58 ? target:15;
        for(integer phase=0;phase<=stop_phase;phase++)begin
            phase_model=phase;
            if(phase>0)begin
                @(negedge clk);child_job_ready=1;tick();@(negedge clk);child_job_ready=0;
            end
            limit=0;while(!child_job_valid && limit<120)begin tick();limit++;end
            require(child_job_valid && !completion_valid,"Validated command was not offered");
            if(phase==0)begin next_model_tag=next_model_tag+4096;reserved_jobs++;end
            random_delay=3+($urandom(random_state)%5);repeat(random_delay)tick();
            if(mode==70)begin reset_all();$display("VOCAB_RESET_JOB job=%0d mode=%0d",jobs,mode);disable job_body;end
            if(mode>=40 && mode<=43 && phase==0)begin
                @(negedge clk);
                if(mode==40)cancel_valid=1;
                if(mode==41)expected_epoch=~model_epoch;
                if(mode==43)begin child_done_valid=1;child_done_sequence=32'hdeadbeef;child_done_epoch=32'hbadcafe0;end
                repeat(mode==42 ? 165:3)begin tick();require(child_job_valid && !child_done_ready && !child_cancel_valid,"Offered command withdrawn or foreign/unowned ACK");end
                @(negedge clk);cancel_valid=0;expected_epoch=model_epoch;child_done_valid=0;
                stop_phase=0;
            end
            @(negedge clk);child_job_ready=1;tick();
            require(active_child && offers==oi+phase+1,"Child admission count");
            @(negedge clk);child_job_ready=0;
            if(mode==71 && phase==target)begin reset_all();$display("VOCAB_RESET_JOB job=%0d mode=%0d",jobs,mode);disable job_body;end
            phase_fault=mode>=40 && mode<=43 || mode>=50 && mode<=58 && phase==target;
            if(mode>=55 && mode<=58 && phase==target)begin
                if(mode==55)cancel_valid=1;
                if(mode==56)expected_epoch=~model_epoch;
                if(mode==57)source_result_address=source_result_address+64;
                repeat(mode==58 ? 165:3)tick();
                @(negedge clk);cancel_valid=0;expected_epoch=model_epoch;source_result_address=model_addr[0];
            end else repeat(3+($urandom(random_state)%5))tick();
            if(mode>=40 && mode<=43 || mode>=55 && mode<=58 && phase==target)accept_cancel();
            good_tag=model_tag+64'(phase*256);
            @(negedge clk);child_done_valid=1;child_done_sequence=good_tag[31:0];child_done_epoch=good_tag[63:32];
            child_done_status=phase_fault ? (mode==50 ? 3:7):0;
            child_done_rows=phase_fault ? 0:(phase==15 ? 5376:8192);
            child_done_bytes=phase_fault ? 0:(phase==15 ? 21504:32768);
            if(mode>=51 && mode<=54 && phase==target)begin
                child_done_status=0;child_done_rows=phase==15 ? 5376:8192;child_done_bytes=phase==15 ? 21504:32768;
                if(mode==51)child_done_bytes=child_done_bytes-64;
                if(mode==52)child_done_rows=child_done_rows-1;
                if(mode==53)child_done_bytes=child_done_bytes+64;
                if(mode==54)begin
                    child_done_epoch=child_done_epoch^32'h00010000;
                    repeat(4)begin tick();require(!child_done_ready && !completion_valid,"Foreign child response acknowledged");end
                    accept_cancel();
                    @(negedge clk);child_done_epoch=good_tag[63:32];
                end
            end
            expect_terminal=phase==stop_phase;
            $display("VOCAB_DONE cycle=%0d job=%0d phase=%0d status=%0d rows=%0d bytes=%0d tag=%016h",cycles,jobs,phase,child_done_status,child_done_rows,child_done_bytes,{child_done_epoch,child_done_sequence});
            tick();
            if(phase<stop_phase)begin
                require(!completion_valid && !result_valid && child_acks==ca+phase+1 && source_acks==sa,"Partial matrices published or source released");
                @(negedge clk);child_done_valid=0;
                if(mode==59 && phase==0)begin
                    repeat(165)tick();expect_terminal=1;finish_public(8,0,1);
                    require(offers==oi+1 && child_acks==ca+1,"Advance watchdog issued another matrix");disable job_body;
                end
            end
        end
        await_public();
        if(mode==72)begin reset_all();$display("VOCAB_RESET_JOB job=%0d mode=%0d",jobs,mode);disable job_body;end
        if(mode>=60 && mode<=69 || mode==73)begin
            require(result_valid && completion_status==0,"Positive pre-revocation result");
            @(negedge clk);
            case(mode)
                60:expected_epoch=~model_epoch;
                61:cancel_valid=1;
                62:source_sequence=source_sequence^64'h0000000100000000;
                63,73:child_done_epoch=child_done_epoch^32'h00010000;
                64:source_completion_valid=0;
                65:child_done_valid=0;
                66:child_done_bytes=child_done_bytes-64;
                67:source_result_address=source_result_address+64;
                68:source_epoch=source_epoch^64'h0000000100000000;
                69:child_done_status=3;
            endcase
            #1;require(!result_valid && completion_revoked,"Immediate held-result revocation");tick();revocations++;
            $display("VOCAB_REVOKE cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            needs_drain=mode>=62 && mode<=65 || mode==68 || mode==73;
            if(needs_drain)begin
                @(negedge clk);completion_ready=1;tick();drains++;
                require(public_seen && !completion_valid && busy && source_acks==sa && child_acks==ca+15,"Public ACK while peer foreign did not enter drain");
                if(mode==73)begin reset_all();$display("VOCAB_RESET_JOB job=%0d mode=%0d",jobs,mode);disable job_body;end
                repeat(3)begin tick();require(!completion_valid && !source_completion_ready && !child_done_ready && !command_ready,"Drain republished or released foreign ownership");end
                @(negedge clk);completion_ready=0;source_sequence=model_sequence;source_epoch=model_epoch;
                source_completion_valid=1;child_done_valid=1;child_done_epoch=good_tag[63:32];
                tick();require(!busy && source_acks==sa+1 && child_acks==ca+16 && publics==pu+1,"Drain release count");
                @(negedge clk);source_completion_valid=0;child_done_valid=0;tick();
                $display("VOCAB_END cycle=%0d job=%0d mode=%0d status=0 source_acks=1",cycles,jobs,mode);disable job_body;
            end
            @(negedge clk);expected_epoch=model_epoch;cancel_valid=0;source_result_address=model_addr[0];
            child_done_bytes=21504;child_done_status=0;tick();
            require(!result_valid && completion_revoked && completion_status==0,"Rollback revived result or changed held status");
            finish_public(0,0,1);
        end else finish_public(expected_status,expected_status==0,1);
        require(offers==oi+stop_phase+1 && child_acks==ca+stop_phase+1 && source_acks==sa+1 && publics==pu+1,"Final lifecycle counters");
        require(cancels==cn+((mode>=40 && mode<=43 || mode>=54 && mode<=58) ? 1:0),"Per-job cancellation count");
        end
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;completion_ready=0;source_completion_valid=0;
        child_done_valid=0;child_job_ready=1;child_cancel_ready=1;cancel_valid=0;
        command_sequence=0;command_epoch=0;expected_epoch=0;drive_addr=0;drive_base=0;drive_bytes=0;
        source_completion_status=0;source_sequence=0;source_epoch=0;source_result_valid=0;source_result_address=0;source_result_bytes=0;
        child_done_status=0;child_done_rows=0;child_done_bytes=0;child_done_sequence=0;child_done_epoch=0;
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;reset_all();
        if(TAG_TEST==1)begin run_job(0,0,0);run_job(90,0,0);run_job(90,0,0);end
        else if(TAG_TEST==2)begin run_job(90,0,0);run_job(90,0,0);end
        else begin
            run_job(0,0,0);
            for(integer m=1;m<=7;m++)run_job(m,0,0);
            for(integer i=0;i<3;i++)for(integer v=0;v<8;v++)run_job(10,i,v);
            for(integer i=2;i<3;i++)for(integer j=0;j<3;j++)if(i!=j)begin run_job(20,i,j);run_job(21,i,j);end
            for(integer m=30;m<=43;m++)if(m<=38 || m>=40)run_job(m,0,0);
            for(integer m=50;m<=58;m++)for(integer i=0;i<16;i++)run_job(m,i,0);
            run_job(59,0,0);
            for(integer m=60;m<=70;m++)run_job(m,0,0);
            for(integer i=0;i<16;i++)run_job(71,i,0);
            run_job(72,0,0);run_job(73,0,0);run_job(0,0,0);
            require(cancels==84 && revocations==11 && drains==6 && resets==20 && reserved_jobs>40 && held_samples>300,"Required scenario coverage");
        end
        $display("VOCAB_PASS seed=%0d tag_test=%0d jobs=%0d offers=%0d child_acks=%0d source_acks=%0d publics=%0d cancels=%0d resets=%0d drains=%0d revoked=%0d reserved=%0d held=%0d cycles=%0d",seed,TAG_TEST,jobs,offers,child_acks,source_acks,publics,cancels,resets,drains,revocations,reserved_jobs,held_samples,cycles);
        $finish;
    end
    initial begin #10000000;$fatal(1,"QKV global timeout");end
endmodule
`default_nettype wire
