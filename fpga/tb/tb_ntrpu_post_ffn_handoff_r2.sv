`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_post_ffn_handoff_r2;
    logic clk,rst_n;
    logic command_valid;
    logic command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic [63:0] command_output_address,command_output_base,command_output_bytes;
    logic cancel_valid;
    logic cancel_ready,busy;
    logic source_completion_valid;
    logic source_completion_ready;
    logic [3:0] source_completion_status;
    logic [63:0] source_sequence,source_epoch;
    logic source_result_valid;
    logic [63:0] source_residual_address,source_ffn_address;
    logic [13:0] source_residual_bytes;
    logic [12:0] source_ffn_bytes;
    logic [31:0] source_root_hls_status,source_post_hls_status;
    logic [9:0] source_post_read_beats;
    logic [7:0] source_residual_beats;
    logic [6:0] source_ffn_beats;
    logic [7:1][63:0] drive_addr,drive_base,drive_bytes,model_addr,model_base,model_bytes;
    logic [7:0][63:0] drive_protected_base,drive_protected_bytes;
    wire [63:0] command_ffn_addr [1:7],command_ffn_base [1:7],command_ffn_bytes [1:7];
    wire [63:0] command_protected_base [0:7],command_protected_bytes [0:7];
    wire [63:0] child_command_addr [0:7],child_command_base [0:7],child_command_bytes [0:7];
    for(genvar i=1;i<8;i++)begin
        assign command_ffn_addr[i]=drive_addr[i];assign command_ffn_base[i]=drive_base[i];assign command_ffn_bytes[i]=drive_bytes[i];
    end
    for(genvar i=0;i<8;i++)begin
        assign command_protected_base[i]=drive_protected_base[i];assign command_protected_bytes[i]=drive_protected_bytes[i];
    end
    logic child_command_valid;
    logic child_command_ready;
    logic [63:0] child_command_sequence,child_command_epoch;
    logic [63:0] child_source_sequence,child_source_epoch;
    logic child_source_valid;
    wire [63:0] child_residual_address,child_residual_base,child_residual_bytes;
    logic [63:0] child_output_address,child_output_base,child_output_bytes;
    logic child_cancel_valid;
    logic child_completion_valid;
    logic child_completion_ready;
    logic [3:0] child_completion_status;
    logic [31:0] child_hls_status,child_ffn_hls_status;
    logic [3:0] child_ffn_status,child_ffn_child_status,child_residual_status;
    logic [2:0] child_ffn_stages;
    logic [1:0] child_ffn_phase;
    logic [16:0] child_ffn_bytes;
    logic [17:0] child_ffn_stage_bytes;
    logic [63:0] child_sequence,child_epoch;
    logic [9:0] child_read_beats;
    logic [7:0] child_write_beats;
    logic child_result_valid;
    logic [63:0] child_result_address;
    logic [13:0] child_result_bytes;
    logic ffn_bus_owned;
    logic completion_valid;
    logic completion_ready;
    logic [3:0] completion_status,completion_source_status,completion_child_status;
    logic [31:0] completion_hls_status,completion_ffn_hls_status;
    logic [3:0] completion_ffn_status,completion_ffn_child_status,completion_residual_status;
    logic [2:0] completion_ffn_stages;
    logic [1:0] completion_ffn_phase;
    logic [16:0] completion_ffn_bytes;
    logic [17:0] completion_ffn_stage_bytes;
    logic [31:0] completion_source_root_hls_status,completion_source_post_hls_status;
    logic [63:0] completion_sequence,completion_epoch;
    logic [9:0] completion_read_beats;
    logic [7:0] completion_write_beats;
    logic result_valid;
    logic [63:0] result_address;
    logic [13:0] result_bytes;
    ntrpu_post_ffn_handoff_r2 dut(.*);
    always #2.5 clk=~clk;
    integer seed=1,random_state=1,cycle=0,jobs=0,mode_now=0,admitted=0,done_count=0,source_acks=0,child_acks=0,publics=0,offered=0,resets=0;
    integer admits_before,source_before,child_before;
    logic [63:0] seq_model,epoch_model,res_model,ffn_model,out_model,base_model,size_model;
    logic child_seen,had_child_response;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"PFHAND_FAIL cycle=%0d job=%0d mode=%0d: %s",cycle,jobs,mode_now,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    always @(posedge clk)begin
        cycle++;
        if(rst_n)begin
            if(busy)require(!command_ready,"Busy command admission");
            if(child_command_valid)begin
                require(ffn_bus_owned && busy,"Child bus ownership");
                require(child_command_sequence==seq_model && child_command_epoch==epoch_model &&
                    child_source_sequence==seq_model && child_source_epoch==epoch_model && child_source_valid,"Captured child identity");
                require(child_residual_address==res_model && child_residual_base==res_model && child_residual_bytes==8192 &&
                    child_command_addr[0]==ffn_model && child_command_base[0]==ffn_model && child_command_bytes[0]==4096 &&
                    child_output_address==out_model && child_output_base==base_model && child_output_bytes==size_model,"Stable captured descriptors");
                for(integer i=1;i<8;i++)require(child_command_addr[i]==model_addr[i] && child_command_base[i]==model_base[i] && child_command_bytes[i]==model_bytes[i],"Captured FFN allocations");
                if(child_command_ready)begin
                    admitted++;child_seen=1;
                    $display("PFHAND_OFFER cycle=%0d job=%0d sequence=%016h epoch=%016h residual=%016h ffn=%016h output=%016h base=%016h bytes=%0d",cycle,jobs,child_command_sequence,child_command_epoch,child_residual_address,child_command_addr[0],child_output_address,child_output_base,child_output_bytes);
                    for(integer i=0;i<8;i++)$display("PFHAND_PORT cycle=%0d job=%0d index=%0d addr=%016h base=%016h bytes=%0d",cycle,jobs,i,child_command_addr[i],child_command_base[i],child_command_bytes[i]);
                end
            end
            if(child_completion_valid && !had_child_response)begin
                had_child_response=1;done_count++;
                $display("PFHAND_CHILD cycle=%0d job=%0d status=%0d sequence=%016h epoch=%016h valid=%0d read=%0d write=%0d hls=%0d address=%016h bytes=%0d ffn_status=%0d ffn_child=%0d residual_status=%0d ffn_hls=%0d stages=%0d phase=%0d ffn_bytes=%0d stage_bytes=%0d",cycle,jobs,child_completion_status,child_sequence,child_epoch,child_result_valid,child_read_beats,child_write_beats,child_hls_status,child_result_address,child_result_bytes,child_ffn_status,child_ffn_child_status,child_residual_status,child_ffn_hls_status,child_ffn_stages,child_ffn_phase,child_ffn_bytes,child_ffn_stage_bytes);
            end
            if(source_completion_ready)begin
                require(completion_valid && completion_ready && source_completion_valid && source_sequence==seq_model && source_epoch==epoch_model,"Foreign/early source ACK");
                if(child_seen)require(child_completion_ready,"Source freed before child ACK");
                source_acks++;
                $display("PFHAND_SOURCE_ACK cycle=%0d job=%0d",cycle,jobs);
            end
            if(child_completion_ready)begin
                require(child_seen && completion_valid && completion_ready && child_completion_valid && child_sequence==seq_model && child_epoch==epoch_model && source_completion_ready,"Foreign/early child ACK");
                child_acks++;
                $display("PFHAND_CHILD_ACK cycle=%0d job=%0d",cycle,jobs);
            end
            if(result_valid)require(completion_valid && completion_status==0 && source_result_valid && child_result_valid && child_completion_valid &&
                child_sequence==seq_model && child_epoch==epoch_model && result_address==out_model && result_bytes==8192,"Invalid publication");
            else require(result_address==0 && result_bytes==0,"Invalid descriptor leak");
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;
        source_completion_valid=0;child_completion_valid=0;child_command_ready=1;child_seen=0;had_child_response=0;resets++;
        repeat(4)tick();@(negedge clk);rst_n=1;tick();
        require(!busy && !completion_valid && !result_valid && !source_completion_ready && !child_completion_ready && !ffn_bus_owned,"Reset ownership");
        $display("PFHAND_RESET cycle=%0d job=%0d",cycle,jobs);
    endtask
    function automatic integer wanted(input integer mode);
        if(mode>=57 && mode<=69)return 5;
        if(mode>=78 && mode<=82 || mode==260)return 11;
        if(mode>=88 && mode<=215)return 1;
        if(mode==256 || mode==257)return 7;
        if(mode==259)return 2;
        case(mode)
            1,2,3,25,53:return 2;
            4,30:return 3;
            5,26,27,28,29,31,32:return 11;
            6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21:return 1;
            22,23,24,52:return 7;
            33,34,35,36,37,38,39,55,56:return 5;
            default:return 0;
        endcase
    endfunction
    task automatic run_job(input integer mode);
        integer limit,di,pi,variant;logic [3:0] status_saved;logic [63:0] addr,span;
        jobs++;mode_now=mode;admits_before=admitted;source_before=source_acks;child_before=child_acks;
        @(negedge clk);
        require(!busy,"Previous job held");
        seq_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        res_model=64'h700000000fc0+64'(jobs*65536);ffn_model=64'h800000000fc0+64'(jobs*65536);out_model=64'h990000000fc0+64'(jobs*65536);
        if(mode==50)begin res_model=64'hfffffffffffe0000;ffn_model=64'hfffffffffffc0000;out_model=64'hfffffffffffa0000;end
        if(mode==51)out_model=res_model+8192;
        base_model=out_model-4032;size_model=12224;
        command_sequence=seq_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_output_address=out_model;command_output_base=base_model;command_output_bytes=size_model;
        source_completion_valid=1;source_completion_status=0;source_sequence=seq_model;source_epoch=epoch_model;source_result_valid=1;
        source_residual_address=res_model;source_ffn_address=ffn_model;source_residual_bytes=8192;source_ffn_bytes=4096;
        source_root_hls_status=0;source_post_hls_status=0;source_post_read_beats=384;source_residual_beats=128;source_ffn_beats=64;
        child_command_ready=1;child_completion_valid=0;child_completion_status=0;child_hls_status=0;
        child_sequence=seq_model;child_epoch=epoch_model;child_read_beats=256;child_write_beats=128;child_result_valid=1;
        child_result_address=out_model;child_result_bytes=8192;
        child_ffn_status=0;child_ffn_child_status=0;child_residual_status=0;child_ffn_hls_status=0;
        child_ffn_stages=4;child_ffn_phase=3;child_ffn_bytes=8192;child_ffn_stage_bytes=155648;
        for(integer i=1;i<8;i++)begin
            drive_addr[i]=64'h200000000fc0+64'(i)*64'h10000000+64'(jobs)*65536;
            drive_base[i]=drive_addr[i]-4032;
            drive_bytes[i]=(i<4 ? 17825792:i<6 ? 65536:i==6 ? 16384:8192)+4032;
        end
        for(integer i=0;i<8;i++)begin drive_protected_base[i]=64'h100000000000+64'(i)*64'h10000000+64'(jobs)*65536;drive_protected_bytes[i]=16384;end
        drive_protected_base[6]=res_model;drive_protected_bytes[6]=8192;
        drive_protected_base[7]=ffn_model;drive_protected_bytes[7]=4096;
        completion_ready=0;cancel_valid=0;child_seen=0;had_child_response=0;
        case(mode)
            1:source_sequence^=64'd1<<40;
            2:source_epoch^=64'd1<<44;
            3:expected_epoch^=64'd1<<48;
            4:begin source_completion_status=3;source_result_valid=0;end
            5:source_result_valid=0;
            6:source_residual_bytes=8191;
            7:source_ffn_bytes=4095;
            8:source_residual_address++;
            9:source_ffn_address++;
            10:source_residual_address=64'hffffffffffffffc0;
            11:source_ffn_address=64'hffffffffffffffc0;
            12:begin command_output_base=out_model;command_output_bytes=8191;end
            13:command_output_address++;
            14:command_output_base=out_model+64;
            15:begin command_output_base=64'hfffffffffffff000;command_output_bytes=8192;end
            16:begin command_output_address=64'hfffffffffffff000;command_output_base=0;command_output_bytes=64'hffffffffffffffff;end
            17:begin command_output_address=res_model;command_output_base=res_model;command_output_bytes=8192;end
            18:begin command_output_address=res_model+8128;command_output_base=res_model+8128;command_output_bytes=8192;end
            19:begin command_output_address=ffn_model;command_output_base=ffn_model;command_output_bytes=8192;end
            20:begin command_output_address=ffn_model+4032;command_output_base=ffn_model+4032;command_output_bytes=8192;end
            21:source_ffn_address=res_model+8128;
            22:cancel_valid=1;
            default:;
        endcase
        if(mode>=57 && mode<=61)source_change(mode-57);
        if(mode==88)drive_protected_base[6]=res_model+64;
        if(mode==89)drive_protected_base[7]=ffn_model+64;
        if(mode>=90 && mode<=105)begin
            pi=(mode-90)/2;
            if((mode-90)%2==0)drive_protected_bytes[pi]=0;
            else begin drive_protected_base[pi]=64'hfffffffffffff000;drive_protected_bytes[pi]=8192;end
        end
        if(mode>=106 && mode<=185)begin
            di=(mode-106)/16;pi=((mode-106)%16)/2;span=di<2 ? 65536:di==2 ? 16384:8192;
            addr=drive_protected_base[pi];
            if((mode-106)%2==1)addr=addr+drive_protected_bytes[pi]-64;
            set_destination(di,addr,addr,span);
        end
        if(mode>=186 && mode<=215)begin
            di=(mode-186)/6;variant=(mode-186)%6;span=di<2 ? 65536:di==2 ? 16384:8192;
            addr=di==4 ? out_model:drive_addr[di+4];
            case(variant)
                0:set_destination(di,addr+1,addr,span+64);
                1:set_destination(di,addr,addr,span-1);
                2:set_destination(di,addr,addr+64,span);
                3:set_destination(di,addr,64'hfffffffffffff000,8192);
                4:set_destination(di,64'hffffffffffffffc0,0,64'hffffffffffffffff);
                5:set_destination(di,addr,addr,0);
                default:;
            endcase
        end
        if(mode>=216 && mode<=255)begin
            di=(mode-216)/8;pi=(mode-216)%8;span=di<2 ? 65536:di==2 ? 16384:8192;
            addr=drive_protected_base[pi]+drive_protected_bytes[pi];
            set_destination(di,addr,addr-4032,span+4032);
            if(di==4)begin out_model=addr;base_model=addr-4032;size_model=span+4032;child_result_address=addr;end
        end
        model_addr=drive_addr;model_base=drive_base;model_bytes=drive_bytes;
        #1;require(command_ready,"Ready admission");
        $display("PFHAND_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h residual=%016h ffn=%016h output=%016h",cycle,jobs,mode,seq_model,epoch_model,res_model,ffn_model,out_model);
        for(integer i=1;i<8;i++)$display("PFHAND_FFN cycle=%0d job=%0d index=%0d addr=%016h base=%016h bytes=%0d",cycle,jobs,i,drive_addr[i],drive_base[i],drive_bytes[i]);
        for(integer i=0;i<8;i++)$display("PFHAND_PROTECT cycle=%0d job=%0d index=%0d base=%016h bytes=%0d",cycle,jobs,i,drive_protected_base[i],drive_protected_bytes[i]);
        $display("PFHAND_SOURCE cycle=%0d job=%0d status=%0d valid=%0d residual=%016h ffn=%016h resbytes=%0d ffnbytes=%0d root_hls=%0d post_hls=%0d reads=%0d residual_beats=%0d ffn_beats=%0d",cycle,jobs,source_completion_status,source_result_valid,source_residual_address,source_ffn_address,source_residual_bytes,source_ffn_bytes,source_root_hls_status,source_post_hls_status,source_post_read_beats,source_residual_beats,source_ffn_beats);
        command_valid=1;tick();@(negedge clk);command_valid=0;cancel_valid=0;
        command_sequence=0;command_epoch=0;command_output_address=0;command_output_base=0;command_output_bytes=0;
        drive_addr=0;drive_base=0;drive_bytes=0;drive_protected_base=0;drive_protected_bytes=0;
        if(mode>=256 && mode<=260)begin
            repeat(mode==256 ? 3:25)tick();@(negedge clk);
            require(busy && !child_command_valid && !ffn_bus_owned,"Protection scan ownership");
            if(mode==256 || mode==257)cancel_valid=1;
            if(mode==259)expected_epoch^=64'd1<<48;
            if(mode==260)source_post_read_beats=383;
            if(mode!=258)begin
                $display("PFHAND_STIM cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
                tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;source_post_read_beats=384;
            end
        end
        if(mode==3)expected_epoch=epoch_model;
        if(mode==23)begin cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if(mode>=24 && mode<=26 || mode==47)begin
            child_command_ready=0;limit=0;
            while(!child_command_valid)begin tick();limit++;require(limit<100,"Offer missing");end
            @(negedge clk);
            if(mode==24)cancel_valid=1;
            if(mode==25)expected_epoch^=64'd1<<48;
            if(mode==26)source_completion_valid=0;
            if(mode!=47)$display("PFHAND_STIM cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
            repeat(4)begin tick();require(child_command_valid && !source_completion_ready && !child_completion_ready,"Withdrawn stalled command");end
            @(negedge clk);cancel_valid=0;expected_epoch=epoch_model;source_completion_valid=1;
            if(mode!=47)begin require(child_cancel_valid,"Sticky child cancellation");child_command_ready=1;end
        end
        if(child_mode(mode) && mode!=47)begin
            limit=0;while(!child_seen)begin tick();limit++;require(limit<100,"Child admission missing");end
            repeat(3+$urandom_range(0,7))tick();@(negedge clk);
            if(mode==27)source_sequence^=64'd1<<40;
            if(mode==28)source_ffn_address+=64;
            if(mode==29)source_result_valid=0;
            if(mode>=78 && mode<=82)source_change(mode-78);
            if(mode>=27 && mode<=29 || mode>=78 && mode<=82)begin
                $display("PFHAND_STIM cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
                tick();@(negedge clk);source_sequence=seq_model;source_ffn_address=ffn_model;source_result_valid=1;
                source_restore();
                require(child_cancel_valid,"Lost source revocation");
            end
            if(mode!=48 && mode!=54)begin
                child_completion_status=child_cancel_valid ? 7:0;child_result_valid=!child_cancel_valid;
                case(mode)
                    30:begin child_completion_status=3;child_result_valid=0;end
                    31:child_sequence^=64'd1<<40;
                    32:child_epoch^=64'd1<<44;
                    33:child_result_address+=64;
                    34:child_read_beats=255;
                    35:child_write_beats=127;
                    36:child_hls_status=3;
                    37:child_result_valid=0;
                    38:child_result_bytes=8191;
                    39:begin child_completion_status=3;child_result_valid=1;end
                    52:cancel_valid=1;
                    53:expected_epoch^=64'd1<<48;
                    55:begin child_completion_status=3;child_result_valid=0;child_read_beats=257;end
                    56:begin child_completion_status=3;child_result_valid=0;child_write_beats=129;end
                    default:;
                endcase
                if(mode>=62 && mode<=69)child_change(mode-62);
                child_completion_valid=1;
                if(mode==31 || mode==32)begin
                    repeat(4)begin tick();require(!completion_valid && !child_completion_ready && !source_completion_ready,"Foreign child response acknowledged");end
                    @(negedge clk);child_sequence=seq_model;child_epoch=epoch_model;
                    $display("PFHAND_RESTORE cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
                end
                tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;
            end
        end
        if(mode==47 || mode==48 || mode==54 || mode==258)begin
            repeat(20)begin tick();require(busy && !completion_valid && source_acks==source_before && child_acks==child_before,"Premature drain/release");end
            $display("PFHAND_ABORT cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);reset_all();
        end else begin
            limit=0;while(!completion_valid)begin tick();limit++;require(limit<100,"Completion missing");end
            require(completion_status==wanted(mode) && completion_sequence==seq_model && completion_epoch==epoch_model,"Completion identity/status");
            require(result_valid==(wanted(mode)==0),"Publication status");
            require(source_acks==source_before && child_acks==child_before,"Peers released before public ACK");
            if(!child_mode(mode))require(admitted==admits_before,"Invalid request offered");
            if(result_valid)offered++;
            publics++;status_saved=completion_status;
            $display("PFHAND_COMPLETE cycle=%0d job=%0d status=%0d source=%0d child=%0d hls=%0d valid=%0d read=%0d write=%0d ffn_status=%0d ffn_child=%0d residual_status=%0d ffn_hls=%0d stages=%0d phase=%0d ffn_bytes=%0d stage_bytes=%0d root_hls=%0d post_hls=%0d",cycle,jobs,completion_status,completion_source_status,completion_child_status,completion_hls_status,result_valid,completion_read_beats,completion_write_beats,completion_ffn_status,completion_ffn_child_status,completion_residual_status,completion_ffn_hls_status,completion_ffn_stages,completion_ffn_phase,completion_ffn_bytes,completion_ffn_stage_bytes,completion_source_root_hls_status,completion_source_post_hls_status);
            repeat(5)begin tick();require(completion_valid && completion_status==status_saved && !command_ready && source_acks==source_before && child_acks==child_before,"Held ownership");end
            if(mode>=40 && mode<=46 || mode>=70 && mode<=77 || mode>=83 && mode<=87)begin
                @(negedge clk);
                case(mode)
                    40:cancel_valid=1;
                    41:expected_epoch^=64'd1<<48;
                    42:source_completion_valid=0;
                    43:begin source_sequence^=64'd1<<40;completion_ready=1;end
                    44:begin child_sequence^=64'd1<<40;completion_ready=1;end
                    45:child_result_valid=0;
                    46:child_result_address+=64;
                    default:;
                endcase
                if(mode>=70 && mode<=77)child_change(mode-70);
                if(mode>=83 && mode<=87)source_change(mode-83);
                #1;require(!result_valid,"Immediate held revocation");
                repeat(3)begin tick();require(busy && source_acks==source_before && child_acks==child_before,"Foreign ACK gate");end
                @(negedge clk);completion_ready=0;cancel_valid=0;expected_epoch=epoch_model;source_completion_valid=1;
                source_sequence=seq_model;child_sequence=seq_model;child_result_valid=1;child_result_address=out_model;
                child_restore();source_restore();
                repeat(3)begin tick();require(!result_valid && completion_valid && completion_status==status_saved,"Sticky held revocation");end
                $display("PFHAND_REVOKE cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
            end
            if(mode==49)begin $display("PFHAND_ABORT cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);reset_all();end
            else begin
                @(negedge clk);completion_ready=1;tick();
                require(!busy,"Public release");
                require(source_acks-source_before==((mode==1 || mode==2) ? 0:1),"Source ACK extent");
                require(child_acks-child_before==(child_seen ? 1:0),"Child ACK extent");
                $display("PFHAND_ACK cycle=%0d job=%0d source=%0d child=%0d",cycle,jobs,source_acks-source_before,child_acks-child_before);
                @(negedge clk);completion_ready=0;source_completion_valid=0;child_completion_valid=0;
                tick();require(!completion_valid && !result_valid && !ffn_bus_owned,"Ownership released");
            end
        end
    endtask
    function automatic bit child_mode(input integer mode);
        return mode==0 || mode>=24 && mode<=56 && mode!=47 || mode==47 || mode>=62 && mode<=87 || mode>=216 && mode<=255;
    endfunction
    task automatic set_destination(input integer d,input logic [63:0] addr,base,bytes);
        if(d==4)begin command_output_address=addr;command_output_base=base;command_output_bytes=bytes;end
        else begin drive_addr[d+4]=addr;drive_base[d+4]=base;drive_bytes[d+4]=bytes;end
    endtask
    task automatic source_change(input integer field);
        case(field)
            0:source_root_hls_status=1;
            1:source_post_hls_status=1;
            2:source_post_read_beats=383;
            3:source_residual_beats=127;
            4:source_ffn_beats=63;
            default:$fatal(1,"Bad source field");
        endcase
    endtask
    task automatic source_restore;
        source_root_hls_status=0;source_post_hls_status=0;source_post_read_beats=384;source_residual_beats=128;source_ffn_beats=64;
    endtask
    task automatic child_change(input integer field);
        case(field)
            0:child_ffn_status=1;
            1:child_ffn_child_status=1;
            2:child_residual_status=1;
            3:child_ffn_hls_status=1;
            4:child_ffn_stages=3;
            5:child_ffn_phase=2;
            6:child_ffn_bytes=8191;
            7:child_ffn_stage_bytes=155647;
            default:$fatal(1,"Bad child field");
        endcase
    endtask
    task automatic child_restore;
        child_ffn_status=0;child_ffn_child_status=0;child_residual_status=0;child_ffn_hls_status=0;
        child_ffn_stages=4;child_ffn_phase=3;child_ffn_bytes=8192;child_ffn_stage_bytes=155648;
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;completion_ready=0;source_completion_valid=0;child_completion_valid=0;
        child_command_ready=1;cancel_valid=0;child_seen=0;had_child_response=0;
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        reset_all();repeat(2)run_job(0);
        for(integer mode=1;mode<=260;mode++)begin run_job(mode);run_job(0);end
        require(jobs==522 && publics==518 && offered==325 && resets==6,"Plan cardinality");
        $display("PFHAND_PASS seed=%0d jobs=%0d publics=%0d offered=%0d resets=%0d admitted=%0d responses=%0d source_acks=%0d child_acks=%0d cycles=%0d",seed,jobs,publics,offered,resets,admitted,done_count,source_acks,child_acks,cycle);
        $finish;
    end
    initial begin #1000000;$fatal(1,"PFHAND_FAIL absolute timeout");end
endmodule
`default_nettype wire
