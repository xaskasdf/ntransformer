`timescale 1ns/1ps
`default_nettype none
// Protocol peers only; this bank does not execute norm or matrix arithmetic.
module tb_ntrpu_input_qkv_handoff_r3;
    logic clk,rst_n;
    logic command_valid;
    wire command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic [63:0] command_source_sequence,command_source_epoch;
    logic command_source_valid;
    logic [8:0][63:0] addr_drive,base_drive,bytes_drive;
    wire [63:0] command_addr [0:8],command_base [0:8],command_bytes [0:8];
    for(genvar i=0;i<9;i++)begin
        assign command_addr[i]=addr_drive[i];assign command_base[i]=base_drive[i];assign command_bytes[i]=bytes_drive[i];
    end
    logic cancel_valid;
    wire cancel_ready,busy;
    wire norm_command_valid;
    logic norm_command_ready,norm_busy;
    wire [63:0] norm_command_sequence,norm_command_epoch;
    wire [63:0] norm_command_addr [0:2],norm_command_base [0:2],norm_command_bytes [0:2];
    wire norm_cancel_valid;
    logic norm_completion_valid;
    wire norm_completion_ready;
    logic [3:0] norm_completion_status;
    logic [31:0] norm_hls_status;
    logic [63:0] norm_sequence,norm_epoch,norm_result_address;
    logic [12:0] norm_result_bytes;
    logic [9:0] norm_read_beats;
    logic [6:0] norm_records;
    logic norm_result_valid;
    wire qkv_command_valid;
    logic qkv_command_ready,qkv_busy;
    wire [63:0] qkv_command_sequence,qkv_command_epoch;
    wire [63:0] qkv_command_addr [0:6],qkv_command_base [0:6],qkv_command_bytes [0:6];
    wire qkv_cancel_valid;
    logic qkv_source_ready;
    logic qkv_completion_valid;
    wire qkv_completion_ready;
    logic [3:0] qkv_completion_status,qkv_source_status,qkv_child_status;
    logic [63:0] qkv_sequence,qkv_epoch;
    logic [1:0] qkv_phase,qkv_matrices;
    logic [14:0] qkv_total_bytes;
    logic qkv_result_valid,qkv_revoked;
    wire [63:0] qkv_result_address [0:2];
    wire [13:0] qkv_result_bytes [0:2];
    wire qkv_bus_owned;
    wire completion_valid;
    logic completion_ready;
    wire [63:0] completion_sequence,completion_epoch;
    wire [3:0] completion_status,completion_norm_status,completion_qkv_status,completion_matrix_status;
    wire [31:0] completion_norm_hls_status;
    wire [9:0] completion_norm_read_beats;
    wire [6:0] completion_norm_records;
    wire [1:0] completion_phase,completion_matrices;
    wire [14:0] completion_total_bytes;
    wire completion_revoked,result_valid;
    wire [63:0] result_address [0:2];
    wire [13:0] result_bytes [0:2];

    localparam integer WD=64;
    ntrpu_input_qkv_handoff_r2 #(.CHILD_TIMEOUT_CYCLES(WD)) dut(.*);
    always #2.5 clk=~clk;
    integer seed=1,random_state=1,cycles=0,jobs=0,mode_now=0;
    integer ns=0,qs=0,ntimer=0,qtimer=0,noffer=0,qoffer=0;
    integer norm_starts=0,qkv_starts=0,nacks=0,qacks=0,publics=0,accepted=0,resets=0;
    integer status_model=0,job_nstarts=0,job_qstarts=0,job_nacks=0,job_qacks=0;
    logic [8:0][63:0] saved_addr,saved_base,saved_bytes;
    logic [63:0] sequence_model,epoch_model;
    logic nrev=0,qrev=0,nforeign=0,qforeign=0,ndrift=0,qdrift=0;
    logic nstalled=0,qstalled=0;
    logic [3:0] nstatus=0,qstatus=0;
    logic [31:0] nhls=0;
    logic [9:0] nread=0;
    logic [6:0] nrecords=0;
    logic [1:0] qphase=0,qmatrices=0;
    logic [14:0] qbytes=0;
    logic [3:0] qmatrix_status=0,qsource_status=0;
    logic [63:0] nseq,nepoch,qseq,qepoch;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"IQH_FAIL cycle=%0d job=%0d mode=%0d: %s",cycles,jobs,mode_now,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [63:0] span(input integer i);
        case(i)0,1,6:return 8192;2:return 4096;3:return 4456448;4,5:return 1114112;default:return 2048;endcase
    endfunction
    function automatic integer wanted(input integer m);
        if(m>=100 || m==36)return 1;
        case(m)
            1,2,3,9,21:return 2;
            4,17,18,24,31:return 11;
            5,6,7,8,19,20:return 7;
            10,22:return 3;
            11,12,13,14,15,16,23,32,33,37,38:return 5;
            27,28:return 8;
            default:return 0;
        endcase
    endfunction
    assign norm_busy=ns!=0;
    assign norm_completion_valid=ns==2;
    assign norm_completion_status=nstatus;
    assign norm_hls_status=nhls;
    assign norm_read_beats=nread;
    assign norm_records=nrecords;
    assign norm_sequence=nseq^(nforeign ? 64'h10000000000:64'd0);
    assign norm_epoch=nepoch;
    assign norm_result_valid=ns==2 && nstatus==0 && !nrev && mode_now!=12;
    assign norm_result_address=norm_result_valid ? saved_addr[2]+((ndrift || mode_now==14) ? 64'd64:64'd0):64'd0;
    assign norm_result_bytes=norm_result_valid ? (mode_now==13 ? 13'd4080:13'd4096):13'd0;
    assign qkv_busy=qs!=0;
    assign qkv_completion_valid=qs==2;
    assign qkv_completion_status=qstatus;
    assign qkv_source_status=qsource_status;
    assign qkv_child_status=qmatrix_status;
    assign qkv_sequence=qseq^(qforeign ? 64'h10000000000:64'd0);
    assign qkv_epoch=qepoch;
    assign qkv_phase=qphase;
    assign qkv_matrices=qmatrices;
    assign qkv_total_bytes=qbytes;
    assign qkv_revoked=qrev;
    assign qkv_result_valid=qs==2 && qstatus==0 && !qrev;
    for(genvar i=0;i<3;i++)begin
        assign qkv_result_address[i]=qkv_result_valid ? saved_addr[i+6]+((qdrift || mode_now==32) && i==1 ? 64'd64:64'd0):64'd0;
        assign qkv_result_bytes[i]=qkv_result_valid ? (i==0 ? 14'd8192:mode_now==33 ? 14'd1024:14'd2048):14'd0;
    end
    // Model QKV's retained source: public ACK is consumed even if its source
    // is foreign; the child then drains until the matching source returns.
    assign qkv_source_ready=(mode_now==31 && qs==1 && qtimer>=2) ||
        ((qs==2 && qkv_completion_ready || qs==3) && norm_completion_valid &&
         norm_sequence==qseq && norm_epoch==qepoch);
    always @(negedge clk)begin
        norm_command_ready=ns==0 && (!norm_command_valid ||
            ((mode_now!=7 || noffer>=8) && $urandom_range(0,3)!=0));
        qkv_command_ready=qs==0 && norm_completion_valid &&
            ((mode_now!=19 || qoffer>=8) && $urandom_range(0,3)!=0);
    end
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin
            ns=0;qs=0;ntimer=0;qtimer=0;noffer=0;qoffer=0;nrev=0;qrev=0;
            nstalled=0;qstalled=0;nstatus=0;qstatus=0;nhls=0;nread=0;nrecords=0;
            qphase=0;qmatrices=0;qbytes=0;qmatrix_status=0;qsource_status=0;
        end else begin
            if(busy)require(completion_sequence==sequence_model && completion_epoch==epoch_model,"Full captured identity");
            if(nstalled)require(norm_command_valid,"Norm offer withdrawn");
            if(qstalled)require(qkv_command_valid,"QKV offer withdrawn");
            nstalled=norm_command_valid && !norm_command_ready;qstalled=qkv_command_valid && !qkv_command_ready;
            if(norm_command_valid)begin
                noffer++;
                require(norm_command_sequence==sequence_model && norm_command_epoch==epoch_model,"Norm identity");
                for(integer i=0;i<3;i++)require(norm_command_addr[i]==saved_addr[i] && norm_command_base[i]==saved_base[i] && norm_command_bytes[i]==saved_bytes[i],"Norm captured descriptors");
            end
            if(qkv_command_valid)begin
                qoffer++;
                require(norm_completion_valid && ns==2,"QKV before norm terminal");
                require(qkv_command_sequence==sequence_model && qkv_command_epoch==epoch_model,"QKV identity");
                for(integer i=0;i<7;i++)require(qkv_command_addr[i]==saved_addr[i+2] && qkv_command_base[i]==saved_base[i+2] && qkv_command_bytes[i]==saved_bytes[i+2],"QKV captured descriptors");
            end
            if(norm_completion_valid && norm_completion_ready)require(norm_sequence==sequence_model && norm_epoch==epoch_model,"Foreign norm ACK");
            if(qkv_completion_ready && qkv_completion_valid)require(qkv_sequence==sequence_model && qkv_epoch==epoch_model,"Foreign QKV ACK");
            if(qkv_source_ready && !(completion_valid && completion_ready) && dut.state!=dut.DRAIN)
                require(!norm_completion_ready,"Premature QKV source release accepted");
            if(result_valid)begin
                require(completion_valid && completion_status==0 && !completion_revoked && norm_result_valid && qkv_result_valid,"Unowned result");
                for(integer i=0;i<3;i++)require(result_address[i]==saved_addr[i+6] && result_bytes[i]==(i==0 ? 8192:2048),"Public descriptors");
            end else for(integer i=0;i<3;i++)require(result_address[i]==0 && result_bytes[i]==0,"Invalid descriptor leak");
            if(completion_valid && completion_ready)begin
                publics++;if(result_valid)accepted++;
                $display("IQH_PUBLIC cycle=%0d job=%0d status=%0d valid=%0d revoked=%0d",cycles,jobs,completion_status,result_valid,completion_revoked);
            end
            // Sample releases before changing the peer states.
            if(norm_completion_valid && norm_completion_ready)begin
                nacks++;job_nacks++;
                $display("IQH_NACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(qkv_completion_valid && qkv_completion_ready)begin
                qacks++;job_qacks++;
                $display("IQH_QACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(ns!=0 && (norm_cancel_valid || expected_epoch!=nepoch))nrev=1;
            if(qs!=0 && (qkv_cancel_valid || expected_epoch!=qepoch))qrev=1;
            case(qs)
                0:if(qkv_command_valid && qkv_command_ready)begin
                    require(ns==2 && norm_completion_valid,"Missing live norm lease");
                    qs<=1;qtimer=0;qseq=qkv_command_sequence;qepoch=qkv_command_epoch;qrev=qkv_cancel_valid;
                    qkv_starts++;job_qstarts++;
                    $display("IQH_QSTART cycle=%0d job=%0d address=%016h",cycles,jobs,qkv_command_addr[0]);
                end
                1:begin
                    qtimer++;
                    if(qtimer>=(mode_now==28 ? WD+8:12))begin
                        qstatus=qrev ? 7:mode_now==22 ? 3:0;
                        qsource_status=mode_now==37 ? 3:0;qmatrix_status=mode_now==38 || mode_now==22 ? 3:0;
                        qphase=2;qmatrices=mode_now==23 ? 2:3;qbytes=12288;qs<=2;
                        $display("IQH_QDONE cycle=%0d job=%0d status=%0d",cycles,jobs,qstatus);
                    end
                end
                2:if(qkv_completion_ready)begin
                    qs<=norm_completion_ready ? 0:3;
                end
                3:if(norm_completion_ready)qs<=0;
                default:;
            endcase
            case(ns)
                0:if(norm_command_valid && norm_command_ready)begin
                    ns<=1;ntimer=0;nseq=norm_command_sequence;nepoch=norm_command_epoch;nrev=norm_cancel_valid;
                    norm_starts++;job_nstarts++;
                    $display("IQH_NSTART cycle=%0d job=%0d address=%016h",cycles,jobs,norm_command_addr[2]);
                end
                1:begin
                    ntimer++;
                    if(ntimer>=(mode_now==27 ? WD+8:12))begin
                        nstatus=nrev ? 7:mode_now==10 ? 3:mode_now==11 ? 5:0;
                        nhls=mode_now==11 ? 4:0;nread=mode_now==15 ? 255:256;nrecords=mode_now==16 ? 63:64;ns<=2;
                        $display("IQH_NDONE cycle=%0d job=%0d status=%0d",cycles,jobs,nstatus);
                    end
                end
                2:if(norm_completion_ready)ns<=0;
                default:;
            endcase
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;command_valid=0;cancel_valid=0;completion_ready=0;resets++;
        repeat(4)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(!busy && command_ready && ns==0 && qs==0,"Coordinated reset");
        $display("IQH_RESET cycle=%0d job=%0d",cycles,jobs);
    endtask
    task automatic run_job(input integer mode);
        integer limit,region,kind,wi,oi;logic [3:0] held_status;
        @(negedge clk);require(command_ready && !busy,"Previous job active");
        jobs++;mode_now=mode;job_nstarts=0;job_qstarts=0;job_nacks=0;job_qacks=0;
        noffer=0;qoffer=0;nforeign=mode==17;qforeign=mode==24;ndrift=0;qdrift=0;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        command_sequence=sequence_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_source_sequence=sequence_model;command_source_epoch=epoch_model;command_source_valid=1;
        for(integer i=0;i<9;i++)begin
            addr_drive[i]=64'h100000000+64'(i)*64'h10000000+64'(jobs)*64'h1000000000;
            base_drive[i]=addr_drive[i];bytes_drive[i]=span(i);
        end
        if(mode>=100 && mode<200)begin
            region=(mode-100)/4;kind=(mode-100)%4;
            case(kind)
                0:bytes_drive[region]=span(region)-64;
                1:addr_drive[region]+=1;
                2:begin base_drive[region]=64'hfffffffffffff000;addr_drive[region]=base_drive[region];bytes_drive[region]=8192;end
                3:base_drive[region]=addr_drive[region]+64;
            endcase
        end else if(mode>=200)begin
            wi=(mode-200)/9;oi=(mode-200)%9;wi=wi==0 ? 2:wi+5;
            base_drive[wi]=base_drive[oi];addr_drive[wi]=base_drive[wi];
            bytes_drive[wi]=span(wi)>span(oi) ? span(wi):span(oi);
        end
        case(mode)
            1:command_source_sequence^=64'd1<<40;
            2:command_source_epoch^=64'd1<<44;
            3:expected_epoch^=64'd1<<48;
            4:command_source_valid=0;
            5:cancel_valid=1;
            36:begin bytes_drive[0]=16384;base_drive[2]=base_drive[0];addr_drive[2]=base_drive[0]+8192;bytes_drive[2]=12288;end
            default:;
        endcase
        saved_addr=addr_drive;saved_base=base_drive;saved_bytes=bytes_drive;
        $display("IQH_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,sequence_model,epoch_model);
        command_valid=1;#1;require(command_ready,"Admission ready");tick();@(negedge clk);command_valid=0;cancel_valid=0;
        command_sequence=0;command_epoch=0;command_source_sequence=0;command_source_epoch=0;command_source_valid=0;
        addr_drive=0;base_drive=0;bytes_drive=0;
        if(mode==6)begin repeat(2)tick();@(negedge clk);end
        if(mode==7)begin wait(norm_command_valid && !norm_command_ready);@(negedge clk);end
        if(mode==8 || mode==9 || mode==29)begin wait(ns==1 && ntimer>=2);@(negedge clk);end
        if(mode==18 || mode==20 || mode==21 || mode==30)begin wait(qs==1 && qtimer>=2);@(negedge clk);end
        if(mode==19)begin wait(qkv_command_valid && !qkv_command_ready);@(negedge clk);end
        if(mode==6 || mode==7 || mode==8 || mode==19 || mode==20)begin cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if(mode==9 || mode==21)begin expected_epoch^=64'd1<<48;tick();@(negedge clk);expected_epoch=epoch_model;end
        if(mode==18)ndrift=1;
        if(mode==29 || mode==30)begin
            $display("IQH_ABORT cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);reset_all();
        end else begin
            limit=0;while(!completion_valid)begin tick();limit++;require(limit<WD*8,"Completion timeout");end
            require(completion_status==wanted(mode),"Completion status");
            require(result_valid==(wanted(mode)==0),"Result publication");
            if(mode>=100 || mode<=6 && mode!=0 || mode==36)require(job_nstarts==0 && job_qstarts==0,"Preflight started child");
            if(mode>=10 && mode<=17 || mode==27 || mode==7 || mode==8 || mode==9)require(job_qstarts==0,"Failed norm launched QKV");
            require(job_nacks==0 && job_qacks==0,"Early child release");
            $display("IQH_COMPLETE cycle=%0d job=%0d status=%0d valid=%0d norm_status=%0d qkv_status=%0d matrices=%0d bytes=%0d",cycles,jobs,completion_status,result_valid,completion_norm_status,completion_qkv_status,completion_matrices,completion_total_bytes);
            held_status=completion_status;
            repeat(5)begin tick();require(completion_valid && completion_status==held_status && !command_ready,"Held completion");end
            if(mode==25 || mode==26 || mode==34 || mode==35 || mode==39 || mode==40)begin
                @(negedge clk);
                case(mode)25:cancel_valid=1;26:expected_epoch^=64'd1<<48;34:qdrift=1;35:ndrift=1;39:qforeign=1;40:nforeign=1;endcase
                #1;require(!result_valid && completion_revoked,"Immediate late revocation");tick();@(negedge clk);
                cancel_valid=0;expected_epoch=epoch_model;qdrift=0;ndrift=0;
                repeat(3)begin tick();require(!result_valid && completion_revoked && completion_status==held_status,"Sticky revocation");end
                $display("IQH_REVOKE cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            end
            @(negedge clk);completion_ready=1;tick();require(!completion_valid,"Public ACK not consumed");
            @(negedge clk);completion_ready=0;
            if(mode==17 || mode==24 || mode==39 || mode==40)begin
                repeat(3)begin tick();require(busy && !command_ready && !completion_valid && job_nacks==0,"Foreign drain retention");end
                @(negedge clk);nforeign=0;qforeign=0;
                $display("IQH_RESTORE cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);
            end
            limit=0;while(busy)begin tick();limit++;require(limit<16,"Drain timeout");require(!completion_valid,"Duplicate response");end
            require(job_nacks==job_nstarts && job_qacks==job_qstarts && command_ready,"Child release cardinality");
            $display("IQH_RETIRED cycle=%0d job=%0d norm=%0d qkv=%0d",cycles,jobs,job_nacks,job_qacks);
        end
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;cancel_valid=0;completion_ready=0;norm_command_ready=0;qkv_command_ready=0;
        command_sequence=0;command_epoch=0;expected_epoch=0;command_source_sequence=0;command_source_epoch=0;command_source_valid=0;
        addr_drive=0;base_drive=0;bytes_drive=0;sequence_model=0;epoch_model=0;nseq=0;nepoch=0;qseq=0;qepoch=0;
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);reset_all();repeat(4)run_job(0);
        for(integer m=1;m<=40;m++)begin run_job(m);run_job(0);end
        for(integer m=100;m<136;m++)begin run_job(m);run_job(0);end
        for(integer wi=0;wi<4;wi++)for(integer oi=0;oi<9;oi++)if(oi!=(wi==0 ? 2:wi+5))begin run_job(200+wi*9+oi);run_job(0);end
        $display("IQH_PASS seed=%0d jobs=%0d publics=%0d accepted=%0d norm_starts=%0d qkv_starts=%0d norm_acks=%0d qkv_acks=%0d resets=%0d cycles=%0d",seed,jobs,publics,accepted,norm_starts,qkv_starts,nacks,qacks,resets,cycles);
        $finish;
    end
    initial begin #10000000;$fatal(1,"IQH_FAIL absolute timeout");end
endmodule
`default_nettype wire
