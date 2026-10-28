`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_qkv_rope_handoff_r4;
    logic clk=0;
    logic rst_n=0;
    logic command_valid=0;
    logic command_ready;
    logic [63:0] command_sequence=0;
    logic [63:0] command_epoch=0;
    logic [63:0] command_lease_base=0;
    logic [63:0] expected_epoch=0;
    logic [31:0] command_layer=0;
    logic [31:0] command_position=0;
    logic [31:0] command_coefficient_position=0;
    logic [63:0] command_addr [0:9];
    logic [63:0] command_base [0:9];
    logic [63:0] command_bytes [0:9];
    logic cancel_valid=0;
    logic cancel_ready;
    logic busy;
    logic cache_exclusive;
    logic cache_idle=0;
    logic cache_valid=0;
    logic cache_metadata_fault=0;
    logic [63:0] cache_base=0;
    logic [63:0] cache_region_base=0;
    logic [63:0] cache_region_bytes=0;
    logic [63:0] cache_sequence=0;
    logic [63:0] cache_epoch=0;
    logic [31:0] cache_capacity=0;
    logic [31:0] cache_committed_tokens=0;
    logic source_busy=0;
    logic source_completion_valid=0;
    logic source_completion_ready;
    logic [63:0] source_sequence=0;
    logic [63:0] source_epoch=0;
    logic [3:0] source_status=0;
    logic [3:0] source_norm_status=0;
    logic [3:0] source_qkv_status=0;
    logic [3:0] source_matrix_status=0;
    logic [31:0] source_hls_status=0;
    logic [9:0] source_norm_read_beats=0;
    logic [6:0] source_norm_records=0;
    logic [1:0] source_phase=0;
    logic [1:0] source_matrices=0;
    logic [14:0] source_total_bytes=0;
    logic source_result_valid=0;
    logic source_revoked=0;
    logic [63:0] source_result_address [0:2];
    logic [13:0] source_result_bytes [0:2];
    logic child_command_valid;
    logic child_command_ready=0;
    logic child_busy=0;
    logic [63:0] child_command_sequence;
    logic [63:0] child_command_epoch;
    logic [63:0] child_command_lease_base;
    logic [31:0] child_command_layer;
    logic [31:0] child_command_position;
    logic [31:0] child_command_coefficient_position;
    logic [63:0] child_command_source_sequence;
    logic [63:0] child_command_source_epoch;
    logic child_command_source_valid;
    logic [31:0] child_command_source_position;
    logic [63:0] child_command_v_source_sequence;
    logic [63:0] child_command_v_source_epoch;
    logic child_command_v_source_valid;
    logic [31:0] child_command_v_source_position;
    wire [63:0] child_command_addr [0:3];
    wire [63:0] child_command_base [0:3];
    wire [63:0] child_command_bytes [0:3];
    logic child_cancel_valid;
    logic child_cancel_ready=0;
    logic child_completion_valid=0;
    logic child_completion_ready;
    logic [63:0] child_sequence=0;
    logic [63:0] child_epoch=0;
    logic [31:0] child_layer=0;
    logic [31:0] child_position=0;
    logic [3:0] child_status=0;
    logic [3:0] child_append_status=0;
    logic [3:0] child_root_status=0;
    logic [3:0] child_capture_status=0;
    logic [3:0] child_rope_status=0;
    logic [31:0] child_rope_hls_status=0;
    logic [9:0] child_rope_beats=0;
    logic [11:0] child_capture_qk_words=0;
    logic [5:0] child_capture_v_beats=0;
    logic [6:0] child_kv_frames=0;
    logic [7:0] child_q_frames=0;
    logic [12:0] child_kv_bytes=0;
    logic [31:0] child_kv_tokens=0;
    logic child_kv_committed=0;
    logic child_result_valid=0;
    logic child_revoked=0;
    logic completion_valid;
    logic completion_ready=0;
    logic [63:0] completion_sequence;
    logic [63:0] completion_epoch;
    logic [31:0] completion_layer;
    logic [31:0] completion_position;
    logic [3:0] completion_status;
    logic [3:0] completion_source_status;
    logic [3:0] completion_child_status;
    logic completion_kv_committed;
    logic [12:0] completion_kv_bytes;
    logic [31:0] completion_kv_tokens;
    logic completion_revoked;
    logic result_valid;
    logic result_ce=0;
    logic [10:0] result_address=0;
    logic child_result_ce;
    logic [10:0] child_result_address;
    logic [31:0] child_result_q=0;
    logic child_result_q_valid=0;
    logic [31:0] result_q;
    logic result_q_valid;
    always #2.5 clk=~clk;
    ntrpu_qkv_rope_handoff_r1 #(.CHILD_TIMEOUT_CYCLES(128)) dut(.*);

    integer cycle=0,seed=1,job=0,mode=0,admissions=0,child_starts=0,child_acks=0,source_acks=0,public_acks=0;
    integer child_drain=0,source_drain=0,resets=0,words=0;
    logic [63:0] seq_model,epoch_model,lease_model,cache_original;
    integer layer_model,pos_model;
    logic [63:0] address_model[0:9],base_model[0:9],bytes_model[0:9];
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"QRL_FAIL cycle=%0d job=%0d mode=%0d %s",cycle,job,mode,why);
    endtask
    function automatic logic [63:0] span(input integer index);
        case(index)0,1,6:return 8192;2:return 4096;3:return 4456448;4,5:return 1114112;9:return 256;default:return 2048;endcase
    endfunction
    function automatic logic [31:0] result_word(input integer index);
        return 32'h3f000000 ^ seq_model[31:0] ^ (32'(index)*32'h00010203);
    endfunction
    task automatic step;@(posedge clk);#2;endtask
    always @(posedge clk)begin
        cycle++;
        if(rst_n)begin
            require(cycle<1000000,"Global deadline");
            require(cache_exclusive==busy,"Exclusive cache reservation covers ownership");
            if(busy)require(!command_ready,"No readmission before both drains");
            if(!result_valid)require(!child_result_ce && !result_q_valid,"Result visibility and RAM request authority");
            child_result_q_valid<=child_result_ce;
            if(child_result_ce)child_result_q<=result_word(int'(child_result_address));
            if(child_drain>0)begin child_drain--;if(child_drain==0)child_busy<=0;end
            if(source_drain>0)begin source_drain--;if(source_drain==0)source_busy<=0;end
            if(command_valid && command_ready)begin
                admissions++;command_valid<=0;
                $display("QRL_ADMIT cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h layer=%0d position=%0d",cycle,job,mode,command_sequence,command_epoch,command_layer,command_position);
            end
            if(child_command_valid && child_command_ready)begin
                require(source_completion_valid && !source_completion_ready && source_result_valid && !source_revoked,"Producer retained at child admission");
                require(child_command_source_valid && child_command_v_source_valid,"Both real source claims retained");
                require(child_command_sequence==seq_model && child_command_epoch==epoch_model && child_command_lease_base==lease_model &&
                    child_command_layer==32'(layer_model) && child_command_position==32'(pos_model) && child_command_coefficient_position==32'(pos_model),"Captured child command identity");
                require(child_command_source_sequence==seq_model && child_command_source_epoch==epoch_model && child_command_source_position==32'(pos_model) &&
                    child_command_v_source_sequence==seq_model && child_command_v_source_epoch==epoch_model && child_command_v_source_position==32'(pos_model),"Full 64-bit source identities");
                for(integer i=0;i<4;i++)begin
                    require(child_command_addr[i]==address_model[i+6] && child_command_base[i]==base_model[i+6] && child_command_bytes[i]==bytes_model[i+6],"Source address/region captured exactly");
                    $display("QRL_REGION cycle=%0d job=%0d index=%0d address=%016h base=%016h bytes=%016h",cycle,job,i,child_command_addr[i],child_command_base[i],child_command_bytes[i]);
                end
                child_starts++;child_busy<=1;
                child_sequence<=seq_model;child_epoch<=epoch_model;child_layer<=32'(layer_model);child_position<=32'(pos_model);
                $display("QRL_CHILD cycle=%0d job=%0d sequence=%016h epoch=%016h layer=%0d position=%0d lease=%016h",cycle,job,child_command_sequence,child_command_epoch,child_command_layer,child_command_position,child_command_lease_base);
            end
            if(child_cancel_valid && child_cancel_ready)$display("QRL_CANCEL cycle=%0d job=%0d",cycle,job);
            if(completion_valid && completion_ready)begin public_acks++;$display("QRL_ACK cycle=%0d job=%0d",cycle,job);end
            if(child_completion_valid && child_completion_ready)begin
                require(child_sequence==seq_model && child_epoch==epoch_model && child_layer==32'(layer_model) && child_position==32'(pos_model),"Never acknowledge foreign child");
                require(!completion_valid && source_completion_valid && !source_completion_ready,"Public ACK precedes child ACK, source still retained");
                child_acks++;child_completion_valid<=0;child_result_valid<=0;child_drain=mode==37 ? 40:5+seed%4;
                $display("QRL_CHILD_ACK cycle=%0d job=%0d",cycle,job);
            end
            if(source_completion_valid && source_completion_ready)begin
                require(source_sequence==seq_model && source_epoch==epoch_model,"Never acknowledge foreign source");
                require(!child_busy && !child_completion_valid && !completion_valid,"Source after full consumer drain");
                source_acks++;source_completion_valid<=0;source_result_valid<=0;source_drain=mode==38 ? 40:7+seed%4;
                $display("QRL_SOURCE_ACK cycle=%0d job=%0d",cycle,job);
            end
        end
    end
    task automatic reset_all;
        rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;result_ce=0;
        source_busy=0;source_completion_valid=0;child_busy=0;child_completion_valid=0;child_result_valid=0;child_result_q_valid=0;
        child_drain=0;source_drain=0;repeat(5)step();rst_n=1;repeat(3)step();
        require(!busy && !completion_valid && !cache_exclusive,"Coordinated idle reset");resets++;
        $display("QRL_RESET cycle=%0d job=%0d id=%0d",cycle,job,resets);
    endtask
    task automatic defaults;
        seq_model=64'h1234567800000000+64'(job);epoch_model=64'hfedcba9800000000+64'(job);lease_model=64'h2000000010000000+64'(job*64);
        layer_model=job%16;pos_model=job%4;
        command_sequence=seq_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_layer=32'(layer_model);command_position=32'(pos_model);command_coefficient_position=32'(pos_model);command_lease_base=lease_model;
        for(integer i=0;i<10;i++)begin
            command_base[i]=(64'(i)+1)<<40;command_addr[i]=command_base[i]+4032;
            command_bytes[i]=((span(i)+4032+4095)/4096)*4096;
        end
        cache_idle=1;cache_valid=1;cache_metadata_fault=0;cache_capacity=8;cache_committed_tokens=32'(pos_model);
        cache_sequence=seq_model;cache_epoch=epoch_model;cache_region_base=64'he00000000000;cache_base=cache_region_base+4032;cache_region_bytes=528384;
        cache_original=cache_base;
        source_sequence=seq_model;source_epoch=epoch_model;source_busy=1;source_completion_valid=1;
        source_status=0;source_norm_status=0;source_qkv_status=0;source_matrix_status=0;source_hls_status=0;
        source_norm_read_beats=256;source_norm_records=64;source_phase=2;source_matrices=3;source_total_bytes=12288;
        source_result_valid=1;source_revoked=0;
        child_status=0;child_append_status=0;child_root_status=0;child_capture_status=0;child_rope_status=0;child_rope_hls_status=0;
        child_rope_beats=164;child_capture_qk_words=2560;child_capture_v_beats=32;child_kv_frames=64;child_q_frames=128;
        child_kv_bytes=4096;child_kv_tokens=32'(pos_model+1);child_kv_committed=1;child_revoked=0;
        child_command_ready=1;child_cancel_ready=1;child_completion_valid=0;child_result_valid=0;child_busy=0;
        cancel_valid=0;completion_ready=0;result_ce=0;
    endtask
    task automatic run_case(input integer which,want,input bit expect_child,reset_case,late);
        integer admit_before,starts,ca,sa,pa,limit,i,kind,write_slot,other;
        begin : case_body
        require(!busy && !source_busy && !child_busy,"Previous children fully idle");job++;mode=which;defaults();
        if(mode>=100 && mode<170)begin
            i=(mode-100)/7;kind=(mode-100)%7;
            case(kind)
                0:command_addr[i]=command_addr[i]+1;
                1:command_base[i]=command_base[i]+1;
                2:command_bytes[i]=command_bytes[i]-1;
                3:command_bytes[i]=0;
                4:command_bytes[i]=64;
                5:command_addr[i]=command_base[i]-64;
                6:begin command_base[i]=64'hfffffffffffff000;command_addr[i]=64'hffffffffffffffc0;command_bytes[i]=8192;end
            endcase
        end
        if(mode>=200 && mode<210)begin i=mode-200;command_base[i]=cache_region_base;command_addr[i]=cache_region_base+4032;end
        if(mode>=300 && mode<340)begin
            i=(mode-300)/10;write_slot=i==0 ? 2:i+5;other=(mode-300)%10;
            if(write_slot!=other)begin
                command_base[write_slot]=command_base[other];command_addr[write_slot]=command_base[other]+4032;
                command_bytes[write_slot]=8388608;command_bytes[other]=8388608;
            end
        end
        case(mode)
            3:begin source_status=6;source_result_valid=0;end
            4:source_matrices=2;
            5:source_hls_status=3;
            7:source_sequence=seq_model^64'h0000010000000000;
            8:source_epoch=epoch_model^64'h0000010000000000;
            9:expected_epoch=epoch_model^64'h0000010000000000;
            10:command_layer=16;
            11:cache_committed_tokens=32'(pos_model+1);
            12:command_coefficient_position=32'(pos_model+1);
            13:command_lease_base=0;
            14:command_lease_base=64'hffffffffffffffe1;
            15:cache_capacity=0;
            16:cache_capacity=131073;
            17:cache_base=cache_base+1;
            18:cache_region_bytes=4096;
            19:begin cache_region_base=64'hfffffffffffff000;cache_base=cache_region_base+4032;cache_region_bytes=8192;end
            20:begin command_base[1]=command_base[0];command_addr[1]=command_addr[0];command_bytes[1]=command_bytes[0];end
            21,26,27,45:child_command_ready=0;
            47:source_phase=1;
            50:cache_sequence=seq_model^64'h0000010000000000;
            51:cache_epoch=epoch_model^64'h0000010000000000;
            52:command_layer=32'h10000000;
            53:begin command_position=131072;cache_capacity=131072;cache_committed_tokens=131072;cache_region_bytes=64'h200001000;end
        endcase
        for(integer k=0;k<10;k++)begin
            address_model[k]=command_addr[k];base_model[k]=command_base[k];bytes_model[k]=command_bytes[k];
            $display("QRL_INPUT cycle=%0d job=%0d index=%0d address=%016h base=%016h bytes=%016h",cycle,job,k,command_addr[k],command_base[k],command_bytes[k]);
        end
        for(integer k=0;k<3;k++)begin source_result_address[k]=command_addr[k+6];source_result_bytes[k]=k==0 ? 8192:2048;end
        if(mode==6)source_result_address[0]=source_result_address[0]^64'h0000010000000000;
        if(mode==48)source_result_bytes[0]=8191;
        if(mode==49)source_result_address[2]=source_result_address[2]^64'h0000010000000000;
        admit_before=admissions;starts=child_starts;ca=child_acks;sa=source_acks;pa=public_acks;command_valid=1;
        while(admissions==admit_before)step();repeat(2)step();
        command_sequence=0;command_epoch=0;command_layer='1;command_position='1;command_lease_base=0;command_coefficient_position='1;
        for(integer k=0;k<10;k++)begin command_addr[k]=0;command_base[k]=0;command_bytes[k]=0;end
        if(mode==21 || mode==26 || mode==27 || mode==45)begin
            limit=0;while(!child_command_valid)begin step();limit++;require(limit<80,"Offer reached");end
            if(mode==21)source_revoked=1;
            if(mode==26)cancel_valid=1;
            if(mode==45)cache_committed_tokens=32'(pos_model+1);
            $display("QRL_INJECT cycle=%0d job=%0d phase=0 mode=%0d",cycle,job,mode);
            step();cancel_valid=0;
        end
        if(expect_child)begin
            limit=0;while(child_starts==starts)begin step();limit++;require(limit<80,"Child offered");end
            repeat(3)step();
            case(mode)
                22:source_result_address[0]=source_result_address[0]^64'h0000010000000000;
                23:source_sequence=source_sequence^64'h0000010000000000;
                24:expected_epoch=epoch_model^64'h0000010000000000;
                25:cancel_valid=1;
                41:source_completion_valid=0;
                42:cache_metadata_fault=1;
                43:cache_base=cache_base^64'h0000010000000000;
                44:cache_capacity=4;
            endcase
            if(mode==39)begin
                reset_all();require(child_acks==ca && source_acks==sa && public_acks==pa,"Reset never fabricates ACKs");
                $display("QRL_RELEASE cycle=%0d job=%0d reset=1",cycle,job);disable case_body;
            end
            $display("QRL_INJECT cycle=%0d job=%0d phase=1 mode=%0d",cycle,job,mode);
            step();cancel_valid=0;
            if(mode==28)begin limit=0;while(!child_cancel_valid)begin step();limit++;require(limit<160,"Timeout requests cancellation");end end
            repeat(4)step();
            if(mode==41)source_completion_valid=1;
            child_completion_valid=1;child_result_valid=1;cache_committed_tokens=32'(pos_model+1);
            if(mode==1 || (want!=0 && mode!=2 && mode!=29 && mode!=46))begin
                child_status=mode==1 ? 3:7;child_result_valid=0;child_kv_committed=0;child_kv_tokens=32'(pos_model);cache_committed_tokens=32'(pos_model);
            end
            if(mode==2)child_kv_bytes=128;
            if(mode==46)child_kv_committed=0;
            if(mode==29)child_epoch=epoch_model^64'h0000010000000000;
            $display("QRL_REPLY cycle=%0d job=%0d status=%0d committed=%0d bytes=%0d tokens=%0d",cycle,job,child_status,child_kv_committed,child_kv_bytes,child_kv_tokens);
            if(mode==29)begin
                repeat(9)begin step();require(!completion_valid && !child_completion_ready,"Foreign completion cannot be used or retired");end
                child_epoch=epoch_model;$display("QRL_RESTORE cycle=%0d job=%0d phase=1",cycle,job);
            end
        end
        limit=0;while(!completion_valid)begin step();limit++;require(limit<200,"Public completion progress");end
        require(completion_status==4'(want) && completion_sequence==seq_model && completion_epoch==epoch_model,"Exact public status/identity");
        require(child_starts-starts==int'(expect_child),"Exact child admission extent");
        require(completion_kv_committed==(expect_child && child_kv_committed) && completion_kv_bytes==(expect_child ? child_kv_bytes:0) && completion_kv_tokens==(expect_child ? child_kv_tokens:0),"Actual commit survives failure certificate");
        $display("QRL_COMPLETE cycle=%0d job=%0d status=%0d source=%0d child=%0d committed=%0d bytes=%0d tokens=%0d",cycle,job,completion_status,completion_source_status,completion_child_status,completion_kv_committed,completion_kv_bytes,completion_kv_tokens);
        if(mode==40)begin
            reset_all();require(child_acks==ca && source_acks==sa && public_acks==pa,"Held reset never fabricates ACKs");
            $display("QRL_RELEASE cycle=%0d job=%0d reset=1",cycle,job);disable case_body;
        end
        if(want==0)begin
            require(result_valid,"Successful owned result");
            for(integer k=0;k<(job==1 ? 2048:8);k++)begin
                i=(k*257+31)%2048;result_ce=1;result_address=11'(i);step();
                require(result_q_valid && result_q===result_word(i),"Synchronous owned public result");
                $display("QRL_WORD cycle=%0d job=%0d index=%0d data=%08h",cycle,job,i,result_q);words++;
            end
            result_ce=0;step();
        end else require(!result_valid && !result_q_valid,"Failure hides results");
        if(late)begin
            case(mode)
                30:child_status=3;31:source_revoked=1;32:expected_epoch=epoch_model^64'h0000010000000000;
                33:cancel_valid=1;34:child_completion_valid=0;
            endcase
            $display("QRL_INJECT cycle=%0d job=%0d phase=2 mode=%0d",cycle,job,mode);
            repeat(3)step();
            child_status=0;source_revoked=0;expected_epoch=epoch_model;cancel_valid=0;child_completion_valid=1;
            repeat(3)begin step();require(completion_revoked && !result_valid && !result_q_valid,"Transient late revocation remains sticky");end
            $display("QRL_AUTH cycle=%0d job=%0d revoked=%0d valid=%0d q_valid=%0d child_ce=%0d status=%0d",cycle,job,completion_revoked,result_valid,result_q_valid,child_result_ce,completion_status);
            $display("QRL_RESTORE cycle=%0d job=%0d phase=2",cycle,job);
        end
        repeat(7)begin step();require(completion_valid && completion_status==4'(want) && child_acks==ca && source_acks==sa,"Held public response retains both peers");end
        if(mode==35)child_sequence=seq_model^64'h0000010000000000;
        if(mode==36)source_epoch=epoch_model^64'h0000010000000000;
        completion_ready=1;step();completion_ready=0;
        if(mode==35)begin
            repeat(9)begin step();require(child_acks==ca && source_acks==sa && !child_completion_ready,"Foreign child held during drain");end
            child_sequence=seq_model;$display("QRL_RESTORE cycle=%0d job=%0d phase=3",cycle,job);
        end
        if(mode==7 || mode==8 || mode==23 || mode==36)begin
            repeat(18)begin step();require(source_acks==sa && !source_completion_ready,"Foreign source held until identity restored");end
            source_sequence=seq_model;source_epoch=epoch_model;$display("QRL_RESTORE cycle=%0d job=%0d phase=4",cycle,job);
        end
        limit=0;while(busy)begin step();limit++;require(limit<90,"Final coordinated drain");end
        require(!source_busy && !child_busy && child_acks-ca==int'(expect_child) && source_acks==sa+1 && public_acks==pa+1,"One ACK per actual owner");
        $display("QRL_RELEASE cycle=%0d job=%0d reset=0",cycle,job);
        end
    endtask
    always @(posedge clk)if(rst_n && command_valid && command_ready && mode>=100)begin
        for(integer d=0;d<10;d++)$display("QRL_INPUT_PORT_DIAG cycle=%0d job=%0d index=%0d external=%016h internal=%016h",cycle,job,d,command_addr[d],dut.command_addr[d]);
        #1;
        for(integer d=0;d<10;d++)$display("QRL_CAPTURE_DIAG cycle=%0d job=%0d index=%0d retained=%016h valid=%0d",cycle,job,d,dut.address[d],dut.valid_span(dut.address[d],dut.region_base[d],dut.region_size[d],span(d)));
    end
    initial begin
        if(!$value$plusargs("SEED=%d",seed))seed=1;reset_all();

        run_case(0,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(1,3,1,0,0);
        run_case(0,0,1,0,0);
        run_case(2,5,1,0,0);
        run_case(0,0,1,0,0);
        run_case(3,6,0,0,0);
        run_case(0,0,1,0,0);
        run_case(4,5,0,0,0);
        run_case(0,0,1,0,0);
        run_case(5,5,0,0,0);
        run_case(0,0,1,0,0);
        run_case(6,5,0,0,0);
        run_case(0,0,1,0,0);
        run_case(7,2,0,0,0);
        run_case(0,0,1,0,0);
        run_case(8,2,0,0,0);
        run_case(0,0,1,0,0);
        run_case(9,2,0,0,0);
        run_case(0,0,1,0,0);
        run_case(10,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(11,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(12,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(13,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(14,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(15,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(16,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(17,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(18,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(19,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(20,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(21,11,0,0,0);
        run_case(0,0,1,0,0);
        run_case(22,11,1,0,0);
        run_case(0,0,1,0,0);
        run_case(23,11,1,0,0);
        run_case(0,0,1,0,0);
        run_case(24,2,1,0,0);
        run_case(0,0,1,0,0);
        run_case(25,7,1,0,0);
        run_case(0,0,1,0,0);
        run_case(26,7,0,0,0);
        run_case(0,0,1,0,0);
        run_case(27,8,0,0,0);
        run_case(0,0,1,0,0);
        run_case(28,8,1,0,0);
        run_case(0,0,1,0,0);
        run_case(29,11,1,0,0);
        run_case(0,0,1,0,0);
        run_case(30,0,1,0,1);
        run_case(0,0,1,0,0);
        run_case(31,0,1,0,1);
        run_case(0,0,1,0,0);
        run_case(32,0,1,0,1);
        run_case(0,0,1,0,0);
        run_case(33,0,1,0,1);
        run_case(0,0,1,0,0);
        run_case(34,0,1,0,1);
        run_case(0,0,1,0,0);
        run_case(35,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(36,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(37,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(38,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(39,0,1,1,0);
        run_case(0,0,1,0,0);
        run_case(40,0,1,1,0);
        run_case(0,0,1,0,0);
        run_case(41,11,1,0,0);
        run_case(0,0,1,0,0);
        run_case(42,11,1,0,0);
        run_case(0,0,1,0,0);
        run_case(43,11,1,0,0);
        run_case(0,0,1,0,0);
        run_case(44,11,1,0,0);
        run_case(0,0,1,0,0);
        run_case(45,11,0,0,0);
        run_case(0,0,1,0,0);
        run_case(46,5,1,0,0);
        run_case(0,0,1,0,0);
        run_case(47,5,0,0,0);
        run_case(0,0,1,0,0);
        run_case(48,5,0,0,0);
        run_case(0,0,1,0,0);
        run_case(49,5,0,0,0);
        run_case(0,0,1,0,0);
        run_case(50,2,0,0,0);
        run_case(0,0,1,0,0);
        run_case(51,2,0,0,0);
        run_case(0,0,1,0,0);
        run_case(52,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(53,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(100,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(101,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(102,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(103,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(104,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(105,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(106,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(107,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(108,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(109,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(110,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(111,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(112,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(113,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(114,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(115,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(116,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(117,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(118,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(119,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(120,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(121,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(122,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(123,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(124,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(125,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(126,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(127,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(128,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(129,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(130,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(131,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(132,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(133,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(134,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(135,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(136,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(137,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(138,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(139,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(140,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(141,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(142,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(143,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(144,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(145,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(146,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(147,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(148,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(149,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(150,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(151,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(152,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(153,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(154,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(155,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(156,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(157,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(158,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(159,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(160,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(161,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(162,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(163,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(164,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(165,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(166,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(167,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(168,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(169,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(200,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(201,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(202,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(203,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(204,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(205,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(206,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(207,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(208,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(209,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(300,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(301,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(302,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(303,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(304,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(305,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(306,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(307,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(308,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(309,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(310,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(311,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(312,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(313,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(314,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(315,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(316,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(317,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(318,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(319,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(320,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(321,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(322,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(323,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(324,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(325,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(326,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(327,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(328,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(329,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(330,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(331,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(332,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(333,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(334,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(335,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(336,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(337,1,0,0,0);
        run_case(0,0,1,0,0);
        run_case(338,0,1,0,0);
        run_case(0,0,1,0,0);
        run_case(339,1,0,0,0);
        run_case(0,0,1,0,0);
        $display("QRL_PASS seed=%0d jobs=%0d admissions=%0d children=%0d child_acks=%0d source_acks=%0d public_acks=%0d resets=%0d words=%0d cycles=%0d",seed,job,admissions,child_starts,child_acks,source_acks,public_acks,resets,words,cycle);
        $finish;
    end
endmodule
`default_nettype wire
