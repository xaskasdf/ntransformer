`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_input_attention_r2;
    logic clk=0;
    logic rst_n=0;
    logic command_valid=0;
    wire command_ready;
    logic [63:0] command_sequence=0;
    logic [63:0] command_epoch=0;
    logic [63:0] command_lease_base=0;
    logic [63:0] expected_epoch=0;
    logic [31:0] command_layer=0;
    logic [31:0] command_position=0;
    logic [31:0] command_coefficient_position=0;
    wire [63:0] command_addr [0:9];
    wire [63:0] command_base [0:9];
    wire [63:0] command_bytes [0:9];
    logic cancel_valid=0;
    wire cancel_ready;
    wire busy;
    wire cache_exclusive;
    wire completion_valid;
    logic completion_ready=0;
    wire [63:0] completion_sequence;
    wire [63:0] completion_epoch;
    wire [31:0] completion_layer;
    wire [31:0] completion_position;
    wire [3:0] completion_status;
    wire [3:0] completion_source_status;
    wire [3:0] completion_child_status;
    wire completion_kv_committed;
    wire [12:0] completion_kv_bytes;
    wire [31:0] completion_kv_tokens;
    wire completion_revoked;
    wire result_valid;
    logic result_ce=0;
    logic [10:0] result_address=0;
    wire [31:0] result_q;
    wire result_q_valid;
    logic [63:0] command_source_sequence=0;
    logic [63:0] command_source_epoch=0;
    logic command_source_valid=0;
    wire [3:0] completion_norm_status;
    wire [3:0] completion_qkv_status;
    wire [3:0] completion_matrix_status;
    wire [31:0] completion_norm_hls_status;
    wire [9:0] completion_norm_read_beats;
    wire [6:0] completion_norm_records;
    wire [1:0] completion_phase;
    wire [1:0] completion_matrices;
    wire [14:0] completion_total_bytes;
    wire norm_tag_exhausted;
    wire qkv_tags_exhausted;
    wire [63:0] input_axi_araddr;
    wire [7:0] input_axi_arlen;
    wire [2:0] input_axi_arsize;
    wire [2:0] input_axi_arprot;
    wire [1:0] input_axi_arburst;
    wire [3:0] input_axi_arid;
    wire [3:0] input_axi_arcache;
    wire input_axi_arlock;
    wire input_axi_arvalid;
    logic input_axi_arready=0;
    logic [511:0] input_axi_rdata=0;
    logic [1:0] input_axi_rresp=0;
    logic [3:0] input_axi_rid=0;
    logic input_axi_rlast=0;
    logic input_axi_rvalid=0;
    wire input_axi_rready;
    wire [63:0] input_axi_awaddr;
    wire [7:0] input_axi_awlen;
    wire [2:0] input_axi_awsize;
    wire [2:0] input_axi_awprot;
    wire [1:0] input_axi_awburst;
    wire [3:0] input_axi_awid;
    wire [3:0] input_axi_awcache;
    wire input_axi_awlock;
    wire input_axi_awvalid;
    logic input_axi_awready=0;
    wire [511:0] input_axi_wdata;
    wire [63:0] input_axi_wstrb;
    wire input_axi_wlast;
    wire input_axi_wvalid;
    logic input_axi_wready=0;
    logic [3:0] input_axi_bid=0;
    logic [1:0] input_axi_bresp=0;
    logic input_axi_bvalid=0;
    wire input_axi_bready;
    wire [63:0] v_axi_araddr;
    wire [7:0] v_axi_arlen;
    wire [2:0] v_axi_arsize;
    wire [2:0] v_axi_arprot;
    wire [1:0] v_axi_arburst;
    wire [3:0] v_axi_arid;
    wire [3:0] v_axi_arcache;
    wire v_axi_arlock;
    wire v_axi_arvalid;
    logic v_axi_arready=0;
    logic [511:0] v_axi_rdata=0;
    logic [1:0] v_axi_rresp=0;
    logic [3:0] v_axi_rid=0;
    logic v_axi_rlast=0;
    logic v_axi_rvalid=0;
    wire v_axi_rready;
    wire [63:0] rope_axi_araddr;
    wire [7:0] rope_axi_arlen;
    wire [2:0] rope_axi_arsize;
    wire [2:0] rope_axi_arprot;
    wire [1:0] rope_axi_arburst;
    wire [3:0] rope_axi_arid;
    wire [3:0] rope_axi_arcache;
    wire rope_axi_arlock;
    wire rope_axi_arvalid;
    logic rope_axi_arready=0;
    logic [511:0] rope_axi_rdata=0;
    logic [1:0] rope_axi_rresp=0;
    logic [3:0] rope_axi_rid=0;
    logic rope_axi_rlast=0;
    logic rope_axi_rvalid=0;
    wire rope_axi_rready;
    logic bind_valid=0;
    wire bind_ready;
    logic [63:0] bind_cache_base=0;
    logic [63:0] bind_region_base=0;
    logic [63:0] bind_region_bytes=0;
    logic [31:0] bind_capacity=0;
    logic [63:0] bind_sequence=0;
    logic [63:0] bind_epoch=0;
    wire cache_valid;
    wire [63:0] cache_base;
    wire [63:0] cache_region_base;
    wire [63:0] cache_region_bytes;
    wire [63:0] cache_sequence;
    wire [63:0] cache_epoch;
    wire [31:0] cache_capacity;
    wire [63:0] kv_axi_awaddr;
    wire [7:0] kv_axi_awlen;
    wire [2:0] kv_axi_awsize;
    wire [2:0] kv_axi_awprot;
    wire [1:0] kv_axi_awburst;
    wire [3:0] kv_axi_awid;
    wire [3:0] kv_axi_awcache;
    wire kv_axi_awlock;
    wire kv_axi_awvalid;
    logic kv_axi_awready=0;
    wire [511:0] kv_axi_wdata;
    wire [63:0] kv_axi_wstrb;
    wire kv_axi_wlast;
    wire kv_axi_wvalid;
    logic kv_axi_wready=0;
    logic [3:0] kv_axi_bid=0;
    logic [1:0] kv_axi_bresp=0;
    logic kv_axi_bvalid=0;
    wire kv_axi_bready;
    wire [63:0] kv_axi_araddr;
    wire [7:0] kv_axi_arlen;
    wire [2:0] kv_axi_arsize;
    wire [2:0] kv_axi_arprot;
    wire [1:0] kv_axi_arburst;
    wire [3:0] kv_axi_arid;
    wire [3:0] kv_axi_arcache;
    wire kv_axi_arlock;
    wire kv_axi_arvalid;
    logic kv_axi_arready=0;
    logic [511:0] kv_axi_rdata=0;
    logic [1:0] kv_axi_rresp=0;
    logic [3:0] kv_axi_rid=0;
    logic kv_axi_rlast=0;
    logic kv_axi_rvalid=0;
    wire kv_axi_rready;
    wire root_cache_lock;
    wire cache_metadata_fault;
    wire [31:0] cache_query_layer;
    wire cache_query_valid;
    wire [31:0] cache_query_committed_tokens;
    wire [31:0] cache_committed_tokens;
    wire bind_completion_valid;
    logic bind_completion_ready=0;
    wire [3:0] bind_completion_status;
    wire [63:0] bind_completion_sequence;
    wire [63:0] bind_completion_epoch;
    logic [9:0][63:0] drive_addr,drive_base,drive_bytes;
    for(genvar i=0;i<10;i++)begin
        assign command_addr[i]=drive_addr[i];assign command_base[i]=drive_base[i];assign command_bytes[i]=drive_bytes[i];
    end
    always #2.5 clk=~clk;
    ntrpu_input_attention_r1 #(.HANDOFF_TIMEOUT_CYCLES(96),.OWNER_TIMEOUT_CYCLES(192)) dut(.*);

    integer seed=1,cycle=0,job=0,mode=0,expected_status=0,expected_producer=0,expected_consumer=0,late=0;
    integer sources=0,consumers=0,source_acks=0,consumer_acks=0,public_acks=0,words=0;
    integer s0=0,c0=0,sa0=0,ca0=0,pa0=0,admit_cycle=0;
    logic admitted=0,public_seen=0,source_ack_seen=0,consumer_ack_seen=0;
    logic [9:0][63:0] saved_addr,saved_base,saved_bytes;
    logic [63:0] seq_model,epoch_model,lease_model;
    integer layer_model=0,pos_model=0;
    task automatic require(input bit good,input string reason);
        if(!good)$fatal(1,"IA_FAIL cycle=%0d job=%0d mode=%0d %s",cycle,job,mode,reason);
    endtask
    task automatic step;
        @(posedge clk);#1;@(negedge clk);
    endtask
    function automatic logic [63:0] span(input integer i);
        case(i)
            0,1,6:return 8192;2:return 4096;3:return 4456448;4,5:return 1114112;9:return 256;default:return 2048;
        endcase
    endfunction
    always @(posedge clk)begin
        cycle++;
        require(cycle<200000,"Global timeout");
        if(rst_n && admitted)begin
            if(busy)require(cache_exclusive && !command_ready && !bind_ready,"Global reservation through drain");
            if(dut.producer_command_valid && dut.producer_command_ready)begin
                require(expected_producer!=0 && sources==s0,"Unexpected or duplicate producer admission");
                require(cycle>=admit_cycle+51,"All allocation checks precede producer admission");
                require(dut.producer_command_sequence==seq_model && dut.producer_command_epoch==epoch_model &&
                    dut.producer_command_source_sequence==seq_model && dut.producer_command_source_epoch==epoch_model && dut.producer_command_source_valid,"Retained producer identity");
                for(integer i=0;i<9;i++)begin
                    require(dut.producer_command_addr[i]==saved_addr[i] && dut.producer_command_base[i]==saved_base[i] && dut.producer_command_bytes[i]==saved_bytes[i],"Producer allocation capture");
                    $display("IA_REGION cycle=%0d job=%0d peer=0 index=%0d address=%016h base=%016h bytes=%016h",cycle,job,i,dut.producer_command_addr[i],dut.producer_command_base[i],dut.producer_command_bytes[i]);
                end
                require(cache_query_layer==32'(layer_model) && cache_query_committed_tokens==32'(pos_model),"Query uses owned layer before producer writes");
                sources++;$display("IA_SOURCE cycle=%0d job=%0d sequence=%016h epoch=%016h",cycle,job,dut.producer_command_sequence,dut.producer_command_epoch);
            end
            if(dut.consumer_command_valid && dut.consumer_command_ready)begin
                require(expected_consumer!=0 && consumers==c0 && sources==s0+1,"Unexpected or duplicate consumer admission");
                require(dut.producer_completion_valid && !dut.producer_completion_ready,"Retain source publication until public retirement");
                require(dut.consumer_command_sequence==seq_model && dut.consumer_command_epoch==epoch_model &&
                    dut.consumer_command_layer==32'(layer_model) && dut.consumer_command_position==32'(pos_model) && dut.consumer_command_lease_base==lease_model,"Retained consumer identity");
                for(integer i=0;i<4;i++)begin
                    require(dut.consumer_command_addr[i]==saved_addr[i==3 ? 9:i+6] && dut.consumer_command_base[i]==saved_base[i==3 ? 9:i+6] && dut.consumer_command_bytes[i]==saved_bytes[i==3 ? 9:i+6],"Consumer physical allocations");
                    $display("IA_REGION cycle=%0d job=%0d peer=1 index=%0d address=%016h base=%016h bytes=%016h",cycle,job,i,dut.consumer_command_addr[i],dut.consumer_command_base[i],dut.consumer_command_bytes[i]);
                end
                consumers++;$display("IA_CONSUMER cycle=%0d job=%0d sequence=%016h epoch=%016h layer=%0d position=%0d lease=%016h",cycle,job,dut.consumer_command_sequence,dut.consumer_command_epoch,dut.consumer_command_layer,dut.consumer_command_position,dut.consumer_command_lease_base);
            end
            if(completion_valid && completion_ready)begin
                require(!public_seen,"Only one public ACK");public_seen=1;public_acks++;
                $display("IA_ACK cycle=%0d job=%0d",cycle,job);
            end
            if(dut.consumer_completion_valid && dut.consumer_completion_ready)begin
                require(public_seen && !consumer_ack_seen && dut.producer_completion_valid && !dut.producer_completion_ready,"Consumer retires after public ACK with source held");
                require(dut.consumer_completion_sequence==seq_model && dut.consumer_completion_epoch==epoch_model,"Never ACK foreign consumer");
                consumer_ack_seen=1;consumer_acks++;$display("IA_CONSUMER_ACK cycle=%0d job=%0d",cycle,job);
            end
            if(dut.producer_completion_valid && dut.producer_completion_ready)begin
                require(public_seen && !source_ack_seen && !dut.consumer_busy && !dut.consumer_completion_valid &&
                    (expected_consumer==0 || consumer_ack_seen),"Source retires after consumer drains");
                require(dut.producer_completion_sequence==seq_model && dut.producer_completion_epoch==epoch_model,"Never ACK foreign producer");
                source_ack_seen=1;source_acks++;$display("IA_SOURCE_ACK cycle=%0d job=%0d",cycle,job);
            end
        end
    end
    task automatic run_case(input integer which,wanted,producer_expected,consumer_expected,late_fault);
        integer limit,i,k,slot,other;
        job++;mode=which;expected_status=wanted;expected_producer=producer_expected;expected_consumer=consumer_expected;late=late_fault;
        require(!busy && !completion_valid,"Previous job fully retired");
        admitted=0;public_seen=0;source_ack_seen=0;consumer_ack_seen=0;
        s0=sources;c0=consumers;sa0=source_acks;ca0=consumer_acks;pa0=public_acks;
        seq_model=64'h1234567800000000+64'(job);epoch_model=64'hfedcba9800000000+64'(job);lease_model=64'h2000000010000000+64'(job*64);
        layer_model=job%16;pos_model=layer_model%3;expected_epoch=epoch_model;
        dut.producer.test_delay=5+seed%4;dut.producer.test_drain=7;dut.producer.test_status=0;dut.producer.test_beats=256;
        dut.producer.test_sequence_mask=0;dut.producer.test_revoked=0;
        dut.consumer.test_delay=5+seed%3;dut.consumer.test_drain=7;dut.consumer.test_status=0;dut.consumer.test_bytes=4096;
        dut.consumer.test_sequence_mask=0;dut.consumer.test_mutate_cache=0;
        bind_valid=1;bind_cache_base=64'he00000000fc0;bind_region_base=64'he00000000000;bind_region_bytes=528384;
        bind_capacity=8;bind_sequence=seq_model;bind_epoch=epoch_model;bind_completion_ready=0;
        limit=0;while(!bind_ready)begin step();limit++;require(limit<20,"Bind ready progress");end
        step();bind_valid=0;limit=0;
        while(!bind_completion_valid)begin step();limit++;require(limit<20,"Bind response progress");end
        require(bind_completion_status==0,"Synthetic metadata bind");bind_completion_ready=1;step();bind_completion_ready=0;step();
        command_sequence=seq_model;command_epoch=epoch_model;command_source_sequence=seq_model;command_source_epoch=epoch_model;command_source_valid=1;
        command_layer=32'(layer_model);command_position=32'(pos_model);command_coefficient_position=32'(pos_model);command_lease_base=lease_model;
        for(i=0;i<10;i++)begin
            drive_base[i]=(64'(i)+1)<<40;drive_addr[i]=drive_base[i]+4032;drive_bytes[i]=((span(i)+4032+4095)/4096)*4096;
        end
        if(mode>=100 && mode<170)begin
            i=(mode-100)/7;k=(mode-100)%7;
            case(k)
                0:drive_addr[i]=drive_addr[i]+1;1:drive_base[i]=drive_base[i]+1;2:drive_bytes[i]=drive_bytes[i]-1;3:drive_bytes[i]=0;
                4:drive_bytes[i]=64;5:drive_addr[i]=drive_base[i]-64;
                6:begin drive_base[i]=64'hfffffffffffff000;drive_addr[i]=64'hffffffffffffffc0;drive_bytes[i]=8192;end
            endcase
        end
        if(mode>=200 && mode<210)begin i=mode-200;drive_base[i]=64'he00000000000;drive_addr[i]=64'he00000000fc0;end
        if(mode>=300 && mode<340)begin
            k=(mode-300)/10;other=(mode-300)%10;slot=k==0 ? 2:k+5;
            if(slot!=other)begin drive_base[slot]=drive_base[other];drive_addr[slot]=drive_base[other]+4032;drive_bytes[slot]=8388608;drive_bytes[other]=8388608;end
        end
        case(mode)
            1:dut.producer.test_status=3;
            2:dut.producer.test_beats=255;
            3:dut.consumer.test_status=3;
            4:dut.consumer.test_bytes=4032;
            8:dut.producer.test_delay=999;
            9:dut.consumer.test_delay=999;
            13:dut.producer.test_sequence_mask=64'h100000000;
            14:dut.consumer.test_sequence_mask=64'h100000000;
            15:begin dut.producer.test_delay=17;dut.consumer.test_delay=19;end
            16:begin dut.producer.test_drain=40;dut.consumer.test_drain=40;end
            17:begin drive_base[1]=drive_base[0];drive_addr[1]=drive_addr[0];drive_bytes[1]=drive_bytes[0];end
            18:command_epoch=epoch_model^64'h100000000;
            19:command_source_sequence=seq_model^64'h100000000;
            20:command_layer=32'h10000000;
            21:command_coefficient_position=32'(pos_model+1);
            22:command_lease_base=64'hffffffffffffffe1;
            23:command_position=32'(pos_model+1);
            24:command_source_valid=0;
            28:command_source_epoch=epoch_model^64'h100000000;
        endcase
        if(mode>=100 && mode<170)begin
            i=(mode-100)/7;k=(mode-100)%7;
            if(k==0)require(drive_addr[i][5:0]==1,"Requested address misalignment is present");
            if(k==1)require(drive_base[i][5:0]==1,"Requested base misalignment is present");
            if(k==2)require(drive_bytes[i][5:0]==63,"Requested size misalignment is present");
        end
        saved_addr=drive_addr;saved_base=drive_base;saved_bytes=drive_bytes;
        for(i=0;i<10;i++)$display("IA_INPUT cycle=%0d job=%0d index=%0d address=%016h base=%016h bytes=%016h",cycle,job,i,drive_addr[i],drive_base[i],drive_bytes[i]);
        command_valid=1;#1;require(command_ready,"Host admission ready");step();command_valid=0;admitted=1;admit_cycle=cycle;
        $display("IA_ADMIT cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h layer=%0d position=%0d",cycle,job,mode,seq_model,epoch_model,command_layer,command_position);
        for(i=0;i<10;i++)$display("IA_BOUNDARY cycle=%0d job=%0d index=%0d address=%016h base=%016h bytes=%016h",cycle,job,i,dut.address[i],dut.region_base[i],dut.region_size[i]);
        command_sequence=0;command_epoch=0;command_source_valid=0;command_source_sequence=0;command_source_epoch=0;
        command_layer=32'hffffffff;command_position=32'hffffffff;command_coefficient_position=32'hffffffff;command_lease_base=0;
        drive_addr=0;drive_base=0;drive_bytes=0;
        if(mode==5)begin step();cancel_valid=1;step();cancel_valid=0;end
        if(mode==6 || mode==26)begin
            limit=0;while(sources==s0)begin step();limit++;require(limit<100,"Source offer progress");end
            if(mode==6)cancel_valid=1;else dut.consumer.test_mutate_cache=1;
            step();cancel_valid=0;dut.consumer.test_mutate_cache=0;
        end
        if(mode==7 || mode==27)begin
            limit=0;while(consumers==c0)begin step();limit++;require(limit<180,"Consumer offer progress");end
            if(mode==7)cancel_valid=1;else dut.consumer.test_mutate_cache=1;
            step();cancel_valid=0;dut.consumer.test_mutate_cache=0;
        end
        if(mode==13 || mode==14)begin
            limit=0;
            while(!(mode==13 ? dut.producer_completion_valid:dut.consumer_completion_valid))begin
                step();limit++;require(limit<180,"Foreign response progress");
            end
            repeat(5)step();require(!completion_valid && !public_seen && source_acks==sa0 && consumer_acks==ca0,"Foreign identity remains owned without ACK");
            dut.producer.test_sequence_mask=0;dut.consumer.test_sequence_mask=0;
            $display("IA_RESTORE cycle=%0d job=%0d peer=%0d",cycle,job,mode==13 ? 0:1);
        end
        limit=0;while(!completion_valid)begin step();limit++;require(limit<450,"Public response progress");end
        require(completion_status==4'(expected_status),"Expected public status");
        require(sources==s0+expected_producer && consumers==c0+expected_consumer,"Expected child command counts");
        require(source_acks==sa0 && consumer_acks==ca0,"No child ACK before public ACK");
        $display("IA_COMPLETE cycle=%0d job=%0d status=%0d source=%0d child=%0d committed=%0d bytes=%0d tokens=%0d",cycle,job,completion_status,completion_source_status,completion_child_status,completion_kv_committed,completion_kv_bytes,completion_kv_tokens);
        if(expected_status==0)begin
            require(result_valid && completion_kv_committed && completion_kv_bytes==4096 && completion_kv_tokens==32'(pos_model+1),"Successful end-to-end protocol certificate");
            for(i=0;i<(job==1 ? 2048:8);i++)begin
                result_address=11'((i*257+31)%2048);result_ce=1;step();
                require(result_q_valid && result_q==(32'hbadd0000^32'(result_address)^(32'(layer_model)<<16)^32'(pos_model)),"Public result address and gating");
                words++;$display("IA_WORD cycle=%0d job=%0d index=%0d data=%08h",cycle,job,result_address,result_q);
            end
            result_ce=0;step();
        end else begin result_ce=1;step();require(!result_valid && !result_q_valid && !dut.consumer_result_ce,"Failed result cannot reach consumer RAM");result_ce=0;end
        if(late!=0)begin
            if(mode==10)expected_epoch=epoch_model+1;
            if(mode==11)dut.consumer.test_mutate_cache=1;
            if(mode==12)cancel_valid=1;
            if(mode==25)dut.producer.test_revoked=1;
            step();expected_epoch=epoch_model;dut.consumer.test_mutate_cache=0;cancel_valid=0;dut.producer.test_revoked=0;
            repeat(3)step();result_ce=1;step();
            require(completion_valid && completion_status==0 && completion_revoked && !result_valid && !result_q_valid && !dut.consumer_result_ce,"Transient late fault cannot resurrect public result");
            $display("IA_REVOKE cycle=%0d job=%0d status=%0d revoked=%0d valid=%0d ce=%0d",cycle,job,completion_status,completion_revoked,result_valid,dut.consumer_result_ce);result_ce=0;
        end
        bind_valid=1;repeat(3)begin step();require(!bind_ready && !command_ready && completion_valid,"Held public response excludes new work");end
        completion_ready=1;step();completion_ready=0;
        limit=0;while(busy)begin require(!bind_ready && !command_ready,"Drain retains reservation");step();limit++;require(limit<180,"Retirement progress");end
        bind_valid=0;#1;
        require(!completion_valid && !cache_exclusive && command_ready && public_acks==pa0+1 &&
            source_acks==sa0+expected_producer && consumer_acks==ca0+expected_consumer,"Full producer and consumer retirement");
        $display("IA_RELEASE cycle=%0d job=%0d sources=%0d consumers=%0d",cycle,job,sources-s0,consumers-c0);admitted=0;
    endtask
    initial begin
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        repeat(5)step();rst_n=1;repeat(3)step();

        run_case(0,0,1,1,0);
        run_case(0,0,1,1,0);
        run_case(1,3,1,0,0);
        run_case(0,0,1,1,0);
        run_case(2,5,1,0,0);
        run_case(0,0,1,1,0);
        run_case(3,3,1,1,0);
        run_case(0,0,1,1,0);
        run_case(4,5,1,1,0);
        run_case(0,0,1,1,0);
        run_case(5,7,0,0,0);
        run_case(0,0,1,1,0);
        run_case(6,7,1,0,0);
        run_case(0,0,1,1,0);
        run_case(7,7,1,1,0);
        run_case(0,0,1,1,0);
        run_case(8,8,1,0,0);
        run_case(0,0,1,1,0);
        run_case(9,8,1,1,0);
        run_case(0,0,1,1,0);
        run_case(10,0,1,1,1);
        run_case(0,0,1,1,0);
        run_case(11,0,1,1,1);
        run_case(0,0,1,1,0);
        run_case(12,0,1,1,1);
        run_case(0,0,1,1,0);
        run_case(13,11,1,0,0);
        run_case(0,0,1,1,0);
        run_case(14,11,1,1,0);
        run_case(0,0,1,1,0);
        run_case(15,0,1,1,0);
        run_case(0,0,1,1,0);
        run_case(16,0,1,1,0);
        run_case(0,0,1,1,0);
        run_case(17,0,1,1,0);
        run_case(0,0,1,1,0);
        run_case(18,2,0,0,0);
        run_case(0,0,1,1,0);
        run_case(19,2,0,0,0);
        run_case(0,0,1,1,0);
        run_case(20,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(21,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(22,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(23,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(24,2,0,0,0);
        run_case(0,0,1,1,0);
        run_case(25,0,1,1,1);
        run_case(0,0,1,1,0);
        run_case(26,11,1,0,0);
        run_case(0,0,1,1,0);
        run_case(27,11,1,1,0);
        run_case(0,0,1,1,0);
        run_case(28,2,0,0,0);
        run_case(0,0,1,1,0);
        run_case(100,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(101,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(102,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(103,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(104,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(105,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(106,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(107,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(108,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(109,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(110,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(111,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(112,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(113,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(114,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(115,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(116,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(117,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(118,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(119,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(120,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(121,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(122,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(123,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(124,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(125,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(126,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(127,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(128,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(129,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(130,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(131,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(132,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(133,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(134,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(135,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(136,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(137,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(138,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(139,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(140,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(141,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(142,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(143,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(144,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(145,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(146,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(147,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(148,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(149,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(150,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(151,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(152,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(153,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(154,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(155,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(156,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(157,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(158,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(159,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(160,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(161,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(162,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(163,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(164,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(165,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(166,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(167,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(168,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(169,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(200,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(201,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(202,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(203,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(204,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(205,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(206,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(207,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(208,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(209,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(300,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(301,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(302,0,1,1,0);
        run_case(0,0,1,1,0);
        run_case(303,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(304,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(305,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(306,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(307,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(308,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(309,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(310,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(311,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(312,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(313,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(314,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(315,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(316,0,1,1,0);
        run_case(0,0,1,1,0);
        run_case(317,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(318,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(319,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(320,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(321,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(322,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(323,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(324,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(325,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(326,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(327,0,1,1,0);
        run_case(0,0,1,1,0);
        run_case(328,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(329,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(330,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(331,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(332,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(333,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(334,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(335,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(336,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(337,1,0,0,0);
        run_case(0,0,1,1,0);
        run_case(338,0,1,1,0);
        run_case(0,0,1,1,0);
        run_case(339,1,0,0,0);
        run_case(0,0,1,1,0);
        $display("IA_PASS seed=%0d jobs=%0d sources=%0d consumers=%0d source_acks=%0d consumer_acks=%0d public_acks=%0d words=%0d cycles=%0d",seed,job,sources,consumers,source_acks,consumer_acks,public_acks,words,cycle);$finish;
    end
endmodule
`default_nettype wire
