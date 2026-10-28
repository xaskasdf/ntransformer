`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_projected_root_native_r1;
    logic clk;
    logic rst_n;
    logic bind_valid;
    logic bind_ready;
    logic [63:0] bind_cache_base;
    logic [63:0] bind_region_base;
    logic [63:0] bind_region_bytes;
    logic [31:0] bind_capacity;
    logic [63:0] bind_sequence;
    logic [63:0] bind_epoch;
    logic [63:0] expected_epoch;
    logic cache_valid;
    logic [63:0] cache_base;
    logic [63:0] cache_region_base;
    logic [63:0] cache_region_bytes;
    logic [63:0] cache_sequence;
    logic [63:0] cache_epoch;
    logic [31:0] cache_capacity;
    logic append_valid;
    logic append_ready;
    logic append_prepared;
    logic [31:0] append_layer;
    logic [31:0] append_position;
    logic [63:0] append_sequence;
    logic [63:0] append_epoch;
    logic data_valid;
    logic data_ready;
    logic [511:0] data_bits;
    logic [5:0] data_index;
    logic data_last;
    logic [63:0] data_sequence;
    logic [63:0] data_epoch;
    logic commit_valid;
    logic commit_ready;
    logic [3:0] commit_status;
    logic [63:0] commit_sequence;
    logic [63:0] commit_epoch;
    logic append_cancel_valid;
    logic append_cancel_ready;
    logic append_done_valid;
    logic append_done_ready;
    logic append_done_bind;
    logic [3:0] append_done_status;
    logic [12:0] append_done_bytes;
    logic [31:0] append_done_committed_tokens;
    logic [63:0] append_done_sequence;
    logic [63:0] append_done_epoch;
    logic lease_active;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize;
    logic [2:0] m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid;
    logic [3:0] m_axi_awcache;
    logic m_axi_awlock;
    logic m_axi_awvalid;
    logic m_axi_awready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast;
    logic m_axi_wvalid;
    logic m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic m_axi_bvalid;
    logic m_axi_bready;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize;
    logic [2:0] m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid;
    logic [3:0] m_axi_arcache;
    logic m_axi_arlock;
    logic m_axi_arvalid;
    logic m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast;
    logic m_axi_rvalid;
    logic m_axi_rready;
    logic command_valid;
    logic command_ready;
    logic [31:0] command_position;
    logic [31:0] command_layer;
    logic [63:0] command_sequence;
    logic [63:0] command_epoch;
    logic cancel_valid;
    logic cancel_ready;
    logic completion_valid;
    logic completion_ready;
    logic [3:0] completion_status;
    logic result_valid;
    logic [63:0] command_lease_base;
    logic q_valid;
    logic q_ready;
    logic [511:0] q_data;
    logic [7:0] q_index;
    logic q_last;
    logic [63:0] q_sequence;
    logic [63:0] q_epoch;
    logic [63:0] completion_sequence;
    logic [63:0] completion_epoch;
    logic cache_lock;
    logic maintenance_busy;
    logic metadata_fault;
    logic [31:0] committed_tokens;
    logic busy;
    logic tag_exhausted;
    logic [31:0] completion_hls_status;
    logic [63:0] result_address;
    logic [16:0] result_bytes;
    logic [63:0] command_weight_addr;
    logic [63:0] command_weight_region_base;
    logic [63:0] command_weight_region_bytes;
    logic [63:0] command_activation_addr;
    logic [63:0] command_activation_region_base;
    logic [63:0] command_activation_region_bytes;
    logic [63:0] command_output_addr;
    logic [63:0] command_output_region_base;
    logic [63:0] command_output_region_bytes;
    ntrpu_attention_projected_root_r1 dut(.*);
    always #2.5 clk=~clk;
    integer seed=1,rng=1,cycles=0,jobs=0,mode_now=0,publics=0,resets=0,root_starts=0,output_starts=0,root_acks=0,output_acks=0;
    integer binds=0,appends=0,aws=0,ws=0,bs=0,ars=0,rs=0,source_reads=0;
    integer job_root_acks=0,job_output_acks=0,job_output_starts=0;
    logic pause_aw=0,pause_w=0,pause_r=0,aw_pending=0,w_pending=0,r_pending=0;
    logic [63:0] write_address,read_address,seq_model,epoch_model;
    logic [511:0] write_data,kv_data,activation_data,output_data;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"PROOT_FAIL cycle=%0d job=%0d mode=%0d: %s",cycles,jobs,mode_now,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    task automatic reset_all;
        @(negedge clk);rst_n=0;command_valid=0;bind_valid=0;append_valid=0;append_done_ready=0;completion_ready=0;cancel_valid=0;q_valid=0;
        dut.root.foreign=0;dut.output_path.foreign=0;dut.output_path.hold_admission=0;dut.root.hold_maintenance=0;
        pause_aw=0;pause_w=0;pause_r=0;
        repeat(3)tick();@(negedge clk);rst_n=1;resets++;repeat(2)tick();
    endtask
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;m_axi_arready=0;m_axi_rvalid=0;end
        else begin
            m_axi_awready=!pause_aw && $urandom_range(0,3)!=0;m_axi_wready=!pause_w && $urandom_range(0,3)!=0;
            m_axi_arready=$urandom_range(0,3)!=0;
            if(aw_pending && w_pending && !m_axi_bvalid && $urandom_range(0,3)!=0)begin m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;end
            if(r_pending && !pause_r && !m_axi_rvalid && $urandom_range(0,3)!=0)begin
                m_axi_rdata=read_address==64'h4000000fc0 ? kv_data:activation_data;
                m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=1;m_axi_rvalid=1;
            end
        end
    end
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin aw_pending=0;w_pending=0;r_pending=0;kv_data='x;activation_data='x;output_data='x;end
        else begin
            if(bind_valid && bind_ready)begin binds++;$display("PROOT_BIND cycle=%0d job=%0d",cycles,jobs);end
            if(append_valid && append_ready)begin appends++;$display("PROOT_APPEND cycle=%0d job=%0d",cycles,jobs);end
            if(busy)require(cache_lock && !bind_ready && !append_ready && !dut.root.bind_valid && !dut.root.append_valid,"Maintenance crossed composed owner");
            if(dut.root.command_valid && dut.root.command_ready)begin
                root_starts++;require(!aw_pending && !w_pending && !r_pending && !dut.output_owned,"Root admission before KV drain");
                $display("PROOT_ROOT cycle=%0d job=%0d sequence=%016h epoch=%016h",cycles,jobs,command_sequence,command_epoch);
            end
            if(dut.output_path.command_valid)begin
                require(dut.root.completion_valid && !dut.root.completion_ready && !aw_pending && !w_pending && !r_pending,"Output offered before retained root/drain");
                require(dut.output_path.command_sequence==seq_model && dut.output_path.command_epoch==epoch_model,"Full output identity");
            end
            if(dut.output_path.command_valid && dut.output_path.command_ready)begin
                output_starts++;job_output_starts++;$display("PROOT_OUTPUT cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.root.result_ce)begin
                source_reads++;require(dut.output_owned && job_root_acks==0 && dut.root.result_valid && dut.root.result_address==seq_model[10:0],"Root memory ownership");
                $display("PROOT_READ cycle=%0d job=%0d index=%0d",cycles,jobs,dut.root.result_address);
            end
            if(dut.root.completion_valid && dut.root.completion_ready)begin
                require(dut.root.completion_sequence==seq_model && dut.root.completion_epoch==epoch_model,"Foreign root ACK");
                require(dut.output_owned || (completion_valid && completion_ready),"Early direct root ACK");
                root_acks++;job_root_acks++;$display("PROOT_ROOT_ACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.output_path.completion_valid && dut.output_path.completion_ready)begin
                require(completion_valid && completion_ready && job_root_acks==1 && !aw_pending && !w_pending && !r_pending,"Early output ACK");
                require(dut.output_path.completion_sequence==seq_model && dut.output_path.completion_epoch==epoch_model,"Foreign output ACK");
                output_acks++;job_output_acks++;$display("PROOT_OUTPUT_ACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awlen==0 && m_axi_awsize==6,"AW routing");write_address=m_axi_awaddr;aw_pending=1;aws++;
                require(dut.output_owned==(m_axi_awaddr!=64'h4000000fc0),"AW owner switch");
                $display("PROOT_AW cycle=%0d job=%0d address=%016h",cycles,jobs,m_axi_awaddr);
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(!w_pending && m_axi_wstrb=='1 && m_axi_wlast,"W routing");write_data=m_axi_wdata;w_pending=1;ws++;
                $display("PROOT_W cycle=%0d job=%0d data=%0128h",cycles,jobs,m_axi_wdata);
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending,"B without AW/W");
                if(write_address==64'h4000000fc0)begin require(!dut.output_owned && write_data=={8{64'h1122334455667788^seq_model}},"KV W content/owner");kv_data=write_data;end
                else if(write_address==64'h8000000fc0)begin require(dut.output_owned && job_root_acks==1 && write_data=={16{32'h3f000000|32'(seq_model[10:0])}},"Quantized W content/owner");activation_data=write_data;end
                else begin require(write_address==64'ha000000fc0 && dut.output_owned && write_data==~activation_data,"Projected W content/owner");output_data=write_data;end
                $display("PROOT_B cycle=%0d job=%0d address=%016h",cycles,jobs,write_address);
                bs++;aw_pending=0;w_pending=0;m_axi_bvalid<=0;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!r_pending && m_axi_arlen==0 && m_axi_arsize==6,"AR routing");read_address=m_axi_araddr;r_pending=1;ars++;
                require(m_axi_araddr==(dut.output_owned ? 64'h8000000fc0:64'h4000000fc0),"AR owner switch");
                $display("PROOT_AR cycle=%0d job=%0d address=%016h",cycles,jobs,m_axi_araddr);
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(r_pending,"R without AR");$display("PROOT_R cycle=%0d job=%0d address=%016h data=%0128h",cycles,jobs,read_address,m_axi_rdata);
                r_pending=0;rs++;m_axi_rvalid<=0;
            end
            if(!result_valid)require(result_address==0 && result_bytes==0,"Private result address leak");
        end
    end
    task automatic maintenance;
        @(negedge clk);bind_valid=1;command_valid=1;dut.root.hold_maintenance=1;
        bind_sequence=seq_model;bind_epoch=epoch_model;bind_cache_base=64'h4000000000;bind_region_base=64'h4000000000;bind_region_bytes=8192;bind_capacity=1;
        repeat(3)begin tick();require(!command_ready && !busy,"Pending maintenance priority");end
        @(negedge clk);dut.root.hold_maintenance=0;#1;require(bind_ready && !command_ready,"Bind priority");
        tick();@(negedge clk);bind_valid=0;command_valid=0;append_done_ready=1;
        tick();@(negedge clk);append_done_ready=0;
        append_sequence=seq_model;append_epoch=epoch_model;append_valid=1;#1;require(append_ready,"Append admission");
        tick();@(negedge clk);append_valid=0;
        wait(append_done_valid);@(negedge clk);append_done_ready=1;tick();@(negedge clk);append_done_ready=0;
        require(!maintenance_busy && !aw_pending && !w_pending,"Maintenance drained");
    endtask
    task automatic run_job(input integer mode);begin: work
        integer expected_status,limit;logic expected_valid;
        @(negedge clk);jobs++;mode_now=mode;job_root_acks=0;job_output_acks=0;job_output_starts=0;
        seq_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);expected_epoch=epoch_model;
        dut.root.test_mode=mode==1 ? 1:mode==2 ? 2:0;
        dut.output_path.test_mode=mode==10 ? 1:mode==11 ? 2:mode==12 ? 3:0;
        dut.root.foreign=mode==8;dut.output_path.foreign=mode==9;dut.output_path.hold_admission=mode==13;
        $display("PROOT_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,seq_model,epoch_model);
        maintenance();
        command_valid=1;command_sequence=seq_model;command_epoch=epoch_model;command_position=0;command_layer=0;command_lease_base=64'h1234000000000000;
        command_weight_addr=64'h100000fc0;command_weight_region_base=64'h100000000;command_weight_region_bytes=4460480;
        command_activation_addr=64'h8000000fc0;command_activation_region_base=64'h8000000000;command_activation_region_bytes=8128;
        command_output_addr=64'ha000000fc0;command_output_region_base=64'ha000000000;command_output_region_bytes=12224;
        #1;require(command_ready,"Composed admission");tick();@(negedge clk);command_valid=0;
        command_sequence=0;command_epoch=0;command_lease_base=0;command_weight_addr=0;command_weight_region_base=0;command_weight_region_bytes=0;
        command_activation_addr=0;command_activation_region_base=0;command_activation_region_bytes=0;command_output_addr=0;command_output_region_base=0;command_output_region_bytes=0;
        bind_valid=1;append_valid=1;
        if(mode==3 || mode==4 || mode==14)pause_r=1;
        if(mode==5 || mode==15)pause_w=1;
        q_valid=1;q_index=0;q_last=1;q_sequence=seq_model;q_epoch=epoch_model;q_data={16{32'hca000000|32'(seq_model[15:0])}};
        #1;require(q_ready,"Root Q admission");tick();@(negedge clk);q_valid=0;
        expected_status=mode==1 || mode==10 ? 3:mode==2 || mode==11 || mode==12 ? 5:mode==3 || mode==5 || mode==13 ? 7:mode==4 ? 2:mode==8 || mode==9 ? 11:0;
        expected_valid=mode==0;
        if(mode==3 || mode==4 || mode==14)begin
            wait(r_pending);@(negedge clk);
            if(mode==14)begin $display("PROOT_RESET cycle=%0d job=%0d phase=1",cycles,jobs);reset_all();disable work;end
            if(mode==3)cancel_valid=1;else expected_epoch=epoch_model^64'h10000000000;
            tick();@(negedge clk);cancel_valid=0;pause_r=0;
        end
        if(mode==5 || mode==15)begin
            wait(dut.output_owned && aw_pending);@(negedge clk);
            if(mode==15)begin $display("PROOT_RESET cycle=%0d job=%0d phase=2",cycles,jobs);reset_all();disable work;end
            cancel_valid=1;tick();@(negedge clk);cancel_valid=0;pause_w=0;
        end
        if(mode==8)begin
            wait(dut.root.completion_valid);repeat(4)begin tick();require(!dut.root.completion_ready && !dut.output_path.command_valid && !completion_valid,"Foreign root held");end
            @(negedge clk);dut.root.foreign=0;
        end
        if(mode==9)begin
            wait(dut.output_path.completion_valid);repeat(4)begin tick();require(!dut.output_path.completion_ready && !completion_valid,"Foreign output held");end
            @(negedge clk);dut.output_path.foreign=0;
        end
        if(mode==13)begin
            wait(dut.output_path.command_valid);@(negedge clk);cancel_valid=1;
            repeat(4)begin tick();require(dut.output_path.command_valid && dut.output_path.command_sequence==seq_model && !dut.root.completion_ready,"Revoked offered command held");end
            @(negedge clk);cancel_valid=0;dut.output_path.hold_admission=0;
        end
        limit=0;while(!completion_valid)begin tick();limit++;require(limit<500,"Completion deadline");end
        if(mode==6 || mode==7)begin
            @(negedge clk);if(mode==6)expected_epoch=epoch_model^64'h10000000000;else cancel_valid=1;
            tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;tick();
        end
        require(completion_status==expected_status && completion_sequence==seq_model && completion_epoch==epoch_model,"Public identity/status");
        require(result_valid==expected_valid && (!result_valid || (result_address==64'ha000000fc0 && result_bytes==8192 && output_data==~activation_data)),"Public result/revocation");
        require(completion_hls_status==(mode==10 ? 42:0),"HLS status propagation");
        repeat(5)begin tick();require(completion_valid && completion_status==expected_status && result_valid==expected_valid && !command_ready && !bind_ready && !append_ready,"Held public owner");end
        $display("PROOT_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d valid=%0d sequence=%016h epoch=%016h root_acks=%0d outputs=%0d",cycles,jobs,mode,completion_status,result_valid,completion_sequence,completion_epoch,job_root_acks,job_output_starts);
        publics++;@(negedge clk);bind_valid=0;append_valid=0;completion_ready=1;tick();@(negedge clk);completion_ready=0;
        require(!busy && !completion_valid && job_root_acks==1 && job_output_acks==job_output_starts,"Final ownership release");
        $display("PROOT_RELEASE cycle=%0d job=%0d root_acks=%0d output_acks=%0d",cycles,jobs,job_root_acks,job_output_acks);
    end endtask
    initial begin
        clk=0;
        rst_n=0;
        bind_valid=0;
        bind_cache_base=0;
        bind_region_base=0;
        bind_region_bytes=0;
        bind_capacity=0;
        bind_sequence=0;
        bind_epoch=0;
        expected_epoch=0;
        append_valid=0;
        append_layer=0;
        append_position=0;
        append_sequence=0;
        append_epoch=0;
        data_valid=0;
        data_bits=0;
        data_index=0;
        data_last=0;
        data_sequence=0;
        data_epoch=0;
        commit_valid=0;
        commit_status=0;
        commit_sequence=0;
        commit_epoch=0;
        append_cancel_valid=0;
        append_done_ready=0;
        m_axi_awready=0;
        m_axi_wready=0;
        m_axi_bid=0;
        m_axi_bresp=0;
        m_axi_bvalid=0;
        m_axi_arready=0;
        m_axi_rdata=0;
        m_axi_rresp=0;
        m_axi_rid=0;
        m_axi_rlast=0;
        m_axi_rvalid=0;
        command_valid=0;
        command_position=0;
        command_layer=0;
        command_sequence=0;
        command_epoch=0;
        cancel_valid=0;
        completion_ready=0;
        command_lease_base=0;
        q_valid=0;
        q_data=0;
        q_index=0;
        q_last=0;
        q_sequence=0;
        q_epoch=0;
        command_weight_addr=0;
        command_weight_region_base=0;
        command_weight_region_bytes=0;
        command_activation_addr=0;
        command_activation_region_base=0;
        command_activation_region_bytes=0;
        command_output_addr=0;
        command_output_region_base=0;
        command_output_region_bytes=0;

        if($value$plusargs("SEED=%d",seed))begin end rng=seed;rng=$urandom(rng);
        reset_all();$display("PROOT_CONFIG seed=%0d",seed);
        run_job(0);for(integer mode=1;mode<=15;mode++)begin run_job(mode);run_job(0);end
        require(jobs==31 && publics==29 && root_starts==31 && root_acks==30 && output_starts==25 && output_acks==24 && resets==3 && binds==31 && appends==31,"Suite accounting");
        $display("PROOT_PASS seed=%0d jobs=%0d publics=%0d roots=%0d outputs=%0d root_acks=%0d output_acks=%0d resets=%0d binds=%0d appends=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d reads=%0d cycles=%0d",seed,jobs,publics,root_starts,output_starts,root_acks,output_acks,resets,binds,appends,aws,ws,bs,ars,rs,source_reads,cycles);
        $finish;
    end
    initial begin #1000000;$fatal(1,"PROOT_FAIL timeout");end
endmodule
`default_nettype wire
