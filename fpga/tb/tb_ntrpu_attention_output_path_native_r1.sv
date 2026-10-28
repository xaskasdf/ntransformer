`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_output_path_native_r1;
    logic clk,rst_n;
    logic command_valid;
    logic command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic [63:0] command_weight_addr,command_weight_region_base,command_weight_region_bytes;
    logic [63:0] command_activation_addr,command_activation_region_base,command_activation_region_bytes;
    logic [63:0] command_output_addr,command_output_region_base,command_output_region_bytes;
    logic cancel_valid;
    logic cancel_ready,busy;
    logic completion_valid;
    logic completion_ready;
    logic [3:0] completion_status;
    logic [63:0] completion_sequence,completion_epoch;
    logic result_valid;
    logic [63:0] result_address;
    logic [16:0] result_bytes;
    logic tag_exhausted;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize, m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid, m_axi_arcache;
    logic m_axi_arlock, m_axi_arvalid;
    logic m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast, m_axi_rvalid;
    logic m_axi_rready;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize, m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid, m_axi_awcache;
    logic m_axi_awlock, m_axi_awvalid;
    logic m_axi_awready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast, m_axi_wvalid;
    logic m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic m_axi_bvalid;
    logic m_axi_bready;
    logic source_completion_valid;
    logic source_completion_ready;
    logic [3:0] source_completion_status;
    logic [63:0] source_sequence,source_epoch;
    logic source_result_valid;
    logic source_result_ce;
    logic [10:0] source_result_address;
    logic [31:0] source_result_q;
    logic source_result_q_valid;
    logic [31:0] completion_hls_status;
    ntrpu_attention_output_path_r1 dut(.*);
    always #2.5 clk=~clk;
    integer seed=1,rng=1,cycles=0,jobs=0,mode_now=0,qacks=0,packs=0,qar=0,par=0,qb=0,pb=0,roots=0,publics=0,resets=0;
    integer qstarts=0,pstarts=0,aws=0,ws=0,bs=0,ars=0,rs=0;
    logic pause_aw=0,pause_w=0,pause_r=0,aw_pending=0,w_pending=0,r_pending=0;
    logic [63:0] bus_address,seq_model,epoch_model;
    logic [511:0] write_data,activation_data,output_data;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"PATH_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,mode_now,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    task automatic reset_all;
        @(negedge clk);rst_n=0;command_valid=0;cancel_valid=0;completion_ready=0;
        source_completion_valid=0;source_result_valid=0;dut.quantize_store.foreign=0;dut.projection.foreign=0;dut.projection.hold_admission=0;
        repeat(3)tick();@(negedge clk);rst_n=1;resets++;repeat(2)tick();
    endtask
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;m_axi_arready=0;m_axi_rvalid=0;end
        else begin
            m_axi_awready=!pause_aw && $urandom_range(0,3)!=0;
            m_axi_wready=!pause_w && $urandom_range(0,3)!=0;
            m_axi_arready=$urandom_range(0,3)!=0;
            if(aw_pending && w_pending && !m_axi_bvalid && $urandom_range(0,3)!=0)begin m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;end
            if(r_pending && !pause_r && !m_axi_rvalid && $urandom_range(0,3)!=0)begin
                m_axi_rdata=activation_data;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=1;m_axi_rvalid=1;
            end
        end
    end
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin aw_pending=0;w_pending=0;r_pending=0;source_result_q_valid<=0;activation_data='x;output_data='x;end
        else begin
            source_result_q_valid<=source_result_ce;
            if(source_result_ce)begin
                require(source_completion_valid && source_result_address==seq_model[10:0],"Source request wiring");
                source_result_q<=32'h3f000000|32'(source_result_address);
            end
            if(source_completion_valid && source_completion_ready)begin
                roots++;source_completion_valid<=0;source_result_valid<=0;
                $display("PATH_ROOT_ACK cycle=%0d job=%0d",cycles,jobs);
            end
            if(dut.quantize_store.command_valid && dut.quantize_store.command_ready)qstarts++;
            if(dut.projection.command_valid)begin
                require(dut.quantize_store.completion_valid && !dut.quantize_store.completion_ready && qb==1 && !aw_pending && !w_pending,"O offered without retained quant region/drain");
                require(dut.projection.command_sequence==seq_model && dut.projection.command_epoch==epoch_model,"Full parent O identity");
            end
            if(dut.projection.command_valid && dut.projection.command_ready)begin pstarts++;par++;end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awlen==0 && m_axi_awsize==6,"AW routing");
                bus_address=m_axi_awaddr;aw_pending=1;aws++;
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(!w_pending && m_axi_wstrb=='1 && m_axi_wlast,"W routing");
                write_data=m_axi_wdata;w_pending=1;ws++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending,"B drain");
                if(bus_address==64'h8000000fc0)begin
                    require(!dut.projection_owned && write_data=={8{64'h0123456789abcdef^seq_model}},"Quant write ownership/bytes");
                    activation_data=write_data;qb++;
                end else begin
                    require(bus_address==64'ha000000fc0 && dut.projection_owned && write_data==~activation_data,"O write ownership/bytes");
                    output_data=write_data;pb++;
                end
                bs++;aw_pending=0;w_pending=0;m_axi_bvalid<=0;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!r_pending && qb==1 && dut.quantize_store.completion_valid && !dut.quantize_store.completion_ready,"Read before published region or after release");
                require(m_axi_araddr==64'h8000000fc0 && m_axi_arlen==0 && m_axi_arsize==6,"AR routing");
                r_pending=1;ars++;
            end
            if(m_axi_rvalid && m_axi_rready)begin require(r_pending,"Unrequested R");r_pending=0;rs++;m_axi_rvalid<=0;end
            if(dut.quantize_store.completion_valid && dut.quantize_store.completion_ready)begin
                require(completion_valid && completion_ready && !aw_pending && !w_pending && !r_pending,"Early quant release");qacks++;
            end
            if(dut.projection.completion_valid && dut.projection.completion_ready)begin
                require(completion_valid && completion_ready && dut.quantize_store.completion_ready,"Separate child release");packs++;
            end
            if(result_valid)require(completion_valid && completion_status==0 && pb==1 && output_data==~activation_data && result_address==64'ha000000fc0 && result_bytes==8192,"Public region");
            else require(result_address==0 && result_bytes==0,"Private region leak");
        end
    end
    task automatic pulse_cancel;
        @(negedge clk);require(cancel_ready,"Cancel ready");cancel_valid=1;tick();@(negedge clk);cancel_valid=0;
    endtask
    task automatic run_job(input integer mode);
        integer limit,start_qack,start_pack;logic [3:0] expected_status;logic expect_projection;
        begin: work
        @(negedge clk);require(!busy,"Prior owner");jobs++;mode_now=mode;qb=0;pb=0;par=0;
        pause_aw=mode==16;pause_w=mode==3;pause_r=mode==4 || mode==5 || mode==17;
        dut.quantize_store.test_mode=mode;dut.projection.test_mode=mode;
        dut.quantize_store.foreign=mode==9;dut.projection.foreign=mode==10;dut.projection.hold_admission=mode==12;
        seq_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        command_sequence=seq_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        source_sequence=seq_model;source_epoch=epoch_model;source_completion_status=0;source_completion_valid=1;source_result_valid=1;
        command_weight_addr=64'h100000fc0;command_weight_region_base=64'h100000000;command_weight_region_bytes=4460480;
        command_activation_addr=64'h8000000fc0;command_activation_region_base=64'h8000000000;command_activation_region_bytes=8128;
        command_output_addr=64'ha000000fc0;command_output_region_base=64'ha000000000;command_output_region_bytes=12224;
        command_valid=1;start_qack=qacks;start_pack=packs;
        $display("PATH_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,seq_model,epoch_model);
        tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;
        command_weight_addr=0;command_weight_region_base=0;command_weight_region_bytes=0;
        command_activation_addr=0;command_activation_region_base=0;command_activation_region_bytes=0;
        command_output_addr=0;command_output_region_base=0;command_output_region_bytes=0;
        if(mode==3)begin wait(aw_pending);pulse_cancel();repeat(3)tick();@(negedge clk);pause_w=0;end
        if(mode==4 || mode==5)begin
            wait(r_pending);
            if(mode==4)pulse_cancel();
            else begin @(negedge clk);expected_epoch=epoch_model^64'h10000000000;repeat(3)tick();@(negedge clk);expected_epoch=epoch_model;end
            repeat(3)tick();@(negedge clk);pause_r=0;
        end
        if(mode==9)begin wait(dut.q_completion_valid);repeat(10)begin tick();require(!completion_valid && qacks==start_qack,"Foreign quant identity released");end @(negedge clk);dut.quantize_store.foreign=0;end
        if(mode==10)begin wait(dut.p_completion_valid);repeat(10)begin tick();require(!completion_valid && packs==start_pack,"Foreign O identity released");end @(negedge clk);dut.projection.foreign=0;end
        if(mode==11)begin wait(!source_completion_valid);@(negedge clk);source_completion_valid=1;source_result_valid=1;source_sequence=seq_model^64'h10000000000;source_epoch=epoch_model^64'h10000000000;end
        if(mode==12)begin wait(dut.projection.command_valid);pulse_cancel();repeat(10)begin tick();require(dut.projection.command_valid,"Offered O withdrawn");end @(negedge clk);dut.projection.hold_admission=0;end
        if(mode==16 || mode==17)begin
            if(mode==16)wait(w_pending);else wait(r_pending);
            require(!completion_valid,"Completion before reset");reset_all();pause_aw=0;pause_w=0;pause_r=0;
            $display("PATH_ABORT cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);disable work;
        end
        limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<500,"Owner completion timeout");end
        expected_status=(mode==1 || mode==7) ? 3:(mode==2 || mode==8 || mode==14 || mode==15) ? 5:
            (mode==3 || mode==4 || mode==12) ? 7:mode==5 ? 2:(mode==9 || mode==10) ? 11:0;
        expect_projection=!(mode==1 || mode==2 || mode==3 || mode==9 || mode==14);
        require(completion_status==expected_status && completion_sequence==seq_model && completion_epoch==epoch_model,"Parent completion");
        require(par==int'(expect_projection) && qb==1 && pb==int'(expect_projection),"Stage execution/drain");
        require(completion_hls_status==(mode==1 ? 2:0),"HLS diagnostic propagation");
        if(mode==6)begin @(negedge clk);expected_epoch=epoch_model^64'h10000000000;repeat(3)tick();@(negedge clk);expected_epoch=epoch_model;end
        if(mode==13)pulse_cancel();
        repeat(7)begin tick();require(completion_valid && !command_ready && qacks==start_qack && packs==start_pack,"Held ownership");end
        require(result_valid==(expected_status==0 && mode!=6 && mode!=13),"Publication/revocation");
        $display("PATH_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d hls=%0d sequence=%016h epoch=%016h projection=%0d valid=%0d bytes=%0d",cycles,jobs,mode,completion_status,completion_hls_status,completion_sequence,completion_epoch,par,result_valid,result_bytes);
        publics++;@(negedge clk);completion_ready=1;tick();@(negedge clk);completion_ready=0;source_completion_valid=0;source_result_valid=0;
        require(!busy && !completion_valid && qacks==start_qack+1 && packs==start_pack+int'(expect_projection),"Atomic child release");
        end
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;expected_epoch=0;
        source_completion_valid=0;source_result_valid=0;source_result_q=0;source_result_q_valid=0;
        m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;m_axi_bresp=0;m_axi_bid=0;m_axi_arready=0;m_axi_rvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;
        if($value$plusargs("SEED=%d",seed))begin end rng=seed;rng=$urandom(rng);
        reset_all();run_job(0);for(integer mode=1;mode<=17;mode++)begin run_job(mode);run_job(0);end
        require(jobs==35 && publics==33 && qacks==33 && packs==28 && resets==3,"Suite totals");
        $display("PATH_PASS seed=%0d jobs=%0d publics=%0d qacks=%0d packs=%0d roots=%0d resets=%0d qstarts=%0d pstarts=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d cycles=%0d",seed,jobs,publics,qacks,packs,roots,resets,qstarts,pstarts,aws,ws,bs,ars,rs,cycles);
        $finish;
    end
    initial begin #1000000;$fatal(1,"PATH_FAIL global timeout");end
endmodule
`default_nettype wire
