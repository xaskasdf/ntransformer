`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_projection_memory_r2;
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
    localparam logic [63:0] WBASE=64'h100000000,ABASE=64'h8000000000,OBASE=64'ha000000000;
    localparam logic [63:0] FIRST_TAG=64'h00000000ffffff80;
    ntrpu_attention_projection_memory_r1 #(.INITIAL_CHILD_TAG(FIRST_TAG)) dut(.*);
    always #2.5 clk=~clk;
    logic [511:0] weight_image[0:69694],activation_image[0:255],destination[0:127];
    logic [63:0] reference64[0:8191];logic [31:0] reference32[0:8191];
    string fixture;
    integer seed=1,random_state=1,cycles=0,jobs=0,mode_now=0,sample_now=0;
    integer ars=0,rs=0,aws=0,ws=0,bs=0,groups=0,tiles=0,computed=0,replies=0;
    integer total_ar=0,total_r=0,total_aw=0,total_w=0,total_b=0,total_results=0,publics=0;
    integer r_index=0,r_beats=0,aw_beats=0,w_pending=0,write_cursor=0,aw_number=0;
    integer quiet=0,max_quiet=0,job_start=0,ar_stalls=0,r_stalls=0,aw_stalls=0,w_stalls=0;
    logic r_active=0,aw_pending=0,hold_ar=0,hold_aw=0,hold_w=0,hold_r=0;
    logic [63:0] r_address,aw_address,sequence_model,epoch_model,tag_model;
    logic [88:0] saved_ar,saved_aw;logic [576:0] saved_w;logic [518:0] saved_r;
    logic [1:0] pending_last;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"OPROJ_FAIL job=%0d mode=%0d cycle=%0d: %s",jobs,mode_now,cycles,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [511:0] read_word(input logic [63:0] address);
        if(address>=WBASE+4032 && address+64<=WBASE+4460480)return weight_image[int'((address-WBASE)>>6)];
        if(!(address>=ABASE+4032 && address+64<=ABASE+8128))
            $fatal(1,"OPROJ_FAIL job=%0d cycle=%0d: Read outside immutable images",jobs,cycles);
        return activation_image[sample_now*64+int'((address-ABASE-4032)>>6)];
    endfunction
    function automatic logic [511:0] output_word(input int index);
        logic [511:0] value;
        for(integer i=0;i<16;i++)value[i*32+:32]=reference32[sample_now*2048+index*16+i];
        return value;
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_arready=0;m_axi_rvalid=0;m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;end
        else begin
            m_axi_arready=$urandom_range(0,3)!=0;m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if(r_active && !m_axi_rvalid && $urandom_range(0,3)!=0)begin
                m_axi_rdata=read_word(r_address+64'(r_index*64));m_axi_rresp=0;m_axi_rid=0;
                m_axi_rlast=r_index==r_beats-1;m_axi_rvalid=1;
            end
            if(aw_pending && w_pending==aw_beats && !m_axi_bvalid && $urandom_range(0,3)!=0)begin
                for(integer i=0;i<w_pending;i++)require(pending_last[i]==(i==aw_beats-1),"Write burst last");
                m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;
            end
        end
    end
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin hold_ar=0;hold_r=0;hold_aw=0;hold_w=0;r_active=0;aw_pending=0;w_pending=0;end
        else begin
            if(dut.control.active_child)begin
                if(dut.child_activity)quiet=0;else quiet++;
                if(quiet>max_quiet)max_quiet=quiet;
            end else quiet=0;
            if(hold_ar)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock}===saved_ar,"Held AR changed");
            if(hold_aw)require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"Held AW changed");
            if(hold_w)require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"Held W changed");
            if(hold_r)require(m_axi_rvalid && {m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast}===saved_r,"Held R changed");
            hold_ar=m_axi_arvalid && !m_axi_arready;hold_aw=m_axi_awvalid && !m_axi_awready;hold_w=m_axi_wvalid && !m_axi_wready;hold_r=m_axi_rvalid && !m_axi_rready;
            saved_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};saved_r={m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast};
            ar_stalls+=int'(hold_ar);aw_stalls+=int'(hold_aw);w_stalls+=int'(hold_w);r_stalls+=int'(hold_r);
            if(dut.child_job_valid && dut.child_job_ready)begin
                require({dut.child_job_epoch,dut.child_job_sequence}==tag_model,"Reserved matrix identity");
                require(dut.child_job_rows==2048 && dut.child_job_k_blocks==64 && dut.child_job_tile_k==32 && dut.child_job_layout==2 && dut.child_job_output_fp32,"Full O shape");
                $display("OPROJ_MATRIX cycle=%0d job=%0d tag=%016h",cycles,jobs,{dut.child_job_epoch,dut.child_job_sequence});
            end
            if(dut.matrix.e_job_valid && dut.matrix.e_job_ready)begin
                require(groups==replies && !r_active && !aw_pending && w_pending==0,"Group overlap");
                require(dut.matrix.e_job_rows==32 && dut.matrix.e_job_output_fp32 && dut.matrix.e_job_k_blocks==64,"Group shape");
                require({dut.matrix.e_job_epoch,dut.matrix.e_job_sequence}==tag_model+64'(groups),"Group identity");
                require(dut.matrix.e_job_output_addr==OBASE+4032+64'(groups*128),"Group output address");
                $display("OPROJ_GROUP cycle=%0d job=%0d group=%0d tag=%016h",cycles,jobs,groups,{dut.matrix.e_job_epoch,dut.matrix.e_job_sequence});groups++;
            end
            if(dut.matrix.e_tile_valid && dut.matrix.e_tile_ready)begin
                require(dut.matrix.e_tile_addr==WBASE+4032+64'(tiles*34816) && dut.matrix.e_tile_k_count==32 && dut.matrix.e_tile_k_base==9'((tiles%2)*32),"O tile sequence");
                $display("OPROJ_TILE cycle=%0d job=%0d tile=%0d address=%016h",cycles,jobs,tiles,dut.matrix.e_tile_addr);tiles++;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!r_active && groups>0 && m_axi_araddr[5:0]==0,"AR admission/alignment");
                require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"AR crosses 4 KiB");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR attributes");
                r_address=m_axi_araddr;r_beats=int'(m_axi_arlen)+1;r_index=0;r_active=1;
                $display("OPROJ_AR cycle=%0d job=%0d ar=%0d address=%016h beats=%0d group=%0d",cycles,jobs,ars,r_address,r_beats,groups-1);ars++;total_ar++;
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(r_active && m_axi_rlast==(r_index==r_beats-1),"Read burst accounting");
                $display("OPROJ_R cycle=%0d job=%0d ar=%0d beat=%0d address=%016h data=%0128h last=%0d",cycles,jobs,ars-1,r_index,r_address+64'(r_index*64),m_axi_rdata,m_axi_rlast);
                r_index++;rs++;total_r++;if(m_axi_rlast)r_active=0;m_axi_rvalid<=0;
            end
            if(dut.matrix.engine.result_valid && dut.matrix.engine.result_ready)begin
                require(dut.matrix.engine.result_row==5'(computed%32),"Result row order");
                require(dut.matrix.engine.result_data===reference64[sample_now*2048+computed],"Real O FP64 arithmetic");
                $display("OPROJ_MAC cycle=%0d job=%0d row=%0d data=%016h",cycles,jobs,computed,dut.matrix.engine.result_data);computed++;total_results++;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awaddr==OBASE+4032+64'(write_cursor*64),"AW global address");
                require(m_axi_awlen<=1 && int'(m_axi_awaddr[11:0])+(int'(m_axi_awlen)+1)*64<=4096,"AW burst/page");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                aw_pending=1;aw_beats=int'(m_axi_awlen)+1;aw_address=m_axi_awaddr;aw_number=aws;write_cursor+=aw_beats;
                $display("OPROJ_AW cycle=%0d job=%0d aw=%0d address=%016h beats=%0d",cycles,jobs,aws,m_axi_awaddr,aw_beats);aws++;total_aw++;
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(ws<128 && w_pending<2 && m_axi_wdata===output_word(ws),"Real O FP32 AXI bytes");
                require(m_axi_wstrb==64'hffffffffffffffff,"FP32 byte mask");
                destination[ws]=m_axi_wdata;pending_last[w_pending]=m_axi_wlast;w_pending++;
                $display("OPROJ_W cycle=%0d job=%0d word=%0d data=%0128h last=%0d",cycles,jobs,ws,m_axi_wdata,m_axi_wlast);ws++;total_w++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending==aw_beats,"B before AW/W drain");
                $display("OPROJ_B cycle=%0d job=%0d aw=%0d response=%0d",cycles,jobs,aw_number,m_axi_bresp);
                bs++;total_b++;aw_pending=0;w_pending=0;m_axi_bvalid<=0;
            end
            if(dut.matrix.e_done_valid && dut.matrix.e_done_ready)begin
                require(!r_active && !aw_pending && w_pending==0 && groups==replies+1,"Group completion before AXI drain");
                $display("OPROJ_GROUP_DONE cycle=%0d job=%0d group=%0d status=%0d bytes=%0d",cycles,jobs,replies,dut.matrix.e_done_status,dut.matrix.e_done_bytes);replies++;
            end
            if(dut.child_done_valid && dut.child_done_ready)begin
                require({dut.child_done_epoch,dut.child_done_sequence}==tag_model,"Matrix completion identity");
                $display("OPROJ_MATRIX_DONE cycle=%0d job=%0d status=%0d bytes=%0d rows=%0d tag=%016h",cycles,jobs,dut.child_done_status,dut.child_done_bytes,dut.child_done_rows,{dut.child_done_epoch,dut.child_done_sequence});
            end
            if(result_valid)begin
                require(completion_valid && computed==2048 && ws==128 && aws==bs && !r_active && !aw_pending && w_pending==0,"Publication before completion/drain");
                require(result_address==OBASE+4032 && result_bytes==8192,"Public output extent");
                for(integer i=0;i<128;i++)require(destination[i]===output_word(i),"Published FP32 memory");
            end else require(result_address==0 && result_bytes==0,"Private result leak");
        end
    end
    task automatic run_job(input integer mode,input integer sample_id);
        integer limit;logic [3:0] status_model;
        @(negedge clk);require(command_ready && !busy,"Previous owner active");jobs++;mode_now=mode;sample_now=sample_id;
        ars=0;rs=0;aws=0;ws=0;bs=0;groups=0;tiles=0;computed=0;replies=0;quiet=0;max_quiet=0;write_cursor=0;
        for(integer i=0;i<128;i++)destination[i]='1;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);tag_model=FIRST_TAG+64'((jobs-1)*64);
        command_sequence=sequence_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_weight_addr=WBASE+4032;command_weight_region_base=WBASE;command_weight_region_bytes=4460480;
        command_activation_addr=ABASE+4032;command_activation_region_base=ABASE;command_activation_region_bytes=8128;
        command_output_addr=OBASE+4032;command_output_region_base=OBASE;command_output_region_bytes=12224-(mode==1 ? 1:0);
        command_valid=1;job_start=cycles;
        $display("OPROJ_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d sequence=%016h epoch=%016h tag=%016h",cycles,jobs,mode,sample_now,sequence_model,epoch_model,tag_model);
        tick();@(negedge clk);command_valid=0;command_sequence=0;command_epoch=0;
        command_weight_addr=0;command_weight_region_base=0;command_weight_region_bytes=0;
        command_activation_addr=0;command_activation_region_base=0;command_activation_region_bytes=0;
        command_output_addr=0;command_output_region_base=0;command_output_region_bytes=0;
        if(mode==2)begin
            wait(rs>=70);@(negedge clk);require(cancel_ready,"Cancel admission");cancel_valid=1;
            $display("OPROJ_CANCEL cycle=%0d job=%0d reads=%0d",cycles,jobs,rs);
            tick();@(negedge clk);cancel_valid=0;
        end
        limit=0;while(!completion_valid)begin @(negedge clk);limit++;require(limit<3000000,"Full O completion timeout");end
        status_model=mode==1 ? 1:mode==2 ? 7:0;
        require(completion_status==status_model && completion_sequence==sequence_model && completion_epoch==epoch_model,"Full parent completion");
        require(result_valid==(mode==0) && !r_active && !aw_pending && w_pending==0 && aws==bs,"Completion publication/drain");
        if(mode==0)require(computed==2048 && groups==64 && replies==64 && tiles==128 && ars==1344 && rs==73728 && aws==66 && ws==128 && bs==66,"Full O coverage");
        else require(computed==0 && aws==0 && ws==0 && bs==0,"Failed O publication");
        if(mode==1)require(ars==0 && rs==0 && groups==0 && tiles==0,"Bounds before reads");
        require(max_quiet<65536,"Parent watchdog activity gap");
        repeat(7)begin tick();require(completion_valid && completion_status==status_model && !command_ready,"Held public completion");end
        $display("OPROJ_COMPLETE cycle=%0d job=%0d mode=%0d status=%0d sequence=%016h epoch=%016h valid=%0d bytes=%0d groups=%0d tiles=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d max_quiet=%0d",cycles,jobs,mode,completion_status,completion_sequence,completion_epoch,result_valid,result_bytes,groups,tiles,computed,ars,rs,aws,ws,bs,max_quiet);
        publics++;@(negedge clk);completion_ready=1;tick();@(negedge clk);completion_ready=0;require(!busy && !completion_valid,"Owner release");
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;expected_epoch=0;
        m_axi_arready=0;m_axi_rvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;
        m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;m_axi_bresp=0;m_axi_bid=0;
        require($value$plusargs("FIXTURE=%s",fixture),"Missing real O fixture");
        $readmemh({fixture,"/weight.hex"},weight_image);$readmemh({fixture,"/activation.hex"},activation_image);
        $readmemh({fixture,"/reference-fp64.hex"},reference64);$readmemh({fixture,"/reference-fp32.hex"},reference32);
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        $display("OPROJ_CONFIG seed=%0d rows=2048 k_blocks=64 tile_k=32 watchdog=65536",seed);
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        for(integer i=0;i<4;i++)run_job(0,i);
        run_job(1,0);run_job(2,1);run_job(0,2);
        require(jobs==7 && publics==7 && total_results==10240 && total_aw==330 && total_w==640 && total_b==330,"Suite coverage");
        $display("OPROJ_PASS seed=%0d jobs=%0d publics=%0d results=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d ar_stalls=%0d r_stalls=%0d aw_stalls=%0d w_stalls=%0d cycles=%0d",seed,jobs,publics,total_results,total_ar,total_r,total_aw,total_w,total_b,ar_stalls,r_stalls,aw_stalls,w_stalls,cycles);
        $finish;
    end
    initial begin #100000000;$fatal(1,"OPROJ_FAIL global timeout");end
endmodule
`default_nettype wire
