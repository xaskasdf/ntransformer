`timescale 1ns/1ps
`default_nettype none
// Actual generated HLS arithmetic; fixture outputs are comparison oracles only.
module tb_ntrpu_post_attention_real_r2;
    logic clk,rst_n;
    logic command_valid;
    wire command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic [63:0] command_source_sequence,command_source_epoch;
    logic command_source_valid;
    logic [63:0] command_input_address,command_input_base,command_input_bytes;
    logic [63:0] command_branch_address,command_branch_base,command_branch_bytes;
    logic [63:0] command_weight_address,command_weight_base,command_weight_bytes;
    logic [63:0] command_residual_address,command_residual_base,command_residual_bytes;
    logic [63:0] command_ffn_address,command_ffn_base,command_ffn_bytes;
    logic cancel_valid;
    wire cancel_ready,busy;
    wire completion_valid;
    logic completion_ready;
    wire [3:0] completion_status;
    wire [31:0] completion_hls_status;
    wire [63:0] completion_sequence,completion_epoch;
    wire [9:0] completion_read_beats;
    wire [7:0] completion_residual_beats;
    wire [6:0] completion_ffn_beats;
    wire result_valid;
    wire [63:0] result_residual_address,result_ffn_address;
    wire [13:0] result_residual_bytes;
    wire [12:0] result_ffn_bytes;
    wire hls_start;
    wire hls_done,hls_ready;
    wire [31:0] hls_return;
    wire [31:0] hls_elements,hls_epsilon_bits;
    wire [2:0] hls_input_ce;
    wire [10:0] hls_input_address [0:2];
    wire [31:0] hls_input_q [0:2];
    wire hls_residual_ce,hls_residual_we;
    wire [10:0] hls_residual_address;
    wire [31:0] hls_residual_d;
    wire hls_ffn_ce,hls_ffn_we;
    wire [8:0] hls_ffn_address;
    wire [63:0] hls_ffn_d;
    wire [63:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize,m_axi_arprot;
    wire [1:0] m_axi_arburst;
    wire [3:0] m_axi_arid,m_axi_arcache;
    wire m_axi_arlock,m_axi_arvalid;
    logic m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast,m_axi_rvalid;
    wire m_axi_rready;
    wire [63:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize,m_axi_awprot;
    wire [1:0] m_axi_awburst;
    wire [3:0] m_axi_awid,m_axi_awcache;
    wire m_axi_awlock,m_axi_awvalid;
    logic m_axi_awready;
    wire [511:0] m_axi_wdata;
    wire [63:0] m_axi_wstrb;
    wire m_axi_wlast,m_axi_wvalid;
    logic m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic m_axi_bvalid;
    wire m_axi_bready;

    localparam integer WD=65536;
    ntrpu_post_attention_real_r2 #(.AXI_WATCHDOG_CYCLES(WD),.HLS_WATCHDOG_CYCLES(262144)) dut(.*);
    assign hls_start=dut.hls_start;
    assign hls_done=dut.hls_done;
    assign hls_ready=dut.hls_ready;
    assign hls_return=dut.hls_return;
    assign hls_elements=dut.hls_elements;
    assign hls_epsilon_bits=dut.hls_epsilon_bits;
    assign hls_input_ce=dut.hls_input_ce;
    for(genvar p=0;p<3;p++)assign hls_input_address[p]=dut.hls_input_address[p];
    for(genvar p=0;p<3;p++)assign hls_input_q[p]=dut.hls_input_q[p];
    assign hls_residual_ce=dut.hls_residual_ce;
    assign hls_residual_we=dut.hls_residual_we;
    assign hls_residual_address=dut.hls_residual_address;
    assign hls_residual_d=dut.hls_residual_d;
    assign hls_ffn_ce=dut.hls_ffn_ce;
    assign hls_ffn_we=dut.hls_ffn_we;
    assign hls_ffn_address=dut.hls_ffn_address;
    assign hls_ffn_d=dut.hls_ffn_d;
    always #2.5 clk=~clk;
    logic [31:0] source_words[0:24575],residual_words[0:8191];
    logic [63:0] abi_words[0:2047];
    logic [511:0] memory[0:191],pending_write;
    logic [63:0] pending_address,read_address;
    logic [63:0] source_address[0:2],sequence_model,epoch_model,residual_model,ffn_model;
    integer seed=1,random_state=1,cycles=0,jobs=0,mode_now=0,sample_now=0;
    integer ars=0,rs=0,aws=0,ws=0,bs=0,read_beat=0,read_beats=0,core_reads=0,core_writes=0;
    integer total_ar=0,total_r=0,total_b=0,starts=0,dones=0,publics=0,resets=0,pairs=0;
    integer core_quiet=0,max_core_quiet=0;
    logic [31:0] core_return=0;
    logic read_active=0,core_busy=0,held_aw=0,held_w=0,held_ar=0;
    logic [88:0] saved_aw;
    logic [576:0] saved_w;
    logic [71:0] saved_ar;
    string fixture;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"POSTREAL_FAIL cycle=%0d job=%0d mode=%0d: %s",cycles,jobs,mode_now,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [31:0] input_word(input integer plane,input integer index);
        return source_words[sample_now*6144+plane*2048+index];
    endfunction
    function automatic logic [31:0] memory_word(input logic [63:0] address);
        for(integer p=0;p<3;p++)if(address>=source_address[p] && address<source_address[p]+8192)
            return input_word(p,int'((address-source_address[p])/4));
        return 32'h7fc12345;
    endfunction
    function automatic logic [511:0] output_word(input integer record_index);
        logic [511:0] value;
        if(record_index<128)for(integer lane=0;lane<16;lane++)
            value[lane*32+:32]=residual_words[sample_now*2048+record_index*16+lane];
        else for(integer lane=0;lane<8;lane++)
            value[lane*64+:64]=abi_words[sample_now*512+(record_index-128)*8+lane];
        return value;
    endfunction
    function automatic logic [63:0] output_address(input integer record_index);
        return record_index<128 ? residual_model+64'(record_index*64):ffn_model+64'((record_index-128)*64);
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_arready=0;m_axi_rvalid=0;m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;end
        else begin
            m_axi_arready=!read_active && $urandom_range(0,3)!=0;
            if(!m_axi_rvalid && read_active && $urandom_range(0,3)!=0)begin
                for(integer lane=0;lane<16;lane++)m_axi_rdata[lane*32+:32]=memory_word(read_address+64'(read_beat*64+lane*4));
                m_axi_rresp=(mode_now==5 && rs==3) ? 2:0;m_axi_rid=0;
                m_axi_rlast=read_beat==read_beats-1;m_axi_rvalid=1;
            end
            m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if(mode_now==19 && bs==0)m_axi_wready=0;
            if(aws>bs && ws>bs && !m_axi_bvalid && $urandom_range(0,3)!=0)begin
                m_axi_bvalid=1;m_axi_bid=0;m_axi_bresp=(mode_now==8 && bs==128) ? 2:0;
            end
        end
    end
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin held_ar=0;held_aw=0;held_w=0;core_busy=0;core_quiet=0;end
        else begin
            if(hls_start)begin
                require(!core_busy && rs==384 && hls_elements==2048 && hls_epsilon_bits==32'h3727c5ac,"Actual HLS start");
                core_busy=1;starts++;core_reads=0;core_writes=0;core_return=0;core_quiet=0;max_core_quiet=0;
                $display("POSTREAL_START cycle=%0d job=%0d",cycles,jobs);
            end
            if(core_busy)begin
                if(|hls_input_ce || (hls_residual_ce && hls_residual_we) || (hls_ffn_ce && hls_ffn_we) || hls_done)core_quiet=0;
                else core_quiet++;
                if(core_quiet>max_core_quiet)max_core_quiet=core_quiet;
                require(max_core_quiet<262144,"Actual HLS quiet interval exceeds watchdog");
                if(|hls_input_ce)core_reads++;
                if(hls_residual_ce && hls_residual_we)begin
                    if(dut.path.owner.fault==0 && !cancel_valid && expected_epoch==epoch_model)
                        require(hls_residual_d===residual_words[sample_now*2048+int'(hls_residual_address)],"Actual HLS residual");
                    core_writes++;
                end
                if(hls_ffn_ce && hls_ffn_we && dut.path.owner.fault==0 && !cancel_valid && expected_epoch==epoch_model)
                    require(hls_ffn_d===abi_words[sample_now*512+int'(hls_ffn_address)],"Actual HLS ABI1");
            end
            if(hls_done)begin
                require(core_busy && hls_ready,"Actual HLS completion protocol");
                core_busy=0;dones++;core_return=hls_return;
                $display("POSTREAL_DONE cycle=%0d job=%0d return=%0d",cycles,jobs,hls_return);
            end
            if(held_ar)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"Held AR");
            if(held_aw)require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"Held AW");
            if(held_w)require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"Held W");
            held_ar=m_axi_arvalid && !m_axi_arready;held_aw=m_axi_awvalid && !m_axi_awready;held_w=m_axi_wvalid && !m_axi_wready;
            saved_ar={m_axi_araddr,m_axi_arlen};saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            if(m_axi_arvalid && m_axi_arready)begin
                require(!read_active && !core_busy && aws==0,"Read stage ownership");
                require(m_axi_araddr==source_address[rs/128]+64'((rs%128)*64),"Read address/order");
                require(m_axi_arlen<64 && int'(m_axi_araddr[11:0])+64*(int'(m_axi_arlen)+1)<=4096,"Read page extent");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR attributes");
                read_active=1;read_address=m_axi_araddr;read_beats=int'(m_axi_arlen)+1;read_beat=0;ars++;total_ar++;
                $display("POSTREAL_AR cycle=%0d job=%0d address=%016h beats=%0d",cycles,jobs,read_address,read_beats);
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(read_active,"R without AR");
                $display("POSTREAL_R cycle=%0d job=%0d index=%0d response=%0d data=%0128h",cycles,jobs,rs,m_axi_rresp,m_axi_rdata);
                rs++;total_r++;read_beat++;if(m_axi_rlast)read_active=0;m_axi_rvalid<=0;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!core_busy && !read_active && !dut.path.loader_busy && !dut.path.owner.hls_start,"Write before source/core drain");
                require(aws==bs && aws<192 && m_axi_awaddr==output_address(aws),"AW capture/address");
                require(m_axi_awlen==0 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW attributes");
                pending_address=m_axi_awaddr;aws++;
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(ws==bs && ws<192 && m_axi_wdata===output_word(ws),"Actual composed output data");
                require(m_axi_wlast && m_axi_wstrb=='1,"W attributes");pending_write=m_axi_wdata;ws++;
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aws==bs+1 && ws==bs+1 && pending_address==output_address(bs),"B ordering");
                if(m_axi_bresp==0 && m_axi_bid==0)memory[bs]=pending_write;
                $display("POSTREAL_B cycle=%0d job=%0d index=%0d address=%016h response=%0d data=%0128h",cycles,jobs,bs,pending_address,m_axi_bresp,pending_write);
                bs++;total_b++;m_axi_bvalid<=0;
            end
            if(result_valid)begin
                require(completion_valid && bs==192 && !core_busy && !read_active && !dut.path.owner_busy && !dut.path.loader_busy,"Publication before complete drain");
                require(result_residual_address==residual_model && result_ffn_address==ffn_model && result_residual_bytes==8192 && result_ffn_bytes==4096,"Result descriptor capture");
                for(integer k=0;k<192;k++)require(memory[k]===output_word(k),"Committed memory pair");
            end else require(result_residual_address==0 && result_ffn_address==0 && result_residual_bytes==0 && result_ffn_bytes==0,"Invalid descriptor leak");
            if(completion_valid)require(aws==bs && ws==bs && !core_busy && !read_active,"Completion before drain");
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;resets++;command_valid=0;cancel_valid=0;completion_ready=0;
        read_active=0;m_axi_rvalid=0;m_axi_bvalid=0;
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(!busy && command_ready && !completion_valid && !result_valid && !core_busy,"Coordinated reset");
        $display("POSTREAL_RESET cycle=%0d job=%0d",cycles,jobs);
    endtask
    function automatic integer wanted(input integer m);
        case(m)
            1,2,11:return 2;
            3:return 11;
            4,7:return 1;
            5,8:return 3;
            6:return 5;
            9,10,12,13,20,21:return 7;
            22:return 8;
            default:return 0;
        endcase
    endfunction
    task automatic run_job(input integer mode);
        integer limit,starts_before;logic [3:0] saved_status;
        jobs++;mode_now=mode;sample_now=(jobs-1)%4;starts_before=starts;
        @(negedge clk);require(command_ready && !busy,"Previous request active");
        ars=0;rs=0;aws=0;ws=0;bs=0;core_reads=0;core_writes=0;core_return=0;max_core_quiet=0;
        for(integer k=0;k<192;k++)memory[k]='1;
        sequence_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        for(integer p=0;p<3;p++)source_address[p]=64'h8000000fc0+64'(p)*64'h100000000+64'(jobs*16384);
        residual_model=64'h2200000fc0+64'(jobs*16384);ffn_model=64'h3300000fc0+64'(jobs*16384);
        command_sequence=sequence_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_source_sequence=sequence_model;command_source_epoch=epoch_model;command_source_valid=1;
        command_input_address=source_address[0];command_input_base=source_address[0];command_input_bytes=8192;
        command_branch_address=source_address[1];command_branch_base=source_address[1];command_branch_bytes=8192;
        command_weight_address=source_address[2];command_weight_base=source_address[2];command_weight_bytes=8192;
        command_residual_address=residual_model;command_residual_base=residual_model;command_residual_bytes=8192;
        command_ffn_address=ffn_model;command_ffn_base=ffn_model;command_ffn_bytes=4096;
        case(mode)
            1:command_source_sequence^=64'd1<<40;
            2:command_source_epoch^=64'd1<<44;
            3:command_source_valid=0;
            4:command_weight_bytes=8191;
            7:command_ffn_address=residual_model;
            20:cancel_valid=1;
            default:;
        endcase
        $display("POSTREAL_BEGIN cycle=%0d job=%0d mode=%0d sample=%0d sequence=%016h epoch=%016h",cycles,jobs,mode,sample_now,sequence_model,epoch_model);
        command_valid=1;tick();@(negedge clk);command_valid=0;cancel_valid=0;
        command_sequence=0;command_epoch=0;command_residual_address=0;command_ffn_address=0;
        command_residual_base=0;command_residual_bytes=0;command_ffn_base=0;command_ffn_bytes=0;
        command_input_address=0;command_branch_address=0;command_weight_address=0;command_input_bytes=0;command_branch_bytes=0;command_weight_bytes=0;
        if(mode==9 || mode==17)begin wait(rs>=8);@(negedge clk);end
        if(mode==10 || mode==11 || mode==18)begin wait(core_reads>=32);@(negedge clk);end
        if(mode==12)begin wait(core_writes>=2048);@(negedge clk);end
        if(mode==13)begin wait(bs>=128);@(negedge clk);end
        if(mode==19)begin wait(aws==1 && ws==0);@(negedge clk);end
        if(mode==21)begin wait(dut.path.loader_valid);@(negedge clk);end
        if(mode==9 || mode==10 || mode==12 || mode==13 || mode==21)begin
            cancel_valid=1;tick();@(negedge clk);cancel_valid=0;
        end
        if(mode==11)begin expected_epoch^=64'd1<<48;tick();@(negedge clk);expected_epoch=epoch_model;end
        if(mode>=17 && mode<=19)begin
            $display("POSTREAL_ABORT cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);reset_all();
        end else begin
            limit=0;while(!completion_valid)begin tick();limit++;require(limit<400000,"Completion timeout");end
            require(completion_status==wanted(mode) && completion_sequence==sequence_model && completion_epoch==epoch_model,"Completion status/identity");
            require(completion_hls_status==core_return && (wanted(mode)!=0 || core_return==0),"HLS diagnostic retained");
            require(result_valid==(wanted(mode)==0),"Result visibility");
            if(mode<=5 && mode!=0 || mode==9 || mode==20)require(starts==starts_before && aws==0,"Invalid/cancelled input started HLS or writes");
            if(wanted(mode)==0)begin require(completion_read_beats==384 && completion_residual_beats==128 && completion_ffn_beats==64,"Success extents");pairs++;end
            publics++;saved_status=completion_status;
            $display("POSTREAL_COMPLETE cycle=%0d job=%0d status=%0d hls=%0d valid=%0d read=%0d residual=%0d abi=%0d ar=%0d r=%0d b=%0d max_quiet=%0d",cycles,jobs,completion_status,completion_hls_status,result_valid,completion_read_beats,completion_residual_beats,completion_ffn_beats,ars,rs,bs,max_core_quiet);
            repeat(5)begin tick();require(completion_valid && completion_status==saved_status && !command_ready,"Held completion");end
            if(mode==14 || mode==15)begin
                @(negedge clk);if(mode==14)expected_epoch^=64'd1<<48;else cancel_valid=1;
                #1;require(!result_valid,"Immediate late revocation");tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;
                repeat(3)begin tick();require(!result_valid && completion_valid && completion_status==saved_status,"Sticky revocation/status");end
            end
            if(mode==16)begin $display("POSTREAL_ABORT cycle=%0d job=%0d mode=%0d",cycles,jobs,mode);reset_all();end
            else begin @(negedge clk);completion_ready=1;tick();@(negedge clk);completion_ready=0;tick();require(command_ready && !busy,"Release all children");end
        end
    endtask
    initial begin
        clk=0;
        rst_n=0;
        command_valid=0;
        command_sequence=0;
        command_epoch=0;
        expected_epoch=0;
        command_source_sequence=0;
        command_source_epoch=0;
        command_source_valid=0;
        command_input_address=0;
        command_input_base=0;
        command_input_bytes=0;
        command_branch_address=0;
        command_branch_base=0;
        command_branch_bytes=0;
        command_weight_address=0;
        command_weight_base=0;
        command_weight_bytes=0;
        command_residual_address=0;
        command_residual_base=0;
        command_residual_bytes=0;
        command_ffn_address=0;
        command_ffn_base=0;
        command_ffn_bytes=0;
        cancel_valid=0;
        completion_ready=0;
        m_axi_arready=0;
        m_axi_rdata=0;
        m_axi_rresp=0;
        m_axi_rid=0;
        m_axi_rlast=0;
        m_axi_rvalid=0;
        m_axi_awready=0;
        m_axi_wready=0;
        m_axi_bid=0;
        m_axi_bresp=0;
        m_axi_bvalid=0;
        if(!$value$plusargs("FIXTURE=%s",fixture))$fatal(1,"Missing fixture");
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        $readmemh({fixture,"/source32.hex"},source_words);
        $readmemh({fixture,"/residual32.hex"},residual_words);
        $readmemh({fixture,"/abi64.hex"},abi_words);
        reset_all();
        repeat(4)run_job(0);
        run_job(10);run_job(0);run_job(13);run_job(0);
        $display("POSTREAL_PASS seed=%0d jobs=%0d publics=%0d pairs=%0d resets=%0d starts=%0d dones=%0d ar=%0d r=%0d b=%0d cycles=%0d",seed,jobs,publics,pairs,resets,starts,dones,total_ar,total_r,total_b,cycles);
        $finish;
    end
    initial begin #40000000;$fatal(1,"POSTREAL_FAIL absolute timeout");end
endmodule
`default_nettype wire
