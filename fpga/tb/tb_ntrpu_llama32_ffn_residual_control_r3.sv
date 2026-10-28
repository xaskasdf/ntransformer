`timescale 1ns/1ps
`default_nettype none
// Protocol-only peers; no FFN or residual numerical qualification.
module tb_ntrpu_llama32_ffn_residual_control_r1;
    logic clk,rst_n;
    logic command_valid;
    logic [63:0] command_sequence;
    logic [63:0] command_epoch;
    logic command_source_valid;
    logic [63:0] command_source_sequence;
    logic [63:0] command_source_epoch;
    logic [63:0] command_addr [0:7];
    logic [63:0] command_base [0:7];
    logic [63:0] command_bytes [0:7];
    logic [63:0] command_residual_address;
    logic [63:0] command_residual_base;
    logic [63:0] command_residual_bytes;
    logic [63:0] expected_epoch;
    logic [63:0] command_output_address,command_output_base,command_output_bytes;
    logic command_ready;
    logic cancel_valid;
    logic cancel_ready,busy;
    logic completion_valid;
    logic completion_ready;
    logic [3:0] completion_status,completion_ffn_status,completion_ffn_child_status,completion_residual_status;
    logic [2:0] completion_ffn_stages;
    logic [1:0] completion_ffn_phase;
    logic [16:0] completion_ffn_bytes;
    logic [17:0] completion_ffn_stage_bytes;
    logic [31:0] completion_ffn_hls_status,completion_hls_status;
    logic [9:0] completion_read_beats;
    logic [7:0] completion_write_beats;
    logic [63:0] completion_sequence,completion_epoch;
    logic result_valid;
    logic [63:0] result_address;
    logic [13:0] result_bytes;
    logic tags_exhausted,activation_selected,residual_selected;
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
    ntrpu_llama32_ffn_residual_r2 dut(.*);
    always #2.5 clk=~clk;
    integer seed=1,random_state=1,cycle=0,jobs=0,mode_now=0,publics=0,offered=0,resets=0,total_reads=0,total_writes=0;
    integer ar_count[0:1],r_count[0:1],aw_count[0:1],w_count[0:1],b_count[0:1],starts[0:1];
    logic [63:0] seq_model,epoch_model,addr_model[0:7],base_model[0:7],size_model[0:7];
    logic [63:0] residual_model,residual_base_model,output_model,output_base_model,output_size_model;
    logic reading=0,aw_pending=0,w_pending=0,ffn_committed=0,res_committed=0,old_select=0,seen_ffn_done=0;
    logic [63:0] read_address,pending_address;
    logic [511:0] pending_data,ffn_memory,res_memory;
    integer read_owner=0,write_owner=0,data_owner=0,b_delay=0;
    logic held_ar=0,held_aw=0,held_w=0;
    logic [88:0] saved_ar,saved_aw;
    logic [576:0] saved_w;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"FFRCTRL_FAIL cycle=%0d job=%0d mode=%0d: %s",cycle,jobs,mode_now,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [511:0] pattern(input logic [63:0] address);
        logic [511:0] value;
        for(integer lane=0;lane<8;lane++)value[lane*64+:64]=address^(64'h13579bdf2468ace0+64'(lane));
        return value;
    endfunction
    function automatic logic [511:0] wanted_write(input integer owner);
        if(owner==0)return pattern(addr_model[0])^pattern(addr_model[1])^{16{32'h46}};
        return pattern(residual_model)^ffn_memory^{16{32'h52}};
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;m_axi_rvalid=0;m_axi_bvalid=0;end
        else begin
            m_axi_arready=!reading && $urandom_range(0,3)!=0;
            m_axi_awready=!aw_pending && $urandom_range(0,3)!=0;
            m_axi_wready=!w_pending && $urandom_range(0,3)!=0;
            if((mode_now==40 && !residual_selected) || (mode_now==42 && residual_selected))m_axi_wready=0;
            if(reading && !m_axi_rvalid && $urandom_range(0,3)!=0)begin
                if(read_owner==1 && read_address==addr_model[7])begin
                    require(ffn_committed,"Residual read before FFN B commit");m_axi_rdata=ffn_memory;
                end else m_axi_rdata=pattern(read_address);
                m_axi_rvalid=1;m_axi_rlast=1;m_axi_rresp=0;m_axi_rid=0;
            end
            if(aw_pending && w_pending && !m_axi_bvalid)begin
                b_delay++;
                if(b_delay>=3 && $urandom_range(0,3)!=0)begin m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;end
            end
        end
    end
    always @(posedge clk)begin
        cycle++;
        if(!rst_n)begin held_ar=0;held_aw=0;held_w=0;old_select=0;end
        else begin
            if(busy)require(!command_ready,"Premature public admission");
            if(dut.ffn_command_valid)begin
                require(dut.ffn_command_sequence==seq_model && dut.ffn_command_epoch==epoch_model && dut.ffn_command_source_valid &&
                    dut.ffn_command_source_sequence==seq_model && dut.ffn_command_source_epoch==epoch_model,"FFN captured identity");
                for(integer i=0;i<8;i++)require(dut.ffn_command_addr[i]==addr_model[i] && dut.ffn_command_base[i]==base_model[i] && dut.ffn_command_bytes[i]==size_model[i],$sformatf("FFN captured allocations index=%0d got=%016h/%016h/%016h captured=%016h want=%016h/%016h/%016h",i,dut.ffn_command_addr[i],dut.ffn_command_base[i],dut.ffn_command_bytes[i],dut.address[i],addr_model[i],base_model[i],size_model[i]));
                require(dut.ffn_command_residual_address==residual_model && dut.ffn_command_residual_base==residual_base_model && dut.ffn_command_residual_bytes==12224,"FFN retained allocation");
                if(dut.ffn_command_ready)begin starts[0]++;$display("FFRCTRL_START cycle=%0d job=%0d owner=0 sequence=%016h epoch=%016h",cycle,jobs,dut.ffn_command_sequence,dut.ffn_command_epoch);end
            end
            if(dut.res_command_valid)begin
                require(residual_selected && dut.ffn_completion_valid && !dut.ffn_completion_ready,"Residual before held FFN completion");
                require(dut.res_command_sequence==seq_model && dut.res_command_epoch==epoch_model && dut.res_command_source_valid &&
                    dut.res_command_source_sequence==seq_model && dut.res_command_source_epoch==epoch_model,"Residual captured identity");
                require(dut.res_command_input_address==residual_model && dut.res_command_input_base==residual_model && dut.res_command_input_bytes==8192 &&
                    dut.res_command_branch_address==addr_model[7] && dut.res_command_branch_base==addr_model[7] && dut.res_command_branch_bytes==8192 &&
                    dut.res_command_residual_address==output_model && dut.res_command_residual_base==output_base_model && dut.res_command_residual_bytes==output_size_model,"Residual captured descriptors");
                if(dut.res_command_ready)begin starts[1]++;$display("FFRCTRL_START cycle=%0d job=%0d owner=1 sequence=%016h epoch=%016h",cycle,jobs,dut.res_command_sequence,dut.res_command_epoch);end
            end
            if(dut.ffn_completion_valid && !seen_ffn_done)begin
                seen_ffn_done=1;require(!reading && !aw_pending && !w_pending && !m_axi_rvalid && !m_axi_bvalid,"FFN response before AXI drain");
                $display("FFRCTRL_FFN_DONE cycle=%0d job=%0d status=%0d",cycle,jobs,dut.ffn_completion_status);
            end
            if(residual_selected!=old_select)begin
                require(!reading && !aw_pending && !w_pending && !m_axi_rvalid && !m_axi_bvalid,"AXI switch before complete drain");
                if(residual_selected)require(dut.ffn_completion_valid && !dut.ffn_completion_ready,"Bus acquired without source ownership");
                $display("FFRCTRL_SWITCH cycle=%0d job=%0d owner=%0d",cycle,jobs,residual_selected);old_select=residual_selected;
            end
            if(held_ar)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock}===saved_ar,"Held AR");
            if(held_aw)require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}===saved_aw,"Held AW");
            if(held_w)require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"Held W");
            held_ar=m_axi_arvalid && !m_axi_arready;held_aw=m_axi_awvalid && !m_axi_awready;held_w=m_axi_wvalid && !m_axi_wready;
            saved_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            if(m_axi_arvalid && m_axi_arready)begin
                require(!reading && ar_count[int'(residual_selected)]<2,"AR ownership/extent");
                require(m_axi_arsize==6 && m_axi_arlen==0 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR metadata mux");
                read_owner=int'(residual_selected);read_address=m_axi_araddr;reading=1;
                require(read_address==(read_owner==0 ? addr_model[ar_count[0]]:ar_count[1]==0 ? residual_model:addr_model[7]),"AR address mux/order");
                $display("FFRCTRL_AR cycle=%0d job=%0d owner=%0d index=%0d address=%016h",cycle,jobs,read_owner,ar_count[read_owner],read_address);ar_count[read_owner]++;
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(reading && residual_selected==read_owner && m_axi_rlast && m_axi_rresp==0 && m_axi_rid==0,"R ownership mux");
                require((read_owner==0 ? dut.ffn_m_axi_rvalid && !dut.res_m_axi_rvalid:dut.res_m_axi_rvalid && !dut.ffn_m_axi_rvalid),"Response isolation");
                $display("FFRCTRL_R cycle=%0d job=%0d owner=%0d index=%0d data=%0128h",cycle,jobs,read_owner,r_count[read_owner],m_axi_rdata);
                r_count[read_owner]++;total_reads++;reading=0;m_axi_rvalid<=0;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && aw_count[int'(residual_selected)]==0,"AW extent");write_owner=int'(residual_selected);pending_address=m_axi_awaddr;aw_pending=1;
                require(m_axi_awaddr==(write_owner==0 ? addr_model[7]:output_model) && m_axi_awlen==0 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW metadata/address mux");
                aw_count[write_owner]++;$display("FFRCTRL_AW cycle=%0d job=%0d owner=%0d address=%016h",cycle,jobs,write_owner,pending_address);
            end
            if(m_axi_wvalid && m_axi_wready)begin
                data_owner=int'(residual_selected);require(!w_pending && w_count[data_owner]==0 && m_axi_wlast && m_axi_wstrb=='1,"W metadata mux");
                require(m_axi_wdata===wanted_write(data_owner),"Probe data through AXI mux");
                pending_data=m_axi_wdata;w_pending=1;w_count[data_owner]++;
                $display("FFRCTRL_W cycle=%0d job=%0d owner=%0d data=%0128h",cycle,jobs,data_owner,pending_data);
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending && write_owner==data_owner && write_owner==residual_selected,"B ownership mux");
                require(write_owner==0 ? dut.ffn_m_axi_bvalid && !dut.res_m_axi_bvalid:dut.res_m_axi_bvalid && !dut.ffn_m_axi_bvalid,"B response isolation");
                if(write_owner==0)begin ffn_memory=pending_data;ffn_committed=1;end else begin res_memory=pending_data;res_committed=1;end
                $display("FFRCTRL_B cycle=%0d job=%0d owner=%0d address=%016h data=%0128h",cycle,jobs,write_owner,pending_address,pending_data);
                b_count[write_owner]++;total_writes++;aw_pending=0;w_pending=0;b_delay=0;m_axi_bvalid<=0;
            end
            if(dut.ffn_completion_ready || dut.res_completion_ready)begin
                require(completion_valid && completion_ready && dut.ffn_completion_ready,"Child released before public ACK");
                if(starts[1]!=0)require(dut.res_completion_ready,"FFN released before residual ACK");
                $display("FFRCTRL_RELEASE cycle=%0d job=%0d ffn=%0d residual=%0d",cycle,jobs,dut.ffn_completion_ready,dut.res_completion_ready);
            end
            if(result_valid)require(completion_valid && completion_status==0 && res_committed && ffn_committed && result_address==output_model && result_bytes==8192 && !reading && !aw_pending && !w_pending,"Public result before drain");
            else require(result_address==0 && result_bytes==0,"Private descriptor leak");
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;m_axi_rvalid=0;m_axi_bvalid=0;
        reading=0;aw_pending=0;w_pending=0;seen_ffn_done=0;b_delay=0;resets++;
        dut.ffn.admission_enable=1;dut.res.admission_enable=1;dut.ffn.injected_error=0;dut.res.injected_error=0;
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(!busy && command_ready && !completion_valid && !result_valid && !residual_selected,"Coordinated reset");
        $display("FFRCTRL_RESET cycle=%0d job=%0d",cycle,jobs);
    endtask
    function automatic integer wanted(input integer mode);
        if(mode>=1 && mode<=18 || mode>=20 && mode<=24)return 1;
        if(mode>=25 && mode<=28 || mode==33 || mode==37)return 2;
        if(mode>=29 && mode<=32 || mode==35 || mode==36)return 7;
        if(mode==34 || mode==38)return 3;
        return 0;
    endfunction
    task automatic run_job(input integer mode);
        integer limit,index;logic [3:0] held_status;
        jobs++;mode_now=mode;
        @(negedge clk);require(command_ready && !busy,"Previous job active");
        for(integer p=0;p<2;p++)begin ar_count[p]=0;r_count[p]=0;aw_count[p]=0;w_count[p]=0;b_count[p]=0;starts[p]=0;end
        ffn_committed=0;res_committed=0;ffn_memory=0;res_memory=0;seen_ffn_done=0;
        seq_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        for(integer i=0;i<8;i++)begin
            base_model[i]=(mode==46 ? 64'hff00000000000000:64'h100000000)+64'(i)*64'h20000000+64'(jobs)*64'h1000000000;
            addr_model[i]=base_model[i]+4032;
            size_model[i]=4032+(i==0 ? 4096:i<4 ? 17825792:i<6 ? 65536:i==6 ? 16384:8192);
            command_addr[i]=addr_model[i];command_base[i]=base_model[i];command_bytes[i]=size_model[i];
        end
        residual_base_model=(mode==46 ? 64'hff10000000000000:64'h700000000000)+64'(jobs*65536);residual_model=residual_base_model+4032;
        output_model=(mode==46 ? 64'hff20000000000fc0:64'h710000000fc0)+64'(jobs*65536);
        if(mode==19)output_model=base_model[0]+size_model[0];
        output_base_model=output_model-4032;output_size_model=12224;
        command_sequence=seq_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_source_valid=1;command_source_sequence=seq_model;command_source_epoch=epoch_model;
        command_residual_address=residual_model;command_residual_base=residual_base_model;command_residual_bytes=12224;
        command_output_address=output_model;command_output_base=output_base_model;command_output_bytes=output_size_model;
        dut.ffn.admission_enable=1;dut.res.admission_enable=1;dut.ffn.injected_error=mode==34 ? 3:0;dut.res.injected_error=mode==38 ? 3:0;
        completion_ready=0;cancel_valid=0;
        if(mode>=1 && mode<=18)begin
            index=(mode-1)%9;
            command_output_address=index==8 ? residual_base_model:base_model[index];
            if(mode>=10)command_output_address+=(index==8 ? 12224:size_model[index])-64;
            command_output_base=command_output_address;command_output_bytes=8192;
        end
        case(mode)
            20:begin command_output_base=output_model;command_output_bytes=8191;end
            21:command_output_address++;
            22:command_output_base=output_model+64;
            23:begin command_output_address=64'hffffffffffffc000;command_output_base=command_output_address;command_output_bytes=65536;end
            24:begin command_output_address=64'hfffffffffffff000;command_output_base=0;command_output_bytes=64'hffffffffffffffff;end
            25:command_source_sequence^=64'd1<<40;
            26:command_source_epoch^=64'd1<<44;
            27:expected_epoch^=64'd1<<48;
            28:command_source_valid=0;
            29:cancel_valid=1;
            default:;
        endcase
        $display("FFRCTRL_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h output=%016h",cycle,jobs,mode,seq_model,epoch_model,output_model);
        command_valid=1;tick();@(negedge clk);command_valid=0;cancel_valid=0;
        command_sequence=0;command_epoch=0;command_source_sequence=0;command_source_epoch=0;command_source_valid=0;
        command_output_address=0;command_output_base=0;command_output_bytes=0;command_residual_address=0;command_residual_base=0;command_residual_bytes=0;
        for(integer i=0;i<8;i++)begin command_addr[i]=0;command_base[i]=0;command_bytes[i]=0;end
        if(mode==27)expected_epoch=epoch_model;
        if(mode==30)begin repeat(4)tick();@(negedge clk);cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if(mode==31)begin
            dut.ffn.admission_enable=0;wait(dut.ffn_command_valid);@(negedge clk);cancel_valid=1;
            repeat(4)begin tick();require(dut.ffn_command_valid && starts[0]==0,"FFN offer withdrawn");end
            @(negedge clk);cancel_valid=0;dut.ffn.admission_enable=1;
        end
        if(mode==32 || mode==39)begin wait(ar_count[0]>=1);@(negedge clk);end
        if(mode==33)begin wait(aw_count[0]>=1);@(negedge clk);end
        if(mode==35)begin
            wait(dut.handoff_busy);@(negedge clk);dut.res.admission_enable=0;
            wait(dut.res_command_valid);@(negedge clk);cancel_valid=1;
            repeat(4)begin tick();require(dut.res_command_valid && starts[1]==0,"Residual offer withdrawn");end
            @(negedge clk);cancel_valid=0;dut.res.admission_enable=1;
        end
        if(mode==36 || mode==41)begin wait(ar_count[1]>=1);@(negedge clk);end
        if(mode==37)begin wait(aw_count[1]>=1);@(negedge clk);end
        if(mode==40)begin wait(aw_count[0]==1 && w_count[0]==0);@(negedge clk);end
        if(mode==42)begin wait(aw_count[1]==1 && w_count[1]==0);@(negedge clk);end
        if(mode==32 || mode==36)begin cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        if(mode==33 || mode==37)begin expected_epoch^=64'd1<<48;tick();@(negedge clk);expected_epoch=epoch_model;end
        if(mode==47)begin wait(starts[0]==1);@(negedge clk);command_valid=1;repeat(4)tick();@(negedge clk);command_valid=0;end
        if(mode>=39 && mode<=42)begin $display("FFRCTRL_ABORT cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);reset_all();end
        else begin
            limit=0;while(!completion_valid)begin tick();limit++;require(limit<500,"Completion missing");end
            require(completion_status==wanted(mode) && completion_sequence==seq_model && completion_epoch==epoch_model,"Final identity/status");
            require(result_valid==(wanted(mode)==0),"Final result visibility");
            if(mode>=1 && mode<=18 || mode>=20 && mode<=30)require(starts[0]==0 && starts[1]==0 && total_reads>=0,"Rejected request launched traffic");
            if(result_valid)offered++;
            publics++;held_status=completion_status;
            $display("FFRCTRL_COMPLETE cycle=%0d job=%0d status=%0d ffn=%0d residual=%0d valid=%0d reads=%0d writes=%0d",cycle,jobs,completion_status,completion_ffn_status,completion_residual_status,result_valid,r_count[0]+r_count[1],b_count[0]+b_count[1]);
            repeat(5)begin tick();require(completion_valid && completion_status==held_status && !command_ready && !dut.ffn_completion_ready && !dut.res_completion_ready,"Held final ownership");end
            if(mode==43 || mode==44)begin
                @(negedge clk);if(mode==43)expected_epoch^=64'd1<<48;else cancel_valid=1;
                #1;require(!result_valid,"Immediate late revocation");tick();@(negedge clk);expected_epoch=epoch_model;cancel_valid=0;
                repeat(3)begin tick();require(!result_valid && completion_status==held_status && completion_valid,"Sticky final revocation");end
                $display("FFRCTRL_REVOKE cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
            end
            if(mode==45)begin $display("FFRCTRL_ABORT cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);reset_all();end
            else begin
                @(negedge clk);completion_ready=1;tick();@(negedge clk);completion_ready=0;
                repeat(3)tick();require(command_ready && !busy,"Final release/reuse");
                $display("FFRCTRL_ACK cycle=%0d job=%0d",cycle,jobs);
            end
        end
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;cancel_valid=0;completion_ready=0;m_axi_rvalid=0;m_axi_bvalid=0;
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        reset_all();repeat(2)run_job(0);
        for(integer mode=1;mode<=47;mode++)begin run_job(mode);run_job(0);end
        require(jobs==96 && publics==92 && offered==55 && resets==6,"Plan cardinality");
        $display("FFRCTRL_PASS seed=%0d jobs=%0d publics=%0d offered=%0d resets=%0d reads=%0d writes=%0d cycles=%0d",seed,jobs,publics,offered,resets,total_reads,total_writes,cycle);
        $finish;
    end
    initial begin #2000000;$fatal(1,"FFRCTRL_FAIL absolute timeout");end
endmodule
`default_nettype wire
