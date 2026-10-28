`timescale 1ns/1ps
`default_nettype none
// Protocol peers only: arithmetic and complete Q/K/V processing are separate gates.
module tb_ntrpu_llama32_attention_ffn_control_r4;
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
    logic [13:0] result_bytes;
    logic [63:0] command_weight_addr;
    logic [63:0] command_weight_region_base;
    logic [63:0] command_weight_region_bytes;
    logic [63:0] command_activation_addr;
    logic [63:0] command_activation_region_base;
    logic [63:0] command_activation_region_bytes;
    logic [63:0] command_output_addr;
    logic [63:0] command_output_region_base;
    logic [63:0] command_output_region_bytes;
    logic [63:0] command_post_input_address;
    logic [63:0] command_post_input_base;
    logic [63:0] command_post_input_bytes;
    logic [63:0] command_post_weight_address;
    logic [63:0] command_post_weight_base;
    logic [63:0] command_post_weight_bytes;
    logic [63:0] command_post_residual_address;
    logic [63:0] command_post_residual_base;
    logic [63:0] command_post_residual_bytes;
    logic [63:0] command_post_ffn_address;
    logic [63:0] command_post_ffn_base;
    logic [63:0] command_post_ffn_bytes;
    logic [31:0] completion_post_hls_status;
    logic [9:0] completion_post_read_beats;
    logic [7:0] completion_residual_beats;
    logic [6:0] completion_ffn_beats;
    logic [3:0] completion_ffn_status;
    logic [3:0] completion_ffn_child_status;
    logic [3:0] completion_residual_status;
    logic [2:0] completion_ffn_stages;
    logic [1:0] completion_ffn_phase;
    logic [16:0] completion_ffn_bytes;
    logic [17:0] completion_ffn_stage_bytes;
    logic [31:0] completion_ffn_hls_status;
    logic [9:0] completion_read_beats;
    logic [7:0] completion_write_beats;
    logic activation_selected;
    logic residual_selected;
    logic [3:0] completion_attention_status;
    logic [3:0] completion_ffn_residual_status;
    logic [31:0] completion_attention_hls_status;
    logic ffn_selected;
    wire [63:0] command_ffn_addr [1:7];
    wire [63:0] command_ffn_base [1:7];
    wire [63:0] command_ffn_bytes [1:7];
    logic [63:0] command_final_address;
    logic [63:0] command_final_base;
    logic [63:0] command_final_bytes;
    logic [15:0][63:0] drive_addr,drive_base,drive_bytes,model_addr,model_base,model_bytes;
    for(genvar i=1;i<8;i++)begin
        assign command_ffn_addr[i]=drive_addr[i+7];assign command_ffn_base[i]=drive_base[i+7];assign command_ffn_bytes[i]=drive_bytes[i+7];
    end
    assign command_weight_addr=drive_addr[1];
    assign command_weight_region_base=drive_base[1];
    assign command_weight_region_bytes=drive_bytes[1];
    assign command_activation_addr=drive_addr[2];
    assign command_activation_region_base=drive_base[2];
    assign command_activation_region_bytes=drive_bytes[2];
    assign command_output_addr=drive_addr[3];
    assign command_output_region_base=drive_base[3];
    assign command_output_region_bytes=drive_bytes[3];
    assign command_post_input_address=drive_addr[4];
    assign command_post_input_base=drive_base[4];
    assign command_post_input_bytes=drive_bytes[4];
    assign command_post_weight_address=drive_addr[5];
    assign command_post_weight_base=drive_base[5];
    assign command_post_weight_bytes=drive_bytes[5];
    assign command_post_residual_address=drive_addr[6];
    assign command_post_residual_base=drive_base[6];
    assign command_post_residual_bytes=drive_bytes[6];
    assign command_post_ffn_address=drive_addr[7];
    assign command_post_ffn_base=drive_base[7];
    assign command_post_ffn_bytes=drive_bytes[7];
    assign command_final_address=drive_addr[15];
    assign command_final_base=drive_base[15];
    assign command_final_bytes=drive_bytes[15];

    ntrpu_llama32_attention_ffn_r4 dut(.*);
    always #2.5 clk=~clk;
    integer seed=1,random_state=1,cycle=0,jobs=0,mode_now=0,publics=0,offered=0,resets=0,total_reads=0,total_writes=0,source_acks=0,child_acks=0;
    logic [1:0][31:0] ar_count,r_count,aw_count,w_count,b_count,starts;
    logic [63:0] seq_model,epoch_model,read_address,pending_address;
    logic [511:0] pending_data,source_memory,result_memory;
    logic reading=0,aw_pending=0,w_pending=0,source_committed=0,result_committed=0,old_select=0;
    logic held_ar=0,held_aw=0,held_w=0;
    logic [88:0] saved_ar,saved_aw;
    logic [576:0] saved_w;
    integer read_owner=0,write_owner=0,data_owner=0,b_delay=0;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"AFC_FAIL cycle=%0d job=%0d mode=%0d: %s",cycle,jobs,mode_now,why);
    endtask
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [511:0] pattern(input logic [63:0] addr);
        logic [511:0] value;
        for(integer i=0;i<8;i++)value[i*64+:64]=addr^(64'h13579bdf2468ace0+64'(i));
        return value;
    endfunction
    function automatic logic [511:0] expected_write(input integer owner);
        if(owner==0)return pattern(model_addr[4])^pattern(model_addr[1])^{16{32'h46}};
        return source_memory^pattern(model_addr[6])^{16{32'h52}};
    endfunction
    function automatic logic [63:0] extent(input integer i);
        case(i)
            0:return 65536;
            1:return 4456448;
            2,7:return 4096;
            8,9,10:return 17825792;
            11,12:return 65536;
            13:return 16384;
            default:return 8192;
        endcase
    endfunction
    function automatic bit early_error(input integer mode);
        return mode>=1 && mode<=14 || mode>=100 && mode<=219 || mode>=340 && mode<=555;
    endfunction
    function automatic bit no_ffn(input integer mode);
        return early_error(mode) || mode==15 || mode==16 || mode==18 || mode>=20 && mode<=23 ||
            mode==26 || mode==27 || mode==37 || mode==40 || mode==42 || mode==43 || mode==50 || mode>=52 && mode<=55 || mode==58;
    endfunction
    function automatic integer expected_status(input integer mode);
        if(mode>=100 && mode<=219 || mode>=340 && mode<=555)return 1;
        case(mode)
            1,2,3,5,13,27,29,43,45,53:return 2;
            4,12,26,28,42,44,52:return 7;
            6,14,18,19,50,51,54,55:return 11;
            7,8,9,10,11:return 1;
            16,17:return 3;
            20,21,22,23,24,25,58,59:return 5;
            default:return 0;
        endcase
    endfunction
    always @(negedge clk)begin
        if(!rst_n)begin m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;m_axi_rvalid=0;m_axi_bvalid=0;end
        else begin
            m_axi_arready=!reading && $urandom_range(0,3)!=0;
            m_axi_awready=!aw_pending && $urandom_range(0,3)!=0;
            m_axi_wready=!w_pending && $urandom_range(0,3)!=0;
            if(reading && !m_axi_rvalid && !(mode_now==37 && read_owner==0) && !(mode_now==38 && read_owner==1) && $urandom_range(0,3)!=0)begin
                if(read_owner==1 && read_address==model_addr[7])begin require(source_committed,"FFN read before source B");m_axi_rdata=source_memory;end
                else m_axi_rdata=pattern(read_address);
                m_axi_rvalid=1;m_axi_rlast=1;m_axi_rresp=0;m_axi_rid=0;
            end
            if(aw_pending && w_pending && !m_axi_bvalid)begin
                b_delay=b_delay+1;
                if(b_delay>=3 && $urandom_range(0,3)!=0)begin m_axi_bvalid=1;m_axi_bresp=0;m_axi_bid=0;end
            end
        end
    end
    always @(posedge clk)begin
        cycle=cycle+1;
        if(!rst_n)begin reading=0;aw_pending=0;w_pending=0;held_ar=0;held_aw=0;held_w=0;old_select=0;b_delay=0;end
        else begin
            if(busy)begin
                require(!command_ready && cache_lock,"Busy ownership");
                require(!bind_ready && !append_ready && !data_ready && !commit_ready && !append_cancel_ready,"Maintenance ready during ownership");
                require(!dut.a_bind_valid && !dut.a_append_valid && !dut.a_data_valid && !dut.a_commit_valid && !dut.a_append_cancel_valid && !dut.a_append_done_ready,"Maintenance reached owned cache");
            end
            if(!dut.attention_launched || ffn_selected)require(!dut.a_q_valid && !q_ready,"Q stream reached unowned attention");
            if(dut.a_command_valid)begin
                require(!early_error(mode_now),"Invalid allocation reached attention");
                require(dut.a_command_sequence==seq_model && dut.a_command_epoch==epoch_model && dut.a_command_layer==0 && dut.a_command_lease_base==64'hfedc000000000100,"Captured attention identity");
                require(dut.a_command_weight_addr==model_addr[1] && dut.a_command_activation_addr==model_addr[2] && dut.a_command_output_addr==model_addr[3] && dut.a_command_post_input_address==model_addr[4] && dut.a_command_post_weight_address==model_addr[5] && dut.a_command_post_residual_address==model_addr[6] && dut.a_command_post_ffn_address==model_addr[7],"Captured attention addresses");
                if(dut.a_command_ready)begin starts[0]=starts[0]+1;$display("AFC_START cycle=%0d job=%0d owner=0 sequence=%016h epoch=%016h",cycle,jobs,dut.a_command_sequence,dut.a_command_epoch);end
            end
            if(dut.f_command_valid)begin
                require(!no_ffn(mode_now),"Rejected work reached FFN");
                require(ffn_selected && dut.a_completion_valid && !dut.a_completion_ready,"FFN started without retained source");
                require(dut.f_command_sequence==seq_model && dut.f_command_epoch==epoch_model && dut.f_command_source_sequence==seq_model && dut.f_command_source_epoch==epoch_model && dut.f_command_source_valid,"Captured FFN identity");
                require(dut.f_command_addr[0]==model_addr[7] && dut.f_command_base[0]==model_addr[7] && dut.f_command_bytes[0]==4096,"Live ABI1 descriptor");
                for(integer i=1;i<8;i++)require(dut.f_command_addr[i]==model_addr[i+7] && dut.f_command_base[i]==model_base[i+7] && dut.f_command_bytes[i]==model_bytes[i+7],"Captured FFN arrays");
                require(dut.f_command_residual_address==model_addr[6] && dut.f_command_residual_base==model_addr[6] && dut.f_command_residual_bytes==8192 && dut.f_command_output_address==model_addr[15] && dut.f_command_output_base==model_base[15] && dut.f_command_output_bytes==model_bytes[15],"Captured residual/final descriptors");
                if(dut.f_command_ready)begin starts[1]=starts[1]+1;$display("AFC_START cycle=%0d job=%0d owner=1 sequence=%016h epoch=%016h",cycle,jobs,dut.f_command_sequence,dut.f_command_epoch);end
            end
            if(ffn_selected!=old_select)begin
                require(!reading && !aw_pending && !w_pending && !m_axi_bvalid && !m_axi_rvalid,"AXI owner changed before drain");
                $display("AFC_SWITCH cycle=%0d job=%0d owner=%0d",cycle,jobs,ffn_selected);old_select=ffn_selected;
            end
            if(ffn_selected)require(!dut.a_m_axi_arready && !dut.a_m_axi_awready && !dut.a_m_axi_wready && !dut.a_m_axi_rvalid && !dut.a_m_axi_bvalid,"Attention sees FFN bus");
            else require(!dut.f_m_axi_arready && !dut.f_m_axi_awready && !dut.f_m_axi_wready && !dut.f_m_axi_rvalid && !dut.f_m_axi_bvalid,"FFN sees attention bus");
            if(held_ar)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock}==saved_ar,"Unstable AR");
            if(held_aw)require(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}==saved_aw,"Unstable AW");
            if(held_w)require(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}==saved_w,"Unstable W");
            held_ar=m_axi_arvalid && !m_axi_arready;saved_ar={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock};
            held_aw=m_axi_awvalid && !m_axi_awready;saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            held_w=m_axi_wvalid && !m_axi_wready;saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            if(m_axi_arvalid && m_axi_arready)begin
                read_owner=ffn_selected;read_address=m_axi_araddr;
                require(!reading && starts[read_owner]==1 && ar_count[read_owner]<2,"AR ownership/extent");
                require(m_axi_araddr==(read_owner==0 ? (ar_count[0]==0 ? model_addr[4]:model_addr[1]):(ar_count[1]==0 ? model_addr[7]:model_addr[6])) && m_axi_arlen==0 && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0,"AR descriptor");
                reading=1;ar_count[read_owner]=ar_count[read_owner]+1;
                $display("AFC_AR cycle=%0d job=%0d owner=%0d address=%016h",cycle,jobs,read_owner,read_address);
            end
            if(m_axi_rvalid && m_axi_rready)begin
                require(reading && read_owner==ffn_selected,"R ownership");
                r_count[read_owner]=r_count[read_owner]+1;total_reads=total_reads+1;reading=0;m_axi_rvalid<=0;
                $display("AFC_R cycle=%0d job=%0d owner=%0d address=%016h data=%0128h",cycle,jobs,read_owner,read_address,m_axi_rdata);
            end
            if(m_axi_awvalid && m_axi_awready)begin
                write_owner=ffn_selected;pending_address=m_axi_awaddr;
                require(!aw_pending && aw_count[write_owner]==0 && m_axi_awaddr==(write_owner==0 ? model_addr[7]:model_addr[15]) && m_axi_awlen==0 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0,"AW descriptor");
                aw_pending=1;aw_count[write_owner]=aw_count[write_owner]+1;
                $display("AFC_AW cycle=%0d job=%0d owner=%0d address=%016h",cycle,jobs,write_owner,pending_address);
            end
            if(m_axi_wvalid && m_axi_wready)begin
                data_owner=ffn_selected;pending_data=m_axi_wdata;
                require(!w_pending && w_count[data_owner]==0 && m_axi_wstrb=='1 && m_axi_wlast && m_axi_wdata==expected_write(data_owner),"W payload/owner");
                w_pending=1;w_count[data_owner]=w_count[data_owner]+1;
                $display("AFC_W cycle=%0d job=%0d owner=%0d data=%0128h",cycle,jobs,data_owner,pending_data);
            end
            if(m_axi_bvalid && m_axi_bready)begin
                require(aw_pending && w_pending && write_owner==data_owner && write_owner==ffn_selected,"B ownership");
                if(write_owner==0)begin source_memory=pending_data;source_committed=1;end
                else begin result_memory=pending_data;result_committed=1;end
                aw_pending=0;w_pending=0;b_delay=0;m_axi_bvalid<=0;b_count[write_owner]=b_count[write_owner]+1;total_writes=total_writes+1;
                $display("AFC_B cycle=%0d job=%0d owner=%0d address=%016h",cycle,jobs,write_owner,pending_address);
            end
            if(dut.a_completion_ready)begin
                require(completion_valid && completion_ready && dut.a_completion_valid && dut.a_completion_sequence==seq_model && dut.a_completion_epoch==epoch_model,"Early/foreign source release");
                if(starts[1])require(dut.f_completion_ready,"Non-atomic child release");
                source_acks=source_acks+1;$display("AFC_RELEASE cycle=%0d job=%0d owner=0",cycle,jobs);
            end
            if(dut.f_completion_ready)begin
                require(dut.a_completion_ready && completion_valid && completion_ready && dut.f_completion_valid && dut.f_completion_sequence==seq_model && dut.f_completion_epoch==epoch_model,"Early/foreign FFN release");
                child_acks=child_acks+1;$display("AFC_RELEASE cycle=%0d job=%0d owner=1",cycle,jobs);
            end
            if(result_valid)require(completion_valid && completion_status==0 && source_committed && result_committed && result_address==model_addr[15] && result_bytes==8192 && !cancel_valid && expected_epoch==epoch_model,"Invalid publication");
            else require(result_address==0 && result_bytes==0,"Leaked descriptor");
        end
    end
    task automatic clear_injections;
        dut.attention.admission_enable=1;dut.ffn.admission_enable=1;
        dut.attention.injected_error=0;dut.ffn.injected_error=0;
        dut.attention.response_sequence_mask=0;dut.ffn.response_sequence_mask=0;
        dut.attention.shape_fault=0;dut.ffn.shape_fault=0;
        dut.attention.mute_result=0;dut.ffn.mute_result=0;dut.attention.mute_completion=0;dut.ffn.mute_completion=0;
        dut.attention.cache_invalid=0;dut.attention.metadata_injected=0;dut.attention.cache_base_mask=0;
    endtask
    task automatic reset_all;
        @(negedge clk);rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;bind_valid=0;append_valid=0;data_valid=0;commit_valid=0;append_cancel_valid=0;append_done_ready=0;q_valid=0;
        clear_injections();resets=resets+1;repeat(4)tick();@(negedge clk);rst_n=1;tick();
        require(!busy && !completion_valid && !result_valid && !ffn_selected,"Reset ownership");
        $display("AFC_RESET cycle=%0d job=%0d",cycle,jobs);
    endtask
    task automatic run_job(input integer mode);
        integer limit,pair_id,left,right,cursor,idx,variant,source_before,child_before;
        logic [3:0] saved_status;logic [63:0] root_base;
        jobs=jobs+1;mode_now=mode;@(negedge clk);require(!busy,"Previous job held");clear_injections();
        seq_model=64'h1234567800000000+64'(jobs);epoch_model=64'hfedcba9800000000+64'(jobs);
        root_base=mode==56 ? 64'hffff000000000000:64'h100000000000;
        for(integer i=0;i<16;i++)begin
            drive_base[i]=root_base+64'(i)*64'h10000000+64'(jobs)*64'h100000;
            drive_addr[i]=drive_base[i]+4032;drive_bytes[i]=extent(i)+4032;
        end
        if(mode>=100 && mode<=339 || mode>=436 && mode<=555)begin
            pair_id=mode>=436 ? mode-436:mode>=220 ? mode-220:mode-100;cursor=0;left=-1;right=-1;
            for(integer l=0;l<15;l++)for(integer r=l+1;r<16;r++)begin if(cursor==pair_id)begin left=l;right=r;end cursor=cursor+1;end
            require(left>=0 && right>left,"Pair plan");
            drive_base[right]=drive_base[left]+(mode>=220 && mode<=339 ? drive_bytes[left]:mode>=436 ? drive_bytes[left]-64:64'd0);
            drive_addr[right]=drive_base[right]+4032;
        end
        if(mode>=340 && mode<=435)begin
            idx=(mode-340)/6;variant=(mode-340)%6;
            case(variant)
                0:drive_addr[idx]=drive_addr[idx]+1;
                1:drive_bytes[idx]=0;
                2:begin drive_base[idx]=drive_addr[idx];drive_bytes[idx]=idx==0 ? 0:extent(idx)-1;end
                3:drive_base[idx]=drive_addr[idx]+64;
                4:begin drive_base[idx]=64'hfffffffffffff000;drive_bytes[idx]=8192;end
                5:begin drive_addr[idx]=64'hffffffffffffffc0;drive_base[idx]=0;drive_bytes[idx]=64'hffffffffffffffff;end
                default:;
            endcase
        end
        model_addr=drive_addr;model_base=drive_base;model_bytes=drive_bytes;
        expected_epoch=epoch_model;command_sequence=seq_model;command_epoch=epoch_model;command_layer=0;command_position=0;command_lease_base=64'hfedc000000000100;
        bind_cache_base=drive_addr[0];bind_region_base=drive_base[0];bind_region_bytes=drive_bytes[0];bind_capacity=mode==8 ? 0:mode==9 ? 131073:1;bind_sequence=seq_model;bind_epoch=epoch_model;
        append_layer=0;append_position=0;append_sequence=seq_model;append_epoch=epoch_model;
        data_bits=0;data_index=0;data_last=0;data_sequence=seq_model;data_epoch=epoch_model;
        commit_status=0;commit_sequence=seq_model;commit_epoch=epoch_model;
        q_data=pattern(seq_model);q_index=0;q_last=0;q_sequence=seq_model;q_epoch=epoch_model;
        starts=0;ar_count=0;r_count=0;aw_count=0;w_count=0;b_count=0;source_committed=0;result_committed=0;
        source_before=source_acks;child_before=child_acks;
        $display("AFC_BEGIN cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h",cycle,jobs,mode,seq_model,epoch_model);
        for(integer i=0;i<16;i++)$display("AFC_REGION cycle=%0d job=%0d index=%0d address=%016h base=%016h bytes=%0d",cycle,jobs,i,model_addr[i],model_base[i],model_bytes[i]);
        bind_valid=1;if(mode==57)command_valid=1;#1;require(bind_ready && !command_ready,"Bind admission priority");tick();@(negedge clk);bind_valid=0;command_valid=0;append_done_ready=1;tick();@(negedge clk);append_done_ready=0;
        case(mode)
            1:command_sequence^=64'd1<<40;
            2:command_epoch^=64'd1<<44;
            3:expected_epoch^=64'd1<<48;
            4:cancel_valid=1;
            5:dut.attention.cache_invalid=1;
            6:dut.attention.metadata_injected=1;
            7:command_layer=16;
            10:command_lease_base=0;
            11:command_lease_base=64'hffffffffffffffe1;
            16:dut.attention.injected_error=3;
            17:dut.ffn.injected_error=3;
            18:dut.attention.response_sequence_mask=64'd1<<40;
            19:dut.ffn.response_sequence_mask=64'd1<<40;
            20,21,22,23:dut.attention.shape_fault=mode-19;
            24,25:dut.ffn.shape_fault=mode-23;
            58:dut.attention.injected_error=5;
            59:dut.ffn.injected_error=5;
            default:;
        endcase
        #1;require(command_ready,"Command admission");command_valid=1;tick();
        $display("AFC_ADMIT cycle=%0d job=%0d",cycle,jobs);
        @(negedge clk);command_valid=0;cancel_valid=0;expected_epoch=epoch_model;drive_addr=0;drive_base=0;drive_bytes=0;
        command_sequence=0;command_epoch=0;command_layer=15;command_position=999;command_lease_base=0;
        bind_valid=1;append_valid=1;data_valid=1;commit_valid=1;append_cancel_valid=1;append_done_ready=1;q_valid=1;
        if(mode==40 || mode==42 || mode==43)dut.attention.admission_enable=0;
        if(mode>=12 && mode<=15)begin
            repeat(3)tick();@(negedge clk);
            if(mode==12)cancel_valid=1;
            if(mode==13)expected_epoch^=64'd1<<48;
            if(mode==14)dut.attention.cache_base_mask=64;
            if(mode!=15)begin tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;dut.attention.cache_base_mask=0;$display("AFC_STIM cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);end
        end
        if(mode==40 || mode==42 || mode==43)begin
            limit=0;while(!dut.a_command_valid)begin tick();limit++;require(limit<250,"Attention offer missing");end
            if(mode!=40)begin
                @(negedge clk);if(mode==42)cancel_valid=1;else expected_epoch^=64'd1<<48;
                repeat(4)begin tick();require(dut.a_command_valid,"Withdrawn attention offer");end
                @(negedge clk);cancel_valid=0;expected_epoch=epoch_model;dut.attention.admission_enable=1;
                $display("AFC_STIM cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
            end
        end
        if(mode==41 || mode==44 || mode==45 || mode>=52 && mode<=55)begin
            limit=0;while(!dut.h_busy)begin tick();limit++;require(limit<400,"Handoff admission missing");end
            @(negedge clk);
            if(mode>=52)begin
                if(mode==52)cancel_valid=1;
                if(mode==53)expected_epoch^=64'd1<<48;
                if(mode==54)dut.attention.response_sequence_mask=64'd1<<40;
                if(mode==55)dut.attention.shape_fault=1;
                repeat(2)tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;dut.attention.response_sequence_mask=0;dut.attention.shape_fault=0;
                $display("AFC_STIM cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
            end else begin
                dut.ffn.admission_enable=0;limit=0;while(!dut.f_command_valid)begin tick();limit++;require(limit<100,"FFN offer missing");end
                if(mode!=41)begin
                    @(negedge clk);if(mode==44)cancel_valid=1;else expected_epoch^=64'd1<<48;
                    repeat(4)begin tick();require(dut.f_command_valid,"Withdrawn FFN offer");end
                    @(negedge clk);cancel_valid=0;expected_epoch=epoch_model;dut.ffn.admission_enable=1;
                    $display("AFC_STIM cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
                end
            end
        end
        if(mode==26 || mode==27 || mode==28 || mode==29 || mode==37 || mode==38 || mode==50 || mode==51)begin
            idx=mode==28 || mode==29 || mode==38 || mode==51 ? 1:0;
            limit=0;while(ar_count[idx]==0)begin tick();limit++;require(limit<500,"Peer AR missing");end
            @(negedge clk);
            if(mode==26 || mode==28)cancel_valid=1;
            if(mode==27 || mode==29)expected_epoch^=64'd1<<48;
            if(mode==50 || mode==51)dut.attention.cache_base_mask=64;
            if(mode!=37 && mode!=38)begin tick();@(negedge clk);cancel_valid=0;expected_epoch=epoch_model;dut.attention.cache_base_mask=0;$display("AFC_STIM cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);end
        end
        if(mode==18 || mode==19)begin
            limit=0;while(!(mode==18 ? dut.a_completion_valid:dut.f_completion_valid))begin tick();limit++;require(limit<500,"Foreign completion missing");end
            repeat(4)begin tick();require(!completion_valid && source_acks==source_before && child_acks==child_before,"Foreign completion released");end
            @(negedge clk);dut.attention.response_sequence_mask=0;dut.ffn.response_sequence_mask=0;
            $display("AFC_RESTORE cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
        end
        if(mode==15 || mode==37 || mode==38 || mode==40 || mode==41)begin
            repeat(4)begin tick();require(busy && !completion_valid && source_acks==source_before && child_acks==child_before,"Premature abort release");end
            $display("AFC_ABORT cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);reset_all();
        end else begin
            limit=0;while(!completion_valid)begin tick();limit++;require(limit<600,"Public completion missing");end
            require(completion_status==expected_status(mode),"Completion status");
            require(completion_sequence==(mode==1 ? seq_model^(64'd1<<40):seq_model) && completion_epoch==(mode==2 ? epoch_model^(64'd1<<44):epoch_model),"Completion identity");
            require(result_valid==(expected_status(mode)==0),"Publication validity");
            if(!early_error(mode))require(starts[0]==1,"Valid command rejected before attention");
            if(!no_ffn(mode))require(starts[1]==1,"Valid command rejected before FFN");
            require(source_acks==source_before && child_acks==child_before,"Peers released before public ACK");
            publics=publics+1;if(result_valid)offered=offered+1;saved_status=completion_status;
            $display("AFC_COMPLETE cycle=%0d job=%0d status=%0d valid=%0d sequence=%016h epoch=%016h address=%016h bytes=%0d attention=%0d ffn=%0d",cycle,jobs,completion_status,result_valid,completion_sequence,completion_epoch,result_address,result_bytes,completion_attention_status,completion_ffn_residual_status);
            repeat(5)begin tick();require(completion_valid && completion_status==saved_status && !command_ready && source_acks==source_before && child_acks==child_before,"Held completion");end
            if(mode>=30 && mode<=36 || mode>=46 && mode<=49 || mode==60)begin
                @(negedge clk);
                case(mode)
                    30,60:cancel_valid=1;
                    31:expected_epoch^=64'd1<<48;
                    32:dut.attention.cache_base_mask=64;
                    33:begin dut.attention.response_sequence_mask=64'd1<<40;completion_ready=1;end
                    34:begin dut.ffn.response_sequence_mask=64'd1<<40;completion_ready=1;end
                    35:dut.ffn.mute_result=1;
                    36:dut.attention.mute_result=1;
                    46:dut.attention.mute_completion=1;
                    47:dut.ffn.mute_completion=1;
                    48:dut.attention.shape_fault=3;
                    49:dut.ffn.shape_fault=1;
                    default:;
                endcase
                #1;require(!result_valid,"Immediate held revocation");
                repeat(3)begin tick();require(busy && source_acks==source_before && child_acks==child_before,"Foreign/held release gate");end
                @(negedge clk);completion_ready=0;cancel_valid=0;expected_epoch=epoch_model;clear_injections();
                repeat(3)begin tick();require(completion_valid && completion_status==saved_status && !result_valid,"Sticky held revocation");end
                $display("AFC_REVOKE cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);
            end
            if(mode==39)begin $display("AFC_ABORT cycle=%0d job=%0d mode=%0d",cycle,jobs,mode);reset_all();end
            else begin
                @(negedge clk);completion_ready=1;tick();
                require(source_acks-source_before==starts[0] && child_acks-child_before==starts[1],"Atomic release extent");
                $display("AFC_ACK cycle=%0d job=%0d source=%0d child=%0d",cycle,jobs,source_acks-source_before,child_acks-child_before);
                @(negedge clk);completion_ready=0;bind_valid=0;append_valid=0;data_valid=0;commit_valid=0;append_cancel_valid=0;append_done_ready=0;q_valid=0;
                limit=0;while(busy)begin tick();limit++;require(limit<15,"Ownership cleanup missing");end
                require(!ffn_selected && !result_valid && !completion_valid,"Ownership cleanup");
                $display("AFC_IDLE cycle=%0d job=%0d",cycle,jobs);
            end
        end
    endtask
    initial begin
        clk=0;rst_n=0;command_valid=0;completion_ready=0;bind_valid=0;append_valid=0;data_valid=0;commit_valid=0;append_cancel_valid=0;append_done_ready=0;cancel_valid=0;q_valid=0;
        m_axi_rvalid=0;m_axi_bvalid=0;
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        reset_all();repeat(2)run_job(0);
        for(integer mode=1;mode<=60;mode++)begin run_job(mode);run_job(0);end
        for(integer mode=100;mode<=555;mode++)begin run_job(mode);run_job(0);end
        require(jobs==1034,"Plan count");
        $display("AFC_PASS seed=%0d jobs=%0d publics=%0d offered=%0d resets=%0d reads=%0d writes=%0d source_acks=%0d child_acks=%0d cycles=%0d",seed,jobs,publics,offered,resets,total_reads,total_writes,source_acks,child_acks,cycle);
        $finish;
    end
    initial begin #10000000;$fatal(1,"AFC_FAIL absolute timeout");end
endmodule
`default_nettype wire
