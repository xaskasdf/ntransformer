`timescale 1ps/1ps
`default_nettype none
// Native PIPE fixture; source reader and attributed ACK remain test stimuli.
module ntrpu_axi_bridge_isolation;
    wire clk=board.AXI_PCIE_EP.axi_aclk_out;
    reg configuration_rst_n=0,mgmt_clk=0;
    always #5000 mgmt_clk=~mgmt_clk;
    wire cold_rst_n,link_up,stop,guard_fault,guard_alive,guard_armed;
    wire [1:0] guard_code;
    wire native_available=board.AXI_PCIE_EP.user_lnk_up && board.AXI_PCIE_EP.axi_aresetn && board.AXI_PCIE_EP.axi_ctl_aresetn;
    ntrpu_axi_bridge_guard guard(.mgmt_clk(mgmt_clk),.axi_clk(clk),.cold_rst_n(configuration_rst_n),
        .native_link_up(board.AXI_PCIE_EP.user_lnk_up),.native_axi_reset_n(board.AXI_PCIE_EP.axi_aresetn),
        .native_ctl_reset_n(board.AXI_PCIE_EP.axi_ctl_aresetn),.perst_n(board.sys_rst_n),
        .client_cold_rst_n(cold_rst_n),.client_link_up(link_up),.client_stop(stop),
        .clock_alive(guard_alive),.armed(guard_armed),.fault(guard_fault),.fault_code(guard_code));
    reg cfg_valid=0;
    wire cfg_ready;
    reg [63:0] cfg_session=0;
    reg [63:0] cfg_generation=0;
    reg [63:0] cfg_axi_base=0;
    reg [63:0] cfg_pcie_base=0;
    wire cfg_rsp_valid;
    reg cfg_rsp_ready=0;
    wire cfg_rsp_error;
    reg cmd_valid=0;
    wire cmd_ready;
    reg [127:0] cmd_tag=0;
    reg [63:0] cmd_generation=0;
    reg [63:0] cmd_source=0;
    reg [63:0] cmd_offset=0;
    reg [12:0] cmd_bytes=0;
    wire reject_valid;
    reg reject_ready=0;
    wire [3:0] reject_code;
    wire [127:0] reject_tag;
    wire src_req_valid;
    reg src_req_ready=0;
    wire [127:0] src_req_tag;
    wire [63:0] src_req_address;
    wire [12:0] src_req_bytes;
    reg src_valid=0;
    reg src_last=0;
    wire src_ready;
    reg [511:0] src_data=0;
    reg [127:0] src_tag=0;
    reg src_cpl_valid=0;
    wire src_cpl_ready;
    reg [127:0] src_cpl_tag=0;
    reg [12:0] src_cpl_bytes=0;
    reg [1:0] src_cpl_status=0;
    wire [63:0] m_awaddr;
    wire [3:0] m_awid;
    wire [7:0] m_awlen;
    wire [2:0] m_awsize;
    wire [1:0] m_awburst;
    wire m_awvalid;
    wire m_awready;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    wire m_wlast;
    wire m_wvalid;
    wire m_wready;
    wire m_bvalid;
    wire m_bready;
    wire [1:0] m_bresp;
    wire [3:0] m_bid;
    reg ack_valid=0;
    wire ack_ready;
    reg [127:0] ack_tag=0;
    reg [63:0] ack_generation=0;
    reg [12:0] ack_bytes=0;
    wire result_valid;
    reg result_ready=0;
    wire [127:0] result_tag;
    wire [12:0] result_bytes;
    wire owned;
    wire fault;
    wire [7:0] fault_code;
    wire window_held;
    wire [31:0] m_axil_awaddr;
    wire [2:0] m_axil_awprot;
    wire m_axil_awvalid;
    wire m_axil_awready;
    wire [31:0] m_axil_wdata;
    wire [3:0] m_axil_wstrb;
    wire m_axil_wvalid;
    wire m_axil_wready;
    wire [1:0] m_axil_bresp;
    wire m_axil_bvalid;
    wire m_axil_bready;
    wire [31:0] m_axil_araddr;
    wire [2:0] m_axil_arprot;
    wire m_axil_arvalid;
    wire m_axil_arready;
    wire [31:0] m_axil_rdata;
    wire [1:0] m_axil_rresp;
    wire m_axil_rvalid;
    wire m_axil_rready;
    wire mapper_fault;
    wire [7:0] mapper_fault_code;
    wire map_verified;
    integer attack_stage=-1,campaigns_done=0,host_rq_count=0,host_rc_count=0,tag_count=0;
    reg [7:0] completion_tag=0,request_tag=0,assigned_tag=0;
    reg [2:0] completion_status=0;
    reg [3:0] completion_error=0,request_type=0;
    reg [10:0] completion_words=0;
    reg [31:0] completion_data=0;
    reg nonposted_pending=0,tag_seen=0;

    // Read-only inspection mux and explicit B backpressure are fixture logic.
    // The mapper remains the only writer of private control registers.
    reg inspect_mode=0,inspect_valid=0,inspect_ready=0,release_b=1;
    reg [31:0] inspect_address=0;
    wire [31:0] c_araddr;
    wire [2:0] c_arprot;
    wire c_arvalid,c_arready,c_rvalid,c_rready,c_bvalid,c_bready;
    assign m_axil_araddr=inspect_mode?inspect_address:c_araddr;
    assign m_axil_arprot=inspect_mode?3'b0:c_arprot;
    assign m_axil_arvalid=inspect_mode?inspect_valid:c_arvalid;
    assign c_arready=!inspect_mode && m_axil_arready;
    assign c_rvalid=!inspect_mode && m_axil_rvalid;
    assign m_axil_rready=inspect_mode?inspect_ready:c_rready;
    assign c_bvalid=release_b && m_axil_bvalid;
    assign m_axil_bready=release_b && c_bready;
    ntrpu_axi_bridge_endpoint_client #(.AXI_ADDR_BITS(64)) dut(
        .m_axil_araddr(c_araddr),.m_axil_arprot(c_arprot),.m_axil_arvalid(c_arvalid),.m_axil_arready(c_arready),
        .m_axil_rvalid(c_rvalid),.m_axil_rready(c_rready),.m_axil_bvalid(c_bvalid),.m_axil_bready(c_bready),.*);
    integer phase=0,case_id=0,reads=0,writes=0,dma_b=0,results=0;
    integer host_bytes[0:1]='{0,0};
    bit written[0:1][0:127];
    reg [7:0] host_memory[0:1][0:127];
    integer host_packets=0,host_offset=0,host_length=0;
    reg host_active=0;
    reg [63:0] host_address=0;
    time previous_edge=0,clock_period=0;
    integer measured_edges=0;
    reg [31:0] last_ar=0;
    reg [127:0] active_tag=0;
    integer expected_bytes=64;
    task require(input bit condition,input string reason);
        if(condition!==1'b1)$fatal(1,"BN CHECK phase=%0d case=%0d %s",phase,case_id,reason);
    endtask
    function automatic [63:0] destination(input integer id);
        destination=id==0?64'h1234567800001000:64'h9abcdef000002000;
    endfunction
    function automatic [7:0] pattern(input integer id,offset);
        pattern=8'((id*13+offset*37)^(offset>>3));
    endfunction
    task tick(input integer count);repeat(count)@(negedge clk);endtask
    always @(posedge clk)if(cold_rst_n && phase>0 && phase<3)begin
        require(link_up===1'b1,"link and both native resets available");
        require(!fault && !mapper_fault && !reject_valid,"healthy client");
        if(previous_edge!=0)begin
            if(clock_period==0)clock_period=$time-previous_edge;
            require($time-previous_edge==clock_period,"stable native clock");
            if(measured_edges<32)$display("BN CLOCK t=%0t period=%0t",$time,$time-previous_edge);
            measured_edges++;
        end
        previous_edge=$time;
        if(cfg_valid && cfg_ready)$display("BN CFG case=%0d session=%h generation=%h axi=%h pcie=%h",case_id,cfg_session,cfg_generation,cfg_axi_base,cfg_pcie_base);
        if(cfg_rsp_valid && cfg_rsp_ready)$display("BN CFG_R case=%0d error=%b",case_id,cfg_rsp_error);
        if(m_axil_awvalid && m_axil_awready)begin
            require(!owned,"no CSR write under lease");
            $display("BN AW case=%0d address=%h",case_id,m_axil_awaddr);
        end
        if(m_axil_wvalid && m_axil_wready)$display("BN W case=%0d data=%h strobe=%h",case_id,m_axil_wdata,m_axil_wstrb);
        if(m_axil_bvalid && m_axil_bready)begin
            require(m_axil_bresp==0,"native mapping write response");writes++;
            $display("BN B case=%0d response=%d",case_id,m_axil_bresp);
        end
        if(!inspect_mode && m_axil_arvalid && m_axil_arready)begin
            require(!owned,"no CSR read under lease");last_ar=m_axil_araddr;
            $display("BN AR case=%0d address=%h",case_id,m_axil_araddr);
        end
        if(!inspect_mode && m_axil_rvalid && m_axil_rready)begin
            reads++;$display("BN R case=%0d data=%h response=%d address=%h",case_id,m_axil_rdata,m_axil_rresp,last_ar);
        end
        if(m_awvalid && m_awready)begin
            require(map_verified && m_awaddr==64'h10000000,"verified aperture before DMA");
            $display("BN DMA_AW case=%0d address=%h len=%d size=%d burst=%d id=%h",case_id,m_awaddr,m_awlen,m_awsize,m_awburst,m_awid);
        end
        if(m_wvalid && m_wready)$display("BN DMA_W case=%0d data=%h strb=%h last=%b",case_id,m_wdata,m_wstrb,m_wlast);
        if(m_bvalid && m_bready)begin
            require(m_bresp==0 && m_bid==0,"native writer response");dma_b++;
            $display("BN DMA_B case=%0d response=%d id=%h",case_id,m_bresp,m_bid);
        end
        if(ack_valid && ack_ready)$display("BN ACK case=%0d tag=%h generation=%h bytes=%d",case_id,ack_tag,ack_generation,ack_bytes);
        if(result_valid && result_ready)begin
            require(result_tag==active_tag && result_bytes==expected_bytes,"attributed result");results++;
            $display("BN RESULT case=%0d tag=%h bytes=%d",case_id,result_tag,result_bytes);
        end
    end
    // Observe actual decoded requests at the native Root Port. No forced internal nets.
    always @(posedge board.RP.user_clk)begin: receive
        integer first,b,id,offset,hits;reg [63:0] address;reg [7:0] byte_value;
        if(phase>0 && phase<3 && board.RP.m_axis_cq_tvalid && board.RP.m_axis_cq_tready)begin
            $display("BN CQ case=%0d t=%0t data=%0128h keep=%04h last=%b user=%046h",case_id,$time,board.RP.m_axis_cq_tdata,board.RP.m_axis_cq_tkeep,board.RP.m_axis_cq_tlast,board.RP.m_axis_cq_tuser);
            first=0;
            if(!host_active)begin
                require(board.RP.m_axis_cq_tuser[81:80]===2'b01 && board.RP.m_axis_cq_tdata[78:75]==1,"single nonstraddled memory write");
                require(board.RP.m_axis_cq_tuser[15:0]==16'h0f0f,"full byte enables");
                host_address=board.RP.m_axis_cq_tdata[63:0];host_length=board.RP.m_axis_cq_tdata[74:64]*4;
                host_offset=0;host_active=1;first=16;
            end else require(board.RP.m_axis_cq_tuser[81:80]===0,"no nested packet");
            require(!board.RP.m_axis_cq_tuser[96],"no discontinue");
            for(b=0;b<64;b++)if(board.RP.m_axis_cq_tuser[16+b]===1'b1)begin
                require(b>=first && board.RP.m_axis_cq_tkeep[b/4] && host_offset<host_length,"valid CQ payload lane");
                address=host_address+host_offset;byte_value=board.RP.m_axis_cq_tdata[b*8+:8];hits=0;
                for(id=0;id<2;id++)if(address>=destination(id) && address<destination(id)+(id==0?64:128))begin
                    offset=address-destination(id);require(id==case_id && !written[id][offset],"current destination and unique byte");
                    host_memory[id][offset]=byte_value;written[id][offset]=1;host_bytes[id]++;hits++;
                end
                require(hits==1,"expected full 64-bit PCIe destination");host_offset++;
            end
            if(board.RP.m_axis_cq_tlast)begin require(host_offset==host_length,"CQ packet length");host_active=0;host_packets++;end
        end
    end
    task configure;
        cfg_session=9;cfg_generation=case_id+1;cfg_axi_base=64'h10000000;cfg_pcie_base=destination(case_id);
        if(case_id==1)release_b=0;
        cfg_valid=1;do @(posedge clk);while(!cfg_ready);
        @(negedge clk);cfg_valid=0;
        if(case_id==1)begin
            wait(m_axil_bvalid && !m_axil_bready);tick(2);
            require(!cfg_rsp_valid && !owned && reads==6 && writes==2,"mapping held after upper write");
            $display("BI MAPPING_HELD reads=%0d writes=%0d bresp=%0d",reads,writes,m_axil_bresp);
            campaign(1);
            require(m_axil_bvalid && !m_axil_bready && !cfg_rsp_valid,"mapping remains pending through host accesses");
            @(negedge clk);release_b=1;
        end
        wait(cfg_rsp_valid);tick(3);
        require(!cfg_rsp_error && map_verified && reads==4*(case_id+1) && writes==2*(case_id+1),"six native transactions verified");
        cfg_rsp_ready=1;@(posedge clk);@(negedge clk);cfg_rsp_ready=0;tick(2);
    endtask
    task transfer;
        expected_bytes=case_id==0?64:128;active_tag={64'd9,64'(case_id+1)};
        cmd_tag=active_tag;cmd_generation=case_id+1;cmd_source=64'h1000;cmd_offset=0;cmd_bytes=expected_bytes;
        cmd_valid=1;do @(posedge clk);while(!cmd_ready);
        @(negedge clk);cmd_valid=0;wait(src_req_valid);
        require(src_req_tag==active_tag && src_req_address==64'h1000 && src_req_bytes==expected_bytes,"source request identity");
        if(case_id==1)begin
            campaign(2);inspect_translation(2);
            require(owned && src_req_valid && src_req_tag==active_tag,"ownership survives hostile accesses");
        end
        tick(2);src_req_ready=1;@(posedge clk);@(negedge clk);src_req_ready=0;
        for(integer offset=0;offset<expected_bytes;offset+=64)begin
            tick(2);
            for(integer b=0;b<64;b++)src_data[b*8+:8]=pattern(case_id,offset+b);
            src_tag=active_tag;src_last=offset+64==expected_bytes;src_valid=1;
            do @(posedge clk);while(!src_ready);
            @(negedge clk);src_valid=0;
        end
        src_cpl_tag=active_tag;src_cpl_bytes=expected_bytes;src_cpl_valid=1;
        do @(posedge clk);while(!src_cpl_ready);
        @(negedge clk);src_cpl_valid=0;
        cfg_generation=case_id+2;cfg_pcie_base=64'hfedcba9800003000;cfg_valid=1;
        wait(dma_b==case_id+1 && host_bytes[case_id]==expected_bytes && !host_active);
        tick(12);
        require(owned && !cfg_ready && !result_valid && reads==4*(case_id+1) && writes==2*(case_id+1),"native B and receipt do not retire lease without ACK");
        for(integer b=0;b<expected_bytes;b++)require(written[case_id][b] && host_memory[case_id][b]===pattern(case_id,b),"native destination bytes");
        ack_tag=active_tag;ack_generation=case_id+1;ack_bytes=expected_bytes;ack_valid=1;
        do @(posedge clk);while(!ack_ready);
        @(negedge clk);ack_valid=0;cfg_valid=0;
        wait(result_valid);tick(8);require(owned && !cfg_ready,"result backpressure retains mapping");
        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;tick(3);
        require(!owned && results==case_id+1,"successful retirement");
        $display("BN CASE_PASS case=%0d bytes=%0d reads=%0d writes=%0d",case_id,expected_bytes,reads,writes);
    endtask
    task run_probe;
        require(native_available===1'b1,"enumerated endpoint with both resets released");
        tick(16);configuration_rst_n=1;wait(link_up && cold_rst_n);phase=1;tick(4);
        configure;campaign(0);inspect_translation(0);transfer;
        case_id=1;phase=2;configure;inspect_translation(1);transfer;
        require(measured_edges>=32 && (clock_period==4000 || clock_period==8000),"measured native AXI clock");
        $display("BN DATA_PASS bytes=192 maps=2 clock_period_ps=%0t",clock_period);
        // Retain a third command across an operational PCIe reset; cold reset stays high.
        phase=3;cmd_tag={64'd9,64'd3};cmd_generation=2;cmd_source=64'h1000;cmd_offset=0;cmd_bytes=64;
        cmd_valid=1;do @(posedge clk);while(!cmd_ready);
        @(negedge clk);cmd_valid=0;wait(src_req_valid);tick(2);
        require(owned && cold_rst_n,"third context owned before PERSTn");
        board.sys_rst_n=0;
        #2000000;
        require(cold_rst_n && owned && fault && mapper_fault && !cmd_ready && !cfg_ready && !result_valid,"native reset retains failed context");
        require(guard_fault && stop && !link_up && guard_code[0],"native PERST reaches independent guard");
        $display("BI GUARD_PASS fault=%b code=%0d cold=%b owned=%b stop=%b link=%b",guard_fault,guard_code,cold_rst_n,owned,stop,link_up);
        $display("BN RESET_PASS owned=%b fault=%b mapper_fault=%b cold_rst_n=%b",owned,fault,mapper_fault,cold_rst_n);
        $display("BRIDGE_NATIVE_PASS bytes=192 maps=2 clock_period_ps=%0t host_isolation=unproven",clock_period);
        require(host_rq_count==48 && host_rc_count==36 && tag_count==36 && campaigns_done==3,"all host campaigns completed");
        $display("BRIDGE_ISOLATION_NATIVE_PASS campaigns=3 rq=48 rc=36 bytes=192 guard=1");
    endtask

    always @(posedge board.RP.user_clk)if(phase>0 && phase<3)begin
        if(board.RP.s_axis_rq_tvalid && board.RP.s_axis_rq_tready)begin
            require(board.RP.s_axis_rq_tlast,"host single-beat requests");
            request_tag=board.RP.s_axis_rq_tdata[103:96];
            request_type=board.RP.s_axis_rq_tdata[78:75];host_rq_count++;
            if(request_type!=1)begin
                require(!nonposted_pending,"one nonposted request at a time");
                nonposted_pending=1;tag_seen=0;
            end
            $display("BI RQ stage=%0d data=%h keep=%h last=%b user=%h",attack_stage,board.RP.s_axis_rq_tdata,board.RP.s_axis_rq_tkeep,board.RP.s_axis_rq_tlast,board.RP.s_axis_rq_tuser);
        end
        // The native Root Port has internal tag management. Its public tag
        // allocation output, not the proposed RQ descriptor tag, identifies RC.
        require(!board.RP.pcie_4_0_rport.pcie_rq_tag_vld1,"one native tag allocation per cycle");
        if(board.RP.pcie_4_0_rport.pcie_rq_tag_vld0)begin
            require(nonposted_pending && !tag_seen,"allocated tag has exactly one outstanding request");
            assigned_tag=board.RP.pcie_4_0_rport.pcie_rq_tag0;tag_seen=1;tag_count++;
            $display("BI TAG stage=%0d tag=%h",attack_stage,assigned_tag);
        end
        if(board.RP.m_axis_rc_tvalid && board.RP.m_axis_rc_tready)begin
            require(board.RP.m_axis_rc_tlast,"host single-beat completion");
            completion_tag=board.RP.m_axis_rc_tdata[71:64];
            completion_status=board.RP.m_axis_rc_tdata[45:43];
            completion_error=board.RP.m_axis_rc_tdata[15:12];
            completion_words=board.RP.m_axis_rc_tdata[42:32];
            completion_data=board.RP.m_axis_rc_tdata[127:96];host_rc_count++;
            require(nonposted_pending && tag_seen && completion_tag==assigned_tag && !board.RP.m_axis_rc_tdata[46],"attributed unpoisoned host completion");
            nonposted_pending=0;tag_seen=0;
            $display("BI RC stage=%0d data=%h keep=%h last=%b user=%h",attack_stage,board.RP.m_axis_rc_tdata,board.RP.m_axis_rc_tkeep,board.RP.m_axis_rc_tlast,board.RP.m_axis_rc_tuser);
        end
    end
    always @(posedge clk)if(phase>0 && phase<3)begin
        if(inspect_mode && m_axil_arvalid && m_axil_arready)
            $display("BI PRIVATE_AR stage=%0d address=%h",attack_stage,m_axil_araddr);
        if(inspect_mode && m_axil_rvalid && m_axil_rready)
            $display("BI PRIVATE_R stage=%0d data=%h response=%0d",attack_stage,m_axil_rdata,m_axil_rresp);
        if(board.AXI_PCIE_EP.m_axi_awvalid && board.AXI_PCIE_EP.m_axi_awready)
            $display("BI IN_AW stage=%0d address=%h len=%0d size=%0d",attack_stage,board.AXI_PCIE_EP.m_axi_awaddr,board.AXI_PCIE_EP.m_axi_awlen,board.AXI_PCIE_EP.m_axi_awsize);
        if(board.AXI_PCIE_EP.m_axi_wvalid && board.AXI_PCIE_EP.m_axi_wready)
            $display("BI IN_W stage=%0d data=%h strb=%h last=%b",attack_stage,board.AXI_PCIE_EP.m_axi_wdata,board.AXI_PCIE_EP.m_axi_wstrb,board.AXI_PCIE_EP.m_axi_wlast);
        if(board.AXI_PCIE_EP.m_axi_bvalid && board.AXI_PCIE_EP.m_axi_bready)
            $display("BI IN_B stage=%0d response=%0d",attack_stage,board.AXI_PCIE_EP.m_axi_bresp);
        if(board.AXI_PCIE_EP.m_axi_arvalid && board.AXI_PCIE_EP.m_axi_arready)
            $display("BI IN_AR stage=%0d address=%h len=%0d size=%0d",attack_stage,board.AXI_PCIE_EP.m_axi_araddr,board.AXI_PCIE_EP.m_axi_arlen,board.AXI_PCIE_EP.m_axi_arsize);
        if(board.AXI_PCIE_EP.m_axi_rvalid && board.AXI_PCIE_EP.m_axi_rready)
            $display("BI IN_R stage=%0d data=%h response=%0d last=%b",attack_stage,board.AXI_PCIE_EP.m_axi_rdata,board.AXI_PCIE_EP.m_axi_rresp,board.AXI_PCIE_EP.m_axi_rlast);
    end
    task wait_completion(input integer previous_count,input bit memory_read);
        integer cycles;
        cycles=0;
        while(host_rc_count==previous_count && cycles<2048)begin @(negedge board.RP.user_clk);cycles++;end
        require(host_rc_count==previous_count+1,"bounded native completion");
        if(memory_read)require(completion_status==0 && completion_error==0 && completion_words==1,"successful BAR read completion");
        else require((completion_status==0 && completion_error==0) ||
                     (completion_status==1 && completion_error==2),"config successful or unsupported, never timeout/CA");
    endtask
    task campaign(input integer stage);
        integer prior_count;reg [31:0] offset,value;reg [11:0] cfg_offset;
        attack_stage=stage;
        $display("BI CAMPAIGN_BEGIN stage=%0d owned=%b verified=%b release_b=%b",stage,owned,map_verified,release_b);
        for(integer i=0;i<4;i++)begin
            offset=(i<2?32'h208:32'h10208)+((i%2)*4);
            value=32'hbad00000+32'(stage*16+i);
            prior_count=host_rq_count;
            board.RP.tx_usrapp.TSK_TX_BAR_WRITE(0,offset,8'h40+8'(i),0,value);
            wait(host_rq_count==prior_count+1);
            prior_count=host_rc_count;
            board.RP.tx_usrapp.TSK_TX_BAR_READ(0,offset,8'h50+8'(i),0);
            wait_completion(prior_count,1);
            require(completion_data==value,"BAR write demonstrably reached its separate target");
            $display("BI BAR_PASS stage=%0d offset=%h data=%h",stage,offset,value);
        end
        for(integer i=0;i<4;i++)begin
            cfg_offset=(i<2?12'h208:12'ha08)+12'((i%2)*4);
            value=32'hfed00000+32'(stage*16+i);
            prior_count=host_rc_count;
            board.RP.tx_usrapp.TSK_TX_TYPE0_CONFIGURATION_WRITE(8'h60+8'(i),cfg_offset,value,4'hf);
            wait_completion(prior_count,0);
            $display("BI CFG_WRITE_PASS stage=%0d offset=%h status=%0d error=%0d",stage,cfg_offset,completion_status,completion_error);
            prior_count=host_rc_count;
            board.RP.tx_usrapp.TSK_TX_TYPE0_CONFIGURATION_READ(8'h70+8'(i),cfg_offset,4'hf);
            wait_completion(prior_count,0);
            $display("BI CFG_READ_PASS stage=%0d offset=%h status=%0d error=%0d data=%h",stage,cfg_offset,completion_status,completion_error,completion_data);
        end
        $display("BI CAMPAIGN_PASS stage=%0d owned=%b verified=%b release_b=%b",stage,owned,map_verified,release_b);
        campaigns_done++;
    endtask
    task inspect_translation(input integer stage);
        reg [31:0] expected;
        @(negedge clk);
        require(!c_arvalid && !m_axil_awvalid && !m_axil_wvalid && !m_axil_bvalid && !m_axil_rvalid,"exclusive read-only inspection");
        attack_stage=stage;inspect_mode=1;
        for(integer word_index=0;word_index<2;word_index++)begin
            inspect_address=word_index==0?32'h208:32'h20c;
            expected=word_index==0?32'(destination(case_id)>>32):32'(destination(case_id));
            inspect_valid=1;do @(posedge clk);while(!m_axil_arready);
            @(negedge clk);inspect_valid=0;inspect_ready=1;
            do @(posedge clk);while(!m_axil_rvalid);
            require(m_axil_rresp==0 && m_axil_rdata==expected,"private translation unchanged after host accesses");
            @(negedge clk);inspect_ready=0;tick(2);
        end
        inspect_mode=0;attack_stage=-1;
        $display("BI TRANSLATION_PASS stage=%0d pcie=%h owned=%b",stage,destination(case_id),owned);
    endtask

    initial begin #250000000;$fatal(1,"BN TIMEOUT phase=%0d case=%0d reads=%0d writes=%0d",phase,case_id,reads,writes);end
endmodule
`default_nettype wire
