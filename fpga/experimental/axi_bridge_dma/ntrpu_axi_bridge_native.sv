`timescale 1ps/1ps
`default_nettype none
// Native PIPE fixture; source reader and attributed ACK remain test stimuli.
module ntrpu_axi_bridge_native;
    wire clk=board.AXI_PCIE_EP.axi_aclk_out;
    reg cold_rst_n=0;
    wire link_up=board.AXI_PCIE_EP.user_lnk_up && board.AXI_PCIE_EP.axi_aresetn && board.AXI_PCIE_EP.axi_ctl_aresetn;
    reg stop=0;
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

    ntrpu_axi_bridge_endpoint_client #(.AXI_ADDR_BITS(64)) dut(.*);
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
        if(m_axil_arvalid && m_axil_arready)begin
            require(!owned,"no CSR read under lease");last_ar=m_axil_araddr;
            $display("BN AR case=%0d address=%h",case_id,m_axil_araddr);
        end
        if(m_axil_rvalid && m_axil_rready)begin
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
        cfg_valid=1;do @(posedge clk);while(!cfg_ready);
        @(negedge clk);cfg_valid=0;
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
        require(link_up===1'b1,"enumerated endpoint with both resets released");
        tick(16);cold_rst_n=1;phase=1;tick(4);
        configure;transfer;
        case_id=1;phase=2;configure;transfer;
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
        $display("BN RESET_PASS owned=%b fault=%b mapper_fault=%b cold_rst_n=%b",owned,fault,mapper_fault,cold_rst_n);
        $display("BRIDGE_NATIVE_PASS bytes=192 maps=2 clock_period_ps=%0t host_isolation=unproven",clock_period);
    endtask
    initial begin #250000000;$fatal(1,"BN TIMEOUT phase=%0d case=%0d reads=%0d writes=%0d",phase,case_id,reads,writes);end
endmodule
`default_nettype wire
