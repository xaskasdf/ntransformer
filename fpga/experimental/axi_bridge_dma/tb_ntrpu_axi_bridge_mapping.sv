`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_axi_bridge_mapping;
    reg clk=0;
    reg cold_rst_n=0;
    reg link_up=0;
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
    wire [47:0] m_awaddr;
    wire [3:0] m_awid;
    wire [7:0] m_awlen;
    wire [2:0] m_awsize;
    wire [1:0] m_awburst;
    wire m_awvalid;
    reg m_awready=0;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    wire m_wlast;
    wire m_wvalid;
    reg m_wready=0;
    reg m_bvalid=0;
    wire m_bready;
    reg [1:0] m_bresp=0;
    reg [3:0] m_bid=0;
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
    reg m_axil_awready=0;
    wire [31:0] m_axil_wdata;
    wire [3:0] m_axil_wstrb;
    wire m_axil_wvalid;
    reg m_axil_wready=0;
    reg [1:0] m_axil_bresp=0;
    reg m_axil_bvalid=0;
    wire m_axil_bready;
    wire [31:0] m_axil_araddr;
    wire [2:0] m_axil_arprot;
    wire m_axil_arvalid;
    reg m_axil_arready=0;
    reg [31:0] m_axil_rdata=0;
    reg [1:0] m_axil_rresp=0;
    reg m_axil_rvalid=0;
    wire m_axil_rready;
    wire mapper_fault;
    wire [7:0] mapper_fault_code;
    wire map_verified;
    always #5 clk=~clk;
    ntrpu_axi_bridge_endpoint_client #(.WAIT_LIMIT(32)) dut(.*);

    integer case_id=0,seed=0,ticks=0,passes=0,reads=0,writes=0,dma_aw=0,dma_bytes=0,dma_results=0;
    integer b_delay=0,r_delay=0,control_aw_stalls=0,control_w_stalls=0,control_ar_stalls=0;
    integer w_wait=0,ar_wait=0;
    reg release_stall=0,aw_seen=0,w_seen=0,r_pending=0,dma_b_done=0,dma_b_issued=0;
    reg [31:0] saved_aw=0,saved_w=0,saved_ar=0,upper=0,lower=0;
    reg [63:0] active_generation=1,expected_pcie=0;
    reg [127:0] active_tag=0;
    integer expected_bytes=64;
    reg was_aw_stall=0,was_w_stall=0,was_ar_stall=0;
    reg [31:0] old_aw=0,old_w=0,old_ar=0;
    task require(input bit condition,input string reason);
        if(!condition)$fatal(1,"MAP CHECK case=%0d seed=%0d %s",case_id,seed,reason);
    endtask
    task tick(input integer count);repeat(count)@(negedge clk);endtask
    function automatic [7:0] pattern(input integer offset);
        pattern=8'((case_id*13+offset*37)^(offset>>3));
    endfunction
    always @* begin
        m_axil_awready=cold_rst_n && !aw_seen && !m_axil_bvalid && ((ticks+seed)%3==0) &&
                       !((case_id==7 || case_id==9) && !release_stall);
        m_axil_wready=cold_rst_n && !w_seen && !m_axil_bvalid && w_wait>=2 && ((ticks+seed)%3==1);
        m_axil_arready=cold_rst_n && !r_pending && !m_axil_rvalid && ar_wait>=2 && ((ticks+seed)%3==2);
        m_awready=cold_rst_n && dma_aw==0 && ((ticks+seed)%3==0);
        m_wready=cold_rst_n && ((ticks+seed)%3!=0);
    end
    always @(posedge clk)begin
        if(!cold_rst_n)begin
            ticks<=0;w_wait<=0;ar_wait<=0;aw_seen<=0;w_seen<=0;r_pending<=0;m_axil_bvalid<=0;m_axil_rvalid<=0;
            b_delay<=0;r_delay<=0;upper<=32'hdeadbeef;lower<=32'habcd0000;
            m_bvalid<=0;m_bresp<=0;m_bid<=0;dma_b_done<=0;dma_b_issued<=0;
            was_aw_stall<=0;was_w_stall<=0;was_ar_stall<=0;
        end else begin
            ticks<=ticks+1;
            if(m_axil_wvalid && !m_axil_wready)w_wait<=w_wait+1;else w_wait<=0;
            if(m_axil_arvalid && !m_axil_arready)ar_wait<=ar_wait+1;else ar_wait<=0;
            if(cfg_valid && cfg_ready)$display("MC CFG case=%0d session=%h generation=%h axi=%h pcie=%h",case_id,cfg_session,cfg_generation,cfg_axi_base,cfg_pcie_base);
            if(cfg_rsp_valid && cfg_rsp_ready)$display("MC CFG_R case=%0d error=%b",case_id,cfg_rsp_error);
            if(was_aw_stall)require(m_axil_awvalid && m_axil_awaddr===old_aw,"AXI-Lite AW stable");
            if(was_w_stall)require(m_axil_wvalid && m_axil_wdata===old_w && m_axil_wstrb==15,"AXI-Lite W stable");
            if(was_ar_stall)require(m_axil_arvalid && m_axil_araddr===old_ar,"AXI-Lite AR stable");
            was_aw_stall<=m_axil_awvalid&&!m_axil_awready;old_aw<=m_axil_awaddr;
            was_w_stall<=m_axil_wvalid&&!m_axil_wready;old_w<=m_axil_wdata;
            was_ar_stall<=m_axil_arvalid&&!m_axil_arready;old_ar<=m_axil_araddr;
            if(m_axil_awvalid&&!m_axil_awready)control_aw_stalls++;
            if(m_axil_wvalid&&!m_axil_wready)control_w_stalls++;
            if(m_axil_arvalid&&!m_axil_arready)control_ar_stalls++;
            if(m_axil_awvalid && m_axil_awready)begin
                require(!owned && !m_axil_awaddr[28] && (m_axil_awaddr==32'h208 || m_axil_awaddr==32'h20c) && m_axil_awprot==0,"only private translation writes");
                aw_seen<=1;saved_aw<=m_axil_awaddr;
                $display("MC AW case=%0d address=%h",case_id,m_axil_awaddr);
            end
            if(m_axil_wvalid && m_axil_wready)begin
                require(!owned && m_axil_wstrb==15,"full mapping word outside lease");
                w_seen<=1;saved_w<=m_axil_wdata;
                $display("MC W case=%0d data=%h",case_id,m_axil_wdata);
            end
            if(aw_seen && w_seen && !m_axil_bvalid)begin
                if(b_delay==2)begin
                    m_axil_bvalid<=1;m_axil_bresp<=case_id==3?2:0;writes++;
                    if(saved_aw==32'h208)upper<=saved_w;else lower<=saved_w;
                end else b_delay<=b_delay+1;
            end
            if(m_axil_bvalid && m_axil_bready)begin
                $display("MC B case=%0d response=%0d",case_id,m_axil_bresp);
                m_axil_bvalid<=0;aw_seen<=0;w_seen<=0;b_delay<=0;
            end
            if(m_axil_arvalid && m_axil_arready)begin
                require(!owned && !m_axil_araddr[28] && m_axil_arprot==0,"private mapping read");
                r_pending<=1;saved_ar<=m_axil_araddr;r_delay<=0;
                $display("MC AR case=%0d address=%h",case_id,m_axil_araddr);
            end
            if(r_pending && !m_axil_rvalid && !(case_id==8 && !release_stall))begin
                if(r_delay==3)begin
                    m_axil_rvalid<=1;m_axil_rresp<=case_id==6?2:0;reads++;
                    case(saved_ar)
                        32'h130:m_axil_rdata<=case_id==2?0:9;
                        32'h204:m_axil_rdata<=32'h03800000;
                        32'h208:m_axil_rdata<=upper ^ (case_id==4?32'h1:0);
                        32'h20c:m_axil_rdata<=lower ^ (case_id==5?32'h1000:0);
                        default:begin m_axil_rdata<=0;m_axil_rresp<=3;end
                    endcase
                end else r_delay<=r_delay+1;
            end
            if(m_axil_rvalid && m_axil_rready)begin
                $display("MC R case=%0d data=%h response=%0d",case_id,m_axil_rdata,m_axil_rresp);
                m_axil_rvalid<=0;r_pending<=0;
            end
            if(m_awvalid && m_awready)begin
                require(map_verified && !mapper_fault && !fault && dma_aw==0,"DMA follows verified map");
                require(m_awaddr==48'h10000000 && m_awid==0 && m_awsize==6 && m_awburst==1 && (m_awlen+1)*64==expected_bytes,"DMA address contract");
                require({upper,lower}==expected_pcie,"DMA targets read-back register image");
                dma_aw++;
                $display("MC DMA_AW case=%0d axi=%h translated=%h bytes=%0d",case_id,m_awaddr,{upper,lower},expected_bytes);
            end
            if(m_wvalid && m_wready)begin
                require(&m_wstrb && dma_bytes<expected_bytes && m_wlast==(dma_bytes+64==expected_bytes),"DMA beat shape");
                for(integer j=0;j<64;j++)require(m_wdata[j*8+:8]===pattern(dma_bytes+j),"DMA bytes");
                $display("MC DMA_W case=%0d offset=%0d last=%b data=%h",case_id,dma_bytes,m_wlast,m_wdata);
                dma_bytes+=64;
            end
            if(dma_aw==1 && dma_bytes==expected_bytes && !dma_b_issued)begin
                dma_b_issued<=1;m_bvalid<=1;
            end
            if(m_bvalid && m_bready)begin m_bvalid<=0;dma_b_done<=1;$display("MC DMA_B case=%0d",case_id);end
            if(ack_valid && ack_ready)$display("MC ACK case=%0d tag=%h generation=%h",case_id,ack_tag,ack_generation);
            if(result_valid && result_ready)begin
                require(result_tag==active_tag && result_bytes==expected_bytes,"attributed DMA result");
                dma_results++;$display("MC RESULT case=%0d bytes=%0d",case_id,result_bytes);
            end
        end
    end
    task reset_case(input integer id);
        @(negedge clk);cold_rst_n=0;case_id=id;link_up=1;stop=0;
        cfg_valid=0;cfg_rsp_ready=0;cmd_valid=0;reject_ready=0;
        src_req_ready=0;src_valid=0;src_cpl_valid=0;src_cpl_status=0;
        ack_valid=0;result_ready=0;release_stall=0;
        cfg_session=9;cfg_generation=1;cfg_axi_base=64'h10000000;cfg_pcie_base=64'h1234567800001000;
        expected_pcie=cfg_pcie_base;active_generation=1;expected_bytes=64;active_tag={64'd9,64'd1};
        reads=0;writes=0;dma_aw=0;dma_bytes=0;dma_results=0;
        tick(3);cold_rst_n=1;tick(2);$display("MAP CASE START id=%0d",case_id);
    endtask
    task start_mapping;
        cfg_valid=1;do @(posedge clk);while(!cfg_ready);
        @(negedge clk);cfg_valid=0;tick(1);
        require(!cmd_ready,"no DMA admission before verification");
    endtask
    task accept_mapping(input bit success);
        wait(cfg_rsp_valid);tick(3);
        require(cfg_rsp_error==!success,"mapping verdict");
        require(!cmd_ready,"mapping report must be consumed");
        if(success)begin
            require(map_verified && !mapper_fault && !fault,"verified successful mapping");
            require(reads==4 && writes==2,"complete read-back sequence");
        end
        else require(mapper_fault && !map_verified,"failed mapping remains unverified");
        cfg_rsp_ready=1;@(posedge clk);@(negedge clk);cfg_rsp_ready=0;tick(2);
    endtask
    task transfer;
        cmd_tag=active_tag;cmd_generation=active_generation;cmd_source=64'h8000000000-4096;
        cmd_offset=0;cmd_bytes=expected_bytes;cmd_valid=1;
        do @(posedge clk);while(!cmd_ready);
        @(negedge clk);cmd_valid=0;
        wait(src_req_valid);tick(2);src_req_ready=1;
        @(posedge clk);@(negedge clk);src_req_ready=0;
        for(integer offset=0;offset<expected_bytes;offset+=64)begin
            tick((offset/64+seed)%3);
            for(integer j=0;j<64;j++)src_data[j*8+:8]=pattern(offset+j);
            src_tag=active_tag;src_last=offset+64==expected_bytes;src_valid=1;
            do @(posedge clk);while(!src_ready);
            @(negedge clk);src_valid=0;
        end
        src_cpl_tag=active_tag;src_cpl_bytes=expected_bytes;src_cpl_valid=1;
        do @(posedge clk);while(!src_cpl_ready);
        @(negedge clk);src_cpl_valid=0;
        cfg_generation=active_generation+1;cfg_pcie_base=64'hfedcba9800003000;cfg_valid=1;
        wait(dma_b_done);tick(4);
        require(owned && !cfg_ready && !result_valid && writes==2 && reads==4,"lease pins mapping through B until ACK");
        ack_tag=active_tag;ack_generation=active_generation;ack_bytes=expected_bytes;ack_valid=1;
        do @(posedge clk);while(!ack_ready);
        @(negedge clk);ack_valid=0;cfg_valid=0;
        wait(result_valid);tick(4);require(owned && !cfg_ready,"result retains lease");
        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;tick(3);
        require(!owned && !fault && !mapper_fault && dma_results==1,"successful composed retirement");
    endtask
    task pass_case;
        passes++;
        $display("MAP CASE PASS id=%0d reads=%0d writes=%0d fault=%h aw=%0d bytes=%0d results=%0d",case_id,reads,writes,mapper_fault_code,dma_aw,dma_bytes,dma_results);
    endtask
    initial begin
        if(!$value$plusargs("seed=%d",seed))seed=0;
        $dumpfile("mapping.vcd");$dumpvars(1,tb_ntrpu_axi_bridge_mapping);
        reset_case(0);start_mapping;accept_mapping(1);transfer;pass_case;
        // A second transfer and remap in the SAME session, with no reset.
        @(negedge clk);case_id=1;reads=0;writes=0;dma_aw=0;dma_bytes=0;dma_results=0;dma_b_done=0;dma_b_issued=0;
        expected_bytes=128;active_generation=2;active_tag={64'd9,64'd2};
        cfg_generation=2;cfg_pcie_base=64'h9abcdef000002000;expected_pcie=cfg_pcie_base;
        $display("MAP CASE START id=1");start_mapping;accept_mapping(1);transfer;pass_case;
        for(integer id=2;id<=10;id++)begin
            reset_case(id);
            if(id==10)cfg_axi_base=64'h20000000;
            start_mapping;
            if(id==7 || id==9)begin
                wait(m_axil_awvalid);
                if(id==9)begin tick(2);link_up=0;tick(4);end
                else tick(45);
                require(mapper_fault && m_axil_awvalid && !cfg_rsp_valid,"fault does not fabricate a mapping response or cancel AW");
                release_stall=1;link_up=1;
            end
            if(id==8)begin
                wait(r_pending);tick(45);
                require(mapper_fault && !cfg_rsp_valid && !m_axil_rvalid,"read timeout awaits a real response");
                release_stall=1;
            end
            accept_mapping(0);tick(4);
            require(fault && mapper_fault && !map_verified && !cmd_ready && !cfg_ready && !owned && dma_aw==0 && dma_bytes==0,"failed programming cannot authorize payload");
            pass_case;
        end
        require(passes==11 && control_aw_stalls>0 && control_w_stalls>0 && control_ar_stalls>0,"case and stall coverage");
        $display("MAP COVER aw=%0d w=%0d ar=%0d",control_aw_stalls,control_w_stalls,control_ar_stalls);
        $display("MAPPING_LOCAL_PASS seed=%0d cases=%0d",seed,passes);$finish;
    end
    initial begin #1000000;$fatal(1,"MAPPING TIMEOUT case=%0d",case_id);end
endmodule
`default_nettype wire
