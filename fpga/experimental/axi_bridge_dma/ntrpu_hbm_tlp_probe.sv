`timescale 1ns/1ps
`define EP board.AXI_PCIE_EP
`define RP board.RP
// Host CSR traffic crosses PIPE; HBM preloading uses public AXI3 write ports only.
module ntrpu_hbm_tlp_probe;
    wire clk=`EP.axi_aclk_out;
    reg cold_rst_n=0,reference_clk=0,reference_enable=0,fabric_reset=1;
    always #5 if(reference_enable)reference_clk=~reference_clk;else reference_clk=0;
    reg [67:0] load_awaddr=0;
    reg [3:0] load_awburst=4'b0101;
    reg [11:0] load_awid=0;
    reg [7:0] load_awlen=0;
    reg [5:0] load_awsize={3'd5,3'd5};
    reg [1:0] load_awvalid=0,load_wlast=0,load_wvalid=0,load_bready=3;
    wire [1:0] load_awready,load_wready,load_bvalid;
    reg [511:0] load_wdata=0;
    reg [63:0] load_wstrb=0;
    wire [11:0] load_bid;
    wire [3:0] load_bresp;
    integer phase=0,case_id=0,checks=0,rq_count=0,rc_count=0,tag_count=0,b_count=0,r_count=0;
    integer csr_w=0,csr_r=0,aw_count=0,ar_count=0;
    bit nonposted_pending=0,tag_seen=0;
    reg [7:0] assigned_tag;
    reg [2:0] completion_status;
    reg [3:0] completion_error;
    reg [10:0] completion_words;
    reg [31:0] completion_data;
    reg [1:0] last_b;
    reg [63:0] bar_base;
    reg [31:0] values[0:7];
    task require(input bit good,input string why);
        checks++;
        if(!good)$fatal(1,"BT CHECK case=%0d %s",case_id,why);
    endtask
    always @(posedge `RP.user_clk)if(phase==1)begin
        if(`RP.s_axis_rq_tvalid&&`RP.s_axis_rq_tready)begin
            require(`RP.s_axis_rq_tlast,"single beat request");rq_count++;
            if(`RP.s_axis_rq_tdata[78:75]!=1)begin
                require(!nonposted_pending,"one pending read");nonposted_pending=1;tag_seen=0;
            end
            $display("BT RQ case=%0d data=%h keep=%h",case_id,`RP.s_axis_rq_tdata,`RP.s_axis_rq_tkeep);
        end
        require(!`RP.pcie_4_0_rport.pcie_rq_tag_vld1,"one allocation per cycle");
        if(`RP.pcie_4_0_rport.pcie_rq_tag_vld0)begin
            require(nonposted_pending&&!tag_seen,"native tag attribution");
            assigned_tag=`RP.pcie_4_0_rport.pcie_rq_tag0;tag_seen=1;tag_count++;
            $display("BT TAG case=%0d tag=%h",case_id,assigned_tag);
        end
        if(`RP.m_axis_rc_tvalid&&`RP.m_axis_rc_tready)begin
            require(`RP.m_axis_rc_tlast&&nonposted_pending&&tag_seen&&
                `RP.m_axis_rc_tdata[71:64]==assigned_tag&&!`RP.m_axis_rc_tdata[46]&&`RP.m_axis_rc_tdata[30],"attributed complete unpoisoned completion");
            completion_status=`RP.m_axis_rc_tdata[45:43];completion_error=`RP.m_axis_rc_tdata[15:12];
            completion_words=`RP.m_axis_rc_tdata[42:32];completion_data=`RP.m_axis_rc_tdata[127:96];
            nonposted_pending=0;tag_seen=0;rc_count++;
            $display("BT RC case=%0d data=%h keep=%h",case_id,`RP.m_axis_rc_tdata,`RP.m_axis_rc_tkeep);
        end
    end
    always @(posedge clk)if(phase==1)begin
        if(`EP.m_axi_awvalid&&`EP.m_axi_awready)begin aw_count++;
            $display("BT AW case=%0d addr=%h len=%0d size=%0d",case_id,`EP.m_axi_awaddr,`EP.m_axi_awlen,`EP.m_axi_awsize);end
        if(`EP.m_axi_wvalid&&`EP.m_axi_wready)$display("BT W case=%0d data=%h strb=%h last=%b",case_id,`EP.m_axi_wdata,`EP.m_axi_wstrb,`EP.m_axi_wlast);
        if(`EP.m_axi_bvalid&&`EP.m_axi_bready)begin b_count++;last_b=`EP.m_axi_bresp;$display("BT B case=%0d resp=%0d",case_id,last_b);end
        if(`EP.m_axi_arvalid&&`EP.m_axi_arready)begin ar_count++;
            $display("BT AR case=%0d addr=%h len=%0d size=%0d",case_id,`EP.m_axi_araddr,`EP.m_axi_arlen,`EP.m_axi_arsize);end
        if(`EP.m_axi_rvalid&&`EP.m_axi_rready)begin r_count++;
            $display("BT R case=%0d data=%h resp=%0d last=%b",case_id,`EP.m_axi_rdata,`EP.m_axi_rresp,`EP.m_axi_rlast);end
        if(`EP.csr_awvalid&&`EP.csr_awready)begin csr_w++;$display("BT CSR_AW case=%0d addr=%h",case_id,`EP.csr_awaddr);end
        if(`EP.csr_arvalid&&`EP.csr_arready)begin csr_r++;$display("BT CSR_AR case=%0d addr=%h",case_id,`EP.csr_araddr);end
    end
    task send_write(input [31:0] offset,input [31:0] value,input [3:0] be,input integer words,input [1:0] expected);
        integer before_b,cycles;
        case_id++;before_b=b_count;
        for(integer i=0;i<8;i++)`RP.tx_usrapp.DATA_STORE[i]=value[(i%4)*8+:8];
        $display("BT WRITE case=%0d offset=%h value=%h be=%h words=%0d expected=%0d",case_id,offset,value,be,words,expected);
        `RP.tx_usrapp.TSK_TX_MEMORY_WRITE_64(8'(case_id),0,11'(words),bar_base+offset,words==1?4'h0:4'hf,be,1'b0);
        cycles=0;
        while(b_count==before_b&&cycles<2048)begin @(negedge clk);cycles++;end
        require(b_count==before_b+1&&last_b==expected,"native write response");
    endtask
    task send_read(input [31:0] offset,input integer words,input bit success);
        integer before_rc,cycles;
        case_id++;before_rc=rc_count;
        $display("BT READ case=%0d offset=%h words=%0d success=%b",case_id,offset,words,success);
        `RP.tx_usrapp.TSK_TX_MEMORY_READ_64(8'(case_id),0,11'(words),bar_base+offset,words==1?4'h0:4'hf,4'hf);
        cycles=0;
        while(rc_count==before_rc&&cycles<2048)begin @(negedge `RP.user_clk);cycles++;end
        require(rc_count==before_rc+1,"bounded completion");
        if(success)require(completion_status==0&&completion_error==0&&completion_words==1,"successful one-word completion");
        else require((completion_status==1||completion_status==4)&&completion_error==2&&completion_words==0,"error completion without payload");
    endtask
    task check_read(input [31:0] offset,input [31:0] expected);
        send_read(offset,1,1);require(completion_data===expected,"exact host read data");
        $display("BT READ_PASS case=%0d offset=%h data=%h",case_id,offset,completion_data);
    endtask
    localparam [63:0] SESSION=64'hcafe002112345678,HOST_BASE=64'h1234567800001000;
    integer transfer_id=0,cfg_responses=0,commands=0,dma_b=0,acks=0,results=0,host_packets=0;
    integer load_b_count[0:1]='{0,0},h_reads[0:1]='{0,0},h_beats[0:1]='{0,0};
    integer host_bytes[0:1]='{0,0};
    bit written[0:1][0:127];
    reg host_active=0;
    reg [63:0] host_address=0;
    integer host_offset=0,host_length=0;
    function automatic [63:0] source_address(input integer id);
        return 64'h1000+(64'(id)<<33);
    endfunction
    function automatic [63:0] destination(input integer id);return HOST_BASE+id*256;endfunction
    function automatic integer length(input integer id);return id==0?64:128;endfunction
    function automatic [7:0] pattern(input integer id,offset);return 8'((id*71+offset*37+91)^(offset>>3));endfunction
    task tick(input integer n);repeat(n)@(negedge clk);endtask

    always @(posedge clk)if(phase==1)begin
        require(`EP.client_cold_rst_n&&`EP.user_lnk_up&&!`EP.fault&&!`EP.mapper_fault&&
            !`EP.source_fault&&!`EP.guard_fault&&!`EP.reader_fault_raw&&!`EP.availability_fault,"healthy integrated path");
        if(`EP.cfg_valid&&`EP.cfg_ready)$display("HBT CFG session=%h generation=%h axi=%h pcie=%h",`EP.cfg_session,`EP.cfg_generation,`EP.cfg_axi_base,`EP.cfg_pcie_base);
        if(`EP.cfg_rsp_valid&&`EP.cfg_rsp_ready)begin
            require(!`EP.cfg_rsp_error&&`EP.map_verified,"native mapper accepted configuration");cfg_responses++;
            $display("HBT CFG_R error=%b verified=%b",`EP.cfg_rsp_error,`EP.map_verified);
        end
        if(`EP.cmd_valid&&`EP.cmd_ready)begin commands++;
            $display("HBT CMD transfer=%0d tag=%h source=%h offset=%h bytes=%0d generation=%h",transfer_id,`EP.cmd_tag,`EP.cmd_source,`EP.cmd_offset,`EP.cmd_bytes,`EP.cmd_generation);end
        if(`EP.m_axil_awvalid&&`EP.m_axil_awready)$display("HBT MAP_AW address=%h",`EP.m_axil_awaddr);
        if(`EP.m_axil_wvalid&&`EP.m_axil_wready)$display("HBT MAP_W data=%h strb=%h",`EP.m_axil_wdata,`EP.m_axil_wstrb);
        if(`EP.m_axil_bvalid&&`EP.m_axil_bready)$display("HBT MAP_B resp=%h",`EP.m_axil_bresp);
        if(`EP.m_axil_arvalid&&`EP.m_axil_arready)$display("HBT MAP_AR address=%h",`EP.m_axil_araddr);
        if(`EP.m_axil_rvalid&&`EP.m_axil_rready)$display("HBT MAP_R data=%h resp=%h",`EP.m_axil_rdata,`EP.m_axil_rresp);
        if(`EP.m_awvalid&&`EP.m_awready)begin
            require(`EP.map_verified&&`EP.m_awaddr==64'h10000000+transfer_id*256,"verified outgoing aperture");
            $display("HBT DMA_AW transfer=%0d address=%h len=%0d size=%0d burst=%0d id=%h",transfer_id,`EP.m_awaddr,`EP.m_awlen,`EP.m_awsize,`EP.m_awburst,`EP.m_awid);
        end
        if(`EP.m_wvalid&&`EP.m_wready)$display("HBT DMA_W transfer=%0d data=%h strb=%h last=%b",transfer_id,`EP.m_wdata,`EP.m_wstrb,`EP.m_wlast);
        if(`EP.m_bvalid&&`EP.m_bready)begin
            require(`EP.m_bresp==0&&`EP.m_bid==0,"native bridge B");dma_b++;
            $display("HBT DMA_B transfer=%0d resp=%h id=%h",transfer_id,`EP.m_bresp,`EP.m_bid);
        end
        if(`EP.ack_valid&&`EP.ack_ready)begin acks++;
            $display("HBT ACK transfer=%0d tag=%h bytes=%0d generation=%h",transfer_id,`EP.ack_tag,`EP.ack_bytes,`EP.ack_generation);end
        if(`EP.result_valid&&`EP.result_ready)begin
            require(`EP.result_tag=={SESSION,64'(transfer_id+1)}&&`EP.result_bytes==length(transfer_id),"result identity");results++;
            $display("HBT RESULT transfer=%0d tag=%h bytes=%0d",transfer_id,`EP.result_tag,`EP.result_bytes);
        end
        require(!`EP.reject_valid,"unexpected rejection");
    end
    always @(posedge `EP.mem_clk)if(cold_rst_n)begin
        for(integer lane=0;lane<2;lane++)begin
            if(load_awvalid[lane]&&load_awready[lane])$display("HBT HAW lane=%0d addr=%h len=%0d size=%0d",lane,load_awaddr[lane*34+:34],load_awlen[lane*4+:4],load_awsize[lane*3+:3]);
            if(load_wvalid[lane]&&load_wready[lane])$display("HBT HW lane=%0d data=%h strb=%h last=%b",lane,load_wdata[lane*256+:256],load_wstrb[lane*32+:32],load_wlast[lane]);
            if(load_bvalid[lane]&&load_bready[lane])begin
                require(load_bresp[lane*2+:2]==0&&load_bid[lane*6+:6]==0,"HBM load response");load_b_count[lane]++;
                $display("HBT HB lane=%0d resp=%h id=%h",lane,load_bresp[lane*2+:2],load_bid[lane*6+:6]);
            end
            if(`EP.h_arvalid[lane]&&`EP.h_arready[lane])begin h_reads[lane]++;
                $display("HBT HAR lane=%0d addr=%h len=%0d id=%h size=%0d burst=%0d",lane,`EP.h_araddr[lane*34+:34],`EP.h_arlen[lane*4+:4],`EP.h_arid[lane*6+:6],`EP.h_arsize[lane*3+:3],`EP.h_arburst[lane*2+:2]);end
            if(`EP.h_rvalid[lane]&&`EP.h_rready[lane])begin h_beats[lane]++;
                require(`EP.h_rresp[lane*2+:2]==0,"native HBM read response");
                $display("HBT HR lane=%0d data=%h resp=%h id=%h last=%b",lane,`EP.h_rdata[lane*256+:256],`EP.h_rresp[lane*2+:2],`EP.h_rid[lane*6+:6],`EP.h_rlast[lane]);end
        end
    end
    always @(posedge `RP.user_clk)if(phase==1&&`RP.m_axis_cq_tvalid&&`RP.m_axis_cq_tready)begin: receive
        integer first,offset;reg [63:0] address;reg [7:0] byte_value;
        $display("HBT CQ transfer=%0d data=%h keep=%h last=%b user=%h",transfer_id,`RP.m_axis_cq_tdata,`RP.m_axis_cq_tkeep,`RP.m_axis_cq_tlast,`RP.m_axis_cq_tuser);
        first=0;
        if(!host_active)begin
            require(`RP.m_axis_cq_tuser[81:80]===1&&`RP.m_axis_cq_tdata[78:75]==1,"non-straddled host write");
            require(`RP.m_axis_cq_tuser[15:0]==16'h0f0f,"full host byte enables");
            host_address=`RP.m_axis_cq_tdata[63:0];host_length=`RP.m_axis_cq_tdata[74:64]*4;
            host_offset=0;host_active=1;first=16;
        end else require(`RP.m_axis_cq_tuser[81:80]===0,"no nested host packet");
        require(!`RP.m_axis_cq_tuser[96],"no discontinued host packet");
        for(integer b=0;b<64;b++)if(`RP.m_axis_cq_tuser[16+b]===1'b1)begin
            require(b>=first&&`RP.m_axis_cq_tkeep[b/4]&&host_offset<host_length,"valid host payload lane");
            address=host_address+host_offset;byte_value=`RP.m_axis_cq_tdata[b*8+:8];
            require(address>=destination(transfer_id)&&address<destination(transfer_id)+length(transfer_id),"full-width host address");
            offset=address-destination(transfer_id);
            require(!written[transfer_id][offset]&&byte_value===pattern(transfer_id,offset),"unique exact HBM byte at host");
            written[transfer_id][offset]=1;host_bytes[transfer_id]++;host_offset++;
        end
        if(`RP.m_axis_cq_tlast)begin require(host_offset==host_length,"complete host packet");host_active=0;host_packets++;end
    end
    task preload(input integer lane);
        @(negedge `EP.mem_clk);
        load_awaddr[lane*34+:34]=34'(source_address(lane));load_awlen[lane*4+:4]=4'(length(lane)/32-1);
        load_awvalid[lane]=1;
        do @(posedge `EP.mem_clk);while(!load_awready[lane]);
        @(negedge `EP.mem_clk);load_awvalid[lane]=0;
        for(integer beat=0;beat<length(lane)/32;beat++)begin
            for(integer b=0;b<32;b++)load_wdata[lane*256+b*8+:8]=pattern(lane,beat*32+b);
            load_wstrb[lane*32+:32]='1;load_wlast[lane]=beat==length(lane)/32-1;load_wvalid[lane]=1;
            do @(posedge `EP.mem_clk);while(!load_wready[lane]);
            @(negedge `EP.mem_clk);load_wvalid[lane]=0;
        end
        wait(load_b_count[lane]==1);
        $display("HBT LOADED lane=%0d bytes=%0d",lane,length(lane));
    endtask
    task stage(input integer base,words,input [351:0] data);
        for(integer i=0;i<words;i++)send_write(base+i*4,data[32*i+:32],15,1,0);
    endtask
    task transfer;
        reg [351:0] command_image;
        command_image={32'(length(transfer_id)),64'(transfer_id*256),source_address(transfer_id),64'd1,SESSION,64'(transfer_id+1)};
        stage('h140,11,command_image);send_write('h10,1,15,1,0);
        wait(dma_b==transfer_id+1&&host_bytes[transfer_id]==length(transfer_id)&&!host_active);
        tick(12);
        require(`EP.owned&&`EP.source_owned&&!`EP.result_valid&&results==transfer_id,"B and receipt cannot retire before ACK");
        $display("HBT BEFORE_ACK transfer=%0d owned=%b source_owned=%b results=%0d bytes=%0d",transfer_id,`EP.owned,`EP.source_owned,results,host_bytes[transfer_id]);
        check_read(8,2);
        stage('h180,7,{32'(length(transfer_id)),64'd1,SESSION,64'(transfer_id+1)});send_write('h14,1,15,1,0);
        wait(results==transfer_id+1);tick(12);
        check_read(8,'h92);check_read('h2c,2);
        check_read('h200,transfer_id+1);check_read('h204,0);check_read('h208,SESSION[31:0]);check_read('h20c,SESSION[63:32]);check_read('h210,length(transfer_id));
        require(!`EP.owned&&!`EP.source_owned,"mailbox acceptance retires both contexts");
        send_write('h1c,1,15,1,0);check_read(8,'h80);
        $display("HBT TRANSFER_PASS transfer=%0d bytes=%0d",transfer_id,length(transfer_id));
    endtask
    task run_probe;
        bar_base={32'b0,`RP.tx_usrapp.BAR_INIT_P_BAR[0][31:0]};
        require(bar_base==64'h80000000,"relocated BAR0");
        reference_enable=1;repeat(16)@(negedge reference_clk);fabric_reset=0;
        wait(`EP.fabric_clock_locked===1);repeat(8)@(negedge reference_clk);cold_rst_n=1;
        wait(`EP.hbm_init_complete===2'b11&&`EP.reader_started_raw&&`EP.client_link_up&&`EP.core.reader_started);
        tick(16);phase=1;$display("BT START base=%h",bar_base);
        $display("HBT BOOT init=%b locked=%b reader=%b client=%b",`EP.hbm_init_complete,`EP.fabric_clock_locked,`EP.reader_started_raw,`EP.client_link_up);
        preload(0);preload(1);
        check_read(0,32'h4e545243);check_read(4,1);
        stage('h100,8,{HOST_BASE,64'h10000000,64'd1,SESSION});send_write('hc,1,15,1,0);
        wait(cfg_responses==1);check_read(8,9);check_read('h28,0);send_write('h18,1,15,1,0);check_read(8,0);
        transfer;transfer_id=1;transfer;
        require(commands==2&&acks==2&&results==2&&dma_b==2&&h_reads[0]>0&&h_reads[1]>0&&h_beats[0]==2&&h_beats[1]==4,"integrated accounting");
        require(!nonposted_pending&&tag_count==rc_count&&aw_count==b_count&&ar_count==r_count,"BAR accounting");
        $display("HBM_TLP_PASS checks=%0d transfers=2 bytes=192 requests=%0d completions=%0d cfg=%0d cmds=%0d ack=%0d results=%0d dma_b=%0d hbm_beats=%0d",checks,rq_count,rc_count,cfg_responses,commands,acks,results,dma_b,h_beats[0]+h_beats[1]);
        phase=2;
    endtask
    initial begin #300000;$fatal(1,"HBT WATCHDOG");end
endmodule
`undef EP
`undef RP
