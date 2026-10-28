`timescale 1ns/1ps
`define EP board.AXI_PCIE_EP
`define RP board.RP
// Only PCIe requests drive the CSR. Downstream offers stay held in this fixture.
module ntrpu_bar_tlp_probe;
    wire clk=`EP.axi_aclk_out;
    reg cold_rst_n=0,mgmt_clk=0,reader_clk=0,mem_clk=0;
    always #5 mgmt_clk=~mgmt_clk;
    always #2.5 reader_clk=~reader_clk;
    always #1.111 mem_clk=~mem_clk;
    wire [3:0] telemetry_clk={mem_clk,reader_clk,mgmt_clk,clk};
    wire [511:0] telemetry_data;
    for(genvar i=0;i<16;i=i+1)assign telemetry_data[32*i+:32]=32'h5a010000+i;
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
    task run_probe;
        integer before_w,before_r;
        require(`RP.tx_usrapp.BAR_INIT_P_BAR_ENABLED[0]==2||`RP.tx_usrapp.BAR_INIT_P_BAR_ENABLED[0]==3,"memory BAR enabled");
        bar_base={`RP.tx_usrapp.BAR_INIT_P_BAR_ENABLED[0]==3?`RP.tx_usrapp.BAR_INIT_P_BAR[1][31:0]:32'b0,`RP.tx_usrapp.BAR_INIT_P_BAR[0][31:0]};
        @(negedge clk);cold_rst_n=1;repeat(8)@(negedge clk);phase=1;
        $display("BT START base=%h",bar_base);
        $display("BT GROUP identity");check_read(0,32'h4e545243);check_read(4,1);check_read('h38,0);
        $display("BT GROUP staging_strobes");
        for(integer i=0;i<8;i++)begin values[i]=32'ha5100000+32'(i*'h10101);send_write('h100+4*i,values[i],15,1,0);check_read('h100+4*i,values[i]);end
        send_write('h104,32'hdeadbeef,5,1,0);values[1]=(values[1]&32'hff00ff00)|32'h00ad00ef;check_read('h104,values[1]);check_read('h38,32'hffffffff);
        $display("BT GROUP rejection");before_w=csr_w;before_r=csr_r;
        send_write('h10104,32'hbad00001,15,1,3);send_read('h10104,1,0);
        send_write('h100,32'hbad00002,15,2,3);send_read('h100,2,0);
        require(csr_w==before_w&&csr_r==before_r,"rejected native requests never reached CSR");
        check_read('h100,values[0]);check_read('h104,values[1]);send_read('hffc,1,0);
        $display("BT GROUP snapshot_lanes");send_write('h20,1,15,1,0);
        send_read(8,1,1);require(completion_data[6]&&!completion_data[5],"snapshot published");check_read('h24,1);
        for(integer i=0;i<16;i++)check_read('h300+4*i,32'h5a010000+i);
        send_write('h30,1,15,1,0);check_read(8,0);
        $display("BT GROUP commit_retention");send_write('hc,1,15,1,0);check_read(8,1);check_read('h38,0);
        for(integer i=0;i<8;i++)check_read('h400+4*i,values[i]);
        send_write('h100,32'habcd1234,15,1,0);check_read('h400,values[0]);send_write('hc,1,15,1,2);
        check_read('h48,1);require(`EP.control.cfg_valid&&`EP.control.cfg_session=={values[1],values[0]},"held immutable configuration offer");
        require(!nonposted_pending&&tag_count==rc_count&&ar_count==r_count&&aw_count==b_count,"complete native accounting");
        $display("BAR_TLP_PASS groups=5 checks=%0d rq=%0d rc=%0d tags=%0d aw=%0d b=%0d ar=%0d r=%0d csr_w=%0d csr_r=%0d",checks,rq_count,rc_count,tag_count,aw_count,b_count,ar_count,r_count,csr_w,csr_r);
        phase=2;
    endtask
    initial begin #240000;$fatal(1,"BT WATCHDOG");end
endmodule
`undef EP
`undef RP
