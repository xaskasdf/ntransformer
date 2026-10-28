`timescale 1ns/1ps
`default_nettype none
// Real V3, unchanged source/writer and frozen readers across 250/200 MHz CDC. AXI memory,
// mapping completion, PCIe response and remote ACK are explicit fixtures.
module tb_axi_bridge_hbm_cdc;
    integer test_id=0,seed=0,cycle=0,txn=0;
    reg clk=0,mgmt_clk=0,configuration_rst_n=0,native_reset_n=0;
    always #2 clk=~clk;
    always #5 mgmt_clk=~mgmt_clk;
    reg reader_clk=0,reader_run=1;
    initial begin #0.7;forever begin #2.5;if(reader_run)reader_clk=~reader_clk;end end
    (* ASYNC_REG="TRUE" *) reg [2:0] reader_reset=0,management_reset=0;
    always @(posedge reader_clk or negedge configuration_rst_n)
        if(!configuration_rst_n)reader_reset<=0;else reader_reset<={reader_reset[1:0],1'b1};
    always @(posedge mgmt_clk or negedge configuration_rst_n)
        if(!configuration_rst_n)management_reset<=0;else management_reset<={management_reset[1:0],1'b1};
    wire cold_rst_n,pcie_link,pcie_stop,guard_fault;
    ntrpu_axi_bridge_guard_v3 #(.SHARED_NATIVE_RESET(1)) guard(
        .mgmt_clk(mgmt_clk),.axi_clk(clk),.cold_rst_n(configuration_rst_n),
        .native_link_up(native_reset_n),.native_axi_reset_n(native_reset_n),
        .native_ctl_reset_n(native_reset_n),.perst_n(native_reset_n),
        .client_cold_rst_n(cold_rst_n),.client_link_up(pcie_link),.client_stop(pcie_stop),
        .clock_alive(),.armed(),.fault(guard_fault),.fault_code());
    reg reader_started_raw=0,reader_fault_in=0,cmd_hold=0,injection=0;
    wire [1:0] frame_fault;
    wire reader_fault_raw=reader_fault_in|| |frame_fault;
    wire reader_fault,reader_started,cdc_reader_stop,reader_clock_alive,reader_guard_armed,reader_management_fault;
    wire source_owned,source_fault,reader_stop;
    localparam [63:0] SESSION=64'h12345678abcdef09;
    reg cfg_valid=0,cfg_rsp_ready=0,cmd_valid=0,map_rsp_valid=0;
    wire cfg_ready,cfg_rsp_valid,cfg_rsp_error,cmd_ready,map_valid,map_rsp_ready;
    wire [63:0] map_session,map_generation,map_axi_base,map_pcie_base;
    reg [127:0] cmd_tag={SESSION,64'h100000001};
    reg [63:0] cmd_source=64'h1000;
    reg [12:0] cmd_bytes=64;
    wire reject_valid;
    reg reject_ready=0;
    wire [3:0] reject_code;
    wire [127:0] reject_tag;
    wire src_req_valid,src_req_ready,src_valid,src_ready,src_last,src_cpl_valid,src_cpl_ready;
    wire [127:0] src_req_tag,src_tag,src_cpl_tag;
    wire [63:0] src_req_address;
    wire [12:0] src_req_bytes,src_cpl_bytes;
    wire [511:0] src_data;
    wire [1:0] src_cpl_status;
    wire [47:0] m_awaddr;
    wire [3:0] m_awid;
    wire [3:0] m_bid=0;
    wire [7:0] m_awlen;
    wire [2:0] m_awsize;
    wire [1:0] m_awburst;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    wire m_awvalid,m_wvalid,m_wlast,m_bready,ack_ready,result_valid,owned,fault,window_held;
    reg m_awready=0,m_wready=0,m_bvalid=0,ack_valid=0,result_ready=0;
    reg [1:0] m_bresp=0;
    reg [127:0] ack_tag=0;
    wire [127:0] result_tag;
    wire [12:0] result_bytes;
    wire [7:0] fault_code;
    wire link_up=pcie_link&&reader_started&&!source_fault&&!reader_fault;
    wire stop=pcie_stop||source_fault||reader_fault;
    ntrpu_axi_bridge_writer #(.SOURCE_ADDR_BITS(34)) writer(
        .cfg_session(SESSION),.cfg_generation(64'd1),.cfg_axi_base(64'h10000000),
        .cfg_pcie_base(64'h1234567800001000),.map_ready(1'b1),.map_rsp_ok(1'b1),
        .cmd_generation(64'd1),.cmd_offset(64'd0),.ack_generation(64'd1),.ack_bytes(cmd_bytes),.*);

    wire [1:0] r_cmd_valid,r_cmd_ready,r_data_valid,r_data_ready,r_data_last,r_cpl_valid,r_cpl_ready;
    wire [127:0] r_cmd_addr,r_cmd_tag,r_data_tag,r_cpl_tag;
    wire [15:0] r_cmd_len;
    wire [1023:0] r_data;
    wire [3:0] r_cpl_status;
    wire [25:0] r_cpl_bytes;
    wire [1:0] f_cmd_ready,f_data_valid,f_data_last,f_cpl_valid;
    wire [127:0] f_data_tag,f_cpl_tag;
    wire [3:0] f_cpl_status;
    wire [25:0] f_cpl_bytes;
    wire lane=cmd_source[33];
    wire [1:0] d_cmd_valid,d_cmd_ready,d_data_valid,d_data_ready,d_data_last,d_cpl_valid,d_cpl_ready;
    wire [127:0] d_cmd_addr,d_cmd_tag,d_data_tag,d_cpl_tag;
    wire [15:0] d_cmd_len;
    wire [1023:0] d_data;
    wire [3:0] d_cpl_status;
    wire [25:0] d_cpl_bytes;
    ntrpu_axi_bridge_hbm_cdc transport(
        .p_clk(clk),.r_clk(reader_clk),.mgmt_clk(mgmt_clk),
        .p_cold_rst_n(cold_rst_n),.r_cold_rst_n(reader_reset[2]),.m_cold_rst_n(management_reset[2]),
        .p_stop(reader_stop),.r_started(reader_started_raw),.r_fault(reader_fault_raw),
        .p_started(reader_started),.p_fault(reader_fault),.r_stop(cdc_reader_stop),
        .clock_alive(reader_clock_alive),.armed(reader_guard_armed),.management_fault(reader_management_fault),
        .p_cmd_valid(r_cmd_valid),.r_cmd_valid(d_cmd_valid),
        .p_cmd_ready(r_cmd_ready),.r_cmd_ready(d_cmd_ready),
        .p_cmd_addr(r_cmd_addr),.r_cmd_addr(d_cmd_addr),
        .p_cmd_tag(r_cmd_tag),.r_cmd_tag(d_cmd_tag),
        .p_cmd_len(r_cmd_len),.r_cmd_len(d_cmd_len),
        .p_data_valid(r_data_valid),.r_data_valid(d_data_valid),
        .p_data_ready(r_data_ready),.r_data_ready(d_data_ready),
        .p_data_last(r_data_last),.r_data_last(d_data_last),
        .p_data(r_data),.r_data(d_data),
        .p_data_tag(r_data_tag),.r_data_tag(d_data_tag),
        .p_cpl_valid(r_cpl_valid),.r_cpl_valid(d_cpl_valid),
        .p_cpl_ready(r_cpl_ready),.r_cpl_ready(d_cpl_ready),
        .p_cpl_tag(r_cpl_tag),.r_cpl_tag(d_cpl_tag),
        .p_cpl_status(r_cpl_status),.r_cpl_status(d_cpl_status),
        .p_cpl_bytes(r_cpl_bytes),.r_cpl_bytes(d_cpl_bytes));
    assign d_cmd_ready=cmd_hold?2'b00:f_cmd_ready;
    assign d_data_valid=(injection&&test_id==8)?(lane?2'b01:2'b10):f_data_valid;
    assign d_data_last=f_data_last ^ ((test_id==12)?(lane?2'b10:2'b01):2'b00);
    assign d_data_tag=f_data_tag ^ ((test_id==4)?(128'd1<<(lane*64+40)):128'd0);
    assign d_cpl_valid=(injection&&(test_id==15||test_id==17))?(lane?2'b10:2'b01):f_cpl_valid;
    assign d_cpl_tag=f_cpl_tag ^ ((test_id==5)?(128'd1<<(lane*64+40)):128'd0);
    assign d_cpl_bytes=f_cpl_bytes ^ ((test_id==6)?(26'd64<<(lane*13)):26'd0);
    assign d_cpl_status=f_cpl_status | ((test_id==7)?(4'd2<<(lane*2)):4'd0);
    wire [127:0] retire_tag=result_tag ^ ((test_id==14)?128'h10000000000000000000000000:128'd0);
    ntrpu_axi_bridge_hbm_source source_join(
        .enable(pcie_link),.stop(pcie_stop||fault),.retire_valid(result_valid&&result_ready),
        .owned(source_owned),.fault(source_fault),.*);
    wire [1:0] arvalid,arready,rvalid,rready,rlast;
    wire [67:0] araddr;
    wire [15:0] arlen;
    wire [1023:0] rdata;
    wire [3:0] rresp;
    for(genvar c=0;c<2;c++)begin: readers
        ntrpu_hbm_read_frames frame(
            .clk(reader_clk),.rst_n(reader_reset[2]),.admit(reader_started_raw&&!cdc_reader_stop),
            .cmd_valid(d_cmd_valid[c]&&!cmd_hold),.cmd_ready(f_cmd_ready[c]),
            .cmd_addr(d_cmd_addr[c*64+:64]),.cmd_tag(d_cmd_tag[c*64+:64]),.cmd_len(d_cmd_len[c*8+:8]),
            .data_valid(f_data_valid[c]),.data_ready(d_data_ready[c]),.data(d_data[c*512+:512]),
            .data_tag(f_data_tag[c*64+:64]),.data_last(f_data_last[c]),
            .cpl_valid(f_cpl_valid[c]),.cpl_ready(d_cpl_ready[c]),.cpl_tag(f_cpl_tag[c*64+:64]),
            .cpl_status(f_cpl_status[c*2+:2]),.cpl_bytes(f_cpl_bytes[c*13+:13]),.idle(),.fault(frame_fault[c]),
            .arvalid(arvalid[c]),.arready(arready[c]),.araddr(araddr[c*34+:34]),.arlen(arlen[c*8+:8]),
            .rvalid(rvalid[c]),.rready(rready[c]),.rdata(rdata[c*512+:512]),.rid(4'd0),
            .rresp(rresp[c*2+:2]),.rlast(rlast[c]));
        hbm_frontier_memory #(.LANE(c)) memory(
            .clk(reader_clk),.rst_n(reader_reset[2]),.seed(seed),.bad_resp(test_id==9),
            .arvalid(arvalid[c]),.arready(arready[c]),.araddr(araddr[c*34+:34]),.arlen(arlen[c*8+:8]),
            .rvalid(rvalid[c]),.rready(rready[c]),.rdata(rdata[c*512+:512]),.rresp(rresp[c*2+:2]),.rlast(rlast[c]));
    end
    task require(input bit ok,input string message);
        if(ok!==1'b1)$fatal(1,"HS CHECK case=%0d seed=%0d %s",test_id,seed,message);
    endtask
    task tick(input integer n);repeat(n)@(negedge clk);endtask
    function automatic [7:0] pattern(input [63:0] address,input integer offset);
        return 8'((offset*37)^(offset>>7)^address[39:32]^address[15:8]^8'h5a);
    endfunction
    integer aw_count=0,w_count=0,b_count=0,ack_count=0,result_count=0,cmd_count=0;
    reg aw_stalled=0,w_stalled=0,read_stalled=0;
    reg [47:0] saved_aw;
    reg [511:0] saved_w;
    reg saved_last;
    reg [127:0] saved_addr,saved_tag;
    reg [15:0] saved_len;
    always @(posedge clk)begin
        cycle++;
        if(cold_rst_n)begin
            if(aw_stalled)require(m_awvalid&&m_awaddr==saved_aw,"AW survives fault/stall");
            if(w_stalled)require(m_wvalid&&m_wdata==saved_w&&m_wlast==saved_last,"W survives fault/stall");
            if(read_stalled)require(|r_cmd_valid&&r_cmd_addr==saved_addr&&r_cmd_tag==saved_tag&&r_cmd_len==saved_len,"HBM command survives fault/stall");
            aw_stalled<=m_awvalid&&!m_awready;saved_aw<=m_awaddr;
            w_stalled<=m_wvalid&&!m_wready;saved_w<=m_wdata;saved_last<=m_wlast;
            read_stalled<=|(r_cmd_valid&~r_cmd_ready);saved_addr<=r_cmd_addr;saved_tag<=r_cmd_tag;saved_len<=r_cmd_len;
            for(integer c=0;c<2;c++)if(r_cmd_valid[c]&&r_cmd_ready[c])begin
                require(c==integer'(cmd_source[33])&&r_cmd_addr[c*64+:64]==cmd_source&&r_cmd_tag[c*64+:64]==cmd_tag[63:0]&&r_cmd_len[c*8+:8]==cmd_bytes/64-1,"HBM command identity/range");
                cmd_count++;$display("HS CMD txn=%0d lane=%0d addr=%h tag=%h len=%0d",txn,c,r_cmd_addr[c*64+:64],cmd_tag,r_cmd_len[c*8+:8]);
            end
            if(m_awvalid&&m_awready)begin
                require(m_awaddr==48'h10000000&&m_awlen==cmd_bytes/64-1&&m_awsize==6&&m_awburst==1&&m_awid==0,"PCIe AW");
                aw_count++;$display("HS AW txn=%0d addr=%h len=%0d",txn,m_awaddr,m_awlen);
            end
            if(m_wvalid&&m_wready)begin
                require(m_wstrb==64'hffffffffffffffff&&m_wlast==(w_count+1==cmd_bytes/64),"PCIe W framing");
                for(integer b=0;b<64;b++)require(m_wdata[b*8+:8]==pattern(cmd_source,w_count*64+b),"HBM to PCIe bytes");
                $display("HS W txn=%0d index=%0d data=%h last=%b",txn,w_count,m_wdata,m_wlast);w_count++;
            end
            if(m_bvalid&&m_bready)b_count++;
            if(ack_valid&&ack_ready)ack_count++;
            if(result_valid&&result_ready)begin
                require(result_tag==cmd_tag&&result_bytes==cmd_bytes,"full result identity");
                result_count++;$display("HS RESULT txn=%0d tag=%h bytes=%0d",txn,result_tag,result_bytes);
            end
        end else begin aw_stalled<=0;w_stalled<=0;read_stalled<=0;end
    end
    task configure;
        cfg_valid=1;do @(posedge clk);while(!cfg_ready);@(negedge clk);cfg_valid=0;
        wait(map_valid);tick(2);map_rsp_valid=1;do @(posedge clk);while(!map_rsp_ready);
        @(negedge clk);map_rsp_valid=0;wait(cfg_rsp_valid);require(!cfg_rsp_error,"mapping fixture");
        tick(2);cfg_rsp_ready=1;@(posedge clk);@(negedge clk);cfg_rsp_ready=0;
    endtask
    task submit;
        tick(2);cmd_valid=1;do @(posedge clk);while(!cmd_ready);@(negedge clk);cmd_valid=0;
    endtask
    task write_and_ack;
        wait(m_awvalid&&m_wvalid);tick(3+seed);m_wready=1;tick(4+seed);m_awready=1;
        wait(aw_count==1&&w_count==cmd_bytes/64);tick(3);
        require(owned&&source_owned&&!result_valid,"local HBM CPL and AXI writes do not retire");
        m_bvalid=1;do @(posedge clk);while(!m_bready);@(negedge clk);m_bvalid=0;
        tick(4);require(owned&&source_owned&&!result_valid,"B alone cannot retire");
        ack_tag=cmd_tag;if(test_id==20)ack_tag=cmd_tag^128'h10000000000000000000000000;
        ack_valid=1;do @(posedge clk);while(!ack_ready);@(negedge clk);ack_valid=0;
    endtask
    task retained;
        wait(source_fault&&fault);tick(8);
        require(owned&&source_owned&&window_held&&!cmd_ready&&cold_rst_n,"failed context remains owned");
        reader_fault_in=0;reader_started_raw=1;injection=0;native_reset_n=1;
        tick(100);require(source_fault&&fault&&owned&&source_owned&&!cmd_ready,"restoring health cannot retire");
        $display("HS RETAIN txn=%0d owned=%b source=%b fault=%b cold=%b",txn,owned,source_owned,fault,cold_rst_n);
    endtask
    initial begin
        if($value$plusargs("case=%d",test_id))begin end
        if($value$plusargs("seed=%d",seed))begin end
`ifndef XSIM
        $dumpfile("hbm_cdc.vcd");$dumpvars(1,tb_axi_bridge_hbm_cdc);
`endif
        tick(12);configuration_rst_n=1;native_reset_n=1;
        wait(pcie_link);tick(5);require(!cfg_ready&&!cmd_ready,"reader readiness gates startup");
        reader_started_raw=1;wait(reader_started);tick(4);configure();
        if(test_id==16)begin
            cmd_source=64'h400000000;submit();wait(reject_valid);tick(2);
            require(reject_code==3&&!owned&&!source_owned&&cmd_count==0,"reject HBM high address before truncation");
            reject_ready=1;tick(2);reject_ready=0;
        end else begin
            for(txn=0;txn<(test_id==0?4:1);txn++)begin
                aw_count=0;w_count=0;b_count=0;ack_count=0;result_count=0;cmd_count=0;
                m_awready=0;m_wready=0;result_ready=0;
                cmd_tag={SESSION,64'h100000001+64'(txn)};
                cmd_source=(txn%2?64'h200000000:64'h100000000)+64'h1000+64'(txn*4096);
                cmd_bytes=test_id==0?13'(64<<txn):13'd128;
                if(test_id==0&&txn==3)cmd_bytes=4096;
                if(test_id==3||test_id==13||test_id==22)cmd_hold=1;
                submit();wait(source_owned);tick(1);
                if(test_id==1)begin reader_fault_in=1;retained();end
                else if(test_id==3||test_id==13)begin
                    wait(|d_cmd_valid);tick(3);
                    if(test_id==3)begin #0.1;native_reset_n=0;#0.1;native_reset_n=1;end
                    else reader_fault_in=1;
                    wait(source_fault);tick(5);require(|d_cmd_valid,"CDC owns stalled command");retained();
                end else if(test_id==22||test_id==23)begin
                    if(test_id==22)begin wait(|d_cmd_valid);@(negedge reader_clk);reader_run=0;end
                    else begin wait(m_awvalid&&m_wvalid);@(posedge reader_clk);#0.1;reader_run=0;end
                    wait(reader_management_fault);tick(5);
                    require(!reader_clock_alive&&source_owned&&owned,"stopped reader clock retains debt");
                    $display("HC WATCHDOG case=%0d level=%b management=%b",test_id,reader_clk,reader_management_fault);
                    reader_run=1;retained();
                    require(reader_fault&&!reader_started,"reader clock restart cannot reopen epoch");
                end else if(test_id==8||test_id==17)begin injection=1;retained();end
                else if((test_id>=4&&test_id<=7)||test_id==9||test_id==12)begin
                    retained();require(aw_count==0&&w_count==0&&!m_awvalid&&!m_wvalid,"bad source cannot launch PCIe");
                end else if(test_id==2||test_id==11||test_id==15)begin
                    wait(m_awvalid&&m_wvalid);tick(3);
                    if(test_id==11)reader_started_raw=0;
                    else if(test_id==15)injection=1;
                    else reader_fault_in=1;
                    wait(source_fault);tick(4);require(m_awvalid&&m_wvalid,"existing offers survive HBM failure");
                    m_awready=1;m_wready=1;wait(aw_count==1&&w_count==cmd_bytes/64);retained();
                end else begin
                    if(test_id==21)m_bresp=2;
                    write_and_ack();
                    if(test_id==20||test_id==21)retained();
                    else begin
                        wait(result_valid);tick(5);require(owned&&source_owned,"result wait owns both contexts");
                        if(test_id==10)begin reader_fault_in=1;wait(source_fault);end
                        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;tick(3);
                        if(test_id==10||test_id==14)retained();
                        else require(!owned&&!source_owned&&!fault&&!source_fault&&result_count==1,"coordinated retirement");
                    end
                end
            end
        end
        $display("HC CLOCKS p=250 r=200 m=100 phase_ps=700");
        $display("HS PASS case=%0d seed=%0d",test_id,seed);$finish;
    end
    initial begin #250000;$fatal(1,"HS TIMEOUT case=%0d seed=%0d",test_id,seed);end
endmodule

module hbm_frontier_memory #(parameter integer LANE=0)(
    input wire clk,rst_n,input wire [31:0] seed,input wire bad_resp,
    input wire arvalid,output wire arready,input wire [33:0] araddr,input wire [7:0] arlen,
    output reg rvalid=0,input wire rready,output reg [511:0] rdata=0,
    output wire [1:0] rresp,output reg rlast=0
);
    integer cycle=0,index=0,count=0;
    reg busy=0;
    reg [33:0] address=0;
    assign arready=rst_n&&!busy&&(cycle+seed+LANE)%4!=0;
    assign rresp=bad_resp?2'd2:2'd0;
    always @(posedge clk)begin
        cycle++;
        if(!rst_n)begin busy<=0;rvalid<=0;index<=0;count<=0;rlast<=0;end
        else begin
            if(arvalid&&arready)begin busy<=1;address<=araddr;count<=integer'(arlen)+1;index<=0;end
            if(rvalid&&rready)begin rvalid<=0;index<=index+1;if(rlast)busy<=0;end
            if(busy&&!rvalid&&(cycle+seed+LANE)%3!=0)begin
                rvalid<=1;rlast<=index+1==count;
                for(integer b=0;b<64;b++)rdata[b*8+:8]<=8'(((index*64+b)*37)^((index*64+b)>>7)^{6'b0,address[33:32]}^address[15:8]^8'h5a);
            end
        end
    end
endmodule
`default_nettype wire
