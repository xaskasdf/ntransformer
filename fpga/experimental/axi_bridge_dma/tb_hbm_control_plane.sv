`timescale 1ns/1ps
`default_nettype none
module tb_hbm_control_plane; hbm_control_harness bench(); endmodule
module tb_hbm_control_writer; hbm_control_harness #(.REAL_WRITER(1)) bench(); endmodule

module hbm_control_harness #(parameter integer REAL_WRITER=0);
    integer seed=0,checks=0,groups=0;
    reg clk=0,cold_rst_n=0,bus_rst_n=0;
    always #2 clk=~clk;
    reg [11:0] s_awaddr=0,s_araddr=0;
    wire [2:0] s_awprot=0,s_arprot=0;
    reg s_awvalid=0,s_wvalid=0,s_bready=0,s_arvalid=0,s_rready=0;
    reg [31:0] s_wdata=0;
    reg [3:0] s_wstrb=0;
    wire s_awready,s_wready,s_arready,s_bvalid,s_rvalid;
    wire [1:0] s_bresp,s_rresp;
    wire [31:0] s_rdata;
    wire cfg_valid,cmd_valid,ack_valid,cfg_ready,cmd_ready,ack_ready;
    wire [63:0] cfg_session,cfg_generation,cfg_axi_base,cfg_pcie_base;
    wire [127:0] cmd_tag,ack_tag;
    wire [63:0] cmd_generation,cmd_source,cmd_offset,ack_generation;
    wire [12:0] cmd_bytes,ack_bytes;
    wire cfg_rsp_valid,cfg_rsp_error,reject_valid,result_valid;
    wire cfg_rsp_ready,reject_ready,result_ready;
    wire [3:0] reject_code;
    wire [127:0] reject_tag,result_tag;
    wire [12:0] result_bytes;
    reg [2:0] other_clocks=0;
    wire [3:0] telemetry_clk={other_clocks,clk};
    reg [3:0] telemetry_cold_rst_n=0,run_clock=4'hf;
    wire [511:0] telemetry_data;
    initial begin #0.01;#(seed*0.17);forever begin #5;if(run_clock[1])other_clocks[0]=~other_clocks[0];else other_clocks[0]=0;end end
    initial begin #0.02;#(seed*0.23);forever begin #3;if(run_clock[2])other_clocks[1]=~other_clocks[1];else other_clocks[1]=0;end end
    initial begin #0.03;#(seed*0.31);forever begin #1.111;if(run_clock[3])other_clocks[2]=~other_clocks[2];else other_clocks[2]=0;end end
    for(genvar d=0;d<4;d=d+1) begin: telemetry_fixture
        reg [31:0] count=0;
        always @(posedge telemetry_clk[d] or negedge telemetry_cold_rst_n[d])
            if(!telemetry_cold_rst_n[d])count<=0;else count<=count+1;
        assign telemetry_data[d*128+:128]={count,~count,count^32'h1eadbeef,32'hfeed0000+32'(d)};
    end
    ntrpu_hbm_control_plane dut(.*);

    reg f_cfg_ready=0,f_cmd_ready=0,f_ack_ready=0;
    reg f_cfg_rsp_valid=0,f_cfg_rsp_error=0,f_reject_valid=0,f_result_valid=0;
    reg [3:0] f_reject_code=0;
    reg [127:0] f_reject_tag=0,f_result_tag=0;
    reg [12:0] f_result_bytes=0;
    wire map_valid,map_rsp_ready,src_req_valid,src_ready,src_cpl_ready;
    reg map_ready=0,map_rsp_valid=0,src_req_ready=0,src_valid=0,src_last=0,src_cpl_valid=0;
    wire [63:0] map_session,map_generation,map_axi_base,map_pcie_base,src_req_address;
    wire [127:0] src_req_tag;
    wire [12:0] src_req_bytes;
    reg [511:0] src_data=0;
    reg [127:0] src_tag=0,src_cpl_tag=0;
    reg [12:0] src_cpl_bytes=0;
    wire [47:0] m_awaddr;
    wire [3:0] m_awid;
    wire [7:0] m_awlen;
    wire [2:0] m_awsize;
    wire [1:0] m_awburst;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    wire m_awvalid,m_wvalid,m_wlast,m_bready,owned,fault,window_held;
    wire [7:0] fault_code;
    reg m_awready=0,m_wready=0,m_bvalid=0;
    if(REAL_WRITER) begin: actual_writer
        ntrpu_axi_bridge_writer #(.SOURCE_ADDR_BITS(34)) writer(
            .link_up(1'b1),.stop(1'b0),.map_rsp_ok(1'b1),.src_cpl_status(2'd0),
            .m_bresp(2'd0),.m_bid(4'd0),.*);
    end else begin: explicit_endpoint_fixture
        assign cfg_ready=f_cfg_ready;assign cmd_ready=f_cmd_ready;assign ack_ready=f_ack_ready;
        assign cfg_rsp_valid=f_cfg_rsp_valid;assign cfg_rsp_error=f_cfg_rsp_error;
        assign reject_valid=f_reject_valid;assign reject_code=f_reject_code;assign reject_tag=f_reject_tag;
        assign result_valid=f_result_valid;assign result_tag=f_result_tag;assign result_bytes=f_result_bytes;
    end

    task check(input bit ok,input string message);
        checks++;
        if(ok!==1'b1)$fatal(1,"CP CHECK writer=%0d seed=%0d %s",REAL_WRITER,seed,message);
    endtask
    task group_done(input string name);
        groups++;$display("CP GROUP writer=%0d seed=%0d name=%s",REAL_WRITER,seed,name);
    endtask
    task tick(input integer n);repeat(n)@(negedge clk);endtask
    task send_aw(input [11:0] address);
        s_awaddr=address;s_awvalid=1;
        do @(posedge clk);while(!s_awready);
        @(negedge clk);s_awvalid=0;
    endtask
    task send_w(input [31:0] data,input [3:0] strobe);
        s_wdata=data;s_wstrb=strobe;s_wvalid=1;
        do @(posedge clk);while(!s_wready);
        @(negedge clk);s_wvalid=0;
    endtask
    task post_write(input [11:0] address,input [31:0] data,input [3:0] strobe,input integer order);
        tick(1);
        fork
            begin tick(order==1?4+seed:0);send_aw(address);end
            begin tick(order==2?4+seed:0);send_w(data,strobe);end
        join
        wait(s_bvalid);@(negedge clk);
    endtask
    task write_reg(input [11:0] address,input [31:0] data,input [3:0] strobe,input [1:0] response,input integer order);
        post_write(address,data,strobe,order);
        check(s_bresp==response,$sformatf("BRESP address=%h expected=%d got=%d",address,response,s_bresp));
        tick(2+seed);s_bready=1;@(posedge clk);@(negedge clk);s_bready=0;
    endtask
    task read_reg(input [11:0] address,input [1:0] response,output [31:0] data);
        tick(1);s_araddr=address;s_arvalid=1;
        do @(posedge clk);while(!s_arready);
        @(negedge clk);s_arvalid=0;wait(s_rvalid);
        check(s_rresp==response,$sformatf("RRESP address=%h",address));data=s_rdata;
        tick(2+seed);s_rready=1;@(posedge clk);@(negedge clk);s_rready=0;
    endtask
    task expect_reg(input [11:0] address,input [31:0] expected);
        reg [31:0] value;read_reg(address,0,value);
        check(value==expected,$sformatf("read address=%h expected=%h got=%h",address,expected,value));
    endtask
    task stage(input [11:0] address,input integer words,input [511:0] image);
        for(integer j=0;j<words;j=j+1)write_reg(address+12'(j*4),image[j*32+:32],15,0,(j+seed)%3);
    endtask
    task hot_reset;
        tick(1);bus_rst_n=0;tick(3+seed);
        check(!s_bvalid&&!s_rvalid&&!s_awready&&!s_wready&&!s_arready,"transport reset blocks bus");
        bus_rst_n=1;tick(2);
    endtask
    task wait_status(input integer bit_index,input bit expected);
        reg [31:0] value;
        for(integer j=0;j<100;j=j+1)begin
            read_reg(12'h008,0,value);
            if(value[bit_index]===expected)return;
        end
        $fatal(1,"CP status timeout bit=%0d",bit_index);
    endtask
    task snapshot_read(output [511:0] value,input bit verify_pattern);
        reg [31:0] word_value,counter;
        for(integer j=0;j<16;j=j+1)begin read_reg(12'h300+12'(j*4),0,word_value);value[j*32+:32]=word_value;end
        if(verify_pattern)for(integer j=0;j<4;j=j+1)begin
            counter=value[j*128+96+:32];
            check(counter>0&&value[j*128+64+:32]==~counter&&
                  value[j*128+32+:32]==(counter^32'h1eadbeef)&&value[j*128+:32]==32'hfeed0000+32'(j),
                  $sformatf("untorn domain %0d",j));
        end
    endtask

    integer cfg_count=0,cmd_count=0,ack_count=0,result_count=0,reject_count=0;
    integer aw_count=0,w_count=0,b_count=0;
    reg prev_b=0,prev_r=0,prev_cfg=0,prev_cmd=0,prev_ack=0,prev_aw=0,prev_w=0;
    reg [1:0] saved_bresp=0,saved_rresp=0;
    reg [31:0] saved_rdata=0;
    reg [255:0] saved_cfg=0;
    reg [332:0] saved_cmd=0;
    reg [204:0] saved_ack=0;
    reg [47:0] saved_aw=0;
    reg [511:0] saved_w=0;
    reg saved_last=0;
    always @(posedge clk)begin
        if(cold_rst_n&&bus_rst_n)begin
            if(prev_b)check(s_bvalid&&s_bresp==saved_bresp,"B response stable under stall");
            if(prev_r)check(s_rvalid&&s_rresp==saved_rresp&&s_rdata==saved_rdata,"R response stable under stall");
            prev_b<=s_bvalid&&!s_bready;saved_bresp<=s_bresp;
            prev_r<=s_rvalid&&!s_rready;saved_rresp<=s_rresp;saved_rdata<=s_rdata;
        end else begin prev_b<=0;prev_r<=0;end
        if(cold_rst_n)begin
            if(prev_cfg)check(cfg_valid&&{cfg_pcie_base,cfg_axi_base,cfg_generation,cfg_session}==saved_cfg,"CFG offer retained");
            if(prev_cmd)check(cmd_valid&&{cmd_bytes,cmd_offset,cmd_source,cmd_generation,cmd_tag}==saved_cmd,"CMD offer retained");
            if(prev_ack)check(ack_valid&&{ack_bytes,ack_generation,ack_tag}==saved_ack,"ACK offer retained");
            prev_cfg<=cfg_valid&&!cfg_ready;saved_cfg<={cfg_pcie_base,cfg_axi_base,cfg_generation,cfg_session};
            prev_cmd<=cmd_valid&&!cmd_ready;saved_cmd<={cmd_bytes,cmd_offset,cmd_source,cmd_generation,cmd_tag};
            prev_ack<=ack_valid&&!ack_ready;saved_ack<={ack_bytes,ack_generation,ack_tag};
            if(cfg_valid&&cfg_ready)cfg_count++;
            if(cmd_valid&&cmd_ready)cmd_count++;
            if(ack_valid&&ack_ready)ack_count++;
            if(result_valid&&result_ready)result_count++;
            if(reject_valid&&reject_ready)reject_count++;
            if(REAL_WRITER)begin
                if(prev_aw)check(m_awvalid&&m_awaddr==saved_aw,"writer AW survives bus reset/stall");
                if(prev_w)check(m_wvalid&&m_wdata==saved_w&&m_wlast==saved_last,"writer W survives bus reset/stall");
                prev_aw<=m_awvalid&&!m_awready;saved_aw<=m_awaddr;
                prev_w<=m_wvalid&&!m_wready;saved_w<=m_wdata;saved_last<=m_wlast;
                if(m_awvalid&&m_awready)begin
                    check(m_awaddr==48'h10000000&&m_awlen==cmd_bytes/64-1&&m_awsize==6&&m_awburst==1&&m_awid==0,"writer AW framing");aw_count++;
                end
                if(m_wvalid&&m_wready)begin
                    check(m_wstrb==64'hffffffffffffffff&&m_wlast==(w_count+1==cmd_bytes/64),"writer W framing");
                    for(integer j=0;j<64;j=j+1)check(m_wdata[j*8+:8]==8'(w_count*64+j+seed),"writer W payload");
                    w_count++;
                end
                if(m_bvalid&&m_bready)b_count++;
            end
        end else begin prev_cfg<=0;prev_cmd<=0;prev_ack<=0;prev_aw<=0;prev_w<=0;end
    end

    localparam [63:0] SESSION=64'h12345678abcdef09;
    reg [255:0] cfg_image={64'h1234567800001000,64'h10000000,64'd1,SESSION};
    reg [351:0] cmd_image=0;
    reg [223:0] ack_image=0;
    reg [31:0] value;
    reg [511:0] snap_first,snap_second,snap_third;
    task set_command(input [63:0] sequence_id,input [63:0] source_address,input [12:0] bytes,input [63:0] generation);
        cmd_image={19'd0,bytes,64'd0,source_address,generation,SESSION,sequence_id};
        stage(12'h140,11,cmd_image);
    endtask
    task set_ack(input [127:0] tag,input [12:0] bytes);
        ack_image={19'd0,bytes,64'd1,tag};stage(12'h180,7,ack_image);
    endtask
    task event_check(input [31:0] kind,input [127:0] tag,input [31:0] info);
        wait_status(4,1);expect_reg(12'h02c,kind);
        for(integer j=0;j<4;j=j+1)expect_reg(12'h200+12'(j*4),tag[j*32+:32]);
        expect_reg(12'h210,info);
        $display("CP EVENT writer=%0d kind=%0d tag=%h info=%0d",REAL_WRITER,kind,tag,info);
    endtask

    task run_unit;
        expect_reg(0,32'h4e545243);expect_reg(4,1);expect_reg(8,0);
        // Independent AXI read and write channels, with opposite AW/W orderings.
        fork
            write_reg(12'h100,32'h44332211,4'h5,0,1);
            begin reg [31:0] data;read_reg(0,0,data);check(data==32'h4e545243,"parallel read");end
        join
        write_reg(12'h100,32'haabbccdd,4'ha,0,2);expect_reg(12'h100,32'haa33cc11);
        write_reg(12'h100,32'hffffffff,0,0,0);expect_reg(12'h100,32'haa33cc11);
        group_done("axil_transport");
        write_reg(12'h00c,1,15,2,0);write_reg(12'h000,1,15,2,1);
        write_reg(12'h101,1,15,2,2);write_reg(12'hffc,1,15,2,0);
        write_reg(12'h020,1,1,2,0);write_reg(12'h020,2,15,2,0);
        read_reg(12'h001,2,value);read_reg(12'hffc,2,value);
        write_reg(12'h018,1,15,2,0);write_reg(12'h01c,1,15,2,0);write_reg(12'h030,1,15,2,0);
        group_done("decode_partial");

        tick(1);send_aw(12'h104);hot_reset();check(!s_bvalid,"partial AW discarded");
        tick(1);send_w(32'hffffffff,15);hot_reset();check(!s_bvalid,"partial W discarded");
        expect_reg(12'h104,0);
        stage(12'h100,8,cfg_image);post_write(12'h00c,1,15,0);
        check(s_bresp==0&&cfg_valid,"commit before lost B response");hot_reset();
        check(cfg_valid&&cfg_count==0&&{cfg_pcie_base,cfg_axi_base,cfg_generation,cfg_session}==cfg_image,"CFG survives lost B");
        write_reg(12'h100,32'h99887766,15,0,0);write_reg(12'h00c,1,15,2,0);
        check({cfg_pcie_base,cfg_axi_base,cfg_generation,cfg_session}==cfg_image,"shadow does not change active CFG");
        expect_reg(12'h400,SESSION[31:0]);expect_reg(12'h038,15);expect_reg(12'h048,1);
        write_reg(12'h400,1,15,2,0);
        f_cfg_ready=1;wait(cfg_count==1);tick(1);f_cfg_ready=0;
        group_done("descriptor_atomic");
        f_cfg_rsp_error=1;f_cfg_rsp_valid=1;do @(posedge clk);while(!cfg_rsp_ready);
        @(negedge clk);f_cfg_rsp_valid=0;wait_status(3,1);hot_reset();expect_reg(12'h028,1);
        check(!cfg_rsp_ready,"full CFG mailbox backpressures");write_reg(12'h00c,1,15,2,0);
        write_reg(12'h018,1,15,0,0);expect_reg(12'h028,0);write_reg(12'h018,1,15,2,0);
        write_reg(12'h00c,1,15,2,0); // only one freshly written CFG word
        group_done("response_mailboxes");

        set_command(64'h100000001,64'h1000,128,1);
        write_reg(12'h168,32'h00200080,15,0,0);write_reg(12'h010,1,15,2,0);
        write_reg(12'h168,128,15,0,0);write_reg(12'h010,1,15,0,0);
        check(cmd_valid&&cmd_tag=={SESSION,64'h100000001}&&cmd_bytes==128,"command fields");
        write_reg(12'h160,64,15,0,0);hot_reset();
        check(cmd_valid&&cmd_offset==0&&cmd_count==0,"command retained across staging/reset");
        expect_reg(12'h460,0);expect_reg(12'h03c,0);expect_reg(12'h040,15);expect_reg(12'h048,4);
        set_ack({SESSION,64'h100000001},128);write_reg(12'h014,1,15,2,0);
        f_cmd_ready=1;wait(cmd_count==1);tick(1);f_cmd_ready=0;
        write_reg(12'h014,1,15,2,0); // endpoint not ready to accept remote ACK
        f_ack_ready=1;write_reg(12'h198,32'h00200080,15,0,0);write_reg(12'h014,1,15,2,0);
        write_reg(12'h198,128,15,0,0);
        f_ack_ready=1;post_write(12'h014,1,15,0);f_ack_ready=0;
        check(s_bresp==0&&ack_valid&&ack_count==0,"ACK published atomically before stall");
        hot_reset();tick(8);check(ack_valid&&ack_tag=={SESSION,64'h100000001},"ACK persists across reset");
        expect_reg(12'h480,32'h00000001);expect_reg(12'h498,128);expect_reg(12'h044,0);
        f_ack_ready=1;wait(ack_count==1);tick(1);f_ack_ready=0;
        set_ack({SESSION,64'h100000001},128);f_ack_ready=1;write_reg(12'h014,1,15,2,0);f_ack_ready=0;
        read_reg(8,0,value);check(value[7]&&!value[2],"ACK publication is recorded after handshake");
        f_result_tag={SESSION,64'h100000001};f_result_bytes=128;f_result_valid=1;
        do @(posedge clk);while(!result_ready);@(negedge clk);f_result_valid=0;
        event_check(2,{SESSION,64'h100000001},128);hot_reset();
        check(!result_ready&&!reject_ready,"event retained with backpressure");
        s_araddr=12'h210;s_arvalid=1;do @(posedge clk);while(!s_arready);
        @(negedge clk);s_arvalid=0;wait(s_rvalid);check(s_rdata==128,"stalled R before reset");hot_reset();
        event_check(2,{SESSION,64'h100000001},128);write_reg(12'h010,1,15,2,0);
        write_reg(12'h01c,1,15,0,0);write_reg(12'h010,1,15,2,0); // incomplete fresh descriptor
        set_command(64'h100000002,64'h2000,64,1);f_cmd_ready=1;write_reg(12'h010,1,15,0,0);f_cmd_ready=0;
        f_reject_tag={SESSION,64'h100000002};f_reject_code=3;f_reject_valid=1;f_result_valid=1;
        tick(1);check(!result_ready,"reject priority prevents dual response acceptance");
        f_reject_valid=0;f_result_valid=0;
        event_check(1,{SESSION,64'h100000002},3);write_reg(12'h01c,1,15,0,0);
        check(cmd_count==2&&ack_count==1&&result_count==1&&reject_count==1,"one downstream handshake per publication");
        group_done("command_ack_retention");

        write_reg(12'h020,1,15,0,0);wait_status(6,1);expect_reg(12'h024,1);snapshot_read(snap_first,1);
        write_reg(12'h020,1,15,2,0);tick(30);snapshot_read(snap_second,1);check(snap_second==snap_first,"published snapshot frozen during software reads");
        write_reg(12'h030,1,15,0,0);write_reg(12'h030,1,15,2,0);
        group_done("snapshot_rounds");
        for(integer stopped=1;stopped<4;stopped=stopped+1)begin
            run_clock[stopped]=0;tick(5);write_reg(12'h020,1,15,0,0);tick(100);
            read_reg(8,0,value);check(value[5]&&!value[6],"stopped source holds whole round busy");
            read_reg(12'h034,0,value);check(value[3:0]==(4'hf^(4'b0001<<stopped)),"progress identifies missing domain");
            $display("CP STOPPED domain=%0d progress=%h",stopped,value[3:0]);
            snapshot_read(snap_second,0);check(snap_second==snap_first,"no partial publication");
            hot_reset();write_reg(12'h020,1,15,2,0);run_clock[stopped]=1;
            wait_status(6,1);expect_reg(12'h024,32'(stopped+1));snapshot_read(snap_third,1);
            check(snap_third!=snap_first,"next round contains new samples");snap_first=snap_third;
            write_reg(12'h030,1,15,0,0);
        end
        group_done("snapshot_stopped_clocks");
        run_clock[3]=0;tick(5);write_reg(12'h020,1,15,0,0);tick(30);
        cold_rst_n=0;telemetry_cold_rst_n=0;bus_rst_n=0;tick(5);
        cold_rst_n=1;bus_rst_n=1;telemetry_cold_rst_n[0]=1;
        @(negedge telemetry_clk[1]);telemetry_cold_rst_n[1]=1;
        @(negedge telemetry_clk[2]);telemetry_cold_rst_n[2]=1;
        run_clock[3]=1;@(negedge telemetry_clk[3]);telemetry_cold_rst_n[3]=1;
        expect_reg(8,0);expect_reg(12'h024,0);expect_reg(12'h100,0);
        snapshot_read(snap_first,0);check(snap_first==0,"cold reset clears published state");
        write_reg(12'h020,1,15,0,0);wait_status(6,1);expect_reg(12'h024,1);snapshot_read(snap_first,1);
        group_done("hot_cold_reset");
    endtask

    task feed_and_write;
        wait(src_req_valid);tick(3);check(owned&&src_req_tag==cmd_tag&&src_req_bytes==cmd_bytes&&src_req_address==cmd_source,"source descriptor from committed command");
        hot_reset();check(owned&&src_req_valid&&!fault,"owned source offer survives bus reset");
        src_req_ready=1;do @(posedge clk);while(!src_req_valid);@(negedge clk);src_req_ready=0;
        for(integer beat=0;beat<cmd_bytes/64;beat=beat+1)begin
            for(integer j=0;j<64;j=j+1)src_data[j*8+:8]=8'(beat*64+j+seed);
            src_tag=cmd_tag;src_last=beat+1==cmd_bytes/64;src_valid=1;
            do @(posedge clk);while(!src_ready);@(negedge clk);src_valid=0;tick(2+seed);
        end
        src_cpl_tag=cmd_tag;src_cpl_bytes=cmd_bytes;src_cpl_valid=1;
        do @(posedge clk);while(!src_cpl_ready);@(negedge clk);src_cpl_valid=0;
        wait(m_awvalid&&m_wvalid);hot_reset();check(owned&&m_awvalid&&m_wvalid&&!fault,"writer offers survive bus reset");
        m_wready=1;wait(w_count==cmd_bytes/64);tick(3);m_awready=1;wait(aw_count==1);tick(2);
        m_wready=0;m_awready=0;check(owned&&!result_valid,"data transfer alone does not retire");
        m_bvalid=1;do @(posedge clk);while(!m_bready);@(negedge clk);m_bvalid=0;
        tick(8);check(owned&&!result_valid&&!fault,"B alone does not retire");
        $display("CP B_ONLY tag=%h owned=%b result=%b fault=%b",cmd_tag,owned,result_valid,fault);
        read_reg(8,0,value);check(value[1]&&!value[4],"no completion mailbox before remote ACK");
    endtask
    task run_writer;
        stage(12'h100,8,cfg_image);write_reg(12'h00c,1,15,0,0);wait(map_valid);
        check({map_pcie_base,map_axi_base,map_generation,map_session}==cfg_image,"mapping fixture identity");
        hot_reset();check(map_valid&&!fault,"mapping retained across bus reset");
        map_ready=1;@(posedge clk);@(negedge clk);map_ready=0;
        map_rsp_valid=1;do @(posedge clk);while(!map_rsp_ready);@(negedge clk);map_rsp_valid=0;
        wait_status(3,1);expect_reg(12'h028,0);write_reg(12'h018,1,15,0,0);group_done("mapping_commit");
        set_command(64'h100000001,64'h1000,128,1);write_reg(12'h010,1,15,0,0);feed_and_write();
        set_ack({SESSION,64'h100000001},128);write_reg(12'h014,1,15,0,0);
        event_check(2,{SESSION,64'h100000001},128);check(!owned&&!fault&&result_count==1&&b_count==1&&ack_count==1,"B plus ACK moves result into retained mailbox");
        hot_reset();event_check(2,{SESSION,64'h100000001},128);write_reg(12'h01c,1,15,0,0);group_done("writer_transfer");
        set_command(64'h100000002,64'h400000000,64,1);write_reg(12'h010,1,15,0,0);
        event_check(1,{SESSION,64'h100000002},3);check(!owned&&!src_req_valid&&!fault,"high source rejected before truncation");write_reg(12'h01c,1,15,0,0);
        set_command(64'h100000003,64'h1000,64,2);write_reg(12'h010,1,15,0,0);
        event_check(1,{SESSION,64'h100000003},1);check(!owned&&!fault,"wrong generation rejected");write_reg(12'h01c,1,15,0,0);group_done("writer_rejections");
        aw_count=0;w_count=0;set_command(64'h100000004,64'h2000,64,1);write_reg(12'h010,1,15,0,0);feed_and_write();
        set_ack({SESSION,64'h100000005},64);write_reg(12'h014,1,15,0,0);tick(10);
        check(owned&&fault&&fault_code==8&&window_held&&!result_valid,"wrong ACK freezes ownership");
        $display("CP BAD_ACK owned=%b fault=%b code=%0d result=%b",owned,fault,fault_code,result_valid);
        hot_reset();read_reg(8,0,value);check(value[1]&&value[7]&&!value[4]&&owned&&fault,"transport recovery cannot release failed context");
        write_reg(12'h010,1,15,2,0);write_reg(12'h00c,1,15,2,0);
        check(cfg_count==1&&cmd_count==4&&ack_count==2&&result_count==1&&reject_count==2,"real writer counts");group_done("writer_ack_fault");
    endtask
    initial begin
        if($value$plusargs("seed=%d",seed))begin end
        tick(8);cold_rst_n=1;bus_rst_n=1;telemetry_cold_rst_n=15;tick(12+seed);
        if(REAL_WRITER)run_writer();else run_unit();
        check(groups==(REAL_WRITER?4:8),"all groups executed");
        $display("CP PASS writer=%0d seed=%0d groups=%0d checks=%0d cfg=%0d cmd=%0d ack=%0d result=%0d reject=%0d",REAL_WRITER,seed,groups,checks,cfg_count,cmd_count,ack_count,result_count,reject_count);
        $finish;
    end
    initial begin #1000000;$fatal(1,"CP WATCHDOG writer=%0d seed=%0d",REAL_WRITER,seed);end
endmodule
`default_nettype wire
