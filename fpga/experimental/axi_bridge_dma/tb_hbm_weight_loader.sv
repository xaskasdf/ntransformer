`timescale 1ns/1ps
module tb_hbm_weight_loader;
    reg clk=0,mem_clk=0,mem_enable=1;
    always #2 clk=~clk;
    always #1.111 if(mem_enable)mem_clk=~mem_clk;else mem_clk=0;
    reg cold_rst_n=0,bus_rst_n=0,mem_cold_rst_n=0,memory_ready_async=0;
    reg [11:0] s_awaddr=0,s_araddr=0;
    reg [2:0] s_awprot=0,s_arprot=0;
    reg s_awvalid=0,s_wvalid=0,s_bready=0,s_arvalid=0,s_rready=0;
    reg [31:0] s_wdata=0;
    reg [3:0] s_wstrb=0;
    wire s_awready,s_wready,s_arready,s_bvalid,s_rvalid;
    wire [1:0] s_bresp,s_rresp;
    wire [31:0] s_rdata;
    wire [11:0] m_awaddr,m_araddr;
    wire [2:0] m_awprot,m_arprot;
    wire [31:0] m_wdata,m_rdata;
    wire [3:0] m_wstrb;
    wire m_awvalid,m_wvalid,m_arvalid,m_awready,m_wready,m_arready,m_bvalid,m_rvalid,m_bready,m_rready;
    wire [1:0] m_bresp,m_rresp;
    wire sealed,cmd_valid;
    wire [67:0] h_awaddr;
    wire [3:0] h_awburst;
    wire [11:0] h_awid;
    wire [7:0] h_awlen;
    wire [5:0] h_awsize;
    wire [1:0] h_awvalid,h_wvalid,h_wlast,h_bready;
    reg [1:0] h_awready=0,h_wready=0,h_bvalid=0;
    wire [511:0] h_wdata;
    wire [63:0] h_wstrb;
    reg [11:0] h_bid=0;
    reg [3:0] h_bresp=0;
    ntrpu_hbm_weight_loader dut(.*);
    ntrpu_hbm_control_plane control(
        .clk(clk),.cold_rst_n(cold_rst_n),.bus_rst_n(bus_rst_n),
        .s_awaddr(m_awaddr),.s_araddr(m_araddr),.s_awprot(m_awprot),.s_arprot(m_arprot),
        .s_awvalid(m_awvalid),.s_wvalid(m_wvalid),.s_bready(m_bready),.s_arvalid(m_arvalid),.s_rready(m_rready),
        .s_wdata(m_wdata),.s_wstrb(m_wstrb),.s_awready(m_awready),.s_wready(m_wready),.s_arready(m_arready),
        .s_bvalid(m_bvalid),.s_rvalid(m_rvalid),.s_bresp(m_bresp),.s_rresp(m_rresp),.s_rdata(m_rdata),
        .cfg_ready(1'b0),.cmd_ready(1'b0),.ack_ready(1'b0),.cmd_valid(cmd_valid),
        .cfg_rsp_valid(1'b0),.cfg_rsp_error(1'b0),.reject_valid(1'b0),.result_valid(1'b0),
        .reject_code(4'b0),.reject_tag(128'b0),.result_tag(128'b0),.result_bytes(13'b0),
        .telemetry_clk({4{clk}}),.telemetry_cold_rst_n({4{cold_rst_n}}),.telemetry_data(512'b0));
    integer checks=0,groups=0,seed=0,cycle=0,aw_count=0,w_count=0,b_count=0;
    integer aw_seen[0:1],w_seen[0:1],delay_b[0:1];
    reg [33:0] received_address[0:1];
    reg [255:0] received_data[0:1];
    reg hold_b=0,hold_aw=0,hold_w=0;
    reg [1:0] response=0;
    reg bad_id=0;
    reg [31:0] value;
    task require(input bit good,input string why);
        checks++;if(!good)$fatal(1,"WL FAIL group=%0d %s",groups,why);
    endtask
    task ticks(input integer n);repeat(n)@(negedge clk);endtask
    function automatic [255:0] pattern(input integer id);
        for(integer i=0;i<32;i++)pattern[i*8+:8]=8'((id*71+i*37+91)^(i>>3));
    endfunction
    always @(negedge mem_clk)begin
        cycle++;
        for(integer i=0;i<2;i++)begin
            h_awready[i]=mem_cold_rst_n&&!hold_aw&&aw_seen[i]==0&&(cycle+seed)%3!=0;
            h_wready[i]=mem_cold_rst_n&&!hold_w&&w_seen[i]==0&&(cycle+seed)%4!=0;
        end
    end
    always @(posedge mem_clk or negedge mem_cold_rst_n)begin
        if(!mem_cold_rst_n)begin
            h_bvalid<=0;h_bid<=0;h_bresp<=0;
            for(integer i=0;i<2;i++)begin aw_seen[i]=0;w_seen[i]=0;delay_b[i]=0;end
        end else for(integer i=0;i<2;i++)begin
            if(h_awvalid[i]&&h_awready[i])begin
                require(!aw_seen[i]&&h_awid[i*6+:6]==0&&h_awlen[i*4+:4]==0&&h_awsize[i*3+:3]==5&&h_awburst[i*2+:2]==1,"HBM AW shape");
                require(h_awaddr[i*34+33]==i,"HBM stack select");
                received_address[i]=h_awaddr[i*34+:34];aw_seen[i]=1;aw_count++;
                $display("WL AW lane=%0d addr=%h",i,received_address[i]);
            end
            if(h_wvalid[i]&&h_wready[i])begin
                require(!w_seen[i]&&h_wlast[i]&&h_wstrb[i*32+:32]==32'hffffffff,"HBM W shape");
                received_data[i]=h_wdata[i*256+:256];w_seen[i]=1;w_count++;
                $display("WL W lane=%0d data=%h",i,received_data[i]);
            end
            if(aw_seen[i]&&w_seen[i]&&!h_bvalid[i]&&!hold_b)begin
                delay_b[i]++;
                if(delay_b[i]>3+seed)begin h_bvalid[i]<=1;h_bresp[i*2+:2]<=response;h_bid[i*6+:6]<=bad_id?6'd1:6'd0;end
            end
            if(h_bvalid[i]&&h_bready[i])begin
                b_count++;h_bvalid[i]<=0;aw_seen[i]=0;w_seen[i]=0;delay_b[i]=0;
                $display("WL B lane=%0d resp=%h id=%h",i,h_bresp[i*2+:2],h_bid[i*6+:6]);
            end
        end
    end
    // AXI offers and payloads must remain stable under independent backpressure.
    reg [1:0] prev_aw=0,prev_w=0;
    reg [67:0] prev_address=0;
    reg [511:0] prev_data=0;
    always @(posedge mem_clk)begin
        if(mem_cold_rst_n)for(integer i=0;i<2;i++)begin
            if(prev_aw[i])require(h_awvalid[i]&&h_awaddr===prev_address,"stable AW while stalled");
            if(prev_w[i])require(h_wvalid[i]&&h_wdata===prev_data,"stable W while stalled");
        end
        prev_aw<=mem_cold_rst_n?(h_awvalid&~h_awready):0;
        prev_w<=mem_cold_rst_n?(h_wvalid&~h_wready):0;
        prev_address<=h_awaddr;prev_data<=h_wdata;
    end
    task address(input [11:0] a);
        @(negedge clk);s_awaddr=a;s_awvalid=1;
        do @(posedge clk);while(!s_awready);
        @(negedge clk);s_awvalid=0;
    endtask
    task data(input [31:0] d,input [3:0] st);
        @(negedge clk);s_wdata=d;s_wstrb=st;s_wvalid=1;
        do @(posedge clk);while(!s_wready);
        @(negedge clk);s_wvalid=0;
    endtask
    task write_reg(input [11:0] a,input [31:0] d,input [3:0] st,input [1:0] expected);
        if((checks+seed)%2==0)begin address(a);ticks(seed+1);data(d,st);end
        else begin data(d,st);ticks(seed+1);address(a);end
        wait(s_bvalid);ticks(3);
        require(s_bvalid&&s_bresp===expected,"retained MMIO B response");
        s_bready=1;@(negedge clk);s_bready=0;ticks(1);
    endtask
    task read_reg(input [11:0] a,output [31:0] d);
        @(negedge clk);s_araddr=a;s_arvalid=1;
        do @(posedge clk);while(!s_arready);
        @(negedge clk);s_arvalid=0;wait(s_rvalid);ticks(2);
        require(s_rvalid&&s_rresp==0,"MMIO read response");d=s_rdata;
        ticks(2);require(s_rvalid&&s_rdata===d,"retained read data");
        s_rready=1;@(negedge clk);s_rready=0;ticks(1);
    endtask
    task expect_reg(input [11:0] a,input [31:0] d);
        reg [31:0] v;read_reg(a,v);require(v===d,"register content");
    endtask
    task stage(input [63:0] a,input [63:0] tag,input integer p);
        reg [383:0] image;image={pattern(p),tag,a};
        for(integer i=0;i<12;i++)write_reg('h840+i*4,image[i*32+:32],15,0);
    endtask
    task completion_wait(input [63:0] tag,input [3:0] result);
        wait(dut.done);expect_reg('h820,tag[31:0]);expect_reg('h824,tag[63:32]);expect_reg('h828,{28'd0,result});
    endtask
    task reset_all;
        @(negedge clk);cold_rst_n=0;bus_rst_n=0;mem_cold_rst_n=0;memory_ready_async=0;
        hold_b=0;hold_aw=0;hold_w=0;response=0;bad_id=0;mem_enable=1;
        ticks(8);cold_rst_n=1;bus_rst_n=1;mem_cold_rst_n=1;memory_ready_async=1;ticks(20);
    endtask
    task group_pass(input string name);groups++;$display("WL GROUP %0d %s",groups,name);endtask
    initial begin
        if(!$value$plusargs("seed=%d",seed))seed=0;
        $dumpfile("loader.vcd");$dumpvars(0,tb_hbm_weight_loader);
        for(integer i=0;i<2;i++)begin aw_seen[i]=0;w_seen[i]=0;delay_b[i]=0;end
        reset_all;
        expect_reg('h800,'h4e544c44);expect_reg('h804,1);expect_reg(0,'h4e545243);
        write_reg('h10,1,15,2);require(!cmd_valid,"commands blocked before seal");
        write_reg('h80c,1,15,2);write_reg('h818,1,15,0);
        stage('h1000,64'h1234567887654321,0);
        write_reg('h80c,1,7,2);write_reg('h818,1,15,0);
        hold_aw=1;write_reg('h80c,1,15,0);ticks(20);
        require(dut.busy&&!dut.done&&w_count==1&&aw_count==0,"W before stalled AW");
        write_reg('h840,'h9999,15,2);write_reg('h80c,1,15,2);write_reg('h814,1,15,2);
        hold_aw=0;completion_wait(64'h1234567887654321,0);
        require(received_address[0]=='h1000&&received_data[0]===pattern(0),"exact PC00 bytes");
        write_reg('h80c,1,15,2);write_reg('h810,1,15,0);write_reg('h818,1,15,0);
        write_reg('h80c,1,15,2); // mask reset prevents replay of a stale line
        write_reg('h818,1,15,0);group_pass("staging_stalls_retention");

        stage(64'h200001000,64'h2222,1);hold_w=1;hold_b=1;write_reg('h80c,1,15,0);ticks(20);
        require(dut.busy&&aw_count==2&&w_count==1,"AW before stalled W");hold_w=0;ticks(20);
        require(dut.busy&&!dut.done,"no completion without B");
        bus_rst_n=0;ticks(10);require(dut.busy&&!sealed,"ownership through transport reset");bus_rst_n=1;ticks(5);
        write_reg('h80c,1,15,2);hold_b=0;completion_wait(64'h2222,0);
        require(received_data[1]===pattern(1)&&received_address[1]==34'h200001000,"exact PC16 bytes");
        bus_rst_n=0;ticks(5);bus_rst_n=1;ticks(5);completion_wait(64'h2222,0);
        write_reg('h810,1,15,0);write_reg('h818,1,15,0);group_pass("bus_reset_and_attribution");

        stage(64'h3ffffffe0,64'h3333,2);mem_enable=0;ticks(5);write_reg('h80c,1,15,0);ticks(30);
        require(dut.busy&&!dut.done,"stopped memory clock retains request");mem_enable=1;
        completion_wait(64'h3333,0);require(received_address[1]==34'h3ffffffe0&&received_data[1]===pattern(2),"last valid HBM line");
        write_reg('h810,1,15,0);write_reg('h814,1,15,0);require(sealed,"seal after successful POP");
        write_reg('h840,0,15,2);write_reg('h80c,1,15,2);
        for(integer i=0;i<11;i++)write_reg('h140+i*4,i==10?32'd64:32'd0,15,0);
        write_reg('h10,1,15,0);require(cmd_valid,"real control command passes after seal");
        bus_rst_n=0;ticks(5);bus_rst_n=1;ticks(5);require(sealed&&cmd_valid,"seal and command survive link reset");
        group_pass("clock_pause_boundary_seal");

        reset_all;stage(64'h400000000,4,3);write_reg('h80c,1,15,2);
        stage('h1001,4,3);write_reg('h80c,1,15,2);require(!dut.busy,"reject range/alignment without truncation");
        memory_ready_async=0;ticks(30);stage('h1000,4,3);write_reg('h80c,1,15,2);
        require(!dut.busy,"reject unavailable memory");group_pass("invalid_address_and_unavailable");

        reset_all;stage('h1000,5,4);response=2;write_reg('h80c,1,15,0);completion_wait(5,2);
        require(dut.failed,"retain bad B response");write_reg('h810,1,15,0);write_reg('h818,1,15,0);
        write_reg('h814,1,15,2);require(!sealed,"failed load cannot seal");group_pass("bad_response_blocks_seal");

        reset_all;stage('h1000,6,5);bad_id=1;write_reg('h80c,1,15,0);completion_wait(6,4);
        require(dut.failed,"reject mismatched response ID");group_pass("bad_id");

        reset_all;stage('h1000,7,6);hold_b=1;write_reg('h80c,1,15,0);ticks(20);
        memory_ready_async=0;ticks(20);require(dut.busy&&!dut.done,"lost availability retains work");
        hold_b=0;completion_wait(7,8);require(dut.failed,"late B does not hide lost availability");group_pass("lost_availability");

        reset_all;stage('h1000,8,7);hold_b=1;write_reg('h80c,1,15,0);ticks(20);
        require(dut.busy,"cold reset test has accepted work");reset_all;
        require(!dut.busy&&!dut.done&&!dut.failed&&!sealed&&dut.written==0,"canonical cold reset clears all state");
        stage('h1000,9,8);write_reg('h80c,1,15,0);completion_wait(9,0);group_pass("cold_reset_recovery");
        $display("WL PASS seed=%0d groups=%0d checks=%0d aw=%0d w=%0d b=%0d",seed,groups,checks,aw_count,w_count,b_count);$finish;
    end
    initial begin #200000;$fatal(1,"WL WATCHDOG");end
endmodule
