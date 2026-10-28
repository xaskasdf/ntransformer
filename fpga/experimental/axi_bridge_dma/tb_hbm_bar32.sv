`timescale 1ns/1ps
`default_nettype none
module tb_hbm_bar32;
    reg clk=0,rst_n=0;always #2 clk=~clk;
    integer seed=0,checks=0,groups=0,cycle=0,down_w=0,down_r=0;
    reg [63:0] s_awaddr=0,s_araddr=0;
    reg [3:0] s_awid=0,s_arid=0;
    reg [7:0] s_awlen=0,s_arlen=0;
    reg [2:0] s_awsize=2,s_arsize=2,s_awprot=0,s_arprot=0;
    reg [1:0] s_awburst=1,s_arburst=1;
    reg s_awlock=0,s_arlock=0,s_awvalid=0,s_wvalid=0,s_wlast=0,s_bready=0,s_arvalid=0,s_rready=0;
    reg [511:0] s_wdata=0;
    reg [63:0] s_wstrb=0;
    wire s_awready,s_wready,s_arready,s_bvalid,s_rvalid,s_rlast;
    wire [3:0] s_bid,s_rid;
    wire [1:0] s_bresp,s_rresp;
    wire [511:0] s_rdata;
    wire [11:0] m_awaddr,m_araddr;
    wire [2:0] m_awprot,m_arprot;
    wire [31:0] m_wdata;
    wire [3:0] m_wstrb;
    wire m_awvalid,m_wvalid,m_arvalid,m_bready,m_rready;
    reg m_bvalid=0,m_rvalid=0;
    reg [1:0] m_bresp=0,m_rresp=0;
    reg [31:0] m_rdata=0;
    reg hold_aw=0,hold_w=0,hold_ar=0,hold_b=0,hold_r=0;
    reg aw_taken=0,w_taken=0,r_taken=0;
    reg [11:0] saved_addr=0,read_addr=0;
    reg [31:0] saved_data=0;
    reg [3:0] saved_strb=0;
    wire m_awready=rst_n&&!hold_aw&&!aw_taken&&!m_bvalid&&(cycle+seed)%3!=0;
    wire m_wready=rst_n&&!hold_w&&!w_taken&&!m_bvalid&&(cycle+seed)%4!=0;
    wire m_arready=rst_n&&!hold_ar&&!r_taken&&!m_rvalid&&(cycle+seed)%3!=0;
    reg [1:0] inject_b=0,inject_r=0;
    reg [31:0] memory[0:1023];
    ntrpu_bar32_adapter dut(.*);
    task check(input bit ok,input string msg);
        checks++;if(ok!==1'b1)$fatal(1,"BAR CHECK seed=%0d %s",seed,msg);
    endtask
    task tick(input integer n);repeat(n)@(negedge clk);endtask
    task group_done(input string name);groups++;$display("BAR GROUP seed=%0d name=%s",seed,name);endtask
    always @(posedge clk)begin
        cycle++;
        if(!rst_n)begin aw_taken<=0;w_taken<=0;r_taken<=0;m_bvalid<=0;m_rvalid<=0;end
        else begin
            if(m_awvalid&&m_awready)begin aw_taken<=1;saved_addr<=m_awaddr;end
            if(m_wvalid&&m_wready)begin w_taken<=1;saved_data<=m_wdata;saved_strb<=m_wstrb;end
            if(aw_taken&&w_taken&&!m_bvalid&&!hold_b)begin
                for(integer b=0;b<4;b++)if(saved_strb[b])memory[saved_addr>>2][b*8+:8]<=saved_data[b*8+:8];
                aw_taken<=0;w_taken<=0;m_bvalid<=1;m_bresp<=inject_b;down_w++;
            end
            if(m_bvalid&&m_bready)m_bvalid<=0;
            if(m_arvalid&&m_arready)begin r_taken<=1;read_addr<=m_araddr;end
            if(r_taken&&!m_rvalid&&!hold_r)begin
                m_rvalid<=1;m_rdata<=memory[read_addr>>2];m_rresp<=inject_r;r_taken<=0;down_r++;
            end
            if(m_rvalid&&m_rready)m_rvalid<=0;
        end
    end
    reg pb=0,pr=0,pa=0,pw=0,par=0;
    reg [5:0] pbv=0;
    reg [518:0] prv=0;
    reg [14:0] pav=0,parv=0;
    reg [35:0] pwv=0;
    always @(posedge clk)begin
        if(rst_n)begin
            if(pb)check(s_bvalid&&{s_bid,s_bresp}==pbv,"upstream B stable");
            if(pr)check(s_rvalid&&{s_rid,s_rresp,s_rlast,s_rdata}==prv,"upstream R stable");
            if(pa)check(m_awvalid&&{m_awaddr,m_awprot}==pav,"downstream AW stable");
            if(pw)check(m_wvalid&&{m_wdata,m_wstrb}==pwv,"downstream W stable");
            if(par)check(m_arvalid&&{m_araddr,m_arprot}==parv,"downstream AR stable");
            pb<=s_bvalid&&!s_bready;pbv<={s_bid,s_bresp};
            pr<=s_rvalid&&!s_rready;prv<={s_rid,s_rresp,s_rlast,s_rdata};
            pa<=m_awvalid&&!m_awready;pav<={m_awaddr,m_awprot};
            pw<=m_wvalid&&!m_wready;pwv<={m_wdata,m_wstrb};
            par<=m_arvalid&&!m_arready;parv<={m_araddr,m_arprot};
        end else begin pb<=0;pr<=0;pa<=0;pw<=0;par<=0;end
    end
    task aw_send;
        s_awvalid=1;do @(posedge clk);while(!s_awready);@(negedge clk);s_awvalid=0;
    endtask
    task w_send;
        s_wvalid=1;do @(posedge clk);while(!s_wready);@(negedge clk);s_wvalid=0;
    endtask
    task write_bus(input [63:0] address,input [31:0] data,input [63:0] strobes,input [1:0] expected,input integer order);
        integer before_count;
        before_count=down_w;tick(1);s_awaddr=address;s_wdata=512'(data)<<(address[5:2]*32);s_wstrb=strobes;
        fork
            begin tick(order==1?5:0);aw_send();end
            begin
                tick(order==2?5:0);
                for(integer beat=0;beat<=s_awlen;beat++)begin s_wlast=beat==s_awlen;w_send();end
            end
        join
        wait(s_bvalid);tick(2+seed);check(s_bresp==expected&&s_bid==s_awid,"B response and preserved ID");
        if(expected==3)check(down_w==before_count,"rejected address/type has no CSR write");
        s_bready=1;@(posedge clk);@(negedge clk);s_bready=0;
    endtask
    task read_bus(input [63:0] address,input [31:0] expected_data,input [1:0] expected);
        integer before_count;reg [511:0] expected_wide;
        before_count=down_r;tick(1);s_araddr=address;s_arvalid=1;
        do @(posedge clk);while(!s_arready);@(negedge clk);s_arvalid=0;
        expected_wide=expected==3?512'd0:512'(expected_data)<<(address[5:2]*32);
        for(integer beat=0;beat<=s_arlen;beat++)begin
            wait(s_rvalid);tick(2+seed);
            check(s_rresp==expected&&s_rid==s_arid&&s_rlast==(beat==s_arlen)&&s_rdata==expected_wide,"R ID/lane/length/zero padding");
            s_rready=1;@(posedge clk);@(negedge clk);s_rready=0;
        end
        if(expected==3)check(down_r==before_count,"rejected read has no CSR access");
    endtask
    task reset_bus;
        tick(1);rst_n=0;tick(4);check(!s_bvalid&&!s_rvalid&&!m_awvalid&&!m_wvalid&&!m_arvalid,"reset clears transport offers");rst_n=1;tick(3);
    endtask
    integer count_before;
    initial begin
        if($value$plusargs("seed=%d",seed))begin end
        for(integer i=0;i<1024;i++)memory[i]=0;
        tick(8);rst_n=1;tick(5);
        for(integer lane=0;lane<16;lane++)begin
            s_awid=4'(lane);s_arid=4'(15-lane);s_awprot=3'(lane);s_arprot=3'(15-lane);
            write_bus(64'h100+64'(lane*4),32'hf00d1000+32'(lane),64'hf<<(lane*4),0,lane%3);
            read_bus(64'h100+64'(lane*4),32'hf00d1000+32'(lane),0);
        end
        group_done("all_lanes_ids");
        write_bus(64'h11c,32'h12345678,64'h5<<28,0,1);read_bus(64'h11c,32'hf0341078,0);
        write_bus(64'h11c,32'hffffffff,0,0,2);read_bus(64'h11c,32'hf0341078,0);
        inject_b=2;write_bus(64'h100,32'haabbccdd,15,2,0);inject_b=0;
        inject_r=2;read_bus(64'h100,32'haabbccdd,2);inject_r=0;
        fork write_bus(64'h104,32'h13579bdf,64'hf0,0,1);read_bus(64'h100,32'haabbccdd,0);join
        group_done("strobes_errors_duplex");
        write_bus(64'h100000100,1,15,3,1);read_bus(64'h100000100,0,3);
        write_bus(64'h1000,1,15,3,2);read_bus(64'h1000,0,3);
        write_bus(64'h102,1,15,3,0);read_bus(64'h102,0,3);
        s_awsize=6;s_arsize=6;write_bus(64'h100,1,15,3,0);read_bus(64'h100,0,3);s_awsize=2;s_arsize=2;
        s_awburst=2;s_arburst=0;write_bus(64'h100,1,15,3,0);read_bus(64'h100,0,3);s_awburst=1;s_arburst=1;
        s_awlock=1;s_arlock=1;write_bus(64'h100,1,15,3,0);read_bus(64'h100,0,3);s_awlock=0;s_arlock=0;
        count_before=down_w;write_bus(64'h104,1,64'hff,2,0);check(down_w==count_before,"out-of-lane strobe cannot write");
        group_done("decode_rejection");
        foreach(memory[i])begin end // Memory is the explicitly modeled CSR target only.
        for(integer length=1;length<=255;length=255)begin
            s_awlen=8'(length);s_arlen=8'(length);s_awsize=6;s_arsize=6;
            write_bus(64'h100,1,15,3,1);read_bus(64'h100,0,3);
            $display("BAR DRAIN beats=%0d no_side_effects=1",length+1);
            if(length==255)break;
        end
        s_awlen=0;s_arlen=0;s_awsize=2;s_arsize=2;
        tick(1);s_awaddr=64'h100;s_wlast=0;s_wstrb=15;count_before=down_w;
        fork aw_send();w_send();join
        wait(s_bvalid);tick(2);check(s_bresp==2&&down_w==count_before,"missing single-beat WLAST fails before CSR");
        s_bready=1;tick(1);s_bready=0;group_done("burst_drain_framing");
        tick(1);s_awaddr=64'h108;aw_send();reset_bus();
        tick(1);s_wlast=1;w_send();reset_bus();
        hold_w=1;tick(1);s_awaddr=64'h108;s_wstrb=64'hf00;s_wdata=512'(32'h76543210)<<64;
        fork aw_send();w_send();join
        wait(aw_taken);tick(4);check(!s_bvalid,"partial CSR write still pending");reset_bus();hold_w=0;
        read_bus(64'h108,32'hf00d1002,0);
        tick(1);s_awaddr=64'h108;s_wdata=512'(32'h76543210)<<64;s_wstrb=64'hf00;s_wlast=1;
        fork aw_send();w_send();join
        wait(s_bvalid);reset_bus();read_bus(64'h108,32'h76543210,0);
        group_done("reset_boundaries");
        write_bus(64'hffc,32'hcafef00d,64'hf000000000000000,0,2);read_bus(64'hffc,32'hcafef00d,0);
        check(!s_bvalid&&!s_rvalid,"drained frontend");group_done("last_word_recovery");
        $display("BAR PASS seed=%0d groups=%0d checks=%0d downstream_w=%0d downstream_r=%0d",seed,groups,checks,down_w,down_r);$finish;
    end
    initial begin #1000000;$fatal(1,"BAR WATCHDOG seed=%0d",seed);end
endmodule
`default_nettype wire
