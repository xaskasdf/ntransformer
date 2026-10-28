`timescale 1ns / 1ps
module tb_ntrpu_pcie_fragment_dual_managed;
    reg clk=0, rst_n=0;
    wire stop;
    reg source_stopped=0, ingress_isolated=0;
    reg [63:0] quiesce_epoch=0;
    wire arm_valid;
    wire arm_ready;
    wire [63:0] arm_epoch, arm_tag, arm_window, arm_destination;
    wire [63:0] epoch;
    wire lease_held, failed;
    wire [3:0] error_code;
    wire transport_idle;
    wire [63:0] valid_bytes;
    wire [31:0] rejected_frames, rejected_identity;
    wire [3:0] last_reject_code;
    wire [6:0] last_reject_faults;
    reg [63:0] s_axi_awaddr=0;
    reg [3:0] s_axi_awid=0;
    reg [7:0] s_axi_awlen=0;
    reg [2:0] s_axi_awsize=0;
    reg [1:0] s_axi_awburst=0;
    reg s_axi_awlock=0, s_axi_awvalid=0;
    wire s_axi_awready;
    reg [511:0] s_axi_wdata=0;
    reg [63:0] s_axi_wstrb=0;
    reg s_axi_wlast=0, s_axi_wvalid=0;
    wire s_axi_wready;
    wire [3:0] s_axi_bid;
    wire [1:0] s_axi_bresp;
    wire s_axi_bvalid;
    reg s_axi_bready=0;
    wire line_valid;
    wire line_ready;
    wire [63:0] line_epoch, line_tag, line_address;
    wire [511:0] line_data;
    wire done_valid;
    wire done_ready;
    wire [63:0] done_epoch, done_tag;
    wire done_error;
    wire result_valid;
    wire result_ready;
    wire [63:0] result_epoch, result_tag;
    wire [3:0] result_status;
    wire arm_channel;
    wire ack_valid;
    wire ack_ready;
    wire [63:0] ack_epoch, ack_tag;
    reg pool_admit=0, pool_restart_required=0, pool_fault=0, pool_drained=0;
    wire cmd_valid;
    reg cmd_ready=0;
    wire cmd_channel;
    wire [63:0] cmd_addr, cmd_tag;
    wire [7:0] cmd_len;
    wire data_valid;
    reg data_ready=0;
    wire [511:0] data;
    reg [1:0] cpl_valid=0;
    wire [1:0] cpl_ready;
    reg [127:0] cpl_tag=0;
    reg [3:0] cpl_status=0;
    wire bridge_lease_held;
    wire [63:0] bridge_epoch;
    wire bridge_idle, bridge_arm_ready, collector_arm_ready, retire_ready;
    wire [2:0] bridge_phase;
    wire [3:0] done_status;
    wire [1:0] done_pool_status;
    wire protocol_fault, session_lost, session_invalid, exhausted;
    wire [63:0] next_cookie;
    wire [31:0] rejected_completions;
    reg [31:0] s_axil_awaddr=0, s_axil_wdata=0, s_axil_araddr=0;
    reg [3:0] s_axil_wstrb=0;
    reg s_axil_awvalid=0, s_axil_wvalid=0, s_axil_bready=0;
    reg s_axil_arvalid=0, s_axil_rready=0;
    wire s_axil_awready, s_axil_wready, s_axil_arready;
    wire s_axil_bvalid, s_axil_rvalid;
    wire [1:0] s_axil_bresp, s_axil_rresp;
    wire [31:0] s_axil_rdata;
    wire irq, control_fault;
    always #2.5 clk=~clk;
    integer cycle=0,job=0,seed=1,arms=0,lines=0,commands=0,writes=0,results=0,acks=0;
    integer bcount=0,rejections=0,job_lines=0;
    integer mmio_writes=0,mmio_reads=0;
    reg [63:0] desired_epoch,desired_destination;
    reg [511:0] expected_data;
    reg [63:0] cookie;
    ntrpu_pcie_fragment_dual_managed #(.FIRST_COOKIE(64'hffffffff)) dut (.*);
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"DUAL_CONTROL_CHECK case=%0d cycle=%0d %s",job,cycle,message);
    endtask
    task automatic idle(input integer n);repeat(n) @(negedge clk);endtask
    function automatic [511:0] pattern(input integer j);
        reg [511:0] v;
        for(integer k=0;k<64;k=k+1) v[k*8+:8]=8'(j*37+k*19+seed*7);
        return v;
    endfunction
    function automatic [511:0] frame(input integer half,input reg [511:0] payload,input reg [63:0] e);
        reg [511:0] v;
        v=0;v[63:0]=64'h313046555052544e;v[127:64]=e;v[191:128]=arm_tag;
        v[255:192]=64'(half*32);v[511:256]=payload[half*256+:256];return v;
    endfunction
    task automatic aw;
        @(negedge clk);s_axi_awaddr=arm_window;s_axi_awid=4'(job);s_axi_awvalid=1;
        do @(posedge clk);while(!s_axi_awready);
        @(negedge clk);s_axi_awvalid=0;
    endtask
    task automatic w(input reg [511:0] payload);
        @(negedge clk);s_axi_wdata=payload;s_axi_wstrb='1;s_axi_wlast=1;s_axi_wvalid=1;
        do @(posedge clk);while(!s_axi_wready);
        @(negedge clk);s_axi_wvalid=0;
    endtask
    task automatic response(input integer resp);
        while(!s_axi_bvalid) @(negedge clk);
        need(s_axi_bresp==resp && s_axi_bid==4'(job),"FRAME_RESPONSE");
        idle(2+seed%3);s_axi_bready=1;@(posedge clk);@(negedge clk);s_axi_bready=0;
        bcount=bcount+1;if(resp!=0) rejections=rejections+1;
    endtask
    task automatic tx(input reg [511:0] payload,input integer resp);
        fork
            begin if((job+seed)%2) idle(3);aw();end
            begin if(!((job+seed)%2)) idle(3);w(payload);end
        join
        response(resp);
    endtask

    task automatic setup(input integer j);
        @(negedge clk);job=j;job_lines=0;
        source_stopped=0;ingress_isolated=1;quiesce_epoch=epoch;
        desired_epoch=epoch==0 ? 64'h100000001 : epoch+1;
        desired_destination=(j%2) ? 64'h1fffffffc0 : 64'h100000040;
        expected_data=pattern(j);
        $display("OWN_CASE id=%0d seed=%0d",job,seed);
        desc(desired_epoch,64'h800000010000abcd,64'h1000fffc0,desired_destination);
        wr('hb0,32'(j%2));rd('hb0,32'(j%2));
    endtask

    task automatic start;
        if(!arm_valid) wr('h10,1);
        source_stopped=1;
        repeat(30) begin @(posedge clk);if(arm_ready && arm_valid) break;end
        need(arm_ready && arm_valid,"JOINT_ARM_TIMEOUT");
        @(negedge clk);source_stopped=0;ingress_isolated=0;
        need(epoch==desired_epoch && bridge_epoch==epoch && bridge_lease_held && lease_held,"JOINT_ARM_STATE");
        need(arm_channel==1'(job%2),"CAPTURED_CHANNEL");
        rd('hb4,32'(job%2));wr('hb0,32'(!(job%2)),15,2);
    endtask
    task automatic fill;
        tx(frame(0,expected_data,epoch),0);tx(frame(1,expected_data,epoch),0);
        while(!cmd_valid) @(negedge clk);
        need(cmd_addr==arm_destination && cmd_channel==arm_channel && cmd_len==0,"COMMAND_CONTEXT");
        cookie=cmd_tag;
    endtask
    task automatic command_accept;
        idle(3+seed%4);cmd_ready=1;@(posedge clk);@(negedge clk);cmd_ready=0;
        need(data_valid && data==expected_data,"PAYLOAD_CONTEXT");
    endtask
    task automatic data_accept;
        idle(4+seed%3);data_ready=1;pool_drained=0;@(posedge clk);@(negedge clk);data_ready=0;
    endtask
    task automatic complete(input integer status);
        idle(3);cpl_tag=0;cpl_tag[arm_channel*64+:64]=cookie;
        cpl_status=0;cpl_status[arm_channel*2+:2]=2'(status);cpl_valid=2'b01<<arm_channel;
        @(posedge clk);need(cpl_ready[arm_channel],"COMPLETION_READY");
        @(negedge clk);cpl_valid=0;idle(4);
        need(!done_valid && !result_valid,"EARLY_DRAIN_RESULT");pool_drained=1;
        idle(5);need(!result_valid,"RESULT_WITHOUT_BARRIER");
    endtask
    task automatic publish(input integer status);
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch-1;
        idle(3);need(!result_valid,"STALE_BARRIER");quiesce_epoch=epoch;
        repeat(15) begin
            @(negedge clk);
            if(result_valid) break;
        end
        need(result_valid && result_status==status && result_epoch==epoch && result_tag==arm_tag,"RESULT_STATUS");
        set_ack(epoch,arm_tag);
        idle(3);need(bridge_lease_held && lease_held,"LEASE_UNTIL_ACK");
    endtask

    task automatic retire;
        need(ack_ready,"RETIRE_READY");wr('h10,4);idle(2);
        need(!result_valid && !bridge_lease_held && bridge_idle && lease_held,"ATOMIC_RETIRE");
    endtask

    task automatic cold;
        @(negedge clk);rst_n=0;
        pool_admit=0;pool_restart_required=0;pool_fault=0;pool_drained=1;
        cpl_valid=0;cmd_ready=0;data_ready=0;
        idle(4);rst_n=1;pool_admit=1;idle(3);wr('h14,1);
    endtask

    task automatic lost;
        pool_admit=0;pool_restart_required=1;idle(4);
        need(session_invalid && session_lost && bridge_lease_held && bridge_epoch==epoch && irq,"LOST_OWNERSHIP_IRQ");
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        set_ack(epoch,arm_tag);wr('h10,4,15,2);wr('h10,1,15,2);wr('hb0,0,15,2);
        need(!ack_ready && !arm_ready && !result_ready,"INVALID_ACK");
        rd('hb8,3);pool_admit=1;pool_restart_required=0;idle(5);rd('hb8,11);
        need(session_invalid && bridge_lease_held && !ack_ready && !arm_ready,"SESSION_REOPENED");
    endtask
    always @(posedge clk) if(rst_n) begin
        if(arm_valid && arm_ready) arms=arms+1;
        if(line_valid && line_ready) begin lines=lines+1;job_lines=job_lines+1;end
        if(cmd_valid && cmd_ready) commands=commands+1;
        if(data_valid && data_ready) writes=writes+1;
        if(done_valid && done_ready) results=results+1;
        if(ack_valid && ack_ready) acks=acks+1;
    end

    task automatic caw(input [31:0] address);
        @(negedge clk);s_axil_awaddr=address;s_axil_awvalid=1;
        do @(posedge clk);while(!s_axil_awready);
        @(negedge clk);s_axil_awvalid=0;s_axil_awaddr=32'hdeadbeef;
    endtask
    task automatic cwd(input [31:0] value,input [3:0] strobe);
        @(negedge clk);s_axil_wdata=value;s_axil_wstrb=strobe;s_axil_wvalid=1;
        do @(posedge clk);while(!s_axil_wready);
        @(negedge clk);s_axil_wvalid=0;s_axil_wdata='1;s_axil_wstrb=0;
    endtask
    task automatic wr(input [31:0] address,input [31:0] value,input [3:0] strobe=15,input [1:0] expected=0);
        integer order;order=(mmio_writes+seed)%3;
        fork
            begin if(order==1) idle(3);caw(address);end
            begin if(order==0) idle(4);cwd(value,strobe);end
        join
        while(!s_axil_bvalid) @(negedge clk);
        need(s_axil_bresp===expected,"CONTROL_WRITE_RESPONSE");
        idle(2+seed%3);s_axil_bready=1;@(posedge clk);@(negedge clk);s_axil_bready=0;
        mmio_writes=mmio_writes+1;
        $display("CTL_WRITE n=%0d addr=%h data=%h strb=%h resp=%h cycle=%0d",mmio_writes,address,value,strobe,expected,cycle);
    endtask
    task automatic rd(input [31:0] address,input [31:0] expected,input [1:0] resp=0,input integer hold_cycles=3);
        @(negedge clk);s_axil_araddr=address;s_axil_arvalid=1;
        do @(posedge clk);while(!s_axil_arready);
        @(negedge clk);s_axil_arvalid=0;s_axil_araddr='1;
        while(!s_axil_rvalid) @(negedge clk);
        need(s_axil_rresp===resp && s_axil_rdata===expected,"CONTROL_READ_RESPONSE");
        idle(hold_cycles+seed%3);s_axil_rready=1;@(posedge clk);@(negedge clk);s_axil_rready=0;
        mmio_reads=mmio_reads+1;
        $display("CTL_READ n=%0d addr=%h data=%h resp=%h cycle=%0d",mmio_reads,address,expected,resp,cycle);
    endtask
    task automatic desc(input [63:0] e,input [63:0] t,input [63:0] win,input [63:0] dest);
        wr('h20,e[31:0]);wr('h24,e[63:32]);wr('h28,t[31:0]);wr('h2c,t[63:32]);
        wr('h30,win[31:0]);wr('h34,win[63:32]);wr('h38,dest[31:0]);wr('h3c,dest[63:32]);
    endtask
    task automatic set_ack(input [63:0] e,input [63:0] t);
        wr('h40,e[31:0]);wr('h44,e[63:32]);wr('h48,t[31:0]);wr('h4c,t[63:32]);
    endtask

    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        $display("OWN_BEGIN seed=%0d first=00000000ffffffff",seed);
        s_axi_awsize=6;s_axi_awburst=1;pool_drained=1;
        idle(40);rst_n=1;
        rd(0,'h4452544e);rd(4,'h10000);rd(8,'hff);rd('h9c,'h5052544e);rd('ha0,'h31304655);
        rd('ha4,64);rd('ha8,32);rd('hac,0);rd('hb0,0);rd('hb4,0);rd('hbc,2);rd('hc0,37);
        rd('hc4,0,3);rd('h100000b0,0,3);rd('hb1,0,3);
        wr('hb0,2,15,2);wr('hb0,'h100,2,2);wr('hb0,1);rd('hb0,1);
        wr('hb0,0,2);rd('hb0,1);wr('hb0,'hffffff00,1);rd('hb0,0);
        wr('hb0,'hffffffff,0);rd('hb0,0);wr('hac,0,15,2);wr('hb4,1,15,2);
        wr('hbc,4,15,2);wr('hc0,39,15,2);wr('hc4,0,15,3);wr('hb1,0,15,3);
        wr('h78,3,15,2);wr('h7c,0,15,2);wr('h80,0,15,2);
        wr('h10,3,15,2);wr('h14,2,15,2);wr('h14,1);wr('h14,0,2);rd('h14,1);
        // Full-width address rejection belongs to control, before queueing.
        setup(1);wr('h3c,'h20);wr('h10,1,15,2);need(!arm_valid,"HIGH_ADDRESS_QUEUED");
        wr('h3c,'h1f);source_stopped=1;wr('h10,1);idle(5);
        need(epoch==0 && bridge_epoch==0 && !lease_held && !bridge_lease_held,"SPLIT_ARM_POOL");
        wr('hb0,0,15,2);wr('h20,2,15,2);wr('h10,2,15,2);wr('h10,4,15,2);
        rd('hb4,1);source_stopped=0;pool_admit=1;idle(5);
        need(epoch==0 && bridge_epoch==0 && !lease_held && !bridge_lease_held,"SPLIT_ARM_COLLECTOR");
        start();fill();command_accept();data_accept();complete(0);publish(0);
        set_ack(epoch,arm_tag^64'h100000000);wr('h10,4,15,2);need(result_valid && bridge_lease_held,"FORGED_ACK_TAG");
        set_ack(epoch^64'h100000000,arm_tag);wr('h10,4,15,2);need(result_valid && bridge_lease_held,"FORGED_ACK_EPOCH");
        set_ack(epoch,arm_tag);source_stopped=0;wr('h10,4,15,2);need(result_valid && bridge_lease_held,"ACK_BARRIER");
        source_stopped=1;ingress_isolated=0;wr('h10,4,15,2);
        ingress_isolated=1;quiesce_epoch=epoch-1;wr('h10,4,15,2);
        quiesce_epoch=epoch;pool_drained=0;wr('h10,4,15,2);
        // Hold an ingress AW through a complete MMIO ACK attempt.
        s_axi_awaddr=arm_window;s_axi_awid=4'(job);s_axi_awvalid=1;pool_drained=1;
        @(posedge clk);need(!ack_ready,"ACK_ARRIVAL");
        @(negedge clk);s_axi_awvalid=0;wr('h10,4,15,2);
        w(frame(0,expected_data,epoch));while(!s_axi_bvalid) @(negedge clk);
        wr('h10,4,15,2);need(result_valid && bridge_lease_held,"ACK_PENDING_B");
        response(2);retire();rd('h18,1);rd('h1c,1);wr('h10,4,15,2);
        wr('h10,8);rd('hac,1);rd('h90,1);rd('h94,0);rd('h98,6);
        setup(2);start();tx(frame(0,expected_data,epoch),0);wr('h10,2);wr('h10,2,15,2);publish(1);
        need(job_lines==0 && next_cookie==64'h100000000,"PARTIAL_NO_LINE_COOKIE");retire();
        setup(3);start();tx(frame(0,expected_data,epoch),0);tx(frame(0,expected_data^512'd1,epoch),2);publish(4);
        need(job_lines==0 && next_cookie==64'h100000000,"CONFLICT_NO_WRITE");retire();
        setup(4);start();tx(frame(0,expected_data,epoch^64'h100000000),2);
        need(!failed && valid_bytes==0 && rejected_identity==1,"FOREIGN_FRAME_ISOLATION");
        fill();command_accept();data_accept();complete(0);publish(0);retire();
        // A read captures the old diagnostic snapshot while another write
        // captures live rejection counters. R must retain its old value.
        fork
            rd('h90,1,0,40);
            begin do @(posedge clk);while(!(s_axil_arvalid && s_axil_arready));@(negedge clk);wr('h10,8);end
        join
        rd('h90,3);rd('h94,1);rd('h98,'h107);
        setup(5);start();fill();wr('h10,2);command_accept();data_accept();complete(0);publish(1);retire();
        setup(6);start();fill();command_accept();wr('h10,2);data_accept();complete(0);publish(1);retire();
        setup(7);start();fill();command_accept();data_accept();complete(3);publish(5);retire();
        setup(8);start();fill();tx(frame(0,expected_data,epoch),2);
        need(failed && error_code==6,"LATE_LEASE_FAILURE");command_accept();data_accept();complete(0);publish(6);retire();
        rd('h18,8);rd('h1c,8);
        setup(9);start();fill();command_accept();data_accept();complete(0);publish(0);
        lost();need(result_valid && result_status==0,"HELD_RESULT_IMMUTABLE");rd('h70,0);rd('h64,1);rd('h6c,'h80000001);
        wr('h10,8);rd('hac,1);rd('h90,4);wr('h14,0);need(!irq,"IRQ_DISABLE");wr('h14,1);need(irq,"INVALID_IRQ_ENABLE");
        cold();setup(10);start();tx(frame(0,expected_data,epoch),0);lost();
        need(valid_bytes==64'hffffffff && next_cookie==64'hffffffff,"PARTIAL_LOSS_LEDGER");
        cold();setup(11);start();fill();lost();
        need(cmd_valid && cmd_tag==cookie && next_cookie==64'h100000000,"EXPOSED_COMMAND_RETAINED");
        command_accept();data_accept();idle(5);
        need(!done_valid && !result_valid && bridge_lease_held && session_invalid,"NO_FALSE_LOSS_COMPLETION");
        cold();setup(12);pool_admit=0;pool_restart_required=1;idle(4);
        wr('h10,1,15,2);wr('hb0,1,15,2);wr('h20,7,15,2);
        rd('hb8,1);pool_admit=1;pool_restart_required=0;idle(3);rd('hb8,9);
        need(session_invalid && irq && !arm_valid && !bridge_lease_held && !lease_held,"INVALID_IDLE_ARM");
        need(arms==11 && lines==8 && commands==8 && writes==8 && results==7 && acks==8 && !control_fault,"FINAL_COUNTS");
        $display("CTL_PASS writes=%0d reads=%0d seed=%0d cycles=%0d",mmio_writes,mmio_reads,seed,cycle);
        $display("OWN_PASS seed=%0d cycles=%0d arms=%0d lines=%0d commands=%0d data=%0d done=%0d acks=%0d frames=%0d rejects=%0d",seed,cycle,arms,lines,commands,writes,results,acks,bcount,rejections);
        $finish;
    end
    initial begin #150000; $fatal(1,"DUAL_CONTROL_TIMEOUT case=%0d",job);end
    always @(posedge clk) begin
        cycle=cycle+1;
`ifdef NTRD_GSR_TRACE
        if(cycle<=45) $display("CTL_GSR cycle=%0d value=%b",cycle,glbl.GSR);
`endif
        $display("OWN_TRACE cycle=%0d case=%0d rst=%h stop=%h stopped=%h isolated=%h qe=%h admit=%h restart=%h pf=%h drained=%h av=%h ar=%h car=%h bar=%h ae=%h at=%h aw=%h aa=%h ac=%h kv=%h kr=%h ke=%h kt=%h rr=%h awv=%h awr=%h awa=%h awi=%h awl=%h aws=%h awb=%h awk=%h wv=%h wr=%h wd=%h ws=%h wl=%h bv=%h br=%h bi=%h bs=%h lv=%h lr=%h le=%h lt=%h la=%h ld=%h cv=%h cr=%h cc=%h ca=%h ct=%h cl=%h dv=%h dr=%h data=%h pv=%h pr=%h pt=%h ps=%h ov=%h ore=%h oe=%h ot=%h os=%h ops=%h err=%h rv=%h rc=%h re=%h rt=%h rs=%h epoch=%h held=%h be=%h bh=%h idle=%h phase=%h fail=%h error=%h bytes=%h ti=%h fault=%h lost=%h invalid=%h exhausted=%h next=%h rejected=%h frames=%h identity=%h code=%h faults=%h",cycle,job,
            rst_n,stop,source_stopped,ingress_isolated,quiesce_epoch,pool_admit,pool_restart_required,pool_fault,pool_drained,arm_valid,arm_ready,collector_arm_ready,bridge_arm_ready,arm_epoch,arm_tag,arm_window,arm_destination,arm_channel,ack_valid,ack_ready,ack_epoch,ack_tag,retire_ready,s_axi_awvalid,s_axi_awready,s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock,s_axi_wvalid,s_axi_wready,s_axi_wdata,s_axi_wstrb,s_axi_wlast,s_axi_bvalid,s_axi_bready,s_axi_bid,s_axi_bresp,line_valid,line_ready,line_epoch,line_tag,line_address,line_data,cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len,data_valid,data_ready,data,cpl_valid,cpl_ready,cpl_tag,cpl_status,done_valid,done_ready,done_epoch,done_tag,done_status,done_pool_status,done_error,result_valid,result_ready,result_epoch,result_tag,result_status,epoch,lease_held,bridge_epoch,bridge_lease_held,bridge_idle,bridge_phase,failed,error_code,valid_bytes,transport_idle,protocol_fault,session_lost,session_invalid,exhausted,next_cookie,rejected_completions,rejected_frames,rejected_identity,last_reject_code,last_reject_faults);
        $display("CTL_TRACE cycle=%0d aw=%h av=%h ar=%h wd=%h ws=%h wv=%h wr=%h bv=%h br=%h bs=%h ra=%h rv=%h rr=%h ov=%h ore=%h od=%h os=%h irq=%h fault=%h",cycle,s_axil_awaddr,s_axil_awvalid,s_axil_awready,s_axil_wdata,s_axil_wstrb,s_axil_wvalid,s_axil_wready,s_axil_bvalid,s_axil_bready,s_axil_bresp,s_axil_araddr,s_axil_arvalid,s_axil_arready,s_axil_rvalid,s_axil_rready,s_axil_rdata,s_axil_rresp,irq,control_fault);
    end
endmodule
