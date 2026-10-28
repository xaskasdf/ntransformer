`timescale 1ns / 1ps
module tb_ntrpu_pcie_fragment_pool;
    reg clk=0, rst_n=0, stop=0;
    reg source_stopped=0, ingress_isolated=0;
    reg [63:0] quiesce_epoch=0;
    reg arm_valid=0;
    wire arm_ready;
    reg [63:0] arm_epoch=0, arm_tag=0, arm_window=0, arm_destination=0;
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
    reg arm_channel=0;
    reg ack_valid=0;
    wire ack_ready;
    reg [63:0] ack_epoch=0, ack_tag=0;
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

    always #2.5 clk=~clk;
    integer cycle=0,job=0,seed=1,arms=0,lines=0,commands=0,writes=0,results=0,acks=0;
    integer bcount=0,rejections=0,job_lines=0;
    reg [511:0] expected_data;
    reg [63:0] cookie;
    ntrpu_pcie_fragment_pool #(.FIRST_COOKIE(64'hffffffff)) dut (.*);
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"OWN_CHECK case=%0d cycle=%0d %s",job,cycle,message);
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
        @(negedge clk);job=j;stop=0;job_lines=0;
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        arm_epoch=epoch==0 ? 64'h100000001 : epoch+1;
        arm_tag=64'h800000010000abcd;arm_window=64'h1000fffc0;
        arm_destination=(j%2) ? 64'h1fffffffc0 : 64'h100000040;
        arm_channel=1'(j%2);expected_data=pattern(j);
        $display("OWN_CASE id=%0d seed=%0d",job,seed);
    endtask
    task automatic start;
        arm_valid=1;
        repeat(30) begin @(posedge clk);if(arm_ready) break;end
        need(arm_ready,"JOINT_ARM_TIMEOUT");
        @(negedge clk);arm_valid=0;source_stopped=0;ingress_isolated=0;
        need(epoch==arm_epoch && bridge_epoch==epoch && bridge_lease_held && lease_held,"JOINT_ARM_STATE");
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
        ack_epoch=epoch;ack_tag=arm_tag;
        idle(3);need(bridge_lease_held && lease_held,"LEASE_UNTIL_ACK");
    endtask
    task automatic retire;
        need(ack_ready,"RETIRE_READY");ack_valid=1;@(posedge clk);
        @(negedge clk);ack_valid=0;idle(2);
        need(!result_valid && !bridge_lease_held && bridge_idle && lease_held,"ATOMIC_RETIRE");
        stop=0;
    endtask
    task automatic cold;
        @(negedge clk);rst_n=0;arm_valid=0;ack_valid=0;stop=0;
        pool_admit=0;pool_restart_required=0;pool_fault=0;pool_drained=1;
        cpl_valid=0;cmd_ready=0;data_ready=0;
        idle(4);rst_n=1;pool_admit=1;idle(3);
    endtask
    task automatic lost;
        pool_admit=0;pool_restart_required=1;idle(4);
        need(session_invalid && session_lost && bridge_lease_held && bridge_epoch==epoch,"LOST_OWNERSHIP");
        ack_epoch=epoch;ack_tag=arm_tag;ack_valid=1;arm_valid=1;arm_epoch=epoch+1;
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        idle(4);need(!ack_ready && !arm_ready && !result_ready,"INVALID_ACK");
        pool_admit=1;pool_restart_required=0;idle(5);
        need(session_invalid && bridge_lease_held && !ack_ready && !arm_ready,"SESSION_REOPENED");
        ack_valid=0;arm_valid=0;
    endtask
    always @(posedge clk) if(rst_n) begin
        if(arm_valid && arm_ready) arms=arms+1;
        if(line_valid && line_ready) begin lines=lines+1;job_lines=job_lines+1;end
        if(cmd_valid && cmd_ready) commands=commands+1;
        if(data_valid && data_ready) writes=writes+1;
        if(done_valid && done_ready) results=results+1;
        if(ack_valid && ack_ready) acks=acks+1;
    end
    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        $display("OWN_BEGIN seed=%0d first=00000000ffffffff",seed);
        s_axi_awsize=6;s_axi_awburst=1;pool_drained=1;
        idle(4);rst_n=1;
        setup(1);arm_valid=1;idle(5);
        need(epoch==0 && bridge_epoch==0 && !lease_held && !bridge_lease_held,"SPLIT_ARM_POOL");
        source_stopped=0;pool_admit=1;idle(5);
        need(epoch==0 && bridge_epoch==0 && !lease_held && !bridge_lease_held,"SPLIT_ARM_COLLECTOR");
        arm_destination=64'h2000000000;source_stopped=1;idle(5);
        need(!arm_ready && epoch==0 && bridge_epoch==0,"RANGE_ARM_SPLIT");
        arm_destination=64'h1fffffffc0;start();fill();command_accept();data_accept();complete(0);publish(0);
        ack_valid=1;ack_tag=arm_tag^64'h100000000;idle(3);need(!ack_ready && result_valid && bridge_lease_held,"FORGED_ACK_TAG");
        ack_tag=arm_tag;ack_epoch=epoch^64'h100000000;idle(3);need(!ack_ready && result_valid && bridge_lease_held,"FORGED_ACK_EPOCH");
        ack_epoch=epoch;source_stopped=0;idle(3);need(!ack_ready && result_valid && bridge_lease_held,"ACK_BARRIER");
        source_stopped=1;ingress_isolated=0;idle(3);need(!ack_ready,"ACK_ISOLATION");
        ingress_isolated=1;quiesce_epoch=epoch-1;idle(3);need(!ack_ready,"ACK_EPOCH_BARRIER");
        quiesce_epoch=epoch;pool_drained=0;idle(3);need(!ack_ready && result_valid && bridge_lease_held,"ACK_DRAIN");
        // A new AW at the same edge as ACK must inhibit retirement, including
        // the interval waiting for W and the retained B response.
        s_axi_awaddr=arm_window;s_axi_awid=4'(job);s_axi_awvalid=1;pool_drained=1;
        @(posedge clk);need(!ack_ready,"ACK_ARRIVAL");
        @(negedge clk);s_axi_awvalid=0;idle(3);need(!ack_ready && result_valid && bridge_lease_held,"ACK_PENDING_AW");
        w(frame(0,expected_data,epoch));while(!s_axi_bvalid) @(negedge clk);
        idle(3);need(!ack_ready && result_valid && bridge_lease_held,"ACK_PENDING_B");
        ack_valid=0;response(2);retire();

        setup(2);start();tx(frame(0,expected_data,epoch),0);stop=1;idle(3);publish(1);
        need(job_lines==0 && next_cookie==64'h100000000,"PARTIAL_NO_LINE_COOKIE");retire();
        setup(3);start();tx(frame(0,expected_data,epoch),0);
        tx(frame(0,expected_data^512'd1,epoch),2);idle(3);publish(4);
        need(job_lines==0 && next_cookie==64'h100000000,"CONFLICT_NO_WRITE");retire();
        setup(4);start();tx(frame(0,expected_data,epoch^64'h100000000),2);
        need(!failed && valid_bytes==0 && rejected_identity==1,"FOREIGN_FRAME_ISOLATION");
        fill();command_accept();data_accept();complete(0);publish(0);retire();
        setup(5);start();fill();stop=1;command_accept();data_accept();complete(0);publish(1);retire();
        setup(6);start();fill();command_accept();stop=1;data_accept();complete(0);publish(1);retire();
        setup(7);start();fill();command_accept();data_accept();complete(3);publish(5);retire();
        setup(8);start();fill();tx(frame(0,expected_data,epoch),2);
        need(failed && error_code==6,"LATE_LEASE_FAILURE");command_accept();data_accept();complete(0);publish(6);retire();
        setup(9);start();fill();command_accept();data_accept();complete(0);publish(0);
        lost();need(result_valid && result_status==0,"HELD_RESULT_IMMUTABLE");
        cold();setup(10);start();tx(frame(0,expected_data,epoch),0);lost();
        need(valid_bytes==64'hffffffff && next_cookie==64'hffffffff,"PARTIAL_LOSS_LEDGER");
        cold();setup(11);start();fill();lost();
        need(cmd_valid && cmd_tag==cookie && next_cookie==64'h100000000,"EXPOSED_COMMAND_RETAINED");
        command_accept();data_accept();idle(5);
        need(!done_valid && !result_valid && bridge_lease_held && session_invalid,"NO_FALSE_LOSS_COMPLETION");
        need(arms==11 && lines==8 && commands==8 && writes==8 && results==7 && acks==8,"FINAL_COUNTS");
        $display("OWN_PASS seed=%0d cycles=%0d arms=%0d lines=%0d commands=%0d data=%0d done=%0d acks=%0d frames=%0d rejects=%0d",seed,cycle,arms,lines,commands,writes,results,acks,bcount,rejections);
        $finish;
    end
    initial begin #100000; $fatal(1,"OWN_TIMEOUT case=%0d",job);end
    always @(posedge clk) begin
        cycle=cycle+1;
        $display("OWN_TRACE cycle=%0d case=%0d rst=%h stop=%h stopped=%h isolated=%h qe=%h admit=%h restart=%h pf=%h drained=%h av=%h ar=%h car=%h bar=%h ae=%h at=%h aw=%h aa=%h ac=%h kv=%h kr=%h ke=%h kt=%h rr=%h awv=%h awr=%h awa=%h awi=%h awl=%h aws=%h awb=%h awk=%h wv=%h wr=%h wd=%h ws=%h wl=%h bv=%h br=%h bi=%h bs=%h lv=%h lr=%h le=%h lt=%h la=%h ld=%h cv=%h cr=%h cc=%h ca=%h ct=%h cl=%h dv=%h dr=%h data=%h pv=%h pr=%h pt=%h ps=%h ov=%h ore=%h oe=%h ot=%h os=%h ops=%h err=%h rv=%h rc=%h re=%h rt=%h rs=%h epoch=%h held=%h be=%h bh=%h idle=%h phase=%h fail=%h error=%h bytes=%h ti=%h fault=%h lost=%h invalid=%h exhausted=%h next=%h rejected=%h frames=%h identity=%h code=%h faults=%h",cycle,job,
            rst_n,stop,source_stopped,ingress_isolated,quiesce_epoch,pool_admit,pool_restart_required,pool_fault,pool_drained,arm_valid,arm_ready,collector_arm_ready,bridge_arm_ready,arm_epoch,arm_tag,arm_window,arm_destination,arm_channel,ack_valid,ack_ready,ack_epoch,ack_tag,retire_ready,s_axi_awvalid,s_axi_awready,s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock,s_axi_wvalid,s_axi_wready,s_axi_wdata,s_axi_wstrb,s_axi_wlast,s_axi_bvalid,s_axi_bready,s_axi_bid,s_axi_bresp,line_valid,line_ready,line_epoch,line_tag,line_address,line_data,cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len,data_valid,data_ready,data,cpl_valid,cpl_ready,cpl_tag,cpl_status,done_valid,done_ready,done_epoch,done_tag,done_status,done_pool_status,done_error,result_valid,result_ready,result_epoch,result_tag,result_status,epoch,lease_held,bridge_epoch,bridge_lease_held,bridge_idle,bridge_phase,failed,error_code,valid_bytes,transport_idle,protocol_fault,session_lost,session_invalid,exhausted,next_cookie,rejected_completions,rejected_frames,rejected_identity,last_reject_code,last_reject_faults);
    end
endmodule
