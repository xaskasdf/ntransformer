`timescale 1ns / 1ps
module ntrpu_dual_coordinated_case #(
    parameter integer CASE_ID=0,SOURCE_PERIOD_PS=5000,LOSS_PHASE=0,BAD_EPOCH=0
)(output reg finished=0);
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
    reg source_enable=1;
    always #(SOURCE_PERIOD_PS/2000.0) begin if(source_enable)clk=~clk;else clk=0;end
    always @(posedge clk)cycle=cycle+1;
    integer cycle=0,job=0,seed=1,arms=0,lines=0,commands=0,writes=0,results=0,acks=0;
    integer bcount=0,rejections=0,job_lines=0;
    integer mmio_writes=0,mmio_reads=0;
    reg [63:0] desired_epoch,desired_destination;
    reg [511:0] expected_data;
    reg [63:0] cookie;
    reg mgmt_clk=0;
    always #5 mgmt_clk=~mgmt_clk;
    wire management_cold_rst_n=rst_n;
    reg close_request=0;
    reg [63:0] generation=64'h100000001,tag=1;
    reg owner_ready_mgmt=1;
    reg [5:0] external_present=0,external_ready=0,external_fault=0,external_ack=0;
    reg [383:0] external_generation=0,external_tag=0;
    wire [6:0] proof_requests,accepted_mask,armed_mask,retained_faults;
    wire [63:0] proof_round_generation,proof_round_tag,complete_generation,complete_tag;
    wire complete_ack,revoked,identity_fault,proof_ack,busy,peer_request,peer_answer,fault_mgmt,revoked_source;
    wire [63:0] proof_generation,proof_ack_tag,peer_generation,peer_tag;
    wire [31:0] obsolete;
    ntrpu_pcie_fragment_dual_coordinated #(.FIRST_COOKIE(64'hffffffff)) dut (.*);
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"MANAGED_OWNER_ASSERT id=%0d phase=%0d cycle=%0d %s",CASE_ID,LOSS_PHASE,cycle,message);
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
        $display("MCTRL_CASE id=%0d job=%0d seed=%0d",CASE_ID,job,seed);
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
        $display("MCTRL_WRITE id=%0d n=%0d addr=%h data=%h strb=%h resp=%h cycle=%0d",CASE_ID,mmio_writes,address,value,strobe,expected,cycle);
    endtask
    task automatic rd(input [31:0] address,input [31:0] expected,input [1:0] resp=0,input integer hold_cycles=3);
        @(negedge clk);s_axil_araddr=address;s_axil_arvalid=1;
        do @(posedge clk);while(!s_axil_arready);
        @(negedge clk);s_axil_arvalid=0;s_axil_araddr='1;
        while(!s_axil_rvalid) @(negedge clk);
        need(s_axil_rresp===resp && s_axil_rdata===expected,"CONTROL_READ_RESPONSE");
        idle(hold_cycles+seed%3);s_axil_rready=1;@(posedge clk);@(negedge clk);s_axil_rready=0;
        mmio_reads=mmio_reads+1;
        $display("MCTRL_READ id=%0d n=%0d addr=%h data=%h resp=%h cycle=%0d",CASE_ID,mmio_reads,address,expected,resp,cycle);
    endtask
    task automatic desc(input [63:0] e,input [63:0] t,input [63:0] win,input [63:0] dest);
        wr('h20,e[31:0]);wr('h24,e[63:32]);wr('h28,t[31:0]);wr('h2c,t[63:32]);
        wr('h30,win[31:0]);wr('h34,win[63:32]);wr('h38,dest[31:0]);wr('h3c,dest[63:32]);
    endtask
    task automatic set_ack(input [63:0] e,input [63:0] t);
        wr('h40,e[31:0]);wr('h44,e[63:32]);wr('h48,t[31:0]);wr('h4c,t[63:32]);
    endtask

    task automatic mark(input string name);
        #0.001;
        $display("MCTRL_EVENT id=%0d name=%s ps=%0.0f phase=%0d bad=%0d period=%0d gen=%h tag=%h epoch=%h proof=%b accepted=%b complete=%b lost=%b invalid=%b bv=%b resp=%h enabled=%b held=%b bridge=%b ack=%b obsolete=%0d",CASE_ID,name,$realtime*1000,LOSS_PHASE,BAD_EPOCH,SOURCE_PERIOD_PS,generation,tag,epoch,proof_ack,accepted_mask,complete_ack,session_lost,session_invalid,s_axil_bvalid,s_axil_bresp,source_enable,lease_held,bridge_lease_held,ack_valid,obsolete);
        #0.001;
    endtask
    task automatic held;
        if(LOSS_PHASE==0)need(cmd_valid && cmd_tag==64'hffffffff,"CMD_RETAINED");
        if(LOSS_PHASE==1)need(data_valid && data==expected_data,"DATA_RETAINED");
        if(LOSS_PHASE==2)need(result_valid && result_epoch==epoch && result_status==0,"RESULT_RETAINED");
    endtask
    initial begin
        generation=64'h100000001+BAD_EPOCH;
        s_axi_awsize=6;s_axi_awburst=1;pool_drained=1;
        idle(4);rst_n=1;pool_admit=1;
        @(negedge mgmt_clk);close_request=1;
        repeat(40)@(negedge mgmt_clk);
        need(epoch==0 && !proof_ack && !complete_ack,"EMPTY_OWNER_CERTIFIED");mark("EMPTY_OWNER_BLOCKED");
        wr('h14,1);rd(0,'h4452544e);rd(8,'hff);rd('h18,0);
        setup(LOSS_PHASE+1);start();mark("CONTROL_ARMED");fill();
        if(LOSS_PHASE>=1)command_accept();
        if(LOSS_PHASE==2)begin data_accept();complete(0);publish(0);end
        repeat(30)@(negedge mgmt_clk);
        need(!proof_ack && !complete_ack && !session_lost,"LIVE_OWNER_CERTIFIED");held();mark("PAYLOAD_HELD");
        // The NTRD response is retained while ownership becomes invalid.
        fork caw('h20);cwd('hdeadbeef,15);join
        while(!s_axil_bvalid)@(negedge clk);
        need(s_axil_bresp==2,"LOCKED_DESCRIPTOR_WRITE");mark("MMIO_B_HELD");
        @(negedge clk);source_enable=0;pool_admit=0;pool_restart_required=1;
        repeat(30)@(negedge mgmt_clk);
        need(session_invalid && !session_lost && !proof_ack && s_axil_bvalid,"SOURCE_STOPPED_BEFORE_LOSS");held();mark("SOURCE_STOPPED_BEFORE_LOSS");
        tag=2;repeat(5)@(negedge mgmt_clk);mark("REQUEST_REPLACED_STOPPED");
        source_enable=1;@(posedge clk);@(negedge clk);source_enable=0;
        repeat(30)@(negedge mgmt_clk);
        need(session_lost && !proof_ack && s_axil_bvalid,"ONE_EDGE_NOT_CERTIFICATE");held();mark("LOSS_LATCHED_SOURCE_STOPPED");
        source_enable=1;
        if(BAD_EPOCH)begin
            repeat(180)@(negedge mgmt_clk);
            need(!proof_ack && accepted_mask==0 && !complete_ack,"WRONG_EPOCH_CERTIFIED");mark("WRONG_EPOCH_BLOCKED");
        end else begin
            wait(proof_ack && accepted_mask[0]);
            need(proof_generation==epoch && proof_ack_tag==2 && obsolete==1,"CERTIFICATE_IDENTITY");mark("OWNER_CERTIFIED");
        end
        need(s_axil_bvalid && s_axil_bresp==2,"MMIO_RESPONSE_LOST");
        @(negedge clk);s_axil_bready=1;@(posedge clk);@(negedge clk);s_axil_bready=0;
        mmio_writes=mmio_writes+1;
        $display("MCTRL_WRITE id=%0d n=%0d addr=00000020 data=deadbeef strb=f resp=2 cycle=%0d",CASE_ID,mmio_writes,cycle);
        mark("MMIO_B_RELEASED");
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        set_ack(epoch,arm_tag);wr('h10,4,15,2);wr('h10,1,15,2);
        wr('h20,2,15,2);wr('hb0,32'(!(job%2)),15,2);
        pool_admit=1;pool_restart_required=0;idle(5);
        rd('h18,1);rd('h1c,0);rd('h50,1);rd('h54,1);rd('hb8,11);
        if(LOSS_PHASE==2)begin rd('h70,0);rd('h64,1);rd('h6c,'h80000001);end
        need(arms==1 && acks==0 && !ack_ready && !arm_ready && !control_fault && irq,"LOST_SESSION_REUSED");held();mark("REUSE_BLOCKED");
        if(!BAD_EPOCH)begin
            @(negedge clk);source_enable=0;
            @(negedge mgmt_clk);tag=3;repeat(40)@(negedge mgmt_clk);
            need(!proof_ack && !accepted_mask[0],"STOPPED_RENEWAL");mark("RENEW_STOPPED");
            source_enable=1;wait(proof_ack && accepted_mask[0]);
            need(proof_ack_tag==3 && proof_generation==epoch,"RENEWED_IDENTITY");mark("OWNER_RENEWED");
        end
        if(LOSS_PHASE==0)command_accept();
        if(LOSS_PHASE<2)data_accept();
        idle(3);cpl_tag=0;cpl_tag[arm_channel*64+:64]=cookie;cpl_valid=2'b01<<arm_channel;
        idle(5);need(cpl_ready==0 && !done_valid,"LATE_COMPLETION_ACCEPTED");cpl_valid=0;
        if(LOSS_PHASE==2)need(result_valid && result_status==0 && result_epoch==epoch,"RESULT_LOST");
        else need(!result_valid,"FALSE_RESULT");
        need(arms==1 && lines==1 && commands==1 && writes==1 && acks==0 && results==(LOSS_PHASE==2?1:0),"OWNER_COUNTS");
        mark("OWNER_PAYLOAD_DRAINED");
        repeat(20)@(negedge mgmt_clk);
        need(!complete_ack && proof_requests[6:1]==0 && accepted_mask[6:1]==0 && retained_faults==0 && !identity_fault,"ABSENT_CONTRACTS");
        need(bridge_lease_held && lease_held && session_invalid && epoch==64'h100000001,"RETAINED_OWNER");
        mark("TERMINAL");
        $display("MCTRL_CASE_PASS id=%0d phase=%0d bad=%0d period=%0d writes=%0d reads=%0d retired=%0d",CASE_ID,LOSS_PHASE,BAD_EPOCH,SOURCE_PERIOD_PS,mmio_writes,mmio_reads,acks);
        finished=1;
    end
endmodule
