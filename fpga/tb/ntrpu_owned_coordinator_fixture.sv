`timescale 1ns / 1ps
// Real collector + reservation owner, observed through public ports only.
module ntrpu_owned_coordinator_fixture #(
    parameter integer SOURCE_PERIOD_PS=5000,LOSS_PHASE=0
)(
    input wire mgmt_clk,source_enable,invalidate_request,
    input wire release_cmd,release_payload,try_reuse,
    input wire management_cold_rst_n,close_request,owner_ready_mgmt,
    input wire [63:0] close_generation,close_tag,
    input wire [5:0] external_present,external_ready,external_fault,external_ack,
    input wire [383:0] external_generation,external_tag,
    output wire [6:0] proof_requests,accepted_mask,armed_mask,retained_faults,
    output wire complete_ack,revoked,identity_fault,
    output wire [63:0] complete_generation,complete_tag,
    output wire [63:0] generation,proof_tag,
    output wire proof_request,
    output wire proof_ack,busy,peer_request,peer_answer,fault_mgmt,revoked_source,
    output wire [63:0] proof_generation,proof_ack_tag,peer_generation,peer_tag,
    output wire [31:0] obsolete,
    output wire source_clk_o,cold_o,lost_o,invalid_o,
    output wire cmd_held_o,data_held_o,result_held_o,
    output reg owner_ready=0,owner_done=0,reuse_tested=0
);
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
    wire pool_admit=rst_n && !invalidate_request;
    wire pool_restart_required=invalidate_request;
    reg pool_fault=0,pool_drained=0;
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

    always #(SOURCE_PERIOD_PS/2000.0) begin if(source_enable)clk=~clk;else clk=0;end
    integer cycle=0,job=0,seed=1,arms=0,lines=0,commands=0,writes=0,results=0,acks=0;
    integer bcount=0,rejections=0,job_lines=0;
    reg [511:0] expected_data;
    reg [63:0] cookie;
    assign proof_request=proof_requests[0];
    ntrpu_owned_drain_coordinator #(.FIRST_COOKIE(64'hffffffff)) dut(
        .generation(close_generation),.tag(close_tag),
        .proof_round_generation(generation),.proof_round_tag(proof_tag),.*);
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"DRAIN_OWNER_ASSERT phase=%0d cycle=%0d %s",LOSS_PHASE,cycle,message);
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

    assign source_clk_o=clk;assign cold_o=rst_n;
    assign lost_o=session_lost;assign invalid_o=session_invalid;
    assign cmd_held_o=cmd_valid;assign data_held_o=data_valid;assign result_held_o=result_valid;
    always @(posedge clk)begin
        cycle=cycle+1;
        if(rst_n)begin
            if(arm_valid && arm_ready)arms=arms+1;
            if(line_valid && line_ready)lines=lines+1;
            if(cmd_valid && cmd_ready)commands=commands+1;
            if(data_valid && data_ready)writes=writes+1;
            if(done_valid && done_ready)results=results+1;
            if(ack_valid && ack_ready)acks=acks+1;
            if(session_lost)begin
                need(epoch==64'h100000001 && bridge_epoch==epoch && lease_held && bridge_lease_held,"LOST_LEDGER_CHANGED");
                need(!arm_ready && !retire_ready && !line_ready && cpl_ready==0,"LOST_OWNER_REOPENED");
            end
        end
    end
    initial begin
        s_axi_awsize=6;s_axi_awburst=1;pool_drained=1;
        idle(4);rst_n=1;idle(3);setup(1);start();fill();
        if(LOSS_PHASE>=1)command_accept();
        if(LOSS_PHASE==2)begin data_accept();complete(0);publish(0);end
        owner_ready=1;
        wait(release_cmd);
        need(session_lost,"COMMAND_RELEASE_BEFORE_LOSS");
        if(LOSS_PHASE==0)command_accept();
        wait(release_payload);
        if(LOSS_PHASE<2)begin
            data_accept();idle(3);cpl_tag[arm_channel*64+:64]=cookie;cpl_valid=2'b01<<arm_channel;
            idle(5);need(cpl_ready==0 && !done_valid && !result_valid,"LOST_COMPLETION_ACCEPTED");cpl_valid=0;
        end else need(result_valid && result_status==0 && result_epoch==epoch && result_tag==arm_tag,"HELD_RESULT_CHANGED");
        need(arms==1 && lines==1 && commands==1 && writes==1 && acks==0 && results==(LOSS_PHASE==2?1:0),"OWNER_COUNTS");
        owner_done=1;
    end
    initial begin
        wait(try_reuse);@(negedge clk);
        ack_epoch=epoch;ack_tag=arm_tag;ack_valid=1;
        arm_epoch=epoch+1;arm_valid=1;
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        idle(8);need(!ack_ready && !arm_ready && !result_ready && arms==1 && acks==0,"RETIRED_OR_REARMED");
        ack_valid=0;arm_valid=0;reuse_tested=1;
    end
endmodule
