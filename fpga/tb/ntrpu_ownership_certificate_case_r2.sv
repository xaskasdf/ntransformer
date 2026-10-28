`timescale 1ns / 1ps
// Real collector + reservation owner, observed through public ports only.
module ntrpu_ownership_certificate_case_r2 #(
    parameter integer ID=0, LOSS_PHASE=0, SOURCE_PERIOD_PS=5000, FAULT_TAIL=0
)(output reg finished=0);
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

    reg source_enable=1,mgmt_clk=0;
    always #(SOURCE_PERIOD_PS/2000.0) begin if(source_enable)clk=~clk;else clk=0;end
    always #5 mgmt_clk=~mgmt_clk;
    integer cycle=0,job=0,seed=1,arms=0,lines=0,commands=0,writes=0,results=0,acks=0;
    integer bcount=0,rejections=0,job_lines=0;
    reg [511:0] expected_data;
    reg [63:0] cookie;
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"OWN_CERT_CHECK id=%0d cycle=%0d %s",ID,cycle,message);
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

    reg proof_request=0;
    reg [63:0] generation=0,proof_tag=1;
    wire proof_ack,busy,peer_request,peer_answer,fault_mgmt,revoked_source;
    wire [63:0] proof_generation,proof_ack_tag,peer_generation,peer_tag;
    wire [31:0] obsolete;
    // Explicit adversarial provider-port probe, separate from real owner state.
    // Only FAULT_TAIL uses it, after all real-owner scenarios have completed.
    reg epoch_probe=0;
    wire [63:0] observed_epoch=epoch ^ (epoch_probe ? 64'h100000000 : 64'd0);
    generate if(FAULT_TAIL==0)begin: integrated
        ntrpu_pcie_fragment_certified_owner #(.FIRST_COOKIE(64'hffffffff)) dut(.*);
    end else begin: public_port_probe
    ntrpu_pcie_fragment_pool #(.FIRST_COOKIE(64'hffffffff)) dut (.*);
    ntrpu_ownership_certificate certificate(
        .mgmt_clk(mgmt_clk),.source_clk(clk),.cold_rst_n(rst_n),
        .req_valid(proof_request),.generation(generation),.tag(proof_tag),
        .session_lost(session_lost),.session_invalid(session_invalid),
        .collector_epoch(observed_epoch),.bridge_epoch(bridge_epoch),
        .collector_lease_held(lease_held),.bridge_lease_held(bridge_lease_held),
        .arm_ready(arm_ready),.retire_ready(retire_ready),.line_ready(line_ready),.cpl_ready(cpl_ready),
        .ack_valid(proof_ack),.ack_generation(proof_generation),.ack_tag(proof_ack_tag),
        .busy(busy),.peer_request(peer_request),.peer_answer(peer_answer),
        .peer_generation(peer_generation),.peer_tag(peer_tag),.fault_mgmt(fault_mgmt),
        .revoked_source(revoked_source),.obsolete_completions(obsolete));
    end endgenerate
    integer checks=0,mgmt_edges=0,positive_edges=0;
    task automatic check(input bit condition,input string message);
        checks=checks+1;need(condition,message);
    endtask
    task automatic event_mark(input string label_name);
        #0.001;
        $display("OC_EVENT id=%0d name=%s ps=%0.0f phase=%0d period=%0d tail=%0d gen=%h tag=%h epoch=%h be=%h lost=%b invalid=%b held=%b bh=%b cmd=%b data=%b result=%b ack=%b busy=%b obsolete=%0d fault=%b revoked=%b enable=%b",ID,label_name,$realtime*1000,LOSS_PHASE,SOURCE_PERIOD_PS,FAULT_TAIL,generation,proof_tag,epoch,bridge_epoch,session_lost,session_invalid,lease_held,bridge_lease_held,cmd_valid,data_valid,result_valid,proof_ack,busy,obsolete,fault_mgmt,revoked_source,source_enable);
    endtask
    task automatic mgmt_wait(input integer count);
        repeat(count)@(negedge mgmt_clk);
    endtask
    task automatic wait_ack;
        repeat(1000)begin @(negedge mgmt_clk);if(proof_ack)break;end
        check(proof_ack && proof_generation==epoch && proof_ack_tag==proof_tag,"FRESH_CERTIFICATE_TIMEOUT_OR_IDENTITY");
    endtask
    task automatic retained;
        check(lease_held && bridge_lease_held && epoch==64'h100000001 && bridge_epoch==epoch,"OWNER_LEDGER_CHANGED");
        check(session_lost && session_invalid && !arm_ready && !retire_ready && !line_ready && cpl_ready==0,"OWNER_REOPENED");
        if(LOSS_PHASE==0)check(cmd_valid && cmd_tag==cookie && cmd_addr==arm_destination,"HELD_COMMAND_CHANGED");
        if(LOSS_PHASE==1)check(data_valid && data==expected_data,"HELD_DATA_CHANGED");
        if(LOSS_PHASE==2)check(result_valid && result_epoch==epoch && result_tag==arm_tag && result_status==0,"HELD_RESULT_CHANGED");
    endtask
    always @(posedge clk)begin
        cycle=cycle+1;
        if(rst_n)begin
            if(arm_valid && arm_ready)arms=arms+1;
            if(line_valid && line_ready)lines=lines+1;
            if(cmd_valid && cmd_ready)commands=commands+1;
            if(data_valid && data_ready)writes=writes+1;
            if(done_valid && done_ready)results=results+1;
            if(ack_valid && ack_ready)acks=acks+1;
        end
    end
    always @(posedge mgmt_clk)if(rst_n)begin
        mgmt_edges=mgmt_edges+1;
        if(proof_ack)begin
            positive_edges=positive_edges+1;
            check(session_lost && session_invalid,"CERTIFIED_LIVE_OWNER");
            check(proof_generation==generation && proof_ack_tag==proof_tag && proof_generation==epoch,"CERTIFIED_WRONG_EPOCH");
            check(!fault_mgmt,"CERTIFIED_REVOKED_OWNER");
        end
    end
    initial begin
        seed=ID+1;s_axi_awsize=6;s_axi_awburst=1;pool_drained=1;
        idle(4);rst_n=1;pool_admit=1;idle(3);
        setup(ID+1);start();fill();
        if(LOSS_PHASE>=1)command_accept();
        if(LOSS_PHASE==2)begin data_accept();complete(0);publish(0);end
        generation=epoch;proof_request=1;
        repeat(1000)begin @(negedge mgmt_clk);if(peer_request)break;end
        check(peer_request,"REQUEST_NOT_DELIVERED");
        mgmt_wait(30);check(!proof_ack && !peer_answer && !session_lost,"LIVE_OWNER_CERTIFIED");
        event_mark("LIVE_OWNER_BLOCKED");
        @(negedge clk);source_enable=0;pool_restart_required=1;pool_admit=0;
        #0.001;check(session_invalid && !session_lost,"LOSS_REQUIRES_SOURCE_EDGE");
        mgmt_wait(30);check(!proof_ack && !peer_answer && !session_lost,"STOPPED_OWNER_CERTIFIED");
        event_mark("STOPPED_BEFORE_LOSS_EDGE");
        proof_tag=2;mgmt_wait(5);event_mark("REPLACED_WHILE_STOPPED");
        source_enable=1;@(posedge clk);@(negedge clk);source_enable=0;
        mgmt_wait(30);check(session_lost && !proof_ack,"ONE_EDGE_NOT_FULL_HANDSHAKE");
        event_mark("REGISTERED_LOSS_SOURCE_STOPPED");
        source_enable=1;wait_ack();
        check(obsolete==1 && !fault_mgmt,"OBSOLETE_REQUEST_ACCOUNTING");retained();
        event_mark("FIRST_CERTIFICATE");
        @(negedge clk);ack_epoch=epoch;ack_tag=arm_tag;ack_valid=1;
        arm_epoch=epoch+1;arm_valid=1;source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        pool_restart_required=0;pool_admit=1;idle(8);retained();
        check(!ack_ready && !result_ready && arms==1 && acks==0,"RETIRED_OR_REARMED_LOST_OWNER");
        ack_valid=0;arm_valid=0;event_mark("RETIRE_AND_REARM_BLOCKED");
        @(negedge clk);source_enable=0;@(negedge mgmt_clk);proof_tag=3;
        mgmt_wait(30);check(!proof_ack,"CACHED_CERTIFICATE_REUSED");
        event_mark("RENEWAL_SOURCE_STOPPED");source_enable=1;wait_ack();retained();
        event_mark("RENEWED_CERTIFICATE");
        if(LOSS_PHASE==0)command_accept();
        if(LOSS_PHASE<2)begin
            data_accept();idle(3);
            cpl_tag[arm_channel*64+:64]=cookie;cpl_valid=2'b01<<arm_channel;
            idle(5);check(cpl_ready==0 && !result_valid && !done_valid,"LOST_COMPLETION_ACCEPTED");cpl_valid=0;
        end
        mgmt_wait(10);check(proof_ack && !fault_mgmt,"OWNERSHIP_PROOF_CONFUSED_WITH_DRAIN");
        event_mark("PENDING_TRAFFIC_PRESERVED");
        if(FAULT_TAIL)begin
            @(negedge clk);epoch_probe=1;@(posedge clk);@(negedge clk);source_enable=0;
            event_mark("PUBLIC_EPOCH_CONTRADICTION");
            mgmt_wait(12);check(revoked_source && fault_mgmt && !proof_ack,"REVOCATION_NOT_PROPAGATED");
            epoch_probe=0;proof_request=0;mgmt_wait(8);proof_tag=4;proof_request=1;
            source_enable=1;mgmt_wait(50);
            check(fault_mgmt && revoked_source && !proof_ack,"REVOCATION_NOT_RETAINED");
            event_mark("REVOCATION_RETAINED");
        end else begin
            @(negedge mgmt_clk);generation=epoch+1;proof_tag=4;
            repeat(1000)begin @(negedge mgmt_clk);if(peer_request && peer_generation==generation)break;end
            check(peer_request && peer_generation==generation,"WRONG_GENERATION_NOT_DELIVERED");
            idle(15);mgmt_wait(15);
            check(!proof_ack && !peer_answer && busy && !fault_mgmt,"WRONG_GENERATION_CERTIFIED");
            event_mark("WRONG_GENERATION_BLOCKED");
        end
        check(arms==1 && lines==1 && commands==1 && writes==1 && acks==0,"REAL_OWNER_TRANSFER_COUNTS");
        check(results==(LOSS_PHASE==2 ? 1 : 0) && bcount==2,"REAL_OWNER_RESULT_COUNTS");
        event_mark("TERMINAL");
        $display("OC_CASE_PASS id=%0d phase=%0d period=%0d tail=%0d checks=%0d positive=%0d source_edges=%0d mgmt_edges=%0d",ID,LOSS_PHASE,SOURCE_PERIOD_PS,FAULT_TAIL,checks,positive_edges,cycle,mgmt_edges);
        finished=1;
    end
endmodule
