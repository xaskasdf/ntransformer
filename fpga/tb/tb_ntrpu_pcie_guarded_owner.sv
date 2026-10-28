`timescale 1ns / 1ps
module tb_ntrpu_pcie_guarded_owner;
    reg clk=0, rst_n=0, host_clk=0, mgmt_clk=0, source_run=1, host_run=1;
    reg link_up_async=0,endpoint_reset_async=1,guard_arm=0,guard_peers_held=0,guard_peers_ready=0;
    wire host_reset_n,transport_reset_n,host_reset_req,host_admit,source_admit,host_alive,source_alive;
    wire guard_invalidate,loss_mgmt,waiting_ack,guard_running,owner_invalid_ack,boundary_invalid;
    wire [4:0] loss_causes;
    wire [63:0] first_loss_tick;
    wire [3:0] guard_state;
    wire [7:0] boundary_fenced;
    wire [15:0] boundary_pending,boundary_fault;
    wire [511:0] boundary_accepted,boundary_delivered,boundary_abandoned;
    integer scenario=-1;
    initial begin #0.731;forever begin #3.65;if(host_run)host_clk=~host_clk;end end
    always #5 mgmt_clk=~mgmt_clk;
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
    always #2.5 if(source_run) clk=~clk;
    integer cycle=0,job=0,seed=1,arms=0,lines=0,commands=0,writes=0,results=0,acks=0;
    integer bcount=0,rejections=0,job_lines=0;
    integer mmio_writes=0,mmio_reads=0;
    reg [63:0] desired_epoch,desired_destination;
    reg [511:0] expected_data;
    reg [63:0] cookie;
    ntrpu_pcie_fragment_guarded_host #(.FIRST_COOKIE(64'hffffffff)) dut (.source_clk(clk),.cold_rst_n(rst_n),.*);
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"DUAL_CONTROL_CHECK case=%0d cycle=%0d %s",job,cycle,message);
    endtask
    task automatic host_idle(input integer n);repeat(n) @(negedge host_clk);endtask
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
        @(negedge host_clk);s_axi_awaddr=arm_window;s_axi_awid=4'(job);s_axi_awvalid=1;
        do @(posedge host_clk);while(!s_axi_awready);
        @(negedge host_clk);s_axi_awvalid=0;
    endtask
    task automatic w(input reg [511:0] payload);
        @(negedge host_clk);s_axi_wdata=payload;s_axi_wstrb='1;s_axi_wlast=1;s_axi_wvalid=1;
        do @(posedge host_clk);while(!s_axi_wready);
        @(negedge host_clk);s_axi_wvalid=0;
    endtask
    task automatic response(input integer resp);
        while(!s_axi_bvalid) @(negedge host_clk);
        need(s_axi_bresp==resp && s_axi_bid==4'(job),"FRAME_RESPONSE");
        host_idle(2+seed%3);s_axi_bready=1;@(posedge host_clk);@(negedge host_clk);s_axi_bready=0;
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


    always @(posedge clk) if(rst_n) begin
        if(arm_valid && arm_ready) arms=arms+1;
        if(line_valid && line_ready) begin lines=lines+1;job_lines=job_lines+1;end
        if(cmd_valid && cmd_ready) commands=commands+1;
        if(data_valid && data_ready) writes=writes+1;
        if(done_valid && done_ready) results=results+1;
        if(ack_valid && ack_ready) acks=acks+1;
    end

    task automatic caw(input [31:0] address);
        @(negedge host_clk);s_axil_awaddr=address;s_axil_awvalid=1;
        do @(posedge host_clk);while(!s_axil_awready);
        @(negedge host_clk);s_axil_awvalid=0;s_axil_awaddr=32'hdeadbeef;
    endtask
    task automatic cwd(input [31:0] value,input [3:0] strobe);
        @(negedge host_clk);s_axil_wdata=value;s_axil_wstrb=strobe;s_axil_wvalid=1;
        do @(posedge host_clk);while(!s_axil_wready);
        @(negedge host_clk);s_axil_wvalid=0;s_axil_wdata='1;s_axil_wstrb=0;
    endtask
    task automatic wr(input [31:0] address,input [31:0] value,input [3:0] strobe=15,input [1:0] expected=0);
        integer order;order=(mmio_writes+seed)%3;
        fork
            begin if(order==1) host_idle(3);caw(address);end
            begin if(order==0) host_idle(4);cwd(value,strobe);end
        join
        while(!s_axil_bvalid) @(negedge host_clk);
        need(s_axil_bresp===expected,"CONTROL_WRITE_RESPONSE");
        host_idle(2+seed%3);s_axil_bready=1;@(posedge host_clk);@(negedge host_clk);s_axil_bready=0;
        mmio_writes=mmio_writes+1;
        $display("CTL_WRITE n=%0d addr=%h data=%h strb=%h resp=%h cycle=%0d",mmio_writes,address,value,strobe,expected,cycle);
    endtask
    task automatic rd(input [31:0] address,input [31:0] expected,input [1:0] resp=0,input integer hold_cycles=3);
        @(negedge host_clk);s_axil_araddr=address;s_axil_arvalid=1;
        do @(posedge host_clk);while(!s_axil_arready);
        @(negedge host_clk);s_axil_arvalid=0;s_axil_araddr='1;
        while(!s_axil_rvalid) @(negedge host_clk);
        need(s_axil_rresp===resp && s_axil_rdata===expected,"CONTROL_READ_RESPONSE");
        host_idle(hold_cycles+seed%3);s_axil_rready=1;@(posedge host_clk);@(negedge host_clk);s_axil_rready=0;
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


    // Cooperating host; it withdraws pending offers only under facing reset.
    always @(negedge host_reset_n) begin
        s_axi_awvalid=0;s_axi_wvalid=0;s_axil_awvalid=0;s_axil_wvalid=0;s_axil_arvalid=0;
    end
    task automatic mark(input string name);
        $display("OWNER_EVENT case=%0d ps=%0d name=%s",scenario,longint'($realtime*1000),name);
    endtask
    task automatic boot(input integer which);
        @(negedge mgmt_clk);scenario=-1;rst_n=0;source_run=1;host_run=1;
        link_up_async=0;endpoint_reset_async=1;guard_arm=0;guard_peers_held=0;guard_peers_ready=0;
        s_axi_awvalid=0;s_axi_wvalid=0;s_axi_bready=0;s_axil_awvalid=0;s_axil_wvalid=0;
        s_axil_bready=0;s_axil_arvalid=0;s_axil_rready=0;
        pool_admit=0;pool_restart_required=0;pool_fault=0;pool_drained=1;
        cmd_ready=0;data_ready=0;cpl_valid=0;cpl_tag=0;cpl_status=0;
        source_stopped=0;ingress_isolated=0;quiesce_epoch=0;
        idle(8);scenario=which;job=which;cycle=0;mmio_writes=0;mmio_reads=0;
        arms=0;lines=0;commands=0;writes=0;results=0;acks=0;bcount=0;rejections=0;
        seed=1;s_axi_awsize=6;s_axi_awburst=1;s_axi_awlen=0;s_axi_awlock=0;
        idle(8);rst_n=1;pool_admit=1;link_up_async=1;endpoint_reset_async=0;
        guard_arm=1;guard_peers_held=1;guard_peers_ready=1;
        wait(guard_running && host_admit && source_admit);host_idle(40);
        rd(0,'h4452544e);rd(4,'h10000);rd(8,'hff);wr('h14,1);
        setup(which);start();
    endtask
    task automatic lose(input integer kind);
        if(kind==2) begin @(negedge host_clk);host_run=0;mark("host_stop");end
        else if(kind==3) begin @(negedge clk);source_run=0;mark("source_stop");end
        else if(kind==4) begin @(negedge clk);pool_restart_required=1;mark("pool_loss");end
        else begin
            @(negedge mgmt_clk);#1;
            if(kind==0) begin mark("link_pulse");link_up_async=0;#1;link_up_async=1;end
            else begin mark("endpoint_pulse");endpoint_reset_async=1;#2;endpoint_reset_async=0;end
        end
        wait(loss_mgmt);
        if(kind==3) begin
            repeat(32) @(negedge mgmt_clk);
            need(!owner_invalid_ack && !host_reset_req && host_reset_n && transport_reset_n,"WAIT_REAL_OWNER_CLOCK");
            source_run=1;mark("source_resume");
        end
        if(kind==2) begin repeat(80) @(negedge mgmt_clk);host_run=1;mark("host_resume");end
        wait(guard_state==8);host_idle(40);
        need(owner_invalid_ack && session_lost && session_invalid && boundary_invalid,"OWNER_INVALID_RETAINED");
        need(!host_admit && !source_admit && host_reset_n && transport_reset_n,"RESTORE_WITHOUT_REARM");
        need(bridge_lease_held && lease_held && bridge_epoch==epoch && epoch==64'h100000001,"PRESERVED_EPOCH_LEASE");
        need(!ack_ready && !arm_ready && !result_ready && cpl_ready==0 && irq,"REUSE_COMPLETION_BLOCKED");
        need(boundary_fault==0 && !protocol_fault && !control_fault,"NO_SPURIOUS_PROTOCOL_FAULT");
        if(kind==4) pool_restart_required=0;
        guard_arm=0;repeat(8) @(negedge mgmt_clk);guard_arm=1;
        host_idle(40);need(guard_state==8 && session_lost && owner_invalid_ack,"NO_WARM_REARM");
    endtask
    task automatic late;
        cpl_tag=0;cpl_tag[arm_channel*64+:64]=cookie;cpl_valid=2'b01<<arm_channel;
        idle(40);need(cpl_ready==0 && !done_valid && !result_valid,"NO_LATE_COMPLETION");
        // Keep the blocked completion until the next cold reset of both peers.
    endtask
    task automatic finish_case;
        host_idle(40);idle(4);
        $display("OWNER_PASS case=%0d ps=%0d arms=%0d lines=%0d commands=%0d data=%0d done=%0d acks=%0d frames=%0d held=%b invalid=%b epoch=%h next=%h",scenario,longint'($realtime*1000),arms,lines,commands,writes,results,acks,bcount,bridge_lease_held,session_invalid,epoch,next_cookie);
    endtask
    initial begin
        boot(0);fill();command_accept();data_accept();complete(0);publish(0);retire();finish_case();
        boot(1);tx(frame(0,expected_data,epoch),0);lose(0);
        need(valid_bytes==64'hffffffff && next_cookie==64'hffffffff,"PARTIAL_FRAME_RETAINED");finish_case();
        boot(2);fill();lose(1);need(cmd_valid && cmd_tag==cookie,"CMD_RETAINED");
        command_accept();data_accept();late();finish_case();
        boot(3);fill();command_accept();lose(2);need(data_valid && data==expected_data,"DATA_RETAINED");
        data_accept();late();finish_case();
        boot(4);fill();lose(3);need(cmd_valid && cmd_tag==cookie,"STOPPED_SOURCE_CMD_RETAINED");
        command_accept();data_accept();late();finish_case();
        boot(5);fill();command_accept();data_accept();complete(0);publish(0);lose(0);
        need(result_valid && result_status==0 && !result_ready,"VISIBLE_RESULT_NOT_RETIRABLE");finish_case();
        boot(6);aw();wait(!dut.k_axi_awready);lose(1);
        need(!transport_idle && !dut.k_axi_awready && next_cookie==64'hffffffff,"PARTIAL_AW_RETAINED");finish_case();
        boot(7);fill();lose(4);need(cmd_valid && cmd_tag==cookie,"POOL_LOSS_CMD_RETAINED");
        command_accept();data_accept();late();finish_case();
        boot(8);fill();caw('h10);wait(!dut.k_axil_awready);lose(1);
        need(!dut.k_axil_awready && !dut.k_axil_bvalid && cmd_valid,"PARTIAL_CONTROL_AW_RETAINED");
        command_accept();data_accept();late();finish_case();
        $display("OWNER_SUITE_PASS cases=9");$finish;
    end
    initial begin #500000;$fatal(1,"OWNER_TIMEOUT case=%0d",scenario);end
    always @(posedge clk) if(scenario>=0) begin
        cycle=cycle+1;
        $display("OWN_TRACE cycle=%0d case=%0d rst=%h stop=%h stopped=%h isolated=%h qe=%h admit=%h restart=%h pf=%h drained=%h av=%h ar=%h car=%h bar=%h ae=%h at=%h aw=%h aa=%h ac=%h kv=%h kr=%h ke=%h kt=%h rr=%h awv=%h awr=%h awa=%h awi=%h awl=%h aws=%h awb=%h awk=%h wv=%h wr=%h wd=%h ws=%h wl=%h bv=%h br=%h bi=%h bs=%h lv=%h lr=%h le=%h lt=%h la=%h ld=%h cv=%h cr=%h cc=%h ca=%h ct=%h cl=%h dv=%h dr=%h data=%h pv=%h pr=%h pt=%h ps=%h ov=%h ore=%h oe=%h ot=%h os=%h ops=%h err=%h rv=%h rc=%h re=%h rt=%h rs=%h epoch=%h held=%h be=%h bh=%h idle=%h phase=%h fail=%h error=%h bytes=%h ti=%h fault=%h lost=%h invalid=%h exhausted=%h next=%h rejected=%h frames=%h identity=%h code=%h faults=%h",cycle,job,
            rst_n,stop,source_stopped,ingress_isolated,quiesce_epoch,pool_admit,(pool_restart_required || boundary_invalid),pool_fault,pool_drained,arm_valid,arm_ready,collector_arm_ready,bridge_arm_ready,arm_epoch,arm_tag,arm_window,arm_destination,arm_channel,ack_valid,ack_ready,ack_epoch,ack_tag,retire_ready,dut.k_axi_awvalid,dut.k_axi_awready,dut.k_axi_awaddr,dut.k_axi_awid,dut.k_axi_awlen,dut.k_axi_awsize,dut.k_axi_awburst,dut.k_axi_awlock,dut.k_axi_wvalid,dut.k_axi_wready,dut.k_axi_wdata,dut.k_axi_wstrb,dut.k_axi_wlast,dut.k_axi_bvalid,dut.k_axi_bready,dut.k_axi_bid,dut.k_axi_bresp,line_valid,line_ready,line_epoch,line_tag,line_address,line_data,cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len,data_valid,data_ready,data,cpl_valid,cpl_ready,cpl_tag,cpl_status,done_valid,done_ready,done_epoch,done_tag,done_status,done_pool_status,done_error,result_valid,result_ready,result_epoch,result_tag,result_status,epoch,lease_held,bridge_epoch,bridge_lease_held,bridge_idle,bridge_phase,failed,error_code,valid_bytes,transport_idle,protocol_fault,session_lost,session_invalid,exhausted,next_cookie,rejected_completions,rejected_frames,rejected_identity,last_reject_code,last_reject_faults);
        $display("CTL_TRACE cycle=%0d aw=%h av=%h ar=%h wd=%h ws=%h wv=%h wr=%h bv=%h br=%h bs=%h ra=%h rv=%h rr=%h ov=%h ore=%h od=%h os=%h irq=%h fault=%h",cycle,dut.k_axil_awaddr,dut.k_axil_awvalid,dut.k_axil_awready,dut.k_axil_wdata,dut.k_axil_wstrb,dut.k_axil_wvalid,dut.k_axil_wready,dut.k_axil_bvalid,dut.k_axil_bready,dut.k_axil_bresp,dut.k_axil_araddr,dut.k_axil_arvalid,dut.k_axil_arready,dut.k_axil_rvalid,dut.k_axil_rready,dut.k_axil_rdata,dut.k_axil_rresp,irq,control_fault);
    end

    always @(posedge clk) if(scenario>=0) begin
        $display("OWNER_SOURCE case=%0d ps=%0d cold=%b loss=%b invalid=%b ack=%b lost=%b rst=%b pending=%h fenced=%h fault=%h accepted=%h delivered=%h abandoned=%h",scenario,longint'($realtime*1000),rst_n,guard_invalidate,boundary_invalid,owner_invalid_ack,session_lost,transport_reset_n,boundary_pending,boundary_fenced,boundary_fault,boundary_accepted,boundary_delivered,boundary_abandoned);
    end
    always @(posedge mgmt_clk) begin
        #0.001;
        if(scenario>=0)$display("OWNER_MGMT case=%0d ps=%0d cold=%b loss=%b invalid=%b ack=%b req=%b hrst=%b srst=%b ha=%b sa=%b causes=%0d tick=%0d state=%0d",scenario,longint'($realtime*1000),rst_n,loss_mgmt,guard_invalidate,owner_invalid_ack,host_reset_req,host_reset_n,transport_reset_n,host_admit,source_admit,loss_causes,first_loss_tick,guard_state);
    end
    always @(posedge host_clk) if(scenario>=0) begin
        $display("OWNER_HOST case=%0d ps=%0d rst=%b awv=%b awr=%b awa=%h awi=%h wv=%b wr=%b wd=%h ws=%h wl=%b bv=%b br=%b bi=%h bs=%h cav=%b car=%b ca=%h cwv=%b cwr=%b cwd=%h cws=%h cbv=%b cbr=%b cbs=%h crv=%b crr=%b cra=%h rv=%b rr=%b rd=%h rs=%h",scenario,longint'($realtime*1000),host_reset_n,s_axi_awvalid,s_axi_awready,s_axi_awaddr,s_axi_awid,s_axi_wvalid,s_axi_wready,s_axi_wdata,s_axi_wstrb,s_axi_wlast,s_axi_bvalid,s_axi_bready,s_axi_bid,s_axi_bresp,s_axil_awvalid,s_axil_awready,s_axil_awaddr,s_axil_wvalid,s_axil_wready,s_axil_wdata,s_axil_wstrb,s_axil_bvalid,s_axil_bready,s_axil_bresp,s_axil_arvalid,s_axil_arready,s_axil_araddr,s_axil_rvalid,s_axil_rready,s_axil_rdata,s_axil_rresp);
    end
endmodule
