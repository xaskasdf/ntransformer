`timescale 1ns/1ps
module managed_lifetime_case #(parameter integer CASE_ID=0)(output reg passed=0);
    reg mgmt_clk=0, source_clk=0, external_reset_async=0, rearm=0;
    reg [1:0] ui_clk=0, calibrated=0, ui_rst=0;
    reg [1:0] m_rvalid=0;
    wire source_reset_n, admit, running, rearm_ready, restart_required;
    wire peers_reset_req, source_alive, lifetime_reset_n, restart_required_source;
    wire [1:0] memory_reset_n, memory_alive;
    wire [7:0] banks_used;
    wire [9:0] pending_counts;
    wire [1:0] m_awvalid, m_wvalid, m_wlast, m_bready;
    reg [1:0] m_awready=0, m_wready=0, m_bvalid=0;
    wire [73:0] m_awaddr;
    wire [15:0] m_awlen;
    wire [7:0] m_awid, m_awcache;
    wire [5:0] m_awsize;
    wire [3:0] m_awburst;
    wire [1023:0] m_wdata;
    wire [127:0] m_wstrb;
    reg [7:0] m_bid=0;
    reg [3:0] m_bresp=0;
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
    wire pool_admit, pool_restart_required, pool_fault, pool_drained;
    wire cmd_valid;
    wire cmd_ready;
    wire cmd_channel;
    wire [63:0] cmd_addr, cmd_tag;
    wire [7:0] cmd_len;
    wire data_valid;
    wire data_ready;
    wire [511:0] data;
    wire [1:0] cpl_valid;
    wire [1:0] cpl_ready;
    wire [127:0] cpl_tag;
    wire [3:0] cpl_status;
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

    wire clk=source_clk,rst_n=lifetime_reset_n;
    reg run_source=1;reg [1:0] run_memory=3;
    always #2.5 if(run_source) source_clk=~source_clk;
    always #5 mgmt_clk=~mgmt_clk;
    initial begin #0.713;forever #2 if(run_memory[0]) ui_clk[0]=~ui_clk[0];end
    initial begin #1.137;forever #3.65 if(run_memory[1]) ui_clk[1]=~ui_clk[1];end
    integer cycle=0,job=-1,seed=CASE_ID+1,mmio_writes=0,mmio_reads=0;
    integer bcount=0,rejections=0,arms=0,lines=0,commands=0,writes=0,returns=0,dones=0,acks=0;
    reg [511:0] expected_data;
    reg [63:0] desired_epoch,desired_destination;
    reg [1:0] hold_b=0;
    integer aw_job[0:1],w_job[0:1],b_job[0:1],mem_cycle[0:1];
    integer commanded[0:3],input_seen[0:3],aw_seen[0:3],w_seen[0:3],b_seen[0:3];
    ntrpu_pcie_fragment_dual_mig #(.FIRST_COOKIE(64'hffffffff)) dut (.*);
    function automatic longint ps;return longint'($realtime/1ps);endfunction
    function automatic [63:0] address(input integer j);return j%2 ? 64'h1fffffffc0:64'h100000040;endfunction
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
        cb_response(address,value,strobe,expected);
    endtask
    task automatic rd(input [31:0] address,input [31:0] expected,input [1:0] resp=0,input integer hold_cycles=3);
        @(negedge clk);s_axil_araddr=address;s_axil_arvalid=1;
        do @(posedge clk);while(!s_axil_arready);
        @(negedge clk);s_axil_arvalid=0;s_axil_araddr='1;
        while(!s_axil_rvalid) @(negedge clk);
        need(s_axil_rresp===resp && s_axil_rdata===expected,"CONTROL_READ_RESPONSE");
        idle(hold_cycles+seed%3);s_axil_rready=1;@(posedge clk);@(negedge clk);s_axil_rready=0;
        mmio_reads=mmio_reads+1;
        $display("NTRD case=%0d CTL_READ n=%0d addr=%h data=%h resp=%h cycle=%0d",CASE_ID,mmio_reads,address,expected,resp,cycle);
    endtask
    task automatic desc(input [63:0] e,input [63:0] t,input [63:0] win,input [63:0] dest);
        wr('h20,e[31:0]);wr('h24,e[63:32]);wr('h28,t[31:0]);wr('h2c,t[63:32]);
        wr('h30,win[31:0]);wr('h34,win[63:32]);wr('h38,dest[31:0]);wr('h3c,dest[63:32]);
    endtask
    task automatic set_ack(input [63:0] e,input [63:0] t);
        wr('h40,e[31:0]);wr('h44,e[63:32]);wr('h48,t[31:0]);wr('h4c,t[63:32]);
    endtask

    task automatic cb_response(input [31:0] address,input [31:0] value,input [3:0] strobe,input [1:0] expected);
        while(!s_axil_bvalid) @(negedge clk);
        need(s_axil_bresp===expected,"CONTROL_WRITE_RESPONSE");
        idle(2+seed%3);s_axil_bready=1;@(posedge clk);@(negedge clk);s_axil_bready=0;
        mmio_writes=mmio_writes+1;
        $display("NTRD case=%0d CTL_WRITE n=%0d addr=%h data=%h strb=%h resp=%h cycle=%0d",CASE_ID,mmio_writes,address,value,strobe,expected,cycle);
    endtask

    task automatic setup(input integer j);
        @(negedge clk);job=j;source_stopped=0;ingress_isolated=1;quiesce_epoch=epoch;
        desired_epoch=64'h100000001+j;desired_destination=address(j);expected_data=pattern(j);
        $display("NTRD case=%0d JOB job=%0d ps=%0d",CASE_ID,job,ps());
        desc(desired_epoch,64'h800000010000abcd,64'h1000fffc0,desired_destination);wr('hb0,32'(j%2));
        wr('h10,1);need(arm_valid && !bridge_lease_held,"ARM_QUEUED");
        source_stopped=1;repeat(30) begin @(posedge clk);if(arm_valid && arm_ready) break;end
        need(arm_valid && arm_ready,"JOINT_ARM_TIMEOUT");
        @(negedge clk);source_stopped=0;ingress_isolated=0;
        need(epoch==desired_epoch && bridge_epoch==epoch && bridge_lease_held && lease_held,"JOINT_ARM_STATE");
    endtask
    task automatic fill;
        tx(frame(0,expected_data,epoch),0);tx(frame(1,expected_data,epoch),0);
    endtask
    task automatic publish(input integer status);
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        repeat(200) begin @(negedge clk);if(result_valid) break;end
        need(result_valid && result_epoch==epoch && result_tag==arm_tag && result_status==status,"RESULT_STATUS");
        set_ack(epoch,arm_tag);idle(2);
    endtask
    task automatic retire;
        need(ack_ready,"RETIRE_READY");wr('h10,4);idle(3);
        need(!result_valid && !bridge_lease_held && bridge_idle && pool_drained && !banks_used && !pending_counts,"RETIRED_DRAIN");
    endtask
    task automatic held_read_begin;
        @(negedge clk);s_axil_araddr='h6c;s_axil_arvalid=1;
        do @(posedge clk);while(!s_axil_arready);
        @(negedge clk);s_axil_arvalid=0;
        need(s_axil_rvalid && s_axil_rdata==32'h80000001 && !s_axil_rresp,"HELD_R_BEGIN");
    endtask
    task automatic held_read_end;
        need(s_axil_rvalid && s_axil_rdata==32'h80000001 && !s_axil_rresp,"HELD_R_SURVIVES");
        s_axil_rready=1;@(posedge clk);@(negedge clk);s_axil_rready=0;
        mmio_reads++;$display("NTRD case=%0d CTL_READ n=%0d addr=0000006c data=80000001 resp=0 cycle=%0d",CASE_ID,mmio_reads,cycle);
    endtask
    always @(posedge clk) if(rst_n && !passed) begin
        if(arm_valid && arm_ready) arms++;
        if(line_valid && line_ready) lines++;
        if(cmd_valid && cmd_ready) begin commands++;commanded[job]=1;end
        if(data_valid && data_ready) begin writes++;input_seen[job]=1;end
        if(|(cpl_valid & cpl_ready)) returns++;
        if(done_valid && done_ready) dones++;
        if(ack_valid && ack_ready) acks++;
    end
    genvar g;
    generate for(g=0;g<2;g++) begin: target
        reg held_aw=0,held_w=0;
        reg [57:0] previous_aw;
        reg [576:0] previous_w;
        always @(negedge ui_clk[g]) begin
            m_awready[g]=memory_reset_n[g] && mem_cycle[g]%4!=0;
            m_wready[g]=memory_reset_n[g] && mem_cycle[g]%7!=0;
            if(!memory_reset_n[g]) m_bvalid[g]=0;
            else if(!m_bvalid[g] && !hold_b[g] && aw_job[g]==job && w_job[g]==job && b_job[g]!=job && mem_cycle[g]%5==0)
                m_bvalid[g]=1;
        end
        always @(posedge ui_clk[g]) begin
            if(!memory_reset_n[g]) begin
                aw_job[g]=-2;w_job[g]=-2;b_job[g]=-2;mem_cycle[g]=0;held_aw=0;held_w=0;
            end else begin
                mem_cycle[g]++;
                if(held_aw && (!m_awvalid[g] || {m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]}!==previous_aw)) $fatal(1,"MEM_AW_STABILITY");
                if(held_w && (!m_wvalid[g] || {m_wlast[g],m_wstrb[g*64+:64],m_wdata[g*512+:512]}!==previous_w)) $fatal(1,"MEM_W_STABILITY");
                held_aw=m_awvalid[g] && !m_awready[g];held_w=m_wvalid[g] && !m_wready[g];
                previous_aw={m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]};
                previous_w={m_wlast[g],m_wstrb[g*64+:64],m_wdata[g*512+:512]};
                if(m_awvalid[g] && m_awready[g]) begin
                    if(job<0 || !commanded[job] || g!=job%2 || aw_seen[job] || 64'(m_awaddr[g*37+:37])!==address(job) || m_awlen[g*8+:8]!=0 || m_awid[g*4+:4]!=0 || m_awsize[g*3+:3]!=6 || m_awburst[g*2+:2]!=1 || m_awcache[g*4+:4]!=0) $fatal(1,"MEMORY_ADDRESS");
                    aw_seen[job]=1;aw_job[g]=job;
                    $display("NTRD case=%0d MEM_AW ps=%0d job=%0d channel=%0d address=%h len=%h id=%h size=%h burst=%h cache=%h",CASE_ID,ps(),job,g,m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]);
                end
                if(m_wvalid[g] && m_wready[g]) begin
                    if(job<0 || !input_seen[job] || g!=job%2 || w_seen[job] || m_wdata[g*512+:512]!==pattern(job) || m_wstrb[g*64+:64]!==64'hffffffffffffffff || !m_wlast[g]) $fatal(1,"MEMORY_PAYLOAD");
                    w_seen[job]=1;w_job[g]=job;
                    $display("NTRD case=%0d MEM_W ps=%0d job=%0d channel=%0d data=%h strb=%h last=%b",CASE_ID,ps(),job,g,m_wdata[g*512+:512],m_wstrb[g*64+:64],m_wlast[g]);
                end
                if(m_bvalid[g] && m_bready[g]) begin
                    if(aw_job[g]!=job || w_job[g]!=job || b_seen[job]) $fatal(1,"MEM_B_CONTEXT");
                    b_seen[job]=1;b_job[g]=job;m_bvalid[g]<=0;
                    $display("NTRD case=%0d MEM_B ps=%0d job=%0d channel=%0d id=%h status=%h",CASE_ID,ps(),job,g,m_bid[g*4+:4],m_bresp[g*2+:2]);
                end
            end
        end
    end endgenerate

    initial begin
        external_reset_async=1;rearm=1;ui_rst=3;
        s_axi_awsize=6;s_axi_awburst=1;
        for(integer j=0;j<4;j++) begin commanded[j]=0;input_seen[j]=0;aw_seen[j]=0;w_seen[j]=0;b_seen[j]=0;end
        #40;external_reset_async=0;
        #1100;calibrated[0]=1;ui_rst[0]=0;
        #1100;need(!admit && !running && !rst_n,"ONE_READY_EARLY");calibrated[1]=1;ui_rst[1]=0;
        wait(admit);idle(4);need(rst_n && pool_drained,"COLD_START");
        rd(0,'h4452544e);rd(4,'h10000);rd(8,'hff);wr('h14,1);
        if(CASE_ID==0) begin
            setup(0);fill();publish(0);retire();
            setup(1);tx(frame(0,expected_data,epoch),0);wr('h10,2);publish(1);retire();
            setup(2);tx(frame(0,expected_data,epoch),0);tx(frame(0,expected_data^512'd1,epoch),2);publish(4);retire();
            setup(3);fill();publish(0);retire();
            rd('h18,4);rd('h1c,4);rd('hb4,1);
            need(!session_invalid && !pool_fault && !control_fault && !protocol_fault,"HEALTHY_STATE");
        end else begin
            setup(CASE_ID%2);
            if(CASE_ID==1 || CASE_ID==7) tx(frame(0,expected_data,epoch),0);
            if(CASE_ID==2) begin hold_b=3;fill();wait(aw_seen[job] && w_seen[job]);end
            if(CASE_ID>=3 && CASE_ID<=5) begin fill();publish(0);end
            if(CASE_ID==4) caw('h10);
            if(CASE_ID==5) held_read_begin();
            if(CASE_ID==6) aw();
            if(CASE_ID==7) begin caw('h14);cwd(0,15);wait(s_axil_bvalid);end
            @(negedge clk);
            $display("NTRD case=%0d FAULT ps=%0d job=%0d epoch=%h cookie=%h result=%b status=%h",CASE_ID,ps(),job,epoch,next_cookie,result_valid,result_status);
            case(CASE_ID)
                1:calibrated[1]=0;
                2,6:external_reset_async=1;
                3:run_source=0;
                4:run_memory[1]=0;
                5:ui_rst[0]=1;
                7:calibrated[0]=0;
            endcase
            #1800;
            need(!admit && !source_reset_n && memory_reset_n==0 && restart_required,"POOL_NOT_CLOSED");
            need(rst_n && bridge_lease_held && lease_held && epoch==desired_epoch && bridge_epoch==epoch,"LOST_LEASE");
            calibrated=3;ui_rst=0;external_reset_async=0;run_source=1;run_memory=3;
            #1800;@(negedge clk);
            if(CASE_ID==4) begin cwd(4,15);cb_response('h10,4,15,2);end
            if(CASE_ID==5) held_read_end();
            if(CASE_ID==6) begin w(frame(0,expected_data,epoch));response(2);need(valid_bytes==0,"LOST_FRAME_REJECTED");end
            if(CASE_ID==7) begin cb_response('h14,0,15,0);rd('h14,0);need(!irq,"MASKED_IRQ_RETAINED");wr('h14,1);end
            source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;idle(4);
            set_ack(epoch,arm_tag);wr('h10,4,15,2);wr('h10,1,15,2);wr('hb0,0,15,2);
            rd('h18,1);rd('h1c,0);rd('hb4,32'(job%2));rd('hb8,3);
            rd('h50,desired_epoch[31:0]);rd('h54,desired_epoch[63:32]);rd('h64,desired_epoch[63:32]);rd('h6c,'h80000001);
            need(rst_n && session_invalid && restart_required_source && bridge_lease_held && lease_held && epoch==desired_epoch && irq &&
                !admit && !source_reset_n && memory_reset_n==0 && !rearm_ready && !ack_ready && !arm_ready && !control_fault,"LOST_SESSION_REOPENED");
            if(CASE_ID>=3 && CASE_ID<=5) need(result_valid && result_status==0,"HELD_RESULT_CHANGED");
            else if(CASE_ID==2) need(!result_valid,"NEW_RESULT_AFTER_LOSS");
            else need(result_valid && result_status==1,"PARTIAL_LOST_RESULT");
        end
        $display("NTRD case=%0d CASE_PASS ps=%0d job=%0d held=%b invalid=%b epoch=%h next=%h arms=%0d lines=%0d commands=%0d data=%0d cpl=%0d done=%0d acks=%0d frames=%0d rejects=%0d",CASE_ID,ps(),job,bridge_lease_held,session_invalid,epoch,next_cookie,arms,lines,commands,writes,returns,dones,acks,bcount,rejections);
        $display("NTRD case=%0d CTL_PASS writes=%0d reads=%0d seed=%0d cycles=%0d",CASE_ID,mmio_writes,mmio_reads,seed,cycle);
        passed=1;
    end
    always @(posedge clk) begin
        cycle=cycle+1;
        if(!passed) $display("NTRD case=%0d LIFE_TRACE cycle=%0d ps=%0d cold=%b poolreset=%b memreset=%h admit=%b restart=%b synced=%b rearm_ready=%b calibrated=%h ui_rst=%h source_run=%b memory_run=%h banks=%h pending=%h",CASE_ID,cycle,ps(),rst_n,source_reset_n,memory_reset_n,admit,restart_required,restart_required_source,rearm_ready,calibrated,ui_rst,run_source,run_memory,banks_used,pending_counts);
        if(!passed) $display("NTRD case=%0d OWN_TRACE cycle=%0d case=%0d rst=%h stop=%h stopped=%h isolated=%h qe=%h admit=%h restart=%h pf=%h drained=%h av=%h ar=%h car=%h bar=%h ae=%h at=%h aw=%h aa=%h ac=%h kv=%h kr=%h ke=%h kt=%h rr=%h awv=%h awr=%h awa=%h awi=%h awl=%h aws=%h awb=%h awk=%h wv=%h wr=%h wd=%h ws=%h wl=%h bv=%h br=%h bi=%h bs=%h lv=%h lr=%h le=%h lt=%h la=%h ld=%h cv=%h cr=%h cc=%h ca=%h ct=%h cl=%h dv=%h dr=%h data=%h pv=%h pr=%h pt=%h ps=%h ov=%h ore=%h oe=%h ot=%h os=%h ops=%h err=%h rv=%h rc=%h re=%h rt=%h rs=%h epoch=%h held=%h be=%h bh=%h idle=%h phase=%h fail=%h error=%h bytes=%h ti=%h fault=%h lost=%h invalid=%h exhausted=%h next=%h rejected=%h frames=%h identity=%h code=%h faults=%h",CASE_ID,cycle,job,
            rst_n,stop,source_stopped,ingress_isolated,quiesce_epoch,pool_admit,pool_restart_required,pool_fault,pool_drained,arm_valid,arm_ready,collector_arm_ready,bridge_arm_ready,arm_epoch,arm_tag,arm_window,arm_destination,arm_channel,ack_valid,ack_ready,ack_epoch,ack_tag,retire_ready,s_axi_awvalid,s_axi_awready,s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock,s_axi_wvalid,s_axi_wready,s_axi_wdata,s_axi_wstrb,s_axi_wlast,s_axi_bvalid,s_axi_bready,s_axi_bid,s_axi_bresp,line_valid,line_ready,line_epoch,line_tag,line_address,line_data,cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len,data_valid,data_ready,data,cpl_valid,cpl_ready,cpl_tag,cpl_status,done_valid,done_ready,done_epoch,done_tag,done_status,done_pool_status,done_error,result_valid,result_ready,result_epoch,result_tag,result_status,epoch,lease_held,bridge_epoch,bridge_lease_held,bridge_idle,bridge_phase,failed,error_code,valid_bytes,transport_idle,protocol_fault,session_lost,session_invalid,exhausted,next_cookie,rejected_completions,rejected_frames,rejected_identity,last_reject_code,last_reject_faults);
        if(!passed) $display("NTRD case=%0d CTL_TRACE cycle=%0d aw=%h av=%h ar=%h wd=%h ws=%h wv=%h wr=%h bv=%h br=%h bs=%h ra=%h rv=%h rr=%h ov=%h ore=%h od=%h os=%h irq=%h fault=%h",CASE_ID,cycle,s_axil_awaddr,s_axil_awvalid,s_axil_awready,s_axil_wdata,s_axil_wstrb,s_axil_wvalid,s_axil_wready,s_axil_bvalid,s_axil_bready,s_axil_bresp,s_axil_araddr,s_axil_arvalid,s_axil_arready,s_axil_rvalid,s_axil_rready,s_axil_rdata,s_axil_rresp,irq,control_fault);
    end
endmodule

module tb_ntrpu_pcie_managed_lifetime;
    wire [7:0] passed;
    for(genvar g=0;g<8;g++) begin: cases
        managed_lifetime_case #(.CASE_ID(g)) instance_case(.passed(passed[g]));
    end
    initial begin wait(&passed);$display("NTRD_LIFETIME_SUITE_PASS cases=8");$finish;end
    initial begin #35000;$fatal(1,"NTRD_LIFETIME_TIMEOUT");end
endmodule
