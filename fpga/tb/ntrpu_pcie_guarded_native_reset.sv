`timescale 1ns/1ps
`ifdef XILINX_SIMULATOR
module short(in1,in1);inout in1;endmodule
`endif
// Native XDMA BAR0/BAR2, two host CDCs and two generic MIG models.
// External isolation and direct UI readback remain trusted test drivers.
module ntrpu_pcie_guarded_native_reset_harness;
    localparam integer CASE_ID=0;
    reg passed=0;
    reg mgmt_clk=0, source_clk=0, external_reset_async=0, rearm=0;
    wire [1:0] ui_clk, calibrated, ui_rst, interrupt_ecc;
    wire [1:0] m_rvalid;
    wire source_reset_n, admit, running, rearm_ready, restart_required;
    wire peers_reset_req, source_alive, lifetime_reset_n, restart_required_source;
    wire [1:0] memory_reset_n, memory_alive;
    wire [7:0] banks_used;
    wire [9:0] pending_counts;
    wire [1:0] m_awvalid, m_wvalid, m_wlast, m_bready;
    wire [1:0] m_awready, m_wready, m_bvalid;
    wire [73:0] m_awaddr;
    wire [15:0] m_awlen;
    wire [7:0] m_awid, m_awcache;
    wire [5:0] m_awsize;
    wire [3:0] m_awburst;
    wire [1023:0] m_wdata;
    wire [127:0] m_wstrb;
    wire [7:0] m_bid;
    wire [3:0] m_bresp;
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
    wire [63:0] s_axi_awaddr=board.EP.m_axib_awaddr;
    wire [3:0] s_axi_awid=board.EP.m_axib_awid;
    wire [7:0] s_axi_awlen=board.EP.m_axib_awlen;
    wire [2:0] s_axi_awsize=board.EP.m_axib_awsize;
    wire [1:0] s_axi_awburst=board.EP.m_axib_awburst;
    wire s_axi_awlock=board.EP.m_axib_awlock;
    wire s_axi_awvalid=board.EP.m_axib_awvalid;
    wire s_axi_awready;
    wire [511:0] s_axi_wdata=board.EP.m_axib_wdata;
    wire [63:0] s_axi_wstrb=board.EP.m_axib_wstrb;
    wire s_axi_wlast=board.EP.m_axib_wlast;
    wire s_axi_wvalid=board.EP.m_axib_wvalid;
    wire s_axi_wready;
    wire [3:0] s_axi_bid;
    wire [1:0] s_axi_bresp;
    wire s_axi_bvalid;
    wire s_axi_bready=board.EP.m_axib_bready;
    assign board.EP.m_axib_awready=s_axi_awready;
    assign board.EP.m_axib_bid=s_axi_bid;
    assign board.EP.m_axib_bresp=s_axi_bresp;
    assign board.EP.m_axib_bvalid=s_axi_bvalid;
    assign board.EP.m_axib_wready=s_axi_wready;
    assign board.EP.m_axib_arready=1'b0;
    assign board.EP.m_axib_rid=4'b0;
    assign board.EP.m_axib_rdata=512'b0;
    assign board.EP.m_axib_rresp=2'b11;
    assign board.EP.m_axib_rlast=1'b1;
    assign board.EP.m_axib_rvalid=1'b0;
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
    wire [31:0] s_axil_awaddr=board.EP.m_axil_awaddr;
    wire [31:0] s_axil_wdata=board.EP.m_axil_wdata;
    wire [31:0] s_axil_araddr=board.EP.m_axil_araddr;
    wire [3:0] s_axil_wstrb=board.EP.m_axil_wstrb;
    wire s_axil_awvalid=board.EP.m_axil_awvalid;
    wire s_axil_wvalid=board.EP.m_axil_wvalid;
    wire s_axil_bready=board.EP.m_axil_bready;
    wire s_axil_arvalid=board.EP.m_axil_arvalid;
    wire s_axil_rready=board.EP.m_axil_rready;
    wire s_axil_awready;
    wire s_axil_wready;
    wire s_axil_arready;
    wire s_axil_bvalid;
    wire s_axil_rvalid;
    wire [1:0] s_axil_bresp;
    wire [1:0] s_axil_rresp;
    wire [31:0] s_axil_rdata;
    assign board.EP.m_axil_arready=s_axil_arready;
    assign board.EP.m_axil_awready=s_axil_awready;
    assign board.EP.m_axil_bresp=s_axil_bresp;
    assign board.EP.m_axil_bvalid=s_axil_bvalid;
    assign board.EP.m_axil_rdata=s_axil_rdata;
    assign board.EP.m_axil_rresp=s_axil_rresp;
    assign board.EP.m_axil_rvalid=s_axil_rvalid;
    assign board.EP.m_axil_wready=s_axil_wready;
    wire irq, control_fault;

    wire host_clk=board.EP.user_clk;
    wire link_up_async=board.EP.user_lnk_up;
    wire endpoint_reset_async=!board.EP.user_resetn || !board.sys_rst_n;
    reg guard_arm=0,guard_peers_held=0,guard_peers_ready=0;
    reg healthy_done=0,loss_test_started=0;
    reg [63:0] loss_cookie,loss_epoch;
    integer loss_window=0;
    // Runtime coordination is absent. Never fabricate its held/ready ACK.
    // A warm link may return while the guard keeps local transport reset.
    wire host_reset_n,transport_reset_n,host_reset_req;
    wire host_admit,source_admit,host_alive,guard_source_alive;
    wire guard_invalidate,loss_mgmt,waiting_ack,guard_running;
    wire [4:0] loss_causes;
    wire [63:0] first_loss_tick;
    wire [3:0] guard_state;
    wire owner_invalid_ack,boundary_invalid;
    wire [7:0] boundary_fenced;
    wire [15:0] boundary_pending,boundary_fault;
    wire [511:0] boundary_accepted,boundary_delivered,boundary_abandoned;
    wire host_rst_n=host_reset_n,transport_rst_n=transport_reset_n;
    // Cold-only peer observation: actual native requests must be inactive.
    // This does not make the native endpoint obey the guard's runtime request.
    wire native_requests_idle=(s_axi_awvalid===1'b0 && s_axi_wvalid===1'b0 &&
        s_axil_awvalid===1'b0 && s_axil_wvalid===1'b0 && s_axil_arvalid===1'b0);
    always @(posedge mgmt_clk) begin
        guard_peers_held<=!loss_test_started && !host_reset_n && !transport_reset_n && native_requests_idle;
        guard_peers_ready<=!loss_test_started && host_reset_n && transport_reset_n &&
            board.EP.user_resetn && board.EP.user_lnk_up;
        if(!passed) $display("NTRD_GUARD_NATIVE ps=%0d cold=%b link=%b epreset=%b perst=%b loss=%b invalid=%b ack=%b req=%b hrst=%b srst=%b ha=%b sa=%b state=%h held=%b ready=%b idle=%b",
            ps(),lifetime_reset_n,board.EP.user_lnk_up,board.EP.user_resetn,board.sys_rst_n,
            loss_mgmt,guard_invalidate,owner_invalid_ack,host_reset_req,host_reset_n,transport_reset_n,
            host_admit,source_admit,guard_state,guard_peers_held,guard_peers_ready,native_requests_idle);
    end
    reg suite_started=0;
    integer host_data_b=0,host_control_b=0,host_control_ar=0,host_control_r=0;
    reg [1:0] last_data_b=0,last_control_b=0,last_control_r=0;
    reg [31:0] last_control_data=0,last_read_address=0;
    integer host_cycle=0;
    integer control_src_aw=0,control_src_w=0,control_src_b=0,control_src_ar=0,control_src_r=0;
    initial begin
        wait(lifetime_reset_n);@(negedge mgmt_clk);guard_arm=1;
    end
    wire clk=source_clk,rst_n=lifetime_reset_n;
    reg run_source=1;reg [1:0] run_memory=3;
    always #2.5 if(run_source) source_clk=~source_clk;
    always #5 mgmt_clk=~mgmt_clk;
    integer cycle=0,job=-1,seed=7,mmio_writes=0,mmio_reads=0;
    integer bcount=0,rejections=0,arms=0,lines=0,commands=0,writes=0,returns=0,dones=0,acks=0;
    reg [511:0] expected_data;
    reg [63:0] desired_epoch,desired_destination;
    reg [1:0] hold_b=0,hold_w=0;
    integer aw_count=0,w_count=0,b_count=0,reads=0,one_ready=0;
    integer aw_stall[0:1],w_stall[0:1];
    longint input_at[0:18];
    reg [73:0] araddr=0;
    reg [1:0] arvalid=0,rready=0;
    wire [1:0] arready,rlast;
    wire [7:0] rid;wire [3:0] rresp;wire [1023:0] rdata;
    integer commanded[0:18],input_seen[0:18],aw_seen[0:18],w_seen[0:18],b_seen[0:18];
    ntrpu_pcie_fragment_dual_host_guarded #(.FIRST_COOKIE(64'hffffffff)) dut (.*);
    function automatic longint ps;return longint'($realtime/1ps);endfunction
    function automatic [63:0] address(input integer j);
        if(j==12) return 64'h1fffffffc0;
        if(j==13) return 64'h100001040;
        if(j==14) return 64'h100001080;
        if(j==17) return 64'h100001040;
        if(j>=15) return 64'h1fffffffc0;
        case(j/2)
            0:return 0;
            1:return 64'h100000040;
            2:return 64'h800000000;
            3:return 64'h1000000000;
            4:return 64'h1800000000;
            5:return 64'h1fffffffc0;
        endcase
        return 0;
    endfunction
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"DUAL_CONTROL_CHECK case=%0d cycle=%0d %s",job,cycle,message);
    endtask
    task automatic idle(input integer n);repeat(n) @(negedge host_clk);endtask
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
    // Root-port tasks generate the packets; these witnesses only wait for
    // endpoint acceptance and never drive or force AXI outputs from XDMA.
    task automatic tx(input reg [511:0] payload,input integer resp);
        integer wanted;wanted=host_data_b+1;
        board.RP.tx_usrapp.NTRPU_FRAME_WRITE(64'h9000fffc0,payload);
        repeat(4000) begin @(negedge host_clk);if(host_data_b==wanted) break;end
        need(host_data_b==wanted && last_data_b==resp,"NATIVE_FRAME_RESPONSE");
        bcount++;if(resp!=0) rejections++;
    endtask
    task automatic wr(input [31:0] address,input [31:0] value,input [3:0] strobe=15,input [1:0] expected=0);
        integer wanted;wanted=host_control_b+1;
        need(strobe==15,"NATIVE_MMIO_STROBE_PROFILE");
        board.RP.tx_usrapp.NTRPU_CTRL_WRITE(address,value);
        repeat(4000) begin @(negedge host_clk);if(host_control_b==wanted) break;end
        need(host_control_b==wanted && last_control_b==expected,"NATIVE_CONTROL_WRITE_RESPONSE");
        mmio_writes++;
        $display("NTRD case=%0d CTL_WRITE n=%0d addr=%h data=%h strb=%h resp=%h cycle=%0d",CASE_ID,mmio_writes,address,value,strobe,expected,cycle);
    endtask
    task automatic rd(input [31:0] address,input [31:0] expected,input [1:0] resp=0,input integer hold_cycles=3);
        integer wanted;wanted=host_control_r+1;
        need(resp==0,"NATIVE_READ_COMPLETION_PROFILE");
        board.RP.tx_usrapp.NTRPU_CTRL_READ(address,expected,'1);
        repeat(4000) begin @(negedge host_clk);if(host_control_r==wanted) break;end
        need(host_control_r==wanted && last_control_r==resp && last_control_data==expected,"NATIVE_CONTROL_READ_RESPONSE");
        mmio_reads++;
        $display("NTRD case=%0d CTL_READ n=%0d addr=%h data=%h resp=%h cycle=%0d",CASE_ID,mmio_reads,address,expected,resp,cycle);
    endtask
    task automatic desc(input [63:0] e,input [63:0] t,input [63:0] win,input [63:0] dest);
        wr('h20,e[31:0]);wr('h24,e[63:32]);wr('h28,t[31:0]);wr('h2c,t[63:32]);
        wr('h30,win[31:0]);wr('h34,win[63:32]);wr('h38,dest[31:0]);wr('h3c,dest[63:32]);
    endtask
    task automatic set_ack(input [63:0] e,input [63:0] t);
        wr('h40,e[31:0]);wr('h44,e[63:32]);wr('h48,t[31:0]);wr('h4c,t[63:32]);
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
        repeat(2000) begin @(negedge clk);if(result_valid) break;end
        need(result_valid && result_epoch==epoch && result_tag==arm_tag && result_status==status,"RESULT_STATUS");
        set_ack(epoch,arm_tag);idle(2);
    endtask
    task automatic retire;
        need(ack_ready,"RETIRE_READY");
        @(negedge clk);source_stopped=0;ingress_isolated=0;
        wr('h10,4,15,2);need(result_valid && bridge_lease_held,"NATIVE_REJECTED_ACK_RETAINED");
        @(negedge clk);source_stopped=1;ingress_isolated=1;
        repeat(4) @(negedge clk);need(ack_ready,"RESTORED_RETIRE_BARRIER");
        wr('h10,4);idle(8);
        need(!result_valid && !bridge_lease_held && bridge_idle && pool_drained && !banks_used && !pending_counts,"RETIRED_DRAIN");
    endtask
    always @(posedge clk) if(rst_n && !passed) begin
        if(arm_valid && arm_ready) arms++;
        if(line_valid && line_ready) lines++;
        if(cmd_valid && cmd_ready) begin commands++;commanded[job]=1;end
        if(data_valid && data_ready) begin writes++;input_seen[job]=1;input_at[job]=ps();end
        if(|(cpl_valid & cpl_ready)) returns++;
        if(done_valid && done_ready) dones++;
        if(ack_valid && ack_ready) acks++;
    end
    for(genvar g=0;g<2;g++) begin: mem
        ntrpu_dual_mig_fixture #(.PORT(g)) model (
            .clk(ui_clk[g]),.calibrated(calibrated[g]),.ui_rst(ui_rst[g]),.interrupt_ecc(interrupt_ecc[g]),
            .rst_n(memory_reset_n[g]),.hold_w(hold_w[g]),.hold_b(hold_b[g]),
            .m_axi_awvalid(m_awvalid[g]),.m_axi_awready(m_awready[g]),.m_axi_awaddr(m_awaddr[g*37+:37]),
            .m_axi_awlen(m_awlen[g*8+:8]),.m_axi_awid(m_awid[g*4+:4]),.m_axi_awcache(m_awcache[g*4+:4]),
            .m_axi_awsize(m_awsize[g*3+:3]),.m_axi_awburst(m_awburst[g*2+:2]),
            .m_axi_wvalid(m_wvalid[g]),.m_axi_wready(m_wready[g]),.m_axi_wlast(m_wlast[g]),
            .m_axi_wdata(m_wdata[g*512+:512]),.m_axi_wstrb(m_wstrb[g*64+:64]),
            .m_axi_bvalid(m_bvalid[g]),.m_axi_bready(m_bready[g]),.m_axi_bid(m_bid[g*4+:4]),.m_axi_bresp(m_bresp[g*2+:2]),
            .m_axi_araddr(araddr[g*37+:37]),.m_axi_arvalid(arvalid[g]),.m_axi_arready(arready[g]),
            .m_axi_rvalid(m_rvalid[g]),.m_axi_rready(rready[g]),.m_axi_rlast(rlast[g]),
            .m_axi_rdata(rdata[g*512+:512]),.m_axi_rresp(rresp[g*2+:2]),.m_axi_rid(rid[g*4+:4])
        );
        reg held_aw=0,held_w=0,held_r=0;
        reg [57:0] previous_aw;
        reg [576:0] previous_w;
        reg [518:0] previous_r;
        longint last_ui=0;integer ui_edges=0;
        integer read_identity=-1,read_index=0;
        always @(posedge ui_clk[g]) begin
            if(calibrated[g] && !ui_rst[g]) begin
                if(ui_edges>0 && ps()-last_ui!=5000) $fatal(1,"UI_PERIOD");
                last_ui=ps();ui_edges++;
                if(ui_edges==4) $display("MANAGED_MIG_CLOCK ps=%0d channel=%0d period=5000 phase=%0d",ps(),g,ps()%5000);
            end
            if(memory_reset_n[g]) begin
                if(held_aw && (!m_awvalid[g] || {m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]}!==previous_aw)) $fatal(1,"AW_STABILITY");
                if(held_w && (!m_wvalid[g] || {m_wlast[g],m_wstrb[g*64+:64],m_wdata[g*512+:512]}!==previous_w)) $fatal(1,"W_STABILITY");
                if(held_r && (!m_rvalid[g] || {rlast[g],rid[g*4+:4],rresp[g*2+:2],rdata[g*512+:512]}!==previous_r)) $fatal(1,"R_STABILITY");
                held_aw=m_awvalid[g] && !m_awready[g];held_w=m_wvalid[g] && !m_wready[g];held_r=m_rvalid[g] && !rready[g];
                previous_aw={m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]};
                previous_w={m_wlast[g],m_wstrb[g*64+:64],m_wdata[g*512+:512]};
                previous_r={rlast[g],rid[g*4+:4],rresp[g*2+:2],rdata[g*512+:512]};
                if(held_aw) aw_stall[g]++;
                if(held_w) w_stall[g]++;
                if(m_awvalid[g] || m_wvalid[g] || arvalid[g])
                    if(!calibrated[g] || ui_rst[g] || !admit) $fatal(1,"MEMORY_NOT_READY");
                if(m_awvalid[g] && m_awready[g]) begin
                    if(job<0 || !commanded[job] || g!=job%2 || aw_seen[job] || 64'(m_awaddr[g*37+:37])!==address(job) || m_awlen[g*8+:8]!=0 || m_awid[g*4+:4]!=0 || m_awsize[g*3+:3]!=6 || m_awburst[g*2+:2]!=1 || m_awcache[g*4+:4]!=0) $fatal(1,"MEMORY_ADDRESS");
                    aw_seen[job]=1;aw_count++;
                    $display("MANAGED_MIG_AW ps=%0d job=%0d channel=%0d address=%h len=%h id=%h size=%h burst=%h cache=%h",ps(),job,g,m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]);
                end
                if(m_wvalid[g] && m_wready[g]) begin
                    if(job<0 || !input_seen[job] || input_at[job]>=ps() || g!=job%2 || w_seen[job] || m_wdata[g*512+:512]!==pattern(job) || m_wstrb[g*64+:64]!==64'hffffffffffffffff || !m_wlast[g]) $fatal(1,"MEMORY_PAYLOAD");
                    w_seen[job]=1;w_count++;
                    $display("MANAGED_MIG_W ps=%0d job=%0d channel=%0d data=%h strb=%h last=%b",ps(),job,g,m_wdata[g*512+:512],m_wstrb[g*64+:64],m_wlast[g]);
                end
                if(m_bvalid[g] && m_bready[g]) begin
                    if(!aw_seen[job] || !w_seen[job] || b_seen[job] || m_bid[g*4+:4]!=0 || m_bresp[g*2+:2]!=0) $fatal(1,"MEMORY_B_CONTEXT");
                    b_seen[job]=1;b_count++;
                    $display("MANAGED_MIG_B ps=%0d job=%0d channel=%0d id=%h status=%h",ps(),job,g,m_bid[g*4+:4],m_bresp[g*2+:2]);
                end
                if(arvalid[g] && arready[g])
                    $display("MANAGED_MIG_AR ps=%0d channel=%0d index=%0d identity=%0d address=%h",ps(),g,read_index,read_identity,araddr[g*37+:37]);
                if(m_rvalid[g] && rready[g]) begin
                    reads++;
                    $display("MANAGED_MIG_R ps=%0d channel=%0d index=%0d identity=%0d data=%h status=%h id=%h last=%b",ps(),g,read_index,read_identity,rdata[g*512+:512],rresp[g*2+:2],rid[g*4+:4],rlast[g]);
                end
            end
        end
        task automatic readback(input integer identity);
            read_identity=identity;read_index++;
            @(negedge ui_clk[g]);araddr[g*37+:37]=37'(address(identity));arvalid[g]=1;
            do @(posedge ui_clk[g]);while(!arready[g]);
            @(negedge ui_clk[g]);arvalid[g]=0;
            while(!m_rvalid[g]) @(negedge ui_clk[g]);
            repeat(3) @(negedge ui_clk[g]);
            if(rid[g*4+:4]!=0 || rresp[g*2+:2]!=0 || !rlast[g] || rdata[g*512+:512]!==pattern(identity)) $fatal(1,"READBACK_BYTES");
            rready[g]=1;@(posedge ui_clk[g]);@(negedge ui_clk[g]);rready[g]=0;
        endtask
    end

    always @(posedge clk) if(rst_n) begin
        if(dut.host.k_axil_awvalid && dut.host.k_axil_awready) control_src_aw++;
        if(dut.host.k_axil_wvalid && dut.host.k_axil_wready) control_src_w++;
        if(dut.host.k_axil_bvalid && dut.host.k_axil_bready) control_src_b++;
        if(dut.host.k_axil_arvalid && dut.host.k_axil_arready) control_src_ar++;
        if(dut.host.k_axil_rvalid && dut.host.k_axil_rready) control_src_r++;
    end
    function automatic bit control_quiet;
        return !s_axil_awvalid && !s_axil_wvalid && !s_axil_bvalid && !s_axil_arvalid && !s_axil_rvalid &&
               !dut.host.k_axil_awvalid && !dut.host.k_axil_wvalid && !dut.host.k_axil_bvalid && !dut.host.k_axil_arvalid && !dut.host.k_axil_rvalid;
    endfunction
    task automatic drain_control;
        repeat(200) begin
            @(negedge clk);
            if(control_src_aw==mmio_writes && control_src_w==mmio_writes && control_src_b==mmio_writes &&
               control_src_ar==mmio_reads && control_src_r==mmio_reads && control_quiet()) break;
        end
        need(control_src_aw==mmio_writes && control_src_w==mmio_writes && control_src_b==mmio_writes &&
             control_src_ar==mmio_reads && control_src_r==mmio_reads && control_quiet(),"CONTROL_CDC_DRAIN");
        $display("NTRD_HOST_DRAIN ps=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d host_writes=%0d host_reads=%0d",ps(),control_src_aw,control_src_w,control_src_b,control_src_ar,control_src_r,mmio_writes,mmio_reads);
        repeat(8) begin @(negedge clk);need(control_quiet(),"CONTROL_CDC_LATE_VALID");end
        $display("NTRD_HOST_DRAIN_QUIET ps=%0d cycles=8",ps());
    endtask
    initial begin external_reset_async=1;#40;external_reset_async=0;end
    initial begin wait(rearm_ready);@(negedge mgmt_clk);rearm=1;repeat(4)@(negedge mgmt_clk);rearm=0;end
    always @(posedge clk) if(calibrated==1 && !ui_rst[0]) begin
        one_ready++;need(!admit && !arm_ready && !cmd_ready,"ONE_READY_EARLY");
    end
    task automatic run_suite;
        need(host_rst_n && rst_n && host_control_r==1 && mmio_reads==1,"ENUMERATION_PROBE_ACCOUNTED");
        suite_started=1;
        for(integer j=0;j<19;j++) begin commanded[j]=0;input_seen[j]=0;aw_seen[j]=0;w_seen[j]=0;b_seen[j]=0;end
        for(integer ch=0;ch<2;ch++) begin aw_stall[ch]=0;w_stall[ch]=0;end
        $display("MANAGED_MIG_BEGIN jobs=18 first=00000000ffffffff");
        wait(admit);idle(4);need(rst_n && pool_drained,"COLD_START");
        wait(guard_running && host_admit && source_admit);idle(8);
        rd(4,'h10000);rd(8,'hff);
        wr('hc4,1,15,3);wr('h04,0,15,2);rd('hbc,2);rd('hc0,37);wr('h14,1);
        for(integer j=0;j<12;j++) begin setup(j);fill();publish(0);retire();end
        fork
            begin for(integer j=0;j<12;j+=2) mem[0].readback(j);end
            begin for(integer j=1;j<12;j+=2) mem[1].readback(j);end
        join
        setup(12);tx(frame(0,expected_data,epoch),0);wr('h10,2);publish(1);retire();
        setup(13);tx(frame(0,expected_data,epoch),0);tx(frame(0,expected_data^512'd1,epoch),2);publish(4);retire();
        fork mem[0].readback(10);mem[1].readback(11);join
        setup(14);fill();publish(0);retire();
        @(negedge ui_clk[1]);hold_b[1]=1;
        setup(15);fill();wait(aw_seen[15] && w_seen[15]);wr('h10,2);
        idle(12);need(!result_valid && bridge_lease_held,"B_HELD_EARLY_RESULT");
        @(negedge ui_clk[1]);hold_b[1]=0;publish(1);retire();
        @(negedge ui_clk[0]);hold_w[0]=1;
        setup(16);fill();wait(m_wvalid[0]);wr('h10,2);
        idle(12);need(!result_valid && bridge_lease_held,"W_HELD_EARLY_RESULT");
        @(negedge ui_clk[0]);hold_w[0]=0;publish(1);retire();
        setup(17);fill();publish(0);retire();
        fork
            begin for(integer j=0;j<10;j+=2) mem[0].readback(j);mem[0].readback(16);mem[0].readback(14);end
            begin for(integer j=1;j<11;j+=2) mem[1].readback(j);mem[1].readback(15);mem[1].readback(17);end
        join
        rd('h18,18);rd('h1c,18);rd('hb4,1);rd('hb8,8);drain_control();
        need(arms==18 && lines==16 && commands==16 && writes==16 && returns==16 && dones==16 && acks==18 &&
            aw_count==16 && w_count==16 && b_count==16 && reads==28 && bcount==35 && rejections==1,"CONSERVATION");
        need(pool_drained && !banks_used && !pending_counts && !bridge_lease_held && bridge_idle &&
            !pool_fault && !session_invalid && !restart_required_source && !control_fault && !protocol_fault && admit && one_ready>=500,"FINAL_DRAINED");
        for(integer rank=0;rank<4;rank++) need(mem[0].model.rank_acts[rank]>0 && mem[1].model.rank_acts[rank]>0,"RANK_COVERAGE");
        fork mem[0].model.verify_periodic_complete();mem[1].model.verify_periodic_complete();join
        @(negedge clk);
        // Place the healthy boundary between native/management rising
        // edges so its prefix is independently replayable without edge races.
        while(ps()%20000!=10000) @(negedge clk);
        $display("NTRD_HOST_PASS cycles=%0d ps=%0d writes=%0d reads=%0d frames=%0d",host_cycle,ps(),mmio_writes,mmio_reads,bcount);
        $display("MANAGED_MIG_PASS arms=%0d lines=%0d commands=%0d data=%0d cpl=%0d done=%0d acks=%0d aw=%0d w=%0d b=%0d reads=%0d frames=%0d rejects=%0d one_ready=%0d end_ps=%0d",arms,lines,commands,writes,returns,dones,acks,aw_count,w_count,b_count,reads,bcount,rejections,one_ready,ps());
        $display("NTRD case=0 CTL_PASS writes=%0d reads=%0d seed=%0d cycles=%0d",mmio_writes,mmio_reads,seed,cycle);
        need(guard_running && host_admit && source_admit && !loss_mgmt && !boundary_invalid &&
             boundary_pending==0 && boundary_fault==0 && boundary_abandoned==0,"GUARDED_NATIVE_FINAL_BOUNDARY");
        $display("NTRD_GUARDED_NATIVE_PASS ps=%0d accepted=%h delivered=%h abandoned=%h",ps(),boundary_accepted,boundary_delivered,boundary_abandoned);
        healthy_done=1;
    endtask
    initial begin #700000;$fatal(1,"MANAGED_MIG_TIMEOUT job=%0d phase=%0d",job,bridge_phase);end
    always @(posedge clk) begin
        cycle=cycle+1;
        if(!passed) $display("NTRD_HOST_RESET cycle=%0d ps=%0d host=%b transport=%b lifetime=%b",cycle,ps(),host_rst_n,transport_rst_n,lifetime_reset_n);
        if(!passed) $display("NTRD case=%0d LIFE_TRACE cycle=%0d ps=%0d cold=%b poolreset=%b memreset=%h admit=%b restart=%b synced=%b rearm_ready=%b calibrated=%h ui_rst=%h source_run=%b memory_run=%h banks=%h pending=%h external=%b rearm=%b",CASE_ID,cycle,ps(),rst_n,source_reset_n,memory_reset_n,admit,restart_required,restart_required_source,rearm_ready,calibrated,ui_rst,run_source,run_memory,banks_used,pending_counts,external_reset_async,rearm);
        if(!passed) $display("NTRD case=%0d OWN_TRACE cycle=%0d case=%0d rst=%h stop=%h stopped=%h isolated=%h qe=%h admit=%h restart=%h pf=%h drained=%h av=%h ar=%h car=%h bar=%h ae=%h at=%h aw=%h aa=%h ac=%h kv=%h kr=%h ke=%h kt=%h rr=%h awv=%h awr=%h awa=%h awi=%h awl=%h aws=%h awb=%h awk=%h wv=%h wr=%h wd=%h ws=%h wl=%h bv=%h br=%h bi=%h bs=%h lv=%h lr=%h le=%h lt=%h la=%h ld=%h cv=%h cr=%h cc=%h ca=%h ct=%h cl=%h dv=%h dr=%h data=%h pv=%h pr=%h pt=%h ps=%h ov=%h ore=%h oe=%h ot=%h os=%h ops=%h err=%h rv=%h rc=%h re=%h rt=%h rs=%h epoch=%h held=%h be=%h bh=%h idle=%h phase=%h fail=%h error=%h bytes=%h ti=%h fault=%h lost=%h invalid=%h exhausted=%h next=%h rejected=%h frames=%h identity=%h code=%h faults=%h",CASE_ID,cycle,job,
            rst_n,stop,source_stopped,ingress_isolated,quiesce_epoch,pool_admit,pool_restart_required,pool_fault,pool_drained,arm_valid,arm_ready,collector_arm_ready,bridge_arm_ready,arm_epoch,arm_tag,arm_window,arm_destination,arm_channel,ack_valid,ack_ready,ack_epoch,ack_tag,retire_ready,dut.host.k_axi_awvalid,dut.host.k_axi_awready,dut.host.k_axi_awaddr,dut.host.k_axi_awid,dut.host.k_axi_awlen,dut.host.k_axi_awsize,dut.host.k_axi_awburst,dut.host.k_axi_awlock,dut.host.k_axi_wvalid,dut.host.k_axi_wready,dut.host.k_axi_wdata,dut.host.k_axi_wstrb,dut.host.k_axi_wlast,dut.host.k_axi_bvalid,dut.host.k_axi_bready,dut.host.k_axi_bid,dut.host.k_axi_bresp,line_valid,line_ready,line_epoch,line_tag,line_address,line_data,cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len,data_valid,data_ready,data,cpl_valid,cpl_ready,cpl_tag,cpl_status,done_valid,done_ready,done_epoch,done_tag,done_status,done_pool_status,done_error,result_valid,result_ready,result_epoch,result_tag,result_status,epoch,lease_held,bridge_epoch,bridge_lease_held,bridge_idle,bridge_phase,failed,error_code,valid_bytes,transport_idle,protocol_fault,session_lost,session_invalid,exhausted,next_cookie,rejected_completions,rejected_frames,rejected_identity,last_reject_code,last_reject_faults);
        if(!passed) $display("NTRD case=%0d CTL_TRACE cycle=%0d aw=%h av=%h ar=%h wd=%h ws=%h wv=%h wr=%h bv=%h br=%h bs=%h ra=%h rv=%h rr=%h ov=%h ore=%h od=%h os=%h irq=%h fault=%h",CASE_ID,cycle,dut.host.k_axil_awaddr,dut.host.k_axil_awvalid,dut.host.k_axil_awready,dut.host.k_axil_wdata,dut.host.k_axil_wstrb,dut.host.k_axil_wvalid,dut.host.k_axil_wready,dut.host.k_axil_bvalid,dut.host.k_axil_bready,dut.host.k_axil_bresp,dut.host.k_axil_araddr,dut.host.k_axil_arvalid,dut.host.k_axil_arready,dut.host.k_axil_rvalid,dut.host.k_axil_rready,dut.host.k_axil_rdata,dut.host.k_axil_rresp,irq,control_fault);
    end
    always @(posedge host_clk) if(host_rst_n && !passed) begin
        if(!loss_test_started) need(board.EP.user_resetn===1'b1,"UNPLANNED_PCIE_RESET");
        if(s_axi_awvalid && s_axi_awready) need(s_axi_awid==0,"NATIVE_XDMA_AWID_PROFILE");
        if(s_axi_bvalid && s_axi_bready) begin
            need(s_axi_bid==0,"NATIVE_XDMA_BID_PROFILE");host_data_b++;last_data_b=s_axi_bresp;
        end
        if(s_axil_bvalid && s_axil_bready) begin host_control_b++;last_control_b=s_axil_bresp;end
        if(s_axil_arvalid && s_axil_arready) begin host_control_ar++;last_read_address=s_axil_araddr;end
        if(s_axil_rvalid && s_axil_rready) begin
            host_control_r++;last_control_r=s_axil_rresp;last_control_data=s_axil_rdata;
            if(!suite_started) begin
                need(host_control_r==1 && last_read_address==0 && s_axil_rresp==0 && s_axil_rdata==32'h4452544e,"ENUMERATION_ID_PROBE");
                mmio_reads++;
                $display("NTRD case=%0d CTL_READ n=%0d addr=%h data=%h resp=%h cycle=%0d",CASE_ID,mmio_reads,last_read_address,s_axil_rdata,s_axil_rresp,cycle);
            end
        end
    end
    always @(posedge host_clk) begin
        if(!host_rst_n && lifetime_reset_n && !loss_test_started)
            need(native_requests_idle,"NATIVE_REQUEST_DURING_COLD_TRANSPORT_RESET");
        host_cycle++;
        if(!passed) $display("NTRD_HOST_DATA cycle=%0d ps=%0d reset=%b av=%b ar=%b addr=%h id=%h len=%h size=%h burst=%h lock=%b wv=%b wr=%b data=%h strb=%h last=%b bv=%b br=%b bid=%h resp=%h",host_cycle,ps(),host_rst_n,s_axi_awvalid,s_axi_awready,s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock,s_axi_wvalid,s_axi_wready,s_axi_wdata,s_axi_wstrb,s_axi_wlast,s_axi_bvalid,s_axi_bready,s_axi_bid,s_axi_bresp);
        if(!passed) $display("NTRD_HOST_CTL cycle=%0d ps=%0d reset=%b aw=%h av=%h ar=%h wd=%h ws=%h wv=%h wr=%h bv=%h br=%h bs=%h ra=%h rv=%h rr=%h ov=%h ore=%h od=%h os=%h",host_cycle,ps(),host_rst_n,s_axil_awaddr,s_axil_awvalid,s_axil_awready,s_axil_wdata,s_axil_wstrb,s_axil_wvalid,s_axil_wready,s_axil_bvalid,s_axil_bready,s_axil_bresp,s_axil_araddr,s_axil_arvalid,s_axil_arready,s_axil_rvalid,s_axil_rready,s_axil_rdata,s_axil_rresp);
    end

    task automatic run_native_reset;
        need(healthy_done && !passed && guard_running && native_requests_idle,"NATIVE_RESET_PREREQUISITE");
        $display("NTRD_NATIVE_RESET_HEALTHY_END ps=%0d",ps());
        @(negedge ui_clk[0]);hold_b[0]=1;
        setup(18);fill();wait(aw_seen[18] && w_seen[18]);
        drain_control();repeat(40) @(negedge clk);
        need(!b_seen[18] && !result_valid && boundary_pending==0 && control_quiet() &&
             native_requests_idle && host_data_b==37,"NATIVE_RESET_READY");
        loss_cookie=next_cookie;loss_epoch=epoch;loss_test_started=1;
        $display("NTRD_NATIVE_RESET_EVENT name=ready ps=%0d cookie=%h epoch=%h",ps(),loss_cookie,loss_epoch);
        @(negedge board.rp_sys_clk_p);
        board.ntrpu_set_perst(1'b0);
        $display("NTRD_NATIVE_RESET_EVENT name=perst_assert ps=%0d",ps());
        fork
            begin
                // Same 500 reference-clock edges as the public cold reset.
                // This is a simulation stimulus, not a board pulse-width spec.
                repeat(500) @(posedge board.rp_sys_clk_p);
            end
            begin
                wait(board.EP.user_resetn===1'b0 && board.EP.user_lnk_up===1'b0);
                $display("NTRD_NATIVE_RESET_EVENT name=endpoint_down ps=%0d",ps());
                wait(guard_state==6);repeat(40) @(negedge clk);
                need(loss_mgmt && owner_invalid_ack && boundary_invalid && session_lost && session_invalid,"NATIVE_RESET_OWNER_ACK");
                need(host_reset_req && !host_reset_n && !transport_reset_n &&
                     !host_admit && !source_admit && !guard_running && !guard_peers_held && !guard_peers_ready,"NATIVE_RESET_WAITING_COORDINATOR");
                need(bridge_lease_held && lease_held && bridge_epoch==loss_epoch && epoch==loss_epoch && next_cookie==loss_cookie,"NATIVE_RESET_OWNER_PRESERVED");
                need(!ack_ready && !arm_ready && !result_ready && cpl_ready==0 && !done_valid && !result_valid,"NATIVE_RESET_PUBLICATION_BLOCKED");
                $display("NTRD_NATIVE_RESET_EVENT name=release_memory_b ps=%0d",ps());
                @(negedge ui_clk[0]);hold_b[0]=0;
                wait(cpl_valid[0]);repeat(80) @(negedge clk);
                need(cpl_valid==1 && cpl_ready==0 && cpl_status[1:0]==0 && cpl_tag[63:0]==loss_cookie-1,"NATIVE_RESET_CPL_QUARANTINED");
                mem[0].readback(18);
                $display("NTRD_NATIVE_RESET_EVENT name=memory_verified ps=%0d",ps());
            end
        join
        @(negedge board.rp_sys_clk_p);
        board.ntrpu_set_perst(1'b1);
        $display("NTRD_NATIVE_RESET_EVENT name=perst_release ps=%0d",ps());
        repeat(10000) begin
            @(negedge mgmt_clk);
            if(board.EP.user_resetn===1'b1 && board.EP.user_lnk_up===1'b1) break;
        end
        need(board.EP.user_resetn===1'b1 && board.EP.user_lnk_up===1'b1,"NATIVE_LINK_RETURN_TIMEOUT");
        $display("NTRD_NATIVE_RESET_EVENT name=link_return ps=%0d",ps());
        // Exercise ARM again without authorizing the missing peer coordinator.
        guard_arm=0;repeat(8) @(negedge mgmt_clk);guard_arm=1;
        repeat(100) @(negedge clk);
        need(guard_state==6 && host_reset_req && !host_reset_n && !transport_reset_n &&
             !host_admit && !source_admit && !guard_running && !guard_peers_held && !guard_peers_ready,"NATIVE_LINK_RETURN_STILL_FENCED");
        need(boundary_pending==0 && boundary_fault==0 && native_requests_idle,"NATIVE_RESET_BOUNDARY_DRAINED");
        need(bridge_lease_held && !bridge_idle && !pool_drained && banks_used==0 && pending_counts==0 &&
             dut.receiver.receiver.receiver.core.occupied[0]==1 && dut.receiver.receiver.receiver.core.occupied[1]==0 &&
             !pool_fault && !restart_required_source,"NATIVE_RESET_DESCRIPTOR_RETAINED");
        need(arms==19 && lines==17 && commands==17 && writes==17 && returns==16 && dones==16 && acks==18 &&
             aw_count==17 && w_count==17 && b_count==17 && reads==29 && bcount==37 && rejections==1 &&
             mmio_writes==304 && mmio_reads==9 && host_data_b==37 && host_control_b==304 && host_control_r==9,"NATIVE_RESET_CONSERVATION");
        fork mem[0].model.verify_periodic_complete();mem[1].model.verify_periodic_complete();join
        need(loss_window>100 && next_cookie==loss_cookie && epoch==loss_epoch && !result_valid && !done_valid && !ack_ready,"NATIVE_RESET_END");
        @(negedge clk);
        $display("NTRD_NATIVE_RESET_RETENTION_PASS ps=%0d arms=%0d commands=%0d data=%0d returns=%0d done=%0d ack=%0d aw=%0d w=%0d b=%0d reads=%0d frames=%0d cookie=%h epoch=%h banks=%h pending=%h occupied0=%0d occupied1=%0d loss_samples=%0d coordination=blocked",
            ps(),arms,commands,writes,returns,dones,acks,aw_count,w_count,b_count,reads,bcount,next_cookie,epoch,banks_used,pending_counts,
            dut.receiver.receiver.receiver.core.occupied[0],dut.receiver.receiver.receiver.core.occupied[1],loss_window);
        passed=1;
    endtask
    always @(posedge clk) if(loss_test_started && !passed) begin
        loss_window++;
        need(lifetime_reset_n && source_reset_n && memory_reset_n==3 && admit && !restart_required && !pool_fault,"MEMORY_LIFETIME_PRESERVED");
        need(next_cookie==loss_cookie && epoch==loss_epoch && bridge_lease_held,"OWNER_NEVER_ERASED");
        need(!done_valid && !result_valid && !ack_ready,"NO_LOSS_PUBLICATION");
        $display("NTRD_NATIVE_RESET_LOSS ps=%0d epoch=%h cookie=%h bh=%b banks=%h pending=%h cv=%h cr=%h ct=%h cs=%h invalid=%b ack=%b boundary=%b hrst=%b srst=%b mrst=%h lifetime=%b fault=%h bp=%h occupied0=%0d occupied1=%0d",
            ps(),epoch,next_cookie,bridge_lease_held,banks_used,pending_counts,cpl_valid,cpl_ready,cpl_tag,cpl_status,session_invalid,owner_invalid_ack,boundary_invalid,host_reset_n,transport_reset_n,memory_reset_n,lifetime_reset_n,boundary_fault,boundary_pending,dut.receiver.receiver.receiver.core.occupied[0],dut.receiver.receiver.receiver.core.occupied[1]);
    end

endmodule
