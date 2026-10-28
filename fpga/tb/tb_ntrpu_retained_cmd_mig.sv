`timescale 1ns/1ps
`ifdef XILINX_SIMULATOR
module short(in1,in1);inout in1;endmodule
`endif
// Actual shared retained sections, NTRD/owner, receiver and two MIGs.
// Host is an AXI/AXI-Lite BFM. Generic DDR4 geometry remains unqualified DIMM.
module tb_ntrpu_retained_cmd_mig #(parameter integer SOURCE_PERIOD_PS=5000);
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

    reg host_clk=0;
    reg run_host=1;
    wire host_rst_n=path_reset_n[0] && path_reset_n[2];
    wire transport_rst_n=path_reset_n[1] && path_reset_n[3];
    reg loss_test_started=0;
    reg [63:0] loss_cookie,loss_epoch;
    integer loss_window=0,host_cycle=0,forensic_reads=0;
    longint command_stop_ps=0,accepted_line_ps=0;
    integer control_src_aw=0,control_src_w=0,control_src_b=0,control_src_ar=0,control_src_r=0;
    initial begin #0.7;forever begin #3.5;if(run_host)host_clk=~host_clk;end end
    reg  close_request='0;
    reg [63:0] generation='0;
    reg [63:0] tag='0;
    reg  owner_ready_mgmt=1'b1;
    wire [6:0] proof_requests;
    wire [63:0] proof_round_generation;
    wire [63:0] proof_round_tag;
    wire  complete_ack;
    wire [63:0] complete_generation;
    wire [63:0] complete_tag;
    wire [6:0] accepted_mask;
    wire [6:0] armed_mask;
    wire [6:0] retained_faults;
    wire  revoked;
    wire  identity_fault;
    wire  proof_ack;
    wire [63:0] proof_generation;
    wire [63:0] proof_ack_tag;
    wire  busy;
    wire  peer_request;
    wire  peer_answer;
    wire [63:0] peer_generation;
    wire [63:0] peer_tag;
    wire  fault_mgmt;
    wire  revoked_source;
    wire [31:0] obsolete;
    reg  invalidate_owner_source='0;
    reg [1:0] upstream_present='0;
    reg [1:0] upstream_ready='0;
    reg [1:0] upstream_fault='0;
    reg [1:0] upstream_ack='0;
    reg [127:0] upstream_generation='0;
    reg [127:0] upstream_tag='0;
    reg [3:0] section_ready_mgmt=4'b1111;
    wire [1:0] section_ack;
    wire [1:0] section_fault;
    wire [1:0] local_ack;
    wire [3:0] local_fault;
    wire [3:0] path_reset_n;
    wire [1:0] local_idle;
    wire [1:0] request_present;
    wire [1:0] reply_pending;
    wire clk=source_clk,rst_n=lifetime_reset_n;
    reg run_source=1;reg [1:0] run_memory=3;
    always #(SOURCE_PERIOD_PS/2000.0) if(run_source) source_clk=~source_clk;
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
    ntrpu_pcie_fragment_retained_mig_r2 #(.FIRST_COOKIE(64'hffffffff)) dut (.*);
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
        idle(2+seed%3);s_axi_bready=1;@(posedge host_clk);@(negedge host_clk);s_axi_bready=0;
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
            begin if(order==1) idle(3);caw(address);end
            begin if(order==0) idle(4);cwd(value,strobe);end
        join
        cb_response(address,value,strobe,expected);
    endtask
    task automatic rd(input [31:0] address,input [31:0] expected,input [1:0] resp=0,input integer hold_cycles=3);
        @(negedge host_clk);s_axil_araddr=address;s_axil_arvalid=1;
        do @(posedge host_clk);while(!s_axil_arready);
        @(negedge host_clk);s_axil_arvalid=0;s_axil_araddr='1;
        while(!s_axil_rvalid) @(negedge host_clk);
        need(s_axil_rresp===resp && s_axil_rdata===expected,"CONTROL_READ_RESPONSE");
        idle(hold_cycles+seed%3);s_axil_rready=1;@(posedge host_clk);@(negedge host_clk);s_axil_rready=0;
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
        while(!s_axil_bvalid) @(negedge host_clk);
        need(s_axil_bresp===expected,"CONTROL_WRITE_RESPONSE");
        idle(2+seed%3);s_axil_bready=1;@(posedge host_clk);@(negedge host_clk);s_axil_bready=0;
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
        repeat(2000) begin @(negedge clk);if(result_valid) break;end
        need(result_valid && result_epoch==epoch && result_tag==arm_tag && result_status==status,"RESULT_STATUS");
        set_ack(epoch,arm_tag);idle(2);
    endtask
    task automatic retire;
        need(ack_ready,"RETIRE_READY");wr('h10,4);idle(3);
        need(!result_valid && !bridge_lease_held && bridge_idle && pool_drained && !banks_used && !pending_counts,"RETIRED_DRAIN");
    endtask
    always @(posedge clk) if(rst_n && !passed) begin
        if(arm_valid && arm_ready) arms++;
        if(line_valid && line_ready) begin lines++;if(job==18)accepted_line_ps=ps();end
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
                    if(!calibrated[g] || ui_rst[g]) $fatal(1,"MEMORY_NOT_READY");
                if((m_awvalid[g] || m_wvalid[g]) && !admit)
                    $fatal(1,"NO_WRITE_FOR_UNACCEPTED_COMMAND");
                // This AR is the independent memory fixture's final readback,
                // not a request admitted through the closed host/receiver.
                // Keep physical readiness and require exactly the retained job.
                if(arvalid[g] && !admit) begin
                    if(g!=0 || job!=18 || read_identity!=16 || !loss_test_started ||
                       !restart_required || running || rearm_ready || peers_reset_req ||
                       !source_reset_n || memory_reset_n!=3 || invalidate_owner_source ||
                       commanded[18] || input_seen[18] || aw_seen[18] || w_seen[18] || b_seen[18] ||
                       araddr[g*37+:37]!==37'(address(18)) ||
                       cpl_valid!=0 || cpl_ready!=0 || !cmd_valid || cmd_ready || cmd_tag!=loss_cookie-1 || bridge_phase!=2 || data_valid ||
                       !session_lost || !session_invalid || !lease_held || !bridge_lease_held ||
                       next_cookie!=loss_cookie || epoch!=loss_epoch || complete_ack ||
                       accepted_mask!=7'b0011001 || section_ack!=3 || result_valid || done_valid)
                        $fatal(1,"FORENSIC_READ_CONTEXT");
                    if(arready[g]) begin
                        if(forensic_reads!=0) $fatal(1,"FORENSIC_READ_DUPLICATE");
                        forensic_reads++;
                        $display("RETAINED_CMD_MIG_FORENSIC_AR ps=%0d channel=%0d identity=%0d address=%h calibrated=%h ui_rst=%h admit=%b restart=%b source_reset=%b memory_reset=%h cpl=%h ready=%h cookie=%h next=%h epoch=%h forced=%b global_ack=%b",ps(),g,read_identity,araddr[g*37+:37],calibrated,ui_rst,admit,restart_required,source_reset_n,memory_reset_n,cpl_valid,cpl_ready,cmd_tag,next_cookie,epoch,invalidate_owner_source,complete_ack);
                    end
                end
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
        if(dut.host.l_awvalid && dut.host.l_awready) control_src_aw++;
        if(dut.host.l_wvalid && dut.host.l_wready) control_src_w++;
        if(dut.host.l_bvalid && dut.host.l_bready) control_src_b++;
        if(dut.host.l_arvalid && dut.host.l_arready) control_src_ar++;
        if(dut.host.l_rvalid && dut.host.l_rready) control_src_r++;
    end
    function automatic bit control_quiet;
        return !s_axil_awvalid && !s_axil_wvalid && !s_axil_bvalid && !s_axil_arvalid && !s_axil_rvalid &&
               !dut.host.l_awvalid && !dut.host.l_wvalid && !dut.host.l_bvalid && !dut.host.l_arvalid && !dut.host.l_rvalid;
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
    initial begin
        s_axi_awsize=6;s_axi_awburst=1;
        for(integer j=0;j<19;j++) begin commanded[j]=0;input_seen[j]=0;aw_seen[j]=0;w_seen[j]=0;b_seen[j]=0;end
        for(integer ch=0;ch<2;ch++) begin aw_stall[ch]=0;w_stall[ch]=0;end
        $display("MANAGED_MIG_BEGIN jobs=18 first=00000000ffffffff");
        wait(admit);wait(&path_reset_n);idle(4);need(rst_n && pool_drained,"COLD_START");
        idle(12);
        rd(0,'h4452544e);rd(4,'h10000);rd(8,'hff);
        rd('hc4,0,3);wr('hc4,1,15,3);wr('h04,0,15,2);rd('hbc,2);rd('hc0,37);wr('h14,1);
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
            !pool_fault && !session_invalid && !restart_required_source && !control_fault && !protocol_fault && admit && one_ready>=2500000/SOURCE_PERIOD_PS,"FINAL_DRAINED");
        for(integer rank=0;rank<4;rank++) need(mem[0].model.rank_acts[rank]>0 && mem[1].model.rank_acts[rank]>0,"RANK_COVERAGE");
        @(negedge clk);
        $display("NTRD_HOST_PASS cycles=%0d ps=%0d writes=%0d reads=%0d frames=%0d",host_cycle,ps(),mmio_writes,mmio_reads,bcount);
        $display("MANAGED_MIG_PASS arms=%0d lines=%0d commands=%0d data=%0d cpl=%0d done=%0d acks=%0d aw=%0d w=%0d b=%0d reads=%0d frames=%0d rejects=%0d one_ready=%0d end_ps=%0d",arms,lines,commands,writes,returns,dones,acks,aw_count,w_count,b_count,reads,bcount,rejections,one_ready,ps());
        $display("NTRD case=0 CTL_PASS writes=%0d reads=%0d seed=%0d cycles=%0d",mmio_writes,mmio_reads,seed,cycle);
        $display("RETAINED_CMD_MIG_HEALTHY_END ps=%0d period=%0d",ps(),SOURCE_PERIOD_PS);
        @(negedge ui_clk[0]);hold_b[0]=1;
        setup(18);
        tx(frame(0,expected_data,epoch),0);
        // Capture real control responses before the final frame creates CMD.
        // No internal IP state or ready/valid wire is forced.
        fork caw('h20);cwd(32'hdeadbeef,15);join
        @(negedge host_clk);s_axil_araddr=0;s_axil_arvalid=1;
        do @(posedge host_clk);while(!s_axil_arready);
        @(negedge host_clk);s_axil_arvalid=0;s_axil_araddr='1;
        wait(s_axil_bvalid && s_axil_rvalid);idle(8);
        fork
            aw();
            w(frame(1,expected_data,epoch));
            begin
                wait(cmd_valid && cmd_ready);
                @(negedge source_clk);
                need(!commanded[18] && !input_seen[18] && cmd_valid && cmd_ready &&
                     !aw_seen[18] && !w_seen[18] && !b_seen[18] && bridge_phase==2 &&
                     cmd_channel==0 && cmd_addr==address(18) && cmd_tag==next_cookie-1 && cmd_len==0 &&
                     data===pattern(18) && banks_used==0 && pending_counts==0,
                     "OFFERED_COMMAND_NOT_TAKEN");
                loss_cookie=next_cookie;loss_epoch=epoch;loss_test_started=1;
                command_stop_ps=ps();run_source=0;
                $display("RETAINED_CMD_MIG_STOP ps=%0d line_ps=%0d cookie=%h epoch=%h cmd=%0d data=%0d aw=%0d w=%0d b=%0d banks=%h pending=%h payload=%h cv=%b cr=%b channel=%b address=%h ctag=%h len=%h",ps(),accepted_line_ps,loss_cookie,loss_epoch,commands,writes,aw_count,w_count,b_count,banks_used,pending_counts,data,cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len);
            end
        join
        repeat(256) begin @(negedge host_clk);if(s_axi_bvalid) break;end
        need(s_axi_bvalid,"CMD_PENDING_HOST_RESPONSE_TIMEOUT");
        need(s_axil_bresp==2 && s_axil_rresp==0 && s_axil_rdata==32'h4452544e && s_axi_bresp==0,"HELD_REAL_RESPONSES");
        need(!commanded[18] && !input_seen[18] && !aw_seen[18] && !w_seen[18] && !b_seen[18] && !result_valid && !complete_ack,"CMD_PENDING_BEFORE_LOSS");
        generation=epoch;tag=1;close_request=1;
        // Source clock is stopped after LINE, before the offered CMD can be accepted.
        // Wait for management admission, never for source-side CDC delivery.
        repeat(16) begin
            @(negedge mgmt_clk);
            if(proof_requests==7'h19 && proof_round_generation==epoch && proof_round_tag==1) break;
        end
        need(proof_requests==7'h19 && proof_round_generation==epoch && proof_round_tag==1 &&
             !peer_request && !proof_ack && !complete_ack && !run_source,
             "MANAGEMENT_REQUEST_WHILE_SOURCE_STOPPED");
        $display("RETAINED_CMD_MIG_MGMT_REQUEST ps=%0d req=%h gen=%h tag=%h peer=%b source=%b",ps(),proof_requests,proof_round_generation,proof_round_tag,peer_request,run_source);
        mark("CLOSE_WITH_PENDING_RESPONSES");
        // Exceed the unchanged 64-cycle clock watchdog; real admission loss
        // must invalidate ownership while preserving receiver/CDC reset rails.
        repeat(96)@(negedge mgmt_clk);
        need(session_invalid && !session_lost && !proof_ack && !complete_ack && restart_required && !admit && source_reset_n && memory_reset_n==3,"WATCHDOG_RETAINS_WITH_NO_SOURCE_EDGE");
        need(cmd_valid && !cmd_ready && !data_valid && data===pattern(18) && !commanded[18] && !input_seen[18] && commands==16 && writes==16 && banks_used==0 && pending_counts==0,"CMD_HELD_THROUGH_WATCHDOG");
        mark("SOURCE_STOPPED_BEFORE_LOSS");
        tag=2;repeat(5)@(negedge mgmt_clk);mark("REPLACED_WHILE_SOURCE_STOPPED");
        run_source=1;@(posedge source_clk);@(negedge source_clk);run_source=0;
        // Exceed the unchanged 64-cycle clock watchdog; real admission loss
        // must invalidate ownership while preserving receiver/CDC reset rails.
        repeat(96)@(negedge mgmt_clk);
        need(session_lost && !proof_ack && !complete_ack,"ONE_EDGE_NOT_PROOF");
        need(!commanded[18] && !input_seen[18] && commands==16 && writes==16 && cmd_valid && !cmd_ready && !data_valid && bridge_phase==2 && cpl_ready==0,"ONE_EDGE_RETAINS_UNACCEPTED_CMD");
        mark("LOSS_LATCHED_SOURCE_STOPPED");
        run_source=1;wait(proof_ack);repeat(8)@(negedge mgmt_clk);
        need(section_ack==0 && accepted_mask==1 && !complete_ack && obsolete==1,"OWNER_NOT_SECTION_DRAIN");
        need(s_axil_bvalid && s_axil_rvalid && s_axi_bvalid,"RESPONSES_SURVIVE_LOSS");mark("OWNER_CERTIFIED_RESPONSES_HELD");
        @(negedge host_clk);run_host=0;
        @(negedge mgmt_clk);tag=3;repeat(40)@(negedge mgmt_clk);
        wait(proof_ack && proof_ack_tag==3);
        need(section_ack==0 && !complete_ack,"HOST_STOPPED_NEW_ID");mark("OWNER_RENEWED_HOST_STOPPED");
        run_host=1;
        response(0);
        cb_response('h20,32'hdeadbeef,15,2);
        need(s_axil_rvalid && s_axil_rresp==0 && s_axil_rdata==32'h4452544e,"HELD_READ_UNCHANGED");
        idle(4);s_axil_rready=1;@(posedge host_clk);@(negedge host_clk);s_axil_rready=0;
        mmio_reads++;
        $display("NTRD case=0 CTL_READ n=%0d addr=00000000 data=4452544e resp=0 cycle=%0d",mmio_reads,cycle);
        wait(section_ack==3);repeat(8)@(negedge mgmt_clk);
        need(accepted_mask==7'b0011001 && !complete_ack && proof_requests[6:5]==0 && local_ack==0 && local_idle==3,"MISSING_UPSTREAM_CONTRACTS");
        mark("SHARED_SECTIONS_DRAINED");
        @(negedge ui_clk[0]);hold_b[0]=0;
        repeat(80)@(negedge source_clk);
        need(cmd_valid && !cmd_ready && cmd_tag==loss_cookie-1 && cmd_addr==address(18) &&
             cmd_channel==0 && cmd_len==0 && !data_valid && data===pattern(18) && cpl_valid==0,
             "UNACCEPTED_COMMAND_RETAINED");
        // Job16 wrote this exact address before loss. The offered job18 must
        // never replace its bytes. Read directly through the memory fixture.
        need(address(16)==address(18) && pattern(16)!==pattern(18),"PRIOR_DATA_REFERENCE");
        mem[0].readback(16);mark("MEMORY_READBACK_WITH_CMD_HELD");
        wr('h10,1,15,2);rd(0,0,2);tx(frame(0,expected_data,epoch),2);
        need(arms==19 && lines==17 && commands==16 && writes==16 && returns==16 && dones==16 && acks==18,"NO_OWNER_REUSE");
        need(forensic_reads==1,"EXACTLY_ONE_FORENSIC_READ");
        need(aw_count==16 && w_count==16 && b_count==16 && reads==29 && bcount==38 && rejections==2,"ENDPOINT_CONSERVATION");
        need(restart_required && !admit && !running && !rearm_ready && invalidate_owner_source==0,"REAL_WATCHDOG_LOSS_STICKY");
        need(section_ack==3 && !section_fault && !local_fault && !retained_faults && !identity_fault,"HEALTHY_SECTION_CLOSURE");
        need(lease_held && bridge_lease_held && !bridge_idle && !pool_drained && !banks_used && !pending_counts && cpl_valid==0 && cmd_valid && !cmd_ready && bridge_phase==2,"CMD_REMAINS_OBLIGATION");
        fork mem[0].model.verify_periodic_complete();mem[1].model.verify_periodic_complete();join
        mark("TERMINAL");
        $display("RETAINED_CMD_MIG_PASS period=%0d arms=%0d lines=%0d commands=%0d data=%0d cpl=%0d done=%0d acks=%0d aw=%0d w=%0d b=%0d reads=%0d frames=%0d rejects=%0d mmio_writes=%0d mmio_reads=%0d accepted=%h complete=%b",SOURCE_PERIOD_PS,arms,lines,commands,writes,returns,dones,acks,aw_count,w_count,b_count,reads,bcount,rejections,mmio_writes,mmio_reads,accepted_mask,complete_ack);
        passed=1;$finish;
    end
    initial begin #400000;$fatal(1,"MANAGED_MIG_TIMEOUT job=%0d phase=%0d",job,bridge_phase);end
    always @(posedge clk) begin
        cycle=cycle+1;
        if(!passed) $display("NTRD_HOST_RESET cycle=%0d ps=%0d host=%b transport=%b lifetime=%b",cycle,ps(),host_rst_n,transport_rst_n,lifetime_reset_n);
        if(!passed) $display("NTRD case=%0d LIFE_TRACE cycle=%0d ps=%0d cold=%b poolreset=%b memreset=%h admit=%b restart=%b synced=%b rearm_ready=%b calibrated=%h ui_rst=%h source_run=%b memory_run=%h banks=%h pending=%h external=%b rearm=%b",CASE_ID,cycle,ps(),rst_n,source_reset_n,memory_reset_n,admit,restart_required,restart_required_source,rearm_ready,calibrated,ui_rst,run_source,run_memory,banks_used,pending_counts,external_reset_async,rearm);
        if(!passed) $display("NTRD case=%0d OWN_TRACE cycle=%0d case=%0d rst=%h stop=%h stopped=%h isolated=%h qe=%h admit=%h restart=%h pf=%h drained=%h av=%h ar=%h car=%h bar=%h ae=%h at=%h aw=%h aa=%h ac=%h kv=%h kr=%h ke=%h kt=%h rr=%h awv=%h awr=%h awa=%h awi=%h awl=%h aws=%h awb=%h awk=%h wv=%h wr=%h wd=%h ws=%h wl=%h bv=%h br=%h bi=%h bs=%h lv=%h lr=%h le=%h lt=%h la=%h ld=%h cv=%h cr=%h cc=%h ca=%h ct=%h cl=%h dv=%h dr=%h data=%h pv=%h pr=%h pt=%h ps=%h ov=%h ore=%h oe=%h ot=%h os=%h ops=%h err=%h rv=%h rc=%h re=%h rt=%h rs=%h epoch=%h held=%h be=%h bh=%h idle=%h phase=%h fail=%h error=%h bytes=%h ti=%h fault=%h lost=%h invalid=%h exhausted=%h next=%h rejected=%h frames=%h identity=%h code=%h faults=%h",CASE_ID,cycle,job,
            rst_n,stop,source_stopped,ingress_isolated,quiesce_epoch,pool_admit,pool_restart_required,pool_fault,pool_drained,arm_valid,arm_ready,collector_arm_ready,bridge_arm_ready,arm_epoch,arm_tag,arm_window,arm_destination,arm_channel,ack_valid,ack_ready,ack_epoch,ack_tag,retire_ready,dut.host.a_awvalid,dut.host.a_awready,dut.host.a_awaddr,dut.host.a_awid,dut.host.a_awlen,dut.host.a_awsize,dut.host.a_awburst,dut.host.a_awlock,dut.host.a_wvalid,dut.host.a_wready,dut.host.a_wdata,dut.host.a_wstrb,dut.host.a_wlast,dut.host.a_bvalid,dut.host.a_bready,dut.host.a_bid,dut.host.a_bresp,line_valid,line_ready,line_epoch,line_tag,line_address,line_data,cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len,data_valid,data_ready,data,cpl_valid,cpl_ready,cpl_tag,cpl_status,done_valid,done_ready,done_epoch,done_tag,done_status,done_pool_status,done_error,result_valid,result_ready,result_epoch,result_tag,result_status,epoch,lease_held,bridge_epoch,bridge_lease_held,bridge_idle,bridge_phase,failed,error_code,valid_bytes,transport_idle,protocol_fault,session_lost,session_invalid,exhausted,next_cookie,rejected_completions,rejected_frames,rejected_identity,last_reject_code,last_reject_faults);
        if(!passed) $display("NTRD case=%0d CTL_TRACE cycle=%0d aw=%h av=%h ar=%h wd=%h ws=%h wv=%h wr=%h bv=%h br=%h bs=%h ra=%h rv=%h rr=%h ov=%h ore=%h od=%h os=%h irq=%h fault=%h",CASE_ID,cycle,dut.host.l_awaddr,dut.host.l_awvalid,dut.host.l_awready,dut.host.l_wdata,dut.host.l_wstrb,dut.host.l_wvalid,dut.host.l_wready,dut.host.l_bvalid,dut.host.l_bready,dut.host.l_bresp,dut.host.l_araddr,dut.host.l_arvalid,dut.host.l_arready,dut.host.l_rvalid,dut.host.l_rready,dut.host.l_rdata,dut.host.l_rresp,irq,control_fault);
    end
    task mark(input string name);
        #0.001;
        $display("RETAINED_CMD_MIG_EVENT name=%s ps=%0d period=%0d gen=%h tag=%h proof=%b accepted=%h sections=%h local=%h fault=%h complete=%b epoch=%h cookie=%h held=%b bridge=%b lost=%b invalid=%b bv=%b rv=%b fb=%b cv=%h cr=%h reset=%h source=%b host=%b admit=%b restart=%b running=%b source_alive=%b receiver_reset=%b memreset=%h peer_reset=%b rearm_ready=%b forced_invalid=%b",name,ps(),SOURCE_PERIOD_PS,generation,tag,proof_ack,accepted_mask,section_ack,local_ack,retained_faults,complete_ack,epoch,next_cookie,lease_held,bridge_lease_held,session_lost,session_invalid,s_axil_bvalid,s_axil_rvalid,s_axi_bvalid,cpl_valid,cpl_ready,path_reset_n,run_source,run_host,admit,restart_required,running,source_alive,source_reset_n,memory_reset_n,peers_reset_req,rearm_ready,invalidate_owner_source);
    endtask
    always @(posedge source_clk)if(loss_test_started && !passed)begin
        loss_window++;
        need(lifetime_reset_n && source_reset_n && memory_reset_n==3 && !pool_fault,"MEMORY_LIFETIME_PRESERVED");
        need(next_cookie==loss_cookie && epoch==loss_epoch && bridge_lease_held,"OWNER_NEVER_ERASED");
        need(!done_valid && !result_valid && !ack_ready,"NO_LOSS_PUBLICATION");
    end
    always @(posedge mgmt_clk)if(lifetime_reset_n && !passed)begin
        if(loss_test_started) $display("RETAINED_CMD_MIG_MGMT ps=%0d req=%h gen=%h tag=%h ack=%b pg=%h pt=%h accepted=%h section=%h local=%h fault=%h complete=%b peer_req=%b peer_answer=%b obsolete=%0d source=%b host=%b admit=%b restart=%b alive=%b receiver_reset=%b memreset=%h peer_reset=%b forced_invalid=%b",ps(),proof_requests,proof_round_generation,proof_round_tag,proof_ack,proof_generation,proof_ack_tag,accepted_mask,section_ack,local_ack,retained_faults,complete_ack,peer_request,peer_answer,obsolete,run_source,run_host,admit,restart_required,source_alive,source_reset_n,memory_reset_n,peers_reset_req,invalidate_owner_source);
        if(loss_test_started) $display("RETAINED_CMD_MIG_PENDING ps=%0d cv=%b cr=%b channel=%b address=%h tag=%h len=%h dv=%b data=%h banks=%h pending=%h cmd=%0d writes=%0d aw=%0d w=%0d b=%0d phase=%0d",ps(),cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len,data_valid,data,banks_used,pending_counts,commands,writes,aw_count,w_count,b_count,bridge_phase);
        if(loss_test_started && restart_required) need(!admit && !running && !rearm_ready && !peers_reset_req && source_reset_n && memory_reset_n==3,"RUNTIME_FAULT_RETAINS_RAILS");
        need(!complete_ack && upstream_present==0 && upstream_ack==0,"NO_FAKE_GLOBAL_DRAIN");
        need(!section_fault && !local_fault && !retained_faults && !identity_fault,"NO_SECTION_FAULT");
    end
    always @(posedge host_clk) begin
        host_cycle++;
        if(!passed) $display("NTRD_HOST_DATA cycle=%0d ps=%0d reset=%b av=%b ar=%b addr=%h id=%h len=%h size=%h burst=%h lock=%b wv=%b wr=%b data=%h strb=%h last=%b bv=%b br=%b bid=%h resp=%h",host_cycle,ps(),host_rst_n,s_axi_awvalid,s_axi_awready,s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock,s_axi_wvalid,s_axi_wready,s_axi_wdata,s_axi_wstrb,s_axi_wlast,s_axi_bvalid,s_axi_bready,s_axi_bid,s_axi_bresp);
        if(!passed) $display("NTRD_HOST_CTL cycle=%0d ps=%0d reset=%b aw=%h av=%h ar=%h wd=%h ws=%h wv=%h wr=%h bv=%h br=%h bs=%h ra=%h rv=%h rr=%h ov=%h ore=%h od=%h os=%h",host_cycle,ps(),host_rst_n,s_axil_awaddr,s_axil_awvalid,s_axil_awready,s_axil_wdata,s_axil_wstrb,s_axil_wvalid,s_axil_wready,s_axil_bvalid,s_axil_bready,s_axil_bresp,s_axil_araddr,s_axil_arvalid,s_axil_arready,s_axil_rvalid,s_axil_rready,s_axil_rdata,s_axil_rresp);
    end

endmodule
