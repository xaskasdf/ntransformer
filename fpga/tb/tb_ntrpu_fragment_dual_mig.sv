`timescale 1ns/1ps
// Real pool/credit guards/AMD CDC; memory and admission supervisor are BFMs.
`ifdef XILINX_SIMULATOR
module short(in1,in1);inout in1;endmodule
`endif
module tb_ntrpu_fragment_dual_mig;
    reg clk=0; always #2.5 clk=~clk;
    wire source_clk=clk;
    reg mgmt_clk=0;always #5 mgmt_clk=~mgmt_clk;
    wire [1:0] mem_clk,mem_rst_n,calibrated,ui_rst,memory_alive,interrupt_ecc;
    wire rst_n,pool_rst_n,admit,running,rearm_ready,restart_required,restart_required_source,peers_reset_req,source_alive;
    reg stop=0,external_reset_async=1,rearm=0;
    initial begin #40;external_reset_async=0;end
    initial begin wait(rearm_ready);@(negedge mgmt_clk);rearm=1;repeat(4)@(negedge mgmt_clk);rearm=0;end
    integer profile=0;
    reg arm_valid=0,arm_channel=0,retire_valid=0,line_valid=0,done_ready=0;
    reg [63:0] arm_epoch=0,arm_tag=0,arm_destination=0,retire_epoch=0,retire_tag=0;
    reg [63:0] line_epoch=0,line_tag=0,line_address=0;
    reg [511:0] line_data=0;
    wire arm_ready,retire_ready,line_ready,done_valid,lease_held;
    wire [63:0] epoch,done_epoch,done_tag,next_cookie;
    wire [3:0] done_status;
    wire [1:0] done_pool_status;
    wire protocol_fault,session_lost,exhausted,session_invalid,idle,pool_fault,pool_drained;
    wire [2:0] phase;
    wire [31:0] rejected_completions;
    wire [7:0] banks_used;
    wire [9:0] pending_counts;
    wire [1:0] m_awvalid,m_wvalid,m_wlast,m_bready;
    wire [1:0] m_awready,m_wready,m_bvalid;
    wire [73:0] m_awaddr;
    wire [15:0] m_awlen;
    wire [7:0] m_awid,m_awcache;
    wire [5:0] m_awsize;
    wire [3:0] m_awburst;
    wire [1023:0] m_wdata;
    wire [127:0] m_wstrb;
    wire [7:0] m_bid;
    wire [3:0] m_bresp;
    reg [73:0] araddr=0;
    reg [1:0] arvalid=0,rready=0;
    wire [1:0] arready,m_rvalid,rlast;
    wire [7:0] rid;
    wire [3:0] rresp;
    wire [1023:0] rdata;
    ntrpu_fragment_dual_mig #(.FIRST_COOKIE(64'hffffffff)) dut (
        .bridge_reset_n(rst_n),.source_reset_n(pool_rst_n),.ui_clk(mem_clk),.memory_reset_n(mem_rst_n),.*
    );
    integer job=-1,cycle=0,arms=0,lines=0,commands=0,inputs=0,returns=0,dones=0,retires=0;
    integer aw_count=0,w_count=0,b_count=0,reads=0,one_ready=0;
    integer aw_stall[0:1],w_stall[0:1];
    integer commanded[0:16],input_seen[0:16],aw_seen[0:16],w_seen[0:16],b_seen[0:16],cpl_seen[0:16];
    longint arm_at[0:16],line_at[0:16],command_at[0:16],input_at[0:16],aw_at[0:16],w_at[0:16],b_at[0:16],cpl_at[0:16];
    reg [1:0] hold_w=0,hold_b=0;
    reg [63:0] expected_cookie=64'hffffffff;
    reg [63:0] cookies[0:16];
    function automatic longint ps;return longint'($realtime/1ps);endfunction
    function automatic [63:0] address(input integer j);
        if(j==12) return 64'h1fffffffc0;
        if(j==13) return 64'h100001040;
        if(j==14) return 64'h100001080;
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
    function automatic [511:0] pattern(input integer j);
        for(integer i=0;i<64;i++) pattern[i*8+:8]=8'(j*37+i*23+profile*11);
    endfunction
    function automatic integer response(input integer j);return 0;endfunction
    task tick(input integer n);repeat(n) @(negedge clk);endtask
    task arm_job(input integer j);
        begin
            @(negedge clk);job=j;arm_epoch=64'h100000001+j;arm_tag=64'h800000010000abcd;
            arm_destination=address(j);arm_channel=j%2;arm_valid=1;
            $display("FRAG_DUAL_JOB ps=%0d job=%0d",ps(),job);
            do @(posedge clk);while(!arm_ready);
            @(negedge clk);arm_valid=0;
        end
    endtask
    task offer_line(input bit bad);
        begin
            line_valid=1;line_epoch=arm_epoch^(bad ? 64'h100000000:0);line_tag=arm_tag;
            line_address=arm_destination;line_data=pattern(job);
            do @(posedge clk);while(!line_ready);
            @(negedge clk);line_valid=0;
        end
    endtask
    task retire_job;
        begin
            retire_epoch=epoch;retire_tag=arm_tag^64'h100000000;retire_valid=1;
            tick(3);if(retire_ready) $fatal(1,"FORGED_RETIRE");
            retire_tag=arm_tag;
            do @(posedge clk);while(!retire_ready);
            @(negedge clk);retire_valid=0;stop=0;tick(3);
            if(!idle || !pool_drained || banks_used || pending_counts) $fatal(1,"LEAKED_POOL_OWNERSHIP");
        end
    endtask
    task finish_job(input integer expected_status,raw_status);
        begin
            wait(done_valid);@(negedge clk);
            repeat(5) begin
                if(!done_valid || done_epoch!==arm_epoch || done_tag!==arm_tag ||
                   done_status!==4'(expected_status) || done_pool_status!==2'(raw_status)) $fatal(1,"DONE_CONTEXT");
                tick(1);
            end
            done_ready=1;tick(1);done_ready=0;tick(2);retire_job();
        end
    endtask
    always @(posedge clk) begin
        cycle++;
        if(calibrated==1 && !ui_rst[0]) begin
            one_ready++;if(arm_ready || dut.cmd_ready || admit) $fatal(1,"ONE_READY_EARLY");
        end
        if(rst_n) begin
            $display("FRAG_DUAL_SRC ps=%0d cycle=%0d job=%0d stop=%b held=%b invalid=%b poolfault=%b drained=%b phase=%h banks=%h pending=%h cv=%b cr=%b ct=%h dv=%b dr=%b pv=%b pr=%b pt=%h psr=%h ov=%b ore=%b os=%h",ps(),cycle,job,stop,lease_held,session_invalid,pool_fault,pool_drained,phase,banks_used,pending_counts,dut.cmd_valid,dut.cmd_ready,dut.cmd_tag,dut.data_valid,dut.data_ready,dut.cpl_valid,dut.cpl_ready,dut.cpl_tag,dut.cpl_status,done_valid,done_ready,done_status);
            if(arm_valid && arm_ready) begin
                arms++;arm_at[job]=ps();
                $display("FRAG_DUAL_ARM ps=%0d job=%0d epoch=%h tag=%h channel=%b address=%h",ps(),job,arm_epoch,arm_tag,arm_channel,arm_destination);
            end
            if(line_valid && line_ready) begin
                lines++;line_at[job]=ps();
                $display("FRAG_DUAL_LINE ps=%0d job=%0d epoch=%h tag=%h address=%h data=%h",ps(),job,line_epoch,line_tag,line_address,line_data);
            end
            if(dut.cmd_valid && dut.cmd_ready) begin
                if(commanded[job] || job==12 || job==16 || dut.cmd_channel!==1'(job%2) || dut.cmd_addr!==address(job) || dut.cmd_len!=0 || dut.cmd_tag!==expected_cookie) $fatal(1,"COMMAND_CONTEXT");
                commanded[job]=1;cookies[job]=expected_cookie;expected_cookie++;commands++;command_at[job]=ps();
                $display("FRAG_DUAL_CMD ps=%0d job=%0d channel=%b address=%h cookie=%h",ps(),job,dut.cmd_channel,dut.cmd_addr,dut.cmd_tag);
            end
            if(dut.data_valid && dut.data_ready) begin
                if(!commanded[job] || command_at[job]>=ps() || input_seen[job] || dut.data!==pattern(job)) $fatal(1,"INPUT_CONTEXT");
                inputs++;input_seen[job]=1;input_at[job]=ps();
                $display("FRAG_DUAL_DATA ps=%0d job=%0d data=%h",ps(),job,dut.data);
            end
            for(integer ch=0;ch<2;ch++) if(dut.cpl_valid[ch] && dut.cpl_ready[ch]) begin
                if(!b_seen[job] || b_at[job]>=ps() || cpl_seen[job] || ch!=job%2 || dut.cpl_tag[ch*64+:64]!==cookies[job] || dut.cpl_status[ch*2+:2]!==2'(response(job))) $fatal(1,"CPL_CONTEXT");
                returns++;cpl_seen[job]=1;cpl_at[job]=ps();
                $display("FRAG_DUAL_CPL ps=%0d job=%0d channel=%0d cookie=%h status=%h",ps(),job,ch,dut.cpl_tag[ch*64+:64],dut.cpl_status[ch*2+:2]);
            end
            if(done_valid && done_ready) begin
                if(job!=16 && (!cpl_seen[job] || cpl_at[job]>=ps() || !pool_drained)) $fatal(1,"EARLY_DONE");
                dones++;
                $display("FRAG_DUAL_DONE ps=%0d job=%0d epoch=%h tag=%h status=%h raw=%h",ps(),job,done_epoch,done_tag,done_status,done_pool_status);
            end
            if(retire_valid && retire_ready) begin
                retires++;
                $display("FRAG_DUAL_RETIRE ps=%0d job=%0d epoch=%h tag=%h",ps(),job,retire_epoch,retire_tag);
            end
        end
    end
    for(genvar g=0;g<2;g++) begin: mem
        ntrpu_dual_mig_fixture #(.PORT(g)) model (
            .clk(mem_clk[g]),.calibrated(calibrated[g]),.ui_rst(ui_rst[g]),.interrupt_ecc(interrupt_ecc[g]),
            .rst_n(mem_rst_n[g]),.hold_w(hold_w[g]),.hold_b(hold_b[g]),
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
        always @(posedge mem_clk[g]) begin
            if(calibrated[g] && !ui_rst[g]) begin
                if(ui_edges>0 && ps()-last_ui!=5000) $fatal(1,"UI_PERIOD");
                last_ui=ps();ui_edges++;
                if(ui_edges==4) $display("FRAG_DUAL_CLOCK ps=%0d channel=%0d period=5000 phase=%0d",ps(),g,ps()%5000);
            end
            if(mem_rst_n[g]) begin
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
                    aw_seen[job]=1;aw_count++;aw_at[job]=ps();
                    $display("FRAG_DUAL_AW ps=%0d job=%0d channel=%0d address=%h len=%h id=%h size=%h burst=%h cache=%h",ps(),job,g,m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]);
                end
                if(m_wvalid[g] && m_wready[g]) begin
                    if(job<0 || !input_seen[job] || input_at[job]>=ps() || g!=job%2 || w_seen[job] || m_wdata[g*512+:512]!==pattern(job) || m_wstrb[g*64+:64]!==64'hffffffffffffffff || !m_wlast[g]) $fatal(1,"MEMORY_PAYLOAD");
                    w_seen[job]=1;w_count++;w_at[job]=ps();
                    $display("FRAG_DUAL_W ps=%0d job=%0d channel=%0d data=%h strb=%h last=%b",ps(),job,g,m_wdata[g*512+:512],m_wstrb[g*64+:64],m_wlast[g]);
                end
                if(m_bvalid[g] && m_bready[g]) begin
                    if(!aw_seen[job] || !w_seen[job] || b_seen[job] || m_bid[g*4+:4]!=0 || m_bresp[g*2+:2]!=0) $fatal(1,"MEMORY_B_CONTEXT");
                    b_seen[job]=1;b_count++;b_at[job]=ps();
                    $display("FRAG_DUAL_B ps=%0d job=%0d channel=%0d id=%h status=%h",ps(),job,g,m_bid[g*4+:4],m_bresp[g*2+:2]);
                end
                if(arvalid[g] && arready[g])
                    $display("FRAG_DUAL_AR ps=%0d channel=%0d index=%0d identity=%0d address=%h",ps(),g,read_index,read_identity,araddr[g*37+:37]);
                if(m_rvalid[g] && rready[g]) begin
                    reads++;
                    $display("FRAG_DUAL_R ps=%0d channel=%0d index=%0d identity=%0d data=%h status=%h id=%h last=%b",ps(),g,read_index,read_identity,rdata[g*512+:512],rresp[g*2+:2],rid[g*4+:4],rlast[g]);
                end
            end
        end
        task automatic readback(input integer identity);
            read_identity=identity;read_index++;
            @(negedge mem_clk[g]);araddr[g*37+:37]=37'(address(identity));arvalid[g]=1;
            do @(posedge mem_clk[g]);while(!arready[g]);
            @(negedge mem_clk[g]);arvalid[g]=0;
            while(!m_rvalid[g]) @(negedge mem_clk[g]);
            repeat(3) @(negedge mem_clk[g]);
            if(rid[g*4+:4]!=0 || rresp[g*2+:2]!=0 || !rlast[g] || rdata[g*512+:512]!==pattern(identity)) $fatal(1,"READBACK_BYTES");
            rready[g]=1;@(posedge mem_clk[g]);@(negedge mem_clk[g]);rready[g]=0;
        endtask
    end
    initial begin
        for(integer j=0;j<=16;j++) begin commanded[j]=0;input_seen[j]=0;aw_seen[j]=0;w_seen[j]=0;b_seen[j]=0;cpl_seen[j]=0;end
        for(integer ch=0;ch<2;ch++) begin aw_stall[ch]=0;w_stall[ch]=0;end
        $display("FRAG_DUAL_BEGIN first=00000000ffffffff jobs=17");
        #100;
        for(integer j=0;j<12;j++) begin arm_job(j);offer_line(0);finish_job(0,0);end
        fork
            begin for(integer j=0;j<12;j+=2) mem[0].readback(j);end
            begin for(integer j=1;j<12;j+=2) mem[1].readback(j);end
        join
        arm_job(12);stop=1;tick(3);stop=0;tick(3);retire_job();
        fork mem[0].readback(10);mem[1].readback(11);join
        arm_job(13);offer_line(0);stop=1;finish_job(1,0);
        @(negedge mem_clk[0]);hold_w[0]=1;
        arm_job(14);offer_line(0);wait(m_wvalid[0]);tick(12);stop=1;
        @(negedge mem_clk[0]);hold_w[0]=0;finish_job(1,0);
        @(negedge mem_clk[1]);hold_b[1]=1;
        arm_job(15);offer_line(0);wait(aw_seen[15] && w_seen[15]);tick(12);stop=1;
        @(negedge mem_clk[1]);hold_b[1]=0;finish_job(1,0);
        arm_job(16);offer_line(1);finish_job(2,0);
        if(!protocol_fault || aw_seen[16] || w_seen[16]) $fatal(1,"BAD_LINE_LEAK");
        fork
            begin for(integer j=0;j<12;j+=2) mem[0].readback(j);mem[0].readback(14);end
            begin for(integer j=1;j<11;j+=2) mem[1].readback(j);mem[1].readback(15);mem[1].readback(13);end
        join
        if(arms!=17 || lines!=16 || commands!=15 || inputs!=15 || returns!=15 || dones!=16 || retires!=17 || aw_count!=15 || w_count!=15 || b_count!=15 || reads!=28) $fatal(1,"CONSERVATION");
        if(!pool_drained || banks_used || pending_counts || lease_held || pool_fault || session_invalid || restart_required_source || !admit || one_ready<500) $fatal(1,"FINAL_NOT_DRAINED");
        for(integer rank=0;rank<4;rank++) if(mem[0].model.rank_acts[rank]==0 || mem[1].model.rank_acts[rank]==0) $fatal(1,"RANK_COVERAGE");
        fork mem[0].model.verify_periodic_complete();mem[1].model.verify_periodic_complete();join
        $display("FRAG_DUAL_PASS arms=%0d lines=%0d commands=%0d data=%0d cpl=%0d done=%0d retires=%0d aw=%0d w=%0d b=%0d reads=%0d one_ready=%0d end_ps=%0d",arms,lines,commands,inputs,returns,dones,retires,aw_count,w_count,b_count,reads,one_ready,ps());
        $finish;
    end
    initial begin #150000;$fatal(1,"FRAG_DUAL_TIMEOUT job=%0d phase=%0d",job,phase);end
endmodule
