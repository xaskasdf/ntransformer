`timescale 1ns/1ps
// Real pool/credit guards/AMD CDC; memory and admission supervisor are BFMs.
module tb_ntrpu_fragment_pool_transport;
    reg clk=0; always #2.5 clk=~clk;
    reg [1:0] mem_clk=0,mem_rst_n=0;
    integer profile=0,half0=2000,half1=3650;
    bit clocks_ready=0;
    initial begin wait(clocks_ready);#0.713;forever #(half0/1000.0) mem_clk[0]=~mem_clk[0];end
    initial begin wait(clocks_ready);#1.137;forever #(half1/1000.0) mem_clk[1]=~mem_clk[1];end
    reg rst_n=0,stop=0,pool_admit=0,pool_restart_required=0;
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
    reg [1:0] m_awready=0,m_wready=0,m_bvalid=0;
    wire [73:0] m_awaddr;
    wire [15:0] m_awlen;
    wire [7:0] m_awid,m_awcache;
    wire [5:0] m_awsize;
    wire [3:0] m_awburst;
    wire [1023:0] m_wdata;
    wire [127:0] m_wstrb;
    reg [7:0] m_bid=0;
    reg [3:0] m_bresp=0;
    ntrpu_fragment_pool_transport #(.FIRST_COOKIE(64'hffffffff)) dut(.*);
    integer job=-1,cycle=0,arms=0,lines=0,commands=0,inputs=0,returns=0,dones=0,retires=0;
    integer aw_count=0,w_count=0,b_count=0,cold_count=0;
    integer aw_job[0:1],w_job[0:1],b_job[0:1],mem_cycle[0:1],aw_stall[0:1],w_stall[0:1];
    integer commanded[0:12],input_seen[0:12],aw_seen[0:12],w_seen[0:12],b_seen[0:12],cpl_seen[0:12];
    longint arm_at[0:12],line_at[0:12],command_at[0:12],input_at[0:12],aw_at[0:12],w_at[0:12],b_at[0:12],cpl_at[0:12];
    reg [1:0] hold_aw=0,hold_w=0,hold_b=0;
    reg [63:0] expected_cookie=64'hffffffff;
    reg [63:0] cookies[0:12];
    function automatic longint ps;return longint'($realtime/1ps);endfunction
    function automatic [63:0] address(input integer j);
        return j%2 ? 64'h1fffffffc0 : 64'h100000040;
    endfunction
    function automatic [511:0] pattern(input integer j);
        for(integer i=0;i<64;i++) pattern[i*8+:8]=8'(j*37+i*23+profile*11);
    endfunction
    function automatic integer response(input integer j);return j==9 ? 2:j==10 ? 3:0;endfunction
    task tick(input integer n);repeat(n) @(negedge clk);endtask
    task cold;
        begin
            @(negedge clk);rst_n=0;mem_rst_n=0;pool_admit=0;pool_restart_required=0;
            arm_valid=0;retire_valid=0;line_valid=0;done_ready=0;stop=0;hold_aw=0;hold_w=0;hold_b=0;
            expected_cookie=64'hffffffff;cold_count++;
            $display("POOL_XPORT_COLD ps=%0d n=%0d",ps(),cold_count);
            tick(100);mem_rst_n=3;tick(100);rst_n=1;tick(100);pool_admit=1;tick(5);
            if(!idle || !pool_drained || pool_fault || protocol_fault || session_invalid) $fatal(1,"COLD_NOT_READY");
        end
    endtask
    task arm_job(input integer j);
        begin
            @(negedge clk);job=j;arm_epoch=64'h100000001+j;arm_tag=64'h800000010000abcd;
            arm_destination=address(j);arm_channel=j%2;arm_valid=1;
            $display("POOL_XPORT_JOB ps=%0d job=%0d",ps(),job);
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
        if(rst_n) begin
            $display("POOL_XPORT_SRC ps=%0d cycle=%0d job=%0d stop=%b held=%b invalid=%b poolfault=%b drained=%b phase=%h banks=%h pending=%h cv=%b cr=%b ct=%h dv=%b dr=%b pv=%b pr=%b pt=%h psr=%h ov=%b ore=%b os=%h",ps(),cycle,job,stop,lease_held,session_invalid,pool_fault,pool_drained,phase,banks_used,pending_counts,dut.cmd_valid,dut.cmd_ready,dut.cmd_tag,dut.data_valid,dut.data_ready,dut.cpl_valid,dut.cpl_ready,dut.cpl_tag,dut.cpl_status,done_valid,done_ready,done_status);
            if(arm_valid && arm_ready) begin
                arms++;arm_at[job]=ps();
                $display("POOL_XPORT_ARM ps=%0d job=%0d epoch=%h tag=%h channel=%b address=%h",ps(),job,arm_epoch,arm_tag,arm_channel,arm_destination);
            end
            if(line_valid && line_ready) begin
                lines++;line_at[job]=ps();
                $display("POOL_XPORT_LINE ps=%0d job=%0d epoch=%h tag=%h address=%h data=%h",ps(),job,line_epoch,line_tag,line_address,line_data);
            end
            if(dut.cmd_valid && dut.cmd_ready) begin
                if(commanded[job] || job==2 || job==8 || dut.cmd_channel!==1'(job%2) || dut.cmd_addr!==address(job) || dut.cmd_len!=0 || dut.cmd_tag!==expected_cookie) $fatal(1,"COMMAND_CONTEXT");
                commanded[job]=1;cookies[job]=expected_cookie;expected_cookie++;commands++;command_at[job]=ps();
                $display("POOL_XPORT_CMD ps=%0d job=%0d channel=%b address=%h cookie=%h",ps(),job,dut.cmd_channel,dut.cmd_addr,dut.cmd_tag);
            end
            if(dut.data_valid && dut.data_ready) begin
                if(!commanded[job] || command_at[job]>=ps() || input_seen[job] || dut.data!==pattern(job)) $fatal(1,"INPUT_CONTEXT");
                inputs++;input_seen[job]=1;input_at[job]=ps();
                $display("POOL_XPORT_DATA ps=%0d job=%0d data=%h",ps(),job,dut.data);
            end
            for(integer ch=0;ch<2;ch++) if(dut.cpl_valid[ch] && dut.cpl_ready[ch]) begin
                if(!b_seen[job] || b_at[job]>=ps() || cpl_seen[job] || ch!=job%2 || dut.cpl_tag[ch*64+:64]!==cookies[job] || dut.cpl_status[ch*2+:2]!==2'(response(job))) $fatal(1,"CPL_CONTEXT");
                returns++;cpl_seen[job]=1;cpl_at[job]=ps();
                $display("POOL_XPORT_CPL ps=%0d job=%0d channel=%0d cookie=%h status=%h",ps(),job,ch,dut.cpl_tag[ch*64+:64],dut.cpl_status[ch*2+:2]);
            end
            if(done_valid && done_ready) begin
                if(job!=8 && (!cpl_seen[job] || cpl_at[job]>=ps() || !pool_drained)) $fatal(1,"EARLY_DONE");
                dones++;
                $display("POOL_XPORT_DONE ps=%0d job=%0d epoch=%h tag=%h status=%h raw=%h",ps(),job,done_epoch,done_tag,done_status,done_pool_status);
            end
            if(retire_valid && retire_ready) begin
                retires++;
                $display("POOL_XPORT_RETIRE ps=%0d job=%0d epoch=%h tag=%h",ps(),job,retire_epoch,retire_tag);
            end
        end
    end
    genvar g;
    generate for(g=0;g<2;g++) begin: target
        reg held_aw=0,held_w=0;
        reg [57:0] previous_aw;
        reg [576:0] previous_w;
        always @(negedge mem_clk[g]) begin
            m_awready[g]=mem_rst_n[g] && !hold_aw[g] && mem_cycle[g]%4!=0;
            m_wready[g]=mem_rst_n[g] && !hold_w[g] && mem_cycle[g]%7!=0;
            if(!mem_rst_n[g]) m_bvalid[g]=0;
            else if(!m_bvalid[g] && !hold_b[g] && aw_job[g]==job && w_job[g]==job && b_job[g]!=job && mem_cycle[g]%5==0) begin
                m_bvalid[g]=1;m_bresp[g*2+:2]=2'(response(job));
            end
        end
        always @(posedge mem_clk[g]) begin
            if(!mem_rst_n[g]) begin
                aw_job[g]=-2;w_job[g]=-2;b_job[g]=-2;mem_cycle[g]=0;held_aw=0;held_w=0;
            end else begin
                mem_cycle[g]++;
                if(held_aw && (!m_awvalid[g] || {m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]}!==previous_aw)) $fatal(1,"AW_STABILITY");
                if(held_w && (!m_wvalid[g] || {m_wlast[g],m_wstrb[g*64+:64],m_wdata[g*512+:512]}!==previous_w)) $fatal(1,"W_STABILITY");
                held_aw=m_awvalid[g] && !m_awready[g];held_w=m_wvalid[g] && !m_wready[g];
                previous_aw={m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]};
                previous_w={m_wlast[g],m_wstrb[g*64+:64],m_wdata[g*512+:512]};
                if(held_aw) aw_stall[g]++;
                if(held_w) w_stall[g]++;
                if(m_awvalid[g] && m_awready[g]) begin
                    if(job<0 || !commanded[job] || g!=job%2 || aw_seen[job] || 64'(m_awaddr[g*37+:37])!==address(job) || m_awlen[g*8+:8]!=0 || m_awid[g*4+:4]!=0 || m_awsize[g*3+:3]!=6 || m_awburst[g*2+:2]!=1 || m_awcache[g*4+:4]!=0) $fatal(1,"MEMORY_ADDRESS");
                    aw_seen[job]=1;aw_job[g]=job;aw_count++;aw_at[job]=ps();
                    $display("POOL_XPORT_AW ps=%0d job=%0d channel=%0d address=%h len=%h id=%h size=%h burst=%h cache=%h",ps(),job,g,m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]);
                end
                if(m_wvalid[g] && m_wready[g]) begin
                    if(job<0 || !input_seen[job] || input_at[job]>=ps() || g!=job%2 || w_seen[job] || m_wdata[g*512+:512]!==pattern(job) || m_wstrb[g*64+:64]!==64'hffffffffffffffff || !m_wlast[g]) $fatal(1,"MEMORY_PAYLOAD");
                    w_seen[job]=1;w_job[g]=job;w_count++;w_at[job]=ps();
                    $display("POOL_XPORT_W ps=%0d job=%0d channel=%0d data=%h strb=%h last=%b",ps(),job,g,m_wdata[g*512+:512],m_wstrb[g*64+:64],m_wlast[g]);
                end
                if(m_bvalid[g] && m_bready[g]) begin
                    if(aw_job[g]!=job || w_job[g]!=job || b_seen[job]) $fatal(1,"MEMORY_B_CONTEXT");
                    b_seen[job]=1;b_job[g]=job;b_count++;b_at[job]=ps();m_bvalid[g]<=0;
                    $display("POOL_XPORT_B ps=%0d job=%0d channel=%0d id=%h status=%h",ps(),job,g,m_bid[g*4+:4],m_bresp[g*2+:2]);
                end
            end
        end
    end endgenerate
    initial begin
        if(!$value$plusargs("PROFILE=%d",profile)) profile=0;
        if(profile==1) begin half0=4250;half1=1750;end
        clocks_ready=1;
        for(integer j=0;j<=12;j++) begin commanded[j]=0;input_seen[j]=0;aw_seen[j]=0;w_seen[j]=0;b_seen[j]=0;cpl_seen[j]=0;end
        for(integer ch=0;ch<2;ch++) begin aw_stall[ch]=0;w_stall[ch]=0;end
        $display("POOL_XPORT_BEGIN profile=%0d half0=%0d half1=%0d",profile,half0,half1);
        cold();
        for(integer j=0;j<2;j++) begin arm_job(j);offer_line(0);finish_job(0,0);end
        arm_job(2);stop=1;tick(3);stop=0;tick(3);retire_job();
        arm_job(3);offer_line(0);stop=1;finish_job(1,0);
        hold_aw=3;arm_job(4);offer_line(0);wait(m_awvalid[0]);tick(9);stop=1;tick(3);hold_aw=0;finish_job(1,0);
        hold_w=3;arm_job(5);offer_line(0);wait(m_wvalid[1]);tick(9);stop=1;tick(3);hold_w=0;finish_job(1,0);
        hold_b=3;arm_job(6);offer_line(0);wait(aw_seen[6] && w_seen[6]);tick(9);stop=1;tick(3);hold_b=0;finish_job(1,0);
        arm_job(7);offer_line(0);wait(done_valid);@(negedge clk);stop=1;finish_job(0,0);
        arm_job(8);offer_line(1);finish_job(2,0);if(!protocol_fault || aw_seen[8] || w_seen[8]) $fatal(1,"BAD_LINE_LEAK");
        cold();arm_job(9);offer_line(0);finish_job(4,2);if(!pool_fault) $fatal(1,"POOL_ERROR_NOT_STICKY");
        cold();arm_job(10);offer_line(0);finish_job(4,3);if(!pool_fault) $fatal(1,"POOL_ERROR_NOT_STICKY");
        cold();hold_b=3;arm_job(11);offer_line(0);wait(aw_seen[11] && w_seen[11]);
        @(negedge clk);pool_admit=0;tick(4);pool_admit=1;hold_b=0;tick(150);
        if(!session_invalid || !session_lost || !lease_held || done_valid || retire_ready || arm_ready || !b_seen[11] || cpl_seen[11]) $fatal(1,"LOST_SESSION_REOPENED");
        $display("POOL_XPORT_LOST ps=%0d job=%0d held=%b invalid=%b cplvalid=%b cplready=%b drained=%b",ps(),job,lease_held,session_invalid,dut.cpl_valid,dut.cpl_ready,pool_drained);
        cold();arm_job(12);offer_line(0);finish_job(0,0);
        if(arms!=13 || lines!=12 || commands!=11 || inputs!=11 || returns!=10 || dones!=11 || retires!=12 || aw_count!=11 || w_count!=11 || b_count!=11) $fatal(1,"CONSERVATION");
        if(!pool_drained || banks_used || pending_counts || lease_held || pool_fault || protocol_fault || session_invalid) $fatal(1,"FINAL_NOT_DRAINED");
        $display("POOL_XPORT_PASS profile=%0d arms=%0d lines=%0d commands=%0d data=%0d cpl=%0d done=%0d retires=%0d aw=%0d w=%0d b=%0d cold=%0d awstall0=%0d awstall1=%0d wstall0=%0d wstall1=%0d end_ps=%0d",profile,arms,lines,commands,inputs,returns,dones,retires,aw_count,w_count,b_count,cold_count,aw_stall[0],aw_stall[1],w_stall[0],w_stall[1],ps());
        $finish;
    end
    initial begin #200000;$fatal(1,"TRANSPORT_TIMEOUT job=%0d phase=%0d stop=%b",job,phase,stop);end
endmodule
