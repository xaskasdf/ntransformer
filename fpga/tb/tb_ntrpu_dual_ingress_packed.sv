`timescale 1ns/1ps
module tb_ntrpu_dual_ingress_packed;
    parameter integer SERIAL_INPUT=0;
    reg clk=0,rst_n=0,stop=0;
    always #2.5 clk=~clk;
    reg [1:0] cmd_valid=0,data_valid=0;
    wire [1:0] cmd_ready,data_ready,reserve_can;
    reg [127:0] cmd_addr=0,cmd_tag=0;
    reg [15:0] cmd_len=0;
    reg [1023:0] data=0;
    reg fence_valid=0,fence_done_ready=0;
    wire fence_ready,fence_done_valid;
    reg [63:0] fence_id=0;
    wire [63:0] fence_done_id;
    wire [1:0] m_awvalid,m_wvalid,m_wlast,m_bready,cpl_valid;
    reg [1:0] m_awready=0,m_wready=0,m_bvalid=0,cpl_ready=0;
    wire [127:0] m_awaddr,m_wstrb,cpl_tag;
    wire [15:0] m_awlen;
    wire [7:0] m_awid,m_awcache;
    wire [5:0] m_awsize;
    wire [3:0] m_awburst,cpl_status;
    wire [1023:0] m_wdata;
    reg [7:0] m_bid=0;
    reg [3:0] m_bresp=0;
    wire fault,drained;
    wire [7:0] banks_used;
    wire [9:0] pending_counts;
    ntrpu_dual_ingress_packed #(.SERIAL_INPUT(SERIAL_INPUT)) dut(.*);

    integer scenario=0,cycle=0,simultaneous=0,case_dual=0,bad_resp_tag=-1;
    integer ncmd[0:1],naw[0:1],nw[0:1],nb[0:1],ncpl[0:1];
    integer tags[0:2047],lengths[0:1023],inputs[0:1023],outputs[0:1023],due[0:2047];
    integer peak=0;
    reg [1:0] block_w=0,block_aw=0,block_b=0,block_cpl=0,block_input=0;
    bit jitter=0;
    reg [1:0] awheld=0,wheld=0,bheld=0,cplheld=0;
    reg [1:0] b_accepted=0;
    reg [169:0] saved_aw=0;
    reg [1153:0] saved_w=0;
    reg [11:0] saved_b=0;
    reg [131:0] saved_cpl=0;
    function automatic longint stamp();return longint'($realtime*1000);endfunction
    function automatic [511:0] pattern(input integer tag,input integer beat);
        for(integer j=0;j<16;j++)pattern[j*32+:32]=(tag*32'h9e3779b9)^(beat*32'h10203)^(j*32'h1010101);
    endfunction
    task automatic need(input bit ok,input string label);
        if(!ok)$fatal(1,"DUAL_FAIL case=%0d cycle=%0d %s",scenario,cycle,label);
    endtask
    task automatic step;@(negedge clk);#0.2;endtask
    always @(negedge clk)begin
        for(integer lane=0;lane<2;lane++)begin
            m_awready[lane]=rst_n&&!block_aw[lane]&&(!jitter||cycle%7!=lane);
            m_wready[lane]=rst_n&&!block_w[lane]&&(!jitter||cycle%9>=2+lane);
            cpl_ready[lane]=rst_n&&!block_cpl[lane]&&(!jitter||cycle%11!=lane);
            if(!rst_n)m_bvalid[lane]=0;
            else if(!m_bvalid[lane] || b_accepted[lane])begin
                m_bvalid[lane]=!block_b[lane]&&nb[lane]<naw[lane]&&nb[lane]<nw[lane]&&cycle>=due[lane*1024+nb[lane]];
                m_bid[lane*4+:4]=0;
                m_bresp[lane*2+:2]=tags[lane*1024+nb[lane]]==bad_resp_tag?2:0;
            end
        end
    end
    always @(posedge clk)begin
        cycle++;
        b_accepted=rst_n ? m_bvalid&m_bready : 0;
        if(!rst_n)begin
            peak=0;case_dual=0;awheld=0;wheld=0;bheld=0;cplheld=0;
            for(integer lane=0;lane<2;lane++)begin ncmd[lane]=0;naw[lane]=0;nw[lane]=0;nb[lane]=0;ncpl[lane]=0;end
            for(integer t=0;t<1024;t++)begin inputs[t]=0;outputs[t]=0;lengths[t]=0;end
        end else begin
            if(scenario!=7 && scenario!=8)need(!fault,"UNEXPECTED_FAULT");
            need(!(data_valid==3&&data_ready==3&&dut.fill_bank[0]==dut.fill_bank[1]),"EXCLUSIVE_WRITERS");
            need(!(dut.read_issue==3&&dut.bank[dut.send_head[0]]==dut.bank[16+dut.send_head[1]]),"EXCLUSIVE_READERS");
            if(|dut.write_request || |dut.read_request)
                $display("CP PORT %0d %0d %0d %0d %0d %0d %0d %0d %0d",stamp(),scenario,
                    dut.write_request,dut.fill_group,dut.write_grant,dut.read_request,dut.read_group,dut.read_grant,banks_used);
            if($countones(banks_used)>peak)peak=$countones(banks_used);
            if((data_valid&data_ready)==3)begin simultaneous++;case_dual++;need(!SERIAL_INPUT,"SERIAL_CONTROL");end
            for(integer lane=0;lane<2;lane++)begin
                need(!awheld[lane]||(m_awvalid[lane]&&{m_awaddr[lane*64+:64],m_awlen[lane*8+:8],m_awid[lane*4+:4],m_awcache[lane*4+:4],m_awsize[lane*3+:3],m_awburst[lane*2+:2]}===saved_aw[lane*85+:85]),"AW_STABLE");
                need(!wheld[lane]||(m_wvalid[lane]&&{m_wdata[lane*512+:512],m_wstrb[lane*64+:64],m_wlast[lane]}===saved_w[lane*577+:577]),"W_STABLE");
                need(!bheld[lane]||(m_bvalid[lane]&&{m_bid[lane*4+:4],m_bresp[lane*2+:2]}===saved_b[lane*6+:6]),"B_STABLE");
                need(!cplheld[lane]||(cpl_valid[lane]&&{cpl_tag[lane*64+:64],cpl_status[lane*2+:2]}===saved_cpl[lane*66+:66]),"CPL_STABLE");
                awheld[lane]=m_awvalid[lane]&&!m_awready[lane];saved_aw[lane*85+:85]={m_awaddr[lane*64+:64],m_awlen[lane*8+:8],m_awid[lane*4+:4],m_awcache[lane*4+:4],m_awsize[lane*3+:3],m_awburst[lane*2+:2]};
                wheld[lane]=m_wvalid[lane]&&!m_wready[lane];saved_w[lane*577+:577]={m_wdata[lane*512+:512],m_wstrb[lane*64+:64],m_wlast[lane]};
                bheld[lane]=m_bvalid[lane]&&!m_bready[lane];saved_b[lane*6+:6]={m_bid[lane*4+:4],m_bresp[lane*2+:2]};
                cplheld[lane]=cpl_valid[lane]&&!cpl_ready[lane];saved_cpl[lane*66+:66]={cpl_tag[lane*64+:64],cpl_status[lane*2+:2]};
                if(cmd_valid[lane]&&cmd_ready[lane])begin
                    integer t;t=int'(cmd_tag[lane*64+:64]);need(t<1024&&lengths[t]==0,"UNIQUE_TAG");
                    tags[lane*1024+ncmd[lane]]=t;ncmd[lane]++;lengths[t]=int'(cmd_len[lane*8+:8])+1;
                    $display("DI CMD %0d %0d %0d %0d %0d %0d",stamp(),lane,t,lengths[t],dut.alloc_bank,dut.alloc_generation);
                end
                if(data_valid[lane]&&data_ready[lane])begin
                    integer t;t=int'(dut.tag[lane*16+dut.fill_slot[lane]]);
                    need(data[lane*512+:512]===pattern(t,inputs[t])&&inputs[t]<lengths[t],"INPUT_DATA");
                    $display("DI IN %0d %0d %0d %0d %h",stamp(),lane,t,inputs[t],data[lane*512+:512]);inputs[t]++;
                end
                if(m_awvalid[lane]&&m_awready[lane])begin
                    integer t;t=tags[lane*1024+naw[lane]];need(naw[lane]<ncmd[lane],"AW_CAUSAL");
                    need(m_awaddr[lane*64+:64]==64'(t)*4096&&int'(m_awlen[lane*8+:8])+1==lengths[t],"AW_METADATA");
                    need(m_awsize[lane*3+:3]==6&&m_awburst[lane*2+:2]==1&&m_awid[lane*4+:4]==0&&m_awcache[lane*4+:4]==0,"AXI_ATTRIBUTES");
                    $display("DI AW %0d %0d %0d",stamp(),lane,t);naw[lane]++;
                end
                if(m_wvalid[lane]&&m_wready[lane])begin
                    integer t;t=tags[lane*1024+nw[lane]];need(nw[lane]<naw[lane],"W_AFTER_AW");
                    need(outputs[t]<inputs[t]&&m_wdata[lane*512+:512]===pattern(t,outputs[t]),"OUTPUT_DATA");
                    need(m_wstrb[lane*64+:64]==64'hffffffffffffffff&&m_wlast[lane]==(outputs[t]+1==lengths[t]),"WLAST_STROBE");
                    $display("DI W %0d %0d %0d %0d %h",stamp(),lane,t,outputs[t],m_wdata[lane*512+:512]);outputs[t]++;
                    if(m_wlast[lane])begin due[lane*1024+nw[lane]]=cycle+3+(jitter?t%13:0);nw[lane]++;end
                end
                if(m_bvalid[lane]&&m_bready[lane])begin
                    integer t;t=tags[lane*1024+nb[lane]];need(nb[lane]<nw[lane],"B_AFTER_W");
                    $display("DI B %0d %0d %0d %0d",stamp(),lane,t,m_bresp[lane*2+:2]);nb[lane]++;
                end
                if(cpl_valid[lane]&&cpl_ready[lane])begin
                    integer t;t=tags[lane*1024+ncpl[lane]];need(ncpl[lane]<nb[lane]&&cpl_tag[lane*64+:64]==t,"CPL_ORDER");
                    need(cpl_status[lane*2+:2]==(t==bad_resp_tag?2:0),"CPL_STATUS");
                    $display("DI CPL %0d %0d %0d %0d",stamp(),lane,t,cpl_status[lane*2+:2]);ncpl[lane]++;
                end
                if(dut.allocator.core.return_valid[lane])
                    $display("DI RETURN %0d %0d %0d %0d",stamp(),lane,dut.allocator.core.return_bank[lane*3+:3],dut.allocator.core.return_generation[lane*64+:64]);
            end
            if(drained)need(banks_used==0&&pending_counts==0&&ncmd[0]==ncpl[0]&&ncmd[1]==ncpl[1],"DRAIN_ALL");
            if(fence_valid&&fence_ready)$display("DI FENCE %0d %0d",stamp(),fence_id);
            if(fence_done_valid&&fence_done_ready)$display("DI FENCED %0d %0d",stamp(),fence_done_id);
        end
    end
    task automatic reset_case(input integer number);
        step;rst_n=0;stop=0;cmd_valid=0;data_valid=0;fence_valid=0;fence_done_ready=0;
        block_w=0;block_aw=0;block_b=0;block_cpl=0;block_input=0;jitter=0;bad_resp_tag=-1;
        repeat(6)step;scenario=number;rst_n=1;$display("DI CASE %0d %0d",stamp(),number);step;
    endtask
    task automatic frame(input integer lane,input integer tag,input integer beats);
        step;cmd_valid[lane]=1;cmd_addr[lane*64+:64]=64'(tag)*4096;cmd_tag[lane*64+:64]=tag;cmd_len[lane*8+:8]=beats-1;
        do @(posedge clk);while(!cmd_ready[lane]);
        step;cmd_valid[lane]=0;
        for(integer beat=0;beat<beats;beat++)begin
            while(block_input[lane])step;
            if(jitter&&beat%7==3)step;
            data_valid[lane]=1;data[lane*512+:512]=pattern(tag,beat);
            do @(posedge clk);while(!data_ready[lane]);
            step;data_valid[lane]=0;
        end
    endtask
    task automatic drain_all;
        for(integer attempt=0;attempt<5000;attempt++)begin step;if(drained)begin $display("DI DRAIN %0d %0d",stamp(),scenario);return;end end
        need(0,"DRAIN_TIMEOUT");
    endtask
    task automatic passed;
        $display("DI PASS %0d %0d %0d %0d %0d %0d",scenario,ncmd[0]+ncmd[1],ncpl[0]+ncpl[1],peak,case_dual,fault);
    endtask
    task automatic pressure(input integer first);
        block_w=3;
        fork
            for(integer j=0;j<8;j++)frame(0,first+2*j,64);
            for(integer j=0;j<8;j++)frame(1,first+2*j+1,64);
            begin
                wait(banks_used==255&&dut.fill_active==0);repeat(100)step;
                need(banks_used==255&&!drained,"ALL_EIGHT_RETAINED");
                $display("DI FULL %0d %0d",stamp(),first);
                block_w=2;wait(ncpl[0]>ncpl[1]);repeat(30)step;
                $display("DI OPEN0 %0d %0d %0d",stamp(),ncpl[0],ncpl[1]);block_w=0;
            end
        join
        drain_all;
    endtask
    task automatic loan_pressure(input integer borrower,input integer first);
        integer other;other=1-borrower;block_w=3;
        // Borrower takes both dedicated banks and the first shared reserve.
        for(integer j=0;j<3;j++)frame(borrower,first+2*j+borrower,64);
        frame(other,first+other,64);
        // For borrower 1 the second shared bank belongs to its own group;
        // advance to bank 6 before overlapping the other channel's payload.
        if(borrower==1)frame(borrower,first+6+borrower,64);
        fork
            frame(borrower,first+(borrower==0?6:8)+borrower,64);
            frame(other,first+2+other,64);
        join
        if(borrower==0)frame(borrower,first+8+borrower,64);
        frame(borrower,first+10+borrower,64);
        repeat(30)step;need(banks_used==255,"LOAN_ALL_EIGHT");
        $display("CP LOAN_FULL %0d %0d %0d",stamp(),scenario,borrower);
        // Delay the other output until the borrower reaches shared banks.
        block_w[borrower]=0;wait(ncpl[borrower]==3);step;block_w=0;
        drain_all;
        need(ncpl[borrower]==6&&ncpl[other]==2,"LOAN_DRAIN");
    endtask
    initial begin
        integer begin_cycle,end_cycle;
        reset_case(1);begin_cycle=cycle;
        fork
            for(integer j=0;j<8;j++)frame(0,100+2*j,64);
            for(integer j=0;j<8;j++)frame(1,101+2*j,64);
        join
        end_cycle=cycle;drain_all;
        need(SERIAL_INPUT?case_dual==0:case_dual>400,"PARALLEL_FILL");
        $display("DI RATE %0d %0d %0d %0d",SERIAL_INPUT,1024, end_cycle-begin_cycle,cycle-begin_cycle);passed;
        reset_case(2);pressure(200);pressure(220);passed;
        reset_case(3);block_input=2;
        fork
            for(integer j=0;j<8;j++)frame(0,300+2*j,9);
            frame(1,301,64);
            begin wait(ncpl[0]==8);need(ncpl[1]==0,"INPUT_LANE_INDEPENDENCE");$display("DI INDEPENDENT %0d",stamp());block_input=0;end
        join
        drain_all;passed;
        reset_case(4);block_cpl=3;
        fork frame(0,400,8);frame(1,401,8);join
        step;fence_valid=1;fence_id=77;do @(posedge clk);while(!fence_ready);step;fence_valid=0;
        wait(cpl_valid==3);repeat(20)step;need(!fence_done_valid&&!drained,"FENCE_AFTER_CPL");
        $display("DI FENCE_BLOCKED %0d",stamp());block_cpl=0;
        wait(fence_done_valid);repeat(10)step;need(fence_done_id==77&&!drained,"FENCE_HELD");
        fence_done_ready=1;step;fence_done_ready=0;drain_all;
        fork frame(0,402,1);frame(1,403,1);join
        drain_all;passed;
        reset_case(5);
        fork
            frame(0,500,64);frame(1,501,64);
            begin wait(inputs[500]>=4&&inputs[501]>=4);step;stop=1;step;stop=0;$display("DI STOP %0d",stamp());end
        join
        drain_all;need(dut.stopped&&cmd_ready==0,"STOP_LATCHED");passed;
        reset_case(6);jitter=1;
        fork
            for(integer j=0;j<12;j++)frame(0,600+2*j,j%3==0?1:j%3==1?9:64);
            for(integer j=0;j<12;j++)frame(1,601+2*j,j%3==0?1:j%3==1?9:64);
        join
        drain_all;passed;
        reset_case(7);bad_resp_tag=700;
        fork frame(0,700,8);frame(1,701,8);join
        drain_all;need(fault&&cmd_ready==0,"RESPONSE_ERROR_CLOSES");passed;
        reset_case(8);cmd_valid=1;cmd_addr=1;cmd_tag=800;cmd_len=7;
        repeat(8)step;need(fault&&cmd_ready==0&&banks_used==0&&ncmd[0]==0,"INVALID_COMMAND_REJECTED");cmd_valid=0;passed;
        reset_case(9);loan_pressure(0,900);passed;
        reset_case(10);loan_pressure(1,1000);passed;
        $display("DI ALL_PASS mode=%0d cases=10 simultaneous=%0d",SERIAL_INPUT,simultaneous);$finish;
    end
    initial begin #500000;$fatal(1,"DUAL_WATCHDOG");end
endmodule
