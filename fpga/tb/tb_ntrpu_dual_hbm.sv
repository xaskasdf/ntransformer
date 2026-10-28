`timescale 1ns/1ps
module tb_ntrpu_dual_hbm;
    parameter integer PREFER_LOCAL=1,NATIVE=0;
    localparam integer FRAME_BEATS=NATIVE?8:64;
    reg clk=0,mem_clk=0,mem_pause=0,cold_reset_n=0,stop=0,fault_in=0;
    realtime mem_half=1.111;
    always #2.5 clk=~clk;
    initial begin #0.731;forever begin #(mem_half);if(!mem_pause)mem_clk=~mem_clk;end end
    wire rst_n,mem_rst_n,started,fault,drained;
    wire [11:0] inflight;
    reg [1:0] cmd_valid=0,data_valid=0,cpl_ready=0;
    wire [1:0] cmd_ready,data_ready,reserve_can,cpl_valid;
    reg [127:0] cmd_addr=0,cmd_tag=0;
    reg [15:0] cmd_len=0;
    reg [1023:0] data=0;
    wire [127:0] cpl_tag;
    wire [3:0] cpl_status;
    reg fence_valid=0,fence_done_ready=0;
    reg [63:0] fence_id=0;
    wire fence_ready,fence_done_valid;
    wire [63:0] fence_done_id;
    wire [7:0] banks_used;
    wire [9:0] pending_counts;
    reg [1:0] init_mask=3;
    wire [1:0] init_complete_async={apb_complete_1,apb_complete_0}&init_mask;
    wire [1:0] h_awvalid,h_awready,h_wvalid,h_wready,h_wlast,h_bvalid,h_bready;
    wire [67:0] h_awaddr;
    wire [7:0] h_awlen;
    wire [11:0] h_awid,h_bid;
    wire [5:0] h_awsize;
    wire [3:0] h_awburst,h_bresp;
    wire [511:0] h_wdata;
    wire [63:0] h_wstrb;
    wire [1:0] adapter_idle,adapter_fault;
    ntrpu_dual_hbm_async #(.PREFER_LOCAL(PREFER_LOCAL)) dut(.*);

    // Generated only from the frozen native binding used in the earlier review.
    `include "ntrpu_dual_hbm_binding.svh"

    integer scenario=0,cycle=0,peak=0,dual=0,early_returns=0,epoch=0;
    integer cmds[0:1],aws[0:1],ws[0:1],bs[0:1],cpls[0:1];
    integer tags[0:2047],lengths[0:4095],ins[0:4095],outs[0:4095];
    integer native_aws[0:1],native_ws[0:1],native_bs[0:1];
    integer nat_tags[0:4095],nat_beats[0:4095],nat_offsets[0:4095];
    integer expected_tag[0:1],expected_offset[0:1],native_index[0:1];
    integer native_read_beats=0,total_cmds=0,total_bytes=0,expected_error=-1;
    reg [1:0] hold_cpl=0;
    reg allow_fault=0;
    reg [1:0] awheld=0,wheld=0,cplheld=0;
    reg [169:0] saved_aw;
    reg [1153:0] saved_w;
    reg [131:0] saved_cpl;
    reg fence_held=0;
    reg [63:0] saved_fence;
    reg [1:0] nah=0,nwh=0,nbh=0,nrh=0;
    reg [97:0] nas;
    reg [577:0] nws;
    reg [15:0] nbs;
    reg [529:0] nrs;
    function automatic longint stamp();return longint'($realtime*1000);endfunction
    function automatic [63:0] destination(input integer tag);
        return (tag%2?64'h200000000:0)+64'h10000+64'(tag/2)*4096;
    endfunction
    function automatic [511:0] pattern(input integer tag,input integer beat);
        for(integer j=0;j<16;j++)pattern[j*32+:32]=(tag*32'h9e3779b9)^(beat*32'h10203)^(j*32'h1010101);
    endfunction
    task automatic need(input bit ok,input string why);
        if(!ok)$fatal(1,"DH_FAIL case=%0d epoch=%0d cycle=%0d %s",scenario,epoch,cycle,why);
    endtask
    task automatic step;@(negedge clk);#0.2;endtask
    always @(negedge clk)cpl_ready=rst_n?~hold_cpl:0;
    always @(posedge clk)begin
        cycle++;
        if(!rst_n)begin
            peak=0;dual=0;early_returns=0;awheld=0;wheld=0;cplheld=0;fence_held=0;
            for(integer c=0;c<2;c++)begin cmds[c]=0;aws[c]=0;ws[c]=0;bs[c]=0;cpls[c]=0;end
            for(integer t=0;t<4096;t++)begin lengths[t]=0;ins[t]=0;outs[t]=0;end
        end else begin
            if(!allow_fault)need(!fault,"UNEXPECTED_FAULT");
            need(!fence_held || (fence_done_valid&&fence_done_id===saved_fence),"FENCE_STABLE");
            fence_held=fence_done_valid&&!fence_done_ready;saved_fence=fence_done_id;
            if(!started)need(cmd_ready==0&&reserve_can==0&&!fence_ready&&!drained,"STARTUP_GATING");
            if($countones(banks_used)>peak)peak=$countones(banks_used);
            if((data_valid&data_ready)==3)dual++;
            for(integer c=0;c<2;c++)begin
                need(inflight[c*6+:6]==aws[c]-bs[c],"CDC_LEDGER");
                need(!awheld[c]||(dut.t_awvalid[c]&&{dut.t_awaddr[c*64+:64],dut.t_awlen[c*8+:8],dut.t_awid[c*4+:4],dut.t_awcache[c*4+:4],dut.t_awsize[c*3+:3],dut.t_awburst[c*2+:2]}===saved_aw[c*85+:85]),"SOURCE_AW_STABLE");
                need(!wheld[c]||(dut.t_wvalid[c]&&{dut.t_wdata[c*512+:512],dut.t_wstrb[c*64+:64],dut.t_wlast[c]}===saved_w[c*577+:577]),"SOURCE_W_STABLE");
                need(!cplheld[c]||(cpl_valid[c]&&{cpl_tag[c*64+:64],cpl_status[c*2+:2]}===saved_cpl[c*66+:66]),"CPL_STABLE");
                awheld[c]=dut.t_awvalid[c]&&!dut.t_awready[c];saved_aw[c*85+:85]={dut.t_awaddr[c*64+:64],dut.t_awlen[c*8+:8],dut.t_awid[c*4+:4],dut.t_awcache[c*4+:4],dut.t_awsize[c*3+:3],dut.t_awburst[c*2+:2]};
                wheld[c]=dut.t_wvalid[c]&&!dut.t_wready[c];saved_w[c*577+:577]={dut.t_wdata[c*512+:512],dut.t_wstrb[c*64+:64],dut.t_wlast[c]};
                cplheld[c]=cpl_valid[c]&&!cpl_ready[c];saved_cpl[c*66+:66]={cpl_tag[c*64+:64],cpl_status[c*2+:2]};
                if(cmd_valid[c]&&cmd_ready[c])begin
                    integer t;t=int'(cmd_tag[c*64+:64]);need(t<4096&&lengths[t]==0,"UNIQUE_COMMAND");
                    tags[c*1024+cmds[c]]=t;cmds[c]++;lengths[t]=int'(cmd_len[c*8+:8])+1;
                    need(cmd_addr[c*64+:64]==destination(t),"COMMAND_ADDRESS");
                    total_cmds++;total_bytes+=lengths[t]*64;
                    $display("DH CMD %0d %0d %0d %0d %0d %h %0d %0d",stamp(),epoch,c,t,lengths[t],cmd_addr[c*64+:64],dut.receiver.alloc_bank,dut.receiver.alloc_generation);
                end
                if(data_valid[c]&&data_ready[c])begin
                    integer t;t=int'(dut.receiver.tag[c*16+dut.receiver.fill_slot[c]]);
                    need(data[c*512+:512]===pattern(t,ins[t])&&ins[t]<lengths[t],"INPUT_DATA");
                    $display("DH IN %0d %0d %0d %0d %0d %h",stamp(),epoch,c,t,ins[t],data[c*512+:512]);ins[t]++;
                end
                if(dut.t_awvalid[c]&&dut.t_awready[c])begin
                    integer t;t=tags[c*1024+aws[c]];need(aws[c]<cmds[c],"SOURCE_AW_CAUSAL");
                    need(dut.t_awaddr[c*64+:64]==destination(t)&&int'(dut.t_awlen[c*8+:8])+1==lengths[t],"SOURCE_AW_METADATA");
                    $display("DH AW %0d %0d %0d %0d",stamp(),epoch,c,t);aws[c]++;
                end
                if(dut.t_wvalid[c]&&dut.t_wready[c])begin
                    integer t;t=tags[c*1024+ws[c]];need(ws[c]<aws[c],"SOURCE_W_CAUSAL");
                    need(outs[t]<ins[t]&&dut.t_wdata[c*512+:512]===pattern(t,outs[t]),"SOURCE_DATA");
                    need(dut.t_wstrb[c*64+:64]==64'hffffffffffffffff&&dut.t_wlast[c]==(outs[t]+1==lengths[t]),"SOURCE_LAST");
                    $display("DH W %0d %0d %0d %0d %0d %h",stamp(),epoch,c,t,outs[t],dut.t_wdata[c*512+:512]);
                    outs[t]++;if(dut.t_wlast[c])ws[c]++;
                end
                if(dut.t_bvalid[c]&&dut.t_bready[c])begin
                    integer t;t=tags[c*1024+bs[c]];need(bs[c]<ws[c],"SOURCE_B_CAUSAL");
                    need(dut.t_bresp[c*2+:2]==(t==expected_error?2:0),"SOURCE_B_STATUS");
                    $display("DH B %0d %0d %0d %0d %0d",stamp(),epoch,c,t,dut.t_bresp[c*2+:2]);bs[c]++;
                end
                if(cpl_valid[c]&&cpl_ready[c])begin
                    integer t;t=tags[c*1024+cpls[c]];need(cpls[c]<bs[c]&&cpl_tag[c*64+:64]==t,"CPL_ORDER");
                    need(cpl_status[c*2+:2]==(t==expected_error?2:0),"CPL_STATUS");
                    $display("DH CPL %0d %0d %0d %0d %0d",stamp(),epoch,c,t,cpl_status[c*2+:2]);cpls[c]++;
                end
                if(dut.receiver.allocator.core.return_valid[c])begin
                    if(native_bs[c]<native_aws[c])early_returns++;
                    $display("DH RETURN %0d %0d %0d %0d %0d",stamp(),epoch,c,dut.receiver.allocator.core.return_bank[c*3+:3],dut.receiver.allocator.core.return_generation[c*64+:64]);
                end
            end
            if(drained)need(banks_used==0&&pending_counts==0&&inflight==0&&cmds[0]==cpls[0]&&cmds[1]==cpls[1],"DRAIN_COMPLETE");
            if(fence_valid&&fence_ready)$display("DH FENCE %0d %0d %0d",stamp(),epoch,fence_id);
            if(fence_done_valid)need(cmds[0]==cpls[0]&&cmds[1]==cpls[1]&&inflight==0,"FENCE_COMPLETE");
            if(fence_done_valid&&fence_done_ready)$display("DH FENCED %0d %0d %0d",stamp(),epoch,fence_done_id);
        end
    end
    always @(posedge mem_clk)begin
        if(!mem_rst_n)begin
            nah=0;nwh=0;nbh=0;nrh=0;
            for(integer c=0;c<2;c++)begin native_aws[c]=0;native_ws[c]=0;native_bs[c]=0;expected_tag[c]=0;expected_offset[c]=0;native_index[c]=0;end
        end else for(integer c=0;c<2;c++)begin
            need(!nah[c]||(h_awvalid[c]&&{h_awaddr[c*34+:34],h_awlen[c*4+:4],h_awid[c*6+:6],h_awsize[c*3+:3],h_awburst[c*2+:2]}===nas[c*49+:49]),"NATIVE_AW_STABLE");
            need(!nwh[c]||(h_wvalid[c]&&{h_wdata[c*256+:256],h_wstrb[c*32+:32],h_wlast[c]}===nws[c*289+:289]),"NATIVE_W_STABLE");
            need(!nbh[c]||(h_bvalid[c]&&{h_bid[c*6+:6],h_bresp[c*2+:2]}===nbs[c*8+:8]),"NATIVE_B_STABLE");
            need(!nrh[c]||(s_rvalid[c]&&{s_rdata[c*256+:256],s_rid[c*6+:6],s_rresp[c*2+:2],s_rlast[c]}===nrs[c*265+:265]),"NATIVE_R_STABLE");
            nah[c]=h_awvalid[c]&&!h_awready[c];nas[c*49+:49]={h_awaddr[c*34+:34],h_awlen[c*4+:4],h_awid[c*6+:6],h_awsize[c*3+:3],h_awburst[c*2+:2]};
            nwh[c]=h_wvalid[c]&&!h_wready[c];nws[c*289+:289]={h_wdata[c*256+:256],h_wstrb[c*32+:32],h_wlast[c]};
            nbh[c]=h_bvalid[c]&&!h_bready[c];nbs[c*8+:8]={h_bid[c*6+:6],h_bresp[c*2+:2]};
            nrh[c]=s_rvalid[c]&&!s_rready[c];nrs[c*265+:265]={s_rdata[c*256+:256],s_rid[c*6+:6],s_rresp[c*2+:2],s_rlast[c]};
            if(h_awvalid[c]&&h_awready[c])begin
                integer t,n,off,q;t=tags[c*1024+expected_tag[c]];n=int'(h_awlen[c*4+:4])+1;off=expected_offset[c];q=c*2048+native_aws[c];
                // The converters may choose any legal contiguous split, not just maximal bursts.
                need(expected_tag[c]<aws[c]&&n>=1&&n<=16&&off+n<=lengths[t]*2,"NATIVE_SPLIT");
                need(h_awaddr[c*34+:34]==destination(t)+off*32&&h_awid[c*6+:6]==0&&h_awsize[c*3+:3]==5&&h_awburst[c*2+:2]==1,"NATIVE_METADATA");
                nat_tags[q]=t;nat_beats[q]=n;nat_offsets[q]=off;native_aws[c]++;
                $display("DH NAW %0d %0d %0d %0d %0d %0d %h",stamp(),epoch,c,t,off,n,h_awaddr[c*34+:34]);
                expected_offset[c]+=n;if(expected_offset[c]==lengths[t]*2)begin expected_tag[c]++;expected_offset[c]=0;end
            end
            if(h_wvalid[c]&&h_wready[c])begin
                integer t,k,q;reg [511:0] want;q=c*2048+native_ws[c];t=nat_tags[q];k=nat_offsets[q]+native_index[c];want=pattern(t,k/2);
                need(native_ws[c]<native_aws[c],"NATIVE_W_CAUSAL");
                need(h_wdata[c*256+:256]==want[(k%2)*256+:256]&&h_wstrb[c*32+:32]==32'hffffffff,"NATIVE_DATA");
                need(h_wlast[c]==(native_index[c]+1==nat_beats[q]),"NATIVE_LAST");
                $display("DH NW %0d %0d %0d %0d %0d %h",stamp(),epoch,c,t,k,h_wdata[c*256+:256]);
                native_index[c]++;if(h_wlast[c])begin native_ws[c]++;native_index[c]=0;end
            end
            if(h_bvalid[c]&&h_bready[c])begin
                integer q;q=c*2048+native_bs[c];need(native_bs[c]<native_ws[c],"NATIVE_B_CAUSAL");
                $display("DH NB %0d %0d %0d %0d %0d %0d",stamp(),epoch,c,nat_tags[q],nat_offsets[q],h_bresp[c*2+:2]);native_bs[c]++;
            end
        end
    end
    task automatic frame(input integer c,input integer t,input integer beats);
        step;cmd_valid[c]=1;cmd_addr[c*64+:64]=destination(t);cmd_tag[c*64+:64]=t;cmd_len[c*8+:8]=beats-1;
        do @(posedge clk);while(!cmd_ready[c]);step;cmd_valid[c]=0;
        for(integer k=0;k<beats;k++)begin
            data_valid[c]=1;data[c*512+:512]=pattern(t,k);
            do @(posedge clk);while(!data_ready[c]);step;data_valid[c]=0;
        end
    endtask
    task automatic drain_all;
        for(integer i=0;i<6000;i++)begin step;if(drained)begin
            need(native_aws[0]==native_bs[0]&&native_aws[1]==native_bs[1],"NATIVE_DRAIN");
            $display("DH DRAIN %0d %0d %0d",stamp(),epoch,scenario);return;end end
        need(0,"DRAIN_TIMEOUT");
    endtask
    task automatic readback(input integer c,input integer t,input integer beats);
        for(integer off=0;off<beats*2;off+=16)begin
            integer n,k;reg [511:0] want;n=beats*2-off>16?16:beats*2-off;k=0;
            @(negedge mem_clk);s_araddr[c*34+:34]=destination(t)+off*32;s_arlen[c*4+:4]=n-1;
            s_arid[c*6+:6]=t%64;s_arsize[c*3+:3]=5;s_arburst[c*2+:2]=1;s_arvalid[c]=1;
            do @(posedge mem_clk);while(!s_arready[c]);@(negedge mem_clk);s_arvalid[c]=0;
            while(k<n)begin
                s_rready[c]=k%3!=1;
                // Every third beat includes a genuine stalled-valid sample.
                if(!s_rready[c])begin
                    wait(s_rvalid[c]);@(posedge mem_clk);need(s_rvalid[c],"READ_STALL");@(negedge mem_clk);s_rready[c]=1;
                end
                do @(posedge mem_clk);while(!s_rvalid[c]);want=pattern(t,(off+k)/2);
                need(s_rdata[c*256+:256]===want[((off+k)%2)*256+:256]&&s_rid[c*6+:6]==t%64&&s_rresp[c*2+:2]==0&&s_rlast[c]==(k+1==n),"READBACK_DATA");
                $display("DH R %0d %0d %0d %0d %0d %h",stamp(),epoch,c,t,off+k,s_rdata[c*256+:256]);native_read_beats++;k++;
                @(negedge mem_clk);
            end
            s_rready[c]=0;
        end
    endtask
    task automatic all_readback;
        fork
            for(integer j=0;j<cmds[0];j++)readback(0,tags[j],lengths[tags[j]]);
            for(integer j=0;j<cmds[1];j++)readback(1,tags[1024+j],lengths[tags[1024+j]]);
        join
    endtask
    task automatic begin_case(input integer number);
        scenario=number;$display("DH CASE %0d %0d %0d",stamp(),epoch,scenario);
    endtask
    task automatic passed;
        $display("DH PASS %0d %0d %0d %0d %0d %0d %0d %0d",scenario,epoch,cmds[0]+cmds[1],cpls[0]+cpls[1],peak,dual,early_returns,fault);
    endtask
    task automatic cold_boot;
        step;cold_reset_n=0;apb_rst_n=0;mem_pause=0;stop=0;fault_in=0;init_mask=3;allow_fault=0;
        cmd_valid=0;data_valid=0;fence_valid=0;fence_done_ready=0;hold_cpl=0;write_open=3;reply_open=3;bad_bid=0;bad_bresp=0;expected_error=-1;
        repeat(NATIVE?940:12)step;epoch++;apb_rst_n=1;cold_reset_n=1;
        wait(started);step;$display("DH BOOT %0d %0d",stamp(),epoch);
    endtask
    task automatic pressure(input integer first);
        step;write_open=0;
        fork
            for(integer j=0;j<12;j++)frame(0,first+2*j,FRAME_BEATS);
            for(integer j=0;j<12;j++)frame(1,first+2*j+1,FRAME_BEATS);
            begin
                wait(banks_used==255&&dut.receiver.fill_active==0);repeat(20)step;
                need(!drained&&inflight!=0,"RETAINED_CDC_WORK");$display("DH FULL %0d %0d %0d",stamp(),epoch,scenario);
                @(negedge mem_clk);write_open=1;wait(cpls[0]>cpls[1]);repeat(10)step;
                @(negedge mem_clk);write_open=3;
            end
        join
        drain_all;
    endtask
    initial begin
        $display("DH CONFIG policy=%0d native=%0d frame_beats=%0d",PREFER_LOCAL,NATIVE,FRAME_BEATS);
        begin_case(1);cold_boot;
        fork
            begin frame(0,0,1);frame(0,2,8);frame(0,4,9);frame(0,6,64);end
            begin frame(1,1,1);frame(1,3,8);frame(1,5,9);frame(1,7,64);end
        join
        drain_all;all_readback;passed;
        begin_case(2);pressure(100);pressure(200);all_readback;passed;
        begin_case(3);step;hold_cpl=3;reply_open=0;
        fork frame(0,300,9);frame(1,301,9);join
        step;fence_id=64'hface;fence_valid=1;
        do @(posedge clk);while(!fence_ready);step;fence_valid=0;
        repeat(30)step;need(!drained&&!fence_done_valid&&inflight!=0,"FENCE_WAIT_B");
        @(negedge mem_clk);reply_open=3;
        wait(cpl_valid==3);repeat(10)step;need(!drained&&!fence_done_valid,"FENCE_WAIT_CPL");
        hold_cpl=0;wait(fence_done_valid);repeat(10)step;
        need(fence_done_valid&&fence_done_id==64'hface&&cmd_ready==0,"FENCE_HOLD");
        fence_done_ready=1;step;fence_done_ready=0;drain_all;readback(0,300,9);readback(1,301,9);passed;
        if(NATIVE)begin
            need(peak==8&&dual>0&&early_returns>0,"COVERAGE_NATIVE");
            $display("DH ALL_PASS cases=3 commands=%0d bytes=%0d reads=%0d",total_cmds,total_bytes,native_read_beats);$finish;
        end
        begin_case(4);step;mem_pause=1;
        fork
            frame(0,400,64);frame(1,401,64);
            begin repeat(120)step;need(!drained,"PAUSED_CLOCK_PENDING");mem_pause=0;mem_half=3.317;end
        join
        drain_all;readback(0,400,64);readback(1,401,64);mem_half=1.111;passed;
        begin_case(5);
        fork
            frame(0,500,64);frame(1,501,64);
            begin wait(dut.receiver.fill_active==3);repeat(5)step;stop=1;end
        join
        drain_all;step;stop=0;repeat(5)step;
        need(cmd_ready==0&&reserve_can==0&&!fence_ready,"STOP_RETAINED");readback(0,500,64);readback(1,501,64);passed;
        cold_boot;begin_case(6);allow_fault=1;
        fork
            frame(0,600,64);frame(1,601,64);
            begin wait(dut.receiver.fill_active==3);repeat(5)step;fault_in=1;step;fault_in=0;end
        join
        drain_all;need(fault&&cmd_ready==0,"EXTERNAL_FAULT_LATCHED");readback(0,600,64);readback(1,601,64);passed;
        cold_boot;begin_case(7);allow_fault=1;
        step;cmd_valid=1;cmd_addr=128'h400000000;cmd_tag=700;cmd_len=0;
        repeat(5)step;need(fault&&cmd_ready==0&&banks_used==0&&cmds[0]==0,"ADDRESS_REJECT");cmd_valid=0;passed;
        cold_boot;begin_case(8);allow_fault=1;bad_bid=1;
        frame(0,800,8);wait(adapter_fault[0]);repeat(20)step;
        need(fault&&!drained&&cmd_ready==0&&cpls[0]==0&&inflight!=0,"BID_QUARANTINE");passed;
        cold_boot;begin_case(9);allow_fault=1;
        fork
            frame(0,900,64);frame(1,901,64);
            begin wait(dut.receiver.fill_active==3);repeat(5)step;init_mask=1;repeat(10)step;init_mask=3;end
        join
        drain_all;need(fault&&cmd_ready==0&&rst_n&&mem_rst_n,"INIT_LOSS_RETAINED");readback(0,900,64);readback(1,901,64);passed;
        cold_boot;begin_case(10);allow_fault=1;expected_error=1000;bad_bresp=1;
        frame(0,1000,9);drain_all;need(fault&&cpls[0]==1,"BRESP_PROPAGATED");readback(0,1000,9);passed;
        cold_boot;begin_case(11);
        step;fence_id=64'habcd;fence_valid=1;
        do @(posedge clk);while(!fence_ready);step;fence_valid=0;
        wait(fence_done_valid);repeat(5)step;allow_fault=1;fault_in=1;step;fault_in=0;
        repeat(5)step;need(fault&&fence_done_valid&&fence_done_id==64'habcd,"ADVERTISED_FENCE_RETAINED");
        fence_done_ready=1;step;fence_done_ready=0;drain_all;passed;
        $display("DH ALL_PASS cases=11 commands=%0d bytes=%0d reads=%0d",total_cmds,total_bytes,native_read_beats);$finish;
    end
    initial begin #200000;need(0,"WATCHDOG");end
endmodule
