`timescale 1ns/1ps
`ifdef XILINX_SIMULATOR
module short(in1,in1);inout in1;endmodule
`endif
module tb_ntrpu_dual_multibuffer_mig;
    reg mgmt_clk=0,source_clk=0;
    initial begin #0.7;forever #2.5 source_clk=~source_clk;end
    initial begin #1.1;forever #5 mgmt_clk=~mgmt_clk;end
    reg external_reset_async=1,rearm=0,stop=0;
    initial begin #40;external_reset_async=0;end
    wire [1:0] ui_clk,calibrated,ui_rst,memory_reset_n,memory_alive,interrupt_ecc;
    wire source_reset_n,admit,running,rearm_ready,restart_required,peers_reset_req,source_alive;
    reg cmd_valid=0,cmd_channel=0,data_valid=0;
    wire cmd_ready,data_ready;
    reg [63:0] cmd_addr=0,cmd_tag=0;
    reg [7:0] cmd_len=0;
    reg [511:0] data=0;
    reg [1:0] cpl_ready=0,hold_cpl=0,hold_w=0,hold_b=0;
    wire [1:0] cpl_valid;
    wire [127:0] cpl_tag;
    wire [3:0] cpl_status;
    wire fault,drained;
    wire [7:0] banks_used;
    wire [9:0] pending_counts;
    wire [1:0] m_awvalid,m_awready,m_wvalid,m_wready,m_wlast,m_bvalid,m_bready,m_rvalid;
    wire [73:0] m_awaddr;
    wire [15:0] m_awlen;
    wire [7:0] m_awid,m_awcache,m_bid;
    wire [5:0] m_awsize;
    wire [3:0] m_awburst,m_bresp;
    wire [1023:0] m_wdata;
    wire [127:0] m_wstrb;
    reg [73:0] araddr=0;
    reg [1:0] arvalid=0,rready=0;
    wire [1:0] arready,rlast;
    wire [7:0] rid;
    wire [3:0] rresp;
    wire [1023:0] rdata;
    ntrpu_dual_multibuffer_mig dut(.*);
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
    end
    function automatic longint ps;return longint'($realtime/1ps);endfunction
    function automatic [511:0] pattern(input integer tag,beat);
        for(integer lane=0;lane<64;lane++)pattern[lane*8+:8]=8'(tag*37+beat*13+lane*19);
    endfunction
    task automatic need(input bit ok,input string message);
        if(!ok)$fatal(1,"DUAL_MIG_CHECK %s",message);
    endtask
    task automatic wait_source(input integer cycles);repeat(cycles)@(negedge source_clk);endtask
    integer accepted=0,completed=0,fill=0,fill_beat=0,cycle=0,one_ready=0,blocked=0;
    integer lengths[0:63],channels[0:63],statuses[0:63];
    reg [63:0] addresses[0:63];
    reg finished[0:63];
    integer queues[0:1][0:63],counts[0:1]='{0,0},cursors[0:1]='{0,0};
    integer aw_cursor[0:1]='{0,0},w_cursor[0:1]='{0,0},b_cursor[0:1]='{0,0},w_beat[0:1]='{0,0};
    integer writes[0:1]='{0,0},reads[0:1]='{0,0},peak[0:1]='{0,0},golden_count[0:1]='{0,0};
    longint aw_time[0:63],w_time[0:63],b_time[0:63];
    reg [63:0] golden_addr[0:1][0:159];
    reg [511:0] golden_data[0:1][0:159];
    reg [7:0] last_banks=0;
    integer max_banks=0;
    always @(negedge source_clk)for(integer ch=0;ch<2;ch++)cpl_ready[ch]=admit && !hold_cpl[ch] && cycle%7!=ch;
    always @(posedge source_clk)begin
        cycle++;
        if(!admit)begin blocked++;need(!cmd_ready && !cpl_valid,"CLOSED_ADMISSION");end
        if(calibrated==1 && ui_rst[0]==0)begin one_ready++;need(!cmd_ready,"ONE_READY_BARRIER");end
        if(source_reset_n)begin
            if(banks_used!=last_banks)begin
                $display("DUAL BANKS %0d mask=%02h",ps(),banks_used);last_banks=banks_used;
                if($countones(banks_used)>max_banks)max_banks=$countones(banks_used);
            end
            if(cmd_valid && cmd_ready)begin
                need(admit && !stop && !fault && calibrated==3 && ui_rst==0,"COMMAND_ADMISSION");
                need(cmd_tag==accepted+1 && fill==0,"COMMAND_IDENTITY");
                accepted++;fill=accepted;fill_beat=0;
                addresses[accepted]=cmd_addr;lengths[accepted]=cmd_len+1;channels[accepted]=cmd_channel;
                statuses[accepted]=({1'b0,cmd_addr}+65'((cmd_len+1)*64)>65'd137438953472)?3:0;
                queues[cmd_channel][counts[cmd_channel]++]=accepted;
                $display("DUAL CMD %0d %0d %0d %h %0d %0d",ps(),cmd_channel,accepted,cmd_addr,cmd_len+1,statuses[accepted]);
            end
            if(data_valid && data_ready)begin
                need(fill>0 && data===pattern(fill,fill_beat),"INPUT_FRAME");
                $display("DUAL IN %0d %0d %0d %h",ps(),fill,fill_beat,data);
                fill_beat++;if(fill_beat==lengths[fill])fill=0;
            end
            for(integer ch=0;ch<2;ch++)if(cpl_valid[ch] && cpl_ready[ch])begin
                integer tag;tag=queues[ch][cursors[ch]++];
                need(cpl_tag[ch*64+:64]==tag && cpl_status[ch*2+:2]==statuses[tag],"COMPLETION_ORDER_STATUS");
                if(statuses[tag]==0)need(b_time[tag]>0 && b_time[tag]<ps(),"COMPLETION_BEFORE_B");
                finished[tag]=1;completed++;
                $display("DUAL CPL %0d %0d %0d %0d",ps(),ch,tag,cpl_status[ch*2+:2]);
            end
        end
    end
    for(genvar g=0;g<2;g++)begin: monitor
        reg aw_held=0,w_held=0,b_held=0,r_held=0,cpl_held=0;
        reg [65:0] saved_cpl;
        reg [57:0] saved_aw;
        reg [576:0] saved_w;
        reg [5:0] saved_b;
        reg [518:0] saved_r;
        longint last_ui=0;integer ui_edges=0;
        always @(posedge source_clk)if(source_reset_n)begin
            if(cpl_held)need(cpl_valid[g] && {cpl_tag[g*64+:64],cpl_status[g*2+:2]}===saved_cpl,"CPL_STABILITY");
            cpl_held=cpl_valid[g]&&!cpl_ready[g];saved_cpl={cpl_tag[g*64+:64],cpl_status[g*2+:2]};
        end
        always @(posedge ui_clk[g])begin
            // MMCM startup edges are not the calibrated UI clock contract.
            if(calibrated[g] && !ui_rst[g])begin
                if(ui_edges>0)need(ps()-last_ui==5000,"UI_PERIOD");
                last_ui=ps();ui_edges++;
                if(ui_edges==4)$display("DUAL CLOCK %0d port=%0d period=5000 phase=%0d",ps(),g,ps()%5000);
            end else begin
                need(ui_edges==0,"READY_CLOCK_LOST");
            end
            if(m_awvalid[g] || (m_wvalid[g]&&!hold_w[g]) || arvalid[g])need(calibrated[g]&&!ui_rst[g]&&memory_reset_n[g],"MIG_BEFORE_READY");
            if(memory_reset_n[g])begin
                need(!interrupt_ecc[g],"ECC_INTERRUPT");
                if(aw_held)need(m_awvalid[g] && {m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awid[g*4+:4],m_awcache[g*4+:4]}===saved_aw,"AW_STABILITY");
                aw_held=m_awvalid[g]&&!m_awready[g];saved_aw={m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awid[g*4+:4],m_awcache[g*4+:4]};
                if(w_held)need(m_wvalid[g] && {m_wdata[g*512+:512],m_wstrb[g*64+:64],m_wlast[g]}===saved_w,"W_STABILITY");
                w_held=m_wvalid[g]&&!m_wready[g];saved_w={m_wdata[g*512+:512],m_wstrb[g*64+:64],m_wlast[g]};
                if(b_held)need(m_bvalid[g] && {m_bid[g*4+:4],m_bresp[g*2+:2]}===saved_b,"B_STABILITY");
                b_held=m_bvalid[g]&&!m_bready[g];saved_b={m_bid[g*4+:4],m_bresp[g*2+:2]};
                if(r_held)need(m_rvalid[g] && {rid[g*4+:4],rresp[g*2+:2],rlast[g],rdata[g*512+:512]}===saved_r,"R_STABILITY");
                r_held=m_rvalid[g]&&!rready[g];saved_r={rid[g*4+:4],rresp[g*2+:2],rlast[g],rdata[g*512+:512]};
                if(m_awvalid[g]&&m_awready[g])begin
                    integer tag;need(aw_cursor[g]<counts[g],"UNAUTHORIZED_AW");tag=queues[g][aw_cursor[g]++];
                    need(statuses[tag]==0 && m_awaddr[g*37+:37]==addresses[tag] && m_awlen[g*8+:8]==lengths[tag]-1 &&
                         m_awsize[g*3+:3]==6 && m_awburst[g*2+:2]==1 && m_awid[g*4+:4]==0,"MIG_AW_FIELDS");
                    aw_time[tag]=ps();if(aw_cursor[g]-b_cursor[g]>peak[g])peak[g]=aw_cursor[g]-b_cursor[g];
                    need(peak[g]<=4,"CREDIT_OVERFLOW");$display("DUAL AW %0d %0d %0d %h %0d",ps(),g,tag,m_awaddr[g*37+:37],m_awlen[g*8+:8]+1);
                end
                if(m_wvalid[g]&&m_wready[g])begin
                    integer tag,slot;reg[63:0]address;
                    need(w_cursor[g]<counts[g],"UNAUTHORIZED_W");tag=queues[g][w_cursor[g]];
                    need(statuses[tag]==0 && m_wdata[g*512+:512]===pattern(tag,w_beat[g]) && m_wstrb[g*64+:64]=='1 &&
                         m_wlast[g]==(w_beat[g]==lengths[tag]-1),"MIG_W_BYTES");
                    address=addresses[tag]+w_beat[g]*64;slot=-1;
                    for(integer k=0;k<golden_count[g];k++)if(golden_addr[g][k]==address)slot=k;
                    if(slot<0)begin slot=golden_count[g]++;golden_addr[g][slot]=address;end
                    golden_data[g][slot]=m_wdata[g*512+:512];writes[g]++;
                    $display("DUAL W %0d %0d %0d %0d %b %h",ps(),g,tag,w_beat[g],m_wlast[g],m_wdata[g*512+:512]);
                    w_beat[g]++;if(m_wlast[g])begin w_time[tag]=ps();w_cursor[g]++;w_beat[g]=0;end
                end
                if(m_bvalid[g]&&m_bready[g])begin
                    integer tag;need(b_cursor[g]<counts[g],"UNAUTHORIZED_B");tag=queues[g][b_cursor[g]++];
                    need(m_bid[g*4+:4]==0 && m_bresp[g*2+:2]==0 && aw_time[tag]>0 && w_time[tag]>0 && aw_time[tag]<ps()&&w_time[tag]<ps(),"MIG_B_ORDER");
                    b_time[tag]=ps();$display("DUAL B %0d %0d %0d 0",ps(),g,tag);
                end
                if(arvalid[g]&&arready[g])
                    $display("DUAL AR %0d %0d %h",ps(),g,araddr[g*37+:37]);
                if(m_rvalid[g]&&rready[g])begin
                    reads[g]++;
                    $display("DUAL R %0d %0d %h %h",ps(),g,araddr[g*37+:37],rdata[g*512+:512]);
                end
            end
        end
    end
    always @(calibrated or ui_rst or memory_reset_n or admit or running)
        $strobe("DUAL STATE %0d calib=%b ui_rst=%b resets=%b admit=%b running=%b",ps(),calibrated,ui_rst,memory_reset_n,admit,running);
    task automatic send(input integer ch,input reg[63:0]address,input integer count,input integer stop_beat=-1);
        integer tag;tag=accepted+1;
        @(negedge source_clk);cmd_channel=ch;cmd_addr=address;cmd_len=count-1;cmd_tag=tag;cmd_valid=1;
        do @(posedge source_clk);while(!cmd_ready);
        @(negedge source_clk);cmd_valid=0;
        for(integer beat=0;beat<count;beat++)begin
            data=pattern(tag,beat);data_valid=1;do @(posedge source_clk);while(!data_ready);
            @(negedge source_clk);data_valid=0;
            if(beat==stop_beat)begin stop=1;$display("DUAL STOP %0d fill port=%0d tag=%0d",ps(),ch,tag);end
        end
    endtask
    task automatic drain;
        integer age;age=0;
        while(!drained || completed!=accepted)begin wait_source(1);age++;need(age<20000,"DRAIN_TIMEOUT");end
        need(fill==0 && banks_used==0,"BANK_DRAIN");$display("DUAL DRAIN %0d",ps());
    endtask
    // Static per-port tasks avoid variable event elaboration and conditional
    // null-event statements that failed the isolated XSIM timing diagnostics.
    for(genvar ch=0;ch<2;ch++)begin: read_port
    task automatic readback(input integer first=0,input integer count=-1);
        integer last;last=(count<0)?golden_count[ch]:first+count;
        $display("DUAL SWEEP %0d port=%0d first=%0d count=%0d",ps(),ch,first,last-first);
        for(integer slot=first;slot<last;slot++)begin
            @(negedge ui_clk[ch]);araddr[ch*37+:37]=golden_addr[ch][slot][36:0];arvalid[ch]=1;
            do @(posedge ui_clk[ch]);while(!arready[ch]);
            @(negedge ui_clk[ch]);arvalid[ch]=0;while(!m_rvalid[ch])@(negedge ui_clk[ch]);
            repeat(3)@(negedge ui_clk[ch]);
            need(rid[ch*4+:4]==0 && rresp[ch*2+:2]==0 && rlast[ch] && rdata[ch*512+:512]===golden_data[ch][slot],"READBACK_BYTES");
            rready[ch]=1;@(posedge ui_clk[ch]);
            @(negedge ui_clk[ch]);rready[ch]=0;
        end
    endtask
    end
    task automatic release_b;
        @(negedge ui_clk[0]);hold_b[0]=0;@(negedge ui_clk[1]);hold_b[1]=0;
    endtask
    initial begin wait(rearm_ready);@(negedge mgmt_clk);rearm=1;repeat(4)@(negedge mgmt_clk);rearm=0;end
    initial begin
        for(integer k=0;k<64;k++)begin aw_time[k]=0;w_time[k]=0;b_time[k]=0;finished[k]=0;end
        #100;$display("DUAL CASE cold_barrier");send(0,0,1);send(1,0,1);drain;
        $display("DUAL CASE sparse_ranks");
        for(integer ch=0;ch<2;ch++)begin
            send(ch,64'h100000040,1);send(ch,64'h800000000,1);send(ch,64'h1000000000,1);
            send(ch,64'h1800000000,1);send(ch,64'h1fffffffc0,1);
        end
        drain;fork read_port[0].readback();read_port[1].readback();join
        $display("DUAL CASE bursts");
        for(integer ch=0;ch<2;ch++)begin send(ch,64'h100001000,64);send(ch,64'h1ffffffe00,8);end
        drain;fork read_port[0].readback();read_port[1].readback();join
        $display("DUAL CASE credits");hold_b=3;
        for(integer k=0;k<4;k++)for(integer ch=0;ch<2;ch++)send(ch,64'h100002000+k*4096,1);
        wait_source(2000);need(aw_cursor[0]-b_cursor[0]==4 && aw_cursor[1]-b_cursor[1]==4,"DUAL_CREDIT_SATURATION");
        $display("DUAL SATURATED %0d pending0=4 pending1=4",ps());stop=1;
        $display("DUAL STOP %0d b_held",ps());wait_source(20);need(!drained&&!cmd_ready,"B_HELD_DRAIN");
        release_b;drain;stop=0;
        $display("DUAL CASE independent_progress");
        @(negedge ui_clk[0]);hold_w[0]=1;send(0,64'h100006000,4);wait(aw_time[25]>0);
        send(1,64'h100006000,4);wait(finished[26]);
        need(!finished[25] && w_time[25]==0 && !drained,"INDEPENDENT_PROGRESS");
        $display("DUAL PROGRESS %0d held=25 completed=26",ps());read_port[1].readback(golden_count[1]-4,4);
        @(negedge ui_clk[0]);hold_w[0]=0;drain;
        $display("DUAL CASE reserve_pool");hold_b=3;
        fork
            begin for(integer k=0;k<24;k++)send(k%2,64'h200001000+(k/2)*4096,4);end
            begin
                wait(banks_used==8'hff);wait_source(20);need(banks_used==8'hff && !drained,"POOL_RETENTION");
                $display("DUAL RESERVE %0d mask=%02h",ps(),banks_used);release_b;
            end
        join
        drain;fork read_port[0].readback(golden_count[0]-48,48);read_port[1].readback(golden_count[1]-48,48);join
        $display("DUAL CASE stop_fill");
        send(0,64'h300000000,8,1);drain;stop=0;send(1,64'h300000000,8,1);drain;stop=0;
        $display("DUAL CASE stop_after_aw");
        @(negedge ui_clk[1]);hold_w[1]=1;send(1,64'h400000000,8);wait(aw_time[53]>0);
        @(negedge source_clk);stop=1;$display("DUAL STOP %0d aw_exposed port=1 tag=53",ps());
        wait_source(20);need(!drained&&!cmd_ready,"W_HELD_DRAIN");
        @(negedge ui_clk[1]);hold_w[1]=0;drain;stop=0;
        $display("DUAL CASE completion_backpressure");hold_cpl=3;
        send(0,64'h500000000,1);send(1,64'h500000000,1);wait(cpl_valid==3);wait_source(20);
        need(!drained,"COMPLETION_OWNERSHIP");hold_cpl=0;drain;fork read_port[0].readback();read_port[1].readback();join
        $display("DUAL CASE range_reject");send(1,64'h2000000000,1);drain;
        need(fault&&!cmd_ready,"RANGE_REJECT_CLOSED");fork read_port[0].readback(0,1);read_port[1].readback(0,1);join
        need(accepted==56 && completed==56 && max_banks==8 && one_ready>500,"FINAL_COVERAGE");
        need(peak[0]==4 && peak[1]==4 && !restart_required,"FINAL_CREDITS_EPOCH");
        for(integer rank=0;rank<4;rank++)need(mem[0].model.rank_acts[rank]>0 && mem[1].model.rank_acts[rank]>0,"RANK_COVERAGE");
        fork mem[0].model.verify_periodic_complete();mem[1].model.verify_periodic_complete();join
        $display("DUAL PASS commands=%0d completions=%0d writes0=%0d writes1=%0d reads0=%0d reads1=%0d locations0=%0d locations1=%0d peak0=%0d peak1=%0d banks=%0d one_ready=%0d blocked=%0d ps=%0d",
                 accepted,completed,writes[0],writes[1],reads[0],reads[1],golden_count[0],golden_count[1],peak[0],peak[1],max_banks,one_ready,blocked,ps());
        $finish;
    end
    initial begin #500000;$fatal(1,"DUAL_MIG_TIMEOUT commands=%0d completions=%0d banks=%h",accepted,completed,banks_used);end
endmodule
