`timescale 1ns/1ps
module tb_ntrpu_read_return;
    parameter integer COMMON=0,NATIVE=0;
    reg clk=0,osc=0,mem_pause=0,cold_reset_n=0,stop=0,fault_in=0;
    realtime mem_half=1.111;
    always #2.5 clk=~clk;
    initial begin #0.731;forever begin #(mem_half);if(!mem_pause)osc=~osc;end end
    wire mem_clk=COMMON?(clk&&!mem_pause):osc;
    wire rst_n,mem_rst_n,started,fault,drained,stopped;
    reg [1:0] init_mask=3;
    wire [1:0] init_complete_async={apb_complete_1,apb_complete_0}&init_mask;
    reg [1:0] cmd_valid=0,data_ready=0,cpl_ready=0;
    wire [1:0] cmd_ready,data_valid,data_last,cpl_valid,frame_idle,adapter_pending,adapter_fault;
    reg [127:0] cmd_addr=0,cmd_tag=0;
    reg [15:0] cmd_len=0;
    wire [1023:0] data;
    wire [127:0] data_tag,cpl_tag;
    wire [3:0] cpl_status;
    wire [25:0] cpl_bytes;
    wire [1:0] h_arvalid,h_arready,h_rvalid,h_rready,h_rlast;
    wire [67:0] h_araddr;
    wire [7:0] h_arlen;
    wire [11:0] h_arid,h_rid;
    wire [5:0] h_arsize;
    wire [3:0] h_arburst,h_rresp;
    wire [511:0] h_rdata;
    ntrpu_hbm_read_return dut(.*);

    reg [1:0] wcmd_valid=0,wdata_valid=0;
    wire [1:0] wcmd_ready,wdata_ready,wcpl_valid;
    reg [127:0] wcmd_addr=0,wcmd_tag=0;
    reg [15:0] wcmd_len=0;
    reg [1023:0] wdata=0;
    wire [127:0] wcpl_tag;
    wire [3:0] wcpl_status;
    wire wstarted,wfault,wdrained,wrst_n,wmem_rst_n;
    wire [1:0] h_awvalid,h_awready,h_wvalid,h_wready,h_wlast,h_bvalid,h_bready;
    wire [67:0] h_awaddr;
    wire [7:0] h_awlen;
    wire [11:0] h_awid,h_bid;
    wire [5:0] h_awsize;
    wire [3:0] h_awburst,h_bresp;
    wire [511:0] h_wdata;
    wire [63:0] h_wstrb;
    ntrpu_dual_hbm_async writer (
        .clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_reset_n),.init_complete_async(init_complete_async),
        .rst_n(wrst_n),.mem_rst_n(wmem_rst_n),.started(wstarted),.inflight(),.stop(1'b0),.fault_in(1'b0),
        .cmd_valid(wcmd_valid),.cmd_ready(wcmd_ready),.reserve_can(),.cmd_addr(wcmd_addr),.cmd_tag(wcmd_tag),.cmd_len(wcmd_len),
        .data_valid(wdata_valid),.data_ready(wdata_ready),.data(wdata),.fence_valid(1'b0),.fence_ready(),.fence_id(64'b0),
        .fence_done_valid(),.fence_done_ready(1'b1),.fence_done_id(),.cpl_valid(wcpl_valid),.cpl_ready(2'b11),.cpl_tag(wcpl_tag),.cpl_status(wcpl_status),
        .fault(wfault),.drained(wdrained),.banks_used(),.pending_counts(),
        .h_awvalid(h_awvalid),.h_awready(h_awready),.h_awaddr(h_awaddr),.h_awlen(h_awlen),.h_awid(h_awid),.h_awsize(h_awsize),.h_awburst(h_awburst),
        .h_wvalid(h_wvalid),.h_wready(h_wready),.h_wdata(h_wdata),.h_wstrb(h_wstrb),.h_wlast(h_wlast),
        .h_bvalid(h_bvalid),.h_bready(h_bready),.h_bid(h_bid),.h_bresp(h_bresp),.adapter_idle(),.adapter_fault());
    reg [1:0] ar_open=3,r_open=3,stray=0;
    reg ready_probe=0;
    integer ready_checks=0;
    wire [1:0] raw_arready,raw_rvalid,raw_rlast;
    wire [511:0] raw_rdata;
    wire [11:0] raw_rid;
    wire [3:0] raw_rresp;
    integer injection=0,inject_index=0;
    integer native_reads[0:1],native_addresses[0:1];
    assign h_arready=(raw_arready&ar_open)^(ready_probe?2'b01:2'b00);
    assign h_rvalid=(raw_rvalid&r_open)|stray|(ready_probe?2'b01:2'b00);
    assign h_rdata=raw_rdata;
    for(genvar c=0;c<2;c++)begin: inject
        wire hit=c==0&&((injection==5)?raw_rlast[0]:(native_reads[0]==inject_index));
        assign h_rid[c*6+:6]=(stray[c]?6'h20:(raw_rid[c*6+:6]^((hit&&injection==3)?6'h20:6'h00)))^((c==0&&ready_probe)?6'h3f:6'h00);
        assign h_rresp[c*2+:2]=(raw_rresp[c*2+:2]|((hit&&injection==1)?2'b10:(hit&&injection==2)?2'b11:(hit&&injection==6)?2'b01:2'b00))^((c==0&&ready_probe)?2'b11:2'b00);
        assign h_rlast[c]=raw_rlast[c]^((hit&&(injection==4||injection==5))?1'b1:1'b0)^(c==0&&ready_probe);
    end
    `include "ntrpu_read_return_binding.svh"

    integer scenario=0,epoch=0,cycles=0,commands=0,writes=0,read_beats=0,output_beats=0,completions=0,dual=0;
    integer expected[0:1],active_tag[0:1],active_len[0:1],delivered[0:1],read_start[0:1];
    integer wdone[0:4095],done[0:4095];
    reg allow_fault=0;
    reg [1:0] held_data=0,held_cpl=0,held_ar=0;
    reg [1153:0] saved_data;
    reg [157:0] saved_cpl;
    reg [101:0] saved_ar;
    function automatic longint stamp();return longint'($realtime*1000);endfunction
    function automatic [63:0] destination(input integer tag);
        return (tag%2?64'h200000000:0)+64'h10000+64'(tag/2)*4096;
    endfunction
    function automatic [511:0] pattern(input integer tag,input integer beat);
        for(integer j=0;j<16;j++)pattern[j*32+:32]=(tag*32'h9e3779b9)^(beat*32'h10203)^(j*32'h1010101);
    endfunction
    task automatic need(input bit ok,input string why);
        if(!ok)$fatal(1,"RR_FAIL case=%0d epoch=%0d cycle=%0d %s",scenario,epoch,cycles,why);
    endtask
    // Perturb inputs entirely between memory edges; no transfer is injected.
    // READY must remain independent even when an offered response is malformed.
    always @(negedge mem_clk)if(!NATIVE&&scenario==1&&mem_rst_n&&h_rready[0]&&ready_checks==0)begin
        #0.02;ready_probe=1;#0.02;need(h_rready[0]===1'b1,"READY_COMBINATORIAL_DEPENDENCE");
        ready_probe=0;#0.02;need(h_rready[0]===1'b1,"READY_COMBINATORIAL_RECOVERY");
        ready_checks++;$display("RR READY_CUT %0d %0d",stamp(),ready_checks);
    end
    task automatic step;@(negedge clk);#0.2;endtask
    always @(posedge clk)begin
        cycles++;
        if(!rst_n)begin
            held_data=0;held_cpl=0;
            for(integer c=0;c<2;c++)begin active_tag[c]=-1;active_len[c]=0;delivered[c]=0;read_start[c]=0;end
        end else begin
            if(!allow_fault)need(!fault&&!wfault,"UNEXPECTED_FAULT");
            if(!started)need(cmd_ready==0&&!drained,"STARTUP_GATING");
            if(stopped)need(cmd_ready==0,"STOP_RETENTION");
            if((data_valid&data_ready)==3)dual++;
            for(integer c=0;c<2;c++)begin
                need(!held_data[c]||(data_valid[c]&&{data_tag[c*64+:64],data_last[c],data[c*512+:512]}===saved_data[c*577+:577]),"DATA_STABILITY");
                need(!held_cpl[c]||(cpl_valid[c]&&{cpl_tag[c*64+:64],cpl_status[c*2+:2],cpl_bytes[c*13+:13]}===saved_cpl[c*79+:79]),"CPL_STABILITY");
                held_data[c]=data_valid[c]&&!data_ready[c];saved_data[c*577+:577]={data_tag[c*64+:64],data_last[c],data[c*512+:512]};
                held_cpl[c]=cpl_valid[c]&&!cpl_ready[c];saved_cpl[c*79+:79]={cpl_tag[c*64+:64],cpl_status[c*2+:2],cpl_bytes[c*13+:13]};
                if(wcmd_valid[c]&&wcmd_ready[c])begin
                    writes++;$display("RR WCMD %0d %0d %0d %0d %0d %h",stamp(),epoch,c,wcmd_tag[c*64+:64],int'(wcmd_len[c*8+:8])+1,wcmd_addr[c*64+:64]);
                end
                if(wdata_valid[c]&&wdata_ready[c])$display("RR WI %0d %0d %0d %h",stamp(),epoch,c,wdata[c*512+:512]);
                if(wcpl_valid[c])begin
                    need(wcpl_status[c*2+:2]==0,"WRITE_COMPLETION");wdone[wcpl_tag[c*64+:64]]=1;
                    $display("RR WCPL %0d %0d %0d %0d",stamp(),epoch,c,wcpl_tag[c*64+:64]);
                end
                if(cmd_valid[c]&&cmd_ready[c])begin
                    need(active_tag[c]==-1,"OVERLAPPING_COMMANDS");active_tag[c]=cmd_tag[c*64+:64];active_len[c]=int'(cmd_len[c*8+:8])+1;delivered[c]=0;read_start[c]=native_reads[c];
                    commands++;$display("RR CMD %0d %0d %0d %0d %0d %h",stamp(),epoch,c,active_tag[c],active_len[c],cmd_addr[c*64+:64]);
                end
                if(data_valid[c])need(native_reads[c]-read_start[c]==2*active_len[c],"PUBLICATION_BEFORE_FULL_FRAME");
                if(data_valid[c]&&data_ready[c])begin
                    need(active_tag[c]>=0&&data_tag[c*64+:64]==active_tag[c]&&expected[c]==0,"DATA_IDENTITY");
                    need(data[c*512+:512]===pattern(active_tag[c],delivered[c])&&data_last[c]==(delivered[c]+1==active_len[c]),"RETURN_DATA");
                    $display("RR DATA %0d %0d %0d %0d %0d %0d %h",stamp(),epoch,c,active_tag[c],delivered[c],data_last[c],data[c*512+:512]);delivered[c]++;output_beats++;
                end
                if(cpl_valid[c]&&cpl_ready[c])begin
                    need(active_tag[c]>=0&&cpl_tag[c*64+:64]==active_tag[c]&&cpl_status[c*2+:2]==expected[c],"COMPLETION_ID_STATUS");
                    need(delivered[c]==(expected[c]==0?active_len[c]:0)&&cpl_bytes[c*13+:13]==delivered[c]*64,"COMPLETION_BYTES");
                    $display("RR CPL %0d %0d %0d %0d %0d %0d",stamp(),epoch,c,active_tag[c],cpl_status[c*2+:2],cpl_bytes[c*13+:13]);
                    done[active_tag[c]]=1;active_tag[c]=-1;completions++;
                end
            end
        end
    end
    always @(posedge mem_clk)begin
        if(!mem_rst_n)begin held_ar=0;for(integer c=0;c<2;c++)begin native_reads[c]=0;native_addresses[c]=0;end end
        else for(integer c=0;c<2;c++)begin
            need(!held_ar[c]||(h_arvalid[c]&&{h_araddr[c*34+:34],h_arlen[c*4+:4],h_arid[c*6+:6],h_arsize[c*3+:3],h_arburst[c*2+:2]}===saved_ar[c*49+:49]),"AR_STABILITY");
            held_ar[c]=h_arvalid[c]&&!h_arready[c];saved_ar[c*49+:49]={h_araddr[c*34+:34],h_arlen[c*4+:4],h_arid[c*6+:6],h_arsize[c*3+:3],h_arburst[c*2+:2]};
            if(h_arvalid[c]&&h_arready[c])begin
                need(h_arid[c*6+:6]==0&&h_arsize[c*3+:3]==5&&h_arburst[c*2+:2]==1,"NATIVE_AR_ATTRIBUTES");
                native_addresses[c]++;$display("RR AR %0d %0d %0d %h %0d",stamp(),epoch,c,h_araddr[c*34+:34],int'(h_arlen[c*4+:4])+1);
            end
            if(h_rvalid[c]&&h_rready[c])begin
                $display("RR NR %0d %0d %0d %0d %0d %0d %h",stamp(),epoch,c,h_rid[c*6+:6],h_rresp[c*2+:2],h_rlast[c],h_rdata[c*256+:256]);
                native_reads[c]++;read_beats++;
            end
        end
    end
    task automatic boot;
        step;cold_reset_n=0;apb_rst_n=0;mem_pause=0;stop=0;fault_in=0;init_mask=3;allow_fault=0;
        ar_open=3;r_open=3;stray=0;injection=0;inject_index=0;cmd_valid=0;wcmd_valid=0;wdata_valid=0;data_ready=3;cpl_ready=3;
        expected[0]=0;expected[1]=0;
        repeat(NATIVE?940:12)step;epoch++;apb_rst_n=1;cold_reset_n=1;
        wait(started&&wstarted);step;need(mem_rst_n&&wmem_rst_n&&apb_complete_0&&apb_complete_1,"BOOT_STATUS");
        $display("RR BOOT %0d %0d %b %b %b %b",stamp(),epoch,apb_complete_0,apb_complete_1,DRAM_0_STAT_CATTRIP,DRAM_1_STAT_CATTRIP);
    endtask
    task automatic write_frame(input integer c,t,n);
        step;wcmd_addr[c*64+:64]=destination(t);wcmd_tag[c*64+:64]=t;wcmd_len[c*8+:8]=n-1;wcmd_valid[c]=1;
        do @(posedge clk);while(!wcmd_ready[c]);step;wcmd_valid[c]=0;
        for(integer i=0;i<n;i++)begin
            wdata_valid[c]=1;wdata[c*512+:512]=pattern(t,i);
            do @(posedge clk);while(!wdata_ready[c]);step;wdata_valid[c]=0;
        end
        wait(wdone[t]);step;
    endtask
    task automatic command_at(input integer c,t,n,input [63:0] a);
        step;cmd_addr[c*64+:64]=a;cmd_tag[c*64+:64]=t;cmd_len[c*8+:8]=n-1;cmd_valid[c]=1;
        do @(posedge clk);while(!cmd_ready[c]);step;cmd_valid[c]=0;
    endtask
    task automatic command(input integer c,t,n);command_at(c,t,n,destination(t));endtask
    task automatic pair_write(input integer t,n);fork write_frame(0,t,n);write_frame(1,t+1,n);join endtask
    task automatic pair_command(input integer t,n);fork command(0,t,n);command(1,t+1,n);join endtask
    task automatic pair_done(input integer t);wait(done[t]&&done[t+1]);step;endtask
    task automatic begin_case(input integer n);scenario=n;$display("RR CASE %0d %0d",n,stamp());endtask
    task automatic passed;
        repeat(3)step;need(drained,"NOT_DRAINED");$display("RR PASS %0d %0d",scenario,stamp());
    endtask
    task automatic quarantined(input integer t);
        wait(fault);repeat(30)step;need(!drained&&!done[t]&&!data_valid[0]&&!cpl_valid[0]&&adapter_pending[0]&&!h_rready[0],"QUARANTINE");
        $display("RR QUARANTINE %0d %0d %0d %0d",stamp(),epoch,t,scenario);$display("RR PASS %0d %0d",scenario,stamp());
    endtask
    initial begin
        $display("RR CONFIG common=%0d native=%0d revision=3",COMMON,NATIVE);
        for(integer i=0;i<4096;i++)begin wdone[i]=0;done[i]=0;end
        begin_case(1);boot;
        for(integer j=0;j<4;j++)begin
            automatic integer n;
            case(j)0:n=1;1:n=8;2:n=9;default:n=64;endcase
            pair_write(j*2,n);pair_command(j*2,n);pair_done(j*2);
        end
        passed;
        begin_case(2);pair_write(100,64);data_ready=0;cpl_ready=0;pair_command(100,64);
        wait(data_valid==3);repeat(25)step;need(!drained&&cmd_ready==0&&native_reads[0]-read_start[0]==128&&native_reads[1]-read_start[1]==128,"FULL_FRAME_RESERVED");
        $display("RR FULL_RESERVED %0d %0d",stamp(),epoch);
        data_ready=1;repeat(80)step;need(cpl_valid[0]&&data_valid[1]&&!drained,"ASYMMETRIC_CONSUMER");
        data_ready=3;wait(cpl_valid==3);repeat(20)step;cpl_ready=1;repeat(10)step;need(!drained,"HELD_COMPLETION");cpl_ready=3;pair_done(100);passed;
        begin_case(3);pair_write(200,9);cpl_ready=0;pair_command(200,9);wait(cpl_valid==3);repeat(15)step;
        need(!drained&&cmd_ready==0,"COMPLETION_RESERVATION");$display("RR CPL_HELD %0d %0d",stamp(),epoch);cpl_ready=3;pair_done(200);passed;
        begin_case(4);pair_write(300,64);r_open=0;pair_command(300,64);wait(adapter_pending==3);step;stop=1;repeat(8)step;stop=0;
        need(stopped&&!drained&&cmd_ready==0,"STOP_PENDING");r_open=3;pair_done(300);passed;
        if(NATIVE)begin $display("RR ALL_PASS cases=4 commands=%0d writes=%0d native_reads=%0d output_beats=%0d completions=%0d dual=%0d",commands,writes,read_beats,output_beats,completions,dual);$finish;end
        begin_case(5);boot;pair_write(400,64);step;mem_pause=1;
        fork pair_command(400,64);begin repeat(120)step;need(!drained,"PAUSED_CLOCK_PENDING");mem_half=3.317;mem_pause=0;end join
        pair_done(400);passed;
        for(integer s=6;s<=10;s++)begin
            begin_case(s);expected[0]=1;
            case(s)
                6:command(0,500,0);
                7:command(0,502,65);
                8:command_at(0,504,1,destination(504)+1);
                9:command_at(0,506,2,destination(506)+4032);
                10:command_at(0,508,1,64'h400000000);
            endcase
            wait(done[500+2*(s-6)]);passed;
        end
        expected[0]=0;
        for(integer e=1;e<=2;e++)begin
            begin_case(10+e);boot;pair_write(600+e*2,64);expected[0]=2;allow_fault=1;injection=e;inject_index=127;
            pair_command(600+e*2,64);pair_done(600+e*2);need(fault,"MISSING_ERROR_LATCH");passed;
        end
        for(integer e=3;e<=6;e++)begin
            begin_case(10+e);boot;write_frame(0,700+e*2,9);allow_fault=1;injection=e;inject_index=(e==5)?15:0;
            command(0,700+e*2,9);quarantined(700+e*2);
        end
        begin_case(17);boot;write_frame(0,800,8);ar_open=0;command(0,800,8);wait(h_arvalid[0]);repeat(5)step;
        allow_fault=1;stray=1;repeat(10)step;need(fault&&h_arvalid[0],"FAULT_MUST_RETAIN_AR");ar_open=3;
        wait(native_addresses[0]==1);quarantined(800);
        begin_case(18);boot;pair_write(900,64);r_open=0;pair_command(900,64);wait(adapter_pending==3);
        step;allow_fault=1;init_mask=2;repeat(12)step;need(fault,"INIT_LOSS");r_open=3;pair_done(900);passed;
        begin_case(19);boot;pair_write(1000,9);r_open=0;pair_command(1000,9);wait(adapter_pending==3);
        step;allow_fault=1;fault_in=1;step;fault_in=0;need(fault,"EXTERNAL_FAULT");r_open=3;pair_done(1000);passed;
        begin_case(20);boot;
        for(integer t=1100;t<1112;t+=2)begin pair_write(t,8);pair_command(t,8);pair_done(t);end
        passed;
        need(dual>0&&ready_checks==1,"DUAL_CHANNEL_AND_READY_CUT_COVERAGE");
        $display("RR ALL_PASS cases=20 commands=%0d writes=%0d native_reads=%0d output_beats=%0d completions=%0d dual=%0d",commands,writes,read_beats,output_beats,completions,dual);$finish;
    end
    initial begin #200000;$fatal(1,"RR_WATCHDOG");end
endmodule
