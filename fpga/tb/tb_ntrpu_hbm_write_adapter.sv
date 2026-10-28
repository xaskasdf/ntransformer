`timescale 1ns/1ps
module tb_ntrpu_hbm_write_adapter;
    reg clk=0;always #2.5 clk=~clk;
    reg rst_n=0,admit=0;
    wire idle,fault;
    reg [63:0] s_axi_awaddr=0;
    reg [7:0] s_axi_awlen=0;
    reg [2:0] s_axi_awsize=6,s_axi_awprot=0;
    reg [1:0] s_axi_awburst=1;
    reg [3:0] s_axi_awid=0,s_axi_awcache=0;
    reg s_axi_awlock=0,s_axi_awvalid=0;
    wire s_axi_awready;
    reg [511:0] s_axi_wdata=0;
    reg [63:0] s_axi_wstrb=0;
    reg s_axi_wlast=0,s_axi_wvalid=0;
    wire s_axi_wready;
    wire [3:0] s_axi_bid;
    wire [1:0] s_axi_bresp;
    wire s_axi_bvalid;
    reg s_axi_bready=0;
    wire [33:0] m_axi_awaddr;
    wire [3:0] m_axi_awlen;
    wire [2:0] m_axi_awsize;
    wire [1:0] m_axi_awburst;
    wire [5:0] m_axi_awid;
    wire m_axi_awvalid;
    reg m_axi_awready=0;
    wire [255:0] m_axi_wdata;
    wire [31:0] m_axi_wstrb;
    wire m_axi_wlast,m_axi_wvalid;
    reg m_axi_wready=0;
    reg [5:0] m_axi_bid=0;
    reg [1:0] m_axi_bresp=0;
    reg m_axi_bvalid=0;
    wire m_axi_bready;
    ntrpu_hbm_write_adapter dut(.*);

    integer cycle=0,scenario=0,njobs=0,origin=0;
    integer sa=0,swjob=0,swbeat=0,sb=0,ma=0,mwburst=0,mwbeat=0,mb=0,scheduled=0;
    integer beats[0:31],sizes[0:31],bursts[0:31],ids[0:31],locks[0:31];
    reg [63:0] addresses[0:31];
    integer due[0:255],blens[0:255];
    integer b_delay=4,hold_reply=0,pause_aw=0,pause_w=0,random_pause=0;
    reg b_sent=0;
    reg aw_held=0,w_held=0,sb_held=0;
    reg [48:0] aw_saved;
    reg [288:0] w_saved;
    reg [5:0] sb_saved;
    task automatic need(input bit ok,input string why);
        if(!ok)$fatal(1,"HBM_%s case=%0d cycle=%0d",why,scenario,cycle);
    endtask
    task automatic step;@(negedge clk);#0.2;endtask
    function automatic [511:0] pattern(input integer job,input integer beat);
        for(integer k=0;k<64;k++)pattern[k*8+:8]=8'(scenario*29+job*17+beat*7+k);
    endfunction
    function automatic [63:0] strobes(input integer job,input integer beat);
        if(scenario==2 || scenario==9)begin
            case((job+beat)%4)
                0:strobes=64'hffff_ffff_ffff_ffff;
                1:strobes=64'h8000_0001_0000_0001;
                2:strobes=64'h0000_0000_ffff_ffff;
                3:strobes=0;
            endcase
        end else strobes=64'hffff_ffff_ffff_ffff;
    endfunction
    function automatic [1:0] response(input integer child);
        if(scenario==3)begin
            response=0;
            case(child/8)
                0:if(child%8==3)response=2;
                1:if(child%8==4)response=3;
                2:begin if(child%8==1)response=2;if(child%8==5)response=3;end
                3:begin if(child%8==1)response=3;if(child%8==5)response=2;end
                4:if(child%8==7)response=2;
                5:if(child%8==7)response=3;
                6:if(child%8==0)response=2;
                default:response=0;
            endcase
        end else if(scenario==11)response=1;
        else response=0;
    endfunction

    always @(negedge clk)begin
        m_axi_awready=rst_n&&(!pause_aw || (cycle-origin)%64>=32)&&(!random_pause || cycle%7!=0);
        m_axi_wready=rst_n&&(!pause_w || (cycle-origin)%71>=37)&&(!random_pause || cycle%5!=0);
        s_axi_bready=rst_n&&!hold_reply&&(!random_pause || cycle%11!=0);
        if(!m_axi_bvalid||b_sent||!rst_n)begin
            m_axi_bvalid=rst_n&&mb<scheduled&&cycle>=due[mb];
            m_axi_bid=scenario==10?6'h10:0;
            m_axi_bresp=response(mb);
        end
    end
    always @(posedge clk)begin
        cycle++;b_sent=m_axi_bvalid&&m_axi_bready;
        if(!rst_n)begin
            sa=0;swjob=0;swbeat=0;sb=0;ma=0;mwburst=0;mwbeat=0;mb=0;scheduled=0;
            aw_held=0;w_held=0;sb_held=0;
        end else begin
            // IDLE describes the inventory before this sampling edge.
            if(idle)need(sa==sb&&ma==mb&&mwburst==ma&&mwbeat==0,"IDLE_INVENTORY");
            need(!aw_held||(m_axi_awvalid&&{m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid}===aw_saved),"AW_STABLE");
            need(!w_held||(m_axi_wvalid&&{m_axi_wdata,m_axi_wstrb,m_axi_wlast}===w_saved),"W_STABLE");
            need(!sb_held||(s_axi_bvalid&&{s_axi_bid,s_axi_bresp}===sb_saved),"PARENT_B_STABLE");
            aw_held=m_axi_awvalid&&!m_axi_awready;aw_saved={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid};
            w_held=m_axi_wvalid&&!m_axi_wready;w_saved={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            sb_held=s_axi_bvalid&&!s_axi_bready;sb_saved={s_axi_bid,s_axi_bresp};
            if(s_axi_awvalid&&s_axi_awready)begin
                need(sa<njobs&&admit&&!fault,"SOURCE_ADMISSION");
                $display("HA SA %0d %0d %h %0d %0d %0d %0d %0d",cycle,sa,s_axi_awaddr,s_axi_awlen+1,s_axi_awsize,s_axi_awburst,s_axi_awid,s_axi_awlock);sa++;
                need(sa-sb<=4,"PARENT_CREDIT_LIMIT");
            end
            if(s_axi_wvalid&&s_axi_wready)begin
                $display("HA SW %0d %0d %0d %h %h %b",cycle,swjob,swbeat,s_axi_wdata,s_axi_wstrb,s_axi_wlast);
                need(s_axi_wlast==(swbeat==beats[swjob]-1),"SOURCE_WLAST");
                swbeat++;if(s_axi_wlast)begin swjob++;swbeat=0;end
            end
            if(m_axi_awvalid&&m_axi_awready)begin
                need(m_axi_awsize==5&&m_axi_awburst==1&&m_axi_awid==0,"HBM_ATTRIBUTES");
                blens[ma]=m_axi_awlen+1;
                $display("HA MA %0d %0d %h %0d %0d %0d %0d",cycle,ma,m_axi_awaddr,m_axi_awlen+1,m_axi_awsize,m_axi_awburst,m_axi_awid);ma++;
                need(ma-mb<=32,"HBM_ACCEPTANCE_LIMIT");
            end
            if(m_axi_wvalid&&m_axi_wready)begin
                $display("HA MW %0d %0d %0d %h %h %b",cycle,mwburst,mwbeat,m_axi_wdata,m_axi_wstrb,m_axi_wlast);
                mwbeat++;if(m_axi_wlast)begin mwburst++;mwbeat=0;end
            end
            while(scheduled<ma&&scheduled<mwburst)begin due[scheduled]=cycle+b_delay;scheduled++;end
            if(m_axi_bvalid&&m_axi_bready)begin
                need(mb<scheduled,"TARGET_B_CAUSAL");
                $display("HA MB %0d %0d %0d %0d",cycle,mb,m_axi_bid,m_axi_bresp);mb++;
            end
            if(s_axi_bvalid&&s_axi_bready)begin
                need(sb<sa&&sb<swjob,"PARENT_B_CAUSAL");
                $display("HA SB %0d %0d %0d %0d",cycle,sb,s_axi_bid,s_axi_bresp);sb++;
            end
        end
    end

    task automatic configure(input integer number);
        rst_n=0;admit=0;s_axi_awvalid=0;s_axi_wvalid=0;
        hold_reply=0;pause_aw=0;pause_w=0;random_pause=0;b_delay=4;
        repeat(32)step;scenario=number;origin=cycle;
        njobs=8;
        for(integer j=0;j<32;j++)begin
            addresses[j]=64'h100000000+64'(j)*8192;
            beats[j]=j%4==0?1:j%4==1?8:j%4==2?9:64;
            sizes[j]=6;bursts[j]=1;ids[j]=0;locks[j]=0;
        end
        if(number==1)addresses[7]=64'h400000000-4096;
        if(number==2)for(integer j=0;j<8;j++)beats[j]=17;
        if(number==3)begin for(integer j=0;j<8;j++)beats[j]=64;b_delay=32;end
        if(number==4)pause_aw=1;
        if(number==5)pause_w=1;
        if(number==6)begin hold_reply=1;b_delay=64;end
        if(number==8)begin
            addresses[0]=64'h400000000;addresses[1]=64'h800000000;
            addresses[2]=64'h100000001;addresses[3]=64'h100000fc0;beats[3]=2;
            sizes[4]=5;bursts[5]=0;ids[6]=3;locks[7]=1;
            addresses[8]=64'hffffffffffffffc0;beats[8]=1;beats[9]=65;
            njobs=11; // A valid successor must drain after local rejects.
        end
        if(number==9)begin njobs=24;random_pause=1;b_delay=41;end
        if(number>=10)begin njobs=1;beats[0]=1;end
        $display("HA CASE %0d %0d %0d %0d",cycle,scenario,njobs,b_delay);
        rst_n=1;admit=1;step;
    endtask
    task automatic addresses_thread;
        if(scenario==4)repeat(20)step; // WVALID may precede AWVALID.
        for(integer j=0;j<njobs;j++)begin
            s_axi_awaddr=addresses[j];s_axi_awlen=beats[j]-1;s_axi_awsize=sizes[j];
            s_axi_awburst=bursts[j];s_axi_awid=ids[j];s_axi_awlock=locks[j];s_axi_awvalid=1;
            @(posedge clk);while(!s_axi_awready)begin step;@(posedge clk);end
            step;s_axi_awvalid=0;
        end
    endtask
    task automatic data_thread;
        for(integer j=0;j<njobs;j++)for(integer beat=0;beat<beats[j];beat++)begin
            s_axi_wdata=pattern(j,beat);s_axi_wstrb=strobes(j,beat);s_axi_wlast=beat==beats[j]-1;s_axi_wvalid=1;
            @(posedge clk);while(!s_axi_wready)begin step;@(posedge clk);end
            step;s_axi_wvalid=0;
        end
    endtask
    initial begin
        #150;
        for(integer number=1;number<=11;number++)begin
            configure(number);
            fork
                addresses_thread;
                data_thread;
                begin
                    if(scenario==6)begin
                        wait(sa==4&&s_axi_bvalid);repeat(64)step;
                        need(sa==4&&!s_axi_awready&&!idle,"HELD_B_OWNS_PARENT_CREDIT");
                        $display("HA RELEASE %0d",cycle);hold_reply=0;
                    end
                    if(scenario==7)begin
                        wait(sa==2);step;admit=0;$display("HA CLOSE %0d %0d",cycle,sa);
                        wait(sb==sa&&idle);repeat(32)step;
                        need(sa==2&&!s_axi_awready,"CLOSED_ADMISSION");
                        $display("HA REOPEN %0d",cycle);admit=1;
                    end
                end
            join
            if(scenario>=10)begin
                wait(m_axi_bvalid);repeat(64)step;
                need(fault&&!idle&&!s_axi_bvalid&&!s_axi_awready&&!m_axi_bready,"FOREIGN_RESPONSE_QUARANTINE");
                $display("HA QUARANTINE %0d %0d %0d %0d %0d",cycle,sa,sb,m_axi_bid,m_axi_bresp);
            end else begin
                wait(sb==njobs&&idle);repeat(4)step;need(!fault,"CLEAN_FINISH");
            end
            $display("HA PASS %0d %0d %0d %0d %0d %0d",scenario,sa,swjob,sb,ma,mb);
        end
        $display("HA ALL_PASS cases=11 cycle=%0d",cycle);$finish;
    end
    initial begin #2000000;$fatal(1,"HBM_WATCHDOG");end
endmodule
