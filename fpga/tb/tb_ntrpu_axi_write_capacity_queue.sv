`timescale 1ns/1ps
// Standalone safety cases not reachable through the prevalidated receiver input.
module tb_ntrpu_axi_write_capacity_queue #(parameter integer DEPTH=4);
    localparam integer MEMORY_ADDR_WIDTH=37, JOBS=36;
    reg clk=0; always #2.5 clk=~clk;
    reg rst_n=0,admit=0;
    wire idle;
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
    wire [36:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize,m_axi_awprot;
    wire [1:0] m_axi_awburst;
    wire [3:0] m_axi_awid,m_axi_awcache;
    wire m_axi_awlock,m_axi_awvalid;
    reg m_axi_awready=0;
    wire [511:0] m_axi_wdata;
    wire [63:0] m_axi_wstrb;
    wire m_axi_wlast,m_axi_wvalid;
    reg m_axi_wready=0;
    reg [3:0] m_axi_bid=0;
    reg [1:0] m_axi_bresp=0;
    reg m_axi_bvalid=0;
    wire m_axi_bready;
    ntrpu_axi_write_capacity_queue #(.DEPTH(DEPTH)) dut(.*);

    reg [63:0] address[0:JOBS-1];
    integer beats[0:JOBS-1],size[0:JOBS-1],burst[0:JOBS-1],locked[0:JOBS-1],id[0:JOBS-1];
    integer response[0:JOBS-1],valid_job[0:JOBS-1],input_beats[0:JOBS-1];
    integer output_beats[0:JOBS-1],aw_time[0:JOBS-1],w_time[0:JOBS-1],b_time[0:JOBS-1],due[0:JOBS-1];
    integer queue[0:JOBS-1],allowed=0,seen=0,finished=0,in_job=0;
    integer aw_count=0,w_count=0,b_count=0,cycle=0,peak=0;
    integer early_w=0,credit_stall=0,admit_stall=0,b_stall=0,bytes=0,negative=0;
    integer t,j,k,n;
    reg held_aw=0,held_w=0,held_b=0;
    reg [61:0] previous_aw;
    reg [576:0] previous_w;
    reg [5:0] previous_b;
    function automatic [511:0] pattern(input integer tag,beat);
        for(integer lane=0;lane<16;lane=lane+1)
            pattern[lane*32+:32]=(tag*32'h9e3779b9)^(beat*32'h10203)^(lane*32'h01010101);
    endfunction
    function automatic [63:0] strobe(input integer tag,beat);
        return (tag+beat)%2==0 ? 64'hffffffffffffffff:64'hff00ff00ff00ff00;
    endfunction
    always @(negedge clk) begin
        admit=rst_n && !(cycle>=150 && cycle<230);
        m_axi_awready=rst_n && cycle>=100 && cycle%4!=0;
        m_axi_wready=rst_n && cycle%7!=0;
        s_axi_bready=rst_n && !(cycle>=300 && cycle<500) && cycle%3!=0;
        if(!rst_n) m_axi_bvalid=0;
        else if(!m_axi_bvalid) begin
            t=queue[b_count];
            m_axi_bvalid=b_count<aw_count && b_count<w_count && due[t]>=0 && cycle>=due[t];
        end
        if(b_count<allowed) m_axi_bresp=2'(response[queue[b_count]]);
    end
    always @(posedge clk) begin
        cycle=cycle+1;
        if(rst_n) begin
            if(!admit && s_axi_awready) $fatal(1,"WRITE_GUARD_ADMIT");
            if(held_aw && (!m_axi_awvalid ||
                {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock}!==previous_aw))
                $fatal(1,"WRITE_GUARD_AW_UNSTABLE");
            if(held_w && (!m_axi_wvalid || {m_axi_wlast,m_axi_wstrb,m_axi_wdata}!==previous_w)) $fatal(1,"WRITE_GUARD_W_UNSTABLE");
            if(held_b && (!s_axi_bvalid || {s_axi_bid,s_axi_bresp}!==previous_b)) $fatal(1,"WRITE_GUARD_B_UNSTABLE");
            held_aw=m_axi_awvalid && !m_axi_awready;
            held_w=m_axi_wvalid && !m_axi_wready;
            held_b=s_axi_bvalid && !s_axi_bready;
            previous_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
            previous_w={m_axi_wlast,m_axi_wstrb,m_axi_wdata}; previous_b={s_axi_bid,s_axi_bresp};
            if(held_b) b_stall=b_stall+1;
            if(s_axi_awvalid && !s_axi_awready && seen-finished==DEPTH) credit_stall=credit_stall+1;
            if(s_axi_awvalid && !admit) admit_stall=admit_stall+1;
            if(s_axi_awvalid && s_axi_awready) begin
                if(seen>=JOBS || s_axi_awaddr!==address[seen]) $fatal(1,"WRITE_GUARD_INPUT_ADDRESS");
                $display("UNIT CMD %0d %0d %h %0d %0d",cycle,seen,address[seen],beats[seen],response[seen]);
                seen=seen+1;
            end
            if(s_axi_wvalid && s_axi_wready) begin
                if(in_job>=seen || s_axi_wdata!==pattern(in_job,input_beats[in_job])) $fatal(1,"WRITE_GUARD_INPUT_DATA");
                input_beats[in_job]=input_beats[in_job]+1;
                if(input_beats[in_job]==beats[in_job]) in_job=in_job+1;
            end
            if(m_axi_awvalid && m_axi_awready) begin
                j=queue[aw_count];
                if(aw_count>=allowed || m_axi_awaddr!==37'(address[j]) || m_axi_awlen!=beats[j]-1 ||
                   m_axi_awsize!=6 || m_axi_awburst!=1 || m_axi_awlock || m_axi_awid!=0 ||
                   m_axi_awcache!=4'(j) || m_axi_awprot!=3'(j)) $fatal(1,"WRITE_GUARD_AW_MISMATCH");
                aw_time[j]=cycle; aw_count=aw_count+1;
                $display("UNIT AW %0d %0d",cycle,j);
            end
            if(m_axi_wvalid && m_axi_wready) begin
                j=queue[w_count];
                if(w_count>=allowed || m_axi_wdata!==pattern(j,output_beats[j]) ||
                   m_axi_wstrb!==strobe(j,output_beats[j]) || m_axi_wlast!==(output_beats[j]==beats[j]-1))
                    $fatal(1,"WRITE_GUARD_DATA_MISMATCH");
                if(aw_time[j]==0) early_w=early_w+1;
                $display("UNIT W %0d %0d %0d %h %h",cycle,j,output_beats[j],m_axi_wstrb,m_axi_wdata);
                output_beats[j]=output_beats[j]+1; bytes=bytes+64;
                if(m_axi_wlast) begin w_time[j]=cycle; w_count=w_count+1; end
            end
            for(k=b_count;k<aw_count && k<w_count;k=k+1)
                if(due[queue[k]]<0) due[queue[k]]=cycle+80;
            if(m_axi_bvalid && m_axi_bready) begin
                j=queue[b_count];
                if(aw_time[j]==0 || w_time[j]==0) $fatal(1,"WRITE_GUARD_EARLY_B");
                b_time[j]=cycle; b_count=b_count+1; m_axi_bvalid<=0;
                $display("UNIT B %0d %0d %0d",cycle,j,response[j]);
            end
            if(s_axi_bvalid && s_axi_bready) begin
                j=finished;
                if(j>=seen || input_beats[j]!=beats[j] || s_axi_bid!=id[j] || s_axi_bresp!=response[j])
                    $fatal(1,"WRITE_GUARD_RESPONSE");
                if(valid_job[j] && (b_time[j]==0 || b_time[j]>=cycle)) $fatal(1,"WRITE_GUARD_EARLY_CPL");
                if(!valid_job[j] && (aw_time[j]!=0 || output_beats[j]!=0)) $fatal(1,"WRITE_GUARD_REJECT_LEAK");
                finished=finished+1;
                $display("UNIT CPL %0d %0d %0d",cycle,j,s_axi_bresp);
            end
            if(seen-finished>DEPTH || finished>seen) $fatal(1,"WRITE_GUARD_CREDIT_OVERFLOW");
            if(seen-finished>peak) peak=seen-finished;
            if(idle && seen!=finished && !(s_axi_awvalid && s_axi_awready)) $fatal(1,"WRITE_GUARD_PREMATURE_IDLE");
        end
    end
    initial begin
        if(!$value$plusargs("NEGATIVE=%d",negative)) negative=0;
        for(n=0;n<JOBS;n=n+1) begin
            address[n]=64'h100000000+(64'(n)<<12); beats[n]=1+n%7;
            size[n]=6; burst[n]=1; locked[n]=0; id[n]=0; response[n]=0; valid_job[n]=1;
            input_beats[n]=0; output_beats[n]=0; aw_time[n]=0; w_time[n]=0; b_time[n]=0; due[n]=-1;
        end
        address[1]=64'h2000000000;
        address[2]=64'h1fffffffc0; beats[2]=1;
        address[3]=64'h100000fc0; beats[3]=2;
        address[4]=64'hffffffffffffffc0; beats[4]=1;
        size[5]=5; burst[6]=0; locked[7]=1; id[8]=7; address[9]=64'h100009001;
        response[11]=2; beats[15]=65; address[16]=64'h2100000000;
        for(n=0;n<JOBS;n=n+1) begin
            if(n==1 || (n>=3 && n<=9) || n==15 || n==16) begin valid_job[n]=0; response[n]=3; end
            if(valid_job[n]) begin queue[allowed]=n; allowed=allowed+1; end
        end
        repeat(20) @(negedge clk); rst_n=1;
        if(negative==1) force dut.allowed={DEPTH{1'b1}};
        if(negative==2) force dut.m_axi_wstrb=64'b0;
        if(negative==3) force dut.s_axi_bresp=2'b0;
        for(n=0;n<JOBS;n=n+1) begin
            @(negedge clk); s_axi_awvalid=1; s_axi_awaddr=address[n]; s_axi_awlen=beats[n]-1;
            s_axi_awsize=3'(size[n]); s_axi_awburst=2'(burst[n]); s_axi_awlock=locked[n];
            s_axi_awid=4'(id[n]); s_axi_awcache=4'(n); s_axi_awprot=3'(n);
            do @(posedge clk); while(!s_axi_awready);
            @(negedge clk); s_axi_awvalid=0;
            for(integer beat=0;beat<beats[n];beat=beat+1) begin
                s_axi_wvalid=1; s_axi_wdata=pattern(n,beat); s_axi_wstrb=strobe(n,beat); s_axi_wlast=beat==beats[n]-1;
                do @(posedge clk); while(!s_axi_wready);
                @(negedge clk);
            end
            s_axi_wvalid=0;
        end
        wait(finished==JOBS); repeat(10) @(negedge clk);
        if(!idle || aw_count!=allowed || w_count!=allowed || b_count!=allowed || peak!=DEPTH ||
           early_w==0 || credit_stall==0 || admit_stall==0 || b_stall==0) $fatal(1,"WRITE_GUARD_COVERAGE");
        $display("WRITE_GUARD_SUITE_PASS depth=%0d commands=%0d accepted_memory=%0d bytes=%0d peak=%0d early_w=%0d credit_stall=%0d admit_stall=%0d b_stall=%0d",
            DEPTH,seen,allowed,bytes,peak,early_w,credit_stall,admit_stall,b_stall);
        $finish;
    end
    initial begin #100000; $fatal(1,"WRITE_GUARD_WATCHDOG"); end
endmodule
