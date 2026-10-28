`timescale 1ns/1ps
// Cross-domain counters below belong to the scoreboard, not hardware CDC.
module tb_ntrpu_peer_multibuffer_credits #(parameter integer GUARD_SLOTS=4);
    localparam integer CHANNELS=2, RESERVE_BANKS=4, BANK_BEATS=64;
    localparam integer DESCRIPTORS=16, MAX_OUTSTANDING=4, MEMORY_ADDR_WIDTH=37;
    localparam integer BANKS=8, COUNT_BITS=5;
    localparam [63:0] CAPACITY=64'd1<<MEMORY_ADDR_WIDTH;
    reg clk=0; always #2.5 clk=~clk;
    reg [1:0] mem_clk=0, mem_rst_n=0, run_memory=3;
    integer profile=0, negative=0, half0=2000, half1=3650, delay_b=3;
    bit clocks_ready=0;
    initial begin wait(clocks_ready); #0.713; forever begin #(half0/1000.0); if(run_memory[0]) mem_clk[0]=~mem_clk[0]; end end
    initial begin wait(clocks_ready); #1.137; forever begin #(half1/1000.0); if(run_memory[1]) mem_clk[1]=~mem_clk[1]; end end
    reg rst_n=0, stop=0, cmd_valid=0, data_valid=0;
    wire cmd_ready, data_ready;
    reg cmd_channel=0;
    reg [63:0] cmd_addr=0,cmd_tag=0;
    reg [7:0] cmd_len=0;
    reg [511:0] data=0;
    wire [1:0] m_awvalid,m_wvalid,m_wlast,m_bready,cpl_valid;
    reg [1:0] m_awready=0,m_wready=0,m_bvalid=0,cpl_ready=0;
    wire [2*37-1:0] m_awaddr;
    wire [127:0] m_wstrb,cpl_tag;
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
    ntrpu_peer_multibuffer_credits #(.GUARD_SLOTS(GUARD_SLOTS)) dut(.*);

    integer cycle=0, next_tag=1, fill_tag=0, fill_beat=0, next_status=0;
    integer issued[0:1], finished[0:1], allowed[0:1];
    integer completion_queue[0:511], memory_queue[0:511];
    integer beats[0:255], response[0:255], state[0:255], destination[0:255];
    reg [63:0] address[0:255];
    integer input_count[0:255], output_count[0:255];
    longint filled_at[0:255], aw_at[0:255], last_w_at[0:255], b_at[0:255];
    integer memory_aw[0:1], memory_w[0:1], memory_b[0:1], memory_beat[0:1];
    integer eligible[0:255];
    integer memory_cycle[0:1], b_due[0:1], w_total[0:1], aw_stalls[0:1], w_stalls[0:1];
    integer accepted=0, completed=0, aborted=0, transferred=0, resets=0;
    integer p,q,t,i,baseline,abort_before,job;
    reg [1:0] hold_aw=0,hold_w=0,hold_b=0;
    reg hold_cpl=0, was_running=0;
    reg [1:0] held_cpl=0;
    reg [65:0] previous_cpl[0:1];
    longint rate_start,rate_end;
    integer rate_bytes;

    function automatic longint ps; return longint'($realtime*1000.0); endfunction
    function automatic [511:0] pattern(input integer identity,beat);
        for(integer lane=0;lane<16;lane=lane+1)
            pattern[lane*32+:32]=(identity*32'h9e3779b9)^(beat*32'h10203)^(lane*32'h01010101);
    endfunction

    always @(negedge clk) begin
        cpl_ready = rst_n && !hold_cpl && cycle%5!=0 ? 2'b11:2'b00;
    end
    always @(posedge clk) begin
        cycle=cycle+1;
        if(!rst_n) begin
            if(was_running) begin
                for(integer tag=1;tag<next_tag;tag=tag+1)
                    if(state[tag]==1) begin state[tag]=3; aborted=aborted+1; end
            end
            was_running=0; fill_tag=0; fill_beat=0; held_cpl=0;
            for(p=0;p<2;p=p+1) begin issued[p]=0; finished[p]=0; allowed[p]=0; end
        end else begin
            was_running=1;
            if((stop || fault) && cmd_valid && cmd_ready) $fatal(1,"MEMORY_CLOSED_ADMISSION");
            if(cmd_valid && cmd_ready) begin
                t=cmd_tag;
                if(t<1 || t>=256 || state[t]!=0 || fill_tag!=0) $fatal(1,"MEMORY_COMMAND_IDENTITY");
                beats[t]=cmd_len+1; address[t]=cmd_addr; destination[t]=cmd_channel;
                response[t]=next_status; state[t]=1; fill_tag=t; fill_beat=0;
                filled_at[t]=0; aw_at[t]=0; last_w_at[t]=0; b_at[t]=0;
                input_count[t]=0; output_count[t]=0; eligible[t]=-1;
                completion_queue[cmd_channel*256+issued[cmd_channel]]=t;
                issued[cmd_channel]=issued[cmd_channel]+1; accepted=accepted+1;
                if(cmd_addr<CAPACITY && {1'b0,cmd_addr}+65'((cmd_len+1)*64)<=CAPACITY) begin
                    memory_queue[cmd_channel*256+allowed[cmd_channel]]=t;
                    allowed[cmd_channel]=allowed[cmd_channel]+1;
                end
                $display("MEM CMD %0d %0d %0d %h %0d %0d",ps(),cmd_channel,t,cmd_addr,cmd_len+1,next_status);
            end
            if(data_valid && data_ready) begin
                if(fill_tag==0 || data!==pattern(fill_tag,fill_beat)) $fatal(1,"MEMORY_INPUT_BYTES");
                $display("MEM IN %0d %0d %0d %h",ps(),fill_tag,fill_beat,data);
                input_count[fill_tag]=input_count[fill_tag]+1;
                fill_beat=fill_beat+1;
                if(fill_beat==beats[fill_tag]) begin filled_at[fill_tag]=ps(); fill_tag=0; end
            end
            for(p=0;p<2;p=p+1) begin
                if(pending_counts[p*5+:5]>4) $fatal(1,"MEMORY_PENDING_OVERFLOW");
                if(held_cpl[p] && (!cpl_valid[p] || {cpl_tag[p*64+:64],cpl_status[p*2+:2]}!==previous_cpl[p]))
                    $fatal(1,"MEMORY_COMPLETION_STABILITY");
                held_cpl[p]=cpl_valid[p] && !cpl_ready[p];
                previous_cpl[p]={cpl_tag[p*64+:64],cpl_status[p*2+:2]};
                if(cpl_valid[p] && cpl_ready[p]) begin
                    t=completion_queue[p*256+finished[p]];
                    if(finished[p]>=issued[p] || state[t]!=1 || cpl_tag[p*64+:64]!==64'(t) ||
                       cpl_status[p*2+:2]!==2'(response[t])) $fatal(1,"MEMORY_COMPLETION_STATUS_ORDER");
                    if(response[t]!=3 && (b_at[t]==0 || b_at[t]>=ps() || output_count[t]!=beats[t]))
                        $fatal(1,"MEMORY_EARLY_SUCCESS");
                    if(response[t]==3 && (aw_at[t]!=0 || output_count[t]!=0)) $fatal(1,"MEMORY_RANGE_LEAK");
                    state[t]=2; finished[p]=finished[p]+1; completed=completed+1;
                    $display("MEM CPL %0d %0d %0d %0d",ps(),p,t,cpl_status[p*2+:2]);
                end
            end
        end
    end

    genvar g;
    generate for(g=0;g<2;g=g+1) begin: target
        integer tag,index;
        reg held_aw=0,held_data=0;
        reg [57:0] previous_aw;
        reg [576:0] previous_w;
        always @(negedge mem_clk[g]) begin
            m_awready[g]=mem_rst_n[g] && !hold_aw[g] && memory_cycle[g]%4!=0;
            m_wready[g]=mem_rst_n[g] && !hold_w[g] && memory_cycle[g]%7!=0;
            // Once offered, B stays stable until accepted, even if hold_b rises.
            if(!mem_rst_n[g]) m_bvalid[g]=0;
            else if(!m_bvalid[g]) begin
                tag=memory_queue[g*256+memory_b[g]];
                m_bvalid[g]=!hold_b[g] && memory_b[g]<memory_aw[g] && memory_b[g]<memory_w[g] &&
                            eligible[tag]>=0 && memory_cycle[g]>=eligible[tag];
            end
            m_bid[g*4+:4]=0;
            m_bresp[g*2+:2]=2'(response[memory_queue[g*256+memory_b[g]]]);
        end
        always @(posedge mem_clk[g]) begin
            if(!mem_rst_n[g]) begin
                memory_aw[g]=0; memory_w[g]=0; memory_b[g]=0; memory_beat[g]=0;
                memory_cycle[g]=0; b_due[g]=-1; held_aw=0; held_data=0;
            end else begin
                memory_cycle[g]=memory_cycle[g]+1;
                if(held_aw && (!m_awvalid[g] ||
                   {m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]}!==previous_aw))
                    $fatal(1,"MEMORY_AW_STABILITY");
                if(held_data && (!m_wvalid[g] || {m_wlast[g],m_wstrb[g*64+:64],m_wdata[g*512+:512]}!==previous_w))
                    $fatal(1,"MEMORY_W_STABILITY");
                held_aw=m_awvalid[g] && !m_awready[g];
                held_data=m_wvalid[g] && !m_wready[g];
                previous_aw={m_awaddr[g*37+:37],m_awlen[g*8+:8],m_awid[g*4+:4],m_awsize[g*3+:3],m_awburst[g*2+:2],m_awcache[g*4+:4]};
                previous_w={m_wlast[g],m_wstrb[g*64+:64],m_wdata[g*512+:512]};
                if(held_aw) aw_stalls[g]=aw_stalls[g]+1;
                if(held_data) w_stalls[g]=w_stalls[g]+1;
                if(m_awvalid[g] && m_awready[g]) begin
                    if(memory_aw[g]>=allowed[g]) $fatal(1,"MEMORY_UNAUTHORIZED_AW");
                    tag=memory_queue[g*256+memory_aw[g]];
                    if(64'(m_awaddr[g*37+:37])!==address[tag] || int'(m_awlen[g*8+:8])+1!=beats[tag] ||
                       m_awid[g*4+:4]!=0 || m_awsize[g*3+:3]!=6 || m_awburst[g*2+:2]!=1 || m_awcache[g*4+:4]!=0)
                        $fatal(1,"MEMORY_ADDRESS_ALIAS_OR_ATTRIBUTES");
                    aw_at[tag]=ps(); memory_aw[g]=memory_aw[g]+1;
                    $display("MEM AW %0d %0d %0d %h %0d",ps(),g,tag,m_awaddr[g*37+:37],m_awlen[g*8+:8]+1);
                end
                if(m_wvalid[g] && m_wready[g]) begin
                    if(memory_w[g]>=allowed[g]) $fatal(1,"MEMORY_UNAUTHORIZED_W");
                    tag=memory_queue[g*256+memory_w[g]];
                    if(filled_at[tag]==0 || filled_at[tag]>=ps() || m_wdata[g*512+:512]!==pattern(tag,memory_beat[g]) ||
                       m_wstrb[g*64+:64]!==64'hffffffffffffffff) $fatal(1,"MEMORY_PAYLOAD_MISMATCH");
                    if(m_wlast[g] !== (memory_beat[g]==beats[tag]-1)) $fatal(1,"MEMORY_WLAST_MISMATCH");
                    $display("MEM W %0d %0d %0d %0d %0d %h",ps(),g,tag,memory_beat[g],m_wlast[g],m_wdata[g*512+:512]);
                    memory_beat[g]=memory_beat[g]+1; output_count[tag]=output_count[tag]+1;
                    w_total[g]=w_total[g]+1; transferred=transferred+64;
                    if(m_wlast[g]) begin last_w_at[tag]=ps(); memory_w[g]=memory_w[g]+1; memory_beat[g]=0; end
                end
                // Every completed AW/W pair earns its own B deadline, independently
                // of older responses. Identical target for all four guard variants.
                for(index=memory_b[g];index<memory_aw[g] && index<memory_w[g];index=index+1) begin
                    tag=memory_queue[g*256+index];
                    if(eligible[tag]<0) eligible[tag]=memory_cycle[g]+delay_b+1;
                end
                if(memory_aw[g]-memory_b[g]>(GUARD_SLOTS==0 ? 1:GUARD_SLOTS))
                    $fatal(1,"CREDIT_DESTINATION_OVERFLOW");
                if(m_bvalid[g] && m_bready[g]) begin
                    tag=memory_queue[g*256+memory_b[g]];
                    if(aw_at[tag]==0 || last_w_at[tag]==0 || ps()<=aw_at[tag] || ps()<=last_w_at[tag]) $fatal(1,"MEMORY_B_BEFORE_WRITE");
                    b_at[tag]=ps(); memory_b[g]=memory_b[g]+1; b_due[g]=-1;
                    m_bvalid[g]<=0;
                    $display("MEM B %0d %0d %0d %0d",ps(),g,tag,response[tag]);
                end
            end
        end
    end endgenerate

    task automatic send_job(input integer port,count,input [63:0] addr,input integer status);
        integer tag;
        begin
            tag=next_tag; next_tag=next_tag+1;
            @(negedge clk); cmd_valid=1; cmd_channel=port; cmd_addr=addr; cmd_tag=tag; cmd_len=count-1; next_status=status;
            do @(posedge clk); while(!cmd_ready);
            @(negedge clk); cmd_valid=0;
            for(integer beat=0;beat<count;beat=beat+1) begin
                data_valid=1; data=pattern(tag,beat);
                do @(posedge clk); while(!data_ready);
                @(negedge clk);
            end
            data_valid=0;
        end
    endtask
    task automatic regular(input integer port,count);
        send_job(port,count,64'h100000000+(64'(next_tag)<<12),0);
    endtask
    task automatic drain_all;
        integer waits;
        begin
            waits=0;
            do begin @(negedge clk); waits=waits+1; if(waits>12000) $fatal(1,"MEMORY_DRAIN_WATCHDOG"); end while(!drained);
            for(integer port=0;port<2;port=port+1)
                if(finished[port]!=issued[port] || memory_b[port]!=allowed[port]) $fatal(1,"MEMORY_PREMATURE_DRAIN");
            $display("MEM DRAIN %0d",ps());
        end
    endtask
    task automatic probe_closed;
        begin
            @(negedge clk); cmd_valid=1; cmd_channel=0; cmd_tag=next_tag; cmd_addr=64'h100000000; cmd_len=0;
            repeat(16) begin @(posedge clk); if(cmd_ready) $fatal(1,"MEMORY_CLOSED_PROBE"); end
            @(negedge clk); cmd_valid=0;
        end
    endtask
    task automatic coordinated_reset;
        begin
            @(negedge clk); cmd_valid=0; data_valid=0; stop=0;
            $display("MEM RESET %0d",ps()); resets=resets+1;
            fork
                begin rst_n=0; repeat(64) @(negedge clk); rst_n=1; end
                begin @(negedge mem_clk[0]); mem_rst_n[0]=0; repeat(64) @(negedge mem_clk[0]); mem_rst_n[0]=1; end
                begin @(negedge mem_clk[1]); mem_rst_n[1]=0; repeat(64) @(negedge mem_clk[1]); mem_rst_n[1]=1; end
            join
            hold_b=0; hold_cpl=0;
            repeat(64) @(negedge clk);
            if(!drained || fault || cpl_valid!=0 || m_awvalid!=0 || m_wvalid!=0) $fatal(1,"MEMORY_RESET_STALE");
        end
    endtask

    initial begin
        if(!$value$plusargs("PROFILE=%d",profile)) profile=0;
        if(!$value$plusargs("NEGATIVE=%d",negative)) negative=0;
        case(profile)
            0: begin half0=2000; half1=3650; delay_b=3; end
            1: begin half0=5750; half1=2150; delay_b=32; end
            2: begin half0=2000; half1=3650; delay_b=160; end
            default: $fatal(1,"MEMORY_PROFILE");
        endcase
        for(i=0;i<256;i=i+1) begin state[i]=0; response[i]=0; eligible[i]=-1; end
        for(i=0;i<512;i=i+1) begin memory_queue[i]=0; completion_queue[i]=0; end
        for(i=0;i<2;i=i+1) begin w_total[i]=0; aw_stalls[i]=0; w_stalls[i]=0; end
        clocks_ready=1;
        $display("MEM CONFIG profile=%0d source_ps=5000 memory0_ps=%0d memory1_ps=%0d delay_b=%0d negative=%0d",profile,2*half0,2*half1,delay_b,negative);
        $display("CREDIT CONFIG slots=%0d",GUARD_SLOTS);
        coordinated_reset;
        if(negative==1) force dut.u_wdata=1024'b0;

        $display("MEM CASE bulk");
        rate_start=ps(); rate_bytes=transferred;
        for(i=0;i<32;i=i+1) regular(i%2,64);
        drain_all;
        rate_end=ps(); rate_bytes=transferred-rate_bytes;
        $display("MEM RATE bytes=%0d elapsed_ps=%0d",rate_bytes,rate_end-rate_start);

        $display("MEM CASE credit_saturation");
        hold_b=3;
        for(i=0;i<8;i=i+1) regular(i%2,1);
        repeat(1500) @(negedge clk);
        for(i=0;i<2;i=i+1)
            if(memory_aw[i]-memory_b[i]!=(GUARD_SLOTS==0 ? 1:GUARD_SLOTS))
                $fatal(1,"CREDIT_SATURATION_NOT_REACHED");
        $display("CREDIT SATURATED slots=%0d pending0=%0d pending1=%0d",GUARD_SLOTS,
                 memory_aw[0]-memory_b[0],memory_aw[1]-memory_b[1]);
        stop=1; probe_closed; hold_b=0; drain_all; stop=0;

        $display("MEM CASE isolation");
        hold_w[0]=1;
        for(i=0;i<6;i=i+1) regular(0,1);
        baseline=finished[1]; regular(1,3); regular(1,17);
        repeat(1500) @(negedge clk);
        if(finished[1]!=baseline+2 || drained) $fatal(1,"MEMORY_PORT_ISOLATION");
        hold_w[0]=0; drain_all;

        $display("MEM CASE stop_fill");
        fork
            regular(0,64);
            begin wait(data_valid && data_ready); @(negedge clk); stop=1; end
        join
        probe_closed; drain_all; stop=0;

        $display("MEM CASE stop_after_aw");
        hold_w[0]=1; baseline=memory_aw[0]; regular(0,64);
        wait(memory_aw[0]>baseline); @(negedge clk); stop=1;
        probe_closed;
        if(drained) $fatal(1,"MEMORY_FALSE_DRAIN_STOP");
        hold_w[0]=0; drain_all; stop=0;

        $display("MEM CASE address_pause");
        hold_aw[0]=1; baseline=w_total[0]; regular(0,3);
        repeat(200) @(negedge clk);
        if(w_total[0]!=baseline+3 || drained) $fatal(1,"MEMORY_W_BEFORE_AW");
        hold_aw[0]=0; drain_all;

        $display("MEM CASE clock_pause");
        hold_b[0]=1; regular(0,4); wait(memory_aw[0]>memory_b[0]);
        @(negedge mem_clk[0]); run_memory[0]=0;
        baseline=finished[1]; regular(1,4);
        repeat(1500) @(negedge clk);
        if(finished[1]!=baseline+1 || drained) $fatal(1,"MEMORY_CLOCK_PAUSE_ISOLATION");
        run_memory[0]=1; hold_b[0]=0; drain_all;

        $display("MEM CASE result_credit");
        hold_cpl=1; regular(0,1); regular(1,1);
        wait(&cpl_valid); repeat(40) @(negedge clk);
        if(drained) $fatal(1,"MEMORY_DRAIN_BEFORE_COMPLETION");
        hold_cpl=0; drain_all;

        $display("MEM CASE address_limits");
        send_job(0,1,64'd0,0); send_job(1,64,CAPACITY-4096,0); drain_all;

        $display("MEM CASE response_error");
        if(negative==2) force dut.u_bresp=4'b0;
        send_job(0,3,64'h100000000+(64'(next_tag)<<12),2);
        wait(fault); probe_closed; drain_all; coordinated_reset;

        $display("MEM CASE mixed_reject");
        hold_b[0]=1;
        regular(0,1); send_job(0,2,CAPACITY,3); regular(0,3);
        stop=1; probe_closed; hold_b[0]=0;
        wait(fault); drain_all; coordinated_reset;

        $display("MEM CASE mixed_error");
        hold_b[0]=1;
        regular(0,3); send_job(0,2,64'h100000000+(64'(next_tag)<<12),2); regular(0,1);
        stop=1; probe_closed; hold_b[0]=0;
        wait(fault); drain_all; coordinated_reset;

        $display("MEM CASE capacity_reject");
        if(negative==3) force dut.memory_port[0].port.d_awaddr=64'b0;
        send_job(0,1,CAPACITY,3); wait(fault); probe_closed; drain_all; coordinated_reset;
        send_job(1,64,CAPACITY+64'h100000000,3); wait(fault); probe_closed; drain_all; coordinated_reset;
        send_job(0,1,64'hffffffffffffffc0,3); wait(fault); probe_closed; drain_all; coordinated_reset;

        $display("MEM CASE reset_busy");
        hold_b[0]=1; regular(0,4); wait(memory_w[0]>memory_b[0]);
        if(drained) $fatal(1,"MEMORY_RESET_CASE_NOT_BUSY");
        abort_before=aborted; coordinated_reset;
        if(aborted!=abort_before+1) $fatal(1,"MEMORY_RESET_ABORT_ACCOUNTING");
        regular(0,3); regular(1,2); drain_all;
        if(accepted!=completed+aborted || aborted!=1 || aw_stalls[0]==0 || w_stalls[0]==0)
            $fatal(1,"MEMORY_COVERAGE");
        $display("MEMORY_SUITE_PASS profile=%0d accepted=%0d completed=%0d aborted=%0d bytes=%0d resets=%0d aw_stall=%0d,%0d w_stall=%0d,%0d",
                 profile,accepted,completed,aborted,transferred,resets,aw_stalls[0],aw_stalls[1],w_stalls[0],w_stalls[1]);
        $finish;
    end
    initial begin #500000; $fatal(1,"MEMORY_GLOBAL_WATCHDOG"); end
endmodule
