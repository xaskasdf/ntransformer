`timescale 1ns / 1ps
module tb_ntrpu_peer_multibuffer;
    parameter integer CHANNELS=2, RESERVE_BANKS=4, BANK_BEATS=64, DESCRIPTORS=16, MAX_OUTSTANDING=4;
    localparam integer BANKS=2*CHANNELS+RESERVE_BANKS;
    localparam integer CHANNEL_BITS=CHANNELS<=1 ? 1 : $clog2(CHANNELS);
    localparam integer COUNT_BITS=$clog2(DESCRIPTORS+1);
    localparam [63:0] BASE=64'h0000000100000000;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0, stop=0, cmd_valid=0, data_valid=0;
    wire cmd_ready, data_ready;
    reg [CHANNEL_BITS-1:0] cmd_channel=0;
    reg [63:0] cmd_addr=0, cmd_tag=0;
    reg [7:0] cmd_len=0;
    reg [511:0] data=0;
    wire [CHANNELS-1:0] m_awvalid,m_wvalid,m_wlast,m_bready,cpl_valid;
    reg [CHANNELS-1:0] m_awready=0,m_wready=0,m_bvalid=0,cpl_ready=0;
    wire [CHANNELS*64-1:0] m_awaddr,m_wstrb,cpl_tag;
    wire [CHANNELS*8-1:0] m_awlen;
    wire [CHANNELS*4-1:0] m_awid,m_awcache;
    wire [CHANNELS*3-1:0] m_awsize;
    wire [CHANNELS*2-1:0] m_awburst,cpl_status;
    wire [CHANNELS*512-1:0] m_wdata;
    reg [CHANNELS*4-1:0] m_bid=0;
    reg [CHANNELS*2-1:0] m_bresp=0;
    wire fault,drained;
    wire [BANKS-1:0] banks_used;
    wire [CHANNELS*COUNT_BITS-1:0] pending_counts;
    ntrpu_peer_multibuffer #(.CHANNELS(CHANNELS),.RESERVE_BANKS(RESERVE_BANKS),
        .BANK_BEATS(BANK_BEATS),.DESCRIPTORS(DESCRIPTORS),.MAX_OUTSTANDING(MAX_OUTSTANDING)) dut(.*);

    integer cycle=0, seed, p, n, k, q, before_other, peak_pending=0;
    integer aw_tag[0:CHANNELS-1], w_count[0:CHANNELS-1], w_total[0:CHANNELS-1];
    integer response_head[0:CHANNELS-1], response_tail[0:CHANNELS-1];
    integer response_tag[0:CHANNELS*512-1], response_due[0:CHANNELS*512-1];
    reg [CHANNELS-1:0] held_aw=0, held_w=0, held_cpl=0;
    reg [63:0] held_addr[0:CHANNELS-1],held_tag[0:CHANNELS-1];
    reg [7:0] held_len[0:CHANNELS-1];
    reg [511:0] held_data[0:CHANNELS-1];
    reg held_last[0:CHANNELS-1];
    reg [1:0] held_status[0:CHANNELS-1];
    reg hold_aw0=0, hold_w0=0, hold_all_w=0, force_w=0, hold_results=0, foreign_pending=0;
    integer error_tag=-1;
    reg [63:0] next_tag=1;
    reg [31:0] rng;
    function [31:0] random_word;
        begin rng=rng*32'd1664525+32'd1013904223; random_word=rng; end
    endfunction
    function [511:0] pattern;
        input [63:0] identity;
        input integer beat;
        integer lane;
        begin
            for(lane=0;lane<16;lane=lane+1)
                pattern[lane*32 +:32]=(identity*32'h9e3779b9) ^ (beat*32'h10203) ^ (lane*32'h01010101);
        end
    endfunction

    // Target BFM: in-order responses delayed 160+ cycles AFTER accepted WLAST.
    always @(negedge clk) begin
        for (n=0;n<CHANNELS;n=n+1) begin
            m_awready[n]=((random_word()>>16)&3)!=0 && !(n==0 && hold_aw0);
            m_wready[n]=(force_w || ((random_word()>>16)&3)!=0) && !(n==0 && hold_w0) && !hold_all_w;
            cpl_ready[n]=((random_word()>>16)&3)!=0 && !hold_results;
            m_bvalid[n]=rst_n && response_head[n]<response_tail[n] &&
                cycle>=response_due[n*512+response_head[n]];
            m_bid[n*4 +:4]=foreign_pending ? 7 : 0;
            m_bresp[n*2 +:2]=(response_tag[n*512+response_head[n]]==error_tag) ? 2 : 0;
        end
    end

    always @(posedge clk) begin
        cycle=cycle+1;
        if (!rst_n) begin
            held_aw=0; held_w=0; held_cpl=0;
            for(p=0;p<CHANNELS;p=p+1) begin
                aw_tag[p]=-1; w_count[p]=0;
                response_head[p]=0; response_tail[p]=0;
            end
        end else begin
            if(cmd_valid && cmd_ready)
                $display("TRACE CMD %0d %0d %0d %h %0d %0d %0d",cycle,cmd_channel,cmd_tag,cmd_addr,cmd_len+1,dut.alloc_bank,dut.alloc_generation);
            if(data_valid && data_ready) $display("TRACE IN %0d %h",cycle,data);
            if(stop && cmd_valid && cmd_ready) $fatal(1,"admission during stop");
            if(fault && cmd_valid && cmd_ready) $fatal(1,"admission during fault");
            if(data_valid && data_ready && dut.ram_read_enable[dut.fill_bank])
                $fatal(1,"simultaneous read and fill of the same bank");
            for(p=0;p<CHANNELS;p=p+1) begin
                if(held_aw[p] && (!m_awvalid[p] || m_awaddr[p*64 +:64]!==held_addr[p] || m_awlen[p*8 +:8]!==held_len[p]))
                    $fatal(1,"AW changed under backpressure");
                if(held_w[p] && (!m_wvalid[p] || m_wdata[p*512 +:512]!==held_data[p] || m_wlast[p]!==held_last[p]))
                    $fatal(1,"W changed under backpressure");
                if(held_cpl[p] && (!cpl_valid[p] || cpl_tag[p*64 +:64]!==held_tag[p] || cpl_status[p*2 +:2]!==held_status[p]))
                    $fatal(1,"completion changed under backpressure");
                held_aw[p]=m_awvalid[p] && !m_awready[p];
                held_addr[p]=m_awaddr[p*64 +:64]; held_len[p]=m_awlen[p*8 +:8];
                held_w[p]=m_wvalid[p] && !m_wready[p];
                held_data[p]=m_wdata[p*512 +:512]; held_last[p]=m_wlast[p];
                held_cpl[p]=cpl_valid[p] && !cpl_ready[p];
                held_tag[p]=cpl_tag[p*64 +:64]; held_status[p]=cpl_status[p*2 +:2];
                if(pending_counts[p*COUNT_BITS +:COUNT_BITS]>MAX_OUTSTANDING)
                    $fatal(1,"outstanding overflow");
                if(pending_counts[p*COUNT_BITS +:COUNT_BITS]>peak_pending)
                    peak_pending=pending_counts[p*COUNT_BITS +:COUNT_BITS];
                if(dut.fifo_count[p]>2 || int'(dut.fifo_count[p])+int'(dut.read_pending[p])>2)
                    $fatal(1,"read pipeline credit overflow");
                if(dut.occupied[p]>DESCRIPTORS) $fatal(1,"descriptor overflow");
                if(dut.read_issue[p] && (!banks_used[dut.bank[p*DESCRIPTORS+dut.send_head[p]]] ||
                    dut.allocator.owner[dut.bank[p*DESCRIPTORS+dut.send_head[p]]] != p))
                    $fatal(1,"RAM read without ownership");
                if(m_awvalid[p] && m_awready[p]) begin
                    if(aw_tag[p]!=-1) $fatal(1,"interleaved AW before prior WLAST");
                    if(m_awsize[p*3 +:3]!=6 || m_awburst[p*2 +:2]!=1 || m_awid[p*4 +:4]!=0 || m_awcache[p*4 +:4]!=0)
                        $fatal(1,"AXI attributes");
                    aw_tag[p]=(m_awaddr[p*64 +:64]-BASE)>>12;
                    w_count[p]=0;
                    $display("TRACE AW %0d %0d %h %0d",cycle,p,m_awaddr[p*64 +:64],m_awlen[p*8 +:8]+1);
                end
                if(m_wvalid[p] && m_wready[p]) begin
                    if(aw_tag[p]<0 || m_wstrb[p*64 +:64]!==64'hffffffffffffffff)
                        $fatal(1,"W without AW or partial strobe");
                    if(m_wdata[p*512 +:512]!==pattern(aw_tag[p],w_count[p]))
                        $fatal(1,"payload mismatch");
                    $display("TRACE W %0d %0d %0d %h",cycle,p,m_wlast[p],m_wdata[p*512 +:512]);
                    w_count[p]=w_count[p]+1; w_total[p]=w_total[p]+1;
                    if(m_wlast[p]) begin
                        q=p*512+response_tail[p];
                        response_tag[q]=aw_tag[p];
                        response_due[q]=cycle+160+(aw_tag[p]%23);
                        response_tail[p]=response_tail[p]+1;
                        aw_tag[p]=-1;
                    end
                end
                if(m_bvalid[p] && m_bready[p]) begin
                    $display("TRACE B %0d %0d %0d %0d",cycle,p,m_bid[p*4 +:4],m_bresp[p*2 +:2]);
                    if(m_bid[p*4 +:4]==0) response_head[p]=response_head[p]+1;
                    else foreign_pending=0;
                end
                if(cpl_valid[p] && cpl_ready[p])
                    $display("TRACE CPL %0d %0d %0d %0d",cycle,p,cpl_tag[p*64 +:64],cpl_status[p*2 +:2]);
            end
        end
    end

    task send_job;
        input integer destination, beats;
        integer beat;
        reg [63:0] identity;
        begin
            identity=next_tag; next_tag=next_tag+1;
            @(negedge clk);
            cmd_valid=1; cmd_channel=destination; cmd_tag=identity;
            cmd_addr=BASE+(identity<<12); cmd_len=beats-1;
            do @(posedge clk); while(!cmd_ready);
            @(negedge clk); cmd_valid=0;
            for(beat=0;beat<beats;beat=beat+1) begin
                if((identity+beat)%7==0) begin data_valid=0; @(negedge clk); end
                data_valid=1; data=pattern(identity,beat);
                do @(posedge clk); while(!data_ready);
                @(negedge clk);
            end
            data_valid=0;
        end
    endtask
    task drain_all;
        begin
            do @(negedge clk); while(!drained);
            for(k=0;k<CHANNELS;k=k+1)
                if(response_head[k]!=response_tail[k] || aw_tag[k]!=-1) $fatal(1,"premature drained");
            $display("TRACE DRAIN %0d",cycle);
        end
    endtask
    task cold_reset;
        begin
            drain_all;
            @(negedge clk); rst_n=0; stop=0; cmd_valid=0; data_valid=0;
            $display("TRACE RESET %0d",cycle);
            repeat(3) @(negedge clk);
            rst_n=1; error_tag=-1; foreign_pending=0;
        end
    endtask
    task probe_closed_admission;
        begin
            @(negedge clk);
            cmd_valid=1; cmd_channel=0; cmd_addr=BASE+(next_tag<<12); cmd_tag=next_tag; cmd_len=0;
            repeat(8) begin @(posedge clk); if(cmd_ready) $fatal(1,"closed admission accepted a command"); end
            @(negedge clk); cmd_valid=0;
        end
    endtask
    task check_bad_command;
        begin
            cmd_valid=1;
            repeat(3) begin @(posedge clk); if(cmd_ready) $fatal(1,"bad descriptor admitted"); end
            @(negedge clk); cmd_valid=0;
            if(!fault || !drained) $fatal(1,"bad descriptor handling");
        end
    endtask

    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        rng=seed;
        for(k=0;k<CHANNELS;k=k+1) begin
            response_head[k]=0; response_tail[k]=0; aw_tag[k]=-1; w_total[k]=0;
        end
        for(k=0;k<CHANNELS*512;k=k+1) begin response_tag[k]=-1; response_due[k]=0; end
        repeat(4) @(negedge clk); rst_n=1;
        $display("CASE traffic_and_isolation");
        if(CHANNELS>1) begin
            hold_w0=1;
            for(k=0;k<2+RESERVE_BANKS;k=k+1) send_job(0,1);
            before_other=w_total[1];
            send_job(1,1); send_job(1,1);
            repeat(40) @(negedge clk);
            if(w_total[1]<=before_other) $fatal(1,"other destination did not progress");
            hold_w0=0;
        end
        for(k=0;k<48;k=k+1) send_job(k%CHANNELS, k%3==0 ? BANK_BEATS : (k%3==1 ? 1 : (BANK_BEATS<4 ? BANK_BEATS : 4)));
        drain_all;
        $display("CASE parallel_ports");
        hold_all_w=1;
        for(k=0;k<CHANNELS;k=k+1) send_job(k,1);
        wait(&m_wvalid);
        @(negedge clk); force_w=1; hold_all_w=0;
        repeat(4) @(negedge clk);
        force_w=0;
        drain_all;
        $display("CASE aw_pause_and_stop");
        hold_aw0=1;
        send_job(0,BANK_BEATS);
        wait(m_awvalid[0]);
        @(negedge clk); stop=1;
        probe_closed_admission;
        hold_aw0=0;
        repeat(12) @(negedge clk);
        stop=0;
        drain_all;
        $display("CASE stop_during_fill");
        fork
            send_job(0,BANK_BEATS);
            begin
                wait(data_valid && data_ready);
                @(negedge clk); stop=1;
                repeat(16) @(negedge clk);
                stop=0;
            end
        join
        drain_all;
        $display("CASE result_credit");
        hold_results=1;
        for(k=0;k<DESCRIPTORS;k=k+1) send_job(0,1);
        fork
            send_job(0,1);
            begin
                wait(cmd_valid);
                repeat(20) begin @(posedge clk); if(cmd_ready) $fatal(1,"result credit not reserved"); end
                @(negedge clk); hold_results=0;
            end
        join
        drain_all;
        $display("CASE error_response");
        cold_reset;
        error_tag=next_tag;
        send_job(0,1); send_job(0,1);
        wait(fault); probe_closed_admission; drain_all;
        $display("CASE foreign_response");
        cold_reset;
        foreign_pending=1;
        send_job(0,1); send_job(0,1);
        wait(fault); probe_closed_admission; drain_all;
        $display("CASE invalid_descriptor");
        cold_reset;
        @(negedge clk); cmd_channel=0; cmd_addr=BASE+1; cmd_len=0;
        check_bad_command;
        cold_reset;
        @(negedge clk); cmd_addr=BASE; cmd_len=BANK_BEATS;
        check_bad_command;
        cold_reset;
        if(BANK_BEATS>=2) begin
            @(negedge clk); cmd_addr=BASE+4032; cmd_len=1;
            check_bad_command;
            cold_reset;
        end
        if((1<<CHANNEL_BITS)>CHANNELS) begin
            @(negedge clk); cmd_addr=BASE; cmd_len=0; cmd_channel=CHANNELS;
            check_bad_command;
        end
        cold_reset;
        if(MAX_OUTSTANDING>1 && peak_pending<2) $fatal(1,"no concurrent writes exercised");
        $display("PASS seed=%0d cycles=%0d peak_pending=%0d",seed,cycle,peak_pending);
        $finish;
    end
    initial begin #5000000; $fatal(1,"watchdog"); end
endmodule
