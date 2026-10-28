`timescale 1ns / 1ps
module tb_ntrpu_rv_elastic;
    parameter integer WIDTH=577;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0, s_valid=0, m_ready=0;
    reg [WIDTH-1:0] s_data=0;
    wire s_ready, m_valid, empty;
    wire [WIDTH-1:0] m_data;
    ntrpu_rv_elastic #(.WIDTH(WIDTH)) dut(.*);
    reg [WIDTH-1:0] expected[0:8191], held_data;
    integer head=0, tail=0, sent=0, received=0, simultaneous=0;
    integer full_cycles=0, stalls=0, resets=0, i, seed;
    reg held=0, prior_ready, accepted=0;
    reg [31:0] rng=1;
    function [31:0] random_word;
        begin rng=rng*32'd1664525+32'd1013904223; random_word=rng; end
    endfunction
    function [WIDTH-1:0] payload;
        input integer n;
        integer b;
        begin for(b=0;b<WIDTH;b=b+1) payload[b]=((n*32'h9e3779b9)>>(b%32)) ^ (b/32); end
    endfunction
    always @(posedge clk) begin
        accepted=s_valid && s_ready;
        if(!rst_n) begin head=0; tail=0; held=0; resets=resets+1; end
        else begin
            if(held && (!m_valid || m_data!==held_data)) $fatal(1,"held payload changed");
            if(empty !== (head==tail)) $fatal(1,"empty mismatch");
            if(m_valid && m_ready) begin
                if(head>=tail || m_data!==expected[head]) $fatal(1,"queue order/data mismatch");
                head=head+1; received=received+1;
            end
            if(s_valid && s_ready) begin expected[tail]=s_data; tail=tail+1; sent=sent+1; end
            if(tail-head>2) $fatal(1,"capacity exceeded");
            if(s_valid && s_ready && m_valid && m_ready) simultaneous=simultaneous+1;
            if(!s_ready) full_cycles=full_cycles+1;
            held=m_valid && !m_ready; held_data=m_data;
            if(held) stalls=stalls+1;
        end
    end
    task step;
        input integer mode;
        begin
            @(negedge clk);
            // A producer holds its item when blocked.
            if(!s_valid || accepted) begin
                s_valid=mode==0 ? 0 : (mode==1 ? 1 : ((random_word()>>16)&1));
                s_data=payload(sent);
            end
            m_ready=mode==1 || (mode==2 && ((random_word()>>16)&3)!=0);
        end
    endtask
    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        rng=seed;
        repeat(3) @(negedge clk); rst_n=1;
        // Fill both slots then toggle READY away from an active clock edge.
        s_valid=1; s_data=payload(sent);
        @(negedge clk); s_data=payload(sent);
        @(negedge clk); s_data=payload(sent);
        #1; prior_ready=s_ready; m_ready=1; #1;
        if(s_ready!==prior_ready) $fatal(1,"combinational ready dependency");
        for(i=0;i<300;i=i+1) begin
            step(1);
            if(!m_valid) $fatal(1,"steady stream bubble");
        end
        for(i=0;i<1500;i=i+1) step(2);
        @(negedge clk); s_valid=0; m_ready=1;
        repeat(4) @(negedge clk);
        if(!empty || head!=tail) $fatal(1,"drain mismatch");
        // Reset with both entries occupied; no old payload may reappear.
        m_ready=0; s_valid=1; s_data=payload(sent);
        @(negedge clk); s_data=payload(sent);
        @(negedge clk); rst_n=0; s_valid=0;
        repeat(2) @(negedge clk);
        if(m_valid || s_ready || !empty) $fatal(1,"reset did not flush");
        rst_n=1;
        for(i=0;i<80;i=i+1) step(1);
        @(negedge clk); s_valid=0; m_ready=1;
        repeat(4) @(negedge clk);
        if(!empty || head!=tail || simultaneous<300 || stalls==0 || full_cycles==0)
            $fatal(1,"unit coverage missing");
        $display("ELASTIC_UNIT_PASS width=%0d seed=%0d sent=%0d received=%0d simultaneous=%0d full=%0d stalls=%0d resets=%0d",
                 WIDTH,seed,sent,received,simultaneous,full_cycles,stalls,resets);
        $finish;
    end
    initial begin #100000; $fatal(1,"unit watchdog"); end
endmodule
