`timescale 1ns/1ps
module tb_ntrpu_peer_read_domain;
    reg clk=0,rst_n=0,s_valid=0,m_ready=0,complete=0;
    reg [63:0] address=0;
    wire s_ready,m_valid,pending,fault;
    integer cycle=0;
    always #2.5 clk=~clk;
    ntrpu_peer_read_domain dut(.*);
    task automatic step(input bit v,r,c,input logic[63:0] a);
        @(negedge clk);#0.001;s_valid=v;m_ready=r;complete=c;address=a;
        @(posedge clk);#0.001;
    endtask
    // External interfaces only: no observation of the implementation counter.
    reg held=0;
    reg [63:0] held_address;
    always @(posedge clk)begin
        if(cycle>0)$display("DM SAMPLE %0d %b %b %b %b %0d %b %b %b %b",cycle,
            rst_n,s_valid,m_ready,complete,address,s_ready,m_valid,pending,fault);
        if(rst_n)begin
            if(held && (!m_valid || address!=held_address))$fatal(1,"DM_FAIL offered command changed");
            held=m_valid&&!m_ready;held_address=address;
        end else held=0;
        cycle++;
    end
    initial begin
        repeat(3)@(negedge clk);#0.001;rst_n=1;
        // Every physical domain: two concurrent commands, a queued change,
        // drain one then both, and hold the newly offered command downstream.
        for(integer pc=0;pc<32;pc++)begin
            step(1,1,0,64'(pc)<<29);
            step(1,1,0,(64'(pc)<<29)+4096);
            repeat(4)step(1,1,0,64'((pc+1)%32)<<29);
            step(1,1,1,64'((pc+1)%32)<<29);
            repeat(3)step(1,1,0,64'((pc+1)%32)<<29);
            step(1,0,1,64'((pc+1)%32)<<29);
            repeat(4)step(1,0,0,64'((pc+1)%32)<<29);
            step(1,1,0,64'((pc+1)%32)<<29);
            // Last completion plus same-domain acceptance preserves one job.
            step(1,1,1,(64'((pc+1)%32)<<29)+8192);
            step(0,0,1,0);step(0,0,0,0);
        end
        // A spurious completion raises a sticky fault without withdrawing VALID.
        step(1,0,0,64'h3fffff000);step(1,0,1,64'h3fffff000);
        repeat(4)step(1,0,0,64'h3fffff000);
        @(negedge clk);#0.001;rst_n=0;s_valid=0;complete=0;
        repeat(3)@(negedge clk);#0.001;rst_n=1;
        step(1,1,0,64'h200000000);step(0,0,1,0);step(0,0,0,0);
        $display("DM ALL_PASS domains=32");$finish;
    end
    initial begin #100000;$fatal(1,"DM_WATCHDOG");end
endmodule
