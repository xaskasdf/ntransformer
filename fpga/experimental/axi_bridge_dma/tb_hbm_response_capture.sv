`timescale 1ns/1ps
module response_capture_case #(parameter integer WIDTH=265, SEED=1)(output reg done=0);
    reg clk=0,rst_n=0,s_valid=0,m_ready=0;
    reg [WIDTH-1:0] s_data=0;
    wire s_ready,m_valid,empty,ref_ready,ref_valid,ref_empty;
    wire [WIDTH-1:0] m_data,ref_data;
    ntrpu_hbm_response_capture_queue #(.WIDTH(WIDTH)) dut(.*);
    ntrpu_hbm_read_response_queue #(.WIDTH(WIDTH)) reference(
        .clk(clk),.rst_n(rst_n),.s_valid(s_valid),.s_ready(ref_ready),.s_data(s_data),
        .m_valid(ref_valid),.m_ready(m_ready),.m_data(ref_data),.empty(ref_empty));
    always #2 clk=~clk;
    reg [WIDTH-1:0] expected[0:1];
    integer occupied=0,checks=0,pushes=0,pops=0,simultaneous=0,discarded=0,speculative=0;
    integer coverage_hits[0:11];
    reg [31:0] rng=SEED;
    function [31:0] next_random(input [31:0] v);
        reg [31:0] x;
        begin x=v^(v<<13);x=x^(x>>17);next_random=x^(x<<5);end
    endfunction
    task check(input bit ok,input string why);
        begin checks=checks+1;if(!ok)$fatal(1,"CAPTURE_QUEUE width=%0d %s",WIDTH,why);end
    endtask
    task cycle(input bit reset_n,input bit valid_in,input bit ready_out);
        reg take,give;reg [WIDTH-1:0] old0,old1;reg old_write;
        integer i,prior;
        begin
            @(negedge clk);rst_n=reset_n;s_valid=valid_in;m_ready=ready_out;
            for(i=0;i<WIDTH;i=i+1)begin rng=next_random(rng);s_data[i]=rng[0];end
            #0.1;
            check({s_ready,m_valid,empty}==={ref_ready,ref_valid,ref_empty},"reference control");
            check(s_ready===(reset_n&&occupied<2),"model ready");
            check(m_valid===(reset_n&&occupied!=0),"model valid");
            if(m_valid)begin
                check(m_data===ref_data,"reference payload");
                check(m_data===expected[0],"scoreboard payload");
            end
            take=s_valid&&s_ready;give=m_valid&&m_ready;prior=occupied;
            old0=dut.slot0;old1=dut.slot1;old_write=dut.write_slot;
            if(reset_n)coverage_hits[occupied*4+valid_in*2+ready_out]=coverage_hits[occupied*4+valid_in*2+ready_out]+1;
            if(!reset_n)begin discarded=discarded+occupied;occupied=0;end
            else begin
                if(give)begin expected[0]=expected[1];occupied=occupied-1;pops=pops+1;end
                if(take)begin expected[occupied]=s_data;occupied=occupied+1;pushes=pushes+1;end
                if(take&&give)simultaneous=simultaneous+1;
                if(s_ready&&!s_valid)speculative=speculative+1;
            end
            @(posedge clk);#0.1;
            check(dut.count===occupied,"model count");
            check({dut.count,dut.read_slot,dut.write_slot}===
                  {reference.count,reference.read_slot,reference.write_slot},"ownership equivalence");
            if(!reset_n||prior==2)begin
                check(dut.slot0===old0&&dut.slot1===old1,"full/reset payload frozen");
            end else if(old_write)begin
                check(dut.slot0===old0,"owned slot0 retained");
                check(dut.slot1===s_data,"free slot1 captured");
            end else begin
                check(dut.slot1===old1,"owned slot1 retained");
                check(dut.slot0===s_data,"free slot0 captured");
            end
        end
    endtask
    integer seq,k,control,i;
    initial begin
        for(i=0;i<12;i=i+1)coverage_hits[i]=0;
        repeat(2)@(posedge clk);
        // Enumerate every 6-cycle VALID/READY sequence from reset (4^6).
        for(seq=0;seq<4096;seq=seq+1)begin
            cycle(0,0,0);control=seq;
            for(k=0;k<6;k=k+1)begin cycle(1,(control&2)!=0,(control&1)!=0);control=control>>2;end
        end
        // Long invalid streams, stalls and full-pop bubble, followed by reset loss.
        cycle(0,0,0);cycle(1,1,0);cycle(1,1,0);
        repeat(32)cycle(1,0,0);
        cycle(1,1,1);cycle(1,1,1);cycle(0,0,0);
        repeat(32)cycle(1,0,1);
        for(k=0;k<20000;k=k+1)begin rng=next_random(rng);cycle((k%997)!=0,rng[2],rng[7]);end
        repeat(4)cycle(1,0,1);
        for(i=0;i<12;i=i+1)check(coverage_hits[i]>0,"occupancy/VALID/READY coverage");
        check(pushes==pops+discarded,"no unexplained accepted losses");
        check(simultaneous>100&&speculative>100&&discarded>100,"activity coverage");
        $display("CAPTURE_QUEUE PASS width=%0d checks=%0d pushes=%0d pops=%0d discarded=%0d simultaneous=%0d speculative=%0d bins=12",WIDTH,checks,pushes,pops,discarded,simultaneous,speculative);
        done=1;
    end
endmodule
module tb_hbm_response_capture;
    wire a,b;
    response_capture_case #(.WIDTH(265),.SEED(32'h31415926)) c265(a);
    response_capture_case #(.WIDTH(519),.SEED(32'h27182818)) c519(b);
    initial begin wait(a&&b);$display("CAPTURE_QUEUE ALL PASS");$finish;end
    initial begin #1000000;$fatal(1,"CAPTURE_QUEUE watchdog");end
endmodule
