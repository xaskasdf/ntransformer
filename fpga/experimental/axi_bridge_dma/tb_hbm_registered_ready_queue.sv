`timescale 1ns/1ps
module ready_queue_case #(parameter integer WIDTH=265,SEED=1)(output reg done=0);
    reg clk=0,rst_n=0,s_valid=0,m_ready=0,permit_next=0,ready_reg=0;
    reg [WIDTH-1:0] s_data=0;
    wire q_space,next_space,m_valid,empty;
    wire [WIDTH-1:0] m_data;
    ntrpu_hbm_ready_capture_queue #(.WIDTH(WIDTH)) dut(
        .clk(clk),.rst_n(rst_n),.s_valid(s_valid&&ready_reg),.s_ready(q_space),
        .s_data(s_data),.m_valid(m_valid),.m_ready(m_ready),.m_data(m_data),
        .empty(empty),.next_space(next_space));
    always #2 clk=~clk;
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)ready_reg<=0;
        else ready_reg<=next_space&&permit_next;
    end
    reg [WIDTH-1:0] expected[0:1];
    integer occupied=0,checks=0,pushes=0,pops=0,discarded=0,simultaneous=0;
    integer coverage_hits[0:23];
    bit permit=0;
    reg [31:0] rng=SEED;
    function [31:0] next_random(input [31:0] v);
        reg [31:0] x;
        begin x=v^(v<<13);x=x^(x>>17);next_random=x^(x<<5);end
    endfunction
    task check(input bit ok,input string why);
        begin checks=checks+1;if(!ok)$fatal(1,"REGISTERED_READY_QUEUE width=%0d %s",WIDTH,why);end
    endtask
    task cycle(input bit reset_n,input bit valid_in,input bit ready_out,input bit allow_next);
        bit take,give;integer prior;reg old_write;reg [WIDTH-1:0] old0,old1;
        begin
            @(negedge clk);rst_n=reset_n;s_valid=valid_in;m_ready=ready_out;permit_next=allow_next;
            rng=next_random(rng);s_data={((WIDTH+31)/32){rng}};
            #0.1;
            check(ready_reg===(reset_n&&permit&&occupied<2),"registered admission model");
            check(m_valid===(reset_n&&occupied!=0),"valid model");
            if(m_valid)check(m_data===expected[0],"ordered payload model");
            prior=occupied;old0=dut.slot0;old1=dut.slot1;old_write=dut.write_slot;
            take=s_valid&&ready_reg;give=m_valid&&m_ready;
            if(reset_n)begin
                coverage_hits[occupied*8+permit*4+valid_in*2+ready_out]=coverage_hits[occupied*8+permit*4+valid_in*2+ready_out]+1;
                check(!take||occupied<2,"reserved capacity before handshake");
                if(give)begin expected[0]=expected[1];occupied=occupied-1;pops=pops+1;end
                if(take)begin expected[occupied]=s_data;occupied=occupied+1;pushes=pushes+1;end
                if(take&&give)simultaneous=simultaneous+1;
                check(next_space===(occupied<2),"predicted space model");
                permit=allow_next;
            end else begin discarded=discarded+occupied;occupied=0;permit=0;end
            @(posedge clk);#0.1;
            check(dut.count===occupied&&occupied<=2,"bounded occupancy model");
            check(ready_reg===(reset_n&&permit&&occupied<2),"post-edge admission model");
            if(!reset_n||prior==2)check(dut.slot0===old0&&dut.slot1===old1,"full/reset payload retained");
            else if(old_write)check(dut.slot0===old0&&dut.slot1===s_data,"free slot1 only");
            else check(dut.slot1===old1&&dut.slot0===s_data,"free slot0 only");
        end
    endtask
    integer seq,k,control,i;
    initial begin
        for(i=0;i<24;i=i+1)coverage_hits[i]=0;
        repeat(2)@(posedge clk);
        // All five-cycle combinations of VALID, downstream READY and next permit.
        for(seq=0;seq<32768;seq=seq+1)begin
            cycle(0,0,0,0);control=seq;
            for(k=0;k<5;k=k+1)begin cycle(1,(control&2)!=0,(control&1)!=0,(control&4)!=0);control=control>>3;end
        end
        for(k=0;k<20000;k=k+1)begin rng=next_random(rng);cycle((k%997)!=0,rng[2],rng[7],rng[13]);end
        repeat(4)cycle(1,0,1,0);
        for(i=0;i<24;i=i+1)check(coverage_hits[i]>0,"occupancy/permit/VALID/READY coverage");
        check(pushes==pops+discarded,"accepted conservation");
        check(simultaneous>100&&discarded>100,"activity coverage");
        $display("REGISTERED_READY_QUEUE PASS width=%0d checks=%0d pushes=%0d pops=%0d discarded=%0d simultaneous=%0d bins=24",WIDTH,checks,pushes,pops,discarded,simultaneous);
        done=1;
    end
endmodule
module tb_hbm_registered_ready_queue;
    wire a,b;
    ready_queue_case #(.WIDTH(265),.SEED(32'h31415926)) c265(a);
    ready_queue_case #(.WIDTH(519),.SEED(32'h27182818)) c519(b);
    initial begin wait(a&&b);$display("REGISTERED_READY_QUEUE ALL PASS");$finish;end
    initial begin #2000000;$fatal(1,"REGISTERED_READY_QUEUE watchdog");end
endmodule
