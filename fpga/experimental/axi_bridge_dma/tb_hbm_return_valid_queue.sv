`timescale 1ns/1ps
module return_valid_case #(parameter integer WIDTH=519,SEED=1)(output reg done=0);
    reg clk=0,rst_n=0,s_valid=0,m_ready=0;
    reg [WIDTH-1:0] s_data=0;
    wire s_ready,m_valid,empty,r_ready,r_valid,r_empty;
    wire [WIDTH-1:0] m_data,r_data;
    ntrpu_hbm_return_valid_queue #(.WIDTH(WIDTH)) dut(.*);
    ntrpu_hbm_read_return_cut #(.WIDTH(WIDTH)) reference(
        .clk(clk),.rst_n(rst_n),.s_valid(s_valid),.s_ready(r_ready),.s_data(s_data),
        .m_valid(r_valid),.m_ready(m_ready),.m_data(r_data),.empty(r_empty));
    always #2 clk=~clk;
    realtime last_edge=-10;
    always @(posedge clk)last_edge=$realtime;
    always @(m_valid)if(rst_n&&$realtime!=last_edge)
        $fatal(1,"RETURN_VALID_QUEUE width=%0d VALID changed between edges",WIDTH);
    reg [WIDTH-1:0] expected[0:1];
    integer occupied=0,checks=0,pushes=0,pops=0,discarded=0,pairs=0,stalls=0,async_resets=0;
    integer coverage_hits[0:11];
    reg [31:0] rng=SEED;
    function [31:0] next_random(input [31:0] v);
        reg [31:0] x;
        begin x=v^(v<<13);x=x^(x>>17);next_random=x^(x<<5);end
    endfunction
    task check(input bit ok,input string why);
        begin checks=checks+1;if(!ok)$fatal(1,"RETURN_VALID_QUEUE width=%0d %s",WIDTH,why);end
    endtask
    task cycle(input bit reset_n,input bit valid_in,input bit ready_out);
        bit take,give;integer prior,b;reg old_write;reg [WIDTH-1:0] old0,old1;
        begin
            @(negedge clk);rst_n=reset_n;s_valid=valid_in;m_ready=ready_out;
            for(b=0;b<WIDTH;b=b+1)begin rng=next_random(rng);s_data[b]=rng[17];end
            #0.1;
            if(!reset_n)check(!m_valid,"asynchronous VALID clear");
            check({s_ready,m_valid,empty}==={r_ready,r_valid,r_empty},"reference controls");
            check(s_ready===(reset_n&&occupied<2)&&m_valid===(reset_n&&occupied!=0),"queue controls model");
            if(m_valid)begin
                check(m_data===expected[0],"ordered payload model");
                check(m_data===r_data,"reference payload");
            end
            prior=occupied;old0=dut.slot0;old1=dut.slot1;old_write=dut.write_slot;
            take=s_valid&&s_ready;give=m_valid&&m_ready;
            if(reset_n)begin
                coverage_hits[occupied*4+valid_in*2+ready_out]=coverage_hits[occupied*4+valid_in*2+ready_out]+1;
                if(give)begin expected[0]=expected[1];occupied=occupied-1;pops=pops+1;end
                if(take)begin expected[occupied]=s_data;occupied=occupied+1;pushes=pushes+1;end
                if(take&&give)pairs=pairs+1;
                if(prior==2&&!ready_out)stalls=stalls+1;
            end else begin discarded=discarded+occupied;occupied=0;end
            @(posedge clk);#0.1;
            check(dut.count===occupied&&occupied<=2,"bounded occupancy");
            check(m_valid===(reset_n&&occupied!=0),"post-edge valid model");
            check(empty===(occupied==0),"pending empty model");
            if(!reset_n||prior==2)check(dut.slot0===old0&&dut.slot1===old1,"full/reset payload retained");
            else if(old_write)check(dut.slot0===old0&&dut.slot1===s_data,"free slot1 only");
            else check(dut.slot1===old1&&dut.slot0===s_data,"free slot0 only");
        end
    endtask
    integer seq,k,control,i;
    initial begin
        for(i=0;i<12;i=i+1)coverage_hits[i]=0;
        repeat(2)@(posedge clk);
        // Every six-cycle VALID/READY word, then longer stalls and resets.
        for(seq=0;seq<4096;seq=seq+1)begin
            cycle(0,0,0);control=seq;
            for(k=0;k<6;k=k+1)begin cycle(1,(control&2)!=0,(control&1)!=0);control=control>>2;end
        end
        for(k=0;k<12000;k=k+1)begin rng=next_random(rng);cycle((k%499)!=0,rng[3],rng[8]);end
        cycle(0,0,0);cycle(1,1,0);cycle(1,1,0);
        repeat(32)cycle(1,1,0);
        // Assert reset halfway between edges while VALID is high and full.
        #0.6;check(m_valid&&occupied==2,"async reset setup");rst_n=0;
        #0.1;check(!m_valid,"asynchronous VALID clear");async_resets=async_resets+1;
        cycle(0,1,1);cycle(0,0,0);cycle(1,0,0);
        repeat(4)cycle(1,0,1);
        for(i=0;i<12;i=i+1)check(coverage_hits[i]>0,"occupancy/VALID/READY coverage");
        check(pushes==pops+discarded,"accepted conservation");
        check(pairs>100&&stalls>100&&discarded>100,"activity coverage");
        $display("RETURN_VALID_QUEUE PASS width=%0d checks=%0d pushes=%0d pops=%0d discarded=%0d simultaneous=%0d stalls=%0d bins=12 async_resets=%0d",WIDTH,checks,pushes,pops,discarded,pairs,stalls,async_resets);
        done=1;
    end
endmodule
module tb_hbm_return_valid_queue;
    wire a,b;
    return_valid_case #(.WIDTH(265),.SEED(32'h452821e6)) c265(a);
    return_valid_case #(.WIDTH(519),.SEED(32'h38d01377)) c519(b);
    initial begin wait(a&&b);$display("RETURN_VALID_QUEUE ALL PASS");$finish;end
    initial begin #1000000;$fatal(1,"RETURN_VALID_QUEUE watchdog");end
endmodule
